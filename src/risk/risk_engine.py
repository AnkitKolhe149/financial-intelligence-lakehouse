"""Risk scoring: combine the rules into a bounded, explainable score.

    score = min(sum(points of rules that fired), max_score)
    level = band(score)

Two properties are guaranteed and tested:

* **Bounded.** The score never exceeds ``risk.max_score``, so "87" always means
  the same thing regardless of how many rules exist.
* **Explainable.** Every point is attributable. ``risk_reasons`` lists the
  evidence for each rule that fired, and ``risk_rule_hits`` lists the rule keys,
  so the score can be reconstructed from its parts. A score with no reasons is a
  bug, and the engine asserts against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.risk.rules import RiskRule, build_rules

log = get_logger(__name__)


@dataclass
class RiskScoringResult:
    """Scored transactions plus per-rule firing statistics."""

    frame: pd.DataFrame
    rule_hits: dict[str, int] = field(default_factory=dict)
    level_counts: dict[str, int] = field(default_factory=dict)
    scored_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def summary(self) -> dict[str, Any]:
        return {
            "records_scored": len(self.frame),
            **{f"level_{k.lower()}": v for k, v in self.level_counts.items()},
            **{f"rule_{k}": v for k, v in self.rule_hits.items()},
        }

    @property
    def high_risk_count(self) -> int:
        return self.level_counts.get("HIGH", 0) + self.level_counts.get("CRITICAL", 0)


class RiskEngine:
    """Applies the configured rules to a feature frame."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.risk
        self.rules: list[RiskRule] = build_rules(self.config)

    def score(self, frame: pd.DataFrame) -> RiskScoringResult:
        """Attach ``risk_score``, ``risk_level``, ``risk_reasons``, ``risk_rule_hits``."""
        if frame.empty:
            scored = frame.copy()
            scored["risk_score"] = pd.Series(dtype="int64")
            scored["risk_level"] = pd.Series(dtype="string")
            scored["risk_reasons"] = pd.Series(dtype="object")
            scored["risk_rule_hits"] = pd.Series(dtype="object")
            scored["risk_scored_at"] = pd.Series(dtype="datetime64[ns, UTC]")
            return RiskScoringResult(scored)

        scored = frame.copy()
        total = pd.Series(0, index=scored.index, dtype="int64")
        # Reasons are accumulated as parallel arrays and zipped at the end;
        # appending to per-row Python lists inside a loop over rules would be
        # an order of magnitude slower at 500k rows.
        fired_masks: list[tuple[str, pd.Series, pd.Series]] = []
        rule_hits: dict[str, int] = {}

        for rule in self.rules:
            if not rule.applicable(scored):
                log.debug(
                    "risk_rule_skipped",
                    rule=rule.key,
                    reason="required features absent",
                    requires=list(rule.requires),
                )
                rule_hits[rule.key] = 0
                continue

            outcome = rule.evaluate(scored)
            total = total + (outcome.mask.astype("int64") * rule.points)
            fired_masks.append((rule.key, outcome.mask, outcome.reasons))
            rule_hits[rule.key] = outcome.fired

        capped = total.clip(upper=self.config.max_score)
        scored["risk_score"] = capped.astype("int64")
        scored["risk_level"] = self._levels(capped)
        reasons, hits = self._collect_explanations(scored.index, fired_masks)
        scored["risk_reasons"] = reasons
        scored["risk_rule_hits"] = hits
        scored["risk_scored_at"] = datetime.now(UTC)

        self._assert_explainable(scored)

        level_counts = scored["risk_level"].value_counts().to_dict()
        result = RiskScoringResult(
            frame=scored,
            rule_hits=rule_hits,
            level_counts={str(k): int(v) for k, v in level_counts.items()},
        )
        log.info("risk_scored", **result.summary())
        return result

    # -- internals ----------------------------------------------------------

    def _levels(self, scores: pd.Series) -> pd.Series:
        """Vectorised band assignment from the configured level boundaries."""
        bands = sorted(self.config.levels.items(), key=lambda item: item[1].min)
        edges = [band.min for _, band in bands]
        names = [name for name, _ in bands]
        # searchsorted finds the band whose lower edge is the greatest one not
        # exceeding the score.
        indices = np.searchsorted(edges, scores.to_numpy(), side="right") - 1
        indices = np.clip(indices, 0, len(names) - 1)
        return pd.Series([names[i] for i in indices], index=scores.index, dtype="string")

    def _collect_explanations(
        self, index: pd.Index, fired: list[tuple[str, pd.Series, pd.Series]]
    ) -> tuple[pd.Series, pd.Series]:
        """Build the per-row reason and rule-key lists."""
        reasons: list[list[str]] = [[] for _ in range(len(index))]
        hits: list[list[str]] = [[] for _ in range(len(index))]
        position = {label: i for i, label in enumerate(index)}

        for rule_key, mask, reason_series in fired:
            if not mask.any():
                continue
            selected = reason_series[mask]
            for label, text in selected.items():
                slot = position[label]
                if text:
                    reasons[slot].append(str(text))
                hits[slot].append(rule_key)

        return (
            pd.Series(reasons, index=index, dtype="object"),
            pd.Series(hits, index=index, dtype="object"),
        )

    def _assert_explainable(self, scored: pd.DataFrame) -> None:
        """Every non-zero score must carry at least one reason.

        This is an internal consistency check, not a data check: if it fires,
        a rule has contributed points without producing an explanation, and the
        output would be unauditable.
        """
        unexplained = scored.loc[
            (scored["risk_score"] > 0) & (scored["risk_reasons"].map(len) == 0)
        ]
        if not unexplained.empty:
            raise AssertionError(
                f"{len(unexplained)} transactions have a risk score with no explanation; "
                "a rule is contributing points without a reason"
            )

    def describe(self) -> list[dict[str, Any]]:
        return [rule.describe() for rule in self.rules]

    @property
    def max_achievable_score(self) -> int:
        """The highest score any single transaction could reach."""
        return min(sum(rule.points for rule in self.rules), self.config.max_score)


def score_risk(frame: pd.DataFrame, settings: Settings | None = None) -> RiskScoringResult:
    """Convenience wrapper used by the pipelines."""
    return RiskEngine(settings).score(frame)


def explain_transaction(scored: pd.DataFrame, transaction_id: str) -> dict[str, Any]:
    """Full explanation for one transaction - what the API and dashboard show."""
    matches = scored.loc[scored["transaction_id"].astype(str) == str(transaction_id)]
    if matches.empty:
        return {"transaction_id": transaction_id, "found": False}

    row = matches.iloc[0]
    return {
        "found": True,
        "transaction_id": str(row["transaction_id"]),
        "account_id": str(row.get("account_id", "")),
        "vendor_id": str(row.get("vendor_id", "")),
        "amount_usd": float(row.get("amount_usd", 0) or 0),
        "transaction_timestamp": str(row.get("transaction_timestamp", "")),
        "risk_score": int(row.get("risk_score", 0) or 0),
        "risk_level": str(row.get("risk_level", "LOW")),
        "risk_reasons": list(row.get("risk_reasons", []) or []),
        "rules_fired": list(row.get("risk_rule_hits", []) or []),
        "anomaly_score": float(row.get("anomaly_score", 0) or 0),
        "is_anomaly": bool(row.get("is_anomaly", False)),
        "model_version": str(row.get("model_version", "")),
        "evidence": {
            "account_prior_transactions": int(row.get("account_txn_count_prior", 0) or 0),
            "account_prior_average_usd": round(
                float(row.get("account_avg_amount_prior", 0) or 0), 2
            ),
            "amount_to_account_average": round(float(row.get("amount_to_account_avg", 0) or 0), 2),
            "vendor_prior_transactions": int(row.get("vendor_txn_count_prior", 0) or 0),
            "transactions_in_prior_5min": int(row.get("account_velocity_5min", 0) or 0),
            "outside_home_region": bool(row.get("is_region_mismatch", 0)),
        },
    }
