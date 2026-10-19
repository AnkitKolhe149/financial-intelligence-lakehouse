"""Deterministic risk rules.

Each rule is a small object with three responsibilities: decide which
transactions it fires on (vectorised), say how many points it contributes, and
explain itself in a sentence containing the actual evidence. The points and
thresholds come from ``config/settings.yaml``, so tuning the risk appetite is a
configuration change reviewed in a pull request, not a code change.

Two design points worth stating:

**Rules read the same features as the model.** They consume the leakage-safe
feature frame from :mod:`src.ml.features`, so "12x the account average" means
the average *before* this transaction. A rule computed against the
whole-dataset average would quietly weaken itself on exactly the transactions it
exists to catch.

**Reasons carry evidence, not labels.** "Amount is 12.4x the account's prior
average of $1,204" is actionable. "AMOUNT_ANOMALY" is not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.config import RiskConfig
from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class RuleOutcome:
    """Which rows fired, and the per-row explanation for those that did."""

    mask: pd.Series
    reasons: pd.Series
    points: int

    @property
    def fired(self) -> int:
        return int(self.mask.sum())


class RiskRule(ABC):
    """Base class for a deterministic risk rule."""

    #: Stable identifier, matching the key in ``risk.rules`` configuration.
    key: str = "unnamed_rule"
    #: Short description used in documentation and the dashboard.
    description: str = ""
    #: Features the rule needs; missing ones cause the rule to skip cleanly.
    requires: tuple[str, ...] = ()

    def __init__(self, points: int, parameters: dict[str, Any] | None = None) -> None:
        self.points = points
        self.parameters = parameters or {}

    def applicable(self, frame: pd.DataFrame) -> bool:
        return all(column in frame.columns for column in self.requires)

    @abstractmethod
    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        """Return which rows fire and why."""

    def _outcome(self, mask: pd.Series, reasons: pd.Series) -> RuleOutcome:
        mask = mask.fillna(False).astype(bool)
        return RuleOutcome(mask=mask, reasons=reasons.where(mask, ""), points=self.points)

    def describe(self) -> dict[str, Any]:
        return {
            "rule": self.key,
            "points": self.points,
            "description": self.description,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


class AmountVersusAccountAverageRule(RiskRule):
    """Payment far larger than this account's own historical norm."""

    key = "amount_vs_account_average"
    description = "Transaction amount is a large multiple of the account's prior average."
    requires = ("amount_to_account_avg", "account_avg_amount_prior", "account_txn_count_prior")

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        threshold = float(self.parameters.get("multiplier_threshold", 10.0))
        ratio = pd.to_numeric(frame["amount_to_account_avg"], errors="coerce")
        history = pd.to_numeric(frame["account_txn_count_prior"], errors="coerce")

        # Require a minimum history: the second transaction on a new account is
        # not "10x the average" in any meaningful sense.
        mask = (ratio >= threshold) & (history >= 5)
        reasons = (
            "Amount is "
            + ratio.round(1).astype("string")
            + "x the account's prior average of $"
            + pd.to_numeric(frame["account_avg_amount_prior"], errors="coerce")
            .round(0)
            .map(lambda v: f"{v:,.0f}" if pd.notna(v) else "0")
        )
        return self._outcome(mask, reasons)


class UnusualHourRule(RiskRule):
    """Activity in the small hours, when nobody should be disbursing funds."""

    key = "unusual_hour"
    description = "Transaction occurred during overnight hours."
    requires = ("hour_of_day",)

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        start = int(self.parameters.get("start_hour", 0))
        end = int(self.parameters.get("end_hour", 4))
        hour = pd.to_numeric(frame["hour_of_day"], errors="coerce")
        mask = hour.between(start, end)
        reasons = (
            "Transaction occurred at "
            + hour.fillna(0).astype(int).astype("string").str.zfill(2)
            + ":00, outside normal disbursement hours"
        )
        return self._outcome(mask, reasons)


class NewVendorRule(RiskRule):
    """Payment to a vendor with little or no history on the platform."""

    key = "new_vendor"
    description = "Vendor has limited prior activity."
    requires = ("vendor_txn_count_prior",)

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        minimum = int(self.parameters.get("min_history_count", 5))
        history = pd.to_numeric(frame["vendor_txn_count_prior"], errors="coerce").fillna(0)
        mask = history < minimum
        reasons = (
            "Vendor has only "
            + history.astype(int).astype("string")
            + f" prior transactions (threshold {minimum})"
        )
        return self._outcome(mask, reasons)


class HighVelocityRule(RiskRule):
    """A burst of transactions from one account in a very short window."""

    key = "high_velocity"
    description = "Unusual number of transactions from the account in a short window."
    requires = ("account_velocity_5min",)

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        maximum = int(self.parameters.get("max_transactions", 5))
        window = int(self.parameters.get("window_minutes", 5))
        velocity = pd.to_numeric(frame["account_velocity_5min"], errors="coerce").fillna(0)
        mask = velocity > maximum
        reasons = (
            "Account made "
            + velocity.astype(int).astype("string")
            + f" transactions in the preceding {window} minutes (threshold {maximum})"
        )
        return self._outcome(mask, reasons)


class GeographicAnomalyRule(RiskRule):
    """Spend outside the account's home region."""

    key = "geographic_anomaly"
    description = "Transaction location is outside the account's home region."
    requires = ("is_region_mismatch",)

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        mask = pd.to_numeric(frame["is_region_mismatch"], errors="coerce").fillna(0) > 0
        reasons = pd.Series(
            "Transaction occurred outside the account's home region",
            index=frame.index,
            dtype="string",
        )
        return self._outcome(mask, reasons)


class MLAnomalyRule(RiskRule):
    """The unsupervised model flagged this transaction.

    Carried as a rule so that the model's contribution to the final score is
    explicit and bounded, rather than the model silently overriding everything
    else. It contributes the largest single weight but cannot reach CRITICAL on
    its own - a deliberate choice, since an unexplainable 30 points should not
    escalate a case by itself.
    """

    key = "ml_anomaly"
    description = "Isolation Forest flagged the transaction as anomalous."
    requires = ("is_anomaly",)

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        mask = frame["is_anomaly"].fillna(False).astype(bool)
        score = pd.to_numeric(frame.get("anomaly_score"), errors="coerce")
        reasons = (
            "ML anomaly detected (score " + score.round(3).astype("string").fillna("n/a") + ")"
        )
        return self._outcome(mask, reasons)


class RoundAmountRule(RiskRule):
    """Suspiciously round amounts just under a reporting threshold.

    Structuring - deliberately keeping payments below a review threshold - shows
    up as clusters of round numbers just beneath it.
    """

    key = "round_amount"
    description = "Round-numbered amount just below a review threshold."
    requires = ("amount_usd",)

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        threshold = float(self.parameters.get("threshold", 10_000))
        amount = pd.to_numeric(frame["amount_usd"], errors="coerce").fillna(0)
        is_round = (amount % 100 == 0) & (amount > 0)
        just_under = amount.between(threshold * 0.9, threshold)
        mask = is_round & just_under
        reasons = (
            "Round amount of $"
            + amount.map(lambda v: f"{v:,.0f}")
            + f" sits just below the ${threshold:,.0f} review threshold"
        )
        return self._outcome(mask, reasons)


class WeekendActivityRule(RiskRule):
    """Disbursement on a weekend, when approval chains are thin."""

    key = "weekend_activity"
    description = "Transaction occurred at the weekend."
    requires = ("is_weekend",)

    def evaluate(self, frame: pd.DataFrame) -> RuleOutcome:
        mask = pd.to_numeric(frame["is_weekend"], errors="coerce").fillna(0) > 0
        reasons = pd.Series(
            "Transaction occurred at the weekend, outside standard approval windows",
            index=frame.index,
            dtype="string",
        )
        return self._outcome(mask, reasons)


#: Rule key -> implementation.
RULE_REGISTRY: dict[str, type[RiskRule]] = {
    AmountVersusAccountAverageRule.key: AmountVersusAccountAverageRule,
    UnusualHourRule.key: UnusualHourRule,
    NewVendorRule.key: NewVendorRule,
    HighVelocityRule.key: HighVelocityRule,
    GeographicAnomalyRule.key: GeographicAnomalyRule,
    MLAnomalyRule.key: MLAnomalyRule,
    RoundAmountRule.key: RoundAmountRule,
    WeekendActivityRule.key: WeekendActivityRule,
}


def build_rules(config: RiskConfig) -> list[RiskRule]:
    """Instantiate the configured rules.

    An unknown key in configuration is a warning, not a crash: a rule that has
    not been implemented yet should not stop the pipeline from scoring with the
    ones that have.
    """
    rules: list[RiskRule] = []
    for key, parameters in config.rules.items():
        implementation = RULE_REGISTRY.get(key)
        if implementation is None:
            log.warning("unknown_risk_rule", rule=key, detail="no implementation registered")
            continue
        points = int(parameters.get("points", 0))
        rules.append(implementation(points=points, parameters=parameters))

    total_possible = sum(rule.points for rule in rules)
    if total_possible < config.max_score:
        log.debug(
            "risk_rules_cannot_reach_max",
            total_points=total_possible,
            max_score=config.max_score,
            detail="no single transaction can reach the maximum score",
        )
    return rules


def describe_rules(config: RiskConfig) -> list[dict[str, Any]]:
    """Serialisable rule documentation, surfaced by the API and the dashboard."""
    return [rule.describe() for rule in build_rules(config)]
