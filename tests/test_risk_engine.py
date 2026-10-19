"""Rule-based risk scoring: correctness, bounds and explainability."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk.risk_engine import RiskEngine, explain_transaction
from src.risk.rules import (
    RULE_REGISTRY,
    AmountVersusAccountAverageRule,
    HighVelocityRule,
    MLAnomalyRule,
    NewVendorRule,
    RoundAmountRule,
    UnusualHourRule,
    describe_rules,
)


@pytest.fixture
def features() -> pd.DataFrame:
    """A feature frame with one deliberately boring baseline row."""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN-BASELINE"],
            "account_id": ["ACC-1"],
            "vendor_id": ["VEN-1"],
            "amount_usd": [500.0],
            "hour_of_day": [14.0],
            "day_of_week": [2.0],
            "is_weekend": [0.0],
            "is_night": [0.0],
            "amount_to_account_avg": [1.0],
            "account_avg_amount_prior": [500.0],
            "account_txn_count_prior": [100.0],
            "account_velocity_5min": [1.0],
            "vendor_txn_count_prior": [50.0],
            "is_region_mismatch": [0.0],
            "is_anomaly": [False],
            "anomaly_score": [0.1],
        }
    )


class TestIndividualRules:
    def test_amount_rule_needs_both_a_multiple_and_a_history(self, features, settings):
        rule = AmountVersusAccountAverageRule(25, settings.risk.rule("amount_vs_account_average"))

        quiet = rule.evaluate(features)
        assert quiet.fired == 0

        spike = features.copy()
        spike["amount_to_account_avg"] = 15.0
        assert rule.evaluate(spike).fired == 1

        # A new account's second payment is not "15x the average" in any
        # meaningful sense, so the rule must stay silent.
        new_account = spike.copy()
        new_account["account_txn_count_prior"] = 2.0
        assert rule.evaluate(new_account).fired == 0

    def test_amount_reason_carries_the_evidence(self, features, settings):
        spike = features.copy()
        spike["amount_to_account_avg"] = 12.4
        outcome = AmountVersusAccountAverageRule(
            25, settings.risk.rule("amount_vs_account_average")
        ).evaluate(spike)
        reason = outcome.reasons.iloc[0]
        assert "12.4x" in reason
        assert "500" in reason

    @pytest.mark.parametrize(
        ("hour", "fires"), [(3.0, True), (0.0, True), (4.0, True), (14.0, False)]
    )
    def test_unusual_hour(self, features, settings, hour, fires):
        frame = features.copy()
        frame["hour_of_day"] = hour
        outcome = UnusualHourRule(10, settings.risk.rule("unusual_hour")).evaluate(frame)
        assert (outcome.fired == 1) is fires

    def test_new_vendor_threshold(self, features, settings):
        rule = NewVendorRule(15, settings.risk.rule("new_vendor"))
        assert rule.evaluate(features).fired == 0

        newcomer = features.copy()
        newcomer["vendor_txn_count_prior"] = 1.0
        assert rule.evaluate(newcomer).fired == 1

    def test_velocity(self, features, settings):
        rule = HighVelocityRule(20, settings.risk.rule("high_velocity"))
        assert rule.evaluate(features).fired == 0

        burst = features.copy()
        burst["account_velocity_5min"] = 12.0
        outcome = rule.evaluate(burst)
        assert outcome.fired == 1
        assert "12 transactions" in outcome.reasons.iloc[0]

    def test_round_amount_only_just_below_the_threshold(self, features, settings):
        rule = RoundAmountRule(5, settings.risk.rule("round_amount"))

        structuring = features.copy()
        structuring["amount_usd"] = 9500.0
        assert rule.evaluate(structuring).fired == 1

        # Round but nowhere near the threshold: not structuring.
        ordinary = features.copy()
        ordinary["amount_usd"] = 100.0
        assert rule.evaluate(ordinary).fired == 0

        # Just below the threshold but not round.
        odd = features.copy()
        odd["amount_usd"] = 9512.37
        assert rule.evaluate(odd).fired == 0

    def test_ml_rule_follows_the_model(self, features, settings):
        rule = MLAnomalyRule(30, settings.risk.rule("ml_anomaly"))
        assert rule.evaluate(features).fired == 0

        flagged = features.copy()
        flagged["is_anomaly"] = True
        flagged["anomaly_score"] = 0.94
        outcome = rule.evaluate(flagged)
        assert outcome.fired == 1
        assert "0.94" in outcome.reasons.iloc[0]

    def test_rule_skips_when_its_features_are_absent(self, settings):
        rule = HighVelocityRule(20, settings.risk.rule("high_velocity"))
        assert not rule.applicable(pd.DataFrame({"transaction_id": ["A"]}))

    def test_every_configured_rule_is_implemented(self, settings):
        for key in settings.risk.rules:
            assert key in RULE_REGISTRY, f"rule '{key}' is configured but not implemented"


class TestRiskEngine:
    def test_baseline_transaction_scores_zero(self, features, settings):
        result = RiskEngine(settings).score(features)
        assert result.frame.iloc[0]["risk_score"] == 0
        assert result.frame.iloc[0]["risk_level"] == "LOW"
        assert result.frame.iloc[0]["risk_reasons"] == []

    def test_scores_accumulate_across_rules(self, features, settings):
        frame = features.copy()
        frame["amount_to_account_avg"] = 15.0  # +25
        frame["hour_of_day"] = 3.0  # +10
        frame["is_night"] = 1.0
        frame["vendor_txn_count_prior"] = 1.0  # +15
        result = RiskEngine(settings).score(frame)
        assert result.frame.iloc[0]["risk_score"] == 50
        assert result.frame.iloc[0]["risk_level"] == "HIGH"
        assert len(result.frame.iloc[0]["risk_reasons"]) == 3

    def test_score_is_capped_at_the_maximum(self, features, settings):
        """Every rule firing at once must not exceed 100."""
        frame = features.copy()
        frame["amount_to_account_avg"] = 50.0
        frame["hour_of_day"] = 2.0
        frame["is_night"] = 1.0
        frame["is_weekend"] = 1.0
        frame["vendor_txn_count_prior"] = 0.0
        frame["account_velocity_5min"] = 30.0
        frame["is_region_mismatch"] = 1.0
        frame["is_anomaly"] = True
        frame["amount_usd"] = 9500.0

        result = RiskEngine(settings).score(frame)
        assert result.frame.iloc[0]["risk_score"] == settings.risk.max_score
        assert result.frame.iloc[0]["risk_level"] == "CRITICAL"

    @pytest.mark.parametrize(
        ("score", "level"),
        [
            (0, "LOW"),
            (24, "LOW"),
            (25, "MEDIUM"),
            (49, "MEDIUM"),
            (50, "HIGH"),
            (74, "HIGH"),
            (75, "CRITICAL"),
            (100, "CRITICAL"),
        ],
    )
    def test_band_boundaries(self, settings, score, level):
        assert settings.risk.level_for(score) == level

    def test_every_scored_transaction_is_explainable(self, features, settings):
        """A score with no reasons would be unauditable - the engine asserts it."""
        frame = pd.concat([features] * 5, ignore_index=True)
        frame["amount_to_account_avg"] = [1.0, 15.0, 1.0, 20.0, 1.0]
        frame["transaction_id"] = [f"TXN-{i}" for i in range(5)]

        result = RiskEngine(settings).score(frame)
        scored = result.frame
        for _, row in scored.iterrows():
            if row["risk_score"] > 0:
                assert row["risk_reasons"], f"{row['transaction_id']} scored with no reason"
                assert row["risk_rule_hits"]

    def test_rule_hits_reconstruct_the_score(self, features, settings):
        frame = features.copy()
        frame["amount_to_account_avg"] = 15.0
        frame["vendor_txn_count_prior"] = 1.0

        result = RiskEngine(settings).score(frame)
        row = result.frame.iloc[0]
        points = {key: cfg["points"] for key, cfg in settings.risk.rules.items()}
        assert sum(points[rule] for rule in row["risk_rule_hits"]) == row["risk_score"]

    def test_empty_frame_produces_the_right_columns(self, features, settings):
        result = RiskEngine(settings).score(features.iloc[0:0])
        for column in ("risk_score", "risk_level", "risk_reasons", "risk_rule_hits"):
            assert column in result.frame.columns

    def test_explanation_payload(self, features, settings):
        frame = features.copy()
        frame["amount_to_account_avg"] = 15.0
        scored = RiskEngine(settings).score(frame).frame

        explanation = explain_transaction(scored, "TXN-BASELINE")
        assert explanation["found"] is True
        assert explanation["risk_score"] == 25
        assert explanation["risk_reasons"]
        assert explanation["evidence"]["account_prior_transactions"] == 100

    def test_unknown_transaction(self, features, settings):
        scored = RiskEngine(settings).score(features).frame
        assert explain_transaction(scored, "TXN-NOPE")["found"] is False

    def test_rules_are_documented_for_the_api(self, settings):
        described = describe_rules(settings.risk)
        assert len(described) == len(settings.risk.rules)
        for entry in described:
            assert entry["description"], f"{entry['rule']} has no description"
            assert entry["points"] > 0

    def test_configuration_drives_the_weights(self, features, settings):
        """Changing a rule's points in configuration changes the score."""
        frame = features.copy()
        frame["vendor_txn_count_prior"] = 1.0

        original = RiskEngine(settings).score(frame).frame.iloc[0]["risk_score"]

        rules = dict(settings.risk.rules)
        rules["new_vendor"] = {**rules["new_vendor"], "points": 40}
        louder = settings.model_copy(
            update={"risk": settings.risk.model_copy(update={"rules": rules})}
        )
        adjusted = RiskEngine(louder).score(frame).frame.iloc[0]["risk_score"]

        assert original == 15
        assert adjusted == 40
