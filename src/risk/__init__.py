"""Explainable, rule-based risk scoring - the counterpart to the ML model.

The platform deliberately runs **both** a statistical model and a deterministic
rule engine, because each covers the other's weakness:

* Isolation Forest finds patterns nobody wrote down, but cannot tell an
  investigator *why* a transaction is suspicious. "Anomaly score 0.94" is not
  something you can put in front of an oversight committee.
* Rules encode what the business already knows, are auditable line by line, and
  produce a human-readable justification - but they only ever find what someone
  thought to write down.

Every score here is decomposable: a transaction scoring 65 has a stored
``risk_reasons`` list showing exactly which rules fired and with what evidence
("Amount is 12.4x the account's prior average"). Nothing contributes to a score
without explaining itself, which is what makes the output defensible.
"""

from src.risk.risk_engine import RiskEngine, RiskScoringResult, score_risk
from src.risk.rules import RULE_REGISTRY, RiskRule, build_rules

__all__ = [
    "RULE_REGISTRY",
    "RiskEngine",
    "RiskRule",
    "RiskScoringResult",
    "build_rules",
    "score_risk",
]
