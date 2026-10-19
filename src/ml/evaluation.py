"""Evaluating an unsupervised detector honestly.

Fraud data is extremely imbalanced - roughly 0.4% of the generated population.
That makes **accuracy meaningless**: a model that flags nothing scores 99.6% and
detects no fraud. This module therefore never reports accuracy. It reports:

``precision@k``
    Of the k highest-scored transactions, how many were genuinely fraudulent?
    This is the metric that matches the operating reality - a review team can
    look at k cases a day, and precision@k is exactly their hit rate.

``recall@k``
    What share of all known fraud appears in that top-k?

``PR-AUC (average precision)``
    Area under the precision-recall curve. The right summary statistic for
    imbalanced problems; ROC-AUC looks flattering on imbalanced data because the
    huge true-negative count dominates the false-positive rate.

``lift``
    How many times better than random selection the model is. A lift of 20 means
    the top-scored bucket contains 20x the fraud density of the population.

``business rule agreement``
    Overlap between ML anomalies and deterministic rule hits. Where they agree,
    confidence is high; where the model fires alone, it may be finding something
    the rules miss - or drifting. Both are worth knowing.

When no labels are available the module still returns the label-free
diagnostics, because that is the situation the platform is designed for.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (10, 50, 100, 500)


def evaluate_model(
    scored: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate scored transactions, with or without ground-truth labels."""
    if scored.empty:
        return {"status": "no_data"}

    report: dict[str, Any] = {
        "records_scored": len(scored),
        "anomalies_flagged": int(scored["is_anomaly"].sum()) if "is_anomaly" in scored else 0,
        "score_distribution": _score_distribution(scored),
    }
    report["anomaly_rate_pct"] = round(
        report["anomalies_flagged"] / report["records_scored"] * 100, 4
    )

    truth = _join_labels(scored, labels)
    if truth is None:
        report["labelled"] = False
        report["note"] = (
            "No ground-truth labels available; unsupervised diagnostics only. "
            "This is the expected production condition."
        )
        return report

    y_true, y_score = truth
    positives = int(y_true.sum())
    report["labelled"] = True
    report["known_fraud_in_scope"] = positives
    report["fraud_base_rate_pct"] = round(positives / len(y_true) * 100, 4)

    if positives == 0:
        report["note"] = "No labelled fraud fell inside the evaluated window"
        return report

    report["precision_at_k"] = {}
    report["recall_at_k"] = {}
    report["lift_at_k"] = {}
    base_rate = positives / len(y_true)

    order = np.argsort(-y_score)
    ranked_truth = y_true[order]
    for k in k_values:
        if k > len(y_true):
            continue
        hits = int(ranked_truth[:k].sum())
        precision = hits / k
        report["precision_at_k"][str(k)] = round(precision, 4)
        report["recall_at_k"][str(k)] = round(hits / positives, 4)
        report["lift_at_k"][str(k)] = round(precision / base_rate, 2) if base_rate else None

    report["pr_auc"] = round(_average_precision(y_true, y_score), 4)
    report["roc_auc"] = round(_roc_auc(y_true, y_score), 4)
    report["confusion_matrix"] = _confusion_matrix(scored, y_true)
    report["metric_note"] = (
        "Accuracy is deliberately not reported: at a "
        f"{report['fraud_base_rate_pct']}% base rate, predicting 'never fraud' "
        "would score above 99% while detecting nothing."
    )

    log.info(
        "model_evaluated",
        records=report["records_scored"],
        known_fraud=positives,
        pr_auc=report["pr_auc"],
        precision_at_50=report["precision_at_k"].get("50"),
    )
    return report


def _join_labels(
    scored: pd.DataFrame, labels: pd.DataFrame | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Attach ground truth to scored rows. Returns ``(y_true, y_score)``."""
    if labels is None or labels.empty or "transaction_id" not in scored.columns:
        return None
    if "transaction_id" not in labels.columns:
        return None

    fraud_ids = set(
        labels.loc[labels.get("is_fraud", True).astype(bool), "transaction_id"].astype(str)
    )
    if not fraud_ids:
        return None

    y_true = scored["transaction_id"].astype(str).isin(fraud_ids).to_numpy().astype(int)
    y_score = pd.to_numeric(scored["anomaly_score"], errors="coerce").fillna(0.0).to_numpy()
    return y_true, y_score


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC via the step-wise average-precision definition."""
    order = np.argsort(-y_score)
    truth = y_true[order]
    cumulative_tp = np.cumsum(truth)
    precision = cumulative_tp / np.arange(1, len(truth) + 1)
    total_positives = truth.sum()
    if total_positives == 0:
        return 0.0
    # Only positions holding a true positive contribute a recall increment.
    return float(np.sum(precision * truth) / total_positives)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the rank statistic (Mann-Whitney U).

    Reported for completeness and comparability, with the caveat above: on a
    0.4% base rate it will look good almost regardless of model quality.
    """
    positives = y_true == 1
    negatives = ~positives
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_positive = ranks[positives].sum()
    return float((rank_sum_positive - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _confusion_matrix(scored: pd.DataFrame, y_true: np.ndarray) -> dict[str, int]:
    if "is_anomaly" not in scored.columns:
        return {}
    predicted = scored["is_anomaly"].fillna(False).to_numpy().astype(int)
    return {
        "true_positive": int(((predicted == 1) & (y_true == 1)).sum()),
        "false_positive": int(((predicted == 1) & (y_true == 0)).sum()),
        "false_negative": int(((predicted == 0) & (y_true == 1)).sum()),
        "true_negative": int(((predicted == 0) & (y_true == 0)).sum()),
    }


def _score_distribution(scored: pd.DataFrame) -> dict[str, float]:
    if "anomaly_score" not in scored.columns:
        return {}
    values = pd.to_numeric(scored["anomaly_score"], errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "min": round(float(values.min()), 6),
        "p50": round(float(values.quantile(0.50)), 6),
        "p95": round(float(values.quantile(0.95)), 6),
        "p99": round(float(values.quantile(0.99)), 6),
        "max": round(float(values.max()), 6),
        "mean": round(float(values.mean()), 6),
    }


def rule_model_agreement(scored: pd.DataFrame) -> dict[str, Any]:
    """Overlap between ML anomalies and rule-based high risk.

    Run after the risk engine. Disagreement is the interesting signal: a large
    ML-only population suggests either genuine novel patterns or model drift,
    and a large rules-only population suggests the model has stopped seeing
    something the business already knows about.
    """
    if scored.empty or not {"is_anomaly", "risk_level"} <= set(scored.columns):
        return {}

    ml_flag = scored["is_anomaly"].fillna(False).astype(bool)
    rule_flag = scored["risk_level"].isin(["HIGH", "CRITICAL"])

    both = int((ml_flag & rule_flag).sum())
    ml_only = int((ml_flag & ~rule_flag).sum())
    rule_only = int((~ml_flag & rule_flag).sum())
    flagged = both + ml_only + rule_only

    return {
        "both": both,
        "ml_only": ml_only,
        "rules_only": rule_only,
        "agreement_pct": round(both / flagged * 100, 2) if flagged else 0.0,
        "interpretation": (
            "High overlap means the model is largely re-deriving known rules; "
            "a large ML-only population is where genuinely novel patterns - or "
            "drift - would show up."
        ),
    }
