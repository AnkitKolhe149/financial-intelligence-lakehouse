"""Feature engineering with enforced temporal integrity.

Every historical feature here answers a question of the form *"what did we know
about this account **before** this transaction?"* - never "what is true of this
account overall". The distinction is the entire ballgame in fraud modelling:

    account_avg_amount computed over the whole dataset  -> leakage
    account_avg_amount computed over strictly prior rows -> correct

The first version lets a transaction influence its own baseline. A large
fraudulent payment raises the account average that it is then compared against,
which shrinks its own deviation and hides it. Offline metrics improve; the model
gets worse. This module computes the second version throughout, using cumulative
arithmetic (O(n)) rather than expanding-window ``apply`` (O(n·k)) so it stays
fast at 500,000 rows.

The same reasoning governs the split: :func:`temporal_split` cuts the data by
time, never at random. A random split would train on Wednesday and test on
Tuesday, which no production model ever gets to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)

#: Features fed to the model, in a fixed order.
#:
#: Order is part of the model contract: it is persisted in the model registry
#: and asserted at scoring time, because silently reordering columns between
#: training and scoring produces a model that runs happily and predicts noise.
#:
#: **Every feature here is stationary.** That constraint is the single most
#: important property of this list, and it was learned the hard way: an earlier
#: version fed the model raw cumulative counts (``account_txn_count_prior``,
#: ``vendor_txn_count_prior``). Those grow monotonically with calendar time, so
#: a model trained on the first 70% of the timeline sees systematically smaller
#: values than it later scores against. Every record in the test period looks
#: mildly unusual for the trivial reason that it is *later*, the flagged rate
#: drifts far above the configured contamination, and genuine anomalies are
#: buried in that drift.
#:
#: The fix is to express history as **ratios and bounded time windows** rather
#: than as running totals: "how does this amount compare to this account's own
#: prior average", "how many transactions in the last 24 hours", "what fraction
#: of this vendor's payers are distinct". Those statements mean the same thing
#: in January and in December, which is what makes a model trained on one usable
#: on the other. Absolute history is retained only as ``log_account_history``,
#: log-compressed so that the difference between 10 and 20 prior transactions
#: matters and the difference between 900 and 1000 does not.
FEATURE_COLUMNS: tuple[str, ...] = (
    # --- amount (stationary: the amount distribution is stable over time) ---
    "amount_usd",
    "log_amount",
    # --- calendar (bounded by construction) ---------------------------------
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    # --- relative amount behaviour ------------------------------------------
    "amount_to_account_avg",
    "amount_zscore_account",
    # --- bounded time windows -----------------------------------------------
    "account_txn_count_24h",
    "account_velocity_5min",
    "vendor_txn_count_7d",
    "log_seconds_since_last_txn",
    # --- ratios rather than running totals -----------------------------------
    "account_vendor_diversity",
    "vendor_account_concentration",
    # --- binary signals -------------------------------------------------------
    "is_new_vendor_for_account",
    "is_region_mismatch",
    # --- log-compressed maturity ---------------------------------------------
    "log_account_history",
)

#: Columns that must never become features. Enforced by :func:`assert_no_leakage`.
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "is_fraud",
        "fraud_type",
        "label_role",
        "labelled_at",
        "labelled_by",
        "anomaly_score",
        "is_anomaly",
        "risk_score",
        "risk_level",
        "risk_reasons",
    }
)


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_columns: list[str]
    rows: int
    #: Rows dropped because they had no prior history to compute features from.
    warmup_rows_flagged: int = 0


def build_features(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame | None = None,
    locations: pd.DataFrame | None = None,
) -> FeatureBuildResult:
    """Build the model feature matrix from Silver transactions.

    ``accounts`` and ``locations`` are optional; when supplied they enable the
    geographic-mismatch feature, which needs each account's home region.
    """
    if transactions.empty:
        empty = pd.DataFrame(columns=[*FEATURE_COLUMNS, "transaction_id"])
        return FeatureBuildResult(empty, list(FEATURE_COLUMNS), 0)

    frame = transactions.copy()
    frame["transaction_timestamp"] = pd.to_datetime(
        frame["transaction_timestamp"], errors="coerce", utc=True
    )
    frame = frame.dropna(subset=["transaction_timestamp"])

    # Chronological order is a precondition for every cumulative feature below.
    # The transaction id breaks ties so the ordering is total and reproducible.
    frame = frame.sort_values(
        ["transaction_timestamp", "transaction_id"], kind="stable"
    ).reset_index(drop=True)

    amount = pd.to_numeric(frame.get("amount_usd", frame.get("amount")), errors="coerce").fillna(
        0.0
    )
    frame["amount_usd"] = amount.astype(float)
    frame["log_amount"] = np.log1p(frame["amount_usd"].clip(lower=0))

    # -- calendar features ---------------------------------------------------
    stamps = frame["transaction_timestamp"]
    frame["hour_of_day"] = stamps.dt.hour.astype(float)
    frame["day_of_week"] = stamps.dt.dayofweek.astype(float)
    frame["is_weekend"] = (stamps.dt.dayofweek >= 5).astype(float)
    frame["is_night"] = stamps.dt.hour.between(0, 4).astype(float)

    # -- account history (strictly prior) ------------------------------------
    frame = _add_prior_account_statistics(frame)

    # -- time-window features ------------------------------------------------
    # Epoch seconds are computed by subtracting the epoch and asking for
    # seconds, NOT by casting to int64 and dividing by 1e9. The cast returns the
    # column's *storage resolution* - nanoseconds on some frames, microseconds
    # on others under pandas 3 - so the divisor is only correct by luck. When it
    # was wrong the window bounds collapsed, every "trailing 5 minutes" count
    # silently became "every prior transaction", and the velocity rule fired on
    # accounts that had simply been around a long time.
    frame["_epoch_seconds"] = (stamps - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    frame["account_txn_count_24h"] = _prior_count_within(frame, "account_id", 86_400)
    frame["account_velocity_5min"] = _prior_count_within(frame, "account_id", 300)
    frame["seconds_since_account_last_txn"] = _seconds_since_previous(frame, "account_id")

    # -- vendor history (strictly prior) -------------------------------------
    frame["vendor_txn_count_prior"] = (
        frame.groupby("vendor_id", sort=False).cumcount().astype(float)
    )
    frame["vendor_txn_count_7d"] = _prior_count_within(frame, "vendor_id", 604_800)
    frame["account_unique_vendors_prior"] = _prior_distinct_count(frame, "account_id", "vendor_id")
    frame["vendor_unique_accounts_prior"] = _prior_distinct_count(frame, "vendor_id", "account_id")
    frame["is_new_vendor_for_account"] = _is_first_pairing(frame, "account_id", "vendor_id")

    # -- geography -----------------------------------------------------------
    frame["is_region_mismatch"] = _region_mismatch(frame, accounts, locations)

    # -- derived ratios ------------------------------------------------------
    avg = frame["account_avg_amount_prior"].replace(0, np.nan)
    frame["amount_to_account_avg"] = (frame["amount_usd"] / avg).fillna(1.0).clip(upper=1000)
    std = frame["account_std_amount_prior"].replace(0, np.nan)
    frame["amount_zscore_account"] = (
        ((frame["amount_usd"] - frame["account_avg_amount_prior"]) / std).fillna(0.0).clip(-50, 50)
    )

    # -- stationary transforms of unbounded history ---------------------------
    # Ratios, not running totals: "what share of this account's payments went to
    # distinct vendors" reads the same in month one and month twenty, whereas
    # "how many vendors has it ever paid" only ever increases.
    account_history = frame["account_txn_count_prior"]
    vendor_history = frame["vendor_txn_count_prior"]
    frame["account_vendor_diversity"] = (
        frame["account_unique_vendors_prior"] / account_history.clip(lower=1)
    ).clip(0, 1)
    frame["vendor_account_concentration"] = (
        frame["vendor_unique_accounts_prior"] / vendor_history.clip(lower=1)
    ).clip(0, 1)
    frame["log_account_history"] = np.log1p(account_history.clip(lower=0))
    # -1 marks an account's first transaction; log1p of the clipped gap keeps
    # that case distinct from "one second later" without an unbounded tail.
    frame["log_seconds_since_last_txn"] = np.where(
        frame["seconds_since_account_last_txn"] < 0,
        -1.0,
        np.log1p(frame["seconds_since_account_last_txn"].clip(lower=0)),
    )

    warmup = int((frame["account_txn_count_prior"] == 0).sum())
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
        frame[column] = frame[column].replace([np.inf, -np.inf], 0.0)

    frame = frame.drop(columns=["_epoch_seconds"], errors="ignore")
    assert_no_leakage(list(FEATURE_COLUMNS))

    log.info(
        "features_built",
        rows=len(frame),
        features=len(FEATURE_COLUMNS),
        warmup_rows=warmup,
    )
    return FeatureBuildResult(frame, list(FEATURE_COLUMNS), len(frame), warmup)


# ---------------------------------------------------------------------------
# prior-history primitives
# ---------------------------------------------------------------------------


def _add_prior_account_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Count, mean and standard deviation of an account's **previous** amounts.

    Computed with cumulative sums rather than ``expanding().mean().shift()``:
    same result, one pass, no Python-level loop per group.
    """
    grouped = frame.groupby("account_id", sort=False)
    amount = frame["amount_usd"]

    count_prior = grouped.cumcount().astype(float)
    sum_inclusive = grouped["amount_usd"].cumsum()
    sum_prior = sum_inclusive - amount

    squares = amount**2
    sum_sq_inclusive = squares.groupby(frame["account_id"], sort=False).cumsum()
    sum_sq_prior = sum_sq_inclusive - squares

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_prior = np.where(count_prior > 0, sum_prior / count_prior, 0.0)
        # Population variance of the prior rows; clipped at zero because
        # floating-point cancellation can produce tiny negatives.
        variance = np.where(
            count_prior > 1,
            np.maximum(sum_sq_prior / count_prior - mean_prior**2, 0.0),
            0.0,
        )

    frame["account_txn_count_prior"] = count_prior
    frame["account_avg_amount_prior"] = mean_prior
    frame["account_std_amount_prior"] = np.sqrt(variance)
    return frame


def _prior_count_within(frame: pd.DataFrame, key: str, window_seconds: int) -> pd.Series:
    """Number of previous rows for ``key`` within the trailing time window.

    Uses ``searchsorted`` over each group's (already sorted) timestamps, which
    is O(n log n) overall and, importantly, counts only rows *before* the current
    one - a centred or forward-looking window would leak the future.
    """
    result = np.zeros(len(frame), dtype=float)
    epoch = frame["_epoch_seconds"].to_numpy()

    for _, positions in frame.groupby(key, sort=False).indices.items():
        times = epoch[positions]
        # positions are ascending because the frame is sorted by time.
        starts = np.searchsorted(times, times - window_seconds, side="left")
        offsets = np.arange(len(times))
        result[positions] = (offsets - starts).astype(float)
    return pd.Series(result, index=frame.index)


def _seconds_since_previous(frame: pd.DataFrame, key: str) -> pd.Series:
    """Gap to the group's previous row; the first row of a group gets -1."""
    epoch = frame["_epoch_seconds"]
    previous = epoch.groupby(frame[key], sort=False).shift(1)
    return (epoch - previous).fillna(-1.0).astype(float)


def _prior_distinct_count(frame: pd.DataFrame, key: str, target: str) -> pd.Series:
    """Distinct ``target`` values seen for ``key`` before the current row.

    Implemented as "count the rows where this pairing appeared for the first
    time, cumulatively, excluding the current row" - which is a running distinct
    count without holding a set per group.
    """
    first_occurrence = ~frame.duplicated(subset=[key, target], keep="first")
    cumulative = first_occurrence.groupby(frame[key], sort=False).cumsum()
    return (cumulative - first_occurrence.astype(int)).astype(float)


def _is_first_pairing(frame: pd.DataFrame, key: str, target: str) -> pd.Series:
    """1.0 when this is the first time ``key`` has transacted with ``target``."""
    return (~frame.duplicated(subset=[key, target], keep="first")).astype(float)


def _region_mismatch(
    frame: pd.DataFrame, accounts: pd.DataFrame | None, locations: pd.DataFrame | None
) -> pd.Series:
    """1.0 when a transaction occurred outside its account's home region."""
    if (
        accounts is None
        or locations is None
        or accounts.empty
        or locations.empty
        or "location_id" not in frame.columns
    ):
        return pd.Series(0.0, index=frame.index)

    home = accounts.set_index("account_id")["region"] if "region" in accounts.columns else None
    where = locations.set_index("location_id")["region"] if "region" in locations.columns else None
    if home is None or where is None:
        return pd.Series(0.0, index=frame.index)

    account_region = frame["account_id"].map(home)
    location_region = frame["location_id"].map(where)
    mismatch = (
        account_region.notna()
        & location_region.notna()
        & (account_region.astype("string") != location_region.astype("string"))
    )
    return mismatch.astype(float)


# ---------------------------------------------------------------------------
# leakage guards and splitting
# ---------------------------------------------------------------------------


def assert_no_leakage(feature_columns: list[str]) -> None:
    """Fail loudly if a label or a model output is being used as a feature.

    A cheap assertion that catches the most expensive mistake in the project.
    """
    offenders = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if offenders:
        raise ValueError(
            "Target leakage: these columns must never be model features: " + ", ".join(offenders)
        )


@dataclass
class TemporalSplit:
    """Chronological train / validation / test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp | None
    validation_end: pd.Timestamp | None

    def summary(self) -> dict[str, object]:
        return {
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "test_rows": len(self.test),
            "train_end": str(self.train_end),
            "validation_end": str(self.validation_end),
        }


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    time_column: str = "transaction_timestamp",
) -> TemporalSplit:
    """Split chronologically - never randomly.

    The cut points are *timestamps*, not row positions, so that all transactions
    from the same instant land on the same side of the boundary. Splitting by
    row index could put two rows of the same second in different partitions,
    which is a small but real leak.
    """
    if frame.empty:
        empty = frame.copy()
        return TemporalSplit(empty, empty.copy(), empty.copy(), None, None)

    ordered = frame.sort_values(time_column, kind="stable")
    stamps = pd.to_datetime(ordered[time_column], utc=True)

    train_end = stamps.quantile(train_fraction)
    validation_end = stamps.quantile(train_fraction + validation_fraction)

    train = ordered[stamps <= train_end]
    validation = ordered[(stamps > train_end) & (stamps <= validation_end)]
    test = ordered[stamps > validation_end]

    log.info(
        "temporal_split",
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_end=str(train_end),
        validation_end=str(validation_end),
    )
    return TemporalSplit(train, validation, test, train_end, validation_end)
