"""Transaction fact generation - realistic patterns, then deliberate defects.

This is the largest generator and the only one written for throughput: at the
``large`` profile it produces 500,000 rows, so everything is vectorised with
NumPy and assembled once into a DataFrame. Per-row Python loops are used only
where the cardinality is small (per-vendor contract assignment loops over
vendors, not transactions).

Three things are modelled that a naive generator would miss, and each exists to
make a downstream component meaningful rather than decorative:

**Per-account spending scale**
    Each account draws its own amount distribution. Without this, "amount is
    12x the account average" is the same statement for every account and the
    risk rule is trivial.

**Vendor affinity**
    An account pays the same handful of vendors most of the time. That is what
    makes a first-time vendor a genuine signal instead of a coin flip, and it
    creates the shared-vendor structure the graph queries traverse.

**Fiscal-year-end surge**
    US federal spending spikes in September as agencies obligate remaining
    budget. This gives the volume check a legitimate seasonal spike that it must
    *not* flag, which is a far better test than uniform data.

Separately, a small cohort of transactions is generated with genuinely
suspicious behaviour and recorded in a **ground-truth label set** written to its
own file (``data/raw/labels/``). Labels are kept out of the transaction feed on
purpose - they represent investigation outcomes that arrive later, they must
never leak into features, and holding them separately is what allows honest
precision@k evaluation of an unsupervised model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.data_generation.base import BAD_CURRENCIES, MERCHANT_CATEGORIES, GeneratorContext

TRANSACTION_TYPES = (
    "PAYMENT",
    "PAYMENT",
    "PAYMENT",
    "PAYMENT",
    "GRANT_DISBURSEMENT",
    "PAYROLL",
    "TRANSFER",
    "REFUND",
    "ADJUSTMENT",
)
PAYMENT_METHODS = ("ACH", "ACH", "ACH", "WIRE", "CARD", "CHECK", "EFT")
CHANNELS = ("BATCH", "BATCH", "ONLINE", "API", "BRANCH", "MOBILE")
STATUSES = ("SETTLED", "SETTLED", "SETTLED", "SETTLED", "SETTLED", "PENDING", "REVERSED", "FAILED")

DESCRIPTIONS = (
    "Monthly services invoice",
    "Equipment purchase order",
    "Consulting engagement fee",
    "Facility maintenance payment",
    "Software licence renewal",
    "Travel reimbursement",
    "Research grant disbursement",
    "Construction milestone payment",
    "Medical supplies order",
    "Training delivery invoice",
    "Telecom services",
    "Fuel and logistics",
    "Emergency response supplies",
    "Cloud hosting charges",
    "Security services invoice",
)

#: Fraud archetypes injected into the population. The names are reused verbatim
#: in the ground-truth label file and in the ML evaluation report.
FRAUD_TYPES = (
    "AMOUNT_SPIKE",
    "VELOCITY_BURST",
    "AFTER_HOURS_NEW_VENDOR",
    "GEOGRAPHIC_ANOMALY",
    "ROUND_AMOUNT_STRUCTURING",
)

#: Share of the transaction population that is genuinely suspicious. Kept close
#: to a realistic base rate - fraud is rare, and a model tuned on 10% fraud
#: would be useless.
SUSPICIOUS_RATE = 0.004


@dataclass
class TransactionGenerationResult:
    """Transactions plus the separately-held investigation labels."""

    transactions: pd.DataFrame
    labels: pd.DataFrame

    @property
    def fraud_count(self) -> int:
        return int(self.labels["is_fraud"].sum()) if len(self.labels) else 0


def generate_transactions(
    ctx: GeneratorContext,
    count: int,
    accounts: pd.DataFrame,
    vendors: pd.DataFrame,
    contracts: pd.DataFrame,
    departments: pd.DataFrame,
    locations: pd.DataFrame,
) -> TransactionGenerationResult:
    """Generate ``count`` transactions with realistic structure and known defects."""
    rng = ctx.rng
    n_accounts = len(accounts)
    n_vendors = len(vendors)
    if n_accounts == 0 or n_vendors == 0:
        raise ValueError("Accounts and vendors are required before generating transactions")

    account_ids = accounts["account_id"].to_numpy()
    vendor_ids = vendors["vendor_id"].to_numpy()
    location_ids = locations["location_id"].to_numpy()

    # ---- account activity concentration ----------------------------------
    # A minority of accounts carry most of the volume, as in real disbursement
    # data. Without concentration, per-account statistics are all identical and
    # velocity analysis has nothing to find.
    activity_weights = rng.lognormal(0.0, 1.1, n_accounts)
    activity_weights /= activity_weights.sum()
    account_idx = rng.choice(n_accounts, size=count, p=activity_weights)

    # ---- per-account amount scale ----------------------------------------
    type_scale = (
        accounts["account_type"]
        .map(
            {
                "PAYROLL": 1.9,
                "CAPITAL": 2.1,
                "PROCUREMENT": 1.4,
                "GRANT": 1.6,
                "OPERATING": 1.0,
                "TRAVEL": 0.35,
            }
        )
        .fillna(1.0)
        .to_numpy(dtype=float)
    )
    account_mu = np.log(np.maximum(type_scale, 0.05)) + rng.normal(6.2, 0.55, n_accounts)
    account_sigma = rng.uniform(0.55, 1.15, n_accounts)

    amounts = np.round(rng.lognormal(account_mu[account_idx], account_sigma[account_idx]), 2)
    amounts = np.clip(amounts, 1.0, 9_000_000.0)

    # ---- vendor affinity ---------------------------------------------------
    affinity_slots = 8
    usual_vendors = rng.integers(0, n_vendors, size=(n_accounts, affinity_slots))
    slot = rng.integers(0, affinity_slots, size=count)
    uses_usual = rng.random(count) < 0.85
    vendor_idx = np.where(
        uses_usual,
        usual_vendors[account_idx, slot],
        rng.integers(0, n_vendors, size=count),
    )

    # ---- timestamps --------------------------------------------------------
    timestamps = _generate_timestamps(ctx, count)

    # ---- geography ---------------------------------------------------------
    # 90% of spend happens in the account's home region; the rest is legitimate
    # travel. The risk rule must live with that false-positive floor.
    region_of_location = locations["region"].to_numpy()
    account_region = accounts["region"].to_numpy()
    location_idx = _choose_locations(
        rng, account_idx, account_region, region_of_location, count, len(location_ids)
    )

    # ---- assemble the base frame ------------------------------------------
    frame = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:012d}" for i in range(1, count + 1)],
            "account_id": account_ids[account_idx],
            "vendor_id": vendor_ids[vendor_idx],
            "amount": amounts,
            "currency": _choose_currencies(ctx, count),
            "transaction_timestamp": timestamps,
            "transaction_type": np.array(TRANSACTION_TYPES)[
                rng.integers(0, len(TRANSACTION_TYPES), count)
            ],
            "payment_method": np.array(PAYMENT_METHODS)[
                rng.integers(0, len(PAYMENT_METHODS), count)
            ],
            "merchant_category": np.array(MERCHANT_CATEGORIES)[
                rng.integers(0, len(MERCHANT_CATEGORIES), count)
            ],
            "description": np.array(DESCRIPTIONS)[rng.integers(0, len(DESCRIPTIONS), count)],
            "location_id": location_ids[location_idx],
            "channel": np.array(CHANNELS)[rng.integers(0, len(CHANNELS), count)],
            "status": np.array(STATUSES)[rng.integers(0, len(STATUSES), count)],
        }
    )

    # ---- dimensional roll-ups ---------------------------------------------
    account_lookup = accounts[["account_id", "department_id"]]
    department_lookup = departments[["department_id", "agency_id"]]
    frame = frame.merge(account_lookup, on="account_id", how="left")
    frame = frame.merge(department_lookup, on="department_id", how="left")

    # ---- contract assignment ----------------------------------------------
    frame["contract_id"] = _assign_contracts(ctx, frame["vendor_id"], contracts)

    # ---- suspicious cohort + ground truth ---------------------------------
    frame, labels = _inject_suspicious_activity(
        ctx, frame, accounts, account_idx, account_mu, locations
    )

    # ---- deliberate data-quality defects -----------------------------------
    frame = _inject_defects(ctx, frame)

    # ---- posted timestamp (derived after defects so it inherits lateness) --
    frame = _add_posted_timestamp(ctx, frame)

    column_order = [
        "transaction_id",
        "account_id",
        "vendor_id",
        "contract_id",
        "department_id",
        "agency_id",
        "amount",
        "currency",
        "transaction_timestamp",
        "posted_timestamp",
        "transaction_type",
        "payment_method",
        "merchant_category",
        "description",
        "location_id",
        "channel",
        "status",
    ]
    frame = frame[column_order]
    return TransactionGenerationResult(transactions=frame, labels=labels)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _generate_timestamps(ctx: GeneratorContext, count: int) -> np.ndarray:
    """Vectorised timestamp generation with weekday, business-hour and
    fiscal-year-end structure. Returns numpy datetime64[s]."""
    rng = ctx.rng
    start = pd.Timestamp(ctx.settings.generation.start_date)
    end = pd.Timestamp(ctx.settings.generation.end_date)
    total_days = max(int((end - start).days), 1)

    day_offset = rng.integers(0, total_days, size=count)
    dates = start + pd.to_timedelta(day_offset, unit="D")

    # Weekday bias: push 85% of weekend activity onto the preceding Friday.
    weekday = dates.dayofweek.to_numpy()
    is_weekend = weekday >= 5
    shift_back = is_weekend & (rng.random(count) < 0.85)
    dates = dates - pd.to_timedelta(np.where(shift_back, weekday - 4, 0), unit="D")

    # Fiscal-year-end surge: 8% of activity is re-dated into September, the
    # month US agencies obligate remaining budget authority.
    surge = rng.random(count) < 0.08
    if surge.any():
        surge_years = dates.year.to_numpy()[surge]
        surge_days = rng.integers(1, 31, size=int(surge.sum()))
        surge_dates = pd.to_datetime(
            {"year": surge_years, "month": np.full(len(surge_years), 9), "day": surge_days}
        )
        # A September date that lands beyond the configured end is moved to the
        # *previous* September rather than clamped to the end date. Clamping
        # piled every out-of-range surge row onto the final day, creating a
        # 20x artificial spike that the volume quality check then dutifully
        # reported - a generator artefact masquerading as a finding.
        overshoot = surge_dates > end
        if overshoot.any():
            surge_dates = surge_dates.where(~overshoot, surge_dates - pd.DateOffset(years=1))
        dates = dates.to_series(index=range(count))
        dates.loc[surge] = surge_dates.to_numpy()
        dates = pd.DatetimeIndex(dates)

    # Business hours with an 8% overnight tail.
    off_hours = rng.random(count) < 0.08
    hours = np.where(off_hours, rng.integers(0, 5, count), rng.integers(8, 19, count))
    minutes = rng.integers(0, 60, count)
    seconds = rng.integers(0, 60, count)

    stamps = (
        dates.normalize()
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )
    # Clamp anything the September re-dating pushed past the configured end.
    stamps = stamps.where(stamps <= end, end)
    return stamps.to_numpy(dtype="datetime64[s]")


def _choose_locations(
    rng: np.random.Generator,
    account_idx: np.ndarray,
    account_region: np.ndarray,
    location_region: np.ndarray,
    count: int,
    n_locations: int,
) -> np.ndarray:
    """Pick a location, preferring the account's home region."""
    # Index locations by region once, then draw within the region.
    region_to_locations: dict[str, np.ndarray] = {}
    for region in np.unique(location_region):
        region_to_locations[str(region)] = np.flatnonzero(location_region == region)

    home_regions = account_region[account_idx]
    result = rng.integers(0, n_locations, size=count)
    stay_home = rng.random(count) < 0.90

    for region, candidates in region_to_locations.items():
        if len(candidates) == 0:
            continue
        selector = stay_home & (home_regions == region)
        hits = int(selector.sum())
        if hits:
            result[selector] = candidates[rng.integers(0, len(candidates), size=hits)]
    return result


def _choose_currencies(ctx: GeneratorContext, count: int) -> np.ndarray:
    """Weighted currency draw. USD dominates, as it must for federal spending."""
    currencies = ctx.settings.generation.currencies
    idx = ctx.rng.integers(0, len(currencies), size=count)
    return np.array(currencies)[idx]


def _assign_contracts(
    ctx: GeneratorContext, vendor_series: pd.Series, contracts: pd.DataFrame
) -> pd.Series:
    """Attach a contract belonging to the paying vendor, where one exists.

    Loops over vendors (thousands) rather than transactions (up to a million),
    assigning each vendor's slice in one vectorised draw.
    """
    result = pd.Series(pd.NA, index=vendor_series.index, dtype="object")
    if contracts.empty:
        return result

    grouped = contracts.groupby("vendor_id")["contract_id"].apply(np.array)
    # ~65% of spend flows under a contract; travel and petty cash do not.
    has_contract = ctx.rng.random(len(vendor_series)) < 0.65

    for vendor_id, positions in vendor_series.groupby(vendor_series).groups.items():
        options = grouped.get(vendor_id)
        if options is None or len(options) == 0:
            continue
        idx = pd.Index(positions)
        mask = has_contract[vendor_series.index.get_indexer(idx)]
        selected_positions = idx[mask]
        if len(selected_positions) == 0:
            continue
        picks = options[ctx.rng.integers(0, len(options), size=len(selected_positions))]
        result.loc[selected_positions] = picks
    return result


def _inject_suspicious_activity(
    ctx: GeneratorContext,
    frame: pd.DataFrame,
    accounts: pd.DataFrame,
    account_idx: np.ndarray,
    account_mu: np.ndarray,
    locations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overwrite a small cohort with genuinely suspicious behaviour.

    Returns the modified frame and a ground-truth label frame. The label frame
    is the *only* record of which transactions are fraudulent; nothing in the
    transaction feed marks them, because a real platform would not know.
    """
    rng = ctx.rng
    count = len(frame)
    n_suspicious = max(int(count * SUSPICIOUS_RATE), 5 if count >= 100 else 1)
    victims = rng.choice(count, size=min(n_suspicious, count), replace=False)

    labels: list[dict] = []
    # Work on numpy views and write back once - avoids chained assignment,
    # which pandas 3's copy-on-write semantics would silently drop.
    amounts = frame["amount"].to_numpy(dtype=float).copy()
    timestamps = frame["transaction_timestamp"].to_numpy(dtype="datetime64[s]").copy()
    vendor_col = frame["vendor_id"].to_numpy(dtype=object).copy()
    location_col = frame["location_id"].to_numpy(dtype=object).copy()

    all_location_ids = locations["location_id"].to_numpy()
    account_home = dict(zip(accounts["account_id"], accounts["region"], strict=False))
    location_region = dict(zip(locations["location_id"], locations["region"], strict=False))
    foreign_locations = {
        region: locations.loc[locations["region"] != region, "location_id"].to_numpy()
        for region in locations["region"].unique()
    }

    fraud_type_choice = rng.integers(0, len(FRAUD_TYPES), size=len(victims))

    for position, victim in enumerate(victims):
        fraud_type = FRAUD_TYPES[int(fraud_type_choice[position])]
        account_id = frame.at[victim, "account_id"]

        if fraud_type == "AMOUNT_SPIKE":
            # 12x-45x the account's own typical amount.
            multiple = float(rng.uniform(12, 45))
            amounts[victim] = round(float(np.exp(account_mu[account_idx[victim]])) * multiple, 2)

        elif fraud_type == "VELOCITY_BURST":
            # Re-time a nearby run of this account's transactions into a
            # five-minute window. The burst is the signal, not any one row.
            same_account = np.flatnonzero(frame["account_id"].to_numpy() == account_id)
            burst = same_account[:8] if len(same_account) >= 4 else same_account
            if len(burst) >= 2:
                anchor = timestamps[victim]
                offsets = rng.integers(0, 300, size=len(burst)).astype("timedelta64[s]")
                timestamps[burst] = anchor + offsets
                for member in burst:
                    if member != victim:
                        labels.append(
                            _label(frame.at[member, "transaction_id"], fraud_type, "burst_member")
                        )

        elif fraud_type == "AFTER_HOURS_NEW_VENDOR":
            # 02:00-04:00 payment to a vendor this account has never used.
            stamp = pd.Timestamp(timestamps[victim])
            timestamps[victim] = np.datetime64(
                stamp.normalize() + pd.Timedelta(hours=int(rng.integers(2, 5))), "s"
            )
            vendor_col[victim] = frame["vendor_id"].iloc[int(rng.integers(0, count))]
            amounts[victim] = round(amounts[victim] * float(rng.uniform(3, 9)), 2)

        elif fraud_type == "GEOGRAPHIC_ANOMALY":
            home = account_home.get(account_id)
            candidates = foreign_locations.get(home, all_location_ids)
            if len(candidates):
                location_col[victim] = candidates[int(rng.integers(0, len(candidates)))]
            amounts[victim] = round(amounts[victim] * float(rng.uniform(4, 12)), 2)

        else:  # ROUND_AMOUNT_STRUCTURING
            # Repeated suspiciously round amounts just under a review threshold.
            amounts[victim] = float(rng.choice([9500.0, 9800.0, 9900.0, 24500.0, 49000.0]))

        labels.append(_label(frame.at[victim, "transaction_id"], fraud_type, "primary"))

    frame["amount"] = amounts
    frame["transaction_timestamp"] = timestamps
    frame["vendor_id"] = vendor_col
    frame["location_id"] = location_col

    label_frame = pd.DataFrame(labels).drop_duplicates(subset=["transaction_id"])
    if label_frame.empty:
        label_frame = pd.DataFrame(
            columns=[
                "transaction_id",
                "is_fraud",
                "fraud_type",
                "label_role",
                "labelled_at",
                "labelled_by",
            ]
        )
    # Unused variable guard: location_region is kept for readability of intent.
    del location_region
    return frame, label_frame


def _label(transaction_id: str, fraud_type: str, role: str) -> dict:
    return {
        "transaction_id": transaction_id,
        "is_fraud": True,
        "fraud_type": fraud_type,
        "label_role": role,
        # Investigations conclude after the fact; the date reflects that.
        "labelled_at": datetime.now(UTC).date().isoformat(),
        "labelled_by": "OFFICE_OF_INSPECTOR_GENERAL",
    }


def _inject_defects(ctx: GeneratorContext, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the configured data-quality defects.

    Timestamps become **strings** here, because every source format the platform
    reads (CSV, JSON, XML) is textual - which is precisely why unparseable
    timestamps are possible at all. Typing them is Silver's job.
    """
    injection = ctx.injection
    frame = frame.copy()
    frame["transaction_timestamp"] = pd.to_datetime(
        frame["transaction_timestamp"], utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")

    if not injection or not injection.enabled:
        return frame

    n = len(frame)
    rng = ctx.rng

    def apply(mask: np.ndarray, column: str, value) -> None:
        if mask.any():
            frame.loc[mask, column] = value

    apply(ctx.mask(n, injection.null_account_rate), "account_id", None)
    apply(ctx.mask(n, injection.null_vendor_rate), "vendor_id", None)
    apply(ctx.mask(n, injection.null_department_rate), "department_id", None)

    # Invalid amounts: negatives and zeroes, which must never reach Gold.
    invalid_amount = ctx.mask(n, injection.invalid_amount_rate)
    if invalid_amount.any():
        hits = int(invalid_amount.sum())
        frame.loc[invalid_amount, "amount"] = np.where(
            rng.random(hits) < 0.7, -np.abs(rng.uniform(10, 5000, hits)).round(2), 0.0
        )

    # Implausibly large amounts - above the platform ceiling, so they are
    # rejected by the contract rather than skewing every aggregate.
    extreme = ctx.mask(n, injection.extreme_amount_rate)
    if extreme.any():
        frame.loc[extreme, "amount"] = np.round(rng.uniform(6e7, 9e8, int(extreme.sum())), 2)

    bad_currency = ctx.mask(n, injection.invalid_currency_rate)
    if bad_currency.any():
        picks = rng.integers(0, len(BAD_CURRENCIES), int(bad_currency.sum()))
        frame.loc[bad_currency, "currency"] = np.array(BAD_CURRENCIES)[picks]

    # Orphans: ids that are well-formed but absent from the dimensions.
    orphan_account = ctx.mask(n, injection.orphan_account_rate)
    if orphan_account.any():
        frame.loc[orphan_account, "account_id"] = [
            f"ACC-9{rng.integers(100000, 999999)}" for _ in range(int(orphan_account.sum()))
        ]
    orphan_vendor = ctx.mask(n, injection.orphan_vendor_rate)
    if orphan_vendor.any():
        frame.loc[orphan_vendor, "vendor_id"] = [
            f"VEN-9{rng.integers(10000, 99999)}" for _ in range(int(orphan_vendor.sum()))
        ]
    bad_contract = ctx.mask(n, injection.invalid_contract_rate)
    if bad_contract.any():
        frame.loc[bad_contract, "contract_id"] = "CON-0000000"

    # Timestamp defects.
    future = ctx.mask(n, injection.future_timestamp_rate)
    if future.any():
        horizon = datetime.now(UTC) + timedelta(days=int(rng.integers(2, 400)))
        frame.loc[future, "transaction_timestamp"] = horizon.strftime("%Y-%m-%dT%H:%M:%S")
    unparseable = ctx.mask(n, injection.invalid_timestamp_rate)
    if unparseable.any():
        garbage = np.array(["not-a-date", "31/13/2026", "0000-00-00", "", "TBD"])
        frame.loc[unparseable, "transaction_timestamp"] = garbage[
            rng.integers(0, len(garbage), int(unparseable.sum()))
        ]

    # Whitespace and casing noise on free-text and coded columns.
    whitespace = ctx.mask(n, injection.whitespace_rate)
    if whitespace.any():
        frame.loc[whitespace, "description"] = [
            ctx.dirty_string(str(value)) for value in frame.loc[whitespace, "description"]
        ]
        frame.loc[whitespace, "currency"] = [
            f" {str(value).lower()} " for value in frame.loc[whitespace, "currency"]
        ]

    # Duplicates, appended last so they carry whatever defects their source row
    # already has. Half are byte-identical re-sends; half are the harder case -
    # a new transaction_id wrapping identical business content.
    if injection.duplicate_rate > 0:
        n_duplicates = int(n * injection.duplicate_rate)
        if n_duplicates > 0:
            victims = frame.sample(n=min(n_duplicates, n), random_state=ctx.seed)
            exact = victims.iloc[: len(victims) // 2].copy()
            reissued = victims.iloc[len(victims) // 2 :].copy()
            reissued["transaction_id"] = [
                f"TXN-R{rng.integers(10**10, 10**11)}" for _ in range(len(reissued))
            ]
            frame = pd.concat([frame, exact, reissued], ignore_index=True)

    return frame


def _add_posted_timestamp(ctx: GeneratorContext, frame: pd.DataFrame) -> pd.DataFrame:
    """Posting lag, including a late-arriving tail.

    Most records post within hours; a small share posts days later. That gap is
    what the late-arrival detection in Silver measures, and it is why Silver
    keys on event time rather than processing time.
    """
    parsed = pd.to_datetime(
        frame["transaction_timestamp"], errors="coerce", utc=True, format="mixed"
    )
    n = len(frame)
    lag_hours = ctx.rng.gamma(shape=2.0, scale=3.0, size=n)
    late = ctx.rng.random(n) < 0.03
    lag_hours = np.where(late, lag_hours + ctx.rng.uniform(48, 240, n), lag_hours)
    posted = parsed + pd.to_timedelta(lag_hours, unit="h")
    frame["posted_timestamp"] = posted.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return frame
