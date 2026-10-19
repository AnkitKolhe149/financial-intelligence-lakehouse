"""Account dimension.

Each account belongs to a department and carries a **home region**. The home
region is not decoration: the geographic-anomaly risk rule compares a
transaction's location region against its account's home region, so the
dimension has to establish what "normal" looks like before the fact table can
deviate from it.
"""

from __future__ import annotations

import pandas as pd

from src.data_generation.base import GeneratorContext

ACCOUNT_TYPES = (
    "OPERATING",
    "OPERATING",
    "PROCUREMENT",
    "PROCUREMENT",
    "GRANT",
    "PAYROLL",
    "CAPITAL",
    "TRAVEL",
)
STATUSES = ("ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "DORMANT", "FROZEN", "CLOSED")
REGIONS = ("NORTHEAST", "SOUTHEAST", "MIDWEST", "SOUTHWEST", "MOUNTAIN", "WEST")


def generate_accounts(ctx: GeneratorContext, count: int, departments: pd.DataFrame) -> pd.DataFrame:
    """Generate ``count`` accounts distributed across departments."""
    department_ids = departments["department_id"].tolist()
    if not department_ids:
        raise ValueError("Cannot generate accounts without departments")

    department_choice = ctx.rng.choice(len(department_ids), size=count)
    rows: list[dict] = []

    for index in range(count):
        account_type = str(ctx.choice(ACCOUNT_TYPES))
        # Credit limits scale with account type: a payroll account legitimately
        # moves far more money than a travel account, and the risk engine must
        # not treat that as anomalous.
        base_limit = {
            "PAYROLL": 5_000_000,
            "CAPITAL": 8_000_000,
            "PROCUREMENT": 3_000_000,
            "GRANT": 4_000_000,
            "OPERATING": 1_000_000,
            "TRAVEL": 150_000,
        }.get(account_type, 1_000_000)
        limit = float(round(base_limit * ctx.rng.uniform(0.4, 2.5), 2))

        rows.append(
            {
                "account_id": f"ACC-{index + 1:07d}",
                # Masked from the start - the platform never holds a full number.
                "account_number": f"****{ctx.rng.integers(1000, 9999)}",
                "department_id": department_ids[int(department_choice[index])],
                "account_type": account_type,
                "currency": "USD",
                "opened_date": ctx.faker.date_between(
                    start_date="-12y", end_date="-60d"
                ).isoformat(),
                "status": str(ctx.choice(STATUSES)),
                "credit_limit_usd": limit,
                "owner_name": ctx.faker.name(),
                "region": str(ctx.choice(REGIONS)),
            }
        )

    return pd.DataFrame(rows)
