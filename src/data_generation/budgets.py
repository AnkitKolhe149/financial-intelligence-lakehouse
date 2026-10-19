"""Budget lines by fiscal year, agency, department and category.

Budgets are generated **after** transactions, from the spend that actually
occurred. That ordering is deliberate and is the difference between a working
metric and a decorative one.

Generating budgets independently - a lognormal draw anchored to nothing - made
allocations three orders of magnitude larger than the transactions that were
supposed to consume them. Every utilisation figure came out at 0.08%, no
department was ever over budget, and the entire "budget overrun" branch of the
Gold layer was dead code exercising no real case.

Deriving allocations from observed spend, then dividing by a target utilisation
ratio drawn per department, produces the intended distribution: most departments
land between 35% and 98% utilised, and a configured minority genuinely overrun.
The overrun detection in ``gold_department_spending`` therefore has real cases to
find, which is the only way to know it works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.base import GeneratorContext

CATEGORIES = ("PERSONNEL", "OPERATIONS", "CAPITAL", "GRANTS", "IT", "TRAVEL", "MAINTENANCE")
QUARTERS = ("Q1", "Q2", "Q3", "Q4")

#: Share of departments generated as over-spent.
OVERSPEND_FRACTION = 0.12


def generate_budgets(
    ctx: GeneratorContext,
    departments: pd.DataFrame,
    fiscal_years: tuple[int, ...],
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One budget line per department, category and fiscal year.

    ``transactions`` is optional: when supplied, allocations are anchored to
    real spend per department and fiscal year. Without it the generator falls
    back to the department's nominal allocation, which keeps the function usable
    in isolation (and in tests) at the cost of unrealistic utilisation.
    """
    if departments.empty:
        return pd.DataFrame(
            columns=[
                "budget_id",
                "fiscal_year",
                "agency_id",
                "department_id",
                "budget_category",
                "allocated_amount_usd",
                "spent_amount_usd",
                "quarter",
            ]
        )

    spend_lookup = _spend_by_department_year(transactions)

    # Chosen once, so overspending is a persistent property of a department
    # rather than noise that changes per row.
    overspending = set(
        departments["department_id"].sample(frac=OVERSPEND_FRACTION, random_state=ctx.seed).tolist()
    )

    rows: list[dict] = []
    counter = 0

    for department in departments.itertuples(index=False):
        nominal = float(department.budget_allocation_usd)
        for fiscal_year in fiscal_years:
            observed_spend = spend_lookup.get((department.department_id, int(fiscal_year)))

            if department.department_id in overspending:
                utilisation = float(ctx.rng.uniform(1.02, 1.45))
            else:
                utilisation = float(ctx.rng.uniform(0.35, 0.98))

            if observed_spend is not None and observed_spend > 0:
                # Allocation is the spend implied by the target utilisation.
                total_allocation = observed_spend / utilisation
            else:
                # No transactions for this department-year (a real occurrence:
                # not every department spends every year). Fall back to the
                # nominal figure and record zero spend.
                total_allocation = nominal * float(ctx.rng.uniform(0.02, 0.08))
                observed_spend = total_allocation * utilisation

            # Split the year's allocation across categories.
            shares = ctx.rng.dirichlet(np.ones(len(CATEGORIES)) * 2.0)
            for category, share in zip(CATEGORIES, shares, strict=True):
                counter += 1
                allocated = round(total_allocation * float(share), 2)
                rows.append(
                    {
                        "budget_id": f"BUD-{counter:07d}",
                        "fiscal_year": int(fiscal_year),
                        "agency_id": department.agency_id,
                        "department_id": department.department_id,
                        "budget_category": category,
                        "allocated_amount_usd": allocated,
                        "spent_amount_usd": round(allocated * utilisation, 2),
                        "quarter": str(ctx.choice(QUARTERS)),
                    }
                )

    return pd.DataFrame(rows)


def _spend_by_department_year(
    transactions: pd.DataFrame | None,
) -> dict[tuple[str, int], float]:
    """Actual spend per (department, fiscal year), from the generated fact data.

    Reads the *pre-defect* transaction frame, so allocations are anchored to
    what was genuinely spent rather than to whatever survived the injected
    nulls and invalid amounts.
    """
    if transactions is None or transactions.empty:
        return {}
    if not {"department_id", "amount", "transaction_timestamp"} <= set(transactions.columns):
        return {}

    frame = transactions[["department_id", "amount", "transaction_timestamp"]].copy()
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    stamps = pd.to_datetime(
        frame["transaction_timestamp"], errors="coerce", utc=True, format="mixed"
    )
    # US federal fiscal year starts on 1 October.
    frame["fiscal_year"] = (stamps.dt.year + (stamps.dt.month >= 10).astype(int)).astype("Int64")
    frame = frame.dropna(subset=["department_id", "amount", "fiscal_year"])
    frame = frame.loc[frame["amount"] > 0]

    grouped = frame.groupby(["department_id", "fiscal_year"])["amount"].sum()
    return {
        (str(department), int(year)): float(total) for (department, year), total in grouped.items()
    }
