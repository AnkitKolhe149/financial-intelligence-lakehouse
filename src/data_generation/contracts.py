"""Contract (award) generation.

Awards are the bridge between the agency side and the vendor side of the graph,
and they carry the oversight-relevant attributes: ceiling value, competition
type and period of performance.

Two realistic structures are built in deliberately:

* **Vendor concentration** - awards are drawn with a Zipf-like preference so a
  small number of vendors hold a large share of award value. Uniform sampling
  would make every concentration metric flat and the Gold layer boring.
* **Sole-source clusters** - a subset of agencies award disproportionately
  sole-source. That is a genuine audit signal, and it gives the SQL analytics
  and graph queries something real to surface.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.data_generation.base import GeneratorContext

CONTRACT_TYPES = (
    "FIXED_PRICE",
    "FIXED_PRICE",
    "COST_PLUS",
    "TIME_AND_MATERIALS",
    "IDIQ",
    "GRANT",
    "PURCHASE_ORDER",
)
COMPETITION_TYPES = ("FULL_AND_OPEN", "FULL_AND_OPEN", "SET_ASIDE", "LIMITED", "SOLE_SOURCE")
STATUSES = ("ACTIVE", "ACTIVE", "ACTIVE", "COMPLETED", "PENDING", "TERMINATED")

DESCRIPTIONS = (
    "IT modernisation and systems integration services",
    "Facility maintenance and grounds keeping",
    "Professional advisory and management support",
    "Medical supplies and equipment procurement",
    "Research and development support services",
    "Construction and renovation of federal property",
    "Logistics, warehousing and distribution",
    "Cybersecurity monitoring and incident response",
    "Training delivery and curriculum development",
    "Telecommunications infrastructure support",
)


def generate_contracts(
    ctx: GeneratorContext,
    count: int,
    vendors: pd.DataFrame,
    agencies: pd.DataFrame,
) -> pd.DataFrame:
    """Generate ``count`` awards linking vendors to agencies."""
    vendor_ids = vendors["vendor_id"].tolist()
    agency_ids = agencies["agency_id"].tolist()

    # Concentration: vendor selection weight decays with rank, so the first
    # vendors win far more awards than the last. Normalised to a probability.
    ranks = ctx.rng.permutation(len(vendor_ids)) + 1
    weights = 1.0 / (ranks**0.85)
    weights = weights / weights.sum()

    # A fifth of agencies are "sole-source heavy" - the oversight red flag.
    sole_source_agencies = set(
        ctx.rng.choice(agency_ids, size=max(1, len(agency_ids) // 5), replace=False).tolist()
    )

    vendor_choice = ctx.rng.choice(len(vendor_ids), size=count, p=weights)
    agency_choice = ctx.rng.choice(len(agency_ids), size=count)
    start_date = ctx.settings.generation.start_date
    end_date = ctx.settings.generation.end_date
    span_days = max((end_date - start_date).days, 1)

    rows: list[dict] = []
    for index in range(count):
        agency_id = agency_ids[int(agency_choice[index])]
        offset = int(ctx.rng.integers(-540, span_days))
        contract_start = start_date + timedelta(days=offset)
        duration = int(ctx.rng.integers(90, 1460))
        contract_end = contract_start + timedelta(days=duration)

        if agency_id in sole_source_agencies:
            competition = str(
                ctx.choice(("SOLE_SOURCE", "SOLE_SOURCE", "LIMITED", "FULL_AND_OPEN"))
            )
        else:
            competition = str(ctx.choice(COMPETITION_TYPES))

        value = float(round(ctx.rng.lognormal(12.6, 1.5), 2))

        rows.append(
            {
                "contract_id": f"CON-{index + 1:07d}",
                "award_id": f"{ctx.rng.integers(1000, 9999)}{chr(65 + int(ctx.rng.integers(0, 26)))}"
                f"{ctx.rng.integers(100000, 999999)}",
                "vendor_id": vendor_ids[int(vendor_choice[index])],
                "agency_id": agency_id,
                "contract_type": str(ctx.choice(CONTRACT_TYPES)),
                "contract_value_usd": value,
                "start_date": contract_start.isoformat(),
                "end_date": contract_end.isoformat(),
                "status": str(ctx.choice(STATUSES)),
                "description": str(ctx.choice(DESCRIPTIONS)),
                "competition_type": competition,
            }
        )

    return pd.DataFrame(rows)
