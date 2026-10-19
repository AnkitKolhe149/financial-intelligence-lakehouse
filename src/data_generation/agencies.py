"""Agency, department and location dimensions.

These three are generated together because departments are meaningless without
their agency and both are anchored to a location. Every id is zero-padded and
prefixed (``AGY-0007``) so that identifiers sort naturally and are visually
distinguishable in logs and quarantine rows.
"""

from __future__ import annotations

import pandas as pd

from src.data_generation.base import (
    AGENCY_NAMES,
    CITIES,
    DEPARTMENT_FUNCTIONS,
    GeneratorContext,
)

AGENCY_TYPES = ("EXECUTIVE", "EXECUTIVE", "EXECUTIVE", "INDEPENDENT", "LEGISLATIVE", "JUDICIAL")


def generate_locations(ctx: GeneratorContext) -> pd.DataFrame:
    """The location dimension - a fixed catalogue, not random noise.

    Locations are a closed set so that "the same city" is genuinely the same
    row, which is what makes the geographic-anomaly risk rule and the graph's
    ``OCCURRED_AT`` edges meaningful.
    """
    rows = [
        {
            "location_id": f"LOC-{index:04d}",
            "city": city,
            "state": state,
            "country": "US",
            "latitude": lat,
            "longitude": lon,
            "region": region,
        }
        for index, (city, state, lat, lon, region) in enumerate(CITIES, start=1)
    ]
    return pd.DataFrame(rows)


def generate_agencies(ctx: GeneratorContext, count: int) -> pd.DataFrame:
    """Generate ``count`` agencies.

    Real agency names are used for the first 30 (they are public information and
    make the dataset readable); beyond that, synthetic bureau names are minted.
    """
    locations = CITIES
    rows: list[dict] = []
    for index in range(1, count + 1):
        if index <= len(AGENCY_NAMES):
            name, abbreviation = AGENCY_NAMES[index - 1]
        else:
            noun = ctx.faker.word().capitalize()
            name = f"Bureau of {noun} Affairs"
            abbreviation = f"B{noun[:2].upper()}{index:02d}"

        _city, state, *_ = locations[int(ctx.rng.integers(0, len(locations)))]
        # Budget authority is heavy-tailed: a handful of agencies dominate.
        budget = float(round(ctx.rng.lognormal(21.0, 1.1), 2))
        rows.append(
            {
                "agency_id": f"AGY-{index:04d}",
                "agency_name": name,
                "agency_abbreviation": abbreviation,
                "agency_type": str(ctx.choice(AGENCY_TYPES)),
                "toptier_code": f"{ctx.rng.integers(10, 100):03d}",
                "established_year": int(ctx.rng.integers(1789, 2015)),
                "headquarters_state": state,
                "annual_budget_usd": budget,
            }
        )
    return pd.DataFrame(rows)


def generate_departments(ctx: GeneratorContext, agencies: pd.DataFrame) -> pd.DataFrame:
    """Between 3 and 8 departments per agency.

    A variable fan-out matters: it produces the uneven group sizes that make
    aggregation bugs (and skewed Spark partitions) visible.
    """
    rows: list[dict] = []
    counter = 0
    for agency in agencies.itertuples(index=False):
        n_departments = int(ctx.rng.integers(3, 9))
        chosen = ctx.rng.choice(len(DEPARTMENT_FUNCTIONS), size=n_departments, replace=False)
        for offset in chosen:
            counter += 1
            function = DEPARTMENT_FUNCTIONS[int(offset)]
            allocation = float(agency.annual_budget_usd) * float(ctx.rng.uniform(0.01, 0.12))
            rows.append(
                {
                    "department_id": f"DEP-{counter:05d}",
                    "department_name": f"{function} - {agency.agency_abbreviation}",
                    "agency_id": agency.agency_id,
                    "cost_center": f"CC-{ctx.rng.integers(1000, 9999)}",
                    "budget_allocation_usd": round(allocation, 2),
                    "manager_name": ctx.faker.name(),
                }
            )
    return pd.DataFrame(rows)
