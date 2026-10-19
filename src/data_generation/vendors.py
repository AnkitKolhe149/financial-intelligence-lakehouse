"""Vendor dimension, including deliberate entity-resolution problems.

Two defect classes are injected here:

* **Near-duplicate vendors** - the same company registered twice under
  ``Acme Inc`` and ``Acme Inc.`` with a different id. This is the single most
  common real-world master-data problem in procurement, and it is what the
  duplicate-vendor quality check and the graph's shared-tax-id query exist to
  find.
* **Malformed contact emails** - nulled by the contract rather than rejected,
  which then shows up as a completeness dip rather than a quarantine spike.
"""

from __future__ import annotations

import pandas as pd

from src.data_generation.base import CITIES, INDUSTRY_CODES, GeneratorContext

VENDOR_TYPES = (
    "SMALL_BUSINESS",
    "SMALL_BUSINESS",
    "LARGE_BUSINESS",
    "NON_PROFIT",
    "SOLE_PROPRIETOR",
    "FOREIGN_ENTITY",
    "GOVERNMENT",
)
RISK_FLAGS = ("NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "WATCH", "SANCTIONED")


def generate_vendors(ctx: GeneratorContext, count: int) -> pd.DataFrame:
    """Generate ``count`` vendors, then inject near-duplicates on top.

    The returned frame therefore has slightly more than ``count`` rows when
    ``duplicate_vendor_rate > 0`` - the duplicates are extra registrations of
    vendors that already exist, which is exactly how they appear in a real
    vendor master.
    """
    injection = ctx.injection
    rows: list[dict] = []

    for index in range(1, count + 1):
        company = ctx.faker.company()
        industry_code, _industry_name = INDUSTRY_CODES[
            int(ctx.rng.integers(0, len(INDUSTRY_CODES)))
        ]
        city, state, *_ = CITIES[int(ctx.rng.integers(0, len(CITIES)))]
        vendor_type = str(ctx.choice(VENDOR_TYPES))
        country = "US" if vendor_type != "FOREIGN_ENTITY" else str(ctx.faker.country_code())

        email = f"contact@{_domain(company)}"
        if injection and ctx.occurs(injection.malformed_email_rate):
            email = ctx.malformed_email(email)

        name = company
        if injection and ctx.occurs(injection.whitespace_rate):
            name = ctx.dirty_string(company)

        rows.append(
            {
                "vendor_id": f"VEN-{index:06d}",
                "vendor_name": name,
                "vendor_legal_name": f"{company} {ctx.choice(('LLC', 'Inc.', 'Corporation', 'LP'))}",
                # Fictional tax id. Deliberately not an EIN-shaped value that
                # could collide with a real one.
                "tax_id": f"TX{ctx.rng.integers(10_000_000, 99_999_999)}",
                "vendor_type": vendor_type,
                "industry_code": industry_code,
                "contact_email": email,
                "phone": f"+1-{ctx.rng.integers(200, 999)}-{ctx.rng.integers(200, 999)}-{ctx.rng.integers(1000, 9999)}",
                "address_line1": ctx.faker.street_address(),
                "city": city,
                "state": state,
                "postal_code": f"{ctx.rng.integers(10_000, 99_999)}",
                "country": country,
                "registration_date": ctx.faker.date_between(
                    start_date="-15y", end_date="-30d"
                ).isoformat(),
                "is_active": bool(ctx.rng.random() > 0.06),
                "risk_flag": str(ctx.choice(RISK_FLAGS)),
            }
        )

    frame = pd.DataFrame(rows)
    if injection and injection.enabled and injection.duplicate_vendor_rate > 0:
        frame = _inject_near_duplicates(ctx, frame)
    return frame


def _domain(company: str) -> str:
    slug = "".join(ch for ch in company.lower() if ch.isalnum() or ch == " ")
    return f"{slug.split()[0] if slug.split() else 'vendor'}.example.com"


def _inject_near_duplicates(ctx: GeneratorContext, frame: pd.DataFrame) -> pd.DataFrame:
    """Re-register a sample of vendors under a lightly different name.

    The duplicate keeps the original ``tax_id``. That is the join key an analyst
    would use to prove two vendor records are one entity, and it gives the graph
    layer a real question to answer.
    """
    n_duplicates = int(len(frame) * ctx.injection.duplicate_vendor_rate)
    if n_duplicates <= 0:
        return frame

    victims = frame.sample(n=min(n_duplicates, len(frame)), random_state=ctx.seed)
    clones = victims.copy()
    start = len(frame) + 1
    clones["vendor_id"] = [f"VEN-{start + i:06d}" for i in range(len(clones))]
    clones["vendor_name"] = [ctx.near_duplicate_name(str(name)) for name in clones["vendor_name"]]
    # A duplicate registration is usually recent, which is what makes it look
    # like a "new vendor" to the risk engine - a genuine oversight signal.
    clones["registration_date"] = [
        ctx.faker.date_between(start_date="-180d", end_date="-1d").isoformat()
        for _ in range(len(clones))
    ]
    return pd.concat([frame, clones], ignore_index=True)
