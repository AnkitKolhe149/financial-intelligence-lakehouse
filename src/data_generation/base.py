"""Shared generation primitives: seeded randomness and defect injection.

All randomness flows through one :class:`numpy.random.Generator` plus one seeded
``Faker``. Nothing calls the global ``random`` module, because a single global
call from an unrelated import would silently break reproducibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from faker import Faker

from src.core.config import InjectionConfig, Settings

T = TypeVar("T")

# US cities used for the location dimension. A fixed list keeps geography
# deterministic and lets the "geographically unusual" risk rule mean something:
# each account gets a home region, and spend outside it is what the rule sees.
CITIES: tuple[tuple[str, str, float, float, str], ...] = (
    ("Washington", "DC", 38.8951, -77.0364, "NORTHEAST"),
    ("New York", "NY", 40.7128, -74.0060, "NORTHEAST"),
    ("Boston", "MA", 42.3601, -71.0589, "NORTHEAST"),
    ("Philadelphia", "PA", 39.9526, -75.1652, "NORTHEAST"),
    ("Pittsburgh", "PA", 40.4406, -79.9959, "NORTHEAST"),
    ("Baltimore", "MD", 39.2904, -76.6122, "NORTHEAST"),
    ("Atlanta", "GA", 33.7490, -84.3880, "SOUTHEAST"),
    ("Miami", "FL", 25.7617, -80.1918, "SOUTHEAST"),
    ("Orlando", "FL", 28.5383, -81.3792, "SOUTHEAST"),
    ("Charlotte", "NC", 35.2271, -80.8431, "SOUTHEAST"),
    ("Nashville", "TN", 36.1627, -86.7816, "SOUTHEAST"),
    ("New Orleans", "LA", 29.9511, -90.0715, "SOUTHEAST"),
    ("Chicago", "IL", 41.8781, -87.6298, "MIDWEST"),
    ("Detroit", "MI", 42.3314, -83.0458, "MIDWEST"),
    ("Minneapolis", "MN", 44.9778, -93.2650, "MIDWEST"),
    ("St. Louis", "MO", 38.6270, -90.1994, "MIDWEST"),
    ("Kansas City", "MO", 39.0997, -94.5786, "MIDWEST"),
    ("Columbus", "OH", 39.9612, -82.9988, "MIDWEST"),
    ("Dallas", "TX", 32.7767, -96.7970, "SOUTHWEST"),
    ("Houston", "TX", 29.7604, -95.3698, "SOUTHWEST"),
    ("Austin", "TX", 30.2672, -97.7431, "SOUTHWEST"),
    ("Phoenix", "AZ", 33.4484, -112.0740, "SOUTHWEST"),
    ("Albuquerque", "NM", 35.0844, -106.6504, "SOUTHWEST"),
    ("Las Vegas", "NV", 36.1699, -115.1398, "SOUTHWEST"),
    ("Denver", "CO", 39.7392, -104.9903, "MOUNTAIN"),
    ("Salt Lake City", "UT", 40.7608, -111.8910, "MOUNTAIN"),
    ("Boise", "ID", 43.6150, -116.2023, "MOUNTAIN"),
    ("Billings", "MT", 45.7833, -108.5007, "MOUNTAIN"),
    ("Los Angeles", "CA", 34.0522, -118.2437, "WEST"),
    ("San Francisco", "CA", 37.7749, -122.4194, "WEST"),
    ("San Diego", "CA", 32.7157, -117.1611, "WEST"),
    ("Seattle", "WA", 47.6062, -122.3321, "WEST"),
    ("Portland", "OR", 45.5152, -122.6784, "WEST"),
    ("Sacramento", "CA", 38.5816, -121.4944, "WEST"),
    ("Anchorage", "AK", 61.2181, -149.9003, "WEST"),
    ("Honolulu", "HI", 21.3069, -157.8583, "WEST"),
)

AGENCY_NAMES: tuple[tuple[str, str], ...] = (
    ("Department of Transportation", "DOT"),
    ("Department of Energy", "DOE"),
    ("Department of Health and Human Services", "HHS"),
    ("Department of Veterans Affairs", "VA"),
    ("Department of Agriculture", "USDA"),
    ("Department of Commerce", "DOC"),
    ("Department of Education", "ED"),
    ("Department of Labor", "DOL"),
    ("Department of the Interior", "DOI"),
    ("Environmental Protection Agency", "EPA"),
    ("General Services Administration", "GSA"),
    ("National Aeronautics and Space Administration", "NASA"),
    ("National Science Foundation", "NSF"),
    ("Small Business Administration", "SBA"),
    ("Social Security Administration", "SSA"),
    ("Department of Homeland Security", "DHS"),
    ("Department of Housing and Urban Development", "HUD"),
    ("Department of Justice", "DOJ"),
    ("Department of State", "DOS"),
    ("Department of the Treasury", "TREAS"),
    ("Federal Communications Commission", "FCC"),
    ("Federal Emergency Management Agency", "FEMA"),
    ("Nuclear Regulatory Commission", "NRC"),
    ("Office of Personnel Management", "OPM"),
    ("National Institutes of Health", "NIH"),
    ("Centers for Disease Control", "CDC"),
    ("Food and Drug Administration", "FDA"),
    ("Federal Aviation Administration", "FAA"),
    ("Bureau of Land Management", "BLM"),
    ("National Park Service", "NPS"),
)

DEPARTMENT_FUNCTIONS: tuple[str, ...] = (
    "Office of Procurement",
    "Office of the Chief Financial Officer",
    "Office of Information Technology",
    "Office of Facilities Management",
    "Office of Grants Administration",
    "Office of Research and Development",
    "Office of Field Operations",
    "Office of Human Capital",
    "Office of Inspector General",
    "Office of Policy and Planning",
    "Office of Emergency Response",
    "Office of Public Affairs",
)

INDUSTRY_CODES: tuple[tuple[str, str], ...] = (
    ("541512", "Computer Systems Design Services"),
    ("236220", "Commercial Building Construction"),
    ("541330", "Engineering Services"),
    ("561210", "Facilities Support Services"),
    ("423430", "Computer Equipment Wholesalers"),
    ("541611", "Management Consulting Services"),
    ("621111", "Offices of Physicians"),
    ("485113", "Bus Transit Systems"),
    ("811310", "Industrial Machinery Repair"),
    ("541990", "Professional Services"),
    ("325412", "Pharmaceutical Preparation"),
    ("336411", "Aircraft Manufacturing"),
    ("517311", "Wired Telecommunications"),
    ("611430", "Professional Training"),
    ("562910", "Remediation Services"),
)

MERCHANT_CATEGORIES: tuple[str, ...] = (
    "PROFESSIONAL_SERVICES",
    "IT_HARDWARE",
    "IT_SOFTWARE",
    "CONSTRUCTION",
    "MEDICAL_SUPPLIES",
    "OFFICE_SUPPLIES",
    "TRAVEL",
    "FUEL",
    "UTILITIES",
    "TRAINING",
    "RESEARCH_GRANT",
    "MAINTENANCE",
    "TELECOM",
    "LOGISTICS",
)

# Invalid currency codes deliberately injected into the transaction feed.
BAD_CURRENCIES: tuple[str, ...] = ("US$", "XXX", "usd ", "DOLLARS", "€", "ZZZ", "")


@dataclass
class GeneratorContext:
    """Seeded randomness plus the injection profile, passed to every generator."""

    settings: Settings
    seed: int = 42
    injection: InjectionConfig | None = None
    rng: np.random.Generator = field(init=False)
    faker: Faker = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.faker = Faker("en_US")
        Faker.seed(self.seed)
        if self.injection is None:
            self.injection = self.settings.injection

    # -- primitive draws ----------------------------------------------------

    def choice(
        self, options: Sequence[T], size: int | None = None, p: Sequence[float] | None = None
    ) -> Any:
        """Draw from ``options``. Indexes rather than passing objects to numpy,
        which would coerce tuples into arrays."""
        idx = self.rng.choice(len(options), size=size, p=p)
        if size is None:
            return options[int(idx)]
        return [options[int(i)] for i in np.atleast_1d(idx)]

    def occurs(self, rate: float) -> bool:
        """True with probability ``rate`` - the injection primitive."""
        if rate <= 0 or not self.injection or not self.injection.enabled:
            return False
        return bool(self.rng.random() < rate)

    def mask(self, size: int, rate: float) -> np.ndarray:
        """Vectorised :meth:`occurs` for whole-column injection."""
        if rate <= 0 or not self.injection or not self.injection.enabled:
            return np.zeros(size, dtype=bool)
        return self.rng.random(size) < rate

    def lognormal_amounts(self, size: int, mean: float = 6.4, sigma: float = 1.35) -> np.ndarray:
        """Transaction amounts follow a heavy right tail, as real spend does.

        A normal distribution would make every large payment an outlier and the
        anomaly model trivially good; the lognormal keeps the problem honest.
        """
        return np.round(self.rng.lognormal(mean, sigma, size), 2)

    def business_hour_timestamps(
        self, size: int, start: date, end: date, off_hours_rate: float = 0.08
    ) -> np.ndarray:
        """Timestamps skewed to business hours, with a configurable night tail.

        The night tail exists so the "unusual hour" risk rule has both true and
        false positives to work with.
        """
        total_days = (end - start).days
        day_offsets = self.rng.integers(0, max(total_days, 1), size=size)

        # Weekday bias: shift ~85% of weekend draws onto the neighbouring weekday.
        base = np.array([start + timedelta(days=int(d)) for d in day_offsets])
        weekday = np.array([d.weekday() for d in base])
        weekend = weekday >= 5
        shift = weekend & (self.rng.random(size) < 0.85)
        base = np.array(
            [
                d - timedelta(days=int(w) - 4) if s else d
                for d, w, s in zip(base, weekday, shift, strict=True)
            ]
        )

        off_hours = self.rng.random(size) < off_hours_rate
        hours = np.where(
            off_hours,
            self.rng.integers(0, 5, size=size),  # 00:00-04:59
            self.rng.integers(8, 19, size=size),  # 08:00-18:59
        )
        minutes = self.rng.integers(0, 60, size=size)
        seconds = self.rng.integers(0, 60, size=size)
        return np.array(
            [
                datetime(d.year, d.month, d.day, int(h), int(m), int(s), tzinfo=UTC)
                for d, h, m, s in zip(base, hours, minutes, seconds, strict=True)
            ]
        )

    # -- string corruption --------------------------------------------------

    def dirty_string(self, value: str) -> str:
        """Apply realistic whitespace/casing noise to a string."""
        style = int(self.rng.integers(0, 4))
        if style == 0:
            return f"  {value} "
        if style == 1:
            return value.upper()
        if style == 2:
            return value.lower()
        return value.replace(" ", "  ")

    def malformed_email(self, value: str) -> str:
        style = int(self.rng.integers(0, 4))
        if style == 0:
            return value.replace("@", "")
        if style == 1:
            return value.replace(".", "", 1)
        if style == 2:
            return f"{value}@"
        return f"not-an-email-{value.split('@')[0]}"

    def near_duplicate_name(self, value: str) -> str:
        """Produce an entity-resolution problem: same company, different string."""
        suffixes = (" Inc", " Inc.", " LLC", " L.L.C.", " Corporation", " Corp", " Co.", "")
        base = value
        for suffix in (" Inc", " LLC", " Corp", " Ltd", " Group", " Inc.", " Corporation"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base.strip()}{self.choice(suffixes)}"


@dataclass
class ReferenceData:
    """The dimension frames generated before the transaction fact table.

    Held together in one object because transaction generation needs consistent
    access to all of them, and because the referential-integrity defects are
    created by *deliberately* referencing ids that are absent from these frames.
    """

    agencies: pd.DataFrame
    departments: pd.DataFrame
    locations: pd.DataFrame
    vendors: pd.DataFrame
    contracts: pd.DataFrame
    accounts: pd.DataFrame
    budgets: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "agencies": self.agencies,
            "departments": self.departments,
            "locations": self.locations,
            "vendors": self.vendors,
            "contracts": self.contracts,
            "accounts": self.accounts,
            "budgets": self.budgets,
        }

    def row_counts(self) -> dict[str, int]:
        return {name: len(frame) for name, frame in self.as_dict().items()}
