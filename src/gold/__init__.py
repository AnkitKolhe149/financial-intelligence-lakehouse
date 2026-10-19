"""Gold: business-ready datasets that answer oversight questions directly.

Gold is where the platform stops describing data and starts answering questions.
Each dataset here maps to a question an oversight analyst actually asks:

===============================  ==================================================
``gold_daily_financial_summary`` What was spent, by whom, and how anomalous was it?
``gold_account_risk``            Which accounts behave unusually?
``gold_vendor_risk``             Which vendors concentrate risk?
``gold_agency_spending``         Which agencies spend the most, and on what?
``gold_department_spending``     Which departments are over budget?
``gold_anomaly_transactions``    Exactly which transactions are suspicious, and why?
``gold_risk_alerts``             What is the review queue?
``gold_transaction_velocity``    Where is activity spiking?
``gold_data_quality_summary``    Can these numbers be trusted?
===============================  ==================================================

Alongside them, :mod:`src.gold.star_schema` publishes a conformed star schema
(``fact_transactions`` plus ``dim_*``) for BI-style consumption, including a
slowly-changing dimension for vendors.

Design rule: Gold tables are **wide, denormalised and pre-aggregated**. A
dashboard should read one table, not join five. The cost is storage, which is
cheap; the benefit is that a query behind a screen an executive is watching does
not depend on a five-way join finishing in time.
"""

from src.gold.base import GoldResult, write_gold
from src.gold.builder import build_all_gold

__all__ = ["GoldResult", "build_all_gold", "write_gold"]
