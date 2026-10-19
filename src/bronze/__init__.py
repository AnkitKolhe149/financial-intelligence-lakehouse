"""Bronze: raw, append-only, source-faithful storage.

Bronze answers one question - *what did the source actually send us?* - and it
must answer it correctly even years later, after the Silver logic has been
rewritten three times. Consequently:

* **No cleaning, no casting, no filtering.** Values stay as text if they arrived
  as text, including the ones that are obviously wrong. A negative amount is
  preserved here and rejected in Silver; deleting it at ingestion would destroy
  the evidence that the source system has a bug.
* **Provenance on every row** - source system, file, format, ingestion run,
  content hash, schema version, row number.
* **Idempotent loads.** Re-running the same file is a no-op, enforced by a
  content-hash ledger rather than by hoping the operator does not re-run it.
* **Schema-on-read.** New source columns are absorbed without a migration.

Because Bronze is complete and immutable, Silver and Gold are always
reconstructible: delete them, re-run, and you get identical output.
"""

from src.bronze.loader import BronzeLoader, BronzeLoadResult

__all__ = ["BronzeLoadResult", "BronzeLoader"]
