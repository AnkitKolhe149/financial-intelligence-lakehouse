"""Silver: validated, typed, deduplicated, referentially-consistent data.

Silver is where the platform takes a position on what the data *means*. Bronze
recorded what arrived; Silver decides which of it is usable, and - critically -
what to do with the rest.

The policy applied throughout:

* **Required field missing** -> quarantine. A transaction with no account cannot
  be attributed to anyone, so it cannot be aggregated.
* **Optional field missing** -> keep the null. Blanket ``fillna`` is how a
  platform quietly invents data; a null that means "we do not know" is more
  honest and more useful than a zero that means the same thing but does not
  look like it.
* **Invalid value** (negative amount, unknown currency, unparseable date) ->
  quarantine, with a specific error code.
* **Broken foreign key** -> quarantine for blocking references (account,
  vendor); keep with a null for non-blocking ones (contract), because petty-cash
  spend legitimately has no contract.
* **Duplicates** -> collapse to one survivor, chosen deterministically.

Nothing is ever silently dropped. Every rejected record lands in the quarantine
dataset with its error code, its source, and the run that rejected it.
"""

from src.silver.spec import SILVER_SPECS, SilverSpec, get_silver_spec
from src.silver.transformations import SilverResult, build_silver

__all__ = ["SILVER_SPECS", "SilverResult", "SilverSpec", "build_silver", "get_silver_spec"]
