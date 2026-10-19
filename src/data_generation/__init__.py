"""Synthetic source-data generation.

The platform's primary data source is generated rather than downloaded, for
three reasons that are worth stating because they are engineering decisions,
not conveniences:

1. **Multi-format coverage.** No real dataset ships as CSV *and* JSON *and* XML
   *and* Excel *and* a REST feed *and* a stream. Generating the same business
   domain into every format is what makes the ingestion layer's normalisation
   work meaningful.
2. **Controlled defects.** A data-quality framework can only be demonstrated
   against data that has known, measurable defects. The generator injects them
   at configured rates, so the expected quarantine volume is known in advance
   and the quality framework can be tested rather than merely exercised.
3. **No sensitive data.** Every person, vendor and account here is fictional.
   Nothing in this repository is derived from a real individual.

Determinism is guaranteed by a single seed (``generation.seed``); the same seed
always produces byte-identical output, which is what makes pipeline runs
reproducible and benchmark comparisons fair.
"""

from src.data_generation.base import GeneratorContext, ReferenceData

__all__ = ["GeneratorContext", "ReferenceData"]
