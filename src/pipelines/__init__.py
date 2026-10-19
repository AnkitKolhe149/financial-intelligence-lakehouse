"""Pipeline orchestration.

Two pipelines, sharing every component below them:

* :mod:`src.pipelines.batch_pipeline` - the full Bronze -> Silver -> Gold load.
* :mod:`src.pipelines.streaming_pipeline` - continuous micro-batch scoring of the
  event stream.

Both are thin. All the logic lives in the layer packages; a pipeline's job is to
sequence stages, thread the run context through them, and make sure metadata is
recorded whether the run succeeds or fails. That last part is why every stage is
wrapped rather than simply called - a pipeline that only records successful runs
is blind to exactly the runs you need to investigate.
"""

from src.pipelines.batch_pipeline import BatchPipeline, run_batch_pipeline

__all__ = ["BatchPipeline", "run_batch_pipeline"]
