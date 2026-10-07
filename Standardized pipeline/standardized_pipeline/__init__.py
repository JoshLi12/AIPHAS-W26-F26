"""Schema-driven pipeline from a raw table to dashboard and model files.

Every model decision carries a confidence score. Scores at or above the
threshold are completed by the model. Scores below it are left unfinished
until a person completes them.
"""

from standardized_pipeline.gate import ConfidenceGate
from standardized_pipeline.pipeline import apply_reviews, run_pipeline

__all__ = ["ConfidenceGate", "apply_reviews", "run_pipeline"]
