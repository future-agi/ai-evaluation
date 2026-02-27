"""Feedback Loop system for improving evaluations over time.

Store developer feedback on metric results, retrieve similar past feedback
as few-shot examples for LLM judges, and calibrate thresholds statistically.
"""

from .types import FeedbackEntry, CalibrationProfile, FeedbackStats
from .store import FeedbackStore, InMemoryFeedbackStore
from .collector import FeedbackCollector
from .retriever import FeedbackRetriever
from .calibrator import ThresholdCalibrator
from .hooks import configure_feedback, get_default_store

# ChromaFeedbackStore requires chromadb — import conditionally
try:
    from .store import ChromaFeedbackStore
except Exception:
    pass

__all__ = [
    "FeedbackEntry",
    "CalibrationProfile",
    "FeedbackStats",
    "FeedbackStore",
    "InMemoryFeedbackStore",
    "ChromaFeedbackStore",
    "FeedbackCollector",
    "FeedbackRetriever",
    "ThresholdCalibrator",
    "configure_feedback",
    "get_default_store",
]
