import inspect

# ---------------------------------------------------------------------------
# Unified evaluate() API (new)
# ---------------------------------------------------------------------------
from .core import evaluate, EvalResult, BatchResult  # noqa: F401

# ---------------------------------------------------------------------------
# Cloud Evaluator + Protect (existing)
# ---------------------------------------------------------------------------
try:
    from .evaluator import Evaluator, list_evaluations  # noqa: F401
    from .protect import Protect, protect  # noqa: F401
    from .templates import *  # noqa: F403, F401
    _evaluator_available = True
except (ImportError, ModuleNotFoundError):
    _evaluator_available = False
    Evaluator = None
    list_evaluations = None
    Protect = None
    protect = None

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
from .streaming import (  # noqa: F401
    StreamingEvaluator,
    StreamingConfig,
    StreamingEvalResult,
    ChunkResult,
    EarlyStopPolicy,
    EarlyStopReason,
    StreamingState,
)

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
_globals = globals()
evaluation_template_names = []
if _evaluator_available:
    evaluation_template_names = [
        name
        for name, obj in _globals.items()
        if inspect.isclass(obj) and obj.__module__ == "fi.evals.templates"
    ]

# New unified API
new_api_names = ["evaluate", "EvalResult", "BatchResult"]

# Existing clients
client_names = ["Evaluator", "Protect", "protect", "list_evaluations"]

# Streaming exports
streaming_names = [
    "StreamingEvaluator",
    "StreamingConfig",
    "StreamingEvalResult",
    "ChunkResult",
    "EarlyStopPolicy",
    "EarlyStopReason",
    "StreamingState",
]

__all__ = sorted(new_api_names + evaluation_template_names + client_names + streaming_names)
