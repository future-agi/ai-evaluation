import inspect

# Optional imports - may fail if fi.api is not available
# This allows guardrails and scanners to work independently
try:
    from .evaluator import Evaluator, evaluate, list_evaluations  # noqa: F401
    from .protect import Protect, protect  # noqa: F401
    from .templates import *  # noqa: F403, F401
    _evaluator_available = True
except (ImportError, ModuleNotFoundError):
    _evaluator_available = False
    Evaluator = None
    evaluate = None
    list_evaluations = None
    Protect = None
    protect = None

# Streaming evaluation imports
from .streaming import (  # noqa: F401
    StreamingEvaluator,
    StreamingConfig,
    StreamingEvalResult,
    ChunkResult,
    EarlyStopPolicy,
    EarlyStopReason,
    StreamingState,
)

# Dynamically generate __all__ from imported templates
_globals = globals()
evaluation_template_names = []
if _evaluator_available:
    evaluation_template_names = [
        name
        for name, obj in _globals.items()
        if inspect.isclass(obj) and obj.__module__ == "fi.evals.templates"
    ]

# Add the clients separately
client_names = ["Evaluator", "Protect", "evaluate", "protect", "list_evaluations"]

# Add streaming exports
streaming_names = [
    "StreamingEvaluator",
    "StreamingConfig",
    "StreamingEvalResult",
    "ChunkResult",
    "EarlyStopPolicy",
    "EarlyStopReason",
    "StreamingState",
]

# Combine and sort for consistency
__all__ = sorted(evaluation_template_names + client_names + streaming_names)
