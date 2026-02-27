import inspect
import warnings

# Suppress Pydantic field-shadowing warnings from our models and the installed fi.api package
warnings.filterwarnings("ignore", message='Field name "json" in .* shadows an attribute in parent')
warnings.filterwarnings("ignore", message='Field name "schema" in .* shadows an attribute in parent')

# ---------------------------------------------------------------------------
# Unified evaluate() API (new)
# ---------------------------------------------------------------------------
from .core import evaluate, EvalResult, BatchResult, Turing  # noqa: F401

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
# Framework (evaluation orchestration + distributed backends)
# ---------------------------------------------------------------------------
try:
    from .framework import (
        FrameworkEvaluator,
        ExecutionMode,
        blocking_evaluator,
        async_evaluator,
        distributed_evaluator,
        resilient_evaluator,
        register_current_span,
        BaseEvaluation,
        register_evaluation,
        custom_eval,
        simple_eval,
        EvalBuilder,
    )
    _framework_available = True
except (ImportError, ModuleNotFoundError):
    _framework_available = False

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
new_api_names = ["evaluate", "EvalResult", "BatchResult", "Turing"]

# Existing clients
client_names = ["Evaluator", "Protect", "protect", "list_evaluations"]

# Framework exports
framework_names = [
    "FrameworkEvaluator",
    "ExecutionMode",
    "blocking_evaluator",
    "async_evaluator",
    "distributed_evaluator",
    "resilient_evaluator",
    "register_current_span",
    "BaseEvaluation",
    "register_evaluation",
    "custom_eval",
    "simple_eval",
    "EvalBuilder",
] if _framework_available else []

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

__all__ = sorted(new_api_names + evaluation_template_names + client_names + framework_names + streaming_names)
