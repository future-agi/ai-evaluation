import inspect

from fi.evals.evaluator import Evaluator, evaluate, list_evaluations  # noqa: F401
from fi.evals.protect import Protect, protect  # noqa: F401
from fi.evals.templates import *  # noqa: F403, F401
from fi.evals.metrics import BLEUScore, ROUGEScore, NumericDiff, LevenshteinDistance, EmbeddingSimilarity, SemanticListContains, AggregatedMetric

# Dynamically generate __all__ from imported templates
_globals = globals()
evaluation_template_names = [
    name
    for name, obj in _globals.items()
    if inspect.isclass(obj) and obj.__module__ == "fi.evals.templates"
]

# Add the clients separately
client_names = ["Evaluator", "Protect", "evaluate", "protect", "list_evaluations"]

# Combine and sort for consistency
__all__ = sorted(evaluation_template_names + client_names)
