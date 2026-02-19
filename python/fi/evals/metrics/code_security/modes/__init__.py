"""
Evaluation Modes for AI Code Security.

Provides specialized evaluation modes for different AI code generation scenarios:
- INSTRUCT: Evaluate code generated from natural language instructions
- AUTOCOMPLETE: Evaluate code completion given partial context
- REPAIR: Evaluate if AI can fix vulnerable code
- ADVERSARIAL: Test resistance to prompts encouraging insecure code

Usage:
    from fi.evals.metrics.code_security.modes import (
        InstructModeEvaluator,
        AutocompleteModeEvaluator,
        RepairModeEvaluator,
        AdversarialModeEvaluator,
    )

    # Instruct mode
    evaluator = InstructModeEvaluator()
    result = evaluator.evaluate(
        instruction="Write a function to query users",
        generated_code=ai_response,
        language="python",
    )

    # Autocomplete mode
    evaluator = AutocompleteModeEvaluator()
    result = evaluator.evaluate(
        code_prefix="def get_user(id):\\n    query = ",
        generated_completion=ai_response,
        language="python",
    )
"""

from .base import (
    BaseModeEvaluator,
    ModeResult,
    InstructModeResult,
    AutocompleteModeResult,
    RepairModeResult,
    AdversarialModeResult,
)

from .instruct import InstructModeEvaluator
from .autocomplete import AutocompleteModeEvaluator
from .repair import RepairModeEvaluator
from .adversarial import AdversarialModeEvaluator


__all__ = [
    # Base
    "BaseModeEvaluator",
    "ModeResult",
    "InstructModeResult",
    "AutocompleteModeResult",
    "RepairModeResult",
    "AdversarialModeResult",
    # Evaluators
    "InstructModeEvaluator",
    "AutocompleteModeEvaluator",
    "RepairModeEvaluator",
    "AdversarialModeEvaluator",
]
