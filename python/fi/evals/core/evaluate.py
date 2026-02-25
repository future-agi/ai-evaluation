"""
evaluate() — the unified entrypoint for all evaluations.

Usage:
    from fi.evals import evaluate

    # Local metric (auto-detected)
    result = evaluate("contains", output="hello world", keyword="hello")

    # Cloud template (turing model → auto-routes to Turing)
    result = evaluate("toxicity", output="hello world", model="turing_flash")

    # Custom prompt on any LLM (use engine="llm", not "turing")
    result = evaluate(
        prompt="Rate the clarity of: {output}",
        output="ML is a subset of AI.",
        engine="llm",
        model="gemini/gemini-2.0-flash",
    )

    # Multiple evals
    results = evaluate(
        ["toxicity", "factual_accuracy"],
        output="Paris is the capital of France",
        model="turing_flash",
    )
"""

from typing import Any, Dict, List, Optional, Union

from .registry import get_unified_registry
from .result import BatchResult, EvalResult
from .engines import LocalEngine, TuringEngine, LLMEngine, Engine


def evaluate(
    eval_name: Optional[Union[str, List[str]]] = None,
    *,
    prompt: Optional[str] = None,
    engine: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    # Turing credentials (optional overrides)
    fi_api_key: Optional[str] = None,
    fi_secret_key: Optional[str] = None,
    fi_base_url: Optional[str] = None,
    **inputs,
) -> Union[EvalResult, BatchResult]:
    """Run one or more evaluations with automatic engine routing.

    Args:
        eval_name: Metric/template name or list of names. Can be None when
                   using a custom prompt.
        prompt: Custom evaluation prompt with {output}/{context}/{input}
                placeholders. Requires explicit engine or a model hint.
                Note: custom prompts only work with engine='llm'.
        engine: Force a specific engine — "local", "turing", or "llm".
        model: Model to use. Turing models (e.g. "turing_flash") auto-route
               to the Turing engine. Other model strings (e.g.
               "gemini/gemini-2.0-flash") auto-route to the LLM engine.
        config: Optional metric/judge config dict.
        fi_api_key: Override FI_API_KEY for Turing engine.
        fi_secret_key: Override FI_SECRET_KEY for Turing engine.
        fi_base_url: Override FI_BASE_URL for Turing engine.
        **inputs: Evaluation inputs (output, context, input, keyword, …).

    Returns:
        EvalResult for a single eval, BatchResult for multiple.
    """
    # --- Batch case: list of eval names --------------------------------
    if isinstance(eval_name, list):
        results = []
        for name in eval_name:
            r = evaluate(
                name,
                prompt=prompt,
                engine=engine,
                model=model,
                config=config,
                fi_api_key=fi_api_key,
                fi_secret_key=fi_secret_key,
                fi_base_url=fi_base_url,
                **inputs,
            )
            results.append(r)
        return BatchResult(results=results)

    # --- Single eval ---------------------------------------------------
    registry = get_unified_registry()

    # Custom prompt with no eval_name
    effective_name = eval_name or "custom_prompt"

    resolved_engine = registry.resolve_engine(
        eval_name,
        model=model,
        prompt=prompt,
        engine=engine,
    )

    # If still ambiguous and we have a prompt, require engine
    if resolved_engine is None and prompt:
        raise ValueError(
            "Cannot auto-detect engine for custom prompts. "
            "Specify engine='turing' or engine='llm', or provide a model "
            "(turing models auto-route to turing, others to llm)."
        )
    if resolved_engine is None:
        raise ValueError(
            f"Cannot auto-detect engine for '{eval_name}'. "
            "Specify engine='local', engine='turing', or engine='llm', "
            "or provide a model (turing models → turing, others → llm)."
        )

    eng = _get_engine(
        resolved_engine,
        fi_api_key=fi_api_key,
        fi_secret_key=fi_secret_key,
        fi_base_url=fi_base_url,
    )

    return eng.run(
        effective_name,
        inputs,
        model=model,
        prompt=prompt,
        config=config,
    )


_ENGINE_FACTORIES = {
    "local": lambda **_: LocalEngine(),
    "turing": lambda **kw: TuringEngine(
        fi_api_key=kw.get("fi_api_key"),
        fi_secret_key=kw.get("fi_secret_key"),
        fi_base_url=kw.get("fi_base_url"),
    ),
    "llm": lambda **_: LLMEngine(),
}


def _get_engine(
    engine_type: str,
    *,
    fi_api_key: Optional[str] = None,
    fi_secret_key: Optional[str] = None,
    fi_base_url: Optional[str] = None,
) -> Engine:
    factory = _ENGINE_FACTORIES.get(engine_type.lower())
    if factory is None:
        raise ValueError(
            f"Unknown engine: '{engine_type}'. "
            f"Use one of: {', '.join(sorted(_ENGINE_FACTORIES))}."
        )
    return factory(fi_api_key=fi_api_key, fi_secret_key=fi_secret_key, fi_base_url=fi_base_url)
