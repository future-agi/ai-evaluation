"""
Auto-generate grading criteria from a short description.

Usage:
    from fi.evals import evaluate

    # Instead of writing a detailed rubric yourself:
    result = evaluate(
        prompt="product description accuracy for e-commerce images",
        output="A red cotton t-shirt with v-neck",
        image_url="https://example.com/tshirt.jpg",
        engine="llm",
        model="gemini/gemini-2.5-flash",
        generate_prompt=True,
    )
"""

import hashlib
from typing import Any, Dict, Optional

_CACHE: Dict[str, str] = {}

_META_PROMPT = """\
You are an expert prompt engineer specializing in LLM evaluation rubrics.

Given a short description of what to evaluate, generate a detailed grading \
criteria that an LLM judge can use to score inputs on a 0.0–1.0 scale.

The criteria MUST:
- Be specific and actionable (not vague)
- Define what 1.0, 0.5, and 0.0 look like
- Reference the input fields the judge will receive: {input_keys}
- Be 4-8 sentences maximum

Description of what to evaluate:
{description}

Return ONLY the grading criteria text. No JSON, no markdown, no preamble.\
"""


def generate_grading_criteria(
    description: str,
    model: str,
    inputs: Dict[str, Any],
    *,
    cache: bool = True,
) -> str:
    """Generate a detailed grading criteria from a short description.

    Args:
        description: Short description of what to evaluate
            (e.g. "product description accuracy for images").
        model: LiteLLM model string (e.g. "gemini/gemini-2.5-flash").
        inputs: The evaluation inputs dict — used to tell the generator
            which fields the judge will receive.
        cache: Cache results per (description, model) for the session.

    Returns:
        A detailed grading criteria string.
    """
    cache_key = hashlib.md5(f"{description}:{model}".encode()).hexdigest()
    if cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    input_keys = ", ".join(sorted(inputs.keys())) or "output"

    prompt = _META_PROMPT.format(
        description=description,
        input_keys=input_keys,
    )

    import litellm
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    criteria = response.choices[0].message.content.strip()

    if cache:
        _CACHE[cache_key] = criteria

    return criteria
