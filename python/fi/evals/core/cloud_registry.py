"""
Cloud eval registry — source of truth for eval metadata.

Fetches the full template list from `/sdk/api/v1/get-evals/` once per
(base_url, api_key) tuple and caches the result. Exposes helpers that
turn user-supplied kwargs into the exact key set the backend will accept,
so the SDK never drifts from the backend when evals are added/renamed.

Why this exists
---------------

The old ``templates.py`` hardcoded Pydantic Input models per template
(``OutputOnly``, ``OutputWithContext``, ``OutputWithExpected``, ...). The
Turing revamp renamed/removed/replaced ~50 templates; the hardcoded
schemas drifted and 28 of 57 stopped working. Going forward, the backend
is authoritative — the SDK reads ``required_keys`` and sends only those.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# Module-level cache: {(base_url, api_key_prefix): {eval_name: info_dict}}
_CACHE: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()

# Aliases user kwargs can use → canonical backend keys.
# Maps are checked in order; first match wins.
_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    # Bidirectional output↔input — some evals only accept one of the two
    # (e.g. prompt_injection wants `input`, toxicity wants `output`).
    # Aliasing only fires when the canonical key isn't already in user_inputs,
    # so we can't accidentally overwrite an explicit user value.
    "output": ("output", "response", "answer", "generated", "input"),
    "input": ("input", "query", "question", "prompt_input", "output"),
    "context": ("context", "contexts"),
    "expected": ("expected", "expected_output", "expected_response", "ground_truth"),
    "expected_value": ("expected_value", "expected_output", "expected_response", "ground_truth"),
    "generated_value": ("generated_value", "output", "response", "answer"),
    "reference": ("reference", "expected_output", "expected_response", "ground_truth"),
    "hypothesis": ("hypothesis", "output", "response"),
    "text": ("text", "output", "content"),
    "conversation": ("conversation", "messages"),
    "prompt": ("prompt", "instructions", "system_prompt"),
    "system_prompt": ("system_prompt", "prompt", "instructions"),
    "image": ("image", "image_url", "input_image_url"),
    "caption": ("caption", "output"),
    "instruction": ("instruction", "prompt"),
    "instructions": ("instructions", "prompt"),
    "images": ("images", "image_urls", "input_image_urls"),
    "input_pdf": ("input_pdf", "pdf"),
    "json_content": ("json_content", "json", "expected_output"),
    "input_audio": ("input_audio", "audio"),
    "audio": ("audio", "input_audio"),
    "generated_audio": ("generated_audio", "audio", "output"),
    "generated_transcript": ("generated_transcript", "transcript", "output"),
}


def _cache_key(base_url: str, api_key: Optional[str]) -> Tuple[str, str]:
    return (base_url.rstrip("/"), (api_key or "")[:12])


def load_registry(
    base_url: str,
    api_key: Optional[str],
    secret_key: Optional[str],
    *,
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch and cache the eval template list. Returns {eval_name: info}."""
    key = _cache_key(base_url, api_key)
    if not force_refresh:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return cached

    # Imported lazily to avoid a circular import at module load time.
    import requests

    url = f"{base_url.rstrip('/')}/sdk/api/v1/get-evals/"
    headers = {}
    if api_key:
        headers["X-Api-Key"] = api_key
    if secret_key:
        headers["X-Secret-Key"] = secret_key

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        log.warning("Failed to load cloud eval registry from %s: %s", url, exc)
        return {}

    items = body.get("result") or []
    by_name: Dict[str, Dict[str, Any]] = {}
    for item in items:
        name = item.get("name")
        if name:
            by_name[name] = item

    with _CACHE_LOCK:
        _CACHE[key] = by_name
    log.debug("Loaded %d cloud eval templates from %s", len(by_name), url)
    return by_name


def get_template_info(
    name: str,
    *,
    base_url: str,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the backend's config block for a given eval, or None."""
    reg = load_registry(base_url, api_key, secret_key)
    return reg.get(name)


def get_required_keys(
    name: str,
    *,
    base_url: str,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> List[str]:
    """List the keys the backend requires for this eval. Empty list if unknown."""
    info = get_template_info(name, base_url=base_url, api_key=api_key, secret_key=secret_key)
    if not info:
        return []
    return list(info.get("config", {}).get("required_keys", []) or [])


def map_inputs_to_backend(
    name: str,
    user_inputs: Dict[str, Any],
    *,
    base_url: str,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce the exact payload the backend expects for ``eval_name``, by:
      1. Looking up the eval's required_keys from the cached registry.
      2. Taking each required key from user_inputs directly if present.
      3. Otherwise resolving via known aliases (e.g. ``output`` → ``response``).
      4. Dropping any keys the backend doesn't accept (the api is strict).

    If the eval isn't in the registry (unknown name, registry load failed),
    falls back to passing user_inputs through unmodified so the backend
    can return its own validation error.
    """
    required = get_required_keys(
        name, base_url=base_url, api_key=api_key, secret_key=secret_key
    )
    if not required:
        return dict(user_inputs)

    mapped: Dict[str, Any] = {}
    for key in required:
        if key in user_inputs:
            mapped[key] = user_inputs[key]
            continue
        for alias in _KEY_ALIASES.get(key, ()):
            if alias in user_inputs and alias != key:
                mapped[key] = user_inputs[alias]
                break
    return mapped


def list_known_names(
    *,
    base_url: str,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Set[str]:
    """Names of all evals the backend currently has registered."""
    return set(load_registry(base_url, api_key, secret_key).keys())
