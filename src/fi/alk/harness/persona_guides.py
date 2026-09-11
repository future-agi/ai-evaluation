"""The persona values the platform understands.

A persona field is only useful if the platform recognises what is in it: an accent it knows
selects a voice, a personality it knows attaches a sentence of behaviour guidance. A value
written in words of its own renders fine and then does nothing, which is how a suite ends up
with callers who all behave the same.

The values are read from the platform's own model when it is mounted, and from the copy carried
with the harness when it is not, so a writer is always offered real ones. The behaviour guidance
itself lives with the prompt builder, next to the code that applies it.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the platform's tables are mounted. Colon-separated so voice and chat guides can both be
# offered; the first file defining a table wins, so voice takes precedence when both are present.
# Where the platform's persona model is mounted, for the values it accepts.
VOCABULARY_ENV = "HARNESS_PERSONA_VOCABULARY"

# The persona fields worth constraining, and the choice class each is drawn from. Only the ones
# that change behaviour or routing: a free-text occupation harms nothing, an accent nobody
# recognises silently loses the voice it was supposed to select.
FIELDS = {
    "gender": "GenderChoices",
    "age_group": "AgeGroupChoices",
    "occupation": "ProfessionChoices",
    "location": "LocationChoices",
    "personality": "PersonalityChoices",
    "communication_style": "CommunicationStyleChoices",
    "accent": "AccentChoices",
    "languages": "LanguageChoices",
}

# Constrained because something downstream reads them. The rest are offered as vocabulary but a
# writer who needs a value outside them is not stopped: an unknown occupation costs nothing,
# an unknown accent costs the voice.
ENFORCED = ("personality", "communication_style", "accent", "languages")


@lru_cache(maxsize=1)
def vocabulary() -> dict[str, list[str]]:
    """What the platform accepts for each persona field.

    Parsed out of the model's ``TextChoices`` classes for the same reason the guidance is read
    rather than restated: the platform is the one that has to understand these values, so it is
    the one that decides what they are. A persona written in words of its own renders fine, gets
    no behaviour guidance, and cannot be grouped with anything on the platform afterwards.
    """
    path = os.environ.get(VOCABULARY_ENV) or ""
    if not path or not Path(path).exists():
        # No model mounted. Fall back to the copy carried with the harness so a writer is always
        # offered real values: an empty vocabulary silently lets it invent an accent that selects
        # no voice and a personality that attaches no guidance.
        return _bundled_vocabulary()
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        logger.warning("persona vocabulary at %s is unreadable; using the bundled copy", path)
        return _bundled_vocabulary()

    by_class: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        values: list[str] = []
        for item in node.body:
            if not isinstance(item, ast.Assign):
                continue
            try:
                held = ast.literal_eval(item.value)
            except ValueError:
                continue
            # ``NAME = "value", "Label"`` is the choices shape; a bare string is also accepted.
            if isinstance(held, tuple) and held and isinstance(held[0], str):
                values.append(held[0])
            elif isinstance(held, str):
                values.append(held)
        if values:
            by_class[node.name] = values

    found = {
        field: by_class[cls] for field, cls in FIELDS.items() if by_class.get(cls)
    }
    if not found:
        # The file parsed but held none of the classes we key on, so it is the wrong file or the
        # classes moved. Silently returning nothing would drop every persona constraint at once.
        logger.warning(
            "persona vocabulary at %s defines none of %s; using the bundled copy",
            path,
            ", ".join(sorted(set(FIELDS.values()))),
        )
        return _bundled_vocabulary()
    return found



@lru_cache(maxsize=1)
def _bundled_vocabulary() -> dict[str, list[str]]:
    """The platform's persona values, carried with the harness.

    Kept so the harness constrains personas out of the box. Languages come from the agent
    definition's set rather than the persona dropdown's two, because nothing on the platform
    enforces the dropdown and a caller is expected to speak more than English and Hindi.
    """
    path = Path(__file__).parent / "data" / "persona_vocabulary.json"
    try:
        by_class = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("bundled persona vocabulary is unreadable; personas stay unconstrained")
        return {}
    return {
        field: list(by_class[cls])
        for field, cls in FIELDS.items()
        if by_class.get(cls)
    }


def offered(field: str) -> list[str]:
    """The values this field accepts, or nothing if the platform's model was not readable."""
    return list(vocabulary().get(field, []))


def unrecognised(persona: dict[str, object]) -> list[str]:
    """Persona values the platform would not recognise, as sentences saying what to use instead.

    Only the fields something downstream actually reads, and only when the vocabulary was found:
    a harness that cannot see the platform's model must not start refusing personas over it.
    """
    known = vocabulary()
    if not known:
        return []
    problems: list[str] = []
    for field in ENFORCED:
        allowed = known.get(field) or []
        if not allowed:
            continue
        held = persona.get(field)
        values = held if isinstance(held, list) else ([held] if held else [])
        lowered = {str(one).strip().lower() for one in allowed}
        for one in values:
            text = str(one).strip()
            if text and text.lower() not in lowered:
                problems.append(
                    f"persona {field} {text!r} is not one the platform knows, so it will not "
                    f"reach the call. Use one of: {', '.join(allowed)}. Anything else this "
                    "person is like belongs in persona.metadata."
                )
    return problems
