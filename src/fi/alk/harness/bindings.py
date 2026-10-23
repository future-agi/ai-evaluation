"""The agent-binding mapping language (`environment-releases.md` §2.4.1, §2.4.2).

A `tool_routed` binding lets an agent keep its own tools: each agent tool maps onto a sequence of
environment operations, declaratively, so the same mapping runs in the tool server and in the
binding proof. Arbitrary code goes through extensions instead (plan D3).

This module imports nothing from `releases` at import time: release models import it.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import jmespath
from jinja2 import StrictUndefined, TemplateError, nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jmespath.exceptions import JMESPathError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_serializer,
    model_validator,
)

if TYPE_CHECKING:
    from .releases import AgentBindingV1, ToolDefinition

_SLUG_PATTERN = r"[a-z][a-z0-9_]*"
_TOOL_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
# Provider function names (Retell and Vapi both allow dashes, up to 64 characters).
_AGENT_TOOL_PATTERN = r"[A-Za-z0-9_-]{1,64}"


class ToolError(Exception):
    """A handler refusing a call. The one class handlers raise, in the tool server and in proofs."""


# --- models ----------------------------------------------------------------------------------------


class _Authored(BaseModel):
    """Dumps only the keys the author wrote, so a binding's digest is over its document as given.

    The platform passes `tools[]` through verbatim; filling optional keys with nulls here would
    give the guest a different digest from the one the platform sealed.
    """

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @model_serializer(mode="wrap")
    def _as_authored(self, handler: Any) -> Any:
        data = handler(self)
        written = {
            self.__class__.model_fields[name].alias or name
            for name in self.model_fields_set
        }
        return {key: value for key, value in data.items() if key in written}


_FORMS = ("arg", "step", "fixture", "const", "concat", "object", "list")


class Value(_Authored):
    arg: str | None = None
    step: str | None = None
    path: str | None = None
    fixture: str | None = None
    const: JsonValue = None
    concat: list[Value] | None = None
    separator: str | None = None
    object_: dict[str, Value] | None = Field(default=None, alias="object")
    list_: list[Value] | None = Field(default=None, alias="list")
    convert: str | None = None
    map_: dict[str, JsonValue] | None = Field(default=None, alias="map")
    alias: str | None = None

    @model_validator(mode="after")
    def _one_form(self) -> Value:
        forms = [form for form in _FORMS if self._has(form)]
        if len(forms) != 1:
            raise ValueError(
                f"a value is exactly one of {', '.join(_FORMS)}; got {forms or 'none'}"
            )
        if self._has("path") and forms[0] != "step":
            raise ValueError("path belongs to a step value")
        if self._has("separator") and forms[0] != "concat":
            raise ValueError("separator belongs to a concat value")
        return self

    def _has(self, form: str) -> bool:
        return {"object": "object_", "list": "list_"}.get(
            form, form
        ) in self.model_fields_set

    @property
    def form(self) -> str:
        return next(form for form in _FORMS if self._has(form))

    def leaves(self) -> list[Value]:
        children = [
            *(self.concat or ()),
            *(self.list_ or ()),
            *(self.object_ or {}).values(),
        ]
        if not children:
            return [self]
        return [leaf for child in children for leaf in child.leaves()]


class Step(_Authored):
    id: str = Field(pattern=_SLUG_PATTERN)
    tool: str = Field(pattern=_TOOL_NAME_PATTERN)
    arguments: dict[str, Value] = Field(default_factory=dict)


class Response(_Authored):
    template: str | None = None
    value: Value | None = None

    @model_validator(mode="after")
    def _one_form(self) -> Response:
        if len(self.model_fields_set & {"template", "value"}) != 1 or (
            self.template is None and self.value is None
        ):
            raise ValueError("a response is exactly one of template or value")
        return self


class MappedToolBinding(_Authored):
    agent_tool: str = Field(pattern=_AGENT_TOOL_PATTERN)
    steps: list[Step]
    response: Response | None = None


class UnsupportedToolBinding(_Authored):
    agent_tool: str = Field(pattern=_AGENT_TOOL_PATTERN)
    unsupported: str = Field(min_length=1)


ToolBinding = MappedToolBinding | UnsupportedToolBinding


class AgentTool(BaseModel):
    """An admitted agent tool (§6.3), as the binding records it. Every field is required."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_AGENT_TOOL_PATTERN)
    description: str
    parameters: dict[str, JsonValue]
    admission: Literal["routed", "replaced"]

    @field_validator("parameters")
    @classmethod
    def _object_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if value.get("type") != "object":
            raise ValueError("parameters must be a JSON Schema object")
        return value


# --- conversions ------------------------------------------------------------------------------------


class _Invalid(Exception):
    """A value that cannot be produced."""


_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_US_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_MONTH_FIRST = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")
_DAY_FIRST = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})")
_CLOCK = re.compile(r"(\d{1,2}):(\d{2})")
_CLOCK_12 = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]\.?")
_HOUR_12 = re.compile(r"(\d{1,2})\s*([AaPp])\.?[Mm]\.?")
_ISO_DATETIME = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?")
_INTEGER = re.compile(r"[+-]?\d+")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def _text(value: Any) -> str:
    """The string form of a scalar, independent of locale and Python's repr of booleans."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return json.dumps(value)
    raise _Invalid(f"{json.dumps(value, default=str)} is not a scalar")


def _string(value: Any) -> str:
    return _text(value)


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        raise _Invalid(f"{value!r} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _INTEGER.fullmatch(value.strip()):
        return int(value.strip())
    raise _Invalid(f"{value!r} is not an integer")


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        raise _Invalid(f"{value!r} is not a number")
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _INTEGER.fullmatch(text):
            return int(text)
        if _NUMBER.fullmatch(text):
            return float(text)
    raise _Invalid(f"{value!r} is not a number")


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise _Invalid(f"{value!r} is not a boolean")


def _string_only(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise _Invalid(f"{name} needs a string, got {value!r}")
    return value


def _digits(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    digits = "".join(c for c in _string_only(value, "digits") if c in "0123456789")
    if not digits:
        raise _Invalid(f"{value!r} has no digits")
    return digits


def _calendar(year: int, month: int, day: int, value: Any) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise _Invalid(f"{value!r} is not a calendar date") from exc


def _month(name: str, value: Any) -> int:
    if (number := _MONTHS.get(name.lower())) is None:
        raise _Invalid(f"{value!r} names no month")
    return number


def _date(value: Any) -> str:
    text = _string_only(value, "date").strip()
    match = _ISO_DATE.fullmatch(text) or (
        _ISO_DATETIME.fullmatch(text) and _ISO_DATE.fullmatch(text[:10])
    )
    if match:
        year, month, day = (int(part) for part in match.groups())
        return _calendar(year, month, day, value)
    if match := _US_DATE.fullmatch(text):
        month, day, year = (int(part) for part in match.groups())
        return _calendar(year, month, day, value)
    if match := _MONTH_FIRST.fullmatch(text):
        return _calendar(int(match[3]), _month(match[1], value), int(match[2]), value)
    if match := _DAY_FIRST.fullmatch(text):
        return _calendar(int(match[3]), _month(match[2], value), int(match[1]), value)
    raise _Invalid(f"{value!r} is not a date this binding can read")


def _clock(hour: int, minute: int, value: Any) -> str:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise _Invalid(f"{value!r} is not a time of day")
    return f"{hour:02d}:{minute:02d}"


def _twelve_hour(hour: int, minute: int, meridiem: str, value: Any) -> str:
    if not 1 <= hour <= 12:
        raise _Invalid(f"{value!r} is not a 12-hour time")
    return _clock(hour % 12 + (12 if meridiem.lower() == "p" else 0), minute, value)


def _time(value: Any) -> str:
    text = _string_only(value, "time").strip()
    if match := _CLOCK.fullmatch(text):
        return _clock(int(match[1]), int(match[2]), value)
    if match := _CLOCK_12.fullmatch(text):
        return _twelve_hour(int(match[1]), int(match[2]), match[3], value)
    if match := _HOUR_12.fullmatch(text):
        return _twelve_hour(int(match[1]), 0, match[2], value)
    raise _Invalid(f"{value!r} is not a time this binding can read")


def _datetime(value: Any) -> str:
    """A date and a wall-clock time. A UTC offset is refused rather than converted: converting
    would need a time zone the binding does not have."""
    text = _string_only(value, "datetime").strip()
    if match := _ISO_DATETIME.fullmatch(text):
        return f"{_date(match[1])}T{_time(match[2])}"
    for separator in re.finditer(r",?\s+(?:at\s+)?", text):
        try:
            day = _date(text[: separator.start()])
            clock = _time(text[separator.end() :])
        except _Invalid:
            continue
        return f"{day}T{clock}"
    raise _Invalid(f"{value!r} is not a date and time this binding can read")


CONVERSIONS: dict[str, Callable[[Any], Any]] = {
    "string": _string,
    "integer": _integer,
    "number": _number,
    "boolean": _boolean,
    "lower": lambda value: _string_only(value, "lower").lower(),
    "upper": lambda value: _string_only(value, "upper").upper(),
    "trim": lambda value: _string_only(value, "trim").strip(),
    "digits": _digits,
    "date": _date,
    "time": _time,
    "datetime": _datetime,
}


def convert(name: str, value: Any) -> Any:
    """Apply one allowlisted conversion; `ValueError` when the value cannot be converted."""
    try:
        return CONVERSIONS[name](value)
    except _Invalid as exc:
        raise ValueError(str(exc)) from exc


# --- templates --------------------------------------------------------------------------------------


def _json_filter(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _alias_back(aliases: Mapping[str, Mapping[str, str]]) -> Callable[[Any, str], Any]:
    def alias_back(value: Any, table: str) -> Any:
        # Several agent values may name one environment value; the first in sorted order is
        # stable whatever order the document's keys arrive in. Presentation only, so an unknown
        # value is shown as it is.
        names = sorted(
            agent
            for agent, environment in aliases[table].items()
            if environment == value
        )
        return names[0] if names else value

    return alias_back


def _templates(
    aliases: Mapping[str, Mapping[str, str]],
) -> ImmutableSandboxedEnvironment:
    environment = ImmutableSandboxedEnvironment(
        undefined=StrictUndefined, autoescape=False
    )
    environment.filters["json"] = _json_filter
    environment.filters["alias_back"] = _alias_back(aliases)
    return environment


def _template_problems(
    source: str, aliases: Mapping[str, Mapping[str, str]]
) -> list[str]:
    environment = _templates(aliases)
    try:
        environment.from_string(source)
        tree = environment.parse(source)
    except TemplateError as exc:
        return [f"does not compile: {exc}"]
    problems = []
    for node in tree.find_all((nodes.Getattr, nodes.Getitem)):
        name = (
            node.attr
            if isinstance(node, nodes.Getattr)
            else getattr(node.arg, "value", None)
        )
        if isinstance(name, str) and name.startswith("_"):
            problems.append(f"reads the private attribute {name!r}")
    for node in tree.find_all(nodes.Filter):
        if node.name != "alias_back":
            continue
        table = node.args[0] if len(node.args) == 1 and not node.kwargs else None
        if not isinstance(table, nodes.Const) or table.value not in aliases:
            problems.append("alias_back needs one known alias table name")
    return problems


# --- static validation (§2.4.2) -------------------------------------------------------------------


def _first_field(expression: str) -> str | None:
    node = jmespath.compile(expression).parsed
    while node.get("type") != "field":
        children = node.get("children") or []
        if (
            node.get("type")
            not in {
                "subexpression",
                "index_expression",
                "projection",
                "value_projection",
                "filter_projection",
                "flatten",
                "pipe",
            }
            or not children
        ):
            return None
        node = children[0]
    return node["value"]


def _value_problems(
    where: str,
    value: Value,
    *,
    agent_parameters: set[str],
    earlier_steps: set[str],
    fixtures: Mapping[str, Any],
    aliases: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    for leaf in value.leaves():
        if leaf.form == "arg":
            try:
                first = _first_field(leaf.arg or "")
            except JMESPathError as exc:
                problems.append(
                    f"{where}: arg path {leaf.arg!r} does not compile: {exc}"
                )
            else:
                if first not in agent_parameters:
                    problems.append(
                        f"{where}: arg path {leaf.arg!r} does not start with a parameter of the "
                        "agent tool"
                    )
        elif leaf.form == "step":
            if leaf.step not in earlier_steps:
                problems.append(f"{where}: step {leaf.step!r} is not an earlier step")
            if leaf.path is not None:
                try:
                    jmespath.compile(leaf.path)
                except JMESPathError as exc:
                    problems.append(
                        f"{where}: path {leaf.path!r} does not compile: {exc}"
                    )
        elif leaf.form == "fixture" and leaf.fixture not in fixtures:
            problems.append(f"{where}: fixture {leaf.fixture!r} is unknown")
    for node in _nodes(value):
        if node.convert is not None and node.convert not in CONVERSIONS:
            problems.append(f"{where}: conversion {node.convert!r} is unknown")
        if node.alias is not None and node.alias not in aliases:
            problems.append(f"{where}: alias table {node.alias!r} is unknown")
    return problems


def _nodes(value: Value) -> list[Value]:
    children = [
        *(value.concat or ()),
        *(value.list_ or ()),
        *(value.object_ or {}).values(),
    ]
    return [value, *(node for child in children for node in _nodes(child))]


def binding_problems(
    *,
    agent_tools: Sequence[AgentTool],
    tools: Sequence[ToolBinding],
    fixtures: Mapping[str, Any],
    aliases: Mapping[str, Mapping[str, str]],
    environment_tools: Sequence[ToolDefinition],
) -> list[str]:
    """Every reason a `tool_routed` binding cannot be sealed (§2.4.2); empty when it can."""
    problems: list[str] = []
    declared = Counter(tool.name for tool in agent_tools)
    mapped = Counter(entry.agent_tool for entry in tools)
    for name in sorted(name for name, count in declared.items() if count > 1):
        problems.append(f"agent tool {name!r} is declared more than once")
    for name in sorted(name for name, count in mapped.items() if count > 1):
        problems.append(f"agent tool {name!r} is mapped more than once")
    for name in sorted(set(declared) - set(mapped)):
        problems.append(f"agent tool {name!r} has no binding entry")
    for name in sorted(set(mapped) - set(declared)):
        problems.append(f"binding entry {name!r} is not an agent tool")
    environment = {tool.name: tool for tool in environment_tools}
    parameters = {
        tool.name: set((tool.parameters.get("properties") or {}).keys())
        for tool in agent_tools
    }
    for entry in tools:
        if isinstance(entry, UnsupportedToolBinding):
            continue
        agent_parameters = parameters.get(entry.agent_tool, set())
        if not entry.steps:
            problems.append(f"{entry.agent_tool}: a mapped entry has no steps")
        seen: set[str] = set()
        for step in entry.steps:
            where = f"{entry.agent_tool}.{step.id}"
            if step.id in seen:
                problems.append(f"{where}: step id is not unique in the entry")
            tool = environment.get(step.tool)
            if tool is None:
                problems.append(
                    f"{where}: tool {step.tool!r} is not in the environment"
                )
            else:
                properties = set((tool.parameters.get("properties") or {}).keys())
                for name in sorted(set(step.arguments) - properties):
                    problems.append(
                        f"{where}: {name!r} is not a parameter of {step.tool}"
                    )
                for name in tool.parameters.get("required") or []:
                    value = step.arguments.get(name)
                    if value is None:
                        problems.append(
                            f"{where}: required argument {name!r} is missing"
                        )
                    elif any(node.form == "const" for node in _nodes(value)):
                        problems.append(
                            f"{where}: required argument {name!r} uses a const"
                        )
                    elif not any(
                        leaf.form in {"arg", "step", "fixture"}
                        for leaf in value.leaves()
                    ):
                        problems.append(
                            f"{where}: required argument {name!r} has no arg, step or fixture"
                        )
            for name, value in step.arguments.items():
                problems.extend(
                    _value_problems(
                        f"{where}.{name}",
                        value,
                        agent_parameters=agent_parameters,
                        earlier_steps=seen,
                        fixtures=fixtures,
                        aliases=aliases,
                    )
                )
            seen.add(step.id)
        if entry.response is None:
            continue
        where = f"{entry.agent_tool}.response"
        if entry.response.template is not None:
            problems.extend(
                f"{where}: template {problem}"
                for problem in _template_problems(entry.response.template, aliases)
            )
        if entry.response.value is not None:
            problems.extend(
                _value_problems(
                    where,
                    entry.response.value,
                    agent_parameters=agent_parameters,
                    earlier_steps=seen,
                    fixtures=fixtures,
                    aliases=aliases,
                )
            )
    return problems


# --- execution (§2.4.1) ----------------------------------------------------------------------------


@dataclass
class OperationRecord:
    step_id: str
    tool: str
    arguments: dict[str, Any]
    result: Any = None
    status: str = "ok"
    error: str = ""


@dataclass
class BoundResult:
    status: str
    content: str
    result: Any = None
    error: str = ""
    operations: list[OperationRecord] = field(default_factory=list)


_MISSING: Any = object()


@dataclass
class _Context:
    args: dict[str, Any]
    steps: dict[str, Any]
    fixtures: Mapping[str, Any]
    aliases: Mapping[str, Mapping[str, str]]


def _resolve(value: Value, context: _Context) -> Any:
    form = value.form
    if form == "arg":
        produced = jmespath.search(value.arg or "", context.args)
    elif form == "step":
        produced = context.steps[value.step]
        if value.path is not None:
            produced = jmespath.search(value.path, produced)
    elif form == "fixture":
        produced = copy.deepcopy(context.fixtures[value.fixture or ""])
    elif form == "const":
        produced = copy.deepcopy(value.const)
    elif form == "concat":
        parts = [_resolve(part, context) for part in value.concat or ()]
        produced = (
            _MISSING
            if any(part is _MISSING for part in parts)
            else (value.separator or "").join(_text(part) for part in parts)
        )
    elif form == "object":
        members = {
            key: _resolve(member, context)
            for key, member in (value.object_ or {}).items()
        }
        produced = {
            key: member for key, member in members.items() if member is not _MISSING
        }
    else:
        items = [_resolve(item, context) for item in value.list_ or ()]
        produced = _MISSING if any(item is _MISSING for item in items) else items
    if produced is None or produced is _MISSING:
        return _MISSING
    if value.convert is not None:
        produced = CONVERSIONS[value.convert](produced)
    if value.map_ is not None:
        key = _text(produced)
        if key not in value.map_:
            raise _Invalid(f"{key!r} is not one of {', '.join(sorted(value.map_))}")
        produced = copy.deepcopy(value.map_[key])
    if value.alias is not None:
        key = _text(produced)
        table = context.aliases[value.alias]
        if key not in table:
            raise _Invalid(f"{key!r} is not a known {value.alias} value")
        produced = table[key]
    return _MISSING if produced is None else produced


def _step_arguments(
    step: Step, tool: ToolDefinition, context: _Context
) -> dict[str, Any]:
    required = set(tool.parameters.get("required") or [])
    produced: dict[str, Any] = {}
    for name, value in step.arguments.items():
        try:
            argument = _resolve(value, context)
        except _Invalid as exc:
            raise _Invalid(f"{step.tool} argument {name!r}: {exc}") from exc
        if argument is not _MISSING:
            produced[name] = argument
    if missing := sorted(required - set(produced)):
        raise _Invalid(
            f"{step.tool} needs {', '.join(missing)}, which the call did not provide"
        )
    return produced


def _content(result: Any) -> str:
    return json.dumps(result, default=str)


def _failed(
    status: str, error: str, operations: list[OperationRecord] | None = None
) -> BoundResult:
    return BoundResult(status, error, None, error, operations or [])


def _operate(
    run_operation: Callable[[str, dict[str, Any]], Any],
    record: OperationRecord,
    operations: list[OperationRecord],
) -> bool:
    operations.append(record)
    try:
        record.result = run_operation(record.tool, dict(record.arguments))
    except ToolError as exc:
        record.status, record.error = "refused", str(exc)
        return False
    except Exception as exc:  # noqa: BLE001 - a handler breaking is reported, not raised
        record.status, record.error = "fault", f"{type(exc).__name__}: {exc}"
        return False
    return True


def _mode(binding: AgentBindingV1 | None) -> str:
    return (
        "canonical"
        if binding is None
        else str(getattr(binding.mode, "value", binding.mode))
    )


def run_bound_call(
    binding: AgentBindingV1 | None,
    agent_tool: str,
    arguments: Any,
    *,
    run_operation: Callable[[str, dict[str, Any]], Any],
    environment_tools: Sequence[ToolDefinition],
) -> BoundResult:
    """Run one agent call through `binding` (§2.4.1); `None` is a canonical binding.

    `run_operation(tool, arguments)` runs one handler inside the caller's transaction and raises
    `ToolError` for a refusal. The caller commits only when the status is `ok`.
    """
    environment = {tool.name: tool for tool in environment_tools}
    mode = _mode(binding)
    if mode == "canonical":
        if agent_tool not in environment:
            return _failed(
                "invalid",
                f"no such tool {agent_tool!r}; this environment has "
                f"{', '.join(sorted(environment))}",
            )
        if not isinstance(arguments, dict):
            return _failed("invalid", "arguments must be a JSON object")
        record = OperationRecord(agent_tool, agent_tool, dict(arguments))
        operations: list[OperationRecord] = []
        if not _operate(run_operation, record, operations):
            return _failed(record.status, record.error, operations)
        return BoundResult("ok", _content(record.result), record.result, "", operations)
    assert binding is not None
    if mode != "tool_routed":
        return _failed("invalid", f"a {mode} binding routes no tools")
    entry = next(
        (item for item in binding.tools if item.agent_tool == agent_tool), None
    )
    if entry is None:
        return _failed(
            "invalid",
            f"no such tool {agent_tool!r}; this agent has "
            f"{', '.join(sorted(item.agent_tool for item in binding.tools))}",
        )
    if isinstance(entry, UnsupportedToolBinding):
        return _failed("unsupported", entry.unsupported)
    if not isinstance(arguments, dict):
        return _failed("invalid", "arguments must be a JSON object")
    context = _Context(dict(arguments), {}, binding.fixtures, binding.aliases)
    operations = []
    try:
        last: Any = None
        for step in entry.steps:
            tool = environment.get(step.tool)
            if tool is None:
                return _failed(
                    "invalid", f"{step.tool} is not in the environment", operations
                )
            try:
                produced = _step_arguments(step, tool, context)
            except _Invalid as exc:
                return _failed("invalid", str(exc), operations)
            record = OperationRecord(step.id, step.tool, produced)
            if not _operate(run_operation, record, operations):
                return _failed(record.status, record.error, operations)
            last = context.steps[step.id] = record.result
        response = entry.response
        if response is None:
            return BoundResult("ok", _content(last), last, "", operations)
        if response.template is not None:
            content = (
                _templates(binding.aliases)
                .from_string(response.template)
                .render(
                    args=context.args,
                    steps=context.steps,
                    fixtures=dict(context.fixtures),
                )
            )
            return BoundResult("ok", content, last, "", operations)
        try:
            value = _resolve(response.value, context)  # type: ignore[arg-type]
        except _Invalid as exc:
            return _failed("invalid", f"response: {exc}", operations)
        if value is _MISSING:
            return _failed("invalid", "response: no value", operations)
        return BoundResult("ok", _content(value), value, "", operations)
    except Exception as exc:  # noqa: BLE001 - the binding itself broke: roll back, report
        return _failed("fault", f"binding: {type(exc).__name__}: {exc}", operations)
