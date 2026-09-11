from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pydantic import BaseModel, Field


THEME_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "agent_benchmarks": ("benchmark", "arena", "bench", "evaluation", "eval", "suite"),
    "world_simulation": ("environment", "simulator", "simulation", "world", "state", "constraint"),
    "tool_use": ("tool", "function calling", "api", "action", "actuator", "mcp"),
    "adversarial_security": (
        "attack",
        "prompt injection",
        "jailbreak",
        "security",
        "safety",
        "poison",
        "hijack",
        "exfiltration",
        "red team",
        "red-team",
    ),
    "memory_learning": ("memory", "reflection", "experience", "episodic", "retrieval"),
    "multi_agent": ("multi-agent", "multi agent", "collaboration", "debate", "society", "role"),
    "optimization": ("optimization", "optimizer", "search", "evolution", "bayesian", "pareto", "prompt"),
    "browser_computer_use": ("browser", "web", "computer use", "gui", "desktop", "osworld"),
    "software_engineering": ("software", "code", "coding", "programming", "swe", "repository", "debug"),
    "long_horizon": ("long-horizon", "long horizon", "multi-step", "planning", "trajectory", "procedure"),
    "observability_traces": ("trace", "trajectory", "telemetry", "log", "span", "monitor"),
    "voice_agents": ("voice", "audio", "speech", "webrtc", "sip", "phone", "turn-taking", "vad"),
}


THEME_IMPLEMENTATION_SIGNALS: Dict[str, Dict[str, List[str]]] = {
    "agent_benchmarks": {
        "components": ["harness", "environment", "evaluator"],
        "config_paths": ["evaluation", "trajectory_templates", "environment.fixtures"],
        "metrics": ["eval_coverage", "agent_goal_accuracy"],
    },
    "world_simulation": {
        "components": ["world", "environment", "harness"],
        "config_paths": ["world.contract", "environment.world", "world.transitions", "world.invariants"],
        "metrics": ["world_contract_coverage", "world_contract_quality"],
    },
    "tool_use": {
        "components": ["tools", "policy", "security"],
        "config_paths": ["tools", "tool_schemas", "tools.permissions", "tools.allowlist"],
        "metrics": ["tool_call_accuracy", "tool_outcome", "tool_fault_tolerance"],
    },
    "adversarial_security": {
        "components": ["security", "policy", "environment", "evaluator"],
        "config_paths": ["security.attack_pack", "red_team.campaign", "evaluation.red_team_campaign_quality"],
        "metrics": ["adversarial_resilience", "red_team_campaign_quality"],
    },
    "memory_learning": {
        "components": ["memory", "autonomy", "framework"],
        "config_paths": ["memory", "memory.isolation", "memory.cross_trial"],
        "metrics": ["memory_correctness", "cross_trial_memory_skill"],
    },
    "multi_agent": {
        "components": ["multi_agent", "orchestration", "planner"],
        "config_paths": ["multi_agent.roles", "multi_agent.handoffs", "orchestration.trace"],
        "metrics": ["multi_agent_coordination_quality", "orchestration_flow_quality"],
    },
    "optimization": {
        "components": ["harness", "evaluator", "multi_agent"],
        "config_paths": ["optimizer.strategy", "optimizer.governance", "optimizer.trace"],
        "metrics": ["optimizer_trace_quality", "trial_reliability"],
    },
    "browser_computer_use": {
        "components": ["browser", "cua", "perception"],
        "config_paths": ["browser.trace", "browser.actions", "browser.screenshot_diff"],
        "metrics": ["browser_action_outcome", "browser_grounding_quality"],
    },
    "software_engineering": {
        "components": ["implementation", "tools", "environment"],
        "config_paths": ["workspace_run.checkout", "execution.commands", "execution.logs"],
        "metrics": ["workspace_run_quality", "tool_outcome"],
    },
    "long_horizon": {
        "components": ["autonomy", "planner", "memory"],
        "config_paths": ["autonomy.loop", "planner", "trajectory.steps"],
        "metrics": ["autonomy_loop_quality", "trial_reliability"],
    },
    "observability_traces": {
        "components": ["framework", "harness", "evaluator"],
        "config_paths": ["framework.trace", "observability.replay", "futureagi.regression_replay"],
        "metrics": ["framework_trace_coverage", "observability_replay_quality"],
    },
    "voice_agents": {
        "components": ["voice", "streaming", "perception"],
        "config_paths": ["voice.trace", "voice.webrtc", "voice.sip", "voice.timing_distribution"],
        "metrics": ["voice_trace_coverage", "voice_timing_distribution_quality"],
    },
}


RED_TEAM_KEYWORDS: Dict[str, tuple[str, tuple[str, ...]]] = {
    "prompt_injection": ("attack_type", ("prompt injection", "indirect prompt", "instruction manipulation")),
    "jailbreak": ("attack_type", ("jailbreak", "policy bypass")),
    "memory_poisoning": ("attack_type", ("memory poison", "poisoned memory", "context poisoning")),
    "tool_abuse": ("attack_type", ("tool abuse", "tool misuse", "actuator", "unauthorized tool")),
    "credential_exfiltration": ("attack_type", ("credential", "api key", "secret", "token", "exfiltration")),
    "distributed_attack": ("attack_type", ("distributed attack", "multi-account", "across accounts")),
    "reward_hacking": ("attack_type", ("reward hack", "benchmark hack", "score exploit")),
    "social_engineering": ("attack_type", ("social engineering", "influence", "persuasion")),
    "hallucination_to_action": ("attack_type", ("hallucination-to-action", "unsupported claim", "unsafe execution")),
    "owasp_llm_top_10": ("taxonomy", ("owasp llm", "llm top 10", "llm01")),
    "owasp_agentic_ai": ("taxonomy", ("agentic ai", "aivss", "agentic security", "asi")),
    "mcp_security": ("taxonomy", ("model context protocol", "mcp", "mcp server", "mcp tool")),
    "mitre_atlas": ("taxonomy", ("mitre atlas", "atlas")),
    "tool": ("surface", ("tool", "function", "api", "actuator")),
    "memory": ("surface", ("memory", "retrieval", "rag", "context")),
    "browser": ("surface", ("browser", "web", "dom", "computer use", "desktop")),
    "voice": ("surface", ("voice", "audio", "speech", "phone", "sip", "webrtc")),
    "code": ("surface", ("code", "coding", "software", "repository", "shell")),
    "multi_agent": ("surface", ("multi-agent", "multi agent", "subagent", "distributed agent")),
    "observability_log": ("surface", ("log", "trace", "telemetry", "monitor")),
    "garak": ("framework", ("garak",)),
    "pyrit": ("framework", ("pyrit",)),
    "inspect": ("framework", ("inspect ai", "inspect_aisi", "inspect")),
}


class ResearchPaper(BaseModel):
    """A normalized scholarly-paper record used by the agent-trinity roadmap."""

    id: str
    title: str
    summary: str = ""
    authors: List[str] = Field(default_factory=list)
    published: str = ""
    updated: str = ""
    primary_category: str = ""
    categories: List[str] = Field(default_factory=list)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    query_tags: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    note: str = ""
    doi: str = ""
    openalex_id: str = ""
    pdf_url: str = ""
    implementation_signals: Dict[str, List[str]] = Field(default_factory=dict)
    red_team_signals: Dict[str, List[str]] = Field(default_factory=dict)


class ResearchCorpusSummary(BaseModel):
    """A compact research-to-implementation summary for roadmap planning."""

    paper_count: int
    recent_paper_count: int
    theme_counts: Dict[str, int] = Field(default_factory=dict)
    query_counts: Dict[str, int] = Field(default_factory=dict)
    implementation_signals: Dict[str, List[str]] = Field(default_factory=dict)
    red_team_campaign: Dict[str, List[str]] = Field(default_factory=dict)
    deep_read_queue: List[ResearchPaper] = Field(default_factory=list)


def normalize_research_paper(payload: Mapping[str, Any]) -> ResearchPaper:
    """Normalize arXiv/OpenAlex/corpus metadata into a stable paper note."""

    raw = dict(payload)
    title = _normalize_space(raw.get("title") or raw.get("display_name") or "")
    summary = _normalize_space(raw.get("summary") or raw.get("abstract") or "")
    paper_id = str(raw.get("id") or raw.get("doi") or raw.get("openalex_id") or title)
    themes = _dedupe(
        [
            _normalize_key(theme)
            for theme in _as_list(raw.get("themes"))
            if _normalize_key(theme)
        ]
    )
    if not themes:
        themes = infer_research_themes({**raw, "title": title, "summary": summary})
    note = str(raw.get("note") or research_note_for({"title": title, "summary": summary, "themes": themes}))
    implementation_signals = merge_implementation_signals(themes)
    red_team_signals = infer_red_team_signals({**raw, "title": title, "summary": summary, "themes": themes})
    return ResearchPaper(
        id=paper_id,
        title=title,
        summary=summary,
        authors=[str(author) for author in _as_list(raw.get("authors")) if str(author).strip()],
        published=str(raw.get("published") or raw.get("publication_date") or "")[:10],
        updated=str(raw.get("updated") or raw.get("updated_date") or "")[:10],
        primary_category=str(raw.get("primary_category") or raw.get("category") or ""),
        categories=[str(category) for category in _as_list(raw.get("categories")) if str(category).strip()],
        links=[dict(link) for link in _as_list(raw.get("links")) if isinstance(link, Mapping)],
        query_tags=_dedupe(str(tag) for tag in _as_list(raw.get("query_tags")) if str(tag).strip()),
        themes=themes,
        note=note,
        doi=str(raw.get("doi") or ""),
        openalex_id=str(raw.get("openalex_id") or ""),
        pdf_url=str(raw.get("pdf_url") or ""),
        implementation_signals=implementation_signals,
        red_team_signals=red_team_signals,
    )


def load_research_papers(source: str | Path | Iterable[Mapping[str, Any]]) -> List[ResearchPaper]:
    """Load normalized papers from JSONL, JSON, or in-memory mappings."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            rows = payload.get("papers", payload) if isinstance(payload, Mapping) else payload
    else:
        rows = list(source)
    return [normalize_research_paper(row) for row in rows if isinstance(row, Mapping)]


def build_agent_research_corpus(
    papers: Iterable[Mapping[str, Any] | ResearchPaper],
    *,
    deep_read_limit: int = 50,
    recent_year: str = "2026",
) -> ResearchCorpusSummary:
    """Summarize a paper corpus into implementation and red-team roadmap signals."""

    normalized = [
        paper if isinstance(paper, ResearchPaper) else normalize_research_paper(paper)
        for paper in papers
    ]
    theme_counts: Counter[str] = Counter(theme for paper in normalized for theme in paper.themes)
    query_counts: Counter[str] = Counter(tag for paper in normalized for tag in paper.query_tags)
    implementation_signals = merge_implementation_signals(
        theme for paper in normalized for theme in paper.themes
    )
    return ResearchCorpusSummary(
        paper_count=len(normalized),
        recent_paper_count=sum(1 for paper in normalized if paper.published.startswith(recent_year)),
        theme_counts=dict(sorted(theme_counts.items())),
        query_counts=dict(sorted(query_counts.items())),
        implementation_signals=implementation_signals,
        red_team_campaign=map_research_to_red_team_campaign(normalized),
        deep_read_queue=build_deep_read_queue(normalized, limit=deep_read_limit),
    )


def build_deep_read_queue(
    papers: Iterable[Mapping[str, Any] | ResearchPaper],
    *,
    limit: int = 50,
    required_themes: Optional[Iterable[str]] = None,
) -> List[ResearchPaper]:
    """Rank papers for manual deep reading by recency and implementation relevance."""

    normalized = [
        paper if isinstance(paper, ResearchPaper) else normalize_research_paper(paper)
        for paper in papers
    ]
    required = {_normalize_key(theme) for theme in required_themes or [] if _normalize_key(theme)}
    if required:
        normalized = [paper for paper in normalized if required & set(paper.themes)]
    deduped: Dict[str, ResearchPaper] = {}
    for paper in sorted(normalized, key=_research_relevance_score, reverse=True):
        key = _normalize_title_key(paper.title) or paper.id
        if key in deduped:
            continue
        deduped[key] = paper
    return list(deduped.values())[:limit]


def map_research_to_red_team_campaign(
    papers: Iterable[Mapping[str, Any] | ResearchPaper],
) -> Dict[str, List[str]]:
    """Map paper evidence into `red_team_campaign` requirement fields."""

    taxonomies: set[str] = set()
    attack_types: set[str] = set()
    surfaces: set[str] = set()
    frameworks: set[str] = set()
    source_paper_ids: set[str] = set()
    for item in papers:
        paper = item if isinstance(item, ResearchPaper) else normalize_research_paper(item)
        signals = paper.red_team_signals or infer_red_team_signals(_paper_mapping(paper))
        if any(signals.values()):
            source_paper_ids.add(paper.id)
        taxonomies.update(signals.get("taxonomies", []))
        attack_types.update(signals.get("attack_types", []))
        surfaces.update(signals.get("surfaces", []))
        frameworks.update(signals.get("frameworks", []))

    if attack_types and not taxonomies:
        taxonomies.add("owasp_llm_top_10")
    required_evidence = {
        "target",
        "attack_pack",
        "scenario",
        "multi_turn",
        "run",
        "finding",
        "artifact",
        "mitigation",
        "observability",
    }
    if frameworks:
        required_evidence.add("framework_run")
    return {
        "required_taxonomies": sorted(taxonomies),
        "required_attack_types": sorted(attack_types),
        "required_surfaces": sorted(surfaces),
        "required_frameworks": sorted(frameworks),
        "required_campaign_evidence": sorted(required_evidence),
        "source_paper_ids": sorted(source_paper_ids),
    }


def infer_research_themes(payload: Mapping[str, Any]) -> List[str]:
    haystack = _paper_haystack(payload)
    themes = [
        theme
        for theme, keywords in THEME_KEYWORDS.items()
        if any(keyword in haystack for keyword in keywords)
    ]
    return themes or ["general_agent_research"]


def infer_red_team_signals(payload: Mapping[str, Any]) -> Dict[str, List[str]]:
    haystack = _paper_haystack(payload)
    by_kind: Dict[str, set[str]] = {
        "taxonomies": set(),
        "attack_types": set(),
        "surfaces": set(),
        "frameworks": set(),
    }
    for key, (kind, keywords) in RED_TEAM_KEYWORDS.items():
        if not any(keyword in haystack for keyword in keywords):
            continue
        if kind == "taxonomy":
            by_kind["taxonomies"].add(key)
        elif kind == "attack_type":
            by_kind["attack_types"].add(key)
        elif kind == "surface":
            by_kind["surfaces"].add(key)
        elif kind == "framework":
            by_kind["frameworks"].add(key)
    if "adversarial_security" in {_normalize_key(theme) for theme in _as_list(payload.get("themes"))}:
        by_kind["taxonomies"].add("owasp_llm_top_10")
    return {key: sorted(values) for key, values in by_kind.items()}


def research_note_for(payload: Mapping[str, Any]) -> str:
    themes = {_normalize_key(theme) for theme in _as_list(payload.get("themes") or infer_research_themes(payload))}
    summary = str(payload.get("summary") or "")
    title = str(payload.get("title") or "")
    first_sentence = re.split(r"(?<=[.!?])\s+", summary.strip())[0] if summary.strip() else title
    implication = "Track as background evidence for the agent-trinity roadmap."
    if "adversarial_security" in themes:
        implication = "Use for red-team campaigns, threat surfaces, canaries, and mitigation gates."
    elif "world_simulation" in themes:
        implication = "Use for replayable environments, state contracts, and simulator fidelity."
    elif "optimization" in themes:
        implication = "Use for candidate search, diagnosis, allocation, and config optimization."
    elif "observability_traces" in themes:
        implication = "Use for trace capture, replay packs, diagnosis, and production regression loops."
    elif "multi_agent" in themes:
        implication = "Use for role allocation and coordination only when metrics show multi-agent failures."
    return _normalize_space(f"{first_sentence} {implication}")


def merge_implementation_signals(themes: Iterable[str]) -> Dict[str, List[str]]:
    merged: Dict[str, set[str]] = {"components": set(), "config_paths": set(), "metrics": set()}
    for theme in themes:
        signals = THEME_IMPLEMENTATION_SIGNALS.get(_normalize_key(theme), {})
        for key in merged:
            merged[key].update(str(item) for item in signals.get(key, []) if str(item).strip())
    return {key: sorted(values) for key, values in merged.items()}


def research_summary_markdown(summary: ResearchCorpusSummary) -> str:
    """Render a compact research summary suitable for internal docs."""

    lines = [
        "# Agent Research Corpus Implementation Summary",
        "",
        f"- Papers: {summary.paper_count}",
        f"- Recent papers: {summary.recent_paper_count}",
        f"- Theme count: {len(summary.theme_counts)}",
        "",
        "## Top Themes",
        "",
    ]
    for theme, count in sorted(summary.theme_counts.items(), key=lambda item: item[1], reverse=True)[:12]:
        lines.append(f"- {theme}: {count}")
    lines.extend(["", "## Implementation Signals", ""])
    for key in ("components", "config_paths", "metrics"):
        values = summary.implementation_signals.get(key, [])
        lines.append(f"- {key}: {', '.join(values)}")
    lines.extend(["", "## Red-Team Campaign Map", ""])
    for key in (
        "required_taxonomies",
        "required_attack_types",
        "required_surfaces",
        "required_frameworks",
        "required_campaign_evidence",
    ):
        values = summary.red_team_campaign.get(key, [])
        lines.append(f"- {key}: {', '.join(values)}")
    lines.extend(["", "## Deep Read Queue", ""])
    for index, paper in enumerate(summary.deep_read_queue[:20], start=1):
        lines.append(f"{index}. {paper.published} [{paper.title}]({paper.id})")
        lines.append(f"   - Themes: {', '.join(paper.themes)}")
        lines.append(f"   - Note: {paper.note}")
    return "\n".join(lines) + "\n"


def _research_relevance_score(paper: ResearchPaper) -> tuple[int, int, int, int, int, str]:
    themes = set(paper.themes)
    priority = {
        "adversarial_security",
        "world_simulation",
        "agent_benchmarks",
        "optimization",
        "long_horizon",
        "observability_traces",
        "tool_use",
        "multi_agent",
        "voice_agents",
    }
    red_team_signal_count = sum(len(values) for values in paper.red_team_signals.values())
    year_score = 3 if paper.published.startswith("2026") else 2 if paper.published.startswith("2025") else 1
    return (
        _agent_focus_score(paper),
        red_team_signal_count,
        len(themes & priority),
        year_score,
        len(paper.query_tags),
        paper.published,
    )


def _paper_haystack(payload: Mapping[str, Any]) -> str:
    values: List[str] = [
        str(payload.get("title") or ""),
        str(payload.get("summary") or payload.get("abstract") or ""),
        " ".join(str(item) for item in _as_list(payload.get("categories"))),
        " ".join(str(item) for item in _as_list(payload.get("query_tags"))),
    ]
    return " ".join(values).lower()


def _agent_focus_score(paper: ResearchPaper) -> int:
    title_and_tags = f"{paper.title} {' '.join(paper.query_tags)}".lower()
    summary_and_categories = f"{paper.summary} {' '.join(paper.categories)}".lower()
    focus_terms = [
        "agent",
        "agentic",
        "autonomous",
        "llm agent",
        "large language model agent",
        "multi-agent",
        "multi agent",
        "model context protocol",
        "mcp",
        "prompt injection",
        "red team",
        "red-team",
        "jailbreak",
        "tool-use",
        "tool use",
        "orchestration trace",
        "trajectory",
        "computer use",
    ]
    title_score = sum(1 for term in focus_terms if term in title_and_tags)
    body_score = sum(1 for term in focus_terms if term in summary_and_categories)
    return title_score * 4 + min(body_score, 4)


def _paper_mapping(paper: ResearchPaper) -> Dict[str, Any]:
    if hasattr(paper, "model_dump"):
        return paper.model_dump()
    return paper.dict()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_title_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
