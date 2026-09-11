"""Dashboard URL construction (Phase 14, ARCH §3) — the W&B "View run at …" link.

Route shapes are VERIFIED from the platform repo (RESEARCH §2.2), not guessed:
  observe project    /dashboard/observe/<project_id>
  observe trace      /dashboard/observe/<project_id>/trace/<trace_id>
App host inverts the verified ``api.→app.`` map (config-global.js:8).

Honesty rule (RESEARCH §2): never return a constructed *deep-link* unless we have
a real project id (from config or a successful name→id resolution). Otherwise
return the project-list view and LABEL it a fallback — a link that lands on a real
page beats a deep-link to an empty/404 view (the false-``synced`` failure mode).

Network-capable imports (``requests``) stay in-function (gate #72).
"""

from __future__ import annotations

from typing import Any

from ..config import DEFAULT_API_URL, AgentLearningConfig

_RESOLVE_PATH = "/tracer/project/list_projects/"


def _app_host(base_url: str) -> tuple[str, bool]:
    """Return (app_base, is_known_app_host). For the managed cloud, api.* → app.*.
    For a self-hosted/collector base, return it unchanged (we do not invent an
    app host that may not exist)."""

    from urllib.parse import urlparse  # lazy: this is a sanctioned network home

    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host.startswith("api."):
        app_host = "app." + host[len("api."):]
        return f"{parsed.scheme}://{app_host}", True
    # unknown / self-hosted: keep the base as the (best-effort) UI host
    return base_url.rstrip("/"), False


def fmt_trace_id(trace_id_hex: str) -> str:
    """Format a 32-char hex trace id to the dashed form the dashboard routes on
    (``default.spans.trace_id`` is stored dashed per the CH25 trace migration).
    The exact format is pinned empirically in the live step; this is the standard
    UUID dashing of the 128-bit id."""

    h = (trace_id_hex or "").strip().lower().replace("-", "")
    if len(h) != 32:
        return trace_id_hex
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _resolve_project_id(project_name: str, config: AgentLearningConfig) -> str | None:
    """Best-effort name→id via an authenticated GET. Any failure → None (caller
    falls back). Never raises into the run."""

    if not (config.api_key and config.secret_key):
        return None
    try:
        import requests

        base = (config.api_url or DEFAULT_API_URL).rstrip("/")
        resp = requests.get(
            f"{base}{_RESOLVE_PATH}",
            headers={"X-Api-Key": config.api_key, "X-Secret-Key": config.secret_key},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        rows = data.get("result") or data.get("results") or data.get("projects") or data
        if isinstance(rows, dict):
            rows = rows.get("projects") or rows.get("result") or []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("name") or row.get("project_name") or "") == project_name:
                pid = row.get("id") or row.get("project_id") or row.get("uuid")
                if pid:
                    return str(pid)
    except Exception:  # noqa: BLE001 — resolution is best-effort, degrade to fallback
        return None
    return None


def build_dashboard_url(
    project_name: str,
    trace_id_hex: str | None,
    *,
    config: AgentLearningConfig | None = None,
) -> dict[str, Any]:
    """Construct the dashboard URL for a keyed run. Returns
    ``{url, kind, project_name}`` where ``kind`` ∈
    {``deep_link``, ``project``, ``list_fallback``}."""

    config = config or AgentLearningConfig.from_env()
    base_url = (config.api_url or DEFAULT_API_URL).rstrip("/")
    app_base, _known = _app_host(base_url)

    # project id: explicit config first (no network), then best-effort resolution.
    project_id = config.project_id or _resolve_project_id(project_name, config)

    if project_id and trace_id_hex:
        return {
            "url": f"{app_base}/dashboard/observe/{project_id}/trace/{fmt_trace_id(trace_id_hex)}",
            "kind": "deep_link",
            "project_name": project_name,
        }
    if project_id:
        return {
            "url": f"{app_base}/dashboard/observe/{project_id}",
            "kind": "project",
            "project_name": project_name,
        }
    return {
        "url": f"{app_base}/dashboard/observe",
        "kind": "list_fallback",
        "project_name": project_name,
    }
