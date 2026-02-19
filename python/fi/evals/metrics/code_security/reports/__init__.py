"""
Security Leaderboard and Reporting.

Provides tools for comparing models on security benchmarks
and generating detailed reports.

Features:
- Model comparison across func@k, sec@k, func-sec@k
- Per-CWE performance breakdown
- Per-language analysis
- Exportable reports (Markdown, JSON, HTML)
- Visualization support

Usage:
    from fi.evals.metrics.code_security.reports import (
        SecurityLeaderboard,
        LeaderboardReport,
        ModelEntry,
    )

    # Create leaderboard
    leaderboard = SecurityLeaderboard()

    # Add model results
    leaderboard.add_result("gpt-4", gpt4_result)
    leaderboard.add_result("claude-3", claude_result)

    # Generate report
    report = leaderboard.generate_report()
    print(report.to_markdown())
"""

from .leaderboard import (
    SecurityLeaderboard,
    ModelEntry,
    LeaderboardReport,
    CWEComparison,
    LanguageComparison,
)

from .generator import (
    ReportGenerator,
    generate_security_report,
)


__all__ = [
    # Leaderboard
    "SecurityLeaderboard",
    "ModelEntry",
    "LeaderboardReport",
    "CWEComparison",
    "LanguageComparison",
    # Generator
    "ReportGenerator",
    "generate_security_report",
]
