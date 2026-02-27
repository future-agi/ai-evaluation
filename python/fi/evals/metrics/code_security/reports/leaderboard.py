"""
Security Leaderboard for Model Comparison.

Compares multiple models on security benchmarks and generates
rankings and detailed analysis.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..benchmarks.types import BenchmarkResult, CWEBreakdown


@dataclass
class ModelEntry:
    """Entry for a model in the leaderboard."""

    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_func_at_k(self) -> float:
        """Average func@k across all results."""
        if not self.results:
            return 0.0
        return sum(r.func_at_k for r in self.results) / len(self.results)

    @property
    def avg_sec_at_k(self) -> float:
        """Average sec@k across all results."""
        if not self.results:
            return 0.0
        return sum(r.sec_at_k for r in self.results) / len(self.results)

    @property
    def avg_func_sec_at_k(self) -> float:
        """Average func-sec@k across all results."""
        if not self.results:
            return 0.0
        return sum(r.func_sec_at_k for r in self.results) / len(self.results)

    @property
    def total_tests(self) -> int:
        """Total tests across all results."""
        return sum(r.total_tests for r in self.results)


@dataclass
class CWEComparison:
    """Comparison of models on a specific CWE."""

    cwe_id: str
    model_scores: Dict[str, float]  # model_name -> secure_rate
    best_model: str
    worst_model: str
    average_score: float


@dataclass
class LanguageComparison:
    """Comparison of models on a specific language."""

    language: str
    model_scores: Dict[str, Dict[str, float]]  # model -> {metric: value}
    best_overall: str
    rankings: List[str]


@dataclass
class LeaderboardReport:
    """Complete leaderboard report with rankings and analysis."""

    # Metadata
    generated_at: datetime
    total_models: int
    total_tests: int

    # Rankings
    overall_ranking: List[str]  # Models ranked by func-sec@k
    func_ranking: List[str]  # Ranked by func@k
    sec_ranking: List[str]  # Ranked by sec@k

    # Detailed scores
    model_scores: Dict[str, Dict[str, float]]  # model -> {metric: value}

    # Breakdowns
    cwe_comparison: List[CWEComparison]
    language_comparison: List[LanguageComparison]

    # Recommendations
    recommendations: Dict[str, str]  # use_case -> recommended_model

    def to_markdown(self) -> str:
        """Export report as Markdown."""
        lines = [
            "# AI Code Security Leaderboard",
            "",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Models: {self.total_models} | Tests: {self.total_tests}*",
            "",
            "## Overall Rankings",
            "",
            "| Rank | Model | func@k | sec@k | func-sec@k |",
            "|------|-------|--------|-------|------------|",
        ]

        for i, model in enumerate(self.overall_ranking, 1):
            scores = self.model_scores[model]
            lines.append(
                f"| {i} | {model} | "
                f"{scores.get('func_at_k', 0):.1%} | "
                f"{scores.get('sec_at_k', 0):.1%} | "
                f"{scores.get('func_sec_at_k', 0):.1%} |"
            )

        lines.extend([
            "",
            "## Key Metrics Comparison",
            "",
            "### Functional Correctness (func@k)",
            "",
        ])
        for i, model in enumerate(self.func_ranking, 1):
            lines.append(
                f"{i}. **{model}**: {self.model_scores[model].get('func_at_k', 0):.1%}"
            )

        lines.extend([
            "",
            "### Security (sec@k)",
            "",
        ])
        for i, model in enumerate(self.sec_ranking, 1):
            lines.append(
                f"{i}. **{model}**: {self.model_scores[model].get('sec_at_k', 0):.1%}"
            )

        if self.cwe_comparison:
            lines.extend([
                "",
                "## CWE Breakdown (Top 5)",
                "",
                "| CWE | Best Model | Avg Score |",
                "|-----|------------|-----------|",
            ])
            for cwe in self.cwe_comparison[:5]:
                lines.append(
                    f"| {cwe.cwe_id} | {cwe.best_model} | {cwe.average_score:.1%} |"
                )

        if self.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                "",
            ])
            for use_case, model in self.recommendations.items():
                lines.append(f"- **{use_case}**: {model}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export report as JSON."""
        data = {
            "generated_at": self.generated_at.isoformat(),
            "total_models": self.total_models,
            "total_tests": self.total_tests,
            "rankings": {
                "overall": self.overall_ranking,
                "functional": self.func_ranking,
                "security": self.sec_ranking,
            },
            "model_scores": self.model_scores,
            "cwe_comparison": [
                {
                    "cwe_id": c.cwe_id,
                    "model_scores": c.model_scores,
                    "best_model": c.best_model,
                    "average_score": c.average_score,
                }
                for c in self.cwe_comparison
            ],
            "recommendations": self.recommendations,
        }
        return json.dumps(data, indent=2)

    def to_html(self) -> str:
        """Export report as HTML."""
        # Convert markdown to basic HTML
        md = self.to_markdown()
        html_lines = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>AI Code Security Leaderboard</title>",
            "<style>",
            "body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f4f4f4; }",
            ".rank-1 { background-color: #ffd700; }",
            ".rank-2 { background-color: #c0c0c0; }",
            ".rank-3 { background-color: #cd7f32; }",
            "</style>",
            "</head><body>",
        ]

        # Simple markdown to HTML conversion
        for line in md.split("\n"):
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("| "):
                # Table handling
                if "---" in line:
                    continue
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if cells[0] == "Rank":
                    html_lines.append("<table><thead><tr>")
                    for cell in cells:
                        html_lines.append(f"<th>{cell}</th>")
                    html_lines.append("</tr></thead><tbody>")
                else:
                    rank = cells[0] if cells else ""
                    rank_class = f"rank-{rank}" if rank in ["1", "2", "3"] else ""
                    html_lines.append(f'<tr class="{rank_class}">')
                    for cell in cells:
                        html_lines.append(f"<td>{cell}</td>")
                    html_lines.append("</tr>")
            elif line.startswith("- "):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith("*") and line.endswith("*"):
                html_lines.append(f"<p><em>{line[1:-1]}</em></p>")
            elif line.strip():
                html_lines.append(f"<p>{line}</p>")

        html_lines.append("</tbody></table></body></html>")
        return "\n".join(html_lines)


class SecurityLeaderboard:
    """
    Compare multiple models on security benchmarks.

    Usage:
        leaderboard = SecurityLeaderboard()

        # Add benchmark results for each model
        leaderboard.add_result("gpt-4", gpt4_benchmark_result)
        leaderboard.add_result("claude-3", claude_benchmark_result)
        leaderboard.add_result("codellama", codellama_benchmark_result)

        # Generate report
        report = leaderboard.generate_report()
        print(report.to_markdown())

        # Export
        with open("leaderboard.json", "w") as f:
            f.write(report.to_json())
    """

    def __init__(self):
        """Initialize empty leaderboard."""
        self.models: Dict[str, ModelEntry] = {}

    def add_result(
        self,
        model_name: str,
        result: BenchmarkResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a benchmark result for a model.

        Args:
            model_name: Name of the model
            result: Benchmark result to add
            metadata: Optional metadata about the model
        """
        if model_name not in self.models:
            self.models[model_name] = ModelEntry(
                name=model_name,
                metadata=metadata or {},
            )

        self.models[model_name].results.append(result)

        if metadata:
            self.models[model_name].metadata.update(metadata)

    def add_model_entry(self, entry: ModelEntry) -> None:
        """Add a complete model entry."""
        self.models[entry.name] = entry

    def get_rankings(self, metric: str = "func_sec_at_k") -> List[str]:
        """
        Get model rankings by metric.

        Args:
            metric: Metric to rank by (func_at_k, sec_at_k, func_sec_at_k)

        Returns:
            List of model names, best to worst
        """
        if metric == "func_at_k":
            key_fn = lambda m: m.avg_func_at_k
        elif metric == "sec_at_k":
            key_fn = lambda m: m.avg_sec_at_k
        else:
            key_fn = lambda m: m.avg_func_sec_at_k

        sorted_models = sorted(
            self.models.values(),
            key=key_fn,
            reverse=True,
        )
        return [m.name for m in sorted_models]

    def get_cwe_comparison(self) -> List[CWEComparison]:
        """Get per-CWE comparison across models."""
        # Collect all CWE data
        cwe_data: Dict[str, Dict[str, List[float]]] = {}

        for model_name, model in self.models.items():
            for result in model.results:
                for cwe_breakdown in result.cwe_breakdown:
                    cwe_id = cwe_breakdown.cwe_id
                    if cwe_id not in cwe_data:
                        cwe_data[cwe_id] = {}
                    if model_name not in cwe_data[cwe_id]:
                        cwe_data[cwe_id][model_name] = []
                    cwe_data[cwe_id][model_name].append(cwe_breakdown.secure_rate)

        # Create comparisons
        comparisons = []
        for cwe_id, model_scores in cwe_data.items():
            # Average scores for each model
            avg_scores = {
                model: sum(scores) / len(scores)
                for model, scores in model_scores.items()
            }

            best = max(avg_scores.items(), key=lambda x: x[1])
            worst = min(avg_scores.items(), key=lambda x: x[1])
            avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0

            comparisons.append(CWEComparison(
                cwe_id=cwe_id,
                model_scores=avg_scores,
                best_model=best[0],
                worst_model=worst[0],
                average_score=avg,
            ))

        # Sort by average score (worst CWEs first)
        return sorted(comparisons, key=lambda c: c.average_score)

    def get_language_comparison(self) -> List[LanguageComparison]:
        """Get per-language comparison across models."""
        # Collect language data
        lang_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

        for model_name, model in self.models.items():
            for result in model.results:
                lang = result.language
                if lang not in lang_data:
                    lang_data[lang] = {}
                if model_name not in lang_data[lang]:
                    lang_data[lang][model_name] = {
                        "func_at_k": [],
                        "sec_at_k": [],
                        "func_sec_at_k": [],
                    }
                lang_data[lang][model_name]["func_at_k"].append(result.func_at_k)
                lang_data[lang][model_name]["sec_at_k"].append(result.sec_at_k)
                lang_data[lang][model_name]["func_sec_at_k"].append(result.func_sec_at_k)

        # Create comparisons
        comparisons = []
        for lang, models in lang_data.items():
            # Average scores
            model_scores = {}
            for model, metrics in models.items():
                model_scores[model] = {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                }

            # Rank by func_sec_at_k
            rankings = sorted(
                model_scores.keys(),
                key=lambda m: model_scores[m].get("func_sec_at_k", 0),
                reverse=True,
            )

            comparisons.append(LanguageComparison(
                language=lang,
                model_scores=model_scores,
                best_overall=rankings[0] if rankings else "",
                rankings=rankings,
            ))

        return comparisons

    def generate_recommendations(self) -> Dict[str, str]:
        """Generate use-case recommendations."""
        recommendations = {}

        if not self.models:
            return recommendations

        # Best for security
        sec_ranking = self.get_rankings("sec_at_k")
        if sec_ranking:
            recommendations["Highest Security"] = sec_ranking[0]

        # Best overall (balanced)
        overall_ranking = self.get_rankings("func_sec_at_k")
        if overall_ranking:
            recommendations["Best Overall (func-sec@k)"] = overall_ranking[0]

        # Best functional
        func_ranking = self.get_rankings("func_at_k")
        if func_ranking:
            recommendations["Highest Correctness"] = func_ranking[0]

        # Best for specific CWEs
        cwe_comparison = self.get_cwe_comparison()
        if cwe_comparison:
            # Find model that handles the worst CWE best
            worst_cwe = cwe_comparison[0]
            recommendations[f"Best for {worst_cwe.cwe_id}"] = worst_cwe.best_model

        return recommendations

    def generate_report(self) -> LeaderboardReport:
        """
        Generate comprehensive leaderboard report.

        Returns:
            LeaderboardReport with rankings, comparisons, and recommendations
        """
        # Compute all rankings
        overall_ranking = self.get_rankings("func_sec_at_k")
        func_ranking = self.get_rankings("func_at_k")
        sec_ranking = self.get_rankings("sec_at_k")

        # Compute model scores
        model_scores = {}
        for model_name, model in self.models.items():
            model_scores[model_name] = {
                "func_at_k": model.avg_func_at_k,
                "sec_at_k": model.avg_sec_at_k,
                "func_sec_at_k": model.avg_func_sec_at_k,
                "total_tests": model.total_tests,
            }

        # Total tests
        total_tests = sum(m.total_tests for m in self.models.values())

        return LeaderboardReport(
            generated_at=datetime.now(),
            total_models=len(self.models),
            total_tests=total_tests,
            overall_ranking=overall_ranking,
            func_ranking=func_ranking,
            sec_ranking=sec_ranking,
            model_scores=model_scores,
            cwe_comparison=self.get_cwe_comparison(),
            language_comparison=self.get_language_comparison(),
            recommendations=self.generate_recommendations(),
        )

    def export_markdown(self) -> str:
        """Export leaderboard as Markdown."""
        return self.generate_report().to_markdown()

    def export_json(self) -> str:
        """Export leaderboard as JSON."""
        return self.generate_report().to_json()

    def export_html(self) -> str:
        """Export leaderboard as HTML."""
        return self.generate_report().to_html()

    @classmethod
    def from_results(
        cls,
        results: Dict[str, List[BenchmarkResult]],
    ) -> "SecurityLeaderboard":
        """
        Create leaderboard from results dictionary.

        Args:
            results: Dict mapping model names to their benchmark results

        Returns:
            SecurityLeaderboard instance
        """
        leaderboard = cls()
        for model_name, model_results in results.items():
            for result in model_results:
                leaderboard.add_result(model_name, result)
        return leaderboard
