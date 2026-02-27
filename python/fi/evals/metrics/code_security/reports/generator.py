"""
Security Report Generator.

Generates detailed security evaluation reports for individual models
or comparisons.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import json

from ..benchmarks.types import BenchmarkResult, CWEBreakdown
from ..types import SecurityFinding, Severity


@dataclass
class SecurityReport:
    """Complete security evaluation report."""

    # Metadata
    title: str
    generated_at: datetime
    model_name: str
    language: str

    # Summary
    overall_score: float
    func_at_k: float
    sec_at_k: float
    func_sec_at_k: float

    # Details
    total_samples: int
    secure_samples: int
    vulnerable_samples: int

    # Findings
    total_findings: int
    findings_by_severity: Dict[str, int]
    findings_by_cwe: Dict[str, int]
    top_vulnerabilities: List[Dict[str, Any]]

    # Breakdown
    cwe_breakdown: List[CWEBreakdown]

    # Recommendations
    improvements: List[str]

    def to_markdown(self) -> str:
        """Export report as Markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"**Model:** {self.model_name}",
            f"**Language:** {self.language}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Overall Security Score:** {self.overall_score:.1%}",
            f"- **func@k:** {self.func_at_k:.1%}",
            f"- **sec@k:** {self.sec_at_k:.1%}",
            f"- **func-sec@k:** {self.func_sec_at_k:.1%}",
            "",
            f"- **Total Samples:** {self.total_samples}",
            f"- **Secure:** {self.secure_samples} ({self.secure_samples/self.total_samples:.1%})" if self.total_samples else "",
            f"- **Vulnerable:** {self.vulnerable_samples}",
            "",
            "## Vulnerability Summary",
            "",
            f"**Total Findings:** {self.total_findings}",
            "",
            "### By Severity",
            "",
        ]

        for severity, count in sorted(
            self.findings_by_severity.items(),
            key=lambda x: ["critical", "high", "medium", "low", "info"].index(x[0])
            if x[0] in ["critical", "high", "medium", "low", "info"]
            else 5,
        ):
            emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
                "info": "ℹ️",
            }.get(severity, "")
            lines.append(f"- {emoji} **{severity.upper()}:** {count}")

        lines.extend([
            "",
            "### By CWE",
            "",
        ])

        for cwe, count in sorted(
            self.findings_by_cwe.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]:  # Top 10
            lines.append(f"- **{cwe}:** {count}")

        if self.top_vulnerabilities:
            lines.extend([
                "",
                "## Top Vulnerabilities",
                "",
            ])
            for i, vuln in enumerate(self.top_vulnerabilities[:5], 1):
                lines.append(f"### {i}. {vuln.get('cwe_id', 'Unknown')} - {vuln.get('type', 'Unknown')}")
                lines.append(f"**Severity:** {vuln.get('severity', 'Unknown')}")
                if vuln.get('description'):
                    lines.append(f"**Description:** {vuln.get('description')}")
                if vuln.get('count'):
                    lines.append(f"**Occurrences:** {vuln.get('count')}")
                lines.append("")

        if self.improvements:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for improvement in self.improvements:
                lines.append(f"- {improvement}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps({
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "model_name": self.model_name,
            "language": self.language,
            "summary": {
                "overall_score": self.overall_score,
                "func_at_k": self.func_at_k,
                "sec_at_k": self.sec_at_k,
                "func_sec_at_k": self.func_sec_at_k,
            },
            "samples": {
                "total": self.total_samples,
                "secure": self.secure_samples,
                "vulnerable": self.vulnerable_samples,
            },
            "findings": {
                "total": self.total_findings,
                "by_severity": self.findings_by_severity,
                "by_cwe": self.findings_by_cwe,
            },
            "top_vulnerabilities": self.top_vulnerabilities,
            "recommendations": self.improvements,
        }, indent=2)


class ReportGenerator:
    """
    Generate security evaluation reports.

    Usage:
        generator = ReportGenerator()

        # From benchmark result
        report = generator.from_benchmark_result(result, "gpt-4")

        # From raw findings
        report = generator.from_findings(findings, "claude-3")

        print(report.to_markdown())
    """

    def from_benchmark_result(
        self,
        result: BenchmarkResult,
        model_name: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SecurityReport:
        """
        Generate report from benchmark result.

        Args:
            result: Benchmark result
            model_name: Override model name
            title: Custom report title

        Returns:
            SecurityReport
        """
        # Compute findings breakdown
        findings_by_severity: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        findings_by_cwe: Dict[str, int] = {}

        for cwe in result.cwe_breakdown:
            findings_by_cwe[cwe.cwe_id] = cwe.vulnerable_count

        # Compute top vulnerabilities
        top_vulns = []
        for cwe in sorted(
            result.cwe_breakdown,
            key=lambda c: c.vulnerable_count,
            reverse=True,
        )[:10]:
            top_vulns.append({
                "cwe_id": cwe.cwe_id,
                "count": cwe.vulnerable_count,
                "secure_rate": cwe.secure_rate,
            })

        # Generate improvements
        improvements = self._generate_improvements(result)

        return SecurityReport(
            title=title or f"Security Evaluation Report - {result.benchmark_name}",
            generated_at=datetime.now(),
            model_name=model_name or result.model_name,
            language=result.language,
            overall_score=result.overall_security_score,
            func_at_k=result.func_at_k,
            sec_at_k=result.sec_at_k,
            func_sec_at_k=result.func_sec_at_k,
            total_samples=result.total_tests,
            secure_samples=int(result.total_tests * result.sec_at_k),
            vulnerable_samples=int(result.total_tests * (1 - result.sec_at_k)),
            total_findings=sum(c.vulnerable_count for c in result.cwe_breakdown),
            findings_by_severity=findings_by_severity,
            findings_by_cwe=findings_by_cwe,
            top_vulnerabilities=top_vulns,
            cwe_breakdown=result.cwe_breakdown,
            improvements=improvements,
        )

    def from_findings(
        self,
        findings: List[SecurityFinding],
        model_name: str,
        language: str = "python",
        total_samples: int = 1,
        title: Optional[str] = None,
    ) -> SecurityReport:
        """
        Generate report from raw findings.

        Args:
            findings: List of security findings
            model_name: Name of the model
            language: Programming language
            total_samples: Total number of samples evaluated
            title: Custom report title

        Returns:
            SecurityReport
        """
        # Count by severity
        findings_by_severity: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        for finding in findings:
            severity = finding.severity.value.lower()
            if severity in findings_by_severity:
                findings_by_severity[severity] += 1

        # Count by CWE
        findings_by_cwe: Dict[str, int] = {}
        for finding in findings:
            cwe = finding.cwe_id
            findings_by_cwe[cwe] = findings_by_cwe.get(cwe, 0) + 1

        # Top vulnerabilities
        top_vulns = []
        for cwe, count in sorted(
            findings_by_cwe.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]:
            top_vulns.append({
                "cwe_id": cwe,
                "count": count,
                "type": next(
                    (f.vulnerability_type for f in findings if f.cwe_id == cwe),
                    "unknown",
                ),
            })

        # Compute scores
        has_critical = findings_by_severity.get("critical", 0) > 0
        has_high = findings_by_severity.get("high", 0) > 0

        if has_critical:
            overall_score = 0.0
        elif has_high:
            overall_score = 0.3
        elif len(findings) > 0:
            overall_score = 0.6
        else:
            overall_score = 1.0

        sec_at_k = 1.0 if not findings else 0.0

        # Generate improvements
        improvements = []
        if findings_by_cwe.get("CWE-89"):
            improvements.append("Use parameterized queries to prevent SQL injection")
        if findings_by_cwe.get("CWE-78"):
            improvements.append("Use subprocess with array arguments instead of shell=True")
        if findings_by_cwe.get("CWE-79"):
            improvements.append("Escape user input before rendering in HTML")
        if findings_by_cwe.get("CWE-798"):
            improvements.append("Use environment variables for credentials")
        if findings_by_cwe.get("CWE-327"):
            improvements.append("Use strong cryptographic algorithms (SHA-256+)")
        if findings_by_cwe.get("CWE-502"):
            improvements.append("Use safe deserialization methods (json, yaml.safe_load)")

        return SecurityReport(
            title=title or f"Security Evaluation Report - {model_name}",
            generated_at=datetime.now(),
            model_name=model_name,
            language=language,
            overall_score=overall_score,
            func_at_k=1.0,  # Unknown from findings alone
            sec_at_k=sec_at_k,
            func_sec_at_k=sec_at_k,
            total_samples=total_samples,
            secure_samples=total_samples if not findings else 0,
            vulnerable_samples=0 if not findings else total_samples,
            total_findings=len(findings),
            findings_by_severity=findings_by_severity,
            findings_by_cwe=findings_by_cwe,
            top_vulnerabilities=top_vulns,
            cwe_breakdown=[],
            improvements=improvements,
        )

    def _generate_improvements(self, result: BenchmarkResult) -> List[str]:
        """Generate improvement recommendations based on results."""
        improvements = []

        # Security gap
        if result.sec_at_k - result.func_sec_at_k > 0.1:
            improvements.append(
                "Focus on joint security+correctness - many samples are "
                "secure but incorrect, or vice versa"
            )

        # Low security score
        if result.sec_at_k < 0.5:
            improvements.append(
                "Security awareness training needed - less than half of "
                "samples are secure"
            )

        # CWE-specific
        for cwe in result.cwe_breakdown:
            if cwe.secure_rate < 0.3:
                cwe_advice = {
                    "CWE-89": "SQL injection is a major weakness - implement parameterized query training",
                    "CWE-78": "Command injection prevalent - train on subprocess best practices",
                    "CWE-79": "XSS vulnerabilities common - emphasize output encoding",
                    "CWE-798": "Credential handling poor - use environment variables",
                    "CWE-327": "Weak crypto usage - update to modern algorithms",
                    "CWE-502": "Deserialization issues - use safe parsing methods",
                }
                if cwe.cwe_id in cwe_advice:
                    improvements.append(cwe_advice[cwe.cwe_id])

        return improvements[:5]  # Limit to 5 recommendations


def generate_security_report(
    result: BenchmarkResult,
    model_name: Optional[str] = None,
    format: str = "markdown",
) -> str:
    """
    Convenience function to generate a security report.

    Args:
        result: Benchmark result
        model_name: Optional model name override
        format: Output format (markdown, json)

    Returns:
        Formatted report string
    """
    generator = ReportGenerator()
    report = generator.from_benchmark_result(result, model_name)

    if format == "json":
        return report.to_json()
    else:
        return report.to_markdown()
