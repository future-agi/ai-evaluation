"""
Pattern-based Security Judge.

Fast, deterministic vulnerability detection using regex patterns
and AST analysis. Designed for <10ms execution time.
"""

import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from .base import BaseJudge, JudgeResult, JudgeFinding
from ..types import Severity, CodeLocation


@dataclass
class PatternRule:
    """A single pattern rule for detection."""

    cwe_id: str
    name: str
    pattern: str
    severity: Severity
    description: str
    languages: Set[str]
    confidence: float = 0.8

    def __post_init__(self):
        self._compiled = re.compile(self.pattern, re.MULTILINE | re.IGNORECASE)

    def matches(self, code: str) -> List[Tuple[int, int, str]]:
        """Return list of (line, column, matched_text) tuples."""
        matches = []
        for match in self._compiled.finditer(code):
            # Calculate line number
            line = code[:match.start()].count('\n') + 1
            # Calculate column
            line_start = code.rfind('\n', 0, match.start()) + 1
            column = match.start() - line_start + 1
            matches.append((line, column, match.group()))
        return matches


class PatternJudge(BaseJudge):
    """
    Fast pattern-based security judge.

    Uses regex patterns and simple heuristics for vulnerability detection.
    Designed for speed (<10ms) with acceptable precision.

    Features:
    - Language-specific patterns
    - Configurable rule sets
    - CWE-aligned findings
    - Line/column location tracking

    Usage:
        judge = PatternJudge()
        result = judge.judge(code, "python")

        # With custom rules
        judge = PatternJudge(additional_rules=[custom_rule])

        # Filter by CWE
        judge = PatternJudge(cwe_filter=["CWE-89", "CWE-78"])
    """

    judge_type = "pattern"

    # Built-in pattern rules
    DEFAULT_RULES: List[PatternRule] = [
        # SQL Injection (CWE-89)
        PatternRule(
            cwe_id="CWE-89",
            name="sql_string_concat",
            pattern=r"(?:execute|cursor\.execute|query)\s*\(\s*['\"].*?%s|(?:execute|cursor\.execute|query)\s*\(\s*f['\"]|(?:execute|cursor\.execute|query)\s*\([^)]*\+",
            severity=Severity.CRITICAL,
            description="SQL query constructed with string formatting/concatenation",
            languages={"python", "java", "javascript", "go"},
            confidence=0.9,
        ),
        PatternRule(
            cwe_id="CWE-89",
            name="sql_format_string",
            pattern=r"(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\s+.*?(?:format|%s|{\w*}|\+\s*\w+)",
            severity=Severity.CRITICAL,
            description="SQL statement with dynamic string construction",
            languages={"python", "java", "javascript", "go"},
            confidence=0.85,
        ),

        # Command Injection (CWE-78)
        PatternRule(
            cwe_id="CWE-78",
            name="os_system_call",
            pattern=r"os\.system\s*\([^)]*[+%f]|subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True",
            severity=Severity.CRITICAL,
            description="Shell command execution with user-controllable input",
            languages={"python"},
            confidence=0.9,
        ),
        PatternRule(
            cwe_id="CWE-78",
            name="shell_exec",
            pattern=r"(?:exec|shell_exec|system|passthru|popen)\s*\(\s*\$",
            severity=Severity.CRITICAL,
            description="Shell execution with variable input",
            languages={"php"},
            confidence=0.9,
        ),
        PatternRule(
            cwe_id="CWE-78",
            name="backtick_execution",
            pattern=r"`[^`]*\$[^`]*`",
            severity=Severity.CRITICAL,
            description="Backtick command execution with variable",
            languages={"bash", "shell", "php"},
            confidence=0.85,
        ),

        # XSS (CWE-79)
        PatternRule(
            cwe_id="CWE-79",
            name="innerHTML_assignment",
            pattern=r"\.innerHTML\s*=\s*[^;]*(?:req\.|request\.|params\.|query\.|\+)",
            severity=Severity.HIGH,
            description="Direct innerHTML assignment with potentially untrusted data",
            languages={"javascript", "typescript"},
            confidence=0.85,
        ),
        PatternRule(
            cwe_id="CWE-79",
            name="document_write",
            pattern=r"document\.write\s*\([^)]*(?:req\.|request\.|params\.|query\.|\+)",
            severity=Severity.HIGH,
            description="document.write with potentially untrusted data",
            languages={"javascript", "typescript"},
            confidence=0.85,
        ),
        PatternRule(
            cwe_id="CWE-79",
            name="flask_render_unescaped",
            pattern=r"render_template_string\s*\([^)]*%|Markup\s*\([^)]*%",
            severity=Severity.HIGH,
            description="Flask template rendering without proper escaping",
            languages={"python"},
            confidence=0.8,
        ),

        # Hardcoded Credentials (CWE-798)
        PatternRule(
            cwe_id="CWE-798",
            name="hardcoded_password",
            pattern=r"(?:password|passwd|pwd|secret|api_key|apikey|auth_token)\s*=\s*['\"][^'\"]{4,}['\"]",
            severity=Severity.HIGH,
            description="Hardcoded credential in source code",
            languages={"python", "java", "javascript", "go", "ruby"},
            confidence=0.75,
        ),
        PatternRule(
            cwe_id="CWE-798",
            name="aws_key",
            pattern=r"(?:AKIA|ASIA)[A-Z0-9]{16}",
            severity=Severity.CRITICAL,
            description="AWS access key detected",
            languages={"python", "java", "javascript", "go", "ruby"},
            confidence=0.95,
        ),

        # Path Traversal (CWE-22)
        PatternRule(
            cwe_id="CWE-22",
            name="path_traversal_open",
            pattern=r"open\s*\([^)]*(?:request\.|params\.|query\.|\+\s*\w+)[^)]*\)",
            severity=Severity.HIGH,
            description="File open with user-controllable path",
            languages={"python"},
            confidence=0.8,
        ),
        PatternRule(
            cwe_id="CWE-22",
            name="path_join_user_input",
            pattern=r"(?:os\.path\.join|path\.join|Path)\s*\([^)]*(?:request\.|params\.|query\.)",
            severity=Severity.MEDIUM,
            description="Path construction with user input (potential traversal)",
            languages={"python", "javascript"},
            confidence=0.7,
        ),

        # Insecure Deserialization (CWE-502)
        PatternRule(
            cwe_id="CWE-502",
            name="pickle_load",
            pattern=r"pickle\.loads?\s*\(",
            severity=Severity.CRITICAL,
            description="Pickle deserialization (arbitrary code execution risk)",
            languages={"python"},
            confidence=0.85,
        ),
        PatternRule(
            cwe_id="CWE-502",
            name="yaml_unsafe_load",
            pattern=r"yaml\.(?:load|unsafe_load)\s*\([^)]*(?:,\s*Loader\s*=\s*(?:yaml\.)?(?:Loader|UnsafeLoader|FullLoader))?[^)]*\)(?!\s*#\s*nosec)",
            severity=Severity.HIGH,
            description="Unsafe YAML loading",
            languages={"python"},
            confidence=0.8,
        ),
        PatternRule(
            cwe_id="CWE-502",
            name="java_deserialization",
            pattern=r"ObjectInputStream\s*\([^)]*\)\.readObject\s*\(",
            severity=Severity.CRITICAL,
            description="Java deserialization of untrusted data",
            languages={"java"},
            confidence=0.85,
        ),

        # Weak Cryptography (CWE-327)
        PatternRule(
            cwe_id="CWE-327",
            name="weak_hash_md5",
            pattern=r"(?:hashlib\.md5|MD5|MessageDigest\.getInstance\s*\(\s*['\"]MD5['\"])",
            severity=Severity.MEDIUM,
            description="MD5 hash algorithm (cryptographically broken)",
            languages={"python", "java", "javascript"},
            confidence=0.9,
        ),
        PatternRule(
            cwe_id="CWE-327",
            name="weak_hash_sha1",
            pattern=r"(?:hashlib\.sha1|SHA1|MessageDigest\.getInstance\s*\(\s*['\"]SHA-?1['\"])",
            severity=Severity.MEDIUM,
            description="SHA1 hash algorithm (deprecated for security)",
            languages={"python", "java", "javascript"},
            confidence=0.85,
        ),
        PatternRule(
            cwe_id="CWE-327",
            name="weak_cipher_des",
            pattern=r"(?:DES|DESede|Blowfish)(?:\.|\s|Cipher)",
            severity=Severity.HIGH,
            description="Weak encryption algorithm",
            languages={"python", "java"},
            confidence=0.9,
        ),

        # Insecure Randomness (CWE-330)
        PatternRule(
            cwe_id="CWE-330",
            name="insecure_random_python",
            pattern=r"random\.(?:random|randint|choice|randrange)\s*\(",
            severity=Severity.MEDIUM,
            description="Non-cryptographic random for security context",
            languages={"python"},
            confidence=0.6,  # Lower confidence - context matters
        ),
        PatternRule(
            cwe_id="CWE-330",
            name="insecure_random_java",
            pattern=r"new\s+Random\s*\(|Math\.random\s*\(",
            severity=Severity.MEDIUM,
            description="Non-cryptographic random for security context",
            languages={"java", "javascript"},
            confidence=0.6,
        ),

        # Code Injection (CWE-94)
        PatternRule(
            cwe_id="CWE-94",
            name="eval_usage",
            pattern=r"\beval\s*\(\s*[a-zA-Z_]",
            severity=Severity.CRITICAL,
            description="eval() with variable input (code injection)",
            languages={"python", "javascript", "php"},
            confidence=0.9,
        ),
        PatternRule(
            cwe_id="CWE-94",
            name="exec_usage",
            pattern=r"\bexec\s*\(\s*[a-zA-Z_]",
            severity=Severity.CRITICAL,
            description="exec() with variable input (code injection)",
            languages={"python"},
            confidence=0.9,
        ),

        # XXE (CWE-611)
        PatternRule(
            cwe_id="CWE-611",
            name="xxe_etree",
            pattern=r"(?:etree\.parse|ET\.parse|xml\.etree\.ElementTree\.parse|fromstring)\s*\([^)]*\)(?!\s*#\s*nosec)",
            severity=Severity.HIGH,
            description="XML parsing without disabling external entities",
            languages={"python"},
            confidence=0.7,
        ),
        PatternRule(
            cwe_id="CWE-611",
            name="xxe_java",
            pattern=r"DocumentBuilderFactory\.newInstance\s*\(\s*\)(?!.*setFeature)",
            severity=Severity.HIGH,
            description="XML parsing without disabling external entities",
            languages={"java"},
            confidence=0.7,
        ),

        # SSRF (CWE-918)
        PatternRule(
            cwe_id="CWE-918",
            name="ssrf_requests",
            pattern=r"requests\.(?:get|post|put|delete|head)\s*\(\s*[a-zA-Z_]",
            severity=Severity.HIGH,
            description="HTTP request with user-controllable URL",
            languages={"python"},
            confidence=0.75,
        ),
        PatternRule(
            cwe_id="CWE-918",
            name="ssrf_urllib",
            pattern=r"urllib\.request\.urlopen\s*\(\s*[a-zA-Z_]",
            severity=Severity.HIGH,
            description="URL open with user-controllable input",
            languages={"python"},
            confidence=0.75,
        ),

        # CSRF - Missing protection (CWE-352)
        PatternRule(
            cwe_id="CWE-352",
            name="csrf_exempt",
            pattern=r"@csrf_exempt|csrf_protect\s*=\s*False",
            severity=Severity.MEDIUM,
            description="CSRF protection disabled",
            languages={"python"},
            confidence=0.85,
        ),

        # Missing Authentication (CWE-306)
        PatternRule(
            cwe_id="CWE-306",
            name="no_auth_decorator",
            pattern=r"@app\.route\s*\([^)]*\)\s*\n\s*def\s+\w+\s*\([^)]*\)\s*:(?!\s*\n\s*(?:@|if\s+(?:not\s+)?(?:session|current_user|request\.user)))",
            severity=Severity.MEDIUM,
            description="Route handler without authentication check",
            languages={"python"},
            confidence=0.5,  # Low confidence - needs context
        ),

        # Improper Input Validation (CWE-20)
        PatternRule(
            cwe_id="CWE-20",
            name="no_input_validation",
            pattern=r"request\.(?:args|form|json)\s*\[\s*['\"][^'\"]+['\"]\s*\](?!\s*\.\s*(?:strip|validate|sanitize))",
            severity=Severity.LOW,
            description="User input used without visible validation",
            languages={"python"},
            confidence=0.5,
        ),
    ]

    def __init__(
        self,
        severity_threshold: Severity = Severity.HIGH,
        min_confidence: float = 0.7,
        additional_rules: Optional[List[PatternRule]] = None,
        cwe_filter: Optional[List[str]] = None,
        exclude_rules: Optional[List[str]] = None,
    ):
        """
        Initialize the pattern judge.

        Args:
            severity_threshold: Minimum severity to flag as insecure
            min_confidence: Minimum confidence to include findings
            additional_rules: Custom rules to add
            cwe_filter: Only check these CWEs (None = all)
            exclude_rules: Rule names to skip
        """
        super().__init__(severity_threshold, min_confidence)

        self.rules = list(self.DEFAULT_RULES)
        if additional_rules:
            self.rules.extend(additional_rules)

        self.cwe_filter = set(cwe_filter) if cwe_filter else None
        self.exclude_rules = set(exclude_rules) if exclude_rules else set()

    def judge(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        """
        Judge code for security vulnerabilities using patterns.

        Args:
            code: Source code to analyze
            language: Programming language
            context: Optional context (unused in pattern judge)

        Returns:
            JudgeResult with findings
        """
        start_time = time.time()

        findings: List[JudgeFinding] = []

        for rule in self.rules:
            # Skip excluded rules
            if rule.name in self.exclude_rules:
                continue

            # Skip if CWE filter active and rule not in filter
            if self.cwe_filter and rule.cwe_id not in self.cwe_filter:
                continue

            # Skip if language doesn't match
            if language.lower() not in rule.languages:
                continue

            # Check for matches
            matches = rule.matches(code)
            for line, column, matched_text in matches:
                finding = JudgeFinding(
                    cwe_id=rule.cwe_id,
                    vulnerability_type=rule.name,
                    description=f"{rule.description}: '{matched_text[:50]}...' " if len(matched_text) > 50 else f"{rule.description}: '{matched_text}'",
                    severity=rule.severity,
                    confidence=rule.confidence,
                    location=CodeLocation(
                        line=line,
                        column=column,
                        snippet=matched_text[:100],
                    ),
                    judge_type=self.judge_type,
                )
                findings.append(finding)

        # Filter by confidence
        filtered_findings = self._filter_findings(findings)

        # Deduplicate findings at same location
        unique_findings = self._deduplicate_findings(filtered_findings)

        execution_time = (time.time() - start_time) * 1000

        return JudgeResult(
            is_secure=self._is_secure(unique_findings),
            security_score=self._compute_score(unique_findings),
            findings=unique_findings,
            judge_type=self.judge_type,
            execution_time_ms=execution_time,
            language=language,
        )

    def _deduplicate_findings(
        self,
        findings: List[JudgeFinding],
    ) -> List[JudgeFinding]:
        """Remove duplicate findings at the same location."""
        seen: Set[Tuple[str, int, str]] = set()
        unique: List[JudgeFinding] = []

        for finding in findings:
            key = (
                finding.cwe_id,
                finding.location.line if finding.location else 0,
                finding.vulnerability_type,
            )
            if key not in seen:
                seen.add(key)
                unique.append(finding)

        return unique

    def get_rules_for_cwe(self, cwe_id: str) -> List[PatternRule]:
        """Get all rules for a specific CWE."""
        return [r for r in self.rules if r.cwe_id == cwe_id]

    def get_rules_for_language(self, language: str) -> List[PatternRule]:
        """Get all rules for a specific language."""
        return [r for r in self.rules if language.lower() in r.languages]

    @classmethod
    def with_strict_rules(cls) -> "PatternJudge":
        """Factory for strict pattern checking (lower confidence threshold)."""
        return cls(
            severity_threshold=Severity.LOW,
            min_confidence=0.5,
        )

    @classmethod
    def with_high_precision(cls) -> "PatternJudge":
        """Factory for high-precision mode (fewer false positives)."""
        return cls(
            severity_threshold=Severity.HIGH,
            min_confidence=0.85,
        )

    @classmethod
    def for_injection_only(cls) -> "PatternJudge":
        """Factory for injection-focused scanning."""
        return cls(
            cwe_filter=["CWE-89", "CWE-78", "CWE-79", "CWE-94"],
        )

    @classmethod
    def for_crypto_only(cls) -> "PatternJudge":
        """Factory for cryptography-focused scanning."""
        return cls(
            cwe_filter=["CWE-327", "CWE-330", "CWE-798"],
        )
