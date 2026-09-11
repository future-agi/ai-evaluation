"""
Code Injection Scanner for Guardrails.

Detects SQL injection, shell injection, path traversal,
and other code injection attacks.
"""

import re
import time
from typing import List, Optional, Tuple

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
)


# SQL Injection patterns
SQL_INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    # Classic SQL injection
    (r"(?i)'\s*(?:OR|AND)\s+['\d]+=\s*['\d]+", "sql_boolean_bypass", 0.95),
    (r"(?i)'\s*;\s*(?:DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT)\s+", "sql_destructive", 0.98),
    (r"(?i)(?:UNION\s+(?:ALL\s+)?SELECT)", "sql_union_select", 0.95),
    (r"(?i)(?:SELECT\s+.*\s+FROM\s+.*\s+WHERE)", "sql_select_where", 0.6),
    (r"(?i)(?:INSERT\s+INTO\s+.*\s+VALUES)", "sql_insert", 0.7),
    (r"(?i)(?:UPDATE\s+.*\s+SET\s+.*\s+WHERE)", "sql_update", 0.7),
    (r"(?i)(?:DELETE\s+FROM\s+.*\s+WHERE)", "sql_delete", 0.8),
    (r"(?i)(?:DROP\s+(?:TABLE|DATABASE|INDEX))", "sql_drop", 0.95),
    (r"(?i)(?:--\s*$|#\s*$|/\*)", "sql_comment", 0.6),
    (r"(?i)(?:EXEC(?:UTE)?\s*\(|xp_cmdshell)", "sql_exec", 0.95),
    (r"(?i)(?:WAITFOR\s+DELAY|SLEEP\s*\()", "sql_time_based", 0.9),
    (r"(?i)(?:BENCHMARK\s*\()", "sql_benchmark", 0.9),
    (r"(?i)(?:LOAD_FILE|INTO\s+(?:OUT|DUMP)FILE)", "sql_file_access", 0.95),
    (r"(?i)(?:information_schema|sys\.)", "sql_schema_access", 0.8),

    # NoSQL Injection
    (r'(?i)\$(?:where|gt|gte|lt|lte|ne|eq|in|nin|or|and|not|regex)', "nosql_operator", 0.85),
    (r'(?i){\s*["\']?\$(?:where|gt|ne)', "nosql_query_injection", 0.9),
]

# Shell Injection patterns
SHELL_INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    # Command chaining
    (r'(?:;|\||&&|\|\|)\s*(?:rm|cat|ls|wget|curl|nc|bash|sh|python|perl|ruby)\b', "shell_chain", 0.95),
    (r'\$\([^)]+\)', "shell_command_substitution", 0.85),
    (r'`[^`]+`', "shell_backtick", 0.85),

    # Dangerous commands
    (r'(?i)\brm\s+(?:-[rf]+\s+)?/', "shell_rm_root", 0.98),
    (r'(?i)\b(?:wget|curl)\s+.*\s*\|\s*(?:bash|sh)', "shell_download_execute", 0.98),
    (r'(?i)\b(?:nc|netcat|ncat)\s+-[elp]', "shell_netcat", 0.95),
    (r'(?i)\b(?:bash|sh|zsh|ksh)\s+-[ci]', "shell_interactive", 0.85),
    (r'(?i)\bchmod\s+[+0-7]*[xst]', "shell_chmod_exec", 0.7),
    (r'(?i)\bchown\s+', "shell_chown", 0.6),

    # Environment manipulation
    (r'(?i)\bexport\s+\w+=', "shell_export", 0.5),
    (r'(?i)\benv\s+-i', "shell_env_clear", 0.7),

    # Reverse shells
    (r'(?i)/dev/tcp/', "shell_dev_tcp", 0.95),
    (r'(?i)mkfifo|mknod', "shell_fifo", 0.85),
]

# Path Traversal patterns
PATH_TRAVERSAL_PATTERNS: List[Tuple[str, str, float]] = [
    (r'(?:\.\.[\\/]){2,}', "path_traversal", 0.9),
    (r'(?:\.\./|\.\.\\){3,}', "deep_path_traversal", 0.95),
    (r'(?i)(?:\.\./|\.\.\\)(?:etc|var|usr|home|windows|system32)', "path_traversal_sensitive", 0.95),
    (r'(?i)/etc/(?:passwd|shadow|hosts)', "unix_sensitive_file", 0.95),
    (r'(?i)(?:C:\\|\\\\)(?:windows|system32|boot)', "windows_sensitive", 0.9),
    (r'%(?:2e|2f|5c){2,}', "encoded_path_traversal", 0.9),
    (r'(?i)file:///', "file_protocol", 0.7),
]

# Template Injection patterns
TEMPLATE_INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    (r'\{\{.*\}\}', "jinja_template", 0.7),
    (r'\$\{[^}]+\}', "variable_interpolation", 0.7),
    (r'<%.*%>', "erb_template", 0.8),
    (r'#\{[^}]+\}', "ruby_interpolation", 0.7),
    (r'\{\%.*\%\}', "jinja_statement", 0.75),
    (r'(?i)\{\{.*(?:__class__|__mro__|__subclasses__|__globals__).*\}\}', "ssti_python", 0.95),
    (r'(?i)\$\{.*(?:Runtime|ProcessBuilder|getClass).*\}', "ssti_java", 0.95),
]

# LDAP Injection patterns
LDAP_INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    (r'\)\s*\(\|', "ldap_or_injection", 0.85),
    (r'\)\s*\(&', "ldap_and_injection", 0.85),
    (r'\*\)\s*\(', "ldap_wildcard", 0.7),
]

# XML/XXE Injection patterns
XML_INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    (r'<!ENTITY\s+', "xxe_entity", 0.9),
    (r'<!DOCTYPE\s+[^>]*\[', "xxe_doctype", 0.85),
    (r'(?i)SYSTEM\s+["\'](?:file|http|ftp)://', "xxe_external", 0.95),
]


@register_scanner("code_injection")
class CodeInjectionScanner(BaseScanner):
    """
    Scanner for detecting code injection attacks.

    Detects:
    - SQL injection (classic, blind, NoSQL)
    - Shell/Command injection
    - Path traversal attacks
    - Template injection (SSTI)
    - LDAP injection
    - XML/XXE injection

    Usage:
        scanner = CodeInjectionScanner()
        result = scanner.scan("'; DROP TABLE users; --")
        if not result.passed:
            print(f"Injection detected: {result.matched_patterns}")
    """

    name = "code_injection"
    category = "code_injection"
    description = "Detects SQL, shell, and other code injection attacks"
    default_action = ScannerAction.BLOCK

    def __init__(
        self,
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
        threshold: float = 0.7,
        check_sql: bool = True,
        check_shell: bool = True,
        check_path: bool = True,
        check_template: bool = True,
        check_ldap: bool = True,
        check_xml: bool = True,
    ):
        """
        Initialize code injection scanner.

        Args:
            action: Action on detection
            enabled: Whether scanner is enabled
            threshold: Minimum confidence to trigger
            check_sql: Check for SQL injection
            check_shell: Check for shell injection
            check_path: Check for path traversal
            check_template: Check for template injection
            check_ldap: Check for LDAP injection
            check_xml: Check for XML/XXE injection
        """
        super().__init__(action, enabled)
        self.threshold = threshold

        # Build pattern list based on enabled checks
        patterns = []
        if check_sql:
            patterns.extend(SQL_INJECTION_PATTERNS)
        if check_shell:
            patterns.extend(SHELL_INJECTION_PATTERNS)
        if check_path:
            patterns.extend(PATH_TRAVERSAL_PATTERNS)
        if check_template:
            patterns.extend(TEMPLATE_INJECTION_PATTERNS)
        if check_ldap:
            patterns.extend(LDAP_INJECTION_PATTERNS)
        if check_xml:
            patterns.extend(XML_INJECTION_PATTERNS)

        # Compile patterns
        self._compiled_patterns = [
            (re.compile(pattern), name, confidence)
            for pattern, name, confidence in patterns
        ]

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content for code injection attacks.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            ScanResult with detection details
        """
        start = time.perf_counter()
        matches = []
        max_confidence = 0.0

        for pattern, name, confidence in self._compiled_patterns:
            for match in pattern.finditer(content):
                matches.append(ScanMatch(
                    pattern_name=name,
                    matched_text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                ))
                max_confidence = max(max_confidence, confidence)

        latency = (time.perf_counter() - start) * 1000

        # Filter by threshold
        significant_matches = [m for m in matches if m.confidence >= self.threshold]

        if significant_matches:
            # Categorize the type of injection
            categories = set()
            for m in significant_matches:
                if m.pattern_name.startswith("sql_") or m.pattern_name.startswith("nosql_"):
                    categories.add("SQL injection")
                elif m.pattern_name.startswith("shell_"):
                    categories.add("Shell injection")
                elif m.pattern_name.startswith("path_") or m.pattern_name.endswith("_sensitive"):
                    categories.add("Path traversal")
                elif m.pattern_name.endswith("_template") or m.pattern_name.startswith("ssti_"):
                    categories.add("Template injection")
                elif m.pattern_name.startswith("ldap_"):
                    categories.add("LDAP injection")
                elif m.pattern_name.startswith("xxe_"):
                    categories.add("XXE injection")

            return self._create_result(
                passed=False,
                matches=significant_matches,
                score=max_confidence,
                reason=f"Code injection detected: {', '.join(categories)}",
                latency_ms=latency,
                metadata={"injection_types": list(categories)},
            )

        return self._create_result(
            passed=True,
            matches=[],
            score=0.0,
            reason="No code injection patterns detected",
            latency_ms=latency,
        )
