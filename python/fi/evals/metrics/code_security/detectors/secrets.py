"""
Secrets Detection - Hardcoded Credentials.

Detects hardcoded secrets and credentials:
- CWE-798: Use of Hard-coded Credentials
- CWE-259: Use of Hard-coded Password
- CWE-321: Use of Hard-coded Cryptographic Key
- CWE-532: Insertion of Sensitive Information into Log File
"""

import re
from typing import List, Optional, Dict

from .base import BaseDetector, register_detector
from ..types import (
    SecurityFinding,
    Severity,
    VulnerabilityCategory,
)
from ..analyzer import AnalysisResult


@register_detector("hardcoded_secrets")
class HardcodedSecretsDetector(BaseDetector):
    """
    Detects hardcoded secrets and credentials (CWE-798, CWE-259, CWE-321).

    Identifies:
    - Hardcoded passwords
    - API keys
    - Secret tokens
    - Private keys
    - Database connection strings

    Examples of vulnerable code:
        password = "secret123"
        API_KEY = "sk-1234567890abcdef"
        conn_str = "postgres://user:pass@host/db"
    """

    name = "hardcoded_secrets"
    cwe_ids = ["CWE-798", "CWE-259", "CWE-321"]
    category = VulnerabilityCategory.SECRETS
    description = "Hardcoded credentials and secrets"
    default_severity = Severity.HIGH

    # Variable names that typically contain secrets
    SECRET_VAR_PATTERNS = [
        r"\b(password|passwd|pwd|pass)\s*=",
        r"\b(secret|token|key|api_key|apikey|auth)\s*=",
        r"\b(credential|cred)\s*=",
        r"\b(private_key|privatekey|priv_key)\s*=",
        r"\b(access_key|secret_key)\s*=",
        r"\b(db_password|database_password)\s*=",
        r"\b(jwt_secret|session_secret)\s*=",
        r"\b(encryption_key|signing_key)\s*=",
    ]

    # Patterns for specific API keys
    API_KEY_PATTERNS = [
        # OpenAI
        (r"sk-[a-zA-Z0-9]{48}", "OpenAI API Key"),
        # AWS
        (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
        (r"[a-zA-Z0-9/+=]{40}", "Potential AWS Secret Key"),
        # Google
        (r"AIza[0-9A-Za-z_-]{35}", "Google API Key"),
        # GitHub
        (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
        (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth Token"),
        (r"ghs_[a-zA-Z0-9]{36}", "GitHub Server Token"),
        # Stripe
        (r"sk_live_[a-zA-Z0-9]{24}", "Stripe Live Secret Key"),
        (r"sk_test_[a-zA-Z0-9]{24}", "Stripe Test Secret Key"),
        # Slack
        (r"xoxb-[a-zA-Z0-9-]+", "Slack Bot Token"),
        (r"xoxp-[a-zA-Z0-9-]+", "Slack User Token"),
        # Generic patterns
        (r"Bearer\s+[a-zA-Z0-9_-]{20,}", "Bearer Token"),
        (r"Basic\s+[a-zA-Z0-9+/=]{20,}", "Basic Auth Credentials"),
    ]

    # Connection string patterns
    CONNECTION_STRING_PATTERNS = [
        (r"(postgres|mysql|mongodb|redis)://[^:]+:[^@]+@", "Database Connection String with Password"),
        (r"mongodb\+srv://[^:]+:[^@]+@", "MongoDB Connection String with Password"),
        (r"amqp://[^:]+:[^@]+@", "RabbitMQ Connection String"),
    ]

    # Private key patterns
    PRIVATE_KEY_PATTERNS = [
        (r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "Private Key"),
        (r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----", "EC Private Key"),
        (r"-----BEGIN\s+DSA\s+PRIVATE\s+KEY-----", "DSA Private Key"),
        (r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----", "OpenSSH Private Key"),
        (r"-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----", "PGP Private Key"),
    ]

    # Values to exclude (not secrets)
    SAFE_VALUE_PATTERNS = [
        r"^[\"']?$",  # Empty string
        r"^[\"']?(None|null|undefined|placeholder|TODO|FIXME)[\"']?$",
        r"^[\"']?<.*>[\"']?$",  # Placeholder like <password>
        r"^[\"']?\$\{.*\}[\"']?$",  # Environment variable
        r"^[\"']?%.*%[\"']?$",  # Windows env var
        r"os\.environ",  # Environment lookup
        r"getenv\(",  # Environment lookup
        r"process\.env",  # Node.js env
        r"config\[",  # Config lookup
        r"settings\.",  # Settings lookup
    ]

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect hardcoded secrets."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*", "/*")):
                continue

            # Check for secret variable assignments
            for pattern in self.SECRET_VAR_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    if not self._is_safe_value(line):
                        # Extract the value being assigned
                        value_match = re.search(r'=\s*["\']([^"\']+)["\']', line)
                        if value_match and len(value_match.group(1)) >= 4:
                            findings.append(self.create_finding(
                                vulnerability_type="Hardcoded Credentials",
                                description=f"Potential hardcoded secret in variable assignment",
                                line=i,
                                snippet=self._redact_secret(line.strip()[:100]),
                                confidence=0.85,
                                cwe_id="CWE-798",
                                suggested_fix="Use environment variables or a secrets manager instead of hardcoding.",
                            ))
                    break

            # Check for API keys
            for pattern, key_type in self.API_KEY_PATTERNS:
                if re.search(pattern, line):
                    findings.append(self.create_finding(
                        vulnerability_type="Hardcoded API Key",
                        description=f"Detected {key_type}",
                        line=i,
                        snippet=self._redact_secret(line.strip()[:100]),
                        severity=Severity.HIGH,
                        confidence=0.95,
                        cwe_id="CWE-798",
                        suggested_fix="Store API keys in environment variables or a secrets manager.",
                    ))
                    break

            # Check for connection strings
            for pattern, conn_type in self.CONNECTION_STRING_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(self.create_finding(
                        vulnerability_type="Hardcoded Connection String",
                        description=f"Detected {conn_type}",
                        line=i,
                        snippet=self._redact_secret(line.strip()[:100]),
                        severity=Severity.HIGH,
                        confidence=0.9,
                        cwe_id="CWE-259",
                        suggested_fix="Use environment variables for database credentials.",
                    ))
                    break

            # Check for private keys
            for pattern, key_type in self.PRIVATE_KEY_PATTERNS:
                if re.search(pattern, line):
                    findings.append(self.create_finding(
                        vulnerability_type="Embedded Private Key",
                        description=f"Detected {key_type} in source code",
                        line=i,
                        snippet=line.strip()[:50] + "...[REDACTED]",
                        severity=Severity.CRITICAL,
                        confidence=0.99,
                        cwe_id="CWE-321",
                        suggested_fix="Never embed private keys in source code. Use secure key management.",
                    ))
                    break

        return findings

    def _is_safe_value(self, line: str) -> bool:
        """Check if the assigned value is a safe pattern (not a hardcoded secret)."""
        return any(re.search(p, line, re.IGNORECASE) for p in self.SAFE_VALUE_PATTERNS)

    def _redact_secret(self, text: str) -> str:
        """Redact potential secrets from text."""
        # Redact values in quotes
        text = re.sub(r'(["\'])([^"\']{4})[^"\']*([^"\']{2})\1', r'\1\2****\3\1', text)
        # Redact API keys
        text = re.sub(r'(sk-|ghp_|gho_|sk_live_|sk_test_)[a-zA-Z0-9]{4}[a-zA-Z0-9]*', r'\1****', text)
        return text


@register_detector("sensitive_logging")
class SensitiveLoggingDetector(BaseDetector):
    """
    Detects logging of sensitive information (CWE-532).

    Identifies:
    - Logging passwords or tokens
    - Printing sensitive variables
    - Debug output with credentials

    Examples of vulnerable code:
        print(f"Password: {password}")
        logger.info(f"Token: {api_token}")
        console.log("Secret:", secret)
    """

    name = "sensitive_logging"
    cwe_ids = ["CWE-532"]
    category = VulnerabilityCategory.INFORMATION
    description = "Sensitive information in log output"
    default_severity = Severity.MEDIUM

    LOGGING_FUNCTIONS = {
        "python": ["print", "logging", "logger", "log"],
        "javascript": ["console.log", "console.info", "console.debug", "console.warn"],
        "java": ["System.out.print", "logger.info", "logger.debug", "log.info"],
    }

    SENSITIVE_TERMS = [
        "password", "passwd", "pwd", "secret", "token", "key",
        "credential", "api_key", "apikey", "auth", "bearer",
        "private", "ssn", "credit_card", "cvv",
    ]

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect sensitive information being logged."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        log_funcs = self.LOGGING_FUNCTIONS.get(lang_lower, self.LOGGING_FUNCTIONS["python"])

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Check if this is a logging line
            is_logging = any(func in line_lower for func in log_funcs)
            if not is_logging:
                continue

            # Check for sensitive terms
            for term in self.SENSITIVE_TERMS:
                if term in line_lower:
                    findings.append(self.create_finding(
                        vulnerability_type="Sensitive Data Logging",
                        description=f"Potentially logging sensitive data: '{term}'",
                        line=i,
                        snippet=line.strip()[:100],
                        confidence=0.7,
                        suggested_fix="Remove sensitive data from log statements or mask the values.",
                    ))
                    break

        return findings


__all__ = [
    "HardcodedSecretsDetector",
    "SensitiveLoggingDetector",
]
