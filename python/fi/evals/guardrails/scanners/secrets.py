"""
Secrets Scanner for Guardrails.

Detects leaked credentials, API keys, tokens, and sensitive data.
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


# API Key patterns
API_KEY_PATTERNS: List[Tuple[str, str, float]] = [
    # OpenAI
    (r'sk-[a-zA-Z0-9]{20,}T3BlbkFJ[a-zA-Z0-9]{20,}', "openai_api_key", 0.99),
    (r'sk-(?:proj-)?[a-zA-Z0-9\-_]{40,}', "openai_api_key_generic", 0.9),

    # AWS
    (r'AKIA[0-9A-Z]{16}', "aws_access_key", 0.95),
    (r'(?i)aws[_\-]?secret[_\-]?(?:access[_\-]?)?key[\s:=]+[\'"]?[A-Za-z0-9/+=]{40}', "aws_secret_key", 0.95),

    # Google Cloud
    (r'AIza[0-9A-Za-z\-_]{35}', "google_api_key", 0.95),
    (r'(?i)google[_\-]?(?:cloud[_\-]?)?(?:api[_\-]?)?key[\s:=]+[\'"]?[A-Za-z0-9\-_]{30,}', "google_cloud_key", 0.85),

    # Azure
    (r'(?i)azure[_\-]?(?:storage[_\-]?)?(?:account[_\-]?)?key[\s:=]+[\'"]?[A-Za-z0-9+/=]{88}', "azure_storage_key", 0.9),
    (r'(?i)azure[_\-]?(?:subscription[_\-]?)?(?:id)?[\s:=]+[\'"]?[0-9a-f\-]{36}', "azure_subscription_id", 0.7),

    # GitHub
    (r'ghp_[0-9a-zA-Z]{36}', "github_pat", 0.99),
    (r'gho_[0-9a-zA-Z]{36}', "github_oauth", 0.99),
    (r'ghu_[0-9a-zA-Z]{36}', "github_user_token", 0.99),
    (r'ghs_[0-9a-zA-Z]{36}', "github_server_token", 0.99),
    (r'ghr_[0-9a-zA-Z]{36}', "github_refresh_token", 0.99),
    (r'github_pat_[0-9a-zA-Z]{22}_[0-9a-zA-Z]{59}', "github_fine_grained_pat", 0.99),

    # GitLab
    (r'glpat-[0-9a-zA-Z\-_]{20,}', "gitlab_pat", 0.95),

    # Slack
    (r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}', "slack_token", 0.95),
    (r'xoxb-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}', "slack_bot_token", 0.95),

    # Stripe
    (r'sk_live_[0-9a-zA-Z]{24}', "stripe_secret_key", 0.99),
    (r'pk_live_[0-9a-zA-Z]{24}', "stripe_publishable_key", 0.85),
    (r'sk_test_[0-9a-zA-Z]{24}', "stripe_test_key", 0.8),

    # Twilio
    (r'SK[0-9a-fA-F]{32}', "twilio_api_key", 0.85),
    (r'AC[a-zA-Z0-9]{32}', "twilio_account_sid", 0.85),

    # SendGrid
    (r'SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}', "sendgrid_api_key", 0.95),

    # Mailchimp
    (r'[0-9a-f]{32}-us[0-9]{1,2}', "mailchimp_api_key", 0.9),

    # Anthropic
    (r'sk-ant-[a-zA-Z0-9\-_]{40,}', "anthropic_api_key", 0.95),

    # HuggingFace
    (r'hf_[a-zA-Z0-9]{34}', "huggingface_token", 0.95),

    # Discord
    (r'[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}', "discord_token", 0.9),

    # Telegram
    (r'[0-9]+:AA[0-9A-Za-z\-_]{33}', "telegram_bot_token", 0.9),

    # Heroku
    (r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', "heroku_api_key", 0.6),

    # Firebase
    (r'(?i)firebase[_\-]?(?:api[_\-]?)?key[\s:=]+[\'"]?AIza[0-9A-Za-z\-_]{35}', "firebase_api_key", 0.9),

    # npm
    (r'npm_[A-Za-z0-9]{36}', "npm_token", 0.95),

    # PyPI
    (r'pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}', "pypi_token", 0.95),
]

# Private Key patterns
PRIVATE_KEY_PATTERNS: List[Tuple[str, str, float]] = [
    (r'-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----', "private_key_header", 0.95),
    (r'-----BEGIN PGP PRIVATE KEY BLOCK-----', "pgp_private_key", 0.95),
    (r'(?i)ssh-(?:rsa|dss|ed25519)\s+[A-Za-z0-9+/=]{100,}', "ssh_public_key", 0.7),
]

# Password patterns
PASSWORD_PATTERNS: List[Tuple[str, str, float]] = [
    (r'(?i)(?:password|passwd|pwd|secret|token)[\s]*[=:]+[\s]*[\'"]?[^\s\'"]{8,}[\'"]?', "password_assignment", 0.8),
    (r'(?i)(?:password|passwd|pwd)[\s]*[=:]+[\s]*[\'"][^\'"]{4,}[\'"]', "quoted_password", 0.85),
    (r'(?i)(?:api[_\-]?key|api[_\-]?secret|access[_\-]?token)[\s]*[=:]+[\s]*[\'"]?[^\s\'"]{16,}', "api_credential", 0.85),
]

# Connection String patterns
CONNECTION_STRING_PATTERNS: List[Tuple[str, str, float]] = [
    (r'(?i)(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp)://[^\s]+:[^\s]+@[^\s]+', "database_url", 0.95),
    (r'(?i)(?:Server|Host)=[^;]+;.*(?:Password|Pwd)=[^;]+', "connection_string", 0.9),
    (r'(?i)jdbc:[a-z]+://[^\s]+', "jdbc_url", 0.7),
]

# JWT patterns
JWT_PATTERNS: List[Tuple[str, str, float]] = [
    (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', "jwt_token", 0.85),
]

# Generic high-entropy patterns (careful with false positives)
ENTROPY_PATTERNS: List[Tuple[str, str, float]] = [
    (r'(?i)(?:secret|token|key|password|credential)[_\-]?(?:value)?[\s]*[=:]+[\s]*[\'"]?[a-zA-Z0-9+/=\-_]{32,}[\'"]?', "generic_secret", 0.7),
]


@register_scanner("secrets")
class SecretsScanner(BaseScanner):
    """
    Scanner for detecting leaked secrets and credentials.

    Detects:
    - API keys (OpenAI, AWS, Google, Azure, GitHub, etc.)
    - Private keys (RSA, SSH, PGP)
    - Passwords in code
    - Database connection strings
    - JWT tokens

    Usage:
        scanner = SecretsScanner()
        result = scanner.scan("api_key = 'sk-xxx...'")
        if not result.passed:
            print(f"Secret leaked: {result.matched_patterns}")
    """

    name = "secrets"
    category = "data_leakage"
    description = "Detects leaked API keys, passwords, and credentials"
    default_action = ScannerAction.BLOCK

    def __init__(
        self,
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
        threshold: float = 0.7,
        check_api_keys: bool = True,
        check_private_keys: bool = True,
        check_passwords: bool = True,
        check_connection_strings: bool = True,
        check_jwt: bool = True,
        check_entropy: bool = False,  # Disabled by default (false positives)
        redact_matches: bool = True,
    ):
        """
        Initialize secrets scanner.

        Args:
            action: Action on detection
            enabled: Whether scanner is enabled
            threshold: Minimum confidence to trigger
            check_api_keys: Check for API keys
            check_private_keys: Check for private keys
            check_passwords: Check for passwords
            check_connection_strings: Check for connection strings
            check_jwt: Check for JWT tokens
            check_entropy: Check generic high-entropy strings
            redact_matches: Whether to redact matched text in results
        """
        super().__init__(action, enabled)
        self.threshold = threshold
        self.redact_matches = redact_matches

        # Build pattern list
        patterns = []
        if check_api_keys:
            patterns.extend(API_KEY_PATTERNS)
        if check_private_keys:
            patterns.extend(PRIVATE_KEY_PATTERNS)
        if check_passwords:
            patterns.extend(PASSWORD_PATTERNS)
        if check_connection_strings:
            patterns.extend(CONNECTION_STRING_PATTERNS)
        if check_jwt:
            patterns.extend(JWT_PATTERNS)
        if check_entropy:
            patterns.extend(ENTROPY_PATTERNS)

        # Compile patterns
        self._compiled_patterns = [
            (re.compile(pattern), name, confidence)
            for pattern, name, confidence in patterns
        ]

    def _redact(self, text: str) -> str:
        """Redact sensitive text, keeping first and last 4 chars."""
        if len(text) <= 12:
            return "*" * len(text)
        return text[:4] + "*" * (len(text) - 8) + text[-4:]

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content for leaked secrets.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            ScanResult with detection details
        """
        start = time.perf_counter()
        matches = []
        max_confidence = 0.0
        secret_types = set()

        for pattern, name, confidence in self._compiled_patterns:
            for match in pattern.finditer(content):
                matched_text = match.group()
                if self.redact_matches:
                    matched_text = self._redact(matched_text)

                matches.append(ScanMatch(
                    pattern_name=name,
                    matched_text=matched_text,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                ))
                max_confidence = max(max_confidence, confidence)

                # Categorize
                if "api_key" in name or "token" in name or "pat" in name:
                    secret_types.add("API Key")
                elif "private_key" in name or "pgp" in name or "ssh" in name:
                    secret_types.add("Private Key")
                elif "password" in name or "credential" in name:
                    secret_types.add("Password")
                elif "url" in name or "connection" in name or "jdbc" in name:
                    secret_types.add("Connection String")
                elif "jwt" in name:
                    secret_types.add("JWT Token")
                else:
                    secret_types.add("Secret")

        latency = (time.perf_counter() - start) * 1000

        # Filter by threshold
        significant_matches = [m for m in matches if m.confidence >= self.threshold]

        if significant_matches:
            return self._create_result(
                passed=False,
                matches=significant_matches,
                score=max_confidence,
                reason=f"Secrets detected: {', '.join(secret_types)}",
                latency_ms=latency,
                metadata={"secret_types": list(secret_types)},
            )

        return self._create_result(
            passed=True,
            matches=[],
            score=0.0,
            reason="No secrets detected",
            latency_ms=latency,
        )
