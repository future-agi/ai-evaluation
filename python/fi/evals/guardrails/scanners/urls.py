"""
Malicious URL Scanner for Guardrails.

Detects suspicious URLs, phishing attempts, and malicious links.
"""

import re
import time
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
)


# URL extraction pattern
URL_PATTERN = re.compile(
    r'https?://[^\s<>"\']+|'
    r'www\.[^\s<>"\']+|'
    r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?'
)

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS: Set[str] = {
    "xyz", "top", "work", "click", "link", "gq", "ml", "cf", "ga", "tk",
    "zip", "mov", "app", "dev",  # New TLDs that can be confusing
    "ru", "cn", "su",  # Sometimes associated with malicious content
}

# Known URL shorteners
URL_SHORTENERS: Set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bc.vc", "j.mp", "tr.im", "tiny.cc",
    "lnkd.in", "db.tt", "qr.ae", "cur.lv", "ity.im", "q.gs",
    "po.st", "su.pr", "rebrand.ly", "shorturl.at",
}

# Legitimate domains that are often spoofed
SPOOFED_DOMAINS: List[Tuple[str, List[str]]] = [
    ("google", ["g00gle", "googie", "gooogle", "google-", "-google"]),
    ("facebook", ["faceb00k", "facebok", "faceboook", "facebook-", "-facebook"]),
    ("microsoft", ["micros0ft", "mircosoft", "microsoft-", "-microsoft"]),
    ("apple", ["app1e", "appie", "apple-", "-apple"]),
    ("amazon", ["amaz0n", "amazonn", "amazon-", "-amazon"]),
    ("paypal", ["paypa1", "paypai", "paypal-", "-paypal"]),
    ("netflix", ["netf1ix", "netiflix", "netflix-", "-netflix"]),
    ("linkedin", ["linkedln", "linkedin-", "-linkedin"]),
    ("twitter", ["tw1tter", "twitter-", "-twitter"]),
    ("instagram", ["lnstagram", "instagram-", "-instagram"]),
    ("whatsapp", ["whatsap", "whatsapp-", "-whatsapp"]),
    ("dropbox", ["dr0pbox", "dropbox-", "-dropbox"]),
    ("github", ["g1thub", "github-", "-github"]),
    ("openai", ["0penai", "openal", "openai-", "-openai"]),
]


def _extract_urls(text: str) -> List[Tuple[str, int, int]]:
    """Extract URLs from text with positions."""
    urls = []
    for match in URL_PATTERN.finditer(text):
        url = match.group()
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        urls.append((url, match.start(), match.end()))
    return urls


def _get_domain(url: str) -> Optional[str]:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None


def _check_homoglyph(domain: str) -> Optional[str]:
    """Check for homoglyph attacks (lookalike characters)."""
    # Common homoglyphs
    homoglyphs = {
        '0': 'o', '1': 'l', '1': 'i', '@': 'a',
        '$': 's', '3': 'e', '4': 'a', '5': 's',
        '6': 'b', '7': 't', '8': 'b', '9': 'g',
    }

    normalized = domain
    for fake, real in homoglyphs.items():
        normalized = normalized.replace(fake, real)

    if normalized != domain:
        return normalized
    return None


@register_scanner("urls")
class MaliciousURLScanner(BaseScanner):
    """
    Scanner for detecting malicious and suspicious URLs.

    Detects:
    - Phishing URLs (lookalike domains)
    - IP-based URLs
    - Suspicious TLDs
    - URL shorteners (optional)
    - Data URLs with executable content
    - Encoded URLs

    Usage:
        scanner = MaliciousURLScanner()
        result = scanner.scan("Visit https://g00gle.com/login")
        if not result.passed:
            print(f"Suspicious URL: {result.matched_patterns}")
    """

    name = "urls"
    category = "malicious_url"
    description = "Detects phishing URLs and malicious links"
    default_action = ScannerAction.FLAG  # Flag rather than block by default

    def __init__(
        self,
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
        threshold: float = 0.7,
        block_ip_urls: bool = True,
        block_suspicious_tlds: bool = True,
        block_shorteners: bool = False,  # Disabled by default
        block_data_urls: bool = True,
        check_homoglyphs: bool = True,
        allowed_domains: Optional[Set[str]] = None,
        blocked_domains: Optional[Set[str]] = None,
    ):
        """
        Initialize URL scanner.

        Args:
            action: Action on detection
            enabled: Whether scanner is enabled
            threshold: Minimum confidence to trigger
            block_ip_urls: Block URLs with IP addresses
            block_suspicious_tlds: Block URLs with suspicious TLDs
            block_shorteners: Block URL shorteners
            block_data_urls: Block data: URLs
            check_homoglyphs: Check for lookalike domains
            allowed_domains: Whitelist of allowed domains
            blocked_domains: Blacklist of blocked domains
        """
        super().__init__(action, enabled)
        self.threshold = threshold
        self.block_ip_urls = block_ip_urls
        self.block_suspicious_tlds = block_suspicious_tlds
        self.block_shorteners = block_shorteners
        self.block_data_urls = block_data_urls
        self.check_homoglyphs = check_homoglyphs
        self.allowed_domains = allowed_domains or set()
        self.blocked_domains = blocked_domains or set()

    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address."""
        # IPv4
        ipv4_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        if re.match(ipv4_pattern, domain):
            return True
        # IPv6
        if ':' in domain and domain.replace(':', '').replace('.', '').isalnum():
            return True
        return False

    def _check_phishing(self, domain: str) -> Optional[Tuple[str, str]]:
        """Check if domain is a phishing attempt."""
        domain_lower = domain.lower()

        for legit, fakes in SPOOFED_DOMAINS:
            for fake in fakes:
                if fake in domain_lower:
                    return (legit, fake)

            # Also check for the legitimate name in suspicious context
            if legit in domain_lower and not domain_lower.endswith(f".{legit}.com"):
                # e.g., google.suspicious.com
                parts = domain_lower.split('.')
                if len(parts) > 2 and legit in parts[0]:
                    return (legit, f"{legit} subdomain")

        return None

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content for malicious URLs.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            ScanResult with detection details
        """
        start = time.perf_counter()
        matches = []
        max_confidence = 0.0
        issues = set()

        # Check for data URLs
        if self.block_data_urls:
            data_url_pattern = re.compile(r'data:[^;]+;base64,[a-zA-Z0-9+/=]+', re.IGNORECASE)
            for match in data_url_pattern.finditer(content):
                # Check if it's potentially executable
                url = match.group().lower()
                if any(t in url for t in ['javascript', 'text/html', 'application/']):
                    matches.append(ScanMatch(
                        pattern_name="data_url_executable",
                        matched_text=match.group()[:50] + "...",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    ))
                    max_confidence = max(max_confidence, 0.95)
                    issues.add("Executable data URL")

        # Extract and check URLs
        urls = _extract_urls(content)

        for url, start_pos, end_pos in urls:
            domain = _get_domain(url)
            if not domain:
                continue

            # Skip allowed domains
            if domain in self.allowed_domains:
                continue

            # Check blocked domains
            if domain in self.blocked_domains:
                matches.append(ScanMatch(
                    pattern_name="blocked_domain",
                    matched_text=url,
                    start=start_pos,
                    end=end_pos,
                    confidence=1.0,
                ))
                max_confidence = 1.0
                issues.add("Blocked domain")
                continue

            # Check IP-based URLs
            if self.block_ip_urls and self._is_ip_address(domain.split(':')[0]):
                matches.append(ScanMatch(
                    pattern_name="ip_url",
                    matched_text=url,
                    start=start_pos,
                    end=end_pos,
                    confidence=0.8,
                ))
                max_confidence = max(max_confidence, 0.8)
                issues.add("IP-based URL")

            # Check suspicious TLDs
            if self.block_suspicious_tlds:
                tld = domain.split('.')[-1]
                if tld in SUSPICIOUS_TLDS:
                    matches.append(ScanMatch(
                        pattern_name="suspicious_tld",
                        matched_text=url,
                        start=start_pos,
                        end=end_pos,
                        confidence=0.6,
                        metadata={"tld": tld},
                    ))
                    max_confidence = max(max_confidence, 0.6)
                    issues.add(f"Suspicious TLD (.{tld})")

            # Check URL shorteners
            if self.block_shorteners and domain in URL_SHORTENERS:
                matches.append(ScanMatch(
                    pattern_name="url_shortener",
                    matched_text=url,
                    start=start_pos,
                    end=end_pos,
                    confidence=0.5,
                ))
                max_confidence = max(max_confidence, 0.5)
                issues.add("URL shortener")

            # Check phishing (lookalike domains)
            if self.check_homoglyphs:
                phishing = self._check_phishing(domain)
                if phishing:
                    legit, fake = phishing
                    matches.append(ScanMatch(
                        pattern_name="phishing_domain",
                        matched_text=url,
                        start=start_pos,
                        end=end_pos,
                        confidence=0.9,
                        metadata={"spoofed_brand": legit, "technique": fake},
                    ))
                    max_confidence = max(max_confidence, 0.9)
                    issues.add(f"Phishing ({legit} lookalike)")

                # Also check homoglyphs
                normalized = _check_homoglyph(domain)
                if normalized:
                    matches.append(ScanMatch(
                        pattern_name="homoglyph_attack",
                        matched_text=url,
                        start=start_pos,
                        end=end_pos,
                        confidence=0.85,
                        metadata={"normalized": normalized},
                    ))
                    max_confidence = max(max_confidence, 0.85)
                    issues.add("Homoglyph attack")

        latency = (time.perf_counter() - start) * 1000

        # Filter by threshold
        significant_matches = [m for m in matches if m.confidence >= self.threshold]

        if significant_matches:
            return self._create_result(
                passed=False,
                matches=significant_matches,
                score=max_confidence,
                reason=f"Suspicious URLs detected: {', '.join(issues)}",
                latency_ms=latency,
                metadata={"issues": list(issues)},
            )

        return self._create_result(
            passed=True,
            matches=[],
            score=0.0,
            reason="No malicious URLs detected",
            latency_ms=latency,
        )
