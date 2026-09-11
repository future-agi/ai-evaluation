"""
Cryptography Vulnerability Detectors.

Detects weak cryptographic practices:
- CWE-327: Use of Broken or Risky Cryptographic Algorithm
- CWE-328: Use of Weak Hash
- CWE-330: Use of Insufficiently Random Values
- CWE-326: Inadequate Encryption Strength
"""

import re
from typing import List, Optional

from .base import BaseDetector, register_detector
from ..types import (
    SecurityFinding,
    Severity,
    VulnerabilityCategory,
)
from ..analyzer import AnalysisResult


@register_detector("weak_crypto")
class WeakCryptoDetector(BaseDetector):
    """
    Detects use of weak cryptographic algorithms (CWE-327, CWE-328).

    Identifies:
    - MD5 usage (broken hash)
    - SHA1 usage (weak hash)
    - DES/3DES usage (weak encryption)
    - RC4/ARC4 usage (broken stream cipher)
    - ECB mode (insecure block cipher mode)

    Examples of vulnerable code:
        hashlib.md5(password)
        Crypto.Cipher.DES.new(key)
        AES.new(key, AES.MODE_ECB)
    """

    name = "weak_crypto"
    cwe_ids = ["CWE-327", "CWE-328"]
    category = VulnerabilityCategory.CRYPTOGRAPHY
    description = "Use of weak or broken cryptographic algorithms"
    default_severity = Severity.MEDIUM

    WEAK_ALGORITHMS = {
        "python": [
            # Hashes
            (r"hashlib\.md5\s*\(", "MD5 hash (broken)", Severity.MEDIUM),
            (r"hashlib\.sha1\s*\(", "SHA1 hash (weak)", Severity.LOW),
            (r"MD5\.new\s*\(", "MD5 hash (broken)", Severity.MEDIUM),
            (r"SHA\.new\s*\(", "SHA1 hash (weak)", Severity.LOW),
            # Ciphers
            (r"DES\.new\s*\(", "DES encryption (weak)", Severity.MEDIUM),
            (r"DES3\.new\s*\(", "3DES encryption (deprecated)", Severity.LOW),
            (r"ARC4\.new\s*\(", "RC4 cipher (broken)", Severity.HIGH),
            (r"Blowfish\.new\s*\(", "Blowfish encryption (deprecated)", Severity.LOW),
            # ECB mode
            (r"MODE_ECB", "ECB mode (insecure)", Severity.MEDIUM),
            (r"AES\.new\s*\([^)]+,\s*AES\.MODE_ECB", "AES with ECB mode", Severity.MEDIUM),
        ],
        "javascript": [
            (r"createHash\s*\(['\"]md5['\"]", "MD5 hash (broken)", Severity.MEDIUM),
            (r"createHash\s*\(['\"]sha1['\"]", "SHA1 hash (weak)", Severity.LOW),
            (r"createCipher\s*\(['\"]des", "DES encryption (weak)", Severity.MEDIUM),
            (r"createCipher\s*\(['\"]rc4", "RC4 cipher (broken)", Severity.HIGH),
        ],
        "java": [
            (r"MessageDigest\.getInstance\s*\(['\"]MD5['\"]", "MD5 hash (broken)", Severity.MEDIUM),
            (r"MessageDigest\.getInstance\s*\(['\"]SHA-1['\"]", "SHA1 hash (weak)", Severity.LOW),
            (r"Cipher\.getInstance\s*\(['\"]DES", "DES encryption (weak)", Severity.MEDIUM),
            (r"Cipher\.getInstance\s*\(['\"].*ECB", "ECB mode (insecure)", Severity.MEDIUM),
            (r"SecretKeySpec\s*\([^)]+['\"]DES['\"]", "DES key (weak)", Severity.MEDIUM),
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect weak cryptographic algorithm usage."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        algorithms = self.WEAK_ALGORITHMS.get(lang_lower, self.WEAK_ALGORITHMS["python"])

        for i, line in enumerate(lines, 1):
            for pattern, description, severity in algorithms:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(self.create_finding(
                        vulnerability_type="Weak Cryptographic Algorithm",
                        description=f"Use of {description}",
                        line=i,
                        snippet=line.strip()[:100],
                        severity=severity,
                        confidence=0.9,
                        suggested_fix="Use SHA-256 or SHA-3 for hashing. Use AES-GCM or ChaCha20-Poly1305 for encryption.",
                    ))

        return findings


@register_detector("insecure_random")
class InsecureRandomDetector(BaseDetector):
    """
    Detects use of insecure random number generation (CWE-330).

    Identifies:
    - Use of random module for security-sensitive operations
    - Predictable random seeds
    - Math.random() for security

    Examples of vulnerable code:
        token = random.randint(0, 999999)
        secret = ''.join(random.choice(chars) for _ in range(32))
        Math.random()
    """

    name = "insecure_random"
    cwe_ids = ["CWE-330"]
    category = VulnerabilityCategory.CRYPTOGRAPHY
    description = "Use of non-cryptographic random number generator"
    default_severity = Severity.MEDIUM

    INSECURE_RANDOM = {
        "python": [
            (r"\brandom\.random\s*\(", "random.random()"),
            (r"\brandom\.randint\s*\(", "random.randint()"),
            (r"\brandom\.choice\s*\(", "random.choice()"),
            (r"\brandom\.choices\s*\(", "random.choices()"),
            (r"\brandom\.randrange\s*\(", "random.randrange()"),
            (r"\brandom\.sample\s*\(", "random.sample()"),
            (r"\brandom\.shuffle\s*\(", "random.shuffle()"),
        ],
        "javascript": [
            (r"Math\.random\s*\(", "Math.random()"),
        ],
        "java": [
            (r"new\s+Random\s*\(", "java.util.Random"),
            (r"Random\s*\(\s*\)", "java.util.Random"),
        ],
    }

    # Context indicating security-sensitive usage
    SECURITY_CONTEXTS = [
        "token", "secret", "key", "password", "session",
        "auth", "nonce", "salt", "iv", "otp", "verification",
        "csrf", "reset", "confirmation",
    ]

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect insecure random number generation."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        random_patterns = self.INSECURE_RANDOM.get(lang_lower, self.INSECURE_RANDOM["python"])

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            for pattern, func_name in random_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if used in security context
                    is_security_context = any(
                        ctx in line_lower for ctx in self.SECURITY_CONTEXTS
                    )

                    if is_security_context:
                        findings.append(self.create_finding(
                            vulnerability_type="Insecure Random Number Generator",
                            description=f"Use of {func_name} in security-sensitive context",
                            line=i,
                            snippet=line.strip()[:100],
                            severity=Severity.MEDIUM,
                            confidence=0.85,
                            suggested_fix="Use secrets module (Python), crypto.randomBytes (Node.js), or SecureRandom (Java).",
                        ))
                    else:
                        # Still flag but with lower confidence
                        findings.append(self.create_finding(
                            vulnerability_type="Insecure Random Number Generator",
                            description=f"Use of non-cryptographic {func_name}",
                            line=i,
                            snippet=line.strip()[:100],
                            severity=Severity.LOW,
                            confidence=0.5,
                            suggested_fix="If used for security purposes, use secrets module instead.",
                        ))

        return findings


@register_detector("weak_key_size")
class WeakKeySizeDetector(BaseDetector):
    """
    Detects inadequate encryption key sizes (CWE-326).

    Identifies:
    - RSA keys < 2048 bits
    - AES keys < 128 bits
    - Symmetric keys that are too short

    Examples of vulnerable code:
        RSA.generate(1024)
        key = os.urandom(8)  # 64 bits for AES
    """

    name = "weak_key_size"
    cwe_ids = ["CWE-326"]
    category = VulnerabilityCategory.CRYPTOGRAPHY
    description = "Inadequate encryption key size"
    default_severity = Severity.MEDIUM

    WEAK_KEY_PATTERNS = {
        "python": [
            # RSA with small key
            (r"RSA\.generate\s*\(\s*(512|768|1024)\s*\)", "RSA key size {0} bits is too small"),
            (r"rsa\.generate_private_key\s*\([^)]*key_size\s*=\s*(512|768|1024)", "RSA key size {0} bits is too small"),
            # Short random key
            (r"os\.urandom\s*\(\s*([1-9]|1[0-5])\s*\)", "Key of {0} bytes is too short"),
            (r"secrets\.token_bytes\s*\(\s*([1-9]|1[0-5])\s*\)", "Key of {0} bytes is too short"),
        ],
        "javascript": [
            (r"randomBytes\s*\(\s*([1-9]|1[0-5])\s*\)", "Key of {0} bytes is too short"),
        ],
        "java": [
            (r"KeyGenerator\.getInstance[^)]+\.init\s*\(\s*(56|64)\s*\)", "Key size {0} bits is too small"),
            (r"KeyPairGenerator[^)]+\.initialize\s*\(\s*(512|768|1024)\s*\)", "RSA key size {0} bits is too small"),
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect weak cryptographic key sizes."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        patterns = self.WEAK_KEY_PATTERNS.get(lang_lower, self.WEAK_KEY_PATTERNS["python"])

        for i, line in enumerate(lines, 1):
            for pattern, description_template in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Get the matched key size
                    key_size = match.group(1) if match.groups() else "unknown"
                    description = description_template.format(key_size)

                    findings.append(self.create_finding(
                        vulnerability_type="Weak Key Size",
                        description=description,
                        line=i,
                        snippet=line.strip()[:100],
                        confidence=0.9,
                        suggested_fix="Use at least RSA-2048 or AES-128. Prefer RSA-4096 or AES-256 for sensitive data.",
                    ))

        return findings


@register_detector("hardcoded_iv")
class HardcodedIVDetector(BaseDetector):
    """
    Detects hardcoded initialization vectors (IVs) for encryption.

    Identifies:
    - Static/hardcoded IVs
    - Reused IVs across encryptions

    Examples of vulnerable code:
        iv = b'0000000000000000'
        cipher = AES.new(key, AES.MODE_CBC, iv=b'static_iv_here!')
    """

    name = "hardcoded_iv"
    cwe_ids = ["CWE-329"]
    category = VulnerabilityCategory.CRYPTOGRAPHY
    description = "Use of hardcoded initialization vector"
    default_severity = Severity.MEDIUM

    IV_PATTERNS = [
        r"iv\s*=\s*b?['\"][^'\"]{8,}['\"]",
        r"IV\s*=\s*b?['\"][^'\"]{8,}['\"]",
        r"nonce\s*=\s*b?['\"][^'\"]{8,}['\"]",
        r"initialization_vector\s*=\s*b?['\"][^'\"]{8,}['\"]",
    ]

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect hardcoded IVs."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in self.IV_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(self.create_finding(
                        vulnerability_type="Hardcoded Initialization Vector",
                        description="IV should be randomly generated for each encryption",
                        line=i,
                        snippet=line.strip()[:100],
                        confidence=0.8,
                        suggested_fix="Generate a random IV for each encryption: os.urandom(16) for AES.",
                    ))
                    break

        return findings


__all__ = [
    "WeakCryptoDetector",
    "InsecureRandomDetector",
    "WeakKeySizeDetector",
    "HardcodedIVDetector",
]
