"""
Base Scanner Classes for Guardrails.

Provides the foundation for lightweight, fast content scanners
that run before model-based backends.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ScannerAction(Enum):
    """Action to take when scanner detects a match."""
    BLOCK = "block"
    FLAG = "flag"
    REDACT = "redact"
    WARN = "warn"


@dataclass
class ScanMatch:
    """A single match found by a scanner."""
    pattern_name: str
    matched_text: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Result from a scanner."""
    passed: bool
    scanner_name: str
    category: str
    matches: List[ScanMatch] = field(default_factory=list)
    score: float = 0.0
    action: ScannerAction = ScannerAction.BLOCK
    reason: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_matches(self) -> bool:
        """Check if any matches were found."""
        return len(self.matches) > 0

    @property
    def matched_patterns(self) -> List[str]:
        """Get list of matched pattern names."""
        return [m.pattern_name for m in self.matches]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "scanner_name": self.scanner_name,
            "category": self.category,
            "matches": [
                {
                    "pattern_name": m.pattern_name,
                    "matched_text": m.matched_text,
                    "start": m.start,
                    "end": m.end,
                    "confidence": m.confidence,
                }
                for m in self.matches
            ],
            "score": self.score,
            "action": self.action.value,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


class BaseScanner(ABC):
    """
    Base class for content scanners.

    Scanners are lightweight, fast detectors that run before
    model-based backends. They should complete in <10ms.

    Usage:
        class MyScanner(BaseScanner):
            name = "my_scanner"
            category = "custom"

            def scan(self, content, context=None):
                # Detection logic
                return ScanResult(passed=True, ...)
    """

    name: str = "base_scanner"
    category: str = "generic"
    description: str = "Base scanner"
    default_action: ScannerAction = ScannerAction.BLOCK

    def __init__(self, action: Optional[ScannerAction] = None, enabled: bool = True):
        """
        Initialize scanner.

        Args:
            action: Action to take on match (default: BLOCK)
            enabled: Whether scanner is enabled
        """
        self.action = action or self.default_action
        self.enabled = enabled

    @abstractmethod
    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content for threats.

        Args:
            content: Content to scan
            context: Optional context (e.g., user query for output scanning)

        Returns:
            ScanResult with detection results
        """
        pass

    async def scan_async(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Async version of scan. Defaults to sync implementation.

        Override for truly async implementations.
        """
        return self.scan(content, context)

    def _create_result(
        self,
        passed: bool,
        matches: Optional[List[ScanMatch]] = None,
        score: float = 0.0,
        reason: Optional[str] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScanResult:
        """Helper to create a ScanResult."""
        return ScanResult(
            passed=passed,
            scanner_name=self.name,
            category=self.category,
            matches=matches or [],
            score=score,
            action=self.action if not passed else ScannerAction.WARN,
            reason=reason,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def _timed_scan(self, func, content: str, context: Optional[str] = None) -> ScanResult:
        """Helper to time a scan function."""
        start = time.perf_counter()
        result = func(content, context)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result


# Scanner registry for custom scanners
_SCANNER_REGISTRY: Dict[str, type] = {}


def register_scanner(name: str):
    """
    Decorator to register a custom scanner.

    Usage:
        @register_scanner("my_scanner")
        class MyScanner(BaseScanner):
            ...
    """
    def decorator(cls):
        _SCANNER_REGISTRY[name] = cls
        return cls
    return decorator


def get_scanner(name: str) -> Optional[type]:
    """Get a registered scanner class by name."""
    return _SCANNER_REGISTRY.get(name)


def list_scanners() -> List[str]:
    """List all registered scanner names."""
    return list(_SCANNER_REGISTRY.keys())
