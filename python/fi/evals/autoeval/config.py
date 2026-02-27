"""Configuration classes for AutoEval."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from copy import deepcopy


@dataclass
class EvalConfig:
    """Configuration for a single evaluation."""

    name: str
    enabled: bool = True
    threshold: float = 0.7
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None  # LLM model for augmented metrics
    augment: bool = False  # Enable local→LLM augmentation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "weight": self.weight,
        }
        if self.params:
            result["params"] = self.params
        if self.model:
            result["model"] = self.model
        if self.augment:
            result["augment"] = self.augment
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            enabled=data.get("enabled", True),
            threshold=data.get("threshold", 0.7),
            weight=data.get("weight", 1.0),
            params=data.get("params", {}),
            model=data.get("model"),
            augment=data.get("augment", False),
        )

    def copy(self) -> "EvalConfig":
        """Create a deep copy."""
        return EvalConfig(
            name=self.name,
            enabled=self.enabled,
            threshold=self.threshold,
            weight=self.weight,
            params=deepcopy(self.params),
            model=self.model,
            augment=self.augment,
        )


@dataclass
class ScannerConfig:
    """Configuration for a single scanner."""

    name: str
    enabled: bool = True
    threshold: float = 0.7
    action: str = "block"  # block, flag, warn, redact
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "action": self.action,
        }
        if self.params:
            result["params"] = self.params
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScannerConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            enabled=data.get("enabled", True),
            threshold=data.get("threshold", 0.7),
            action=data.get("action", "block"),
            params=data.get("params", {}),
        )

    def copy(self) -> "ScannerConfig":
        """Create a deep copy."""
        return ScannerConfig(
            name=self.name,
            enabled=self.enabled,
            threshold=self.threshold,
            action=self.action,
            params=deepcopy(self.params),
        )


@dataclass
class AutoEvalConfig:
    """Complete configuration for an auto-generated evaluation pipeline."""

    name: str
    description: str = ""
    version: str = "1.0.0"

    # Analysis metadata
    app_category: str = "unknown"
    risk_level: str = "medium"
    domain_sensitivity: str = "general"

    # Evaluations
    evaluations: List[EvalConfig] = field(default_factory=list)

    # Scanners
    scanners: List[ScannerConfig] = field(default_factory=list)

    # Execution settings
    execution_mode: str = "non_blocking"  # blocking, non_blocking, distributed
    parallel_workers: int = 4
    timeout_seconds: int = 30
    fail_fast: bool = False

    # Thresholds
    global_pass_rate: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "metadata": {
                "app_category": self.app_category,
                "risk_level": self.risk_level,
                "domain_sensitivity": self.domain_sensitivity,
                "generated_by": "autoeval",
            },
            "execution": {
                "mode": self.execution_mode,
                "parallel_workers": self.parallel_workers,
                "timeout_seconds": self.timeout_seconds,
                "fail_fast": self.fail_fast,
            },
            "thresholds": {
                "global_pass_rate": self.global_pass_rate,
            },
            "evaluations": [e.to_dict() for e in self.evaluations],
            "scanners": [s.to_dict() for s in self.scanners],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoEvalConfig":
        """Create from dictionary (YAML import)."""
        metadata = data.get("metadata", {})
        execution = data.get("execution", {})
        thresholds = data.get("thresholds", {})

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            app_category=metadata.get("app_category", "unknown"),
            risk_level=metadata.get("risk_level", "medium"),
            domain_sensitivity=metadata.get("domain_sensitivity", "general"),
            evaluations=[
                EvalConfig.from_dict(e) for e in data.get("evaluations", [])
            ],
            scanners=[
                ScannerConfig.from_dict(s) for s in data.get("scanners", [])
            ],
            execution_mode=execution.get("mode", "non_blocking"),
            parallel_workers=execution.get("parallel_workers", 4),
            timeout_seconds=execution.get("timeout_seconds", 30),
            fail_fast=execution.get("fail_fast", False),
            global_pass_rate=thresholds.get("global_pass_rate", 0.8),
        )

    def copy(self) -> "AutoEvalConfig":
        """Create a deep copy of the configuration."""
        return AutoEvalConfig(
            name=self.name,
            description=self.description,
            version=self.version,
            app_category=self.app_category,
            risk_level=self.risk_level,
            domain_sensitivity=self.domain_sensitivity,
            evaluations=[e.copy() for e in self.evaluations],
            scanners=[s.copy() for s in self.scanners],
            execution_mode=self.execution_mode,
            parallel_workers=self.parallel_workers,
            timeout_seconds=self.timeout_seconds,
            fail_fast=self.fail_fast,
            global_pass_rate=self.global_pass_rate,
        )

    def get_eval(self, name: str) -> Optional[EvalConfig]:
        """Get evaluation config by name."""
        for e in self.evaluations:
            if e.name == name:
                return e
        return None

    def get_scanner(self, name: str) -> Optional[ScannerConfig]:
        """Get scanner config by name."""
        for s in self.scanners:
            if s.name == name:
                return s
        return None

    def summary(self) -> str:
        """Get a summary of the configuration."""
        lines = [
            f"AutoEval Config: {self.name}",
            f"  Category: {self.app_category}",
            f"  Risk Level: {self.risk_level}",
            f"  Domain: {self.domain_sensitivity}",
            "",
            f"Evaluations ({len(self.evaluations)}):",
        ]

        for e in self.evaluations:
            status = "enabled" if e.enabled else "disabled"
            lines.append(f"  - {e.name} (threshold: {e.threshold}, {status})")

        lines.append("")
        lines.append(f"Scanners ({len(self.scanners)}):")

        for s in self.scanners:
            status = "enabled" if s.enabled else "disabled"
            lines.append(f"  - {s.name} (action: {s.action}, {status})")

        lines.append("")
        lines.append(f"Execution: {self.execution_mode}, workers: {self.parallel_workers}")
        lines.append(f"Pass Rate Threshold: {self.global_pass_rate:.0%}")

        return "\n".join(lines)
