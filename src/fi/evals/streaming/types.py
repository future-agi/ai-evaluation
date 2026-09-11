"""Type definitions for Streaming Evaluation.

Provides types for evaluating LLM outputs in real-time as tokens are generated.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone


class EarlyStopReason(Enum):
    """Reasons for early stopping during streaming evaluation."""

    NONE = "none"  # No early stop
    TOXICITY = "toxicity"  # Toxic content detected
    SAFETY = "safety"  # Safety violation detected
    PII = "pii"  # PII detected
    JAILBREAK = "jailbreak"  # Jailbreak attempt detected
    MAX_TOKENS = "max_tokens"  # Maximum token limit reached
    MAX_CHARS = "max_chars"  # Maximum character limit reached
    THRESHOLD = "threshold"  # Score dropped below threshold
    CUSTOM = "custom"  # Custom stop condition triggered
    TIMEOUT = "timeout"  # Evaluation timeout
    ERROR = "error"  # Evaluation error


class StreamingState(Enum):
    """State of the streaming evaluation."""

    IDLE = "idle"  # Not started
    STREAMING = "streaming"  # Actively processing chunks
    PAUSED = "paused"  # Temporarily paused
    STOPPED = "stopped"  # Early stopped
    COMPLETED = "completed"  # Finished normally
    ERROR = "error"  # Error occurred


@dataclass
class ChunkResult:
    """Result of evaluating a single chunk."""

    chunk_index: int
    chunk_text: str
    cumulative_text: str
    scores: Dict[str, float]  # eval_name -> score
    flags: Dict[str, bool]  # eval_name -> passed
    should_stop: bool = False
    stop_reason: EarlyStopReason = EarlyStopReason.NONE
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        """Check if all evaluations passed for this chunk."""
        return all(self.flags.values()) if self.flags else True

    @property
    def min_score(self) -> float:
        """Get the minimum score across all evaluations."""
        return min(self.scores.values()) if self.scores else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text,
            "scores": self.scores,
            "flags": self.flags,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StreamingEvalResult:
    """Final result of streaming evaluation."""

    passed: bool
    final_text: str
    total_chunks: int
    chunk_results: List[ChunkResult]
    final_scores: Dict[str, float]  # eval_name -> final score
    early_stopped: bool = False
    stop_reason: EarlyStopReason = EarlyStopReason.NONE
    stopped_at_chunk: Optional[int] = None
    total_latency_ms: float = 0.0
    state: StreamingState = StreamingState.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def average_chunk_latency_ms(self) -> float:
        """Average latency per chunk."""
        if not self.chunk_results:
            return 0.0
        return sum(c.latency_ms for c in self.chunk_results) / len(self.chunk_results)

    @property
    def min_score_history(self) -> List[float]:
        """Get minimum score at each chunk for plotting."""
        return [c.min_score for c in self.chunk_results]

    @property
    def score_by_eval(self) -> Dict[str, List[float]]:
        """Get score history by evaluation name."""
        result: Dict[str, List[float]] = {}
        for chunk in self.chunk_results:
            for eval_name, score in chunk.scores.items():
                if eval_name not in result:
                    result[eval_name] = []
                result[eval_name].append(score)
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Streaming Evaluation: {'PASSED' if self.passed else 'FAILED'}",
            f"  Total Chunks: {self.total_chunks}",
            f"  Final Text Length: {len(self.final_text)} chars",
            f"  Total Latency: {self.total_latency_ms:.2f}ms",
            f"  Avg Chunk Latency: {self.average_chunk_latency_ms:.2f}ms",
        ]

        if self.early_stopped:
            lines.append(f"  Early Stopped: Yes (at chunk {self.stopped_at_chunk})")
            lines.append(f"  Stop Reason: {self.stop_reason.value}")

        lines.append("  Final Scores:")
        for eval_name, score in self.final_scores.items():
            lines.append(f"    {eval_name}: {score:.3f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "final_text": self.final_text,
            "total_chunks": self.total_chunks,
            "chunk_results": [c.to_dict() for c in self.chunk_results],
            "final_scores": self.final_scores,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason.value,
            "stopped_at_chunk": self.stopped_at_chunk,
            "total_latency_ms": self.total_latency_ms,
            "state": self.state.value,
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming evaluation."""

    # Chunk processing
    min_chunk_size: int = 1  # Minimum characters to trigger evaluation
    max_chunk_size: int = 100  # Maximum characters per chunk before forced eval
    eval_interval_ms: int = 100  # Minimum ms between evaluations

    # Token limits
    max_tokens: Optional[int] = None  # Maximum tokens before stopping
    max_chars: Optional[int] = None  # Maximum characters before stopping

    # Timeouts
    chunk_timeout_ms: int = 5000  # Timeout for single chunk evaluation
    total_timeout_ms: int = 60000  # Total timeout for streaming

    # Early stopping
    enable_early_stop: bool = True
    stop_on_first_failure: bool = False  # Stop immediately on any failure

    # Thresholds for early stopping
    toxicity_threshold: float = 0.7  # Stop if toxicity score exceeds this
    safety_threshold: float = 0.3  # Stop if safety score drops below this

    # Evaluation frequency
    eval_every_n_chunks: int = 1  # Run evaluation every N chunks
    eval_on_sentence_end: bool = True  # Also evaluate at sentence boundaries

    # Callbacks
    on_chunk_callback: Optional[Callable[[ChunkResult], None]] = None
    on_stop_callback: Optional[Callable[[EarlyStopReason, str], None]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callbacks)."""
        return {
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "eval_interval_ms": self.eval_interval_ms,
            "max_tokens": self.max_tokens,
            "max_chars": self.max_chars,
            "chunk_timeout_ms": self.chunk_timeout_ms,
            "total_timeout_ms": self.total_timeout_ms,
            "enable_early_stop": self.enable_early_stop,
            "stop_on_first_failure": self.stop_on_first_failure,
            "toxicity_threshold": self.toxicity_threshold,
            "safety_threshold": self.safety_threshold,
            "eval_every_n_chunks": self.eval_every_n_chunks,
            "eval_on_sentence_end": self.eval_on_sentence_end,
        }


@dataclass
class EarlyStopCondition:
    """Defines a condition for early stopping."""

    name: str
    eval_name: str  # Which evaluation to check
    threshold: float  # Threshold value
    comparison: str = "below"  # "below" or "above"
    consecutive_chunks: int = 1  # How many consecutive chunks must fail
    enabled: bool = True

    def check(self, score: float, consecutive_count: int) -> bool:
        """Check if this condition triggers early stop."""
        if not self.enabled:
            return False

        if consecutive_count < self.consecutive_chunks:
            return False

        if self.comparison == "below":
            return score < self.threshold
        else:  # above
            return score > self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "eval_name": self.eval_name,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "consecutive_chunks": self.consecutive_chunks,
            "enabled": self.enabled,
        }
