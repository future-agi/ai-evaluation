"""Chunk buffer for streaming evaluation.

Accumulates tokens/chunks and manages when to trigger evaluations.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .types import StreamingConfig


@dataclass
class BufferState:
    """Current state of the chunk buffer."""

    total_text: str = ""
    pending_text: str = ""
    chunk_count: int = 0
    token_count: int = 0
    last_eval_time: float = 0.0
    last_eval_position: int = 0


class ChunkBuffer:
    """
    Buffer that accumulates streaming tokens and determines when to evaluate.

    Manages the accumulation of tokens from a streaming LLM response and
    decides when enough content has been collected to trigger an evaluation.

    Example:
        buffer = ChunkBuffer(config)

        for token in stream:
            buffer.add(token)
            if buffer.should_evaluate():
                chunk = buffer.get_chunk()
                result = evaluator.evaluate_chunk(chunk, buffer.get_cumulative())
                buffer.mark_evaluated()
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]\s*$')
    PARTIAL_SENTENCE = re.compile(r'[.!?]\s+')

    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize the buffer.

        Args:
            config: Streaming configuration (uses defaults if None)
        """
        self.config = config or StreamingConfig()
        self._state = BufferState()
        self._start_time = time.perf_counter()

    def add(self, token: str) -> None:
        """
        Add a token to the buffer.

        Args:
            token: The token text to add
        """
        self._state.total_text += token
        self._state.pending_text += token
        self._state.token_count += 1

    def add_chunk(self, chunk: str) -> None:
        """
        Add a larger chunk of text to the buffer.

        Args:
            chunk: The chunk text to add
        """
        self._state.total_text += chunk
        self._state.pending_text += chunk
        # Estimate token count (rough approximation)
        self._state.token_count += len(chunk.split())

    def should_evaluate(self) -> bool:
        """
        Check if we should trigger an evaluation.

        Returns:
            True if evaluation should be triggered
        """
        pending_len = len(self._state.pending_text)

        # Check minimum chunk size
        if pending_len < self.config.min_chunk_size:
            return False

        # Check time interval
        current_time = time.perf_counter()
        time_since_last = (current_time - self._state.last_eval_time) * 1000

        # Force evaluation if max chunk size reached
        if pending_len >= self.config.max_chunk_size:
            return True

        # Check time-based interval
        if time_since_last < self.config.eval_interval_ms:
            return False

        # Check sentence boundary if enabled
        if self.config.eval_on_sentence_end:
            if self.SENTENCE_ENDINGS.search(self._state.pending_text):
                return True

        # Check chunk count interval
        if self.config.eval_every_n_chunks > 1:
            # Only evaluate every N chunks
            next_chunk = self._state.chunk_count + 1
            if next_chunk % self.config.eval_every_n_chunks != 0:
                return pending_len >= self.config.max_chunk_size

        return pending_len >= self.config.min_chunk_size

    def get_chunk(self) -> str:
        """
        Get the pending chunk for evaluation.

        Returns:
            The pending text that should be evaluated
        """
        return self._state.pending_text

    def get_cumulative(self) -> str:
        """
        Get all accumulated text so far.

        Returns:
            The complete accumulated text
        """
        return self._state.total_text

    def mark_evaluated(self) -> None:
        """Mark the current pending content as evaluated."""
        self._state.chunk_count += 1
        self._state.last_eval_time = time.perf_counter()
        self._state.last_eval_position = len(self._state.total_text)
        self._state.pending_text = ""

    def get_chunk_index(self) -> int:
        """Get the current chunk index."""
        return self._state.chunk_count

    def get_token_count(self) -> int:
        """Get the total token count."""
        return self._state.token_count

    def get_char_count(self) -> int:
        """Get the total character count."""
        return len(self._state.total_text)

    def should_stop_for_limits(self) -> Tuple[bool, str]:
        """
        Check if we should stop due to limits.

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check token limit
        if self.config.max_tokens and self._state.token_count >= self.config.max_tokens:
            return True, "max_tokens"

        # Check character limit
        if self.config.max_chars and len(self._state.total_text) >= self.config.max_chars:
            return True, "max_chars"

        # Check total timeout
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        if elapsed_ms >= self.config.total_timeout_ms:
            return True, "timeout"

        return False, ""

    def reset(self) -> None:
        """Reset the buffer to initial state."""
        self._state = BufferState()
        self._start_time = time.perf_counter()

    @property
    def state(self) -> BufferState:
        """Get the current buffer state."""
        return self._state

    @property
    def is_empty(self) -> bool:
        """Check if buffer has no content."""
        return len(self._state.total_text) == 0

    @property
    def has_pending(self) -> bool:
        """Check if there's pending unevaluated content."""
        return len(self._state.pending_text) > 0

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        return {
            "total_chars": len(self._state.total_text),
            "pending_chars": len(self._state.pending_text),
            "chunk_count": self._state.chunk_count,
            "token_count": self._state.token_count,
            "elapsed_ms": elapsed_ms,
            "avg_chunk_size": (
                len(self._state.total_text) / self._state.chunk_count
                if self._state.chunk_count > 0
                else 0
            ),
        }
