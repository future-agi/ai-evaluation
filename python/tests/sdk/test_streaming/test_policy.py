"""Tests for EarlyStopPolicy."""

import pytest

from fi.evals.streaming.policy import EarlyStopPolicy, PolicyState
from fi.evals.streaming.types import ChunkResult, EarlyStopReason


class TestPolicyState:
    """Tests for PolicyState dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        state = PolicyState()
        assert state.consecutive_failures == {}
        assert state.total_failures == {}
        assert state.triggered_conditions == []


class TestEarlyStopPolicy:
    """Tests for EarlyStopPolicy."""

    def test_create_empty(self):
        """Should create with no conditions."""
        policy = EarlyStopPolicy()
        chunk = ChunkResult(0, "", "", {"toxicity": 0.9}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is False
        assert reason == EarlyStopReason.NONE

    def test_add_condition(self):
        """Should add threshold-based condition."""
        policy = EarlyStopPolicy()
        policy.add_condition(
            name="toxicity_stop",
            eval_name="toxicity",
            threshold=0.7,
            comparison="above",
        )

        # Below threshold, should not stop
        chunk = ChunkResult(0, "", "", {"toxicity": 0.5}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is False

        # Above threshold, should stop
        chunk = ChunkResult(1, "", "", {"toxicity": 0.9}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True
        assert reason == EarlyStopReason.TOXICITY

    def test_add_condition_chaining(self):
        """add_condition should return self for chaining."""
        policy = EarlyStopPolicy()
        result = policy.add_condition("a", "a", 0.5)
        assert result is policy

    def test_add_toxicity_stop(self):
        """add_toxicity_stop should create toxicity condition."""
        policy = EarlyStopPolicy()
        policy.add_toxicity_stop(threshold=0.5)

        chunk = ChunkResult(0, "", "", {"toxicity": 0.8}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True
        assert reason == EarlyStopReason.TOXICITY

    def test_add_safety_stop(self):
        """add_safety_stop should create safety condition."""
        policy = EarlyStopPolicy()
        policy.add_safety_stop(threshold=0.5)

        # Safety uses "below" comparison
        chunk = ChunkResult(0, "", "", {"safety": 0.3}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True
        assert reason == EarlyStopReason.SAFETY

    def test_add_quality_stop(self):
        """add_quality_stop should create quality condition."""
        policy = EarlyStopPolicy()
        policy.add_quality_stop(threshold=0.4, consecutive=2)

        # First chunk below threshold
        chunk1 = ChunkResult(0, "", "", {"quality": 0.2}, {})
        should_stop, _ = policy.check(chunk1)
        assert should_stop is False  # Need 2 consecutive

        # Second chunk below threshold
        chunk2 = ChunkResult(1, "", "", {"quality": 0.3}, {})
        should_stop, reason = policy.check(chunk2)
        assert should_stop is True
        assert reason == EarlyStopReason.THRESHOLD

    def test_consecutive_chunks_requirement(self):
        """Should require consecutive failures before triggering."""
        policy = EarlyStopPolicy()
        policy.add_condition(
            name="test",
            eval_name="score",
            threshold=0.5,
            comparison="below",
            consecutive_chunks=3,
        )

        # First two below threshold
        for i in range(2):
            chunk = ChunkResult(i, "", "", {"score": 0.3}, {})
            should_stop, _ = policy.check(chunk)
            assert should_stop is False

        # Third below threshold, should trigger
        chunk = ChunkResult(2, "", "", {"score": 0.3}, {})
        should_stop, _ = policy.check(chunk)
        assert should_stop is True

    def test_consecutive_resets_on_pass(self):
        """Consecutive count should reset when condition passes."""
        policy = EarlyStopPolicy()
        policy.add_condition(
            name="test",
            eval_name="score",
            threshold=0.5,
            comparison="below",
            consecutive_chunks=3,
        )

        # Two below threshold
        for i in range(2):
            chunk = ChunkResult(i, "", "", {"score": 0.3}, {})
            policy.check(chunk)

        # One above threshold (reset)
        chunk = ChunkResult(2, "", "", {"score": 0.7}, {})
        policy.check(chunk)

        # Two more below threshold (should not trigger yet)
        for i in range(2):
            chunk = ChunkResult(3 + i, "", "", {"score": 0.3}, {})
            should_stop, _ = policy.check(chunk)
            assert should_stop is False

    def test_missing_eval_score(self):
        """Should handle missing evaluation scores."""
        policy = EarlyStopPolicy()
        policy.add_condition("test", "missing_eval", 0.5, "below")

        # Score not present, should not trigger
        chunk = ChunkResult(0, "", "", {"other": 0.3}, {})
        should_stop, _ = policy.check(chunk)
        assert should_stop is False

    def test_add_custom_check(self):
        """Should support custom check functions."""
        def custom_check(chunk_result):
            if "dangerous" in chunk_result.chunk_text:
                return EarlyStopReason.CUSTOM
            return None

        policy = EarlyStopPolicy()
        policy.add_custom_check(custom_check)

        # Safe text
        chunk = ChunkResult(0, "Hello world", "Hello world", {}, {})
        should_stop, _ = policy.check(chunk)
        assert should_stop is False

        # Dangerous text
        chunk = ChunkResult(1, "dangerous content", "dangerous content", {}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True
        assert reason == EarlyStopReason.CUSTOM

    def test_reset(self):
        """reset should clear policy state."""
        policy = EarlyStopPolicy()
        policy.add_condition("test", "score", 0.5, "below", consecutive_chunks=3)

        # Accumulate some failures
        for i in range(2):
            chunk = ChunkResult(i, "", "", {"score": 0.3}, {})
            policy.check(chunk)

        policy.reset()

        # After reset, need to start over
        chunk = ChunkResult(0, "", "", {"score": 0.3}, {})
        should_stop, _ = policy.check(chunk)
        assert should_stop is False

    def test_enable_disable_condition(self):
        """Should enable/disable conditions by name."""
        policy = EarlyStopPolicy()
        policy.add_condition("test", "score", 0.5, "above")

        # Disable condition
        policy.disable_condition("test")
        chunk = ChunkResult(0, "", "", {"score": 0.9}, {})
        should_stop, _ = policy.check(chunk)
        assert should_stop is False

        # Re-enable condition
        policy.enable_condition("test")
        should_stop, _ = policy.check(chunk)
        assert should_stop is True

    def test_get_stats(self):
        """get_stats should return policy statistics."""
        policy = EarlyStopPolicy()
        policy.add_condition("test1", "score", 0.5, "above")
        policy.add_condition("test2", "other", 0.3, "below")

        chunk = ChunkResult(0, "", "", {"score": 0.9, "other": 0.2}, {})
        policy.check(chunk)

        stats = policy.get_stats()
        assert "conditions" in stats
        assert "consecutive_failures" in stats
        assert "total_failures" in stats
        assert "triggered_conditions" in stats
        assert "custom_checks" in stats
        assert len(stats["conditions"]) == 2


class TestEarlyStopPolicyPresets:
    """Tests for policy presets."""

    def test_default_policy(self):
        """default() should create policy with toxicity and safety stops."""
        policy = EarlyStopPolicy.default()

        # Toxicity above 0.7 should stop
        chunk = ChunkResult(0, "", "", {"toxicity": 0.8}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True
        assert reason == EarlyStopReason.TOXICITY

        policy.reset()

        # Safety below 0.3 should stop
        chunk = ChunkResult(0, "", "", {"safety": 0.2}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True
        assert reason == EarlyStopReason.SAFETY

    def test_strict_policy(self):
        """strict() should have lower thresholds."""
        policy = EarlyStopPolicy.strict()

        # Toxicity above 0.5 should stop (stricter than default 0.7)
        chunk = ChunkResult(0, "", "", {"toxicity": 0.6}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True

    def test_permissive_policy(self):
        """permissive() should have higher thresholds."""
        policy = EarlyStopPolicy.permissive()

        # Toxicity at 0.8 should NOT stop (threshold is 0.9)
        chunk = ChunkResult(0, "", "", {"toxicity": 0.8}, {})
        should_stop, _ = policy.check(chunk)
        assert should_stop is False

        # Need 2 consecutive for toxicity in permissive
        chunk = ChunkResult(1, "", "", {"toxicity": 0.95}, {})
        should_stop, _ = policy.check(chunk)
        assert should_stop is False  # Only 1 so far

        chunk = ChunkResult(2, "", "", {"toxicity": 0.95}, {})
        should_stop, reason = policy.check(chunk)
        assert should_stop is True


class TestEarlyStopPolicyReasonMapping:
    """Tests for reason mapping from condition names."""

    def test_toxicity_reason(self):
        """Should map toxicity conditions to TOXICITY reason."""
        policy = EarlyStopPolicy()
        policy.add_condition("toxic_content", "toxic_score", 0.5, "above")

        chunk = ChunkResult(0, "", "", {"toxic_score": 0.9}, {})
        _, reason = policy.check(chunk)
        assert reason == EarlyStopReason.TOXICITY

    def test_safety_reason(self):
        """Should map safety conditions to SAFETY reason."""
        policy = EarlyStopPolicy()
        policy.add_condition("safety_check", "safe_score", 0.5, "below")

        chunk = ChunkResult(0, "", "", {"safe_score": 0.3}, {})
        _, reason = policy.check(chunk)
        assert reason == EarlyStopReason.SAFETY

    def test_pii_reason(self):
        """Should map PII conditions to PII reason."""
        policy = EarlyStopPolicy()
        policy.add_condition("pii_detection", "pii_score", 0.5, "above")

        chunk = ChunkResult(0, "", "", {"pii_score": 0.9}, {})
        _, reason = policy.check(chunk)
        assert reason == EarlyStopReason.PII

    def test_jailbreak_reason(self):
        """Should map jailbreak conditions to JAILBREAK reason."""
        policy = EarlyStopPolicy()
        policy.add_condition("jailbreak_detect", "jb", 0.5, "above")

        chunk = ChunkResult(0, "", "", {"jb": 0.9}, {})
        _, reason = policy.check(chunk)
        assert reason == EarlyStopReason.JAILBREAK

    def test_generic_reason(self):
        """Should use THRESHOLD for unrecognized conditions."""
        policy = EarlyStopPolicy()
        policy.add_condition("my_custom_check", "custom", 0.5, "above")

        chunk = ChunkResult(0, "", "", {"custom": 0.9}, {})
        _, reason = policy.check(chunk)
        assert reason == EarlyStopReason.THRESHOLD
