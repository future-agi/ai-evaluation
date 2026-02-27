"""
Distributed backend example for fi-evals.

Demonstrates submitting evaluation tasks to Celery, Temporal, or Ray
backends. Each backend requires its infrastructure to be running first
(see python/docker/ for Docker Compose files).

Usage:
    # Start infrastructure first:
    cd python/docker/celery && docker compose up -d

    # Then run:
    poetry run python examples/10_distributed_backends.py celery
    poetry run python examples/10_distributed_backends.py temporal
    poetry run python examples/10_distributed_backends.py ray
"""

import sys
import time


# ── Sample scoring function ─────────────────────────────────────────────────
# This is the function that gets serialized and sent to the remote worker.

def score_text(text: str) -> dict:
    """A trivial scorer that counts words and characters."""
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "score": min(len(words) / 10.0, 1.0),
    }


SAMPLE_INPUT = "The quick brown fox jumps over the lazy dog"


# ── Backend launchers ───────────────────────────────────────────────────────

def run_celery():
    from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

    config = CeleryConfig(
        broker_url="redis://localhost:6379/0",
        result_backend="redis://localhost:6379/1",
        task_queue="eval_tasks",
    )
    backend = CeleryBackend(config)
    print(f"[celery] Connected to broker: {config.broker_url}")

    handle = backend.submit(score_text, args=(SAMPLE_INPUT,))
    print(f"[celery] Submitted task: {handle.task_id}")

    result = backend.get_result(handle, timeout=30)
    print(f"[celery] Result: {result}")
    backend.shutdown()


def run_temporal():
    from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig

    config = TemporalConfig(
        host="localhost:7233",
        namespace="default",
        task_queue="eval-tasks",
    )
    backend = TemporalBackend(config)
    print(f"[temporal] Connected to: {config.host}")

    handle = backend.submit(score_text, args=(SAMPLE_INPUT,))
    print(f"[temporal] Started workflow: {handle.task_id}")

    result = backend.get_result(handle, timeout=30)
    print(f"[temporal] Result: {result}")
    backend.shutdown()


def run_ray():
    from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

    config = RayConfig(address="ray://localhost:10001")
    backend = RayBackend(config)
    print(f"[ray] Connected to: {config.address}")

    handle = backend.submit(score_text, args=(SAMPLE_INPUT,))
    print(f"[ray] Submitted task: {handle.task_id}")

    result = backend.get_result(handle, timeout=30)
    print(f"[ray] Result: {result}")
    backend.shutdown()


# ── Main ────────────────────────────────────────────────────────────────────

BACKENDS = {
    "celery": run_celery,
    "temporal": run_temporal,
    "ray": run_ray,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BACKENDS:
        print(f"Usage: python {sys.argv[0]} [{' | '.join(BACKENDS)}]")
        print()
        print("Start the infrastructure first:")
        print("  cd python/docker/celery   && docker compose up -d")
        print("  cd python/docker/temporal && docker compose up -d")
        print("  cd python/docker/ray      && docker compose up -d")
        sys.exit(1)

    backend_name = sys.argv[1]
    print(f"Running {backend_name} backend example...")
    start = time.time()

    try:
        BACKENDS[backend_name]()
        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Is the {backend_name} infrastructure running?")
        print(f"  cd python/docker/{backend_name} && docker compose up -d")
        sys.exit(1)


if __name__ == "__main__":
    main()
