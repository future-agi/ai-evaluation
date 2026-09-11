#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

.venv/bin/ruff check \
  src/fi/alk/harness \
  tests/test_harness_architecture.py \
  tests/test_harness_service_environments.py

.venv/bin/pytest -q tests/test_harness*.py tests/runtime/test_livekit_engine.py

if [[ "${RUN_INTEGRATION:-0}" == "1" ]]; then
  RUN_INTEGRATION=1 .venv/bin/pytest -q \
    tests/test_harness_service_environments.py::test_sealed_bundle_restarts_the_admitted_source_not_the_mutated_checkout
fi
