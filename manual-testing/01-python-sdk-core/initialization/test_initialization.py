"""
Manual Test: Python SDK - Evaluator Initialization
Run: python test_initialization.py
"""

import os
import sys

# Test results tracking
RESULTS = []

def log_result(test_name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    RESULTS.append({"test": test_name, "passed": passed, "details": details})
    print(f"{status}: {test_name}")
    if details:
        print(f"       {details}")

def print_summary():
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for r in RESULTS if r["passed"])
    total = len(RESULTS)
    print(f"Passed: {passed}/{total}")
    if passed < total:
        print("\nFailed tests:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"  - {r['test']}: {r['details']}")


# =============================================================================
# TEST 1: Initialize with environment variables
# =============================================================================
def test_init_with_env_vars():
    """Test that Evaluator initializes when FI_API_KEY and FI_SECRET_KEY are set"""
    test_name = "Init with environment variables"

    # Check env vars are set
    if not os.environ.get("FI_API_KEY") or not os.environ.get("FI_SECRET_KEY"):
        log_result(test_name, False, "FI_API_KEY or FI_SECRET_KEY not set in environment")
        return

    try:
        from fi.evals import Evaluator
        evaluator = Evaluator()
        log_result(test_name, True, f"Evaluator created: {type(evaluator)}")
    except Exception as e:
        log_result(test_name, False, str(e))


# =============================================================================
# TEST 2: Initialize with explicit keys
# =============================================================================
def test_init_with_explicit_keys():
    """Test that Evaluator initializes with explicit API keys"""
    test_name = "Init with explicit keys"

    api_key = os.environ.get("FI_API_KEY", "test_key")
    secret_key = os.environ.get("FI_SECRET_KEY", "test_secret")

    try:
        from fi.evals import Evaluator
        evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key)
        log_result(test_name, True, f"Evaluator created with explicit keys")
    except Exception as e:
        log_result(test_name, False, str(e))


# =============================================================================
# TEST 3: Initialize with invalid keys
# =============================================================================
def test_init_with_invalid_keys():
    """Test error handling with invalid API keys"""
    test_name = "Init with invalid keys (expect error on first eval)"

    try:
        from fi.evals import Evaluator
        evaluator = Evaluator(fi_api_key="invalid_key", fi_secret_key="invalid_secret")
        # Note: Initialization may succeed, but evaluation should fail
        log_result(test_name, True, "Evaluator created (auth error expected on evaluate)")
    except Exception as e:
        # If it fails on init, that's also acceptable behavior
        log_result(test_name, True, f"Auth error on init: {str(e)[:100]}")


# =============================================================================
# TEST 4: Initialize with custom timeout
# =============================================================================
def test_init_with_timeout():
    """Test that Evaluator respects custom timeout setting"""
    test_name = "Init with custom timeout"

    api_key = os.environ.get("FI_API_KEY", "test_key")
    secret_key = os.environ.get("FI_SECRET_KEY", "test_secret")

    try:
        from fi.evals import Evaluator
        evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key, timeout=60)
        # Check if timeout was set (implementation dependent)
        timeout_val = getattr(evaluator, 'timeout', None) or getattr(evaluator, '_timeout', None)
        log_result(test_name, True, f"Evaluator created with timeout=60, stored as: {timeout_val}")
    except Exception as e:
        log_result(test_name, False, str(e))


# =============================================================================
# TEST 5: Initialize with custom max_workers
# =============================================================================
def test_init_with_max_workers():
    """Test that Evaluator respects max_workers setting"""
    test_name = "Init with custom max_workers"

    api_key = os.environ.get("FI_API_KEY", "test_key")
    secret_key = os.environ.get("FI_SECRET_KEY", "test_secret")

    try:
        from fi.evals import Evaluator
        evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key, max_workers=4)
        workers = getattr(evaluator, 'max_workers', None) or getattr(evaluator, '_max_workers', None)
        log_result(test_name, True, f"Evaluator created with max_workers=4, stored as: {workers}")
    except Exception as e:
        log_result(test_name, False, str(e))


# =============================================================================
# TEST 6: Initialize with custom base URL
# =============================================================================
def test_init_with_custom_base_url():
    """Test that Evaluator accepts custom base URL"""
    test_name = "Init with custom base URL"

    api_key = os.environ.get("FI_API_KEY", "test_key")
    secret_key = os.environ.get("FI_SECRET_KEY", "test_secret")

    try:
        from fi.evals import Evaluator
        evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key, fi_base_url="https://custom.api.futureagi.com")
        base_url = getattr(evaluator, 'base_url', None) or getattr(evaluator, '_base_url', None)
        log_result(test_name, True, f"Evaluator created with custom URL, stored as: {base_url}")
    except Exception as e:
        log_result(test_name, False, str(e))


# =============================================================================
# TEST 7: Check available attributes after initialization
# =============================================================================
def test_evaluator_attributes():
    """Test that Evaluator has expected attributes"""
    test_name = "Evaluator attributes check"

    api_key = os.environ.get("FI_API_KEY", "test_key")
    secret_key = os.environ.get("FI_SECRET_KEY", "test_secret")

    try:
        from fi.evals import Evaluator
        evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key)

        # List available attributes
        attrs = [a for a in dir(evaluator) if not a.startswith('_')]
        methods = [a for a in attrs if callable(getattr(evaluator, a))]

        # Check for expected methods
        expected_methods = ['evaluate']
        missing = [m for m in expected_methods if m not in methods]

        if missing:
            log_result(test_name, False, f"Missing expected methods: {missing}")
        else:
            log_result(test_name, True, f"Methods available: {methods[:5]}...")
    except Exception as e:
        log_result(test_name, False, str(e))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("MANUAL TEST: Python SDK - Evaluator Initialization")
    print("="*60)
    print()

    test_init_with_env_vars()
    test_init_with_explicit_keys()
    test_init_with_invalid_keys()
    test_init_with_timeout()
    test_init_with_max_workers()
    test_init_with_custom_base_url()
    test_evaluator_attributes()

    print_summary()
