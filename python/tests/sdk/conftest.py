"""Pytest configuration and fixtures for SDK tests."""

import os
import pytest


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch):
    """Automatically set mock API keys for all tests."""
    monkeypatch.setenv("FI_API_KEY", "test_api_key")
    monkeypatch.setenv("FI_SECRET_KEY", "test_secret_key")


@pytest.fixture
def clean_env(monkeypatch):
    """Remove API keys from environment."""
    monkeypatch.delenv("FI_API_KEY", raising=False)
    monkeypatch.delenv("FI_SECRET_KEY", raising=False)


@pytest.fixture
def sample_rag_inputs():
    """Sample RAG evaluation inputs."""
    return {
        "context": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was constructed from 1887 to 1889 and is 324 meters tall.",
        "query": "How tall is the Eiffel Tower?",
        "output": "The Eiffel Tower is 324 meters tall."
    }


@pytest.fixture
def sample_safety_inputs():
    """Sample safety evaluation inputs."""
    return {
        "text": "This is a safe and appropriate message for all audiences."
    }


@pytest.fixture
def sample_conversation():
    """Sample conversation for evaluation."""
    return {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
            {"role": "user", "content": "Can you tell me about the weather?"},
            {"role": "assistant", "content": "I don't have access to real-time weather data, but I'd recommend checking a weather service like Weather.com or your local forecast."}
        ]
    }


@pytest.fixture
def sample_json_output():
    """Sample JSON output for format validation."""
    return {
        "response": '{"name": "John", "age": 30, "city": "New York"}'
    }


@pytest.fixture
def sample_translation():
    """Sample translation inputs."""
    return {
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?",
        "source_language": "English",
        "target_language": "French"
    }
