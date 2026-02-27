"""
Backend Discovery for Guardrails.

Auto-detects available backends based on environment variables,
API keys, and hardware capabilities.
"""

import os
from typing import Dict, List, Optional, Tuple

from fi.evals.guardrails.config import GuardrailModel
from fi.evals.guardrails.registry import MODEL_REGISTRY, ModelInfo, get_model_info


class BackendDiscovery:
    """
    Auto-discover available guardrail backends.

    Checks for:
    - API keys (OpenAI, Azure, FutureAGI)
    - VLLM servers (via environment variables)
    - GPU availability for local models

    Usage:
        discovery = BackendDiscovery()
        available = discovery.discover()
        print(f"Available backends: {[m.value for m in available]}")

        # Get details
        details = discovery.get_availability_details()
        for model, info in details.items():
            print(f"{model}: {info['status']} - {info['reason']}")
    """

    def __init__(self):
        """Initialize discovery."""
        self._cache: Optional[List[GuardrailModel]] = None
        self._details_cache: Optional[Dict[str, Dict]] = None

    def discover(self, force_refresh: bool = False) -> List[GuardrailModel]:
        """
        Discover available backends.

        Args:
            force_refresh: Bypass cache and re-check

        Returns:
            List of available GuardrailModel values
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        available = []

        # Check API backends
        if self._check_fi_credentials():
            available.append(GuardrailModel.TURING_FLASH)
            available.append(GuardrailModel.TURING_SAFETY)

        if self._check_openai_key():
            available.append(GuardrailModel.OPENAI_MODERATION)

        if self._check_azure_credentials():
            available.append(GuardrailModel.AZURE_CONTENT_SAFETY)

        # Check local models via VLLM servers
        for model_value, info in MODEL_REGISTRY.items():
            if info.model_type == "local":
                vllm_url = self._get_vllm_url(model_value)
                if vllm_url and self._check_vllm_health(vllm_url):
                    available.append(info.model)

        self._cache = available
        return available

    def get_availability_details(self) -> Dict[str, Dict]:
        """
        Get detailed availability information for all models.

        Returns:
            Dict mapping model names to availability details
        """
        if self._details_cache is not None:
            return self._details_cache

        details = {}

        for model_value, info in MODEL_REGISTRY.items():
            status = "unavailable"
            reason = ""

            if info.model_type == "api":
                if model_value.startswith("turing"):
                    if self._check_fi_credentials():
                        status = "available"
                        reason = "FutureAGI credentials found"
                    else:
                        reason = "Missing FI_API_KEY or FI_SECRET_KEY"
                elif model_value == "openai-moderation":
                    if self._check_openai_key():
                        status = "available"
                        reason = "OPENAI_API_KEY found"
                    else:
                        reason = "Missing OPENAI_API_KEY"
                elif model_value == "azure-content-safety":
                    if self._check_azure_credentials():
                        status = "available"
                        reason = "Azure credentials found"
                    else:
                        reason = "Missing AZURE_CONTENT_SAFETY_ENDPOINT or AZURE_CONTENT_SAFETY_KEY"

            elif info.model_type == "local":
                vllm_url = self._get_vllm_url(model_value)
                if vllm_url:
                    if self._check_vllm_health(vllm_url):
                        status = "available"
                        reason = f"VLLM server at {vllm_url}"
                    else:
                        status = "unavailable"
                        reason = f"VLLM server at {vllm_url} not responding"
                else:
                    gpu_status = self._check_gpu_available()
                    if gpu_status[0]:
                        vram = gpu_status[1]
                        if info.vram_required_gb and vram and vram >= info.vram_required_gb:
                            status = "available"
                            reason = f"GPU available ({vram:.1f}GB VRAM)"
                        elif info.vram_required_gb:
                            status = "unavailable"
                            reason = f"Insufficient VRAM ({vram:.1f}GB < {info.vram_required_gb}GB required)"
                        else:
                            status = "available"
                            reason = "GPU available"
                    else:
                        status = "unavailable"
                        reason = "No GPU available and no VLLM server configured"

            details[model_value] = {
                "status": status,
                "reason": reason,
                "model_type": info.model_type,
                "description": info.description,
                "hf_model": info.hf_model_name,
                "vram_required": info.vram_required_gb,
                "is_gated": info.is_gated,
            }

        self._details_cache = details
        return details

    def _check_fi_credentials(self) -> bool:
        """Check if FutureAGI credentials are available."""
        return bool(
            os.environ.get("FI_API_KEY") and
            os.environ.get("FI_SECRET_KEY")
        )

    def _check_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _check_azure_credentials(self) -> bool:
        """Check if Azure credentials are available."""
        return bool(
            os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT") and
            os.environ.get("AZURE_CONTENT_SAFETY_KEY")
        )

    def _get_vllm_url(self, model_value: str) -> Optional[str]:
        """Get VLLM server URL for a model."""
        # Try model-specific env var
        env_var = f"VLLM_{model_value.upper().replace('-', '_')}_URL"
        url = os.environ.get(env_var)
        if url:
            return url

        # Fall back to generic VLLM_SERVER_URL
        return os.environ.get("VLLM_SERVER_URL")

    def _check_vllm_health(self, url: str) -> bool:
        """Check if VLLM server is healthy."""
        try:
            import httpx
            base = url.rstrip('/')
            with httpx.Client(timeout=5.0) as client:
                # Try /health (VLLM), fall back to / (ollama)
                response = client.get(f"{base}/health")
                if response.status_code == 200:
                    return True
                response = client.get(base)
                return response.status_code == 200
        except Exception:
            return False

    def _check_gpu_available(self) -> Tuple[bool, Optional[float]]:
        """
        Check if GPU is available and get VRAM.

        Returns:
            Tuple of (is_available, vram_gb)
        """
        try:
            import torch

            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return (True, vram)

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't report VRAM, estimate based on typical Apple Silicon
                return (True, 16.0)  # Conservative estimate

        except ImportError:
            pass

        return (False, None)

    def _check_hf_token(self) -> bool:
        """Check if HuggingFace token is available."""
        return bool(
            os.environ.get("HF_TOKEN") or
            os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )


def discover_backends() -> List[GuardrailModel]:
    """
    Convenience function to discover available backends.

    Returns:
        List of available GuardrailModel values
    """
    return BackendDiscovery().discover()


def get_backend_details() -> Dict[str, Dict]:
    """
    Convenience function to get detailed availability info.

    Returns:
        Dict mapping model names to availability details
    """
    return BackendDiscovery().get_availability_details()
