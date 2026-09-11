from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class RequestConfig(BaseModel):
    """Configuration for an HTTP request."""

    method: HttpMethod
    url: str
    headers: Optional[Dict[str, str]] = {}
    params: Optional[Dict[str, Any]] = {}
    files: Optional[Dict[str, Any]] = {}
    data: Optional[Dict[str, Any]] = {}
    json: Optional[Dict[str, Any]] = {}
    timeout: Optional[int] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0

    model_config = ConfigDict(protected_namespaces=())
