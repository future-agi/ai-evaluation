"""Auth + HTTP client primitives.

``APIKeyAuth`` is the common base for every cloud client in the SDK. It
reads credentials from env vars (``FI_API_KEY`` / ``FI_SECRET_KEY``)
unless explicit keys are passed.
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, TypeVar, Union

from requests import Response
from requests_futures.sessions import FuturesSession

from fi.api.types import RequestConfig
from fi.utils.constants import (
    API_KEY_ENVVAR_NAME,
    DEFAULT_MAX_QUEUE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIMEOUT,
    SECRET_KEY_ENVVAR_NAME,
    get_base_url,
)
from fi.utils.errors import DatasetNotFoundError, MissingAuthError
from fi.utils.executor import BoundedExecutor

T = TypeVar("T")
U = TypeVar("U")


class ResponseHandler(Generic[T, U], ABC):
    """Parses + validates a requests.Response into a typed result."""

    @classmethod
    def parse(cls, response: Response) -> Union[T, U]:
        if not response.ok or response.status_code != 200:
            cls._handle_error(response)
        return cls._parse_success(response)

    @classmethod
    @abstractmethod
    def _parse_success(cls, response: Response) -> Union[T, U]:
        ...

    @classmethod
    @abstractmethod
    def _handle_error(cls, response: Response) -> None:
        ...


class HttpClient:
    """Thin async-capable HTTP client with retries."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        session: Optional[FuturesSession] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self._base_url = (base_url or get_base_url()).rstrip("/")
        self._session = session or FuturesSession(
            executor=BoundedExecutor(
                bound=kwargs.get("max_queue", DEFAULT_MAX_QUEUE),
                max_workers=kwargs.get("max_workers", DEFAULT_MAX_WORKERS),
            ),
        )
        self._default_headers = default_headers or {}
        self._default_timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)

    def request(
        self,
        config: RequestConfig,
        response_handler: Optional[ResponseHandler[T, U]] = None,
    ) -> Union[Response, T]:
        url = config.url
        headers = {**self._default_headers, **(config.headers or {})}
        params = config.params or {}
        json_body = config.json or {}
        timeout = config.timeout or self._default_timeout
        files = config.files or {}
        data = config.data or {}

        for attempt in range(config.retry_attempts):
            try:
                response = self._session.request(
                    method=config.method.value,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    data=data,
                    files=files,
                    timeout=timeout,
                ).result()

                if response_handler:
                    return response_handler.parse(response=response)
                return response

            except Exception as exc:
                if isinstance(exc, DatasetNotFoundError):
                    raise
                if attempt == config.retry_attempts - 1:
                    raise
                time.sleep(config.retry_delay)

    def close(self) -> None:
        self._session.close()


class APIKeyAuth(HttpClient):
    """HTTP client that injects FutureAGI API + secret key headers."""

    _fi_api_key: Optional[str] = None
    _fi_secret_key: Optional[str] = None

    def __init__(
        self,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.__class__._fi_api_key = fi_api_key or os.environ.get(API_KEY_ENVVAR_NAME)
        self.__class__._fi_secret_key = fi_secret_key or os.environ.get(SECRET_KEY_ENVVAR_NAME)
        if self._fi_api_key is None or self._fi_secret_key is None:
            raise MissingAuthError(self._fi_api_key, self._fi_secret_key)

        super().__init__(
            base_url=fi_base_url,
            default_headers={
                "X-Api-Key": self._fi_api_key,
                "X-Secret-Key": self._fi_secret_key,
            },
            **kwargs,
        )
