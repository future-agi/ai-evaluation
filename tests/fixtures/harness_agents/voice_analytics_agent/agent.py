"""Tiny submitted worker used to prove environment wiring, not a harness-owned tool."""

import base64
import http.client
import os
import socket
import time
from urllib.parse import urlsplit


def clickhouse_value() -> str:
    endpoint = urlsplit(os.environ.get("CLICKHOUSE_URL", "http://localhost:8123"))
    headers = {}
    if endpoint.username:
        token = base64.b64encode(
            f"{endpoint.username}:{endpoint.password or ''}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {token}"
    connection = http.client.HTTPConnection(
        endpoint.hostname, endpoint.port or 8123, timeout=5
    )
    connection.request(
        "GET", "/?query=SELECT%20city%20FROM%20voice.calls", headers=headers
    )
    response = connection.getresponse()
    return response.read().decode().strip()


def redis_ping() -> str:
    endpoint = urlsplit(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    with socket.create_connection(
        (endpoint.hostname, endpoint.port or 6379), timeout=5
    ) as client:
        client.sendall(b"*1\r\n$4\r\nPING\r\n")
        return client.recv(64).decode().strip()


print(f"AGENT_READY clickhouse={clickhouse_value()} redis={redis_ping()}", flush=True)
while True:
    time.sleep(60)
