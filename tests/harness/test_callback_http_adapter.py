import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ADAPTER = (
    Path(__file__).resolve().parents[2] / "src/fi/alk/harness/callback_http_adapter.py"
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def adapter(tmp_path):
    (tmp_path / "chat_agent.py").write_text(
        "def reply(agent_input):\n"
        "    return {'content': 'thread ' + agent_input.thread_id}\n",
        encoding="utf-8",
    )
    # Bundles embed the adapter's source and run it from the bundle, not from the package.
    script = tmp_path / "adapter.py"
    script.write_text(ADAPTER.read_text(encoding="utf-8"), encoding="utf-8")
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(script)],
        env={
            **os.environ,
            "ALK_CALLBACK_ENTRYPOINT": "chat_agent:reply",
            "PORT": str(port),
            "PYTHONPATH": str(tmp_path),
        },
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            break
        except OSError:
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            time.sleep(0.2)
    yield base
    process.terminate()
    process.wait(timeout=10)


def test_a_callback_answers_from_the_request_thread(adapter):
    """The request thread is not the main thread; nothing imported there may require it."""
    body = json.dumps(
        {
            "thread_id": "t1",
            "messages": [{"role": "user", "content": "hello"}],
            "new_message": {"role": "user", "content": "hello"},
        }
    ).encode()
    request = urllib.request.Request(
        f"{adapter}/invoke", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        assert json.loads(response.read())["content"] == "thread t1"
