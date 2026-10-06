import os
import socket
import subprocess
import sys
import time

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_server_factory():
    """Start GB10-simulator servers (fast time scale) and tear them down at the end."""
    procs = []

    def start(*flags: str) -> str:
        port = _free_port()
        env = {**os.environ, "SERVEBENCH_MOCK_TIME_SCALE": "0.02"}
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "servebench.mock",
                "Qwen/Qwen3-8B",
                "--port",
                str(port),
                "--served-model-name",
                "qwen3-8b",
                *flags,
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    return url
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError("mock server did not start")

    yield start
    for p in procs:
        p.terminate()
        p.wait(timeout=10)
