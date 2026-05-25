"""Same server-spawning fixture as the skeg-py tests."""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
TARGET = REPO_ROOT / "target" / "release"
DEFAULT_SKEG = TARGET / "skeg"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_tcp(port: int, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="session")
def binary_server() -> dict:
    skeg = os.environ.get("SKEG_BIN") or str(DEFAULT_SKEG)
    if not Path(skeg).exists():
        pytest.skip(f"skeg binary not found at {skeg}")
    data_dir = Path(tempfile.mkdtemp(prefix="skeg-llamaindex-"))
    port = _free_port()
    proc = subprocess.Popen(
        [skeg, "--data-dir", str(data_dir), "--addr", f"127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not _wait_tcp(port):
        proc.terminate()
        pytest.fail("skeg-server did not start")
    yield {"host": "127.0.0.1", "port": port, "data_dir": data_dir}
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)
