"""
test_engine.py — C executable subprocess boundary test suite.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch

from backend.engine import (
    execute_engine,
    ping_engine,
    get_engine_path,
    EngineError,
    EngineNotFoundError,
    EngineTimeoutError,
    EngineCrashError
)


class TestEngineSubprocess:
    """Test process lifecycle and boundaries for the C engine."""

    def test_engine_executable_exists(self):
        exe = get_engine_path()
        assert exe.is_file(), f"Expected binary at {exe}"

    def test_ping_engine_success(self):
        info = ping_engine()
        assert info["healthy"] is True
        assert info["status"] == "PONG"
        assert info["version"] == "1.0.0"

    def test_engine_not_found_raises(self):
        fake_path = Path("D:/nonexistent/path/zeroform_missing.exe")
        with pytest.raises(EngineNotFoundError, match="executable not found"):
            execute_engine("CMD PING v1\nEND\n", exe_path=fake_path)

    def test_engine_timeout_raises(self):
        # Mock subprocess.run raising TimeoutExpired
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["zeroform.exe"], timeout=0.01)):
            with pytest.raises(EngineTimeoutError, match="timed out"):
                execute_engine("CMD PING v1\nEND\n", timeout=0.01)

    def test_engine_crash_raises(self):
        # Mock subprocess returning non-zero exit code
        mock_proc = subprocess.CompletedProcess(
            args=["zeroform.exe"],
            returncode=-11,  # SIGSEGV simulation
            stdout="",
            stderr="Segmentation fault (core dumped)"
        )
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(EngineCrashError) as exc_info:
                execute_engine("CMD PING v1\nEND\n")
            assert exc_info.value.exit_code == -11
            assert "Segmentation fault" in exc_info.value.stderr

    def test_engine_handles_multiple_sequential_invocations(self):
        # Verify statelessness: one request per process invocation
        for _ in range(5):
            stdout = execute_engine("CMD PING v1\nEND\n")
            assert "OK" in stdout
            assert "PONG" in stdout

    def test_get_engine_path_env_override(self, monkeypatch, tmp_path):
        fake_binary = tmp_path / "custom_zeroform_engine"
        fake_binary.write_text("#!/bin/sh\n")
        monkeypatch.setenv("ZEROFORM_EXECUTABLE", str(fake_binary))
        assert get_engine_path() == fake_binary.resolve()

    def test_get_engine_path_legacy_env_override(self, monkeypatch, tmp_path):
        fake_binary = tmp_path / "legacy_zeroform_engine.exe"
        fake_binary.write_text("dummy")
        monkeypatch.delenv("ZEROFORM_EXECUTABLE", raising=False)
        monkeypatch.setenv("ZEROFORM_EXE", str(fake_binary))
        assert get_engine_path() == fake_binary.resolve()

    def test_get_engine_path_platform_awareness(self, monkeypatch):
        monkeypatch.delenv("ZEROFORM_EXECUTABLE", raising=False)
        monkeypatch.delenv("ZEROFORM_EXE", raising=False)
        with patch("sys.platform", "linux"):
            path_linux = get_engine_path()
            # On Linux without zeroform compiled, should default towards zeroform
            assert path_linux.name in ("zeroform", "zeroform.exe")
