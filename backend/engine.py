"""
engine.py — Subprocess execution wrapper for ZEROFORM C CLI engine.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.

Enforces:
1. Subprocess boundary with shell=False.
2. Bounded execution with explicit timeout.
3. Separate capture of stdout and stderr.
4. Clean exception hierarchy.
5. Absolute or securely resolved executable path.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class EngineError(Exception):
    """Base exception for C engine execution issues."""
    pass


class EngineNotFoundError(EngineError):
    """Raised when zeroform.exe is not found or not executable."""
    pass


class EngineTimeoutError(EngineError):
    """Raised when the C engine process exceeds timeout."""
    pass


class EngineCrashError(EngineError):
    """Raised when the C engine process crashes or exits with a nonzero code."""
    def __init__(self, exit_code: int, stderr: str):
        super().__init__(f"C engine process exited with code {exit_code}: {stderr.strip()}")
        self.exit_code = exit_code
        self.stderr = stderr.strip()

def get_engine_path() -> Path:
    """
    Resolve the absolute path to the C engine executable.

    Resolution order:
    1. ZEROFORM_EXECUTABLE environment variable (if set and points to an existing file).
    2. ZEROFORM_EXE environment variable (alias/legacy, if set and points to an existing file).
    3. Platform-specific default relative to project root:
       - Windows (win32): <project_root>/c_engine/zeroform.exe
       - Linux/POSIX:     <project_root>/c_engine/zeroform
       If the primary platform file does not exist, checks the alternate name as fallback.
    """
    env_override = os.environ.get("ZEROFORM_EXECUTABLE") or os.environ.get("ZEROFORM_EXE")
    if env_override:
        p = Path(env_override).resolve()
        if p.is_file():
            return p

    project_root = Path(__file__).resolve().parent.parent
    c_engine_dir = project_root / "c_engine"

    if sys.platform == "win32":
        primary = c_engine_dir / "zeroform.exe"
        fallback = c_engine_dir / "zeroform"
    else:
        primary = c_engine_dir / "zeroform"
        fallback = c_engine_dir / "zeroform.exe"

    if primary.is_file():
        return primary
    if fallback.is_file():
        return fallback

    return primary


def execute_engine(request_text: str, timeout: float = 5.0, exe_path: Optional[Path] = None) -> str:
    """
    Run zeroform.exe with the given request text piped into stdin.
    Captures stdout and stderr separately.
    Returns stdout string on success (exit code 0).
    
    Raises:
        EngineNotFoundError: If executable does not exist.
        EngineTimeoutError: If execution exceeds timeout.
        EngineCrashError: If executable exits with non-zero code.
    """
    target_exe = exe_path if exe_path is not None else get_engine_path()
    
    if not target_exe.is_file():
        raise EngineNotFoundError(f"ZEROFORM engine executable not found at: {target_exe}")

    try:
        proc = subprocess.run(
            [str(target_exe)],
            input=request_text,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )
    except FileNotFoundError as exc:
        raise EngineNotFoundError(f"Failed to execute {target_exe}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EngineTimeoutError(
            f"ZEROFORM engine timed out after {timeout:.1f} seconds"
        ) from exc
    except Exception as exc:
        raise EngineError(f"Unexpected subprocess error: {exc}") from exc

    if proc.returncode != 0:
        raise EngineCrashError(proc.returncode, proc.stderr)

    return proc.stdout


def ping_engine(timeout: float = 2.0, exe_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Execute a PING command against the C engine.
    Returns parsed status dictionary or raises EngineError.
    """
    request_text = "CMD PING v1\nEND\n"
    stdout = execute_engine(request_text, timeout=timeout, exe_path=exe_path)
    
    # Quick inline verification of PING response
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines or lines[0] != "OK" or lines[-1] != "END":
        raise EngineError("Invalid PING response framing from engine")
    
    info: Dict[str, Any] = {"healthy": True}
    for line in lines[1:-1]:
        if line.startswith("D STATUS "):
            info["status"] = line.split(" ", 2)[2]
        elif line.startswith("D VERSION "):
            info["version"] = line.split(" ", 2)[2]
            
    return info
