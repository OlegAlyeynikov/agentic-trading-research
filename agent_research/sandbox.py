from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_workspace_script(
    script_path: str | Path,
    workspace_dir: str | Path,
    data_dir: str,
    available_timeframes: list[str],
    symbols: list[str],
    start_date: str | None,
    timeout_seconds: int = 300,
    funding_rates_dir: str | None = None,
) -> dict:
    script_file = Path(script_path).resolve()
    workspace = Path(workspace_dir).resolve()
    stdout_path = workspace / "stdout.json"
    stderr_path = workspace / "stderr.log"
    setups_csv = workspace / "setups.csv"

    env = {
        "PYTHONPATH": str(Path(__file__).parent.parent),
        "AGENT_RESEARCH_WORKSPACE_DIR": str(workspace),
        "AGENT_RESEARCH_DATA_DIR": data_dir,
        "AGENT_RESEARCH_AVAILABLE_TIMEFRAMES": json.dumps(available_timeframes),
        "AGENT_RESEARCH_SYMBOLS": json.dumps(symbols),
        "AGENT_RESEARCH_SETUPS_CSV": str(setups_csv),
        "AGENT_RESEARCH_STDOUT_JSON": str(stdout_path),
    }
    if start_date:
        env["AGENT_RESEARCH_START_DATE"] = start_date
    if funding_rates_dir:
        env["AGENT_RESEARCH_FUNDING_RATES_DIR"] = funding_rates_dir
    max_open = os.environ.get("MAX_OPEN_POSITIONS", "")
    if max_open:
        env["AGENT_RESEARCH_MAX_OPEN_POSITIONS"] = max_open
    risk_pct = os.environ.get("RISK_PCT_PER_TRADE", "")
    if risk_pct:
        env["AGENT_RESEARCH_RISK_PCT"] = risk_pct

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_file)],
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        runtime_seconds = round(time.monotonic() - start, 3)
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        runtime_seconds = float(timeout_seconds)
        timeout_msg = f"Script timed out after {timeout_seconds}s — use vectorized O(n) operations, avoid per-candle loops"
        stderr_path.write_text(timeout_msg, encoding="utf-8")
        if exc.output:
            stdout_path.write_text(exc.output if isinstance(exc.output, str) else exc.output.decode(errors="replace"), encoding="utf-8")
        return {
            "returncode": 1,
            "runtime_seconds": runtime_seconds,
            "stdout_json": {},
            "stderr_path": str(stderr_path),
            "stdout_path": str(stdout_path),
            "setups_csv": str(setups_csv),
            "parse_error": timeout_msg,
        }

    stdout_json = {}
    parse_error = ""
    payload = (proc.stdout or "").strip()
    if payload:
        try:
            stdout_json = json.loads(payload)
        except json.JSONDecodeError as exc:
            parse_error = f"stdout is not valid JSON: {exc}"
    elif stdout_path.exists():
        try:
            stdout_json = json.loads(stdout_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parse_error = f"stdout file is not valid JSON: {exc}"
    else:
        parse_error = "script produced no JSON output"

    return {
        "returncode": proc.returncode,
        "runtime_seconds": runtime_seconds,
        "stdout_json": stdout_json,
        "stderr_path": str(stderr_path),
        "stdout_path": str(stdout_path),
        "setups_csv": str(setups_csv),
        "parse_error": parse_error,
    }
