"""
Code executor — writes a standalone workspace script, validates it,
executes it in a sandboxed subprocess, and captures generated setups.csv.
"""
import subprocess
import sys
import re
from pathlib import Path

import pandas as pd

from agent_research.state import ResearchState
from agent_research.logging_config import get_logger, AgentTimer
from agent_research.config_utils import load_full_config
from agent_research.memory import ExperimentStore
from agent_research.agents.coder import generate_script_with_coder
from agent_research.code_runtime import (
    create_workspace_dir,
    script_hash,
    static_script_checks,
    lookahead_bias_flags,
    normalize_script_chunks,
    write_workspace_helpers,
)
from agent_research.sandbox import run_workspace_script

_log = get_logger("code_executor")
MAX_CODE_REPAIR_ATTEMPTS = 3


def _requires_execution_bridge(state: ResearchState, script_contents: str) -> bool:
    flags = dict(state.get("strategy_flags") or {})
    proposal = dict(state.get("code_change_proposal") or {})
    proposal_flags = dict(proposal.get("strategy_flags") or {})
    funding_strategy = bool(
        flags.get("requires_funding_model")
        or proposal_flags.get("requires_funding_model")
    )
    return funding_strategy and "generate_single_asset_signals(" in script_contents


def _missing_execution_bridge_errors(script_contents: str) -> list[str]:
    errors: list[str] = []
    if "load_execution_price_data(" not in script_contents:
        errors.append(
            "funding single-asset strategy must call load_execution_price_data() before signal generation"
        )
    if "build_execution_bridge(" not in script_contents:
        errors.append(
            "funding single-asset strategy must call build_execution_bridge() instead of using same-timeframe fills"
        )
    if "execution_timestamps=" not in script_contents or "execution_prices=" not in script_contents:
        errors.append(
            "funding single-asset strategy must pass execution_timestamps and execution_prices into generate_single_asset_signals()"
        )
    return errors


def _funding_single_asset_static_errors(script_contents: str) -> list[str]:
    """Catch common funding-strategy script bugs before they become false data gaps."""
    errors: list[str] = []
    lower = script_contents.lower()
    search_markers = (
        "grid search",
        "random search",
        "parameter sweep",
        "brute force",
        "itertools.combinations",
        "itertools.product",
        "from itertools import combinations",
        "from itertools import product",
    )
    if any(marker in lower for marker in search_markers):
        errors.append(
            "workspace scripts must test one concrete configuration only; "
            "grid/random/bruteforce parameter or symbol searches belong in "
            "`python -m agent_research.funding_grid_search`, not in coder-generated scripts"
        )
    if "generate_single_asset_signals(" not in script_contents:
        return errors

    if re.search(r"\bprices\s*\[[^\]]+\]\.index\b", script_contents):
        errors.append(
            "funding single-asset scripts must use prices[sym]['timestamp'], not prices[sym].index; "
            "load_price_data() resets the DataFrame index to 0..N"
        )
    if re.search(r"\bprice_df\.index\b|\bdf\.index\b", script_contents) and "compute_peer_funding_zscore(" in script_contents:
        errors.append(
            "funding z-score alignment must use the price timestamp column, not price_df.index/df.index"
        )
    if re.search(r"timestamps\s*=\s*[^,\n]*\.index\b", script_contents):
        errors.append(
            "generate_single_asset_signals(timestamps=...) must receive the price timestamp column, not a DataFrame index"
        )
    if any(action in script_contents for action in ("EXIT_LONG", "STOP_LONG", "EXIT_SHORT", "STOP_SHORT")):
        errors.append(
            "generate_single_asset_signals() emits ENTER_LONG, ENTER_SHORT, EXIT, STOP; "
            "do not filter for EXIT_LONG/STOP_LONG/EXIT_SHORT/STOP_SHORT"
        )
    if re.search(r"except\s+Exception\s*:\s*\n\s*continue\b", script_contents):
        errors.append(
            "do not hide signal-generation failures with `except Exception: continue`; "
            "emit_blocked() with diagnostics or let the exception surface for repair"
        )
    if re.search(r"\bclose_row\s*=|Force close|force[-_ ]close", script_contents, re.IGNORECASE):
        errors.append(
            "do not manually append force-close EXIT rows; use max_holding_candles/canonical helper "
            "so every signal row has valid execution fields"
        )
    if "compute_peer_funding_zscore(" in script_contents and (
        "pseudo_sharpe" in script_contents or re.search(r"\bmean_z\s*/\s*std_z\b", script_contents)
    ):
        errors.append(
            "do not use funding_z mean/std as a pseudo sharpe pre-filter; generate signals first and let the backtest/reviewer evaluate performance"
        )
    return errors


def _is_repairable_blocked_report(stdout_json: dict, script_contents: str) -> bool:
    """Blocked no-signal reports from generated scripts often indicate code bugs, not market data gaps."""
    notes = str(stdout_json.get("notes") or "").lower()
    missing = " ".join(str(v).lower() for v in stdout_json.get("missing_requirements") or [])
    text = f"{notes} {missing}"
    if not text.strip():
        return False
    no_signal_like = (
        "no signals generated" in text
        or "no signals after" in text
        or "no fallback symbol" in text
        or "zero signals" in text
    )
    if not no_signal_like:
        return False
    has_diagnostics = bool((stdout_json.get("metrics") or {}).get("diagnostics"))
    if not has_diagnostics:
        return True
    diagnostics = (stdout_json.get("metrics") or {}).get("diagnostics") or {}
    symbols = diagnostics.get("symbols") or {}
    raw_triggers = sum(int((d or {}).get("count_beyond_entry") or 0) for d in symbols.values())
    if raw_triggers > 0:
        return True
    return bool(_funding_single_asset_static_errors(script_contents))


def run_code_executor(state: ResearchState) -> ResearchState:
    exp_id = state.get("current_experiment_id", "?")
    script_name = state.get("code_script_filename") or "workspace_strategy.py"
    script_contents = state.get("code_script_contents") or ""

    with AgentTimer(_log, "code_executor", exp_id, state.get("iteration", 0)):
        if state.get("dry_run"):
            print(f"[code_executor] {exp_id}  [dry-run] no sandbox execution")
            return {
                **state,
                "workspace_dir": "",
                "code_script_path": "",
                "generated_setups_csv": "",
                "script_hash": "dry-run",
                "setups_csv_hash": "",
                "lookahead_flags": [],
                "sandbox_result": {"returncode": 0, "stdout_json": {}, "runtime_seconds": 0.0},
                "validation_result": {
                    "passed": True,
                    "errors": [],
                    "compile_ok": True,
                    "static_checks_ok": True,
                    "lookahead_ok": True,
                    "sandbox_ok": True,
                },
            }

        workspace_dir = create_workspace_dir(exp_id)
        write_workspace_helpers(workspace_dir)
        if not script_contents.strip():
            return _fail(state, ["No workspace script contents provided by coder"])

        full_config = load_full_config(state["config_path"])
        current_scope = dict(state.get("current_scope") or {})
        selected_symbols = list(current_scope.get("selected_symbols") or full_config.get("symbols") or [])
        return _validate_and_repair(
            state=state,
            workspace_dir=workspace_dir,
            script_name=script_name,
            script_contents=script_contents,
            selected_symbols=selected_symbols,
            full_config=full_config,
        )


def _validate_and_repair(
    *,
    state: ResearchState,
    workspace_dir: Path,
    script_name: str,
    script_contents: str,
    selected_symbols: list[str],
    full_config: dict,
) -> ResearchState:
    exp_id = state.get("current_experiment_id", "?")
    proposal = dict(state.get("code_change_proposal") or {})
    store = ExperimentStore()
    repair_attempt = 0
    current_name = script_name
    current_contents = script_contents
    last_failure: dict | None = None

    while True:
        current_contents = normalize_script_chunks(current_contents)
        script_path = workspace_dir / current_name
        script_path.write_text(current_contents, encoding="utf-8")
        hash_value = script_hash(current_contents)
        _log.info(
            "code_executor_stage",
            exp_id=exp_id,
            stage="script_written",
            script_path=str(script_path),
            script_hash=hash_value,
            repair_attempt=repair_attempt,
        )
        print(f"[code_executor] {exp_id} stage=script_written repair={repair_attempt} hash={hash_value}")

        bridge_errors: list[str] = []
        if _requires_execution_bridge(state, current_contents):
            bridge_errors = _missing_execution_bridge_errors(current_contents)

        static_errors = (
            bridge_errors
            + _funding_single_asset_static_errors(current_contents)
            + static_script_checks(current_contents)
        )
        if static_errors:
            last_failure = {
                "errors": static_errors,
                "workspace_dir": workspace_dir,
                "script_path": script_path,
                "hash_value": hash_value,
                "static_checks_ok": False,
                "compile_ok": True,
                "lookahead_ok": True,
                "sandbox_ok": False,
                "sandbox_result": {},
                "lookahead_flags": [],
            }
        else:
            _log.info("code_executor_stage", exp_id=exp_id, stage="static_checks_passed")
            print(f"[code_executor] {exp_id} stage=static_checks_passed")
            lookahead_flags = lookahead_bias_flags(current_contents)
            if lookahead_flags:
                last_failure = {
                    "errors": [f"lookahead bias risk: {flag}" for flag in lookahead_flags],
                    "workspace_dir": workspace_dir,
                    "script_path": script_path,
                    "hash_value": hash_value,
                    "static_checks_ok": True,
                    "compile_ok": True,
                    "lookahead_ok": False,
                    "sandbox_ok": False,
                    "sandbox_result": {},
                    "lookahead_flags": lookahead_flags,
                }
            else:
                _log.info("code_executor_stage", exp_id=exp_id, stage="lookahead_checks_passed")
                print(f"[code_executor] {exp_id} stage=lookahead_checks_passed")
                compile_res = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(script_path)],
                    capture_output=True,
                    text=True,
                )
                if compile_res.returncode != 0:
                    last_failure = {
                        "errors": [compile_res.stderr.strip()[:500]],
                        "workspace_dir": workspace_dir,
                        "script_path": script_path,
                        "hash_value": hash_value,
                        "static_checks_ok": True,
                        "compile_ok": False,
                        "lookahead_ok": True,
                        "sandbox_ok": False,
                        "sandbox_result": {},
                        "lookahead_flags": [],
                    }
                else:
                    _log.info("code_executor_stage", exp_id=exp_id, stage="compile_passed")
                    print(f"[code_executor] {exp_id} stage=compile_passed")
                    prior = store.find_by_script_hash(hash_value)
                    if prior and prior.get("experiment_id") != exp_id:
                        print(
                            f"[code_executor] {exp_id}  DUPLICATE SCRIPT blocked"
                            f"  script_hash={hash_value}  matches={prior['experiment_id']}"
                        )
                        return {
                            **state,
                            "workspace_dir": str(workspace_dir),
                            "code_script_path": str(script_path),
                            "code_script_filename": current_name,
                            "code_script_contents": current_contents,
                            "generated_setups_csv": "",
                            "script_hash": hash_value,
                            "setups_csv_hash": "",
                            "lookahead_flags": [],
                            "sandbox_result": {},
                            "duplicate_of_experiment_id": prior["experiment_id"],
                            "validation_result": {
                                "passed": False,
                                "errors": [f"duplicate script: matches {prior['experiment_id']}"],
                                "compile_ok": True,
                                "static_checks_ok": True,
                                "lookahead_ok": True,
                                "sandbox_ok": False,
                                "repair_attempts": repair_attempt,
                            },
                        }
                    _log.info("code_executor_stage", exp_id=exp_id, stage="sandbox_started")
                    print(f"[code_executor] {exp_id} stage=sandbox_started")
                    sandbox_result = run_workspace_script(
                        script_path=script_path,
                        workspace_dir=workspace_dir,
                        data_dir=full_config["data_dir"],
                        available_timeframes=full_config.get("available_timeframes", ["1h"]),
                        symbols=selected_symbols,
                        start_date=full_config.get("start_date"),
                        funding_rates_dir=full_config.get("funding_rates_dir"),
                    )
                    _log.info(
                        "code_executor_stage",
                        exp_id=exp_id,
                        stage="sandbox_finished",
                        returncode=sandbox_result.get("returncode"),
                        runtime_seconds=sandbox_result.get("runtime_seconds"),
                        parse_error=bool(sandbox_result.get("parse_error")),
                    )
                    print(
                        f"[code_executor] {exp_id} stage=sandbox_finished "
                        f"rc={sandbox_result.get('returncode')} runtime={sandbox_result.get('runtime_seconds')}"
                    )
                    if sandbox_result["returncode"] != 0:
                        stderr_excerpt = _stderr_excerpt(sandbox_result.get("stderr_path", ""))
                        last_failure = {
                            "errors": [f"sandbox execution failed: {stderr_excerpt}"],
                            "workspace_dir": workspace_dir,
                            "script_path": script_path,
                            "hash_value": hash_value,
                            "static_checks_ok": True,
                            "compile_ok": True,
                            "lookahead_ok": True,
                            "sandbox_ok": False,
                            "sandbox_result": sandbox_result,
                            "lookahead_flags": [],
                        }
                    elif sandbox_result.get("parse_error"):
                        last_failure = {
                            "errors": [sandbox_result["parse_error"]],
                            "workspace_dir": workspace_dir,
                            "script_path": script_path,
                            "hash_value": hash_value,
                            "static_checks_ok": True,
                            "compile_ok": True,
                            "lookahead_ok": True,
                            "sandbox_ok": False,
                            "sandbox_result": sandbox_result,
                            "lookahead_flags": [],
                        }
                    else:
                        stdout_json = dict(sandbox_result.get("stdout_json") or {})
                        if stdout_json.get("status") == "blocked":
                            if _is_repairable_blocked_report(stdout_json, current_contents):
                                notes = str(stdout_json.get("notes") or "workspace script reported blocked/no-signal")
                                last_failure = {
                                    "errors": [
                                        "workspace script emitted a repairable blocked/no-signal report: "
                                        f"{notes}. Treat this as a possible script bug until diagnostics prove a true data gap."
                                    ],
                                    "workspace_dir": workspace_dir,
                                    "script_path": script_path,
                                    "hash_value": hash_value,
                                    "static_checks_ok": True,
                                    "compile_ok": True,
                                    "lookahead_ok": True,
                                    "sandbox_ok": False,
                                    "sandbox_result": sandbox_result,
                                    "lookahead_flags": [],
                                }
                                _log.info(
                                    "code_executor_repairable_blocked_report",
                                    exp_id=exp_id,
                                    notes=notes[:300],
                                    script_hash=hash_value,
                                )
                            else:
                                _log.info(
                                    "code_executor_blocked_report",
                                    exp_id=exp_id,
                                    workspace_dir=str(workspace_dir),
                                    script_path=str(script_path),
                                    script_hash=hash_value,
                                )
                                print(f"[code_executor] {exp_id}  sandbox blocked-report  script={current_name}  hash={hash_value}")
                                return {
                                    **state,
                                    "workspace_dir": str(workspace_dir),
                                    "code_script_path": str(script_path),
                                    "code_script_filename": current_name,
                                    "code_script_contents": current_contents,
                                    "generated_setups_csv": "",
                                    "script_hash": hash_value,
                                    "lookahead_flags": [],
                                    "sandbox_result": sandbox_result,
                                    "validation_result": {
                                        "passed": True,
                                        "errors": [],
                                        "compile_ok": True,
                                        "static_checks_ok": True,
                                        "lookahead_ok": True,
                                        "sandbox_ok": True,
                                        "repair_attempts": repair_attempt,
                                    },
                                }
                        if stdout_json.get("status") == "blocked":
                            pass
                        else:
                            setups_csv = Path(sandbox_result["setups_csv"])
                            if not setups_csv.exists():
                                last_failure = {
                                    "errors": [f"workspace script did not create setups CSV: {setups_csv}"],
                                    "workspace_dir": workspace_dir,
                                    "script_path": script_path,
                                    "hash_value": hash_value,
                                    "static_checks_ok": True,
                                    "compile_ok": True,
                                    "lookahead_ok": True,
                                    "sandbox_ok": False,
                                    "sandbox_result": sandbox_result,
                                    "lookahead_flags": [],
                                }
                            else:
                                _log.info(
                                    "code_executor_stage",
                                    exp_id=exp_id,
                                    stage="csv_validation_started",
                                    setups_csv=str(setups_csv),
                                )
                                print(f"[code_executor] {exp_id} stage=csv_validation_started")
                                csv_validation_errors = _validate_generated_signals_csv(
                                    setups_csv,
                                    available_timeframes=full_config.get("available_timeframes", ["1h"]),
                                )
                                if csv_validation_errors:
                                    _log.info(
                                        "code_executor_stage",
                                        exp_id=exp_id,
                                        stage="csv_validation_failed",
                                        errors=csv_validation_errors,
                                    )
                                    print(
                                        f"[code_executor] {exp_id} stage=csv_validation_failed "
                                        f"errors={csv_validation_errors[:3]}"
                                    )
                                    last_failure = {
                                        "errors": csv_validation_errors,
                                        "workspace_dir": workspace_dir,
                                        "script_path": script_path,
                                        "hash_value": hash_value,
                                        "static_checks_ok": True,
                                        "compile_ok": True,
                                        "lookahead_ok": True,
                                        "sandbox_ok": False,
                                        "sandbox_result": sandbox_result,
                                        "lookahead_flags": [],
                                    }
                                else:
                                    _log.info("code_executor_stage", exp_id=exp_id, stage="csv_validation_passed")
                                    print(f"[code_executor] {exp_id} stage=csv_validation_passed")
                                    csv_hash = script_hash(setups_csv.read_text(encoding="utf-8"))
                                    _log.info(
                                        "code_executor_ok",
                                        exp_id=exp_id,
                                        workspace_dir=str(workspace_dir),
                                        script_path=str(script_path),
                                        script_hash=hash_value,
                                        setups_csv_hash=csv_hash,
                                        repair_attempts=repair_attempt,
                                    )
                                    print(
                                        f"[code_executor] {exp_id}  sandbox ok  script={current_name}  hash={hash_value}"
                                        f"  csv_hash={csv_hash}  repairs={repair_attempt}"
                                    )
                                    return {
                                        **state,
                                        "workspace_dir": str(workspace_dir),
                                        "code_script_path": str(script_path),
                                        "code_script_filename": current_name,
                                        "code_script_contents": current_contents,
                                        "generated_setups_csv": str(setups_csv),
                                        "script_hash": hash_value,
                                        "setups_csv_hash": csv_hash,
                                        "lookahead_flags": [],
                                        "sandbox_result": sandbox_result,
                                        "validation_result": {
                                            "passed": True,
                                            "errors": [],
                                            "compile_ok": True,
                                            "static_checks_ok": True,
                                            "lookahead_ok": True,
                                            "sandbox_ok": True,
                                            "repair_attempts": repair_attempt,
                                        },
                                    }

                        if last_failure:
                            pass
                        else:
                            _log.info(
                                "code_executor_blocked_report",
                                exp_id=exp_id,
                                workspace_dir=str(workspace_dir),
                                script_path=str(script_path),
                                script_hash=hash_value,
                            )
                            print(f"[code_executor] {exp_id}  sandbox blocked-report  script={current_name}  hash={hash_value}")

        if repair_attempt >= MAX_CODE_REPAIR_ATTEMPTS or not last_failure:
            return _fail(
                state,
                last_failure["errors"] if last_failure else ["workspace script validation failed"],
                workspace_dir=last_failure.get("workspace_dir") if last_failure else workspace_dir,
                script_path=last_failure.get("script_path") if last_failure else None,
                hash_value=last_failure.get("hash_value", "") if last_failure else "",
                sandbox_result=last_failure.get("sandbox_result", {}) if last_failure else {},
                compile_ok=last_failure.get("compile_ok", True) if last_failure else True,
                static_checks_ok=last_failure.get("static_checks_ok", True) if last_failure else True,
                lookahead_ok=last_failure.get("lookahead_ok", True) if last_failure else True,
                sandbox_ok=last_failure.get("sandbox_ok", False) if last_failure else False,
                lookahead_flags=last_failure.get("lookahead_flags", []) if last_failure else [],
                repair_attempts=repair_attempt,
            )

        repair_attempt += 1
        repair_instructions = _build_repair_instructions(
            last_failure["errors"],
            last_failure.get("sandbox_result", {}),
        )
        seed_script = {
            "experiment_id": exp_id,
            "script_name": current_name,
            "script_path": str(last_failure["script_path"]),
            "script_text": current_contents,
            "validation_errors": list(last_failure["errors"]),
            "reviewer_notes": f"Automatic repair attempt {repair_attempt} after validation/runtime failure.",
        }
        history = store.summarize_for_llm(5)
        current_name, current_contents, _ = generate_script_with_coder(
            state,
            proposal,
            history=history,
            seed_script=seed_script,
            exp_id=exp_id,
            iteration=state.get("iteration", 0),
            repair_instructions=repair_instructions,
        )
        _log.info(
            "code_executor_repair_attempt",
            exp_id=exp_id,
            repair_attempt=repair_attempt,
            errors=last_failure["errors"],
            filename=current_name,
        )
        print(f"[code_executor] repair attempt {repair_attempt}/{MAX_CODE_REPAIR_ATTEMPTS}: {last_failure['errors'][0][:100]}")


def _fail(
    state: ResearchState,
    errors: list[str],
    workspace_dir: Path | None = None,
    script_path: Path | None = None,
    hash_value: str = "",
    sandbox_result: dict | None = None,
    compile_ok: bool = True,
    static_checks_ok: bool = True,
    lookahead_ok: bool = True,
    sandbox_ok: bool = False,
    lookahead_flags: list[str] | None = None,
    repair_attempts: int = 0,
) -> ResearchState:
    _log.warning("code_executor_failed", errors=errors)
    print(f"[code_executor] FAILED: {errors[0][:120]}")
    return {
        **state,
        "workspace_dir": str(workspace_dir) if workspace_dir else "",
        "code_script_path": str(script_path) if script_path else "",
        "generated_setups_csv": "",
        "script_hash": hash_value,
        "setups_csv_hash": "",
        "lookahead_flags": list(lookahead_flags or []),
        "sandbox_result": sandbox_result or {},
        "validation_result": {
            "passed": False,
            "errors": errors,
            "compile_ok": compile_ok,
            "static_checks_ok": static_checks_ok,
            "lookahead_ok": lookahead_ok,
            "sandbox_ok": sandbox_ok,
            "repair_attempts": repair_attempts,
        },
    }


def _stderr_excerpt(stderr_path: str) -> str:
    path = Path(stderr_path)
    if not path.exists():
        return "see stderr log"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return "see stderr log"
    return " ".join(text.split())[:400]


def _build_repair_instructions(errors: list[str], sandbox_result: dict) -> str:
    parts = ["The previous script failed validation/runtime. Repair it without rewriting unrelated logic."]
    if errors:
        parts.append("Observed errors:")
        parts.extend(f"- {error}" for error in errors[:6])
    stderr_path = str(sandbox_result.get("stderr_path") or "")
    stderr_excerpt = _stderr_excerpt(stderr_path) if stderr_path else ""
    if stderr_excerpt and stderr_excerpt != "see stderr log":
        parts.append(f"Sandbox stderr excerpt: {stderr_excerpt}")
    error_text = " ".join(errors[:3] + [stderr_excerpt]).lower()
    if "shape" in error_text or "broadcast" in error_text or "reshape" in error_text:
        parts.append(
            "NumPy shape hint: ensure all array operations preserve consistent shapes. "
            "Common fixes: use K * e instead of K.flatten() * e to keep (N,1) shape; "
            "use np.dot(K, e) for scalar results; add explicit .reshape(-1,1) after operations that collapse dimensions."
        )
    if "importerror" in error_text or "modulenotfounderror" in error_text or "no module named" in error_text:
        parts.append(
            "Import error hint: do not import unavailable third-party packages. "
            "Replace with stdlib, numpy, pandas, or workspace_helpers equivalents."
        )
    stdout_json = dict(sandbox_result.get("stdout_json") or {})
    if stdout_json.get("status") == "blocked":
        parts.append(f"Blocked stdout JSON: {stdout_json}")
    if "no signals" in error_text or "no fallback symbol" in error_text:
        parts.append(
            "No-signal repair hint for funding strategies: verify timestamp alignment and action names before changing thresholds. "
            "load_price_data() returns a 'timestamp' column and a RangeIndex; align funding_z with price_df['timestamp'], "
            "pass timestamps=price_df['timestamp'], and keep action values ENTER_LONG/ENTER_SHORT/EXIT/STOP. "
            "Do not catch Exception and continue silently. Emit diagnostics: funding_z min/max, counts beyond entry threshold, "
            "price_rows, funding_rows, alignment_non_null_count."
        )
    if "grid/random/bruteforce" in error_text or "grid" in error_text or "parameter" in error_text:
        parts.append(
            "Search repair hint: do not implement grid search, random search, all symbol combinations, "
            "or parameter sweeps in a workspace script. Replace the script with exactly one concrete "
            "configuration using the canonical helper."
        )
    if "invalid execution/signal timestamp" in error_text or "invalid execution" in error_text:
        parts.append(
            "Execution timestamp repair hint: every CSV row must come from generate_single_asset_signals() "
            "or generate_pair_signals() with build_execution_bridge() fields. Do not append manual force-close "
            "EXIT rows. If restricting traded symbols, use run_funding_divergence_strategy(..., entry_symbols=[...], "
            "peer_symbols=[...]) instead of custom rows."
        )
    parts.append(
        "Do not introduce new third-party dependencies. If a module is missing, replace it with stdlib/numpy/pandas logic or reuse workspace_helpers functions."
    )
    parts.append(
        "The revised script must contain exactly one main() and one __main__ guard, and must emit one valid JSON object."
    )
    return "\n".join(parts)


def _validate_generated_signals_csv(
    setups_csv: Path,
    *,
    available_timeframes: list[str],
) -> list[str]:
    try:
        df = pd.read_csv(setups_csv)
    except Exception as exc:
        return [f"unable to read generated signals CSV: {exc}"]

    if df.empty:
        return []

    errors: list[str] = []
    strategy_type = str(df.get("strategy_type", pd.Series(["single_asset"])).dropna().iloc[0])
    finest_secs = min((_timeframe_to_seconds(tf) for tf in available_timeframes), default=3600)
    requires_5m_or_finer = finest_secs <= 300

    if strategy_type == "single_asset":
        required = [
            "signal_timestamp",
            "signal_close_timestamp",
            "execution_timestamp",
            "execution_price",
            "execution_timeframe",
            "reduced_fidelity",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return [f"signals CSV missing required execution-model columns: {missing}"]

        exec_ts = pd.to_datetime(df["execution_timestamp"], utc=True, errors="coerce")
        signal_ts = pd.to_datetime(df["signal_timestamp"], utc=True, errors="coerce")
        signal_close_ts = pd.to_datetime(df["signal_close_timestamp"], utc=True, errors="coerce")

        if exec_ts.isna().any() or signal_ts.isna().any() or signal_close_ts.isna().any():
            errors.append("signals CSV contains invalid execution/signal timestamps")

        if (signal_close_ts <= signal_ts).fillna(False).any():
            errors.append("signal_close_timestamp must be strictly after signal_timestamp")

        if "timestamp" in df.columns:
            legacy_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if not legacy_ts.equals(exec_ts):
                errors.append("timestamp column must match execution_timestamp exactly")

        if "price" in df.columns:
            legacy_px = pd.to_numeric(df["price"], errors="coerce")
            exec_px = pd.to_numeric(df["execution_price"], errors="coerce")
            if not legacy_px.equals(exec_px):
                errors.append("price column must match execution_price exactly")

        if (exec_ts <= signal_ts).fillna(False).any():
            errors.append("execution_timestamp must be strictly after signal_timestamp")

        if (exec_ts < signal_close_ts).fillna(False).any():
            errors.append("execution_timestamp must be at or after signal_close_timestamp")

        exec_tfs = {str(v).strip() for v in df["execution_timeframe"].dropna().unique() if str(v).strip()}
        if not exec_tfs:
            errors.append("execution_timeframe must be recorded for single-asset signals")

        if requires_5m_or_finer:
            if exec_tfs:
                bad_tfs = [tf for tf in exec_tfs if _timeframe_to_seconds(tf) > 300]
                if bad_tfs:
                    errors.append(
                        f"execution_timeframe must be 5m-or-finer when such data exists; found {sorted(bad_tfs)}"
                    )
            reduced_vals = {
                str(v).strip().lower()
                for v in df["reduced_fidelity"].dropna().unique()
            }
            if reduced_vals - {"false", "0"}:
                errors.append("reduced_fidelity must be false when 5m-or-finer data exists")
    elif strategy_type == "pairs":
        required = [
            "signal_timestamp",
            "signal_close_timestamp",
            "execution_timestamp",
            "execution_price_a",
            "execution_price_b",
            "execution_timeframe",
            "reduced_fidelity",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return [f"pairs signals CSV missing required execution-model columns: {missing}"]

        exec_ts = pd.to_datetime(df["execution_timestamp"], utc=True, errors="coerce")
        signal_ts = pd.to_datetime(df["signal_timestamp"], utc=True, errors="coerce")
        signal_close_ts = pd.to_datetime(df["signal_close_timestamp"], utc=True, errors="coerce")
        if exec_ts.isna().any() or signal_ts.isna().any() or signal_close_ts.isna().any():
            errors.append("pairs signals CSV contains invalid execution/signal timestamps")

        if (signal_close_ts <= signal_ts).fillna(False).any():
            errors.append("pairs signal_close_timestamp must be strictly after signal_timestamp")

        if "timestamp" in df.columns:
            legacy_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if not legacy_ts.equals(exec_ts):
                errors.append("pairs timestamp column must match execution_timestamp exactly")

        for legacy_col, exec_col in (("price_a", "execution_price_a"), ("price_b", "execution_price_b")):
            if legacy_col in df.columns:
                legacy_px = pd.to_numeric(df[legacy_col], errors="coerce")
                exec_px = pd.to_numeric(df[exec_col], errors="coerce")
                if not legacy_px.equals(exec_px):
                    errors.append(f"{legacy_col} column must match {exec_col} exactly")

        if (exec_ts <= signal_ts).fillna(False).any():
            errors.append("pairs execution_timestamp must be strictly after signal_timestamp")

        if (exec_ts < signal_close_ts).fillna(False).any():
            errors.append("pairs execution_timestamp must be at or after signal_close_timestamp")

        exec_tfs = {str(v).strip() for v in df["execution_timeframe"].dropna().unique() if str(v).strip()}
        if not exec_tfs:
            errors.append("execution_timeframe must be recorded for pairs signals")

        if requires_5m_or_finer:
            if exec_tfs:
                bad_tfs = [tf for tf in exec_tfs if _timeframe_to_seconds(tf) > 300]
                if bad_tfs:
                    errors.append(
                        f"pairs execution_timeframe must be 5m-or-finer when such data exists; found {sorted(bad_tfs)}"
                    )
            reduced_vals = {
                str(v).strip().lower()
                for v in df["reduced_fidelity"].dropna().unique()
            }
            if reduced_vals - {"false", "0"}:
                errors.append("pairs reduced_fidelity must be false when 5m-or-finer data exists")

    return errors


def _timeframe_to_seconds(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    mapping = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
        "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
    }
    return mapping.get(tf, 3600)
