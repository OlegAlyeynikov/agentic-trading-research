import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from agent_research.state import ResearchState
from agent_research.memory import ExperimentStore
from agent_research.config_utils import load_full_config, compute_research_score
from agent_research.logging_config import get_logger, AgentTimer

_log = get_logger("executor")

_MOCK_RESULT = {
    "status": "success",
    "stats": {
        "total_trades": 42,
        "win_rate": 0.62,
        "total_pnl_pct": 8.4,
        "avg_pnl_pct": 0.200,
        "profit_factor": 1.82,
        "max_drawdown_pct": -5.2,
        "by_exit_reason": {
            "EXIT": {"count": 30, "avg_pnl_pct": 0.45, "win_rate": 0.78},
            "STOP": {"count": 12, "avg_pnl_pct": -0.62, "win_rate": 0.17},
        },
    },
    "data_quality": {
        "orphan_exits": 0,
        "duplicate_enters": 0,
        "open_trades_unclosed": 0,
        "is_clean": True,
    },
    "pairs": [
        {
            "pair_id": "BTCUSDT/ETHUSDT",
            "trades": 42,
            "win_rate": 0.62,
            "total_pnl_pct": 8.4,
            "avg_pnl_pct": 0.200,
            "sharpe": 1.34,
            "max_drawdown_pct": -5.2,
            "stop_rate": 0.286,
        }
    ],
}

_EMPTY_RESULT = {
    "status": "success",
    "stats": {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl_pct": 0.0,
        "avg_pnl_pct": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "by_exit_reason": {},
    },
    "data_quality": {"orphan_exits": 0, "duplicate_enters": 0, "open_trades_unclosed": 0, "is_clean": True},
    "pairs": [],
}

_DUPLICATE_BLOCKED_RESULT = {
    "status": "blocked",
    "stats": {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl_pct": 0.0,
        "avg_pnl_pct": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "by_exit_reason": {},
    },
    "data_quality": {
        "orphan_exits": 0,
        "duplicate_enters": 0,
        "open_trades_unclosed": 0,
        "is_clean": False,
    },
    "pairs": [],
}


def _resolve_funding_dir(state: ResearchState) -> str:
    configured = str(state.get("funding_rates_dir") or "").strip()
    if configured:
        return configured
    data_dir = str(state.get("data_dir") or "").strip()
    if not data_dir:
        return ""
    return str(Path(data_dir) / "new_data" / "funding_rates")


def run_executor(state: ResearchState) -> ResearchState:
    experiment_id = state["current_experiment_id"]
    store = ExperimentStore()

    with AgentTimer(_log, "executor", experiment_id, state.get("iteration", 0)):
        if state.get("dry_run"):
            result = dict(_MOCK_RESULT)
            _persist(store, state, result, experiment_id)
            _log.info("executor_dry_run", experiment_id=experiment_id, trades=42)
            print(f"[executor] {experiment_id}  trades=42  total=+8.4%  [dry-run]")
            return {
                **state,
                "backtest_result": result,
                "experiment_ids": [*state.get("experiment_ids", []), experiment_id],
            }

        result = _run_live(state)
        output_dup = result.pop("_duplicate_of_experiment_id", "")
        if output_dup:
            state = {**state, "duplicate_of_experiment_id": output_dup}
        _persist(store, state, result, experiment_id)
        total = result["stats"]["total_trades"]
        pnl = result["stats"].get("total_pnl_pct") or 0.0
        win = result["stats"].get("win_rate") or 0.0
        _log.info(
            "executor_done",
            experiment_id=experiment_id,
            total_trades=total,
            total_pnl_pct=pnl,
            win_rate=win,
            is_clean=result.get("data_quality", {}).get("is_clean"),
        )
        print(f"[executor] {experiment_id}  trades={total}  total={pnl:+.1f}%")

    return {
        **state,
        "backtest_result": result,
        "experiment_ids": [*state.get("experiment_ids", []), experiment_id],
    }


def _run_live(state: ResearchState) -> dict:
    current_scope = dict(state.get("current_scope") or {})
    funding_required = _requires_funding_model(state)
    strategy_flags = dict(state.get("strategy_flags") or {})

    if state.get("duplicate_of_experiment_id"):
        result = dict(_DUPLICATE_BLOCKED_RESULT)
        result["research_scope"] = current_scope
        result["diagnostics_summary"] = _empty_diagnostics()
        result["execution_blocked_reason"] = (
            f"duplicate_of={state.get('duplicate_of_experiment_id')}; "
            "researcher proposed an already-tested config/scope"
        )
        _log.warning(
            "executor_blocked_duplicate",
            experiment_id=state.get("current_experiment_id"),
            duplicate_of=state.get("duplicate_of_experiment_id"),
        )
        return result

    csv_hash = state.get("setups_csv_hash") or ""
    if csv_hash:
        store = ExperimentStore()
        prior = store.find_by_setups_hash(csv_hash)
        if prior and prior.get("experiment_id") != state.get("current_experiment_id"):
            result = dict(_DUPLICATE_BLOCKED_RESULT)
            result["research_scope"] = current_scope
            result["diagnostics_summary"] = _empty_diagnostics()
            result["execution_blocked_reason"] = (
                f"duplicate_of={prior['experiment_id']}; "
                "generated signals CSV is identical to a previously run experiment"
            )
            _log.warning(
                "executor_blocked_duplicate_output",
                experiment_id=state.get("current_experiment_id"),
                duplicate_of=prior["experiment_id"],
                csv_hash=csv_hash,
            )
            print(
                f"[executor] {state.get('current_experiment_id')}  "
                f"DUPLICATE OUTPUT blocked  csv_hash={csv_hash}  "
                f"matches={prior['experiment_id']}"
            )
            return {**result, "_duplicate_of_experiment_id": prior["experiment_id"]}

    setups_csv = state.get("generated_setups_csv") or ""
    validation_result = dict(state.get("validation_result") or {})
    sandbox_result = dict(state.get("sandbox_result") or {})
    sandbox_json = dict(sandbox_result.get("stdout_json") or {})

    if not validation_result.get("passed"):
        result = dict(_EMPTY_RESULT)
        result["status"] = "blocked"
        result["research_scope"] = current_scope
        result["execution_blocked_reason"] = "workspace script validation failed"
        result["diagnostics_summary"] = _diagnostics_from_sandbox(sandbox_json)
        return result

    if sandbox_json.get("status") == "blocked":
        result = dict(_EMPTY_RESULT)
        result["status"] = "blocked"
        result["research_scope"] = current_scope
        result["execution_blocked_reason"] = sandbox_json.get("notes", "workspace script reported blocked status")
        result["data_gap_report"] = sandbox_json
        result["diagnostics_summary"] = _diagnostics_from_sandbox(sandbox_json)
        return result

    if not setups_csv or not os.path.exists(setups_csv):
        result = dict(_EMPTY_RESULT)
        result["status"] = "blocked"
        result["research_scope"] = current_scope
        result["execution_blocked_reason"] = "workspace script did not produce signals CSV"
        result["diagnostics_summary"] = _empty_diagnostics()
        return result

    from backtest.engine import run_backtest
    funding_dir = _resolve_funding_dir(state) or None
    _log.info(
        "executor_funding_context",
        experiment_id=state.get("current_experiment_id"),
        funding_required=funding_required,
        configured_funding_rates_dir=str(state.get("funding_rates_dir") or ""),
        configured_data_dir=str(state.get("data_dir") or ""),
        resolved_funding_dir=funding_dir or "",
        strategy_flags=strategy_flags,
    )
    if funding_required and not funding_dir:
        result = dict(_EMPTY_RESULT)
        result["status"] = "blocked"
        result["research_scope"] = current_scope
        result["execution_blocked_reason"] = "funding strategy requires funding_rates_dir"
        result["diagnostics_summary"] = _empty_diagnostics()
        return result

    risk_pct = float(os.environ.get("RISK_PCT_PER_TRADE", "0.02") or 0.02)
    result = run_backtest(
        signals_csv=setups_csv,
        fee_rate=0.0005,
        funding_dir=funding_dir,
        risk_pct_per_trade=risk_pct,
    )
    if funding_required and "funding_paid_pct" not in (result.get("stats") or {}):
        result = dict(_EMPTY_RESULT)
        result["status"] = "blocked"
        result["research_scope"] = current_scope
        result["execution_blocked_reason"] = "funding strategy result missing funding_paid_pct"
        result["diagnostics_summary"] = _empty_diagnostics()
        return result
    if (
        funding_required
        and strategy_flags.get("require_positive_funding_carry")
        and (result.get("stats") or {}).get("funding_paid_pct", 0.0) > 0
    ):
        result = dict(_EMPTY_RESULT)
        result["status"] = "blocked"
        result["research_scope"] = current_scope
        result["execution_blocked_reason"] = (
            "strategy requires positive funding carry, but funding_paid_pct indicates net funding cost"
        )
        result["diagnostics_summary"] = _empty_diagnostics()
        return result
    result["research_scope"] = current_scope
    result["diagnostics_summary"] = {
        "signal_count": result.get("input", {}).get("total_signals", 0),
        "symbols_used": _symbols_from_result(result),
    }
    return result


def _empty_diagnostics() -> dict:
    return {"signal_count": 0, "symbols_used": []}


def _diagnostics_from_sandbox(sandbox_json: dict) -> dict:
    metrics = sandbox_json.get("metrics") or {}
    diagnostics = metrics.get("diagnostics") or {}
    if not diagnostics:
        return _empty_diagnostics()
    symbols = diagnostics.get("symbols") or {}
    raw_triggers = sum(int((d or {}).get("count_beyond_entry") or 0) for d in symbols.values())
    return {
        "signal_count": int(metrics.get("total_enter_signals") or 0),
        "symbols_used": [],
        "blocked_reason": diagnostics.get("blocked_reason", ""),
        "raw_trigger_count": raw_triggers,
        "symbols_with_raw_triggers": sum(
            1 for d in symbols.values() if int((d or {}).get("count_beyond_entry") or 0) > 0
        ),
        "execution_timeframe": diagnostics.get("execution_timeframe", ""),
        "reduced_fidelity": bool(diagnostics.get("reduced_fidelity", False)),
    }


def _symbols_from_result(result: dict) -> list[str]:
    pairs = result.get("pairs") or []
    return [p["pair_id"] for p in pairs if p.get("pair_id")]


def _requires_funding_model(state: ResearchState) -> bool:
    flags = dict(state.get("strategy_flags") or {})
    proposal = dict(state.get("code_change_proposal") or {})
    proposal_flags = dict(proposal.get("strategy_flags") or {})
    if (
        flags.get("requires_funding_model")
        or proposal_flags.get("requires_funding_model")
    ):
        return True

    # Fallback for partially propagated state: funding strategies should still
    # be recognized from their direction/script metadata even if flags are missing.
    haystacks = [
        str(state.get("goal") or ""),
        str(state.get("code_direction") or ""),
        str(state.get("code_script_filename") or ""),
        str(state.get("code_script_contents") or "")[:4000],
        str((state.get("sandbox_result") or {}).get("stdout_json", {}).get("notes") or ""),
    ]
    text = "\n".join(haystacks).lower()
    funding_markers = (
        "funding divergence",
        "compute_peer_funding_zscore",
        "load_funding_rates",
        "funding_z",
        "funding rate",
    )
    return any(marker in text for marker in funding_markers)


def _persist(store: ExperimentStore, state: ResearchState, result: dict, exp_id: str) -> None:
    stats = result.get("stats", {})
    total = stats.get("total_trades") or 0
    pairs = result.get("pairs", [])
    goal_contract = dict(state.get("goal_contract") or {})

    if pairs and total > 0:
        stop_rate = sum(
            p.get("stop_rate", 0) * p.get("trades", 0)
            for p in pairs
        ) / total
    else:
        stop_count = stats.get("by_exit_reason", {}).get("STOP", {}).get("count", 0)
        stop_rate = stop_count / total if total > 0 else 0.0

    sharpe_vals = [p.get("sharpe", 0) for p in pairs if p.get("sharpe") is not None]
    pair_drawdowns = [p.get("max_drawdown_pct", 0) for p in pairs if p.get("max_drawdown_pct") is not None]
    pairs_count = len([p for p in pairs if (p.get("trades") or 0) > 0])

    metrics = {
        "total_trades": total,
        "win_rate": stats.get("win_rate") or 0.0,
        "total_pnl_pct": stats.get("total_pnl_pct") or 0.0,
        "sum_trade_pnl_pct": stats.get("sum_trade_pnl_pct") or stats.get("total_pnl_pct") or 0.0,
        "estimated_equity_return_pct": stats.get("estimated_equity_return_pct") or 0.0,
        "avg_pnl_pct": stats.get("avg_pnl_pct") or 0.0,
        "sharpe_proxy": max(sharpe_vals) if sharpe_vals else 0.0,
        "profit_factor": stats.get("profit_factor") or 0.0,
        "max_drawdown_pct": (
            min(pair_drawdowns)
            if pair_drawdowns else
            (stats.get("max_drawdown_pct") or 0.0)
        ),
        "stop_rate": round(stop_rate, 4),
        "pairs_count": pairs_count,
        "is_clean": result.get("data_quality", {}).get("is_clean", True),
        "signal_count": result.get("diagnostics_summary", {}).get("signal_count", 0),
        "symbols_used_count": len(result.get("diagnostics_summary", {}).get("symbols_used", [])),
        "avg_duration_hours": stats.get("avg_duration_hours") or 0.0,
        "median_duration_hours": stats.get("median_duration_hours") or 0.0,
        "max_duration_hours": stats.get("max_duration_hours") or 0.0,
        "p95_duration_hours": stats.get("p95_duration_hours") or 0.0,
        "time_exit_rate": stats.get("time_exit_rate") or 0.0,
    }
    metrics["research_score"] = compute_research_score(metrics, goal_contract)

    store.save(exp_id, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_hypothesis_id": state.get("active_hypothesis_id", ""),
        "code_direction": state.get("code_direction", ""),
        "research_scope": state.get("current_scope", {}),
        "result": result,
        "reviewer_verdict": "",
        "reviewer_notes": "",
        "reviewer_diagnosis": {},
        "diagnostics_summary": result.get("diagnostics_summary", {}),
        "experiment_type": "code_research",
        "workspace_dir": state.get("workspace_dir", ""),
        "code_script_path": state.get("code_script_path", ""),
        "generated_setups_csv": state.get("generated_setups_csv", ""),
        "script_hash": state.get("script_hash", ""),
        "setups_csv_hash": state.get("setups_csv_hash", ""),
        "lookahead_flags": list(state.get("lookahead_flags") or []),
        "validation_result": dict(state.get("validation_result") or {}),
        "sandbox_result": dict(state.get("sandbox_result") or {}),
        "duplicate_of_experiment_id": state.get("duplicate_of_experiment_id", ""),
        "goal_contract": dict(state.get("goal_contract") or {}),
        "metrics": metrics,
    })
