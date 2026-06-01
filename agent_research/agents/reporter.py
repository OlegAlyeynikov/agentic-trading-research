"""
Reporter agent — writes a human-readable .log file after every experiment.

Output format: plain text, structured, no JSON.
Files written to: reports/ directory.
One file per experiment + rolling session summary.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from agent_research.state import ResearchState
from agent_research.memory import ExperimentStore
from agent_research.logging_config import get_logger, AgentTimer

_log = get_logger("reporter")


def _fmt_pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}%}"


def _fmt_signed_pct(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.{digits}f}%"


def _fmt_float(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def run_reporter(state: ResearchState) -> ResearchState:
    exp_id = state.get("current_experiment_id", "?")
    reports_dir = Path(state.get("reports_dir") or "agent_research/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with AgentTimer(_log, "reporter", exp_id, state.get("iteration", 0)):
        report_text = _build_report(state)
        report_path = reports_dir / f"report_{exp_id}.log"
        report_path.write_text(report_text, encoding="utf-8")

        _update_session_summary(state, reports_dir)
        final_artifact_dir = _write_final_artifact_bundle(state, report_path)
        promising_dir = _write_promising_bundle(state, report_path)

        _store_top_pairs(state, exp_id)

        _log.info("reporter_wrote", path=str(report_path), exp_id=exp_id, final_artifact_dir=final_artifact_dir or "")
        print(f"[reporter] wrote {report_path}")
        if final_artifact_dir:
            print(f"[reporter] final bundle: {final_artifact_dir}")
        if promising_dir:
            print(f"[reporter] promising run saved: {promising_dir}")

    return {**state, "report_path": str(report_path), "final_artifact_dir": final_artifact_dir}


def _build_report(state: ResearchState) -> str:
    exp_id = state.get("current_experiment_id", "?")
    result = state.get("backtest_result") or {}
    record = ExperimentStore().load(exp_id) or {}
    persisted_metrics = dict(record.get("metrics") or {})
    stats = result.get("stats", {})
    diag = result.get("diagnostics_summary", {})
    pairs = result.get("pairs", [])
    scope = state.get("current_scope", {})
    code_direction = state.get("code_direction", "")
    verdict = state.get("reviewer_verdict", "?")
    notes = state.get("reviewer_notes", "")
    hypothesis = state.get("active_hypothesis_id", "?")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 0)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    router_msg = state.get("router_message", "")
    goal_contract = dict(state.get("goal_contract") or {})
    workspace_dir = state.get("workspace_dir", "")
    code_script_path = state.get("code_script_path", "")
    generated_setups_csv = state.get("generated_setups_csv", "")
    script_hash = state.get("script_hash", "")
    lookahead_flags = list(state.get("lookahead_flags") or [])
    sandbox_result = dict(state.get("sandbox_result") or {})
    validation_result = dict(state.get("validation_result") or {})
    data_gap_report = dict(result.get("data_gap_report") or {})
    blocked_reason = result.get("execution_blocked_reason", "")

    scope_lines = [f"  {k:<28} {v}" for k, v in sorted(scope.items()) if v is not None]

    pair_lines = []
    for p in pairs[:10]:
        pair_lines.append(
            f"  {p.get('pair_id', '?'):<25} "
            f"trades={p.get('trades', '?'):<5} "
            f"win={_fmt_pct(p.get('win_rate', 0), 0)}  "
            f"pnl={_fmt_signed_pct(p.get('total_pnl_pct', 0), 2)}  "
            f"sharpe={_fmt_float(p.get('sharpe', 0), 2)}  "
            f"stop={_fmt_pct(p.get('stop_rate', 0), 0)}"
        )

    total_trades = stats.get("total_trades", 0) or 0
    if pairs and total_trades > 0:
        aggregate_stop_rate = sum(
            (p.get("stop_rate", 0) or 0.0) * (p.get("trades", 0) or 0)
            for p in pairs
        ) / total_trades
    else:
        stop_count = (stats.get("by_exit_reason", {}).get("STOP", {}) or {}).get("count", 0) or 0
        aggregate_stop_rate = (stop_count / total_trades) if total_trades > 0 else 0.0

    sep = "=" * 80
    thin = "-" * 80

    lines = [
        sep,
        "EXPERIMENT REPORT",
        sep,
        f"Experiment ID  : {exp_id}",
        f"Timestamp      : {ts}",
        f"Hypothesis     : {hypothesis}",
        f"Iteration      : {iteration} / {max_iter}",
        f"Goal           : {state.get('goal', '')}",
        "",
        thin,
        "GOAL CONTRACT",
        thin,
        *[f"  {k:<28} {v}" for k, v in sorted(goal_contract.items())],
        "",
        thin,
        "CODE DIRECTION",
        thin,
        f"  {code_direction or 'n/a'}",
        "",
        "RESEARCH SCOPE",
        *scope_lines,
        "",
        thin,
        "CODE SCRIPT",
        thin,
        f"  Workspace dir            : {workspace_dir or 'n/a'}",
        f"  Script path              : {code_script_path or 'n/a'}",
        f"  Script hash              : {script_hash or 'n/a'}",
        f"  Generated setups CSV     : {generated_setups_csv or 'n/a'}",
    ]

    if sandbox_result:
        lines.append(f"  Sandbox returncode       : {sandbox_result.get('returncode', '?')}")
        lines.append(f"  Sandbox runtime_seconds  : {sandbox_result.get('runtime_seconds', '?')}")
        lines.append(f"  Sandbox stderr           : {sandbox_result.get('stderr_path', 'n/a')}")
    if lookahead_flags:
        lines.append(f"  Lookahead flags          : {'; '.join(lookahead_flags[:3])}")
    val_errors = validation_result.get("errors", [])
    if val_errors:
        lines.append(f"  Validation errors        : {'; '.join(val_errors[:3])}")
    lines.append("")

    if diag:
        lines += [
            thin,
            "SCREENING RESULTS",
            thin,
            f"  Pairs tested             : {diag.get('pairs_tested', '?')}",
            f"  Pairs passed coint test  : {diag.get('pairs_passed', '?')}",
            f"  Pass rate                : {diag.get('pass_rate', 0):.1%}",
            f"  Setups generated         : {diag.get('setups_rows', '?')}",
            "",
        ]
        if diag.get("top_pairs_by_pvalue"):
            lines.append("  Top pairs by cointegration p-value:")
            for tp in diag["top_pairs_by_pvalue"][:5]:
                pval = tp.get("coint_pvalue")
                hl = tp.get("halflife")
                pval_str = f"{pval:.5f}" if pval is not None else "?"
                hl_str = f"{hl:.1f}h" if hl is not None else "?"
                lines.append(f"    {tp.get('pair_id', '?'):<25} p={pval_str}  hl={hl_str}")
            lines.append("")

    lines += [
        thin,
        "BACKTEST RESULTS",
        thin,
        f"  Total trades             : {stats.get('total_trades', '?')}",
        f"  Pairs with trades        : {len([p for p in pairs if (p.get('trades', 0) or 0) > 0])}",
        f"  Win rate                 : {_fmt_pct(stats.get('win_rate', 0), 1)}",
        f"  Stop rate                : {_fmt_pct(aggregate_stop_rate, 1)}",
        f"  Sum trade P&L            : {_fmt_signed_pct(stats.get('sum_trade_pnl_pct', stats.get('total_pnl_pct', 0)), 2)}",
        f"  Est. equity return       : {_fmt_signed_pct(stats.get('estimated_equity_return_pct', 0), 2)}",
        f"  Avg P&L per trade        : {_fmt_signed_pct(stats.get('avg_pnl_pct', 0), 4)}",
        f"  Profit factor            : {_fmt_float(stats.get('profit_factor', 0), 2)}",
        f"  Max drawdown             : {_fmt_signed_pct(stats.get('max_drawdown_pct', 0), 2)}",
        f"  Max duration             : {_fmt_float(stats.get('max_duration_hours', 0), 2)}h",
        f"  P95 duration             : {_fmt_float(stats.get('p95_duration_hours', 0), 2)}h",
        f"  Time-exit rate           : {_fmt_pct(stats.get('time_exit_rate', 0), 1)}",
        f"  Research score           : {_fmt_float(persisted_metrics.get('research_score', 0), 2)}",
        "",
    ]

    if pairs:
        lines += [
            "  Per-pair breakdown:",
            f"  {'Pair':<25} {'Trades':<8} {'Win%':<7} {'Total%':<10} {'Sharpe':<9} {'Stop%'}",
            "  " + "-" * 65,
            *pair_lines,
            "",
        ]

    by_exit = stats.get("by_exit_reason", {})
    if by_exit:
        lines.append("  Exit reason breakdown:")
        for reason, s in by_exit.items():
            lines.append(
                f"    {reason:<10} count={s.get('count', '?'):<5} "
                f"avg_pnl={_fmt_signed_pct(s.get('avg_pnl_pct'), 3)}  "
                f"win={_fmt_pct(s.get('win_rate'), 0)}"
            )
        lines.append("")

    by_exit_detail = stats.get("by_exit_reason_detail", {})
    if by_exit_detail:
        lines.append("  Exit detail breakdown:")
        for reason, s in by_exit_detail.items():
            lines.append(
                f"    {reason:<15} count={s.get('count', '?'):<5} "
                f"avg_pnl={_fmt_signed_pct(s.get('avg_pnl_pct'), 3)}  "
                f"win={_fmt_pct(s.get('win_rate'), 0)}"
            )
        lines.append("")

    if data_gap_report:
        lines += [
            thin,
            "DATA GAP REPORT",
            thin,
            f"  Status                   : {data_gap_report.get('status', 'blocked')}",
            f"  Notes                    : {data_gap_report.get('notes', blocked_reason or 'n/a')}",
        ]
        missing = list(data_gap_report.get("missing_requirements") or [])
        if missing:
            lines.append("  Missing requirements:")
            for item in missing:
                lines.append(f"    - {item}")
        lines.append("")
    elif blocked_reason:
        lines += [
            thin,
            "BLOCKED EXECUTION",
            thin,
            f"  Reason                   : {blocked_reason}",
            "",
        ]

    lines += [
        thin,
        f"REVIEWER VERDICT: {verdict.upper()}",
        thin,
        notes if notes else "(no notes)",
        "",
        thin,
        "ROUTER DECISION",
        thin,
        "  (pending — written by router after this report)",
        "",
        sep,
        "",
    ]

    return "\n".join(lines)


def _update_session_summary(state: ResearchState, reports_dir: Path) -> None:
    store = ExperimentStore()
    records = store.list_recent(50)
    approved = [r for r in records if r.get("reviewer_verdict") == "approve"]
    rejected = [r for r in records if r.get("reviewer_verdict") == "reject"]
    goal_contract = dict(state.get("goal_contract") or {})
    primary_metric = str(goal_contract.get("primary_metric") or "research_score")
    best = store.find_best(primary_metric)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sep = "=" * 80
    thin = "-" * 80

    lines = [
        sep,
        "SESSION SUMMARY",
        sep,
        f"Updated        : {ts}",
        f"Goal           : {state.get('goal', '')}",
        f"Iteration      : {state.get('iteration', 0)} / {state.get('max_iterations', 0)}",
        f"Active H       : {state.get('active_hypothesis_id', '?')}",
        f"Completed H    : {', '.join(state.get('completed_hypothesis_ids') or []) or 'none'}",
        "",
        thin,
        "PROGRESS",
        thin,
        f"  Total experiments  : {len(records)}",
        f"  Approved           : {len(approved)}",
        f"  Rejected           : {len(rejected)}",
        "",
    ]

    if best:
        m = best.get("metrics", {})
        direction = (best.get("code_direction") or "")[:120]
        lines += [
            thin,
            "BEST APPROVED EXPERIMENT",
            thin,
            f"  Experiment ID  : {best.get('experiment_id', '?')}",
            f"  Hypothesis     : {best.get('active_hypothesis_id') or '?'}",
            f"  Direction      : {direction}",
            f"  Total trades   : {m.get('total_trades', '?')}",
            f"  Research score : {_fmt_float(m.get('research_score', 0), 2)}",
            f"  Sharpe proxy   : {_fmt_float(m.get('sharpe_proxy', 0), 2)}",
            f"  Profit factor  : {_fmt_float(m.get('profit_factor', 0), 2)}",
            f"  Max drawdown   : {_fmt_signed_pct(m.get('max_drawdown_pct', 0), 2)}",
            f"  Human metrics  : equity={_fmt_signed_pct(m.get('estimated_equity_return_pct', 0), 2)}  "
            f"sum_trade_pnl={_fmt_signed_pct(m.get('sum_trade_pnl_pct', m.get('total_pnl_pct', 0)), 2)}  "
            f"win={_fmt_pct(m.get('win_rate', 0), 1)}  stop={_fmt_pct(m.get('stop_rate', 0), 1)}",
            "",
        ]

    blocked_recent = [
        r for r in records
        if (r.get("result", {}) or {}).get("status") == "blocked"
    ][-10:]
    if blocked_recent:
        lines += [thin, "RECENT BLOCKED / DATA-GAP EXPERIMENTS", thin]
        for rec in blocked_recent:
            result = rec.get("result", {}) or {}
            gap = result.get("data_gap_report", {}) or {}
            reason = gap.get("notes") or result.get("execution_blocked_reason") or "blocked"
            compact_reason = " ".join(str(reason).split())[:120]
            lines.append(
                f"  {rec.get('experiment_id', '?'):<24} "
                f"H={rec.get('active_hypothesis_id') or '?':<4} "
                f"reason={compact_reason}"
            )
        lines.append("")

    promising_records = [r for r in records if r.get("promising")]
    if promising_records:
        lines += [thin, "PROMISING RUNS (rejected but worth revisiting)", thin]
        for rec in promising_records:
            m = rec.get("metrics") or {}
            score = float(m.get("research_score") or 0)
            reason = str(rec.get("promising_reason") or "")[:60]
            lines.append(
                f"  {rec.get('experiment_id', '?'):<24} "
                f"H={rec.get('active_hypothesis_id') or '?':<4} "
                f"score={score:.1f}  reason={reason}"
            )
        lines.append("")

    lines += [thin, "PER-HYPOTHESIS SUMMARY", thin]
    hyp_records: dict = {}
    for r in records:
        h = r.get("active_hypothesis_id") or "unknown"
        if h not in hyp_records:
            hyp_records[h] = []
        hyp_records[h].append(r)
    for h_id in sorted(hyp_records.keys()):
        h_recs = hyp_records[h_id]
        h_approved = [r for r in h_recs if r.get("reviewer_verdict") == "approve"]
        h_blocked = [
            r for r in h_recs
            if (r.get("result", {}) or {}).get("status") == "blocked"
        ]
        h_promising = [r for r in h_recs if r.get("promising")]
        best_score = max(
            (r.get("metrics", {}).get("research_score", 0) for r in h_recs if not r.get("result", {}).get("status") == "blocked"),
            default=0,
        )
        lines.append(
            f"  {h_id:<6} experiments={len(h_recs):<4} "
            f"approved={len(h_approved):<3} "
            f"blocked={len(h_blocked):<3} "
            f"promising={len(h_promising):<3} "
            f"best_score={best_score:.2f}"
        )
    lines += ["", sep, ""]

    summary_path = reports_dir / "session_summary.log"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_artifact_bundle(state: ResearchState, report_path: Path) -> str | None:
    if state.get("reviewer_verdict") != "approve":
        return None

    exp_id = state.get("current_experiment_id", "?")
    record = ExperimentStore().load(exp_id) or {}
    if not record:
        return None

    artifacts_root = Path(__file__).resolve().parent.parent / "approved"
    bundle_dir = artifacts_root / exp_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(state.get("config_path") or "")
    current_scope = dict(state.get("current_scope") or {})
    goal_contract = dict(state.get("goal_contract") or {})
    result = dict(state.get("backtest_result") or {})
    metrics = dict(record.get("metrics") or {})
    code_direction = state.get("code_direction", "")
    script_path = state.get("code_script_path") or ""

    base_full_config = {}
    if config_path.exists():
        base_full_config = json.loads(config_path.read_text(encoding="utf-8"))

    reproducibility = {
        "experiment_id": exp_id,
        "goal": state.get("goal", ""),
        "code_direction": code_direction,
        "hypothesis_id": state.get("active_hypothesis_id", ""),
        "reviewer_verdict": state.get("reviewer_verdict", ""),
        "router_decision": state.get("router_decision", ""),
        "config_path": str(config_path.resolve()) if config_path else "",
        "report_path": str(report_path.resolve()),
    }

    _write_json(bundle_dir / "reproducibility.json", reproducibility)
    _write_json(bundle_dir / "goal_contract.json", goal_contract)
    _write_json(bundle_dir / "research_scope.json", current_scope)
    _write_json(bundle_dir / "result.json", result)
    _write_json(bundle_dir / "metrics.json", metrics)
    _write_json(bundle_dir / "record.json", record)

    if base_full_config:
        _write_json(bundle_dir / "base_config_snapshot.json", base_full_config)

    shutil.copy2(report_path, bundle_dir / "report.log")

    if script_path and Path(script_path).exists():
        shutil.copy2(script_path, bundle_dir / Path(script_path).name)

    setups_csv = state.get("generated_setups_csv") or ""
    if setups_csv and Path(setups_csv).exists():
        shutil.copy2(setups_csv, bundle_dir / Path(setups_csv).name)

    reproduce_text = _build_reproduce_text(
        exp_id=exp_id,
        code_direction=code_direction,
        config_path=str(config_path.resolve()) if config_path else "",
        scope=current_scope,
        bundle_dir=bundle_dir,
        script_path=script_path,
        setups_csv=setups_csv,
    )
    (bundle_dir / "REPRODUCE.md").write_text(reproduce_text, encoding="utf-8")

    ExperimentStore().update(exp_id, {"final_artifact_dir": str(bundle_dir.resolve())})
    return str(bundle_dir.resolve())


_PROMISING_KEYWORDS = (
    "promising", "shows potential", "worth revisiting", "good signal",
    "high win rate", "positive pnl", "profitable exits",
)
_PROMISING_SCORE_THRESHOLD = 90.0

# Hard gates: experiment must pass ALL of these before even checking score/keywords.
# These prevent obvious failures (losing strategies, catastrophic drawdowns) from
# cluttering promising_runs even if their research_score happens to be elevated.
_PROMISING_HARD_GATES = {
    "min_sharpe_proxy":    0.25,   # must show some positive signal
    "min_profit_factor":   1.20,   # must be net profitable (gross profit > gross loss × 1.2)
    "max_drawdown_pct":   -30.0,   # no catastrophic drawdowns
    "min_total_trades":    20,     # enough data to be statistically meaningful
}


def _is_promising(state: ResearchState, record: dict) -> bool:
    if state.get("reviewer_verdict") == "approve":
        return False  # approved runs go to approved/, not promising_runs
    metrics = dict((record.get("metrics") or {}))
    # Apply hard gates first — fail any gate → not promising regardless of score/keywords
    if float(metrics.get("sharpe_proxy") or 0) < _PROMISING_HARD_GATES["min_sharpe_proxy"]:
        return False
    if float(metrics.get("profit_factor") or 0) < _PROMISING_HARD_GATES["min_profit_factor"]:
        return False
    if float(metrics.get("max_drawdown_pct") or 0) < _PROMISING_HARD_GATES["max_drawdown_pct"]:
        return False
    if int(metrics.get("total_trades") or 0) < _PROMISING_HARD_GATES["min_total_trades"]:
        return False
    # Passed hard gates — now check score or keywords
    notes = str(state.get("reviewer_notes") or "").lower()
    if any(kw in notes for kw in _PROMISING_KEYWORDS):
        return True
    score = float(metrics.get("research_score") or 0)
    return score >= _PROMISING_SCORE_THRESHOLD


def _write_promising_bundle(state: ResearchState, report_path: Path) -> str | None:
    exp_id = state.get("current_experiment_id", "?")
    record = ExperimentStore().load(exp_id) or {}
    if not record or not _is_promising(state, record):
        return None

    promising_root = Path(__file__).resolve().parent.parent / "promising_runs"
    bundle_dir = promising_root / exp_id
    if bundle_dir.exists():
        return str(bundle_dir.resolve())
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metrics = dict(record.get("metrics") or {})
    score = metrics.get("research_score", 0)
    why = "score >= threshold" if score >= _PROMISING_SCORE_THRESHOLD else "reviewer noted promising signal"

    _write_json(bundle_dir / "metrics.json", metrics)
    _write_json(bundle_dir / "record.json", record)
    _write_json(bundle_dir / "summary.json", {
        "experiment_id": exp_id,
        "hypothesis_id": state.get("active_hypothesis_id", ""),
        "code_direction": state.get("code_direction", ""),
        "reviewer_verdict": state.get("reviewer_verdict", ""),
        "reviewer_notes": state.get("reviewer_notes", ""),
        "research_score": score,
        "promising_reason": why,
    })
    shutil.copy2(report_path, bundle_dir / "report.log")

    script_path = state.get("code_script_path") or ""
    if script_path and Path(script_path).exists():
        shutil.copy2(script_path, bundle_dir / Path(script_path).name)

    setups_csv = state.get("generated_setups_csv") or ""
    if setups_csv and Path(setups_csv).exists():
        shutil.copy2(setups_csv, bundle_dir / Path(setups_csv).name)

    ExperimentStore().update(exp_id, {"promising": True, "promising_reason": why})
    return str(bundle_dir.resolve())


def _store_top_pairs(state: ResearchState, exp_id: str) -> None:
    """Persist per-pair stats snapshot to JSONL so router can aggregate edge pairs."""
    result = state.get("backtest_result") or {}
    pairs = result.get("pairs") or []
    if not pairs:
        return
    top_pairs = [
        {
            "pair_id": p.get("pair_id", ""),
            "trades": int(p.get("trades") or 0),
            "win_rate": float(p.get("win_rate") or 0),
            "total_pnl_pct": float(p.get("total_pnl_pct") or 0),
            "sharpe": float(p.get("sharpe") or 0),
            "stop_rate": float(p.get("stop_rate") or 0),
        }
        for p in pairs[:10]
        if int(p.get("trades") or 0) > 0
    ]
    if top_pairs:
        ExperimentStore().update(exp_id, {"top_pairs": top_pairs})


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _build_reproduce_text(
    exp_id: str,
    code_direction: str,
    config_path: str,
    scope: dict,
    bundle_dir: Path,
    script_path: str,
    setups_csv: str,
) -> str:
    script_name = Path(script_path).name if script_path else "script.py"
    setups_name = Path(setups_csv).name if setups_csv else "<generated_setups.csv>"

    lines = [
        f"# Reproduce {exp_id}",
        "",
        f"Research direction: {code_direction}",
        f"Base config: `{config_path}`",
        "",
        "Artifacts in this bundle:",
        "- `goal_contract.json`",
        "- `research_scope.json`",
        "- `result.json`",
        "- `metrics.json`",
        "- `record.json`",
        "- `report.log`",
        f"- `{script_name}` (signal generation script)",
        "",
        "To reproduce:",
        "1. Ensure research_config.json data_dir points to valid price data.",
        f"2. Re-run the standalone workspace script: `python {script_name}`",
        "3. The script regenerates the signals CSV.",
        f"4. Run `backtest/engine.py` on that signals CSV (example: `{setups_name}`).",
        "",
    ]

    if scope.get("selected_symbols"):
        lines.append(f"Selected symbols: `{', '.join(scope['selected_symbols'])}`")
        lines.append("")

    lines.append(f"Bundle directory: `{bundle_dir.resolve()}`")
    lines.append("")
    return "\n".join(lines)
