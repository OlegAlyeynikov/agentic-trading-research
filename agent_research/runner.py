"""
CLI entrypoint for the AI multi-agent trading research platform.

Usage:
    # Dry-run:
    python -m agent_research.runner \\
        --config research_config.json \\
        --goal "Find profitable pairs strategy" \\
        --max-iterations 3 \\
        --dry-run

    # Full run with LangSmith tracing:
    LANGSMITH_TRACING=true \\
    LANGSMITH_API_KEY=... \\
    LANGSMITH_PROJECT=trading-research \\
    python -m agent_research.runner \\
        --config research_config.json \\
        --goal "Find best stat-arb configuration" \\
        --max-iterations 10
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path


ENV_ALIASES = {
    "config": ("AGENT_RESEARCH_CONFIG",),
    "goal": ("AGENT_RESEARCH_GOAL", "AGENT_GOAL"),
    "max_iterations": ("AGENT_RESEARCH_MAX_ITERATIONS", "AGENT_MAX_ITER"),
    "max_iterations_per_hypothesis": ("AGENT_RESEARCH_MAX_ITERATIONS_PER_HYPOTHESIS",),
    "dry_run": ("AGENT_RESEARCH_DRY_RUN",),
    "log_level": ("AGENT_RESEARCH_LOG_LEVEL",),
    "log_file": ("AGENT_RESEARCH_LOG_FILE",),
    "json_logs": ("AGENT_RESEARCH_JSON_LOGS",),
    "hypotheses":  ("AGENT_RESEARCH_HYPOTHESES",),
    "reports_dir": ("AGENT_RESEARCH_REPORTS_DIR",),
    "code_budget": ("AGENT_RESEARCH_CODE_BUDGET",),
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass


def _apply_langsmith_env_aliases() -> None:
    """Mirror LANGSMITH_* vars to LANGCHAIN_* for LangGraph tracing."""
    tracing = os.environ.get("LANGSMITH_TRACING")
    if tracing is not None and "LANGCHAIN_TRACING_V2" not in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing.lower() == "true" else "false"

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if api_key and "LANGCHAIN_API_KEY" not in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = api_key

    project = os.environ.get("LANGSMITH_PROJECT")
    if project and "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = project

    endpoint = os.environ.get("LANGSMITH_ENDPOINT")
    if endpoint and "LANGCHAIN_ENDPOINT" not in os.environ:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return None


def _env_bool(*names: str) -> bool | None:
    value = _env_value(*names)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(*names: str) -> int | None:
    value = _env_value(*names)
    if value is None:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI multi-agent trading research runner.")
    p.add_argument("--config", default=None, help="Path to research_config.json")
    p.add_argument("--goal", default=None, help="Natural-language research goal")
    p.add_argument("--max-iterations", type=int, default=None, dest="max_iterations")
    p.add_argument(
        "--max-iterations-per-hypothesis",
        type=int,
        default=None,
        dest="max_iterations_per_hypothesis",
        help="Hard cap per hypothesis before forcing switch",
    )
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--log-level", dest="log_level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", dest="log_file", default=None,
                   help="Log file path (default: agent_research/logs/run_<timestamp>.jsonl)")
    p.add_argument("--json-logs", action="store_true", dest="json_logs",
                   help="Output JSON logs to stderr instead of colored console")
    p.add_argument(
        "--hypotheses", default=None, dest="hypotheses",
        help="Path to hypotheses.md file (default: agent_research/hypotheses.md)"
    )
    p.add_argument(
        "--reports-dir", default=None, dest="reports_dir",
        help="Directory for .log reports (default: agent_research/reports)"
    )
    p.add_argument(
        "--code-budget", type=int, default=None, dest="code_budget",
        help="Max number of code experiments per run (default: 100)"
    )
    args = p.parse_args()

    args.config = args.config or _env_value(*ENV_ALIASES["config"])
    args.goal = args.goal or _env_value(*ENV_ALIASES["goal"])
    args.max_iterations = (
        args.max_iterations
        if args.max_iterations is not None
        else (_env_int(*ENV_ALIASES["max_iterations"]) or 10)
    )
    args.max_iterations_per_hypothesis = (
        args.max_iterations_per_hypothesis
        if args.max_iterations_per_hypothesis is not None
        else (_env_int(*ENV_ALIASES["max_iterations_per_hypothesis"]) or 6)
    )
    if not args.dry_run:
        args.dry_run = _env_bool(*ENV_ALIASES["dry_run"]) or False
    args.log_level = args.log_level or _env_value(*ENV_ALIASES["log_level"]) or "INFO"
    args.log_file = args.log_file or _env_value(*ENV_ALIASES["log_file"])
    if not args.json_logs:
        args.json_logs = _env_bool(*ENV_ALIASES["json_logs"]) or False
    args.hypotheses = args.hypotheses or _env_value(*ENV_ALIASES["hypotheses"])
    args.reports_dir = args.reports_dir or _env_value(*ENV_ALIASES["reports_dir"])
    args.code_budget = args.code_budget or _env_int(*ENV_ALIASES["code_budget"]) or 100

    if not args.config:
        p.error("--config is required (or set AGENT_RESEARCH_CONFIG)")
    if not args.goal:
        p.error("--goal is required (or set AGENT_RESEARCH_GOAL / AGENT_GOAL)")

    return args


def _default_log_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    return str(log_dir / f"run_{ts}.jsonl")


def _langsmith_project_url() -> str | None:
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return None
    project = os.environ.get("LANGSMITH_PROJECT", "trading-research")
    return f"https://smith.langchain.com/projects/{project}"


def _print_banner(
    goal: str,
    max_iter: int,
    max_iter_per_hypothesis: int,
    dry_run: bool,
    log_file: str,
    reports_dir: str = "",
    code_budget: int = 100,
) -> None:
    from agent_research._llm import get_model, _get_reasoning_effort
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    ls_url = _langsmith_project_url()

    print("=" * 60)
    print("  AI TRADING RESEARCH PLATFORM")
    print("=" * 60)
    print(f"  Goal:           {goal}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Per hypothesis: {max_iter_per_hypothesis}")
    print(f"  Code budget:    {code_budget}")
    if reports_dir:
        print(f"  Reports dir:    {reports_dir}")
    if dry_run:
        print("  Models:         n/a (dry-run)")
    else:
        role_models = {
            "strategist": get_model(agent_name="strategist"),
            "researcher":  get_model(agent_name="researcher"),
            "reviewer":    get_model(agent_name="reviewer"),
            "coder":       get_model(agent_name="coder"),
        }
        print("  Models:")
        for role, model in role_models.items():
            effort = _get_reasoning_effort(role)
            reasoning_suffix = f"  reasoning={effort}" if effort and effort != "none" else ""
            print(f"    {role:<10} {model}{reasoning_suffix}")
    print(f"  API key:        {'set' if has_key else 'NOT SET'}")
    print(f"  Mode:           {'DRY-RUN (mock)' if dry_run else 'LIVE'}")
    print(f"  LangSmith:      {ls_url if ls_url else 'disabled'}")
    print(f"  Log file:       {log_file}")
    print("=" * 60)


def _print_final_summary(final_state: dict | None, experiment_ids: list, goal: str, goal_contract: dict | None = None) -> None:
    from agent_research.memory import ExperimentStore
    from agent_research.logging_config import get_logger
    _log = get_logger("runner")

    store = ExperimentStore()
    goal_contract = dict(goal_contract or {})
    primary_metric = str(goal_contract.get("primary_metric") or "research_score")
    best = store.find_best(primary_metric, goal_contract=goal_contract)
    n = int((final_state or {}).get("iteration", len(experiment_ids)) or 0)
    raw_attempts = len(experiment_ids)
    approved_ids = list((final_state or {}).get("approved_experiment_ids") or [])

    print("\n" + "=" * 50)
    print(f"RESEARCH COMPLETE — {n} iteration{'s' if n != 1 else ''}")
    print("=" * 50)
    print(f"Goal:    {goal}")
    if raw_attempts != n:
        print(f"Attempts: {raw_attempts} total, {n} counted iterations")

    if approved_ids:
        print(f"Approved: {len(approved_ids)} experiment(s) met all goal-contract criteria")
        for aid in approved_ids:
            rec = store.load(aid)
            if rec:
                m = rec.get("metrics", {})
                artifact_dir = rec.get("final_artifact_dir", "")
                print(
                    f"  [{aid}]  "
                    f"score={m.get('research_score', 0):.1f}  "
                    f"sharpe={m.get('sharpe_proxy', 0):.2f}  "
                    f"pf={m.get('profit_factor', 0):.2f}  "
                    f"maxDD={m.get('max_drawdown_pct', 0):+.1f}%  "
                    f"equity={m.get('estimated_equity_return_pct', 0):+.2f}%"
                )
                if artifact_dir:
                    print(f"    Bundle: {artifact_dir}")

    if best:
        m = best.get("metrics", {})
        notes = best.get("reviewer_notes", "")
        direction = (best.get("code_direction") or "")[:120]
        ls_url = _langsmith_project_url()
        print(f"Best:    {best.get('experiment_id', '?')}")
        if direction:
            print(f"Direction: {direction}")
        print(
            f"Metrics: trades={m.get('total_trades', '?')}  "
            f"score={m.get('research_score', 0):.1f}  "
            f"sharpe={m.get('sharpe_proxy', 0):.2f}  "
            f"pf={m.get('profit_factor', 0):.2f}  "
            f"maxDD={m.get('max_drawdown_pct', 0):+.1f}%"
        )
        print(
            f"Human:   win_rate={m.get('win_rate', 0):.1%}  "
            f"stop_rate={m.get('stop_rate', 0):.1%}  "
            f"equity_return={m.get('estimated_equity_return_pct', 0):+.2f}%  "
            f"sum_trade_pnl={m.get('sum_trade_pnl_pct', m.get('total_pnl_pct', 0)):+.1f}%"
        )
        artifact_dir = best.get("final_artifact_dir", "")
        if artifact_dir:
            print(f"Bundle:  {artifact_dir}")
        if notes:
            print(f"Notes:   {notes[:200]}")
        if ls_url:
            print(f"Traces:  {ls_url}")
        _log.info(
            "research_complete",
            iterations=n,
            best_experiment_id=best.get("experiment_id"),
            best_research_score=m.get("research_score"),
            best_sharpe=m.get("sharpe_proxy"),
            approved_count=len(approved_ids),
        )
    else:
        print("Best:    no approved experiment found")

    print("=" * 50 + "\n")


def main() -> int:
    _load_dotenv()
    _apply_langsmith_env_aliases()
    args = parse_args()
    from agent_research.config_utils import load_full_config, load_goal_contract
    full_config = load_full_config(args.config)
    goal_contract = load_goal_contract(args.config)

    log_file = args.log_file or _default_log_file()

    from agent_research.logging_config import setup_logging
    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        json_logs=args.json_logs,
    )

    from agent_research.logging_config import get_logger
    _log = get_logger("runner")
    _log.info(
        "runner_start",
        goal=args.goal,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
        log_file=log_file,
        langsmith_enabled=bool(_langsmith_project_url()),
    )

    from pathlib import Path as _Path
    hypotheses_path = args.hypotheses or str(_Path(__file__).parent / "hypotheses.md")
    hypotheses_text = ""
    if _Path(hypotheses_path).exists():
        hypotheses_text = _Path(hypotheses_path).read_text(encoding="utf-8")

    reports_dir = args.reports_dir or str(_Path(__file__).parent / "reports")

    _print_banner(
        args.goal, args.max_iterations, args.max_iterations_per_hypothesis, args.dry_run,
        log_file, reports_dir, args.code_budget,
    )

    from agent_research.graph import compile_graph
    graph = compile_graph()
    thread_config = {"configurable": {"thread_id": "research_1"}}

    initial_state: dict = {
        "goal": args.goal,
        "goal_contract": goal_contract,
        "data_dir": full_config.get("data_dir", ""),
        "funding_rates_dir": full_config.get("funding_rates_dir", ""),
        "iteration": 0,
        "max_iterations": args.max_iterations,
        "max_iterations_per_hypothesis": args.max_iterations_per_hypothesis,
        "research_cycle": 1,
        "code_direction": "",
        "current_scope": {
            "selected_symbols": full_config.get("symbols") or [],
            "start_date": full_config.get("start_date"),
        },
        "current_experiment_id": "",
        "backtest_result": None,
        "reviewer_verdict": "",
        "reviewer_notes": "",
        "reviewer_diagnosis": {},
        "router_decision": "",
        "router_message": "",
        "duplicate_of_experiment_id": "",
        "experiment_ids": [],
        "best_experiment_id": None,
        "config_path": args.config,
        "dry_run": args.dry_run,
        "langsmith_run_url": _langsmith_project_url(),
        "hypotheses_text": hypotheses_text,
        "active_hypothesis_id": "",
        "next_hypothesis_id": "",
        "completed_hypothesis_ids": [],
        "hypothesis_iterations": {},
        "analyst_memo": "",
        "analyst_new_hypotheses": [],
        "reports_dir": reports_dir,
        "report_path": None,
        "final_artifact_dir": None,
        "approved_experiment_ids": [],
        # Code pipeline state
        "allow_code_changes": True,
        "code_change_budget": args.code_budget,
        "code_change_attempts": 0,
        "code_change_proposal": {},
        "strategy_flags": {},
        "code_script_filename": "",
        "code_script_contents": "",
        "validation_result": {},
        "workspace_dir": "",
        "code_script_path": "",
        "generated_setups_csv": "",
        "sandbox_result": {},
        "script_hash": "",
        "setups_csv_hash": "",
        "lookahead_flags": [],
    }

    final_state = _run_graph(graph, initial_state, thread_config)
    experiment_ids = final_state.get("experiment_ids", []) if final_state else []
    _print_final_summary(final_state, experiment_ids, args.goal, goal_contract)
    return 0


def _run_graph(graph, initial_state: dict, thread_config: dict) -> dict | None:
    from agent_research.logging_config import get_logger
    _log = get_logger("runner")

    state_snapshot = None
    try:
        for event in graph.stream(initial_state, config=thread_config, stream_mode="values"):
            state_snapshot = event
    except KeyboardInterrupt:
        _log.warning("runner_interrupted")
        print("\n[runner] interrupted by user")
    return state_snapshot


if __name__ == "__main__":
    raise SystemExit(main())
