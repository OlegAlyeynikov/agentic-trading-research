from typing import TypedDict, Optional


class ResearchState(TypedDict):
    # ── Research goal and progress ────────────────────────────────
    goal: str
    data_dir: str
    funding_rates_dir: str
    iteration: int
    max_iterations: int
    max_iterations_per_hypothesis: int
    research_cycle: int
    config_path: str
    dry_run: bool
    langsmith_run_url: Optional[str]
    goal_contract: dict

    # ── Hypothesis tracking ───────────────────────────────────────
    hypotheses_text: str            # full content of hypotheses.md
    active_hypothesis_id: str       # e.g. "H1", "H2", ...
    completed_hypothesis_ids: list  # hypotheses fully explored
    hypothesis_iterations: dict     # {hypothesis_id: iteration_count}

    # ── Current experiment ────────────────────────────────────────
    code_direction: str             # researcher's natural-language direction for coder
    current_scope: dict             # {selected_symbols, start_date, ...}
    current_experiment_id: str
    backtest_result: Optional[dict]
    duplicate_of_experiment_id: str

    # ── Agent verdicts ────────────────────────────────────────────
    reviewer_verdict: str           # "approve" | "reject"
    reviewer_notes: str
    reviewer_diagnosis: dict        # structured root-cause dict from ReviewerResponse
    router_decision: str            # "next_iteration" | "switch_hypothesis" | "done"
    router_message: str

    # ── Memory ───────────────────────────────────────────────────
    experiment_ids: list
    best_experiment_id: Optional[str]
    edge_pairs: list               # pairs with cross-exp edge: [{pair_id, appearances, avg_win_rate, avg_pnl_pct, avg_sharpe, avg_trades}]
    portfolio_candidate: dict      # best untested portfolio: {symbols, pairs, expected_trades, note}

    # ── Router hint for strategist ────────────────────────────────
    next_hypothesis_id: str          # set by router on switch_hypothesis

    # ── Strategist synthesis ──────────────────────────────────────
    analyst_memo: str                # cross-experiment synthesis written by strategist
    analyst_new_hypotheses: list     # H-IDs of hypotheses appended this cycle

    # ── Code execution pipeline ───────────────────────────────────
    allow_code_changes: bool         # always True — kept for compatibility
    code_change_budget: int          # max total code experiments per run
    code_change_attempts: int        # count of code experiments so far
    code_change_proposal: dict       # researcher's structured code goal
    strategy_flags: dict             # explicit execution/model flags for the current strategy
    code_script_filename: str        # standalone workspace Python script
    code_script_contents: str        # full generated script contents
    workspace_dir: str               # agent_research/workspace/<experiment_id>
    code_script_path: str            # absolute path to generated script
    generated_setups_csv: str        # signals CSV produced by sandboxed script
    validation_result: dict          # {passed, errors, compile_ok, static_checks_ok, lookahead_ok, sandbox_ok}
    sandbox_result: dict             # {returncode, stdout_json, stderr_path, runtime_seconds}
    script_hash: str                 # sha256[:12] of generated script
    setups_csv_hash: str             # sha256[:12] of signals CSV — output-level dedup key
    lookahead_flags: list            # heuristic static warnings for future leakage

    # ── Reporting ─────────────────────────────────────────────────
    reports_dir: str                # path to reports/ folder
    report_path: Optional[str]      # path to this experiment's .log file
    final_artifact_dir: Optional[str]  # path to reproducibility bundle for approved experiment

    # ── Approved experiments ──────────────────────────────────────
    approved_experiment_ids: list   # IDs of experiments that satisfied goal_contract this session
