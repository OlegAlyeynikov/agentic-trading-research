import json
from agent_research.state import ResearchState
from agent_research._llm import load_prompt, call_llm
from agent_research.memory import ExperimentStore
from agent_research.config_utils import goal_contract_progress, goal_contract_summary
from agent_research.logging_config import get_logger, AgentTimer

_log = get_logger("reviewer")

_MOCK_APPROVE = {"verdict": "approve", "notes": "[dry-run] Mock approval — result looks reasonable."}
_MOCK_REJECT  = {"verdict": "reject",  "notes": "[dry-run] Mock rejection — 0 trades."}

_DIAGNOSIS_KEYS = (
    "root_cause", "failed_dimensions", "passing_dimensions",
    "diagnostic_insight", "suggested_direction", "cross_strategy_note",
)


def run_reviewer(state: ResearchState) -> ResearchState:
    result = state.get("backtest_result") or {}
    exp_id = state.get("current_experiment_id", "?")

    with AgentTimer(_log, "reviewer", exp_id, state.get("iteration", 0)):
        if state.get("dry_run"):
            _update_store(exp_id, {"reviewer_verdict": "approve", "reviewer_notes": _MOCK_APPROVE["notes"]})
            _log.info("reviewer_dry_run", verdict="approve", experiment_id=exp_id)
            print(f"[reviewer] {exp_id}  verdict=approve  [dry-run]")
            return {**state, "reviewer_verdict": "approve", "reviewer_notes": _MOCK_APPROVE["notes"],
                    "reviewer_diagnosis": {}}

        store = ExperimentStore()
        store_summary = store.summarize_for_llm(5)
        goal_contract = dict(state.get("goal_contract") or {})
        record = store.load(exp_id) or {}
        metrics = dict(record.get("metrics") or {})
        goal_progress = goal_contract_progress(metrics, goal_contract)

        pipeline_reject = _pipeline_or_script_bug_diagnosis(state, result)
        if pipeline_reject:
            verdict = "reject"
            notes = pipeline_reject["notes"]
            diagnosis = {k: pipeline_reject[k] for k in _DIAGNOSIS_KEYS if pipeline_reject.get(k) is not None}
            _update_store(exp_id, {
                "reviewer_verdict": verdict,
                "reviewer_notes": notes,
                "reviewer_diagnosis": diagnosis,
            })
            _log.info(
                "reviewer_verdict",
                experiment_id=exp_id,
                verdict=verdict,
                root_cause=diagnosis.get("root_cause"),
                total_trades=result.get("stats", {}).get("total_trades"),
                total_pnl=result.get("stats", {}).get("total_pnl_pct"),
            )
            print(f"[reviewer] {exp_id}  verdict={verdict}  root_cause={diagnosis.get('root_cause', '?')}")
            print(f"[reviewer] notes: {notes[:100]}")
            return {
                **state,
                "reviewer_verdict": verdict,
                "reviewer_notes": notes,
                "reviewer_diagnosis": diagnosis,
            }

        system_prompt = load_prompt("reviewer")
        user_message = (
            f"Backtest result:\n{json.dumps(result, indent=2)}\n\n"
            f"Goal contract:\n{json.dumps(goal_contract, indent=2)}\n\n"
            f"Goal contract summary:\n{goal_contract_summary(goal_contract)}\n\n"
            f"Derived metrics / goal progress:\n{json.dumps(goal_progress, indent=2)}\n\n"
            f"Recent experiment history for context:\n{store_summary}"
        )

        response = call_llm(
            system_prompt, user_message, temperature=0.0,
            agent_name="reviewer", experiment_id=exp_id,
            iteration=state.get("iteration", 0),
        )
        verdict = response.get("verdict", "reject")
        notes = response.get("notes", "")
        diagnosis = {k: response[k] for k in _DIAGNOSIS_KEYS if response.get(k) is not None}

        _update_store(exp_id, {
            "reviewer_verdict": verdict,
            "reviewer_notes": notes,
            "reviewer_diagnosis": diagnosis,
        })
        _log.info(
            "reviewer_verdict",
            experiment_id=exp_id,
            verdict=verdict,
            root_cause=diagnosis.get("root_cause"),
            total_trades=result.get("stats", {}).get("total_trades"),
            total_pnl=result.get("stats", {}).get("total_pnl_pct"),
        )
        print(f"[reviewer] {exp_id}  verdict={verdict}  root_cause={diagnosis.get('root_cause', '?')}")
        if notes:
            print(f"[reviewer] notes: {notes[:100]}")
        if diagnosis.get("suggested_direction"):
            print(f"[reviewer] suggested: {diagnosis['suggested_direction'][:100]}")

    return {
        **state,
        "reviewer_verdict": verdict,
        "reviewer_notes": notes,
        "reviewer_diagnosis": diagnosis,
    }


def _update_store(exp_id: str, updates: dict) -> None:
    ExperimentStore().update(exp_id, updates)


def _pipeline_or_script_bug_diagnosis(state: ResearchState, result: dict) -> dict:
    validation = dict(state.get("validation_result") or {})
    errors = list(validation.get("errors") or [])
    blocked_reason = str(result.get("execution_blocked_reason") or "")
    generated_csv = str(state.get("generated_setups_csv") or "")
    stats = dict(result.get("stats") or {})
    total_trades = int(stats.get("total_trades") or 0)

    validation_failed = validation and not bool(validation.get("passed", True))
    no_csv_zero_trades = not generated_csv and total_trades == 0
    repairable_no_signal = any(
        "repairable blocked/no-signal" in str(error).lower()
        or "timestamp alignment" in str(error).lower()
        or "action values" in str(error).lower()
        for error in errors
    )
    blocked_validation = "workspace script validation failed" in blocked_reason.lower()

    if not (validation_failed or repairable_no_signal or (blocked_validation and no_csv_zero_trades)):
        return {}

    detail = "; ".join(str(error) for error in errors[:3]) or blocked_reason or "no setups CSV produced"
    return {
        "verdict": "reject",
        "notes": (
            "Pipeline/script failure, not a market signal conclusion: no usable setups CSV was produced. "
            f"Primary issue: {detail}"
        ),
        "root_cause": "pipeline_or_script_bug",
        "failed_dimensions": ["script_validation", "setups_generation"],
        "passing_dimensions": [],
        "diagnostic_insight": (
            "Do not lower thresholds or change universe based on this result. Fix script generation/validation first, "
            "especially timestamp alignment, action names, hidden exceptions, and blocked/no-signal diagnostics."
        ),
        "suggested_direction": (
            "Repair coder/code_executor path; rerun the same hypothesis only after the script generates a valid setups CSV."
        ),
        "cross_strategy_note": None,
    }
