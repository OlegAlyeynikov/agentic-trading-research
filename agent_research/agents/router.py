"""
Router agent — decides research flow in autonomous mode.

Decisions:
  next_iteration    — continue testing within the current hypothesis
  switch_hypothesis — current hypothesis exhausted, move to next
  done              — success criteria met OR all hypotheses exhausted OR max_iterations reached
"""
import re
from pathlib import Path
from agent_research.state import ResearchState
from agent_research.memory import ExperimentStore
from agent_research.config_utils import goal_contract_satisfied
from agent_research.logging_config import get_logger, AgentTimer

_SEP = "=" * 80
_THIN = "-" * 80

_log = get_logger("router")
_RECYCLE_HYPOTHESIS = "H8"
_RETIRED_AGENT_HYPOTHESES: frozenset[str] = frozenset({"H4", "H5"})
_HYPOTHESES_PATH = Path(__file__).parent.parent / "hypotheses.md"

# Pairs that consistently destroy portfolio drawdown when combined with others.
# These are kept in edge_pairs for informational context but excluded from portfolio candidates.
_PORTFOLIO_BLACKLIST: frozenset[str] = frozenset({"AVAXUSDT-SOLUSDT", "SOLUSDT-AVAXUSDT"})


def _load_all_hypotheses() -> list[str]:
    if not _HYPOTHESES_PATH.exists():
        return ["H1", "H2", "H3", "H7", "H8", "H9", "H10", "H11"]
    text = _HYPOTHESES_PATH.read_text(encoding="utf-8")
    ids = re.findall(r"^## (H\d+)", text, re.MULTILINE)
    seen = set()
    return [h for h in ids if h not in _RETIRED_AGENT_HYPOTHESES and not (h in seen or seen.add(h))]


def _is_duplicate_blocked(state: ResearchState) -> bool:
    if state.get("duplicate_of_experiment_id"):
        return True
    result = state.get("backtest_result") or {}
    if result.get("status") != "blocked":
        return False
    reason = str(result.get("execution_blocked_reason") or "")
    return "duplicate_of=" in reason


def _is_coder_fallback_blocked_record(record: dict) -> bool:
    result = record.get("result", {}) or {}
    if result.get("status") != "blocked":
        return False

    gap = result.get("data_gap_report", {}) or {}
    notes = " ".join(
        [
            str(result.get("execution_blocked_reason") or ""),
            str(gap.get("notes") or ""),
            str(record.get("reviewer_notes") or ""),
        ]
    ).lower()
    return (
        "coder model returned an unusable response" in notes
        or "no standalone script could be generated" in notes
        or "valid coder llm output containing executable python script contents" in notes
    )


def _is_coder_blocked(state: ResearchState) -> bool:
    """True when the current experiment failed due to coder LLM producing unusable output."""
    result = state.get("backtest_result") or {}
    if result.get("status") != "blocked":
        return False
    gap = result.get("data_gap_report", {}) or {}
    notes = " ".join([
        str(result.get("execution_blocked_reason") or ""),
        str(gap.get("notes") or ""),
        str(state.get("reviewer_notes") or ""),
    ]).lower()
    return (
        "coder model returned an unusable response" in notes
        or "no standalone script could be generated" in notes
        or "valid coder llm output containing executable python script contents" in notes
    )


def _recent_coder_fallback_count(store: ExperimentStore, hypothesis_id: str, limit: int = 6) -> int:
    records = [
        r for r in store.list_recent(50)
        if (r.get("active_hypothesis_id") or "") == hypothesis_id
    ]
    return sum(1 for r in records[-limit:] if _is_coder_fallback_blocked_record(r))


def _is_pipeline_bug_record(record: dict) -> bool:
    diagnosis = record.get("reviewer_diagnosis") or {}
    if diagnosis.get("root_cause") == "pipeline_or_script_bug":
        return True
    validation = record.get("validation_result") or {}
    if validation and validation.get("passed") is False:
        return True
    result = record.get("result") or {}
    reason = str(result.get("execution_blocked_reason") or "").lower()
    if result.get("status") == "blocked" and any(
        marker in reason
        for marker in (
            "validation failed",
            "did not produce signals csv",
            "workspace script",
            "sandbox",
            "invalid json",
        )
    ):
        return True
    return False


def _recent_pipeline_bug_count(store: ExperimentStore, hypothesis_id: str, limit: int = 4) -> int:
    records = store.list_records_for_hypothesis(hypothesis_id, limit=limit)
    count = 0
    for rec in reversed(records):
        if _is_pipeline_bug_record(rec):
            count += 1
        else:
            break
    return count


def _recent_consecutive_duplicate_count(store: ExperimentStore, hypothesis_id: str) -> int:
    records = store.list_records_for_hypothesis(hypothesis_id, limit=10)
    count = 0
    for rec in reversed(records):
        reason = str(rec.get("execution_blocked_reason") or "")
        dup_id = str(rec.get("duplicate_of_experiment_id") or "")
        if "duplicate_of=" in reason or dup_id:
            count += 1
        else:
            break
    return count


def _find_edge_pairs(store: ExperimentStore, limit: int = 60) -> list[dict]:
    """Return pairs with consistent profitability across ≥2 experiments."""
    pair_stats: dict[str, list[dict]] = {}
    for rec in store.list_recent(limit):
        result = rec.get("result") or {}
        if result.get("status") == "blocked":
            continue
        pairs = rec.get("top_pairs") or result.get("pairs") or []
        for p in pairs:
            pid = p.get("pair_id", "")
            if not pid:
                continue
            pnl = float(p.get("total_pnl_pct") or 0)
            win_rate = float(p.get("win_rate") or 0)
            sharpe = float(p.get("sharpe") or 0)
            trades = int(p.get("trades") or 0)
            if trades >= 3 and pnl > 5.0 and win_rate > 0.5:
                pair_stats.setdefault(pid, []).append(
                    {"pnl": pnl, "win_rate": win_rate, "sharpe": sharpe, "trades": trades}
                )
    edge: list[dict] = []
    for pid, appearances in pair_stats.items():
        if len(appearances) < 2:
            continue
        n = len(appearances)
        avg_pnl = sum(a["pnl"] for a in appearances) / n
        avg_win = sum(a["win_rate"] for a in appearances) / n
        avg_sharpe = sum(a["sharpe"] for a in appearances) / n
        avg_trades = sum(a["trades"] for a in appearances) / n
        if avg_pnl > 10.0 and avg_win > 0.6:
            edge.append({
                "pair_id": pid,
                "appearances": n,
                "avg_win_rate": round(avg_win, 3),
                "avg_pnl_pct": round(avg_pnl, 2),
                "avg_sharpe": round(avg_sharpe, 3),
                "avg_trades": round(avg_trades, 1),
            })
    return sorted(edge, key=lambda x: (-x["appearances"], -x["avg_pnl_pct"]))


def _build_portfolio_candidate(edge_pairs: list[dict]) -> dict:
    """Propose a combined portfolio from all confirmed edge pairs.

    Returns an empty dict if fewer than 2 edge pairs are known.
    Pairs in _PORTFOLIO_BLACKLIST are excluded even if they have edge.
    """
    viable = [ep for ep in edge_pairs if ep.get("pair_id") not in _PORTFOLIO_BLACKLIST]
    if len(viable) < 2:
        return {}
    symbols: set[str] = set()
    for ep in viable:
        pid = ep.get("pair_id", "")
        if "-" in pid:
            a, b = pid.split("-", 1)
            symbols.add(a)
            symbols.add(b)
    expected_trades = sum(int(ep.get("avg_trades") or 0) for ep in viable)
    pairs_list = [ep["pair_id"] for ep in viable]
    return {
        "symbols": sorted(symbols),
        "pairs": pairs_list,
        "expected_trades": expected_trades,
        "note": (
            f"{len(pairs_list)} confirmed edge pairs × "
            f"~{expected_trades // len(pairs_list)} avg_trades = "
            f"~{expected_trades} combined trades"
        ),
    }


def _recent_structural_scope_failures(
    store: ExperimentStore, hypothesis_id: str, limit: int = 6
) -> int:
    """Count scope failures scoped to current hypothesis only.

    Counting globally (across all hypotheses) causes H2/H3/H4 failures to
    pollute H1's counter and trigger a premature H8 switch.
    """
    count = 0
    records = store.list_records_for_hypothesis(hypothesis_id, limit=limit * 2)
    for rec in reversed(records):
        if rec.get("reviewer_verdict") != "reject":
            continue
        notes = str(rec.get("reviewer_notes") or "").lower()
        scope_fail = (
            "structurally invalid" in notes
            or "current symbol universe" in notes
            or "further tuning on this scope is not recommended" in notes
            or "further tuning on this set is not recommended" in notes
            or ("further tuning on this" in notes and "not recommended" in notes)
            or ("symbol scope is invalid" in notes)
            or ("scope is not viable" in notes)
        )
        if not scope_fail:
            continue
        # False-positive guard: if any pair shows genuine edge (sharpe > 0.3) the real
        # problem is parameter tuning (e.g. entry_z too selective), not a broken scope.
        # Threshold 0.3 (was 0.4) so borderline-profitable experiments are protected.
        pairs = (rec.get("top_pairs") or (rec.get("result") or {}).get("pairs") or [])
        if any(float(p.get("sharpe") or 0) > 0.3 for p in pairs):
            continue
        count += 1
        if count >= limit:
            return count
    return count


def _derive_hypotheses_from_state(state: ResearchState) -> list[str]:
    text = state.get("hypotheses_text") or ""
    if text:
        ids = re.findall(r"^## (H\d+)", text, re.MULTILINE)
        seen: set[str] = set()
        return [h for h in ids if h not in _RETIRED_AGENT_HYPOTHESES and not (h in seen or seen.add(h))]
    return _load_all_hypotheses()


def _append_decision_to_report(state: ResearchState, decision: str, message: str) -> None:
    report_path = state.get("report_path") or ""
    if not report_path:
        return
    p = Path(report_path)
    if not p.exists():
        return
    try:
        existing = p.read_text(encoding="utf-8")
        placeholder = f"{_THIN}\nROUTER DECISION\n{_THIN}\n  (pending — written by router after this report)\n\n{_SEP}\n"
        replacement_lines = [_THIN, "ROUTER DECISION", _THIN, f"  Decision  : {decision}"]
        if message:
            replacement_lines.append(f"  Message   : {message}")
        replacement_lines += ["", _SEP, ""]
        replacement = "\n".join(replacement_lines)
        updated = existing.replace(placeholder, replacement)
        if updated == existing:
            updated = existing.rstrip("\n") + "\n" + replacement
        p.write_text(updated, encoding="utf-8")
    except Exception as exc:
        _log.warning("router_report_append_failed", error=str(exc))


def run_router(state: ResearchState) -> ResearchState:
    result = _run_router_inner(state)
    _append_decision_to_report(
        state,
        decision=result.get("router_decision", ""),
        message=result.get("router_message", ""),
    )
    return result


def _run_router_inner(state: ResearchState) -> ResearchState:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    verdict = state.get("reviewer_verdict", "reject")
    exp_id = state.get("current_experiment_id", "?")
    active_h = state.get("active_hypothesis_id", "H1")
    hyp_iters = dict(state.get("hypothesis_iterations") or {})
    completed = list(state.get("completed_hypothesis_ids") or [])
    code_attempts = state.get("code_change_attempts", 0)
    code_budget = state.get("code_change_budget", 100)
    goal_contract = dict(state.get("goal_contract") or {})
    max_iter_per_hypothesis = int(state.get("max_iterations_per_hypothesis", 6) or 6)
    duplicate_blocked = _is_duplicate_blocked(state)
    research_cycle = int(state.get("research_cycle", 1) or 1)
    approved_experiment_ids = list(state.get("approved_experiment_ids") or [])

    all_hypotheses = _derive_hypotheses_from_state(state)
    coder_blocked = _is_coder_blocked(state)
    infra_failure = duplicate_blocked or coder_blocked
    effective_iteration = iteration - 1 if infra_failure and iteration > 0 else iteration
    effective_hyp_iters = dict(hyp_iters)
    if infra_failure and active_h in effective_hyp_iters and effective_hyp_iters[active_h] > 0:
        effective_hyp_iters[active_h] -= 1

    new_code_attempts = code_attempts + 1 if not infra_failure else code_attempts
    edge_pairs: list = []
    portfolio_candidate: dict = {}

    with AgentTimer(_log, "router", exp_id, iteration):
        if effective_iteration >= max_iter:
            _log.info("router_max_iter_reached", iteration=effective_iteration)
            print(f"[router] max iterations reached ({effective_iteration}/{max_iter})")
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": effective_hyp_iters,
                "router_decision": "done",
                "router_message": "Max iterations reached.",
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        if state.get("dry_run"):
            if effective_hyp_iters.get(active_h, 0) >= 2:
                decision, message = "switch_hypothesis", ""
            else:
                decision, message = "next_iteration", ""
            _log.info("router_dry_run", decision=decision, iteration=iteration)
            print(f"[router] iter={effective_iteration}/{max_iter}  decision={decision}  [dry-run]")
            new_completed = completed + [active_h] if decision == "switch_hypothesis" else completed
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": effective_hyp_iters,
                "router_decision": decision,
                "router_message": message,
                "completed_hypothesis_ids": new_completed,
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        store = ExperimentStore()
        edge_pairs = _find_edge_pairs(store)
        portfolio_candidate = _build_portfolio_candidate(edge_pairs)
        primary_metric = str(goal_contract.get("primary_metric") or "research_score")
        best = store.find_best(primary_metric, goal_contract=goal_contract)
        recent_coder_fallbacks = _recent_coder_fallback_count(store, active_h)
        recent_pipeline_bugs = _recent_pipeline_bug_count(store, active_h)
        structural_scope_failures = _recent_structural_scope_failures(store, active_h)
        consecutive_dupes = _recent_consecutive_duplicate_count(store, active_h)
        scope_fail_threshold = 5 if edge_pairs else 3

        if active_h in _RETIRED_AGENT_HYPOTHESES:
            message = (
                f"Hypothesis {active_h} is retired from the agent loop; "
                "use python -m agent_research.funding_grid_search for this parameter/symbol search."
            )
            new_completed = completed + ([active_h] if active_h not in completed else [])
            _log.info("router_switch_retired_hypothesis", hypothesis=active_h)
            print(f"[router] {message}")
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": effective_hyp_iters,
                "router_decision": "switch_hypothesis",
                "router_message": message,
                "next_hypothesis_id": "",
                "completed_hypothesis_ids": new_completed,
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        if best:
            m = best.get("metrics", {})
            best_exp_id = best.get("experiment_id", "")
            if goal_contract_satisfied(m, goal_contract) and best_exp_id not in approved_experiment_ids:
                new_approved = approved_experiment_ids + [best_exp_id]
                msg = (
                    f"AUTO-APPROVED: {best_exp_id} "
                    f"score={m.get('research_score', 0):.1f} "
                    f"sharpe={m.get('sharpe_proxy', 0):.2f} "
                    f"pf={m.get('profit_factor', 0):.2f} "
                    f"maxDD={m.get('max_drawdown_pct', 0):+.2f}% "
                    f"avg_dur={m.get('avg_duration_hours', 0):.1f}h — "
                    f"continuing research on other symbols/hypotheses."
                )
                _log.info(
                    "router_auto_approved",
                    experiment_id=best_exp_id,
                    approved_count=len(new_approved),
                    **{k: v for k, v in m.items() if isinstance(v, (int, float))},
                )
                print(f"[router] {msg}")
                new_completed = (
                    completed + [active_h] if active_h not in completed else list(completed)
                )
                remaining = [h for h in all_hypotheses if h not in new_completed]
                decision = "switch_hypothesis" if remaining and effective_iteration < max_iter else "done"
                return {
                    **state,
                    "iteration": effective_iteration,
                    "hypothesis_iterations": effective_hyp_iters,
                    "router_decision": decision,
                    "router_message": msg,
                    "next_hypothesis_id": "",
                    "completed_hypothesis_ids": new_completed,
                    "approved_experiment_ids": new_approved,
                    "code_change_attempts": new_code_attempts,
                    "edge_pairs": edge_pairs,
                    "portfolio_candidate": portfolio_candidate,
                }

        all_hypotheses_completed = all(h in completed for h in all_hypotheses)
        if all_hypotheses_completed:
            if approved_experiment_ids:
                message = (
                    f"All hypotheses explored. "
                    f"{len(approved_experiment_ids)} approved experiment(s): "
                    f"{', '.join(approved_experiment_ids)}. Research complete."
                )
                _log.info(
                    "router_all_hypotheses_done",
                    approved_count=len(approved_experiment_ids),
                    approved_ids=approved_experiment_ids,
                )
                print(f"[router] {message}")
                return {
                    **state,
                    "iteration": effective_iteration,
                    "hypothesis_iterations": effective_hyp_iters,
                    "router_decision": "done",
                    "router_message": message,
                    "code_change_attempts": new_code_attempts,
                    "edge_pairs": edge_pairs,
                    "portfolio_candidate": portfolio_candidate,
                    "approved_experiment_ids": approved_experiment_ids,
                }
            # No approved experiments after full exploration — recycle
            message = (
                f"Completed research cycle {research_cycle} with no approved experiment; "
                f"starting cycle {research_cycle + 1} from {_RECYCLE_HYPOTHESIS} to rebuild the symbol universe."
            )
            _log.info(
                "router_recycle_cycle",
                cycle=research_cycle,
                next_cycle=research_cycle + 1,
                next_hypothesis=_RECYCLE_HYPOTHESIS,
            )
            print(f"[router] {message}")
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": {},
                "router_decision": "switch_hypothesis",
                "router_message": message,
                "next_hypothesis_id": _RECYCLE_HYPOTHESIS,
                "completed_hypothesis_ids": [],
                "research_cycle": research_cycle + 1,
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        incremented_hyp_iters = dict(effective_hyp_iters)
        if not infra_failure:
            incremented_hyp_iters[active_h] = incremented_hyp_iters.get(active_h, 0) + 1
        h_iter_count = incremented_hyp_iters.get(active_h, 0)
        force_switch = h_iter_count >= max_iter_per_hypothesis

        if consecutive_dupes >= 3:
            message = (
                f"Hypothesis {active_h} produced {consecutive_dupes} consecutive duplicate-blocked experiments; "
                "forcing hypothesis switch to break the loop."
            )
            new_completed = completed + ([active_h] if active_h not in completed else [])
            _log.info(
                "router_switch_consecutive_dupes",
                hypothesis=active_h,
                consecutive_dupes=consecutive_dupes,
            )
            print(f"[router] {message}")
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": incremented_hyp_iters,
                "router_decision": "switch_hypothesis",
                "router_message": message,
                "next_hypothesis_id": "",
                "completed_hypothesis_ids": new_completed,
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        if recent_coder_fallbacks >= 2:
            message = (
                f"Hypothesis {active_h} hit {recent_coder_fallbacks} coder-fallback blocked runs; "
                "switching hypothesis instead of repeating the same code-generation failure."
            )
            new_completed = completed
            if active_h not in completed:
                new_completed = completed + [active_h]
            _log.info(
                "router_switch_coder_fallbacks",
                hypothesis=active_h,
                fallback_runs=recent_coder_fallbacks,
            )
            print(f"[router] {message}")
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": incremented_hyp_iters,
                "router_decision": "switch_hypothesis",
                "router_message": message,
                "completed_hypothesis_ids": new_completed,
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        if recent_pipeline_bugs >= 2:
            message = (
                f"Hypothesis {active_h} hit {recent_pipeline_bugs} consecutive pipeline/script failures; "
                "switching hypothesis instead of spending more iterations on broken plumbing."
            )
            new_completed = completed + ([active_h] if active_h not in completed else [])
            _log.info(
                "router_switch_pipeline_bugs",
                hypothesis=active_h,
                pipeline_bug_runs=recent_pipeline_bugs,
            )
            print(f"[router] {message}")
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": incremented_hyp_iters,
                "router_decision": "switch_hypothesis",
                "router_message": message,
                "completed_hypothesis_ids": new_completed,
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        if structural_scope_failures >= scope_fail_threshold:
            if active_h != _RECYCLE_HYPOTHESIS:
                next_h_id = _RECYCLE_HYPOTHESIS
                message = (
                    f"Recent experiments show {structural_scope_failures} structural scope failures "
                    f"(threshold={scope_fail_threshold}); "
                    f"forcing {_RECYCLE_HYPOTHESIS} to rebuild the symbol universe before more parameter tuning."
                )
                log_event = "router_force_h8_rebuild"
            else:
                next_h_id = ""
                message = (
                    f"{_RECYCLE_HYPOTHESIS} accumulated {structural_scope_failures} scope failures "
                    f"(threshold={scope_fail_threshold}); "
                    f"universe rebuild did not resolve the issue — switching to next hypothesis."
                )
                log_event = "router_h8_scope_deadlock"
            new_completed = completed
            if active_h not in completed:
                new_completed = completed + [active_h]
            _log.info(log_event, hypothesis=active_h, structural_scope_failures=structural_scope_failures)
            print(f"[router] {message}")
            return {
                **state,
                "iteration": effective_iteration,
                "hypothesis_iterations": incremented_hyp_iters,
                "router_decision": "switch_hypothesis",
                "router_message": message,
                "next_hypothesis_id": next_h_id,
                "completed_hypothesis_ids": new_completed,
                "code_change_attempts": new_code_attempts,
                "edge_pairs": edge_pairs,
                "portfolio_candidate": portfolio_candidate,
                "approved_experiment_ids": approved_experiment_ids,
            }

        # Pure Python routing: iterate until per-hypothesis limit, then switch.
        if force_switch:
            decision = "switch_hypothesis"
            message = f"Hypothesis {active_h} exhausted after {h_iter_count} iterations."
        else:
            decision = "next_iteration"
            message = ""

        new_completed = completed
        if decision == "switch_hypothesis" and active_h not in completed:
            new_completed = completed + [active_h]

        if edge_pairs:
            ep_summary = ", ".join(
                f"{ep['pair_id']}(n={ep['appearances']},win={ep['avg_win_rate']:.0%})"
                for ep in edge_pairs[:3]
            )
            _log.info("router_edge_pairs_detected", pairs=ep_summary)
            print(f"[router] edge pairs detected: {ep_summary}")

        _log.info(
            "router_decision",
            decision=decision,
            iteration=effective_iteration,
            hypothesis=active_h,
            h_iter_count=h_iter_count,
            verdict=verdict,
        )
        print(f"[router] iter={effective_iteration}/{max_iter}  hypothesis={active_h}  decision={decision}")
        if message:
            print(f"[router] {message[:120]}")

        return {
            **state,
            "iteration": effective_iteration,
            "hypothesis_iterations": incremented_hyp_iters,
            "router_decision": decision,
            "router_message": message,
            "next_hypothesis_id": "",
            "completed_hypothesis_ids": new_completed,
            "code_change_attempts": new_code_attempts,
            "edge_pairs": edge_pairs,
            "portfolio_candidate": portfolio_candidate,
            "approved_experiment_ids": approved_experiment_ids,
        }
