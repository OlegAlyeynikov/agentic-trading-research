"""
Strategist agent — synthesizes evidence and selects the next hypothesis.

Called at START and on every switch_hypothesis.
Part 1: cross-experiment synthesis, optionally writes new hypotheses to hypotheses.md.
Part 2: selects next hypothesis and provides code_direction for the researcher.
"""
import re
import json
from pathlib import Path

from agent_research.state import ResearchState
from agent_research._llm import load_prompt, call_llm
from agent_research.config_utils import load_full_config
from agent_research.memory import ExperimentStore
from agent_research.logging_config import get_logger, AgentTimer
from agent_research.agents.router import _find_edge_pairs, _build_portfolio_candidate

_log = get_logger("strategist")
_HYPOTHESES_PATH = Path(__file__).parent.parent / "hypotheses.md"
_RETIRED_AGENT_HYPOTHESES = {"H4", "H5"}

_MOCK_CODE_DIRECTION = "Implement stat-arb pairs strategy on 1h with entry_z=2.0, stop_z=3.5, exit_z=0.5"


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def run_strategist(state: ResearchState) -> ResearchState:
    exp_id = state.get("current_experiment_id", "init")
    full_config = load_full_config(state["config_path"])

    hypotheses_text = state.get("hypotheses_text") or _read_hypotheses()
    completed = list(state.get("completed_hypothesis_ids") or [])
    hypothesis_iterations = dict(state.get("hypothesis_iterations") or {})
    router_next_h = state.get("next_hypothesis_id", "")

    with AgentTimer(_log, "strategist", exp_id, state.get("iteration", 0)):
        if state.get("dry_run"):
            dry_h = router_next_h if (router_next_h and router_next_h not in completed) else "H1"
            _log.info("strategist_dry_run", hypothesis=dry_h)
            print(f"[strategist] [dry-run] hypothesis={dry_h}")
            return _build_return(
                state, _MOCK_CODE_DIRECTION, {}, dry_h,
                hypotheses_text, completed, hypothesis_iterations,
                research_memo="", new_h_ids=[], edge_pairs=[], portfolio_candidate={},
            )

        store = ExperimentStore()
        all_records = store.list_recent(50)
        has_experiments = len(all_records) >= 3
        edge_pairs = _find_edge_pairs(store)
        portfolio_candidate = _build_portfolio_candidate(edge_pairs)

        diagnoses_block = _build_diagnoses_block(all_records) if has_experiments else "No experiments yet — first run."
        existing_h_ids = _extract_hypothesis_ids(hypotheses_text)

        router_hint_line = (
            f"Router's suggested next hypothesis: {router_next_h} (REQUIRED — use this unless already completed)\n"
            if router_next_h else
            "Router's suggested next hypothesis: none — select lowest-numbered incomplete\n"
        )

        edge_pairs_block = _build_edge_pairs_block(edge_pairs)
        portfolio_block = _build_portfolio_candidate_block(portfolio_candidate, edge_pairs)

        system_prompt = load_prompt("strategist")
        user_message = (
            f"Research goal: {state['goal']}\n\n"
            f"Total experiments so far: {len(all_records)}\n"
            f"Existing hypothesis IDs: {', '.join(sorted(existing_h_ids))}\n"
            f"Already completed hypotheses: {completed or 'none'}\n"
            f"{router_hint_line}"
            f"Currently completing hypothesis: {state.get('active_hypothesis_id') or 'none (start)'}\n\n"
            f"{edge_pairs_block}"
            f"{portfolio_block}"
            f"Experiment diagnoses:\n{diagnoses_block}\n\n"
            f"Available hypotheses:\n{hypotheses_text}\n\n"
            f"Available timeframes: {full_config.get('available_timeframes', ['1h'])}\n"
            f"Available symbols: {full_config.get('symbols', [])}\n\n"
            f"Current research scope:\n{json.dumps(state.get('current_scope') or {}, indent=2)}\n"
        )

        result = call_llm(
            system_prompt, user_message, temperature=0.3,
            agent_name="strategist", experiment_id=exp_id, iteration=state.get("iteration", 0),
        )

        active_h = result.get("active_hypothesis_id") or "H1"
        if router_next_h and router_next_h not in completed:
            active_h = router_next_h
        if active_h in _RETIRED_AGENT_HYPOTHESES:
            active_h = next(
                (h for h in sorted(existing_h_ids) if h not in completed and h not in _RETIRED_AGENT_HYPOTHESES),
                "H1",
            )

        code_direction = result.get("code_direction") or ""
        research_scope = _as_dict(result.get("research_scope"))

        if active_h == "H8":
            research_scope = {}

        research_memo = result.get("research_memo", "")
        new_h_proposals = result.get("new_hypotheses") or []

        new_h_ids: list[str] = []
        if new_h_proposals and has_experiments:
            next_id = _next_hypothesis_id(existing_h_ids)
            proposal = new_h_proposals[0]
            try:
                h_block = _format_hypothesis_block(next_id, proposal)
                with _HYPOTHESES_PATH.open("a", encoding="utf-8") as f:
                    f.write(f"\n---\n\n{h_block}\n")
                new_h_ids.append(next_id)
                hypotheses_text = _HYPOTHESES_PATH.read_text(encoding="utf-8")
                _log.info("strategist_hypothesis_added", id=next_id, title=proposal.get("title", "?"))
                print(f"[strategist] added hypothesis {next_id}: {proposal.get('title', '?')}")
            except Exception as exc:
                _log.error("strategist_write_failed", error=str(exc))

        _log.info(
            "strategist_done",
            active_hypothesis=active_h,
            new_hypotheses=new_h_ids,
            rationale=result.get("rationale", "")[:120],
        )
        print(f"[strategist] hypothesis={active_h}  new_h={new_h_ids}  | {result.get('rationale', '')[:80]}")
        if research_memo:
            print(f"[strategist] memo: {research_memo[:100]}")

        return _build_return(
            state, code_direction, research_scope, active_h,
            hypotheses_text, completed, hypothesis_iterations,
            research_memo=research_memo, new_h_ids=new_h_ids,
            edge_pairs=edge_pairs, portfolio_candidate=portfolio_candidate,
        )


def _build_return(
    state: ResearchState,
    code_direction: str,
    research_scope: dict,
    active_h: str,
    hypotheses_text: str,
    completed: list,
    hypothesis_iterations: dict,
    *,
    research_memo: str,
    new_h_ids: list,
    edge_pairs: list,
    portfolio_candidate: dict,
) -> ResearchState:
    return {
        **state,
        "code_direction": code_direction,
        "current_scope": research_scope,
        "active_hypothesis_id": active_h,
        "next_hypothesis_id": "",
        "completed_hypothesis_ids": completed,
        "hypothesis_iterations": hypothesis_iterations,
        "hypotheses_text": hypotheses_text,
        "analyst_memo": research_memo,
        "analyst_new_hypotheses": new_h_ids,
        "edge_pairs": edge_pairs,
        "portfolio_candidate": portfolio_candidate,
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
        "duplicate_of_experiment_id": "",
    }


def _build_portfolio_candidate_block(candidate: dict, edge_pairs: list) -> str:
    if not candidate or not candidate.get("pairs"):
        return ""
    pairs_lines = []
    ep_by_id = {ep["pair_id"]: ep for ep in edge_pairs}
    for pid in candidate["pairs"]:
        ep = ep_by_id.get(pid, {})
        pairs_lines.append(
            f"    {pid:<28} appearances={ep.get('appearances','?')}  "
            f"avg_win={ep.get('avg_win_rate',0):.0%}  "
            f"avg_trades={ep.get('avg_trades',0):.0f}"
        )
    lines = [
        "PORTFOLIO CANDIDATE — confirmed edge pairs, not yet tested together on 4h with strict quality filter:",
        "\n".join(pairs_lines),
        f"  Universe symbols : {', '.join(candidate['symbols'])}",
        f"  Expected trades  : ~{candidate['expected_trades']} ({candidate['note']})",
        "  ACTION: propose a 4h experiment with this full universe. Use entry_z=1.5.",
        "  Per-pair quality gate: if any pair fails cointegration p<0.05, skip it silently.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _build_edge_pairs_block(edge_pairs: list) -> str:
    if not edge_pairs:
        return ""
    lines = [
        "KNOWN EDGE PAIRS (consistent profitability across multiple experiments):",
        "  These pairs have demonstrated real edge. EXPLOIT them before switching hypothesis.",
    ]
    for ep in edge_pairs[:5]:
        lines.append(
            f"  {ep['pair_id']:<28} appearances={ep['appearances']}  "
            f"avg_win={ep['avg_win_rate']:.0%}  avg_pnl={ep['avg_pnl_pct']:+.1f}%  "
            f"avg_trades={ep['avg_trades']:.0f}  avg_sharpe={ep['avg_sharpe']:.2f}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _read_hypotheses() -> str:
    return _HYPOTHESES_PATH.read_text(encoding="utf-8") if _HYPOTHESES_PATH.exists() else "No hypotheses file found."


def _build_diagnoses_block(records: list[dict]) -> str:
    lines = []
    for rec in records:
        exp_id = rec.get("experiment_id", "?")
        h_id = rec.get("active_hypothesis_id") or "?"
        verdict = rec.get("reviewer_verdict", "?")
        d = rec.get("reviewer_diagnosis") or {}
        metrics = rec.get("metrics") or {}

        root_cause = d.get("root_cause") or "none"
        insight = (d.get("diagnostic_insight") or "").strip()
        suggested = (d.get("suggested_direction") or "").strip()
        direction = (rec.get("code_direction") or "").strip()

        line = (
            f"  {exp_id}  H={h_id}  verdict={verdict}"
            f"  trades={metrics.get('total_trades', '?')}  sharpe={metrics.get('sharpe_proxy', 0):.2f}"
            f"  root_cause={root_cause}"
        )
        if direction:
            line += f"\n    direction: {direction[:100]}"
        if insight:
            line += f"\n    insight: {insight[:120]}"
        if suggested:
            line += f"\n    suggested: {suggested[:100]}"
        lines.append(line)
    return "\n".join(lines) if lines else "No experiments yet."


def _extract_hypothesis_ids(hypotheses_text: str) -> set[str]:
    return set(re.findall(r"^## (H\d+)", hypotheses_text, re.MULTILINE))


def _next_hypothesis_id(existing_ids: set[str]) -> str:
    nums = [int(re.search(r"\d+", h).group()) for h in existing_ids if re.search(r"\d+", h)]
    return f"H{max(nums, default=11) + 1}"


def _format_hypothesis_block(h_id: str, proposal: dict) -> str:
    title = proposal.get("title", "Untitled")
    questions_md = "\n".join(f"- {q}" for q in (proposal.get("research_questions") or []))
    return (
        f"## {h_id} — {title} [STRATEGIST-GENERATED]\n"
        f"**Claim:** {proposal.get('claim', '')}\n\n"
        f"**Derived from pattern:** {proposal.get('rationale', '')}\n\n"
        f"**Research questions:**\n{questions_md}\n\n"
        f"**Success criteria:** {proposal.get('success_criteria', '')}\n"
    )
