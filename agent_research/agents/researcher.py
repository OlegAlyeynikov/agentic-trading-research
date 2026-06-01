"""
Researcher agent — proposes next code_direction within the active hypothesis.

Uses experiment history + hypothesis text to decide what to research next.
Produces code_direction (natural language instruction) for the coder agent.
Never re-proposes a direction already tested.
"""
import json
from agent_research.state import ResearchState
from agent_research._llm import load_prompt, call_llm
from agent_research.config_utils import load_full_config
from agent_research.memory import ExperimentStore, generate_experiment_id
from agent_research.logging_config import get_logger, AgentTimer

_log = get_logger("researcher")

_MOCK_DIRECTIONS = [
    "Implement stat-arb on 1h with entry_z=1.8, stop_z=3.5, exit_z=0.3, OLS hedge ratio, lookback=720",
    "Try Kalman filter hedge ratio instead of OLS, entry_z=2.0, exit_z=0.5, timeframe=4h",
    "Test log price transform with tighter coint threshold p<0.005, entry_z=1.5",
    "Use ATR regime filter — only enter when 14-bar ATR is below 20-day mean, entry_z=2.0",
    "Try momentum strategy on 4h: 20-day return z-score entry, stop at 3.0",
]


def _direction_fingerprint(direction: str) -> str:
    return direction.strip().lower()


def _find_duplicate_direction(
    store: ExperimentStore,
    direction: str,
    scope: dict,
    active_hypothesis_id: str,
) -> str | None:
    fp = _direction_fingerprint(direction)
    selected = frozenset((scope.get("selected_symbols") or []))
    for rec in store._iter_all():
        if rec.get("active_hypothesis_id") != active_hypothesis_id:
            continue
        prior_dir = _direction_fingerprint(rec.get("code_direction") or "")
        if prior_dir != fp:
            continue
        prior_selected = frozenset((rec.get("research_scope") or {}).get("selected_symbols") or [])
        if prior_selected == selected:
            return rec.get("experiment_id")
    return None


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _normalize_symbols(symbols: list[str] | None) -> list[str]:
    return [str(s).strip() for s in (symbols or []) if str(s).strip()]


def _contains_disallowed_search(direction: str) -> bool:
    text = direction.lower()
    markers = (
        "grid search",
        "random search",
        "parameter sweep",
        "brute force",
        "all combinations",
        "all symbol",
        "itertools.combinations",
        "itertools.product",
        "top 5",
        "top 10",
    )
    return any(marker in text for marker in markers)


def run_researcher(state: ResearchState) -> ResearchState:
    iteration = state.get("iteration", 0)
    experiment_id = generate_experiment_id()
    active_h = state.get("active_hypothesis_id", "H1")
    full_config = load_full_config(state["config_path"])
    hyp_iters = dict(state.get("hypothesis_iterations") or {})
    hyp_iters[active_h] = hyp_iters.get(active_h, 0) + 1

    with AgentTimer(_log, "researcher", experiment_id, iteration):
        if state.get("dry_run"):
            idx = iteration % len(_MOCK_DIRECTIONS)
            mock_dir = _MOCK_DIRECTIONS[idx]
            _log.info("researcher_dry_run", iteration=iteration + 1, direction=mock_dir[:60])
            print(f"[researcher] iter={iteration+1}  hypothesis={active_h}  [dry-run]")
            return {
                **state,
                "iteration": iteration + 1,
                "code_direction": mock_dir,
                "current_scope": dict(state.get("current_scope") or {}),
                "current_experiment_id": experiment_id,
                "hypothesis_iterations": hyp_iters,
                "code_change_proposal": {},
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

        store = ExperimentStore()
        memory_summary = store.summarize_for_llm(15)
        hypotheses_text = state.get("hypotheses_text", "")
        active_h_section = _extract_hypothesis_section(hypotheses_text, active_h)

        tested_directions = [
            rec.get("code_direction", "")
            for rec in store.list_records_for_hypothesis(active_h, limit=20)
            if rec.get("code_direction")
        ]
        tested_str = (
            "\n".join(f"  {d[:120]}" for d in tested_directions[:20])
            if tested_directions else "  none"
        )

        code_budget = state.get("code_change_budget", 100)
        code_attempts = state.get("code_change_attempts", 0)

        diagnoses_summary = store.get_diagnoses_for_hypothesis(active_h, n=5)

        system_prompt = load_prompt("researcher")
        user_message = (
            f"Research goal: {state['goal']}\n\n"
            f"Active hypothesis: {active_h}\n\n"
            f"Hypothesis details:\n{active_h_section}\n\n"
            f"Iteration: {iteration + 1} / {state['max_iterations']}\n"
            f"Iterations on this hypothesis: {hyp_iters.get(active_h, 0)}\n"
            f"Code experiments budget: {code_budget - code_attempts} remaining\n\n"
            f"Available timeframes: {full_config.get('available_timeframes', ['1h'])}\n"
            f"Available symbols: {full_config.get('symbols', [])}\n\n"
            f"Reviewer diagnoses for {active_h} — USE THESE to reason about root causes:\n{diagnoses_summary}\n\n"
            f"Directions already tested for {active_h} — MUST NOT repeat exactly:\n{tested_str}\n\n"
            f"Strategist direction (starting point):\n{state.get('code_direction', '')}\n\n"
            f"Current research scope:\n{json.dumps(state.get('current_scope', {}), indent=2)}\n\n"
            f"All experiment history:\n{memory_summary}\n"
        )

        duplicate_of = ""
        result = {}
        code_direction = ""
        merged_scope: dict = {}
        code_change_proposal: dict = {}

        for attempt in range(2):
            result = call_llm(
                system_prompt, user_message, temperature=0.3,
                agent_name="researcher", experiment_id=experiment_id, iteration=iteration,
            )
            code_direction = (result.get("code_direction") or "").strip()
            proposed_scope = _as_dict(result.get("research_scope"))
            merged_scope = {**(state.get("current_scope") or {}), **proposed_scope}

            if _contains_disallowed_search(code_direction):
                _log.warning(
                    "researcher_disallowed_search_direction",
                    direction=code_direction[:160],
                    retry=attempt + 1,
                )
                if attempt == 0:
                    user_message += (
                        "\n\nThe proposed direction asks the coder to run a grid/random/search sweep. "
                        "That is forbidden in agent workspace scripts. Propose exactly ONE concrete "
                        "configuration instead. If a sweep is needed, mention "
                        "`python -m agent_research.funding_grid_search` in rationale only."
                    )
                    continue

            raw_proposal = result.get("code_change_proposal") or {}
            if isinstance(raw_proposal, dict) and raw_proposal.get("script_name"):
                code_change_proposal = raw_proposal
            else:
                code_change_proposal = {}

            duplicate_of = _find_duplicate_direction(
                store, code_direction, merged_scope, active_h,
            )
            if not duplicate_of:
                break

            _log.warning(
                "researcher_duplicate_proposed",
                duplicate_of=duplicate_of,
                direction=code_direction[:80],
                retry=attempt + 1,
            )
            if attempt == 0:
                user_message += (
                    f"\n\nThe proposed direction is too similar to experiment {duplicate_of}. "
                    "Propose a meaningfully different research direction now."
                )

        if duplicate_of:
            print(f"[researcher] WARNING: duplicate direction blocked ({duplicate_of})")

        _log.info(
            "researcher_proposed",
            iteration=iteration + 1,
            hypothesis=active_h,
            direction=code_direction[:100],
            scope=merged_scope,
            duplicate_of=duplicate_of,
            rationale=result.get("rationale", "")[:100],
        )
        print(
            f"[researcher] iter={iteration+1}  hypothesis={active_h}  "
            f"| {code_direction[:100]}"
        )

        return {
            **state,
            "iteration": iteration + 1,
            "code_direction": code_direction,
            "current_scope": merged_scope,
            "current_experiment_id": experiment_id,
            "hypothesis_iterations": hyp_iters,
            "code_change_proposal": code_change_proposal,
            "strategy_flags": dict(code_change_proposal.get("strategy_flags") or {}),
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
            "duplicate_of_experiment_id": duplicate_of,
        }


def _extract_hypothesis_section(hypotheses_text: str, hypothesis_id: str) -> str:
    lines = hypotheses_text.split("\n")
    in_section = False
    section_lines = []
    for line in lines:
        if line.startswith(f"## {hypothesis_id} "):
            in_section = True
        elif in_section and line.startswith("## H") and not line.startswith(f"## {hypothesis_id} "):
            break
        if in_section:
            section_lines.append(line)
    return "\n".join(section_lines) if section_lines else f"Hypothesis {hypothesis_id} not found."
