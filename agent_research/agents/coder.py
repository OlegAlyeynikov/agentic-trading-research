"""
Coder agent — writes a standalone workspace Python script for a code hypothesis.

The script must run inside agent_research/workspace/<experiment_id>/ and output JSON.
It must never modify project source files.
"""
import json
import re
from pathlib import Path

from agent_research.state import ResearchState
from agent_research._llm import load_prompt, call_llm_text, get_coder_fallback_script
from agent_research.memory import ExperimentStore
from agent_research.logging_config import get_logger, AgentTimer
from agent_research.code_runtime import normalize_script_chunks

_log = get_logger("coder")


MAX_CODE_CHUNKS = 2
MAX_REPAIR_NOTES_CHARS = 2500
MAX_CODER_RETRIES = 2

WORKSPACE_HELPERS_API_SUMMARY = """Available from workspace_helpers.py:
- load_runtime_inputs()
- filter_available_symbols(data_dir, symbols, timeframe)
- load_price_data(data_dir, symbols, timeframe, start_date)
- load_execution_price_data(data_dir, symbols, available_timeframes, start_date="")
- build_execution_bridge(signal_df, execution_df, signal_timeframe, execution_timeframe)
- build_next_bar_execution_frame(df)
- load_funding_rates(data_dir, symbol)
- compute_peer_funding_zscore(data_dir, symbols, lookback_payments=90)
- generate_funding_divergence_signals(...)
- run_funding_divergence_strategy(inp, ...)
- candles_per_day(timeframe)
- rolling_zscore(series, window)
- generate_single_asset_signals(...)
- generate_pair_signals(...)
- apply_position_limit(df, max_open_positions)
- write_signals_csv(df, path)
- emit_success(path, metrics=..., notes=...)
- emit_blocked(notes, missing_requirements=[...])

Execution policy:
- Compute signals on the strategy timeframe.
- Execute on 5m-or-finer data using load_execution_price_data + build_execution_bridge.
- Pass execution_timestamps, execution_prices, stop_high, stop_low,
  signal_close_timestamps, execution_timeframe, reduced_fidelity to
  generate_single_asset_signals() or generate_pair_signals().

Funding/single-asset timestamp contract:
- For Funding Divergence hypotheses, prefer run_funding_divergence_strategy(inp, ...)
  instead of writing custom loops. It handles timestamp alignment, execution bridge,
  direction-specific entries, position limits, diagnostics, CSV writing, and JSON output.
  Use direction="short_high" for SHORT when funding_z is high; pass funding_z directly.
  Use entry_symbols=[...] to restrict traded symbols while keeping peer_symbols=[...] for
  cross-sectional z-score computation. Legacy symbols=[...] is accepted as an entry alias.
  Keep require_carry_sign=True unless the hypothesis explicitly tests price-only reversion.
- load_price_data() returns a DataFrame with a "timestamp" column and a RangeIndex.
- NEVER use prices[sym].index, price_df.index, or df.index as signal timestamps.
- Align funding to prices with price_df["timestamp"]:
    z = funding_z[sym].reindex(price_df["timestamp"], method="ffill").fillna(0.0)
- Pass timestamps=price_df["timestamp"] into generate_single_asset_signals().
- generate_single_asset_signals() emits action values: ENTER_LONG, ENTER_SHORT, EXIT, STOP.
  Do not filter for EXIT_LONG/STOP_LONG or EXIT_SHORT/STOP_SHORT.
- Never manually append force-close rows. They often lack execution_timestamp/execution_price
  and will be rejected; use max_holding_candles or canonical funding helper output.
- Do not hide signal-generation errors with broad `except Exception: continue`; emit_blocked()
  with diagnostics instead.
"""

# Parameter names the coder must not deviate from.
_ENFORCED_PARAMS: list[tuple[str, str]] = [
    ("stop_z",      r"\bstop_z\s*=\s*([0-9][0-9.]*)"),
    ("entry_z",     r"\bentry_z\s*=\s*([0-9][0-9.]*)"),
    ("exit_z",      r"\bexit_z\s*=\s*([0-9][0-9.]*)"),
    ("lookback",    r"\blookback\s*=\s*([0-9]+)(?:\s*#.*)?$"),
    ("p_threshold", r"\bp_threshold\s*=\s*([0-9][0-9.]*)"),
]


def _extract_direction_params(code_direction: str) -> dict[str, str]:
    """Parse numeric parameters that the researcher explicitly set in code_direction."""
    params: dict[str, str] = {}
    for name, _ in _ENFORCED_PARAMS:
        # Match "name=value" or "name = value" in prose
        m = re.search(rf"\b{name}\s*[=<>]\s*([0-9][0-9.]*)", code_direction, re.IGNORECASE)
        if m:
            params[name] = m.group(1)
    return params


def _enforce_parameters(script: str, params: dict[str, str]) -> tuple[str, list[str]]:
    """Replace wrong parameter assignments in the generated script. Returns (script, changes)."""
    changes: list[str] = []
    for name, pattern in _ENFORCED_PARAMS:
        if name not in params:
            continue
        expected = params[name]
        def _replace(m: re.Match, n=name, v=expected) -> str:
            actual = m.group(1)
            if actual != v:
                changes.append(f"{n}: {actual} → {v}")
            return m.group(0).replace(actual, v, 1)
        script = re.sub(pattern, _replace, script, flags=re.MULTILINE)
    return script, changes


def _format_params_block(params: dict[str, str]) -> str:
    if not params:
        return ""
    lines = ["MANDATORY PARAMETER VALUES — copy these verbatim into your script:"]
    for name, value in params.items():
        lines.append(f"    {name} = {value}")
    lines.append("Do NOT use any other value for these variables. The experiment is defined by these numbers.")
    return "\n".join(lines) + "\n\n"


def _join_chunks(chunks: list[str]) -> str:
    cleaned = [chunk.strip("\n") for chunk in chunks if chunk.strip()]
    return normalize_script_chunks("\n\n".join(cleaned))


def _fallback_plan_text(filename: str) -> str:
    return (
        f"filename: {filename}\n"
        "entrypoint: main\n"
        "risk_level: low\n"
        "total_chunks: 1\n"
        "change_summary: fallback plan after unusable model response\n"
    )


def _parse_plan_text(plan_text: str, default_filename: str) -> dict:
    text = (plan_text or "").strip()
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    filename = fields.get("filename") or default_filename
    if not filename.endswith(".py"):
        filename = default_filename

    entrypoint = fields.get("entrypoint") or "main"
    risk_level = (fields.get("risk_level") or "medium").lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "medium"

    total_chunks_raw = fields.get("total_chunks") or "1"
    total_chunks = 1
    m = re.search(r"\d+", total_chunks_raw)
    if m:
        total_chunks = max(1, min(int(m.group(0)), MAX_CODE_CHUNKS))

    change_summary = fields.get("change_summary") or "text plan parsed"
    return {
        "filename": filename,
        "entrypoint": entrypoint,
        "expected_output_schema": {
            "status": "success|blocked",
            "setups_csv": "string",
            "metrics": {},
            "notes": "string",
        },
        "change_summary": change_summary,
        "risk_level": risk_level,
        "total_chunks": total_chunks,
    }


def _is_coder_fallback(filename: str, contents: str, plan: dict) -> bool:
    fallback_script = get_coder_fallback_script().strip()
    summary = str(plan.get("change_summary", "")).lower()
    return (
        contents.strip() == fallback_script
        or "fallback" in summary
        or "unusable response" in summary
        or (
            filename == "workspace_strategy.py"
            and "could be generated" in contents
            and "missing_requirements" in contents
        )
    )


def _load_seed_script(store: ExperimentStore, hypothesis_id: str) -> dict:
    record = store.latest_code_record_for_hypothesis(hypothesis_id)
    if not record:
        return {}

    raw_path = str(record.get("code_script_path") or "").strip()
    script_path = Path(raw_path) if raw_path else None
    script_text = ""
    if script_path and script_path.exists() and script_path.is_file():
        script_text = script_path.read_text(encoding="utf-8")
    errors = list((record.get("validation_result") or {}).get("errors") or [])
    return {
        "experiment_id": record.get("experiment_id", ""),
        "script_name": record.get("code_change_proposal", {}).get("script_name")
        or record.get("code_script_filename")
        or (script_path.name if script_path else ""),
        "script_path": str(script_path) if script_path else "",
        "script_text": script_text,
        "validation_errors": errors,
        "reviewer_notes": record.get("reviewer_notes", ""),
    }


def _build_plan_message(
    state: ResearchState,
    proposal: dict,
    history: str,
    seed_script: dict,
    repair_instructions: str = "",
) -> str:
    seed_text = ""
    if seed_script:
        seed_text = (
            f"Previous candidate script to revise instead of rewriting from scratch:\n"
            f"- experiment_id: {seed_script.get('experiment_id', '')}\n"
            f"- script_name: {seed_script.get('script_name', '')}\n"
            f"- validation_errors: {json.dumps(seed_script.get('validation_errors', []), ensure_ascii=True)}\n"
            f"- reviewer_notes: {seed_script.get('reviewer_notes', '')[:300]}\n"
            "Previous script body (truncate and reuse what is good; fix only the broken parts):\n"
            "```python\n"
            f"{(seed_script.get('script_text') or '')[-5000:]}\n"
            "```\n\n"
        )
    repair_text = ""
    if repair_instructions.strip():
        repair_text = (
            "Repair objective for this revision:\n"
            f"{repair_instructions[:MAX_REPAIR_NOTES_CHARS]}\n\n"
            "Fix the listed issues with the smallest possible diff. "
            "Do not rewrite unrelated sections.\n\n"
        )
    code_direction = str(state.get("code_direction") or "")
    params = _extract_direction_params(code_direction)
    params_block = _format_params_block(params)
    direction_block = (
        f"RESEARCH DIRECTION (implement EXACTLY as written — every number is mandatory):\n"
        f"{code_direction}\n\n"
        f"{params_block}"
    ) if code_direction else ""
    return (
        "Stage: plan\n"
        "Return only a compact implementation plan for a standalone workspace script in raw text.\n"
        "Do not include any code yet.\n"
        "Choose the smallest workable script that tests the hypothesis.\n"
        "Prefer editing the previous candidate script if one is provided.\n"
        "Keep the final script under 140 lines and avoid nonessential comments. "
        "Set total_chunks=1 unless the script would still be too large or risky; "
        "use total_chunks=2 for cleaner split rather than truncating one huge JSON field.\n\n"
        "Return exactly these lines and nothing else:\n"
        "filename: <name>.py\n"
        "entrypoint: main\n"
        "risk_level: low|medium|high\n"
        "total_chunks: 1 or 2\n"
        "change_summary: <one sentence>\n\n"
        f"Active hypothesis: {state.get('active_hypothesis_id', '?')}\n\n"
        f"{direction_block}"
        f"Code change proposal:\n{json.dumps(proposal, indent=2)}\n\n"
        f"Current research scope:\n{json.dumps(state.get('current_scope') or {}, indent=2)}\n\n"
        "workspace_helpers.py already exists in the workspace — import from it, do NOT rewrite its functions.\n"
        f"{WORKSPACE_HELPERS_API_SUMMARY}\n"
        f"{repair_text}"
        f"{seed_text}"
        f"Recent experiment history:\n{history}\n"
    )


def _build_chunk_message(
    state: ResearchState,
    proposal: dict,
    plan: dict,
    chunk_index: int,
    total_chunks: int,
    assembled_prefix: str,
    seed_script: dict,
    repair_instructions: str = "",
) -> str:
    seed_text = ""
    if seed_script and chunk_index == 1:
        seed_text = (
            "Prefer adapting this previous candidate instead of rewriting everything:\n"
            "```python\n"
            f"{(seed_script.get('script_text') or '')[-5000:]}\n"
            "```\n\n"
        )
    repair_text = ""
    if repair_instructions.strip() and chunk_index == 1:
        repair_text = (
            "Specific issues to fix in this revision:\n"
            f"{repair_instructions[:MAX_REPAIR_NOTES_CHARS]}\n\n"
        )
    code_direction = str(state.get("code_direction") or "")
    params = _extract_direction_params(code_direction)
    params_block = _format_params_block(params)
    direction_block = (
        f"RESEARCH DIRECTION (every number below is mandatory — do not substitute defaults):\n"
        f"{code_direction}\n\n"
        f"{params_block}"
    ) if code_direction else ""
    return (
        "Stage: chunk\n"
        "Return raw Python code only for this chunk.\n"
        "Generate only the requested slice of the script.\n"
        "Keep the chunk concise and self-contained.\n"
        "Do not repeat previously generated code.\n\n"
        f"Active hypothesis: {state.get('active_hypothesis_id', '?')}\n"
        f"Filename: {plan.get('filename', proposal.get('script_name', 'workspace_strategy.py'))}\n"
        f"Chunk: {chunk_index} of {total_chunks}\n"
        f"Entrypoint: {plan.get('entrypoint', 'main')}\n"
        f"Change summary: {plan.get('change_summary', '')}\n\n"
        f"{direction_block}"
        f"Code change proposal:\n{json.dumps(proposal, indent=2)}\n\n"
        "workspace_helpers.py already exists in the same directory — import from it, do NOT rewrite its functions.\n"
        f"{WORKSPACE_HELPERS_API_SUMMARY}\n"
        f"{repair_text}"
        f"{seed_text}"
        "Already generated prefix:\n"
        "```python\n"
        f"{assembled_prefix[-2500:]}\n"
        "```\n"
        "Return raw Python code only. No JSON object. No markdown fences. "
        "Minimize comments and blank lines to reduce truncation risk.\n"
    )


def generate_script_with_coder(
    state: ResearchState,
    proposal: dict,
    *,
    history: str,
    seed_script: dict,
    exp_id: str,
    iteration: int,
    repair_instructions: str = "",
) -> tuple[str, str, dict]:
    system_prompt = load_prompt("coder")
    default_filename = proposal.get("script_name", "workspace_strategy.py")
    plan_text = call_llm_text(
        system_prompt,
        _build_plan_message(state, proposal, history, seed_script, repair_instructions),
        temperature=0.1,
        agent_name="coder",
        experiment_id=exp_id,
        iteration=iteration,
        max_tokens=900,
        fallback_text=_fallback_plan_text(default_filename),
    )
    plan = _parse_plan_text(plan_text, default_filename)
    _log.info(
        "coder_substage",
        exp_id=exp_id,
        substage="plan_text_ok",
        filename=plan.get("filename", default_filename),
        total_chunks=plan.get("total_chunks", 1),
    )
    filename = plan.get("filename", default_filename)
    total_chunks = min(int(plan.get("total_chunks", 1) or 1), MAX_CODE_CHUNKS)
    chunks: list[str] = []
    for chunk_index in range(1, total_chunks + 1):
        chunk_text = call_llm_text(
            system_prompt,
            _build_chunk_message(
                state,
                proposal,
                plan,
                chunk_index,
                total_chunks,
                "".join(chunks),
                seed_script,
                repair_instructions,
            ),
            temperature=0.1,
            agent_name="coder",
            experiment_id=exp_id,
            iteration=iteration,
            max_tokens=3500,
        )
        if not str(chunk_text).strip():
            _log.error(
                "coder_substage",
                exp_id=exp_id,
                substage="chunk_text_failed",
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
            chunks = [get_coder_fallback_script()]
            filename = "workspace_strategy.py"
            total_chunks = 1
            break
        _log.info(
            "coder_substage",
            exp_id=exp_id,
            substage="chunk_text_ok",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chars=len(str(chunk_text)),
        )
        chunks.append(str(chunk_text))

    contents = _join_chunks(chunks).strip()
    if not contents:
        contents = get_coder_fallback_script()
        filename = "workspace_strategy.py"
    return filename, contents, plan


def run_coder(state: ResearchState) -> ResearchState:
    exp_id = state.get("current_experiment_id", "?")
    proposal = state.get("code_change_proposal") or {}

    with AgentTimer(_log, "coder", exp_id, state.get("iteration", 0)):
        if state.get("dry_run"):
            _log.info("coder_dry_run", exp_id=exp_id)
            print(f"[coder] {exp_id}  [dry-run] no workspace script generated")
            return {
                **state,
                "code_script_filename": proposal.get("script_name", "workspace_strategy.py"),
                "code_script_contents": "",
            }

        store = ExperimentStore()
        history = store.summarize_for_llm(5)
        seed_script = _load_seed_script(store, state.get("active_hypothesis_id", ""))
        filename = proposal.get("script_name", "workspace_strategy.py")
        contents = ""
        plan: dict = {}
        total_chunks = 1
        repair_notes = ""

        for attempt in range(MAX_CODER_RETRIES + 1):
            filename, contents, plan = generate_script_with_coder(
                state,
                proposal,
                history=history,
                seed_script=seed_script,
                exp_id=exp_id,
                iteration=state.get("iteration", 0),
                repair_instructions=repair_notes,
            )
            total_chunks = min(int(plan.get("total_chunks", 1) or 1), MAX_CODE_CHUNKS)
            if not _is_coder_fallback(filename, contents, plan):
                break
            if attempt >= MAX_CODER_RETRIES:
                break
            repair_notes = (
                "Previous coder attempt collapsed into fallback because the model returned invalid or "
                "truncated raw text. Retry with a smaller script, fewer comments, and stricter formatting. "
                "Return only the requested raw-text plan or raw Python chunk. Do not use markdown fences. "
                "If needed, use total_chunks=2 to avoid truncation."
            )
            _log.warning(
                "coder_fallback_retry",
                exp_id=exp_id,
                attempt=attempt + 1,
                filename=filename,
            )
            print(f"[coder] retrying after unusable JSON response (attempt {attempt + 1}/{MAX_CODER_RETRIES})")

        # Enforce parameters from code_direction regardless of what the LLM wrote.
        code_direction = str(state.get("code_direction") or "")
        if code_direction and contents:
            params = _extract_direction_params(code_direction)
            contents, enforced = _enforce_parameters(contents, params)
            if enforced:
                _log.info("coder_params_enforced", exp_id=exp_id, corrections=enforced)
                print(f"[coder] parameter corrections applied: {enforced}")

        if _is_coder_fallback(filename, contents, plan):
            _log.error("coder_fallback_used", exp_id=exp_id, filename=filename)
            print("[coder] fallback blocked script used after repeated unusable model responses")

        _log.info(
            "coder_proposed",
            exp_id=exp_id,
            filename=filename,
            summary=plan.get("change_summary", "")[:120],
            risk=plan.get("risk_level", "?"),
            chunks=total_chunks,
        )
        print(
            f"[coder] {exp_id}  file={filename}  chunks={total_chunks}  risk={plan.get('risk_level', '?')}"
            f"  | {plan.get('change_summary', '')[:80]}"
        )

    return {
        **state,
        "code_script_filename": filename,
        "code_script_contents": contents,
    }
