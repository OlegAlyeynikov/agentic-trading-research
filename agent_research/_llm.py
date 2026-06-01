"""
LLM helper — OpenRouter backend with LangSmith tracing.

LLM agents (role → required env var):
  strategist  → STRATEGIST_MODEL  (synthesis + hypothesis selection)
  researcher  → RESEARCHER_MODEL  (next experiment proposal)
  reviewer    → REVIEWER_MODEL    (result evaluation + diagnosis)
  coder       → CODER_MODEL       (script generation, code mode only)

Per-role optional overrides:
  {ROLE}_REASONING_EFFORT  — none | minimal | low | medium | high | xhigh
  {ROLE}_MAX_TOKENS        — integer

Global reasoning fallback:
  AGENT_RESEARCH_REASONING_EFFORT — applies to all roles without a specific override

Connection:
  OPENROUTER_API_KEY      — required
  OPENROUTER_BASE_URL     — required (https://openrouter.ai/api/v1)

LangSmith tracing (all required to enable):
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=...
  LANGSMITH_PROJECT=stat-arb-research
  LANGSMITH_ENDPOINT=https://api.smith.langchain.com

Error policy:
  Transport errors and unrecoverable JSON failures raise RuntimeError.
  The pipeline stops rather than recording fake experiment results.
  Exception: coder agent returns a blocked-script sentinel (valid outcome).
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from agent_research.logging_config import get_logger
from agent_research.schemas_llm import validate_llm_response, format_validation_error

PROMPTS_DIR = Path(__file__).parent / "prompts"

_log = get_logger("llm")

ROLE_MODEL_ENV_VARS = {
    "strategist": ("STRATEGIST_MODEL",),
    "researcher": ("RESEARCHER_MODEL",),
    "reviewer":   ("REVIEWER_MODEL",),
    "coder":      ("CODER_MODEL",),
}

ROLE_REASONING_ENV_VARS = {
    "strategist": ("STRATEGIST_REASONING_EFFORT", "AGENT_RESEARCH_REASONING_EFFORT"),
    "researcher": ("RESEARCHER_REASONING_EFFORT", "AGENT_RESEARCH_REASONING_EFFORT"),
    "reviewer":   ("REVIEWER_REASONING_EFFORT", "AGENT_RESEARCH_REASONING_EFFORT"),
    "coder":      ("CODER_REASONING_EFFORT", "AGENT_RESEARCH_REASONING_EFFORT"),
}

# Generous defaults: reasoning models emit <think> blocks before JSON,
# consuming tokens before the actual response.  2048 is too small.
ROLE_MAX_TOKENS_ENV_VARS: dict[str, tuple[str, ...]] = {
    "strategist": ("STRATEGIST_MAX_TOKENS",),
    "researcher": ("RESEARCHER_MAX_TOKENS",),
    "reviewer":   ("REVIEWER_MAX_TOKENS",),
    "coder":      ("CODER_MAX_TOKENS",),
}
ROLE_MAX_TOKENS_DEFAULTS: dict[str, int] = {
    "strategist": 8192,
    "researcher": 8192,
    "reviewer":   4096,
    "coder":      8192,
}

VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _get_role_model(agent_name: str | None) -> str | None:
    if not agent_name:
        return None

    for env_name in ROLE_MODEL_ENV_VARS.get(agent_name.lower(), ()):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def get_model(
    model: str | None = None,
    agent_name: str | None = None,
) -> str:
    resolved_model = model or _get_role_model(agent_name)
    if resolved_model:
        return resolved_model

    role_label = agent_name or "unknown"
    raise RuntimeError(
        f"Model is not configured for agent '{role_label}'. "
        f"Set the corresponding env var ({role_label.upper()}_MODEL)."
    )


def _get_role_max_tokens(agent_name: str | None, explicit: int | None) -> int:
    """Return max_tokens for the role.

    Priority: explicit arg > env var > role default > 2048.
    """
    if explicit is not None:
        return explicit
    role = (agent_name or "").lower()
    for env_name in ROLE_MAX_TOKENS_ENV_VARS.get(role, ()):
        raw = os.environ.get(env_name, "").strip()
        if raw.isdigit():
            return int(raw)
    return ROLE_MAX_TOKENS_DEFAULTS.get(role, 2048)


def _get_reasoning_effort(agent_name: str | None) -> str | None:
    if not agent_name:
        return None
    for env_name in ROLE_REASONING_ENV_VARS.get(agent_name.lower(), ()):
        value = (os.environ.get(env_name) or "").strip().lower()
        if value:
            if value in VALID_REASONING_EFFORTS:
                return value
            _log.warning(
                "llm_invalid_reasoning_effort",
                agent=agent_name,
                env_var=env_name,
                value=value,
            )
            return None
    return None


def _client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Export it: export OPENROUTER_API_KEY=..."
        )
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "OPENROUTER_BASE_URL is not set. "
            "Export it: export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1"
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def _langsmith_enabled() -> bool:
    return (
        os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        and bool(os.environ.get("LANGSMITH_API_KEY"))
    )


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    extracted = _extract_first_json_object(cleaned)
    if extracted:
        return json.loads(extracted)

    return json.loads(cleaned)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:python|py|json)?\s*\n?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _parse_and_validate_response(text: str, agent_name: str, schema_name: str | None = None) -> dict:
    payload = parse_json_response(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return validate_llm_response(agent_name, payload, schema_name=schema_name)


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None


def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    model: str | None = None,
    agent_name: str = "unknown",
    experiment_id: str = "",
    iteration: int = 0,
    response_schema: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Call the LLM via OpenRouter with optional LangSmith tracing.

    Extra kwargs (agent_name, experiment_id, iteration) are for observability only.
    """
    resolved_model = get_model(model=model, agent_name=agent_name)
    resolved_max_tokens = _get_role_max_tokens(agent_name, max_tokens)
    client = _client()
    t0 = time.monotonic()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    _log.debug(
        "llm_call_start",
        agent=agent_name,
        model=resolved_model,
        experiment_id=experiment_id,
        iteration=iteration,
        temperature=temperature,
        prompt_chars=len(system_prompt) + len(user_message),
    )

    if _langsmith_enabled():
        result = _call_with_langsmith(
            client, messages, resolved_model, temperature,
            agent_name, experiment_id, iteration, response_schema, resolved_max_tokens,
        )
    else:
        result = _call_direct(
            client, messages, resolved_model, temperature, agent_name, response_schema, resolved_max_tokens
        )

    duration_ms = int((time.monotonic() - t0) * 1000)
    _log.info(
        "llm_call_done",
        agent=agent_name,
        model=resolved_model,
        experiment_id=experiment_id,
        iteration=iteration,
        duration_ms=duration_ms,
        response_keys=list(result.keys()) if isinstance(result, dict) else [],
    )

    return result


def call_llm_text(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    model: str | None = None,
    agent_name: str = "unknown",
    experiment_id: str = "",
    iteration: int = 0,
    max_tokens: int | None = None,
    fallback_text: str | None = None,
) -> str:
    resolved_model = get_model(model=model, agent_name=agent_name)
    resolved_max_tokens = _get_role_max_tokens(agent_name, max_tokens)
    client = _client()
    t0 = time.monotonic()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    _log.debug(
        "llm_text_call_start",
        agent=agent_name,
        model=resolved_model,
        experiment_id=experiment_id,
        iteration=iteration,
        temperature=temperature,
        prompt_chars=len(system_prompt) + len(user_message),
    )

    result = _call_direct_text(
        client, messages, resolved_model, temperature, agent_name, resolved_max_tokens, fallback_text
    )

    duration_ms = int((time.monotonic() - t0) * 1000)
    _log.info(
        "llm_text_call_done",
        agent=agent_name,
        model=resolved_model,
        experiment_id=experiment_id,
        iteration=iteration,
        duration_ms=duration_ms,
        chars=len(result),
    )
    return result


def _call_direct(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float,
    agent_name: str,
    response_schema: str | None = None,
    max_tokens: int = 2048,
) -> dict:
    """Raw OpenRouter call without tracing."""
    reasoning_effort = _get_reasoning_effort(agent_name)

    def _request(msgs: list) -> str:
        request_kwargs = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort and reasoning_effort != "none":
            request_kwargs["extra_body"] = {
                "reasoning": {
                    "effort": reasoning_effort,
                }
            }
        resp = client.chat.completions.create(
            **request_kwargs,
        )
        return resp.choices[0].message.content or ""

    text = _request(messages)
    try:
        return _parse_and_validate_response(text, agent_name, schema_name=response_schema)
    except (json.JSONDecodeError, ValidationError, ValueError) as first_error:
        _log.warning("llm_json_parse_retry", agent=agent_name, schema=response_schema, raw_response=text[:1000])
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (
                "Your previous response was not valid for the required schema. "
                "Return exactly one complete JSON object only. "
                "Do not use markdown fences. "
                "Do not include commentary before or after the JSON. "
                f"Validation issue: {_format_retry_error(first_error)}"
            ),
        })
        text = _request(messages)
        try:
            return _parse_and_validate_response(text, agent_name, schema_name=response_schema)
        except (json.JSONDecodeError, ValidationError, ValueError):
            schema_key = (response_schema or agent_name or "").lower()
            _log.error(
                "llm_json_parse_failed",
                agent=agent_name,
                schema=response_schema,
                raw_response=text[:1000],
            )
            if agent_name.lower() == "coder" or schema_key.startswith("coder"):
                return _fallback_response(agent_name, response_schema=response_schema)
            raise RuntimeError(
                f"Agent '{agent_name}' returned invalid JSON after 2 attempts. "
                "Check llm_json_parse_failed log above."
            )


def _call_direct_text(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float,
    agent_name: str,
    max_tokens: int = 2048,
    fallback_text: str | None = None,
) -> str:
    reasoning_effort = _get_reasoning_effort(agent_name)

    def _request(msgs: list) -> str:
        request_kwargs = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort and reasoning_effort != "none":
            request_kwargs["extra_body"] = {
                "reasoning": {
                    "effort": reasoning_effort,
                }
            }
        resp = client.chat.completions.create(**request_kwargs)
        return resp.choices[0].message.content or ""

    text = strip_code_fences(_request(messages))
    if text.strip():
        return text

    _log.warning("llm_text_retry", agent=agent_name, raw_response=text[:1000])
    messages.append({"role": "assistant", "content": text})
    messages.append({
        "role": "user",
        "content": (
            "Your previous response was empty or malformed. "
            "Return raw Python code only. "
            "No markdown fences. No JSON. No commentary."
        ),
    })
    text = strip_code_fences(_request(messages))
    if text.strip():
        return text

    _log.error("llm_text_failed", agent=agent_name, raw_response=text[:1000])
    if fallback_text is not None:
        return fallback_text
    if agent_name.lower() == "coder":
        return get_coder_fallback_script()
    raise RuntimeError(
        f"Agent '{agent_name}' returned empty text after 2 attempts."
    )


def _format_retry_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return format_validation_error(error)
    return str(error)


def get_coder_fallback_script() -> str:
    return """import json
import os


def main() -> None:
    payload = {
        "status": "blocked",
        "setups_csv": "",
        "metrics": {},
        "notes": (
            "Coder model returned an unusable response, so no standalone script "
            "could be generated for this hypothesis in this iteration."
        ),
        "missing_requirements": [
            "Valid coder LLM output containing executable Python script contents"
        ],
    }
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
"""


def _fallback_response(agent_name: str, response_schema: str | None = None) -> dict:
    """Only valid for coder: a blocked script is a legitimate outcome (no signal found)."""
    schema_key = (response_schema or agent_name or "").lower()
    if schema_key == "coder_plan":
        return {
            "filename": "workspace_strategy.py",
            "entrypoint": "main",
            "expected_output_schema": {
                "status": "blocked",
                "setups_csv": "",
                "metrics": {},
                "notes": "string",
            },
            "change_summary": "LLM returned an unusable response; using a blocked-report fallback script instead.",
            "risk_level": "low",
            "total_chunks": 1,
        }
    if schema_key == "coder_chunk":
        return {
            "chunk_index": 1,
            "total_chunks": 1,
            "content_chunk": get_coder_fallback_script(),
        }
    if agent_name.lower() == "coder":
        return {
            "filename": "workspace_strategy.py",
            "contents": get_coder_fallback_script(),
            "entrypoint": "main",
            "expected_output_schema": {
                "status": "blocked",
                "setups_csv": "",
                "metrics": {},
                "notes": "string",
            },
            "change_summary": "LLM returned unusable response; using a blocked-report fallback script instead.",
            "risk_level": "low",
        }
    raise RuntimeError(
        f"_fallback_response called for non-coder agent '{agent_name}' — this is a bug. "
        "Transport errors and JSON failures should have raised before reaching here."
    )


def _call_with_langsmith(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float,
    agent_name: str,
    experiment_id: str,
    iteration: int,
    response_schema: str | None,
    max_tokens: int,
) -> dict:
    """OpenRouter call wrapped in a LangSmith run for tracing."""
    try:
        from langsmith import traceable
    except ImportError:
        _log.warning("langsmith_not_installed", msg="pip install langsmith")
        return _call_direct(
            client, messages, model, temperature, agent_name, response_schema, max_tokens
        )

    @traceable(
        name=f"{agent_name}_llm_call",
        run_type="llm",
        tags=["stat-arb", agent_name],
        metadata={
            "experiment_id": experiment_id,
            "iteration": iteration,
            "model": model,
        },
    )
    def _traced_call(msgs: list) -> dict:
        return _call_direct(
            client, msgs, model, temperature, agent_name, response_schema, max_tokens
        )

    return _traced_call(messages)
