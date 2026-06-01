"""
Structured logging for agent_research.

Uses structlog for machine-readable JSON logs alongside human-readable console output.
Every agent call is logged with: agent, iteration, experiment_id, duration_ms, outcome.

Log levels:
  DEBUG   — LLM prompt/response content, internal decisions
  INFO    — agent entry/exit, key metrics, graph transitions
  WARNING — reviewer flags, data quality issues, retries
  ERROR   — exceptions, failed LLM calls, backtest errors
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import structlog


LOG_DIR = Path(__file__).parent / "logs"


def _format_duration_ms(duration_ms: Any) -> str:
    try:
        return f"{float(duration_ms) / 1000:.1f}s"
    except Exception:
        return f"{duration_ms}ms"


def _human_event_message(event_dict: dict[str, Any]) -> str:
    event = str(event_dict.get("event", "log"))
    agent = event_dict.get("agent")
    iteration = event_dict.get("iteration")
    exp_id = event_dict.get("experiment_id")

    if event == "agent_start":
        return f"{agent} started (iter={iteration}, exp={exp_id or '-'})"
    if event == "agent_done":
        return f"{agent} finished in {_format_duration_ms(event_dict.get('duration_ms'))}"
    if event == "agent_error":
        return f"{agent} failed after {_format_duration_ms(event_dict.get('duration_ms'))}: {event_dict.get('error', '')}"
    if event == "llm_call_done":
        keys = ", ".join(event_dict.get("response_keys", []) or [])
        return (
            f"{agent} LLM responded in {_format_duration_ms(event_dict.get('duration_ms'))}"
            f"{f' with keys [{keys}]' if keys else ''}"
        )
    if event == "llm_json_parse_retry":
        return f"{agent or 'llm'} returned invalid JSON, retrying"
    if event == "llm_json_parse_fallback":
        return f"{agent or 'llm'} returned unusable output twice, using safe fallback"
    if event == "planner_done":
        return f"planner proposed config={event_dict.get('proposed_config', {})} scope={event_dict.get('research_scope', {})}"
    if event == "researcher_proposed":
        return f"researcher proposed config={event_dict.get('proposed', {})} scope={event_dict.get('proposed_scope', {})}"
    if event == "executor_done":
        return (
            f"executor finished: trades={event_dict.get('total_trades')} "
            f"pnl={event_dict.get('total_pnl_pct')} win_rate={event_dict.get('win_rate')} "
            f"clean={event_dict.get('is_clean')}"
        )
    if event == "executor_blocked_missing_selected_symbols":
        return "executor blocked: selected_symbols are required before iterative execution"
    if event == "executor_blocked_symbol_filter":
        return "executor blocked: selected_symbols did not match available market data files"
    if event == "reviewer_verdict":
        return (
            f"reviewer verdict={event_dict.get('verdict')} "
            f"trades={event_dict.get('total_trades')} pnl={event_dict.get('total_pnl')}"
        )
    if event == "router_decision":
        return (
            f"router decision={event_dict.get('decision')} "
            f"(iter={event_dict.get('iteration')}, verdict={event_dict.get('verdict')})"
        )
    if event == "graph_routing":
        return f"graph routed to {event_dict.get('decision')} after verdict={event_dict.get('verdict')}"
    if event == "runner_start":
        return (
            f"runner started goal='{event_dict.get('goal')}' "
            f"max_iterations={event_dict.get('max_iterations')} dry_run={event_dict.get('dry_run')}"
        )
    if event == "research_complete":
        return (
            f"research complete after {event_dict.get('iterations')} iterations "
            f"best={event_dict.get('best_experiment_id')} pnl={event_dict.get('best_pnl')}"
        )
    if event == "human_input_received":
        return "human input received"
    if event == "human_node_received":
        return "human node resumed graph execution"
    if event == "graph_compiled":
        return f"graph compiled (project={event_dict.get('project')}, tracing={event_dict.get('tracing')})"
    if event == "runner_interrupted":
        return "runner interrupted by user"
    return event.replace("_", " ")


def _human_console_processor(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    event_dict = dict(event_dict)
    event_dict["event"] = _human_event_message(event_dict)
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    json_logs: bool = False,
) -> None:
    """
    Configure structlog + stdlib logging.

    Args:
        log_level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
        log_file:  if set, also write to this file (in addition to stderr)
        json_logs: if True, output JSON to stderr; if False, colored console output
    """
    LOG_DIR.mkdir(exist_ok=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            _human_console_processor,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [stderr_handler]

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    for h in handlers:
        root_logger.addHandler(h)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger."""
    return structlog.get_logger(name)


class AgentTimer:
    """Context manager that logs agent execution time."""

    def __init__(self, logger: Any, agent: str, experiment_id: str, iteration: int):
        self._log = logger
        self._agent = agent
        self._exp_id = experiment_id
        self._iter = iteration
        self._start = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        self._log.info(
            "agent_start",
            agent=self._agent,
            experiment_id=self._exp_id,
            iteration=self._iter,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((time.monotonic() - self._start) * 1000)
        if exc_type:
            self._log.error(
                "agent_error",
                agent=self._agent,
                experiment_id=self._exp_id,
                iteration=self._iter,
                duration_ms=duration_ms,
                error=str(exc_val),
            )
        else:
            self._log.info(
                "agent_done",
                agent=self._agent,
                experiment_id=self._exp_id,
                iteration=self._iter,
                duration_ms=duration_ms,
            )
        return False
