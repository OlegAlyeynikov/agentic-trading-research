"""
LangGraph StateGraph — autonomous research topology.

Flow:
  START → strategist → researcher → coder → code_executor → executor
  executor → reviewer → reporter → router
  router: next_iteration    → researcher
          switch_hypothesis → strategist
          done              → END

LLM agents (4): strategist, researcher, reviewer, coder
Pure Python: executor, code_executor, reporter, router
"""
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_research.state import ResearchState
from agent_research.agents.strategist import run_strategist
from agent_research.agents.researcher import run_researcher
from agent_research.agents.coder import run_coder
from agent_research.agents.code_executor import run_code_executor
from agent_research.agents.executor import run_executor
from agent_research.agents.reviewer import run_reviewer
from agent_research.agents.reporter import run_reporter
from agent_research.agents.router import run_router
from agent_research.logging_config import get_logger

_log = get_logger("graph")


def _route_decision(state: ResearchState) -> str:
    decision = state.get("router_decision", "done")
    _log.info(
        "graph_routing",
        decision=decision,
        iteration=state.get("iteration", 0),
        verdict=state.get("reviewer_verdict", ""),
        hypothesis=state.get("active_hypothesis_id", ""),
    )
    return decision


def build_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)

    workflow.add_node("strategist",    run_strategist)
    workflow.add_node("researcher",    run_researcher)
    workflow.add_node("coder",         run_coder)
    workflow.add_node("code_executor", run_code_executor)
    workflow.add_node("executor",      run_executor)
    workflow.add_node("reviewer",      run_reviewer)
    workflow.add_node("reporter",      run_reporter)
    workflow.add_node("router",        run_router)

    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist",    "researcher")
    workflow.add_edge("researcher",    "coder")
    workflow.add_edge("coder",         "code_executor")
    workflow.add_edge("code_executor", "executor")
    workflow.add_edge("executor",      "reviewer")
    workflow.add_edge("reviewer",      "reporter")
    workflow.add_edge("reporter",      "router")

    workflow.add_conditional_edges(
        "router",
        _route_decision,
        {
            "next_iteration":    "researcher",
            "switch_hypothesis": "strategist",
            "done":              END,
        },
    )

    return workflow


def compile_graph():
    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    _log.info(
        "graph_compiled",
        tracing=os.environ.get("LANGSMITH_TRACING", "false"),
        project=os.environ.get("LANGSMITH_PROJECT", ""),
    )
    return graph
