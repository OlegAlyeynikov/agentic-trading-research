# Agent Research Platform - Workflow Reference

## Overview

`agent_research` is an autonomous experiment loop for historical trading research.

The current implementation is code-first:

- agents propose a hypothesis-driven experiment
- the coder writes a standalone workspace script
- the script is validated and sandbox-executed
- generated signals are backtested
- results are diagnosed and stored

Agents do not modify core repository source files during experiments.

## Current Graph Topology

```text
START
  -> strategist
  -> researcher
  -> coder
  -> code_executor
  -> executor
  -> reviewer
  -> reporter
  -> router

router:
  next_iteration    -> researcher
  switch_hypothesis -> strategist
  done              -> END
```

Implemented in `agent_research/graph.py`.

## Node Roles

### 1. Strategist

- loads and selects the active hypothesis
- reads recent experiment history
- may append new hypotheses to `hypotheses.md`
- sets the initial research direction and scope for the next cycle

### 2. Researcher

- proposes the next experiment direction inside the active hypothesis
- uses recent diagnoses and experiment memory
- avoids repeating previously tested directions for the same hypothesis
- creates a new `experiment_id`

### 3. Coder

- turns the research direction into a standalone Python script
- targets the workspace only
- may revise the latest failed script instead of rewriting from scratch

### 4. Code Executor

- creates `agent_research/workspace/<experiment_id>/`
- writes the script and helper file
- runs static checks, look-ahead checks, `py_compile`, and sandbox execution
- blocks duplicate scripts and duplicate generated CSV outputs

### 5. Executor

- reads the generated signals CSV
- runs `backtest.engine.run_backtest(...)`
- computes derived metrics and `research_score`
- persists the experiment record to `experiments/experiments.jsonl`

### 6. Reviewer

- compares the result against the goal contract
- returns `approve` or `reject`
- records structured diagnosis:
  - `root_cause`
  - failed and passing dimensions
  - diagnostic insight
  - suggested next direction

### 7. Reporter

- writes human-readable report logs for the experiment and session

### 8. Router

- decides whether to continue, switch hypothesis, recycle the universe, or stop
- stops on max iterations or when a stored approved run satisfies the full goal contract
- forces switches after repeated duplicate-blocked or coder-fallback failures

## Runtime State

The graph carries a shared `ResearchState` with:

- goal and goal contract
- current hypothesis and iteration counters
- current scope and experiment id
- generated script paths and sandbox outputs
- reviewer verdict and diagnosis
- experiment ids, reports, and best-run tracking

See `agent_research/state.py`.

## Storage and Artifacts

- experiment memory: `agent_research/experiments/experiments.jsonl`
- per-run reports: `agent_research/reports/report_<experiment_id>.log`
- session summary: `agent_research/reports/session_summary.log`
- sandbox workspace: `agent_research/workspace/<experiment_id>/`

## Launch

Direct CLI:

```bash
python -m agent_research.runner \
  --config research_config.json \
  --goal "Find a robust signal idea" \
  --max-iterations 10 \
  --dry-run
```

Make targets:

```bash
AGENT_RESEARCH_CONFIG=research_config.json \
AGENT_RESEARCH_GOAL="Find a robust signal idea" \
make agent-research-dry-run
```

```bash
AGENT_RESEARCH_CONFIG=research_config.json \
AGENT_RESEARCH_GOAL="Find a robust signal idea" \
OPENROUTER_API_KEY=... \
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 \
STRATEGIST_MODEL=... \
RESEARCHER_MODEL=... \
REVIEWER_MODEL=... \
make agent-research-run
```

## Environment Variables

Required for live runs:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
RESEARCHER_MODEL=...
REVIEWER_MODEL=...
STRATEGIST_MODEL=...
```

Optional:

```bash
CODER_MODEL=...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=trading-research
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
AGENT_RESEARCH_LOG_LEVEL=INFO
AGENT_RESEARCH_JSON_LOGS=false
```

## Important Note

Older documentation may mention a separate `config` experiment branch or a broader `stat_arb` pipeline. That is not the current execution path in this repository.

Today, every live experiment flows through generated code, sandbox validation, and the universal backtest engine.
