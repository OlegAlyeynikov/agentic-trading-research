# Agentic Trading Research

Autonomous research and backtesting sandbox for crypto trading ideas.

This project lets an LLM-driven research loop propose hypotheses, generate
isolated Python signal scripts, validate those scripts, run backtests, and keep
structured diagnostics about what worked and what failed.

This is not a live trading system and not financial advice. Backtests can be
wrong, overfit, or invalid for live execution.

## What It Does

- Runs a multi-agent research loop over trading hypotheses.
- Generates standalone experiment scripts in isolated workspaces.
- Validates generated code for basic safety and look-ahead risks.
- Converts generated signals into setup CSVs.
- Backtests setup CSVs with fees, slippage, drawdown, duration, and funding
  accounting where funding data is available.
- Stores experiment records, reports, and diagnostics locally.
- Provides deterministic helper scripts for funding-divergence grid search and
  symbol selection.

## Repository Layout

```text
agent_research/     Multi-agent research loop, prompts, runner, memory, tools
backtest/           Reusable backtest engine
docs/               Strategy notes and execution assumptions
research_config.example.json
                    Safe example config for local setup
funding_download.py Optional funding-rate downloader helper
```

Generated artifacts are intentionally ignored:

```text
agent_research/workspace/
agent_research/reports/
agent_research/experiments/*.jsonl
agent_research/approved/
agent_research/param_search/
```

## Install

Use Python 3.11+.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r agent_research/requirements.txt
```

or install from `pyproject.toml`:

```bash
uv pip install --python .venv/bin/python -e .
```

## Configure

Create local files from the examples:

```bash
cp .env.example .env
cp research_config.example.json research_config.json
```

Then edit:

- `.env` for model/API settings.
- `research_config.json` for local candle and funding data paths.

Market data is not included. See `docs/DATA.md` for the expected local layout.

## Dry Run

Dry-run mode validates runner wiring without calling live models:

```bash
AGENT_RESEARCH_CONFIG=research_config.json \
AGENT_RESEARCH_GOAL="Find a robust signal idea for the current symbol set" \
make agent-research-dry-run
```

Direct CLI:

```bash
python -m agent_research.runner \
  --config research_config.json \
  --goal "Find a robust signal idea" \
  --dry-run
```

## Live Agent Run

Live runs require OpenRouter-compatible API settings and role models:

```bash
AGENT_RESEARCH_CONFIG=research_config.json \
AGENT_RESEARCH_GOAL="Find a robust signal idea for the current symbol set" \
OPENROUTER_API_KEY="..." \
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1" \
STRATEGIST_MODEL="..." \
RESEARCHER_MODEL="..." \
REVIEWER_MODEL="..." \
CODER_MODEL="..." \
make agent-research-run
```

Optional LangSmith tracing can be enabled with `LANGSMITH_*` variables in
`.env`.

## Deterministic Research Utilities

Funding-divergence grid/random search:

```bash
python -m agent_research.funding_grid_search \
  --config research_config.json \
  --output-dir agent_research/param_search/funding_divergence_4h \
  --timeframe 4h \
  --random-sample 100
```

Funding symbol selection:

```bash
python -m agent_research.funding_symbol_selection \
  --config research_config.json \
  --timeframe 4h \
  --execution-timeframe 5m \
  --output-dir agent_research/param_search/symbol_selection_4h
```

Approved portfolio replay:

```bash
python -m agent_research.approved_portfolio_backtest \
  --config research_config.json
```

## Verification

Basic import/compile check:

```bash
make check
```

Manual equivalent:

```bash
python -m py_compile \
  backtest/engine.py \
  agent_research/runner.py \
  agent_research/funding_grid_search.py \
  agent_research/funding_symbol_selection.py \
  agent_research/approved_portfolio_backtest.py
```

## Safety Notes

- Keep secrets in `.env`; never commit API keys.
- Keep datasets outside Git or under ignored `data/`.
- Treat all generated experiment outputs as local artifacts unless manually
  curated and scrubbed.
- Backtest metrics are research diagnostics, not live trading guarantees.

## License

MIT. See `LICENSE`.
