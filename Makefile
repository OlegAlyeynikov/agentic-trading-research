ifneq (,$(wildcard .env))
include .env
export
endif

UV := uv
VENV_PYTHON := .venv/bin/python
AGENT_RESEARCH_UV_DEPS := openai langgraph langchain-core pydantic pandas python-dotenv langsmith

.PHONY: agent-research-install agent-research-run agent-research-dry-run check

agent-research-install:
	$(UV) pip install --python $(VENV_PYTHON) $(AGENT_RESEARCH_UV_DEPS)

agent-research-dry-run:
	@if [ -z "$(AGENT_RESEARCH_CONFIG)" ]; then echo "AGENT_RESEARCH_CONFIG is required"; exit 1; fi
	@if [ -z "$(AGENT_RESEARCH_GOAL)" ]; then echo "AGENT_RESEARCH_GOAL is required"; exit 1; fi
	$(VENV_PYTHON) -m agent_research.runner --config "$(AGENT_RESEARCH_CONFIG)" --goal "$(AGENT_RESEARCH_GOAL)" --dry-run --max-iterations 1 --max-iterations-per-hypothesis 1

agent-research-run:
	@if [ -z "$(AGENT_RESEARCH_CONFIG)" ]; then echo "AGENT_RESEARCH_CONFIG is required"; exit 1; fi
	@if [ -z "$(AGENT_RESEARCH_GOAL)" ]; then echo "AGENT_RESEARCH_GOAL is required"; exit 1; fi
	@if [ -z "$(OPENROUTER_API_KEY)" ]; then echo "OPENROUTER_API_KEY is required"; exit 1; fi
	@if [ -z "$(OPENROUTER_BASE_URL)" ]; then echo "OPENROUTER_BASE_URL is required"; exit 1; fi
	@if [ -z "$(STRATEGIST_MODEL)" ]; then echo "STRATEGIST_MODEL is required"; exit 1; fi
	@if [ -z "$(RESEARCHER_MODEL)" ]; then echo "RESEARCHER_MODEL is required"; exit 1; fi
	@if [ -z "$(REVIEWER_MODEL)" ]; then echo "REVIEWER_MODEL is required"; exit 1; fi
	$(VENV_PYTHON) -m agent_research.runner --config "$(AGENT_RESEARCH_CONFIG)" --goal "$(AGENT_RESEARCH_GOAL)"

check:
	$(VENV_PYTHON) -m py_compile \
		backtest/engine.py \
		agent_research/runner.py \
		agent_research/funding_grid_search.py \
		agent_research/funding_symbol_selection.py \
		agent_research/approved_portfolio_backtest.py \
		funding_download.py
