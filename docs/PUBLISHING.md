# Public Publishing Checklist

Use this checklist before pushing the repository to a public remote.

1. Confirm local artifacts are ignored:

```bash
git status --short --ignored
```

2. Scan public files for secrets and private paths:

```bash
rg -n -i "api[_-]?key|secret|token|password|bearer|sk-[A-Za-z0-9]|ls__[A-Za-z0-9]|/Users/|PycharmProjects" .
```

3. Verify core modules compile:

```bash
make check
```

4. Ensure only safe config is committed:

```text
commit:     research_config.example.json
do not:     research_config.json
do not:     .env
```

5. Do not commit:

```text
data/
draft/
agent_research/workspace/
agent_research/reports/
agent_research/approved/
agent_research/promising_runs/
agent_research/param_search/
agent_research/experiments/*.jsonl
```

6. If sharing results, manually curate a small scrubbed artifact in `examples/`
or `docs/` and explain that it is illustrative, not a live-trading result.
