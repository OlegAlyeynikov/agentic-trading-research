# Security

Do not commit secrets or private data.

Keep these values in a local `.env` only:

- `OPENROUTER_API_KEY`
- `LANGSMITH_API_KEY`
- model-provider API keys
- private dataset paths

Before publishing or opening a pull request, run:

```bash
rg -n -i "api[_-]?key|secret|token|password|bearer|sk-[A-Za-z0-9]|ls__[A-Za-z0-9]" .
git status --short --ignored
```

Generated artifacts, datasets, virtualenvs, IDE files, and experiment outputs are
ignored by `.gitignore`. If an artifact needs to be shared, move a scrubbed copy
to `examples/` or `docs/` and remove local paths, API traces, and private data.
