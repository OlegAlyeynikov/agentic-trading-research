# Data Layout

This repository does not include market data. Bring your own candle and funding
rate datasets, then point `research_config.json` at them.

Recommended local layout:

```text
data/
  1m/
    BTCUSDT_...pkl
    ETHUSDT_...pkl
  5m/
  15m/
  1h/
  4h/
  new_data/
    funding_rates/
      BTCUSDT.csv
      ETHUSDT.csv
```

The exact filename parser is intentionally permissive in the research helpers,
but each candle file must contain enough OHLCV columns for the strategy being
tested. Funding-rate files are expected to be local historical funding snapshots
for perpetual futures.

Do not commit downloaded exchange data, generated setup CSVs, experiment logs,
or workspace outputs. They are intentionally ignored by `.gitignore`.
