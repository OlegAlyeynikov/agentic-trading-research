# Backtest Execution Specification

## Goal

This document defines the required execution model for all research backtests.
The design is intentionally pessimistic and live-oriented. A strategy is not
valid unless its backtest respects these rules.

## Principles

1. Signals are computed on closed bars only.
2. Orders are executed on the next executable bar, never on the signal bar.
3. If a pessimistic and an optimistic interpretation are both possible, the
   backtest must choose the pessimistic one.
4. Funding, fees, and slippage are mandatory for perpetual strategies.
5. Execution timeframe is capped at `5m` for precision unless finer data exists.

## Inputs

- `signal_timeframe`: timeframe used to compute the strategy signal, e.g. `4h`.
- `execution_timeframe`: actual fill/stop evaluation timeframe.
- `peer_symbols`: universe used to normalize relative signals such as funding z-score.
- `entry_symbols`: symbols allowed to trade.
- `funding_rates_dir`: required for perp strategies.

## Timeframe Policy

Execution timeframe must be selected as follows:

- If the smallest available timeframe is below `5m`, use that smaller timeframe.
- Else if `5m` exists, use `5m`.
- Else use the smallest available timeframe above `5m` and mark the run as reduced-fidelity.

Examples:

- signal on `4h`, data available on `5m` -> execute on `5m`
- signal on `1h`, data available on `1m` -> execute on `1m`
- signal on `15m`, no `5m`, only `15m` -> execute on `15m`, reduced-fidelity flag

## Signal Alignment

If source data uses Binance kline open timestamps:

- `bar_open_time[t]` is the stored timestamp
- `bar_close_time[t] = bar_open_time[t] + bar_duration`
- signal is evaluated only after `bar_close_time[t]`
- first legal execution point is `bar_open_time[t+1]`

Therefore:

- `signal_bar = t`
- `execution_bar = t + 1`
- `entry_price = open[t+1]`
- `exit_price = open[t+1]` for normal signal exits

Using `close[t]` as the execution price for a decision made from bar `t` is forbidden.

## Entry Rules

- Entry conditions are evaluated on the closed signal bar.
- If the condition is true, generate a pending order.
- Execute at the next execution bar open.
- Apply entry slippage pessimistically.

Required recorded fields:

- `signal_bar_time`
- `execution_bar_time`
- `signal_timeframe`
- `execution_timeframe`
- `entry_price_raw`
- `entry_price_filled`

## Exit Rules

### Signal Exit

- Exit condition is evaluated on the closed signal bar.
- Fill occurs at the next execution bar open.
- Apply exit slippage pessimistically.

### Time Stop

- Time stop is triggered when holding duration reaches the configured limit.
- Fill occurs at the first execution bar open after the limit is reached.

### Stop Loss

Stop evaluation must use execution-bar intrabar extremes:

- LONG: if `low <= stop_price`, stop is considered hit
- SHORT: if `high >= stop_price`, stop is considered hit

Fill price must be pessimistic:

- If the next execution bar opens through the stop, fill at the worse open
- Else fill at stop price minus/plus stop slippage

### Same-Bar Conflict Rule

If both favorable exit and stop are reachable within the same execution bar:

- choose the worse outcome for the strategy

This rule is mandatory.

## Slippage Model

Slippage cannot be zero in live-oriented backtests.

Minimum model:

- normal entry/exit: configurable fixed bps, default `3 bps`
- stop execution: configurable fixed bps, default `8 bps`

If a symbol-specific slippage model exists, prefer the worse of:

- fixed model
- observed spread-based estimate

## Funding Model

For perpetual strategies, funding must always be included.

Rules:

- missing funding data -> experiment invalid
- funding accrues across all settlement events strictly after entry and up to exit
- funding must be stored in trade-level and run-level outputs

Required result fields:

- `funding_paid_pct`
- `slippage_paid_pct`
- `net_pnl_after_costs_pct`

## Trade Independence

Backtests must not inflate trade count by slicing one persistent regime into many
independent trades.

At least one of these rules must hold after a time-stop exit:

- cooldown before re-entry
- re-entry only after signal resets to neutral band
- regime stitching with shared `independent_signal_id`

Recommended default:

- after time-stop, re-entry is forbidden until signal crosses back inside neutral band

## Peer Group Separation

`peer_symbols` and `entry_symbols` are distinct objects and must be configured explicitly.

Forbidden:

- implicitly using the trade universe as the peer universe
- implicitly using the peer universe as the trade universe

## Validation Invariants

A run must fail validation if any of the following is true:

- execution uses same-bar close after a signal from that bar
- stop-loss ignores intrabar high/low
- funding is missing for a perp strategy
- slippage is disabled
- execution timeframe is above `5m` while `5m` or finer data exists
- repeated time-stop re-entry occurs without reset/cooldown

## Migration Plan

1. Add explicit `signal_time` and `execution_time` to signal records.
2. Refactor `generate_single_asset_signals()` to emit pending orders, not same-bar fills.
3. Add a multi-timeframe execution layer that maps signal bars to `5m` or finer bars.
4. Move stop handling from close-only logic to intrabar pessimistic logic using `high/low`.
5. Make `funding_dir` mandatory for funding-driven strategies and persist `funding_paid_pct`.
6. Add slippage fields and pessimistic same-bar conflict resolution.
7. Add anti-reentry rules for time-stop dominated strategies.
8. Re-run all previously approved funding experiments under the new engine.

## Immediate Application To Funding Strategies

For the current funding-divergence research line:

- signal can still be computed on `4h`
- execution must use `5m` if available
- entry after `funding_z <= -2.0` or `>= 2.0` must occur on next `5m` bar open
- stop must be checked on `5m` highs/lows
- funding carry must be part of the saved result, not an optional extra
