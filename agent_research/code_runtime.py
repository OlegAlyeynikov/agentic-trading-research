from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
AGENT_RESEARCH_ROOT = Path(__file__).parent
WORKSPACE_ROOT = AGENT_RESEARCH_ROOT / "workspace"

WORKSPACE_HELPERS_FILENAME = "workspace_helpers.py"
WORKSPACE_HELPERS_CONTENTS = """from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Available libraries ───────────────────────────────────────────────────────
# Pre-installed and safe to import in workspace scripts:
#   numpy             — array math
#   pandas            — DataFrames
#   scipy 1.17        — scipy.signal, scipy.stats, scipy.linalg
#   statsmodels 0.14  — statsmodels.tsa.stattools (adfuller, coint)
#   arch 8.0          — arch.univariate (GARCH, EGARCH) for volatility regime detection
#   hurst 0.0.5       — hurst.compute_Hc(series) for Hurst exponent (mean-reversion quality)
#   ta 0.11           — ta.volatility.AverageTrueRange, ta.momentum.RSIIndicator, etc.
# Do NOT import: requests, httpx, sklearn, torch, filterpy, pykalman, tensorflow, etc.
#
# ── Timeframe selection ───────────────────────────────────────────────────────
# inp["available_timeframes"] contains all available timeframes: e.g. ["1m","5m","15m","1h","4h"]
# Choose the timeframe your theory requires — do NOT use inp["timeframe"] (removed).
# Example:  timeframe = "4h"   # mean-reversion on 4h tends to have longer half-lives
#           timeframe = "1h"   # default for stat-arb; 720 candles ≈ 30 days


# ── Signal CSV contracts ──────────────────────────────────────────────────────
# Two supported schemas — engine auto-detects from columns.
#
# PAIRS strategy (price_a + price_b + hedge_ratio present):
#   T, timestamp, symbol, pair_id, asset_a, asset_b, action,
#   price_a, price_b, hedge_ratio, spread, z_score, strategy_type="pairs"
#
# SINGLE-ASSET strategy (no price_a/price_b):
#   T, timestamp, symbol, action, price, strategy_type="single_asset"
#
# action values: ENTER_LONG | ENTER_SHORT | EXIT | STOP
# T = unix seconds (int) — required for signal ordering

REQUIRED_PAIRS_COLUMNS = [
    "T", "timestamp", "symbol", "pair_id", "asset_a", "asset_b",
    "action", "price_a", "price_b", "hedge_ratio",
]
REQUIRED_SINGLE_ASSET_COLUMNS = ["T", "timestamp", "symbol", "action", "price"]

# Legacy alias — kept for backward compatibility with old workspace scripts
REQUIRED_SETUP_COLUMNS = REQUIRED_PAIRS_COLUMNS


# ── Runtime inputs ────────────────────────────────────────────────────────────
def load_runtime_inputs() -> dict:
    \"\"\"Return all sandbox environment variables as a plain dict.

    Keys:
        workspace_dir        — path to this experiment's workspace dir
        data_dir             — root data directory with timeframe subdirs
        funding_rates_dir    — path to funding rate pkl files (data_dir/new_data/funding_rates)
        available_timeframes — list of available timeframes, e.g. ["1m","5m","15m","1h","4h"]
        symbols              — list[str] of symbols to research
        start_date           — ISO date string or empty string
        signals_csv          — path where your script must write the signals CSV
        setups_csv           — legacy alias for signals_csv
        max_open_positions   — int: max simultaneous open positions across all symbols/pairs
        risk_pct_per_trade   — float: fraction of equity risked per trade (default 0.02)

    Choose timeframe from available_timeframes based on your theory:
        timeframe = inp["available_timeframes"][-1]   # e.g. "4h" for slower mean-reversion
        timeframe = "1h"                               # explicit choice
    \"\"\"
    csv_path = os.environ["AGENT_RESEARCH_SETUPS_CSV"]
    risk_pct = float(os.environ.get("AGENT_RESEARCH_RISK_PCT", "0.02"))
    max_open_raw = os.environ.get("AGENT_RESEARCH_MAX_OPEN_POSITIONS", "")
    max_open = int(max_open_raw) if max_open_raw.strip() else max(1, int(1.0 / risk_pct))
    data_dir = os.environ["AGENT_RESEARCH_DATA_DIR"]
    funding_rates_dir = os.environ.get(
        "AGENT_RESEARCH_FUNDING_RATES_DIR",
        str(Path(data_dir) / "new_data" / "funding_rates"),
    )
    return {
        "workspace_dir":        os.environ["AGENT_RESEARCH_WORKSPACE_DIR"],
        "data_dir":             data_dir,
        "funding_rates_dir":    funding_rates_dir,
        "available_timeframes": json.loads(os.environ.get("AGENT_RESEARCH_AVAILABLE_TIMEFRAMES", '["1h"]')),
        "symbols":              json.loads(os.environ.get("AGENT_RESEARCH_SYMBOLS", "[]")),
        "start_date":           os.environ.get("AGENT_RESEARCH_START_DATE", ""),
        "signals_csv":          csv_path,
        "setups_csv":           csv_path,
        "max_open_positions":   max_open,
        "risk_pct_per_trade":   risk_pct,
    }


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_one_symbol(data_dir: str, symbol: str, timeframe: str) -> "pd.DataFrame":
    \"\"\"Load a single symbol from pkl or parquet. Tries parquet first (1m), then pkl.\"\"\"
    sym_upper = symbol.strip().upper()
    tf_dir = Path(data_dir) / timeframe
    for pattern, loader in [
        (f"{sym_upper}_*.parquet", pd.read_parquet),
        (f"{sym_upper}_*.pkl",     pd.read_pickle),
    ]:
        matches = sorted(tf_dir.glob(pattern))
        if matches:
            return loader(str(matches[0]))
    raise FileNotFoundError(f"No data file found for {sym_upper} in {data_dir}/{timeframe}/")


def load_price_data(
    data_dir: str,
    symbols: list,
    timeframe: str,
    start_date: str = "",
) -> dict:
    \"\"\"Load OHLCV Binance kline files for each symbol.

    Supports both .pkl (1h and other timeframes) and .parquet (1m) files.
    File path: {data_dir}/{timeframe}/{SYMBOL}_*.pkl  or  ...parquet
    Raw columns are Binance kline indices '0'..'11':
      '0'=timestamp(ms), '1'=open, '2'=high, '3'=low, '4'=close, '5'=volume
    Returns {symbol: DataFrame} with named columns and UTC datetime timestamp.
    Raises FileNotFoundError if no matching file is found for a symbol.
    \"\"\"
    result = {}
    for symbol in symbols:
        df = _load_one_symbol(data_dir, symbol, timeframe)
        df = df.rename(columns={
            "0": "timestamp", "1": "open", "2": "high",
            "3": "low",       "4": "close", "5": "volume",
        })
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype("int64"), unit="ms", utc=True
        )
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if start_date:
            cutoff = pd.to_datetime(start_date, utc=True, errors="coerce")
            df = df[df["timestamp"] >= cutoff]
        df = df.sort_values("timestamp").reset_index(drop=True)
        result[symbol] = df
    return result


def timeframe_to_seconds(timeframe: str) -> int:
    \"\"\"Return timeframe length in seconds.

    Examples:
        "1m" -> 60
        "5m" -> 300
        "1h" -> 3600
        "4h" -> 14400
    \"\"\"
    tf = timeframe.strip().lower()
    mapping = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
        "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
    }
    return mapping.get(tf, 3600)


def choose_execution_timeframe(available_timeframes: list[str]) -> str:
    \"\"\"Choose a live-safe execution timeframe.

    Policy:
    - Prefer 5m when available (standard pessimistic execution precision).
    - If 5m is absent but finer data exists, use the finest available below 5m.
    - Else use the finest available timeframe above 5m and mark reduced_fidelity.
    \"\"\"
    if not available_timeframes:
        return "5m"
    normalized = {str(tf).strip().lower(): str(tf).strip() for tf in available_timeframes}
    if "5m" in normalized:
        return normalized["5m"]
    ranked = sorted(
        [(timeframe_to_seconds(tf), tf) for tf in available_timeframes],
        key=lambda x: x[0],
    )
    finer = [tf for secs, tf in ranked if secs < 300]
    if finer:
        return finer[0]
    coarser = [tf for secs, tf in ranked if secs > 300]
    if coarser:
        return coarser[0]
    return ranked[0][1]


def load_execution_price_data(
    data_dir: str,
    symbols: list,
    available_timeframes: list[str],
    start_date: str = "",
) -> tuple[dict, str, bool]:
    \"\"\"Load execution-timeframe OHLCV data using the live-safe timeframe policy.

    Returns:
        execution_prices: dict[symbol, DataFrame]
        execution_timeframe: selected timeframe string
        reduced_fidelity: True when no 5m-or-finer data exists
    \"\"\"
    execution_timeframe = choose_execution_timeframe(available_timeframes)
    reduced_fidelity = timeframe_to_seconds(execution_timeframe) > 300
    execution_prices = load_price_data(data_dir, symbols, execution_timeframe, start_date)
    return execution_prices, execution_timeframe, reduced_fidelity


def _worse_stop_fill_price(entry_price: float, stop_price: float, next_exec_price: float, direction: str) -> float:
    \"\"\"Return a pessimistic stop fill against the next executable open.

    LONG stop:
      - normal stop price is below entry
      - if next executable open is even lower, assume fill at that worse open

    SHORT stop:
      - normal stop price is above entry
      - if next executable open is even higher, assume fill at that worse open
    \"\"\"
    if direction == "LONG":
        return float(min(stop_price, next_exec_price))
    return float(max(stop_price, next_exec_price))


def build_next_bar_execution_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    \"\"\"Return per-row execution fields for next-bar-open fills.

    Assumes df rows are ordered by bar open timestamp. For a signal computed on
    row t (using information from the just-closed bar), the earliest legal fill
    is the open of row t+1.

    Returned columns:
        signal_timestamp
        signal_close_timestamp
        execution_timestamp
        execution_price
        stop_high
        stop_low
    \"\"\"
    out = pd.DataFrame(index=df.index)
    out["signal_timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    inferred_step = out["signal_timestamp"].shift(-1) - out["signal_timestamp"]
    out["signal_close_timestamp"] = out["signal_timestamp"] + inferred_step
    out["execution_timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").shift(-1)
    out["execution_price"] = pd.to_numeric(df["open"], errors="coerce").shift(-1)
    out["stop_high"] = pd.to_numeric(df["high"], errors="coerce")
    out["stop_low"] = pd.to_numeric(df["low"], errors="coerce")
    return out


def build_execution_bridge(
    signal_df: "pd.DataFrame",
    execution_df: "pd.DataFrame",
    signal_timeframe: str,
    execution_timeframe: str,
) -> "pd.DataFrame":
    \"\"\"Project signal bars onto an execution timeframe.

    Semantics:
    - signal row t is evaluated after signal bar close
    - execution row t is the first execution bar whose open is at or after
      signal_close[t]
    - stop_high/stop_low on row t summarize the worst intrabar move during the
      interval from signal_close[t-1] to signal_close[t]

    This lets a strategy compute signals on 1h/4h data while executing and
    checking stops on 5m-or-finer data.
    \"\"\"
    sig = signal_df.copy().sort_values("timestamp").reset_index(drop=True)
    exe = execution_df.copy().sort_values("timestamp").reset_index(drop=True)

    sig_open = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
    exe_open = pd.to_datetime(exe["timestamp"], utc=True, errors="coerce")
    sig_close = sig_open + pd.to_timedelta(timeframe_to_seconds(signal_timeframe), unit="s")

    exe_open_arr = exe_open.to_numpy(dtype="datetime64[ns]")
    exe_open_px = pd.to_numeric(exe["open"], errors="coerce").to_numpy(dtype=float)
    exe_high = pd.to_numeric(exe["high"], errors="coerce").to_numpy(dtype=float)
    exe_low = pd.to_numeric(exe["low"], errors="coerce").to_numpy(dtype=float)
    sig_close_arr = sig_close.to_numpy(dtype="datetime64[ns]")

    fill_idx = np.searchsorted(exe_open_arr, sig_close_arr, side="left")

    execution_timestamp = []
    execution_price = []
    stop_high = []
    stop_low = []

    for i, idx in enumerate(fill_idx):
        if idx < len(exe_open_arr):
            execution_timestamp.append(pd.Timestamp(exe_open_arr[idx], tz="UTC"))
            execution_price.append(float(exe_open_px[idx]))
        else:
            execution_timestamp.append(pd.NaT)
            execution_price.append(float("nan"))

        if i == 0:
            stop_high.append(float("nan"))
            stop_low.append(float("nan"))
            continue

        left = int(np.searchsorted(exe_open_arr, sig_close_arr[i - 1], side="left"))
        right = int(np.searchsorted(exe_open_arr, sig_close_arr[i], side="left"))
        if right <= left:
            stop_high.append(float("nan"))
            stop_low.append(float("nan"))
            continue

        interval_high = exe_high[left:right]
        interval_low = exe_low[left:right]
        stop_high.append(
            float(np.nanmax(interval_high)) if len(interval_high) and not np.isnan(interval_high).all() else float("nan")
        )
        stop_low.append(
            float(np.nanmin(interval_low)) if len(interval_low) and not np.isnan(interval_low).all() else float("nan")
        )

    out = pd.DataFrame(index=sig.index)
    out["signal_timestamp"] = sig_open
    out["signal_close_timestamp"] = sig_close
    out["execution_timestamp"] = execution_timestamp
    out["execution_price"] = execution_price
    out["stop_high"] = stop_high
    out["stop_low"] = stop_low
    out["execution_timeframe"] = execution_timeframe
    out["reduced_fidelity"] = timeframe_to_seconds(execution_timeframe) > 300
    return out


# ── T column ──────────────────────────────────────────────────────────────────
def add_t_column(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Derive integer unix-seconds column T from the timestamp column.

    backtest.run_backtest() requires T for signal sorting and write-back.
    write_setups_csv() calls this automatically if T is absent.
    \"\"\"
    df = df.copy()
    # to_numpy(dtype="datetime64[ns]") forces nanosecond int64 regardless of
    # internal pandas resolution (pandas 3.x stores datetime64[ms] from unit="ms")
    ts_ns = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )
    df["T"] = ts_ns // 1_000_000_000
    return df


# ── CSV writing ───────────────────────────────────────────────────────────────
def _infer_strategy_type(df: "pd.DataFrame") -> str:
    if {"price_a", "price_b", "hedge_ratio"}.issubset(df.columns):
        return "pairs"
    return "single_asset"


def write_signals_csv(df: "pd.DataFrame", path: str) -> str:
    \"\"\"Universal signal CSV writer — works for pairs and single-asset strategies.

    Adds T column (unix seconds) if absent.
    Adds strategy_type column if absent (auto-detected from columns).
    Validates minimum required columns.
    Always use this instead of df.to_csv() directly.
    \"\"\"
    df = df.copy()
    if "T" not in df.columns:
        df = add_t_column(df)
    if "strategy_type" not in df.columns:
        df["strategy_type"] = _infer_strategy_type(df)

    strategy_type = str(df["strategy_type"].iloc[0]) if len(df) > 0 else "single_asset"
    if strategy_type == "pairs":
        missing = [c for c in REQUIRED_PAIRS_COLUMNS if c not in df.columns]
    else:
        missing = [c for c in REQUIRED_SINGLE_ASSET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required signal columns for {strategy_type}: {missing}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_setups_csv(df: "pd.DataFrame", path: str) -> str:
    \"\"\"Legacy alias for write_signals_csv — backward compatible with pairs scripts.\"\"\"
    return write_signals_csv(df, path)


# ── JSON output ───────────────────────────────────────────────────────────────
def emit_success(setups_csv: str, metrics: dict | None = None, notes: str = "") -> None:
    \"\"\"Print the success JSON payload to stdout (read by sandbox).\"\"\"
    print(json.dumps(
        {"status": "success", "setups_csv": setups_csv,
         "metrics": metrics or {}, "notes": notes},
        ensure_ascii=True,
    ))


def emit_blocked(notes: str, missing_requirements: list | None = None, metrics: dict | None = None) -> None:
    \"\"\"Print a blocked JSON payload to stdout (no setups CSV produced).\"\"\"
    print(json.dumps(
        {"status": "blocked", "setups_csv": "", "metrics": metrics or {},
         "notes": notes, "missing_requirements": missing_requirements or []},
        ensure_ascii=True,
    ))


def filter_available_symbols(data_dir: str, symbols: list, timeframe: str) -> list:
    \"\"\"Return only the symbols that have data files in data_dir/timeframe/.

    Use this before load_price_data to avoid FileNotFoundError when the agent
    selects symbols that may not exist on disk.

    Example:
        symbols = filter_available_symbols(inp["data_dir"], inp["symbols"], timeframe)
        if len(symbols) < 2:
            emit_blocked("Not enough symbols with data.", missing_requirements=[...])
            return
        prices = load_price_data(inp["data_dir"], symbols, timeframe, inp["start_date"])
    \"\"\"
    tf_dir = Path(data_dir) / timeframe
    available = []
    for sym in symbols:
        sym_upper = sym.strip().upper()
        has_file = (
            bool(list(tf_dir.glob(f"{sym_upper}_*.parquet")))
            or bool(list(tf_dir.glob(f"{sym_upper}_*.pkl")))
        )
        if has_file:
            available.append(sym)
        else:
            print(f"[filter_available_symbols] skipping {sym_upper}: no file in {tf_dir}")
    return available


def apply_position_limit(
    signals_df: "pd.DataFrame",
    max_open_positions: int = 1,
) -> "pd.DataFrame":
    \"\"\"Filter ENTER signals to enforce a global cap on concurrent open positions.

    Call this on the fully assembled signals DataFrame BEFORE write_signals_csv().
    It drops ENTER signals that would exceed the position limit and drops any
    EXIT/STOP signals that no longer have a matching ENTER (orphan exits).

    Selection when multiple ENTERs arrive at the same timestamp: the signal with
    the largest abs(z_score) is preferred (strongest conviction first).

    Args:
        signals_df:          DataFrame of all signals (pairs or single-asset).
        max_open_positions:  Maximum simultaneous open positions. Use inp["max_open_positions"].

    Returns a new filtered DataFrame — the original is not modified.

    Example:
        signals = generate_single_asset_signals(...)  # per symbol in a loop
        df = pd.DataFrame(signals)
        df = apply_position_limit(df, inp["max_open_positions"])
        write_signals_csv(df, inp["signals_csv"])
    \"\"\"
    import math
    if signals_df.empty or max_open_positions <= 0:
        return signals_df

    df = signals_df.copy()
    group_col = "pair_id" if "pair_id" in df.columns else "symbol"

    # Ensure T column exists for sorting
    if "T" not in df.columns:
        df = add_t_column(df)

    # Sort: time asc, then STOP/EXIT before ENTER, then strongest z_score first for ENTERs
    _pri = {"STOP": 0, "EXIT": 1, "ENTER_LONG": 2, "ENTER_SHORT": 3}
    df["_pri"] = df["action"].map(_pri).fillna(99).astype(int)
    z_col = df.get("z_score", pd.Series(0.0, index=df.index))
    df["_zabs"] = pd.to_numeric(z_col, errors="coerce").fillna(0.0).abs()
    df = df.sort_values(["T", "_pri", "_zabs"], ascending=[True, True, False]).reset_index(drop=True)

    open_groups: set = set()
    keep = []

    for _, row in df.iterrows():
        action = str(row["action"])
        grp = str(row.get(group_col, ""))

        if action in ("EXIT", "STOP"):
            if grp in open_groups:
                keep.append(True)
                open_groups.discard(grp)
            else:
                keep.append(False)  # orphan — no matching ENTER was emitted
        elif action in ("ENTER_LONG", "ENTER_SHORT"):
            can_open = grp not in open_groups and len(open_groups) < max_open_positions
            keep.append(can_open)
            if can_open:
                open_groups.add(grp)
        else:
            keep.append(True)

    return df[keep].drop(columns=["_pri", "_zabs"]).reset_index(drop=True)


# ── Signal generation helpers ─────────────────────────────────────────────────
def candles_per_day(timeframe: str) -> int:
    \"\"\"Return number of candles in one calendar day for a given timeframe string.

    Examples: '1h' -> 24, '4h' -> 6, '15m' -> 96, '1d' -> 1.
    Always use this instead of hardcoding 24 — timeframe comes from the env.
    \"\"\"
    tf = timeframe.strip().lower()
    mapping = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
               "1h": 24, "2h": 12, "4h": 6, "6h": 4, "8h": 3, "12h": 2,
               "1d": 1, "3d": 1, "1w": 1}
    return mapping.get(tf, 24)


def load_funding_rates(data_dir: str, symbol: str) -> "pd.Series":
    \"\"\"Load perpetual funding rates for a symbol as a time-indexed Series.

    Returns a Series indexed by UTC datetime with float fundingRate values.
    Funding payments occur every 8 hours (00:00, 08:00, 16:00 UTC).
    Returns an empty Series if no funding data is found (graceful degradation).

    Usage:
        funding = load_funding_rates(inp["data_dir"], "SOLUSDT")
        # Check current regime: high positive = market overlong = contrarian signal
        recent_funding = funding.last("7D").mean()
        if recent_funding > 0.0003:  # 0.03%/8h = extremely high
            # Market is overlong — good for short entries (fade)
            pass

    Funding rate interpretation:
        rate > 0: longs pay shorts (bullish market, longs are dominant)
        rate < 0: shorts pay longs (bearish market, shorts are dominant)
        Typical range: -0.05% to +0.05% per 8h (extremes: -2% to +0.33%)
        Historical mean (2024): SOL +0.012%/8h, NEAR +0.013%/8h, AVAX +0.010%/8h

    Note: The backtest engine automatically deducts funding costs from trade PnL.
    This function is for using funding as a SIGNAL in your strategy logic.
    \"\"\"
    from pathlib import Path
    funding_dir_env = os.environ.get("AGENT_RESEARCH_FUNDING_RATES_DIR", "").strip()
    funding_dir = Path(funding_dir_env) if funding_dir_env else Path(data_dir) / "new_data" / "funding_rates"
    matches = sorted(funding_dir.glob(f"{symbol}_*.pkl"))
    if not matches:
        return pd.Series(dtype=float)
    try:
        df = pd.read_pickle(str(matches[0]))
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        return df.set_index("ts")["fundingRate"].sort_index()
    except Exception:
        return pd.Series(dtype=float)


def compute_peer_funding_zscore(
    data_dir: str,
    symbols: list,
    lookback_payments: int = 90,
) -> "pd.DataFrame":
    \"\"\"Compute contemporaneous funding z-score relative to peer symbols.

    For each settlement timestamp t, compute:
        z[symbol, t] = (funding[symbol, t] - peer_mean[t]) / peer_std[t]

    where peer_mean/std are computed from the *other* peer symbols at the same
    timestamp. This is a cross-sectional spike detector, not a rolling regime
    score.

    `lookback_payments` is accepted for backward compatibility with older
    workspace scripts but is intentionally ignored by this implementation.
    \"\"\"
    _ = lookback_payments
    all_series = {}
    for sym in symbols:
        s = load_funding_rates(data_dir, sym)
        if not s.empty:
            all_series[sym] = s

    if not all_series:
        return pd.DataFrame()

    # Align all series to a common settlement index.
    df = pd.DataFrame(all_series)
    df = df.sort_index().dropna(how="all")

    # Cross-sectional peer mean and std at each timestamp (exclude self).
    result = {}
    for sym in df.columns:
        peers = df.drop(columns=[sym])
        if peers.empty:
            result[sym] = pd.Series(0.0, index=df.index)
            continue
        peer_mean = peers.mean(axis=1)
        peer_std = peers.std(axis=1, ddof=0).replace(0, float("nan"))
        result[sym] = (df[sym] - peer_mean) / peer_std

    return pd.DataFrame(result)


def generate_funding_divergence_signals(
    data_dir: str,
    symbols: list,
    available_timeframes: list[str],
    start_date: str = "",
    timeframe: str = "4h",
    direction: str = "short_high",
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_pct: float | None = 0.07,
    stop_z: float = 99.0,
    max_holding_candles: int = 18,
    exclude_entry_symbols: list[str] | None = None,
    entry_symbols: list[str] | None = None,
    peer_symbols: list[str] | None = None,
    max_open_positions: int = 1,
    require_carry_sign: bool = True,
    price_z_threshold: float | None = None,
    price_z_window: int = 90,
) -> tuple["pd.DataFrame", dict]:
    \"\"\"Canonical funding-divergence signal builder.

    This helper is intentionally strict so generated workspace scripts do not
    reimplement timestamp alignment, execution bridging, or action filtering.

    direction:
      - "short_high": SHORT when peer funding_z >= entry_z
      - "long_low":   LONG  when peer funding_z <= -entry_z

    Returns (signals_df, diagnostics). signals_df is empty when no usable
    signals exist; diagnostics explains whether this is a real data gap or a
    threshold/universe issue.
    \"\"\"
    exclude = {str(s).upper() for s in (exclude_entry_symbols or [])}
    direction = str(direction).strip().lower()
    if direction not in {"short_high", "long_low"}:
        raise ValueError("direction must be 'short_high' or 'long_low'")

    available = filter_available_symbols(data_dir, symbols, timeframe)
    diagnostics: dict = {
        "timeframe": timeframe,
        "direction": direction,
        "entry_z": float(entry_z),
        "exit_z": float(exit_z),
        "require_carry_sign": bool(require_carry_sign),
        "price_z_threshold": price_z_threshold,
        "price_z_window": int(price_z_window),
        "available_symbols": len(available),
        "symbols": {},
    }
    if len(available) < 2:
        diagnostics["blocked_reason"] = "fewer_than_two_price_symbols"
        return pd.DataFrame(), diagnostics

    peer_universe = [s for s in (peer_symbols or available) if s in available]
    if len(peer_universe) < 2:
        diagnostics["blocked_reason"] = "fewer_than_two_peer_symbols"
        return pd.DataFrame(), diagnostics

    funding_z = compute_peer_funding_zscore(data_dir, peer_universe)
    diagnostics["funding_rows"] = int(len(funding_z))
    diagnostics["funding_symbols"] = int(len(funding_z.columns)) if not funding_z.empty else 0
    if funding_z.empty:
        diagnostics["blocked_reason"] = "funding_z_empty"
        return pd.DataFrame(), diagnostics

    entry_universe = [s for s in (entry_symbols or available) if s in available]
    trade_symbols = [s for s in entry_universe if str(s).upper() not in exclude and s in funding_z.columns]
    diagnostics["trade_symbols"] = len(trade_symbols)
    diagnostics["entry_symbols"] = entry_universe
    diagnostics["excluded_entry_symbols"] = sorted(exclude)
    if not trade_symbols:
        diagnostics["blocked_reason"] = "no_trade_symbols"
        return pd.DataFrame(), diagnostics

    prices = load_price_data(data_dir, trade_symbols, timeframe, start_date)
    exec_prices, exec_tf, reduced_fidelity = load_execution_price_data(
        data_dir, trade_symbols, available_timeframes, start_date
    )
    diagnostics["execution_timeframe"] = exec_tf
    diagnostics["reduced_fidelity"] = bool(reduced_fidelity)

    all_signals: list[dict] = []
    for sym in trade_symbols:
        price_df = prices.get(sym)
        exec_df = exec_prices.get(sym)
        if price_df is None or price_df.empty or exec_df is None or exec_df.empty:
            diagnostics["symbols"][sym] = {"blocked_reason": "missing_price_or_execution_data"}
            continue

        price_ts = pd.to_datetime(price_df["timestamp"], utc=True, errors="coerce")
        raw_z = funding_z[sym].reindex(price_ts, method="ffill")
        raw_funding = load_funding_rates(data_dir, sym).reindex(price_ts, method="ffill")
        aligned_non_null = int(raw_z.notna().sum())
        z = raw_z.fillna(0.0).reset_index(drop=True)
        funding = raw_funding.reset_index(drop=True)

        if direction == "short_high":
            carry_ok = funding.gt(0) if require_carry_sign else pd.Series(True, index=z.index)
            threshold_ok = z >= float(entry_z)
        else:
            carry_ok = funding.lt(0) if require_carry_sign else pd.Series(True, index=z.index)
            threshold_ok = z <= -float(entry_z)

        price_gate_ok = pd.Series(True, index=z.index)
        price_z = pd.Series(np.nan, index=z.index)
        if price_z_threshold is not None:
            window = max(int(price_z_window), 2)
            close = pd.to_numeric(price_df["close"], errors="coerce").reset_index(drop=True)
            roll_mean = close.rolling(window, min_periods=window).mean()
            roll_std = close.rolling(window, min_periods=window).std()
            price_z = (close - roll_mean) / roll_std.where(roll_std > 1e-12, other=float("nan"))
            if direction == "short_high":
                price_gate_ok = price_z >= float(price_z_threshold)
            else:
                price_gate_ok = price_z <= -float(price_z_threshold)

        entry_allowed = (carry_ok & price_gate_ok).fillna(False)
        count_beyond = int((threshold_ok & entry_allowed).sum())

        sym_diag = {
            "price_rows": int(len(price_df)),
            "execution_rows": int(len(exec_df)),
            "alignment_non_null_count": aligned_non_null,
            "raw_funding_non_null_count": int(raw_funding.notna().sum()),
            "funding_z_min": float(z.min()) if len(z) else None,
            "funding_z_max": float(z.max()) if len(z) else None,
            "count_beyond_entry": count_beyond,
            "count_carry_sign_ok": int(carry_ok.sum()) if len(z) else 0,
            "count_price_z_ok": int(price_gate_ok.fillna(False).sum()) if len(z) else 0,
            "price_z_min": float(price_z.min()) if price_z.notna().any() else None,
            "price_z_max": float(price_z.max()) if price_z.notna().any() else None,
        }
        diagnostics["symbols"][sym] = sym_diag
        if count_beyond == 0:
            continue

        bridge = build_execution_bridge(price_df, exec_df, timeframe, exec_tf)
        signals = generate_single_asset_signals(
            timestamps=price_ts.reset_index(drop=True),
            prices=price_df["close"].reset_index(drop=True),
            z_scores=z.reset_index(drop=True),
            symbol=sym,
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=float(stop_z),
            stop_pct=stop_pct,
            max_holding_candles=int(max_holding_candles),
            execution_timestamps=bridge["execution_timestamp"],
            execution_prices=bridge["execution_price"],
            stop_high=bridge["stop_high"],
            stop_low=bridge["stop_low"],
            signal_close_timestamps=bridge["signal_close_timestamp"],
            execution_timeframe=exec_tf,
            reduced_fidelity=reduced_fidelity,
            entry_allowed_mask=entry_allowed.reset_index(drop=True),
        )
        if direction == "short_high":
            signals = [s for s in signals if s["action"] in {"ENTER_SHORT", "EXIT", "STOP"}]
        else:
            signals = [s for s in signals if s["action"] in {"ENTER_LONG", "EXIT", "STOP"}]
        sym_diag["signals"] = len(signals)
        all_signals.extend(signals)

    if not all_signals:
        diagnostics["blocked_reason"] = "no_signals_generated"
        return pd.DataFrame(), diagnostics

    df = pd.DataFrame(all_signals)
    df = apply_position_limit(df, max_open_positions)
    if df.empty:
        diagnostics["blocked_reason"] = "no_signals_after_position_limit"
        return df, diagnostics

    diagnostics["total_rows"] = int(len(df))
    diagnostics["enter_signals"] = int(df["action"].isin(["ENTER_LONG", "ENTER_SHORT"]).sum())
    diagnostics["symbols_with_signals"] = int(df["symbol"].nunique())
    return df, diagnostics


def run_funding_divergence_strategy(
    inp: dict,
    *,
    timeframe: str = "4h",
    direction: str = "short_high",
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_pct: float | None = 0.07,
    stop_z: float = 99.0,
    max_holding_candles: int = 18,
    exclude_entry_symbols: list[str] | None = None,
    entry_symbols: list[str] | None = None,
    symbols: list[str] | None = None,
    peer_symbols: list[str] | None = None,
    require_carry_sign: bool = True,
    price_z_threshold: float | None = None,
    price_z_window: int = 90,
    notes: str = "",
) -> None:
    \"\"\"Build, write, and emit a funding-divergence strategy result JSON.\"\"\"
    if entry_symbols is None and symbols is not None:
        entry_symbols = symbols
    df, diagnostics = generate_funding_divergence_signals(
        data_dir=inp["data_dir"],
        symbols=inp["symbols"],
        available_timeframes=inp["available_timeframes"],
        start_date=inp.get("start_date", ""),
        timeframe=timeframe,
        direction=direction,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_pct=stop_pct,
        stop_z=stop_z,
        max_holding_candles=max_holding_candles,
        exclude_entry_symbols=exclude_entry_symbols,
        entry_symbols=entry_symbols,
        peer_symbols=peer_symbols,
        max_open_positions=int(inp.get("max_open_positions", 1)),
        require_carry_sign=require_carry_sign,
        price_z_threshold=price_z_threshold,
        price_z_window=price_z_window,
    )
    metrics = {
        "diagnostics": diagnostics,
        "total_enter_signals": int(df["action"].isin(["ENTER_LONG", "ENTER_SHORT"]).sum()) if not df.empty else 0,
        "symbols_traded": int(df["symbol"].nunique()) if not df.empty else 0,
    }
    if df.empty:
        emit_blocked(
            f"No funding divergence signals generated: {diagnostics.get('blocked_reason', 'unknown')}.",
            missing_requirements=[],
            metrics=metrics,
        )
        return
    csv_path = inp["signals_csv"]
    write_signals_csv(df, csv_path)
    emit_success(csv_path, metrics=metrics, notes=notes or f"funding divergence {direction}")


def ols_hedge_ratio(price_a: "pd.Series", price_b: "pd.Series") -> float:
    \"\"\"Compute OLS hedge ratio beta: price_b ~ beta * price_a.

    Returns beta = cov(a, b) / var(a).  Both series must be aligned.
    Consistent with screen_pairs_cointegration and backtest engine convention:
        spread = price_b - beta * price_a
    \"\"\"
    import numpy as np
    b = price_b.values
    a = price_a.values
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return 1.0
    b_clean, a_clean = b[mask], a[mask]
    var_a = np.var(a_clean)
    return float(np.cov(a_clean, b_clean)[0, 1] / var_a) if var_a > 0 else 1.0


def kalman_hedge_ratio(price_a: "pd.Series", price_b: "pd.Series",
                       delta: float = 1e-4, ve: float = 0.001) -> "pd.Series":
    \"\"\"Compute time-varying hedge ratio using a Kalman filter regression.

    Models:  price_b[t] = beta[t] * price_a[t] + epsilon[t]
    Returns a Series of beta estimates (one per candle) aligned with price_a.index.
    Consistent with screen_pairs_cointegration and backtest engine convention:
        spread = price_b - beta * price_a

    PERFORMANCE WARNING — call ONCE on the full price series, NOT inside a loop:
        hr = kalman_hedge_ratio(merged["close_a"], merged["close_b"])   # O(n), correct
        # NOT:  for t in range(n): hr_t = kalman_hedge_ratio(window_a, window_b)  <- O(n^2), TIMEOUT

    Uses raw numpy — no filterpy shape bugs.
    delta controls how fast beta can change (larger = faster adaptation).
    ve    controls measurement noise (larger = smoother beta).
    \"\"\"
    import numpy as np
    n = len(price_a)
    beta = np.zeros(n)
    P = 1.0          # state variance
    Vw = delta / (1 - delta)   # process noise

    a = price_a.values
    b = price_b.values

    for t in range(n):
        if np.isnan(a[t]) or np.isnan(b[t]) or a[t] == 0:
            beta[t] = beta[t - 1] if t > 0 else 0.0
            continue
        # Predict
        P = P + Vw
        # Update: observation model is b[t] = beta * a[t]
        H = a[t]
        innovation = b[t] - H * beta[t - 1] if t > 0 else 0.0
        S = H * P * H + ve
        K = P * H / S
        beta[t] = (beta[t - 1] if t > 0 else 0.0) + K * innovation
        P = (1 - K * H) * P

    return pd.Series(beta, index=price_a.index)


def rolling_zscore(spread: "pd.Series", window: int) -> "pd.Series":
    \"\"\"Vectorized rolling z-score — use this for walk-forward normalization.

    Equivalent to:  (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    but handles edge cases (std=0 → NaN) cleanly.

    Example for 30-day OOS window on 1h data:
        hr     = kalman_hedge_ratio(close_a, close_b)   # call once
        spread = close_b - hr * close_a                 # b = beta * a convention
        z      = rolling_zscore(spread, window=30 * candles_per_day("1h"))
    \"\"\"
    mean = spread.rolling(window, min_periods=window).mean()
    std  = spread.rolling(window, min_periods=window).std()
    return (spread - mean) / std.where(std > 1e-8, other=float("nan"))


def generate_pair_signals(
    timestamps: "pd.Series",
    price_a: "pd.Series",
    price_b: "pd.Series",
    hedge_ratios: "pd.Series",
    spreads: "pd.Series",
    z_scores: "pd.Series",
    asset_a: str,
    asset_b: str,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_candles: int | None = None,
    execution_timestamps: "pd.Series | None" = None,
    execution_price_a: "pd.Series | None" = None,
    execution_price_b: "pd.Series | None" = None,
    signal_close_timestamps: "pd.Series | None" = None,
    execution_timeframe: "str | None" = None,
    reduced_fidelity: "bool | None" = None,
    require_reset_after_time_stop: bool = True,
) -> list:
    \"\"\"Generate ENTER/EXIT signals with a strict per-pair state machine.

    Rules enforced:
    - Only one open position at a time per pair.
    - Entries only when entry_z <= |z| < stop_z (prevents immediate-stop entries).
    - EXIT / STOP are only emitted when a position is open.
    - If max_holding_candles is set, force EXIT after N candles regardless of z-score.
    - Open position at end of data is closed with EXIT.
    - No lookahead: all decisions use data at candle t.

    Returns list of signal dicts ready for pd.DataFrame(setups).
    \"\"\"
    pair_id = f"{asset_a}-{asset_b}"
    position = None   # None | "long" | "short"
    entry_candle = None
    reentry_locked = False
    setups = []

    ts_list   = list(timestamps)
    pa_list   = list(price_a)
    pb_list   = list(price_b)
    hr_list   = list(hedge_ratios)
    sp_list   = list(spreads)
    zs_list   = list(z_scores)
    exec_ts_list = list(execution_timestamps) if execution_timestamps is not None else None
    exec_pa_list = list(execution_price_a) if execution_price_a is not None else None
    exec_pb_list = list(execution_price_b) if execution_price_b is not None else None
    signal_close_ts_list = list(signal_close_timestamps) if signal_close_timestamps is not None else None

    def _coerce_ts(v):
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return str(v)

    def _can_execute(t: int) -> bool:
        if exec_ts_list is not None and exec_pa_list is not None and exec_pb_list is not None:
            if t >= len(exec_ts_list) or t >= len(exec_pa_list) or t >= len(exec_pb_list):
                return False
            return pd.notna(exec_ts_list[t]) and pd.notna(exec_pa_list[t]) and pd.notna(exec_pb_list[t])
        return t + 1 < len(ts_list)

    def _exec_ts(t: int):
        if exec_ts_list is not None:
            return exec_ts_list[t]
        return ts_list[t + 1]

    def _exec_pa(t: int) -> float:
        if exec_pa_list is not None:
            return float(exec_pa_list[t])
        return float(pa_list[t + 1])

    def _exec_pb(t: int) -> float:
        if exec_pb_list is not None:
            return float(exec_pb_list[t])
        return float(pb_list[t + 1])

    def _row(t, action, exit_reason_detail: str = ""):
        signal_ts = ts_list[t]
        signal_close_ts = (
            signal_close_ts_list[t]
            if signal_close_ts_list is not None and t < len(signal_close_ts_list)
            else signal_ts
        )
        exec_ts = _exec_ts(t)
        exec_pa = _exec_pa(t)
        exec_pb = _exec_pb(t)
        return {
            "pair_id":    pair_id,
            "asset_a":    asset_a,
            "asset_b":    asset_b,
            "symbol":     pair_id,
            "action":     action,
            "exit_reason_detail": exit_reason_detail,
            "timestamp":  _coerce_ts(exec_ts),
            "hedge_ratio": float(hr_list[t]),
            "price_a":    exec_pa,
            "price_b":    exec_pb,
            "spread":     float(sp_list[t]),
            "z_score":    float(zs_list[t]),
            "signal_timestamp": _coerce_ts(signal_ts),
            "signal_close_timestamp": _coerce_ts(signal_close_ts),
            "signal_price_a": float(pa_list[t]),
            "signal_price_b": float(pb_list[t]),
            "execution_timestamp": _coerce_ts(exec_ts),
            "execution_price_a": exec_pa,
            "execution_price_b": exec_pb,
            "execution_timeframe": execution_timeframe or "",
            "reduced_fidelity": bool(reduced_fidelity) if reduced_fidelity is not None else False,
        }

    for t in range(len(ts_list)):
        z = zs_list[t]
        import math
        if math.isnan(z):
            continue

        if position is None:
            if reentry_locked:
                if abs(z) <= exit_z:
                    reentry_locked = False
                else:
                    continue
            if not _can_execute(t):
                continue
            # Only enter when z is within [entry_z, stop_z) — prevents immediate-stop entries.
            # Entries at |z| >= stop_z would fire STOP on the very next candle (wrong direction).
            if z <= -entry_z and z > -stop_z:
                setups.append(_row(t, "ENTER_LONG"))
                position = "long"
                entry_candle = t
            elif z >= entry_z and z < stop_z:
                setups.append(_row(t, "ENTER_SHORT"))
                position = "short"
                entry_candle = t
        elif position == "long":
            held = t - entry_candle if entry_candle is not None else 0
            if max_holding_candles is not None and held >= max_holding_candles and _can_execute(t):
                setups.append(_row(t, "EXIT", "TIME_EXIT"))
                if require_reset_after_time_stop:
                    reentry_locked = True
                position = None
                entry_candle = None
            elif abs(z) >= stop_z and _can_execute(t):
                setups.append(_row(t, "STOP", "STOP"))
                position = None
                entry_candle = None
            elif z >= -exit_z and _can_execute(t):
                setups.append(_row(t, "EXIT", "MEAN_REVERSION"))
                position = None
                entry_candle = None
        elif position == "short":
            held = t - entry_candle if entry_candle is not None else 0
            if max_holding_candles is not None and held >= max_holding_candles and _can_execute(t):
                setups.append(_row(t, "EXIT", "TIME_EXIT"))
                if require_reset_after_time_stop:
                    reentry_locked = True
                position = None
                entry_candle = None
            elif abs(z) >= stop_z and _can_execute(t):
                setups.append(_row(t, "STOP", "STOP"))
                position = None
                entry_candle = None
            elif z <= exit_z and _can_execute(t):
                setups.append(_row(t, "EXIT", "MEAN_REVERSION"))
                position = None
                entry_candle = None

    return setups


def generate_single_asset_signals(
    timestamps: "pd.Series",
    prices: "pd.Series",
    z_scores: "pd.Series",
    symbol: str,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    stop_pct: "float | None" = None,
    max_holding_candles: "int | None" = None,
    execution_timestamps: "pd.Series | None" = None,
    execution_prices: "pd.Series | None" = None,
    stop_high: "pd.Series | None" = None,
    stop_low: "pd.Series | None" = None,
    signal_close_timestamps: "pd.Series | None" = None,
    execution_timeframe: "str | None" = None,
    reduced_fidelity: "bool | None" = None,
    require_reset_after_time_stop: bool = True,
    entry_allowed_mask: "pd.Series | list | None" = None,
) -> list:
    \"\"\"Generate ENTER/EXIT signals for a single-asset strategy.

    Stop priority (checked in order, first hit wins):
    1. stop_pct (absolute %): fires when price moves stop_pct against entry.
       LONG: fires when price ≤ entry_price × (1 - stop_pct).
       SHORT: fires when price ≥ entry_price × (1 + stop_pct).
       Recommended for crypto: 0.04–0.06 (4–6%). Protects against flash crashes
       without exiting during normal mean-reversion deepening.
    2. stop_z (z-score): fires when |z| ≥ stop_z. Use stop_z=99 to disable.
    3. max_holding_candles: force EXIT after N candles regardless of z or price.
       1h → 24 (24h), 4h → 30 (5 days).
    4. exit_z: normal exit when z returns to ±exit_z (set 0.0 for full reversion).

    Entry guard: only enters when entry_z ≤ |z| < stop_z (prevents immediate stop).

    Execution model:
    - signal is evaluated on bar t
    - ENTER/EXIT fills use next-bar execution by default
    - if execution_timestamps/execution_prices are provided, they are used directly
    - otherwise the function falls back to timestamps[t+1] / prices[t+1]

    Output columns use execution timestamps/prices:
    - timestamp, price           → execution fields used by backtest engine
    - execution_timestamp/price  → explicit aliases for validation/audit
    - signal_timestamp           → bar that triggered the decision
    - signal_close_timestamp     → when the signal bar became knowable
    - signal_price               → source-bar price used in signal evaluation
    \"\"\"
    position = None
    entry_price = None
    entry_candle = None
    reentry_locked = False
    signals = []

    ts_list = list(timestamps)
    px_list = list(prices)
    zs_list = list(z_scores)
    entry_allowed_list = list(entry_allowed_mask) if entry_allowed_mask is not None else None
    exec_ts_list = list(execution_timestamps) if execution_timestamps is not None else None
    exec_px_list = list(execution_prices) if execution_prices is not None else None
    hi_list = list(stop_high) if stop_high is not None else None
    lo_list = list(stop_low) if stop_low is not None else None
    signal_close_ts_list = list(signal_close_timestamps) if signal_close_timestamps is not None else None

    def _coerce_ts(v):
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return str(v)

    def _can_execute(t: int) -> bool:
        if exec_ts_list is not None and exec_px_list is not None:
            if t >= len(exec_ts_list) or t >= len(exec_px_list):
                return False
            return pd.notna(exec_ts_list[t]) and pd.notna(exec_px_list[t])
        return t + 1 < len(ts_list)

    def _exec_ts(t: int):
        if exec_ts_list is not None:
            return exec_ts_list[t]
        return ts_list[t + 1]

    def _exec_px(t: int) -> float:
        if exec_px_list is not None:
            return float(exec_px_list[t])
        return float(px_list[t + 1])

    def _entry_allowed(t: int) -> bool:
        if entry_allowed_list is None:
            return True
        if t >= len(entry_allowed_list):
            return False
        return bool(entry_allowed_list[t])

    def _row(t: int, action: str, exit_reason_detail: str = "") -> dict:
        signal_ts = ts_list[t]
        exec_ts = _exec_ts(t)
        signal_close_ts = (
            signal_close_ts_list[t]
            if signal_close_ts_list is not None and t < len(signal_close_ts_list)
            else signal_ts
        )
        exec_px = _exec_px(t)
        return {
            "symbol":        symbol,
            "action":        action,
            "exit_reason_detail": exit_reason_detail,
            "timestamp":     _coerce_ts(exec_ts),
            "price":         exec_px,
            "strategy_type": "single_asset",
            "z_score":       float(zs_list[t]),
            "signal_timestamp": _coerce_ts(signal_ts),
            "signal_close_timestamp": _coerce_ts(signal_close_ts),
            "signal_price":     float(px_list[t]),
            "execution_timestamp": _coerce_ts(exec_ts),
            "execution_price":     exec_px,
            "execution_timeframe": execution_timeframe or "",
            "reduced_fidelity": bool(reduced_fidelity) if reduced_fidelity is not None else False,
        }

    import math
    for t in range(len(ts_list)):
        z = zs_list[t]
        if math.isnan(z):
            continue

        if position is None:
            if reentry_locked:
                if abs(z) <= exit_z:
                    reentry_locked = False
                else:
                    continue
            if not _entry_allowed(t):
                continue
            if not _can_execute(t):
                continue
            if z <= -entry_z and z > -stop_z:
                signals.append(_row(t, "ENTER_LONG"))
                position = "long"
                entry_price = _exec_px(t)
                entry_candle = t
            elif z >= entry_z and z < stop_z:
                signals.append(_row(t, "ENTER_SHORT"))
                position = "short"
                entry_price = _exec_px(t)
                entry_candle = t

        elif position == "long":
            held = t - entry_candle
            bar_low = float(lo_list[t]) if lo_list is not None and t < len(lo_list) and not math.isnan(lo_list[t]) else float(px_list[t])

            # 1. Absolute % stop (flash crash protection)
            if stop_pct is not None and entry_price and bar_low <= entry_price * (1.0 - stop_pct):
                stop_row = _row(t, "STOP", "STOP")
                stop_fill = _worse_stop_fill_price(
                    entry_price=entry_price,
                    stop_price=float(entry_price * (1.0 - stop_pct)),
                    next_exec_price=_exec_px(t),
                    direction="LONG",
                )
                stop_row["price"] = stop_fill
                stop_row["execution_price"] = stop_fill
                signals.append(stop_row)
                position = entry_price = entry_candle = None

            # 2. Z-score stop (extreme deviation guard)
            elif abs(z) >= stop_z:
                signals.append(_row(t, "STOP", "STOP"))
                position = entry_price = entry_candle = None

            # 3. Time stop
            elif max_holding_candles is not None and held >= max_holding_candles and _can_execute(t):
                signals.append(_row(t, "EXIT", "TIME_EXIT"))
                if require_reset_after_time_stop:
                    reentry_locked = True
                position = entry_price = entry_candle = None

            # 4. Normal mean-reversion exit
            elif z >= -exit_z and _can_execute(t):
                signals.append(_row(t, "EXIT", "MEAN_REVERSION"))
                position = entry_price = entry_candle = None

        elif position == "short":
            held = t - entry_candle
            bar_high = float(hi_list[t]) if hi_list is not None and t < len(hi_list) and not math.isnan(hi_list[t]) else float(px_list[t])

            # 1. Absolute % stop
            if stop_pct is not None and entry_price and bar_high >= entry_price * (1.0 + stop_pct):
                stop_row = _row(t, "STOP", "STOP")
                stop_fill = _worse_stop_fill_price(
                    entry_price=entry_price,
                    stop_price=float(entry_price * (1.0 + stop_pct)),
                    next_exec_price=_exec_px(t),
                    direction="SHORT",
                )
                stop_row["price"] = stop_fill
                stop_row["execution_price"] = stop_fill
                signals.append(stop_row)
                position = entry_price = entry_candle = None

            # 2. Z-score stop
            elif abs(z) >= stop_z:
                signals.append(_row(t, "STOP", "STOP"))
                position = entry_price = entry_candle = None

            # 3. Time stop
            elif max_holding_candles is not None and held >= max_holding_candles and _can_execute(t):
                signals.append(_row(t, "EXIT", "TIME_EXIT"))
                if require_reset_after_time_stop:
                    reentry_locked = True
                position = entry_price = entry_candle = None

            # 4. Normal exit
            elif z <= exit_z and _can_execute(t):
                signals.append(_row(t, "EXIT", "MEAN_REVERSION"))
                position = entry_price = entry_candle = None

    return signals


def compute_halflife(spread: "pd.Series") -> float:
    \"\"\"Estimate Ornstein-Uhlenbeck half-life of mean reversion.

    Fits AR(1): Δspread[t] = λ * spread[t-1] + ε
    half_life = -ln(2) / λ

    Returns float('inf') if series is not mean-reverting (λ >= 0).
    Rule of thumb: useful range is 5–200 candles.
    \"\"\"
    import numpy as np
    vals = np.asarray(spread, dtype=float)
    if len(vals) < 31:
        return float("inf")
    spread_lag  = vals[:-1]
    spread_diff = np.diff(vals)
    X = np.column_stack((np.ones(len(spread_lag)), spread_lag))
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, spread_diff, rcond=None)
        lam = float(coeffs[1])
    except Exception:
        return float("inf")
    return float(-np.log(2) / lam) if lam < 0 else float("inf")


def screen_pairs_cointegration(
    prices: dict,
    p_threshold: float = 0.05,
    lookback: int = 720,
) -> list:
    \"\"\"Screen all symbol pairs for cointegration using Engle-Granger test.

    Args:
        prices: {symbol: DataFrame} as returned by load_price_data()
        p_threshold: cointegration p-value threshold (0.05 = 5%)
        lookback: number of recent candles to use for the test

    Returns list of dicts with keys:
        pair_id, asset_a, asset_b, coint_pvalue, hedge_ratio, halflife

    Only pairs passing p_threshold AND with finite half-life are returned.
    Sorted by coint_pvalue ascending (most cointegrated first).

    Example:
        data = load_price_data(inp["data_dir"], inp["symbols"], inp["timeframe"])
        pairs = screen_pairs_cointegration(data, p_threshold=0.01, lookback=720)
        for p in pairs[:5]:  # top 5
            print(p["pair_id"], p["halflife"])
    \"\"\"
    import numpy as np
    from itertools import combinations
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        raise ImportError("statsmodels is required for screen_pairs_cointegration")

    syms = sorted(prices.keys())
    candidates = []

    for a, b in combinations(syms, 2):
        df_a = prices[a].dropna(subset=["close"]).tail(lookback)
        df_b = prices[b].dropna(subset=["close"]).tail(lookback)
        merged = df_a[["timestamp", "close"]].merge(
            df_b[["timestamp", "close"]], on="timestamp", suffixes=("_a", "_b")
        ).dropna()
        if len(merged) < max(lookback // 2, 60):
            continue

        ya = merged["close_a"].values
        yb = merged["close_b"].values

        if np.std(ya) < 1e-8 or np.std(yb) < 1e-8:
            continue
        try:
            _, pval, _ = coint(ya, yb)
        except Exception:
            continue
        if pval >= p_threshold:
            continue

        # OLS hedge ratio
        X = np.column_stack((np.ones(len(ya)), ya))
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, yb, rcond=None)
            beta = float(coeffs[1])
            intercept = float(coeffs[0])
        except Exception:
            continue

        spread = pd.Series(yb - beta * ya - intercept, index=merged["timestamp"])
        hl = compute_halflife(spread)
        if not np.isfinite(hl):
            continue

        candidates.append({
            "pair_id":     f"{a}/{b}",
            "asset_a":     a,
            "asset_b":     b,
            "coint_pvalue": float(pval),
            "hedge_ratio": beta,
            "intercept":   intercept,
            "halflife":    hl,
        })

    return sorted(candidates, key=lambda x: x["coint_pvalue"])


def rolling_returns(prices: "pd.Series", window: int) -> "pd.Series":
    \"\"\"Rolling percentage return over `window` candles.

    return[t] = (prices[t] / prices[t - window] - 1)
    Returns NaN for the first `window` rows.
    Use for momentum signal construction.
    \"\"\"
    return prices.pct_change(periods=window)


def atr(
    high: "pd.Series",
    low: "pd.Series",
    close: "pd.Series",
    window: int = 14,
) -> "pd.Series":
    \"\"\"Average True Range — standard volatility measure for breakout filters.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = rolling mean of True Range over `window` periods.
    \"\"\"
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def compute_asset_quality_score(
    prices_df: "pd.DataFrame",
    funding_series: "pd.Series",
    funding_z_aligned: "pd.Series",
    candles_per_day_count: int = 6,
    lookback_candles: int = 180,
    funding_lookback: int = 90,
    vol_ratio_max: float = 3.0,
    price_surge_max: float = 0.20,
    funding_spike_max: float = 8.0,
    funding_cv_max: float = 3.5,
    liquidity_ratio_min: float = 0.30,
    edge_win_rate_min: float = 0.52,
    edge_min_samples: int = 5,
    edge_horizon_candles: int = 18,
    entry_z_threshold: float = 2.0,
) -> "pd.DataFrame":
    \"\"\"Per-candle asset quality filter for funding-divergence SHORT strategies.

    Distinguishes 'quality extremes' (temporary crowding → reversion edge) from
    'toxic extremes' (structural pathology → dangerous to short).  Use quality_pass
    as a hard gate BEFORE opening any position on a high funding_z signal.

    Returns a DataFrame (same index as prices_df) with boolean columns:
        vol_regime_ok       — 24h realized vol < vol_ratio_max × 30d baseline
        price_surge_ok      — abs(24h price return) < price_surge_max (no pump)
        liquidity_ok        — current volume > liquidity_ratio_min × rolling avg
        funding_spike_ok    — raw funding rate not a sudden spike vs 90d history
        funding_cv_ok       — funding historically consistent (no chronic spikes)
        historical_edge_ok  — past high-funding SHORTs on this asset were profitable
        quality_pass        — AND of all six filters (the entry gate)

    Parameters
    ----------
    prices_df              OHLCV DataFrame from load_price_data.
    funding_series         Raw 8h rates from load_funding_rates (datetime-indexed).
    funding_z_aligned      Per-candle peer z-score aligned to prices_df rows.
                           Pass funding_z_df[symbol] after ffill to the price index.
    candles_per_day_count  6 for 4h (default), 24 for 1h, 96 for 15m.
    lookback_candles       Rolling window for price-based filters. 180 = 30d on 4h.
    funding_lookback       Rolling window for funding filters in 8h payments. 90 = 30d.
    vol_ratio_max          Reject if vol_24h > vol_ratio_max × vol_30d (default 3.0).
    price_surge_max        Reject if abs(24h return) > this (default 0.20 = 20%).
    funding_spike_max      Reject if rate > funding_spike_max × avg_rate (default 8.0).
    funding_cv_max         Reject if std(rate) / mean(rate) > this (default 3.5).
    liquidity_ratio_min    Reject if volume < this × avg_volume (default 0.30).
    edge_win_rate_min      Require this historical short win rate (default 0.52).
    edge_min_samples       Bypass edge filter if fewer than this past events (default 5).
    edge_horizon_candles   Look-back horizon for historical edge. 18 = 72h on 4h.
    entry_z_threshold      z-score that defines 'high funding' for edge check (2.0).

    No lookahead: all rolling windows use only past data at each candle.

    USAGE:
        prices  = load_price_data(inp["data_dir"], symbols, timeframe, inp["start_date"])
        peer_fz = compute_peer_funding_zscore(inp["data_dir"], peer_symbols, 90)
        price_idx = prices[sym]["timestamp"]
        fz_aligned = peer_fz.reindex(price_idx, method="ffill").fillna(0.0)

        for sym in symbols:
            funding_raw = load_funding_rates(inp["data_dir"], sym)
            quality = compute_asset_quality_score(
                prices_df=prices[sym],
                funding_series=funding_raw,
                funding_z_aligned=fz_aligned[sym],
                candles_per_day_count=candles_per_day(timeframe),
            )
            # Entry gate: funding signal AND quality pass
            entry_mask = (fz_aligned[sym] > 2.0) & quality["quality_pass"]
    \"\"\"
    df     = prices_df.reset_index(drop=True)
    close  = pd.to_numeric(df["close"],  errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    cpd    = candles_per_day_count
    n      = len(df)
    min4   = max(5, lookback_candles // 4)

    # ── 1. Volatility regime ─────────────────────────────────────────────────
    # Reject when short-term vol is anomalously elevated vs the 30d baseline.
    ret      = close.pct_change()
    vol_now  = ret.rolling(cpd, min_periods=max(2, cpd // 2)).std()
    vol_base = ret.rolling(lookback_candles, min_periods=min4).std()
    vol_regime_ok = (vol_now / vol_base.clip(lower=1e-9)) < vol_ratio_max

    # ── 2. Price surge ────────────────────────────────────────────────────────
    # Reject if price moved sharply in 24h — catches meme pumps and events
    # where the funding spike is caused by a structural move, not crowding.
    price_surge_ok = close.pct_change(periods=cpd).abs() < price_surge_max

    # ── 3. Liquidity ──────────────────────────────────────────────────────────
    # Reject if the asset is trading on unusually thin volume.
    avg_vol      = volume.rolling(lookback_candles, min_periods=min4).mean()
    liquidity_ok = (volume / avg_vol.clip(lower=1e-9)) >= liquidity_ratio_min

    # ── 4. Funding spike ──────────────────────────────────────────────────────
    # Align raw 8h funding payments to the price candle index (forward-fill).
    price_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if not funding_series.empty:
        fr = funding_series.reindex(price_ts, method="pad").fillna(0.0)
        fr = pd.Series(fr.values, dtype=float)
    else:
        fr = pd.Series(np.zeros(n), dtype=float)

    abs_fr       = fr.abs()
    min_fr       = max(3, funding_lookback // 4)
    roll_avg_abs = abs_fr.rolling(funding_lookback, min_periods=min_fr).mean()
    # Reject if absolute funding rate is an abrupt multiple of its 90d average.
    # A sudden 8× spike signals a flash event or pathological flow, not crowding.
    funding_spike_ok = (abs_fr / roll_avg_abs.clip(lower=1e-9)) < funding_spike_max

    # ── 5. Funding consistency (coefficient of variation) ────────────────────
    # Reject if the asset exhibits chronic spike-then-zero funding behaviour.
    # High CV = erratic funding = structural toxicity, not mean-reverting crowding.
    roll_std  = abs_fr.rolling(funding_lookback, min_periods=min_fr).std()
    roll_mean = abs_fr.rolling(funding_lookback, min_periods=min_fr).mean()
    funding_cv_ok = (roll_std / roll_mean.clip(lower=1e-9)) < funding_cv_max

    # ── 6. Historical edge (strict no-lookahead) ──────────────────────────────
    # At time t, ask: "when funding_z was high [horizon] candles ago,
    # did price fall over that horizon?" — both quantities are in the past.
    #
    #   was_entry[t]  = funding_z[t − horizon] > threshold  (past signal)
    #   was_winner[t] = close[t] < close[t − horizon]       (past outcome)
    #
    # Rolling win rate = historical success rate of the SHORT thesis.
    # If fewer than edge_min_samples events in the window → insufficient history,
    # so the filter passes optimistically (True) to avoid blocking new assets.
    fz_arr = np.asarray(
        funding_z_aligned.values if hasattr(funding_z_aligned, "values")
        else list(funding_z_aligned),
        dtype=float,
    )
    if len(fz_arr) != n:
        fz_arr = np.zeros(n)

    fz        = pd.Series(fz_arr).fillna(0.0)
    was_entry = (fz.shift(edge_horizon_candles) > entry_z_threshold).astype(float)
    was_win   = (close.pct_change(periods=edge_horizon_candles) < 0.0).astype(float)

    entry_cnt = was_entry.rolling(lookback_candles, min_periods=1).sum()
    win_cnt   = (was_entry * was_win).rolling(lookback_candles, min_periods=1).sum()
    win_rate  = win_cnt / entry_cnt.clip(lower=1.0)
    historical_edge_ok = (win_rate >= edge_win_rate_min) | (entry_cnt < edge_min_samples)

    # ── Composite ─────────────────────────────────────────────────────────────
    quality_pass = (
        vol_regime_ok.fillna(False)
        & price_surge_ok.fillna(False)
        & liquidity_ok.fillna(False)
        & funding_spike_ok.fillna(False)
        & funding_cv_ok.fillna(False)
        & historical_edge_ok.fillna(True)
    )

    return pd.DataFrame(
        {
            "vol_regime_ok":      vol_regime_ok.values,
            "price_surge_ok":     price_surge_ok.values,
            "liquidity_ok":       liquidity_ok.values,
            "funding_spike_ok":   funding_spike_ok.values,
            "funding_cv_ok":      funding_cv_ok.values,
            "historical_edge_ok": historical_edge_ok.values,
            "quality_pass":       quality_pass.values,
        },
        index=prices_df.index,
    )


# ── Typical usage pattern — PAIRS strategy ───────────────────────────────────
# import pandas as pd
# from workspace_helpers import (
#     load_runtime_inputs, load_price_data, candles_per_day,
#     kalman_hedge_ratio, ols_hedge_ratio, rolling_zscore,
#     generate_pair_signals, write_signals_csv, emit_success, emit_blocked,
# )
#
# def main() -> None:
#     inp = load_runtime_inputs()
#     cpd = candles_per_day(inp["timeframe"])  # 24 for 1h
#     try:
#         prices = load_price_data(inp["data_dir"], inp["symbols"], inp["timeframe"], inp["start_date"])
#     except FileNotFoundError as exc:
#         emit_blocked(str(exc), missing_requirements=[str(exc)])
#         return
#
#     signals = []
#     syms = list(prices.keys())
#     for i in range(len(syms)):
#         for j in range(i + 1, len(syms)):
#             a, b = syms[i], syms[j]
#             merged = prices[a][["timestamp","close"]].merge(
#                 prices[b][["timestamp","close"]], on="timestamp", suffixes=("_a","_b")
#             ).dropna()
#             if len(merged) < cpd * 30:
#                 continue
#             hr     = kalman_hedge_ratio(merged["close_a"], merged["close_b"])  # call ONCE — O(n)
#             spread = merged["close_b"] - hr * merged["close_a"]                # b = beta * a
#             z      = rolling_zscore(spread, cpd * 30)
#             signals += generate_pair_signals(
#                 merged["timestamp"], merged["close_a"], merged["close_b"],
#                 hr, spread, z, asset_a=a, asset_b=b,
#             )
#     if not signals:
#         emit_blocked("No signals generated.")
#         return
#     df = pd.DataFrame(signals)
#     write_signals_csv(df, inp["signals_csv"])
#     emit_success(inp["signals_csv"], metrics={"total_signals": len(df)})
#
# if __name__ == "__main__":
#     main()

# ── Typical usage pattern — SINGLE-ASSET strategy (momentum example) ─────────
# import pandas as pd
# from workspace_helpers import (
#     load_runtime_inputs, load_price_data, candles_per_day,
#     rolling_returns, rolling_zscore, atr,
#     generate_single_asset_signals, write_signals_csv, emit_success, emit_blocked,
# )
#
# def main() -> None:
#     inp = load_runtime_inputs()
#     cpd = candles_per_day(inp["timeframe"])
#     try:
#         prices = load_price_data(inp["data_dir"], inp["symbols"], inp["timeframe"], inp["start_date"])
#     except FileNotFoundError as exc:
#         emit_blocked(str(exc), missing_requirements=[str(exc)])
#         return
#
#     signals = []
#     for symbol, df in prices.items():
#         ret    = rolling_returns(df["close"], window=20 * cpd)   # 20-day returns
#         z      = rolling_zscore(ret, window=60 * cpd)            # z-score vs 60d window
#         # Optional regime filter: only trade when ATR is in normal range
#         # atr_series = atr(df["high"], df["low"], df["close"], window=14)
#         signals += generate_single_asset_signals(
#             df["timestamp"], df["close"], z, symbol=symbol,
#             entry_z=1.5, exit_z=0.3, stop_z=3.0,
#         )
#     if not signals:
#         emit_blocked("No signals generated.")
#         return
#     df_out = pd.DataFrame(signals)
#     write_signals_csv(df_out, inp["signals_csv"])
#     emit_success(inp["signals_csv"], metrics={"total_signals": len(df_out)})
#
# if __name__ == "__main__":
#     main()
"""

FORBIDDEN_IMPORT_TOKENS = (
    "requests",
    "httpx",
    "socket",
    "subprocess",
    "websocket",
    "paramiko",
)

FORBIDDEN_CALL_TOKENS = (
    "os.system(",
    "subprocess.",
    "shutil.rmtree(",
    "Path.unlink(",
    ".unlink(",
)

LOOKAHEAD_TOKENS = (
    "shift(-",
    "iloc[i+1",
    "iloc[i + 1",
    ".lead(",
    "rolling(",
)


def create_workspace_dir(experiment_id: str) -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{experiment_id}_", dir=str(WORKSPACE_ROOT))).resolve()


def write_workspace_helpers(workspace_dir: str | Path) -> Path:
    workspace = Path(workspace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    helper_path = workspace / WORKSPACE_HELPERS_FILENAME
    helper_path.write_text(WORKSPACE_HELPERS_CONTENTS, encoding="utf-8")
    return helper_path


def script_hash(contents: str) -> str:
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()[:12]


def dump_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True)


def static_script_checks(contents: str) -> list[str]:
    errors: list[str] = []
    lower = contents.lower()
    for token in FORBIDDEN_IMPORT_TOKENS:
        if f"import {token}" in lower or f"from {token} import" in lower:
            errors.append(f"forbidden import: {token}")
    for token in FORBIDDEN_CALL_TOKENS:
        if token.lower() in lower:
            errors.append(f"forbidden call pattern: {token}")
    errors.extend(structural_script_checks(contents))
    return errors


def normalize_script_chunks(contents: str) -> str:
    normalized = contents.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"([^ \t\n])\b(class\s+[A-Za-z_])", r"\1\n\n\2", normalized)
    normalized = re.sub(r"([^ \t\n])\b(def\s+[A-Za-z_])", r"\1\n\n\2", normalized)
    normalized = re.sub(r"([^ \t\n])\b(if __name__\s*==\s*[\"']__main__[\"'])", r"\1\n\n\2", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def structural_script_checks(contents: str) -> list[str]:
    errors: list[str] = []
    main_count = len(re.findall(r"^\s*def\s+main\s*\(", contents, flags=re.MULTILINE))
    if main_count != 1:
        errors.append(f"script must contain exactly one main() definition, found {main_count}")

    name_main_count = len(re.findall(r'if __name__\s*==\s*["\']__main__["\']', contents))
    if name_main_count != 1:
        errors.append(f"script must contain exactly one __main__ guard, found {name_main_count}")

    if re.search(r"return[^\n]*\b(class|def)\s+[A-Za-z_]", contents):
        errors.append("chunk boundary corruption detected near return/class or return/def")
    if re.search(r"\)[^\S\n]+(class|def)\s+[A-Za-z_]", contents):
        errors.append("missing newline before class/def declaration")

    repeated_mains = contents.count("def main():")
    if repeated_mains > 1:
        errors.append(f"duplicate main() blocks detected: {repeated_mains}")

    main_guard_matches = list(re.finditer(r'if __name__\s*==\s*["\']__main__["\']', contents))
    if main_guard_matches:
        after_guard = contents[main_guard_matches[-1].end():]
        if re.search(r'^\s*(def|class)\s+[A-Za-z_]', after_guard, re.MULTILINE):
            errors.append("orphaned function or class definition after __main__ guard (chunk boundary corruption)")

    return errors


def lookahead_bias_flags(contents: str) -> list[str]:
    compact = contents.replace(" ", "")
    flags: list[str] = []
    if "shift(-" in compact:
        if (
            "execution_timestamp" in contents
            or "execution_price" in contents
            or "signal_close_timestamp" in contents
            or "build_execution_bridge(" in contents
            or "build_next_bar_execution_frame(" in contents
        ):
            flags.append(
                "inline shift(-N) detected near execution alignment; do not hand-roll next-bar logic in the strategy script — use build_execution_bridge() or build_next_bar_execution_frame() from workspace_helpers"
            )
        else:
            flags.append("shift(-N) detected")
    if "iloc[i+1" in compact or "iloc[i+2" in compact:
        flags.append("forward iloc indexing detected")
    if ".rolling(" in contents and ".shift(-" in contents:
        flags.append("rolling window combined with forward shift")
    return flags
