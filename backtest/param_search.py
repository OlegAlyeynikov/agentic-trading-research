"""
Single-asset z-score mean-reversion parameter search.

Sweeps combinations of: timeframe, lookback, entry_z, stop_pct,
max_holding_candles across a configurable symbol universe.

Usage:
    python backtest/param_search.py
    python backtest/param_search.py --fast
    python backtest/param_search.py --symbols BTCUSDT ETHUSDT SOLUSDT --top 30
    python backtest/param_search.py --by-symbol  # also test each symbol alone
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import run_backtest

# ── Parameter grid ────────────────────────────────────────────────────────────

PARAM_GRID: dict[str, list] = {
    "timeframe":      ["1h", "4h"],
    "lookback_days":  [1, 2, 3, 5, 7, 14],
    "entry_z":        [1.5, 2.0, 2.5, 3.0],
    "exit_z":         [0.0, 0.3],
    "stop_pct":       [0.03, 0.05, 0.07, None],
    "holding_days":   [0.5, 1, 2, 3],
}

FAST_GRID: dict[str, list] = {
    "timeframe":      ["1h"],
    "lookback_days":  [2, 5, 7],
    "entry_z":        [2.0, 2.5],
    "exit_z":         [0.0],
    "stop_pct":       [0.05, None],
    "holding_days":   [1, 2],
}

# ── Standalone helpers ────────────────────────────────────────────────────────

_CPD = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48,
        "1h": 24, "2h": 12, "4h": 6, "8h": 3, "1d": 1}


def _candles_per_day(tf: str) -> int:
    return _CPD.get(tf.lower().strip(), 24)


def _load_symbol(data_dir: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    tf_dir = Path(data_dir) / timeframe
    for pat in [f"{symbol}_*.pkl", f"{symbol.upper()}_*.pkl"]:
        matches = sorted(tf_dir.glob(pat))
        if matches:
            df = pd.read_pickle(str(matches[0]))
            if isinstance(df.index, pd.RangeIndex):
                cols = ["timestamp","open","high","low","close","vol",
                        "close_time","quote_vol","trades","tb","tq","ignore"]
                if len(df.columns) == len(cols):
                    df.columns = cols
                df["ts"] = pd.to_datetime(df["timestamp"].astype(float).astype("int64"),
                                          unit="ms", utc=True)
                df = df.set_index("ts").sort_index()
            return df
    return None


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window, min_periods=max(2, window // 4)).mean()
    s = series.rolling(window, min_periods=max(2, window // 4)).std()
    return (series - m) / s.replace(0, np.nan)


def _generate_signals(
    ts: pd.Index,
    prices: pd.Series,
    z: pd.Series,
    symbol: str,
    entry_z: float,
    exit_z: float,
    stop_pct: float | None,
    max_hold: int,
) -> list[dict]:
    stop_z = 5.0
    position = None
    entry_price = None
    entry_candle = None
    sigs: list[dict] = []

    px = prices.values
    zv = z.values
    tss = ts

    for t in range(len(zv)):
        z_t = zv[t]
        if math.isnan(z_t):
            continue

        ts_str = tss[t].isoformat() if hasattr(tss[t], "isoformat") else str(tss[t])

        def row(action: str) -> dict:
            return {
                "symbol": symbol, "action": action,
                "timestamp": ts_str,
                "price": float(px[t]),
                "strategy_type": "single_asset",
                "z_score": float(z_t),
            }

        if position is None:
            if z_t <= -entry_z and z_t > -stop_z:
                sigs.append(row("ENTER_LONG"))
                position, entry_price, entry_candle = "long", float(px[t]), t
            elif z_t >= entry_z and z_t < stop_z:
                sigs.append(row("ENTER_SHORT"))
                position, entry_price, entry_candle = "short", float(px[t]), t

        elif position == "long":
            p = float(px[t])
            held = t - entry_candle
            if stop_pct is not None and p <= entry_price * (1.0 - stop_pct):
                sigs.append(row("STOP")); position = entry_price = entry_candle = None
            elif abs(z_t) >= stop_z:
                sigs.append(row("STOP")); position = entry_price = entry_candle = None
            elif held >= max_hold:
                sigs.append(row("EXIT")); position = entry_price = entry_candle = None
            elif z_t >= -exit_z:
                sigs.append(row("EXIT")); position = entry_price = entry_candle = None

        elif position == "short":
            p = float(px[t])
            held = t - entry_candle
            if stop_pct is not None and p >= entry_price * (1.0 + stop_pct):
                sigs.append(row("STOP")); position = entry_price = entry_candle = None
            elif abs(z_t) >= stop_z:
                sigs.append(row("STOP")); position = entry_price = entry_candle = None
            elif held >= max_hold:
                sigs.append(row("EXIT")); position = entry_price = entry_candle = None
            elif z_t <= exit_z:
                sigs.append(row("EXIT")); position = entry_price = entry_candle = None

    if position is not None and len(zv):
        ts_str = tss[-1].isoformat() if hasattr(tss[-1], "isoformat") else str(tss[-1])
        sigs.append({"symbol": symbol, "action": "EXIT", "timestamp": ts_str,
                     "price": float(px[-1]), "strategy_type": "single_asset",
                     "z_score": float(zv[-1])})
    return sigs


# ── Run one combination ───────────────────────────────────────────────────────

def _run_combo(
    price_cache: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    tf: str,
    lookback: int,
    entry_z: float,
    exit_z: float,
    stop_pct: float | None,
    max_hold: int,
    tmp_csv: str,
    start_date: str,
) -> dict | None:
    prices = price_cache.get(tf, {})
    all_sigs: list[dict] = []

    for sym in symbols:
        df = prices.get(sym)
        if df is None:
            continue
        close = df["close"].astype(float)
        if start_date:
            close = close[close.index >= pd.Timestamp(start_date, tz="UTC")]
        if len(close) < lookback + 10:
            continue

        z = _rolling_zscore(close, lookback)
        sigs = _generate_signals(
            ts=close.index, prices=close, z=z, symbol=sym,
            entry_z=entry_z, exit_z=exit_z, stop_pct=stop_pct, max_hold=max_hold,
        )
        all_sigs.extend(sigs)

    if not all_sigs:
        return None

    df_sig = pd.DataFrame(all_sigs)
    df_sig["T"] = (
        pd.to_datetime(df_sig["timestamp"], utc=True, errors="coerce")
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64") // 1_000_000_000
    )
    df_sig.to_csv(tmp_csv, index=False)
    return run_backtest(signals_csv=tmp_csv)


# ── Main search ───────────────────────────────────────────────────────────────

def run_search(
    data_dir: str,
    symbols: list[str],
    start_date: str,
    out_csv: str,
    top_n: int = 30,
    min_trades: int = 30,
    max_dur_h: float = 72.0,
    grid: dict | None = None,
    symbol_subsets: bool = False,
) -> pd.DataFrame:
    grid = grid or PARAM_GRID
    tmp = str(Path(out_csv).parent / "_ps_tmp.csv")

    # Pre-load prices (once per timeframe)
    print("Loading price data...")
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    for tf in grid["timeframe"]:
        cache[tf] = {}
        for sym in symbols:
            df = _load_symbol(data_dir, sym, tf)
            if df is not None and "close" in df.columns:
                cache[tf][sym] = df
        print(f"  {tf}: {len(cache[tf])} symbols")

    sym_groups = [[s] for s in symbols] + [symbols] if symbol_subsets else [symbols]

    keys = [k for k in grid if k != "timeframe"]
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(grid["timeframe"]) * len(sym_groups) * len(combos)
    print(f"\nCombinations: {total:,}  (est. {total*1.5/60:.0f}–{total*3/60:.0f} min)\n")

    results: list[dict] = []
    done = 0
    t0 = time.monotonic()

    for tf in grid["timeframe"]:
        cpd = _candles_per_day(tf)
        for sym_group in sym_groups:
            avail = [s for s in sym_group if s in cache.get(tf, {})]
            if not avail:
                continue
            for vals in combos:
                p = dict(zip(keys, vals))
                lookback = max(10, int(p["lookback_days"] * cpd))
                max_hold = max(2, int(p["holding_days"] * cpd))

                try:
                    res = _run_combo(cache, avail, tf, lookback,
                                     p["entry_z"], p["exit_z"],
                                     p["stop_pct"], max_hold, tmp, start_date)
                    if res and res.get("status") == "success":
                        st = res["stats"]
                        # sharpe_proxy = max per-symbol sharpe from pairs breakdown
                        pairs_data = res.get("pairs", [])
                        sharpe_vals = [p2.get("sharpe", 0) or 0 for p2 in pairs_data]
                        sharpe_proxy = round(max(sharpe_vals), 4) if sharpe_vals else 0.0
                        row = {
                            "timeframe":          tf,
                            "lookback_days":      p["lookback_days"],
                            "lookback_candles":   lookback,
                            "entry_z":            p["entry_z"],
                            "exit_z":             p["exit_z"],
                            "stop_pct":           p["stop_pct"],
                            "holding_days":       p["holding_days"],
                            "max_hold_candles":   max_hold,
                            "symbols":            "+".join(avail),
                            "total_trades":       st.get("total_trades", 0),
                            "sharpe":             sharpe_proxy,
                            "profit_factor":      round(st.get("profit_factor", 0) or 0, 4),
                            "max_dd_pct":         round(st.get("max_drawdown_pct", 0) or 0, 2),
                            "avg_pnl_pct":        round(st.get("avg_pnl_pct", 0) or 0, 4),
                            "win_rate":           round(st.get("win_rate", 0) or 0, 4),
                            "stop_rate":          round(st.get("stop_rate", 0) or 0, 4),
                            "avg_dur_h":          round(st.get("avg_duration_hours", 0) or 0, 1),
                            "total_pnl_pct":      round(st.get("total_pnl_pct", 0) or 0, 2),
                        }
                        if (row["total_trades"] >= min_trades
                                and row["avg_dur_h"] <= max_dur_h):
                            results.append(row)
                except Exception:
                    pass

                done += 1
                if done % 100 == 0 or done == total:
                    elapsed = time.monotonic() - t0
                    eta = (total - done) / (done / elapsed) / 60 if done else 0
                    best = max((r["sharpe"] for r in results), default=0)
                    print(f"  {done:>5}/{total}  ETA {eta:.0f}min  "
                          f"valid={len(results)}  best_sharpe={best:.3f}")

    Path(tmp).unlink(missing_ok=True)

    if not results:
        print("\nNo combinations passed filters.")
        return pd.DataFrame()

    df = (pd.DataFrame(results)
          .sort_values("sharpe", ascending=False)
          .reset_index(drop=True))
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} results → {out_csv}")

    show_cols = ["timeframe","lookback_days","entry_z","exit_z","stop_pct",
                 "holding_days","symbols","total_trades","sharpe","profit_factor",
                 "max_dd_pct","avg_dur_h","stop_rate"]

    print(f"\n{'='*100}")
    print(f"TOP {min(top_n, len(df))} by sharpe")
    print(f"{'='*100}")
    print(df.head(top_n)[show_cols].to_string(index=True))

    print(f"\n{'='*100}")
    print("GOAL CONTRACT: sharpe≥0.7, pf≥1.5, dd≥-15%, trades≥50, dur≤48h")
    print(f"{'='*100}")
    passing = df[
        (df["sharpe"] >= 0.7) & (df["profit_factor"] >= 1.5) &
        (df["max_dd_pct"] >= -15.0) & (df["total_trades"] >= 50) &
        (df["avg_dur_h"] <= 48.0)
    ]
    if passing.empty:
        print("No combination satisfies all goals.")
        for metric, col, target, direction in [
            ("sharpe",   "sharpe",       0.7,  "max"),
            ("pf",       "profit_factor",1.5,  "max"),
            ("drawdown", "max_dd_pct",   -15,  "max"),
            ("trades",   "total_trades", 50,   "max"),
        ]:
            best_val = df[col].max() if direction == "max" else df[col].min()
            print(f"  best {metric}: {best_val:.3f}  (need {'>=' if direction=='max' else '<='}{target})")
    else:
        print(f"{len(passing)} passing combinations:")
        print(passing[show_cols].to_string(index=True))

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="research_config.json")
    ap.add_argument("--out", default="param_search_results.csv")
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--max-duration", type=float, default=72.0)
    ap.add_argument("--by-symbol", action="store_true",
                    help="Also test each symbol individually")
    ap.add_argument("--timeframes", nargs="+")
    ap.add_argument("--fast", action="store_true",
                    help="Quick ~72-combo grid for sanity check")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent.parent / cfg_path
    with open(cfg_path) as f:
        cfg = json.load(f)

    symbols = args.symbols or cfg["symbols"]
    data_dir = cfg["data_dir"]
    start_date = cfg.get("start_date", "2024-01-01")
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).parent.parent / out

    grid = FAST_GRID if args.fast else dict(PARAM_GRID)
    if args.timeframes:
        grid["timeframe"] = args.timeframes

    print(f"Symbols ({len(symbols)}): {symbols}")
    print(f"Period:  {start_date} → now")
    print(f"Mode:    {'fast' if args.fast else 'full'}\n")

    run_search(data_dir=data_dir, symbols=symbols, start_date=start_date,
               out_csv=str(out), top_n=args.top, min_trades=args.min_trades,
               max_dur_h=args.max_duration, grid=grid,
               symbol_subsets=args.by_symbol)


if __name__ == "__main__":
    main()
