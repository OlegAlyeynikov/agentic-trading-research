"""
Universal backtest engine — supports pairs trading and single-asset strategies.

Fee model: 0.0005 per leg (0.05% futures taker)
  - single_asset: 2 legs (entry + exit)  = 0.10% per round-trip
  - pairs:        4 legs (2 entry + 2 exit) = 0.20% per round-trip

Signal CSV auto-detection:
  - columns {price_a, price_b, hedge_ratio} present  → pairs
  - column strategy_type="pairs"                     → pairs
  - otherwise                                        → single_asset

Result dict is backward-compatible with stat_arb/backtest.py run_backtest() output:
  keys: status, stats, data_quality, pairs (per-symbol breakdown)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


FEE_RATE_DEFAULT = 0.0005  # 0.05% taker per leg
ENTRY_SLIPPAGE_BPS_DEFAULT = 3.0
EXIT_SLIPPAGE_BPS_DEFAULT = 3.0
STOP_SLIPPAGE_BPS_DEFAULT = 8.0

_ACTION_ORDER = {"STOP": 0, "EXIT": 1, "ENTER_LONG": 2, "ENTER_SHORT": 3}


# ── Funding rate helpers ───────────────────────────────────────────────────────

def _find_funding_file(funding_dir: str, symbol: str) -> "str | None":
    from pathlib import Path
    d = Path(funding_dir)
    if not d.exists():
        return None
    matches = sorted(d.glob(f"{symbol}_*.pkl"))
    return str(matches[0]) if matches else None


def _load_funding_series(funding_dir: str, symbol: str) -> "pd.Series | None":
    """Load funding rates as Series indexed by UTC datetime (value = rate per 8h payment)."""
    path = _find_funding_file(funding_dir, symbol)
    if path is None:
        return None
    try:
        df = pd.read_pickle(path)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        return df.set_index("ts")["fundingRate"].sort_index()
    except Exception:
        return None


def _funding_cost_pct(
    entry_T: int,
    exit_T: int,
    direction: str,
    funding_a: "pd.Series | None",
    funding_b: "pd.Series | None" = None,
) -> float:
    """Compute net funding cost in % points (positive = cost to strategy).

    Single asset LONG: pay when rate > 0 → cost = sum(rates) × 100.
    Single asset SHORT: receive when rate > 0 → cost = -sum(rates) × 100.
    Pairs LONG (long A, short B): cost = (sum_A - sum_B) × 100.
    Pairs SHORT (short A, long B): cost = (sum_B - sum_A) × 100.
    """
    if funding_a is None and funding_b is None:
        return 0.0

    entry_dt = pd.Timestamp(entry_T, unit="s", tz="UTC")
    exit_dt = pd.Timestamp(exit_T, unit="s", tz="UTC")

    def _sum(series: "pd.Series | None") -> float:
        if series is None:
            return 0.0
        mask = (series.index > entry_dt) & (series.index <= exit_dt)
        return float(series[mask].sum())

    rate_a = _sum(funding_a)
    rate_b = _sum(funding_b)

    if funding_b is None:
        return (rate_a if direction == "LONG" else -rate_a) * 100.0
    return (rate_a - rate_b if direction == "LONG" else rate_b - rate_a) * 100.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    direction: str        # "LONG" | "SHORT"
    entry_timestamp: str
    exit_timestamp: str
    exit_reason: str      # "EXIT" | "STOP"
    pnl_pct: float
    entry_T: int
    exit_T: int
    slippage_pct: float = 0.0
    exit_reason_detail: str = ""
    asset_a: str = ""
    asset_b: str = ""


@dataclass
class DataQuality:
    orphan_exits: int = 0
    duplicate_enters: int = 0
    open_trades_unclosed: int = 0

    @property
    def is_clean(self) -> bool:
        return (
            self.orphan_exits == 0
            and self.duplicate_enters == 0
            and self.open_trades_unclosed == 0
        )

    def to_dict(self) -> dict:
        return {
            "orphan_exits": self.orphan_exits,
            "duplicate_enters": self.duplicate_enters,
            "open_trades_unclosed": self.open_trades_unclosed,
            "is_clean": self.is_clean,
        }


@dataclass
class SymbolSummary:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    quality: DataQuality = field(default_factory=DataQuality)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_pct > 0) / len(self.trades)

    @property
    def total_pnl_pct(self) -> float:
        return sum(t.pnl_pct for t in self.trades)

    @property
    def avg_pnl_pct(self) -> float:
        return self.total_pnl_pct / self.n_trades if self.trades else 0.0

    @property
    def sharpe(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        pnls = [t.pnl_pct for t in self.trades]
        std = float(np.std(pnls, ddof=1))
        return self.avg_pnl_pct / std if std > 0 else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.trades:
            return 0.0
        equity = np.cumsum([t.pnl_pct for t in self.trades])
        peak = np.maximum.accumulate(equity)
        return float((equity - peak).min())

    @property
    def stop_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.exit_reason == "STOP") / len(self.trades)

    @property
    def time_exit_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.exit_reason_detail == "TIME_EXIT") / len(self.trades)

    @property
    def max_duration_hours(self) -> float:
        durations = [(t.exit_T - t.entry_T) / 3600.0 for t in self.trades if t.exit_T > t.entry_T]
        return max(durations) if durations else 0.0

    def to_dict(self) -> dict:
        exit_pnls = [t.pnl_pct for t in self.trades if t.exit_reason == "EXIT"]
        stop_pnls = [t.pnl_pct for t in self.trades if t.exit_reason == "STOP"]
        legacy_exit_pnls = [t.pnl_pct for t in self.trades if t.exit_reason_detail == "EXIT"]
        time_exit_pnls = [t.pnl_pct for t in self.trades if t.exit_reason_detail == "TIME_EXIT"]
        mean_reversion_pnls = [t.pnl_pct for t in self.trades if t.exit_reason_detail == "MEAN_REVERSION"]
        return {
            "pair_id": self.symbol,
            "trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "avg_pnl_pct": round(self.avg_pnl_pct, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "stop_rate": round(self.stop_rate, 4),
            "time_exit_rate": round(self.time_exit_rate, 4),
            "max_duration_hours": round(self.max_duration_hours, 2),
            "data_quality": self.quality.to_dict(),
            "by_exit_reason": {
                "EXIT": _reason_stats(exit_pnls),
                "STOP": _reason_stats(stop_pnls),
            },
            "by_exit_reason_detail": {
                "EXIT": _reason_stats(legacy_exit_pnls),
                "MEAN_REVERSION": _reason_stats(mean_reversion_pnls),
                "TIME_EXIT": _reason_stats(time_exit_pnls),
                "STOP": _reason_stats(stop_pnls),
            },
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reason_stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"count": 0, "avg_pnl_pct": None, "win_rate": None}
    return {
        "count": len(pnls),
        "avg_pnl_pct": round(float(np.mean(pnls)), 4),
        "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4),
    }


def _sort_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_order"] = df["action"].map(_ACTION_ORDER).fillna(99).astype(int)
    return df.sort_values(["T", "_order"]).drop(columns=["_order"]).reset_index(drop=True)


def _detect_strategy_type(df: pd.DataFrame) -> str:
    if "strategy_type" in df.columns:
        vals = df["strategy_type"].dropna().unique()
        if len(vals) >= 1 and str(vals[0]) == "pairs":
            return "pairs"
    if {"price_a", "price_b", "hedge_ratio"}.issubset(df.columns):
        return "pairs"
    return "single_asset"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_time_value(row: pd.Series, field: str = "timestamp") -> str:
    if field == "timestamp" and "execution_timestamp" in row.index:
        value = row.get("execution_timestamp")
        if pd.notna(value):
            return str(value)
    value = row.get(field, "")
    return "" if pd.isna(value) else str(value)


def _get_numeric_value(
    row: pd.Series, primary: str, fallback: str, default: float = 0.0
) -> float:
    if primary in row.index:
        value = row.get(primary)
        if pd.notna(value):
            return float(value)
    value = row.get(fallback, default)
    return default if pd.isna(value) else float(value)


def _get_exit_reason_detail(row: pd.Series) -> str:
    action = str(row.get("action", ""))
    value = row.get("exit_reason_detail", "")
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    return "STOP" if action == "STOP" else "EXIT"


def _bps_multiplier(bps: float, worsen_up: bool) -> float:
    offset = bps / 10_000.0
    return 1.0 + offset if worsen_up else 1.0 - offset


def _apply_single_asset_fill_price(
    price: float, direction: str, is_entry: bool, slippage_bps: float
) -> float:
    # LONG entry / SHORT exit are buys → pessimistically worse at higher prices.
    is_buy = direction == "LONG" if is_entry else direction == "SHORT"
    return price * _bps_multiplier(slippage_bps, worsen_up=is_buy)


def _apply_pairs_fill_prices(
    pa: float,
    pb: float,
    direction: str,
    is_entry: bool,
    slippage_bps: float,
) -> tuple[float, float]:
    if direction == "LONG":
        # LONG spread = short A + long B
        side_a = "SELL" if is_entry else "BUY"
        side_b = "BUY" if is_entry else "SELL"
    else:
        # SHORT spread = long A + short B
        side_a = "BUY" if is_entry else "SELL"
        side_b = "SELL" if is_entry else "BUY"

    pa_adj = pa * _bps_multiplier(slippage_bps, worsen_up=(side_a == "BUY"))
    pb_adj = pb * _bps_multiplier(slippage_bps, worsen_up=(side_b == "BUY"))
    return pa_adj, pb_adj


# ── Trade extraction ──────────────────────────────────────────────────────────

def _extract_pairs_trades(
    df_group: pd.DataFrame, fee_rate: float
) -> tuple[list[Trade], DataQuality]:
    signals = _sort_signals(df_group)
    trades: list[Trade] = []
    quality = DataQuality()
    open_entry: Optional[pd.Series] = None

    for _, row in signals.iterrows():
        action = row["action"]

        if action in ("ENTER_LONG", "ENTER_SHORT"):
            if open_entry is not None:
                quality.duplicate_enters += 1
                continue
            open_entry = row

        elif action in ("EXIT", "STOP"):
            if open_entry is None:
                quality.orphan_exits += 1
                continue

            direction = "LONG" if open_entry["action"] == "ENTER_LONG" else "SHORT"
            beta = float(open_entry.get("hedge_ratio", 1.0))
            pa_entry = _get_numeric_value(open_entry, "execution_price_a", "price_a")
            pb_entry = _get_numeric_value(
                open_entry, "execution_price_b", "price_b", default=1.0
            )
            pa_exit = _get_numeric_value(row, "execution_price_a", "price_a")
            pb_exit = _get_numeric_value(row, "execution_price_b", "price_b", default=1.0)

            base_spread_entry = pb_entry - beta * pa_entry
            base_spread_exit = pb_exit - beta * pa_exit
            base_pnl_raw = (
                base_spread_exit - base_spread_entry
                if direction == "LONG"
                else base_spread_entry - base_spread_exit
            )
            notional = pb_entry + beta * pa_entry
            fee_pct = 4.0 * fee_rate * 100
            base_pnl_pct = (base_pnl_raw / notional * 100 - fee_pct) if notional > 0 else 0.0

            exit_slippage_bps = (
                STOP_SLIPPAGE_BPS_DEFAULT if action == "STOP" else EXIT_SLIPPAGE_BPS_DEFAULT
            )
            pa_entry_slip, pb_entry_slip = _apply_pairs_fill_prices(
                pa_entry, pb_entry, direction, is_entry=True, slippage_bps=ENTRY_SLIPPAGE_BPS_DEFAULT
            )
            pa_exit_slip, pb_exit_slip = _apply_pairs_fill_prices(
                pa_exit, pb_exit, direction, is_entry=False, slippage_bps=exit_slippage_bps
            )

            spread_entry = pb_entry_slip - beta * pa_entry_slip
            spread_exit = pb_exit_slip - beta * pa_exit_slip
            pnl_raw = (
                spread_exit - spread_entry
                if direction == "LONG"
                else spread_entry - spread_exit
            )
            pnl_pct = (pnl_raw / notional * 100 - fee_pct) if notional > 0 else 0.0
            slippage_pct = max(0.0, base_pnl_pct - pnl_pct)

            sym = str(
                open_entry.get("pair_id", open_entry.get("symbol", "unknown"))
            )
            trades.append(Trade(
                symbol=sym,
                direction=direction,
                entry_timestamp=_get_time_value(open_entry),
                exit_timestamp=_get_time_value(row),
                exit_reason=action,
                pnl_pct=round(pnl_pct, 4),
                entry_T=int(open_entry.get("T", 0)),
                exit_T=int(row.get("T", 0)),
                slippage_pct=round(slippage_pct, 4),
                exit_reason_detail=_get_exit_reason_detail(row),
                asset_a=str(open_entry.get("asset_a", "")),
                asset_b=str(open_entry.get("asset_b", "")),
            ))
            open_entry = None

    if open_entry is not None:
        quality.open_trades_unclosed += 1

    return trades, quality


def _extract_single_asset_trades(
    df_group: pd.DataFrame, fee_rate: float
) -> tuple[list[Trade], DataQuality]:
    signals = _sort_signals(df_group)
    trades: list[Trade] = []
    quality = DataQuality()
    open_entry: Optional[pd.Series] = None

    for _, row in signals.iterrows():
        action = row["action"]

        if action in ("ENTER_LONG", "ENTER_SHORT"):
            if open_entry is not None:
                quality.duplicate_enters += 1
                continue
            open_entry = row

        elif action in ("EXIT", "STOP"):
            if open_entry is None:
                quality.orphan_exits += 1
                continue

            direction = "LONG" if open_entry["action"] == "ENTER_LONG" else "SHORT"
            entry_price = _get_numeric_value(open_entry, "execution_price", "price", default=1.0)
            exit_price = _get_numeric_value(row, "execution_price", "price", default=1.0)
            base_sign = 1.0 if direction == "LONG" else -1.0
            base_gross_pct = (exit_price / entry_price - 1.0) * base_sign * 100.0
            fee_pct = 2.0 * fee_rate * 100  # entry leg + exit leg
            base_pnl_pct = base_gross_pct - fee_pct

            exit_slippage_bps = (
                STOP_SLIPPAGE_BPS_DEFAULT if action == "STOP" else EXIT_SLIPPAGE_BPS_DEFAULT
            )
            entry_price_slip = _apply_single_asset_fill_price(
                entry_price, direction, is_entry=True, slippage_bps=ENTRY_SLIPPAGE_BPS_DEFAULT
            )
            exit_price_slip = _apply_single_asset_fill_price(
                exit_price, direction, is_entry=False, slippage_bps=exit_slippage_bps
            )

            sign = 1.0 if direction == "LONG" else -1.0
            gross_pct = (exit_price_slip / entry_price_slip - 1.0) * sign * 100.0
            pnl_pct = gross_pct - fee_pct
            slippage_pct = max(0.0, base_pnl_pct - pnl_pct)

            trades.append(Trade(
                symbol=str(open_entry.get("symbol", "unknown")),
                direction=direction,
                entry_timestamp=_get_time_value(open_entry),
                exit_timestamp=_get_time_value(row),
                exit_reason=action,
                pnl_pct=round(pnl_pct, 4),
                entry_T=int(open_entry.get("T", 0)),
                exit_T=int(row.get("T", 0)),
                slippage_pct=round(slippage_pct, 4),
                exit_reason_detail=_get_exit_reason_detail(row),
            ))
            open_entry = None

    if open_entry is not None:
        quality.open_trades_unclosed += 1

    return trades, quality


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def _aggregate_stats(all_trades: list[Trade], risk_pct_per_trade: float = 1.0) -> dict:
    if not all_trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "total_pnl_pct": 0.0,
            "sum_trade_pnl_pct": 0.0, "estimated_equity_return_pct": 0.0,
            "avg_pnl_pct": 0.0, "profit_factor": 0.0,
            "max_drawdown_pct": 0.0, "max_drawdown_equity_pct": 0.0,
            "avg_duration_hours": 0.0, "median_duration_hours": 0.0,
            "max_duration_hours": 0.0, "p95_duration_hours": 0.0,
            "time_exit_rate": 0.0, "by_exit_reason": {}, "by_exit_reason_detail": {},
            "slippage_paid_pct": 0.0, "slippage_paid_equity_pct": 0.0,
        }
    pnls = [t.pnl_pct for t in all_trades]
    wins = sum(1 for p in pnls if p > 0)
    gross_profit = float(sum(p for p in pnls if p > 0))
    gross_loss = float(abs(sum(p for p in pnls if p < 0)))
    if gross_loss == 0.0:
        profit_factor = 999.0 if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min())
    risk = max(float(risk_pct_per_trade or 0.0), 0.0)
    exit_pnls = [t.pnl_pct for t in all_trades if t.exit_reason == "EXIT"]
    stop_pnls = [t.pnl_pct for t in all_trades if t.exit_reason == "STOP"]
    legacy_exit_pnls = [t.pnl_pct for t in all_trades if t.exit_reason_detail == "EXIT"]
    time_exit_pnls = [t.pnl_pct for t in all_trades if t.exit_reason_detail == "TIME_EXIT"]
    mean_reversion_pnls = [t.pnl_pct for t in all_trades if t.exit_reason_detail == "MEAN_REVERSION"]
    total_slippage = float(sum(t.slippage_pct for t in all_trades))
    durations_h = [(t.exit_T - t.entry_T) / 3600.0 for t in all_trades if t.exit_T > t.entry_T]
    avg_dur = round(float(np.mean(durations_h)), 2) if durations_h else 0.0
    sorted_durs = sorted(durations_h)
    n = len(sorted_durs)
    median_dur = round(float((sorted_durs[n // 2] + sorted_durs[(n - 1) // 2]) / 2), 2) if sorted_durs else 0.0
    p95_dur = round(float(np.percentile(durations_h, 95)), 2) if durations_h else 0.0
    max_dur = round(float(max(durations_h)), 2) if durations_h else 0.0
    sum_trade_pnl = float(np.sum(pnls))
    return {
        "total_trades": len(all_trades),
        "win_rate": round(wins / len(all_trades), 4),
        "total_pnl_pct": round(sum_trade_pnl, 4),
        "sum_trade_pnl_pct": round(sum_trade_pnl, 4),
        "estimated_equity_return_pct": round(sum_trade_pnl * risk, 4),
        "avg_pnl_pct": round(float(np.mean(pnls)), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown_pct": round(max_dd, 4),
        "max_drawdown_equity_pct": round(max_dd * risk, 4),
        "avg_duration_hours": avg_dur,
        "median_duration_hours": median_dur,
        "max_duration_hours": max_dur,
        "p95_duration_hours": p95_dur,
        "time_exit_rate": round(len(time_exit_pnls) / len(all_trades), 4),
        "slippage_paid_pct": round(total_slippage, 4),
        "slippage_paid_equity_pct": round(total_slippage * risk, 4),
        "by_exit_reason": {
            "EXIT": _reason_stats(exit_pnls),
            "STOP": _reason_stats(stop_pnls),
        },
        "by_exit_reason_detail": {
            "EXIT": _reason_stats(legacy_exit_pnls),
            "MEAN_REVERSION": _reason_stats(mean_reversion_pnls),
            "TIME_EXIT": _reason_stats(time_exit_pnls),
            "STOP": _reason_stats(stop_pnls),
        },
    }


def _aggregate_quality(summaries: list[SymbolSummary]) -> dict:
    return {
        "orphan_exits": sum(s.quality.orphan_exits for s in summaries),
        "duplicate_enters": sum(s.quality.duplicate_enters for s in summaries),
        "open_trades_unclosed": sum(s.quality.open_trades_unclosed for s in summaries),
        "is_clean": all(s.quality.is_clean for s in summaries),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_backtest(
    signals_csv: str,
    fee_rate: float = FEE_RATE_DEFAULT,
    funding_dir: "str | None" = None,
    risk_pct_per_trade: float = 1.0,
) -> dict:
    """Run backtest on a signals CSV produced by a workspace strategy script.

    Auto-detects strategy type from CSV columns:
      - {price_a, price_b, hedge_ratio} present OR strategy_type="pairs" → pairs
      - otherwise → single_asset

    funding_dir: optional path to funding rate pkl files. If provided, funding
    costs are deducted from each trade's pnl_pct and reported as funding_paid_pct.

    Returns a dict compatible with stat_arb/backtest.py for downstream consumers
    (reviewer, reporter, executor metrics).
    """
    started_at = _utc_now_iso()

    path = Path(signals_csv)
    if not path.exists():
        return {
            "status": "blocked",
            "stats": {"total_trades": 0, "win_rate": 0.0, "total_pnl_pct": 0.0,
                      "sum_trade_pnl_pct": 0.0, "estimated_equity_return_pct": 0.0,
                      "avg_pnl_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0,
                      "by_exit_reason": {}},
            "data_quality": {"orphan_exits": 0, "duplicate_enters": 0,
                             "open_trades_unclosed": 0, "is_clean": False},
            "pairs": [],
            "error": f"signals CSV not found: {signals_csv}",
        }

    df = pd.read_csv(signals_csv)
    total_signals = len(df)

    if total_signals == 0 or "action" not in df.columns:
        finished_at = _utc_now_iso()
        return {
            "status": "success",
            "started_at": started_at,
            "finished_at": finished_at,
            "strategy_type": "unknown",
            "input": {"csv_path": str(path.resolve()), "fee_rate": fee_rate, "total_signals": 0},
            "stats": {"total_trades": 0, "win_rate": 0.0, "total_pnl_pct": 0.0,
                      "sum_trade_pnl_pct": 0.0, "estimated_equity_return_pct": 0.0,
                      "avg_pnl_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0,
                      "by_exit_reason": {}},
            "data_quality": {"orphan_exits": 0, "duplicate_enters": 0,
                             "open_trades_unclosed": 0, "is_clean": True},
            "pairs": [],
        }

    # Prefer explicit execution timestamps when present.
    timestamp_col = "execution_timestamp" if "execution_timestamp" in df.columns else "timestamp"

    # Ensure T column (unix seconds) exists
    if "T" not in df.columns:
        df["T"] = (
            pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64") // 1_000_000_000
        )

    strategy_type = _detect_strategy_type(df)
    extract_fn = _extract_pairs_trades if strategy_type == "pairs" else _extract_single_asset_trades

    # Determine grouping column
    if strategy_type == "pairs" and "pair_id" in df.columns:
        group_col = "pair_id"
    elif "symbol" in df.columns:
        group_col = "symbol"
    else:
        group_col = None

    summaries: list[SymbolSummary] = []
    all_trades: list[Trade] = []

    if group_col and group_col in df.columns:
        for grp_key, group in df.groupby(group_col):
            trades, quality = extract_fn(group, fee_rate)
            summaries.append(SymbolSummary(symbol=str(grp_key), trades=trades, quality=quality))
            all_trades.extend(trades)
    else:
        trades, quality = extract_fn(df, fee_rate)
        summaries.append(SymbolSummary(symbol="portfolio", trades=trades, quality=quality))
        all_trades.extend(trades)

    # ── Apply funding cost adjustments ──────────────────────────────────────
    total_funding_paid = 0.0
    if funding_dir:
        _funding_cache: dict[str, "pd.Series | None"] = {}

        def _get_funding(sym: str) -> "pd.Series | None":
            if sym not in _funding_cache:
                _funding_cache[sym] = _load_funding_series(funding_dir, sym)
            return _funding_cache[sym]

        for trade in all_trades:
            sym = trade.symbol
            if trade.asset_a and trade.asset_b:
                parts = [trade.asset_a, trade.asset_b]
                fa = _get_funding(parts[0])
                fb = _get_funding(parts[1])
                cost = _funding_cost_pct(trade.entry_T, trade.exit_T, trade.direction, fa, fb)
            elif "-" in sym or "/" in sym:
                sep = "-" if "-" in sym else "/"
                parts = sym.split(sep, 1)
                fa = _get_funding(parts[0])
                fb = _get_funding(parts[1])
                cost = _funding_cost_pct(trade.entry_T, trade.exit_T, trade.direction, fa, fb)
            else:
                fa = _get_funding(sym)
                cost = _funding_cost_pct(trade.entry_T, trade.exit_T, trade.direction, fa)
            trade.pnl_pct = round(trade.pnl_pct - cost, 4)
            total_funding_paid += cost

    finished_at = _utc_now_iso()
    legs = 4 if strategy_type == "pairs" else 2
    stats = _aggregate_stats(all_trades, risk_pct_per_trade=risk_pct_per_trade)
    if funding_dir:
        stats["funding_paid_pct"] = round(total_funding_paid, 4)
        stats["funding_paid_equity_pct"] = round(total_funding_paid * max(float(risk_pct_per_trade or 0.0), 0.0), 4)

    return {
        "status": "success",
        "started_at": started_at,
        "finished_at": finished_at,
        "strategy_type": strategy_type,
        "input": {
            "csv_path": str(path.resolve()),
            "fee_rate": fee_rate,
            "fee_pct_per_roundtrip": round(fee_rate * legs * 100, 4),
            "entry_slippage_bps": ENTRY_SLIPPAGE_BPS_DEFAULT,
            "exit_slippage_bps": EXIT_SLIPPAGE_BPS_DEFAULT,
            "stop_slippage_bps": STOP_SLIPPAGE_BPS_DEFAULT,
            "total_signals": total_signals,
            "risk_pct_per_trade": risk_pct_per_trade,
            "execution_timestamp_column": timestamp_col,
            "uses_explicit_execution_fields": bool(
                "execution_timestamp" in df.columns or "execution_price" in df.columns
            ),
        },
        "stats": stats,
        "data_quality": _aggregate_quality(summaries),
        "pairs": [
            s.to_dict()
            for s in sorted(summaries, key=lambda x: x.total_pnl_pct, reverse=True)
        ],
    }
