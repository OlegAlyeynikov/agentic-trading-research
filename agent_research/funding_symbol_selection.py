from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent_research.code_runtime import write_workspace_helpers
from agent_research.config_utils import compute_research_score, load_full_config, load_goal_contract
from backtest.engine import run_backtest


def _parse_csv(value: str | None) -> list[str]:
    return [item.strip().upper() for item in (value or "").split(",") if item.strip()]


def _load_workspace_helpers(output_dir: Path):
    runtime_dir = output_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_workspace_helpers(runtime_dir)
    helper_path = runtime_dir / "workspace_helpers.py"
    spec = importlib.util.spec_from_file_location("funding_symbol_selection_workspace_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import workspace helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_position_limit(signals_df: pd.DataFrame, max_open_positions: int = 1) -> pd.DataFrame:
    if signals_df.empty or max_open_positions <= 0:
        return signals_df
    df = signals_df.copy()
    if "T" not in df.columns:
        ts_ns = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        df["T"] = ts_ns // 1_000_000_000
    group_col = "pair_id" if "pair_id" in df.columns else "symbol"
    action_order = {"STOP": 0, "EXIT": 1, "ENTER_LONG": 2, "ENTER_SHORT": 3}
    df["_action_order"] = df["action"].map(action_order).fillna(99).astype(int)
    df["_zabs"] = pd.to_numeric(df.get("z_score", 0.0), errors="coerce").fillna(0.0).abs()
    df = df.sort_values(["T", "_action_order", "_zabs"], ascending=[True, True, False]).reset_index(drop=True)

    open_groups: set[str] = set()
    keep: list[bool] = []
    for _, row in df.iterrows():
        action = str(row["action"])
        group = str(row.get(group_col, ""))
        if action in {"EXIT", "STOP"}:
            should_keep = group in open_groups
            keep.append(should_keep)
            if should_keep:
                open_groups.discard(group)
        elif action in {"ENTER_LONG", "ENTER_SHORT"}:
            should_keep = group not in open_groups and len(open_groups) < max_open_positions
            keep.append(should_keep)
            if should_keep:
                open_groups.add(group)
        else:
            keep.append(True)
    return df[keep].drop(columns=["_action_order", "_zabs"]).reset_index(drop=True)


def _symbol_from_data_file(path: Path) -> str:
    return path.name.split("_", 1)[0].upper()


def _dates_from_data_file(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    stem = path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return None, None
    start = pd.to_datetime(parts[-2], format="%d-%m-%Y", utc=True, errors="coerce")
    end = pd.to_datetime(parts[-1], format="%d-%m-%Y", utc=True, errors="coerce")
    return (None if pd.isna(start) else start, None if pd.isna(end) else end)


def _discover_symbols(data_dir: str, timeframe: str, execution_timeframe: str, funding_dir: str) -> list[str]:
    root = Path(data_dir)
    signal_symbols = {
        _symbol_from_data_file(p)
        for p in (root / timeframe).glob("*")
        if p.suffix in {".pkl", ".parquet"}
    }
    execution_symbols = {
        _symbol_from_data_file(p)
        for p in (root / execution_timeframe).glob("*")
        if p.suffix in {".pkl", ".parquet"}
    }
    funding_symbols = {_symbol_from_data_file(p) for p in Path(funding_dir).glob("*.pkl")}
    return sorted(signal_symbols & execution_symbols & funding_symbols)


def _file_metadata_by_symbol(directory: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for path in directory.glob("*"):
        if path.suffix not in {".pkl", ".parquet"}:
            continue
        sym = _symbol_from_data_file(path)
        start, end = _dates_from_data_file(path)
        days = float((end - start).total_seconds() / 86400.0) if start is not None and end is not None else 0.0
        metadata[sym] = {
            "ok": start is not None and end is not None,
            "file": str(path),
            "start": start.isoformat() if start is not None else "",
            "end": end.isoformat() if end is not None else "",
            "days": days,
        }
    return metadata


def _funding_file_metadata(funding_dir: str) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for path in Path(funding_dir).glob("*.pkl"):
        sym = _symbol_from_data_file(path)
        start, end = _dates_from_data_file(path)
        days = float((end - start).total_seconds() / 86400.0) if start is not None and end is not None else 0.0
        metadata[sym] = {
            "ok": start is not None and end is not None,
            "file": str(path),
            "start": start.isoformat() if start is not None else "",
            "end": end.isoformat() if end is not None else "",
            "days": days,
            "approx_funding_rows": int(days * 3) if days else 0,
        }
    return metadata


def _generate_symbol_signals(
    *,
    helpers,
    data_dir: str,
    available_timeframes: list[str],
    start_date: str,
    timeframe: str,
    direction: str,
    symbol: str,
    funding_z: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    stop_pct: float,
    max_holding_candles: int,
    require_carry_sign: bool,
) -> pd.DataFrame:
    prices = helpers.load_price_data(data_dir, [symbol], timeframe, start_date)
    exec_prices, exec_tf, reduced_fidelity = helpers.load_execution_price_data(
        data_dir, [symbol], available_timeframes, start_date
    )
    price_df = prices[symbol]
    exec_df = exec_prices[symbol]
    price_ts = pd.to_datetime(price_df["timestamp"], utc=True, errors="coerce")

    raw_z = funding_z[symbol].reindex(price_ts, method="ffill")
    z = raw_z.fillna(0.0).reset_index(drop=True)
    funding = helpers.load_funding_rates(data_dir, symbol).reindex(price_ts, method="ffill").reset_index(drop=True)

    if direction == "short_high":
        entry_allowed = funding.gt(0) if require_carry_sign else pd.Series(True, index=z.index)
    elif direction == "long_low":
        entry_allowed = funding.lt(0) if require_carry_sign else pd.Series(True, index=z.index)
    else:
        raise ValueError("direction must be short_high or long_low")

    bridge = helpers.build_execution_bridge(price_df, exec_df, timeframe, exec_tf)
    signals = helpers.generate_single_asset_signals(
        timestamps=price_ts.reset_index(drop=True),
        prices=price_df["close"].reset_index(drop=True),
        z_scores=z,
        symbol=symbol,
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        stop_z=99.0,
        stop_pct=float(stop_pct),
        max_holding_candles=int(max_holding_candles),
        execution_timestamps=bridge["execution_timestamp"],
        execution_prices=bridge["execution_price"],
        stop_high=bridge["stop_high"],
        stop_low=bridge["stop_low"],
        signal_close_timestamps=bridge["signal_close_timestamp"],
        execution_timeframe=exec_tf,
        reduced_fidelity=reduced_fidelity,
        entry_allowed_mask=entry_allowed.fillna(False),
    )
    allowed_actions = {"EXIT", "STOP", "ENTER_SHORT" if direction == "short_high" else "ENTER_LONG"}
    return pd.DataFrame([s for s in signals if s["action"] in allowed_actions])


def _metrics_from_result(result: dict, goal_contract: dict) -> dict[str, Any]:
    stats = result.get("stats") or {}
    pairs = result.get("pairs") or []
    sharpe_vals = [float(p.get("sharpe") or 0.0) for p in pairs if p.get("sharpe") is not None]
    metrics = {
        "total_trades": int(stats.get("total_trades") or 0),
        "win_rate": float(stats.get("win_rate") or 0.0),
        "sum_trade_pnl_pct": float(stats.get("sum_trade_pnl_pct") or stats.get("total_pnl_pct") or 0.0),
        "estimated_equity_return_pct": float(stats.get("estimated_equity_return_pct") or 0.0),
        "avg_pnl_pct": float(stats.get("avg_pnl_pct") or 0.0),
        "profit_factor": float(stats.get("profit_factor") or 0.0),
        "max_drawdown_pct": float(stats.get("max_drawdown_pct") or 0.0),
        "max_drawdown_equity_pct": float(stats.get("max_drawdown_equity_pct") or 0.0),
        "avg_duration_hours": float(stats.get("avg_duration_hours") or 0.0),
        "max_duration_hours": float(stats.get("max_duration_hours") or 0.0),
        "p95_duration_hours": float(stats.get("p95_duration_hours") or 0.0),
        "time_exit_rate": float(stats.get("time_exit_rate") or 0.0),
        "funding_paid_pct": float(stats.get("funding_paid_pct") or 0.0),
        "funding_paid_equity_pct": float(stats.get("funding_paid_equity_pct") or 0.0),
        "is_clean": bool((result.get("data_quality") or {}).get("is_clean", False)),
        "sharpe_proxy": max(sharpe_vals) if sharpe_vals else 0.0,
    }
    metrics["research_score"] = compute_research_score(metrics, goal_contract)
    return metrics


def _passes_symbol_gate(row: dict, args: argparse.Namespace) -> bool:
    return (
        int(row["total_trades"]) >= args.min_symbol_trades
        and float(row["sum_trade_pnl_pct"]) > 0
        and float(row["profit_factor"]) >= args.min_symbol_profit_factor
        and float(row["max_drawdown_pct"]) >= args.min_symbol_max_drawdown
        and float(row["avg_duration_hours"]) <= args.max_avg_duration_hours
        and bool(row["is_clean"])
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    config = load_full_config(args.config)
    goal_contract = load_goal_contract(args.config)
    data_dir = str(config["data_dir"])
    funding_dir = args.funding_dir or config.get("funding_rates_dir") or str(Path(data_dir) / "new_data" / "funding_rates")
    output_dir = Path(args.output_dir)
    setups_dir = output_dir / "setups"
    output_dir.mkdir(parents=True, exist_ok=True)
    setups_dir.mkdir(parents=True, exist_ok=True)
    helpers = _load_workspace_helpers(output_dir)

    discovered = _discover_symbols(data_dir, args.timeframe, args.execution_timeframe, funding_dir)
    include = set(_parse_csv(args.include_symbols))
    exclude = set(_parse_csv(args.exclude_symbols))
    symbols = [s for s in discovered if (not include or s in include) and s not in exclude]
    symbols = helpers.filter_available_symbols(data_dir, symbols, args.timeframe)
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]

    price_meta = _file_metadata_by_symbol(Path(data_dir) / args.timeframe)
    execution_meta = _file_metadata_by_symbol(Path(data_dir) / args.execution_timeframe)
    funding_meta = _funding_file_metadata(funding_dir)
    universe = [
        s for s in symbols
        if price_meta.get(s, {}).get("ok")
        and execution_meta.get(s, {}).get("ok")
        and funding_meta.get(s, {}).get("ok")
        and price_meta[s]["days"] >= args.min_history_days
        and execution_meta[s]["days"] >= args.min_history_days
        and funding_meta[s]["days"] >= args.min_history_days
        and funding_meta[s]["approx_funding_rows"] >= args.min_funding_rows
    ]
    if not universe:
        raise SystemExit("No symbols passed data-quality filters.")

    print(f"universe discovered={len(discovered)} filtered={len(symbols)} data_quality={len(universe)}", flush=True)
    print("computing funding z-score...", flush=True)
    funding_z = helpers.compute_peer_funding_zscore(data_dir, universe)
    if funding_z.empty:
        raise SystemExit("funding z-score is empty.")

    availability_rows: list[dict] = []
    test_symbols: list[str] = []
    for sym in universe:
        z = funding_z[sym].dropna()
        if args.direction == "short_high":
            trigger_count = int((z >= args.entry_z).sum())
        else:
            trigger_count = int((z <= -args.entry_z).sum())
        row = {
            "symbol": sym,
            "trigger_count": trigger_count,
            "price_start": price_meta[sym]["start"],
            "price_end": price_meta[sym]["end"],
            "price_days": price_meta[sym]["days"],
            "execution_days": execution_meta[sym]["days"],
            "funding_start": funding_meta[sym]["start"],
            "funding_end": funding_meta[sym]["end"],
            "funding_days": funding_meta[sym]["days"],
            "approx_funding_rows": funding_meta[sym]["approx_funding_rows"],
            "funding_z_min": float(z.min()) if len(z) else 0.0,
            "funding_z_max": float(z.max()) if len(z) else 0.0,
        }
        availability_rows.append(row)
        if trigger_count >= args.min_raw_triggers:
            test_symbols.append(sym)

    availability_rows.sort(key=lambda r: int(r["trigger_count"]), reverse=True)
    _write_csv(output_dir / "symbol_availability.csv", availability_rows)
    print(f"symbols with >= {args.min_raw_triggers} raw triggers: {len(test_symbols)}", flush=True)

    per_symbol_rows: list[dict] = []
    result_by_symbol: dict[str, dict] = {}
    for i, sym in enumerate(test_symbols, start=1):
        if i % 25 == 0:
            print(f"backtesting {i}/{len(test_symbols)} {sym}", flush=True)
        try:
            signals = _generate_symbol_signals(
                helpers=helpers,
                data_dir=data_dir,
                available_timeframes=config.get("available_timeframes", ["5m", "4h"]),
                start_date=config.get("start_date", ""),
                timeframe=args.timeframe,
                direction=args.direction,
                symbol=sym,
                funding_z=funding_z,
                entry_z=args.entry_z,
                exit_z=args.exit_z,
                stop_pct=args.stop_pct,
                max_holding_candles=args.max_holding_candles,
                require_carry_sign=not args.disable_carry_sign_gate,
            )
        except Exception as exc:
            per_symbol_rows.append({"symbol": sym, "error": str(exc)})
            continue
        if signals.empty:
            per_symbol_rows.append({"symbol": sym, "error": "no_signals_after_state_machine"})
            continue
        signals = _apply_position_limit(signals, max_open_positions=1)
        setups_csv = setups_dir / f"{sym}.csv"
        helpers.write_signals_csv(signals, str(setups_csv))
        result = run_backtest(
            signals_csv=str(setups_csv),
            fee_rate=args.fee_rate,
            funding_dir=funding_dir,
            risk_pct_per_trade=args.risk_pct_per_trade,
        )
        result_by_symbol[sym] = result
        metrics = _metrics_from_result(result, goal_contract)
        row = {
            "symbol": sym,
            "setups_csv": str(setups_csv),
            "raw_trigger_count": next(r["trigger_count"] for r in availability_rows if r["symbol"] == sym),
            **metrics,
            "candidate_pass": False,
        }
        row["candidate_pass"] = _passes_symbol_gate(row, args)
        per_symbol_rows.append(row)

    ranked = [
        r for r in per_symbol_rows
        if not r.get("error")
    ]
    ranked.sort(
        key=lambda r: (
            bool(r.get("candidate_pass")),
            float(r.get("research_score") or 0.0),
            float(r.get("profit_factor") or 0.0),
            float(r.get("sum_trade_pnl_pct") or 0.0),
        ),
        reverse=True,
    )
    _write_csv(output_dir / "per_symbol_results.csv", ranked)
    (output_dir / "per_symbol_results.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    candidates = [r for r in ranked if r.get("candidate_pass")]
    candidate_symbols = [str(r["symbol"]) for r in candidates[: args.max_portfolio_candidates]]
    portfolio_rows: list[dict] = []
    selected: list[str] = []
    best_result: dict | None = None
    best_score = float("-inf")

    for sym in candidate_symbols:
        trial = [*selected, sym]
        frames = [pd.read_csv(setups_dir / f"{s}.csv") for s in trial]
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged = _apply_position_limit(merged, max_open_positions=args.max_open_positions)
        merged_path = output_dir / f"portfolio_{len(portfolio_rows) + 1:02d}_{'_'.join(trial)}.csv"
        helpers.write_signals_csv(merged, str(merged_path))
        result = run_backtest(
            signals_csv=str(merged_path),
            fee_rate=args.fee_rate,
            funding_dir=funding_dir,
            risk_pct_per_trade=args.risk_pct_per_trade,
        )
        metrics = _metrics_from_result(result, goal_contract)
        row = {
            "trial_symbols": ",".join(trial),
            "added_symbol": sym,
            "accepted": False,
            "setups_csv": str(merged_path),
            **metrics,
        }
        improves = float(metrics["research_score"]) > best_score
        meets_minimum = (
            int(metrics["total_trades"]) >= args.min_portfolio_trades
            and float(metrics["profit_factor"]) >= args.min_portfolio_profit_factor
            and float(metrics["max_drawdown_pct"]) >= args.min_portfolio_max_drawdown
            and bool(metrics["is_clean"])
        )
        if improves and (not selected or meets_minimum or int(metrics["total_trades"]) < args.min_portfolio_trades):
            selected = trial
            best_result = result
            best_score = float(metrics["research_score"])
            row["accepted"] = True
        portfolio_rows.append(row)

    _write_csv(output_dir / "portfolio_trials.csv", portfolio_rows)
    summary = {
        "config": vars(args),
        "universe_count": len(universe),
        "tested_symbols": len(test_symbols),
        "candidate_count": len(candidates),
        "top_candidates": candidates[:25],
        "selected_symbols": selected,
        "best_portfolio_result": best_result,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'symbol_availability.csv'}")
    print(f"wrote {output_dir / 'per_symbol_results.csv'}")
    print(f"wrote {output_dir / 'portfolio_trials.csv'}")
    print(f"candidates={len(candidates)} selected={selected}")
    if candidates:
        print("top candidates:", ", ".join(r["symbol"] for r in candidates[:10]))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic per-symbol selection for funding divergence.")
    parser.add_argument("--config", default="research_config.json")
    parser.add_argument("--output-dir", default="agent_research/param_search/funding_symbol_selection")
    parser.add_argument("--funding-dir", default="")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--execution-timeframe", default="5m")
    parser.add_argument("--direction", choices=["short_high", "long_low"], default="short_high")
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-pct", type=float, default=0.07)
    parser.add_argument("--max-holding-candles", type=int, default=18)
    parser.add_argument("--risk-pct-per-trade", type=float, default=0.02)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--disable-carry-sign-gate", action="store_true")
    parser.add_argument("--include-symbols", default="")
    parser.add_argument("--exclude-symbols", default="SOLUSDT")
    parser.add_argument("--limit-symbols", type=int, default=0)
    parser.add_argument("--min-history-days", type=float, default=365.0)
    parser.add_argument("--min-funding-rows", type=int, default=900)
    parser.add_argument("--min-raw-triggers", type=int, default=10)
    parser.add_argument("--min-symbol-trades", type=int, default=10)
    parser.add_argument("--min-symbol-profit-factor", type=float, default=1.2)
    parser.add_argument("--min-symbol-max-drawdown", type=float, default=-25.0)
    parser.add_argument("--max-avg-duration-hours", type=float, default=72.0)
    parser.add_argument("--max-open-positions", type=int, default=1)
    parser.add_argument("--max-portfolio-candidates", type=int, default=30)
    parser.add_argument("--min-portfolio-trades", type=int, default=30)
    parser.add_argument("--min-portfolio-profit-factor", type=float, default=1.3)
    parser.add_argument("--min-portfolio-max-drawdown", type=float, default=-15.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
