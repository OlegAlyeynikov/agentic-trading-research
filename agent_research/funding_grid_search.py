from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import random
from pathlib import Path
from typing import Any

from agent_research.code_runtime import write_workspace_helpers
from agent_research.config_utils import (
    compute_research_score,
    goal_contract_satisfied,
    load_full_config,
    load_goal_contract,
)


DEFAULT_PEER_SYMBOLS = [
    "APTUSDT",
    "AVAXUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "DOTUSDT",
    "ETHUSDT",
    "BTCUSDT",
    "LINKUSDT",
    "AAVEUSDT",
]


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_workspace_helpers(output_dir: Path):
    runtime_dir = output_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_workspace_helpers(runtime_dir)
    helper_path = runtime_dir / "workspace_helpers.py"
    spec = importlib.util.spec_from_file_location("funding_grid_workspace_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import workspace helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_name(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _symbols_from_pairs(pairs: list[dict]) -> list[str]:
    return [str(p.get("pair_id")) for p in pairs if p.get("pair_id")]


def _summarize_diagnostics(diagnostics: dict) -> dict:
    symbols = diagnostics.get("symbols") or {}
    raw_triggers = sum(int((d or {}).get("count_beyond_entry") or 0) for d in symbols.values())
    price_ok = sum(int((d or {}).get("count_price_z_ok") or 0) for d in symbols.values())
    return {
        "blocked_reason": diagnostics.get("blocked_reason", ""),
        "raw_trigger_count": raw_triggers,
        "price_z_ok_count": price_ok,
        "symbols_with_raw_triggers": sum(
            1 for d in symbols.values() if int((d or {}).get("count_beyond_entry") or 0) > 0
        ),
        "execution_timeframe": diagnostics.get("execution_timeframe", ""),
        "reduced_fidelity": bool(diagnostics.get("reduced_fidelity", False)),
    }


def _row_from_result(
    *,
    candidate_id: str,
    params: dict,
    result: dict,
    diagnostics: dict,
    setups_csv: str,
    goal_contract: dict,
) -> dict:
    stats = result.get("stats") or {}
    pairs = result.get("pairs") or []
    diag = _summarize_diagnostics(diagnostics)
    metrics = {
        "total_trades": int(stats.get("total_trades") or 0),
        "win_rate": float(stats.get("win_rate") or 0.0),
        "total_pnl_pct": float(stats.get("total_pnl_pct") or 0.0),
        "sum_trade_pnl_pct": float(stats.get("sum_trade_pnl_pct") or stats.get("total_pnl_pct") or 0.0),
        "estimated_equity_return_pct": float(stats.get("estimated_equity_return_pct") or 0.0),
        "profit_factor": float(stats.get("profit_factor") or 0.0),
        "max_drawdown_pct": float(stats.get("max_drawdown_pct") or 0.0),
        "avg_duration_hours": float(stats.get("avg_duration_hours") or 0.0),
        "median_duration_hours": float(stats.get("median_duration_hours") or 0.0),
        "max_duration_hours": float(stats.get("max_duration_hours") or 0.0),
        "p95_duration_hours": float(stats.get("p95_duration_hours") or 0.0),
        "time_exit_rate": float(stats.get("time_exit_rate") or 0.0),
        "pairs_count": len([p for p in pairs if int(p.get("trades") or 0) > 0]),
        "is_clean": bool((result.get("data_quality") or {}).get("is_clean", False)),
    }
    sharpe_vals = [float(p.get("sharpe") or 0.0) for p in pairs if p.get("sharpe") is not None]
    metrics["sharpe_proxy"] = max(sharpe_vals) if sharpe_vals else 0.0
    metrics["research_score"] = compute_research_score(metrics, goal_contract)
    metrics["meets_min_trades"] = metrics["total_trades"] >= int(goal_contract.get("min_total_trades", 0) or 0)
    metrics["passes_goal_contract"] = goal_contract_satisfied(metrics, goal_contract)
    metrics["eligible_for_selection"] = bool(
        metrics["passes_goal_contract"]
        and metrics["meets_min_trades"]
        and metrics["is_clean"]
    )
    return {
        "candidate_id": candidate_id,
        **params,
        **metrics,
        **diag,
        "symbols_used": ",".join(_symbols_from_pairs(pairs)),
        "setups_csv": setups_csv,
    }


def run_search(args: argparse.Namespace) -> list[dict]:
    from backtest.engine import run_backtest

    config = load_full_config(args.config)
    goal_contract = load_goal_contract(args.config)
    output_dir = Path(args.output_dir)
    setups_dir = output_dir / "setups"
    output_dir.mkdir(parents=True, exist_ok=True)
    setups_dir.mkdir(parents=True, exist_ok=True)
    helpers = _load_workspace_helpers(output_dir)

    config_symbols = list(config.get("symbols") or [])
    excluded_entry_symbols = set(_parse_csv(args.exclude_entry_symbols))
    entry_symbols = _parse_csv(args.entry_symbols) or [
        s for s in config_symbols if s not in excluded_entry_symbols
    ]
    peer_symbols = _parse_csv(args.peer_symbols) or [
        s for s in [*DEFAULT_PEER_SYMBOLS, *entry_symbols] if s in config_symbols
    ]
    if not peer_symbols:
        peer_symbols = config_symbols

    search_config = {
        "config": str(Path(args.config).resolve()),
        "timeframe": args.timeframe,
        "config_symbols_count": len(config_symbols),
        "entry_symbols_count": len(entry_symbols),
        "peer_symbols_count": len(peer_symbols),
        "excluded_entry_symbols": sorted(excluded_entry_symbols),
        "entry_symbols": entry_symbols,
        "peer_symbols": peer_symbols,
    }
    (output_dir / "funding_grid_search_config.json").write_text(
        json.dumps(search_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "search universe "
        f"config_symbols={len(config_symbols)} "
        f"entry_symbols={len(entry_symbols)} "
        f"peer_symbols={len(peer_symbols)} "
        f"excluded={','.join(sorted(excluded_entry_symbols)) or '-'}"
    )

    price_z_values: list[float | None] = [None if v.lower() == "none" else float(v) for v in _parse_csv(args.price_z_thresholds)]
    grid = list(itertools.product(
        _parse_csv(args.directions),
        _parse_float_list(args.funding_z_thresholds),
        price_z_values,
        _parse_int_list(args.price_z_windows),
        _parse_float_list(args.exit_funding_z),
        _parse_int_list(args.max_holding_candles),
        _parse_float_list(args.stop_pct),
        _parse_float_list(args.risk_pct_per_trade),
    ))
    if args.random_sample and args.random_sample < len(grid):
        rng = random.Random(args.seed)
        grid = rng.sample(grid, args.random_sample)

    rows: list[dict] = []
    for i, combo in enumerate(grid, start=1):
        (
            direction,
            funding_z_threshold,
            price_z_threshold,
            price_z_window,
            exit_z,
            max_holding,
            stop_pct,
            risk_pct,
        ) = combo
        candidate_id = (
            f"fd_{i:04d}_{direction}_fz{_safe_name(funding_z_threshold)}_"
            f"pz{_safe_name(price_z_threshold)}_w{price_z_window}_h{max_holding}_s{_safe_name(stop_pct)}"
        )
        setups_csv = str(setups_dir / f"{candidate_id}.csv")
        signals, diagnostics = helpers.generate_funding_divergence_signals(
            data_dir=config["data_dir"],
            symbols=config_symbols,
            available_timeframes=config.get("available_timeframes", ["4h"]),
            start_date=config.get("start_date", ""),
            timeframe=args.timeframe,
            direction=direction,
            entry_z=funding_z_threshold,
            exit_z=exit_z,
            stop_pct=stop_pct,
            stop_z=99.0,
            max_holding_candles=max_holding,
            exclude_entry_symbols=sorted(excluded_entry_symbols),
            entry_symbols=entry_symbols,
            peer_symbols=peer_symbols,
            max_open_positions=args.max_open_positions,
            require_carry_sign=not args.disable_carry_sign_gate,
            price_z_threshold=price_z_threshold,
            price_z_window=price_z_window,
        )
        params = {
            "direction": direction,
            "funding_z_threshold": funding_z_threshold,
            "price_z_threshold": "" if price_z_threshold is None else price_z_threshold,
            "price_z_window": price_z_window,
            "exit_funding_z": exit_z,
            "max_holding_candles": max_holding,
            "stop_pct": stop_pct,
            "risk_pct_per_trade": risk_pct,
            "entry_symbols": ",".join(entry_symbols),
            "peer_symbols": ",".join(peer_symbols),
        }
        if signals.empty:
            result = {
                "stats": {"total_trades": 0},
                "data_quality": {"is_clean": False},
                "pairs": [],
            }
            rows.append(_row_from_result(
                candidate_id=candidate_id,
                params=params,
                result=result,
                diagnostics=diagnostics,
                setups_csv="",
                goal_contract=goal_contract,
            ))
            continue

        helpers.write_signals_csv(signals, setups_csv)
        result = run_backtest(
            signals_csv=setups_csv,
            fee_rate=args.fee_rate,
            funding_dir=config.get("funding_rates_dir") or str(Path(config["data_dir"]) / "new_data" / "funding_rates"),
            risk_pct_per_trade=risk_pct,
        )
        rows.append(_row_from_result(
            candidate_id=candidate_id,
            params=params,
            result=result,
            diagnostics=diagnostics,
            setups_csv=setups_csv,
            goal_contract=goal_contract,
        ))

    rows.sort(
        key=lambda r: (
            bool(r.get("eligible_for_selection")),
            bool(r.get("meets_min_trades")),
            float(r.get("research_score") or 0.0),
            int(float(r.get("total_trades") or 0)),
        ),
        reverse=True,
    )
    csv_path = output_dir / "funding_grid_results.csv"
    json_path = output_dir / "funding_grid_results.json"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    if rows:
        best = rows[0]
        print(
            "best "
            f"{best['candidate_id']} trades={best['total_trades']} "
            f"score={float(best['research_score']):.1f} "
            f"sharpe={float(best['sharpe_proxy']):.2f} "
            f"pf={float(best['profit_factor']):.2f} "
            f"equity={float(best['estimated_equity_return_pct']):+.2f}%"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic funding-divergence grid/random search.")
    parser.add_argument("--config", default="research_config.json")
    parser.add_argument("--output-dir", default="agent_research/param_search/funding_divergence")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--directions", default="short_high,long_low")
    parser.add_argument("--funding-z-thresholds", default="1.5,1.8,2.0,2.5,3.0")
    parser.add_argument("--price-z-thresholds", default="none,0.5,1.0,1.5")
    parser.add_argument("--price-z-windows", default="60,90,180")
    parser.add_argument("--exit-funding-z", default="0.25,0.5,0.75")
    parser.add_argument("--max-holding-candles", default="6,12,18")
    parser.add_argument("--stop-pct", default="0.05,0.07,0.10")
    parser.add_argument("--risk-pct-per-trade", default="0.02")
    parser.add_argument("--entry-symbols", default="")
    parser.add_argument("--peer-symbols", default="")
    parser.add_argument("--exclude-entry-symbols", default="SOLUSDT,DOTUSDT")
    parser.add_argument("--max-open-positions", type=int, default=1)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--disable-carry-sign-gate", action="store_true")
    parser.add_argument("--random-sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    run_search(parse_args())


if __name__ == "__main__":
    main()
