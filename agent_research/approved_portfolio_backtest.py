from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agent_research.config_utils import goal_contract_satisfied, load_full_config, load_goal_contract
from agent_research.code_runtime import write_workspace_helpers
from agent_research.sandbox import run_workspace_script
from backtest.engine import run_backtest


ACTION_ORDER = {"STOP": 0, "EXIT": 1, "ENTER_LONG": 2, "ENTER_SHORT": 3}
DEFAULT_FLOAT_TOLERANCE = 1e-6


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_exp_ids(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _to_unix_seconds(series: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(series, utc=True, errors="coerce")
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
        // 1_000_000_000
    )


def _find_strategy_script(bundle_dir: Path) -> Path | None:
    candidates = [
        p for p in sorted(bundle_dir.glob("*.py"))
        if p.name != "workspace_helpers.py" and not p.name.startswith("__")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _infer_strategy_family(bundle_dir: Path) -> str:
    text_parts: list[str] = []
    for name in ("reproducibility.json", "record.json"):
        path = bundle_dir / name
        if path.exists():
            text_parts.append(path.read_text(encoding="utf-8", errors="ignore").lower())
    script_path = _find_strategy_script(bundle_dir)
    if script_path is not None:
        text_parts.append(script_path.read_text(encoding="utf-8", errors="ignore").lower())
    text = "\n".join(text_parts)
    if "run_funding_divergence_strategy" in text or "funding divergence" in text:
        return "funding_divergence"
    if "pairs" in text and ("hedge_ratio" in text or "spread" in text):
        return "pairs_stat_arb"
    return "unknown"


def _canonicalize_setups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "T" not in out.columns:
        timestamp_col = "execution_timestamp" if "execution_timestamp" in out.columns else "timestamp"
        out["T"] = _to_unix_seconds(out[timestamp_col])
    out["_action_order"] = out["action"].map(ACTION_ORDER).fillna(99).astype(int)
    sort_cols = [c for c in ["T", "_action_order", "symbol", "pair_id", "action"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True).drop(columns=["_action_order"])
    return out.reindex(sorted(out.columns), axis=1)


def _values_equal(a: Any, b: Any, *, tolerance: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return str(a) == str(b)


def _compare_dataframes(expected: pd.DataFrame, actual: pd.DataFrame, *, tolerance: float) -> list[str]:
    errors: list[str] = []
    exp = _canonicalize_setups(expected)
    act = _canonicalize_setups(actual)
    if list(exp.columns) != list(act.columns):
        missing = sorted(set(exp.columns) - set(act.columns))
        extra = sorted(set(act.columns) - set(exp.columns))
        errors.append(f"setups columns differ: missing={missing} extra={extra}")
        common = sorted(set(exp.columns) & set(act.columns))
        exp = exp[common]
        act = act[common]
    if len(exp) != len(act):
        errors.append(f"setups row count differs: expected={len(exp)} actual={len(act)}")
        return errors
    max_examples = 5
    for row_idx in range(len(exp)):
        for col in exp.columns:
            if not _values_equal(exp.iloc[row_idx][col], act.iloc[row_idx][col], tolerance=tolerance):
                errors.append(
                    f"setups value differs at row={row_idx} col={col}: "
                    f"expected={exp.iloc[row_idx][col]!r} actual={act.iloc[row_idx][col]!r}"
                )
                if len(errors) >= max_examples:
                    return errors
    return errors


def _compare_json_subset(expected: Any, actual: Any, *, path: str, tolerance: float) -> list[str]:
    """Compare keys present in expected; actual may contain additional newer metrics."""
    errors: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path} type differs: expected=dict actual={type(actual).__name__}"]
        for key, exp_value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key} missing in reproduced result")
                continue
            errors.extend(_compare_json_subset(exp_value, actual[key], path=f"{path}.{key}", tolerance=tolerance))
            if len(errors) >= 10:
                return errors
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path} type differs: expected=list actual={type(actual).__name__}"]
        if len(expected) != len(actual):
            return [f"{path} length differs: expected={len(expected)} actual={len(actual)}"]
        for idx, exp_value in enumerate(expected):
            errors.extend(_compare_json_subset(exp_value, actual[idx], path=f"{path}[{idx}]", tolerance=tolerance))
            if len(errors) >= 10:
                return errors
        return errors
    if not _values_equal(expected, actual, tolerance=tolerance):
        return [f"{path} differs: expected={expected!r} actual={actual!r}"]
    return []


def _run_with_env(env_updates: dict[str, str], fn) -> Any:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env_updates}
    try:
        os.environ.update(env_updates)
        return fn()
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _validate_reproducibility(
    *,
    bundle_dir: Path,
    output_dir: Path,
    tolerance: float,
    default_risk_pct_per_trade: float,
) -> dict:
    exp_id = bundle_dir.name
    validation_dir = output_dir / "reproducibility" / exp_id
    validation_dir.mkdir(parents=True, exist_ok=True)

    script_path = _find_strategy_script(bundle_dir)
    if script_path is None:
        return {"passed": False, "errors": ["expected exactly one strategy .py file in approved bundle"]}

    stored_setups_path = bundle_dir / "setups.csv"
    stored_result_path = bundle_dir / "result.json"
    base_config_path = bundle_dir / "base_config_snapshot.json"
    research_scope_path = bundle_dir / "research_scope.json"
    if not stored_setups_path.exists() or not stored_result_path.exists() or not base_config_path.exists():
        return {"passed": False, "errors": ["missing setups.csv, result.json, or base_config_snapshot.json"]}

    base_config = _load_json(base_config_path)
    research_scope = _load_json(research_scope_path) if research_scope_path.exists() else {}
    expected_result = _load_json(stored_result_path)
    expected_input = dict(expected_result.get("input") or {})
    risk_pct = float(expected_input.get("risk_pct_per_trade") or default_risk_pct_per_trade)
    fee_rate = float(expected_input.get("fee_rate") or 0.0005)
    funding_dir = base_config.get("funding_rates_dir") or str(Path(base_config["data_dir"]) / "new_data" / "funding_rates")

    repro_script = validation_dir / script_path.name
    shutil.copy2(script_path, repro_script)
    write_workspace_helpers(validation_dir)

    selected_symbols = list(research_scope.get("selected_symbols") or base_config.get("symbols") or [])
    available_timeframes = list(base_config.get("available_timeframes") or ["1h"])
    start_date = base_config.get("start_date")

    def _run_script() -> dict:
        return run_workspace_script(
            script_path=repro_script,
            workspace_dir=validation_dir,
            data_dir=base_config["data_dir"],
            available_timeframes=available_timeframes,
            symbols=selected_symbols,
            start_date=start_date,
            funding_rates_dir=funding_dir,
        )

    sandbox_result = _run_with_env({"RISK_PCT_PER_TRADE": str(risk_pct)}, _run_script)
    errors: list[str] = []
    if sandbox_result.get("returncode") != 0:
        errors.append(f"strategy script failed rc={sandbox_result.get('returncode')} stderr={sandbox_result.get('stderr_path')}")
    if sandbox_result.get("parse_error"):
        errors.append(f"strategy script JSON parse error: {sandbox_result['parse_error']}")
    reproduced_setups_path = Path(sandbox_result.get("setups_csv") or validation_dir / "setups.csv")
    if not reproduced_setups_path.exists():
        errors.append(f"strategy script did not reproduce setups.csv at {reproduced_setups_path}")
    if errors:
        return {"passed": False, "errors": errors, "sandbox_result": sandbox_result}

    stored_setups = pd.read_csv(stored_setups_path)
    reproduced_setups = pd.read_csv(reproduced_setups_path)
    errors.extend(_compare_dataframes(stored_setups, reproduced_setups, tolerance=tolerance))

    reproduced_result = run_backtest(
        signals_csv=str(reproduced_setups_path),
        fee_rate=fee_rate,
        funding_dir=funding_dir,
        risk_pct_per_trade=risk_pct,
    )
    reproduced_result_path = validation_dir / "reproduced_result.json"
    reproduced_result_path.write_text(json.dumps(reproduced_result, indent=2, ensure_ascii=False), encoding="utf-8")

    comparable_expected = {
        "status": expected_result.get("status"),
        "strategy_type": expected_result.get("strategy_type"),
        "input": {
            key: value
            for key, value in expected_input.items()
            if key not in {"csv_path"}
        },
        "stats": expected_result.get("stats") or {},
        "data_quality": expected_result.get("data_quality") or {},
        "pairs": expected_result.get("pairs") or [],
    }
    comparable_actual = {
        "status": reproduced_result.get("status"),
        "strategy_type": reproduced_result.get("strategy_type"),
        "input": {
            key: value
            for key, value in (reproduced_result.get("input") or {}).items()
            if key not in {"csv_path"}
        },
        "stats": reproduced_result.get("stats") or {},
        "data_quality": reproduced_result.get("data_quality") or {},
        "pairs": reproduced_result.get("pairs") or [],
    }
    errors.extend(_compare_json_subset(comparable_expected, comparable_actual, path="result", tolerance=tolerance))

    report = {
        "experiment_id": exp_id,
        "passed": not errors,
        "errors": errors[:20],
        "strategy_script": str(script_path),
        "validation_dir": str(validation_dir),
        "stored_setups_csv": str(stored_setups_path),
        "reproduced_setups_csv": str(reproduced_setups_path),
        "reproduced_result_json": str(reproduced_result_path),
        "backtest_inputs": {
            "fee_rate": fee_rate,
            "funding_dir": funding_dir,
            "risk_pct_per_trade": risk_pct,
            "symbols": selected_symbols,
            "available_timeframes": available_timeframes,
            "start_date": start_date,
        },
        "sandbox_result": sandbox_result,
    }
    (validation_dir / "reproducibility_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _discover_approved(
    approved_dir: Path,
    *,
    output_dir: Path,
    current_goal_contract: dict,
    explicit_ids: set[str],
    include_legacy_approved: bool,
    validate_reproducibility: bool,
    tolerance: float,
    default_risk_pct_per_trade: float,
) -> tuple[list[dict], list[dict]]:
    included: list[dict] = []
    excluded: list[dict] = []

    for bundle_dir in sorted(p for p in approved_dir.iterdir() if p.is_dir()):
        exp_id = bundle_dir.name
        if explicit_ids and exp_id not in explicit_ids:
            continue

        setups_csv = bundle_dir / "setups.csv"
        metrics_path = bundle_dir / "metrics.json"
        result_path = bundle_dir / "result.json"
        if not setups_csv.exists() or not metrics_path.exists():
            excluded.append({
                "experiment_id": exp_id,
                "reason": "missing setups.csv or metrics.json",
                "path": str(bundle_dir),
            })
            continue

        metrics = _load_json(metrics_path)
        result = _load_json(result_path) if result_path.exists() else {}
        passes_current_contract = goal_contract_satisfied(metrics, current_goal_contract)
        is_clean = bool(metrics.get("is_clean", False))
        has_trades = int(metrics.get("total_trades") or 0) > 0

        reasons = []
        if not passes_current_contract:
            reasons.append("fails current goal_contract")
        if not is_clean:
            reasons.append("metrics.is_clean is false")
        if not has_trades:
            reasons.append("total_trades is zero")

        if reasons and not include_legacy_approved:
            excluded.append({
                "experiment_id": exp_id,
                "reason": "; ".join(reasons),
                "metrics": metrics,
                "path": str(bundle_dir),
            })
            continue

        reproducibility = {"passed": True, "errors": [], "skipped": True}
        if validate_reproducibility:
            reproducibility = _validate_reproducibility(
                bundle_dir=bundle_dir,
                output_dir=output_dir,
                tolerance=tolerance,
                default_risk_pct_per_trade=default_risk_pct_per_trade,
            )
            if not reproducibility.get("passed"):
                excluded.append({
                    "experiment_id": exp_id,
                    "reason": "reproducibility validation failed: " + "; ".join(reproducibility.get("errors") or []),
                    "metrics": metrics,
                    "path": str(bundle_dir),
                    "reproducibility": reproducibility,
                })
                continue

        included.append({
            "experiment_id": exp_id,
            "bundle_dir": str(bundle_dir),
            "setups_csv": str(setups_csv),
            "metrics": metrics,
            "strategy_family": _infer_strategy_family(bundle_dir),
            "result_status": result.get("status", ""),
            "passes_current_contract": passes_current_contract,
            "included_with_warnings": bool(reasons),
            "warnings": reasons,
            "reproducibility": reproducibility,
        })

    return included, excluded


def _load_and_merge_setups(experiments: list[dict]) -> tuple[pd.DataFrame, dict]:
    frames: list[pd.DataFrame] = []
    strategy_types: set[str] = set()
    strategy_families: set[str] = {str(exp.get("strategy_family") or "unknown") for exp in experiments}
    per_source_rows: dict[str, int] = {}

    for exp in experiments:
        exp_id = exp["experiment_id"]
        df = pd.read_csv(exp["setups_csv"])
        if df.empty:
            per_source_rows[exp_id] = 0
            continue

        df = df.copy()
        df["source_experiment_id"] = exp_id
        if "strategy_type" not in df.columns:
            df["strategy_type"] = "pairs" if {"price_a", "price_b", "hedge_ratio"}.issubset(df.columns) else "single_asset"
        strategy_types.update(str(v) for v in df["strategy_type"].dropna().unique())

        timestamp_col = "execution_timestamp" if "execution_timestamp" in df.columns else "timestamp"
        if "T" not in df.columns:
            df["T"] = _to_unix_seconds(df[timestamp_col])
        df["_action_order"] = df["action"].map(ACTION_ORDER).fillna(99).astype(int)
        frames.append(df)
        per_source_rows[exp_id] = len(df)

    if not frames:
        return pd.DataFrame(), {"per_source_rows": per_source_rows, "strategy_types": sorted(strategy_types)}

    if len(strategy_types) > 1:
        raise ValueError(f"Mixed strategy_type is not supported in one merged backtest: {sorted(strategy_types)}")
    if len(strategy_families) > 1:
        raise ValueError(f"Mixed strategy_family is not supported in one merged backtest: {sorted(strategy_families)}")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values(["T", "_action_order", "source_experiment_id"]).drop(columns=["_action_order"])
    merged = merged.reset_index(drop=True)
    return merged, {
        "per_source_rows": per_source_rows,
        "strategy_types": sorted(strategy_types),
        "strategy_families": sorted(strategy_families),
    }


def _write_summary(
    *,
    output_dir: Path,
    merged_csv: Path,
    included: list[dict],
    excluded: list[dict],
    merge_info: dict,
    result: dict,
    current_goal_contract: dict,
    include_legacy_approved: bool,
) -> Path:
    stats = result.get("stats") or {}
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "include_legacy_approved" if include_legacy_approved else "current_contract_only",
        "current_goal_contract": current_goal_contract,
        "merged_signals_csv": str(merged_csv.resolve()),
        "included_experiments": included,
        "excluded_experiments": excluded,
        "merge_info": merge_info,
        "portfolio_result": result,
        "headline": {
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0.0),
            "sum_trade_pnl_pct": stats.get("sum_trade_pnl_pct", stats.get("total_pnl_pct", 0.0)),
            "estimated_equity_return_pct": stats.get("estimated_equity_return_pct", 0.0),
            "profit_factor": stats.get("profit_factor", 0.0),
            "max_drawdown_pct": stats.get("max_drawdown_pct", 0.0),
            "max_drawdown_equity_pct": stats.get("max_drawdown_equity_pct", 0.0),
            "avg_duration_hours": stats.get("avg_duration_hours", 0.0),
            "max_duration_hours": stats.get("max_duration_hours", 0.0),
            "time_exit_rate": stats.get("time_exit_rate", 0.0),
            "funding_paid_equity_pct": stats.get("funding_paid_equity_pct", 0.0),
            "slippage_paid_equity_pct": stats.get("slippage_paid_equity_pct", 0.0),
            "data_quality": result.get("data_quality", {}),
        },
    }
    summary_path = output_dir / "portfolio_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    text_lines = [
        "APPROVED PORTFOLIO BACKTEST",
        "=" * 80,
        f"Mode: {summary['mode']}",
        f"Merged CSV: {merged_csv.resolve()}",
        "",
        "Included experiments:",
    ]
    for exp in included:
        warning = f" WARNINGS={exp['warnings']}" if exp.get("warnings") else ""
        text_lines.append(f"  {exp['experiment_id']} current_contract={exp['passes_current_contract']}{warning}")
    if excluded:
        text_lines += ["", "Excluded experiments:"]
        for exp in excluded:
            text_lines.append(f"  {exp['experiment_id']} reason={exp['reason']}")
    text_lines += [
        "",
        "Portfolio metrics:",
        f"  total_trades              {stats.get('total_trades', 0)}",
        f"  win_rate                  {stats.get('win_rate', 0.0):.4f}",
        f"  sum_trade_pnl_pct         {stats.get('sum_trade_pnl_pct', stats.get('total_pnl_pct', 0.0)):+.4f}",
        f"  estimated_equity_return   {stats.get('estimated_equity_return_pct', 0.0):+.4f}",
        f"  profit_factor             {stats.get('profit_factor', 0.0):.4f}",
        f"  max_drawdown_pct          {stats.get('max_drawdown_pct', 0.0):+.4f}",
        f"  max_drawdown_equity_pct   {stats.get('max_drawdown_equity_pct', 0.0):+.4f}",
        f"  avg_duration_hours        {stats.get('avg_duration_hours', 0.0):.2f}",
        f"  max_duration_hours        {stats.get('max_duration_hours', 0.0):.2f}",
        f"  time_exit_rate            {stats.get('time_exit_rate', 0.0):.4f}",
        f"  funding_paid_equity_pct   {stats.get('funding_paid_equity_pct', 0.0):+.4f}",
        f"  slippage_paid_equity_pct  {stats.get('slippage_paid_equity_pct', 0.0):+.4f}",
        f"  data_quality              {result.get('data_quality', {})}",
        "",
        "Per-symbol / pair breakdown:",
    ]
    for pair in result.get("pairs") or []:
        text_lines.append(
            f"  {pair.get('pair_id', '?'):<18} "
            f"trades={pair.get('trades', 0):<5} "
            f"win={pair.get('win_rate', 0.0):.3f} "
            f"pnl={pair.get('total_pnl_pct', 0.0):+.2f}% "
            f"sharpe={pair.get('sharpe', 0.0):+.3f} "
            f"dd={pair.get('max_drawdown_pct', 0.0):+.2f}% "
            f"stop={pair.get('stop_rate', 0.0):.3f}"
        )
    report_path = output_dir / "portfolio_report.log"
    report_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return summary_path


def run(args: argparse.Namespace) -> dict:
    approved_dir = Path(args.approved_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_full_config(args.config)
    current_goal_contract = load_goal_contract(args.config)
    funding_dir = args.funding_dir or config.get("funding_rates_dir") or str(Path(config["data_dir"]) / "new_data" / "funding_rates")
    explicit_ids = _parse_exp_ids(args.experiments)

    included, excluded = _discover_approved(
        approved_dir,
        output_dir=output_dir,
        current_goal_contract=current_goal_contract,
        explicit_ids=explicit_ids,
        include_legacy_approved=args.include_legacy_approved,
        validate_reproducibility=not args.skip_reproducibility_validation,
        tolerance=args.tolerance,
        default_risk_pct_per_trade=args.risk_pct_per_trade,
    )
    if not included:
        raise SystemExit("No approved experiments selected for portfolio backtest.")

    merged, merge_info = _load_and_merge_setups(included)
    if merged.empty:
        raise SystemExit("Selected experiments produced no setup rows.")

    merged_csv = output_dir / "merged_approved_setups.csv"
    merged.to_csv(merged_csv, index=False)

    result = run_backtest(
        signals_csv=str(merged_csv),
        fee_rate=args.fee_rate,
        funding_dir=funding_dir,
        risk_pct_per_trade=args.risk_pct_per_trade,
    )
    result_path = output_dir / "portfolio_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = _write_summary(
        output_dir=output_dir,
        merged_csv=merged_csv,
        included=included,
        excluded=excluded,
        merge_info=merge_info,
        result=result,
        current_goal_contract=current_goal_contract,
        include_legacy_approved=args.include_legacy_approved,
    )

    stats = result.get("stats") or {}
    print(f"wrote {merged_csv}")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(
        "portfolio "
        f"experiments={len(included)} trades={stats.get('total_trades', 0)} "
        f"equity={stats.get('estimated_equity_return_pct', 0.0):+.2f}% "
        f"sum_pnl={stats.get('sum_trade_pnl_pct', stats.get('total_pnl_pct', 0.0)):+.2f}% "
        f"pf={stats.get('profit_factor', 0.0):.2f} "
        f"maxDD={stats.get('max_drawdown_pct', 0.0):+.2f}% "
        f"equityDD={stats.get('max_drawdown_equity_pct', 0.0):+.2f}% "
        f"clean={result.get('data_quality', {}).get('is_clean')}"
    )
    if excluded:
        print(f"excluded={len(excluded)} (see portfolio_summary.json)")
    return result


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge approved experiment setups and run one portfolio backtest.")
    parser.add_argument("--config", default=defaults["config"])
    parser.add_argument("--approved-dir", default=defaults["approved_dir"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--funding-dir", default=defaults["funding_dir"])
    parser.add_argument(
        "--experiments",
        default=defaults["experiments"],
        help="Comma-separated full experiment ids. Empty = auto-discover approved.",
    )
    parser.add_argument(
        "--include-legacy-approved",
        action="store_true",
        default=defaults["include_legacy_approved"],
        help="Include approved bundles that fail the current goal_contract.",
    )
    parser.add_argument("--risk-pct-per-trade", type=float, default=defaults["risk_pct_per_trade"])
    parser.add_argument("--fee-rate", type=float, default=defaults["fee_rate"])
    parser.add_argument("--tolerance", type=float, default=defaults["tolerance"])
    parser.add_argument(
        "--skip-reproducibility-validation",
        action="store_true",
        default=defaults["skip_reproducibility_validation"],
        help="Skip per-experiment setup/backtest reproduction checks before merging.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    run(args)


if __name__ == "__main__":
    CONFIG = "research_config.json"
    APPROVED_DIR = "agent_research/approved"
    OUTPUT_DIR = "agent_research/portfolio_backtests/approved_portfolio"
    FUNDING_DIR = ""
    EXPERIMENTS = ""
    INCLUDE_LEGACY_APPROVED = False
    RISK_PCT_PER_TRADE = 0.02
    FEE_RATE = 0.0005
    TOLERANCE = DEFAULT_FLOAT_TOLERANCE
    SKIP_REPRODUCIBILITY_VALIDATION = False

    DEFAULTS = {
        "config": CONFIG,
        "approved_dir": APPROVED_DIR,
        "output_dir": OUTPUT_DIR,
        "funding_dir": FUNDING_DIR,
        "experiments": EXPERIMENTS,
        "include_legacy_approved": INCLUDE_LEGACY_APPROVED,
        "risk_pct_per_trade": RISK_PCT_PER_TRADE,
        "fee_rate": FEE_RATE,
        "tolerance": TOLERANCE,
        "skip_reproducibility_validation": SKIP_REPRODUCIBILITY_VALIDATION,
    }
    main(parse_args(DEFAULTS))
