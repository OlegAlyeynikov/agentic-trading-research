"""Shared config loading helpers — used by researcher, executor, agents."""
import json


DEFAULT_GOAL_CONTRACT = {
    "primary_metric": "research_score",
    "min_sharpe_proxy": 0.6,
    "min_profit_factor": 1.3,
    "max_drawdown_pct": -15.0,
    "min_total_trades": 30,
    "max_avg_duration_hours": 72.0,
    "min_pairs": 0,
    "require_clean_data": True,
}


def load_full_config(config_path: str) -> dict:
    """Load research_config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_goal_contract(config_path: str) -> dict:
    """Load configurable research success criteria."""
    full = load_full_config(config_path)
    raw = full.get("goal_contract") or {}
    return {**DEFAULT_GOAL_CONTRACT, **raw}


def compute_research_score(metrics: dict, goal_contract: dict | None = None) -> float:
    """
    Composite research score for ranking experiments.

    The score emphasizes risk-adjusted behavior first, then sample quality.
    It is normalized against the active goal contract so "100" roughly means
    "met the contract on average", while higher scores indicate stronger runs.
    """
    goal = {**DEFAULT_GOAL_CONTRACT, **(goal_contract or {})}

    sharpe_target = max(float(goal.get("min_sharpe_proxy", 1.0) or 1.0), 1e-9)
    pf_target = max(float(goal.get("min_profit_factor", 1.5) or 1.5), 1e-9)
    trades_target = max(int(goal.get("min_total_trades", 50) or 50), 1)
    min_pairs = int(goal.get("min_pairs", 0) or 0)
    dd_limit = abs(float(goal.get("max_drawdown_pct", -20.0) or -20.0))
    dd_limit = max(dd_limit, 1e-9)

    sharpe = float(metrics.get("sharpe_proxy") or 0.0)
    profit_factor = float(metrics.get("profit_factor") or 0.0)
    drawdown = abs(float(metrics.get("max_drawdown_pct") or 0.0))
    trades = int(metrics.get("total_trades") or 0)
    pairs = int(metrics.get("pairs_count") or 0)
    is_clean = bool(metrics.get("is_clean", False))
    clean_required = bool(goal.get("require_clean_data", True))

    if trades <= 0:
        return 0.0
    if min_pairs > 0 and pairs <= 0:
        return 0.0

    sharpe_score = min(max(sharpe / sharpe_target, 0.0), 2.0)
    pf_score = min(max(profit_factor / pf_target, 0.0), 2.0)
    if drawdown == 0:
        dd_score = 2.0
    else:
        dd_score = min(max(dd_limit / drawdown, 0.0), 2.0)
    trades_score = min(max(trades / trades_target, 0.0), 1.5)
    pairs_score = min(max(pairs / max(min_pairs, 1), 0.0), 1.5) if min_pairs > 0 else 1.0
    clean_score = 1.0 if (is_clean or not clean_required) else 0.0

    weighted = (
        sharpe_score * 0.35
        + pf_score * 0.25
        + dd_score * 0.20
        + trades_score * 0.10
        + pairs_score * 0.05
        + clean_score * 0.05
    )
    return round(weighted * 100, 4)


def goal_contract_satisfied(metrics: dict, goal_contract: dict | None = None) -> bool:
    goal = {**DEFAULT_GOAL_CONTRACT, **(goal_contract or {})}

    if float(metrics.get("sharpe_proxy") or 0.0) < float(goal.get("min_sharpe_proxy", 0.7)):
        return False
    if float(metrics.get("profit_factor") or 0.0) < float(goal.get("min_profit_factor", 1.3)):
        return False
    if float(metrics.get("max_drawdown_pct") or 0.0) < float(goal.get("max_drawdown_pct", -15.0)):
        return False
    if int(metrics.get("total_trades") or 0) < int(goal.get("min_total_trades", 50)):
        return False
    max_dur = goal.get("max_avg_duration_hours")
    if max_dur is not None:
        actual_dur = metrics.get("avg_duration_hours")
        if actual_dur is not None and float(actual_dur) > float(max_dur):
            return False
    if int(goal.get("min_pairs", 0) or 0) > 0 and int(metrics.get("pairs_count") or 0) < int(goal.get("min_pairs", 0)):
        return False
    if bool(goal.get("require_clean_data", True)) and not bool(metrics.get("is_clean", False)):
        return False
    return True


def goal_contract_summary(goal_contract: dict | None = None) -> str:
    goal = {**DEFAULT_GOAL_CONTRACT, **(goal_contract or {})}
    return (
        f"primary_metric={goal.get('primary_metric', 'research_score')} "
        f"sharpe>={goal.get('min_sharpe_proxy', 1.0):.2f} "
        f"profit_factor>={goal.get('min_profit_factor', 1.5):.2f} "
        f"maxDD>={goal.get('max_drawdown_pct', -20.0):.2f}% "
        f"trades>={goal.get('min_total_trades', 50)} "
        f"pairs>={goal.get('min_pairs', 3)} "
        f"clean={goal.get('require_clean_data', True)}"
    )


def goal_contract_progress(metrics: dict, goal_contract: dict | None = None) -> dict:
    goal = {**DEFAULT_GOAL_CONTRACT, **(goal_contract or {})}
    progress = {
        "research_score": compute_research_score(metrics, goal),
        "sharpe_proxy": float(metrics.get("sharpe_proxy") or 0.0),
        "profit_factor": float(metrics.get("profit_factor") or 0.0),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct") or 0.0),
        "total_trades": int(metrics.get("total_trades") or 0),
        "pairs_count": int(metrics.get("pairs_count") or 0),
        "is_clean": bool(metrics.get("is_clean", False)),
        "goal_satisfied": goal_contract_satisfied(metrics, goal),
    }
    return progress
