import json
import threading
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

_FILE_LOCK = threading.Lock()


class ExperimentStore:
    def __init__(self, store_path: str | Path | None = None):
        self._path = Path(store_path) if store_path else EXPERIMENTS_DIR / "experiments.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def save(self, experiment_id: str, record: dict) -> None:
        entry = {"experiment_id": experiment_id, **record}
        with _FILE_LOCK:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    def load(self, experiment_id: str) -> dict | None:
        for record in self._iter_all():
            if record.get("experiment_id") == experiment_id:
                return record
        return None

    def list_recent(self, n: int = 20) -> list[dict]:
        return list(self._iter_all())[-n:]

    def find_best(self, metric: str = "total_pnl_pct", goal_contract: dict | None = None) -> dict | None:
        """Return the best approved experiment by metric.

        Fallback: if no approved experiment exists, also check rejected experiments whose
        metrics actually satisfy the goal contract (guards against false-reject bugs in reviewer).
        """
        from agent_research.config_utils import goal_contract_satisfied  # local import avoids circular
        best = None
        best_val = float("-inf")
        for record in self._iter_all():
            if record.get("reviewer_verdict") != "approve":
                continue
            val = record.get("metrics", {}).get(metric, float("-inf"))
            if val is None:
                val = float("-inf")
            if val > best_val:
                best_val = val
                best = record
        if best is not None:
            return best
        # Fallback: find rejected experiments that actually satisfy the goal contract metrics.
        # This catches cases where the reviewer incorrectly rejected a goal-satisfying result.
        if goal_contract:
            for record in self._iter_all():
                if record.get("reviewer_verdict") == "approve":
                    continue
                m = record.get("metrics") or {}
                if not m or not goal_contract_satisfied(m, goal_contract):
                    continue
                val = m.get(metric, float("-inf"))
                if val is None:
                    val = float("-inf")
                if val > best_val:
                    best_val = val
                    best = record
        return best

    def summarize_for_llm(self, n: int = 10) -> str:
        records = self.list_recent(n)
        if not records:
            return "No experiments run yet."
        lines = ["Recent experiments (oldest → newest):"]
        for rec in records:
            exp_id = rec.get("experiment_id", "?")
            metrics = rec.get("metrics", {})
            config = rec.get("config", {})
            scope = rec.get("research_scope", {})
            verdict = rec.get("reviewer_verdict", "?")
            experiment_type = rec.get("experiment_type", "iterative_research")
            notes = rec.get("reviewer_notes", "") or rec.get("notes", "")
            h_id = rec.get("config", {}).get("active_hypothesis_id") or rec.get("active_hypothesis_id") or ""
            is_clean = metrics.get("is_clean", True)
            clean_tag = "" if is_clean else " UNCLEAN"
            lines.append(
                f"  {exp_id}: "
                f"H={h_id or '?'} "
                f"trades={metrics.get('total_trades', '?')} "
                f"pairs={metrics.get('pairs_count', 0)} "
                f"sharpe={metrics.get('sharpe_proxy', 0):.2f} "
                f"pf={metrics.get('profit_factor', 0):.2f} "
                f"maxDD={metrics.get('max_drawdown_pct', 0):+.2f}% "
                f"score={metrics.get('research_score', 0):.1f} "
                f"verdict={verdict}{clean_tag} "
                f"type={experiment_type} "
                f"| {_config_summary(config)} "
                f"{_scope_summary(scope)} "
                f"{_notes_summary(notes)}"
            )
        return "\n".join(lines)

    def update(self, experiment_id: str, updates: dict) -> None:
        with _FILE_LOCK:
            lines = self._path.read_text(encoding="utf-8").splitlines()
            updated = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("experiment_id") == experiment_id:
                        record.update(updates)
                    updated.append(json.dumps(record, ensure_ascii=True))
                except json.JSONDecodeError:
                    updated.append(line)
            self._path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def list_configs_for_hypothesis(self, hypothesis_id: str) -> list[dict]:
        configs = []
        for rec in self._iter_all():
            if rec.get("active_hypothesis_id") == hypothesis_id:
                cfg = {k: v for k, v in rec.get("config", {}).items() if v is not None}
                if cfg:
                    configs.append(cfg)
        return configs

    def list_records_for_hypothesis(self, hypothesis_id: str, limit: int | None = None) -> list[dict]:
        records = [rec for rec in self._iter_all() if rec.get("active_hypothesis_id") == hypothesis_id]
        if limit is not None:
            return records[-limit:]
        return records

    def latest_code_record_for_hypothesis(self, hypothesis_id: str) -> dict | None:
        records = self.list_records_for_hypothesis(hypothesis_id)
        return records[-1] if records else None

    def find_by_setups_hash(self, csv_hash: str) -> dict | None:
        """Return the first experiment whose setups_csv produced this hash, or None."""
        for rec in self._iter_all():
            if rec.get("setups_csv_hash") == csv_hash:
                return rec
        return None

    def get_diagnoses_for_hypothesis(self, hypothesis_id: str, n: int = 5) -> str:
        records = self.list_records_for_hypothesis(hypothesis_id, limit=n * 3)
        diagnosed = [r for r in records if r.get("reviewer_diagnosis")]
        diagnosed = diagnosed[-n:]
        if not diagnosed:
            return f"No diagnoses recorded yet for {hypothesis_id}."
        lines = [f"Last {len(diagnosed)} diagnoses for {hypothesis_id} (oldest → newest):"]
        for rec in diagnosed:
            exp_id = rec.get("experiment_id", "?")
            d = rec.get("reviewer_diagnosis", {})
            root_cause = d.get("root_cause") or "unknown"
            insight = (d.get("diagnostic_insight") or "").strip()
            suggested = (d.get("suggested_direction") or "").strip()
            failed = ", ".join(d.get("failed_dimensions") or [])
            passing = ", ".join(d.get("passing_dimensions") or [])
            line = f"  {exp_id}: root_cause={root_cause}"
            if failed:
                line += f" | failed=[{failed}]"
            if passing:
                line += f" | passing=[{passing}]"
            if insight:
                line += f"\n    insight: {insight[:150]}"
            if suggested:
                line += f"\n    suggested: {suggested[:150]}"
            lines.append(line)
        return "\n".join(lines)

    def find_by_script_hash(self, script_hash: str) -> dict | None:
        """Return the first successfully executed experiment with this script hash, or None."""
        for rec in self._iter_all():
            if rec.get("script_hash") == script_hash and rec.get("script_hash"):
                v = rec.get("validation_result") or {}
                if v.get("sandbox_ok"):
                    return rec
        return None

    def recent_rejected_symbol_sets(self, limit: int = 12) -> list[list[str]]:
        result: list[list[str]] = []
        for rec in reversed(self.list_recent(limit * 3)):
            if rec.get("reviewer_verdict") != "reject":
                continue
            scope = rec.get("research_scope", {}) or {}
            selected = [str(s).strip() for s in (scope.get("selected_symbols") or []) if str(s).strip()]
            if selected:
                result.append(selected)
            if len(result) >= limit:
                break
        return result

    def _iter_all(self):
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def _config_summary(config: dict) -> str:
    keys = ["entry_z", "exit_z", "stop_z", "coint_p_threshold",
            "halflife_max_candles", "price_transform"]
    parts = [f"{k}={config[k]}" for k in keys if k in config]
    return " ".join(parts)


def _scope_summary(scope: dict) -> str:
    if not scope:
        return ""
    selected = scope.get("selected_symbols") or []
    if selected:
        preview = ",".join(selected[:5])
        suffix = "..." if len(selected) > 5 else ""
        return f"| selected_symbols[{len(selected)}]={preview}{suffix}"
    return ""


def _notes_summary(notes: str) -> str:
    if not notes:
        return ""
    compact = " ".join(str(notes).split())
    return f"| notes={compact[:120]}"


def generate_experiment_id() -> str:
    store = ExperimentStore()
    n = len(list(store._iter_all())) + 1
    now = datetime.now(timezone.utc)
    return f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{n:04d}"
