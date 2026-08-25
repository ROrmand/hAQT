"""Persist and reload benchmark run artifacts under outputs/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.profiler import ProfileMetrics


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_run(
    metrics: list[ProfileMetrics],
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write JSON + CSV for one benchmark run. Returns the JSON path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or utc_run_id()

    rows = [m.to_dict() for m in metrics]
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "metrics": rows,
    }

    json_path = output_dir / f"run_{run_id}.json"
    csv_path = output_dir / f"run_{run_id}.csv"
    latest_json = output_dir / "latest.json"
    latest_csv = output_dir / "latest.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    flat_rows = [_flatten_row(r) for r in rows]
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    pd.DataFrame(flat_rows).to_csv(latest_csv, index=False)
    return json_path


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {k: v for k, v in row.items() if k != "task_scores"}
    scores = row.get("task_scores") or {}
    for key, value in scores.items():
        flat[f"score_{key}"] = value
    notes = row.get("notes") or []
    flat["notes"] = "; ".join(notes) if isinstance(notes, list) else notes
    return flat


def load_run(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_latest_run(output_dir: str | Path) -> dict[str, Any] | None:
    """Load outputs/latest.json, or the newest run_*.json if latest is missing."""
    output_dir = Path(output_dir)
    latest = output_dir / "latest.json"
    if latest.is_file():
        return load_run(latest)

    runs = sorted(output_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    return load_run(runs[-1])


def metrics_from_run(payload: dict[str, Any]) -> list[ProfileMetrics]:
    out: list[ProfileMetrics] = []
    for row in payload.get("metrics") or []:
        out.append(
            ProfileMetrics(
                model_size=str(row.get("model_size", "")),
                precision=str(row.get("precision", "")),
                peak_vram_mb=row.get("peak_vram_mb"),
                ttft_ms=row.get("ttft_ms"),
                tokens_per_sec=row.get("tokens_per_sec"),
                total_latency_ms=row.get("total_latency_ms"),
                num_tokens=int(row.get("num_tokens") or 0),
                notes=list(row.get("notes") or []),
                task_scores=dict(row.get("task_scores") or {}),
            )
        )
    return out
