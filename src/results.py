"""Persist and reload benchmark run artifacts under outputs/."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.profiler import ProfileMetrics

RUN_ID_WIDTH = 3
_MANIFEST_NAME = "run_manifest.json"
_NUMBERED_RUN_RE = re.compile(r"^run_(\d+)$", re.IGNORECASE)


def format_run_id(run_number: int) -> str:
    """Zero-padded run id used in filenames, e.g. ``001``."""
    return str(run_number).zfill(RUN_ID_WIDTH)


def parse_run_number_from_stem(stem: str) -> int | None:
    """Extract run number from ``run_001``; ``None`` for legacy timestamp ids."""
    match = _NUMBERED_RUN_RE.match(stem)
    if not match:
        return None
    return int(match.group(1))


def is_numbered_run_path(path: Path) -> bool:
    return parse_run_number_from_stem(path.stem) is not None


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / _MANIFEST_NAME


def load_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Load ``outputs/run_manifest.json`` or return an empty manifest."""
    path = _manifest_path(Path(output_dir))
    if not path.is_file():
        return {"next_run_number": 1, "runs": []}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("next_run_number", 1)
    data.setdefault("runs", [])
    return data


def save_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    path = _manifest_path(Path(output_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _max_numbered_run_on_disk(output_dir: Path) -> int:
    highest = 0
    for path in output_dir.glob("run_*.json"):
        n = parse_run_number_from_stem(path.stem)
        if n is not None:
            highest = max(highest, n)
    return highest


def allocate_run_number(output_dir: str | Path) -> int:
    """Return the next sequential run number (does not write manifest)."""
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir)
    on_disk = _max_numbered_run_on_disk(output_dir)
    next_from_manifest = int(manifest.get("next_run_number") or 1)
    return max(next_from_manifest, on_disk + 1)


def register_run(
    output_dir: str | Path,
    *,
    run_number: int,
    created_at: str,
    label: str | None = None,
    legacy_id: str | None = None,
) -> None:
    """Append a run entry to the manifest and bump ``next_run_number``."""
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir)
    run_id = format_run_id(run_number)
    entry = {
        "run_number": run_number,
        "run_id": run_id,
        "created_at": created_at,
        "file": f"run_{run_id}.json",
        "label": label,
    }
    if legacy_id:
        entry["legacy_id"] = legacy_id
    runs: list[dict[str, Any]] = list(manifest.get("runs") or [])
    runs = [r for r in runs if r.get("run_number") != run_number]
    runs.append(entry)
    runs.sort(key=lambda r: int(r.get("run_number") or 0))
    manifest["runs"] = runs
    manifest["next_run_number"] = max(
        int(manifest.get("next_run_number") or 1),
        run_number + 1,
    )
    save_manifest(output_dir, manifest)


def run_display_name(
    payload: dict[str, Any],
    *,
    include_file: bool = False,
) -> str:
    """Human label for UI: ``Run 13 — golden``."""
    run_number = payload.get("run_number")
    run_id = payload.get("run_id", "—")
    if run_number is None and isinstance(run_id, str):
        n = parse_run_number_from_stem(f"run_{run_id}")
        run_number = n
    label = (payload.get("meta") or {}).get("label") or payload.get("label")
    parts = [f"Run {run_number}" if run_number is not None else f"Run {run_id}"]
    if label:
        parts.append(str(label))
    if include_file:
        parts.append(f"({payload.get('created_at', '')[:10]})")
    return " — ".join(parts) if len(parts) > 1 else parts[0]


def save_run(
    metrics: list[ProfileMetrics],
    output_dir: str | Path,
    *,
    run_number: int | None = None,
    run_id: str | None = None,  # deprecated alias; ignored when run_number set
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write JSON + CSV for one benchmark run. Returns the JSON path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(meta or {})
    if run_number is None:
        if run_id is not None and parse_run_number_from_stem(f"run_{run_id}") is not None:
            run_number = int(run_id)
        else:
            run_number = allocate_run_number(output_dir)

    run_id = format_run_id(run_number)
    created_at = datetime.now(timezone.utc).isoformat()
    label = meta.get("label")

    rows = [m.to_dict() for m in metrics]
    payload: dict[str, Any] = {
        "run_number": run_number,
        "run_id": run_id,
        "created_at": created_at,
        "meta": meta,
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

    register_run(
        output_dir,
        run_number=run_number,
        created_at=created_at,
        label=label,
    )
    return json_path


def scores_path(output_dir: str | Path, run_id: str) -> Path:
    return Path(output_dir) / f"scores_{run_id}.json"


def save_scores(
    output_dir: str | Path,
    *,
    run_number: int,
    run_id: str,
    created_at: str,
    example_index: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> Path:
    """Write per-example predictions and scores for bootstrap / error analysis."""
    output_dir = Path(output_dir)
    path = scores_path(output_dir, run_id)
    payload = {
        "run_number": run_number,
        "run_id": run_id,
        "created_at": created_at,
        "example_index": example_index,
        "cells": cells,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_scores(output_dir: str | Path, run_id: str) -> dict[str, Any] | None:
    path = scores_path(output_dir, run_id)
    if not path.is_file():
        return None
    return load_run(path)


def scores_for_run_payload(
    output_dir: str | Path,
    run_payload: dict[str, Any],
) -> dict[str, Any] | None:
    run_id = str(run_payload.get("run_id", ""))
    if not run_id:
        return None
    return load_scores(output_dir, run_id)


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


def _run_sort_key(path: Path) -> tuple[int, int | float]:
    """Numbered runs first (high→low), then legacy timestamp runs by mtime."""
    n = parse_run_number_from_stem(path.stem)
    if n is not None:
        return (0, -n)
    return (1, -path.stat().st_mtime)


def list_runs(output_dir: str | Path) -> list[Path]:
    """Return run JSON paths newest-first (by run number, then legacy mtime)."""
    output_dir = Path(output_dir)
    paths = [p for p in output_dir.glob("run_*.json") if p.name != "latest.json"]
    return sorted(paths, key=_run_sort_key)


def load_latest_run(output_dir: str | Path) -> dict[str, Any] | None:
    """Load outputs/latest.json, or the newest numbered run if latest is missing."""
    output_dir = Path(output_dir)
    latest = output_dir / "latest.json"
    if latest.is_file():
        return load_run(latest)

    runs = list_runs(output_dir)
    if not runs:
        return None
    return load_run(runs[0])


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


def migrate_legacy_runs(output_dir: str | Path, *, dry_run: bool = False) -> list[str]:
    """
    Rename timestamped ``run_YYYYMMDDTHHMMSSZ`` files to ``run_001`` … sequentially.

    Returns a log of actions taken (or that would be taken in dry-run mode).
    """
    output_dir = Path(output_dir)
    log: list[str] = []

    legacy: list[Path] = []
    for path in output_dir.glob("run_*.json"):
        if path.name == "latest.json":
            continue
        if is_numbered_run_path(path):
            continue
        legacy.append(path)

    legacy.sort(key=lambda p: p.stat().st_mtime)

    start = _max_numbered_run_on_disk(output_dir) + 1
    if not legacy:
        log.append("No legacy timestamp runs to migrate.")
        return log

    renames: list[tuple[Path, Path, int, str]] = []
    for offset, old_json in enumerate(legacy):
        run_number = start + offset
        run_id = format_run_id(run_number)
        new_json = output_dir / f"run_{run_id}.json"
        legacy_id = old_json.stem.removeprefix("run_")
        renames.append((old_json, new_json, run_number, legacy_id))

    for old_json, new_json, run_number, legacy_id in renames:
        old_csv = old_json.with_suffix(".csv")
        new_csv = new_json.with_suffix(".csv")
        old_table_md = output_dir / f"table_{legacy_id}.md"
        new_table_md = output_dir / f"table_{format_run_id(run_number)}.md"
        old_table_tex = output_dir / f"table_{legacy_id}.tex"
        new_table_tex = output_dir / f"table_{format_run_id(run_number)}.tex"

        if dry_run:
            log.append(f"Would migrate {old_json.name} -> {new_json.name}")
            continue

        payload = load_run(old_json)
        payload["run_number"] = run_number
        payload["run_id"] = format_run_id(run_number)
        meta = payload.setdefault("meta", {})
        meta["legacy_id"] = legacy_id
        label = meta.get("label")
        if not label:
            if meta.get("n_records") == 500:
                label = "golden"
            elif meta.get("dry_run"):
                label = "dry-run"
            elif meta.get("n_records", 0) <= 8:
                label = "smoke"
            if label:
                meta["label"] = label

        with new_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if old_csv.is_file():
            pd.read_csv(old_csv).to_csv(new_csv, index=False)
            old_csv.unlink()

        if old_table_md.is_file():
            old_table_md.rename(new_table_md)
        if old_table_tex.is_file():
            old_table_tex.rename(new_table_tex)

        old_json.unlink()
        register_run(
            output_dir,
            run_number=run_number,
            created_at=str(payload.get("created_at") or ""),
            label=label,
            legacy_id=legacy_id,
        )
        log.append(f"Migrated {legacy_id} -> run_{format_run_id(run_number)}")

    if not dry_run and renames:
        highest = renames[-1][2]
        latest_src = output_dir / f"run_{format_run_id(highest)}.json"
        if latest_src.is_file():
            (output_dir / "latest.json").write_text(
                latest_src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            csv_src = latest_src.with_suffix(".csv")
            if csv_src.is_file():
                pd.read_csv(csv_src).to_csv(output_dir / "latest.csv", index=False)
        log.append(f"Updated latest.json -> run_{format_run_id(highest)}")

    return log
