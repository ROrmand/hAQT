#!/usr/bin/env python3
"""Run the hAQT multi-precision GPU benchmark matrix and write outputs/.

Examples:
  python scripts/run_benchmark.py --limit 4 --models 2b --precisions int4_nf4
  python scripts/run_benchmark.py --limit 8 --models 2b,nemotron-mini-4b --vram-gb 16
  python scripts/run_benchmark.py --data data/cve_normalized.jsonl --limit 500 \\
      --models 2b,nemotron-mini-4b --label golden --save-scores
  python scripts/run_benchmark.py --dry-run --save-scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cve_context import build_cve_context  # noqa: E402
from src.data_loader import load_benchmark_dataset  # noqa: E402
from src.profiler import ProfileMetrics, profile_prompt_batch  # noqa: E402
from src.quantizer import (  # noqa: E402
    LoadConfig,
    ModelSize,
    Precision,
    recommended_precisions,
)
from src.results import save_run, save_scores  # noqa: E402
from src.scoring import reports_to_flat_metrics, score_example, score_predictions  # noqa: E402
from src.tasks import BenchmarkExample, TaskName, build_all_examples  # noqa: E402


def _parse_models(raw: str) -> list[ModelSize]:
    sizes: list[ModelSize] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        sizes.append(ModelSize(part))
    return sizes or [ModelSize.GEMMA2_2B]


def _parse_precisions(raw: str | None) -> list[Precision] | None:
    if not raw:
        return None
    out: list[Precision] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if part:
            out.append(Precision(part))
    return out


def build_matrix(
    models: list[ModelSize],
    vram_gb: float,
    precision_override: list[Precision] | None,
) -> list[LoadConfig]:
    configs: list[LoadConfig] = []
    for size in models:
        precisions = precision_override or recommended_precisions(size, vram_gb)
        for precision in precisions:
            configs.append(LoadConfig(model_size=size, precision=precision))
    return configs


def _gold_prediction(example: BenchmarkExample) -> str:
    if example.task == TaskName.CWE_CLASSIFICATION:
        return example.gold_cwe or "UNKNOWN"
    if example.gold_versions:
        return ", ".join(example.gold_versions)
    return "NONE"


def _build_example_index(
    examples: list[BenchmarkExample],
    cve_context: dict[str, dict],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        ctx = cve_context.get(ex.cve_id, {})
        rows.append(
            {
                "i": i,
                "cve_id": ex.cve_id,
                "task": ex.task.value,
                "severity": ctx.get("severity"),
                "published_year": ctx.get("published_year"),
                "gold_cwe": ex.gold_cwe,
                "gold_versions": list(ex.gold_versions),
            }
        )
    return rows


def _cell_items(
    examples: list[BenchmarkExample],
    generations: list[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i, (ex, pred) in enumerate(zip(examples, generations, strict=True)):
        scored = score_example(ex, pred)
        row = scored.to_dict()
        row["i"] = i
        row["prediction_text"] = pred
        items.append(row)
    return items


def run_matrix(
    configs: list[LoadConfig],
    examples: list[BenchmarkExample],
    *,
    max_new_tokens: int,
    dry_run: bool = False,
    save_scores: bool = False,
) -> tuple[list[ProfileMetrics], list[dict[str, Any]]]:
    prompts = [e.prompt for e in examples]
    results: list[ProfileMetrics] = []
    score_cells: list[dict[str, Any]] = []

    for i, config in enumerate(configs, start=1):
        label = f"{config.model_size.value}/{config.precision.value}"
        print(f"[{i}/{len(configs)}] {label} ...", flush=True)

        if dry_run:
            results.append(
                ProfileMetrics(
                    model_size=config.model_size.value,
                    precision=config.precision.value,
                    peak_vram_mb=None,
                    ttft_ms=None,
                    tokens_per_sec=None,
                    total_latency_ms=None,
                    notes=["dry-run"],
                )
            )
            if save_scores:
                generations = [_gold_prediction(ex) for ex in examples]
                score_cells.append(
                    {
                        "model_size": config.model_size.value,
                        "precision": config.precision.value,
                        "items": _cell_items(examples, generations),
                    }
                )
            continue

        profiled = profile_prompt_batch(
            config,
            prompts,
            max_new_tokens=max_new_tokens,
        )
        metrics = profiled.metrics

        if profiled.generations and len(profiled.generations) == len(examples):
            reports = score_predictions(examples, profiled.generations)
            metrics.task_scores = reports_to_flat_metrics(reports)
            if save_scores:
                score_cells.append(
                    {
                        "model_size": config.model_size.value,
                        "precision": config.precision.value,
                        "items": _cell_items(examples, profiled.generations),
                    }
                )
        elif profiled.generations:
            metrics.notes.append(
                f"Generation count mismatch "
                f"({len(profiled.generations)} vs {len(examples)}); skipped scoring."
            )

        if metrics.notes:
            print(f"  notes: {'; '.join(metrics.notes)}", flush=True)
        if metrics.task_scores:
            print(f"  scores: {metrics.task_scores}", flush=True)
        if metrics.peak_vram_mb is not None:
            print(
                f"  peak_vram={metrics.peak_vram_mb:.0f} MB  "
                f"tps={metrics.tokens_per_sec}",
                flush=True,
            )
        results.append(metrics)

    return results, score_cells


def main() -> int:
    parser = argparse.ArgumentParser(description="hAQT GPU benchmark runner")
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "cve_sample.json"),
        help="Path to NVD/CVE JSON sample or feed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Max CVE records to load (default: 4)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap total prompts after building CWE+version examples",
    )
    parser.add_argument(
        "--models",
        default="2b",
        help="Comma-separated models: 2b, nemotron-mini-4b, 9b (default: 2b)",
    )
    parser.add_argument(
        "--vram-gb",
        type=float,
        default=16.0,
        help="Assumed GPU VRAM for precision recommendations",
    )
    parser.add_argument(
        "--precisions",
        default=None,
        help="Optional override, e.g. fp16,int8,int4_nf4",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs"),
        help="Directory for JSON/CSV artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matrix and write placeholder metrics (no model load)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional tag stored in manifest (e.g. golden, smoke, cloud-9b)",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Write scores_<run_id>.json with per-example predictions for bootstrap CIs",
    )
    args = parser.parse_args()

    models = _parse_models(args.models)
    precision_override = _parse_precisions(args.precisions)
    configs = build_matrix(models, args.vram_gb, precision_override)

    records = load_benchmark_dataset(args.data, limit=args.limit)
    examples = build_all_examples(records)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    if not examples:
        print("No benchmark examples built from dataset; aborting.", file=sys.stderr)
        return 1

    print(
        f"Loaded {len(records)} CVEs -> {len(examples)} prompts; "
        f"{len(configs)} matrix cells",
        flush=True,
    )
    for cfg in configs:
        print(f"  - {cfg.model_size.value} / {cfg.precision.value}", flush=True)

    metrics, score_cells = run_matrix(
        configs,
        examples,
        max_new_tokens=args.max_new_tokens,
        dry_run=args.dry_run,
        save_scores=args.save_scores,
    )

    meta = {
        "data": str(args.data),
        "limit": args.limit,
        "n_records": len(records),
        "n_examples": len(examples),
        "models": [m.value for m in models],
        "vram_gb": args.vram_gb,
        "max_new_tokens": args.max_new_tokens,
        "dry_run": args.dry_run,
        "label": args.label,
        "has_scores": args.save_scores and bool(score_cells),
    }

    path = save_run(metrics, args.output_dir, meta=meta)

    if args.save_scores and score_cells:
        from src.results import load_run

        run_payload = load_run(path)
        run_number = int(run_payload["run_number"])
        run_id = str(run_payload["run_id"])
        cve_context = build_cve_context(records)
        scores_path_written = save_scores(
            args.output_dir,
            run_number=run_number,
            run_id=run_id,
            created_at=str(run_payload.get("created_at", "")),
            example_index=_build_example_index(examples, cve_context),
            cells=score_cells,
        )
        print(f"Wrote {scores_path_written}", flush=True)

    print(f"Wrote {path} (+ latest.json / CSV)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
