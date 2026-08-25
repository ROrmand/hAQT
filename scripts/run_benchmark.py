#!/usr/bin/env python3
"""Run the hAQT multi-precision GPU benchmark matrix and write outputs/.

Examples:
  python scripts/run_benchmark.py --limit 4 --models 2b --precisions int4_nf4
  python scripts/run_benchmark.py --vram-gb 16 --limit 8
  python scripts/run_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_benchmark_dataset  # noqa: E402
from src.profiler import ProfileMetrics, profile_prompt_batch  # noqa: E402
from src.quantizer import (  # noqa: E402
    LoadConfig,
    ModelSize,
    Precision,
    recommended_precisions,
)
from src.results import save_run  # noqa: E402
from src.scoring import reports_to_flat_metrics, score_predictions  # noqa: E402
from src.tasks import build_all_examples  # noqa: E402


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


def run_matrix(
    configs: list[LoadConfig],
    examples,
    *,
    max_new_tokens: int,
    dry_run: bool = False,
) -> list[ProfileMetrics]:
    prompts = [e.prompt for e in examples]
    results: list[ProfileMetrics] = []

    for i, config in enumerate(configs, start=1):
        label = f"{config.model_size.value}/{config.precision.value}"
        print(f"[{i}/{len(configs)}] {label} …", flush=True)

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

    return results


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
        help="Comma-separated model sizes: 2b,9b (default: 2b)",
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
        f"Loaded {len(records)} CVEs → {len(examples)} prompts; "
        f"{len(configs)} matrix cells",
        flush=True,
    )
    for cfg in configs:
        print(f"  - {cfg.model_size.value} / {cfg.precision.value}", flush=True)

    metrics = run_matrix(
        configs,
        examples,
        max_new_tokens=args.max_new_tokens,
        dry_run=args.dry_run,
    )

    path = save_run(
        metrics,
        args.output_dir,
        meta={
            "data": str(args.data),
            "limit": args.limit,
            "n_records": len(records),
            "n_examples": len(examples),
            "models": [m.value for m in models],
            "vram_gb": args.vram_gb,
            "max_new_tokens": args.max_new_tokens,
            "dry_run": args.dry_run,
        },
    )
    print(f"Wrote {path} (+ latest.json / CSV)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
