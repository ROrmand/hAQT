#!/usr/bin/env python3
"""Analyze a benchmark run: deltas, bootstrap CIs, stratified slices, quantization tax."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import (  # noqa: E402
    bootstrap_all_cells,
    format_quantization_tax_report,
    quantization_tax,
    stratified_breakdown,
)
from src.export_tables import metrics_to_dataframe, quantization_deltas  # noqa: E402
from src.results import load_latest_run, load_run, scores_for_run_payload  # noqa: E402
from src.tasks import TaskName  # noqa: E402


def _print_deltas(payload: dict) -> None:
    run_no = payload.get("run_number", payload.get("run_id"))
    print(f"Run {run_no} | meta: {json.dumps(payload.get('meta') or {})}")
    print()
    df = metrics_to_dataframe(payload)
    delta_df = quantization_deltas(df)
    if delta_df.empty:
        print("No quantization deltas (need FP16 + quantized rows).")
        return
    cols = [
        "model_label",
        "precision_label",
        "delta_vram_mb",
        "vram_reduction_pct",
        "delta_tok_s",
        "delta_cwe_acc",
        "delta_version_f1",
    ]
    present = [c for c in cols if c in delta_df.columns]
    print(delta_df[present].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _print_bootstrap(scores: dict) -> None:
    rows = bootstrap_all_cells(scores)
    if not rows:
        print("No per-example scores for bootstrap CIs.")
        return
    print()
    print("Bootstrap 95% CIs (per-example):")
    print(
        f"{'Model':<18} {'Precision':<10} {'Task':<20} {'Mean':>6} {'CI lo':>6} {'CI hi':>6} {'n':>5}"
    )
    for row in rows:
        task = "CWE acc" if row["task"] == TaskName.CWE_CLASSIFICATION.value else "Version F1"
        print(
            f"{row['model_size']:<18} {row['precision']:<10} {task:<20} "
            f"{row['mean']:6.3f} {row['ci_lo']:6.3f} {row['ci_hi']:6.3f} {int(row['n']):5d}"
        )


def _print_tax(scores: dict) -> None:
    rows = quantization_tax(scores)
    if not rows:
        return
    print()
    print("Quantization tax vs FP16 (per-example means):")
    print(
        f"{'Model':<18} {'Task':<12} {'Prec':<10} {'FP16':>6} {'Quant':>6} {'Delta':>7}"
    )
    for row in rows:
        task = "CWE" if row["task"] == TaskName.CWE_CLASSIFICATION.value else "Version"
        print(
            f"{row['model_size']:<18} {task:<12} {row['precision']:<10} "
            f"{row['baseline_mean']:6.3f} {row['mean']:6.3f} {row['delta_mean']:+7.3f}"
        )


def _print_stratified(scores: dict, *, by: str) -> None:
    print()
    print(f"Stratified breakdown by {by} (version F1, first cell with data):")
    for cell in scores.get("cells") or []:
        rows = stratified_breakdown(
            scores,
            cell,
            TaskName.VERSION_PARSING.value,
            by=by,
        )
        if not rows:
            continue
        print(
            f"  {cell['model_size']}/{cell['precision']} — "
            + ", ".join(f"{r['slice']}: {r['mean']:.2f} (n={r['n']})" for r in rows[:6])
        )
        break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write quantization_tax_<run_id>.md report",
    )
    parser.add_argument(
        "--stratify",
        choices=("severity", "year", "cwe"),
        default=None,
        help="Print one stratified breakdown slice",
    )
    args = parser.parse_args()

    if args.run is None:
        payload = load_latest_run(args.output_dir)
        if payload is None:
            print("No run JSON found.", file=sys.stderr)
            return 1
    else:
        payload = load_run(args.run)

    _print_deltas(payload)
    scores = scores_for_run_payload(args.output_dir, payload)
    if scores:
        _print_bootstrap(scores)
        _print_tax(scores)
        if args.stratify:
            _print_stratified(scores, by=args.stratify)
        if args.report:
            report = format_quantization_tax_report(payload, scores)
            run_id = str(payload.get("run_id", "unknown"))
            out = args.output_dir / f"quantization_tax_{run_id}.md"
            out.write_text(report, encoding="utf-8")
            print()
            print(f"Wrote {out}")
    else:
        run_id = payload.get("run_id", "?")
        print()
        print(
            f"No scores_{run_id}.json — re-run with --save-scores to enable bootstrap CIs.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
