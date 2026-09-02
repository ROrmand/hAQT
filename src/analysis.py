"""Statistical analysis: bootstrap CIs, stratified breakdowns, quantization tax."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.stats import bootstrap_ci
from src.tasks import TaskName


def _metric_value(item: dict[str, Any], task: str) -> float:
    if task == TaskName.CWE_CLASSIFICATION.value:
        return float(item.get("detail", {}).get("exact_match", 0.0))
    if task == TaskName.VERSION_PARSING.value:
        return float(item.get("detail", {}).get("f1", 0.0))
    return float(item.get("correct", 0.0))


def _task_items(cell: dict[str, Any], task: str) -> list[dict[str, Any]]:
    return [it for it in cell.get("items") or [] if it.get("task") == task]


def bootstrap_task_ci(
    items: list[dict[str, Any]],
    task: str,
    *,
    n_resamples: int = 2000,
) -> dict[str, float]:
    values = [_metric_value(it, task) for it in items]
    if not values:
        return {"n": 0.0, "mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    mean, lo, hi = bootstrap_ci(values, n_resamples=n_resamples)
    return {"n": float(len(values)), "mean": mean, "ci_lo": lo, "ci_hi": hi}


def bootstrap_all_cells(
    scores: dict[str, Any],
    *,
    n_resamples: int = 2000,
) -> list[dict[str, Any]]:
    """95% bootstrap CIs per matrix cell and task."""
    rows: list[dict[str, Any]] = []
    for cell in scores.get("cells") or []:
        for task in (TaskName.CWE_CLASSIFICATION.value, TaskName.VERSION_PARSING.value):
            items = _task_items(cell, task)
            if not items:
                continue
            ci = bootstrap_task_ci(items, task, n_resamples=n_resamples)
            metric = "accuracy" if task == TaskName.CWE_CLASSIFICATION.value else "f1"
            rows.append(
                {
                    "model_size": cell.get("model_size"),
                    "precision": cell.get("precision"),
                    "task": task,
                    "metric": metric,
                    **ci,
                }
            )
    return rows


def _example_lookup(scores: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(ex["i"]): ex for ex in scores.get("example_index") or []}


def stratified_breakdown(
    scores: dict[str, Any],
    cell: dict[str, Any],
    task: str,
    *,
    by: str,
) -> list[dict[str, Any]]:
    """Break down a cell's task metric by severity, published_year, or gold_cwe."""
    lookup = _example_lookup(scores)
    buckets: dict[str, list[float]] = defaultdict(list)

    for item in _task_items(cell, task):
        meta = lookup.get(int(item["i"]), {})
        if by == "severity":
            key = str(meta.get("severity") or "UNKNOWN")
        elif by == "year":
            year = meta.get("published_year")
            key = str(year) if year is not None else "UNKNOWN"
        elif by == "cwe":
            key = str(meta.get("gold_cwe") or "UNKNOWN")
        else:
            raise ValueError(f"Unknown breakdown dimension: {by}")
        buckets[key].append(_metric_value(item, task))

    rows: list[dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        values = buckets[key]
        mean, lo, hi = bootstrap_ci(values)
        rows.append(
            {
                "slice": key,
                "n": len(values),
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return rows


def quantization_tax(
    scores: dict[str, Any],
    *,
    baseline_precision: str = "fp16",
    n_resamples: int = 2000,
) -> list[dict[str, Any]]:
    """
    Per-model task means at FP16 vs INT8/INT4 with bootstrap CIs and deltas.
    """
    cells = scores.get("cells") or []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        by_key[(str(cell["model_size"]), str(cell["precision"]))] = cell

    rows: list[dict[str, Any]] = []
    models = sorted({str(c["model_size"]) for c in cells})
    for model in models:
        base = by_key.get((model, baseline_precision))
        if not base:
            continue
        for task in (TaskName.CWE_CLASSIFICATION.value, TaskName.VERSION_PARSING.value):
            base_items = _task_items(base, task)
            if not base_items:
                continue
            base_ci = bootstrap_task_ci(base_items, task, n_resamples=n_resamples)
            for precision in ("int8", "int4_nf4"):
                cell = by_key.get((model, precision))
                if not cell:
                    continue
                items = _task_items(cell, task)
                if not items:
                    continue
                ci = bootstrap_task_ci(items, task, n_resamples=n_resamples)
                rows.append(
                    {
                        "model_size": model,
                        "task": task,
                        "baseline_precision": baseline_precision,
                        "precision": precision,
                        "baseline_mean": base_ci["mean"],
                        "baseline_ci_lo": base_ci["ci_lo"],
                        "baseline_ci_hi": base_ci["ci_hi"],
                        "mean": ci["mean"],
                        "ci_lo": ci["ci_lo"],
                        "ci_hi": ci["ci_hi"],
                        "delta_mean": ci["mean"] - base_ci["mean"],
                    }
                )
    return rows


def format_quantization_tax_report(
    run_payload: dict[str, Any],
    scores: dict[str, Any],
) -> str:
    """Markdown report for Tier 5.1 — quantization tax on security tasks."""
    run_no = run_payload.get("run_number", run_payload.get("run_id", "?"))
    meta = run_payload.get("meta") or {}
    tax_rows = quantization_tax(scores)
    bootstrap_rows = bootstrap_all_cells(scores)

    lines = [
        f"# Quantization tax report — Run {run_no}",
        "",
        f"- **Dataset:** `{meta.get('data', '—')}`",
        f"- **CVEs:** {meta.get('n_records', '—')} | **Prompts:** {meta.get('n_examples', '—')}",
        f"- **Models:** {', '.join(meta.get('models') or [])}",
        "",
        "## Summary (Tier 5.1)",
        "",
        "Compares **INT8 / INT4 NF4** task quality vs **FP16** using per-example bootstrap 95% CIs.",
        "CWE = exact-match accuracy; Version = per-example F1.",
        "",
        "### Delta vs FP16 (mean point estimate)",
        "",
        "| Model | Task | Precision | FP16 mean | Quant mean | Delta |",
        "|-------|------|-----------|-----------|------------|-------|",
    ]
    for row in tax_rows:
        task_label = "CWE acc." if row["task"] == TaskName.CWE_CLASSIFICATION.value else "Version F1"
        lines.append(
            f"| {row['model_size']} | {task_label} | {row['precision']} "
            f"| {row['baseline_mean']:.3f} | {row['mean']:.3f} | {row['delta_mean']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "### Bootstrap 95% CIs (all cells)",
            "",
            "| Model | Precision | Task | Mean | CI low | CI high | n |",
            "|-------|-----------|------|------|--------|---------|---|",
        ]
    )
    for row in bootstrap_rows:
        task_label = "CWE" if row["task"] == TaskName.CWE_CLASSIFICATION.value else "Version"
        lines.append(
            f"| {row['model_size']} | {row['precision']} | {task_label} "
            f"| {row['mean']:.3f} | {row['ci_lo']:.3f} | {row['ci_hi']:.3f} | {int(row['n'])} |"
        )

    # Narrative bullets from tax data
    lines.extend(["", "## Interpretation", ""])
    cwe_deltas = [r["delta_mean"] for r in tax_rows if r["task"] == TaskName.CWE_CLASSIFICATION.value]
    ver_deltas = [r["delta_mean"] for r in tax_rows if r["task"] == TaskName.VERSION_PARSING.value]
    if cwe_deltas:
        max_cwe = max(abs(d) for d in cwe_deltas)
        lines.append(
            f"- **CWE classification:** max |delta| vs FP16 = {max_cwe:.3f} — "
            + ("quantization-neutral in practice." if max_cwe < 0.02 else "measurable CWE degradation at some precisions.")
        )
    if ver_deltas:
        max_ver = max(abs(d) for d in ver_deltas)
        lines.append(
            f"- **Version parsing:** max |delta| vs FP16 = {max_ver:.3f} — "
            + ("small parsing tax under INT4." if max_ver < 0.05 else "version parsing more sensitive to quantization.")
        )
    lines.append("")
    return "\n".join(lines)


def error_items(
    scores: dict[str, Any],
    *,
    model_size: str | None = None,
    precision: str | None = None,
    task: str | None = None,
    errors_only: bool = True,
) -> list[dict[str, Any]]:
    """Flat list of per-example rows for drill-down tables."""
    lookup = _example_lookup(scores)
    rows: list[dict[str, Any]] = []
    for cell in scores.get("cells") or []:
        if model_size and cell.get("model_size") != model_size:
            continue
        if precision and cell.get("precision") != precision:
            continue
        for item in cell.get("items") or []:
            if task and item.get("task") != task:
                continue
            if errors_only and item.get("correct"):
                continue
            meta = lookup.get(int(item["i"]), {})
            rows.append(
                {
                    "model_size": cell.get("model_size"),
                    "precision": cell.get("precision"),
                    "cve_id": meta.get("cve_id", item.get("cve_id")),
                    "task": item.get("task"),
                    "severity": meta.get("severity"),
                    "published_year": meta.get("published_year"),
                    "gold_cwe": meta.get("gold_cwe"),
                    "gold": item.get("gold"),
                    "predicted": item.get("predicted"),
                    "prediction_text": item.get("prediction_text"),
                    "correct": item.get("correct"),
                }
            )
    return rows
