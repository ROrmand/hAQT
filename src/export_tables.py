"""Format benchmark run artifacts as comparison tables for reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

PRECISION_ORDER = ["fp16", "int8", "int4_nf4"]
PRECISION_LABELS = {
    "fp16": "FP16",
    "int8": "INT8",
    "int4_nf4": "INT4 NF4",
}
MODEL_LABELS = {
    "2b": "Gemma-2 2B",
    "nemotron-mini-4b": "Nemotron-Mini-4B",
    "9b": "Gemma-2 9B",
}

SUMMARY_COLUMNS = [
    ("model_size", "Model"),
    ("precision", "Precision"),
    ("peak_vram_mb", "Peak VRAM (MB)"),
    ("tokens_per_sec", "Throughput (tok/s)"),
    ("ttft_ms", "TTFT (ms)"),
    ("cwe_accuracy", "CWE accuracy"),
    ("version_f1", "Version F1"),
]


def metrics_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten run JSON metrics into a chart-friendly DataFrame."""
    rows: list[dict[str, Any]] = []
    for row in payload.get("metrics") or []:
        scores = row.get("task_scores") or {}
        rows.append(
            {
                "model_size": str(row.get("model_size", "")),
                "precision": str(row.get("precision", "")),
                "peak_vram_mb": row.get("peak_vram_mb"),
                "tokens_per_sec": row.get("tokens_per_sec"),
                "ttft_ms": row.get("ttft_ms"),
                "total_latency_ms": row.get("total_latency_ms"),
                "cwe_accuracy": scores.get("cwe_classification.accuracy"),
                "version_f1": scores.get("version_parsing.f1"),
                "version_accuracy": scores.get("version_parsing.accuracy"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["model_label"] = df["model_size"].map(lambda k: MODEL_LABELS.get(k, k))
    df["precision_label"] = df["precision"].map(
        lambda p: PRECISION_LABELS.get(p, str(p).upper())
    )
    df["precision"] = pd.Categorical(
        df["precision"], categories=PRECISION_ORDER, ordered=True
    )
    return df.sort_values(["model_size", "precision"], kind="stable")


def quantization_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Δ vs FP16 baseline per model for VRAM, throughput, and task scores."""
    if df.empty:
        return df
    baseline = df[df["precision"].astype(str) == "fp16"].set_index("model_size")
    deltas: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        model = str(row["model_size"])
        precision = str(row["precision"])
        if precision == "fp16":
            continue
        base = baseline.loc[model] if model in baseline.index else None
        entry: dict[str, Any] = {
            "model_label": row.get("model_label", model),
            "precision_label": row.get("precision_label", precision),
            "precision": precision,
        }
        for col, delta_name in (
            ("peak_vram_mb", "delta_vram_mb"),
            ("tokens_per_sec", "delta_tok_s"),
            ("cwe_accuracy", "delta_cwe_acc"),
            ("version_f1", "delta_version_f1"),
        ):
            val = row.get(col)
            base_val = base.get(col) if base is not None else None
            if pd.notna(val) and base is not None and pd.notna(base_val):
                if col == "peak_vram_mb":
                    entry[delta_name] = float(val) - float(base_val)
                else:
                    entry[delta_name] = float(val) - float(base_val)
            else:
                entry[delta_name] = None
        if base is not None and pd.notna(row.get("peak_vram_mb")) and pd.notna(
            base.get("peak_vram_mb")
        ):
            entry["vram_reduction_pct"] = (
                1.0 - float(row["peak_vram_mb"]) / float(base["peak_vram_mb"])
            ) * 100.0
        deltas.append(entry)
    return pd.DataFrame(deltas)


def _fmt_num(value: Any, *, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if pct:
        return f"{float(value):.1%}"
    if isinstance(value, float) and abs(value) < 1 and not pct:
        return f"{float(value):.3f}"
    return f"{float(value):.1f}"


def format_markdown_summary(
    payload: dict[str, Any],
    *,
    include_deltas: bool = True,
) -> str:
    """One-page Markdown table for README / slides."""
    meta = payload.get("meta") or {}
    df = metrics_to_dataframe(payload)
    lines = [
        f"# hAQT benchmark — {payload.get('run_number', payload.get('run_id', 'unknown'))}",
        "",
        f"- **Dataset:** `{meta.get('data', '—')}`",
        f"- **CVEs:** {meta.get('n_records', '—')} | **Prompts:** {meta.get('n_examples', '—')}",
        f"- **Models:** {', '.join(meta.get('models') or [])}",
        f"- **VRAM budget:** {meta.get('vram_gb', '—')} GB",
        "",
        "## VRAM / speed / accuracy",
        "",
        "| Model | Precision | Peak VRAM (MB) | tok/s | CWE acc. | Version F1 |",
        "|-------|-----------|----------------|-------|----------|------------|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| {model} | {prec} | {vram} | {tps} | {cwe} | {vf1} |".format(
                model=row.get("model_label", row["model_size"]),
                prec=row.get("precision_label", row["precision"]),
                vram=_fmt_num(row.get("peak_vram_mb")),
                tps=_fmt_num(row.get("tokens_per_sec")),
                cwe=_fmt_num(row.get("cwe_accuracy"), pct=True),
                vf1=_fmt_num(row.get("version_f1")),
            )
        )
    if include_deltas:
        delta_df = quantization_deltas(df)
        if not delta_df.empty:
            lines.extend(
                [
                    "",
                    "## Quantization delta vs FP16",
                    "",
                    "| Model | Precision | d VRAM (MB) | VRAM saved | d tok/s | d CWE | d Version F1 |",
                    "|-------|-----------|-------------|------------|---------|-------|--------------|",
                ]
            )
            for _, row in delta_df.iterrows():
                lines.append(
                    "| {model} | {prec} | {dvram} | {saved} | {dtps} | {dcwe} | {dvf1} |".format(
                        model=row["model_label"],
                        prec=row["precision_label"],
                        dvram=_fmt_num(row.get("delta_vram_mb")),
                        saved=(
                            f"{row['vram_reduction_pct']:.0f}%"
                            if pd.notna(row.get("vram_reduction_pct"))
                            else "—"
                        ),
                        dtps=_fmt_num(row.get("delta_tok_s")),
                        dcwe=_fmt_num(row.get("delta_cwe_acc")),
                        dvf1=_fmt_num(row.get("delta_version_f1")),
                    )
                )
    lines.append("")
    return "\n".join(lines)


def format_latex_summary(payload: dict[str, Any]) -> str:
    """LaTeX tabular for papers."""
    df = metrics_to_dataframe(payload)
    run_id = payload.get("run_id", "unknown")
    lines = [
        f"% hAQT run {run_id}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{VRAM, throughput, and task quality by model and precision.}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Precision & VRAM (MB) & tok/s & CWE acc. & Version F1 \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row.get('model_label', row['model_size'])} & "
            f"{row.get('precision_label', row['precision'])} & "
            f"{_fmt_num(row.get('peak_vram_mb'))} & "
            f"{_fmt_num(row.get('tokens_per_sec'))} & "
            f"{_fmt_num(row.get('cwe_accuracy'), pct=True)} & "
            f"{_fmt_num(row.get('version_f1'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)
