"""hAQT Streamlit workbench — comparative VRAM / speed / accuracy dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_benchmark_dataset
from src.profiler import empty_matrix
from src.quantizer import ModelSize, Precision, recommended_precisions
from src.results import load_latest_run, metrics_from_run
from src.scoring import reports_to_flat_metrics, score_predictions
from src.tasks import build_all_examples

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "cve_sample.json"
OUTPUTS_DIR = ROOT / "outputs"

st.set_page_config(page_title="hAQT Benchmarker", layout="wide")
st.title("hAQT")
st.caption("Hardware-Aware Quantization & Security Benchmarker")

with st.sidebar:
    st.header("Run config")
    model_choices = st.multiselect(
        "Models",
        options=[m.value for m in ModelSize],
        default=[ModelSize.GEMMA2_2B.value, ModelSize.GEMMA2_9B.value],
    )
    vram_gb = st.slider("Assumed GPU VRAM (GB)", 8, 80, 16)
    data_path = st.text_input("CVE dataset path", str(SAMPLE_PATH))
    st.markdown(
        "Fill metrics with:\n"
        "`python scripts/run_benchmark.py --limit 4 --models 2b`"
    )

records = []
load_error = None
try:
    records = load_benchmark_dataset(data_path, limit=50)
except Exception as exc:
    load_error = str(exc)

col1, col2, col3 = st.columns(3)
col1.metric("Loaded CVEs", len(records))
col2.metric("With CWE labels", sum(1 for r in records if r.primary_cwe))
col3.metric("With version mentions", sum(1 for r in records if r.version_mentions))

if load_error:
    st.error(f"Failed to load dataset: {load_error}")
elif records:
    st.subheader("Sample normalized records")
    st.dataframe(
        pd.DataFrame([r.to_dict() for r in records]),
        use_container_width=True,
        hide_index=True,
    )

    examples = build_all_examples(records)
    st.subheader("Task examples")
    st.caption(f"{len(examples)} prompts (CWE classification + version parsing)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "task": e.task.value,
                    "cve_id": e.cve_id,
                    "gold_cwe": e.gold_cwe,
                    "gold_versions": ", ".join(e.gold_versions),
                    "prompt_preview": e.prompt[:160] + ("…" if len(e.prompt) > 160 else ""),
                }
                for e in examples
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Offline sanity check of the scorer (no model required).
    demo_preds = []
    for e in examples:
        if e.task.value == "cwe_classification":
            demo_preds.append(e.gold_cwe or "UNKNOWN")
        else:
            demo_preds.append(", ".join(e.gold_versions) if e.gold_versions else "NONE")
    demo_reports = score_predictions(examples, demo_preds)
    st.subheader("Scorer smoke test (gold echoed as predictions)")
    st.json(reports_to_flat_metrics(demo_reports))

sizes = [ModelSize(m) for m in model_choices] or [ModelSize.GEMMA2_2B]
precisions: list[Precision] = []
for size in sizes:
    for p in recommended_precisions(size, float(vram_gb)):
        if p not in precisions:
            precisions.append(p)

run_payload = load_latest_run(OUTPUTS_DIR)
st.subheader("Benchmark matrix")
if run_payload:
    st.caption(
        f"Loaded `outputs/` run `{run_payload.get('run_id')}` "
        f"({run_payload.get('created_at')})"
    )
    matrix = metrics_from_run(run_payload)
    meta = run_payload.get("meta") or {}
    if meta:
        st.json(meta)
else:
    st.info(
        "No `outputs/latest.json` yet — showing placeholders. "
        "Run `python scripts/run_benchmark.py` first."
    )
    matrix = empty_matrix(sizes, precisions)

df = pd.DataFrame([m.to_dict() for m in matrix])
# Expand task_scores into columns for the table.
if "task_scores" in df.columns:
    score_df = pd.json_normalize(df["task_scores"].tolist()).add_prefix("score_")
    df = pd.concat([df.drop(columns=["task_scores"]), score_df], axis=1)

st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Charts")
chart_df = df.copy()
chart_df["label"] = (
    chart_df["model_size"].astype(str) + "/" + chart_df["precision"].astype(str)
)
chart_df["peak_vram_mb"] = pd.to_numeric(chart_df["peak_vram_mb"], errors="coerce")
chart_df["tokens_per_sec"] = pd.to_numeric(chart_df["tokens_per_sec"], errors="coerce")
c1, c2 = st.columns(2)
with c1:
    st.caption("Peak VRAM (MB)")
    st.bar_chart(chart_df.set_index("label")["peak_vram_mb"])
with c2:
    st.caption("Throughput (tokens/sec)")
    st.bar_chart(chart_df.set_index("label")["tokens_per_sec"])

score_cols = [c for c in chart_df.columns if c.startswith("score_") and c.endswith(".accuracy")]
if score_cols:
    st.caption("Task accuracy")
    st.bar_chart(chart_df.set_index("label")[score_cols])
