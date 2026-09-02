"""hAQT Streamlit workbench — model-centric VRAM / speed / accuracy dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.analysis import (
    bootstrap_all_cells,
    error_items,
    format_quantization_tax_report,
    quantization_tax,
)
from src.data_loader import load_benchmark_dataset
from src.export_tables import (
    format_latex_summary,
    format_markdown_summary,
    metrics_to_dataframe,
    quantization_deltas,
)
from src.profiler import empty_matrix
from src.quantizer import MODEL_IDS, ModelSize, Precision, recommended_precisions
from src.results import (
    list_runs,
    load_latest_run,
    load_run,
    metrics_from_run,
    run_display_name,
    scores_for_run_payload,
)
from src.scoring import reports_to_flat_metrics, score_predictions
from src.tasks import build_all_examples

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "cve_sample.json"
NORMALIZED_PATH = ROOT / "data" / "cve_normalized.jsonl"
OUTPUTS_DIR = ROOT / "outputs"

PRECISION_ORDER = ["fp16", "int8", "int4_nf4"]
PRECISION_LABELS = {
    "fp16": "FP16",
    "int8": "INT8",
    "int4_nf4": "INT4 NF4",
}
PRECISION_COLORS = {
    "fp16": "#2563eb",
    "int8": "#059669",
    "int4_nf4": "#d97706",
}

MODEL_META: dict[str, dict[str, str]] = {
    ModelSize.GEMMA2_2B.value: {
        "name": "Gemma-2 2B",
        "family": "Gemma",
        "hf_id": MODEL_IDS[ModelSize.GEMMA2_2B],
        "role": "Local baseline",
    },
    ModelSize.GEMMA2_9B.value: {
        "name": "Gemma-2 9B",
        "family": "Gemma",
        "hf_id": MODEL_IDS[ModelSize.GEMMA2_9B],
        "role": "Cloud scale-up",
    },
    ModelSize.NEMOTRON_MINI_4B.value: {
        "name": "Nemotron-Mini-4B",
        "family": "Nemotron",
        "hf_id": MODEL_IDS[ModelSize.NEMOTRON_MINI_4B],
        "role": "Local comparison",
    },
}
MODEL_COLORS = {
    ModelSize.GEMMA2_2B.value: "#4f46e5",
    ModelSize.GEMMA2_9B.value: "#7c3aed",
    ModelSize.NEMOTRON_MINI_4B.value: "#0d9488",
}

SCORE_COLUMNS = {
    "score_cwe_classification.accuracy": "CWE accuracy",
    "score_version_parsing.f1": "Version F1",
    "score_version_parsing.accuracy": "Version accuracy",
}

st.set_page_config(page_title="hAQT Benchmarker", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_display(key: str) -> str:
    return MODEL_META.get(key, {}).get("name", key)


def _model_hf_id(key: str) -> str:
    if key in MODEL_META:
        return MODEL_META[key]["hf_id"]
    try:
        return MODEL_IDS[ModelSize(key)]
    except (KeyError, ValueError):
        return key


def _prepare_chart_df(matrix_rows) -> pd.DataFrame:
    df = pd.DataFrame([m.to_dict() for m in matrix_rows])
    if df.empty:
        return df
    if "task_scores" in df.columns:
        score_df = pd.json_normalize(df["task_scores"].tolist()).add_prefix("score_")
        df = pd.concat([df.drop(columns=["task_scores"]), score_df], axis=1)
    df["model_label"] = df["model_size"].map(lambda k: _model_display(str(k)))
    df["profile"] = df["precision"].astype(str)
    df["profile_label"] = df["profile"].map(
        lambda p: PRECISION_LABELS.get(p, str(p).upper())
    )
    df["label"] = df["model_label"] + " · " + df["profile_label"]
    for col in ("peak_vram_mb", "tokens_per_sec", "ttft_ms", "total_latency_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["profile"] = pd.Categorical(df["profile"], categories=PRECISION_ORDER, ordered=True)
    return df.sort_values(["model_size", "profile"], kind="stable")


def _show_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> None:
    if frame.empty:
        st.info("No rows to display.")
        return
    view = frame if max_rows is None else frame.head(max_rows)
    safe = view.copy()
    for col in safe.columns:
        if str(safe[col].dtype) == "category":
            safe[col] = safe[col].astype(str)
    html = safe.to_html(index=False, border=0, classes="haqt-table", na_rep="")
    st.markdown(
        """
<style>
table.haqt-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
table.haqt-table th, table.haqt-table td {
  border-bottom: 1px solid rgba(49, 51, 63, 0.2);
  padding: 0.35rem 0.5rem;
  text-align: left;
  vertical-align: top;
}
table.haqt-table th { font-weight: 600; }
.haqt-model-card {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
}
.haqt-model-card h4 { margin: 0 0 0.25rem 0; font-size: 1rem; }
.haqt-model-card p { margin: 0; font-size: 0.82rem; opacity: 0.85; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown(html, unsafe_allow_html=True)


def _has_numeric(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _precision_bar(
    frame: pd.DataFrame,
    y: str,
    title: str,
    y_label: str,
    *,
    height: int = 340,
) -> go.Figure:
    fig = px.bar(
        frame,
        x="profile_label",
        y=y,
        color="profile",
        color_discrete_map=PRECISION_COLORS,
        category_orders={"profile_label": [PRECISION_LABELS[p] for p in PRECISION_ORDER]},
        title=title,
        labels={"profile_label": "Precision profile", y: y_label, "profile": "Profile"},
        text_auto=".1f",
    )
    fig.update_layout(
        legend_title_text="Profile",
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        showlegend=False,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def _model_header(model_key: str, vram_gb: float) -> None:
    meta = MODEL_META.get(model_key, {})
    name = meta.get("name", model_key)
    family = meta.get("family", "")
    role = meta.get("role", "")
    hf_id = meta.get("hf_id", _model_hf_id(model_key))
    try:
        rec = recommended_precisions(ModelSize(model_key), float(vram_gb))
        prec_hint = ", ".join(PRECISION_LABELS.get(p.value, p.value) for p in rec)
    except ValueError:
        prec_hint = "—"
    color = MODEL_COLORS.get(model_key, "#64748b")
    st.markdown(
        f'<div class="haqt-model-card" style="border-left: 4px solid {color};">'
        f"<h4>{name}</h4>"
        f"<p><code>{hf_id}</code> · {family} · {role}</p>"
        f"<p>Recommended at {vram_gb:.0f} GB VRAM: {prec_hint}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _model_kpis(model_df: pd.DataFrame) -> None:
    if model_df.empty:
        st.info("No benchmark rows for this model yet.")
        return
    cols = st.columns(len(model_df))
    for col, (_, row) in zip(cols, model_df.iterrows()):
        profile = PRECISION_LABELS.get(str(row["profile"]), str(row["profile"]))
        vram = row.get("peak_vram_mb")
        tps = row.get("tokens_per_sec")
        cwe = row.get("score_cwe_classification.accuracy")
        with col:
            st.markdown(f"**{profile}**")
            if pd.notna(vram):
                st.metric("Peak VRAM", f"{vram:.0f} MB")
            else:
                st.metric("Peak VRAM", "—")
            if pd.notna(tps):
                st.metric("Throughput", f"{tps:.1f} tok/s")
            if pd.notna(cwe):
                st.metric("CWE accuracy", f"{cwe:.0%}")


def _render_model_profiles(model_df: pd.DataFrame, model_key: str) -> None:
    """Precision-profile comparison for a single model."""
    if model_df.empty:
        st.warning("Run the benchmark for this model to populate profile charts.")
        return

    has_hw = _has_numeric(model_df, "peak_vram_mb") or _has_numeric(
        model_df, "tokens_per_sec"
    )
    if not has_hw:
        st.warning("No hardware metrics yet — run `scripts/run_benchmark.py` for this model.")
        return

    _model_kpis(model_df)

    g1, g2 = st.columns(2)
    with g1:
        if _has_numeric(model_df, "peak_vram_mb"):
            st.plotly_chart(
                _precision_bar(
                    model_df,
                    "peak_vram_mb",
                    "Peak VRAM by precision profile",
                    "Peak VRAM (MB)",
                ),
                use_container_width=True,
            )
    with g2:
        if _has_numeric(model_df, "tokens_per_sec"):
            st.plotly_chart(
                _precision_bar(
                    model_df,
                    "tokens_per_sec",
                    "Throughput by precision profile",
                    "Tokens / sec",
                ),
                use_container_width=True,
            )

    g3, g4 = st.columns(2)
    with g3:
        if _has_numeric(model_df, "ttft_ms"):
            st.plotly_chart(
                _precision_bar(
                    model_df,
                    "ttft_ms",
                    "Time-to-first-token by profile",
                    "TTFT (ms)",
                ),
                use_container_width=True,
            )
    with g4:
        if _has_numeric(model_df, "total_latency_ms"):
            st.plotly_chart(
                _precision_bar(
                    model_df,
                    "total_latency_ms",
                    "Total generation latency by profile",
                    "Latency (ms)",
                ),
                use_container_width=True,
            )

    present = [(c, label) for c, label in SCORE_COLUMNS.items() if c in model_df.columns]
    score_rows = []
    for _, row in model_df.iterrows():
        for col, label in present:
            val = row.get(col)
            if pd.isna(val):
                continue
            score_rows.append(
                {
                    "profile_label": row["profile_label"],
                    "profile": str(row["profile"]),
                    "metric": label,
                    "value": float(val),
                }
            )
    if score_rows:
        score_df = pd.DataFrame(score_rows)
        fig_scores = px.bar(
            score_df,
            x="metric",
            y="value",
            color="profile",
            barmode="group",
            color_discrete_map=PRECISION_COLORS,
            title="Task quality by precision profile",
            labels={"value": "Score", "metric": "Metric", "profile": "Profile"},
            range_y=[0, 1.05],
            text_auto=".2f",
        )
        fig_scores.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40))
        fig_scores.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_scores, use_container_width=True)

    # Multi-metric radar: one trace per precision profile.
    radar_metrics = [
        ("peak_vram_mb", "VRAM (inv)", True),
        ("tokens_per_sec", "Throughput", False),
        ("score_cwe_classification.accuracy", "CWE acc.", False),
        ("score_version_parsing.f1", "Version F1", False),
    ]
    usable = [
        (col, label, invert)
        for col, label, invert in radar_metrics
        if _has_numeric(model_df, col)
    ]
    if len(usable) >= 3:
        categories = [label for _, label, _ in usable]
        fig_radar = go.Figure()
        for _, row in model_df.iterrows():
            vals = []
            for col, _, invert in usable:
                v = float(row[col])
                if invert:
                    v = 1.0 / max(v, 1.0)
                vals.append(v)
            max_v = max(vals) or 1.0
            norm = [v / max_v for v in vals]
            profile = str(row["profile"])
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=norm + [norm[0]],
                    theta=categories + [categories[0]],
                    name=PRECISION_LABELS.get(profile, profile),
                    line_color=PRECISION_COLORS.get(profile, "#94a3b8"),
                    fill="toself",
                    opacity=0.55,
                )
            )
        fig_radar.update_layout(
            title="Normalized profile shape (higher = better per axis)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend_title_text="Profile",
        )
        st.plotly_chart(fig_radar, use_container_width=True)


def _render_cross_model(df: pd.DataFrame, precision: str) -> None:
    """Compare models at a single precision profile."""
    subset = df[df["profile"].astype(str) == precision].copy()
    if subset.empty:
        st.info(f"No rows at **{PRECISION_LABELS.get(precision, precision)}** yet.")
        return

    st.caption(
        f"Head-to-head at **{PRECISION_LABELS.get(precision, precision)}** "
        "— same precision, different model families."
    )

    g1, g2 = st.columns(2)
    with g1:
        if _has_numeric(subset, "peak_vram_mb"):
            fig = px.bar(
                subset,
                x="model_label",
                y="peak_vram_mb",
                color="model_size",
                color_discrete_map=MODEL_COLORS,
                title="Peak VRAM",
                labels={"model_label": "Model", "peak_vram_mb": "MB"},
                text_auto=".0f",
            )
            fig.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig, use_container_width=True)
    with g2:
        if _has_numeric(subset, "tokens_per_sec"):
            fig = px.bar(
                subset,
                x="model_label",
                y="tokens_per_sec",
                color="model_size",
                color_discrete_map=MODEL_COLORS,
                title="Throughput",
                labels={"model_label": "Model", "tokens_per_sec": "Tokens / sec"},
                text_auto=".1f",
            )
            fig.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig, use_container_width=True)

    present = [(c, label) for c, label in SCORE_COLUMNS.items() if c in subset.columns]
    long_rows = []
    for _, row in subset.iterrows():
        for col, label in present:
            val = row.get(col)
            if pd.isna(val):
                continue
            long_rows.append(
                {
                    "model_label": row["model_label"],
                    "metric": label,
                    "value": float(val),
                }
            )
    if long_rows:
        fig_scores = px.bar(
            pd.DataFrame(long_rows),
            x="metric",
            y="value",
            color="model_label",
            barmode="group",
            title="Task quality",
            labels={"value": "Score", "metric": "Metric", "model_label": "Model"},
            range_y=[0, 1.05],
            text_auto=".2f",
        )
        fig_scores.update_layout(height=360)
        st.plotly_chart(fig_scores, use_container_width=True)


def _render_overview(df: pd.DataFrame, run_payload: dict | None) -> None:
    if run_payload:
        meta = run_payload.get("meta") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Run", run_display_name(run_payload))
        c2.metric("CVE limit", meta.get("limit", "—"))
        c3.metric("Examples scored", meta.get("n_examples", "—"))
        c4.metric("VRAM budget", f"{meta.get('vram_gb', '—')} GB")
        if meta:
            with st.expander("Run metadata", expanded=False):
                st.json(meta)
    else:
        st.info(
            "No `outputs/latest.json` yet — showing placeholders. "
            "Run `python scripts/run_benchmark.py` first."
        )

    if df.empty:
        return

    display_cols = [
        c
        for c in [
            "model_label",
            "profile_label",
            "peak_vram_mb",
            "tokens_per_sec",
            "ttft_ms",
            "total_latency_ms",
            "score_cwe_classification.accuracy",
            "score_version_parsing.f1",
        ]
        if c in df.columns
    ]
    st.markdown("**Full benchmark matrix**")
    _show_table(df[display_cols])

    scatter_df = df.dropna(subset=["peak_vram_mb", "tokens_per_sec"]).copy()
    if not scatter_df.empty:
        acc_col = "score_cwe_classification.accuracy"
        if acc_col in scatter_df.columns:
            scatter_df["bubble"] = (
                pd.to_numeric(scatter_df[acc_col], errors="coerce").fillna(0.2) * 40 + 8
            )
        else:
            scatter_df["bubble"] = 18
        fig_eff = px.scatter(
            scatter_df,
            x="peak_vram_mb",
            y="tokens_per_sec",
            color="model_label",
            symbol="profile_label",
            size="bubble",
            color_discrete_map={_model_display(k): v for k, v in MODEL_COLORS.items()},
            title="Efficiency frontier — VRAM vs throughput",
            labels={
                "peak_vram_mb": "Peak VRAM (MB)",
                "tokens_per_sec": "Tokens / sec",
                "model_label": "Model",
                "profile_label": "Profile",
            },
            hover_data={"label": True},
        )
        fig_eff.update_layout(height=420, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_eff, use_container_width=True)

    # Compact strip: models on x-axis, metrics faceted, grouped by precision.
    if _has_numeric(df, "peak_vram_mb"):
        summary = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Peak VRAM (MB)", "Tokens / sec", "CWE accuracy"),
        )
        for profile, color in PRECISION_COLORS.items():
            subset = df[df["profile"].astype(str) == profile]
            if subset.empty:
                continue
            x = subset["model_label"]
            summary.add_trace(
                go.Bar(
                    name=PRECISION_LABELS.get(profile, profile),
                    x=x,
                    y=subset["peak_vram_mb"],
                    marker_color=color,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            summary.add_trace(
                go.Bar(
                    name=PRECISION_LABELS.get(profile, profile),
                    x=x,
                    y=subset["tokens_per_sec"],
                    marker_color=color,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            acc_col = "score_cwe_classification.accuracy"
            if acc_col in subset.columns:
                summary.add_trace(
                    go.Bar(
                        name=PRECISION_LABELS.get(profile, profile),
                        x=x,
                        y=subset[acc_col],
                        marker_color=color,
                        showlegend=False,
                    ),
                    row=1,
                    col=3,
                )
        summary.update_layout(
            barmode="group",
            height=360,
            legend_title_text="Precision profile",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(summary, use_container_width=True)


def _render_export_tab(
    run_payload: dict | None,
    *,
    all_runs: list[Path],
) -> None:
    """Quantization deltas, run comparison, and Markdown/LaTeX export."""
    if not run_payload:
        st.info("Load a benchmark run to export tables and compare precision profiles.")
        return

    export_df = metrics_to_dataframe(run_payload)
    delta_df = quantization_deltas(export_df)

    st.markdown("### Quantization delta vs FP16")
    if delta_df.empty:
        st.caption("No FP16 baseline rows in this run.")
    else:
        display = delta_df[
            [
                "model_label",
                "precision_label",
                "delta_vram_mb",
                "vram_reduction_pct",
                "delta_tok_s",
                "delta_cwe_acc",
                "delta_version_f1",
            ]
        ].copy()
        display.columns = [
            "Model",
            "Precision",
            "d VRAM (MB)",
            "VRAM saved (%)",
            "d tok/s",
            "d CWE acc.",
            "d Version F1",
        ]
        _show_table(display)

    st.markdown("### Export for README / slides")
    md_text = format_markdown_summary(run_payload)
    tex_text = format_latex_summary(run_payload)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Markdown",
            md_text,
            file_name=f"table_{run_payload.get('run_id', 'run')}.md",
            mime="text/markdown",
        )
        st.text_area("Markdown preview", md_text, height=280)
    with c2:
        st.download_button(
            "Download LaTeX",
            tex_text,
            file_name=f"table_{run_payload.get('run_id', 'run')}.tex",
            mime="text/plain",
        )
        st.text_area("LaTeX preview", tex_text, height=280)

    if len(all_runs) >= 2:
        st.markdown("### Compare two runs")
        st.caption("Side-by-side version F1 at INT4 (or pick profiles below).")
        run_names = [p.name for p in all_runs]
        c1, c2, c3 = st.columns(3)
        with c1:
            run_a_name = st.selectbox("Run A", run_names, index=0, key="cmp_run_a")
        with c2:
            run_b_name = st.selectbox(
                "Run B", run_names, index=min(1, len(run_names) - 1), key="cmp_run_b"
            )
        with c3:
            cmp_prec = st.selectbox(
                "Precision",
                PRECISION_ORDER,
                index=PRECISION_ORDER.index("int4_nf4"),
                format_func=lambda p: PRECISION_LABELS.get(p, p),
                key="cmp_prec",
            )
        if run_a_name and run_b_name:
            df_a = metrics_to_dataframe(load_run(all_runs[run_names.index(run_a_name)]))
            df_b = metrics_to_dataframe(load_run(all_runs[run_names.index(run_b_name)]))
            sub_a = df_a[df_a["precision"].astype(str) == cmp_prec]
            sub_b = df_b[df_b["precision"].astype(str) == cmp_prec]
            rows = []
            for model in sorted(
                set(sub_a["model_size"].tolist()) | set(sub_b["model_size"].tolist())
            ):
                ra = sub_a[sub_a["model_size"] == model]
                rb = sub_b[sub_b["model_size"] == model]
                rows.append(
                    {
                        "Model": MODEL_META.get(model, {}).get("name", model),
                        "Run A VRAM (MB)": ra["peak_vram_mb"].iloc[0]
                        if not ra.empty
                        else None,
                        "Run B VRAM (MB)": rb["peak_vram_mb"].iloc[0]
                        if not rb.empty
                        else None,
                        "Run A Version F1": ra["version_f1"].iloc[0]
                        if not ra.empty
                        else None,
                        "Run B Version F1": rb["version_f1"].iloc[0]
                        if not rb.empty
                        else None,
                    }
                )
            if rows:
                _show_table(pd.DataFrame(rows))


def _render_errors_tab(run_payload: dict | None, output_dir: Path) -> None:
    """Per-CVE error drill-down when scores_<run_id>.json exists."""
    if not run_payload:
        st.info("Load a benchmark run first.")
        return

    scores = scores_for_run_payload(output_dir, run_payload)
    if not scores:
        run_id = run_payload.get("run_id", "?")
        st.warning(
            f"No `scores_{run_id}.json` for this run. "
            "Re-run with `--save-scores` to enable error analysis and bootstrap CIs."
        )
        return

    st.markdown("### Quantization tax (bootstrap vs FP16)")
    tax_df = pd.DataFrame(quantization_tax(scores))
    if not tax_df.empty:
        show = tax_df[
            ["model_size", "task", "precision", "baseline_mean", "mean", "delta_mean"]
        ].copy()
        show.columns = ["Model", "Task", "Precision", "FP16", "Quant", "Delta"]
        _show_table(show)

    st.markdown("### Bootstrap 95% CIs")
    boot_df = pd.DataFrame(bootstrap_all_cells(scores))
    if not boot_df.empty:
        _show_table(
            boot_df[
                ["model_size", "precision", "task", "mean", "ci_lo", "ci_hi", "n"]
            ]
        )

    report = format_quantization_tax_report(run_payload, scores)
    st.download_button(
        "Download quantization tax report",
        report,
        file_name=f"quantization_tax_{run_payload.get('run_id', 'run')}.md",
        mime="text/markdown",
    )

    st.markdown("### Per-CVE errors")
    models = sorted({c["model_size"] for c in scores.get("cells") or []})
    precisions = sorted({c["precision"] for c in scores.get("cells") or []})
    tasks = ["cwe_classification", "version_parsing"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        filt_model = st.selectbox("Model", ["(all)"] + models, key="err_model")
    with c2:
        filt_prec = st.selectbox("Precision", ["(all)"] + precisions, key="err_prec")
    with c3:
        filt_task = st.selectbox("Task", ["(all)"] + tasks, key="err_task")
    with c4:
        errors_only = st.checkbox("Errors only", value=True, key="err_only")

    rows = error_items(
        scores,
        model_size=None if filt_model == "(all)" else filt_model,
        precision=None if filt_prec == "(all)" else filt_prec,
        task=None if filt_task == "(all)" else filt_task,
        errors_only=errors_only,
    )
    st.caption(f"{len(rows)} rows")
    if rows:
        _show_table(pd.DataFrame(rows), max_rows=50)
    else:
        st.success("No rows match the current filters.")


# ---------------------------------------------------------------------------
# Sidebar & data load
# ---------------------------------------------------------------------------

default_data = str(NORMALIZED_PATH if NORMALIZED_PATH.is_file() else SAMPLE_PATH)

with st.sidebar:
    st.header("Run config")
    available_runs = list_runs(OUTPUTS_DIR)

    def _run_label(filename: str) -> str:
        if filename == "latest":
            return "latest (most recent)"
        path = OUTPUTS_DIR / filename
        if path.is_file():
            return f"{filename} — {run_display_name(load_run(path))}"
        return filename

    run_options = ["latest"] + [p.name for p in available_runs]
    selected_run = st.selectbox(
        "Benchmark run",
        run_options,
        index=0,
        format_func=_run_label,
    )
    model_choices = st.multiselect(
        "Models",
        options=[m.value for m in ModelSize],
        format_func=_model_display,
        default=[ModelSize.GEMMA2_2B.value, ModelSize.NEMOTRON_MINI_4B.value],
    )
    vram_gb = st.slider("Assumed GPU VRAM (GB)", 8, 80, 16)
    data_path = st.text_input("CVE dataset path", default_data)
    st.divider()
    st.markdown("**Model catalog**")
    for key, meta in MODEL_META.items():
        st.caption(f"**{meta['name']}** — `{meta['hf_id']}`")
    st.divider()
    st.markdown(
        "Populate metrics:\n"
        "```\npython scripts/run_benchmark.py "
        "--data data/cve_normalized.jsonl \\\n"
        "  --limit 32 --models 2b,nemotron-mini-4b \\\n"
        "  --vram-gb 16\n```"
    )

if selected_run == "latest":
    run_payload = load_latest_run(OUTPUTS_DIR)
else:
    run_payload = load_run(OUTPUTS_DIR / selected_run)

records = []
load_error = None
try:
    records = load_benchmark_dataset(data_path, limit=50)
except Exception as exc:
    load_error = str(exc)

sizes = [ModelSize(m) for m in model_choices] or [ModelSize.GEMMA2_2B]
precisions: list[Precision] = []
for size in sizes:
    for p in recommended_precisions(size, float(vram_gb)):
        if p not in precisions:
            precisions.append(p)

if run_payload:
    matrix = metrics_from_run(run_payload)
else:
    matrix = empty_matrix(sizes, precisions)

df = _prepare_chart_df(matrix)
if model_choices:
    df = df[df["model_size"].isin(model_choices)].copy()

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("hAQT")
st.caption("Hardware-Aware Quantization & Security Benchmarker")

k1, k2, k3 = st.columns(3)
k1.metric("Loaded CVEs", len(records))
k2.metric("With CWE labels", sum(1 for r in records if r.primary_cwe))
k3.metric("Matrix cells", len(df))

if load_error:
    st.error(f"Failed to load dataset: {load_error}")

models_in_data = list(dict.fromkeys(df["model_size"].astype(str).tolist()))
if not models_in_data:
    models_in_data = model_choices or [ModelSize.GEMMA2_2B.value]

tab_overview, tab_profiles, tab_cross, tab_export, tab_errors, tab_data = st.tabs(
    ["Overview", "Model profiles", "Cross-model", "Compare & export", "Error analysis", "Dataset"]
)

with tab_overview:
    _render_overview(df, run_payload)

with tab_profiles:
    st.markdown(
        "Compare **FP16 / INT8 / INT4 NF4** within each model — "
        "the quantization trade-off for that checkpoint."
    )
    profile_tabs = st.tabs([_model_display(m) for m in models_in_data])
    for model_key, tab in zip(models_in_data, profile_tabs):
        with tab:
            _model_header(model_key, vram_gb)
            model_df = df[df["model_size"] == model_key].copy()
            _render_model_profiles(model_df, model_key)

with tab_cross:
    st.markdown(
        "Hold precision fixed and compare **model families** at the same profile."
    )
    available_prec = [
        p
        for p in PRECISION_ORDER
        if p in df["profile"].astype(str).values
    ] or PRECISION_ORDER
    chosen_prec = st.radio(
        "Precision profile",
        options=available_prec,
        format_func=lambda p: PRECISION_LABELS.get(p, p),
        horizontal=True,
    )
    _render_cross_model(df, chosen_prec)

with tab_export:
    _render_export_tab(run_payload, all_runs=available_runs)

with tab_errors:
    _render_errors_tab(run_payload, OUTPUTS_DIR)

with tab_data:
    if load_error:
        st.error(f"Dataset unavailable: {load_error}")
    elif records:
        st.markdown("**Normalized CVE records**")
        _show_table(pd.DataFrame([r.to_dict() for r in records]), max_rows=15)
        examples = build_all_examples(records)
        st.caption(f"{len(examples)} prompts (CWE classification + version parsing)")
        _show_table(
            pd.DataFrame(
                [
                    {
                        "task": e.task.value,
                        "cve_id": e.cve_id,
                        "gold_cwe": e.gold_cwe,
                        "gold_versions": ", ".join(e.gold_versions),
                        "prompt_preview": e.prompt[:160]
                        + ("..." if len(e.prompt) > 160 else ""),
                    }
                    for e in examples
                ]
            ),
            max_rows=15,
        )
        demo_preds = []
        for e in examples:
            if e.task.value == "cwe_classification":
                demo_preds.append(e.gold_cwe or "UNKNOWN")
            else:
                demo_preds.append(
                    ", ".join(e.gold_versions) if e.gold_versions else "NONE"
                )
        st.caption("Scorer smoke test (gold echoed as predictions)")
        st.json(reports_to_flat_metrics(score_predictions(examples, demo_preds)))
    else:
        st.info("No CVE records loaded.")
