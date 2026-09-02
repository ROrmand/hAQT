# hAQT
Hardware-Aware Quantization & Security Benchmarker

**Contest deadline: September 10, 2026**

End-to-end GPU profiling workbench for contest-allowed open-weight LLMs — **Gemma-2** and **Nemotron-Mini-4B** for local runs on the 5070 Ti, plus **Gemma-2 9B** (and larger Nemotron tiers) on cloud GPUs — across **FP16 / INT8 / INT4_NF4**, measured on NVD/CVE security workloads.

## Roadmap tier status

**Maintain this section.** When a tier (or individual item) is finished, update the status table and add a short **Done** note under that tier — what shipped, where artifacts live, and any follow-ups. Do not delete completed items; mark them complete so the history stays visible.

| Tier | Status | Notes |
|------|--------|-------|
| **Tier 1** — Finish the story | In progress | Golden run done; **Path C** — per-CVE scores + bootstrap CIs (1.4) |
| **Tier 2** — Expand the benchmark | Not started | |
| **Tier 3** — Hardware & quantization | Not started | Cloud 9B after scored golden re-run |
| **Tier 4** — Product & engineering | In progress | Error analysis tab; scored runs pending |
| **Tier 5** — Research / narrative | In progress | Quantization tax report (5.1) from `--save-scores` |

---

## Roadmap

### Tier 1 — Finish the story (highest leverage)

Make the project credible before widening scope.

| # | Item | Status |
|---|------|--------|
| 1.1 | **Golden run + reproducibility pack** — fixed dataset slice (e.g. 500 CWE-labeled rows), documented `outputs/run_*.json` from **Gemma-2 2B** and **Nemotron-Mini-4B** (FP16/INT8/INT4 on the 5070 Ti), one-page VRAM vs speed vs accuracy table with cross-family comparison | Done |
| 1.5 | **Wire Nemotron-Mini-4B** — add `nvidia/Nemotron-Mini-4B-Instruct` to `quantizer.py`, CLI (`--models`), and Streamlit; validate chat template + BitsAndBytes on 16GB | Done |
| 1.2 | **True TTFT + batching** — streaming first-token timing in `profiler.py`; optional batch-size sweeps (1 / 4 / 8 prompts) | Not started |
| 1.3 | **Gold-label quality pass** — human-reviewed subset (50–100 CVEs); report label precision/recall and stratified slices (clean vs messy version mentions) | Not started |
| 1.4 | **Statistical rigor** — bootstrap CIs, breakdowns by severity/year/CWE family, quantization delta (Δaccuracy / ΔF1 vs FP16) | In progress |

**Done** *(add notes here as items ship)*

- **1.1** — Golden run **`run_015`** (`outputs/run_015.json` + `scores_015.json`, label `golden`): 500 CVEs / 921 prompts, ~62 min on 5070 Ti. Supersedes `run_013`. **INT4 VRAM savings:** Gemma 53%, Nemotron 42%. **Quantization tax:** CWE flat (bootstrap CIs overlap); Version F1 delta ≤0.04 vs FP16.
- **1.4** *(partial)* — `--save-scores` → `scores_<run_id>.json`; `analyze_run.py` bootstrap CIs + stratified slices; quantization deltas vs FP16 in golden run.
- **1.5** — Nemotron-Mini-4B wired; 5070 Ti smoke **`run_012`** (label `smoke`).

---

### Tier 2 — Expand the benchmark (security domain)

| # | Item | Status |
|---|------|--------|
| 2.1 | **More security tasks** — severity/CVSS bucket, affected product/vendor from CPE, exploitability triage, patch priority, ATT&CK mapping | Not started |
| 2.2 | **Richer CVE context in prompts** — CPE blocks, reference URLs, CWE descriptions from MITRE | Not started |
| 2.3 | **Adversarial / robustness slice** — format tricks (markdown, JSON, chit-chat); score format compliance separately from task accuracy | Not started |
| 2.4 | **Cross-dataset validation** — compare NVD against curated MITRE sets or synthetic CVE-like text | Not started |

**Done** *(add notes here as items ship)*

---

### Tier 3 — Expand hardware & quantization

| # | Item | Status |
|---|------|--------|
| 3.1 | **More contest-allowed models** — Nemotron 3 Nano/Lightning on cloud GPUs; Cosmos Reasoner only if the rubric needs multimodal; keep local matrix on Gemma-2 2B + Nemotron-Mini-4B | Not started |
| 3.2 | **More quantization backends** — GPTQ/AWQ, GGUF/llama.cpp, TensorRT-LLM / vLLM | Not started |
| 3.3 | **Multi-GPU / cloud SKU matrix** — L4, A10, A100, H100; cost-per-1k-CVE metric | Not started |
| 3.4 | **Power draw** — joules-per-CVE via `pynvml` where supported | Not started |

**Done** *(add notes here as items ship)*

---

### Tier 4 — Product & engineering maturity

| # | Item | Status |
|---|------|--------|
| 4.1 | **Unit tests** — `data_loader`, `scoring`, parser edge cases; dry-run integration test for CI | Done |
| 4.2 | **Interactive Streamlit runner** — pick run history, trigger small benchmarks, per-CVE error drill-down | In progress |
| 4.3 | **Run comparison mode** — side-by-side runs (e.g. Gemma-2 2B vs Nemotron-Mini-4B at same precision; or local int4 vs cloud 9B fp16) | Done |
| 4.4 | **Export for talks/papers** — chart PNG export, LaTeX/Markdown tables, repro card YAML | In progress |

**Done** *(add notes here as items ship)*

- **4.1** — `tests/test_scoring.py`, `tests/test_stats.py`, `tests/test_integration.py` (dry-run). Run: `python -m unittest discover -s tests`.
- **4.3** — Streamlit **Cross-model** tab + **Compare & export** tab (run history picker, two-run diff, quantization delta table).
- **4.2** *(partial)* — Streamlit **Error analysis** tab: bootstrap CIs, quantization tax table, per-CVE error filters (requires `scores_<run_id>.json`).
- **4.4** *(partial)* — `scripts/export_run.py` + `outputs/table_<run_id>.md/.tex`; download buttons in Streamlit.

---

### Tier 5 — Research / narrative angles

| # | Item | Status |
|---|------|--------|
| 5.1 | **Quantization tax on security reasoning** — does INT4 hurt version parsing more than CWE classification? | In progress |
| 5.2 | **Pareto-optimal precision per task** — lowest precision within X% of FP16 accuracy | Not started |
| 5.3 | **Model size vs quantization interaction** — Gemma-2 2B INT4 vs Nemotron-Mini-4B INT4 at the same VRAM budget; cloud follow-up with 9B / Nemotron 3 | Not started |
| 5.4 | **Temporal drift** — eval split by `published` year; generalization across evolving CVE language | Not started |

**Done** *(add notes here as items ship)*

- **5.1** — `outputs/quantization_tax_015.md`: INT4 quantization-neutral on CWE (Δ=0); Version F1 tax ≤0.04 for Gemma, −0.036 for Nemotron Mini at INT4.

---

### Schedule to 9/10/2026

| When | Milestone |
|------|-----------|
| **Now → 9/1** | ~~Re-run golden with `--save-scores`~~ ✓ `run_015` + `quantization_tax_015.md` |
| **9/1 → 9/5** | Cloud Gemma 9B (`--label cloud-9b --save-scores`); compare vs local INT4 in Streamlit |
| **9/5 → 9/8** | Demo polish: pin `run_012`/`run_014`, Streamlit screenshots, prune old runs |
| **9/8 → 9/10** | Final export tables + submission buffer |

---

### Suggested order

```
✓ Wire Nemotron-Mini-4B + smoke run (Gemma 2B + Nemotron Mini FP16/INT8/INT4) on 5070 Ti
✓ Fixed 500-CVE slice + golden outputs (run_013)
✓ Streamlit compare/export + unit tests
  → Re-run golden with --save-scores (run_014) + quantization tax report   [Path C — NOW]
  → Cloud Gemma 9B (3.1) by 9/5
  → TTFT streaming (1.2) if time after cloud
```

| Quick (days) | Medium (1–2 weeks) | Bigger (multi-week) |
|--------------|--------------------|---------------------|
| ~~Nemotron-Mini-4B in `quantizer.py`~~ ✓ | Human-reviewed label subset | Nemotron 3 on cloud (vLLM) |
| ~~Golden `outputs/` for Gemma + Nemotron Mini~~ ✓ | True TTFT streaming | Full NVD-year benchmark |
| ~~Unit tests for scoring/parser~~ ✓ | True TTFT streaming | Nemotron 3 on cloud (vLLM) |
| ~~Streamlit compare + export tables~~ ✓ | Gemma-2 9B cloud matrix | Paper-style ablation study |

---

## Cloud readiness (local-first)

Pull onto a Google Cloud GPU VM only after:

1. **Local pipeline runs end-to-end** on the RTX 5070 Ti for **Gemma-2 2B** and **Nemotron-Mini-4B** (data load → multi-precision profile on a small batch → Streamlit charts) with no crashes/OOM on the intended precision matrix. ✓ Smoke matrix passed (`run_012`).
2. **Clean repo**: `requirements.txt`, `.gitignore` (no raw feeds / weights), and a runnable `demo_run.ipynb`.
3. **Ready to record**: cloud time is for final metrics, screenshots, and a short demo — not debugging.

## Layout

```
hAQT/
├── data/cve_sample.json          # tiny checked-in sample
├── scripts/
│   ├── fetch_nvd.py              # download + normalize NVD year feeds
│   ├── run_benchmark.py          # GPU matrix → outputs/
│   ├── export_run.py             # Markdown/LaTeX tables from run JSON
│   ├── analyze_run.py            # quantization deltas (+ bootstrap if scores JSON)
│   └── migrate_run_ids.py        # one-time UTC id → run_001 migration
├── src/
│   ├── data_loader.py            # NVD parse + CWE/version label derivation
│   ├── tasks.py                  # CWE + version prompt builders
│   ├── scoring.py                # Parse generations + accuracy/F1 metrics
│   ├── export_tables.py          # report table formatting
│   ├── analysis.py               # bootstrap CIs, stratified breakdowns, tax report
│   ├── cve_context.py            # severity/year context for stratification
│   ├── stats.py                  # bootstrap CI helper
│   ├── quantizer.py              # BitsAndBytes load configs (Gemma 2B/9B, Nemotron Mini 4B)
│   ├── profiler.py               # NVML + timing hooks
│   └── results.py                # save/load outputs/ artifacts
├── tests/                        # unittest: scoring, stats, dry-run integration
├── app.py                        # Streamlit dashboard
├── demo_run.ipynb                # Colab / Deep Learning VM notebook
└── requirements.txt
```

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Smoke-test data path
python -c "from src.data_loader import load_benchmark_dataset as L; print(L('data/cve_sample.json'))"

# Golden dataset: 500 CWE-labeled rows (2024+2025 feeds, gitignored raw files)
python scripts/fetch_nvd.py --years 2024 2025 --limit 500 --require-cwe

# Dry-run (no GPU): writes placeholder rows under outputs/
python scripts/run_benchmark.py --dry-run --models 2b

# First real cell on 16GB: Gemma-2 2B INT4 on a tiny batch
python scripts/run_benchmark.py --limit 4 --models 2b --precisions int4_nf4

# Same smoke for Nemotron-Mini-4B
python scripts/run_benchmark.py --limit 4 --models nemotron-mini-4b --precisions int4_nf4

# Full local matrix: both contest local models (FP16/INT8/INT4)
python scripts/run_benchmark.py --limit 8 --models 2b,nemotron-mini-4b --vram-gb 16

# Golden run with per-example scores (~64 min; enables bootstrap CIs + error analysis)
python scripts/run_benchmark.py --data data/cve_normalized.jsonl --limit 500 \\
  --models 2b,nemotron-mini-4b --vram-gb 16 --label golden --save-scores

# Analysis: deltas + bootstrap CIs + quantization tax report
python scripts/analyze_run.py --report
python scripts/analyze_run.py --stratify severity

# Tests
python -m unittest discover -s tests

streamlit run app.py
```

### NVD storage (approximate)

| Scope | Compressed | Uncompressed |
|-------|------------|--------------|
| 1 year feed | ~23 MB | ~150–250 MB |
| 3 recent years | ~70 MB | ~0.5–1 GB |
| Normalized 500–2k rows | — | a few MB |

## Models & precision policy

Contest-allowed families: **Gemma**, **Nemotron**, **Cosmos**. Local development targets the two small instruct models below; cloud runs add Gemma 9B and larger Nemotron tiers.

| Model | HF ID | Role | 16GB local (5070 Ti) | Large cloud GPU (L4/A100) |
|-------|-------|------|----------------------|---------------------------|
| Gemma-2 2B | `google/gemma-2-2b-it` | Local baseline | FP16, INT8, INT4_NF4 | same |
| Nemotron-Mini-4B | `nvidia/Nemotron-Mini-4B-Instruct` | Local comparison (contest Nemotron) | FP16, INT8, INT4_NF4 | same |
| Gemma-2 9B | `google/gemma-2-9b-it` | Cloud scale-up | INT8, INT4_NF4 (FP16 optional/gated) | FP16 + quantized |
| Nemotron 3 Nano/Lightning | e.g. `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Cloud scale-up (MoE; vLLM likely) | Not planned for 16GB | BF16 / NVFP4 per NVIDIA docs |

**Cosmos** (physical-AI omni-model) is out of scope for the CVE text benchmark unless the contest rubric requires it — see Tier 3.1.

Quantization stack for v1: **BitsAndBytes** (INT8 + NF4) on Gemma + Nemotron-Mini-4B. AutoAWQ and vLLM backends are deferred (see Tier 3).

## Derived labels

From each NVD record we extract:

- **primary_cwe** / `cwe_ids` from `weaknesses`
- **version_mentions** from description patterns and CPE URIs

These become gold targets for classification / version-parsing accuracy.

## Tasks & scoring

- **CWE classification** — model must reply `CWE-###` (or `UNKNOWN`); scored with exact-match accuracy (+ soft “in gold CWE set”).
- **Version parsing** — model replies with comma-separated versions (or `NONE`); scored with set precision / recall / F1.

```bash
python -c "from src.data_loader import load_benchmark_dataset; from src.tasks import build_all_examples; from src.scoring import score_predictions, reports_to_flat_metrics; r=load_benchmark_dataset('data/cve_sample.json'); e=build_all_examples(r); p=[(x.gold_cwe or 'UNKNOWN') if x.task.value=='cwe_classification' else (', '.join(x.gold_versions) or 'NONE') for x in e]; print(reports_to_flat_metrics(score_predictions(e,p)))"
```

## Benchmark outputs

`scripts/run_benchmark.py` walks the recommended precision matrix, profiles each cell, scores CWE/version generations, and writes:

- `outputs/run_001.json` / `.csv` — sequential run number (auto-incremented)
- `outputs/latest.json` / `latest.csv` — copy of the most recent run (what Streamlit loads by default)
- `outputs/run_manifest.json` — index of all runs (`run_number`, `label`, `legacy_id`, timestamps)

- `outputs/scores_<run_id>.json` — per-example predictions + scores (with `--save-scores`)
- `outputs/quantization_tax_<run_id>.md` — Tier 5.1 report (`analyze_run.py --report`)

Tag important runs: `--label golden`, `--label smoke`, `--label cloud-9b`. Always pass **`--save-scores`** on final benchmark runs.

### Keeping runs (recommended)

| What | Policy |
|------|--------|
| **Git** | Keep `outputs/` **gitignored** (already). Artifacts are local/regenerable except long GPU runs. |
| **Pin forever** | Label with `--label golden` / `smoke` / `cloud-9b`. Copy `run_NNN.json` + `table_NNN.md` into a `pinned/` folder or attach to releases if you need permanence. |
| **Day-to-day** | Keep `latest.json` + manifest. Prune unlabeled smoke/dry-run rows (`run_001`–`011` etc.) once you're confident — keep **`run_012`** (smoke) and **`run_013`** (golden) as anchors. |
| **Logs** | `outputs/golden_run.log` can be deleted after a successful run; metrics live in JSON. |
| **Migrate old UTC ids** | `python scripts/migrate_run_ids.py` (already run once; safe to re-run — no-op if nothing legacy). |

**Practical minimum for demo/contest:** `run_012`, `run_014` (golden + scores), `scores_014.json`, `quantization_tax_014.md`, `latest.json`, `run_manifest.json`.

## Current status

**Deadline: 9/10/2026.** Scored golden complete (`run_015` + `scores_015.json` + `quantization_tax_015.md`). **Next:** cloud Gemma 9B (`--label cloud-9b --save-scores`) or Streamlit demo screenshots.
