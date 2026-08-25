# hAQT
Hardware-Aware Quantization & Security Benchmarker

End-to-end GPU profiling workbench for open-weight LLMs (Gemma-2 **2B** and **9B**) across **FP16 / INT8 / INT4_NF4**, measured on NVD/CVE security workloads.

## Cloud readiness (local-first)

Pull onto a Google Cloud GPU VM only after:

1. **Local pipeline runs end-to-end** on the RTX 5070 Ti (data load → multi-precision profile on a small batch → Streamlit charts) with no crashes/OOM on the intended precision matrix.
2. **Clean repo**: `requirements.txt`, `.gitignore` (no raw feeds / weights), and a runnable `demo_run.ipynb`.
3. **Ready to record**: cloud time is for final metrics, screenshots, and a short demo — not debugging.

## Layout

```
hAQT/
├── data/cve_sample.json          # tiny checked-in sample
├── scripts/
│   ├── fetch_nvd.py              # download + normalize NVD year feeds
│   └── run_benchmark.py          # GPU matrix → outputs/
├── src/
│   ├── data_loader.py            # NVD parse + CWE/version label derivation
│   ├── tasks.py                  # CWE + version prompt builders
│   ├── scoring.py                # Parse generations + accuracy/F1 metrics
│   ├── quantizer.py              # BitsAndBytes load configs (2B/9B)
│   ├── profiler.py               # pynvml + timing hooks
│   └── results.py                # save/load outputs/ artifacts
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

# Optional: pull a year feed and normalize (gitignored raw files)
python scripts/fetch_nvd.py --years 2025 --limit 500 --require-cwe

# Dry-run (no GPU): writes placeholder rows under outputs/
python scripts/run_benchmark.py --dry-run --models 2b

# First real cell on 16GB: Gemma-2 2B INT4 on a tiny batch
python scripts/run_benchmark.py --limit 4 --models 2b --precisions int4_nf4

# Full local matrix for 2B (FP16/INT8/INT4)
python scripts/run_benchmark.py --limit 8 --models 2b --vram-gb 16

streamlit run app.py
```

### NVD storage (approximate)

| Scope | Compressed | Uncompressed |
|-------|------------|--------------|
| 1 year feed | ~23 MB | ~150–250 MB |
| 3 recent years | ~70 MB | ~0.5–1 GB |
| Normalized 500–2k rows | — | a few MB |

## Models & precision policy

| Model | 16GB local (5070 Ti) | Large cloud GPU (L4/A100) |
|-------|----------------------|---------------------------|
| `google/gemma-2-2b-it` | FP16, INT8, INT4_NF4 | same |
| `google/gemma-2-9b-it` | INT8, INT4_NF4 (FP16 optional/gated) | FP16 + quantized |

Quantization stack for v1: **BitsAndBytes** (INT8 + NF4). AutoAWQ is deferred.

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

- `outputs/run_<UTC>.json` / `.csv`
- `outputs/latest.json` / `latest.csv` (what Streamlit loads)

## Status

Data → tasks → scoring → GPU runner → `outputs/` → Streamlit are wired. Next: prove a local 2B INT4 smoke run on the RTX 5070 Ti, then expand the matrix and record cloud demos.
