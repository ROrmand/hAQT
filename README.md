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
├── scripts/fetch_nvd.py          # download + normalize NVD year feeds
├── src/
│   ├── data_loader.py            # NVD parse + CWE/version label derivation
│   ├── quantizer.py              # BitsAndBytes load configs (2B/9B)
│   └── profiler.py               # pynvml + timing hooks
├── app.py                        # Streamlit dashboard (skeleton)
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

These become gold targets for classification / version-parsing accuracy later.

## Status

Skeleton increment: interfaces, sample data, fetch script, and dashboard shell. Next: task prompts + scoring, then real GPU benchmark runs writing to `outputs/`.
