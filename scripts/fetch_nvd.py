#!/usr/bin/env python3
"""Download NVD JSON 2.0 year feeds and write a normalized benchmark file.

Examples:
  python scripts/fetch_nvd.py --years 2025 --limit 500
  python scripts/fetch_nvd.py --years 2024 2025 --require-cwe

Storage (approx, from NVD feeds):
  - One year .gz: ~23 MB compressed, ~150-250 MB uncompressed
  - Recent 3 years: ~70 MB compressed, ~0.5-1 GB uncompressed
  - Normalized subset (500-2000 rows): a few MB
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    iter_normalized,
    load_nvd_json,
    save_normalized_jsonl,
)

NVD_YEAR_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}.json.gz"


def download_year(year: int, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_path = dest_dir / f"nvdcve-2.0-{year}.json.gz"
    json_path = dest_dir / f"nvdcve-2.0-{year}.json"
    url = NVD_YEAR_URL.format(year=year)

    if json_path.exists():
        print(f"[skip] {json_path} already exists")
        return json_path

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with gz_path.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    out.write(chunk)

    print(f"[decompress] {gz_path} -> {json_path}")
    with gzip.open(gz_path, "rb") as src, json_path.open("wb") as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            dst.write(block)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2025],
        help="NVD feed years to download (default: 2025)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Where to store raw NVD feeds (gitignored)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "cve_normalized.jsonl",
        help="Normalized JSONL output path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max normalized rows")
    parser.add_argument(
        "--require-cwe",
        action="store_true",
        help="Keep only records with at least one CWE label",
    )
    args = parser.parse_args()

    records = []
    for year in args.years:
        feed_path = download_year(year, args.raw_dir)
        raw = load_nvd_json(feed_path)
        for item in iter_normalized(raw):
            if args.require_cwe and not item.primary_cwe:
                continue
            records.append(item)
            if args.limit is not None and len(records) >= args.limit:
                break
        if args.limit is not None and len(records) >= args.limit:
            break

    n = save_normalized_jsonl(records, args.out)
    print(f"[done] wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
