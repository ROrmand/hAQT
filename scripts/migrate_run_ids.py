#!/usr/bin/env python3
"""Migrate timestamp run ids (run_20260829T045959Z) to sequential run_001 …"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.results import migrate_legacy_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for line in migrate_legacy_runs(args.output_dir, dry_run=args.dry_run):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
