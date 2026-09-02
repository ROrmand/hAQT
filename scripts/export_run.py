#!/usr/bin/env python3
"""Export a benchmark run as Markdown or LaTeX tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export_tables import format_latex_summary, format_markdown_summary  # noqa: E402
from src.results import load_latest_run, load_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Path to run JSON (default: outputs/latest.json)",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "latex", "both"),
        default="markdown",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Write export files here when --write is set",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write table_<run_id>.md / .tex under --out-dir",
    )
    args = parser.parse_args()

    if args.run is None:
        payload = load_latest_run(ROOT / "outputs")
        if payload is None:
            print("No run JSON found under outputs/", file=sys.stderr)
            return 1
    else:
        payload = load_run(args.run)

    run_id = payload.get("run_id", "unknown")
    md = format_markdown_summary(payload)
    tex = format_latex_summary(payload)

    if args.format in ("markdown", "both"):
        print(md)
    if args.format in ("latex", "both"):
        if args.format == "both":
            print("\n--- LaTeX ---\n")
        print(tex)

    if args.write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        md_path = args.out_dir / f"table_{run_id}.md"
        tex_path = args.out_dir / f"table_{run_id}.tex"
        md_path.write_text(md, encoding="utf-8")
        tex_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {md_path} and {tex_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
