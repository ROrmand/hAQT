"""Dry-run integration test for the benchmark pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestBenchmarkDryRun(unittest.TestCase):
    def test_dry_run_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_benchmark.py"),
                    "--dry-run",
                    "--limit",
                    "2",
                    "--models",
                    "2b",
                    "--output-dir",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            latest = out / "latest.json"
            self.assertTrue(latest.is_file())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertIn("metrics", payload)
            self.assertGreaterEqual(len(payload["metrics"]), 1)
            self.assertTrue(payload["meta"]["dry_run"])


if __name__ == "__main__":
    unittest.main()
