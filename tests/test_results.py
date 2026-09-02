"""Tests for sequential run numbering and manifest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.profiler import ProfileMetrics
from src.results import (
    allocate_run_number,
    format_run_id,
    list_runs,
    load_run,
    migrate_legacy_runs,
    parse_run_number_from_stem,
    save_run,
)


class TestRunNumbering(unittest.TestCase):
    def test_format_run_id(self) -> None:
        self.assertEqual(format_run_id(1), "001")
        self.assertEqual(format_run_id(42), "042")

    def test_parse_run_number(self) -> None:
        self.assertEqual(parse_run_number_from_stem("run_007"), 7)
        self.assertIsNone(parse_run_number_from_stem("run_20260829T045959Z"))

    def test_sequential_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            m = ProfileMetrics("2b", "fp16", None, None, None, None)
            p1 = save_run([m], out, meta={"label": "smoke"})
            p2 = save_run([m], out)
            self.assertEqual(p1.name, "run_001.json")
            self.assertEqual(p2.name, "run_002.json")
            payload = load_run(p2)
            self.assertEqual(payload["run_number"], 2)
            self.assertEqual(payload["run_id"], "002")
            runs = list_runs(out)
            self.assertEqual(runs[0].name, "run_002.json")

    def test_migrate_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            legacy = {
                "run_id": "20260829T034103Z",
                "created_at": "2026-08-29T03:41:03+00:00",
                "meta": {"label": "smoke"},
                "metrics": [],
            }
            path = out / "run_20260829T034103Z.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            log = migrate_legacy_runs(out)
            self.assertTrue(any("Migrated" in line for line in log))
            self.assertTrue((out / "run_001.json").is_file())
            migrated = load_run(out / "run_001.json")
            self.assertEqual(migrated["run_number"], 1)
            self.assertEqual(migrated["meta"]["legacy_id"], "20260829T034103Z")


if __name__ == "__main__":
    unittest.main()
