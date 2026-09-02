"""Tests for per-example scores and analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis import bootstrap_all_cells, error_items, quantization_tax
from src.results import load_scores, save_scores
from src.tasks import TaskName


def _sample_scores() -> dict:
    return {
        "run_number": 1,
        "run_id": "001",
        "example_index": [
            {"i": 0, "cve_id": "CVE-A", "task": "cwe_classification", "severity": "HIGH", "published_year": 2024, "gold_cwe": "CWE-79"},
            {"i": 1, "cve_id": "CVE-A", "task": "version_parsing", "severity": "HIGH", "published_year": 2024, "gold_cwe": "CWE-79"},
        ],
        "cells": [
            {
                "model_size": "2b",
                "precision": "fp16",
                "items": [
                    {"i": 0, "cve_id": "CVE-A", "task": "cwe_classification", "correct": True, "detail": {"exact_match": 1.0}, "prediction_text": "CWE-79"},
                    {"i": 1, "cve_id": "CVE-A", "task": "version_parsing", "correct": True, "detail": {"f1": 1.0}, "prediction_text": "1.0"},
                ],
            },
            {
                "model_size": "2b",
                "precision": "int4_nf4",
                "items": [
                    {"i": 0, "cve_id": "CVE-A", "task": "cwe_classification", "correct": False, "detail": {"exact_match": 0.0}, "prediction_text": "CWE-89"},
                    {"i": 1, "cve_id": "CVE-A", "task": "version_parsing", "correct": True, "detail": {"f1": 0.8}, "prediction_text": "1.0"},
                ],
            },
        ],
    }


class TestScoresPersistence(unittest.TestCase):
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_scores(
                out,
                run_number=1,
                run_id="001",
                created_at="2026-01-01T00:00:00+00:00",
                example_index=[],
                cells=[],
            )
            loaded = load_scores(out, "001")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["run_id"], "001")


class TestAnalysis(unittest.TestCase):
    def test_bootstrap_rows(self) -> None:
        rows = bootstrap_all_cells(_sample_scores(), n_resamples=100)
        self.assertGreaterEqual(len(rows), 2)

    def test_quantization_tax(self) -> None:
        tax = quantization_tax(_sample_scores(), n_resamples=100)
        self.assertTrue(any(r["task"] == TaskName.CWE_CLASSIFICATION.value for r in tax))
        cwe = next(r for r in tax if r["task"] == TaskName.CWE_CLASSIFICATION.value)
        self.assertAlmostEqual(cwe["delta_mean"], -1.0)

    def test_error_items(self) -> None:
        errs = error_items(_sample_scores(), precision="int4_nf4", errors_only=True)
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["task"], "cwe_classification")


if __name__ == "__main__":
    unittest.main()
