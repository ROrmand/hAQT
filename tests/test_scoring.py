"""Tests for prediction parsing and scoring."""

from __future__ import annotations

import unittest

from src.scoring import (
    aggregate_cwe,
    aggregate_version,
    parse_cwe_prediction,
    parse_version_prediction,
    score_cwe_example,
    score_version_example,
)
from src.tasks import BenchmarkExample, TaskName


class TestParseCwe(unittest.TestCase):
    def test_plain_cwe(self) -> None:
        self.assertEqual(parse_cwe_prediction("CWE-79"), "CWE-79")

    def test_prose_before_cwe(self) -> None:
        self.assertEqual(
            parse_cwe_prediction("The weakness is CWE-89 due to SQL injection."),
            "CWE-89",
        )

    def test_unknown(self) -> None:
        self.assertIsNone(parse_cwe_prediction("UNKNOWN"))

    def test_empty(self) -> None:
        self.assertIsNone(parse_cwe_prediction(""))

    def test_no_cwe_token(self) -> None:
        self.assertIsNone(parse_cwe_prediction("cross-site scripting"))


class TestParseVersion(unittest.TestCase):
    def test_comma_list(self) -> None:
        self.assertEqual(parse_version_prediction("1.4.0, 3.2.1"), ("1.4.0", "3.2.1"))

    def test_none(self) -> None:
        self.assertEqual(parse_version_prediction("NONE"), ())

    def test_strips_v_prefix(self) -> None:
        self.assertEqual(parse_version_prediction("v2.0.1"), ("2.0.1",))


class TestScoreExamples(unittest.TestCase):
    def test_cwe_exact_match(self) -> None:
        ex = BenchmarkExample(
            task=TaskName.CWE_CLASSIFICATION,
            cve_id="CVE-TEST",
            prompt="x",
            gold_cwe="CWE-79",
            gold_cwes=("CWE-79", "CWE-80"),
        )
        scored = score_cwe_example(ex, "CWE-79")
        self.assertTrue(scored.correct)
        self.assertEqual(scored.detail["in_gold_set"], 1.0)

    def test_cwe_in_set_not_primary(self) -> None:
        ex = BenchmarkExample(
            task=TaskName.CWE_CLASSIFICATION,
            cve_id="CVE-TEST",
            prompt="x",
            gold_cwe="CWE-79",
            gold_cwes=("CWE-79", "CWE-80"),
        )
        scored = score_cwe_example(ex, "CWE-80")
        self.assertFalse(scored.correct)
        self.assertEqual(scored.detail["in_gold_set"], 1.0)

    def test_version_partial_f1(self) -> None:
        ex = BenchmarkExample(
            task=TaskName.VERSION_PARSING,
            cve_id="CVE-TEST",
            prompt="x",
            gold_versions=("1.0", "2.0"),
        )
        scored = score_version_example(ex, "1.0, 9.9")
        self.assertFalse(scored.correct)
        self.assertAlmostEqual(scored.detail["precision"], 0.5)
        self.assertAlmostEqual(scored.detail["recall"], 0.5)


class TestAggregate(unittest.TestCase):
    def test_cwe_accuracy(self) -> None:
        from src.scoring import ExampleScore

        scores = [
            ExampleScore("a", "cwe", True, "CWE-1", "CWE-1"),
            ExampleScore("b", "cwe", False, "CWE-2", "CWE-1"),
        ]
        report = aggregate_cwe(scores)
        self.assertEqual(report.n, 2)
        self.assertAlmostEqual(report.accuracy or 0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
