"""Bootstrap CI helper tests."""

from __future__ import annotations

import unittest

from src.stats import bootstrap_ci


class TestBootstrapCi(unittest.TestCase):
    def test_constant_values(self) -> None:
        mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_empty(self) -> None:
        mean, lo, hi = bootstrap_ci([])
        self.assertEqual((mean, lo, hi), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
