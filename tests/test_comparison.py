"""Deterministic tests for the public comparison page."""
import unittest

import comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"ticker": "AAA", "momentum": 90, "recovery": 20, "date": "2026-11-23"},
            {"ticker": "BBB", "momentum": 40, "recovery": 80, "date": "2026-11-23"},
            {"ticker": "CCC", "momentum": 60, "recovery": 60, "date": "2026-11-23"},
        ]

    def test_parse_tickers(self):
        self.assertEqual(comparison.parse_comparison_tickers("nvda, amd, nvda, msft"),
                         ["NVDA", "AMD", "MSFT"])
        self.assertEqual(comparison.parse_comparison_tickers("brk.b, brk-b"),
                         ["BRK.B", "BRK-B"])

    def test_invalid_tickers(self):
        for raw in ("", "NVDA", "A,B,C,D", "A,A", "NVDA, @@@"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                comparison.parse_comparison_tickers(raw)

    def test_weight_changes_rank_but_not_original_scores(self):
        trend = comparison.build_comparison(self.rows, 100)
        recovery = comparison.build_comparison(self.rows, 0)
        self.assertEqual(trend.iloc[0]["Ticker"], "AAA")
        self.assertEqual(recovery.iloc[0]["Ticker"], "BBB")
        self.assertEqual(trend.loc[trend.Ticker == "AAA", "Momentum"].iloc[0], 90)
        self.assertEqual(recovery.loc[recovery.Ticker == "AAA", "Momentum"].iloc[0], 90)
        self.assertEqual(trend.loc[trend.Ticker == "AAA", "Standard Hybrid (60/40)"].iloc[0], 62)
        self.assertEqual(recovery.loc[recovery.Ticker == "AAA", "Standard Hybrid (60/40)"].iloc[0], 62)

    def test_weight_validation(self):
        for weight in (-1, 101, 50.5):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                comparison.build_comparison(self.rows, weight)


if __name__ == "__main__":
    unittest.main()
