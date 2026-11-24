"""Historical score trend tests: shared calculations and no lookahead."""
import unittest

import numpy as np
import pandas as pd

import score_history
import signals


class ScoreHistoryTests(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range("2024-01-02", periods=400)
        price = 100 + np.linspace(0, 60, len(idx)) + 4*np.sin(np.arange(len(idx))/17)
        self.history = pd.DataFrame({"Close": price}, index=idx)

    def test_history_matches_shared_indicator(self):
        trend = score_history.score_history(self.history, 60, 6)
        mom, rec, hybrid = signals.indicators(
            pd.DataFrame({"Stock":self.history["Close"]}))
        self.assertFalse(trend.empty)
        self.assertAlmostEqual(trend.iloc[-1]["Momentum"], mom["Stock"].iloc[-1])
        self.assertAlmostEqual(trend.iloc[-1]["Recovery"], rec["Stock"].iloc[-1])
        self.assertAlmostEqual(trend.iloc[-1]["Standard Hybrid"], hybrid["Stock"].iloc[-1])
        self.assertAlmostEqual(trend.iloc[-1]["Custom Hybrid"], hybrid["Stock"].iloc[-1])

    def test_weight_changes_only_custom_hybrid(self):
        a = score_history.score_history(self.history, 10, 3)
        b = score_history.score_history(self.history, 90, 3)
        for name in ("Momentum", "Recovery", "Standard Hybrid"):
            np.testing.assert_allclose(a[name], b[name])
        self.assertGreater(float(abs(a["Custom Hybrid"]-b["Custom Hybrid"]).max()), 1e-3)

    def test_no_future_lookahead(self):
        all_scores = score_history.score_history(self.history, 65, 6)
        earlier = score_history.score_history(self.history.iloc[:-1], 65, 6)
        matching = all_scores.loc[earlier.index]
        for name in earlier:
            np.testing.assert_allclose(matching[name], earlier[name], rtol=0, atol=1e-10)

    def test_missing_or_short_history(self):
        self.assertTrue(score_history.score_history(pd.DataFrame(), 60, 3).empty)
        self.assertTrue(score_history.score_history(self.history.iloc[:75], 60, 3).empty)

    def test_bad_settings(self):
        for pct in (-1, 101, 32.5):
            with self.assertRaises(ValueError):
                score_history.score_history(self.history, pct, 3)
        with self.assertRaises(ValueError):
            score_history.score_history(self.history, 60, 2)


if __name__ == "__main__":
    unittest.main()
