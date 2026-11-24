"""Watchlist behavior remains deterministic and session-local."""
import unittest

import watchlist


class WatchlistTests(unittest.TestCase):
    def test_add_dedupe_case_and_remove(self):
        saved = watchlist.add_ticker([], " nvda ")
        self.assertEqual(saved, ["NVDA"])
        self.assertEqual(watchlist.add_ticker(saved, "NVDA"), ["NVDA"])
        saved = watchlist.add_ticker(saved, "brk.b")
        self.assertEqual(saved, ["NVDA", "BRK.B"])
        self.assertEqual(watchlist.remove_ticker(saved, "NvDa"), ["BRK.B"])

    def test_rejects_bad_tickers(self):
        for ticker in ("", "@@@", "TOO_LONG_TICKER", "A B", "A,B"):
            with self.subTest(ticker=ticker), self.assertRaises(ValueError):
                watchlist.add_ticker([], ticker)

    def test_limit(self):
        saved = [f"T{i}" for i in range(12)]
        with self.assertRaises(ValueError):
            watchlist.add_ticker(saved, "NVDA")
        self.assertEqual(watchlist.add_ticker(saved, "T2"), saved)

    def test_clean_and_no_input_mutation(self):
        original = ["nvda", "NVDA", " msft "]
        cleaned = watchlist.clean_watchlist(original)
        self.assertEqual(cleaned, ["NVDA", "MSFT"])
        self.assertEqual(original, ["nvda", "NVDA", " msft "])


if __name__ == "__main__":
    unittest.main()
