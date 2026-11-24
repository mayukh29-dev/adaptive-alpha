"""Sector discovery contract and ranking tests."""
import unittest

import sector_discovery


class SectorDiscoveryTests(unittest.TestCase):
    def test_curated_sector_coverage(self):
        self.assertEqual(len(sector_discovery.SECTOR_UNIVERSE), 11)
        for sector, tickers in sector_discovery.SECTOR_UNIVERSE.items():
            self.assertEqual(len(tickers), 6, sector)
            self.assertEqual(len(set(tickers)), len(tickers), sector)

    def test_custom_slider_changes_order(self):
        rows = [
            {"ticker":"AAA","momentum":90,"recovery":20,"date":"2026-11-24"},
            {"ticker":"BBB","momentum":30,"recovery":95,"date":"2026-11-24"},
        ]
        momentum = sector_discovery.rank_sector(rows,100,"Custom Hybrid")
        recovery = sector_discovery.rank_sector(rows,0,"Custom Hybrid")
        self.assertEqual(momentum.iloc[0]["Ticker"],"AAA")
        self.assertEqual(recovery.iloc[0]["Ticker"],"BBB")
        self.assertEqual(momentum.loc[momentum.Ticker=="AAA","Standard Hybrid"].iloc[0],
                         recovery.loc[recovery.Ticker=="AAA","Standard Hybrid"].iloc[0])

    def test_fixed_score_rank(self):
        rows = [
            {"ticker":"AAA","momentum":30,"recovery":90,"date":"2026-11-24"},
            {"ticker":"BBB","momentum":80,"recovery":40,"date":"2026-11-24"},
        ]
        ranked=sector_discovery.rank_sector(rows,60,"Momentum")
        self.assertEqual(ranked.iloc[0]["Ticker"],"BBB")

    def test_all_sector_tables_and_leaders(self):
        groups = {
            "Information Technology": [
                {"ticker":"AAA","momentum":90,"recovery":20,"date":"2026-11-24"},
                {"ticker":"BBB","momentum":30,"recovery":95,"date":"2026-11-24"},
            ],
            "Financials": [
                {"ticker":"CCC","momentum":60,"recovery":75,"date":"2026-11-24"},
            ],
        }
        momentum=sector_discovery.all_sector_tables(groups,100,"Custom Hybrid")
        recovery=sector_discovery.all_sector_tables(groups,0,"Custom Hybrid")
        self.assertEqual(len(momentum),11)
        self.assertEqual(momentum["Information Technology"].iloc[0]["Ticker"],"AAA")
        self.assertEqual(recovery["Information Technology"].iloc[0]["Ticker"],"BBB")
        leaders=sector_discovery.sector_leaders(momentum,"Custom Hybrid")
        self.assertEqual(len(leaders),11)
        self.assertEqual(leaders.loc[leaders.Sector=="Financials","Leader"].iloc[0],"CCC")
        self.assertEqual(leaders.loc[leaders.Sector=="Energy","Leader"].iloc[0],"Unavailable")

    def test_empty_and_invalid_options(self):
        self.assertTrue(sector_discovery.rank_sector([],60,"Momentum").empty)
        with self.assertRaises(ValueError):
            sector_discovery.rank_sector([],101,"Momentum")
        with self.assertRaises(ValueError):
            sector_discovery.rank_sector([],60,"Forecast")


if __name__ == "__main__":
    unittest.main()
