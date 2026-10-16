import unittest

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from src.features import (
    FEATURE_COLS,
    adjust_price_scale,
    build_panel,
    latest_market_row,
    next_trading_day,
    price_features,
    seasonal_profile,
)


class PanelTests(unittest.TestCase):
    def test_alias_weekend_sector_and_macro(self):
        days = pd.bdate_range("2024-01-03", periods=45)
        dates = [day.strftime("%Y-%m-%d") for day in days]
        frames = []
        for ticker, start in (("SBER", 100.0), ("GAZP", 200.0)):
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "trade_date": dates,
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": np.linspace(start, start + 40, len(dates)),
                        "volume": np.full(len(dates), 1_000.0),
                        "value": np.full(len(dates), 1_000_000.0),
                    }
                )
            )
        prices = pd.concat(frames, ignore_index=True)
        mapped = next_trading_day(dates, "2024-02-17")
        self.assertIsNotNone(mapped)
        news = pd.DataFrame(
            [
                {
                    "channel": "rian_ru",
                    "msg_id": 1,
                    "published_at": "2024-02-17T15:00:00+03:00",
                    "text": "Сбербанк",
                    "sentiment": 0.5,
                    "relevant": 1,
                    "is_macro": 0,
                },
                {
                    "channel": "mash",
                    "msg_id": 2,
                    "published_at": f"{dates[30]}T10:00:00+03:00",
                    "text": "нефть",
                    "sentiment": -0.4,
                    "relevant": 1,
                    "is_macro": 0,
                },
                {
                    "channel": "rian_ru",
                    "msg_id": 3,
                    "published_at": f"{dates[30]}T11:00:00+03:00",
                    "text": "ЦБ",
                    "sentiment": 0.2,
                    "relevant": 1,
                    "is_macro": 1,
                },
            ]
        )
        links = pd.DataFrame(
            [
                {"channel": "rian_ru", "msg_id": 1, "ticker": "SBER", "match_type": "alias", "weight": 1.0, "match_term": "сбербанк"},
                {"channel": "mash", "msg_id": 2, "ticker": "GAZP", "match_type": "sector", "weight": 0.5, "match_term": "нефт"},
            ]
        )
        panel = build_panel(prices, news, links)
        weekend = panel.loc[(panel["ticker"] == "SBER") & (panel["trade_date"] == mapped)].iloc[0]
        self.assertEqual(weekend["news_count_alias"], 1)
        self.assertAlmostEqual(weekend["sent_mean_alias"], 0.5)
        oil_day = dates[30]
        gazp = panel.loc[(panel["ticker"] == "GAZP") & (panel["trade_date"] == oil_day)].iloc[0]
        self.assertAlmostEqual(gazp["news_count_sector"], 0.5)
        self.assertAlmostEqual(gazp["sent_sum_sector"], -0.2)
        self.assertAlmostEqual(gazp["sent_mean_macro"], 0.2)
        self.assertEqual(gazp["news_count_macro"], 1)
        sber = panel.loc[(panel["ticker"] == "SBER") & (panel["trade_date"] == oil_day)].iloc[0]
        self.assertEqual(sber["news_count_alias"], 0)
        self.assertAlmostEqual(sber["sent_mean_macro"], 0.2)
        self.assertTrue(panel["target"].notna().all())
        closes = prices.loc[prices["ticker"] == "SBER"].set_index("trade_date")["close"]
        expected = float(closes.shift(-1).loc[mapped] / closes.loc[mapped] - 1)
        self.assertAlmostEqual(float(weekend["target"]), expected)

    def test_holdout_fit(self):
        try:
            from src.train import _fit
        except ImportError:
            self.skipTest("lightgbm или scikit-learn не установлены")
        days = pd.bdate_range("2023-10-12", "2026-10-12")
        dates = [day.strftime("%Y-%m-%d") for day in days]
        frames = []
        for ticker, start in (("SBER", 100.0), ("GAZP", 200.0)):
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "trade_date": dates,
                        "close": np.linspace(start, start + len(dates), len(dates)),
                        "volume": np.linspace(1_000, 2_000, len(dates)),
                    }
                )
            )
        news = pd.DataFrame(columns=["channel", "msg_id", "published_at", "text", "sentiment", "relevant", "is_macro"])
        links = pd.DataFrame(columns=["channel", "msg_id", "ticker", "match_type", "weight", "match_term"])
        panel = build_panel(pd.concat(frames, ignore_index=True), news, links)
        result = _fit(panel, "2025-10-12")
        train_n = float((panel["trade_date"] < "2025-10-12").sum())
        test_n = float((panel["trade_date"] >= "2025-10-12").sum())
        self.assertEqual(result["metrics"]["n_train"], train_n)
        self.assertEqual(result["metrics"]["n_test"], test_n)
        self.assertGreater(train_n, 100)
        self.assertGreater(test_n, 100)
        actual = panel.loc[panel["trade_date"] >= "2025-10-12", "target"].to_numpy()
        self.assertAlmostEqual(result["metrics"]["mae_zero"], float(mean_absolute_error(actual, np.zeros_like(actual))))
        self.assertTrue(np.isfinite(result["metrics"]["mae"]))
        self.assertTrue(np.isfinite(result["metrics"]["dir_acc"]))
        self.assertTrue(np.isfinite(result["metrics"]["dir_yesterday"]))
        self.assertIn("ret_1", result["importance"])
        sample = panel[FEATURE_COLS + ["ticker"]].iloc[[0]].copy()
        sample["ticker"] = pd.Categorical(sample["ticker"], categories=result["ticker_categories"])
        predicted = result["model"].predict(sample)
        self.assertEqual(len(predicted), 1)
        self.assertTrue(np.isfinite(predicted[0]))

    def test_seasonal_profile_uses_daily_returns(self):
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = pd.DataFrame(
            {
                "ticker": "SBER",
                "trade_date": [day.strftime("%Y-%m-%d") for day in dates],
                "close": np.linspace(100, 130, len(dates)),
            }
        )
        profile = seasonal_profile(prices, ["SBER"])
        self.assertFalse(profile.empty)
        self.assertEqual(set(profile["ticker"]), {"SBER"})
        manual = prices.copy()
        manual["ret"] = manual["close"].pct_change()
        manual["month"] = pd.to_datetime(manual["trade_date"]).dt.month
        expected = manual.dropna(subset=["ret"]).groupby("month")["ret"].mean()
        for row in profile.itertuples(index=False):
            self.assertAlmostEqual(row.mean_return, float(expected.loc[row.month]))
        self.assertTrue((profile["n"] > 0).all())

    def test_share_consolidation_does_not_become_a_return(self):
        prices = pd.DataFrame(
            {
                "ticker": ["VTBR", "VTBR", "VTBR"],
                "trade_date": ["2024-07-12", "2024-07-15", "2024-07-16"],
                "close": [0.01992, 92.95, 95.32],
                "volume": [0.0, 100.0, 80.0],
            }
        )
        adjusted = adjust_price_scale(prices)
        self.assertAlmostEqual(float(adjusted["close"].iloc[0]), 92.95)
        returns = price_features(adjusted)
        self.assertLess(float(returns["ret_1"].abs().max()), 0.05)

    def test_forecast_row_needs_twenty_one_closes(self):
        short = _price_frame("RAGR", 20)
        self.assertIsNone(latest_market_row(short, "RAGR"))
        ready = latest_market_row(_price_frame("YDEX", 21), "YDEX")
        self.assertIsNotNone(ready)
        self.assertTrue(np.isfinite(ready["ret_20"]))
        self.assertIn("month_sin", ready)


def _price_frame(ticker: str, n: int) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": [day.strftime("%Y-%m-%d") for day in days],
            "close": np.linspace(100, 110, n),
            "volume": np.full(n, 1_000.0),
        }
    )


if __name__ == "__main__":
    unittest.main()
