"""Панель «бумага × день» и сезонный профиль."""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from src.link import Hit, classify
from src.universe import Universe

FEATURE_COLS = [
    "sent_mean_alias",
    "sent_sum_alias",
    "news_count_alias",
    "sent_mean_sector",
    "sent_sum_sector",
    "news_count_sector",
    "sent_mean_macro",
    "news_count_macro",
    "ret_1",
    "ret_5",
    "ret_20",
    "volume_z",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]
ZERO_NEWS = [
    "sent_mean_alias",
    "sent_sum_alias",
    "news_count_alias",
    "sent_mean_sector",
    "sent_sum_sector",
    "news_count_sector",
    "sent_mean_macro",
    "news_count_macro",
]


def next_trading_day(calendar: list[str], day: str) -> str | None:
    index = bisect.bisect_left(calendar, day)
    if index >= len(calendar):
        return None
    return calendar[index]


def prepare_links(conn, universe: Universe) -> tuple[int, int]:
    """Пересобирает привязки. Тональность уже посчитанных постов не стирает."""
    news = pd.read_sql_query("SELECT channel, msg_id, text FROM news", conn)
    if news.empty:
        raise SystemExit("В базе нет новостей. Сначала выполните: python -m src.collect_news")
    conn.execute("DELETE FROM news_links")
    flags: list[tuple] = []
    links: list[tuple] = []
    relevant_n = 0
    for row in news.itertuples(index=False):
        hits, is_macro = classify(row.text, universe)
        relevant = bool(hits) or is_macro
        relevant_n += int(relevant)
        flags.append((int(relevant), int(is_macro), row.channel, int(row.msg_id)))
        for hit in hits:
            links.append((row.channel, int(row.msg_id), hit.ticker, hit.match_type, hit.weight, hit.term))
    conn.executemany(
        "UPDATE news SET relevant = ?, is_macro = ? WHERE channel = ? AND msg_id = ?",
        flags,
    )
    conn.executemany(
        """
        INSERT INTO news_links (channel, msg_id, ticker, match_type, weight, match_term)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        links,
    )
    conn.commit()
    return relevant_n, len(links)


def _map_frame(frame: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    mapped = frame.copy()
    days = mapped["published_at"].astype(str).str.slice(0, 10)
    mapped["trade_date"] = [next_trading_day(calendar, day) for day in days]
    return mapped.dropna(subset=["trade_date"])


def _alias_agg(links: pd.DataFrame) -> pd.DataFrame:
    if links.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "news_count_alias", "sent_sum_alias", "sent_mean_alias"])
    frame = links.loc[links["match_type"] == "alias"].copy()
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "news_count_alias", "sent_sum_alias", "sent_mean_alias"])
    frame["sentiment"] = frame["sentiment"].fillna(0.0)
    grouped = frame.groupby(["ticker", "trade_date"], as_index=False).agg(
        news_count_alias=("msg_id", "count"),
        sent_sum_alias=("sentiment", "sum"),
        sent_mean_alias=("sentiment", "mean"),
    )
    return grouped


def _sector_agg(links: pd.DataFrame) -> pd.DataFrame:
    columns = ["ticker", "trade_date", "news_count_sector", "sent_sum_sector", "sent_mean_sector"]
    if links.empty:
        return pd.DataFrame(columns=columns)
    frame = links.loc[links["match_type"] == "sector"].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["sentiment"] = frame["sentiment"].fillna(0.0)
    frame["weighted"] = frame["sentiment"] * frame["weight"]
    grouped = frame.groupby(["ticker", "trade_date"], as_index=False).agg(
        news_count_sector=("weight", "sum"),
        sent_sum_sector=("weighted", "sum"),
        weight_sum=("weight", "sum"),
    )
    grouped["sent_mean_sector"] = grouped["sent_sum_sector"] / grouped["weight_sum"]
    return grouped.drop(columns=["weight_sum"])


def _macro_agg(news: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    columns = ["trade_date", "news_count_macro", "sent_mean_macro"]
    if news.empty or "is_macro" not in news.columns:
        return pd.DataFrame(columns=columns)
    frame = news.loc[news["is_macro"] == 1].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _map_frame(frame, calendar)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["sentiment"] = frame["sentiment"].fillna(0.0)
    return frame.groupby("trade_date", as_index=False).agg(
        news_count_macro=("msg_id", "count"),
        sent_mean_macro=("sentiment", "mean"),
    )


def _volume_z(series: pd.Series) -> pd.Series:
    mean = series.rolling(20, min_periods=20).mean()
    std = series.rolling(20, min_periods=20).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0.0)


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values(["ticker", "trade_date"]).copy()
    frame = frame.loc[frame["close"].astype(float) > 0].copy()
    frame["ret_1"] = frame.groupby("ticker")["close"].transform(lambda series: series.pct_change())
    frame["ret_5"] = frame.groupby("ticker")["close"].transform(lambda series: series.pct_change(5))
    frame["ret_20"] = frame.groupby("ticker")["close"].transform(lambda series: series.pct_change(20))
    frame["target"] = frame.groupby("ticker")["close"].transform(lambda series: series.shift(-1)) / frame["close"] - 1
    frame["volume_z"] = frame.groupby("ticker")["volume"].transform(_volume_z)
    dates = pd.to_datetime(frame["trade_date"])
    frame["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    frame["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    return frame


def market_calendar(prices: pd.DataFrame) -> list[str]:
    counts = prices.groupby("ticker")["trade_date"].nunique()
    if counts.empty:
        return []
    anchor = counts.idxmax()
    return sorted(prices.loc[prices["ticker"] == anchor, "trade_date"].astype(str).unique())


def build_panel(prices: pd.DataFrame, news: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    features = price_features(prices)
    calendar = market_calendar(prices)
    calendars = {
        ticker: sorted(part["trade_date"].astype(str).unique())
        for ticker, part in prices.groupby("ticker")
    }
    if links.empty:
        linked = links.copy()
    else:
        linked = links.merge(
            news[["channel", "msg_id", "published_at", "sentiment"]],
            on=["channel", "msg_id"],
            how="left",
        )
        parts = []
        for ticker, part in linked.groupby("ticker"):
            cal = calendars.get(ticker, calendar)
            parts.append(_map_frame(part, cal))
        linked = pd.concat(parts, ignore_index=True) if parts else linked.iloc[0:0]
    panel = features.merge(_alias_agg(linked), on=["ticker", "trade_date"], how="left")
    panel = panel.merge(_sector_agg(linked), on=["ticker", "trade_date"], how="left")
    panel = panel.merge(_macro_agg(news, calendar), on="trade_date", how="left")
    for column in ZERO_NEWS:
        if column not in panel.columns:
            panel[column] = 0.0
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
    panel = panel.dropna(subset=["ret_1", "ret_5", "ret_20", "target"])
    return panel.reset_index(drop=True)


def load_frames(conn) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_sql_query("SELECT * FROM prices", conn)
    news = pd.read_sql_query(
        "SELECT channel, msg_id, published_at, text, sentiment, relevant, is_macro FROM news",
        conn,
    )
    links = pd.read_sql_query("SELECT * FROM news_links", conn)
    return prices, news, links


def seasonal_profile(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frame = prices.loc[prices["ticker"].isin(tickers)].sort_values(["ticker", "trade_date"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "month", "mean_return", "n"])
    frame["ret"] = frame.groupby("ticker")["close"].pct_change()
    frame["month"] = pd.to_datetime(frame["trade_date"]).dt.month
    frame = frame.dropna(subset=["ret"])
    return (
        frame.groupby(["ticker", "month"], as_index=False)
        .agg(mean_return=("ret", "mean"), n=("ret", "size"))
    )


def latest_market_row(prices: pd.DataFrame, ticker: str) -> dict | None:
    """Признаки последнего закрытия, по которым можно строить прогноз следующего дня."""
    features = price_features(prices.loc[prices["ticker"] == ticker])
    ready = features.dropna(subset=["ret_1", "ret_5", "ret_20"])
    if ready.empty:
        return None
    row = ready.iloc[-1]
    return {column: float(row[column]) for column in ("ret_1", "ret_5", "ret_20", "volume_z", "month_sin", "month_cos", "dow_sin", "dow_cos")}


def hits_to_feature_patch(hits: list[Hit], sentiment: float, is_macro: bool) -> dict[str, dict]:
    """Для каждой бумаги — новостные признаки одного текста."""
    patches: dict[str, dict] = {}
    for hit in hits:
        patch = {column: 0.0 for column in ZERO_NEWS}
        if hit.match_type == "alias":
            patch["sent_mean_alias"] = sentiment
            patch["sent_sum_alias"] = sentiment
            patch["news_count_alias"] = 1.0
        else:
            patch["sent_mean_sector"] = sentiment
            patch["sent_sum_sector"] = sentiment * hit.weight
            patch["news_count_sector"] = hit.weight
        if is_macro:
            patch["sent_mean_macro"] = sentiment
            patch["news_count_macro"] = 1.0
        patch["match_type"] = hit.match_type
        patch["match_term"] = hit.term
        patches[hit.ticker] = patch
    return patches
