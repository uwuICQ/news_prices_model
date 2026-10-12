"""Прогноз доходности следующего дня по тексту новости."""

from __future__ import annotations

import joblib
import pandas as pd

from src.db import MODEL_PATH, connect
from src.features import FEATURE_COLS, ZERO_NEWS, hits_to_feature_patch, latest_market_row
from src.link import classify
from src.sentiment import SentimentScorer
from src.universe import load_universe

_SCORER: SentimentScorer | None = None
_BUNDLE: dict | None = None


def _scorer() -> SentimentScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = SentimentScorer()
    return _SCORER


def _bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Модель не обучена. Выполните: python -m src.train")
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def reset_cache() -> None:
    global _SCORER, _BUNDLE
    _SCORER = None
    _BUNDLE = None


def direction(expected_return: float) -> str:
    if expected_return > 0.001:
        return "Рост"
    if expected_return < -0.001:
        return "Снижение"
    return "Около нуля"


def _reason(match_type: str, term: str) -> str:
    if match_type == "alias":
        return f"алиас: {term}"
    return f"сегмент: {term}"


def forecast_texts(items: list[dict]) -> list[dict]:
    """items: словари с ключом text и необязательными channel, published_at."""
    universe = load_universe()
    parsed = []
    pending_texts: list[str] = []
    for item in items:
        hits, is_macro = classify(item.get("text") or "", universe)
        parsed.append((item, hits, is_macro))
        if hits:
            pending_texts.append(item.get("text") or "")
    if not pending_texts:
        return [
            {
                "channel": item.get("channel"),
                "published_at": item.get("published_at"),
                "text": item.get("text") or "",
                "unlinked": True,
                "forecasts": [],
            }
            for item, hits, is_macro in parsed
        ]
    bundle = _bundle()
    categories = list(bundle["ticker_categories"])
    conn = connect()
    try:
        prices = pd.read_sql_query("SELECT * FROM prices", conn)
    finally:
        conn.close()
    sentiments = _scorer().score(pending_texts)
    sentiment_at = 0
    market_cache: dict[str, dict | None] = {}
    results: list[dict] = []
    for item, hits, is_macro in parsed:
        if not hits:
            results.append(
                {
                    "channel": item.get("channel"),
                    "published_at": item.get("published_at"),
                    "text": item.get("text") or "",
                    "unlinked": True,
                    "forecasts": [],
                }
            )
            continue
        sentiment = sentiments[sentiment_at]
        sentiment_at += 1
        patches = hits_to_feature_patch(hits, sentiment, is_macro)
        forecasts = []
        batch_rows = []
        batch_meta = []
        for ticker, patch in patches.items():
            info = universe.get(ticker)
            if ticker not in market_cache:
                market_cache[ticker] = latest_market_row(prices, ticker)
            market = market_cache[ticker]
            base = {
                "ticker": ticker,
                "name": info.name,
                "sector": universe.sector_name(info.sector),
                "sentiment": sentiment,
                "reason": _reason(patch["match_type"], patch["match_term"]),
            }
            if market is None:
                forecasts.append({**base, "direction": "—", "expected_return": None, "note": "Недостаточно истории цен: для доходности за 20 дней нужно хотя бы 21 закрытие."})
                continue
            if ticker not in categories:
                forecasts.append({**base, "direction": "—", "expected_return": None, "note": "Бумага не входила в обучение."})
                continue
            row = {**market, **{column: float(patch[column]) for column in ZERO_NEWS}, "ticker": ticker}
            batch_rows.append(row)
            batch_meta.append(base)
        if batch_rows:
            frame = pd.DataFrame(batch_rows)
            frame["ticker"] = pd.Categorical(frame["ticker"], categories=categories)
            predicted = bundle["model"].predict(frame[FEATURE_COLS + ["ticker"]])
            for base, value in zip(batch_meta, predicted):
                expected = float(value)
                forecasts.append(
                    {
                        **base,
                        "direction": direction(expected),
                        "expected_return": expected,
                        "note": "От последнего закрытия в базе к следующему торговому дню.",
                    }
                )
        forecasts.sort(key=lambda row: row["ticker"])
        results.append(
            {
                "channel": item.get("channel"),
                "published_at": item.get("published_at"),
                "text": item.get("text") or "",
                "unlinked": False,
                "forecasts": forecasts,
            }
        )
    return results


def recent_news(channel: str | None, limit: int = 10) -> list[dict]:
    conn = connect()
    try:
        if channel:
            rows = conn.execute(
                """
                SELECT channel, published_at, text
                FROM news
                WHERE channel = ?
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (channel, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT channel, published_at, text
                FROM news
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    finally:
        conn.close()
    return [{"channel": row["channel"], "published_at": row["published_at"], "text": row["text"]} for row in rows]
