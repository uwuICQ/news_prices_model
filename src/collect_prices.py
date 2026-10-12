"""Дневные итоги TQBR с публичного ISS Мосбиржи."""

from __future__ import annotations

import sys
import time

import requests

from src.db import connect
from src.universe import load_universe

ISS_URL = (
    "https://iss.moex.com/iss/history/engines/stock/markets/shares/"
    "boards/TQBR/securities/{ticker}.json"
)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "moex-news-study/1.0"


def _setup_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def _pick_close(row: dict) -> float | None:
    for key in ("CLOSE", "LEGALCLOSEPRICE", "WAPRICE"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def fetch_history(ticker: str, start: str, end: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        payload = _get_page(ticker, start, end, offset)
        history = payload["history"]
        columns = history["columns"]
        data = history["data"]
        for raw in data:
            item = dict(zip(columns, raw))
            close = _pick_close(item)
            if close is None or close <= 0 or not item.get("TRADEDATE"):
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": item["TRADEDATE"],
                    "open": item.get("OPEN"),
                    "high": item.get("HIGH"),
                    "low": item.get("LOW"),
                    "close": close,
                    "volume": item.get("VOLUME"),
                    "value": item.get("VALUE"),
                }
            )
        cursor = dict(zip(payload["history.cursor"]["columns"], payload["history.cursor"]["data"][0]))
        offset = int(cursor["INDEX"]) + int(cursor["PAGESIZE"])
        if offset >= int(cursor["TOTAL"]) or not data:
            break
        time.sleep(0.15)
    return rows


def _get_page(ticker: str, start: str, end: str, offset: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = SESSION.get(
                ISS_URL.format(ticker=ticker),
                params={"from": start, "till": end, "start": offset, "iss.meta": "off"},
                timeout=40,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ISS не ответил по {ticker}") from last_error


def store_ticker(conn, ticker: str, start: str, end: str) -> int:
    try:
        rows = fetch_history(ticker, start, end)
    except Exception as exc:
        conn.execute(
            """
            INSERT INTO ticker_status (ticker, n_rows, first_date, last_date, ok, note)
            VALUES (?, 0, NULL, NULL, 0, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                n_rows=0, first_date=NULL, last_date=NULL, ok=0, note=excluded.note
            """,
            (ticker, str(exc)),
        )
        conn.commit()
        return 0
    if not rows:
        note = "ISS не вернул свечи за окно"
        conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
        conn.execute(
            """
            INSERT INTO ticker_status (ticker, n_rows, first_date, last_date, ok, note)
            VALUES (?, 0, NULL, NULL, 0, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                n_rows=0, first_date=NULL, last_date=NULL, ok=0, note=excluded.note
            """,
            (ticker, note),
        )
        conn.commit()
        return 0
    dates = sorted(row["trade_date"] for row in rows)
    note = ""
    if dates[0] > start:
        note = f"ряд начинается с {dates[0]}"
    conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
    conn.executemany(
        """
        INSERT INTO prices (ticker, trade_date, open, high, low, close, volume, value)
        VALUES (:ticker, :trade_date, :open, :high, :low, :close, :volume, :value)
        """,
        rows,
    )
    conn.execute(
        """
        INSERT INTO ticker_status (ticker, n_rows, first_date, last_date, ok, note)
        VALUES (?, ?, ?, ?, 1, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            n_rows=excluded.n_rows,
            first_date=excluded.first_date,
            last_date=excluded.last_date,
            ok=1,
            note=excluded.note
        """,
        (ticker, len(rows), dates[0], dates[-1], note),
    )
    conn.commit()
    return len(rows)


def main() -> None:
    _setup_stdout()
    universe = load_universe()
    conn = connect()
    try:
        for ticker in universe.tickers:
            print(f"цены {ticker.ticker} {ticker.name}", flush=True)
            count = store_ticker(conn, ticker.ticker, universe.start, universe.end)
            print(f"  строк: {count}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
