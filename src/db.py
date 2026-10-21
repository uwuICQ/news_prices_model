from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market.sqlite"
MODEL_PATH = ROOT / "models" / "forecast.joblib"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    value REAL,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS news (
    channel TEXT NOT NULL,
    msg_id INTEGER NOT NULL,
    published_at TEXT NOT NULL,
    text TEXT NOT NULL,
    sentiment REAL,
    relevant INTEGER NOT NULL DEFAULT 0,
    is_macro INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (channel, msg_id)
);

CREATE TABLE IF NOT EXISTS news_links (
    channel TEXT NOT NULL,
    msg_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    match_type TEXT NOT NULL,
    weight REAL NOT NULL,
    match_term TEXT NOT NULL,
    PRIMARY KEY (channel, msg_id, ticker)
);

CREATE TABLE IF NOT EXISTS ticker_status (
    ticker TEXT PRIMARY KEY,
    n_rows INTEGER,
    first_date TEXT,
    last_date TEXT,
    ok INTEGER,
    note TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    name TEXT PRIMARY KEY,
    value REAL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS feature_importance (
    feature TEXT PRIMARY KEY,
    importance REAL
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at);
CREATE INDEX IF NOT EXISTS idx_links_ticker ON news_links(ticker);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn
