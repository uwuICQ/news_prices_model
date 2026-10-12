"""Привязка текста новости к бумагам по алиасам и словам сегмента."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.universe import Universe

# Окончания, при которых стема ещё нет: портфель, банкет, газпромнефть.
BLOCK_SUFFIXES: dict[str, tuple[str, ...]] = {
    "порт": ("фел", "ал", "рет", "ье", "уг"),
    "банк": ("ет",),
    "газпром": ("нефт",),
    "сбер": ("еж", "ег", "книж"),
}


@dataclass(frozen=True)
class Hit:
    ticker: str
    match_type: str
    weight: float
    term: str


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def term_found(text: str, term: str) -> bool:
    """Стем с левой границы слова. text уже нормализован."""
    needle = normalize(term)
    if not needle:
        return False
    blocks = BLOCK_SUFFIXES.get(needle, ())
    pattern = _compiled(needle)
    for match in pattern.finditer(text):
        tail = text[match.end(): match.end() + 6]
        if any(tail.startswith(block) for block in blocks):
            continue
        return True
    return False


@lru_cache(maxsize=None)
def _compiled(needle: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![0-9a-zа-я]){re.escape(needle)}")


def classify(text: str, universe: Universe) -> tuple[list[Hit], bool]:
    """Алиас важнее сегмента для той же бумаги. Сегмент — вес 0.5."""
    norm = normalize(text)
    hits: list[Hit] = []
    matched: set[str] = set()
    for ticker in universe.tickers:
        found = ""
        for alias in sorted(ticker.aliases, key=len, reverse=True):
            if term_found(norm, alias):
                found = alias
                break
        if found:
            hits.append(Hit(ticker.ticker, "alias", 1.0, found))
            matched.add(ticker.ticker)
    for ticker in universe.tickers:
        if ticker.ticker in matched:
            continue
        found = ""
        for keyword in universe.sectors[ticker.sector].keywords:
            if term_found(norm, keyword):
                found = keyword
                break
        if found:
            hits.append(Hit(ticker.ticker, "sector", 0.5, found))
    is_macro = any(term_found(norm, keyword) for keyword in universe.macro_keywords)
    return hits, is_macro
