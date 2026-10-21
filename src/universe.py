from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.db import ROOT

CONFIG_PATH = ROOT / "config" / "universe.yaml"


@dataclass(frozen=True)
class Ticker:
    ticker: str
    name: str
    sector: str
    seasonal: bool
    aliases: tuple[str, ...]
    history_note: str = ""


@dataclass(frozen=True)
class Sector:
    id: str
    name: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Universe:
    start: str
    end: str
    train_end: str
    channels: tuple[str, ...]
    macro_keywords: tuple[str, ...]
    sectors: dict[str, Sector]
    tickers: tuple[Ticker, ...]

    def get(self, ticker: str) -> Ticker:
        for item in self.tickers:
            if item.ticker == ticker:
                return item
        raise KeyError(ticker)

    def sector_name(self, sector_id: str) -> str:
        return self.sectors[sector_id].name


@lru_cache(maxsize=1)
def load_universe(path: str | None = None) -> Universe:
    raw = yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    sectors = {
        key: Sector(id=key, name=value["name"], keywords=tuple(value.get("keywords") or ()))
        for key, value in raw["sectors"].items()
    }
    tickers = tuple(
        Ticker(
            ticker=item["ticker"],
            name=item["name"],
            sector=item["sector"],
            seasonal=bool(item.get("seasonal", True)),
            aliases=tuple(item.get("aliases") or ()),
            history_note=item.get("history_note") or "",
        )
        for item in raw["tickers"]
    )
    unknown = {item.sector for item in tickers} - set(sectors)
    if unknown:
        raise ValueError(f"Неизвестные сегменты: {sorted(unknown)}")
    return Universe(
        start=raw["window"]["start"],
        end=raw["window"]["end"],
        train_end=raw["train_end"],
        channels=tuple(raw["channels"]),
        macro_keywords=tuple(raw.get("macro_keywords") or ()),
        sectors=sectors,
        tickers=tickers,
    )
