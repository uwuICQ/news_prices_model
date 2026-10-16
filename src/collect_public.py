"""Новости без ключей Telegram: поиск РИА и лента mash.ru.

Те же каналы rian_ru и mash, то же окно и та же таблица news.
Повторный запуск догружает только новые материалы. Если сбор оборвался,
следующий продолжается с сохранённого смещения в data/public_progress.json.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from html import unescape

import requests

from src.collect_news import MSK, window_bounds
from src.db import ROOT, connect
from src.universe import Universe, load_universe

RIA_SEARCH = "https://ria.ru/services/search/getmore/"
MASH_POSTS = "https://mash.ru/api/posts/"
PROGRESS_PATH = ROOT / "data" / "public_progress.json"
PAGE_LIMIT = 100
# Короткие слова цепляют посторонние материалы и раздувают выгрузку.
SKIP_QUERIES = {"vk", "банк", "порт"}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
SESSION.headers["Accept-Language"] = "ru-RU,ru;q=0.9"
SESSION.headers["Referer"] = "https://ria.ru/"

# Поиск РИА ищет словоформы, а в конфиге лежат стемы для привязки текста.
SEARCH_FORMS = {
    "инарктик": "инарктика",
    "пшениц": "пшеница",
    "зернов": "зерно",
    "сахарн": "сахар",
    "свинин": "свинина",
    "удобрен": "удобрения",
    "посевн": "посевная",
    "свекл": "свекла",
    "аквакультур": "аквакультура",
    "нефт": "нефть",
    "дизел": "дизель",
    "нефтепродукт": "нефтепродукты",
    "баррел": "баррель",
    "газодобыч": "газодобыча",
    "судоход": "судоходство",
    "грузоперевоз": "грузоперевозки",
    "логистик": "логистика",
    "ипоте": "ипотека",
    "вкладов": "вклады",
    "кредитован": "кредитование",
    "кибербезопас": "кибербезопасность",
    "ит-компани": "ит-компания",
    "интернет-компани": "интернет-компания",
    "санкц": "санкции",
    "инфляц": "инфляция",
    "курс рубл": "курс рубля",
    "т-технолог": "т-технологии",
}

_ITEM_RE = re.compile(
    r'href="(?:https://ria.ru)?/(?P<day>\d{8})/(?P<slug>[^"]+)\.html"'
    r'\s+class="list-item__title[^"]*"[^>]*>(?P<title>[^<]*)</a>'
    r".{0,2500}?data-type=\"date\">(?P<label>[^<]*)</div>",
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


def plain_text(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def published_from_ria(day: str, label: str) -> datetime:
    """День берётся из адреса материала, время — из подписи РИА. Часовой пояс Москвы."""
    hour, minute = 12, 0
    found = _TIME_RE.search(label or "")
    if found:
        hour = min(int(found.group(1)), 23)
        minute = min(int(found.group(2)), 59)
    return datetime(
        int(day[0:4]),
        int(day[4:6]),
        int(day[6:8]),
        hour,
        minute,
        tzinfo=MSK,
    )


def parse_ria_page(html: str) -> tuple[list[dict], int | None]:
    page = unescape(html)
    if "m-notfound" in page:
        return [], 0
    count_match = re.search(r'data-count="(\d+)"', page)
    total = int(count_match.group(1)) if count_match else None
    items: list[dict] = []
    seen: set[int] = set()
    for match in _ITEM_RE.finditer(page):
        slug = match.group("slug")
        id_match = re.search(r"(\d+)$", slug)
        if not id_match:
            continue
        msg_id = int(id_match.group(1))
        if msg_id in seen:
            continue
        seen.add(msg_id)
        title = plain_text(match.group("title"))
        if not title:
            continue
        items.append(
            {
                "msg_id": msg_id,
                "published": published_from_ria(match.group("day"), match.group("label")),
                "text": title,
            }
        )
    return items, total


def search_queries(universe: Universe) -> list[str]:
    found: list[str] = []
    for ticker in universe.tickers:
        found.extend(ticker.aliases)
    for sector in universe.sectors.values():
        found.extend(sector.keywords)
    found.extend(universe.macro_keywords)
    unique: list[str] = []
    seen: set[str] = set()
    for raw in found:
        query = SEARCH_FORMS.get(raw.strip().lower().replace("ё", "е"), raw.strip().lower().replace("ё", "е"))
        if len(query) < 3 or query in SKIP_QUERIES or query in seen:
            continue
        seen.add(query)
        unique.append(query)
    return unique


def _load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"ria": {}, "mash": {}}
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def _get(url: str, params: dict) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            response = SESSION.get(url, params=params, timeout=40)
            if response.status_code == 429 or not response.content:
                last_error = requests.RequestException(f"пустой или ограниченный ответ {url}")
                time.sleep(4 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Не ответил {url}") from last_error


def _insert(conn, channel: str, rows: list[tuple]) -> int:
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM news WHERE channel = ?", (channel,)).fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO news (channel, msg_id, published_at, text)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM news WHERE channel = ?", (channel,)).fetchone()[0]
    return int(after - before)


def _in_window(moment: datetime, start: datetime, end_exclusive: datetime) -> bool:
    return start <= moment < end_exclusive


def _halves(start: str, end: str) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Поиск РИА отдаёт не больше 200 материалов на один интервал дат."""
    left = date.fromisoformat(start)
    right = date.fromisoformat(end)
    if right <= left:
        return None
    mid = left + (right - left) // 2
    if mid < left or mid >= right:
        return None
    return (start, mid.isoformat()), ((mid + timedelta(days=1)).isoformat(), end)


def _ria_page(query: str, date_from: str, date_to: str, offset: int) -> tuple[list[dict], int | None]:
    response = _get(
        RIA_SEARCH,
        {
            "query": query,
            "offset": offset,
            "limit": PAGE_LIMIT,
            "date_from": date_from,
            "date_to": date_to,
        },
    )
    return parse_ria_page(response.text)


def _store_ria(conn, items: list[dict], start: datetime, end_exclusive: datetime) -> int:
    rows = [
        ("rian_ru", item["msg_id"], item["published"].isoformat(timespec="seconds"), item["text"])
        for item in items
        if _in_window(item["published"], start, end_exclusive)
    ]
    return _insert(conn, "rian_ru", rows)


def collect_ria(conn, universe: Universe, progress: dict) -> int:
    start, end_exclusive = window_bounds(universe.start, universe.end)
    queries = search_queries(universe)
    saved = 0
    ria_state = progress.setdefault("ria", {})
    for index, query in enumerate(queries, start=1):
        state = ria_state.get(query) or {}
        pending = list(state.get("pending") or [])
        if state.get("done"):
            pending = [[universe.start, universe.end]]
        elif not pending:
            pending = [[universe.start, universe.end]]
        print(f"риа [{index}/{len(queries)}] {query}", flush=True)
        while pending:
            date_from, date_to = pending[0]
            items, total = _ria_page(query, date_from, date_to, 0)
            if total is None:
                raise RuntimeError(f"РИА не отдала счётчик по запросу {query} {date_from}…{date_to}")
            if total > 200:
                halves = _halves(date_from, date_to)
                if halves is None:
                    fresh = _store_ria(conn, items, start, end_exclusive)
                    saved += fresh
                    if total > PAGE_LIMIT:
                        more, _more_total = _ria_page(query, date_from, date_to, PAGE_LIMIT)
                        saved += _store_ria(conn, more, start, end_exclusive)
                    print(f"  {date_from}…{date_to}: день длиннее лимита поиска, взято до 200", flush=True)
                    pending.pop(0)
                else:
                    print(f"  {date_from}…{date_to}: в поиске {total}, делю интервал", flush=True)
                    pending[0:1] = [list(halves[0]), list(halves[1])]
                    ria_state[query] = {"pending": pending, "done": False}
                    _save_progress(progress)
                    time.sleep(0.15)
                    continue
            else:
                saved += _store_ria(conn, items, start, end_exclusive)
                loaded = len(items)
                offset = PAGE_LIMIT
                while loaded < total:
                    more, _more_total = _ria_page(query, date_from, date_to, offset)
                    if not more:
                        break
                    saved += _store_ria(conn, more, start, end_exclusive)
                    loaded += len(more)
                    offset += PAGE_LIMIT
                    time.sleep(0.15)
                print(f"  {date_from}…{date_to}: в поиске {total}", flush=True)
                pending.pop(0)
            ria_state[query] = {"pending": pending, "done": False}
            _save_progress(progress)
            time.sleep(0.15)
        ria_state[query] = {"pending": [], "done": True}
        _save_progress(progress)
    return saved


def collect_mash(conn, universe: Universe, progress: dict) -> int:
    start, end_exclusive = window_bounds(universe.start, universe.end)
    state = progress.setdefault("mash", {})
    done = bool(state.get("done"))
    cursor = None if done else state.get("cursor")
    saved = 0
    seen_cursors: set[str] = set()
    print("mash", flush=True)
    while True:
        params: dict = {"limit": PAGE_LIMIT, "order": "-publishedAt", "type": "news"}
        if cursor:
            params["cursor"] = cursor
        payload = _get(MASH_POSTS, params).json()
        items = payload.get("items") or []
        rows = []
        reached_past = False
        for item in items:
            raw_date = item.get("publishedAt")
            title = item.get("title") or {}
            text = plain_text(title.get("data") or title.get("seoTitle") or "")
            if not raw_date or not text or item.get("id") is None:
                continue
            published = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).astimezone(MSK)
            if published >= end_exclusive:
                continue
            if published < start:
                reached_past = True
                continue
            rows.append(("mash", int(item["id"]), published.isoformat(timespec="seconds"), text))
        fresh = _insert(conn, "mash", rows)
        saved += fresh
        cursor = payload.get("afterCursor")
        print(f"  страница {len(items)}, новых {fresh}", flush=True)
        if cursor and cursor in seen_cursors:
            cursor = None
        if cursor:
            seen_cursors.add(cursor)
        if reached_past or not items or not cursor or (done and fresh == 0):
            progress["mash"] = {"cursor": None, "done": True}
            _save_progress(progress)
            break
        progress["mash"] = {"cursor": cursor, "done": False}
        _save_progress(progress)
        time.sleep(0.15)
    return saved


def collect_public() -> None:
    universe = load_universe()
    progress = _load_progress()
    conn = connect()
    try:
        mash_saved = collect_mash(conn, universe, progress)
        ria_saved = collect_ria(conn, universe, progress)
        print(f"сохранено новых: mash {mash_saved}, риа {ria_saved}", flush=True)
    finally:
        conn.close()
