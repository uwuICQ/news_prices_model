"""История публичных каналов @rian_ru и @mash через Telethon.

Первый запуск интерактивный: код входа приходит в Telegram.
Повторный запуск догружает сообщения новее уже просмотренных.
Если сбор оборвался, история дочитывается со старого края, а не заново с самых новых постов.
Сессия лежит в data/ и в git не попадает.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.db import ROOT, connect
from src.universe import load_universe

MSK = ZoneInfo("Europe/Moscow")
SESSION = ROOT / "data" / "telegram"
_FLUSH_EVERY = 100
CLEAR = object()


def _setup_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def window_bounds(start_iso: str, end_iso: str) -> tuple[datetime, datetime]:
    """Начало включительно, конец — полночь следующих суток по Москве."""
    start = datetime.fromisoformat(start_iso).replace(tzinfo=MSK)
    end_exclusive = datetime.fromisoformat(end_iso).replace(tzinfo=MSK) + timedelta(days=1)
    return start, end_exclusive


def classify_published(published: datetime, start: datetime, end_exclusive: datetime) -> str:
    if published < start:
        return "older"
    if published >= end_exclusive:
        return "newer"
    return "keep"


@dataclass
class SyncState:
    oldest: int | None = None
    newest: int | None = None
    history_complete: bool = False
    forward_lo: int | None = None
    forward_hi: int | None = None


def _prefer_min(old: int | None, new: int | None) -> int | None:
    if new is None:
        return old
    if old is None:
        return new
    return min(old, new)


def _prefer_max(old: int | None, new: int | None) -> int | None:
    if new is None:
        return old
    if old is None:
        return new
    return max(old, new)


def merge_sync(old: SyncState, patch: dict) -> SyncState:
    """patch может сбрасывать курсор догрузки значением CLEAR."""

    def pick(name: str, prefer):
        if name not in patch:
            return getattr(old, name)
        value = patch[name]
        if value is CLEAR:
            return None
        return prefer(getattr(old, name), value)

    return SyncState(
        oldest=pick("oldest", _prefer_min),
        newest=pick("newest", _prefer_max),
        history_complete=old.history_complete or bool(patch.get("history_complete")),
        forward_lo=pick("forward_lo", _prefer_min),
        forward_hi=pick("forward_hi", _prefer_max),
    )


def _sync_state(conn, channel: str) -> SyncState:
    row = conn.execute(
        """
        SELECT oldest_scanned_id, newest_scanned_id, history_complete, forward_lo, forward_hi
        FROM news_sync
        WHERE channel = ?
        """,
        (channel,),
    ).fetchone()
    if row is None:
        return SyncState()

    def _num(name: str) -> int | None:
        value = row[name]
        return None if value is None else int(value)

    return SyncState(
        oldest=_num("oldest_scanned_id"),
        newest=_num("newest_scanned_id"),
        history_complete=bool(row["history_complete"]),
        forward_lo=_num("forward_lo"),
        forward_hi=_num("forward_hi"),
    )


def _write_state(conn, channel: str, state: SyncState) -> None:
    conn.execute(
        """
        INSERT INTO news_sync (
            channel, oldest_scanned_id, newest_scanned_id, history_complete, forward_lo, forward_hi
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(channel) DO UPDATE SET
            oldest_scanned_id = excluded.oldest_scanned_id,
            newest_scanned_id = excluded.newest_scanned_id,
            history_complete = excluded.history_complete,
            forward_lo = excluded.forward_lo,
            forward_hi = excluded.forward_hi
        """,
        (
            channel,
            state.oldest,
            state.newest,
            int(state.history_complete),
            state.forward_lo,
            state.forward_hi,
        ),
    )


def _save_progress(conn, channel: str, batch: list[tuple], patch: dict) -> None:
    if batch:
        conn.executemany(
            """
            INSERT OR IGNORE INTO news (channel, msg_id, published_at, text)
            VALUES (?, ?, ?, ?)
            """,
            batch,
        )
    if patch:
        _write_state(conn, channel, merge_sync(_sync_state(conn, channel), patch))
    if batch or patch:
        conn.commit()


def _as_msk(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ZoneInfo("UTC"))
    return moment.astimezone(MSK)


def _checkpoint_patch(
    mode: str,
    scanned_lo: int | None,
    scanned_hi: int | None,
    *,
    done: bool,
    prior_forward_hi: int | None,
) -> dict:
    if mode == "history":
        patch: dict = {}
        if scanned_lo is not None:
            patch["oldest"] = scanned_lo
        if scanned_hi is not None:
            patch["newest"] = scanned_hi
        if done:
            patch["history_complete"] = True
        return patch
    if not done:
        patch = {}
        if scanned_lo is not None:
            patch["forward_lo"] = scanned_lo
        if scanned_hi is not None:
            patch["forward_hi"] = scanned_hi
        return patch
    # Догрузка идёт от новых к старым. Верхнюю границу двигаем только когда
    # промежуток до уже сохранённого края просмотрен целиком, иначе дыра теряется.
    patch = {"forward_lo": CLEAR, "forward_hi": CLEAR}
    candidates = [value for value in (prior_forward_hi, scanned_hi) if value is not None]
    if candidates:
        patch["newest"] = max(candidates)
    return patch


async def _open_channel(client, channel: str):
    from telethon.errors import FloodWaitError, UserAlreadyParticipantError
    from telethon.tl.functions.channels import JoinChannelRequest

    try:
        entity = await client.get_entity(channel)
    except Exception as exc:
        raise SystemExit(f"Не найден канал @{channel}: {exc}") from exc
    try:
        await client(JoinChannelRequest(entity))
    except UserAlreadyParticipantError:
        pass
    except FloodWaitError as exc:
        print(f"  пауза Telegram {exc.seconds} с", flush=True)
        await asyncio.sleep(exc.seconds + 1)
    except Exception as exc:
        print(f"  @{channel}: читаю без вступления ({exc})", flush=True)
    return entity


async def _walk(
    client,
    conn,
    entity,
    channel: str,
    start: datetime,
    end_exclusive: datetime,
    *,
    mode: str,
    min_id: int,
    max_id: int,
    stop_when_older: bool,
) -> tuple[int, bool]:
    """Обход от новых к старым. max_id и min_id исключают уже просмотренные id."""
    from telethon.errors import FloodWaitError

    saved = 0
    visited = 0
    printed = 0
    batch: list[tuple] = []
    scanned_lo: int | None = None
    scanned_hi: int | None = None
    cursor_max = max_id
    history_done = False
    prior_forward_hi = _sync_state(conn, channel).forward_hi

    def checkpoint(done: bool) -> None:
        nonlocal batch
        patch = _checkpoint_patch(
            mode,
            scanned_lo,
            scanned_hi,
            done=done,
            prior_forward_hi=prior_forward_hi,
        )
        _save_progress(conn, channel, batch, patch)
        batch = []

    while True:
        kwargs: dict = {"reverse": False}
        if min_id:
            kwargs["min_id"] = min_id
        if cursor_max:
            kwargs["max_id"] = cursor_max
        try:
            async for message in client.iter_messages(entity, **kwargs):
                if message.id is None:
                    continue
                mid = int(message.id)
                scanned_lo = mid if scanned_lo is None else min(scanned_lo, mid)
                scanned_hi = mid if scanned_hi is None else max(scanned_hi, mid)
                cursor_max = scanned_lo
                visited += 1
                if message.date is not None:
                    published = _as_msk(message.date)
                    kind = classify_published(published, start, end_exclusive)
                    if kind == "older" and stop_when_older:
                        history_done = True
                        break
                    if kind == "keep":
                        text = (message.message or "").strip()
                        if text:
                            batch.append((channel, mid, published.isoformat(timespec="seconds"), text))
                            saved += 1
                if visited % _FLUSH_EVERY == 0:
                    checkpoint(False)
                    if saved // 500 > printed:
                        printed = saved // 500
                        print(f"  {channel}: {saved}", flush=True)
            else:
                if stop_when_older:
                    history_done = True
            break
        except FloodWaitError as exc:
            checkpoint(False)
            print(f"  пауза Telegram {exc.seconds} с", flush=True)
            await asyncio.sleep(exc.seconds + 1)
        except Exception:
            checkpoint(False)
            raise
    checkpoint(history_done if mode == "history" else True)
    return saved, history_done


async def _collect_channel(client, conn, channel: str, start: datetime, end_exclusive: datetime) -> int:
    entity = await _open_channel(client, channel)
    saved = 0
    state = _sync_state(conn, channel)
    if not state.history_complete:
        print("  история до начала окна", flush=True)
        count, _done = await _walk(
            client,
            conn,
            entity,
            channel,
            start,
            end_exclusive,
            mode="history",
            min_id=0,
            max_id=state.oldest or 0,
            stop_when_older=True,
        )
        saved += count
        state = _sync_state(conn, channel)
    if state.history_complete and (state.newest or state.forward_lo):
        print("  новые сообщения", flush=True)
        count, _done = await _walk(
            client,
            conn,
            entity,
            channel,
            start,
            end_exclusive,
            mode="forward",
            min_id=state.newest or 0,
            max_id=state.forward_lo or 0,
            stop_when_older=False,
        )
        saved += count
    return saved


async def run() -> None:
    from telethon import TelegramClient

    load_dotenv(ROOT / ".env")
    api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    if not api_id or not api_hash:
        raise SystemExit(
            "Нет TELEGRAM_API_ID и TELEGRAM_API_HASH. Скопируйте .env.example в .env "
            "и вставьте ключи с https://my.telegram.org"
        )
    try:
        api_id_int = int(api_id)
    except ValueError as exc:
        raise SystemExit("TELEGRAM_API_ID должен быть числом с https://my.telegram.org") from exc
    universe = load_universe()
    start, end_exclusive = window_bounds(universe.start, universe.end)
    SESSION.parent.mkdir(parents=True, exist_ok=True)
    print(f"сессия: {SESSION}.session", flush=True)
    conn = connect()
    client = TelegramClient(str(SESSION), api_id_int, api_hash)
    try:
        try:
            await client.start()
        except Exception as exc:
            raise SystemExit(f"Не удалось войти в Telegram: {exc}") from exc
        for channel in universe.channels:
            print(f"канал @{channel}", flush=True)
            count = await _collect_channel(client, conn, channel, start, end_exclusive)
            print(f"  сохранено сообщений: {count}", flush=True)
    finally:
        await client.disconnect()
        conn.close()


def main() -> None:
    _setup_stdout()
    asyncio.run(run())


if __name__ == "__main__":
    main()
