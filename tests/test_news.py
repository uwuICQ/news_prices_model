import asyncio
import sqlite3
import unittest
from datetime import datetime, timezone

from src.collect_news import (
    CLEAR,
    SyncState,
    _collect_channel,
    _sync_state,
    _walk,
    classify_published,
    merge_sync,
    window_bounds,
)
from src.db import SCHEMA

MSK_SAMPLE = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _msg(mid: int, moment: datetime, text: str):
    return type("Msg", (), {"id": mid, "date": moment, "message": text})()


class _Client:
    def __init__(self, messages, fail_after: int | None = None):
        self.messages = list(messages)
        self.fail_after = fail_after

    async def get_entity(self, channel):
        return channel

    async def __call__(self, request):
        return None

    def iter_messages(self, entity, min_id=0, max_id=0, **kwargs):
        fail_after = self.fail_after
        messages = self.messages

        async def _gen():
            yielded = 0
            for message in messages:
                if min_id and message.id <= min_id:
                    continue
                if max_id and message.id >= max_id:
                    continue
                if fail_after is not None and yielded >= fail_after:
                    raise RuntimeError("обрыв")
                yielded += 1
                yield message

        return _gen()


def _history_messages():
    return [
        _msg(50, datetime(2026, 10, 13, 0, 30, tzinfo=timezone.utc), "после окна"),
        _msg(40, datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc), "последний день"),
        _msg(30, datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc), "погода в сочи"),
        _msg(20, datetime(2023, 10, 11, 21, 0, tzinfo=timezone.utc), "на границе"),
        _msg(10, datetime(2023, 10, 11, 20, 0, tzinfo=timezone.utc), "слишком рано"),
    ]


class NewsWindowTests(unittest.TestCase):
    def test_window_includes_last_calendar_day(self):
        start, end_exclusive = window_bounds("2023-10-12", "2026-10-12")
        self.assertEqual(classify_published(start, start, end_exclusive), "keep")
        last = datetime(2026, 10, 12, 23, 0, tzinfo=start.tzinfo)
        after = datetime(2026, 10, 13, 0, 0, tzinfo=start.tzinfo)
        before = datetime(2023, 10, 11, 23, 59, tzinfo=start.tzinfo)
        self.assertEqual(classify_published(last, start, end_exclusive), "keep")
        self.assertEqual(classify_published(after, start, end_exclusive), "newer")
        self.assertEqual(classify_published(before, start, end_exclusive), "older")
        self.assertEqual(classify_published(MSK_SAMPLE, start, end_exclusive), "keep")

    def test_merge_keeps_forward_gap_until_cleared(self):
        base = SyncState(oldest=10, newest=30, history_complete=True)
        partial = merge_sync(base, {"forward_lo": 50, "forward_hi": 50})
        self.assertEqual(partial.newest, 30)
        self.assertEqual(partial.forward_lo, 50)
        lowered = merge_sync(partial, {"forward_lo": 40, "forward_hi": 55})
        self.assertEqual(lowered.forward_lo, 40)
        self.assertEqual(lowered.forward_hi, 55)
        self.assertEqual(lowered.newest, 30)
        done = merge_sync(lowered, {"forward_lo": CLEAR, "forward_hi": CLEAR, "newest": 55})
        self.assertIsNone(done.forward_lo)
        self.assertIsNone(done.forward_hi)
        self.assertEqual(done.newest, 55)
        self.assertTrue(done.history_complete)
        self.assertEqual(done.oldest, 10)

    def test_history_keeps_unrelated_posts_inside_window(self):
        asyncio.run(self._history())

    async def _history(self):
        conn = _db()
        try:
            start, end_exclusive = window_bounds("2023-10-12", "2026-10-12")
            saved, done = await _walk(
                _Client(_history_messages()),
                conn,
                "rian_ru",
                "rian_ru",
                start,
                end_exclusive,
                mode="history",
                min_id=0,
                max_id=0,
                stop_when_older=True,
            )
            self.assertTrue(done)
            self.assertEqual(saved, 3)
            stored = [row["text"] for row in conn.execute("SELECT text FROM news ORDER BY msg_id")]
            self.assertEqual(stored, ["на границе", "погода в сочи", "последний день"])
            state = _sync_state(conn, "rian_ru")
            self.assertTrue(state.history_complete)
            self.assertEqual(state.oldest, 10)
            self.assertEqual(state.newest, 50)
        finally:
            conn.close()

    def test_interrupted_history_resumes_below_saved_edge(self):
        asyncio.run(self._resume_history())

    async def _resume_history(self):
        conn = _db()
        try:
            start, end_exclusive = window_bounds("2023-10-12", "2026-10-12")
            messages = [
                _msg(30, datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc), "свежее"),
                _msg(20, datetime(2024, 1, 10, 9, 0, tzinfo=timezone.utc), "середина"),
                _msg(10, datetime(2023, 10, 11, 20, 0, tzinfo=timezone.utc), "рано"),
            ]
            with self.assertRaises(RuntimeError):
                await _walk(
                    _Client(messages, fail_after=1),
                    conn,
                    "mash",
                    "mash",
                    start,
                    end_exclusive,
                    mode="history",
                    min_id=0,
                    max_id=0,
                    stop_when_older=True,
                )
            state = _sync_state(conn, "mash")
            self.assertFalse(state.history_complete)
            self.assertEqual(state.newest, 30)
            self.assertEqual(state.oldest, 30)
            saved, done = await _walk(
                _Client(messages),
                conn,
                "mash",
                "mash",
                start,
                end_exclusive,
                mode="history",
                min_id=0,
                max_id=state.oldest or 0,
                stop_when_older=True,
            )
            self.assertTrue(done)
            self.assertEqual(saved, 1)
            stored = [row["text"] for row in conn.execute("SELECT text FROM news ORDER BY msg_id")]
            self.assertEqual(stored, ["середина", "свежее"])
            self.assertEqual(_sync_state(conn, "mash").newest, 30)
        finally:
            conn.close()

    def test_forward_resume_does_not_skip_the_gap(self):
        asyncio.run(self._forward_gap())

    async def _forward_gap(self):
        conn = _db()
        try:
            start, end_exclusive = window_bounds("2023-10-12", "2026-10-12")
            await _walk(
                _Client([
                    _msg(30, datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc), "база"),
                    _msg(10, datetime(2023, 10, 11, 20, 0, tzinfo=timezone.utc), "рано"),
                ]),
                conn,
                "rian_ru",
                "rian_ru",
                start,
                end_exclusive,
                mode="history",
                min_id=0,
                max_id=0,
                stop_when_older=True,
            )
            fresh = [
                _msg(50, datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc), "самая новая"),
                _msg(40, datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc), "пропущенная"),
                _msg(30, datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc), "база"),
            ]
            with self.assertRaises(RuntimeError):
                await _walk(
                    _Client(fresh, fail_after=1),
                    conn,
                    "rian_ru",
                    "rian_ru",
                    start,
                    end_exclusive,
                    mode="forward",
                    min_id=30,
                    max_id=0,
                    stop_when_older=False,
                )
            state = _sync_state(conn, "rian_ru")
            self.assertEqual(state.newest, 30)
            self.assertEqual(state.forward_lo, 50)
            self.assertEqual(state.forward_hi, 50)
            saved, _done = await _walk(
                _Client(fresh),
                conn,
                "rian_ru",
                "rian_ru",
                start,
                end_exclusive,
                mode="forward",
                min_id=state.newest or 0,
                max_id=state.forward_lo or 0,
                stop_when_older=False,
            )
            self.assertEqual(saved, 1)
            stored = [row["text"] for row in conn.execute("SELECT text FROM news ORDER BY msg_id")]
            self.assertEqual(stored, ["база", "пропущенная", "самая новая"])
            done_state = _sync_state(conn, "rian_ru")
            self.assertEqual(done_state.newest, 50)
            self.assertIsNone(done_state.forward_lo)
            self.assertIsNone(done_state.forward_hi)
        finally:
            conn.close()

    def test_second_run_loads_only_new_messages(self):
        asyncio.run(self._second_run())

    async def _second_run(self):
        conn = _db()
        try:
            start, end_exclusive = window_bounds("2023-10-12", "2026-10-12")
            client = _Client(_history_messages())
            first = await _collect_channel(client, conn, "rian_ru", start, end_exclusive)
            self.assertEqual(first, 3)
            client.messages.insert(0, _msg(60, datetime(2026, 10, 12, 20, 0, tzinfo=timezone.utc), "догрузка"))
            second = await _collect_channel(client, conn, "rian_ru", start, end_exclusive)
            self.assertEqual(second, 1)
            stored = [row["text"] for row in conn.execute("SELECT text FROM news ORDER BY msg_id")]
            self.assertEqual(
                stored,
                ["на границе", "погода в сочи", "последний день", "догрузка"],
            )
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
