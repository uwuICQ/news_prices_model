import unittest
from datetime import datetime

from src.collect_public import _halves, parse_ria_page, plain_text, published_from_ria, search_queries
from src.universe import load_universe


SAMPLE = """
<div class="list-items-loaded" data-count="2">
<a href="https://ria.ru/20231012/izrail-1902404413.html" class="list-item__title color-font-hover-only">США пересмотрят решение</a>
</div><div class="list-item__info"><div class="list-item__info-item" data-type="date">12 октября 2023, 23:56</div>
<a href="https://ria.ru/20231012/izrail-1902404413.html" class="list-item__title color-font-hover-only">США пересмотрят решение</a>
</div><div class="list-item__info"><div class="list-item__info-item" data-type="date">12 октября 2023, 23:56</div>
"""


class PublicParseTests(unittest.TestCase):
    def test_ria_item_keeps_moscow_time_and_drops_duplicate(self):
        items, total = parse_ria_page(SAMPLE)
        self.assertEqual(total, 2)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["msg_id"], 1902404413)
        self.assertEqual(items[0]["text"], "США пересмотрят решение")
        self.assertEqual(items[0]["published"], datetime(2023, 10, 12, 23, 56, tzinfo=items[0]["published"].tzinfo))
        self.assertEqual(published_from_ria("20231012", "без времени"), datetime(2023, 10, 12, 12, 0, tzinfo=items[0]["published"].tzinfo))

    def test_plain_text_strips_markup(self):
        self.assertEqual(plain_text("<p>Сбербанк повысил <b>ставку</b></p>"), "Сбербанк повысил ставку")

    def test_queries_skip_broad_stems(self):
        queries = search_queries(load_universe())
        self.assertIn("сбербанк", queries)
        self.assertIn("ключевая ставка", queries)
        self.assertIn("нефть", queries)
        self.assertIn("санкции", queries)
        self.assertNotIn("нефт", queries)
        self.assertNotIn("санкц", queries)
        self.assertNotIn("банк", queries)
        self.assertNotIn("порт", queries)
        self.assertNotIn("vk", queries)

    def test_search_window_splits_when_ria_caps_results(self):
        halves = _halves("2023-10-12", "2026-10-17")
        self.assertIsNotNone(halves)
        left, right = halves
        self.assertEqual(left[0], "2023-10-12")
        self.assertLess(left[1], right[0])
        self.assertEqual(right[1], "2026-10-17")
        self.assertIsNone(_halves("2024-01-01", "2024-01-01"))


if __name__ == "__main__":
    unittest.main()
