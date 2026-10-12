import unittest

from src.link import classify
from src.predict import direction
from src.universe import load_universe


class LinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = load_universe()

    def tickers(self, text: str) -> set[str]:
        hits, _ = classify(text, self.universe)
        return {hit.ticker for hit in hits}

    def types_for(self, text: str, ticker: str) -> list[str]:
        hits, _ = classify(text, self.universe)
        return [hit.match_type for hit in hits if hit.ticker == ticker]

    def test_sber_alias_does_not_spill(self):
        self.assertEqual(self.tickers("Сбербанк отчитался о прибыли"), {"SBER"})
        self.assertEqual(self.types_for("Сбербанк отчитался о прибыли", "SBER"), ["alias"])

    def test_banks_sector(self):
        self.assertEqual(self.tickers("Банки подняли ставки по вкладам"), {"SBER", "VTBR", "T"})

    def test_oil_sector_and_company(self):
        self.assertEqual(self.tickers("Цены на нефть выросли"), {"GAZP", "LKOH", "ROSN"})
        self.assertEqual(self.tickers("Роснефть увеличила добычу"), {"ROSN"})
        self.assertEqual(self.types_for("Роснефть увеличила добычу", "ROSN"), ["alias"])

    def test_gazpromneft_is_not_gazprom(self):
        self.assertNotIn("GAZP", self.tickers("газпромнефть сократила поставки"))

    def test_gazprom_alias(self):
        self.assertEqual(self.tickers("Газпром увеличил поставки газа"), {"GAZP"})

    def test_yandex_and_vk(self):
        self.assertEqual(self.tickers("Яндекс запустил сервис"), {"YDEX"})
        self.assertIn("VKCO", self.tickers("VK обновил ленту новостей"))

    def test_unrelated(self):
        hits, is_macro = classify("Погода в Сочи отличная", self.universe)
        self.assertEqual(hits, [])
        self.assertFalse(is_macro)

    def test_harvest_maps_to_agriculture(self):
        self.assertEqual(self.tickers("Урожай пшеницы выше плана"), {"GCHE", "AQUA", "PHOR", "RAGR"})

    def test_rate_is_macro_and_banks(self):
        hits, is_macro = classify("Ключевая ставка сохранена", self.universe)
        self.assertTrue(is_macro)
        self.assertEqual({hit.ticker for hit in hits}, {"SBER", "VTBR", "T"})
        self.assertTrue(all(hit.match_type == "sector" for hit in hits))

    def test_false_friends(self):
        self.assertEqual(self.tickers("портфель акций вырос"), set())
        self.assertEqual(self.tickers("банкет в честь юбилея"), set())
        self.assertEqual(self.tickers("сбережения населения выросли"), set())
        self.assertEqual(self.tickers("Португалия увеличила экспорт"), set())

    def test_port_and_sber_stems(self):
        self.assertEqual(self.tickers("в порту выросла перевалка"), {"FESH", "NMTP", "FLOT"})
        self.assertEqual(self.types_for("Сбер повысил ставки", "SBER"), ["alias"])

    def test_nmtp_alias(self):
        self.assertEqual(self.types_for("НМТП увеличил перевалку", "NMTP"), ["alias"])

    def test_direction_bands(self):
        self.assertEqual(direction(0.01), "Рост")
        self.assertEqual(direction(-0.01), "Снижение")
        self.assertEqual(direction(0.0001), "Около нуля")


if __name__ == "__main__":
    unittest.main()
