"""Prueba de integración offline del bucle en vivo (BingX y Telegram simulados)."""
from __future__ import annotations

import asyncio
import re
import unittest
from unittest import mock

from models import Candle, SymbolMeta
from scanner import Scanner
from tracker import Tracker
from test_pullback import LTF, mk_cfg

NOW0 = 1_800_000_000  # segundos (instante arbitrario)


class FakeBingX:
    def __init__(self):
        self.candles = {}

    async def klines(self, sym, interval, limit=500, **kw):
        return list(self.candles[sym])[-limit:]


class FakeTG:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)
        return True


def series_ending_at(boundary_ms, n=300):
    """Tendencia alcista con n velas cerradas; la última cierra en `boundary_ms`."""
    rows = []
    for k in range(n):
        t = boundary_ms - (n - k) * LTF
        o = 100 + 0.5 * k
        c = o + 0.8
        rows.append(Candle(t, o, c + 0.1, o - 0.1, c, 1000.0))
    return rows


class TestLiveLoop(unittest.TestCase):
    def test_signal_then_flip_exit(self):
        cfg = mk_cfg()
        boundary = (NOW0 * 1000 // LTF) * LTF  # cierre de la última vela
        rows = series_ending_at(boundary)
        last = rows[-1]
        # última vela cerrada: ROJA que no pierde el mínimo anterior -> señal LONG
        rows[-1] = Candle(last.t, last.o + 0.8 + 0.0, last.o + 0.8 + 0.1, last.o + 0.5 - 0.1, last.o + 0.5, 1000.0)

        bingx, tg = FakeBingX(), FakeTG()
        bingx.candles["TEST-USDT"] = rows
        tracker = Tracker(cfg)
        sc = Scanner(cfg, bingx, tg, tracker)
        sc.universe = {"TEST-USDT": SymbolMeta("TEST-USDT", price_prec=2)}
        sc._universe_ts = 1e18

        with mock.patch("time.time", return_value=boundary / 1000 + 5):
            asyncio.run(sc.scan_once())
        self.assertEqual(len(tg.sent), 1, tg.sent)
        self.assertIn("LONG · TEST-USDT", tg.sent[0])
        self.assertIsNotNone(tracker.has_open("TEST-USDT"))
        sl = tracker.has_open("TEST-USDT").sl

        # Siguiente vela: cierra por debajo del mínimo anterior sin tocar el SL -> CBC pasa a rojo
        prev = rows[-1]
        c = prev.l - 0.02
        assert c - 0.01 > sl
        nxt = Candle(prev.t + LTF, prev.c, prev.c, c - 0.01, c, 1000.0)
        bingx.candles["TEST-USDT"] = rows + [nxt]
        with mock.patch("time.time", return_value=(boundary + LTF) / 1000 + 5):
            asyncio.run(sc.scan_once())
        self.assertEqual(len(tg.sent), 2, tg.sent)
        self.assertIn("CBC cambió a ROJO", tg.sent[1])
        self.assertIsNone(tracker.has_open("TEST-USDT"))
        self.assertEqual(tracker.stats["n"], 1)

    def test_open_trade_still_tracked_when_symbol_leaves_universe(self):
        cfg = mk_cfg()
        boundary = (NOW0 * 1000 // LTF) * LTF
        rows = series_ending_at(boundary)
        last = rows[-1]
        rows[-1] = Candle(last.t, last.o + 0.8, last.o + 0.9, last.o + 0.4, last.o + 0.5, 1000.0)
        bingx, tg = FakeBingX(), FakeTG()
        bingx.candles["TEST-USDT"] = rows
        bingx.candles["OTRO-USDT"] = series_ending_at(boundary)
        tracker = Tracker(cfg)
        sc = Scanner(cfg, bingx, tg, tracker)
        sc.universe = {"TEST-USDT": SymbolMeta("TEST-USDT", price_prec=2)}
        sc._universe_ts = 1e18
        with mock.patch("time.time", return_value=boundary / 1000 + 5):
            asyncio.run(sc.scan_once())
        self.assertIsNotNone(tracker.has_open("TEST-USDT"))

        # el símbolo sale del universo; la vela siguiente rompe el mínimo (CBC -> rojo)
        sc.universe = {"OTRO-USDT": SymbolMeta("OTRO-USDT", price_prec=2)}
        prev = rows[-1]
        c = prev.l - 0.02
        bingx.candles["TEST-USDT"] = rows + [Candle(prev.t + LTF, prev.c, prev.c, c - 0.01, c, 1000.0)]
        bingx.candles["OTRO-USDT"] = series_ending_at(boundary + LTF)
        with mock.patch("time.time", return_value=(boundary + LTF) / 1000 + 5):
            asyncio.run(sc.scan_once())
        self.assertIsNone(tracker.has_open("TEST-USDT"))
        self.assertTrue(any("CBC cambió a ROJO" in m for m in tg.sent), tg.sent)

    def _red_pullback(self, boundary):
        rows = series_ending_at(boundary)
        last = rows[-1]
        rows[-1] = Candle(last.t, last.o + 0.8, last.o + 0.9, last.o + 0.4, last.o + 0.5, 1000.0)
        return rows

    def _scanner_with_six(self, cfg, boundary):
        """6 pares con la MISMA vela de señal; solo cambia el volumen 24h (=> el ranking)."""
        bingx, tg = FakeBingX(), FakeTG()
        sc = Scanner(cfg, bingx, tg, Tracker(cfg))
        sc.universe = {}
        for k, vol in enumerate([2e6, 5e6, 5e7, 2e8, 5e8, 2e9]):
            sym = f"C{k}-USDT"
            bingx.candles[sym] = self._red_pullback(boundary)
            sc.universe[sym] = SymbolMeta(sym, price_prec=2, quote_volume=vol)
        sc._universe_ts = 1e18
        return sc, tg

    def test_only_top_n_sent_in_rank_order(self):
        cfg = mk_cfg(top_n=4)
        boundary = (NOW0 * 1000 // LTF) * LTF
        sc, tg = self._scanner_with_six(cfg, boundary)
        with mock.patch("time.time", return_value=boundary / 1000 + 5):
            asyncio.run(sc.scan_once())
        # 1 resumen TOP + 4 señales (las 2 menos líquidas / de menor ranking se quedan fuera)
        self.assertEqual(len(tg.sent), 5, tg.sent)
        self.assertIn("TOP 4 entradas", tg.sent[0])
        order = [re.search(r"(?:LONG|SHORT) · (C\d-USDT)", m).group(1) for m in tg.sent[1:]]
        self.assertEqual(order, ["C5-USDT", "C4-USDT", "C3-USDT", "C2-USDT"])
        self.assertIn("Ranking #1 de 4", tg.sent[1])
        self.assertIn("Ranking #4 de 4", tg.sent[4])
        self.assertEqual(sorted(sc.tracker.trades), ["C2-USDT", "C3-USDT", "C4-USDT", "C5-USDT"])

    def test_next_best_are_picked_when_top_already_in_trade(self):
        cfg = mk_cfg(top_n=2)
        boundary = (NOW0 * 1000 // LTF) * LTF
        sc, tg = self._scanner_with_six(cfg, boundary)
        with mock.patch("time.time", return_value=boundary / 1000 + 5):
            asyncio.run(sc.scan_once())
        self.assertEqual(sorted(sc.tracker.trades), ["C4-USDT", "C5-USDT"])
        # misma vela otra vez (p.ej. reinicio): C4/C5 ya están en operación o alertadas -> salen C3 y C2
        sc.tracker.last_alert.clear()
        tg.sent.clear()
        with mock.patch("time.time", return_value=boundary / 1000 + 20):
            asyncio.run(sc.scan_once())
        self.assertEqual(sorted(sc.tracker.trades), ["C2-USDT", "C3-USDT", "C4-USDT", "C5-USDT"])

    def test_min_rank_blocks_everything(self):
        cfg = mk_cfg(top_n=4, min_rank=99.0)
        boundary = (NOW0 * 1000 // LTF) * LTF
        sc, tg = self._scanner_with_six(cfg, boundary)
        with mock.patch("time.time", return_value=boundary / 1000 + 5):
            asyncio.run(sc.scan_once())
        self.assertEqual(tg.sent, [])
        self.assertEqual(sc.tracker.trades, {})

    def test_market_bias_from_reference_symbol(self):
        cfg = mk_cfg(market_bias=True)
        boundary = (NOW0 * 1000 // LTF) * LTF
        sc = Scanner(cfg, FakeBingX(), FakeTG(), Tracker(cfg))
        now_ms = boundary + 5000
        self.assertEqual(sc.market_bias_for(series_ending_at(boundary), now_ms), 1)
        self.assertEqual(sc.market_bias_for(None, now_ms), 0)
        self.assertEqual(sc.market_bias_for(series_ending_at(boundary)[-30:], now_ms), 0)  # pocas velas

    def test_no_signal_on_green_candle(self):
        cfg = mk_cfg()
        boundary = (NOW0 * 1000 // LTF) * LTF
        bingx, tg = FakeBingX(), FakeTG()
        bingx.candles["TEST-USDT"] = series_ending_at(boundary)  # última vela VERDE
        sc = Scanner(cfg, bingx, tg, Tracker(cfg))
        sc.universe = {"TEST-USDT": SymbolMeta("TEST-USDT", price_prec=2)}
        sc._universe_ts = 1e18
        with mock.patch("time.time", return_value=boundary / 1000 + 5):
            asyncio.run(sc.scan_once())
        self.assertEqual(tg.sent, [])


if __name__ == "__main__":
    unittest.main()
