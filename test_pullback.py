"""Pruebas offline (sin red) de la estrategia PULLBACK:

  CBC verde  + vela roja  cerrada -> LONG
  CBC rojo   + vela verde cerrada -> SHORT

Ejecutar:  python -m unittest discover -p "test_*.py"
"""
from __future__ import annotations

import os
import random
import unittest
from collections import Counter
from dataclasses import replace
from unittest import mock

from backtest import backtest_symbol, make_symdata, simulate
from config import load_config
from messages import format_event, format_signal, format_startup, format_top
from models import Candle, SymbolMeta
from strategy import (MIN_BARS, RANK_WEIGHTS, base_of, build_series, cbc_against, evaluate,
                      market_bias, select_top)
from tracker import Tracker, cooldown_active
from universe import build_universe

DAY = 86_400_000
LTF = 900_000  # 15m
T0 = (1_700_000_000_000 // DAY) * DAY  # inicio de un día UTC


def mk_cfg(**over):
    with mock.patch.dict(os.environ, {}, clear=True):
        cfg = load_config()
    # Para probar solo el DISPARADOR: sin filtros de confluencia, espacio, ranking ni cooldown
    base = dict(min_score=0, min_room_r=0.0, state_file="", dry_run=True,
                min_rank=0.0, cooldown_bars=0, market_bias=False)
    base.update(over)
    return replace(cfg, **base)


def candle(k, o, c, wick=0.1):
    return Candle(T0 + k * LTF, o, max(o, c) + wick, min(o, c) - wick, c, 1000.0)


def uptrend(n=300):
    """Velas verdes seguidas; cada cierre rompe el máximo anterior -> CBC verde. ATR ≈ 1."""
    rows = []
    for k in range(n):
        o = 100 + 0.5 * k
        rows.append(candle(k, o, o + 0.8))
    return rows


def downtrend(n=300):
    rows = []
    for k in range(n):
        o = 300 - 0.5 * k
        rows.append(candle(k, o, o - 0.8))
    return rows


def evaluate_last(rows, cfg):
    S = build_series(rows, cfg)
    return evaluate(S, len(rows) - 1, cfg, "TEST-USDT"), S


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = mk_cfg()
        self.assertEqual(cfg.strategy, "pullback")
        self.assertEqual(cfg.ltf, "15m")
        self.assertTrue(cfg.exit_on_cbc_flip)
        self.assertFalse(cfg.pullback_require_htf)

    def test_top_n_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_config()
        self.assertEqual(cfg.top_n, 4)
        self.assertEqual(cfg.min_rank, 50.0)
        self.assertEqual(cfg.cooldown_bars, 4)
        self.assertTrue(cfg.market_bias)
        self.assertEqual(cfg.market_symbol, "BTC-USDT")
        # por defecto analiza TODOS los pares: sin mínimo de volumen, con TradFi y con USDC-M
        self.assertEqual(cfg.min_quote_volume, 0.0)
        self.assertTrue(cfg.include_tradfi and cfg.include_oil)
        self.assertEqual(cfg.quote_assets, ("USDT", "USDC"))
        with mock.patch.dict(os.environ, {"TOP_N": "2", "MIN_RANK": "140"}, clear=True):
            cfg = load_config()
        self.assertEqual((cfg.top_n, cfg.min_rank), (2, 100.0))  # MIN_RANK se acota a 0-100

    def test_invalid_strategy(self):
        with mock.patch.dict(os.environ, {"STRATEGY": "otra"}, clear=True):
            with self.assertRaises(ValueError):
                load_config()


class TestPullbackTrigger(unittest.TestCase):
    def setUp(self):
        self.cfg = mk_cfg()

    def test_cbc_is_green_in_uptrend(self):
        S = build_series(uptrend(), self.cfg)
        self.assertTrue(all(S.cbc[5:]))

    def test_long_on_red_candle_in_green_cbc(self):
        rows = uptrend()
        last = rows[-1]
        # vela ROJA que NO pierde el mínimo anterior -> el CBC sigue verde
        rows.append(candle(len(rows), last.c, last.c - 0.3))
        (sig, why), S = evaluate_last(rows, self.cfg)
        self.assertTrue(S.cbc[-1] and S.cbc[-2])
        self.assertEqual(why, "ok")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "LONG")
        self.assertLess(sig.sl, sig.entry)
        self.assertGreater(sig.tps[0], sig.entry)
        # SL = por debajo del mínimo de la vela de señal (nivel de flip del CBC)
        self.assertLess(sig.sl, rows[-1].l)
        self.assertIn("vela roja", sig.trigger)

    def test_no_long_on_green_candle_in_green_cbc(self):
        (sig, why), _ = evaluate_last(uptrend(), self.cfg)  # la última vela es verde
        self.assertIsNone(sig)
        self.assertEqual(why, "no_setup")

    def test_no_long_if_red_candle_breaks_previous_low(self):
        rows = uptrend()
        last = rows[-1]
        # vela roja que cierra POR DEBAJO del mínimo anterior -> el CBC pasa a rojo (flip)
        rows.append(candle(len(rows), last.c, last.l - 0.5))
        (sig, why), S = evaluate_last(rows, self.cfg)
        self.assertFalse(S.cbc[-1])
        self.assertIsNone(sig)  # tampoco SHORT: la vela es roja, no verde
        self.assertEqual(why, "no_setup")

    def test_short_on_green_candle_in_red_cbc(self):
        rows = downtrend()
        last = rows[-1]
        rows.append(candle(len(rows), last.c, last.c + 0.3))  # vela VERDE sin romper el máximo anterior
        (sig, why), S = evaluate_last(rows, self.cfg)
        self.assertFalse(S.cbc[-1] or S.cbc[-2])
        self.assertEqual(why, "ok")
        self.assertEqual(sig.side, "SHORT")
        self.assertGreater(sig.sl, sig.entry)
        self.assertLess(sig.tps[0], sig.entry)
        self.assertGreater(sig.sl, rows[-1].h)

    def test_no_short_on_red_candle_in_red_cbc(self):
        (sig, why), _ = evaluate_last(downtrend(), self.cfg)
        self.assertIsNone(sig)
        self.assertEqual(why, "no_setup")

    def test_wide_stop_is_skipped(self):
        rows = uptrend()
        last = rows[-1]
        # vela ROJA con mecha inferior larguísima (low muy lejos), pero que cierra sin perder el
        # mínimo anterior -> CBC sigue verde y el stop natural queda > MAX_SL_ATR x ATR
        rows.append(Candle(T0 + len(rows) * LTF, last.c, last.c + 0.05, last.c - 5.0, last.c - 0.3, 1000.0))
        (sig, why), S = evaluate_last(rows, self.cfg)
        self.assertTrue(S.cbc[-1])
        self.assertIsNone(sig)
        self.assertEqual(why, "sl_wide")
        # con SKIP_WIDE_SL=false la señal se acota a MAX_SL_ATR
        cfg = replace(self.cfg, skip_wide_sl=False)
        (sig, why), _ = evaluate_last(rows, cfg)
        self.assertIsNotNone(sig)
        self.assertAlmostEqual(sig.risk, cfg.max_sl_atr * sig.atr)

    def test_min_score_filter(self):
        rows = uptrend()
        last = rows[-1]
        rows.append(candle(len(rows), last.c, last.c - 0.3))
        cfg = replace(self.cfg, min_score=5)  # imposible (máx. 4)
        (sig, why), _ = evaluate_last(rows, cfg)
        self.assertIsNone(sig)
        self.assertEqual(why, "score")

    def test_require_htf(self):
        # tendencia bajista larga + un rebote verde: el CBC HTF (1h) es rojo
        rows = downtrend()
        last = rows[-1]
        rows.append(candle(len(rows), last.c, last.c + 0.3))
        cfg = replace(self.cfg, pullback_require_htf=True)
        (sig, why), _ = evaluate_last(rows, cfg)
        self.assertIsNotNone(sig)  # CBC 15m rojo y CBC 1h rojo: coinciden
        self.assertEqual(sig.side, "SHORT")

    def test_cbc_against(self):
        self.assertTrue(cbc_against("LONG", False))
        self.assertFalse(cbc_against("LONG", True))
        self.assertTrue(cbc_against("SHORT", True))
        self.assertFalse(cbc_against("SHORT", False))

    def test_original_flip_strategy_still_works(self):
        cfg = mk_cfg(strategy="flip")
        # bajista y de golpe sube fuerte: nace el acuerdo CBC alcista
        rows = downtrend(260)
        n = len(rows)
        px = rows[-1].c
        for j in range(30):
            o = px + 1.5 * j
            rows.append(candle(n + j, o, o + 1.5))
        S = build_series(rows, cfg)
        sigs = [evaluate(S, i, cfg, "T")[0] for i in range(n, len(rows))]
        sigs = [s for s in sigs if s is not None]
        self.assertTrue(sigs)
        self.assertEqual(sigs[0].side, "LONG")
        self.assertIn("Flip", sigs[0].trigger)


class TestTrackerFlipExit(unittest.TestCase):
    def _open_long(self, cfg):
        rows = uptrend()
        last = rows[-1]
        rows.append(candle(len(rows), last.c, last.c - 0.3))
        (sig, _), _ = evaluate_last(rows, cfg)
        self.assertIsNotNone(sig)
        tr = Tracker(cfg)
        tr.register(sig, 2)
        return tr, rows, sig

    def test_flip_closes_the_trade_with_FLIP(self):
        cfg = mk_cfg()
        tr, rows, sig = self._open_long(cfg)
        prev_low = rows[-1].l
        # siguiente vela: cierra apenas por debajo del mínimo anterior, sin tocar el SL
        c = prev_low - 0.02
        rows.append(Candle(T0 + len(rows) * LTF, rows[-1].c, rows[-1].c, c - 0.01, c, 1000.0))
        self.assertGreater(c - 0.01, sig.sl)  # el SL no se toca
        S = build_series(rows, cfg)
        self.assertFalse(S.cbc[-1])
        evs = tr.update("TEST-USDT", rows, S.cbc)
        self.assertEqual([e.kind for _, e in evs], ["FLIP"])
        self.assertTrue(evs[0][1].final)
        self.assertLess(evs[0][1].total_r, 0)
        self.assertIsNone(tr.has_open("TEST-USDT"))
        self.assertEqual(tr.stats["n"], 1)
        txt = format_event(evs[0][0], evs[0][1], cfg, tr.stats_line())
        self.assertIn("CBC cambió a ROJO", txt)

    def test_no_flip_exit_when_disabled(self):
        cfg = mk_cfg(exit_on_cbc_flip=False)
        tr, rows, sig = self._open_long(cfg)
        c = rows[-1].l - 0.02
        rows.append(Candle(T0 + len(rows) * LTF, rows[-1].c, rows[-1].c, c - 0.01, c, 1000.0))
        S = build_series(rows, cfg)
        self.assertEqual(tr.update("TEST-USDT", rows, S.cbc), [])
        self.assertIsNotNone(tr.has_open("TEST-USDT"))

    def test_take_profit_then_flip(self):
        cfg = mk_cfg()
        tr, rows, sig = self._open_long(cfg)
        # vela que alcanza TP1 (verde, el CBC sigue verde)
        rows.append(Candle(T0 + len(rows) * LTF, rows[-1].c, sig.tps[0] + 0.1, rows[-1].c - 0.01, sig.tps[0] + 0.05, 1000.0))
        S = build_series(rows, cfg)
        evs = tr.update("TEST-USDT", rows, S.cbc)
        self.assertEqual([e.kind for _, e in evs], ["TP"])
        self.assertAlmostEqual(tr.has_open("TEST-USDT").sl, sig.entry)  # SL a breakeven


class TestBacktestInvariants(unittest.TestCase):
    def test_random_walk(self):
        rnd = random.Random(7)
        rows, px = [], 100.0
        for k in range(1500):
            o = px
            c = o * (1 + rnd.gauss(0, 0.003))
            h = max(o, c) * (1 + abs(rnd.gauss(0, 0.001)))
            l = min(o, c) * (1 - abs(rnd.gauss(0, 0.001)))
            rows.append(Candle(T0 + k * LTF, o, h, l, c, rnd.uniform(500, 1500)))
            px = c
        cfg = mk_cfg(min_score=2, min_room_r=0.0)
        trades = backtest_symbol(rows, cfg, "RW-USDT", 0.0005, 0)
        self.assertGreater(len(trades), 20)
        kinds = {t["kind"] for t in trades}
        self.assertTrue(kinds <= {"TP", "SL", "BE", "FLIP", "OPPOSITE", "TIMEOUT"}, kinds)
        self.assertIn("FLIP", kinds)
        for t in trades:
            self.assertLess(t["r"], t["gross_r"] + 1e-12)  # la comisión siempre resta

        # Invariante de la regla: toda señal LONG = CBC verde (antes y después) + vela roja
        S = build_series(rows, cfg)
        n_sig = 0
        for i in range(MIN_BARS, len(rows)):
            sig, _ = evaluate(S, i, cfg, "RW-USDT")
            if sig is None:
                continue
            n_sig += 1
            red, green = rows[i].c < rows[i].o, rows[i].c > rows[i].o
            if sig.side == "LONG":
                self.assertTrue(S.cbc[i] and S.cbc[i - 1] and red)
            else:
                self.assertTrue((not S.cbc[i]) and (not S.cbc[i - 1]) and green)
        self.assertGreater(n_sig, 20)

    def test_no_lookahead(self):
        """La señal en la vela i no cambia si se añaden velas futuras."""
        rnd = random.Random(11)
        rows, px = [], 100.0
        for k in range(700):
            o = px
            c = o * (1 + rnd.gauss(0, 0.003))
            rows.append(Candle(T0 + k * LTF, o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000.0))
            px = c
        cfg = mk_cfg(min_score=0)
        S_full = build_series(rows, cfg)
        for i in range(300, 600, 7):
            S_cut = build_series(rows[: i + 1], cfg)
            a, _ = evaluate(S_full, i, cfg, "X")
            b, _ = evaluate(S_cut, i, cfg, "X")
            self.assertEqual((a is None), (b is None), f"i={i}")
            if a is not None:
                self.assertEqual((a.side, round(a.sl, 9), round(a.entry, 9)), (b.side, round(b.sl, 9), round(b.entry, 9)))
                self.assertEqual(a.rank, b.rank, f"ranking con lookahead en i={i}")


class TestRuleIndependentCheck(unittest.TestCase):
    """Reimplementa la regla desde cero (sin usar strategy.py) y exige que coincida 1:1."""

    def test_signals_match_independent_rule(self):
        rnd = random.Random(2024)
        rows, px = [], 100.0
        for k in range(2500):
            o = px
            c = o * (1 + rnd.gauss(0, 0.004))
            h = max(o, c) * (1 + abs(rnd.gauss(0, 0.0015)))
            l = min(o, c) * (1 - abs(rnd.gauss(0, 0.0015)))
            rows.append(Candle(T0 + k * LTF, o, h, l, c, 1000.0))
            px = c
        # sin filtros: solo el disparador puro
        cfg = mk_cfg(min_score=0, min_room_r=0.0, skip_wide_sl=False)
        S = build_series(rows, cfg)

        # CBC independiente: verde si cierra > máx. anterior, rojo si cierra < mín. anterior
        state, cbc = False, [False]
        for i in range(1, len(rows)):
            if rows[i].c > rows[i - 1].h:
                state = True
            elif rows[i].c < rows[i - 1].l:
                state = False
            cbc.append(state)

        longs = shorts = 0
        for i in range(MIN_BARS, len(rows)):
            red, green = rows[i].c < rows[i].o, rows[i].c > rows[i].o
            exp = None
            if cbc[i] and cbc[i - 1] and red:
                exp = "LONG"
            elif (not cbc[i]) and (not cbc[i - 1]) and green:
                exp = "SHORT"
            sig, _ = evaluate(S, i, cfg, "X")
            self.assertEqual(sig.side if sig else None, exp, f"vela {i}")
            longs += exp == "LONG"
            shorts += exp == "SHORT"
        self.assertGreater(longs, 50)
        self.assertGreater(shorts, 50)


def walk(seed, n=900, vol=0.003):
    rnd = random.Random(seed)
    rows, px = [], 100.0
    for k in range(n):
        o = px
        c = o * (1 + rnd.gauss(0, vol))
        h = max(o, c) * (1 + abs(rnd.gauss(0, 0.001)))
        l = min(o, c) * (1 - abs(rnd.gauss(0, 0.001)))
        rows.append(Candle(T0 + k * LTF, o, h, l, c, rnd.uniform(500, 1500)))
        px = c
    return rows


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.cfg = mk_cfg()

    def _long_sig(self, **kw):
        rows = uptrend()
        last = rows[-1]
        rows.append(candle(len(rows), last.c, last.c - 0.3))
        S = build_series(rows, self.cfg)
        sig, why = evaluate(S, len(rows) - 1, self.cfg, "T-USDT", **kw)
        self.assertEqual(why, "ok")
        return sig

    def test_weights_sum_100(self):
        self.assertEqual(sum(RANK_WEIGHTS.values()), 100)

    def test_rank_in_range_and_parts(self):
        sig = self._long_sig()
        self.assertTrue(0 <= sig.rank <= 100, sig.rank)
        self.assertEqual(set(sig.rank_parts), set(RANK_WEIGHTS))
        for v in sig.rank_parts.values():
            self.assertTrue(0 <= v <= 100)

    def test_market_bias_moves_rank(self):
        up, neutral, down = (self._long_sig(bias=b) for b in (1, 0, -1))
        self.assertGreater(up.rank, neutral.rank)
        self.assertGreater(neutral.rank, down.rank)
        self.assertAlmostEqual(up.rank - down.rank, RANK_WEIGHTS["mercado"], delta=0.11)

    def test_liquidity_moves_rank(self):
        hi, lo = self._long_sig(liq=1e9), self._long_sig(liq=1e6)
        self.assertAlmostEqual(hi.rank - lo.rank, RANK_WEIGHTS["liquidez"], delta=0.11)
        # liquidez desconocida (0 / None) = neutra, entre ambas
        self.assertLess(lo.rank, self._long_sig().rank)
        self.assertLess(self._long_sig().rank, hi.rank)

    def test_dead_contract_gets_zero_liquidity_but_unknown_is_neutral(self):
        dead, floor, unknown = self._long_sig(liq=0.0), self._long_sig(liq=1e6), self._long_sig(liq=None)
        self.assertEqual(dead.rank, floor.rank)      # 0 y 1M USDT: ambos 0 puntos de liquidez
        self.assertEqual(dead.rank_parts["liquidez"], 0)
        self.assertEqual(unknown.rank_parts["liquidez"], 50)

    def test_short_mirrors_long(self):
        rows = downtrend()
        last = rows[-1]
        rows.append(candle(len(rows), last.c, last.c + 0.3))
        S = build_series(rows, self.cfg)
        sig, why = evaluate(S, len(rows) - 1, self.cfg, "T-USDT", bias=-1)  # mercado bajista = a favor
        self.assertEqual((why, sig.side), ("ok", "SHORT"))
        against, _ = evaluate(S, len(rows) - 1, self.cfg, "T-USDT", bias=1)
        self.assertGreater(sig.rank, against.rank)

    def test_market_bias_function(self):
        S_up = build_series(uptrend(), self.cfg)
        S_dn = build_series(downtrend(), self.cfg)
        self.assertEqual(market_bias(S_up, len(S_up.c) - 1), 1)
        self.assertEqual(market_bias(S_dn, len(S_dn.c) - 1), -1)

    def test_rank_text_in_messages(self):
        sig = self._long_sig(liq=5e8, bias=1)
        txt = format_signal(sig, SymbolMeta("T-USDT", price_prec=2), self.cfg, pos=2, total=4)
        self.assertIn("Ranking #2 de 4", txt)
        self.assertIn(f"{sig.rank:.0f}/100", txt)


class TestSelectTop(unittest.TestCase):
    def _sigs(self, ranks):
        rows = uptrend()
        rows.append(candle(len(rows), rows[-1].c, rows[-1].c - 0.3))
        cfg = mk_cfg()
        sig, _ = evaluate(build_series(rows, cfg), len(rows) - 1, cfg, "X")
        return [replace(sig, symbol=f"S{k}-USDT", rank=float(r)) for k, r in enumerate(ranks)]

    def test_top_n_and_order(self):
        cfg = mk_cfg(top_n=4)
        chosen, low, cut = select_top(self._sigs([40, 90, 70, 55, 80, 60]), cfg)
        self.assertEqual([s.rank for s in chosen], [90, 80, 70, 60])
        self.assertEqual((low, cut), (0, 2))

    def test_min_rank_filter(self):
        cfg = mk_cfg(top_n=4, min_rank=65.0)
        chosen, low, cut = select_top(self._sigs([40, 90, 70, 55]), cfg)
        self.assertEqual([s.rank for s in chosen], [90, 70])
        self.assertEqual((low, cut), (2, 0))

    def test_fewer_than_n_is_not_padded(self):
        chosen, _, _ = select_top(self._sigs([80, 75]), mk_cfg(top_n=4))
        self.assertEqual(len(chosen), 2)

    def test_modes(self):
        sigs = self._sigs([10, 20, 30, 40, 50, 60])
        cfg = mk_cfg(top_n=2, min_rank=25.0)
        allm, low, cut = select_top(sigs, cfg, "all")
        self.assertEqual((len(allm), low, cut), (4, 2, 0))
        rnd, _, _ = select_top(sigs, cfg, "random", random.Random(3))
        self.assertEqual(len(rnd), 2)
        self.assertTrue(all(s.rank >= 25 for s in rnd))
        unlimited, _, _ = select_top(sigs, mk_cfg(top_n=0), "rank")
        self.assertEqual(len(unlimited), 6)

    def test_one_signal_per_underlying(self):
        sigs = self._sigs([80, 70, 60, 50])
        sigs = [replace(sigs[0], symbol="BTC-USDT"), replace(sigs[1], symbol="BTC-USDC"),
                replace(sigs[2], symbol="ETH-USDT"), replace(sigs[3], symbol="ETH-USDC")]
        chosen, _, cut = select_top(sigs, mk_cfg(top_n=4))
        self.assertEqual([s.symbol for s in chosen], ["BTC-USDT", "ETH-USDT"])  # la mejor de cada subyacente
        self.assertEqual(cut, 2)
        self.assertEqual(base_of("1000PEPE-USDT"), "1000PEPE")

    def test_busy_bases_are_skipped(self):
        sigs = self._sigs([80, 70, 60])
        sigs = [replace(sigs[0], symbol="BTC-USDC"), replace(sigs[1], symbol="SOL-USDT"),
                replace(sigs[2], symbol="XRP-USDT")]
        chosen, low, _ = select_top(sigs, mk_cfg(top_n=2), busy_bases={"BTC"})  # BTC ya está en operación
        self.assertEqual([s.symbol for s in chosen], ["SOL-USDT", "XRP-USDT"])
        self.assertEqual(low, 0)

    def test_top_summary_message(self):
        cfg = mk_cfg(top_n=4)
        chosen, _, _ = select_top(self._sigs([90, 80, 70]), cfg)
        txt = format_top(chosen, cfg, n_valid=7, n_eval=300)
        self.assertIn("TOP 3 entradas", txt)
        self.assertIn("1. 🟢", txt)
        self.assertIn("7 señales válidas entre 300 pares", txt)


class TestUniverseAllSymbols(unittest.TestCase):
    CONTRACTS = [
        {"symbol": "BTC-USDT", "status": 1}, {"symbol": "BTC-USDC", "status": 1},
        {"symbol": "TINY-USDT", "status": 1},                       # sin volumen
        {"symbol": "DEAD-USDT", "status": 0},                       # fuera de línea
        {"symbol": "NCCOOILWTI2USD-USDT", "status": 1},             # petróleo
        {"symbol": "NCSKTSLA2USD-USDT", "status": 1},               # acción (TradFi)
        {"symbol": "NCCOGOLD2USD-USDT", "status": 1},               # materia prima (TradFi)
        {"symbol": "BTC-USD", "status": 1},                         # coin-M: otra API, no se admite
    ]
    VOLUMES = {"BTC-USDT": 3e8, "BTC-USDC": 1e8, "TINY-USDT": 2e4}

    def test_default_includes_everything_tradable(self):
        uni = build_universe(self.CONTRACTS, self.VOLUMES, mk_cfg())
        self.assertEqual(set(uni), {"BTC-USDT", "BTC-USDC", "TINY-USDT", "NCCOOILWTI2USD-USDT",
                                    "NCSKTSLA2USD-USDT", "NCCOGOLD2USD-USDT"})
        self.assertEqual(uni["NCCOOILWTI2USD-USDT"].kind, "oil")
        self.assertEqual(uni["NCSKTSLA2USD-USDT"].kind, "tradfi")

    def test_filters_still_work_when_asked(self):
        uni = build_universe(self.CONTRACTS, self.VOLUMES, mk_cfg(min_quote_volume=1e6, include_tradfi=False))
        self.assertNotIn("TINY-USDT", uni)
        self.assertNotIn("NCSKTSLA2USD-USDT", uni)
        self.assertIn("NCCOOILWTI2USD-USDT", uni)
        uni = build_universe(self.CONTRACTS, self.VOLUMES, mk_cfg(quote_assets=("USDT",)))
        self.assertNotIn("BTC-USDC", uni)


class TestCooldown(unittest.TestCase):
    def test_cooldown_window(self):
        X = 100 * LTF  # fin de la vela en la que se cerró la operación
        self.assertFalse(cooldown_active(None, X, LTF, 4))
        for k in range(4):  # velas que abren en X, X+1, X+2, X+3 -> bloqueadas
            self.assertTrue(cooldown_active(X, X + k * LTF, LTF, 4), k)
        self.assertFalse(cooldown_active(X, X + 4 * LTF, LTF, 4))
        self.assertFalse(cooldown_active(X, X, LTF, 0))  # 0 = desactivado

    def _open_long(self, cfg):
        rows = uptrend()
        rows.append(candle(len(rows), rows[-1].c, rows[-1].c - 0.3))
        (sig, _), _ = evaluate_last(rows, cfg)
        tr = Tracker(cfg)
        tr.register(sig, 2)
        return tr, sig

    def test_tracker_cooldown_only_after_non_win(self):
        cfg = mk_cfg(cooldown_bars=4)
        tr, sig = self._open_long(cfg)
        end = sig.t + 3 * LTF
        tr.close_opposite("TEST-USDT", sig.entry - sig.risk, end_ms=end)  # -1R
        self.assertTrue(tr.in_cooldown("TEST-USDT", end))
        self.assertFalse(tr.in_cooldown("TEST-USDT", end + 4 * LTF))
        tr2, sig2 = self._open_long(cfg)
        tr2.close_opposite("TEST-USDT", sig2.entry + 2 * sig2.risk, end_ms=end)  # +2R
        self.assertFalse(tr2.in_cooldown("TEST-USDT", end))


class TestPortfolioSimulation(unittest.TestCase):
    def setUp(self):
        self.cfg = mk_cfg(min_score=0, top_n=2)
        self.data = {f"S{k}-USDT": make_symdata(f"S{k}-USDT", walk(100 + k), self.cfg, 1e7 * (k + 1))
                     for k in range(6)}

    def test_never_more_than_top_n_per_candle(self):
        trades, info = simulate(self.data, self.cfg, 0.0005, 0, mode="rank")
        self.assertGreater(len(trades), 20)
        self.assertLessEqual(max(Counter(t["t"] for t in trades).values()), 2)
        self.assertGreater(info["recortadas"], 0)
        self.assertGreater(info["candidatas"], info["elegidas"])

    def test_all_mode_takes_more_trades(self):
        rank, _ = simulate(self.data, self.cfg, 0.0005, 0, mode="rank")
        allm, _ = simulate(self.data, self.cfg, 0.0005, 0, mode="all")
        self.assertGreater(len(allm), len(rank))

    def test_random_mode_is_reproducible(self):
        a, _ = simulate(self.data, self.cfg, 0.0005, 0, mode="random", seed=5)
        b, _ = simulate(self.data, self.cfg, 0.0005, 0, mode="random", seed=5)
        self.assertEqual([(t["symbol"], t["t"], round(t["r"], 9)) for t in a],
                         [(t["symbol"], t["t"], round(t["r"], 9)) for t in b])

    def test_cooldown_reduces_reentries(self):
        free, _ = simulate(self.data, replace(self.cfg, top_n=0), 0.0005, 0, mode="all")
        waits, _ = simulate(self.data, replace(self.cfg, top_n=0, cooldown_bars=8), 0.0005, 0, mode="all")
        self.assertLess(len(waits), len(free))

    def test_market_bias_changes_ranks(self):
        cfg = replace(self.cfg, market_bias=True)
        market = make_symdata("BTC-USDT", walk(999), cfg)
        t_off, _ = simulate(self.data, replace(cfg, market_bias=False), 0.0005, 0, "all", market=market)
        t_on, _ = simulate(self.data, cfg, 0.0005, 0, "all", market=market)
        self.assertNotEqual([t["rank"] for t in t_off], [t["rank"] for t in t_on])


class TestMessages(unittest.TestCase):
    def test_signal_text_long_and_short(self):
        cfg = mk_cfg(min_score=0)
        meta = SymbolMeta("TEST-USDT", price_prec=2)
        rows = uptrend()
        rows.append(candle(len(rows), rows[-1].c, rows[-1].c - 0.3))
        (sig, _), _ = evaluate_last(rows, cfg)
        txt = format_signal(sig, meta, cfg)
        self.assertIn("LONG · TEST-USDT", txt)
        self.assertIn("CBC 15m VERDE", txt)
        self.assertIn("cambia a ROJO", txt)

        rows = downtrend()
        rows.append(candle(len(rows), rows[-1].c, rows[-1].c + 0.3))
        (sig, _), _ = evaluate_last(rows, cfg)
        txt = format_signal(sig, meta, cfg)
        self.assertIn("SHORT · TEST-USDT", txt)
        self.assertIn("CBC 15m ROJO", txt)
        self.assertIn("cambia a VERDE", txt)

    def test_startup(self):
        txt = format_startup(mk_cfg(), 100, [], 0)
        self.assertIn("compra velas rojas", txt)


if __name__ == "__main__":
    unittest.main()
