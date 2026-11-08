import hashlib, hmac, os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("STATE_DIR", "/tmp/lce_test_state")

import indicators as ind
import backtest
from bingx import sign_query
from config import Config
from risk import RiskGuard, fmt_num, round_down, size_position
from state import State
from strategy import Signal


# ---------- firma y formato ----------
def test_sign_query_ordena_y_firma_el_mismo_string():
    qs = sign_query({"b": 2, "a": 1}, "secreto")
    base, sig = qs.rsplit("&signature=", 1)
    assert base == "a=1&b=2"
    assert sig == hmac.new(b"secreto", base.encode(), hashlib.sha256).hexdigest()


def test_fmt_num_sin_notacion_cientifica():
    assert fmt_num(0.00000123, 8) == "0.00000123"
    assert "e" not in fmt_num(1e-7, 10).lower()


def test_round_down_no_redondea_hacia_arriba():
    assert round_down(1.999, 2) == 1.99


# ---------- tamaño de posición ----------
def test_size_position_arriesga_el_porcentaje():
    r = size_position(1000, 100, 98, 1.0, 10, 3)       # arriesga 10 USDT, SL a 2
    assert r["qty"] == pytest.approx(5.0)
    assert r["risk_usdt"] == pytest.approx(10.0)


def test_size_position_respeta_tope_de_apalancamiento():
    r = size_position(1000, 100, 99.9, 1.0, 3, 3)      # SL muy cerca -> toparía
    assert r["notional"] <= 3000 + 1e-6


def test_size_position_rechaza_minimos_y_datos_malos():
    assert size_position(0, 100, 98, 1, 5, 3) is None
    assert size_position(1000, 100, 100, 1, 5, 3) is None
    assert size_position(10, 100, 98, 1, 5, 3, min_usdt=50) is None


# ---------- estado ----------
def test_state_solo_acepta_dict_y_persiste(tmp_path):
    s = State(str(tmp_path))
    with pytest.raises(TypeError):
        s.put("x", [1])
    s.put("trades", {"A": 1})
    assert State(str(tmp_path)).get("trades") == {"A": 1}


# ---------- límites de riesgo ----------
def _guard(tmp_path, **env):
    for k, v in env.items():
        os.environ[k] = str(v)
    try:
        cfg = Config()
    finally:
        for k in env:
            del os.environ[k]
    return RiskGuard(cfg, State(str(tmp_path)))


def test_riskguard_limite_operaciones_diarias(tmp_path):
    g = _guard(tmp_path, MAX_TRADES_DAY=2)
    assert g.can_open(1000)[0]
    g.on_open(); g.on_open()
    assert not g.can_open(1000)[0]


def test_riskguard_perdida_diaria_y_pausa(tmp_path):
    g = _guard(tmp_path, MAX_DAY_LOSS=3, MAX_CONS_LOSS=2)
    assert g.can_open(1000)[0]
    g.on_close(-31)                                   # -3.1% del día
    assert not g.can_open(1000)[0]
    g2 = _guard(tmp_path / "b", MAX_CONS_LOSS=2)
    g2.on_close(-1); g2.on_close(-1)
    assert not g2.can_open(1000)[0]


# ---------- cerrojos de live ----------
def test_live_exige_confirmacion_y_limites():
    os.environ.update(TRADE_MODE="live")
    try:
        probs = Config().live_problems()
    finally:
        del os.environ["TRADE_MODE"]
    assert any("CONFIRM_LIVE" in p for p in probs)
    assert any("MAX_DAY_LOSS" in p for p in probs)


def test_bingx_demo_manda_sobre_live():
    os.environ.update(TRADE_MODE="live", BINGX_DEMO="true")
    try:
        assert Config().mode == "demo"
    finally:
        del os.environ["TRADE_MODE"], os.environ["BINGX_DEMO"]


# ---------- indicadores ----------
def test_ema_y_sma_basicos():
    x = np.arange(1.0, 21.0)
    assert ind.sma(x, 5)[-1] == pytest.approx(18.0)
    assert np.isnan(ind.sma(x, 5)[3])
    assert ind.ema(x, 5)[-1] < x[-1]


def test_rsi_en_tendencia_alcista_es_100():
    assert ind.rsi(np.arange(1.0, 60.0), 14)[-1] == pytest.approx(100.0)


def test_pivot_high_detecta_el_maximo():
    x = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1.0])
    ph = ind.pivot_high(x, 2, 2)
    assert ph[5] == 9.0                                 # confirmado 2 velas después


# ---------- backtest ----------
def _d(highs, lows, closes=None):
    n = len(highs)
    closes = closes or [(h + l) / 2 for h, l in zip(highs, lows)]
    return {"t": np.arange(n) * 300_000, "h": np.array(highs, float), "l": np.array(lows, float),
            "c": np.array(closes, float), "o": np.array(closes, float), "v": np.ones(n)}


CFG = SimpleNamespace(max_bars=0, risk_pct=1.0, max_lev=5.0)


def test_backtest_long_toma_tp():
    d = _d([100, 100, 103], [100, 100, 100])
    s = Signal(0, 0, "LONG", "BO", 100, 98, 102, 2.0, 1.0)
    tr = backtest.simulate(CFG, d, [s], 0.0, 0.0)
    assert tr[0]["why"] == "TP" and tr[0]["r"] == pytest.approx(1.0)


def test_backtest_sl_gana_si_coinciden_en_la_misma_vela():
    d = _d([100, 105], [100, 97])
    s = Signal(0, 0, "LONG", "RV", 100, 98, 102, 1.0, 1.0)
    tr = backtest.simulate(CFG, d, [s], 0.0, 0.0)
    assert tr[0]["why"] == "SL" and tr[0]["r"] == pytest.approx(-1.0)


def test_backtest_short_y_comisiones_restan():
    d = _d([100, 100, 100], [100, 97, 97])
    s = Signal(0, 0, "SHORT", "RV", 100, 102, 98, 1.0, 1.0)
    sin = backtest.simulate(CFG, d, [s], 0.0, 0.0)[0]["r"]
    con = backtest.simulate(CFG, d, [s], 0.05, 0.0)[0]["r"]
    assert sin == pytest.approx(1.0) and con < sin


def test_backtest_no_solapa_operaciones_del_mismo_simbolo():
    d = _d([100, 100, 100, 103, 103], [100, 100, 100, 100, 100])
    s1 = Signal(0, 0, "LONG", "BO", 100, 98, 102, 2.0, 1.0)
    s2 = Signal(1, 0, "LONG", "BO", 100, 98, 102, 2.0, 1.0)
    assert len(backtest.simulate(CFG, d, [s1, s2], 0.0, 0.0)) == 1


def test_portfolio_respeta_max_posiciones():
    tr = [{"t_in": 0, "t_out": 10, "ret": 0.01, "r": 1, "kind": "BO", "side": "LONG"},
          {"t_in": 1, "t_out": 10, "ret": 0.01, "r": 1, "kind": "BO", "side": "LONG"},
          {"t_in": 2, "t_out": 10, "ret": 0.01, "r": 1, "kind": "BO", "side": "LONG"}]
    kept, eq, mdd = backtest.portfolio(tr, 2)
    assert len(kept) == 2 and eq == pytest.approx(1.01 ** 2) and mdd == 0


def test_fetch_pagina_hacia_atras():
    class FakeBX:
        def __init__(self): self.calls = []
        def klines(self, sym, itv, limit, end_time=None):
            self.calls.append(end_time)
            top = (end_time // 300_000 + 1) if end_time else 3_000_000   # endTime es inclusivo
            return [(i * 300_000, 1, 1, 1, 1, 1) for i in range(max(0, top - 1000), top)]
    bx = FakeBX()
    d = backtest.fetch(bx, "X-USDT", "5m", 300_000, 2500)
    assert len(d["t"]) == 2500 and len(bx.calls) >= 3
    assert (np.diff(d["t"]) == 300_000).all()
