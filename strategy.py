"""Motor de señales: CBC (LTF + HTF) + confluencias + plan de trade (SL / TP / R:R)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import MAX_SCORE, Config
from indicators import (NAN, atr_wilder, bucketize, cbc_states, ema, is_nan,
                         rel_volume, trend_efficiency, vwap_daily)
from models import Candle, Signal

DAY_MS = 86_400_000
MIN_BARS = 250  # mínimo de velas cerradas para evaluar (EMA200 / HTF estables)


@dataclass
class Series:
    t: List[int]
    o: List[float]
    h: List[float]
    l: List[float]
    c: List[float]
    v: List[float]
    ema9: List[float]
    ema20: List[float]
    ema200: List[float]
    atr: List[float]
    vwap: List[float]
    rvol: List[float]
    cbc: List[bool]
    htf_state: List[Optional[bool]]
    htf_cloud: List[Optional[bool]]
    pdh: List[float]
    pdl: List[float]
    pdc: List[float]
    er: List[float]  # eficiencia direccional con signo (20 velas), en [-1, 1]


def htf_series(t, o, h, l, c, htf_ms: int, use_forming: bool):
    """Estado CBC y nube EMA 9/20 del HTF visto desde cada vela LTF.

    use_forming=True  -> incluye la vela HTF en formación (igual que lookahead_on del
                         indicador en tiempo real: el HTF puede girar con el cierre LTF actual).
    use_forming=False -> solo la última vela HTF CERRADA (más lento, más conservador).
    Sin mirar al futuro: la vela HTF en formación solo usa el cierre LTF actual.
    """
    n = len(c)
    st: List[Optional[bool]] = [None] * n
    cl: List[Optional[bool]] = [None] * n
    buckets = bucketize(t, o, h, l, c, htf_ms)
    a9, a20 = 2.0 / 10.0, 2.0 / 21.0

    states: List[bool] = []
    e9s: List[float] = []
    e20s: List[float] = []
    state = False
    e9: Optional[float] = None
    e20: Optional[float] = None
    for k, b in enumerate(buckets):
        if k > 0:
            pb = buckets[k - 1]
            if state and b.c < pb.l:
                state = False
            elif (not state) and b.c > pb.h:
                state = True
        states.append(state)
        e9 = b.c if e9 is None else a9 * b.c + (1 - a9) * e9
        e20 = b.c if e20 is None else a20 * b.c + (1 - a20) * e20
        e9s.append(e9)
        e20s.append(e20)

    for k in range(1, len(buckets)):
        b = buckets[k]
        pb = buckets[k - 1]
        for i in range(b.first, b.last + 1):
            s = states[k - 1]
            if use_forming:
                px = c[i]
                if s and px < pb.l:
                    s = False
                elif (not s) and px > pb.h:
                    s = True
                x9 = a9 * px + (1 - a9) * e9s[k - 1]
                x20 = a20 * px + (1 - a20) * e20s[k - 1]
            else:
                x9, x20 = e9s[k - 1], e20s[k - 1]
            st[i] = s
            cl[i] = x9 >= x20
    return st, cl


def prev_day_levels(t, h, l, c, ltf_ms: int):
    """Máximo / mínimo / cierre del día UTC anterior COMPLETO (PDH / PDL / PDC)."""
    n = len(t)
    pdh = [NAN] * n
    pdl = [NAN] * n
    pdc = [NAN] * n
    if n == 0:
        return pdh, pdl, pdc
    first_partial = (t[0] % DAY_MS) >= ltf_ms  # los datos empiezan a mitad de día
    cur_day = None
    d_hi = d_lo = d_cl = 0.0
    days_seen = 0
    prev = None
    for i in range(n):
        d = t[i] // DAY_MS
        if d != cur_day:
            if cur_day is not None:
                if not (days_seen == 0 and first_partial):
                    prev = (d_hi, d_lo, d_cl)
                days_seen += 1
            cur_day = d
            d_hi, d_lo = h[i], l[i]
        else:
            d_hi = max(d_hi, h[i])
            d_lo = min(d_lo, l[i])
        d_cl = c[i]
        if prev is not None:
            pdh[i], pdl[i], pdc[i] = prev
    return pdh, pdl, pdc


def build_series(candles: List[Candle], cfg: Config) -> Series:
    t = [x.t for x in candles]
    o = [x.o for x in candles]
    h = [x.h for x in candles]
    l = [x.l for x in candles]
    c = [x.c for x in candles]
    v = [x.v for x in candles]
    hs, hc = htf_series(t, o, h, l, c, cfg.htf_sec * 1000, cfg.htf_use_forming)
    pdh, pdl, pdc = prev_day_levels(t, h, l, c, cfg.ltf_sec * 1000)
    return Series(
        t=t, o=o, h=h, l=l, c=c, v=v,
        ema9=ema(c, 9), ema20=ema(c, 20), ema200=ema(c, 200),
        atr=atr_wilder(h, l, c, cfg.atr_period),
        vwap=vwap_daily(t, h, l, c, v),
        rvol=rel_volume(v, 20),
        cbc=cbc_states(h, l, c),
        htf_state=hs, htf_cloud=hc,
        pdh=pdh, pdl=pdl, pdc=pdc,
        er=trend_efficiency(c, 20),
    )


def _trigger_flip(S: Series, i: int, cfg: Config):
    """Estrategia ORIGINAL: la señal nace cuando el "acuerdo CBC" (LTF + HTF) empieza."""
    hs, hp = S.htf_state[i], S.htf_state[i - 1]
    if hs is None or hp is None:
        return None, "", False, "nodata"
    ls, lp = S.cbc[i], S.cbc[i - 1]
    long_now, long_prev = (ls and hs), (lp and hp)
    short_now, short_prev = ((not ls) and (not hs)), ((not lp) and (not hp))
    if long_now and not long_prev:
        side = "LONG"
    elif short_now and not short_prev:
        side = "SHORT"
    else:
        return None, "", False, "no_setup"
    flip_now = ls != lp
    trigger = f"Flip CBC {cfg.ltf}" if flip_now else f"Flip CBC {cfg.htf} (HTF)"
    reversal = flip_now and (S.cbc[i - 1] != S.cbc[i - 2])  # doble flip = "CBC Reversal"
    return side, trigger, reversal, "ok"


def _trigger_pullback(S: Series, i: int, cfg: Config):
    """Estrategia PULLBACK (la que pidió el usuario), sobre el LTF (15m):

      CBC VERDE (alcista) + cierra una vela ROJA   -> LONG  (compra el retroceso)
      CBC ROJO  (bajista) + cierra una vela VERDE  -> SHORT (vende el rebote)

    - Color de la vela = cierre vs apertura (roja: close < open, verde: close > open).
    - El CBC debe ser del MISMO color antes y después de cerrar la vela: así la vela es un
      retroceso DENTRO de la tendencia CBC y no la vela que la rompe (si una vela roja
      cerrara por debajo del mínimo anterior, el CBC pasaría a rojo y NO se compra).
    - Opcional (PULLBACK_REQUIRE_HTF): el CBC del HTF debe ir en la misma dirección.
    """
    ls, lp = S.cbc[i], S.cbc[i - 1]
    if ls != lp:
        return None, "", False, "no_setup"  # esta vela cambió el CBC: es un flip, no un retroceso
    if ls and S.c[i] < S.o[i]:
        side = "LONG"
    elif (not ls) and S.c[i] > S.o[i]:
        side = "SHORT"
    else:
        return None, "", False, "no_setup"
    if cfg.pullback_require_htf:
        hs = S.htf_state[i]
        if hs is None:
            return None, "", False, "nodata"
        if hs != ls:
            return None, "", False, "htf_contra"
    if side == "LONG":
        trigger = f"CBC {cfg.ltf} verde + vela roja (retroceso)"
    else:
        trigger = f"CBC {cfg.ltf} rojo + vela verde (rebote)"
    return side, trigger, False, "ok"


def cbc_against(side: str, cbc_state: bool) -> bool:
    """True si el CBC (True = verde/alcista) está en contra de la operación."""
    return (not cbc_state) if side == "LONG" else bool(cbc_state)


# ---------------------------------------------------------------------------------------------
# RANKING: puntuación de calidad 0-100 para elegir las N mejores entradas de cada cierre de vela.
# Cada componente vale 0..1 y se multiplica por su peso (los pesos suman 100). Son heurísticas
# razonadas, NO optimizadas: calibra con `python backtest.py --compare` antes de fiarte de ellas.
# ---------------------------------------------------------------------------------------------
RANK_WEIGHTS = {
    "confluencia": 20,  # EMA cloud + VWAP + EMA200 + EMA cloud HTF a favor (0-4)
    "htf": 12,          # el CBC del HTF va en la misma dirección
    "tendencia": 14,    # tendencia limpia: eficiencia direccional 20v + pendiente de la EMA20
    "retroceso": 14,    # el retroceso llega a "zona de valor" (cerca de la EMA20), no sobreextendido
    "espacio": 10,      # recorrido libre hasta PDH/PDL/PDC (3R o más = máximo)
    "stop": 8,          # stop ajustado (<= 1.0 ATR = máximo, 0 en MAX_SL_ATR): más R por % de movimiento
    "vela": 8,          # vela de señal con rechazo (cierra lejos de su extremo) y cuerpo pequeño
    "liquidez": 8,      # volumen 24h (1M USDT = 0 · 1B USDT = máximo): menos slippage / mechas
    "mercado": 6,       # alineada con el CBC + nube EMA del mercado de referencia (BTC)
}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def market_bias(S: "Series", i: int) -> int:
    """+1 / -1 si el CBC y la nube EMA 9/20 del mercado de referencia (BTC) coinciden; 0 = mixto."""
    if i < 1 or i >= len(S.c):
        return 0
    if S.cbc[i] and S.ema9[i] >= S.ema20[i]:
        return 1
    if (not S.cbc[i]) and S.ema9[i] < S.ema20[i]:
        return -1
    return 0


def rank_parts(S: Series, i: int, is_long: bool, *, score: int, risk: float, atr: float,
               room_r: Optional[float], liq: Optional[float], bias: int, cfg: Config) -> Dict[str, float]:
    """Componentes del ranking (cada uno en 0..1)."""
    sgn = 1.0 if is_long else -1.0
    close = S.c[i]

    hs = S.htf_state[i]
    htf = 0.5 if hs is None else (1.0 if hs == is_long else 0.0)

    er_dir = max(0.0, sgn * S.er[i])
    slope = max(0.0, sgn * (S.ema20[i] - S.ema20[max(0, i - 8)]) / atr)
    tendencia = 0.6 * _clamp01(er_dir / 0.5) + 0.4 * _clamp01(slope / 1.5)

    if cfg.strategy == "pullback":
        d = sgn * (close - S.ema20[i]) / atr  # distancia a la EMA20 en ATR, a favor = positiva
        if d >= 0:
            retroceso = 1.0 if d <= 0.8 else _clamp01(1.0 - (d - 0.8) / 1.7)  # 0 a partir de 2.5 ATR
        else:
            retroceso = _clamp01(1.0 + d / 1.5)  # por debajo de la EMA20: retroceso demasiado profundo
    else:
        retroceso = 0.5  # en la estrategia "flip" no hay retroceso que puntuar

    espacio = 1.0 if room_r is None else _clamp01(room_r / 3.0)

    r_atr = risk / atr
    if r_atr <= 1.0:
        stop = 1.0
    else:
        stop = _clamp01(1.0 - (r_atr - 1.0) / max(cfg.max_sl_atr - 1.0, 0.1))

    rng = S.h[i] - S.l[i]
    if rng > 0:
        pos = (close - S.l[i]) / rng if is_long else (S.h[i] - close) / rng
    else:
        pos = 0.5
    body = abs(S.c[i] - S.o[i])
    vela = 0.6 * _clamp01(pos) + 0.4 * (1.0 - _clamp01(body / (1.5 * atr)))

    liquidez = 0.5 if (not liq or liq <= 0) else _clamp01(math.log10(liq / 1e6) / 3.0)
    mercado = 0.5 if bias == 0 else (1.0 if (bias > 0) == is_long else 0.0)

    return {
        "confluencia": score / MAX_SCORE, "htf": htf, "tendencia": tendencia,
        "retroceso": retroceso, "espacio": espacio, "stop": stop, "vela": vela,
        "liquidez": liquidez, "mercado": mercado,
    }


def select_top(cands: List[Signal], cfg: Config, mode: str = "rank",
               rng=None) -> Tuple[List[Signal], int, int]:
    """Elige las mejores señales de UN cierre de vela. Devuelve (elegidas, bajo_ranking_mínimo, recortadas).

    mode="rank"   -> las TOP_N de mayor ranking (lo que hace el bot en vivo)
    mode="random" -> TOP_N al azar entre las que superan MIN_RANK (línea base para el backtest)
    mode="all"    -> todas las que superan MIN_RANK, sin límite (línea base para el backtest)
    """
    pool = [s for s in cands if s.rank >= cfg.min_rank]
    low = len(cands) - len(pool)
    if mode == "random":
        (rng if rng is not None else random).shuffle(pool)
    else:
        pool.sort(key=lambda s: (s.rank, s.score, s.rvol), reverse=True)
    if mode == "all" or cfg.top_n <= 0:
        return pool, low, 0
    chosen = pool[:cfg.top_n]
    return chosen, low, len(pool) - len(chosen)


def evaluate(S: Series, i: int, cfg: Config, symbol: str, liq: Optional[float] = None,
             bias: int = 0) -> Tuple[Optional[Signal], str]:
    """Evalúa la vela CERRADA i. Devuelve (señal | None, motivo_de_rechazo).

    liq  = volumen 24h en USDT del par (solo para el ranking). bias = +1/-1/0 del mercado de referencia.
    El ranking NO descarta nada aquí: solo ordena (el corte lo hace `select_top`)."""
    if i < 2 or i >= len(S.c):
        return None, "nodata"
    trig = _trigger_pullback if cfg.strategy == "pullback" else _trigger_flip
    side, trigger, reversal, why = trig(S, i, cfg)
    if side is None:
        return None, why
    is_long = side == "LONG"

    atr = S.atr[i]
    if is_nan(atr) or atr <= 0:
        return None, "nodata"
    close = S.c[i]

    htf_cloud = bool(S.htf_cloud[i])
    conds = {
        "ema_cloud": (S.ema9[i] >= S.ema20[i]) if is_long else (S.ema9[i] < S.ema20[i]),
        "vwap": (close > S.vwap[i]) if is_long else (close < S.vwap[i]),
        "ema200": (close > S.ema200[i]) if is_long else (close < S.ema200[i]),
        "htf_cloud": htf_cloud if is_long else (not htf_cloud),
    }
    score = sum(1 for x in conds.values() if x)
    if score < cfg.min_score:
        return None, "score"

    # ---- Plan de trade -------------------------------------------------------------
    # Stop = nivel de flip CBC contrario (extremo de la vela de señal) + colchón ATR,
    # acotado entre MIN_SL_ATR y MAX_SL_ATR veces el ATR.
    # En pullback es justo la invalidez: si la vela siguiente cierra más allá del extremo de
    # la vela de señal, el CBC cambia de color y la idea (CBC verde + compra) deja de valer.
    entry = close
    buf = cfg.sl_buffer_atr * atr
    risk = (entry - (S.l[i] - buf)) if is_long else ((S.h[i] + buf) - entry)
    min_risk, max_risk = cfg.min_sl_atr * atr, cfg.max_sl_atr * atr
    if risk > max_risk:
        if cfg.skip_wide_sl:
            return None, "sl_wide"
        risk = max_risk
    if risk < min_risk:
        risk = min_risk
    sign = 1.0 if is_long else -1.0
    sl = entry - sign * risk
    tps = [entry + sign * r * risk for r in cfg.tp_r]

    # Espacio libre hasta el siguiente nivel clave (PDH / PDL / PDC del día UTC anterior)
    levels = [(n, p) for n, p in (("PDH", S.pdh[i]), ("PDL", S.pdl[i]), ("PDC", S.pdc[i])) if not is_nan(p)]
    if is_long:
        cands = [(n, p) for n, p in levels if p > entry]
        nearest = min(cands, key=lambda x: x[1]) if cands else None
    else:
        cands = [(n, p) for n, p in levels if p < entry]
        nearest = max(cands, key=lambda x: x[1]) if cands else None
    room_r = abs(nearest[1] - entry) / risk if nearest else None
    if room_r is not None and room_r < cfg.min_room_r:
        return None, "room"

    parts = rank_parts(S, i, is_long, score=score, risk=risk, atr=atr, room_r=room_r,
                       liq=liq, bias=bias, cfg=cfg)
    rank = round(sum(RANK_WEIGHTS[k] * v for k, v in parts.items()), 1)

    rv = S.rvol[i]
    return Signal(
        symbol=symbol, side=side, t=S.t[i], entry=entry, sl=sl, tps=tps, risk=risk,
        score=score, max_score=MAX_SCORE, conds=conds, trigger=trigger, reversal=reversal,
        atr=atr, rvol=(0.0 if is_nan(rv) else rv), room_r=room_r,
        level_name=nearest[0] if nearest else None,
        level_price=nearest[1] if nearest else None,
        pdh=None if is_nan(S.pdh[i]) else S.pdh[i],
        pdl=None if is_nan(S.pdl[i]) else S.pdl[i],
        ltf=cfg.ltf, htf=cfg.htf,
        rank=rank, rank_parts={k: int(round(100 * v)) for k, v in parts.items()},
    ), "ok"
