"""Indicadores y lógica CBC portados del Pine Script "MapleStax CBC v2.00".

Todo es CAUSAL (cada valor solo usa datos hasta esa vela), así que el mismo código
sirve para el bot en vivo y para el backtest sin sesgo de mirar al futuro.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

NAN = float("nan")


def ema(values: Sequence[float], period: int) -> List[float]:
    """EMA estilo Pine ta.ema (semilla = primer valor)."""
    alpha = 2.0 / (period + 1.0)
    out: List[float] = []
    prev = None
    for x in values:
        prev = x if prev is None else alpha * x + (1.0 - alpha) * prev
        out.append(prev)
    return out


def atr_wilder(h: Sequence[float], l: Sequence[float], c: Sequence[float], period: int = 14) -> List[float]:
    """ATR con suavizado de Wilder (ta.atr / RMA)."""
    n = len(c)
    out = [NAN] * n
    if n < period or n == 0:
        return out
    tr = [h[0] - l[0]]
    for i in range(1, n):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    a = sum(tr[:period]) / period
    out[period - 1] = a
    for i in range(period, n):
        a = (a * (period - 1) + tr[i]) / period
        out[i] = a
    return out


def vwap_daily(t: Sequence[int], h: Sequence[float], l: Sequence[float],
               c: Sequence[float], v: Sequence[float]) -> List[float]:
    """VWAP de hlc3 reiniciado cada día UTC (equivale a ta.vwap(hlc3) en mercado 24/7)."""
    out: List[float] = []
    day = None
    pv = vv = tp_sum = 0.0
    cnt = 0
    for i in range(len(t)):
        d = t[i] // 86_400_000
        if d != day:
            day = d
            pv = vv = tp_sum = 0.0
            cnt = 0
        tp = (h[i] + l[i] + c[i]) / 3.0
        pv += tp * v[i]
        vv += v[i]
        tp_sum += tp
        cnt += 1
        out.append(pv / vv if vv > 0 else tp_sum / cnt)
    return out


def cbc_states(h: Sequence[float], l: Sequence[float], c: Sequence[float]) -> List[bool]:
    """Máquina de estados CBC (idéntica a f_cbc_state del Pine, rupturas estrictas).

    Alcista (True)  -> pasa a bajista si close < low[1]
    Bajista (False) -> pasa a alcista si close > high[1]
    """
    n = len(c)
    if n == 0:
        return []
    state = False
    out = [False]
    for i in range(1, n):
        if state and c[i] < l[i - 1]:
            state = False
        elif (not state) and c[i] > h[i - 1]:
            state = True
        out.append(state)
    return out


def rel_volume(v: Sequence[float], n: int = 20) -> List[float]:
    """Volumen relativo = volumen / media de las n velas ANTERIORES."""
    out = [NAN] * len(v)
    for i in range(n, len(v)):
        avg = sum(v[i - n:i]) / n
        out[i] = v[i] / avg if avg > 0 else NAN
    return out


def trend_efficiency(c: Sequence[float], n: int = 20) -> List[float]:
    """Eficiencia direccional (Kaufman) CON SIGNO, en [-1, 1]. Causal.

    (c[i] - c[i-n]) / suma(|Δc|) de las últimas n velas.
    ~0 = lateral / ruido · +1 = subida limpia · -1 = bajada limpia. Antes de n velas devuelve 0.
    """
    out = [0.0] * len(c)
    if n < 1 or len(c) <= n:
        return out
    diffs = [0.0] + [abs(c[k] - c[k - 1]) for k in range(1, len(c))]
    for i in range(n, len(c)):
        path = sum(diffs[i - n + 1:i + 1])
        out[i] = (c[i] - c[i - n]) / path if path > 1e-12 else 0.0
    return out


@dataclass
class Bucket:
    key: int
    o: float
    h: float
    l: float
    c: float
    first: int
    last: int


def bucketize(t: Sequence[int], o: Sequence[float], h: Sequence[float], l: Sequence[float],
              c: Sequence[float], bucket_ms: int) -> List[Bucket]:
    """Agrupa velas LTF en velas HTF alineadas a la época UTC (igual que BingX)."""
    out: List[Bucket] = []
    for i in range(len(t)):
        k = t[i] // bucket_ms
        if out and out[-1].key == k:
            b = out[-1]
            b.h = max(b.h, h[i])
            b.l = min(b.l, l[i])
            b.c = c[i]
            b.last = i
        else:
            out.append(Bucket(k, o[i], h[i], l[i], c[i], i, i))
    return out


def is_nan(x: float) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
