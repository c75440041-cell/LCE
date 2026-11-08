"""Indicadores equivalentes a las funciones ta.* de Pine Script (numpy puro)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _first_valid(x):
    idx = np.flatnonzero(~np.isnan(x))
    return int(idx[0]) if idx.size else -1


def sma(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).mean(axis=1)
    return out


def _smooth(x, n, alpha):
    """EMA/RMA con semilla SMA (igual que Pine)."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    s = _first_valid(x)
    if s < 0 or len(x) - s < n:
        return out
    out[s + n - 1] = x[s:s + n].mean()
    for i in range(s + n, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema(x, n):
    return _smooth(x, n, 2.0 / (n + 1))


def rma(x, n):
    return _smooth(x, n, 1.0 / n)


def true_range(h, l, c):
    h, l, c = (np.asarray(a, float) for a in (h, l, c))
    pc = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr(h, l, c, n):
    return rma(true_range(h, l, c), n)


def stdev(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).std(axis=1)   # poblacional, como Pine
    return out


def highest(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).max(axis=1)
    return out


def lowest(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).min(axis=1)
    return out


def shift(x, k=1):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def rsi(c, n):
    c = np.asarray(c, float)
    d = np.diff(c, prepend=np.nan)
    up = np.where(np.isnan(d), np.nan, np.maximum(d, 0.0))
    dn = np.where(np.isnan(d), np.nan, np.maximum(-d, 0.0))
    ru, rd = rma(up, n), rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + ru / rd)
    out = np.where((rd == 0) & ~np.isnan(ru), 100.0, out)
    return out


def dmi(h, l, c, n=14):
    h, l, c = (np.asarray(a, float) for a in (h, l, c))
    up = h - np.roll(h, 1)
    dn = np.roll(l, 1) - l
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus[0] = minus[0] = np.nan
    tr = true_range(h, l, c)
    tr[0] = np.nan
    a = rma(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        dip = 100.0 * rma(plus, n) / a
        dim = 100.0 * rma(minus, n) / a
        s = dip + dim
        dx = 100.0 * np.abs(dip - dim) / np.where(s == 0, np.nan, s)
    return dip, dim, rma(dx, n)


def pivot_high(x, left, right):
    """out[i] = valor del pivote (barra i-right) confirmado en la barra i."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for i in range(left + right, len(x)):
        w = x[i - left - right:i + 1]
        if x[i - right] == w.max():
            out[i] = x[i - right]
    return out


def pivot_low(x, left, right):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for i in range(left + right, len(x)):
        w = x[i - left - right:i + 1]
        if x[i - right] == w.min():
            out[i] = x[i - right]
    return out
