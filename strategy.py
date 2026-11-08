"""Port a Python de la estrategia Pine LCE v8 (nucleo LCE v2 + modulos).

`run()` repasa TODAS las velas cerradas en orden, igual que Pine en historico:
zonas, estructura, divergencias, kernel y cooldown se reconstruyen en cada ciclo.
Ademas simula la operacion virtual (SL/TP/parcial/BE/trailing/tiempo/estructura)
porque en Pine `canSignal` exige `position_size == 0`: mientras hay operacion
abierta no hay senales nuevas. Solo se opera la senal de la ULTIMA vela cerrada.

Limites que NO se replican en el historico (dependen del equity real): perdida
diaria, maximo de operaciones por dia y pausa por perdidas seguidas. El bot los
aplica en vivo a nivel de cartera (risk.RiskGuard).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import indicators as ind


@dataclass
class Signal:
    idx: int
    time: int
    side: str            # LONG | SHORT
    kind: str            # BO | RV | SW
    entry: float
    sl: float
    tp: float            # objetivo final (curTP2): nivel, o entry +- rrTrailCap*R si hay trailing
    rr: float            # R:R hasta el nivel (rrL / rrS de Pine)
    atr: float
    tp1: Optional[float] = None
    risk_mult: float = 1.0
    score: int = 0
    htf: str = "n/d"
    rel_vol: float = 0.0
    adx: float = 0.0


@dataclass
class Analysis:
    signals: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)
    last_time: int = 0
    last_close: float = float("nan")
    struct_break_up: bool = False
    struct_break_dn: bool = False
    hh3: float = float("nan")
    ll3: float = float("nan")
    atr: float = float("nan")
    n: int = 0
    counts: dict = field(default_factory=dict)


def map_htf(cfg, t, htf):
    """EMA del TF superior de la ultima vela CERRADA (equivale a [1] + lookahead_on)."""
    n = len(t)
    f = np.full(n, np.nan)
    s = np.full(n, np.nan)
    if htf is None or len(htf["t"]) < cfg.ema_slow + 2:
        return f, s
    ef = ind.ema(htf["c"], cfg.ema_fast)
    es = ind.ema(htf["c"], cfg.ema_slow)
    starts = (np.asarray(t, np.int64) // cfg.htf_ms) * cfg.htf_ms
    idx = np.searchsorted(np.asarray(htf["t"], np.int64), starts, side="left") - 1
    ok = idx >= 0
    f[ok] = ef[idx[ok]]
    s[ok] = es[idx[ok]]
    return f, s


def prev_day_levels(t, h, l):
    t = np.asarray(t, np.int64)
    day = t // 86_400_000
    hi, lo = {}, {}
    for d in np.unique(day):
        m = day == d
        hi[int(d)] = float(np.max(h[m]))
        lo[int(d)] = float(np.min(l[m]))
    pdh = np.array([hi.get(int(d) - 1, np.nan) for d in day])
    pdl = np.array([lo.get(int(d) - 1, np.nan) for d in day])
    return pdh, pdl


def _norm_cdf(z):
    """Misma aproximacion logistica que el Pine: 1 / (1 + exp(-1.702 z))."""
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-1.702 * z))


def kde_prob(samples, x, bw_mult=1.0):
    """Probabilidad (0-1) segun el KDE gaussiano, ancho de banda de Silverman (stdev poblacional)."""
    n = len(samples)
    if n < 2:
        return None
    s = np.asarray(samples, float)
    sd = float(s.std())
    h = max(1.06 * sd * n ** (-0.2) * bw_mult, 0.0001)
    return sum(_norm_cdf((x - v) / h) for v in s) / n


def _in_window(spec, t_ms):
    a, b = spec.split("-")
    s, e = int(a[:2]) * 60 + int(a[2:]), int(b[:2]) * 60 + int(b[2:])
    m = (int(t_ms) // 60_000) % 1440
    return (s <= m < e) if s <= e else (m >= s or m < e)


def _in_session(cfg, t_ms):
    return (not cfg.use_session) or _in_window(cfg.session_utc, t_ms)


def run(cfg, d, htf=None, mintick=1e-9, count_from=0) -> Analysis:
    t = np.asarray(d["t"], np.int64)
    o, h, l, c, v = (np.asarray(d[k], float) for k in ("o", "h", "l", "c", "v"))
    n = len(c)
    an = Analysis(n=n)

    def bump(k):
        an.counts[k] = an.counts.get(k, 0) + 1

    if n < 60:
        an.diag = {"reason": "pocas_velas"}
        return an

    atr = ind.atr(h, l, c, cfg.atr_len)
    ef, es = ind.ema(c, cfg.ema_fast), ind.ema(c, cfg.ema_slow)
    volavg = ind.sma(v, 20)
    e200 = ind.ema(c, cfg.ema200_len)
    _, _, adx = ind.dmi(h, l, c, 14)
    rsi = ind.rsi(c, cfg.rsi_len)
    hh3, ll3 = ind.highest(h, 3), ind.lowest(l, 3)
    ph_z, pl_z = ind.pivot_high(h, cfg.piv_len, cfg.piv_len), ind.pivot_low(l, cfg.piv_len, cfg.piv_len)
    ph_d, pl_d = ind.pivot_high(h, cfg.div_piv, cfg.div_piv), ind.pivot_low(l, cfg.div_piv, cfg.div_piv)
    ph_s, pl_s = ind.pivot_high(h, cfg.struct_len, cfg.struct_len), ind.pivot_low(l, cfg.struct_len, cfg.struct_len)
    htf_f, htf_s = map_htf(cfg, t, htf)
    pdh, pdl = prev_day_levels(t, h, l)
    hh_prev = ind.shift(ind.highest(h, cfg.sweep_look), 1)
    ll_prev = ind.shift(ind.lowest(l, cfg.sweep_look), 1)
    rng_a = np.maximum(h - l, mintick)
    buy_v, sell_v = v * (c - l) / rng_a, v * (h - c) / rng_a
    with np.errstate(divide="ignore", invalid="ignore"):
        buy_rel = ind.ema(buy_v, 10) / ind.ema(buy_v, 25)
        sell_rel = ind.ema(sell_v, 10) / ind.ema(sell_v, 25)
    bb_b, bb_d = ind.sma(c, 20), 2.0 * ind.stdev(c, 20)
    kc_b, kc_r = ind.ema(c, 20), 1.5 * ind.atr(h, l, c, 20)
    with np.errstate(invalid="ignore"):
        sqz_on = ((bb_b + bb_d) < (kc_b + kc_r)) & ((bb_b - bb_d) > (kc_b - kc_r))

    zones: list = []
    kde_samples: list = []
    last_sig = -1000
    # divergencia RSI
    last_plp = last_plr = last_php = last_phr = None
    last_plb = last_phb = 0
    bull_div_bar = bear_div_bar = None
    # estructura
    last_sh = last_sl = None
    sh_broken = sl_broken = True
    m_trend = 0
    # niveles de la sesion de referencia
    prev_in_l = False
    sess_hi = sess_lo = sess_last_hi = sess_last_lo = None
    # operacion virtual (equivale a dirT / curSL / curTP1 / curTP2 de Pine)
    v_dir = 0
    v_sl = v_tp1 = v_tp2 = v_entry = 0.0
    v_bar = 0
    v_tp1_done = False

    def add_zone(top, bot, ty, born):
        for z in zones:
            if z["ty"] == ty and z["act"] and bot < z["top"] and top > z["bot"]:
                return
        zones.append({"top": top, "bot": bot, "ty": ty, "act": True, "born": born})
        if len(zones) > cfg.max_zones:
            zones.pop(0)

    for i in range(1, n):
        # ---- divergencia RSI ----
        dp = cfg.div_piv
        bull_now = bear_now = False
        if not math.isnan(pl_d[i]):
            pi = i - dp
            if not math.isnan(rsi[pi]):
                if (last_plp is not None and l[pi] < last_plp and rsi[pi] > last_plr
                        and rsi[pi] <= cfg.div_low and (pi - last_plb) <= cfg.div_max_bars):
                    bull_now = True
                last_plp, last_plr, last_plb = l[pi], rsi[pi], pi
        if not math.isnan(ph_d[i]):
            pi = i - dp
            if not math.isnan(rsi[pi]):
                if (last_php is not None and h[pi] > last_php and rsi[pi] < last_phr
                        and rsi[pi] >= cfg.div_high and (pi - last_phb) <= cfg.div_max_bars):
                    bear_now = True
                last_php, last_phr, last_phb = h[pi], rsi[pi], pi
        if bull_now:
            bull_div_bar = i
        if bear_now:
            bear_div_bar = i
        bull_div_recent = bull_div_bar is not None and i - bull_div_bar <= cfg.div_valid
        bear_div_recent = bear_div_bar is not None and i - bear_div_bar <= cfg.div_valid

        # ---- estructura BOS / CHoCH ----
        if not math.isnan(ph_s[i]):
            last_sh, sh_broken = h[i - cfg.struct_len], False
        if not math.isnan(pl_s[i]):
            last_sl, sl_broken = l[i - cfg.struct_len], False
        sh_now = sl_now = False
        if last_sh is not None and not sh_broken and c[i] > last_sh:
            sh_now, sh_broken, m_trend = True, True, 1
        if last_sl is not None and not sl_broken and c[i] < last_sl:
            sl_now, sl_broken, m_trend = True, True, -1
        if i == n - 1:
            an.struct_break_up, an.struct_break_dn = sh_now, sl_now

        # ---- max/min de la ultima sesion de referencia ----
        if cfg.use_sess_lvl:
            in_l = _in_window(cfg.sess_lvl_utc, t[i])
            if in_l:
                if not prev_in_l:
                    sess_hi, sess_lo = h[i], l[i]
                else:
                    sess_hi = h[i] if sess_hi is None else max(sess_hi, h[i])
                    sess_lo = l[i] if sess_lo is None else min(sess_lo, l[i])
            elif prev_in_l:
                sess_last_hi, sess_last_lo = sess_hi, sess_lo
            prev_in_l = in_l

        if math.isnan(atr[i]) or math.isnan(es[i]) or math.isnan(ef[i]):
            continue
        a = atr[i]

        # ---- zonas nuevas ----
        if not math.isnan(ph_z[i]):
            add_zone(ph_z[i], ph_z[i] - a * cfg.zone_mult, 1, i)
        if not math.isnan(pl_z[i]):
            add_zone(pl_z[i] + a * cfg.zone_mult, pl_z[i], -1, i)

        rng = max(h[i] - l[i], mintick)
        up_w = h[i] - max(o[i], c[i])
        dn_w = min(o[i], c[i]) - l[i]
        strong = abs(c[i] - o[i]) >= a * cfg.body_mult

        # ---- deteccion sobre zonas ----
        brk_up = brk_dn = rev_short = rev_long = False
        rev_short_top = rev_long_bot = next_sup = next_dem = None
        for z in zones:
            if not z["act"]:
                continue
            if cfg.zone_max_age > 0 and i - z["born"] > cfg.zone_max_age:
                z["act"] = False
                continue
            top, bot = z["top"], z["bot"]
            if z["ty"] == 1:
                if c[i] > top and c[i - 1] <= top:
                    brk_up = True
                if h[i] >= bot and c[i] < bot and c[i] < o[i] and up_w >= cfg.wick_ratio * rng:
                    rev_short = True
                    rev_short_top = top if rev_short_top is None else max(rev_short_top, top)
                if bot > c[i] and (next_sup is None or bot < next_sup):
                    next_sup = bot
                if c[i] > top:
                    z["act"] = False
            else:
                if c[i] < bot and c[i - 1] >= bot:
                    brk_dn = True
                if l[i] <= top and c[i] > top and c[i] > o[i] and dn_w >= cfg.wick_ratio * rng:
                    rev_long = True
                    rev_long_bot = bot if rev_long_bot is None else min(rev_long_bot, bot)
                if top < c[i] and (next_dem is None or top > next_dem):
                    next_dem = top
                if c[i] < bot:
                    z["act"] = False

        # ---- kernel (KDE) del volumen relativo ----
        va = volavg[i]
        rel_vol = (v[i] / va) if (not math.isnan(va) and va > 0) else 1.0
        raw_brk = brk_up or brk_dn
        kde_p = kde_prob(kde_samples, rel_vol, cfg.kde_bw) if raw_brk else None
        if cfg.vol_mode == "kernel" and len(kde_samples) >= cfg.kde_min:
            vol_raw = kde_p is not None and kde_p * 100.0 >= cfg.kde_thr
        else:
            vol_raw = (not math.isnan(va)) and v[i] > va * cfg.vol_mult
        if raw_brk:
            kde_samples.append(rel_vol)
            if len(kde_samples) > cfg.kde_len:
                kde_samples.pop(0)

        # ---- ingredientes ----
        slope_up = i >= cfg.slope_bars and es[i] > es[i - cfg.slope_bars]
        slope_dn = i >= cfg.slope_bars and es[i] < es[i - cfg.slope_bars]
        bull_cloud = ef[i] > es[i] and c[i] > es[i]
        bear_cloud = ef[i] < es[i] and c[i] < es[i]
        hv = not (math.isnan(htf_f[i]) or math.isnan(htf_s[i]))
        raw_bull = hv and htf_f[i] > htf_s[i]
        raw_bear = hv and htf_f[i] < htf_s[i]
        htf_bull = raw_bull if cfg.use_htf else True
        htf_bear = raw_bear if cfg.use_htf else True

        e_ok = not math.isnan(e200[i])
        e200_bo_l = (not cfg.use_e200_dir or (e_ok and c[i] > e200[i])) and \
                    (not cfg.use_e200_ext or (e_ok and (c[i] - e200[i]) <= cfg.max_ext200 * a))
        e200_bo_s = (not cfg.use_e200_dir or (e_ok and c[i] < e200[i])) and \
                    (not cfg.use_e200_ext or (e_ok and (e200[i] - c[i]) <= cfg.max_ext200 * a))
        e200_rv_l = not cfg.use_e200_rv or (e_ok and (e200[i] - c[i]) >= cfg.min_stretch * a)
        e200_rv_s = not cfg.use_e200_rv or (e_ok and (c[i] - e200[i]) >= cfg.min_stretch * a)
        adx_ok = (not math.isnan(adx[i])) and adx[i] >= cfg.adx_min
        flow_l = (1.0 if math.isnan(buy_rel[i]) else buy_rel[i]) >= cfg.flow_min
        flow_s = (1.0 if math.isnan(sell_rel[i]) else sell_rel[i]) >= cfg.flow_min
        sqz_raw = bool(sqz_on[max(0, i - cfg.sqz_look + 1):i + 1].any())
        adx_range = (not math.isnan(adx[i])) and adx[i] < cfg.adx_range_max

        sweep_s = ((not math.isnan(pdh[i]) and h[i] > pdh[i] and c[i] < pdh[i])
                   or (not math.isnan(hh_prev[i]) and h[i] > hh_prev[i] and c[i] < hh_prev[i])
                   or (cfg.use_sess_lvl and sess_last_hi is not None and h[i] > sess_last_hi and c[i] < sess_last_hi))
        sweep_l = ((not math.isnan(pdl[i]) and l[i] < pdl[i] and c[i] > pdl[i])
                   or (not math.isnan(ll_prev[i]) and l[i] < ll_prev[i] and c[i] > ll_prev[i])
                   or (cfg.use_sess_lvl and sess_last_lo is not None and l[i] < sess_last_lo and c[i] > sess_last_lo))

        score_bo_l = sum((raw_bull, vol_raw, strong, slope_up, adx_ok, flow_l, sqz_raw, m_trend == 1))
        score_bo_s = sum((raw_bear, vol_raw, strong, slope_dn, adx_ok, flow_s, sqz_raw, m_trend == -1))
        score_rv_l = sum((raw_bull, sweep_l, flow_l, adx_range, bull_div_recent))
        score_rv_s = sum((raw_bear, sweep_s, flow_s, adx_range, bear_div_recent))

        extra_l = (not cfg.use_adx or adx_ok) and (not cfg.use_flow or flow_l) and (not cfg.use_sqz or sqz_raw)
        extra_s = (not cfg.use_adx or adx_ok) and (not cfg.use_flow or flow_s) and (not cfg.use_sqz or sqz_raw)

        if cfg.use_score:
            long_bo = cfg.use_breakout and brk_up and bull_cloud and score_bo_l >= cfg.min_score_bo
            short_bo = cfg.use_breakout and brk_dn and bear_cloud and score_bo_s >= cfg.min_score_bo
            long_rv = cfg.use_reversion and rev_long and score_rv_l >= cfg.min_score_rv
            short_rv = cfg.use_reversion and rev_short and score_rv_s >= cfg.min_score_rv
        else:
            long_bo = (cfg.use_breakout and brk_up and bull_cloud and htf_bull and (not cfg.use_vol or vol_raw)
                       and strong and (not cfg.use_slope or slope_up) and extra_l)
            short_bo = (cfg.use_breakout and brk_dn and bear_cloud and htf_bear and (not cfg.use_vol or vol_raw)
                        and strong and (not cfg.use_slope or slope_dn) and extra_s)
            long_rv = (cfg.use_reversion and rev_long and (not cfg.rev_with_htf or raw_bull)
                       and (not cfg.rev_need_sweep or sweep_l))
            short_rv = (cfg.use_reversion and rev_short and (not cfg.rev_with_htf or raw_bear)
                        and (not cfg.rev_need_sweep or sweep_s))

        struct_l = (not cfg.use_struct_bo) or m_trend == 1
        struct_s = (not cfg.use_struct_bo) or m_trend == -1
        div_rv_l = (not cfg.rev_need_div) or bull_div_recent
        div_rv_s = (not cfg.rev_need_div) or bear_div_recent
        long_bo = long_bo and e200_bo_l and struct_l
        short_bo = short_bo and e200_bo_s and struct_s
        adx_v = adx[i]
        rv_ok = ((cfg.rv_min_vol <= 0 or rel_vol >= cfg.rv_min_vol)
                 and (cfg.rv_min_adx <= 0 or (not math.isnan(adx_v) and adx_v >= cfg.rv_min_adx))
                 and (cfg.rv_max_adx <= 0 or (not math.isnan(adx_v) and adx_v <= cfg.rv_max_adx)))
        long_rv = long_rv and e200_rv_l and div_rv_l and rv_ok
        short_rv = short_rv and e200_rv_s and div_rv_s and rv_ok
        long_sw = (cfg.use_sweep_sig and sweep_l and c[i] > o[i] and dn_w >= cfg.wick_ratio * rng
                   and (not cfg.rev_with_htf or raw_bull) and e200_rv_l and div_rv_l and rv_ok)
        short_sw = (cfg.use_sweep_sig and sweep_s and c[i] < o[i] and up_w >= cfg.wick_ratio * rng
                    and (not cfg.rev_with_htf or raw_bear) and e200_rv_s and div_rv_s and rv_ok)

        if t[i] >= count_from:                       # diagnostico: por que se descartan rupturas / rechazos
            if brk_up or brk_dn:
                bump("rupturas")
                up = brk_up
                if not cfg.use_breakout:
                    bump("BO_desactivado")
                for nm, ok in (("nube", bull_cloud if up else bear_cloud), ("htf", htf_bull if up else htf_bear),
                               ("volumen", (not cfg.use_vol or vol_raw)), ("cuerpo", strong),
                               ("pendiente", (not cfg.use_slope or (slope_up if up else slope_dn))),
                               ("adx/flow/squeeze", extra_l if up else extra_s), ("ema200", e200_bo_l if up else e200_bo_s),
                               ("estructura", struct_l if up else struct_s)):
                    if not ok:
                        bump("BO_falla_" + nm)
            if rev_long or rev_short:
                bump("rechazos")
                lg = rev_long
                if not cfg.use_reversion:
                    bump("RV_desactivado")
                for nm, ok in (("htf", not cfg.rev_with_htf or (raw_bull if lg else raw_bear)),
                               ("barrido", not cfg.rev_need_sweep or (sweep_l if lg else sweep_s)),
                               ("ema200", e200_rv_l if lg else e200_rv_s), ("divergencia", div_rv_l if lg else div_rv_s),
                               ("filtros_rv", rv_ok)):
                    if not ok:
                        bump("RV_falla_" + nm)

        long_cand = long_bo or long_rv or long_sw
        short_cand = (short_bo or short_rv or short_sw) and not long_cand

        # ---- gestion de la operacion virtual (antes de permitir una nueva) ----
        if v_dir != 0 and i > v_bar:
            closed = False
            if v_dir == 1:
                if l[i] <= v_sl or h[i] >= v_tp2:
                    closed = True
                elif cfg.use_partial and not v_tp1_done and h[i] >= v_tp1:
                    v_tp1_done = True
                    if cfg.use_be:
                        v_sl = v_entry
            else:
                if h[i] >= v_sl or l[i] <= v_tp2:
                    closed = True
                elif cfg.use_partial and not v_tp1_done and l[i] <= v_tp1:
                    v_tp1_done = True
                    if cfg.use_be:
                        v_sl = v_entry
            if not closed:
                if cfg.use_trail and (v_tp1_done or not cfg.use_partial) and not math.isnan(hh3[i]):
                    v_sl = max(v_sl, hh3[i] - a * cfg.trail_atr) if v_dir == 1 else min(v_sl, ll3[i] + a * cfg.trail_atr)
                if cfg.use_struct_exit and ((v_dir == 1 and sl_now) or (v_dir == -1 and sh_now)):
                    closed = True
                if cfg.max_bars > 0 and i - v_bar >= cfg.max_bars:
                    closed = True
            if closed:
                v_dir = 0

        can_signal = (i - last_sig > cfg.cooldown) and _in_session(cfg, t[i]) and v_dir == 0
        side = "LONG" if long_cand else ("SHORT" if short_cand else None)
        diag = {"brk": raw_brk, "rev": rev_long or rev_short, "cand": side is not None,
                "cooldown": side is not None and not can_signal, "in_trade": v_dir != 0,
                "side": side, "blocked": None, "signal": False}

        if side and can_signal:
            entry = c[i]
            if side == "LONG":
                if long_bo:
                    sl = es[i] - a * cfg.sl_buffer
                elif long_rv:
                    sl = (l[i] - a * cfg.sl_buffer) if rev_long_bot is None else (rev_long_bot - a * cfg.sl_buffer)
                else:
                    sl = l[i] - a * cfg.sl_buffer
                if sl >= entry:
                    sl = entry - a
                risk = entry - sl
                room = next_sup is None or (next_sup - entry) >= risk * cfg.min_rr
                tp = entry + risk * cfg.rr_fallback if (next_sup is None or cfg.fixed_tp) else next_sup - a * cfg.tp_buffer
                if cfg.use_e200_tp and not long_bo and e_ok and e200[i] > entry \
                        and (e200[i] - entry) >= risk * cfg.min_rr and e200[i] < tp:
                    tp = e200[i]
                kind = "BO" if long_bo else ("RV" if long_rv else "SW")
                allowed = cfg.allow_long
                norm = score_bo_l / 8.0 if long_bo else score_rv_l / 5.0
                score = score_bo_l if long_bo else score_rv_l
            else:
                if short_bo:
                    sl = es[i] + a * cfg.sl_buffer
                elif short_rv:
                    sl = (h[i] + a * cfg.sl_buffer) if rev_short_top is None else (rev_short_top + a * cfg.sl_buffer)
                else:
                    sl = h[i] + a * cfg.sl_buffer
                if sl <= entry:
                    sl = entry + a
                risk = sl - entry
                room = next_dem is None or (entry - next_dem) >= risk * cfg.min_rr
                tp = entry - risk * cfg.rr_fallback if (next_dem is None or cfg.fixed_tp) else next_dem + a * cfg.tp_buffer
                if cfg.use_e200_tp and not short_bo and e_ok and e200[i] < entry \
                        and (entry - e200[i]) >= risk * cfg.min_rr and e200[i] > tp:
                    tp = e200[i]
                kind = "BO" if short_bo else ("RV" if short_rv else "SW")
                allowed = cfg.allow_short
                norm = score_bo_s / 8.0 if short_bo else score_rv_s / 5.0
                score = score_bo_s if short_bo else score_rv_s
            sgn = 1.0 if side == "LONG" else -1.0
            rr = (tp - entry) * sgn / risk
            mult = (cfg.size_min + (cfg.size_max - cfg.size_min) * norm) if cfg.size_by_score else 1.0

            blocked = None
            if not allowed:
                blocked = "no_permitido"
            elif not room:
                blocked = "sin_espacio"
            elif cfg.min_risk_pct > 0 and risk / entry * 100.0 < cfg.min_risk_pct:
                blocked = "riesgo_corto"
            elif cfg.max_risk_atr > 0 and risk > a * cfg.max_risk_atr:
                blocked = "riesgo_ancho"
            elif cfg.cost_mult > 0 and (risk * cfg.tp1_r / entry * 100.0) < 2.0 * cfg.fee_pct * cfg.cost_mult:
                blocked = "costes"
            elif rr <= 0 or mult <= 0:
                blocked = "rr_invalido"
            if blocked:
                diag["blocked"] = blocked
            else:
                tp2 = entry + sgn * risk * cfg.rr_trail_cap if cfg.use_trail else tp
                tp1 = entry + sgn * risk * cfg.tp1_r
                an.signals.append(Signal(idx=i, time=int(t[i]), side=side, kind=kind, entry=entry, sl=sl,
                                         tp=tp2, rr=rr, atr=a, tp1=tp1 if cfg.use_partial else None,
                                         risk_mult=mult, score=int(score),
                                         htf=("alcista" if raw_bull else ("bajista" if raw_bear else "n/d")),
                                         rel_vol=float(rel_vol), adx=0.0 if math.isnan(adx[i]) else float(adx[i])))
                diag["signal"] = True
                last_sig = i
                v_dir = 1 if side == "LONG" else -1
                v_sl, v_tp1, v_tp2, v_entry, v_bar, v_tp1_done = sl, tp1, tp2, entry, i, False
        an.diag = diag
        if t[i] >= count_from:
            if diag["cand"]:
                bump("candidatas")
            if diag["cooldown"]:
                bump("en_cooldown_o_trade")
            if diag["blocked"]:
                bump("bloq_" + diag["blocked"])
            if diag["signal"]:
                bump("senales")

    an.last_time = int(t[-1])
    an.last_close = float(c[-1])
    an.hh3, an.ll3, an.atr = float(hh3[-1]), float(ll3[-1]), float(atr[-1])
    return an
