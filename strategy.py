"""Port a Python de LCE (Nivel -> Confirmacion -> Ejecucion), nucleo v2 + modulos v8.

`run()` repasa TODAS las velas cerradas en orden (igual que Pine en historico):
las zonas, el cooldown y las muestras del kernel se reconstruyen en cada ciclo,
asi que el bot no depende de estado persistido para la logica de senales.
Solo se opera la senal de la ULTIMA vela cerrada.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import indicators as ind


@dataclass
class Signal:
    idx: int
    time: int            # open time (ms) de la vela de la senal
    side: str            # LONG | SHORT
    kind: str            # BO | RV | SW
    entry: float
    sl: float
    tp: float
    rr: float
    atr: float
    tp1: Optional[float] = None


@dataclass
class Analysis:
    signals: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)
    last_time: int = 0
    last_close: float = float("nan")
    struct_break_up: bool = False
    struct_break_dn: bool = False
    n: int = 0


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


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


def kde_percentile(samples, x, bw_mult=1.0):
    """Percentil suavizado (CDF con nucleo gaussiano, ancho de banda de Silverman)."""
    s = np.asarray(samples, float)
    n = len(s)
    sd = float(s.std())
    h = bw_mult * 1.06 * sd * n ** (-0.2) if sd > 1e-12 else 1e-6
    h = max(h, 1e-9)
    z = (x - s) / (h * math.sqrt(2.0))
    return 100.0 * sum(0.5 * (1.0 + math.erf(v)) for v in z) / n


def _in_session(cfg, t_ms):
    if not cfg.use_session:
        return True
    a, b = cfg.session_utc.split("-")
    s, e = int(a[:2]) * 60 + int(a[2:]), int(b[:2]) * 60 + int(b[2:])
    m = (int(t_ms) // 60_000) % 1440
    return (s <= m < e) if s <= e else (m >= s or m < e)


def run(cfg, d, htf=None, mintick=1e-9) -> Analysis:
    t = np.asarray(d["t"], np.int64)
    o, h, l, c, v = (np.asarray(d[k], float) for k in ("o", "h", "l", "c", "v"))
    n = len(c)
    an = Analysis(n=n)
    if n < 60:
        an.diag = {"reason": "pocas_velas"}
        return an

    atr = ind.atr(h, l, c, cfg.atr_len)
    ef, es = ind.ema(c, cfg.ema_fast), ind.ema(c, cfg.ema_slow)
    volavg = ind.sma(v, 20)
    e200 = ind.ema(c, cfg.ema200_len)
    _, _, adx = ind.dmi(h, l, c, 14)
    rsi = ind.rsi(c, cfg.rsi_len)
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

    zones: list = []          # dicts: top, bot, ty (1 oferta / -1 demanda), act, born
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

    def add_zone(top, bot, ty, born):
        for z in zones:
            if z["ty"] == ty and z["act"] and bot < z["top"] and top > z["bot"]:
                return
        zones.append({"top": top, "bot": bot, "ty": ty, "act": True, "born": born})
        if len(zones) > cfg.max_zones:
            zones.pop(0)

    for i in range(1, n):
        # ---- divergencia RSI (estado) ----
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

        if math.isnan(atr[i]) or math.isnan(es[i]) or math.isnan(ef[i]):
            continue

        # ---- zonas nuevas ----
        if not math.isnan(ph_z[i]):
            add_zone(ph_z[i], ph_z[i] - atr[i] * cfg.zone_mult, 1, i - cfg.piv_len)
        if not math.isnan(pl_z[i]):
            add_zone(pl_z[i] + atr[i] * cfg.zone_mult, pl_z[i], -1, i - cfg.piv_len)
        if cfg.zone_max_age > 0:
            for z in zones:
                if z["act"] and i - z["born"] > cfg.zone_max_age:
                    z["act"] = False

        rng = max(h[i] - l[i], mintick)
        up_w = h[i] - max(o[i], c[i])
        dn_w = min(o[i], c[i]) - l[i]
        body = abs(c[i] - o[i])
        strong = body >= atr[i] * cfg.body_mult

        # ---- deteccion sobre zonas ----
        brk_up = brk_dn = rev_short = rev_long = False
        rev_short_top = rev_long_bot = next_sup = next_dem = None
        for z in zones:
            if not z["act"]:
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

        # ---- volumen (multiplo o kernel) ----
        rel_vol = v[i] / volavg[i] if volavg[i] and not math.isnan(volavg[i]) else float("nan")
        if not cfg.use_vol:
            vol_ok = True
        elif cfg.vol_mode == "kernel" and len(kde_samples) >= cfg.kde_min and not math.isnan(rel_vol):
            vol_ok = kde_percentile(kde_samples, rel_vol, cfg.kde_bw) >= cfg.kde_thr
        else:
            vol_ok = (not math.isnan(volavg[i])) and v[i] > volavg[i] * cfg.vol_mult
        if (brk_up or brk_dn) and not math.isnan(rel_vol):
            kde_samples.append(rel_vol)
            if len(kde_samples) > cfg.kde_len:
                kde_samples.pop(0)

        # ---- filtros de modulos ----
        slope_up = es[i] > es[i - cfg.slope_bars] if i >= cfg.slope_bars else False
        slope_dn = es[i] < es[i - cfg.slope_bars] if i >= cfg.slope_bars else False
        bull_cloud = ef[i] > es[i] and c[i] > es[i]
        bear_cloud = ef[i] < es[i] and c[i] < es[i]
        if cfg.use_htf:
            hv = not (math.isnan(htf_f[i]) or math.isnan(htf_s[i]))
            htf_bull = hv and htf_f[i] > htf_s[i]
            htf_bear = hv and htf_f[i] < htf_s[i]
        else:
            htf_bull = htf_bear = True
        raw_bull = (not math.isnan(htf_f[i])) and htf_f[i] > htf_s[i]
        raw_bear = (not math.isnan(htf_f[i])) and htf_f[i] < htf_s[i]

        e_ok = not math.isnan(e200[i])
        a = atr[i]
        e200_bo_l = (not cfg.use_e200_dir or (e_ok and c[i] > e200[i])) and \
                    (not cfg.use_e200_ext or (e_ok and (c[i] - e200[i]) <= cfg.max_ext200 * a))
        e200_bo_s = (not cfg.use_e200_dir or (e_ok and c[i] < e200[i])) and \
                    (not cfg.use_e200_ext or (e_ok and (e200[i] - c[i]) <= cfg.max_ext200 * a))
        e200_rv_l = not cfg.use_e200_rv or (e_ok and (e200[i] - c[i]) >= cfg.min_stretch * a)
        e200_rv_s = not cfg.use_e200_rv or (e_ok and (c[i] - e200[i]) >= cfg.min_stretch * a)
        adx_ok = (not cfg.use_adx) or (not math.isnan(adx[i]) and adx[i] >= cfg.adx_min)
        flow_l = (not cfg.use_flow) or (1.0 if math.isnan(buy_rel[i]) else buy_rel[i]) >= cfg.flow_min
        flow_s = (not cfg.use_flow) or (1.0 if math.isnan(sell_rel[i]) else sell_rel[i]) >= cfg.flow_min
        sqz_recent = (not cfg.use_sqz) or bool(sqz_on[max(0, i - cfg.sqz_look + 1):i + 1].any())
        struct_l = (not cfg.use_struct_bo) or m_trend == 1
        struct_s = (not cfg.use_struct_bo) or m_trend == -1

        lv_lo = [x for x in (pdl[i], ll_prev[i]) if not math.isnan(x)]
        lv_hi = [x for x in (pdh[i], hh_prev[i]) if not math.isnan(x)]
        sweep_l = any(l[i] < x < c[i] for x in lv_lo)
        sweep_s = any(h[i] > x > c[i] for x in lv_hi)

        div_rv_l = (not cfg.rev_need_div) or bull_div_recent
        div_rv_s = (not cfg.rev_need_div) or bear_div_recent

        long_bo = (cfg.use_breakout and brk_up and bull_cloud and htf_bull and vol_ok and strong
                   and (not cfg.use_slope or slope_up) and e200_bo_l and adx_ok and flow_l
                   and sqz_recent and struct_l)
        short_bo = (cfg.use_breakout and brk_dn and bear_cloud and htf_bear and vol_ok and strong
                    and (not cfg.use_slope or slope_dn) and e200_bo_s and adx_ok and flow_s
                    and sqz_recent and struct_s)
        long_rv = (cfg.use_reversion and rev_long and (not cfg.rev_with_htf or raw_bull)
                   and (not cfg.rev_need_sweep or sweep_l) and e200_rv_l and div_rv_l)
        short_rv = (cfg.use_reversion and rev_short and (not cfg.rev_with_htf or raw_bear)
                    and (not cfg.rev_need_sweep or sweep_s) and e200_rv_s and div_rv_s)
        long_sw = cfg.use_sweep_sig and sweep_l and c[i] > o[i]
        short_sw = cfg.use_sweep_sig and sweep_s and c[i] < o[i]

        long_cand = long_bo or long_rv or long_sw
        short_cand = (short_bo or short_rv or short_sw) and not long_cand

        can_signal = (i - last_sig > cfg.cooldown) and _in_session(cfg, t[i])
        side = "LONG" if long_cand else ("SHORT" if short_cand else None)
        diag = {"brk": brk_up or brk_dn, "rev": rev_long or rev_short, "cand": side is not None,
                "cooldown": side is not None and not can_signal, "side": side, "blocked": None,
                "signal": False}
        if side and can_signal:
            entry = c[i]
            if side == "LONG":
                if long_bo:
                    sl = es[i] - a * cfg.sl_buffer
                else:
                    sl = (l[i] - a * cfg.sl_buffer) if rev_long_bot is None else (rev_long_bot - a * cfg.sl_buffer)
                if sl >= entry:
                    sl = entry - a
                risk = entry - sl
                lvl = next_sup
                room = lvl is None or (lvl - entry) >= risk * cfg.min_rr
                tp_dist = (lvl - a * cfg.tp_buffer - entry) if lvl is not None else risk * cfg.rr_fallback
                kind = "BO" if long_bo else ("RV" if long_rv else "SW")
                allowed = cfg.allow_long
            else:
                if short_bo:
                    sl = es[i] + a * cfg.sl_buffer
                else:
                    sl = (h[i] + a * cfg.sl_buffer) if rev_short_top is None else (rev_short_top + a * cfg.sl_buffer)
                if sl <= entry:
                    sl = entry + a
                risk = sl - entry
                lvl = next_dem
                room = lvl is None or (entry - lvl) >= risk * cfg.min_rr
                tp_dist = (entry - lvl - a * cfg.tp_buffer) if lvl is not None else risk * cfg.rr_fallback
                kind = "BO" if short_bo else ("RV" if short_rv else "SW")
                allowed = cfg.allow_short
            sgn = 1.0 if side == "LONG" else -1.0

            blocked = None
            if not allowed:
                blocked = "no_permitido"
            elif not room:
                blocked = "sin_espacio"
            else:
                if cfg.use_partial and cfg.use_trail:
                    tp_dist = min(tp_dist, cfg.rr_trail_cap * risk)
                if cfg.use_e200_tp and kind == "RV" and e_ok:
                    de = (e200[i] - entry) * sgn
                    if 0 < de < tp_dist:
                        tp_dist = de
                        if tp_dist < risk * cfg.min_rr:
                            blocked = "rr_ema200"
                first = cfg.tp1_r * risk if cfg.use_partial else tp_dist
                if blocked is None and tp_dist <= 0:
                    blocked = "tp_invalido"
                if blocked is None and cfg.max_risk_atr > 0 and risk > cfg.max_risk_atr * a:
                    blocked = "riesgo_ancho"
                if blocked is None and cfg.cost_mult > 0 and first / entry * 100.0 < cfg.cost_mult * 2.0 * cfg.fee_pct:
                    blocked = "costes"
            if blocked:
                diag["blocked"] = blocked
            else:
                sig = Signal(idx=i, time=int(t[i]), side=side, kind=kind, entry=entry,
                             sl=sl, tp=entry + sgn * tp_dist, rr=tp_dist / risk, atr=a,
                             tp1=(entry + sgn * cfg.tp1_r * risk) if cfg.use_partial else None)
                an.signals.append(sig)
                diag["signal"] = True
                last_sig = i
        an.diag = diag

    an.last_time = int(t[-1])
    an.last_close = float(c[-1])
    return an
