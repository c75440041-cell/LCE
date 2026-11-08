"""Port a Python del indicador Pine v6 "QS + SSMT 2 Etapas + tCISD | Senales 15m".

Flujo (HTF -> LTF): 1) alineacion de cuartos (369 / cadena)  2) llave: SSMT de 2 etapas
3) gatillo tCISD  4) SL tras el extremo del cuarto, TP en extension "137" o RR fijo.
Hora de referencia: Nueva York; el dia de trading empieza a las 18:00 NY.

SUPUESTOS heredados del indicador (no verificados): "137" = 1.37 x rango del cuarto anterior;
triada correlacionada (aqui BTC/ETH/SOL por defecto; el original usa NQ/ES/YM).
`run()` repasa todas las velas cerradas; solo se opera la senal de la ULTIMA vela.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from . import indicators as ind
from .strategy import Analysis, Signal

NY = ZoneInfo("America/New_York")
DAY_MS = 86_400_000
WEEK_MS = 604_800_000


@dataclass
class QSSignal(Signal):
    info: str = ""
    to: str = ""


# ---------------------------------------------------------------- tiempo (reloj de Nueva York)
_mq1_cache: dict = {}


def mq1_start(y, mo):
    """Inicio (ms) del primer 'Q1 mensual': domingo 18:00 NY previo al primer lunes del mes."""
    k = (y, mo)
    if k not in _mq1_cache:
        first_dow = (datetime(y, mo, 1, 12, 0, tzinfo=NY).isoweekday() % 7) + 1      # Pine: domingo = 1
        first_mon = 1 + ((2 - first_dow + 7) % 7)
        _mq1_cache[k] = int(datetime(y, mo, first_mon, 18, 0, tzinfo=NY).timestamp() * 1000) - DAY_MS
    return _mq1_cache[k]


def monthly_q(t_ms):
    dt = datetime.fromtimestamp(t_ms / 1000.0, NY)
    y, mo = dt.year, dt.month
    start = mq1_start(y, mo)
    if t_ms < start:
        mo -= 1
        if mo == 0:
            mo, y = 12, y - 1
        start = mq1_start(y, mo)
    else:
        ny, nmo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        ns = mq1_start(ny, nmo)
        if t_ms >= ns:
            start = ns
    return min(math.floor((t_ms - start) / WEEK_MS) + 1, 4)


def time_fields(t, fri_q4=False):
    """Cuartos y marcas de cambio por vela. Todo en hora NY (las desviaciones DST se calculan por hora)."""
    t = np.asarray(t, np.int64)
    n = len(t)
    hrs = t // 3_600_000
    uh = np.unique(hrs)
    hh = np.empty(len(uh), np.int64)
    dw = np.empty(len(uh), np.int64)
    mo = np.empty(len(uh), np.int64)
    mq = np.empty(len(uh), np.int64)
    for j, h in enumerate(uh):
        dt = datetime.fromtimestamp(int(h) * 3600, NY)
        hh[j] = dt.hour
        dw[j] = (dt.isoweekday() % 7) + 1                 # Pine: domingo=1 ... sabado=7
        mo[j] = dt.month
        mq[j] = monthly_q(int(h) * 3_600_000)
    idx = np.searchsorted(uh, hrs)
    sod = hh[idx] * 3600 + (t // 1000) % 3600             # segundos desde 00:00 NY
    msd = ((sod - 64800 + 86400) % 86400) * 1000          # ms desde las 18:00 NY
    qd = (msd // 21_600_000) + 1
    q90 = (msd // 5_400_000) % 4 + 1
    sd = dw[idx] + (hh[idx] >= 18).astype(np.int64)
    qw = np.select([sd == 2, sd == 3, sd == 4, sd == 5, sd == 6], [1, 2, 3, 4, 4 if fri_q4 else 0], 0)
    qy = np.where(mo[idx] <= 3, 1, np.where(mo[idx] <= 6, 2, np.where(mo[idx] <= 9, 3, 4)))
    day_id = t - msd
    chg = lambda a: np.concatenate([[False], a[1:] != a[:-1]])
    return {"sod": sod, "dow": dw[idx], "qd": qd, "q90": q90, "qw": qw, "qm": mq[idx], "qy": qy,
            "new_d": chg(qd), "new_n": chg(q90), "new_w": chg(day_id)}


# ---------------------------------------------------------------- alineacion
def is369(a, b, c):
    ok = all(1 <= x <= 4 for x in (a, b, c))
    return ok and (((a + b + c) - 1) % 9 + 1) % 3 == 0


def valid_pair(a, b, br):
    return (a == b and 1 <= a <= 4) or (br and ((a == 4 and b == 1) or (a == 1 and b == 4)))


def chain_len(q, en, br):
    best = run = prev = 0
    for qi, e in zip(q, en):
        if not e:
            continue
        if 1 <= qi <= 4:
            if run > 0 and valid_pair(prev, qi, br):
                run += 1
            else:
                run = 1
            prev = qi
        else:
            run = prev = 0
        best = max(best, run)
    return best


def align_ok(mode, a369, chain_ok):
    return {"369_o_cadena": a369 or chain_ok, "solo_369": a369, "solo_cadena": chain_ok,
            "369_y_cadena": a369 and chain_ok}.get(mode, True)


# ---------------------------------------------------------------- SSMT
class _Track:
    __slots__ = ("ch", "cl", "ph", "pl")

    def __init__(self):
        self.ch = self.cl = self.ph = self.pl = None

    def update(self, new_q, hi, lo):
        if new_q:
            self.ph, self.pl = self.ch, self.cl
            self.ch, self.cl = hi, lo
        else:
            self.ch = hi if self.ch is None else (None if hi is None else max(self.ch, hi))
            self.cl = lo if self.cl is None else (None if lo is None else min(self.cl, lo))


def _div(c0, p0, c1, p1, c2, p2, is_high):
    """-> (hay divergencia, el activo 1 barrio el extremo previo, el activo 1 es el 'diverger')."""
    v = [c0 is not None and p0 is not None, c1 is not None and p1 is not None, c2 is not None and p2 is not None]
    sw = [False, False, False]
    for k, (c, p) in enumerate(((c0, p0), (c1, p1), (c2, p2))):
        if v[k]:
            sw[k] = (c > p) if is_high else (c < p)
    nv, ns = sum(v), sum(sw)
    dv = v[0] and nv >= 2 and ns >= 1 and ns < nv
    d1 = (not v[1]) or sw[0] != sw[1]
    d2 = (not v[2]) or sw[0] != sw[2]
    return dv, sw[0], dv and d1 and d2


def align_hl(t_ref, d):
    """Alinea high/low de otro activo a las marcas de tiempo del activo 1 (rellena hacia delante)."""
    n = len(t_ref)
    h, l = np.full(n, np.nan), np.full(n, np.nan)
    if d is None:
        return h, l
    idx = np.searchsorted(np.asarray(d["t"], np.int64), np.asarray(t_ref, np.int64), side="right") - 1
    ok = idx >= 0
    h[ok], l[ok] = np.asarray(d["h"], float)[idx[ok]], np.asarray(d["l"], float)[idx[ok]]
    return h, l


def _opt(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ---------------------------------------------------------------- estrategia
def run(cfg, d1, h2r, l2r, h3r, l3r, mintick=1e-9, count_from=0) -> Analysis:
    t = np.asarray(d1["t"], np.int64)
    o, h, l, c = (np.asarray(d1[k], float) for k in ("o", "h", "l", "c"))
    n = len(c)
    an = Analysis(n=n)
    if n < 60:
        an.diag = {"reason": "pocas_velas"}
        return an

    def bump(k):
        an.counts[k] = an.counts.get(k, 0) + 1

    tf = time_fields(t, cfg.qs_fri_q4)
    atr = ind.atr(h, l, c, cfg.atr_len)
    h2r, l2r, h3r, l3r = (np.asarray(x, float) for x in (h2r, l2r, h3r, l3r))
    h2, l2 = (-l2r, -h2r) if cfg.qs_inv2 else (h2r, l2r)
    h3, l3 = (-l3r, -h3r) if cfg.qs_inv3 else (h3r, l3r)

    seq_wd = cfg.qs_seq == "wd90"
    stage_d = cfg.qs_stage == "daily"
    en = (cfg.qs_use_year, cfg.qs_use_month, True, True, True)
    trD = [_Track() for _ in range(3)]
    trN = [_Track() for _ in range(3)]
    trW = [_Track() for _ in range(3)]

    tdo = tso = two = None
    ref_hi_body_low = ref_lo_body_high = None
    ref_hi_bar = ref_lo_bar = -10 ** 9
    done_bear = done_bull = False
    sod_prev = -1
    trig = lambda cur, prev, tgt: cur >= tgt and (prev < tgt or cur < prev)

    for i in range(n):
        hi3 = (float(h[i]), _opt(h2[i]), _opt(h3[i]))
        lo3 = (float(l[i]), _opt(l2[i]), _opt(l3[i]))
        for k in range(3):
            trD[k].update(bool(tf["new_d"][i]), hi3[k], lo3[k])
            trN[k].update(bool(tf["new_n"][i]), hi3[k], lo3[k])
            trW[k].update(bool(tf["new_w"][i]), hi3[k], lo3[k])
        if math.isnan(atr[i]):
            sod_prev = int(tf["sod"][i])
            continue

        # ---- alineacion ----
        qy, qm, qw, qd, q90 = (int(tf[k][i]) for k in ("qy", "qm", "qw", "qd", "q90"))
        sa, sb, sc = (qw, qd, q90) if seq_wd else (qm, qw, qd)
        a369 = is369(sa, sb, sc)
        chain = chain_len((qy, qm, qw, qd, q90), en, cfg.qs_bridge)
        chain_ok = chain >= cfg.qs_min_chain
        aligned = align_ok(cfg.qs_align, a369, chain_ok)

        # ---- SSMT: ciclo elegido y ciclo superior ----
        def dv(tr, is_high):
            return _div(tr[0].ch if is_high else tr[0].cl, tr[0].ph if is_high else tr[0].pl,
                        tr[1].ch if is_high else tr[1].cl, tr[1].ph if is_high else tr[1].pl,
                        tr[2].ch if is_high else tr[2].cl, tr[2].ph if is_high else tr[2].pl, is_high)

        st_tr, pa_tr = (trD, trW) if stage_d else (trN, trD)
        st_bear, _, st_bear_dv = dv(st_tr, True)
        st_bull, _, st_bull_dv = dv(st_tr, False)
        pa_bear, _, _ = dv(pa_tr, True)
        pa_bull, _, _ = dv(pa_tr, False)
        key_bear = st_bear and (not cfg.qs_use2 or pa_bear)
        key_bull = st_bull and (not cfg.qs_use2 or pa_bull)
        q_hi, q_lo, p_hi, p_lo = st_tr[0].ch, st_tr[0].cl, st_tr[0].ph, st_tr[0].pl
        new_qs = bool(tf["new_d"][i] if stage_d else tf["new_n"][i])

        # ---- True Opens ----
        sod = int(tf["sod"][i])
        if trig(sod, sod_prev, 0):
            tdo = float(o[i])
        if any(trig(sod, sod_prev, x) for x in (70200, 5400, 27000, 48600)):
            tso = float(o[i])
        if trig(sod, sod_prev, 64800) and int(tf["dow"][i]) == 2:
            two = float(o[i])
        sod_prev = sod
        disc = sum(1 for x in (tdo, tso, two) if x is not None and c[i] < x)
        prem = sum(1 for x in (tdo, tso, two) if x is not None and c[i] > x)
        to_ok_l = (not cfg.qs_use_to) or disc >= cfg.qs_min_to
        to_ok_s = (not cfg.qs_use_to) or prem >= cfg.qs_min_to

        # ---- tCISD ----
        if new_qs:
            done_bear = done_bull = False
        if new_qs or (q_hi is not None and h[i] >= q_hi):
            ref_hi_body_low, ref_hi_bar = float(min(o[i], c[i])), i
        if new_qs or (q_lo is not None and l[i] <= q_lo):
            ref_lo_body_high, ref_lo_bar = float(max(o[i], c[i])), i
        tc_bear = ref_hi_body_low is not None and c[i] < ref_hi_body_low and i > ref_hi_bar and i - ref_hi_bar <= cfg.qs_max_wait
        tc_bull = ref_lo_body_high is not None and c[i] > ref_lo_body_high and i > ref_lo_bar and i - ref_lo_bar <= cfg.qs_max_wait
        bear_bias = aligned and key_bear and not key_bull
        bull_bias = aligned and key_bull and not key_bear

        if t[i] >= count_from:
            if aligned:
                bump("alineado")
            if key_bear:
                bump("ssmt_bajista")
            if key_bull:
                bump("ssmt_alcista")
            if bear_bias:
                bump("sesgo_bajista")
            if bull_bias:
                bump("sesgo_alcista")
            if bear_bias and tc_bear:
                bump("tcisd_bajista")
            if bull_bias and tc_bull:
                bump("tcisd_alcista")

        diag = {"align": aligned, "key_bear": key_bear, "key_bull": key_bull, "bias": bear_bias or bull_bias,
                "cand": (bear_bias and tc_bear) or (bull_bias and tc_bull), "signal": False, "blocked": None,
                "brk": key_bear or key_bull, "rev": False, "cooldown": False}

        # ---- senales ----
        for side in ("SHORT", "LONG"):
            if side == "SHORT":
                go = cfg.allow_short and bear_bias and tc_bear and not done_bear
                to_ok, held = to_ok_s, (not cfg.qs_req_div or st_bear_dv)
                if not go or q_hi is None:
                    continue
                sl = q_hi + atr[i] * cfg.qs_sl_buf
                risk = sl - c[i]
                rng_prev = (p_hi - p_lo) if (p_hi is not None and p_lo is not None) else None
                t137 = (q_hi - cfg.qs_fib_ext * rng_prev) if rng_prev is not None else None
                use137 = cfg.qs_tp_mode == "ext137" and t137 is not None and t137 < c[i]
                tp = t137 if use137 else c[i] - risk * cfg.qs_rr_fixed
                rr = (c[i] - tp) / risk if risk > 0 else None
            else:
                go = cfg.allow_long and bull_bias and tc_bull and not done_bull
                to_ok, held = to_ok_l, (not cfg.qs_req_div or st_bull_dv)
                if not go or q_lo is None:
                    continue
                sl = q_lo - atr[i] * cfg.qs_sl_buf
                risk = c[i] - sl
                rng_prev = (p_hi - p_lo) if (p_hi is not None and p_lo is not None) else None
                t137 = (q_lo + cfg.qs_fib_ext * rng_prev) if rng_prev is not None else None
                use137 = cfg.qs_tp_mode == "ext137" and t137 is not None and t137 > c[i]
                tp = t137 if use137 else c[i] + risk * cfg.qs_rr_fixed
                rr = (tp - c[i]) / risk if risk > 0 else None
            blocked = None
            if not to_ok:
                blocked = "true_opens"
            elif not (risk > 0 and risk <= cfg.qs_max_risk_atr * atr[i]):
                blocked = "riesgo"
            elif rr is None or rr < cfg.qs_min_rr:
                blocked = "rr"
            elif not held:
                blocked = "diverger"
            if blocked:
                diag["blocked"] = blocked
                if t[i] >= count_from:
                    bump("bloq_" + blocked)
                continue
            info = (f"QS {sa}-{sb}-{sc}" + (" [369]" if a369 else "") + (f" [C{chain}]" if chain_ok else "")
                    + f" · SSMT {'Diario' if stage_d else '90m'}" + (f"+{'Semanal' if stage_d else 'Diario'}" if cfg.qs_use2 else "")
                    + " · tCISD" + (" · TP 137" if use137 else f" · TP {cfg.qs_rr_fixed}R"))
            an.signals.append(QSSignal(idx=i, time=int(t[i]), side=side, kind="QS", entry=float(c[i]), sl=float(sl),
                                       tp=float(tp), rr=float(rr), atr=float(atr[i]), info=info,
                                       to=f"prima {prem} / descuento {disc}"))
            diag["signal"] = True
            if t[i] >= count_from:
                bump("senales")
            if side == "SHORT":
                done_bear = True
            else:
                done_bull = True
        an.diag = diag

    an.last_time = int(t[-1])
    an.last_close = float(c[-1])
    return an
