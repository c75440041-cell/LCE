"""RUN_MODE=backtest -> barrido de parametros con validacion train/test (semanas de historial).
RUN_MODE=dump     -> lista de senales de UN simbolo (CSV) para compararla con TradingView.

Criterio de resultado: primer toque SL/TP de la senal (misma vela = SL), neto de comision +
slippage de ida y vuelta (FEE_PCT + SLIP_PCT por lado). Sin parciales ni trailing.
AVISO: elegir el mejor de muchas configuraciones sobre poca historia es sobreajuste casi
seguro; por eso se valida en un tramo (test) que no se uso para ordenar.
"""
import copy
import csv
import io
import json
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np

from . import strategy
from .config import INTERVAL_MS, PRESETS, clean_config
from .telegram import esc
from .tracker import cost_r, fmt_wr, resolve

log = logging.getLogger("backtest")
DAY = 86_400_000
TRAIN_FRAC = 0.6


# ---------------------------------------------------------------- datos
def _cache_dir(cfg):
    d = os.path.join(cfg.state_dir if os.path.isdir(cfg.state_dir) else ".", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def load_candles(bx, cfg, sym, interval, days, extra_bars=0, use_cache=True):
    """Velas CERRADAS de los ultimos `days` dias (+ extra_bars de calentamiento) o None."""
    ms = INTERVAL_MS[interval]
    now = int(time.time() * 1000)
    end = (now // ms) * ms                       # inicio de la vela en formacion
    start = end - int(days * DAY) - extra_bars * ms
    path = os.path.join(_cache_dir(cfg), f"{sym}_{interval}_{days}_{extra_bars}_{now // (6 * 3_600_000)}.npz")
    if use_cache and os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in "tohlcv"}
    rows = [r for r in bx.klines_range(sym, interval, start, end - 1) if r[0] + ms <= now]
    if len(rows) < 120:
        return None
    a = np.asarray(rows, float)
    d = {"t": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": a[:, 5]}
    if use_cache:
        np.savez(path, **d)
    return d


def covered_days(d):
    return 0.0 if d is None or len(d["t"]) < 2 else (int(d["t"][-1]) - int(d["t"][0])) / DAY


# ---------------------------------------------------------------- evaluacion
def _eval_symbol(args):
    """Worker: ejecuta TODAS las configuraciones sobre un simbolo. -> (sym, {nombre: [filas]})."""
    sym, d, htf, cfgs, mintick, timeout, window_start = args
    out = {}
    for name, cfg in cfgs:
        try:
            an = strategy.run(cfg, d, htf, mintick=mintick)
        except Exception as e:                     # noqa: BLE001
            log.warning("%s %s: %s", sym, name, e)
            out[name] = []
            continue
        c_pct = cfg.fee_pct + cfg.slip_pct
        rows = []
        for sg in an.signals:
            if sg.time < window_start:
                continue
            res, r = resolve(sg.side, sg.entry, sg.sl, sg.tp, sg.time, d, timeout)
            if res in ("win", "loss"):
                rows.append((sg.time, sg.kind, 1 if res == "win" else 0, r, cost_r(sg.entry, sg.sl, c_pct)))
        out[name] = rows
    return sym, out


def metrics(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "w": 0, "wr": None, "e": 0.0, "pf": 0.0, "rnet": 0.0, "rg": 0.0}
    nets = [r[3] - r[4] for r in rows]
    gw = sum(x for x in nets if x > 0)
    gl = -sum(x for x in nets if x < 0)
    w = sum(r[2] for r in rows)
    return {"n": n, "w": w, "wr": 100.0 * w / n, "e": sum(nets) / n, "pf": (gw / gl) if gl > 0 else float("inf"),
            "rnet": sum(nets), "rg": sum(r[3] for r in rows)}


def split(rows, split_t):
    return [r for r in rows if r[0] < split_t], [r for r in rows if r[0] >= split_t]


def clone(cfg, ov):
    c = copy.copy(cfg)
    c.apply_overrides(ov, respect_env=False)
    return c


def run_configs(cfg_list, data, htf, info, timeout, window_start, workers):
    """cfg_list: [(nombre, Config)]. -> {nombre: [filas de todos los simbolos]}."""
    tasks = [(sym, d, htf.get(sym), cfg_list, 10.0 ** (-info.get(sym, {}).get("pp", 8)), timeout, window_start)
             for sym, d in data.items()]
    agg = {name: [] for name, _ in cfg_list}
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            results = pool.map(_eval_symbol, tasks, chunksize=1)
    else:
        results = [_eval_symbol(t) for t in tasks]
    for _, out in results:
        for name, rows in out.items():
            agg[name].extend(rows)
    return agg


# ---------------------------------------------------------------- rejilla
MODES = (("BO+RV", {"use_breakout": True, "use_reversion": True}),
         ("BO", {"use_breakout": True, "use_reversion": False}),
         ("RV", {"use_breakout": False, "use_reversion": True}))
# (nombre, overrides, ambito): solo se prueba un extra donde tiene efecto (BO / RV / ambos)
EXTRAS = (("adx>=20", {"use_adx": True, "adx_min": 20.0}, "bo"),
          ("RV+EMA200 estirada", {"use_e200_rv": True, "min_stretch": 2.5, "use_e200_tp": True}, "rv"),
          ("RV a favor HTF", {"rev_with_htf": True}, "rv"),
          ("RV vol>=1x", {"rv_min_vol": 1.0}, "rv"),
          ("RV con barrido", {"rev_need_sweep": True}, "rv"),
          ("BO a favor estructura", {"use_struct_bo": True}, "bo"),
          ("TP fijo 1.5R", {"fixed_tp": True, "rr_fallback": 1.5}, "both"))


def stage1():
    out = [("BASE (Pine v8 fiel)", {})]
    for mode, mov in MODES:
        for mr in (0.0, 0.6, 1.0):
            for rr in (1.2, 1.5, 2.0):
                out.append((f"{mode} · riesgo>={mr}% · RR>={rr}", dict(mov, min_risk_pct=mr, min_rr=rr)))
    for name, ov in PRESETS.items():
        if ov:
            out.append((f"preset {name}", dict(ov)))
    return out


def stage2(best):
    out = []
    for name, ov in best:
        bo, rv = ov.get("use_breakout", True), ov.get("use_reversion", True)
        for ename, eov, scope in EXTRAS:
            if (scope == "bo" and bo) or (scope == "rv" and rv) or scope == "both":
                out.append((f"{name} + {ename}", dict(ov, **eov)))
    return out


def env_text(ov):
    lines = []
    for k, v in ov.items():
        lines.append(f"{k.upper()}={str(v).lower() if isinstance(v, bool) else v}")
    return "\n".join(lines)


def passes(tr, te, min_n):
    return (tr["n"] >= min_n and te["n"] >= max(10, min_n // 2) and tr["e"] > 0 and te["e"] > 0 and te["pf"] > 1.1)


def _line(label, m):
    wr = "s/d" if m["wr"] is None else f"{m['wr']:.0f}%"
    pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return f"{label} n={m['n']} WR {wr} E {m['e']:+.2f}R PF {pf}"


# ---------------------------------------------------------------- barrido
def run_sweep(bot):
    live_cfg, tg, bx = bot.cfg, bot.tg, bot.bx
    cfg = clean_config()                      # base = valores por defecto de la estrategia (ignora variables de entorno)
    for k in ("backtest_days", "backtest_symbols", "backtest_workers", "backtest_min_n"):
        setattr(cfg, k, getattr(live_cfg, k))
    ignored = live_cfg.active_env()
    days, nsym = cfg.backtest_days, cfg.backtest_symbols
    bot.refresh_universe(force=True)
    syms = bot.universe[:nsym]
    tg.send(f"🔬 <b>Barrido iniciado</b>: {days} días · {len(syms)} símbolos. Descargando historial…")
    t0 = time.time()
    data, htf = {}, {}
    for sym in syms:
        try:
            d = load_candles(bx, cfg, sym, cfg.timeframe, days, extra_bars=300)
            if d is None:
                continue
            data[sym] = d
            if cfg.use_htf:
                htf[sym] = load_candles(bx, cfg, sym, cfg.htf, days + 10)
        except Exception as e:                     # noqa: BLE001
            log.warning("descarga %s: %s", sym, e)
    if not data:
        tg.send("⛔ No pude descargar historial de ningún símbolo.")
        return
    cov = float(np.median([covered_days(d) for d in data.values()]))
    end_t = max(int(d["t"][-1]) for d in data.values())
    window_start = end_t - int(days * DAY)
    split_t = window_start + int(TRAIN_FRAC * days * DAY)
    note = ""
    if cov < days * 0.9:
        note = f"\n⚠️ El servidor devolvió solo ~{cov:.1f} días (la paginación no llegó a {days}). Resultados sobre ese tramo."
    log.info("historial listo en %.0fs: %d simbolos, cobertura %.1f d", time.time() - t0, len(data), cov)

    timeout = cfg.signal_timeout_bars
    c1 = [(n, clone(cfg, ov)) for n, ov in stage1()]
    ov_by = {n: ov for n, ov in stage1()}
    agg = run_configs(c1, data, htf, bot.info, timeout, window_start, cfg.backtest_workers)
    scored = {}
    for n, rows in agg.items():
        tr, te = split(rows, split_t)
        scored[n] = (metrics(tr), metrics(te), metrics(rows))
    ranked = sorted([n for n in scored if scored[n][0]["n"] >= cfg.backtest_min_n and n != "BASE (Pine v8 fiel)"],
                    key=lambda n: -scored[n][0]["e"])
    tg.send(f"✅ Etapa 1: {len(c1)} configuraciones evaluadas en {time.time() - t0:.0f}s. Afinando las 3 mejores…")

    top3 = [(n, ov_by[n]) for n in ranked[:3]]
    if top3:
        c2 = [(n, clone(cfg, ov)) for n, ov in stage2(top3)]
        for n, ov in stage2(top3):
            ov_by[n] = ov
        agg2 = run_configs(c2, data, htf, bot.info, timeout, window_start, cfg.backtest_workers)
        for n, rows in agg2.items():
            tr, te = split(rows, split_t)
            scored[n] = (metrics(tr), metrics(te), metrics(rows))
    allnames = [n for n in scored if n != "BASE (Pine v8 fiel)" and scored[n][0]["n"] >= cfg.backtest_min_n]
    allnames.sort(key=lambda n: -scored[n][0]["e"])

    cost_t = 2 * (cfg.fee_pct + cfg.slip_pct)
    lines = [f"🔬 <b>Barrido de parámetros</b> · {cov:.0f} d · {len(data)} símbolos · {len(scored)} configs",
             "Base: valores por defecto de la estrategia" + (f" (variables ignoradas: {', '.join(ignored[:8])})" if ignored else ""),
             f"Coste {cost_t:.2f}% ida/vuelta · Train {int(TRAIN_FRAC * 100)}% / Test {int((1 - TRAIN_FRAC) * 100)}% (test no se usa para ordenar)" + note,
             "", "<b>BASE</b> " + _line("train", scored["BASE (Pine v8 fiel)"][0]),
             "       " + _line("test ", scored["BASE (Pine v8 fiel)"][1]), "", "<b>Top 6 por E_neto en train</b>"]
    winners = []
    for i, n in enumerate(allnames[:6], 1):
        tr, te, _ = scored[n]
        ok = passes(tr, te, cfg.backtest_min_n)
        if ok:
            winners.append(n)
        lines += [f"{i}) {esc(n)} {'✅' if ok else '✗'}", "   " + _line("train", tr), "   " + _line("test ", te)]
    if winners:
        winners.sort(key=lambda n: -min(scored[n][0]["e"], scored[n][1]["e"]))
        best = winners[0]
        lines += ["", f"✅ Pasan train Y test: {len(winners)}. Mejor y más estable: <b>{esc(best)}</b>",
                  "Variables a pegar en Railway (RUN_MODE=bot):", "<code>" + env_text(ov_by[best]) + "</code>",
                  "⚠️ Aun así: pocas semanas y muchas configs probadas = riesgo de sobreajuste. Observa en signals varias semanas."]
    else:
        lines += ["", "❌ <b>Ninguna configuración es rentable neta de costes en train Y test.</b>",
                  "Eso es una señal para NO operar esta estrategia con estos parámetros/universo. Prueba el universo `majors` o más días."]
    tg.send("\n".join(lines))

    dump = {"days": cov, "symbols": list(data), "split_t": split_t,
            "configs": {n: {"train": scored[n][0], "test": scored[n][1], "all": scored[n][2], "ov": ov_by.get(n, {})}
                        for n in scored}}
    tg.send_document("sweep.json", json.dumps(dump, indent=1, default=str).encode(), "Resultados completos del barrido")
    log.info("barrido terminado en %.0fs", time.time() - t0)


# ---------------------------------------------------------------- volcado TradingView
def run_dump(bot):
    cfg, tg, bx = bot.cfg, bot.tg, bot.bx
    sym, days = cfg.dump_symbol, cfg.dump_days
    d = load_candles(bx, cfg, sym, cfg.timeframe, days, extra_bars=400, use_cache=False)
    if d is None:
        tg.send(f"⛔ Sin datos para {esc(sym)}")
        return
    htf = load_candles(bx, cfg, sym, cfg.htf, days + 10, use_cache=False) if cfg.use_htf else None
    bot.refresh_universe(force=True)
    mt = 10.0 ** (-bot.info.get(sym, {}).get("pp", 8))
    win_start = int(d["t"][-1]) - int(days * DAY)
    an = strategy.run(cfg, d, htf, mintick=mt, count_from=win_start)
    ms = cfg.bar_ms
    rows, buf = [], io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["vela_open_utc", "vela_close_utc", "side", "tipo", "entrada", "sl", "tp", "rr", "resultado", "R_bruto", "R_neto"])
    fmt = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    wins = losses = 0
    for sg in an.signals:
        if sg.time < win_start:
            continue
        res, r = resolve(sg.side, sg.entry, sg.sl, sg.tp, sg.time, d, cfg.signal_timeout_bars)
        net = (r - cost_r(sg.entry, sg.sl, cfg.fee_pct + cfg.slip_pct)) if res in ("win", "loss") else 0.0
        wins += res == "win"
        losses += res == "loss"
        wr.writerow([fmt(sg.time), fmt(sg.time + ms), sg.side, sg.kind, sg.entry, sg.sl, sg.tp, f"{sg.rr:.2f}", res,
                     f"{r:.2f}", f"{net:.2f}"])
        ico = {"win": "✅", "loss": "❌", "open": "⏳", "expired": "⌛"}[res]
        rows.append(f"{fmt(sg.time)} {sg.side[0]} {sg.kind} e={sg.entry:g} sl={sg.sl:g} tp={sg.tp:g} {ico}")
    head = (f"🧾 <b>Volcado {esc(sym)}</b> · {days} d · {len(rows)} señales · WR {fmt_wr(wins, losses)}\n"
            f"Config: preset {cfg.preset} · TF {cfg.timeframe} · HTF {cfg.htf if cfg.use_htf else 'off'} · vol {cfg.vol_mode}\n"
            "Para comparar: en TradingView usa BINGX:" + esc(sym.replace('-', '')) + ".P, zona horaria UTC, mismos inputs que las variables, "
            "y abre 'Lista de operaciones'. La hora aquí es la de APERTURA de la vela de señal (entrada al cierre).")
    body = "\n".join(rows[-25:]) if rows else "Sin señales en la ventana."
    cn = an.counts
    funnel = " · ".join(f"{k} {v}" for k, v in sorted(cn.items())) or "sin eventos de zona en la ventana"
    act = cfg.active_env()
    diag = (f"\n\n🔎 Embudo ({days} d): {funnel}\n"
            f"⚙️ Variables fijadas: {', '.join(act) if act else 'ninguna (valores por defecto)'}\n"
            f"Velas 5m: {len(d['t'])} ({covered_days(d):.1f} d) · HTF: {'sí' if htf is not None else 'no'} · huella {cfg.fingerprint()}")
    tg.send(head + diag + "\n\n<code>" + esc(body) + "</code>")
    tg.send_document(f"senales_{sym}.csv", buf.getvalue().encode(), f"{sym}: señales y resultado ({days} d)")
