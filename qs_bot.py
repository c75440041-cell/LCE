"""STRATEGY=qs: bot de la estrategia QS + SSMT 2 etapas + tCISD (15m).

Cada triada (3 simbolos correlacionados) se evalua tres veces: cada miembro hace de "activo del
grafico" y los otros dos de activos 2 y 3. Reutiliza todo el bot LCE: Telegram con win rate,
protecciones, ejecucion en BingX, estadisticas y backtest.
"""
import csv
import io
import json
import logging
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from . import qs_strategy as qs
from .backtest import (DAY, TRAIN_FRAC, _line, clone, covered_days, env_text, load_candles, metrics, passes, split)
from .config import clean_config
from .main import Bot, to_dict
from .risk import fmt_num
from .telegram import esc
from .tracker import cost_r, fmt_wr, resolve

log = logging.getLogger("qs")


def parse_triads(text):
    out = []
    for grp in str(text).split(";"):
        syms = []
        for s in grp.split(","):
            s = s.strip().upper()
            if s:
                syms.append(s if "-" in s else s.replace("USDT", "-USDT"))
        if len(syms) == 3:
            out.append(syms)
        elif syms:
            raise ValueError(f"cada triada necesita exactamente 3 simbolos: {grp}")
    if not out:
        raise ValueError("TRIADS vacio")
    return out


class QSBot(Bot):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.triad_list = parse_triads(self.cfg.triads)

    def refresh_universe(self, force=False):
        super().refresh_universe(force=force)
        members = sorted({s for t in self.triad_list for s in t})
        miss = [s for s in members if s not in self.info]
        if miss:
            log.warning("simbolos de TRIADS no disponibles en BingX: %s", miss)
        self.universe = [s for s in members if s in self.info]

    def _fetch(self, sym):
        try:
            return to_dict(self.bx.klines(sym, self.cfg.timeframe, self.cfg.candles), self.cfg.bar_ms)
        except Exception as e:                                # noqa: BLE001
            log.warning("klines %s: %s", sym, e)
            return None

    def scan(self):
        c = self.cfg
        t0 = time.time()
        self.refresh_universe()
        with ThreadPoolExecutor(max_workers=c.workers) as ex:
            data = dict(zip(self.universe, ex.map(self._fetch, self.universe)))
        funnel, events = Counter(), []
        now = int(time.time() * 1000)
        for tr in self.triad_list:
            for m in tr:
                others = [x for x in tr if x != m]
                d1 = data.get(m)
                if d1 is None or len(d1["t"]) < 200 or others[0] not in data or others[1] not in data:
                    funnel["sin_datos"] += 1
                    continue
                if now - (int(d1["t"][-1]) + c.bar_ms) > 2 * c.bar_ms:
                    funnel["datos_viejos"] += 1
                    continue
                h2, l2 = qs.align_hl(d1["t"], data[others[0]])
                h3, l3 = qs.align_hl(d1["t"], data[others[1]])
                try:
                    an = qs.run(c, d1, h2, l2, h3, l3, mintick=10.0 ** (-self.info.get(m, {}).get("pp", 8)))
                except Exception:                             # noqa: BLE001
                    log.exception("qs.run %s", m)
                    funnel["error"] += 1
                    continue
                funnel["ok"] += 1
                events.extend(self.stats.update(m, d1, c.signal_timeout_bars))
                if not self._hist_done:
                    self._hist_syms += 1
                    for hs in an.signals:
                        self.hist.add(hs, d1, an.n, c.signal_timeout_bars)
                dg = an.diag
                funnel["alineado"] += bool(dg.get("align"))
                funnel["cand"] += bool(dg.get("bias"))
                funnel["tcisd"] += bool(dg.get("cand"))
                if dg.get("blocked"):
                    funnel["bloq_" + dg["blocked"]] += 1
                if an.signals and an.signals[-1].idx == an.n - 1:
                    funnel["senales"] += 1
                    try:
                        self.handle_signal(m, an.signals[-1])
                    except Exception:                         # noqa: BLE001
                        log.exception("handle_signal %s", m)
        self.send_events(events)
        if not self._hist_done:
            self._hist_done = True
            self.tg.send(self.hist.text(c.candles * c.bar_ms / 86_400_000, self._hist_syms))
        self.cycles += 1
        self.funnel_total.update(funnel)
        log.info("Ciclo %d en %.1fs | %s", self.cycles, time.time() - t0, dict(funnel))

    def format_signal(self, sym, sig):
        c = self.cfg
        pp = self.info.get(sym, {}).get("pp", 6)
        pct = lambda x: abs(x - sig.entry) / sig.entry * 100
        when = datetime.fromtimestamp((sig.time + c.bar_ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"{'🟢' if sig.side == 'LONG' else '🔴'} <b>{sig.side}</b> · <b>{esc(sym)}</b> · {c.timeframe} [{self.label()}]",
                 "Tipo: QS + SSMT 2 etapas + tCISD", f"🕒 {when}",
                 f"Entrada: <code>{fmt_num(sig.entry, pp)}</code>",
                 f"🛑 SL: <code>{fmt_num(sig.sl, pp)}</code> (-{pct(sig.sl):.2f}%)",
                 f"🎯 TP: <code>{fmt_num(sig.tp, pp)}</code> (+{pct(sig.tp):.2f}%)",
                 f"R:R {sig.rr:.2f} · ATR {fmt_num(sig.atr, pp)}", esc(getattr(sig, "info", "")),
                 f"True Opens: {getattr(sig, 'to', '')}",
                 f"Riesgo {c.risk_pct * sig.risk_mult:.2f}% del equity" + (f" · Lev {c.leverage}x" if self.trading else ""),
                 "📊 " + self.stats.summary()]
        h = self.hist
        if h.w + h.l:
            lines.append(f"🧪 Backtest reciente: {fmt_wr(h.w, h.l)} · R neto {h.nw - h.nl:+.1f}")
        return "\n".join(lines)


# ---------------------------------------------------------------- backtest / barrido
def _eval_qs(args):
    label, d1, h2, l2, h3, l3, cfgs, mintick, timeout, window_start = args
    out = {}
    for name, cfg in cfgs:
        try:
            an = qs.run(cfg, d1, h2, l2, h3, l3, mintick=mintick)
        except Exception as e:                                   # noqa: BLE001
            log.warning("%s %s: %s", label, name, e)
            out[name] = []
            continue
        c_pct = cfg.fee_pct + cfg.slip_pct
        rows = []
        for sg in an.signals:
            if sg.time < window_start:
                continue
            res, r = resolve(sg.side, sg.entry, sg.sl, sg.tp, sg.time, d1, timeout)
            if res in ("win", "loss"):
                rows.append((sg.time, sg.kind, 1 if res == "win" else 0, r, cost_r(sg.entry, sg.sl, c_pct)))
        out[name] = rows
    return label, out


def stage1():
    out = [("BASE (indicador por defecto)", {})]
    for al in ("369_o_cadena", "solo_369", "solo_cadena", "369_y_cadena", "sin_filtro"):
        for st in ("daily", "m90"):
            for u2 in (True, False):
                for tp in ("ext137", "rr"):
                    out.append((f"{al} · {st}{'+sup' if u2 else ''} · {tp}",
                                {"qs_align": al, "qs_stage": st, "qs_use2": u2, "qs_tp_mode": tp}))
    return out


QS_EXTRAS = (("sin True Opens", {"qs_use_to": False}), ("True Opens>=2", {"qs_min_to": 2}),
             ("solo diverger", {"qs_req_div": True}), ("espera 8", {"qs_max_wait": 8}),
             ("espera 32", {"qs_max_wait": 32}), ("RR>=1.5", {"qs_min_rr": 1.5}), ("RR>=2", {"qs_min_rr": 2.0}),
             ("sec. M-S-D", {"qs_seq": "mwd"}))


def _prepare(bot, cfg, days, use_cache=True, extra_bars=700):
    triads = parse_triads(cfg.triads)
    members = sorted({s for t in triads for s in t})
    data = {}
    for sym in members:
        try:
            d = load_candles(bot.bx, cfg, sym, cfg.timeframe, days, extra_bars=extra_bars, use_cache=use_cache)
        except Exception as e:                                   # noqa: BLE001
            log.warning("descarga %s: %s", sym, e)
            d = None
        if d is not None:
            data[sym] = d
    return triads, data


def _tasks(triads, data, cfg_list, info, timeout, window_start, chunk=6):
    tasks = []
    chunks = [cfg_list[i:i + chunk] for i in range(0, len(cfg_list), chunk)]
    for tr in triads:
        for m in tr:
            others = [x for x in tr if x != m]
            if any(x not in data for x in (m, *others)):
                continue
            h2, l2 = qs.align_hl(data[m]["t"], data[others[0]])
            h3, l3 = qs.align_hl(data[m]["t"], data[others[1]])
            mt = 10.0 ** (-info.get(m, {}).get("pp", 8))
            for ch in chunks:
                tasks.append((m, data[m], h2, l2, h3, l3, ch, mt, timeout, window_start))
    return tasks


def _run_pool(tasks, names, workers):
    agg = {n: [] for n in names}
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            results = pool.map(_eval_qs, tasks, chunksize=1)
    else:
        results = [_eval_qs(t) for t in tasks]
    for _, out in results:
        for n, rows in out.items():
            agg[n].extend(rows)
    return agg


def run_backtest(bot):
    live_cfg, tg = bot.cfg, bot.tg
    cfg = clean_config()
    for k in ("backtest_days", "backtest_workers", "backtest_min_n", "triads"):
        setattr(cfg, k, getattr(live_cfg, k))
    ignored = live_cfg.active_env()
    days = cfg.backtest_days
    bot.refresh_universe(force=True)
    t0 = time.time()
    tg.send(f"🔬 <b>Barrido QS iniciado</b>: {days} días · {len(parse_triads(cfg.triads))} triada(s). Descargando…")
    triads, data = _prepare(bot, cfg, days)
    if not data:
        tg.send("⛔ No pude descargar historial.")
        return
    cov = float(np.median([covered_days(d) for d in data.values()]))
    end_t = max(int(d["t"][-1]) for d in data.values())
    window_start = end_t - int(days * DAY)
    split_t = window_start + int(TRAIN_FRAC * days * DAY)
    note = f"\n⚠️ Solo ~{cov:.1f} días disponibles (paginación)." if cov < days * 0.9 else ""
    timeout = cfg.signal_timeout_bars

    c1 = [(n, clone(cfg, ov)) for n, ov in stage1()]
    ov_by = dict(stage1())
    agg = _run_pool(_tasks(triads, data, c1, bot.info, timeout, window_start), [n for n, _ in c1], cfg.backtest_workers)
    scored = {n: (metrics(split(r, split_t)[0]), metrics(split(r, split_t)[1]), metrics(r)) for n, r in agg.items()}
    base = "BASE (indicador por defecto)"
    ranked = sorted([n for n in scored if n != base and scored[n][0]["n"] >= cfg.backtest_min_n],
                    key=lambda n: -scored[n][0]["e"])
    tg.send(f"✅ Etapa 1: {len(c1)} configs en {time.time() - t0:.0f}s. Afinando las 3 mejores…")
    top = [(n, ov_by[n]) for n in ranked[:3]]
    if top:
        c2 = [(f"{n} + {en}", clone(cfg, dict(ov, **eov))) for n, ov in top for en, eov in QS_EXTRAS]
        for n, ov in top:
            for en, eov in QS_EXTRAS:
                ov_by[f"{n} + {en}"] = dict(ov, **eov)
        agg2 = _run_pool(_tasks(triads, data, c2, bot.info, timeout, window_start), [n for n, _ in c2], cfg.backtest_workers)
        for n, r in agg2.items():
            scored[n] = (metrics(split(r, split_t)[0]), metrics(split(r, split_t)[1]), metrics(r))
    names = sorted([n for n in scored if n != base and scored[n][0]["n"] >= cfg.backtest_min_n], key=lambda n: -scored[n][0]["e"])

    lines = [f"🔬 <b>Barrido QS+SSMT+tCISD</b> · {cov:.0f} d · {len(data)} símbolos · {len(scored)} configs",
             "Base: valores por defecto del indicador" + (f" (variables ignoradas: {', '.join(ignored[:6])})" if ignored else ""),
             f"Coste {2 * (cfg.fee_pct + cfg.slip_pct):.2f}% ida/vuelta · Train {int(TRAIN_FRAC * 100)}% / Test {int((1 - TRAIN_FRAC) * 100)}%" + note,
             "", "<b>BASE</b> " + _line("train", scored[base][0]), "       " + _line("test ", scored[base][1]), "",
             "<b>Top 6 por E_neto en train</b>"]
    winners = []
    for i, n in enumerate(names[:6], 1):
        tr, te, _ = scored[n]
        ok = passes(tr, te, cfg.backtest_min_n)
        winners += [n] if ok else []
        lines += [f"{i}) {esc(n)} {'✅' if ok else '✗'}", "   " + _line("train", tr), "   " + _line("test ", te)]
    if winners:
        best = max(winners, key=lambda n: min(scored[n][0]["e"], scored[n][1]["e"]))
        lines += ["", f"✅ Pasan train Y test: {len(winners)}. Mejor y más estable: <b>{esc(best)}</b>",
                  "Variables (STRATEGY=qs):", "<code>" + env_text(ov_by[best]) + "</code>",
                  "⚠️ Pocas semanas y muchas configs probadas = riesgo de sobreajuste."]
    else:
        lines += ["", "❌ <b>Ninguna configuración es rentable neta de costes en train Y test.</b>",
                  "No operar esta estrategia con este universo. Prueba más días (BACKTEST_DAYS) u otra triada (TRIADS)."]
    tg.send("\n".join(lines))
    dump = {"days": cov, "split_t": split_t, "triads": triads,
            "configs": {n: {"train": scored[n][0], "test": scored[n][1], "all": scored[n][2], "ov": ov_by.get(n, {})} for n in scored}}
    tg.send_document("sweep_qs.json", json.dumps(dump, indent=1, default=str).encode(), "Resultados completos del barrido QS")


# ---------------------------------------------------------------- volcado para TradingView
def run_dump(bot):
    cfg, tg = bot.cfg, bot.tg
    sym, days = cfg.dump_symbol, cfg.dump_days
    triads = parse_triads(cfg.triads)
    tr = next((t for t in triads if sym in t), None)
    if tr is None:
        tg.send(f"⛔ {esc(sym)} no pertenece a ninguna triada de TRIADS ({esc(cfg.triads)}).")
        return
    bot.refresh_universe(force=True)
    _, data = _prepare(bot, cfg, days, use_cache=False)
    others = [x for x in tr if x != sym]
    if any(x not in data for x in (sym, *others)):
        tg.send("⛔ Sin datos de algún miembro de la triada.")
        return
    d1 = data[sym]
    h2, l2 = qs.align_hl(d1["t"], data[others[0]])
    h3, l3 = qs.align_hl(d1["t"], data[others[1]])
    win_start = int(d1["t"][-1]) - int(days * DAY)
    an = qs.run(cfg, d1, h2, l2, h3, l3, mintick=10.0 ** (-bot.info.get(sym, {}).get("pp", 8)), count_from=win_start)
    fmt = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["vela_open_utc", "vela_close_utc", "side", "entrada", "sl", "tp", "rr", "info", "resultado", "R_bruto", "R_neto"])
    rows, wins, losses = [], 0, 0
    for sg in an.signals:
        if sg.time < win_start:
            continue
        res, r = resolve(sg.side, sg.entry, sg.sl, sg.tp, sg.time, d1, cfg.signal_timeout_bars)
        net = (r - cost_r(sg.entry, sg.sl, cfg.fee_pct + cfg.slip_pct)) if res in ("win", "loss") else 0.0
        wins += res == "win"
        losses += res == "loss"
        wr.writerow([fmt(sg.time), fmt(sg.time + cfg.bar_ms), sg.side, sg.entry, sg.sl, sg.tp, f"{sg.rr:.2f}", sg.info, res, f"{r:.2f}", f"{net:.2f}"])
        rows.append(f"{fmt(sg.time)} {sg.side[0]} e={sg.entry:g} sl={sg.sl:g} tp={sg.tp:g} {sg.rr:.1f}R "
                    f"{ {'win': '✅', 'loss': '❌', 'open': '⏳', 'expired': '⌛'}[res] }")
    funnel = " · ".join(f"{k} {v}" for k, v in sorted(an.counts.items())) or "sin eventos"
    act = cfg.active_env()
    head = (f"🧾 <b>Volcado QS {esc(sym)}</b> · {days} d · {len(rows)} señales · WR {fmt_wr(wins, losses)}\n"
            f"Triada: {esc(', '.join(tr))}\n🔎 Embudo: {funnel}\n"
            f"⚙️ Variables fijadas: {', '.join(act) if act else 'ninguna (por defecto)'} · huella {cfg.fingerprint()}\n"
            "Compara con el indicador en TradingView (15m, zona horaria Nueva York o UTC, mismos símbolos). "
            "Hora = apertura de la vela de señal.")
    tg.send(head + "\n\n<code>" + esc("\n".join(rows[-25:]) if rows else "Sin señales en la ventana.") + "</code>")
    tg.send_document(f"senales_QS_{sym}.csv", buf.getvalue().encode(), f"{sym}: señales QS ({days} d)")
