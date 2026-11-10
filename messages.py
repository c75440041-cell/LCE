"""Formato de los mensajes de Telegram (HTML)."""
from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from typing import Optional

from config import Config
from models import Event, Signal, SymbolMeta, Trade


def esc(x) -> str:
    return html.escape(str(x), quote=False)


def fmt_price(x: float, prec: Optional[int]) -> str:
    if prec is None:
        prec = 2 if x >= 100 else 4 if x >= 1 else 6 if x >= 0.01 else 8
    return f"{x:.{prec}f}"


def fmt_qty(q: float, prec: Optional[int]) -> str:
    if prec is None:
        prec = 4 if q < 100 else 2
    f = 10 ** prec
    return f"{math.floor(q * f) / f:.{prec}f}"


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M UTC")


RANK_LABELS = {
    "confluencia": "Confl.", "htf": "HTF", "tendencia": "Tend.", "retroceso": "Zona",
    "espacio": "Espacio", "stop": "Stop", "vela": "Vela", "liquidez": "Liq.", "mercado": "Mercado",
}


def format_top(sigs: list, cfg: Config, n_valid: int, n_eval: int) -> str:
    """Resumen del TOP de un cierre de vela (se envía antes de los detalles de cada señal)."""
    L = [f"🏆 <b>TOP {len(sigs)} entradas · {cfg.ltf}</b> · cierre {_utc(sigs[0].t + cfg.ltf_sec * 1000)}"]
    for k, s in enumerate(sigs, 1):
        emoji = "🟢" if s.side == "LONG" else "🔴"
        sgn = 1 if s.side == "LONG" else -1
        sl_pct = s.risk / s.entry * 100
        tp1_pct = cfg.tp_r[0] * s.risk / s.entry * 100
        L.append(f"{k}. {emoji} <b>{s.side}</b> {esc(s.symbol)} · <b>{s.rank:.0f}</b>/100 · "
                 f"SL −{sl_pct:.2f}% · TP1 {sgn * tp1_pct:+.2f}%")
    L.append("")
    L.append(f"<i>{n_valid} señales válidas entre {n_eval} pares · solo las {len(sigs)} de mayor ranking "
             f"(mín. {cfg.min_rank:g}/100). Detalle de cada una a continuación.</i>")
    return "\n".join(L)


def format_signal(sig: Signal, meta: SymbolMeta, cfg: Config,
                  pos: Optional[int] = None, total: Optional[int] = None) -> str:
    long = sig.side == "LONG"
    pp = meta.price_prec
    p = lambda x: fmt_price(x, pp)  # noqa: E731
    sgn = 1 if long else -1
    emoji = "🟢" if long else "🔴"
    opp = "SHORT" if long else "LONG"
    sl_pct = sig.risk / sig.entry * 100
    kind = {"oil": " 🛢", "tradfi": " 🏦"}.get(meta.kind, "")

    L = []
    L.append(f"{emoji} <b>{sig.side} · {esc(sig.symbol)}</b>{kind}")
    if pos is not None:
        L.append(f"🏆 <b>Ranking #{pos} de {total}</b> · puntuación <b>{sig.rank:.0f}/100</b>")
    else:
        L.append(f"🏆 Puntuación <b>{sig.rank:.0f}/100</b>")
    rev = " · ⚡ CBC Reversal" if sig.reversal else ""
    L.append(f"⏱ {cfg.ltf} (HTF {cfg.htf}) · Confluencia <b>{sig.score}/{sig.max_score}</b>{rev}")
    L.append(f"🕒 Cierre de vela: {_utc(sig.t + cfg.ltf_sec * 1000)} · {esc(sig.trigger)}")
    L.append("")
    L.append(f"📥 <b>Entrada:</b> <code>{p(sig.entry)}</code> (mercado / cierre de vela)")
    L.append(f"🛑 <b>Stop Loss:</b> <code>{p(sig.sl)}</code> (−{sl_pct:.2f}%)")
    for i, tp in enumerate(sig.tps):
        r = cfg.tp_r[i]
        pct = r * sig.risk / sig.entry * 100
        L.append(f"🎯 <b>TP{i + 1}:</b> <code>{p(tp)}</code> ({r:g}R · {sgn * pct:+.2f}%)")
    rr = " · ".join(f"1:{r:g}" for r in cfg.tp_r)
    wr = sum(f * r for f, r in zip(cfg.tp_split, cfg.tp_r))
    L.append("")
    L.append(f"⚖️ <b>Riesgo/Beneficio:</b> {rr}  (medio si cumple todos los TP: 1:{wr:.2f})")
    if sig.room_r is not None and sig.level_name:
        L.append(f"📏 Espacio libre hasta {sig.level_name} (<code>{p(sig.level_price)}</code>): {sig.room_r:.1f}R")
    else:
        L.append("📏 Sin PDH/PDL/PDC por delante (recorrido libre)")
    L.append(f"📊 ATR {fmt_price(sig.atr, pp)} · Volumen relativo x{sig.rvol:.1f}")
    if sig.rank_parts:
        L.append("🔎 " + " · ".join(f"{RANK_LABELS.get(k, k)} {v}%" for k, v in sig.rank_parts.items()))

    names = {
        "ema_cloud": f"EMA 9/20 {cfg.ltf} a favor",
        "vwap": "Precio vs VWAP a favor",
        "ema200": "Precio vs EMA 200 a favor",
        "htf_cloud": f"EMA 9/20 {cfg.htf} a favor",
    }
    L.append("")
    L.append("✅ <b>Confluencias</b>")
    if cfg.strategy == "pullback":
        if long:
            L.append(f"• CBC {cfg.ltf} VERDE (alcista) + vela ROJA {cfg.ltf} cerrada = compra del retroceso")
        else:
            L.append(f"• CBC {cfg.ltf} ROJO (bajista) + vela VERDE {cfg.ltf} cerrada = venta del rebote")
        if cfg.pullback_require_htf:
            L.append(f"• CBC {cfg.htf} en la misma dirección")
    else:
        L.append(f"• CBC {cfg.ltf} + CBC {cfg.htf} en acuerdo {'alcista' if long else 'bajista'}")
    for k, v in sig.conds.items():
        L.append(f"• {'✅' if v else '▫️'} {names[k]}")

    L.append("")
    L.append("🚪 <b>Salida / gestión</b>")
    for i, f in enumerate(cfg.tp_split):
        extra = " → mover SL a entrada" if (i == 0 and cfg.move_sl_to_be) else ""
        L.append(f"• TP{i + 1}: cerrar {f * 100:.0f}%{extra}")
    if cfg.strategy == "pullback" and cfg.exit_on_cbc_flip:
        L.append(f"• Cerrar todo si el CBC {cfg.ltf} cambia a {'ROJO' if long else 'VERDE'} (cierre de vela)")
    elif cfg.exit_on_opposite:
        if cfg.strategy == "pullback":
            L.append(f"• Cerrar todo si aparece señal {opp} (CBC {cfg.ltf} contrario)")
        else:
            L.append(f"• Cerrar todo si aparece señal {opp} (acuerdo CBC {cfg.ltf}+{cfg.htf} contrario)")
    if cfg.max_trade_hours > 0:
        L.append(f"• Tiempo máximo: {cfg.max_trade_hours:g} h")

    if cfg.account_usdt > 0 and cfg.risk_pct > 0:
        risk_usd = cfg.account_usdt * cfg.risk_pct / 100
        qty = risk_usd / sig.risk
        notional = qty * sig.entry
        lev = max(1, math.ceil(notional / cfg.account_usdt))
        L.append("")
        L.append(f"💰 Riesgo {cfg.risk_pct:g}% de {cfg.account_usdt:,.0f} USDT = {risk_usd:,.2f} USDT")
        L.append(f"   Tamaño ≈ <code>{fmt_qty(qty, meta.qty_prec)}</code> ({notional:,.0f} USDT) · apalancamiento mín. ≈ {lev}x")

    L.append("")
    L.append("<i>Señal informativa, no es asesoría financiera. Entrada manual bajo tu responsabilidad.</i>")
    return "\n".join(L)


def format_event(tr: Trade, ev: Event, cfg: Config, stats_line: str) -> str:
    pp = tr.price_prec
    p = lambda x: fmt_price(x, pp)  # noqa: E731
    head = f"<b>{tr.side} · {esc(tr.symbol)}</b>"
    L = []
    if ev.kind == "TP":
        L.append(f"🎯 <b>TP{ev.n} alcanzado</b> · {head}")
        L.append(f"Precio <code>{p(ev.price)}</code> · +{ev.r:g}R")
        frac = cfg.tp_split[ev.n - 1] if ev.n - 1 < len(cfg.tp_split) else 0
        L.append(f"→ Cierra {frac * 100:.0f}% de la posición")
        if ev.n == 1 and cfg.move_sl_to_be and not ev.final:
            L.append(f"→ Mueve el SL a la entrada (<code>{p(tr.entry)}</code>)")
        if ev.final:
            L.append(f"🏁 Todos los TP cumplidos · resultado total <b>{ev.total_r:+.2f}R</b>")
    elif ev.kind == "SL":
        L.append(f"🛑 <b>Stop Loss</b> · {head}")
        L.append(f"Precio <code>{p(ev.price)}</code> · resultado total <b>{ev.total_r:+.2f}R</b>")
    elif ev.kind == "BE":
        L.append(f"⚖️ <b>Cierre en breakeven</b> · {head}")
        L.append(f"Stop en entrada <code>{p(ev.price)}</code> · resultado total <b>{ev.total_r:+.2f}R</b>")
    elif ev.kind == "OPPOSITE":
        L.append(f"🔄 <b>Salida por señal contraria</b> · {head}")
        L.append(f"Cierra lo que quede cerca de <code>{p(ev.price)}</code> · resultado total <b>{ev.total_r:+.2f}R</b>")
    elif ev.kind == "FLIP":
        color = "ROJO" if tr.side == "LONG" else "VERDE"
        L.append(f"🔄 <b>CBC cambió a {color}</b> · {head}")
        L.append(f"Cierra lo que quede cerca de <code>{p(ev.price)}</code> · resultado total <b>{ev.total_r:+.2f}R</b>")
    elif ev.kind == "TIMEOUT":
        L.append(f"⏳ <b>Tiempo máximo alcanzado</b> · {head}")
        L.append(f"Cierra cerca de <code>{p(ev.price)}</code> · resultado total <b>{ev.total_r:+.2f}R</b>")
    if ev.final:
        L.append("")
        L.append(f"<i>{esc(stats_line)}</i>")
        L.append("<i>(seguimiento simulado por el bot con velas de BingX)</i>")
    return "\n".join(L)


def format_startup(cfg: Config, n_crypto: int, oil: list, n_other: int) -> str:
    oil_txt = ", ".join(oil) if oil else "NO encontrado (usa EXTRA_SYMBOLS)"
    if cfg.strategy == "pullback":
        regla = f"📐 Estrategia: CBC {cfg.ltf} verde → compra velas rojas · CBC rojo → vende velas verdes"
    else:
        regla = f"📐 Estrategia: flip del acuerdo CBC {cfg.ltf} + {cfg.htf}"
    return "\n".join([
        "✅ <b>Bot MapleStax CBC iniciado</b>",
        regla,
        f"⏱ LTF {cfg.ltf} · HTF {cfg.htf} · confluencia mín. {cfg.min_score}/4",
        (f"🏆 Cada cierre de {cfg.ltf}: solo las {cfg.top_n} mejores entradas por ranking (mín. {cfg.min_rank:g}/100)"
         if cfg.top_n > 0 else f"🏆 Todas las señales con ranking ≥ {cfg.min_rank:g}/100"),
        f"🔭 Analiza {n_crypto + len(oil) + n_other} pares de BingX",
        f"🪙 {n_crypto} criptos · 🛢 Petróleo: {esc(oil_txt)}" + (f" · +{n_other} otros (TradFi/extra)" if n_other else ""),
        f"🎯 TP: {' / '.join(f'{r:g}R' for r in cfg.tp_r)} · SL = flip CBC contrario (ATR {cfg.min_sl_atr:g}–{cfg.max_sl_atr:g}x)",
        "Recibirás señales al cierre de cada vela " + cfg.ltf + ".",
    ])
