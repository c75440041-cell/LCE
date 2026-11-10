"""Seguimiento de operaciones: avisa de TP / SL / breakeven / salida por señal contraria.

`step_trade` es una función pura que usan tanto el bot en vivo como el backtest.
Regla conservadora: si en una misma vela se tocan SL y TP, se asume que se tocó el SL primero.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from config import Config
from models import Candle, Event, Signal, Trade
from strategy import cbc_against

log = logging.getLogger("tracker")


def new_trade(sig: Signal, ltf_ms: int, price_prec: Optional[int] = None) -> Trade:
    return Trade(
        symbol=sig.symbol, side=sig.side, entry=sig.entry, sl=sig.sl, sl0=sig.sl,
        tps=list(sig.tps), risk=sig.risk, t_signal=sig.t, t_last=sig.t,
        opened_ms=sig.t + ltf_ms, hits=[False] * len(sig.tps), price_prec=price_prec,
    )


def _r_at(tr: Trade, price: float) -> float:
    d = (price - tr.entry) if tr.side == "LONG" else (tr.entry - price)
    return d / tr.risk


def step_trade(tr: Trade, c: Candle, cfg: Config) -> List[Event]:
    """Procesa UNA vela cerrada posterior a la entrada."""
    events: List[Event] = []
    if tr.closed or c.t <= tr.t_last:
        return events
    tr.t_last = c.t
    is_long = tr.side == "LONG"

    # 1) Stop primero (conservador)
    if (c.l <= tr.sl) if is_long else (c.h >= tr.sl):
        exit_r = _r_at(tr, tr.sl)
        total = tr.realized_r + tr.remaining * exit_r
        kind = "SL" if exit_r < -1e-9 else "BE"
        tr.closed = True
        events.append(Event(kind, price=tr.sl, r=exit_r, final=True, total_r=total))
        return events

    # 2) Take profits en orden
    for idx, tp in enumerate(tr.tps):
        if tr.hits[idx]:
            continue
        reached = (c.h >= tp) if is_long else (c.l <= tp)
        if not reached:
            break
        tr.hits[idx] = True
        frac = cfg.tp_split[idx] if idx < len(cfg.tp_split) else 1.0 / len(tr.tps)
        r_i = cfg.tp_r[idx] if idx < len(cfg.tp_r) else abs(_r_at(tr, tp))
        tr.realized_r += frac * r_i
        tr.remaining = max(0.0, tr.remaining - frac)
        last = all(tr.hits)
        if last:
            tr.closed = True
        events.append(Event("TP", n=idx + 1, price=tp, r=r_i, final=last,
                            total_r=tr.realized_r if last else 0.0))
        if idx == 0 and cfg.move_sl_to_be:
            tr.sl = tr.entry
    return events


def close_trade_at(tr: Trade, price: float, kind: str) -> Event:
    """Cierra lo que quede de la operación a `price` (señal contraria / flip CBC / tiempo máximo)."""
    exit_r = _r_at(tr, price)
    total = tr.realized_r + tr.remaining * exit_r
    tr.closed = True
    return Event(kind, price=price, r=exit_r, final=True, total_r=total)


def cooldown_active(last_close_end_ms: Optional[int], sig_t: int, ltf_ms: int, bars: int) -> bool:
    """True si la señal (vela que ABRE en sig_t) cae dentro de las `bars` velas de espera tras un
    cierre no ganador cuya vela terminó en `last_close_end_ms`."""
    if bars <= 0 or last_close_end_ms is None:
        return False
    return (sig_t + ltf_ms - last_close_end_ms) <= bars * ltf_ms


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ltf_ms = cfg.ltf_sec * 1000
        self.trades: Dict[str, Trade] = {}
        self.last_alert: Dict[str, int] = {}
        self.last_close: Dict[str, int] = {}  # par -> fin de la vela del último cierre NO ganador
        self.stats = {"n": 0, "wins": 0, "losses": 0, "be": 0, "sum_r": 0.0}
        self.load()

    # ---- persistencia ----------------------------------------------------------------
    def load(self) -> None:
        path = self.cfg.state_file
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.trades = {k: Trade(**v) for k, v in d.get("trades", {}).items()}
            self.last_alert = {k: int(v) for k, v in d.get("last_alert", {}).items()}
            self.last_close = {k: int(v) for k, v in d.get("last_close", {}).items()}
            self.stats.update(d.get("stats", {}))
            log.info("Estado cargado: %d operaciones abiertas", len(self.trades))
        except Exception as e:  # estado corrupto: se ignora
            log.warning("No se pudo leer %s: %s", path, e)

    def save(self) -> None:
        if self.cfg.dry_run or not self.cfg.state_file:
            return
        data = {
            "trades": {k: asdict(v) for k, v in self.trades.items()},
            "last_alert": self.last_alert,
            "last_close": self.last_close,
            "stats": self.stats,
        }
        tmp = self.cfg.state_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.cfg.state_file)
        except Exception as e:
            log.warning("No se pudo guardar el estado: %s", e)

    # ---- API ---------------------------------------------------------------------------
    def has_open(self, symbol: str) -> Optional[Trade]:
        return self.trades.get(symbol)

    def already_alerted(self, symbol: str, side: str, t: int) -> bool:
        return self.last_alert.get(f"{symbol}|{side}", -1) >= t

    def register(self, sig: Signal, price_prec: Optional[int]) -> None:
        self.last_alert[f"{sig.symbol}|{sig.side}"] = sig.t
        if self.cfg.track_trades:
            self.trades[sig.symbol] = new_trade(sig, self.ltf_ms, price_prec)

    def in_cooldown(self, symbol: str, sig_t: int) -> bool:
        return cooldown_active(self.last_close.get(symbol), sig_t, self.ltf_ms, self.cfg.cooldown_bars)

    def _finalize(self, tr: Trade, ev: Event, end_ms: Optional[int] = None) -> None:
        self.trades.pop(tr.symbol, None)
        self.stats["n"] += 1
        self.stats["sum_r"] += ev.total_r
        if ev.total_r > 0.05:
            self.stats["wins"] += 1
        else:
            if ev.total_r < -0.05:
                self.stats["losses"] += 1
            else:
                self.stats["be"] += 1
            if end_ms is not None:  # cierre no ganador: el par espera COOLDOWN_BARS velas
                self.last_close[tr.symbol] = end_ms

    def update(self, symbol: str, closed: List[Candle],
               cbc: Optional[List[bool]] = None, now_ms: Optional[int] = None) -> List[Tuple[Trade, Event]]:
        """Procesa las velas nuevas de `symbol`. Devuelve eventos para notificar.

        `cbc` (opcional): estado CBC LTF alineado con `closed`. Si la estrategia es pullback y
        EXIT_ON_CBC_FLIP está activo, la operación se cierra al cierre de la primera vela en la
        que el CBC cambia de color en contra (mismo criterio que el backtest).
        `now_ms` (opcional, solo en vivo): reloj real, para que el tiempo máximo también se
        cumpla si el contrato se queda sin velas nuevas (mercado parado).
        """
        tr = self.trades.get(symbol)
        out: List[Tuple[Trade, Event]] = []
        if tr is None or not closed:
            return out
        first = len(closed)
        while first > 0 and closed[first - 1].t > tr.t_last:
            first -= 1
        flip_exit = (cbc is not None and len(cbc) == len(closed)
                     and self.cfg.strategy == "pullback" and self.cfg.exit_on_cbc_flip)
        for k in range(first, len(closed)):
            cd = closed[k]
            for ev in step_trade(tr, cd, self.cfg):
                out.append((tr, ev))
            if tr.closed:
                break
            if flip_exit and cbc_against(tr.side, cbc[k]):
                out.append((tr, close_trade_at(tr, cd.c, "FLIP")))
                break
        if not tr.closed and self.cfg.max_trade_hours > 0:
            now_edge = closed[-1].t + self.ltf_ms
            if now_ms is not None:
                now_edge = max(now_edge, now_ms)
            if now_edge - tr.opened_ms >= self.cfg.max_trade_hours * 3_600_000:
                out.append((tr, close_trade_at(tr, closed[-1].c, "TIMEOUT")))
        for t_, ev in out:
            if ev.final:
                self._finalize(t_, ev, closed[-1].t + self.ltf_ms)
        return out

    def close_opposite(self, symbol: str, price: float, kind: str = "OPPOSITE",
                       end_ms: Optional[int] = None) -> Optional[Tuple[Trade, Event]]:
        """Cierra lo que quede a `price`. kind: OPPOSITE (señal contraria) | FLIP (CBC cambió de color)."""
        tr = self.trades.get(symbol)
        if tr is None:
            return None
        ev = close_trade_at(tr, price, kind)
        self._finalize(tr, ev, end_ms)
        return tr, ev

    def stats_line(self) -> str:
        s = self.stats
        if s["n"] == 0:
            return "Estadística del bot: sin operaciones cerradas aún"
        wr = 100.0 * s["wins"] / s["n"]
        return (f"Estadística del bot: {s['n']} cerradas · {s['wins']}W/{s['losses']}L/{s['be']}BE "
                f"· winrate {wr:.0f}% · acumulado {s['sum_r']:+.2f}R")
