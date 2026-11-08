"""Tamano de posicion, formato de numeros y limites diarios."""
import logging
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

log = logging.getLogger("risk")


def fmt_num(x, decimals):
    """Sin notacion cientifica (tokens de precio muy bajo)."""
    q = Decimal(1).scaleb(-int(decimals))
    return format(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP), "f")


def round_down(x, decimals):
    q = Decimal(1).scaleb(-int(decimals))
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_DOWN))


def size_position(equity, entry, sl, risk_pct, max_lev, qty_prec, min_qty=0.0, min_usdt=0.0):
    """Cantidad para arriesgar risk_pct del equity, limitada a max_lev x equity de nocional."""
    dist = abs(entry - sl)
    if equity <= 0 or entry <= 0 or dist <= 0:
        return None
    qty = (equity * risk_pct / 100.0) / dist
    if qty * entry > equity * max_lev:
        qty = equity * max_lev / entry
    qty = round_down(qty, qty_prec)
    if qty <= 0 or qty < min_qty or qty * entry < min_usdt:
        return None
    return {"qty": qty, "notional": qty * entry, "risk_usdt": qty * dist}


class RiskGuard:
    """Limites diarios y pausa por perdidas seguidas. Estado en la clave 'risk' (dict)."""

    def __init__(self, cfg, state):
        self.cfg = cfg
        self.state = state
        self.r = state.get("risk", {})

    def _save(self):
        self.state.put("risk", self.r)

    def _roll(self, equity):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.r.get("day") != day:
            self.r.update(day=day, day_start_equity=equity, day_pnl=0.0, day_trades=0)
            self._save()

    def can_open(self, equity):
        c = self.cfg
        self._roll(equity)
        if c.max_trades_day and self.r.get("day_trades", 0) >= c.max_trades_day:
            return False, f"maximo {c.max_trades_day} operaciones hoy"
        start = self.r.get("day_start_equity") or equity
        if c.max_day_loss and equity <= start * (1.0 - c.max_day_loss / 100.0):
            return False, f"perdida diaria maxima {c.max_day_loss}%"
        if time.time() < self.r.get("pause_until", 0):
            return False, "en pausa por perdidas seguidas"
        return True, ""

    def on_open(self):
        self.r["day_trades"] = self.r.get("day_trades", 0) + 1
        self._save()

    def on_close(self, pnl, risk_usdt=0.0):
        c = self.cfg
        self.r["day_pnl"] = self.r.get("day_pnl", 0.0) + pnl
        if pnl < -0.3 * risk_usdt if risk_usdt > 0 else pnl < 0:   # igual que Pine: perdida > 0.3R
            self.r["cons_losses"] = self.r.get("cons_losses", 0) + 1
            if c.max_cons_loss and self.r["cons_losses"] >= c.max_cons_loss:
                self.r["pause_until"] = time.time() + c.pause_bars * c.bar_ms / 1000.0
                self.r["cons_losses"] = 0
        else:
            self.r["cons_losses"] = 0
        self._save()
