"""Tamano de posicion, formato de numeros y limites diarios."""
import logging
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
    """Limites diarios, pausa por perdidas seguidas y drawdown de equity.

    Estado en la clave 'risk' (dict). on_close devuelve True si acaba de activar pausa.
    """

    def __init__(self, cfg, state):
        self.cfg = cfg
        self.state = state
        self.r = state.get("risk", {})
        # defaults para campos nuevos (compatibilidad con estados antiguos)
        self.r.setdefault("peak_equity", 0.0)
        self.r.setdefault("max_dd_pct", 0.0)
        self.r.setdefault("cons_losses", 0)
        self.r.setdefault("day_trades", 0)
        self.r.setdefault("day_pnl", 0.0)
        self.r.setdefault("pause_until", 0.0)
        self.r.setdefault("total_closes", 0)
        self.r.setdefault("total_pnl", 0.0)

    def _save(self):
        self.state.put("risk", self.r)

    def _roll(self, equity):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.r.get("day") != day:
            self.r.update(day=day, day_start_equity=equity, day_pnl=0.0, day_trades=0)
            self._save()

    def update_equity(self, equity):
        """Actualiza pico y drawdown (llamar periodicamente con equity real)."""
        if equity <= 0:
            return
        peak = max(float(self.r.get("peak_equity") or 0.0), equity)
        self.r["peak_equity"] = peak
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > float(self.r.get("max_dd_pct") or 0.0):
                self.r["max_dd_pct"] = dd
        self._save()

    def can_open(self, equity):
        c = self.cfg
        self._roll(equity)
        self.update_equity(equity)
        if c.max_trades_day and self.r.get("day_trades", 0) >= c.max_trades_day:
            return False, f"maximo {c.max_trades_day} operaciones hoy"
        start = self.r.get("day_start_equity") or equity
        if c.max_day_loss and equity <= start * (1.0 - c.max_day_loss / 100.0):
            return False, f"perdida diaria maxima {c.max_day_loss}%"
        pause_until = float(self.r.get("pause_until") or 0)
        if time.time() < pause_until:
            mins = max(0, int((pause_until - time.time()) / 60))
            return False, f"en pausa por perdidas seguidas ({mins} min restantes)"
        return True, ""

    def on_open(self):
        self.r["day_trades"] = self.r.get("day_trades", 0) + 1
        self._save()

    def on_close(self, pnl, risk_usdt=0.0):
        """Registra cierre. Devuelve True si se acaba de activar la pausa por rachas."""
        c = self.cfg
        self.r["day_pnl"] = self.r.get("day_pnl", 0.0) + pnl
        self.r["total_closes"] = self.r.get("total_closes", 0) + 1
        self.r["total_pnl"] = self.r.get("total_pnl", 0.0) + pnl
        paused_now = False
        # igual que Pine: perdida > 0.3R cuenta como racha
        if (pnl < -0.3 * risk_usdt) if risk_usdt > 0 else (pnl < 0):
            self.r["cons_losses"] = self.r.get("cons_losses", 0) + 1
            if c.max_cons_loss and self.r["cons_losses"] >= c.max_cons_loss:
                self.r["pause_until"] = time.time() + c.pause_bars * c.bar_ms / 1000.0
                self.r["cons_losses"] = 0
                paused_now = True
                log.warning("Pausa por %d perdidas seguidas hasta %s",
                            c.max_cons_loss,
                            datetime.fromtimestamp(self.r["pause_until"], timezone.utc).isoformat())
        else:
            self.r["cons_losses"] = 0
        self._save()
        return paused_now

    def summary(self, equity=None):
        r = self.r
        parts = [f"día PnL {r.get('day_pnl', 0):+.2f} USDT",
                 f"ops hoy {r.get('day_trades', 0)}"]
        if r.get("peak_equity"):
            parts.append(f"DD máx {r.get('max_dd_pct', 0):.1f}%")
        if equity is not None and equity > 0:
            parts.append(f"equity {equity:.2f}")
        pause_until = float(r.get("pause_until") or 0)
        if time.time() < pause_until:
            mins = max(0, int((pause_until - time.time()) / 60))
            parts.append(f"⏸ pausa {mins}m")
        return " · ".join(parts)
