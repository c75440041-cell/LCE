"""LCE bot: escanea BingX (perpetuos USDT) cada vela de 5m, envia senales a Telegram
y (segun TRADE_MODE) opera en demo/live. Un solo hilo ejecuta ordenes; el
analisis de simbolos se paraleliza (solo lectura)."""
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import strategy
from bingx import BingX, BingXError
from config import Config, INTERVAL_MS
from risk import RiskGuard, fmt_num, round_down, size_position
from state import State
from telegram import Telegram, esc

log = logging.getLogger("lce")


def to_dict(rows, bar_ms, now_ms=None):
    """Lista de tuplas -> dict de arrays, descartando la vela en formacion."""
    now_ms = now_ms or int(time.time() * 1000)
    rows = [r for r in rows if r[0] + bar_ms <= now_ms]
    if not rows:
        return None
    a = np.asarray(rows, float)
    return {"t": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": a[:, 5]}


class Bot:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.tg = Telegram(c.tg_token, c.tg_chat)
        self.bx = BingX(c.api_key, c.api_secret, demo=c.demo)
        self.state = State(c.state_dir)
        self.risk = RiskGuard(c, self.state)
        self.trades = self.state.get("trades", {})
        self.sig_ts = self.state.get("signals", {})
        self.info, self.vol, self.universe = {}, {}, []
        self._uni_exp = 0.0
        self._htf_cache = {}
        self._lev_done, self._mt_done = set(), set()
        self.funnel_total = Counter()
        self.cycles = self.signals_sent = 0
        self._last_hb = time.time()
        self._last_err_msg = 0.0
        self.trading = c.mode in ("demo", "live")

    # ------------------------------------------------------------------ util
    def _save_trades(self):
        self.state.put("trades", self.trades)

    def label(self):
        return {"signals": "SEÑAL", "demo": "DEMO", "live": "LIVE"}[self.cfg.mode]

    # ------------------------------------------------------------------ arranque
    def startup(self):
        c = self.cfg
        log.info("Modo=%s | base URL=%s | TF=%s | estado=%s (persistente=%s)",
                 c.mode, self.bx.base, c.timeframe, self.state.path, self.state.persistent)
        if self.trading:
            miss = c.missing_for_trading()
            if miss:
                raise SystemExit(f"Faltan variables para operar: {', '.join(miss)}")
            if c.mode == "live":
                probs = c.live_problems()
                if not self.state.persistent:
                    probs.append("sin Volume en /data: el estado se perderia al redeploy")
                if probs:
                    msg = "Live bloqueado: " + "; ".join(probs)
                    self.tg.send("⛔ " + esc(msg))
                    raise SystemExit(msg)
            bal0 = self.bx.balance()                  # valida credenciales ya
            if c.mode == "live" and bal0["equity"] <= 0:
                raise SystemExit("Live bloqueado: equity 0 en BingX (cuenta de futuros sin fondos)")
            try:
                if not self.bx.position_mode():
                    log.warning("Cuenta en modo one-way: activo hedge (requiere sin posiciones abiertas)")
                    self.bx.set_position_mode(True)
            except BingXError as e:
                log.error("No pude fijar modo hedge: %s", e)
            if c.mode == "live":
                try:
                    hedge_ok = self.bx.position_mode()
                except BingXError:
                    hedge_ok = False
                if not hedge_ok:
                    raise SystemExit("Live bloqueado: la cuenta no esta en modo hedge (cierra posiciones y activalo en BingX)")
            self.reconcile()
        self.refresh_universe(force=True)
        self.tg.send(
            f"🤖 <b>LCE bot iniciado</b> [{self.label()}]\n"
            f"TF {c.timeframe} · HTF {c.htf if c.use_htf else 'off'} · {len(self.universe)} símbolos\n"
            f"Endpoint: {esc(self.bx.base)}\n"
            f"Riesgo {c.risk_pct}% · Lev {c.leverage}x · Máx pos {c.max_positions}\n"
            f"Estado persistente: {'sí' if self.state.persistent else 'NO (monta un Volume)'}")

    def reconcile(self):
        """Compara trades guardados con posiciones reales del exchange."""
        live = {(p["symbol"], p["side"]): p for p in self.bx.positions()}
        for sym, tr in list(self.trades.items()):
            if (sym, tr["side"]) not in live:
                log.warning("Trade %s guardado pero sin posicion real -> cierro registro", sym)
                self._finish_trade(sym, tr, reason="reconciliacion")
        known = {(s, t["side"]) for s, t in self.trades.items()}
        orphans = [k for k in live if k not in known]
        if orphans:
            self.tg.send("⚠️ Posiciones abiertas sin registro del bot (no las gestiono): "
                         + ", ".join(f"{s} {d}" for s, d in orphans))

    # ------------------------------------------------------------------ universo
    def refresh_universe(self, force=False):
        c = self.cfg
        if not force and time.time() < self._uni_exp:
            return
        info = {}
        for ct in self.bx.contracts():
            sym = str(ct.get("symbol", ""))
            if not sym.endswith("-USDT"):
                continue
            if str(ct.get("apiStateOpen", "true")).lower() != "true" or int(ct.get("status", 1) or 1) != 1:
                continue
            if any(sym.upper().startswith(p) for p in c.non_crypto_prefixes):
                continue
            info[sym] = {"pp": int(ct.get("pricePrecision", 4) or 4),
                         "qp": int(ct.get("quantityPrecision", 0) or 0),
                         "min_qty": float(ct.get("tradeMinQuantity", 0) or 0),
                         "min_usdt": float(ct.get("tradeMinUSDT", 0) or 0)}
        vols = {}
        try:
            for t in self.bx.tickers():
                vols[t.get("symbol")] = float(t.get("quoteVolume") or 0)
        except (BingXError, ValueError) as e:
            log.warning("tickers no disponibles: %s", e)
        wanted = None
        if c.symbols.upper() != "ALL":
            wanted = {s.strip().upper().replace("USDT", "-USDT") if "-" not in s else s.strip().upper()
                      for s in c.symbols.split(",") if s.strip()}
        syms = [s for s in info if wanted is None or s in wanted]
        if wanted is None and vols:
            syms = [s for s in syms if vols.get(s, 0) >= c.min_quote_volume]
        syms.sort(key=lambda s: -vols.get(s, 0))
        self.info, self.vol, self.universe = info, vols, syms[:c.max_symbols]
        self._uni_exp = time.time() + 3600
        log.info("Universo: %d simbolos (de %d contratos)", len(self.universe), len(info))

    # ------------------------------------------------------------------ analisis (hilos)
    def _get_htf(self, sym):
        c = self.cfg
        bucket = int(time.time() * 1000) // c.htf_ms
        cached = self._htf_cache.get(sym)
        if cached and cached[0] == bucket:
            return cached[1]
        d = to_dict(self.bx.klines(sym, c.htf, 300), c.htf_ms)
        self._htf_cache[sym] = (bucket, d)
        return d

    def analyze_symbol(self, sym):
        c = self.cfg
        try:
            d = to_dict(self.bx.klines(sym, c.timeframe, c.candles), c.bar_ms)
            if d is None or len(d["t"]) < 120:
                return sym, "pocas_velas", None
            if int(time.time() * 1000) - (int(d["t"][-1]) + c.bar_ms) > 2 * c.bar_ms:
                return sym, "datos_viejos", None
            htf = self._get_htf(sym) if c.use_htf else None
            mt = 10.0 ** (-self.info.get(sym, {}).get("pp", 8))
            return sym, "ok", strategy.run(c, d, htf, mintick=mt)
        except Exception as e:                        # un simbolo malo no tumba el ciclo
            log.warning("analisis %s fallo: %s", sym, e)
            return sym, "error", None

    # ------------------------------------------------------------------ escaneo
    def scan(self):
        c = self.cfg
        t0 = time.time()
        self.refresh_universe()
        funnel = Counter()
        with ThreadPoolExecutor(max_workers=c.workers) as ex:
            results = list(ex.map(self.analyze_symbol, self.universe))
        found = []
        for sym, status, an in results:
            funnel[status] += 1
            if status != "ok":
                continue
            dg = an.diag
            funnel["brk"] += bool(dg.get("brk"))
            funnel["rev"] += bool(dg.get("rev"))
            funnel["cand"] += bool(dg.get("cand"))
            funnel["cooldown"] += bool(dg.get("cooldown"))
            if dg.get("blocked"):
                funnel["bloq_" + dg["blocked"]] += 1
            if c.use_struct_exit and sym in self.trades:
                tr = self.trades[sym]
                if (tr["side"] == "LONG" and an.struct_break_dn) or (tr["side"] == "SHORT" and an.struct_break_up):
                    self.close_trade(sym, "estructura rota en contra")
            if an.signals and an.signals[-1].idx == an.n - 1:
                found.append((sym, an.signals[-1]))
        funnel["senales"] = len(found)
        found.sort(key=lambda x: -x[1].rr)
        for sym, sig in found:
            self.handle_signal(sym, sig)
        self.cycles += 1
        self.funnel_total.update(funnel)
        log.info("Ciclo %d en %.1fs | %s", self.cycles, time.time() - t0, dict(funnel))

    # ------------------------------------------------------------------ senales
    def handle_signal(self, sym, sig):
        if sig.time <= self.sig_ts.get(sym, 0):
            return
        self.sig_ts[sym] = sig.time
        self.state.put("signals", self.sig_ts)
        self.signals_sent += 1
        pp = self.info.get(sym, {}).get("pp", 6)
        arrow = "🟢" if sig.side == "LONG" else "🔴"
        pct = lambda x: abs(x - sig.entry) / sig.entry * 100
        msg = (f"{arrow} <b>{sig.side}</b> · <b>{esc(sym)}</b> · {self.cfg.timeframe} · {sig.kind} [{self.label()}]\n"
               f"Entrada: <code>{fmt_num(sig.entry, pp)}</code>\n"
               f"SL: <code>{fmt_num(sig.sl, pp)}</code> (-{pct(sig.sl):.2f}%)\n"
               f"TP: <code>{fmt_num(sig.tp, pp)}</code> (+{pct(sig.tp):.2f}%)\n"
               + (f"TP1: <code>{fmt_num(sig.tp1, pp)}</code>\n" if sig.tp1 else "")
               + f"R:R {sig.rr:.2f}")
        self.tg.send(msg)
        if self.trading:
            try:
                self.open_trade(sym, sig)
            except BingXError as e:
                log.error("open_trade %s: %s", sym, e)
                self.tg.send(f"⛔ No ejecutada {esc(sym)}: {esc(e)}")

    # ------------------------------------------------------------------ ejecucion
    def _exit_side(self, side):
        return "SELL" if side == "LONG" else "BUY"

    def _place_exit(self, sym, side, kind, qty, price):
        inf = self.info[sym]
        typ = "STOP_MARKET" if kind == "SL" else "TAKE_PROFIT_MARKET"
        for attempt in range(3):
            try:
                o = self.bx.place_order(symbol=sym, side=self._exit_side(side), positionSide=side, type=typ,
                                        quantity=fmt_num(qty, inf["qp"]), stopPrice=fmt_num(price, inf["pp"]),
                                        workingType="MARK_PRICE")
                oid = o.get("orderId")
                if oid:
                    return str(oid)
                log.error("%s %s sin orderId: %s", kind, sym, o)
            except BingXError as e:
                log.error("%s %s intento %d: %s", kind, sym, attempt + 1, e)
            time.sleep(0.6 * (attempt + 1))
        return None

    def _close_market(self, sym, side, qty):
        inf = self.info[sym]
        return self.bx.place_order(symbol=sym, side=self._exit_side(side), positionSide=side, type="MARKET",
                                   quantity=fmt_num(qty, inf["qp"]))

    def _confirm_fill(self, sym, side, oid):
        """Devuelve (precio_medio, cantidad) SOLO si de verdad se lleno."""
        for _ in range(8):
            time.sleep(0.5)
            try:
                o = self.bx.get_order(sym, oid)
            except BingXError:
                continue
            st = str(o.get("status", "")).upper()
            qty = float(o.get("executedQty") or 0)
            if st in ("CANCELED", "CANCELLED", "FAILED", "EXPIRED", "REJECTED"):
                return None
            if st in ("FILLED", "PARTIALLY_FILLED") and qty > 0:
                return float(o.get("avgPrice") or 0), qty
        for p in self.bx.positions(sym):                 # ultimo recurso: la posicion real
            if p["side"] == side and p["amt"] > 0:
                return p["avg"], p["amt"]
        return None

    def open_trade(self, sym, sig):
        c = self.cfg
        inf = self.info.get(sym)
        if inf is None:
            return
        if c.kill_switch:
            self.tg.send(f"ℹ️ {esc(sym)} no ejecutada: KILL_SWITCH activo")
            return
        if sym in self.trades:
            log.info("%s ya tiene operacion abierta", sym)
            return
        if len(self.trades) >= c.max_positions:
            self.tg.send(f"ℹ️ {esc(sym)} no ejecutada: máximo de {c.max_positions} posiciones")
            return
        bal = self.bx.balance()
        ok, why = self.risk.can_open(bal["equity"])
        if not ok:
            self.tg.send(f"ℹ️ {esc(sym)} no ejecutada: {why}")
            return
        px = self.bx.price(sym)
        if px <= 0:
            return
        sgn = 1 if sig.side == "LONG" else -1
        risk_d = (px - sig.sl) * sgn
        reward = (sig.tp - px) * sgn
        if risk_d <= 0 or reward <= 0 or reward / risk_d < c.min_live_rr:
            self.tg.send(f"ℹ️ {esc(sym)} no ejecutada: el precio ya se movió (R:R real {reward / risk_d if risk_d > 0 else 0:.2f})")
            return
        sz = size_position(bal["equity"], px, sig.sl, c.risk_pct, c.max_lev, inf["qp"], inf["min_qty"], inf["min_usdt"])
        if not sz:
            self.tg.send(f"ℹ️ {esc(sym)} no ejecutada: tamaño por debajo del mínimo")
            return
        if sz["notional"] / c.leverage > bal["available"] * 0.95:
            self.tg.send(f"ℹ️ {esc(sym)} no ejecutada: margen disponible insuficiente")
            return
        qty = sz["qty"]
        n1 = 0.0
        if c.use_partial and sig.tp1:
            n1 = round_down(qty * c.tp1_pct / 100.0, inf["qp"])
            if n1 < inf["min_qty"] or qty - n1 < inf["min_qty"] or n1 <= 0:
                n1 = 0.0

        # apalancamiento en AMBOS lados (hedge) y margen aislado, una vez por simbolo
        if sym not in self._lev_done:
            lev_ok = 0
            for s in ("LONG", "SHORT"):
                try:
                    self.bx.set_leverage(sym, s, c.leverage)
                    lev_ok += 1
                except BingXError as e:
                    log.warning("leverage %s %s: %s", sym, s, e)
            if lev_ok == 0 and c.mode == "live":
                self.tg.send(f"⛔ {esc(sym)} no ejecutada: no pude fijar el apalancamiento {c.leverage}x")
                return
            self._lev_done.add(sym)
        if sym not in self._mt_done:
            try:
                self.bx.set_margin_type(sym, "ISOLATED")
            except BingXError as e:
                log.info("marginType %s: %s", sym, e)
            self._mt_done.add(sym)

        o = self.bx.place_order(symbol=sym, side="BUY" if sgn > 0 else "SELL", positionSide=sig.side,
                                type="MARKET", quantity=fmt_num(qty, inf["qp"]))
        oid = o.get("orderId")
        fill = self._confirm_fill(sym, sig.side, oid) if oid else None
        if not fill:
            self.tg.send(f"⛔ {esc(sym)}: la orden de entrada NO se confirmó como ejecutada")
            return
        avg, fqty = fill
        fqty = round_down(fqty, inf["qp"])

        sl_id = self._place_exit(sym, sig.side, "SL", fqty, sig.sl)
        if not sl_id:                                   # sin SL no se queda abierta
            try:
                self._close_market(sym, sig.side, fqty)
            except BingXError as e:
                log.error("cierre de emergencia %s fallo: %s", sym, e)
            self.tg.send(f"🚨 {esc(sym)}: no pude colocar el SL; posición cerrada por seguridad")
            return
        tp_id = tp1_id = None
        if n1 > 0 and fqty - n1 > 0:
            tp1_id = self._place_exit(sym, sig.side, "TP", n1, sig.tp1)
            tp_id = self._place_exit(sym, sig.side, "TP", round_down(fqty - n1, inf["qp"]), sig.tp)
        else:
            n1 = 0.0
            tp_id = self._place_exit(sym, sig.side, "TP", fqty, sig.tp)
        if not tp_id:
            self.tg.send(f"⚠️ {esc(sym)}: SL puesto pero NO pude colocar el TP; revisa la posición")

        self.trades[sym] = {
            "side": sig.side, "kind": sig.kind, "entry": avg, "sl": sig.sl, "tp": sig.tp, "tp1": sig.tp1 if n1 else None,
            "qty": fqty, "n1": n1, "atr": sig.atr, "risk_d": abs(avg - sig.sl), "opened": time.time(),
            "sl_id": sl_id, "tp_id": tp_id, "tp1_id": tp1_id, "tp1_done": False, "trail_ts": 0.0,
        }
        self._save_trades()
        self.risk.on_open()
        self.tg.send(f"✅ <b>{self.label()} abierta</b> {sig.side} {esc(sym)}\n"
                     f"Qty {fmt_num(fqty, inf['qp'])} @ {fmt_num(avg, inf['pp'])} · riesgo ≈ {sz['risk_usdt']:.2f} USDT\n"
                     f"SL {'✔' if sl_id else '✖'} · TP {'✔' if tp_id else '✖'}")

    # ------------------------------------------------------------------ gestion
    def _replace_sl(self, sym, tr, new_sl, qty):
        inf = self.info[sym]
        new_id = self._place_exit(sym, tr["side"], "SL", qty, new_sl)
        if not new_id:
            return False
        old = tr.get("sl_id")
        tr["sl_id"], tr["sl"] = new_id, new_sl
        if old:
            try:
                self.bx.cancel_order(sym, old)
            except BingXError as e:
                log.info("cancel SL viejo %s: %s", sym, e)
        return True

    def _finish_trade(self, sym, tr, reason=""):
        pnl = None
        try:
            pnl = self.bx.income(sym, int(tr["opened"] * 1000) - 1000) if self.trading else None
        except BingXError as e:
            log.warning("income %s: %s", sym, e)
        if self.trading:
            try:
                self.bx.cancel_all(sym)
            except BingXError as e:
                log.info("cancel_all %s: %s", sym, e)
        self.trades.pop(sym, None)
        self._save_trades()
        if pnl is not None:
            self.risk.on_close(pnl)
        r = (pnl / (tr["risk_d"] * tr["qty"])) if pnl is not None and tr.get("risk_d") and tr.get("qty") else None
        self.tg.send(f"🏁 <b>Cerrada</b> {tr['side']} {esc(sym)} {('· ' + reason) if reason else ''}\n"
                     + (f"PnL realizado: {pnl:+.2f} USDT" + (f" ({r:+.2f} R)" if r is not None else "")
                        if pnl is not None else "PnL: no disponible"))

    def close_trade(self, sym, reason):
        tr = self.trades.get(sym)
        if not tr:
            return
        try:
            pos = [p for p in self.bx.positions(sym) if p["side"] == tr["side"]]
            if pos:
                self._close_market(sym, tr["side"], pos[0]["amt"])
        except BingXError as e:
            log.error("close_trade %s: %s", sym, e)
            self.tg.send(f"🚨 No pude cerrar {esc(sym)} ({esc(e)}). Revisa manualmente.")
            return
        time.sleep(1.0)
        self._finish_trade(sym, tr, reason)

    def manage(self):
        if not self.trades:
            return
        c = self.cfg
        try:
            live = {(p["symbol"], p["side"]): p for p in self.bx.positions()}
        except BingXError as e:
            log.warning("manage: positions fallo: %s", e)
            return
        for sym, tr in list(self.trades.items()):
            pos = live.get((sym, tr["side"]))
            if pos is None:
                if time.time() - tr["opened"] > 20:      # margen para que aparezca tras abrir
                    self._finish_trade(sym, tr, "SL/TP")
                continue
            try:
                if c.max_bars and (time.time() - tr["opened"]) / (c.bar_ms / 1000) >= c.max_bars:
                    self.close_trade(sym, f"stop por tiempo ({c.max_bars} velas)")
                    continue
                inf = self.info.get(sym, {"qp": 0, "pp": 6})
                sgn = 1 if tr["side"] == "LONG" else -1
                if tr.get("n1") and not tr["tp1_done"] and pos["amt"] <= tr["qty"] - tr["n1"] * 0.999:
                    tr["tp1_done"] = True
                    self.tg.send(f"🎯 TP1 alcanzado {esc(sym)}")
                    if c.use_be:
                        be = tr["entry"] + sgn * tr["entry"] * (2 * c.fee_pct / 100.0)
                        if self._replace_sl(sym, tr, be, round_down(pos["amt"], inf["qp"])):
                            self.tg.send(f"🔒 SL a break-even {esc(sym)}")
                    self._save_trades()
                if tr["tp1_done"] and c.use_trail and time.time() - tr["trail_ts"] > 60:
                    mark = pos["mark"] or self.bx.price(sym)
                    new_sl = mark - sgn * c.trail_atr * tr["atr"]
                    better = (new_sl - tr["sl"]) * sgn >= 0.1 * tr["atr"]
                    safe = (mark - new_sl) * sgn > 0
                    if better and safe and self._replace_sl(sym, tr, new_sl, round_down(pos["amt"], inf["qp"])):
                        tr["trail_ts"] = time.time()
                        self._save_trades()
            except BingXError as e:
                log.warning("manage %s: %s", sym, e)

    # ------------------------------------------------------------------ bucle
    def heartbeat_if_due(self):
        c = self.cfg
        if c.heartbeat_hours <= 0 or time.time() - self._last_hb < c.heartbeat_hours * 3600:
            return
        self._last_hb = time.time()
        f = self.funnel_total
        eq = ""
        if self.trading:
            try:
                eq = f"\nEquity: {self.bx.balance()['equity']:.2f} USDT"
            except BingXError:
                pass
        self.tg.send(f"💓 LCE [{self.label()}] · ciclos {self.cycles} · señales {self.signals_sent} · abiertas {len(self.trades)}"
                     f"{eq}\nEmbudo: ok {f['ok']} · cand {f['cand']} · cooldown {f['cooldown']} · "
                     f"sin_espacio {f['bloq_sin_espacio']} · errores {f['error']} · viejos {f['datos_viejos']}")
        self.funnel_total = Counter()

    def run(self):
        c = self.cfg
        self.startup()
        bar_s = c.bar_ms / 1000.0
        nxt = (int(time.time() // bar_s) + 1) * bar_s + c.scan_delay
        if c.scan_on_start:
            nxt = time.time()
        last_manage = 0.0
        while True:
            try:
                now = time.time()
                if now >= nxt:
                    self.scan()
                    nxt = (int(time.time() // bar_s) + 1) * bar_s + c.scan_delay
                if self.trading and time.time() - last_manage >= c.manage_interval:
                    self.manage()
                    last_manage = time.time()
                self.heartbeat_if_due()
            except Exception as e:                       # noqa: BLE001 - el bucle no debe morir
                log.exception("error en bucle principal")
                if time.time() - self._last_err_msg > 900:
                    self._last_err_msg = time.time()
                    self.tg.send(f"⚠️ Error en el bot: {esc(repr(e)[:300])}")
            time.sleep(1.0)


def main():
    cfg = Config()
    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    Bot(cfg).run()


if __name__ == "__main__":
    main()
