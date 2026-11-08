"""LCE bot: escanea BingX (perpetuos USDT) cada vela de 5m, envia senales a Telegram
y (segun TRADE_MODE) opera en demo/live. Un solo hilo ejecuta ordenes; el
analisis de simbolos se paraleliza (solo lectura)."""
import logging
import math
import sys
import time
from datetime import datetime, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import indicators as ind
from . import strategy
from .bingx import BingX, BingXError
from .config import Config, INTERVAL_MS
from .risk import RiskGuard, fmt_num, round_down, size_position
from .state import State
from .tracker import History, Stats, fmt_wr
from .telegram import Telegram, esc

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
        self.state = State(c.state_dir, name=c.state_file)
        self.risk = RiskGuard(c, self.state)
        # si cambia la configuracion de la estrategia, las estadisticas en vivo dejan de ser comparables: se reinician
        meta, fp = self.state.get("meta", {}), c.fingerprint()
        self.cfg_changed = None
        if meta.get("fp") and meta["fp"] != fp:
            old = Stats(self.state, c.fee_pct + c.slip_pct)
            self.cfg_changed = old.summary()
            self.state.put("stats_prev", old.s)
            self.state.put("stats", {})
        if meta.get("fp") != fp:
            self.state.put("meta", {"fp": fp})
        self.stats = Stats(self.state, c.fee_pct + c.slip_pct)
        self.hist = History(c.fee_pct + c.slip_pct)
        self._hist_done = False
        self._hist_syms = 0
        self.trades = self.state.get("trades", {})
        self.sig_ts = self.state.get("signals", {})
        self.info, self.vol, self.universe = {}, {}, []
        self._uni_exp = 0.0
        self._htf_cache = {}
        self._lev_done, self._mt_done = set(), set()
        self.funnel_total = Counter()
        self.skipped = Counter()
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
            self.bx.balance()                         # valida credenciales ya
            try:
                if not self.bx.position_mode():
                    log.warning("Cuenta en modo one-way: activo hedge (requiere sin posiciones abiertas)")
                    self.bx.set_position_mode(True)
            except BingXError as e:
                log.error("No pude fijar modo hedge: %s", e)
            self.reconcile()
        self.refresh_universe(force=True)
        self.tg.send(
            f"🤖 <b>LCE bot iniciado</b> [{self.label()}]\n"
            f"TF {c.timeframe} · HTF {c.htf if c.use_htf else 'off'} · {len(self.universe)} símbolos\n"
            f"Endpoint: {esc(self.bx.base)}\n"
            f"Riesgo {c.risk_pct}% · Lev {c.leverage}x · Máx pos {c.max_positions} ({c.max_per_side}/sentido)\n"
            f"Protecciones: régimen {c.regime_symbol if c.regime_filter else 'off'} · pérdida diaria {c.max_day_loss}% · "
            f"pausa tras {c.max_cons_loss} pérdidas · kill_switch {'ON' if c.kill_switch else 'off'}\n"
            f"Preset: {c.preset} · config {c.fingerprint()}\n"
            + (f"Variables fijadas: {', '.join(c.active_env()[:14])}\n" if c.active_env() else "Variables de estrategia: todas por defecto\n")
            +
            f"Estado persistente: {'sí' if self.state.persistent else 'NO (monta un Volume)'}"
            + (f"\n♻️ Configuración cambiada: estadísticas en vivo reiniciadas.\nAnteriores: {self.cfg_changed}"
               if self.cfg_changed else ""))

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
                return sym, "pocas_velas", None, None
            if int(time.time() * 1000) - (int(d["t"][-1]) + c.bar_ms) > 2 * c.bar_ms:
                return sym, "datos_viejos", None, None
            htf = self._get_htf(sym) if c.use_htf else None
            mt = 10.0 ** (-self.info.get(sym, {}).get("pp", 8))
            return sym, "ok", strategy.run(c, d, htf, mintick=mt), d
        except Exception as e:                        # un simbolo malo no tumba el ciclo
            log.warning("analisis %s fallo: %s", sym, e)
            return sym, "error", None, None

    # ------------------------------------------------------------------ escaneo
    def scan(self):
        """Analiza en paralelo y procesa cada simbolo en cuanto termina (las senales entran antes)."""
        c = self.cfg
        t0 = time.time()
        self.refresh_universe()
        funnel = Counter()
        events = []
        extra = set(self.trades) | self.stats.open_symbols()
        syms = list(self.universe) + [x for x in extra if x not in self.universe and x in self.info]
        with ThreadPoolExecutor(max_workers=c.workers) as ex:
            futs = [ex.submit(self.analyze_symbol, s_) for s_ in syms]
            for fut in as_completed(futs):
                sym, status, an, d = fut.result()
                funnel[status] += 1
                if status != "ok":
                    continue
                events.extend(self.stats.update(sym, d, c.signal_timeout_bars))
                if not self._hist_done:
                    self._hist_syms += 1
                    for hs in an.signals:
                        self.hist.add(hs, d, an.n, c.signal_timeout_bars)
                dg = an.diag
                funnel["brk"] += bool(dg.get("brk"))
                funnel["rev"] += bool(dg.get("rev"))
                funnel["cand"] += bool(dg.get("cand"))
                funnel["cooldown"] += bool(dg.get("cooldown"))
                if dg.get("blocked"):
                    funnel["bloq_" + dg["blocked"]] += 1
                try:
                    if c.use_struct_exit and sym in self.trades:
                        tr = self.trades[sym]
                        if (tr["side"] == "LONG" and an.struct_break_dn) or (tr["side"] == "SHORT" and an.struct_break_up):
                            self.close_trade(sym, "estructura rota en contra")
                    if c.use_trail and sym in self.trades:
                        self.trail_trade(sym, an)
                    if an.signals and an.signals[-1].idx == an.n - 1:
                        funnel["senales"] += 1
                        self.handle_signal(sym, an.signals[-1])
                except Exception:                    # noqa: BLE001 - un simbolo no tumba el ciclo
                    log.exception("procesando %s", sym)
        self.send_events(events)
        if not self._hist_done:
            self._hist_done = True
            self.tg.send(self.hist.text(c.candles * c.bar_ms / 86_400_000, self._hist_syms))
        self.cycles += 1
        self.funnel_total.update(funnel)
        log.info("Ciclo %d en %.1fs | %s", self.cycles, time.time() - t0, dict(funnel))

    # ------------------------------------------------------------------ senales
    def send_events(self, events):
        if not events:
            return
        def one(e):
            ico = {"win": "✅", "loss": "❌", "expired": "⌛"}[e["result"]]
            txt = {"win": "TP alcanzado", "loss": "SL alcanzado", "expired": "expirada sin resolver"}[e["result"]]
            r = f" {e['r']:+.2f}R" if e["result"] != "expired" else ""
            return f"{ico} {esc(e['sym'])} {e['side']} {e['kind']} · {txt}{r}"
        lines = [one(e) for e in events]
        if len(lines) > 6:
            lines = lines[:6] + [f"… y {len(lines) - 6} más"]
        self.tg.send("\n".join(lines) + "\n📊 " + self.stats.summary())

    def format_signal(self, sym, sig):
        c = self.cfg
        pp = self.info.get(sym, {}).get("pp", 6)
        pct = lambda x: abs(x - sig.entry) / sig.entry * 100
        names = {"BO": "Ruptura", "RV": "Reversión en zona", "SW": "Barrido de liquidez"}
        when = datetime.fromtimestamp((sig.time + c.bar_ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        arrow = "🟢" if sig.side == "LONG" else "🔴"
        lines = [f"{arrow} <b>{sig.side}</b> · <b>{esc(sym)}</b> · {c.timeframe} [{self.label()}]",
                 f"Tipo: {names[sig.kind]} ({sig.kind})" + (f" · puntos {sig.score}" if c.use_score or c.size_by_score else ""),
                 f"🕒 {when}",
                 f"Entrada: <code>{fmt_num(sig.entry, pp)}</code>",
                 f"🛑 SL: <code>{fmt_num(sig.sl, pp)}</code> (-{pct(sig.sl):.2f}%)",
                 f"🎯 TP: <code>{fmt_num(sig.tp, pp)}</code> (+{pct(sig.tp):.2f}%)"]
        if sig.tp1:
            lines.append(f"🎯 TP1: <code>{fmt_num(sig.tp1, pp)}</code> ({c.tp1_pct:.0f}% de la posición)")
        lines.append(f"R:R {sig.rr:.2f} · ATR {fmt_num(sig.atr, pp)}")
        lines.append(f"HTF {c.htf}: {sig.htf} · Vol rel {sig.rel_vol:.1f}x · ADX {sig.adx:.0f}")
        lines.append(f"Riesgo {c.risk_pct * sig.risk_mult:.2f}% del equity" + (f" · Lev {c.leverage}x" if self.trading else ""))
        lines.append("📊 " + self.stats.summary())
        h = self.hist
        if h.w + h.l:
            lines.append(f"🧪 Backtest reciente: {fmt_wr(h.w, h.l)} · R neto {h.nw - h.nl:+.1f}")
        return "\n".join(lines)

    def handle_signal(self, sym, sig):
        if sig.time <= self.sig_ts.get(sym, 0):
            return
        self.sig_ts[sym] = sig.time
        self.state.put("signals", self.sig_ts)
        self.signals_sent += 1
        self.stats.add_open(sym, sig)
        msg = self.format_signal(sym, sig)
        self.tg.send(msg)
        if self.trading:
            try:
                self.open_trade(sym, sig)
            except BingXError as e:
                log.error("open_trade %s: %s", sym, e)
                self.tg.send(f"⛔ No ejecutada {esc(sym)}: {esc(e)}")

    # ------------------------------------------------------------------ ejecucion
    def _skip(self, sym, key, detail=""):
        """Operacion omitida por una proteccion: solo log + contador (el resumen va en el heartbeat)."""
        self.skipped[key] += 1
        log.info("%s no ejecutada: %s %s", sym, key, detail)

    def regime(self):
        """Regimen de BTC en el TF superior: bull / bear / neutral (None si no hay datos)."""
        c = self.cfg
        try:
            d = self._get_htf(c.regime_symbol)
        except BingXError:
            return None
        if d is None or len(d["t"]) < c.ema_slow + 2:
            return None
        ef, es = ind.ema(d["c"], c.ema_fast)[-1], ind.ema(d["c"], c.ema_slow)[-1]
        px = d["c"][-1]
        if ef > es and px > es:
            return "bull"
        if ef < es and px < es:
            return "bear"
        return "neutral"

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
            return self._skip(sym, "kill_switch")
        if sym in self.trades:
            log.info("%s ya tiene operacion abierta", sym)
            return
        if len(self.trades) >= c.max_positions:
            return self._skip(sym, "max_posiciones", f"({c.max_positions})")
        if c.max_per_side and sum(1 for t_ in self.trades.values() if t_["side"] == sig.side) >= c.max_per_side:
            return self._skip(sym, "max_por_sentido", f"({sig.side} {c.max_per_side})")
        if c.regime_filter:
            rg = self.regime()
            if rg is None:
                return self._skip(sym, "regimen_desconocido")
            if (sig.side == "LONG" and rg == "bear") or (sig.side == "SHORT" and rg == "bull"):
                return self._skip(sym, "contra_regimen_BTC", f"({sig.side} con BTC {rg})")
        bal = self.bx.balance()
        ok, why = self.risk.can_open(bal["equity"])
        if not ok:
            return self._skip(sym, "limite_riesgo", why)
        px = self.bx.price(sym)
        if px <= 0:
            return
        sl_pct = abs(px - sig.sl) / px * 100.0
        lims = [x for x in (c.max_sl_pct, c.max_sl_liq_frac * 100.0 / c.leverage if c.max_sl_liq_frac > 0 and c.leverage > 0 else 0) if x > 0]
        if lims and sl_pct > min(lims):
            return self._skip(sym, "sl_demasiado_lejos", f"(SL {sl_pct:.1f}% > {min(lims):.1f}%)")
        sgn = 1 if sig.side == "LONG" else -1
        risk_d = (px - sig.sl) * sgn
        reward = (sig.tp - px) * sgn
        if risk_d <= 0 or reward <= 0 or reward / risk_d < c.min_live_rr:
            return self._skip(sym, "precio_movido", f"(R:R real {reward / risk_d if risk_d > 0 else 0:.2f})")
        sz = size_position(bal["equity"], px, sig.sl, c.risk_pct * sig.risk_mult, c.max_lev, inf["qp"], inf["min_qty"], inf["min_usdt"])
        if not sz:
            return self._skip(sym, "tamano_minimo")
        if sz["notional"] / c.leverage > bal["available"] * 0.95:
            return self._skip(sym, "margen_insuficiente")
        qty = sz["qty"]
        n1 = 0.0
        if c.use_partial and sig.tp1:
            n1 = round_down(qty * c.tp1_pct / 100.0, inf["qp"])
            if n1 < inf["min_qty"] or qty - n1 < inf["min_qty"] or n1 <= 0:
                n1 = 0.0

        # apalancamiento en AMBOS lados (hedge) y margen aislado, una vez por simbolo
        if sym not in self._lev_done:
            for s in ("LONG", "SHORT"):
                try:
                    self.bx.set_leverage(sym, s, c.leverage)
                except BingXError as e:
                    log.warning("leverage %s %s: %s", sym, s, e)
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
            "risk_usdt": sz["risk_usdt"], "sl_id": sl_id, "tp_id": tp_id, "tp1_id": tp1_id, "tp1_done": False, "trail_ts": 0.0,
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

    def trail_trade(self, sym, an):
        """Trailing de Pine: SL = max(SL, hh3 - ATR*trailATR) (short: min(SL, ll3 + ATR*trailATR)), por vela cerrada."""
        c = self.cfg
        tr = self.trades.get(sym)
        if not tr or not self.trading or math.isnan(an.hh3) or math.isnan(an.atr):
            return
        if c.use_partial and tr.get("n1") and not tr["tp1_done"]:
            return
        if tr["side"] == "LONG":
            new = max(tr["sl"], an.hh3 - an.atr * c.trail_atr)
            ok = new > tr["sl"] and new < an.last_close
        else:
            new = min(tr["sl"], an.ll3 + an.atr * c.trail_atr)
            ok = new < tr["sl"] and new > an.last_close
        if not ok:
            return
        try:
            pos = [p for p in self.bx.positions(sym) if p["side"] == tr["side"]]
            if pos and self._replace_sl(sym, tr, new, round_down(pos[0]["amt"], self.info[sym]["qp"])):
                self._save_trades()
        except BingXError as e:
            log.warning("trail %s: %s", sym, e)

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
            self.risk.on_close(pnl, tr.get("risk_usdt", 0.0))
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
                        be = tr["entry"]
                        if self._replace_sl(sym, tr, be, round_down(pos["amt"], inf["qp"])):
                            self.tg.send(f"🔒 SL a break-even {esc(sym)}")
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
                     f"{eq}\n📊 {self.stats.summary()}\n🛡 Omitidas por protecciones: {dict(self.skipped) or 'ninguna'}\nEmbudo: ok {f['ok']} · cand {f['cand']} · cooldown {f['cooldown']} · "
                     f"sin_espacio {f['bloq_sin_espacio']} · errores {f['error']} · viejos {f['datos_viejos']}")
        self.funnel_total = Counter()
        self.skipped = Counter()

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


def idle(msg):
    """Tras un backtest/volcado el servicio queda en espera (evita reinicios en bucle)."""
    while True:
        log.info(msg)
        time.sleep(3600)


def main():
    cfg = Config()
    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    if cfg.strategy == "qs":
        from .qs_bot import QSBot
        bot_cls = QSBot
    else:
        bot_cls = Bot
    if cfg.run_mode in ("backtest", "dump"):
        if cfg.strategy == "qs":
            from . import qs_bot as bt_mod
            fns = (bt_mod.run_backtest, bt_mod.run_dump)
        else:
            from . import backtest as bt_mod
            fns = (bt_mod.run_sweep, bt_mod.run_dump)
        bot = bot_cls(cfg)
        try:
            (fns[0] if cfg.run_mode == "backtest" else fns[1])(bot)
        except Exception as e:                       # noqa: BLE001
            log.exception("fallo en %s", cfg.run_mode)
            bot.tg.send(f"⛔ Error en {cfg.run_mode}: {esc(repr(e)[:300])}")
        bot.tg.send("ℹ️ Terminado. Cambia RUN_MODE=bot en Railway para volver a las señales en vivo.")
        idle(f"RUN_MODE={cfg.run_mode} terminado; pon RUN_MODE=bot")
    else:
        bot_cls(cfg).run()


if __name__ == "__main__":
    main()
