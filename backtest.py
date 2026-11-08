"""Backtest de la estrategia con los MISMOS módulos que el bot (strategy.run).

Uso:
    python backtest.py --symbols BTC-USDT,ETH-USDT,SOL-USDT --days 30
    python backtest.py --top 20 --days 60 --slip 0.03

Descarga velas públicas de BingX (no necesita claves) y respeta tu configuración
(variables de entorno, igual que el bot). Costes: comisión FEE_PCT por lado + slippage.

Simplificaciones (los resultados reales serán algo peores):
  * Entrada al cierre de la vela de señal (+ slippage adverso).
  * Si SL y TP caen en la misma vela, se asume que salta el SL.
  * No simula parciales/BE/trailing (solo SL, TP y MAX_BARS); sin funding.
  * Posiciones simultáneas limitadas a MAX_POSITIONS, orden cronológico.
"""
import argparse
import math
import sys
import time

import numpy as np

import strategy
from bingx import BingX, BingXError
from config import Config
from main import to_dict


def fetch(bx, sym, interval, bar_ms, bars):
    """Pagina hacia atrás hasta reunir `bars` velas cerradas."""
    rows, end = {}, None
    for _ in range(math.ceil(bars / 1000) + 1):
        chunk = bx.klines(sym, interval, 1000, end_time=end)
        if not chunk:
            break
        for r in chunk:
            rows[r[0]] = r
        first = chunk[0][0]
        if end is not None and first >= end:
            break
        end = first - 1
        if len(rows) >= bars:
            break
    return to_dict([rows[k] for k in sorted(rows)][-bars:], bar_ms)


def simulate(cfg, d, signals, fee_pct, slip_pct):
    """Recorre las señales de un símbolo. Devuelve lista de trades (dicts)."""
    h, l, t = d["h"], d["l"], d["t"]
    n, out, free_from = len(t), [], -1
    for s in sorted(signals, key=lambda x: x.idx):
        if s.idx <= free_from or s.idx >= n - 1:
            continue
        sgn = 1 if s.side == "LONG" else -1
        entry = s.entry * (1 + sgn * slip_pct / 100.0)
        risk_d = (entry - s.sl) * sgn
        if risk_d <= 0 or (s.tp - entry) * sgn <= 0:
            continue
        exit_px, exit_i, why = None, None, None
        last = n - 1
        for j in range(s.idx + 1, n):
            hit_sl = l[j] <= s.sl if sgn > 0 else h[j] >= s.sl
            hit_tp = h[j] >= s.tp if sgn > 0 else l[j] <= s.tp
            if hit_sl:                                    # SL manda si coinciden
                exit_px, exit_i, why = s.sl * (1 - sgn * slip_pct / 100.0), j, "SL"
                break
            if hit_tp:
                exit_px, exit_i, why = s.tp, j, "TP"
                break
            if cfg.max_bars and j - s.idx >= cfg.max_bars:
                exit_px, exit_i, why = d["c"][j], j, "TIEMPO"
                break
        if exit_px is None:
            continue                                      # sigue abierta al final: se descarta
        move = (exit_px - entry) * sgn
        cost = (fee_pct / 100.0) * (entry + exit_px)      # comisión de entrada y salida
        net = move - cost
        notional_frac = min(cfg.risk_pct / 100.0 * entry / risk_d, cfg.max_lev)
        out.append({"t_in": int(t[s.idx]), "t_out": int(t[exit_i]), "side": s.side, "kind": s.kind,
                    "why": why, "r": net / risk_d, "ret": notional_frac * net / entry})
        free_from = exit_i
    return out


def portfolio(trades, max_positions):
    """Filtra por MAX_POSITIONS en orden cronológico y calcula la curva de capital."""
    kept, open_out = [], []
    for tr in sorted(trades, key=lambda x: x["t_in"]):
        open_out = [o for o in open_out if o > tr["t_in"]]
        if len(open_out) >= max_positions:
            continue
        open_out.append(tr["t_out"])
        kept.append(tr)
    eq, peak, mdd, curve = 1.0, 1.0, 0.0, []
    for tr in sorted(kept, key=lambda x: x["t_out"]):
        eq *= 1.0 + tr["ret"]
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        curve.append(eq)
    return kept, eq, mdd


def stats(trades):
    if not trades:
        return None
    r = np.array([x["r"] for x in trades])
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(r), "win": float((r > 0).mean() * 100), "avg_r": float(r.mean()),
            "pf": float(gp / gl) if gl > 0 else float("inf"), "best": float(r.max()), "worst": float(r.min())}


def report(title, trades):
    s = stats(trades)
    if not s:
        print(f"{title}: sin operaciones")
        return
    print(f"{title:<14} n={s['n']:<4} acierto={s['win']:5.1f}%  R medio={s['avg_r']:+.3f}  "
          f"PF={s['pf']:.2f}  mejor={s['best']:+.2f}R peor={s['worst']:+.2f}R")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--top", type=int, default=0, help="usar los N contratos con más volumen")
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--slip", type=float, default=0.03, help="slippage %% por lado (def. 0.03)")
    a = ap.parse_args()
    cfg = Config()
    bx = BingX(demo=False)
    bars = int(a.days * 86_400_000 / cfg.bar_ms)
    info = {}
    for ct in bx.contracts():
        sy = str(ct.get("symbol", ""))
        if sy.endswith("-USDT") and not any(sy.upper().startswith(p) for p in cfg.non_crypto_prefixes):
            info[sy] = int(ct.get("pricePrecision", 4) or 4)
    if a.symbols:
        syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    else:
        vol = {t.get("symbol"): float(t.get("quoteVolume") or 0) for t in bx.tickers()}
        syms = sorted((s for s in info), key=lambda s: -vol.get(s, 0))[:a.top or 10]
    print(f"Backtest {cfg.timeframe} · {a.days:g} días (~{bars} velas) · {len(syms)} símbolos · "
          f"fee {cfg.fee_pct}%/lado · slip {a.slip}%/lado · riesgo {cfg.risk_pct}% · lev máx {cfg.max_lev}x\n")
    allt = []
    for sy in syms:
        try:
            d = fetch(bx, sy, cfg.timeframe, cfg.bar_ms, bars)
            if d is None or len(d["t"]) < 200:
                print(f"{sy}: pocas velas, salto")
                continue
            htf = fetch(bx, sy, cfg.htf, cfg.htf_ms, int(bars * cfg.bar_ms / cfg.htf_ms) + 300)
            an = strategy.run(cfg, d, htf, mintick=10 ** -info.get(sy, 4))
        except (BingXError, KeyError, ValueError) as e:
            print(f"{sy}: error {e}")
            continue
        tr = simulate(cfg, d, an.signals, cfg.fee_pct, a.slip)
        allt += tr
        s = stats(tr)
        print(f"{sy:<14} {('n=%d R medio=%+.3f' % (s['n'], s['avg_r'])) if s else 'sin operaciones'}")
        time.sleep(0.2)
    kept, eq, mdd = portfolio(allt, cfg.max_positions)
    print("\n=== RESULTADO (cartera, con MAX_POSITIONS) ===")
    report("TOTAL", kept)
    for k in ("BO", "RV", "SW"):
        report(f"tipo {k}", [x for x in kept if x["kind"] == k])
    for sd in ("LONG", "SHORT"):
        report(sd, [x for x in kept if x["side"] == sd])
    print(f"\nRetorno compuesto: {(eq - 1) * 100:+.1f}%   Drawdown máximo: {mdd * 100:.1f}%")
    if len(kept) < 100:
        print("\n⚠ Menos de 100 operaciones: la muestra es demasiado pequeña para sacar conclusiones.")
    print("⚠ Un backtest favorable NO garantiza resultados futuros. Valida igualmente en signals/demo.")


if __name__ == "__main__":
    sys.exit(main())
