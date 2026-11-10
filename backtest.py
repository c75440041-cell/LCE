"""Backtest con el MISMO motor que el bot en vivo (sin mirar al futuro) y con su misma SELECCIÓN TOP-N.

En cada cierre de vela reúne las señales de TODOS los pares, las ordena por ranking y abre solo
las TOP_N mejores (igual que el bot en vivo). Con --compare enfrenta tres formas de elegir sobre
exactamente las mismas señales:  ranking (TOP_N mejores)  vs  azar (TOP_N al azar)  vs  todas.
Si "ranking" no gana claramente a "azar", el ranking no está aportando nada: ajusta RANK_WEIGHTS.

Ejemplos:
  python backtest.py --top 30 --days 14 --compare          # ¿el ranking aporta ventaja?
  python backtest.py --top 40 --days 14                    # TOP_N por ranking (TOP_N del entorno)
  python backtest.py --top 40 --days 14 --top-n 2 --min-rank 60
  python backtest.py --symbols BTC-USDT,ETH-USDT,SOL-USDT --days 14 --mode all
  STRATEGY=flip LTF=5m HTF=15m python backtest.py --top 30 --days 14   # estrategia original

Las variables de entorno (LTF, HTF, MIN_SCORE, TOP_N, MIN_RANK, TP_R, MAX_SL_ATR...) se respetan,
así puedes probar distintas configuraciones antes de usar el bot.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import aiohttp

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from bingx import BingXClient
from config import Config, load_config
from models import Candle, Trade
from strategy import MIN_BARS, Series, build_series, cbc_against, evaluate, market_bias, select_top
from tracker import close_trade_at, cooldown_active, new_trade, step_trade
from universe import build_universe


@dataclass
class SymData:
    symbol: str
    candles: List[Candle]
    S: Series
    idx: Dict[int, int]  # hora de apertura de la vela -> posición
    liq: float = 0.0     # volumen 24h en USDT (solo para el ranking)


def make_symdata(symbol: str, candles: List[Candle], cfg: Config, liq: float = 0.0) -> SymData:
    return SymData(symbol, candles, build_series(candles, cfg), {c.t: k for k, c in enumerate(candles)}, liq)


async def fetch_history(bingx: BingXClient, symbol: str, interval: str, ltf_ms: int, days: float) -> List[Candle]:
    """Descarga `days` días paginando hacia atrás (máx. 1440 velas por petición)."""
    now_ms = int(time.time() * 1000)
    target = now_ms - int(days * 86_400_000) - 500 * ltf_ms  # + calentamiento de indicadores
    end = now_ms
    out: Dict[int, Candle] = {}
    for _ in range(60):
        chunk = await bingx.klines(symbol, interval, 1440, end_time=end)
        if not chunk:
            break
        for c in chunk:
            out[c.t] = c
        first = chunk[0].t
        if first <= target or first >= end:
            break
        end = first - 1
    candles = [out[k] for k in sorted(out)]
    return [c for c in candles if c.t + ltf_ms <= now_ms]  # solo velas cerradas


def simulate(data: Dict[str, SymData], cfg: Config, fee_rate: float, eval_from_ms: int,
             mode: str = "rank", seed: int = 1, market: Optional[SymData] = None) -> Tuple[List[dict], dict]:
    """Simula el bot en vivo sobre varios pares a la vez, vela a vela.

    En cada cierre: 1) avanza las operaciones abiertas (SL / TP / FLIP / tiempo máx.), 2) evalúa cada
    par, 3) elige con `select_top` (rank | random | all) y abre las elegidas al cierre de la vela.
    Si en una vela se tocan SL y TP se asume SL primero (conservador).
    """
    ltf_ms = cfg.ltf_sec * 1000
    rng = random.Random(seed)
    active: Dict[str, Trade] = {}
    ranks: Dict[str, float] = {}
    last_close: Dict[str, int] = {}
    trades: List[dict] = []
    info = {"candidatas": 0, "elegidas": 0, "bajo_ranking": 0, "recortadas": 0}

    def record(sym: str, tr: Trade, ev, end_ms: int) -> None:
        fee_r = 2.0 * fee_rate * tr.entry / tr.risk  # comisión ida y vuelta en R
        trades.append({"symbol": sym, "side": tr.side, "t": tr.t_signal, "kind": ev.kind,
                       "r": ev.total_r - fee_r, "gross_r": ev.total_r, "rank": ranks.get(sym, 0.0)})
        if ev.total_r <= 0.05:  # cierre no ganador -> el par espera COOLDOWN_BARS velas
            last_close[sym] = end_ms

    all_ts = sorted({t for d in data.values() for t in d.idx if t >= eval_from_ms})
    for ts in all_ts:
        end_ms = ts + ltf_ms
        bias = 0
        if cfg.market_bias and market is not None and ts in market.idx:
            bias = market_bias(market.S, market.idx[ts])
        cands = []
        for sym, d in data.items():
            i = d.idx.get(ts)
            if i is None:
                continue
            c = d.candles[i]
            tr = active.get(sym)
            if tr is not None:
                final = None
                for ev in step_trade(tr, c, cfg):
                    if ev.final:
                        final = ev
                if final is None and cfg.max_trade_hours > 0 and \
                        (end_ms - tr.opened_ms) >= cfg.max_trade_hours * 3_600_000:
                    final = close_trade_at(tr, c.c, "TIMEOUT")
                if final is None and cfg.strategy == "pullback" and cfg.exit_on_cbc_flip \
                        and cbc_against(tr.side, d.S.cbc[i]):
                    final = close_trade_at(tr, c.c, "FLIP")  # el CBC cambió de color en contra
                if final is not None:
                    record(sym, tr, final, end_ms)
                    del active[sym]
                    tr = None
            if i < max(MIN_BARS, 3):
                continue
            sig, _ = evaluate(d.S, i, cfg, sym, liq=d.liq, bias=bias)
            if sig is None:
                continue
            if cooldown_active(last_close.get(sym), sig.t, ltf_ms, cfg.cooldown_bars):
                continue
            if tr is not None:
                if tr.side == sig.side or not cfg.exit_on_opposite:
                    continue
                record(sym, tr, close_trade_at(tr, sig.entry, "OPPOSITE"), end_ms)
                del active[sym]
            cands.append(sig)

        chosen, low, cut = select_top(cands, cfg, mode, rng)
        info["candidatas"] += len(cands)
        info["elegidas"] += len(chosen)
        info["bajo_ranking"] += low
        info["recortadas"] += cut
        for sig in chosen:
            active[sig.symbol] = new_trade(sig, ltf_ms)
            ranks[sig.symbol] = sig.rank
    return trades, info


def backtest_symbol(candles: List[Candle], cfg: Config, symbol: str, fee_rate: float,
                    eval_from_ms: int) -> List[dict]:
    """Backtest de un solo par (todas las señales, sin límite TOP-N)."""
    d = make_symdata(symbol, candles, cfg)
    return simulate({symbol: d}, cfg, fee_rate, eval_from_ms, mode="all")[0]


def summarize(trades: List[dict], title: str = "RESULTADO", info: Optional[dict] = None) -> None:
    print(f"\n================ {title} (neto de comisiones) ================")
    if info:
        print(f"Señales candidatas: {info['candidatas']}  ·  elegidas: {info['elegidas']}  ·  "
              f"bajo ranking mínimo: {info['bajo_ranking']}  ·  fuera del TOP: {info['recortadas']}")
    if not trades:
        print("Sin operaciones.")
        return
    rs = [t["r"] for t in trades]
    wins = [r for r in rs if r > 0.05]
    losses = [r for r in rs if r < -0.05]
    gp, gl = sum(r for r in rs if r > 0), -sum(r for r in rs if r < 0)
    eq = peak = dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    print(f"Operaciones:        {len(rs)}   (LONG {sum(t['side']=='LONG' for t in trades)} / SHORT {sum(t['side']=='SHORT' for t in trades)})")
    print(f"Ganadoras/Perdedoras: {len(wins)} / {len(losses)}   Winrate: {100*len(wins)/len(rs):.1f}%")
    print(f"Expectativa:        {sum(rs)/len(rs):+.3f} R por operación")
    print(f"Total:              {sum(rs):+.2f} R")
    print(f"Profit factor:      {gp/gl:.2f}" if gl > 0 else "Profit factor:      n/a")
    print(f"Máx. drawdown:      {dd:.2f} R")
    kinds: Dict[str, int] = {}
    for t in trades:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print("Cierres por tipo:   " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    by_sym: Dict[str, float] = {}
    for t in trades:
        by_sym[t["symbol"]] = by_sym.get(t["symbol"], 0.0) + t["r"]
    best = sorted(by_sym.items(), key=lambda x: x[1], reverse=True)
    print("Mejores símbolos:   " + ", ".join(f"{s} {r:+.1f}R" for s, r in best[:5]))
    print("Peores símbolos:    " + ", ".join(f"{s} {r:+.1f}R" for s, r in best[::-1][:5]))
    for side in ("LONG", "SHORT"):
        sr = [t["r"] for t in trades if t["side"] == side]
        if sr:
            print(f"  {side:5s}: {len(sr)} ops · {sum(sr):+.1f}R · {sum(sr)/len(sr):+.3f} R/op · "
                  f"winrate {100*sum(r > 0.05 for r in sr)/len(sr):.0f}%")
    buckets = [(0, 60), (60, 70), (70, 80), (80, 101)]
    parts = []
    for lo, hi in buckets:
        br = [t["r"] for t in trades if lo <= t.get("rank", 0.0) < hi]
        if br:
            parts.append(f"{lo}-{min(hi, 100)}: {len(br)} ops {sum(br)/len(br):+.3f}R")
    if parts:
        print("R medio por ranking: " + " | ".join(parts))
    print("================================================================")


async def amain(args: argparse.Namespace) -> None:
    cfg: Config = load_config()
    cfg = replace(cfg, dry_run=True)
    if args.top_n is not None:
        cfg = replace(cfg, top_n=max(0, args.top_n))
    if args.min_rank is not None:
        cfg = replace(cfg, min_rank=max(0.0, min(100.0, args.min_rank)))
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    ltf_ms = cfg.ltf_sec * 1000
    async with aiohttp.ClientSession(headers={"User-Agent": "bingx-cbc-bot/2.0"}) as session:
        bingx = BingXClient(session, cfg.requests_per_second)
        try:
            volumes = await bingx.tickers()
        except Exception as e:
            print(f"Aviso: sin volumen 24h ({e}); la liquidez puntuará neutra")
            volumes = {}
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            uni = build_universe(await bingx.contracts(), volumes, cfg)
            symbols = list(uni)[: args.top]
        print(f"Backtest {len(symbols)} símbolos · {args.days} días · estrategia {cfg.strategy} · "
              f"LTF {cfg.ltf} / HTF {cfg.htf} · MIN_SCORE {cfg.min_score} · TOP_N {cfg.top_n} · "
              f"MIN_RANK {cfg.min_rank:g} · comisión {args.fee_bps} bps/lado")
        eval_from = int(time.time() * 1000) - int(args.days * 86_400_000)
        data: Dict[str, SymData] = {}
        for n, sym in enumerate(symbols, 1):
            try:
                candles = await fetch_history(bingx, sym, cfg.ltf, ltf_ms, args.days)
            except Exception as e:
                print(f"[{n}/{len(symbols)}] {sym}: error {e}")
                continue
            if len(candles) < MIN_BARS + 50:
                print(f"[{n}/{len(symbols)}] {sym}: pocas velas ({len(candles)})")
                continue
            data[sym] = make_symdata(sym, candles, cfg, volumes.get(sym, 0.0))
            print(f"[{n}/{len(symbols)}] {sym}: {len(candles)} velas")
        market = data.get(cfg.market_symbol)
        if cfg.market_bias and market is None:
            try:
                mc = await fetch_history(bingx, cfg.market_symbol, cfg.ltf, ltf_ms, args.days)
                if len(mc) >= MIN_BARS + 50:
                    market = make_symdata(cfg.market_symbol, mc, cfg)
            except Exception as e:
                print(f"Aviso: sin datos de {cfg.market_symbol} ({e}); sin sesgo de mercado")
    if not data:
        print("Sin datos.")
        return
    fee = args.fee_bps / 10_000.0
    if args.compare:
        label = {"rank": f"TOP {cfg.top_n} POR RANKING", "random": f"TOP {cfg.top_n} AL AZAR",
                 "all": "TODAS LAS SEÑALES (sin límite)"}
        for mode in ("rank", "random", "all"):
            tr, info = simulate(data, cfg, fee, eval_from, mode, args.seed, market)
            summarize(tr, label[mode], info)
        print("\nLectura: si 'POR RANKING' no supera a 'AL AZAR' en expectativa (R/op) con bastantes "
              "operaciones, el ranking no aporta ventaja con estos datos.")
    else:
        tr, info = simulate(data, cfg, fee, eval_from, args.mode, args.seed, market)
        summarize(tr, f"RESULTADO · modo {args.mode}", info)
    print("Aviso: el pasado no garantiza resultados futuros. Si en una vela se tocan SL y TP,")
    print("se asume SL primero (conservador). No incluye slippage ni funding. La liquidez usa el")
    print("volumen 24h ACTUAL (no el de la época de cada señal).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest de la estrategia CBC + selección TOP-N en BingX")
    ap.add_argument("--symbols", default="", help="lista separada por comas, p.ej. BTC-USDT,ETH-USDT")
    ap.add_argument("--top", type=int, default=30, help="si no hay --symbols: top N pares por volumen")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--fee-bps", type=float, default=5.0, help="comisión por lado en puntos básicos (5 = 0.05%)")
    ap.add_argument("--top-n", type=int, default=None, help="señales por cierre de vela (por defecto TOP_N del entorno)")
    ap.add_argument("--min-rank", type=float, default=None, help="ranking mínimo 0-100 (por defecto MIN_RANK)")
    ap.add_argument("--mode", choices=("rank", "random", "all"), default="rank",
                    help="rank = TOP_N mejores · random = TOP_N al azar · all = todas")
    ap.add_argument("--compare", action="store_true", help="compara ranking vs azar vs todas")
    ap.add_argument("--seed", type=int, default=1, help="semilla del modo random")
    asyncio.run(amain(ap.parse_args()))
