"""Punto de entrada.

  python main.py                      -> bot en marcha (Railway)
  python main.py --once --dry-run     -> un análisis ahora mismo, imprime señales en consola
  python main.py --test-telegram      -> envía un mensaje de prueba a Telegram
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from dataclasses import replace

import aiohttp

try:  # opcional: leer .env en local
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from bingx import BingXClient
from config import load_config
from messages import format_startup
from scanner import Scanner
from telegram import Telegram
from tracker import Tracker


async def amain(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.dry_run:
        cfg = replace(cfg, dry_run=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    log = logging.getLogger("main")

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=50),
        headers={"User-Agent": "bingx-cbc-bot/2.0"},
    ) as session:
        tg = Telegram(session, cfg.telegram_token, cfg.telegram_chat_ids, cfg.dry_run)

        if args.test_telegram:
            ok = await tg.send("✅ <b>Prueba</b>: el bot MapleStax CBC puede escribir en este chat.")
            log.info("Mensaje de prueba %s", "enviado" if ok else "FALLÓ")
            return

        bingx = BingXClient(session, cfg.requests_per_second)
        tracker = Tracker(cfg)
        scanner = Scanner(cfg, bingx, tg, tracker)

        # Reintenta el arranque (BingX / red) en vez de morir
        for attempt in range(1, 6):
            try:
                await scanner.refresh_universe()
                break
            except Exception as e:
                log.error("Arranque intento %d/5 falló: %s", attempt, e)
                if attempt == 5:
                    raise
                await asyncio.sleep(10 * attempt)

        if args.once:
            await scanner.scan_once(once=True)
            return

        n_c, oil, n_other = scanner.universe_summary()
        await tg.send(format_startup(cfg, n_c, oil, n_other))
        log.info("Bot en marcha. LTF=%s HTF=%s", cfg.ltf, cfg.htf)

        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(scanner.run())
        for sig in (signal.SIGINT, signal.SIGTERM):  # Railway envía SIGTERM al redeploy
            try:
                loop.add_signal_handler(sig, task.cancel)
            except NotImplementedError:  # Windows
                pass
        try:
            await task
        except asyncio.CancelledError:
            log.info("Apagando…")
        finally:
            tracker.save()


def main() -> None:
    ap = argparse.ArgumentParser(description="Bot de señales MapleStax CBC para BingX -> Telegram")
    ap.add_argument("--once", action="store_true", help="analiza una vez y termina")
    ap.add_argument("--dry-run", action="store_true", help="no envía a Telegram, imprime en consola")
    ap.add_argument("--test-telegram", action="store_true", help="envía un mensaje de prueba")
    args = ap.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
