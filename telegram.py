"""Envío de mensajes a Telegram (Bot API)."""
from __future__ import annotations

import asyncio
import html
import logging
import re
from typing import Tuple

import aiohttp

log = logging.getLogger("telegram")

_TAG = re.compile(r"<[^>]+>")


def _plain(text: str) -> str:
    return html.unescape(_TAG.sub("", text))


class Telegram:
    def __init__(self, session: aiohttp.ClientSession, token: str, chat_ids: Tuple[str, ...],
                 dry_run: bool = False):
        self.session = session
        self.token = token
        self.chat_ids = chat_ids
        self.dry_run = dry_run or not token or not chat_ids
        if not dry_run and (not token or not chat_ids):
            log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID no configurados: modo consola (dry-run).")

    async def send(self, text: str) -> bool:
        if self.dry_run:
            print("\n" + "=" * 60 + "\n" + _plain(text) + "\n" + "=" * 60, flush=True)
            return True
        ok_any = False
        for chat in self.chat_ids:
            ok_any = (await self._send_one(chat, text)) or ok_any
            await asyncio.sleep(1.1)  # Telegram: ~1 mensaje/segundo por chat
        return ok_any

    async def _send_one(self, chat_id: str, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        for attempt in range(4):
            try:
                async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    data = await r.json(content_type=None)
                    if r.status == 200 and data.get("ok"):
                        return True
                    if r.status == 429:
                        wait = int(data.get("parameters", {}).get("retry_after", 5)) + 1
                        log.warning("Telegram 429: esperando %ss", wait)
                        await asyncio.sleep(wait)
                        continue
                    if r.status == 400 and "parse" in str(data.get("description", "")).lower():
                        payload = {**payload, "text": _plain(text)}
                        payload.pop("parse_mode", None)
                        continue
                    log.error("Telegram error %s: %s", r.status, data)
                    return False
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                log.warning("Telegram fallo de red (%s), reintento", e)
                await asyncio.sleep(2 * (attempt + 1))
        return False
