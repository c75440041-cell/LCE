"""Cliente de la API PÚBLICA de BingX (perpetuos USDT-M). No necesita API keys."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from models import Candle

log = logging.getLogger("bingx")

BASE_URL = "https://open-api.bingx.com"
RATE_LIMIT_CODES = {100410, 100413}


class BingXError(Exception):
    pass


class RateLimiter:
    """Espacia las peticiones para no superar N por segundo."""

    def __init__(self, rps: float):
        self.interval = 1.0 / max(rps, 0.1)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)


def _parse_kline(x: Any) -> Optional[Candle]:
    try:
        if isinstance(x, dict):
            c = Candle(int(x["time"]), float(x["open"]), float(x["high"]),
                       float(x["low"]), float(x["close"]), float(x.get("volume", 0) or 0))
        else:  # formato lista [time, open, high, low, close, volume]
            c = Candle(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]),
                       float(x[5]) if len(x) > 5 else 0.0)
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    if c.h < c.l or c.c <= 0 or c.o <= 0:
        return None
    return c


class BingXClient:
    def __init__(self, session: aiohttp.ClientSession, rps: float = 10.0, base_url: str = BASE_URL):
        self.session = session
        self.base_url = base_url
        self.limiter = RateLimiter(rps)

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Any:
        url = self.base_url + path
        last_err = "?"
        for attempt in range(retries):
            await self.limiter.wait()
            q = dict(params or {})
            q["timestamp"] = int(time.time() * 1000)
            try:
                async with self.session.get(url, params=q, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    if r.status == 429:
                        last_err = "HTTP 429"
                        await asyncio.sleep(2 ** attempt)
                        continue
                    text = await r.text()
                    if r.status >= 500:
                        last_err = f"HTTP {r.status}"
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    data = json.loads(text)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            code = data.get("code", 0)
            if code == 0:
                return data.get("data")
            if code in RATE_LIMIT_CODES:
                last_err = f"rate limit code={code}"
                await asyncio.sleep(2 ** attempt)
                continue
            raise BingXError(f"{path} code={code} msg={data.get('msg')}")
        raise BingXError(f"{path} falló tras {retries} intentos ({last_err})")

    async def contracts(self) -> List[Dict[str, Any]]:
        data = await self._get("/openApi/swap/v2/quote/contracts")
        return data if isinstance(data, list) else []

    async def tickers(self) -> Dict[str, float]:
        """Volumen 24h en USDT por símbolo (una sola petición para todos)."""
        data = await self._get("/openApi/swap/v2/quote/ticker")
        if isinstance(data, dict):
            data = [data]
        out: Dict[str, float] = {}
        for d in data or []:
            try:
                sym = str(d["symbol"]).upper()
                qv = d.get("quoteVolume")
                if qv in (None, ""):
                    qv = float(d.get("volume", 0) or 0) * float(d.get("lastPrice", 0) or 0)
                out[sym] = float(qv)
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def klines(self, symbol: str, interval: str, limit: int = 500,
                     start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Candle]:
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        data = await self._get("/openApi/swap/v3/quote/klines", params)
        by_t: Dict[int, Candle] = {}
        for x in data or []:
            c = _parse_kline(x)
            if c is not None:
                by_t[c.t] = c
        return [by_t[k] for k in sorted(by_t)]
