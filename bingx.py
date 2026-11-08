"""Cliente REST minimo de BingX USDT-M perpetuos (swap v2/v3).

Reglas aprendidas en la flota:
  * La URL base se decide en UN solo sitio (demo -> open-api-vst.bingx.com).
  * Firma: el MISMO string `urlencode(sorted(params.items()))` se firma y se envia.
  * Los POST/DELETE no se reintentan por timeout (evita ordenes duplicadas).
"""
import hashlib
import hmac
import logging
import threading
import time
from urllib.parse import urlencode

import requests

log = logging.getLogger("bingx")

LIVE_URL = "https://open-api.bingx.com"
DEMO_URL = "https://open-api-vst.bingx.com"


class BingXError(Exception):
    def __init__(self, code, msg, path=""):
        super().__init__(f"BingX {code} {msg} ({path})")
        self.code = code
        self.msg = msg
        self.path = path


def sign_query(params: dict, secret: str) -> str:
    """Devuelve el query string firmado (ordenado alfabeticamente)."""
    qs = urlencode(sorted(params.items()))
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


class BingX:
    def __init__(self, api_key="", api_secret="", demo=False, min_interval=0.11, timeout=15):
        self.key = api_key
        self.secret = api_secret
        self.demo = demo
        self.base = DEMO_URL if demo else LIVE_URL
        self.min_interval = min_interval
        self.timeout = timeout
        self._lock = threading.Lock()
        self._last = 0.0
        self.session = requests.Session()

    # ------------------------------------------------------------------ core
    def _throttle(self):
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _request(self, method, path, params=None, signed=False, retry=None):
        retry = (method == "GET") if retry is None else retry
        last_err = None
        for attempt in range(4):
            p = dict(params or {})
            headers = {}
            if signed:
                if not self.key or not self.secret:
                    raise BingXError(-1, "faltan BINGX_API_KEY / BINGX_API_SECRET", path)
                p["timestamp"] = int(time.time() * 1000)
                p["recvWindow"] = 5000
                qs = sign_query(p, self.secret)
                headers["X-BX-APIKEY"] = self.key
            else:
                qs = urlencode(sorted(p.items()))
            url = f"{self.base}{path}" + (f"?{qs}" if qs else "")
            self._throttle()
            try:
                r = self.session.request(method, url, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                if not retry:
                    raise BingXError(-2, f"red: {e}", path)
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            try:
                data = r.json()
            except ValueError:
                last_err = f"respuesta no JSON HTTP {r.status_code}"
                if r.status_code >= 500 and retry:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise BingXError(-3, str(last_err), path)
            code = data.get("code", 0)
            if code in (100410,):                      # rate limit
                time.sleep(2.0 * (attempt + 1))
                continue
            if code != 0:
                raise BingXError(code, data.get("msg", ""), path)
            return data.get("data")
        raise BingXError(-4, f"sin respuesta tras reintentos: {last_err}", path)

    # ------------------------------------------------------------------ mercado
    def contracts(self):
        return self._request("GET", "/openApi/swap/v2/quote/contracts") or []

    def tickers(self):
        d = self._request("GET", "/openApi/swap/v2/quote/ticker") or []
        return d if isinstance(d, list) else [d]

    def klines(self, symbol, interval, limit=1000, end_time=None):
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = int(end_time)
        data = self._request("GET", "/openApi/swap/v3/quote/klines", params) or []
        rows = {}
        for k in data:
            if isinstance(k, dict):
                r = (int(k["time"]), _f(k["open"]), _f(k["high"]), _f(k["low"]), _f(k["close"]), _f(k["volume"]))
            else:
                r = (int(k[0]), _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4]), _f(k[5]))
            rows[r[0]] = r
        return [rows[k] for k in sorted(rows)]

    def klines_range(self, symbol, interval, start_ms, end_ms=None, page=1000, max_pages=80):
        """Historial paginado hacia atras con endTime (devuelve velas >= start_ms, ascendente).
        Si el servidor ignorase endTime, corta en la primera pagina (el llamador detecta la cobertura)."""
        cursor = int(end_ms or time.time() * 1000)
        rows = {}
        for _ in range(max_pages):
            batch = self.klines(symbol, interval, page, end_time=cursor)
            if not batch:
                break
            for r in batch:
                rows[r[0]] = r
            first = batch[0][0]
            if first <= start_ms or first >= cursor:
                break
            cursor = first - 1
        return [rows[k] for k in sorted(rows) if k >= start_ms]

    def price(self, symbol):
        d = self._request("GET", "/openApi/swap/v2/quote/price", {"symbol": symbol})
        if isinstance(d, list):
            d = d[0] if d else {}
        return _f((d or {}).get("price"))

    # ------------------------------------------------------------------ cuenta
    def balance(self):
        d = self._request("GET", "/openApi/swap/v2/user/balance", signed=True) or {}
        b = d.get("balance", d) if isinstance(d, dict) else d
        if isinstance(b, list):
            b = next((x for x in b if str(x.get("asset", "USDT")).upper() == "USDT"), b[0] if b else {})
        equity = _f(b.get("equity"), _f(b.get("balance")) + _f(b.get("unrealizedProfit")))
        avail = _f(b.get("availableMargin"), _f(b.get("availableBalance"), equity))
        return {"equity": equity, "available": avail, "unrealized": _f(b.get("unrealizedProfit"))}

    def positions(self, symbol=None):
        p = {"symbol": symbol} if symbol else {}
        d = self._request("GET", "/openApi/swap/v2/user/positions", p, signed=True) or []
        out = []
        for x in d:
            amt = abs(_f(x.get("positionAmt")))
            if amt <= 0:
                continue
            out.append({"symbol": x.get("symbol"), "side": str(x.get("positionSide", "")).upper(),
                        "amt": amt, "avg": _f(x.get("avgPrice")), "mark": _f(x.get("markPrice"))})
        return out

    def position_mode(self):
        d = self._request("GET", "/openApi/swap/v1/positionSide/dual", signed=True) or {}
        return str(d.get("dualSidePosition", "")).lower() == "true"

    def set_position_mode(self, dual=True):
        return self._request("POST", "/openApi/swap/v1/positionSide/dual",
                             {"dualSidePosition": "true" if dual else "false"}, signed=True)

    def set_leverage(self, symbol, side, leverage):
        return self._request("POST", "/openApi/swap/v2/trade/leverage",
                             {"symbol": symbol, "side": side, "leverage": int(leverage)}, signed=True)

    def set_margin_type(self, symbol, mtype="ISOLATED"):
        return self._request("POST", "/openApi/swap/v2/trade/marginType",
                             {"symbol": symbol, "marginType": mtype}, signed=True)

    def income(self, symbol, start_ms, kind="REALIZED_PNL"):
        d = self._request("GET", "/openApi/swap/v2/user/income",
                          {"symbol": symbol, "incomeType": kind, "startTime": int(start_ms), "limit": 100},
                          signed=True) or []
        return sum(_f(x.get("income")) for x in d)

    # ------------------------------------------------------------------ ordenes
    def place_order(self, **params):
        d = self._request("POST", "/openApi/swap/v2/trade/order", params, signed=True, retry=False)
        return (d or {}).get("order", d) or {}

    def get_order(self, symbol, order_id):
        d = self._request("GET", "/openApi/swap/v2/trade/order",
                          {"symbol": symbol, "orderId": order_id}, signed=True) or {}
        return d.get("order", d)

    def open_orders(self, symbol):
        d = self._request("GET", "/openApi/swap/v2/trade/openOrders", {"symbol": symbol}, signed=True) or {}
        return d.get("orders", []) if isinstance(d, dict) else d

    def cancel_order(self, symbol, order_id):
        return self._request("DELETE", "/openApi/swap/v2/trade/order",
                             {"symbol": symbol, "orderId": order_id}, signed=True, retry=False)

    def cancel_all(self, symbol):
        return self._request("DELETE", "/openApi/swap/v2/trade/allOpenOrders",
                             {"symbol": symbol}, signed=True, retry=False)
