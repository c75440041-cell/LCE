"""Configuración por variables de entorno (ver .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

# Intervalos soportados por BingX (el "10m" del indicador original NO existe en BingX).
TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
    "12h": 43200, "1d": 86400,
}

MAX_SCORE = 4  # EMA cloud + VWAP + EMA200 + EMA cloud HTF


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _bool(name: str, default: bool) -> bool:
    v = _env(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "y", "on", "si", "sí")


def _int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _list(name: str, default: str) -> Tuple[str, ...]:
    raw = _env(name, default)
    return tuple(x.strip().upper() for x in raw.split(",") if x.strip())


def _floats(name: str, default: str) -> Tuple[float, ...]:
    out = []
    for x in _env(name, default).split(","):
        x = x.strip()
        if x:
            try:
                out.append(float(x))
            except ValueError:
                pass
    return tuple(out)


@dataclass(frozen=True)
class Config:
    # Telegram
    telegram_token: str
    telegram_chat_ids: Tuple[str, ...]
    # Estrategia: "pullback" (CBC verde -> compra velas rojas / CBC rojo -> vende velas verdes)
    #             "flip"     (original: entra cuando nace el acuerdo CBC LTF+HTF)
    strategy: str
    pullback_require_htf: bool  # pullback: exige además que el CBC del HTF vaya en la misma dirección
    exit_on_cbc_flip: bool      # pullback: cierra la operación si el CBC LTF cambia de color en contra
    # Timeframes
    ltf: str
    htf: str
    ltf_sec: int
    htf_sec: int
    htf_use_forming: bool
    # Filtros de señal
    min_score: int
    atr_period: int
    sl_buffer_atr: float
    min_sl_atr: float
    max_sl_atr: float
    skip_wide_sl: bool
    tp_r: Tuple[float, ...]
    tp_split: Tuple[float, ...]
    min_room_r: float
    move_sl_to_be: bool
    exit_on_opposite: bool
    max_trade_hours: float
    track_trades: bool
    # Selección TOP-N por cierre de vela
    top_n: int            # máx. de señales nuevas por cierre de vela (0 = sin límite)
    min_rank: float       # ranking mínimo (0-100) para poder entrar en el TOP
    cooldown_bars: int    # velas de espera en un par tras cerrar una operación NO ganadora
    market_bias: bool     # el ranking premia las señales alineadas con el mercado de referencia
    market_symbol: str    # mercado de referencia (BTC-USDT)
    # Universo
    min_quote_volume: float
    max_symbols: int
    include_oil: bool
    include_tradfi: bool
    oil_keywords: Tuple[str, ...]
    tradfi_prefixes: Tuple[str, ...]
    extra_symbols: Tuple[str, ...]
    exclude_symbols: Tuple[str, ...]
    # Riesgo (solo para calcular tamaño sugerido en el mensaje)
    account_usdt: float
    risk_pct: float
    # Ejecución
    requests_per_second: float
    concurrency: int
    scan_delay_sec: float
    universe_refresh_min: int
    state_file: str
    dry_run: bool
    log_level: str


def load_config() -> Config:
    ltf = _env("LTF", "15m").lower()
    htf = _env("HTF", "1h").lower()
    strategy = _env("STRATEGY", "pullback").lower()
    if strategy not in ("pullback", "flip"):
        raise ValueError(f"STRATEGY '{strategy}' no válida. Usa: pullback | flip")
    if ltf not in TF_SECONDS:
        raise ValueError(f"LTF '{ltf}' no soportado. Usa: {', '.join(TF_SECONDS)}")
    if htf not in TF_SECONDS:
        raise ValueError(f"HTF '{htf}' no soportado. Usa: {', '.join(TF_SECONDS)}")
    ltf_sec, htf_sec = TF_SECONDS[ltf], TF_SECONDS[htf]
    if htf_sec < ltf_sec or htf_sec % ltf_sec != 0:
        raise ValueError(f"HTF ({htf}) debe ser múltiplo de LTF ({ltf}).")

    tp_r = tuple(sorted(x for x in _floats("TP_R", "1,2,3") if x > 0)) or (1.0, 2.0, 3.0)
    split = _floats("TP_SPLIT", "40,30,30")
    if len(split) != len(tp_r) or sum(split) <= 0 or any(x < 0 for x in split):
        split = tuple(1.0 for _ in tp_r)
    total = sum(split)
    tp_split = tuple(x / total for x in split)

    chat_ids = tuple(x.strip() for x in _env("TELEGRAM_CHAT_ID", "").split(",") if x.strip())

    return Config(
        telegram_token=_env("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_ids=chat_ids,
        strategy=strategy,
        pullback_require_htf=_bool("PULLBACK_REQUIRE_HTF", False),
        exit_on_cbc_flip=_bool("EXIT_ON_CBC_FLIP", True),
        ltf=ltf,
        htf=htf,
        ltf_sec=ltf_sec,
        htf_sec=htf_sec,
        htf_use_forming=_bool("HTF_USE_FORMING", True),
        min_score=max(0, min(MAX_SCORE, _int("MIN_SCORE", 3))),
        atr_period=max(2, _int("ATR_PERIOD", 14)),
        sl_buffer_atr=max(0.0, _float("SL_BUFFER_ATR", 0.15)),
        min_sl_atr=max(0.1, _float("MIN_SL_ATR", 0.8)),
        max_sl_atr=max(0.5, _float("MAX_SL_ATR", 2.5)),
        skip_wide_sl=_bool("SKIP_WIDE_SL", True),
        tp_r=tp_r,
        tp_split=tp_split,
        min_room_r=max(0.0, _float("MIN_ROOM_R", 1.0)),
        move_sl_to_be=_bool("MOVE_SL_TO_BE", True),
        exit_on_opposite=_bool("EXIT_ON_OPPOSITE", True),
        max_trade_hours=max(0.0, _float("MAX_TRADE_HOURS", 24.0)),
        track_trades=_bool("TRACK_TRADES", True),
        top_n=max(0, _int("TOP_N", 4)),
        min_rank=max(0.0, min(100.0, _float("MIN_RANK", 50.0))),
        cooldown_bars=max(0, _int("COOLDOWN_BARS", 4)),
        market_bias=_bool("USE_MARKET_BIAS", True),
        market_symbol=_env("MARKET_SYMBOL", "BTC-USDT").upper(),
        min_quote_volume=max(0.0, _float("MIN_QUOTE_VOLUME_USDT", 1_000_000.0)),
        max_symbols=max(0, _int("MAX_SYMBOLS", 0)),
        include_oil=_bool("INCLUDE_OIL", True),
        include_tradfi=_bool("INCLUDE_TRADFI", False),
        oil_keywords=_list("OIL_KEYWORDS", "OILWTI,OILBRENT,WTI,BRENT,CRUDE,OIL"),
        tradfi_prefixes=_list("TRADFI_PREFIXES", "NCCO,NCFX,NCSK,NCSI"),
        extra_symbols=_list("EXTRA_SYMBOLS", ""),
        exclude_symbols=_list("EXCLUDE_SYMBOLS", ""),
        account_usdt=max(0.0, _float("ACCOUNT_USDT", 0.0)),
        risk_pct=max(0.0, _float("RISK_PCT", 1.0)),
        requests_per_second=max(1.0, _float("REQUESTS_PER_SECOND", 10.0)),
        concurrency=max(1, _int("CONCURRENCY", 20)),
        scan_delay_sec=max(0.0, _float("SCAN_DELAY_SEC", 4.0)),
        universe_refresh_min=max(5, _int("UNIVERSE_REFRESH_MIN", 60)),
        state_file=_env("STATE_FILE", "state.json"),
        dry_run=_bool("DRY_RUN", False),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )
