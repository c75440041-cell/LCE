"""Configuración 100 % por variables de entorno (Railway -> Variables).

Los nombres replican los inputs de la estrategia Pine LCE v8. Con todo en
valores por defecto, las señales son las del núcleo LCE v2.
"""
import hashlib
import json
import os

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "12h": 43_200_000, "1d": 86_400_000,
}


# Presets = hipotesis a evaluar con el barrido (RUN_MODE=backtest); NO son parametros validados.
# Las variables de entorno explicitas siempre ganan al preset.
PRESETS = {
    "fiel": {},
    "calidad": {"min_risk_pct": 0.8, "min_rr": 1.5, "use_adx": True, "adx_min": 20.0, "use_e200_rv": True,
                "min_stretch": 2.5, "use_e200_tp": True, "rv_min_vol": 1.0},
    "rupturas": {"use_reversion": False, "min_risk_pct": 0.6, "min_rr": 1.5, "use_adx": True,
                 "use_e200_dir": True, "use_struct_bo": True},
    "reversion": {"use_breakout": False, "min_risk_pct": 0.8, "min_rr": 1.5, "use_e200_rv": True,
                  "rev_need_sweep": True, "rv_min_vol": 1.0, "use_e200_tp": True},
    "majors": {"symbols": "BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT,XRP-USDT,DOGE-USDT,ADA-USDT,LINK-USDT,"
                          "AVAX-USDT,TON-USDT,SUI-USDT,LTC-USDT", "min_risk_pct": 0.4, "min_rr": 1.5},
}

# Campos que NO cambian la calidad de las senales (no reinician las estadisticas)
_FP_SKIP = {"api_key", "api_secret", "tg_token", "tg_chat", "state_dir", "log_level", "heartbeat_hours", "workers",
            "scan_delay", "scan_on_start", "manage_interval", "mode", "demo", "run_mode", "max_positions",
            "risk_pct", "max_lev", "leverage", "min_live_rr", "state_file", "max_per_side", "regime_filter", "regime_symbol",
            "max_sl_pct", "max_sl_liq_frac", "kill_switch", "live_confirm", "max_day_loss", "max_trades_day",
            "max_cons_loss", "pause_bars"}


def _raw(name):
    v = os.getenv(name)
    return None if v is None or v.strip() == "" else v.strip()


def _s(name, d):
    v = _raw(name)
    return d if v is None else v


def _i(name, d):
    v = _raw(name)
    return d if v is None else int(float(v))


def _f(name, d):
    v = _raw(name)
    return d if v is None else float(v)


def _b(name, d):
    v = _raw(name)
    if v is None:
        return d
    return v.lower() in ("1", "true", "yes", "y", "on", "si", "sí")


def _first(names, d=""):
    for n in names:
        v = _raw(n)
        if v:
            return v
    return d


class Config:
    def __init__(self):
        # ---------------- Operativa ----------------
        self.mode = _s("TRADE_MODE", "signals").lower()   # signals | demo | live
        if _b("BINGX_DEMO", False) and self.mode == "live":
            self.mode = "demo"                              # seguridad: demo manda
        if self.mode not in ("signals", "demo", "live"):
            raise ValueError("TRADE_MODE debe ser signals, demo o live")
        self.demo = self.mode == "demo"
        self.api_key = _first(["BINGX_API_KEY", "BINGX_KEY"])
        self.api_secret = _first(["BINGX_API_SECRET", "BINGX_SECRET_KEY", "BINGX_SECRET"])
        self.tg_token = _first(["TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN"])
        self.tg_chat = _first(["TELEGRAM_CHAT_ID", "TELEGRAM_CHAT"])

        self.symbols = _s("SYMBOLS", "ALL")
        self.min_quote_volume = _f("MIN_QUOTE_VOLUME", 5_000_000.0)   # USDT 24h
        self.max_symbols = _i("MAX_SYMBOLS", 150)
        self.non_crypto_prefixes = tuple(
            p.strip().upper() for p in _s("NON_CRYPTO_PREFIXES", "NCSK,NCFX,NCCO,NCSI").split(",") if p.strip())
        self.max_positions = _i("MAX_POSITIONS", 4)
        self.max_per_side = _i("MAX_PER_SIDE", 2)                 # max. largos y max. cortos a la vez (0 = sin limite)
        self.regime_filter = _b("REGIME_FILTER", True)            # no largos con BTC bajista / no cortos con BTC alcista
        self.regime_symbol = _s("REGIME_SYMBOL", "BTC-USDT")
        self.max_sl_pct = _f("MAX_SL_PCT", 0.0)                   # SL maximo (% del precio), 0 = solo el limite de liquidacion
        self.max_sl_liq_frac = _f("MAX_SL_LIQ_FRAC", 0.5)         # SL <= esta fraccion de la distancia de liquidacion (100/lev %)
        self.kill_switch = _b("KILL_SWITCH", False)               # true = no abre operaciones nuevas
        self.live_confirm = _s("LIVE_CONFIRM", "")                # live exige LIVE_CONFIRM=SI
        self.strategy = _s("STRATEGY", "lce").lower()              # lce | qs
        if self.strategy not in ("lce", "qs"):
            raise ValueError("STRATEGY debe ser lce o qs")
        self.state_file = _s("STATE_FILE", "qs_state.json" if self.strategy == "qs" else "lce_state.json")
        self.timeframe = _s("TIMEFRAME", "15m" if self.strategy == "qs" else "5m")
        self.bar_ms = INTERVAL_MS[self.timeframe]
        self.candles = _i("CANDLES", 800)
        self.workers = _i("WORKERS", 4)
        self.scan_delay = _f("SCAN_DELAY_S", 4.0)
        self.scan_on_start = _b("SCAN_ON_START", True)
        self.manage_interval = _f("MANAGE_INTERVAL_S", 15.0)
        self.heartbeat_hours = _f("HEARTBEAT_HOURS", 6.0)
        self.state_dir = _s("STATE_DIR", "/data")
        self.log_level = _s("LOG_LEVEL", "INFO").upper()
        self.signal_timeout_bars = _i("SIGNAL_TIMEOUT_BARS", 288)   # senal sin resolver -> expirada
        self.min_live_rr = _f("MIN_LIVE_RR", 1.0)
        self.preset = _s("PRESET", "fiel").lower()
        if self.preset not in PRESETS:
            raise ValueError(f"PRESET desconocido: {self.preset} (opciones: {', '.join(PRESETS)})")
        self.run_mode = _s("RUN_MODE", "bot").lower()               # bot | backtest | dump
        self.backtest_days = _i("BACKTEST_DAYS", 30)
        self.backtest_symbols = _i("BACKTEST_SYMBOLS", 40)
        self.backtest_workers = _i("BACKTEST_WORKERS", 3)
        self.backtest_min_n = _i("BACKTEST_MIN_N", 30)
        self.dump_symbol = _s("DUMP_SYMBOL", "BTC-USDT").upper()
        self.dump_days = _i("DUMP_DAYS", 3)     # R:R minimo con el precio real al entrar

        # ---------------- QS + SSMT 2 etapas + tCISD (STRATEGY=qs) ----------------
        self.triads = _s("TRIADS", "BTC-USDT,ETH-USDT,SOL-USDT")   # triadas separadas por ';'
        self.qs_inv2 = _b("QS_INV2", False)
        self.qs_inv3 = _b("QS_INV3", False)
        self.qs_seq = _s("QS_SEQ", "wd90").lower()                  # wd90 = Semanal-Diario-90m | mwd = Mensual-Semanal-Diario
        self.qs_align = _s("QS_ALIGN", "369_o_cadena").lower()      # 369_o_cadena|solo_369|solo_cadena|369_y_cadena|sin_filtro
        self.qs_min_chain = _i("QS_MIN_CHAIN", 3)
        self.qs_bridge = _b("QS_BRIDGE", True)
        self.qs_fri_q4 = _b("QS_FRI_Q4", False)
        self.qs_use_month = _b("QS_USE_MONTH", True)
        self.qs_use_year = _b("QS_USE_YEAR", False)
        self.qs_stage = _s("QS_STAGE", "daily").lower()             # daily | m90
        self.qs_use2 = _b("QS_USE2", True)
        self.qs_req_div = _b("QS_REQ_DIV", False)
        self.qs_use_to = _b("QS_USE_TO", True)
        self.qs_min_to = _i("QS_MIN_TO", 1)
        self.qs_max_wait = _i("QS_MAX_WAIT", 16)
        self.qs_sl_buf = _f("QS_SL_BUF", 0.1)
        self.qs_max_risk_atr = _f("QS_MAX_RISK_ATR", 8.0)
        self.qs_tp_mode = _s("QS_TP_MODE", "ext137").lower()        # ext137 | rr
        self.qs_fib_ext = _f("QS_FIB_EXT", 1.37)
        self.qs_rr_fixed = _f("QS_RR_FIXED", 2.0)
        self.qs_min_rr = _f("QS_MIN_RR", 1.0)

        # ---------------- Zonas ----------------
        self.piv_len = _i("PIV_LEN", 10)
        self.zone_mult = _f("ZONE_MULT", 0.6)
        self.max_zones = _i("MAX_ZONES", 10)
        self.zone_max_age = _i("ZONE_MAX_AGE", 0)

        # ---------------- Nube EMA ----------------
        self.ema_fast = _i("EMA_FAST", 9)
        self.ema_slow = _i("EMA_SLOW", 21)
        self.slope_bars = _i("SLOPE_BARS", 3)

        # ---------------- Tipos de operación ----------------
        self.allow_long = _b("ALLOW_LONG", True)
        self.allow_short = _b("ALLOW_SHORT", True)
        self.use_breakout = _b("USE_BREAKOUT", True)
        self.use_reversion = _b("USE_REVERSION", True)
        self.use_sweep_sig = _b("USE_SWEEP_SIG", False)

        # ---------------- Filtros del núcleo ----------------
        self.use_htf = _b("USE_HTF", True)
        self.htf = _s("HTF_INTERVAL", "1h")
        self.htf_ms = INTERVAL_MS[self.htf]
        self.use_vol = _b("USE_VOL", True)
        self.vol_mult = _f("VOL_MULT", 1.2)
        self.body_mult = _f("BODY_MULT", 0.5)
        self.wick_ratio = _f("WICK_RATIO", 0.35)
        self.use_slope = _b("USE_SLOPE", True)
        self.rev_with_htf = _b("REV_WITH_HTF", False)

        # ---------------- Kernel (KDE) ----------------
        self.vol_mode = _s("VOL_MODE", "Multiplo").lower()      # multiplo | kernel
        self.kde_len = _i("KDE_LEN", 100)
        self.kde_min = _i("KDE_MIN", 20)
        self.kde_thr = _f("KDE_THR", 60.0)
        self.kde_bw = _f("KDE_BW", 1.0)

        # ---------------- Módulos extra ----------------
        self.use_adx = _b("USE_ADX", False)
        self.adx_min = _f("ADX_MIN", 18.0)
        self.use_flow = _b("USE_FLOW", False)
        self.flow_min = _f("FLOW_MIN", 1.0)
        self.use_sqz = _b("USE_SQZ", False)
        self.sqz_look = _i("SQZ_LOOK", 12)
        self.rev_need_sweep = _b("REV_NEED_SWEEP", False)
        self.sweep_look = _i("SWEEP_LOOK", 48)

        # ---------------- EMA200 y divergencia RSI ----------------
        self.ema200_len = _i("EMA200_LEN", 200)
        self.use_e200_dir = _b("USE_E200_DIR", False)
        self.use_e200_ext = _b("USE_E200_EXT", False)
        self.max_ext200 = _f("MAX_EXT200", 6.0)
        self.use_e200_rv = _b("USE_E200_RV", False)
        self.min_stretch = _f("MIN_STRETCH", 3.0)
        self.use_e200_tp = _b("USE_E200_TP", False)
        self.rev_need_div = _b("REV_NEED_DIV", False)
        self.rsi_len = _i("RSI_LEN", 14)
        self.div_piv = _i("DIV_PIV", 3)
        self.div_max_bars = _i("DIV_MAX_BARS", 30)
        self.div_valid = _i("DIV_VALID", 10)
        self.div_low = _f("DIV_LOW", 40.0)
        self.div_high = _f("DIV_HIGH", 60.0)

        # ---------------- Estructura ----------------
        self.use_sess_lvl = _b("USE_SESS_LVL", False)
        self.sess_lvl_utc = _s("SESS_LVL_UTC", "0000-0800")
        self.size_by_score = _b("SIZE_BY_SCORE", False)
        self.size_min = _f("SIZE_MIN", 0.5)
        self.size_max = _f("SIZE_MAX", 1.25)
        self.use_score = _b("USE_SCORE", False)
        self.min_score_bo = _i("MIN_SCORE_BO", 5)
        self.min_score_rv = _i("MIN_SCORE_RV", 2)
        self.adx_range_max = _f("ADX_RANGE_MAX", 25.0)
        self.struct_len = _i("STRUCT_LEN", 5)
        self.use_struct_bo = _b("USE_STRUCT_BO", False)
        self.use_struct_exit = _b("USE_STRUCT_EXIT", False)
        # filtros nuevos (apagados por defecto: con todo apagado = Pine v8)
        self.min_risk_pct = _f("MIN_RISK_PCT", 0.0)     # distancia minima al SL (% del precio)
        self.fixed_tp = _b("FIXED_TP", False)           # TP = entrada +- RR_FALLBACK x R (ignora el siguiente nivel)
        self.rv_min_vol = _f("RV_MIN_VOL", 0.0)         # reversion: volumen relativo minimo
        self.rv_min_adx = _f("RV_MIN_ADX", 0.0)         # reversion: ADX minimo (0 = off)
        self.rv_max_adx = _f("RV_MAX_ADX", 0.0)         # reversion: ADX maximo (0 = off)

        # ---------------- Gestión y riesgo ----------------
        self.atr_len = _i("ATR_LEN", 14)
        self.sl_buffer = _f("SL_BUFFER", 0.3)
        self.tp_buffer = _f("TP_BUFFER", 0.0)
        self.min_rr = _f("MIN_RR", 1.2)
        self.rr_fallback = _f("RR_FALLBACK", 2.0)
        self.max_risk_atr = _f("MAX_RISK_ATR", 0.0)
        self.cooldown = _i("COOLDOWN", 6)
        self.risk_pct = _f("RISK_PCT", 1.0)
        self.max_lev = _f("MAX_LEV", 5.0)
        self.leverage = _i("LEVERAGE", int(self.max_lev))
        self.use_partial = _b("USE_PARTIAL", False)
        self.tp1_r = _f("TP1_R", 1.0)
        self.tp1_pct = _f("TP1_PCT", 50.0)
        self.use_be = _b("USE_BE", False)
        self.use_trail = _b("USE_TRAIL", False)
        self.trail_atr = _f("TRAIL_ATR", 2.0)
        self.rr_trail_cap = _f("RR_TRAIL_CAP", 6.0)
        self.max_bars = _i("MAX_BARS", 0)
        self.max_day_loss = _f("MAX_DAY_LOSS", 3.0)
        self.max_trades_day = _i("MAX_TRADES_DAY", 12)
        self.max_cons_loss = _i("MAX_CONS_LOSS", 3)
        self.pause_bars = _i("PAUSE_BARS", 36)
        self.fee_pct = _f("FEE_PCT", 0.05)
        self.cost_mult = _f("COST_MULT", 0.0)
        self.slip_pct = _f("SLIP_PCT", 0.02)        # slippage por lado (%) para el R neto de las estadisticas
        self.use_session = _b("USE_SESSION", False)
        self.session_utc = _s("SESSION_UTC", "0900-1730")
        self.apply_overrides(PRESETS[self.preset], respect_env=True)

    def live_problems(self):
        """Condiciones minimas para permitir TRADE_MODE=live."""
        p = []
        if self.live_confirm.upper() != "SI":
            p.append("falta LIVE_CONFIRM=SI")
        if self.max_day_loss <= 0:
            p.append("MAX_DAY_LOSS debe ser > 0")
        if self.risk_pct > 2.0:
            p.append("RISK_PCT > 2 %")
        if self.max_positions > 6:
            p.append("MAX_POSITIONS > 6")
        return p

    def apply_overrides(self, ov, respect_env=True):
        for k, v in ov.items():
            if respect_env and _raw(k.upper()) is not None:
                continue
            setattr(self, k, v)

    def active_env(self):
        """Variables de entorno de la estrategia que difieren de 'no definidas' (sin secretos ni operativas)."""
        names = {k.upper() for k in vars(self)} | {"HTF_INTERVAL"}
        skip = {"PRESET", "STATE_DIR", "LOG_LEVEL", "RUN_MODE", "HEARTBEAT_HOURS", "WORKERS", "SCAN_DELAY_S",
                "SCAN_ON_START", "MANAGE_INTERVAL_S", "MAX_POSITIONS", "RISK_PCT", "MAX_LEV", "LEVERAGE"}
        return [f"{n}={os.environ[n]}" for n in sorted(names)
                if n in os.environ and os.environ[n].strip() != "" and n not in skip
                and not n.startswith(("BACKTEST", "DUMP"))]

    def fingerprint(self):
        d = {k: v for k, v in sorted(vars(self).items())
             if k not in _FP_SKIP and not k.startswith(("backtest", "dump"))}
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def missing_for_trading(self):
        miss = []
        if not self.api_key:
            miss.append("BINGX_API_KEY")
        if not self.api_secret:
            miss.append("BINGX_API_SECRET")
        return miss


# Variables que el barrido NO debe ignorar (datos, costes y operativa); el resto son parametros de estrategia
_KEEP_ENV = {"STATE_DIR", "LOG_LEVEL", "STRATEGY", "TIMEFRAME", "HTF_INTERVAL", "CANDLES", "SYMBOLS", "MIN_QUOTE_VOLUME",
             "MAX_SYMBOLS", "NON_CRYPTO_PREFIXES", "SIGNAL_TIMEOUT_BARS", "FEE_PCT", "SLIP_PCT", "STATE_FILE", "TRIADS",
             "RUN_MODE", "WORKERS"}


def clean_config():
    """Config con los valores por defecto de la ESTRATEGIA: ignora las variables de entorno de estrategia
    (COST_MULT, MIN_RR, USE_*, PRESET...) para que el barrido parta de una base limpia."""
    names = {k.upper() for k in vars(Config())} - _KEEP_ENV - {k.upper() for k in _FP_SKIP}
    names = {n for n in names if not n.startswith(("BACKTEST", "DUMP"))} | {"PRESET"}
    saved = {n: os.environ.pop(n) for n in list(names) if n in os.environ}
    try:
        cfg = Config()
    finally:
        os.environ.update(saved)
    return cfg
