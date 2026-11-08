"""Configuración 100 % por variables de entorno (Railway -> Variables).

Los nombres replican los inputs de la estrategia Pine LCE v8. Con todo en
valores por defecto, las señales son las del núcleo LCE v2.
"""
import os

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "12h": 43_200_000, "1d": 86_400_000,
}


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
        self.max_positions = _i("MAX_POSITIONS", 5)
        self.timeframe = _s("TIMEFRAME", "5m")
        self.bar_ms = INTERVAL_MS[self.timeframe]
        self.candles = _i("CANDLES", 1000)
        self.workers = _i("WORKERS", 4)
        self.scan_delay = _f("SCAN_DELAY_S", 8.0)
        self.scan_on_start = _b("SCAN_ON_START", True)
        self.manage_interval = _f("MANAGE_INTERVAL_S", 15.0)
        self.heartbeat_hours = _f("HEARTBEAT_HOURS", 6.0)
        self.state_dir = _s("STATE_DIR", "/data")
        self.log_level = _s("LOG_LEVEL", "INFO").upper()
        self.confirm_live = _s("CONFIRM_LIVE", "")      # debe ser SI_ENTIENDO_EL_RIESGO para operar en live
        self.kill_switch = _b("KILL_SWITCH", False)     # true = no abre operaciones nuevas (gestiona las abiertas)
        self.min_live_rr = _f("MIN_LIVE_RR", 1.0)     # R:R minimo con el precio real al entrar

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
        self.struct_len = _i("STRUCT_LEN", 5)
        self.use_struct_bo = _b("USE_STRUCT_BO", False)
        self.use_struct_exit = _b("USE_STRUCT_EXIT", False)

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
        self.max_day_loss = _f("MAX_DAY_LOSS", 0.0)
        self.max_trades_day = _i("MAX_TRADES_DAY", 0)
        self.max_cons_loss = _i("MAX_CONS_LOSS", 0)
        self.pause_bars = _i("PAUSE_BARS", 36)
        self.fee_pct = _f("FEE_PCT", 0.05)
        self.cost_mult = _f("COST_MULT", 0.0)
        self.use_session = _b("USE_SESSION", False)
        self.session_utc = _s("SESSION_UTC", "0900-1730")

    LIVE_PHRASE = "SI_ENTIENDO_EL_RIESGO"

    def live_problems(self):
        """Condiciones obligatorias para operar con dinero real."""
        p = []
        if self.confirm_live != self.LIVE_PHRASE:
            p.append(f"falta CONFIRM_LIVE={self.LIVE_PHRASE}")
        if not (self.tg_token and self.tg_chat):
            p.append("Telegram sin configurar (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        if self.risk_pct > 2.0:
            p.append(f"RISK_PCT={self.risk_pct} es demasiado alto para live (maximo 2)")
        if self.max_lev > 10:
            p.append(f"MAX_LEV={self.max_lev} es demasiado alto para live (maximo 10)")
        if self.max_day_loss <= 0:
            p.append("MAX_DAY_LOSS debe ser > 0 en live (limite de perdida diaria)")
        return p

    def missing_for_trading(self):
        miss = []
        if not self.api_key:
            miss.append("BINGX_API_KEY")
        if not self.api_secret:
            miss.append("BINGX_API_SECRET")
        return miss
