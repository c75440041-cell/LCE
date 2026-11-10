"""Construye la lista de símbolos a analizar: todas las criptos USDT-M + petróleo."""
from __future__ import annotations

from typing import Any, Dict, List

from config import Config
from models import SymbolMeta


def _int_or_none(x: Any):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def classify(sym: str, cfg: Config) -> str:
    """crypto | oil | tradfi (acciones, forex, otras materias primas)."""
    is_tradfi = any(sym.startswith(p) for p in cfg.tradfi_prefixes)
    if is_tradfi:
        return "oil" if any(k in sym for k in cfg.oil_keywords) else "tradfi"
    # Sin prefijo TradFi solo aceptamos las claves "fuertes" para no confundir con tokens cripto.
    if any(k in sym for k in ("OILWTI", "OILBRENT")):
        return "oil"
    return "crypto"


def build_universe(contracts: List[Dict[str, Any]], volumes: Dict[str, float],
                   cfg: Config) -> Dict[str, SymbolMeta]:
    exclude = set(cfg.exclude_symbols)
    extra = set(cfg.extra_symbols)
    crypto: List[SymbolMeta] = []
    special: List[SymbolMeta] = []
    seen = set()

    for ct in contracts:
        sym = str(ct.get("symbol", "")).upper()
        if not sym or sym in seen or sym in exclude or not sym.endswith("-USDT"):
            continue
        status = ct.get("status")
        if status is not None and _int_or_none(status) not in (1, None):
            continue  # fuera de línea / prelanzamiento
        seen.add(sym)
        kind = classify(sym, cfg)
        vol = float(volumes.get(sym, 0.0))
        meta = SymbolMeta(
            symbol=sym, kind=kind,
            price_prec=_int_or_none(ct.get("pricePrecision")),
            qty_prec=_int_or_none(ct.get("quantityPrecision")),
            quote_volume=vol,
        )
        if sym in extra:
            meta.kind = "extra"
            special.append(meta)
        elif kind == "oil":
            if cfg.include_oil:
                special.append(meta)
        elif kind == "tradfi":
            if cfg.include_tradfi:
                special.append(meta)
        else:
            if not volumes or vol >= cfg.min_quote_volume:  # sin ticker => sin filtro de volumen
                crypto.append(meta)

    crypto.sort(key=lambda m: m.quote_volume, reverse=True)
    if cfg.max_symbols > 0:
        crypto = crypto[:cfg.max_symbols]
    return {m.symbol: m for m in special + crypto}
