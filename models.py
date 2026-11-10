"""Modelos de datos del bot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional


class Candle(NamedTuple):
    t: int  # hora de APERTURA de la vela (ms, UTC)
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class SymbolMeta:
    symbol: str
    kind: str = "crypto"  # crypto | oil | tradfi | extra
    price_prec: Optional[int] = None
    qty_prec: Optional[int] = None
    quote_volume: float = 0.0


@dataclass
class Signal:
    symbol: str
    side: str  # LONG | SHORT
    t: int  # apertura de la vela de señal (ms)
    entry: float
    sl: float
    tps: List[float]
    risk: float  # distancia absoluta entrada -> SL (1R)
    score: int
    max_score: int
    conds: Dict[str, bool]
    trigger: str
    reversal: bool
    atr: float
    rvol: float
    room_r: Optional[float]
    level_name: Optional[str]
    level_price: Optional[float]
    pdh: Optional[float]
    pdl: Optional[float]
    ltf: str
    htf: str
    rank: float = 0.0  # puntuación de calidad 0-100 (se usa para elegir el TOP-N de cada cierre)
    rank_parts: Dict[str, int] = field(default_factory=dict)  # cada componente del ranking, 0-100


@dataclass
class Trade:
    """Operación 'virtual' que el bot sigue para avisar de TP / SL / salida."""

    symbol: str
    side: str
    entry: float
    sl: float  # stop actual (puede moverse a breakeven)
    sl0: float  # stop original
    tps: List[float]
    risk: float
    t_signal: int
    t_last: int  # última vela procesada (apertura, ms)
    opened_ms: int  # momento de la entrada (cierre de la vela de señal)
    hits: List[bool] = field(default_factory=list)
    closed: bool = False
    realized_r: float = 0.0
    remaining: float = 1.0
    price_prec: Optional[int] = None


@dataclass
class Event:
    kind: str  # TP | SL | BE | OPPOSITE | FLIP | TIMEOUT
    n: int = 0  # número de TP (solo para TP)
    price: float = 0.0
    r: float = 0.0  # R de este evento
    final: bool = False
    total_r: float = 0.0  # R total de la operación (solo si final)
