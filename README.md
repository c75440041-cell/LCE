# LCE Bot — BingX perpetuos + Telegram (Railway)

Port a Python de la estrategia **LCE v8** (núcleo LCE v2: zonas oferta/demanda, nube EMA 9/21,
rupturas BO + rechazos RV, filtro HTF sin repintado). Opcionalmente **QS + SSMT + tCISD**
(`STRATEGY=qs`).

Escanea cada vela cerrada, envía la señal a Telegram y, según `TRADE_MODE`, opera en BingX.

## Modos

| TRADE_MODE | Qué hace |
|---|---|
| `signals` | Solo Telegram. No necesita claves BingX. **Empieza aquí.** |
| `demo` | Sandbox VST (`open-api-vst.bingx.com`). |
| `live` | Dinero real (cerrojos obligatorios). |
| `RUN_MODE=backtest` | Barrido de parámetros (Telegram + JSON). |
| `RUN_MODE=dump` | Volcado CSV de señales de un símbolo. |

## Mejoras de esta versión

- Sin condición de carrera al abrir trades (análisis paralelo, órdenes en serie + lock).
- Apagado limpio con SIGTERM/SIGINT (Railway redeploy).
- Live bloqueado si faltan protecciones (`LIVE_CONFIRM=SI` o `CONFIRM_LIVE=SI_ENTIENDO_EL_RIESGO`,
  `MAX_DAY_LOSS > 0`, riesgo/leverage acotados, volume, equity, hedge).
- Drawdown de equity y de R neto en heartbeat / stats.
- Pausa por rachas notificada en Telegram.
- Sincronización de cantidad con la posición real del exchange.
- Cliente BingX endurecido (rate-limit, User-Agent).

Ver `CHANGELOG_MEJORAS.md`.

## Despliegue Railway

1. Sube este repo a GitHub (archivos en la **raíz**, sin subcarpeta).
2. Telegram: @BotFather → token; escribe al bot y lee `getUpdates` para el `chat_id`.
3. Railway → Deploy from GitHub. Usa el `Procfile` (`worker: python main.py`).
   **No** definas `startCommand` en `railway.toml` (provoca crash-loop).
4. **Volume montado en `/data`** (`STATE_DIR=/data`).
5. Variables de `RAILWAY_VARIABLES.env` (o el perfil live).
6. Secuencia: `signals` (días) → `demo` → `live` con riesgo bajo.

## Live: cerrojos

El bot **no arranca en live** sin:
- `LIVE_CONFIRM=SI` **o** `CONFIRM_LIVE=SI_ENTIENDO_EL_RIESGO`
- Telegram configurado
- Volume persistente en `/data`
- `MAX_DAY_LOSS > 0`, `RISK_PCT <= 2`, `LEVERAGE/MAX_LEV` razonables
- equity > 0 y cuenta en **modo hedge**

`KILL_SWITCH=true` bloquea nuevas entradas sin cerrar las abiertas.
Perfil conservador: `RAILWAY_VARIABLES_LIVE.env`.

## API BingX

Permisos: **lectura + trading de futuros. Sin retiros.**

## Estructura

```
main.py          # bucle, escaneo, ejecución
strategy.py      # LCE v8
qs_strategy.py   # QS + SSMT + tCISD (STRATEGY=qs)
qs_bot.py
indicators.py
bingx.py
config.py
risk.py
state.py
tracker.py       # win rate / R neto en vivo
telegram.py
backtest.py      # barrido (RUN_MODE=backtest)
Procfile
requirements.txt
RAILWAY_VARIABLES.env
RAILWAY_VARIABLES_LIVE.env
```

## Variables útiles

```
TRADE_MODE=signals|demo|live
STRATEGY=lce|qs
RUN_MODE=bot|backtest|dump
TIMEFRAME=5m
HTF_INTERVAL=1h
RISK_PCT=1.0
MAX_POSITIONS=4
MAX_DAY_LOSS=3
KILL_SWITCH=false
```

## Aviso

Trading de futuros implica riesgo de pérdida total. Un backtest favorable no garantiza
resultados futuros. Valida en `signals` y `demo` antes de live.
