# LCE Bot — BingX perpetuos + Telegram (Railway)

Port a Python de la estrategia **LCE v8** (núcleo **LCE v2**: zonas de oferta/demanda por pivotes,
nube EMA 9/21, rupturas BO + rechazos RV, filtro HTF sin repintado, R:R mínimo hasta el siguiente nivel).
Escanea cada vela cerrada de 5m, envía la señal a Telegram y, según `TRADE_MODE`, opera en BingX.

## Modos
| TRADE_MODE | Qué hace |
|---|---|
| `signals` | Solo envía señales a Telegram. No necesita claves de BingX. **Empieza aquí.** |
| `demo` | Opera en el sandbox VST (`open-api-vst.bingx.com`). Necesita claves VST. |
| `live` | Opera con dinero real. |

## Despliegue
1. Sube este repo a GitHub.
2. Telegram: crea el bot con @BotFather (token) y obtén tu `chat_id` (escríbele al bot y abre `https://api.telegram.org/bot<TOKEN>/getUpdates`).
3. Railway → New Project → Deploy from GitHub repo. Railway usa el `Procfile` (`worker: python main.py`).
   **No definas `startCommand` en railway.toml** (choca con el Procfile y provoca crash-loop).
4. **Añade un Volume montado en `/data`** (si no, se pierde el estado en cada redeploy).
5. Pega las variables de `RAILWAY_VARIABLES.env`.
6. Secuencia recomendada: `signals` (días) → `demo` → `live` con `RISK_PCT` bajo.

## Dinero real (live): cerrojos obligatorios
El bot **se niega a arrancar en `live`** si falta alguna de estas condiciones:
`CONFIRM_LIVE=SI_ENTIENDO_EL_RIESGO`, Telegram configurado, Volume en `/data`, `MAX_DAY_LOSS > 0`,
`RISK_PCT <= 2`, `MAX_LEV <= 10`, equity > 0 y cuenta en modo hedge.
Si no puede fijar el apalancamiento no abre la operación. `KILL_SWITCH=true` bloquea nuevas entradas sin tocar las abiertas.
Perfil conservador listo en `RAILWAY_VARIABLES_LIVE.env`.

## API key de BingX
Permisos: **lectura + trading de futuros. Sin retiros.** Cuenta en **modo hedge** (el bot lo comprueba al arrancar).

## Lógica de ejecución
- Entrada a mercado; confirma que la orden está `FILLED` (no acepta cancelaciones como fills).
- SL y TP como órdenes separadas `STOP_MARKET` / `TAKE_PROFIT_MARKET` (MARK_PRICE) con `positionSide`.
  Si no se puede colocar el SL → cierra la posición al instante.
- Apalancamiento fijado en LONG **y** SHORT, margen aislado. Tamaño = `RISK_PCT` del equity / distancia al SL, tope `MAX_LEV`×equity.
- Reconciliación al arrancar contra las posiciones reales. Estado en JSON atómico (`/data/lce_state.json`).
- Opcional: parcial en TP1, BE, trailing ATR, stop por tiempo, límites diarios, pausa por pérdidas seguidas.
- Firma HMAC: `urlencode(sorted(params))`, el mismo string que se envía.

## Qué NO está portado (el script pegado se corta antes)
El Pine pegado termina en la definición del máximo/mínimo de sesión, así que **no** incluí: modo puntuación
(confluencia), tamaño por puntuación, barrido del máx/mín de sesión, ni la lógica exacta de entradas/salidas
de `strategy.*` de v8. El bot implementa el núcleo v2 completo + los módulos cuyos inputs sí aparecen
(kernel KDE, ADX, flow, squeeze, EMA200, divergencia RSI, estructura BOS/CHoCH, barrido día anterior/N velas,
parcial/BE/trailing). Si pegas el resto del script, lo completo.

## Backtest (hazlo ANTES de operar)
`python backtest.py --top 20 --days 60` (o `--symbols BTC-USDT,ETH-USDT`). Usa la misma estrategia y tu configuración,
con comisiones y slippage (`--slip`). Muestra acierto, R medio, profit factor, retorno compuesto y drawdown.
Corre en tu PC o en la consola de Railway; no necesita claves. Menos de ~100 operaciones = muestra insuficiente.

## Tests
`pip install -r requirements-dev.txt && pytest -q`  (20 tests: firma, tamaño, riesgo, cerrojos live, indicadores, backtest)

## Estructura
Todos los archivos van en la raíz del repo (sin carpetas): `main.py`, `strategy.py`, `indicators.py`, `bingx.py`, `config.py`, `risk.py`, `state.py`, `telegram.py`, `backtest.py`, `test_lce.py`, `Procfile`, `requirements.txt`, `.python-version`, `.gitignore`.
