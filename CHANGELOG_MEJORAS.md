# Mejoras aplicadas al bot LCE / QS

## Criticas (bugs / seguridad)

1. **Condicion de carrera al abrir trades**  
   El analisis paralelo llamaba a `handle_signal` / `open_trade` desde varios hilos a la vez, pudiendo saltarse `max_positions` y `max_per_side`.  
   Ahora el analisis es paralelo (solo lectura) y la gestion de senales/ordenes es **estrictamente serial**. Ademas hay un `threading.Lock` y reserva de hueco al abrir.

2. **TRADE_MODE=live sin protecciones**  
   Si faltaba `LIVE_CONFIRM=SI` o habia parametros peligrosos (riesgo alto, leverage >10, kill_switch, etc.), el bot arrancaba igual. Ahora **bloquea el arranque** en live con mensaje claro.

3. **Apagado limpio**  
   `SIGTERM` / `SIGINT` (Railway redeploy, Ctrl+C) activan un flag y el bucle termina sin cortar a medias un ciclo. Avisa por Telegram.

## Robustez operativa

4. **Pausa por rachas** notificada en Telegram cuando se activa (antes solo log).
5. **Sincronizacion de cantidad** con la posicion real del exchange en `manage` (parciales / redondeos).
6. **Reserva de hueco** al abrir: si la orden falla, se libera; si otro ciclo intenta el mismo simbolo, no duplica.
7. **Estado JSON**: escritura atomica mas segura, `put_many`, validacion de raiz dict, limpieza de `.tmp` si falla.
8. **BingX**: mas codigos de rate-limit, User-Agent, mensajes de error mas claros.

## Metricas y riesgo

9. **Drawdown de equity** (pico y DD max %) en `RiskGuard`, visible en heartbeat.
10. **Drawdown de R neto** en estadisticas de senales en vivo.
11. Heartbeat enriquecido: equity + resumen de riesgo del dia.

## Config

12. Validacion de `TIMEFRAME` y `HTF_INTERVAL` contra `INTERVAL_MS`.
13. `live_problems` incluye leverage y kill_switch.

## Notas

- La logica de la estrategia Pine (LCE v8 / QS) **no se ha alterado** a proposito: cambios de senales sin validacion seria sobreajuste.
- Para operar live: `TRADE_MODE=live`, `LIVE_CONFIRM=SI`, volume en `/data`, limites de perdida diarios > 0.
- `requirements.txt` anadido (`numpy`, `requests`).
