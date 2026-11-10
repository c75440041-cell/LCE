FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Comprobación en el build: si falta algún archivo o hay un import roto, el build falla aquí
# (se ve en "Build Logs") en vez de entrar en bucle de reinicios en producción.
RUN python -c "import main, backtest, scanner, strategy, tracker, messages, universe, bingx, telegram, config, models, indicators; print('Imports OK')"

CMD ["python", "main.py"]
