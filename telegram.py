"""Envio a Telegram (HTML). Nunca lanza excepciones."""
import html
import logging
import time

import requests

log = logging.getLogger("telegram")


class Telegram:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat = chat_id

    @property
    def enabled(self):
        return bool(self.token and self.chat)

    def send(self, text):
        if not self.enabled:
            log.info("[Telegram desactivado] %s", text.replace("\n", " | "))
            return False
        ok = True
        for i in range(0, len(text), 3900):
            ok &= self._post(text[i:i + 3900])
        return ok

    def send_document(self, filename, content, caption=""):
        """Envia un archivo (bytes) como documento. Nunca lanza."""
        if not self.enabled:
            log.info("[Telegram desactivado] documento %s (%d bytes)", filename, len(content))
            return False
        try:
            r = requests.post(f"https://api.telegram.org/bot{self.token}/sendDocument",
                              data={"chat_id": self.chat, "caption": caption[:1000]},
                              files={"document": (filename, content)}, timeout=60)
            if not r.ok:
                log.error("Telegram documento HTTP %s: %s", r.status_code, r.text[:200])
            return r.ok
        except requests.RequestException as e:
            log.error("Telegram documento error: %s", e)
            return False

    def _post(self, chunk):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        body = {"chat_id": self.chat, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, timeout=15)
                if r.status_code == 429:
                    time.sleep(float(r.json().get("parameters", {}).get("retry_after", 3)) + 1)
                    continue
                if r.ok:
                    return True
                log.error("Telegram HTTP %s: %s", r.status_code, r.text[:200])
                if r.status_code in (400, 401, 403, 404):
                    return False
            except (requests.RequestException, ValueError) as e:
                log.error("Telegram error: %s", e)
            time.sleep(2 * (attempt + 1))
        return False


def esc(x):
    return html.escape(str(x))
