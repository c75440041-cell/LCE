"""Persistencia JSON atomica. En Railway, monta un Volume en /data (STATE_DIR)."""
import copy
import json
import logging
import os
import threading

log = logging.getLogger("state")


class State:
    def __init__(self, directory="/data", name="lce_state.json"):
        self.dir = directory
        try:
            os.makedirs(directory, exist_ok=True)
            probe = os.path.join(directory, ".w")
            with open(probe, "w") as f:
                f.write("1")
            os.remove(probe)
        except OSError:
            log.warning("STATE_DIR %s no es escribible -> uso ./state (SE PERDERA al redeploy; monta un Volume)", directory)
            self.dir = os.path.abspath("state")
            os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, name)
        self._lock = threading.Lock()
        self._data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._data = raw
                else:
                    log.error("estado raiz no es dict, empiezo vacio")
            except (OSError, ValueError):
                log.exception("estado corrupto, empiezo vacio")
        self.persistent = self.dir == directory

    def get(self, key, default=None):
        with self._lock:
            return copy.deepcopy(self._data.get(key, default))

    def put(self, key, value):
        # Siempre dict: evita el bug de guardar listas donde se esperan dicts.
        if not isinstance(value, dict):
            raise TypeError(f"State.put('{key}') requiere dict, recibi {type(value).__name__}")
        with self._lock:
            self._data[key] = copy.deepcopy(value)
            self._flush_unlocked()

    def put_many(self, mapping):
        """Actualiza varias claves en una sola escritura atomica."""
        for k, v in mapping.items():
            if not isinstance(v, dict):
                raise TypeError(f"State.put_many('{k}') requiere dict, recibi {type(v).__name__}")
        with self._lock:
            for k, v in mapping.items():
                self._data[k] = copy.deepcopy(v)
            self._flush_unlocked()

    def _flush_unlocked(self):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
        except OSError:
            log.exception("no pude persistir estado en %s", self.path)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
