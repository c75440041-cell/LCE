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
                    self._data = json.load(f)
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
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
