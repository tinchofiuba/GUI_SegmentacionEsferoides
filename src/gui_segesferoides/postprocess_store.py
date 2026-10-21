"""Reglas de post-procesado que el usuario editó, por backend. Se guardan en
~/.config/gui_segesferoides/postprocess.json y se mandan al backend en cada
predicción; si un backend no tiene entrada, usa las reglas de su modelo.
Sin Qt.
"""

import json
import os

from gui_segesferoides.calibration import default_path as calibration_path


def default_path():
    return os.path.join(os.path.dirname(calibration_path()), "postprocess.json")


class PostprocessStore:
    def __init__(self, path=None):
        self.path = path or default_path()
        self.configs = {}      # {nombre de backend: {"rules": [...], "fill_holes": [...]}}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.configs = {k: v for k, v in data.items()
                            if isinstance(v, dict) and isinstance(v.get("rules"), list)}
        except (OSError, ValueError, AttributeError):
            pass   # sin archivo o dañado: reglas del modelo

    def get(self, backend):
        return self.configs.get(backend)

    def set(self, backend, config):
        """`config` None borra la edición del usuario (vuelve a las reglas del modelo)."""
        if config is None:
            self.configs.pop(backend, None)
        else:
            self.configs[backend] = {"rules": config["rules"], "fill_holes": config.get("fill_holes", [])}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.configs, f, indent=2, ensure_ascii=False)
