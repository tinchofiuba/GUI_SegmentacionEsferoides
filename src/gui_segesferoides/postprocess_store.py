"""Reglas de post-procesado que el usuario editó. Son de la GUI, no de un modelo:
valen para cualquier backend y se guardan en ~/.config/gui_segesferoides/postprocess.json.
Sin entrada, se usan las de fábrica (postprocess.DEFAULT_POSTPROCESS). Sin Qt.
"""

import json
import os

from gui_segesferoides import postprocess
from gui_segesferoides.calibration import default_path as calibration_path


def default_path():
    return os.path.join(os.path.dirname(calibration_path()), "postprocess.json")


class PostprocessStore:
    def __init__(self, path=None):
        self.path = path or default_path()
        self.config = None      # {"rules": [...], "fill_holes": [...]} editado por el usuario
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if "rules" not in data:    # formato viejo: un dict por backend; se queda con la primera edición válida
                data = next((v for v in data.values() if self._valid(v)), {})
            postprocess.validate(data)
            self.config = {"rules": data["rules"], "fill_holes": data.get("fill_holes", [])}
        except (OSError, ValueError, AttributeError):
            pass

    @staticmethod
    def _valid(config):
        try:
            postprocess.validate(config)
        except ValueError:
            return False
        return True

    def get(self):
        """Las reglas vigentes: las del usuario o, sin ellas, las de fábrica."""
        return self.config or postprocess.defaults()

    def set(self, config):
        """`config` None borra la edición del usuario (vuelve a las de fábrica)."""
        self.config = None if config is None else {"rules": config["rules"],
                                                   "fill_holes": config.get("fill_holes", [])}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.config or {}, f, indent=2, ensure_ascii=False)
