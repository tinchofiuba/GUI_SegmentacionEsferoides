"""Backend falso en proceso (solo numpy/cv2), para testear la GUI sin ningún modelo.
Predice un círculo de clase 2 centrado en la región pedida. Misma interfaz que OnnxBackend."""

import cv2
import numpy as np

from gui_segesferoides.backend import BackendError, BackendInfo, ClassInfo, ModelEntry


class FakeBackend:
    def __init__(self, spec=None):
        self.info = BackendInfo(
            name="fake",
            classes=[ClassInfo(1, "chicas", (0, 255, 0)), ClassInfo(2, "grandes", (255, 255, 0)),
                     ClassInfo(5, "raras", (0, 255, 255))],
            magnifications=["4x", "10x"],
            supports_bbox=True,
        )
        self.model = None
        self.crash_on_predict = False
        self.options_set = []     # [(id, valor)] de las llamadas a set_option

    def list_models(self):
        return [ModelEntry("a", "Modelo A", current=self.model == "a"),
                ModelEntry("b", "Modelo B", "en curso", current=self.model == "b")]

    def load_model(self, model_id):
        if model_id == "malo":
            raise BackendError("no existe ese modelo")
        self.model = model_id
        return f"fake:{model_id}"

    def predict(self, image_path, magnification, bbox=None):
        if self.crash_on_predict:
            raise BackendError("el backend falló")
        img = cv2.imread(image_path)
        if img is None:
            raise BackendError("no se pudo leer la imagen")
        h, w = img.shape[:2]
        x, y, bw, bh = bbox or (0, 0, w, h)
        mask = np.zeros((bh, bw), np.uint8)
        cv2.circle(mask, (bw // 2, bh // 2), max(1, min(bw, bh) // 4), 2, -1)
        return mask

    def set_option(self, option_id, value):
        self.options_set.append((option_id, value))

    def close(self):
        pass
