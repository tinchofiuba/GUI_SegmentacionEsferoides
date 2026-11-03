"""Adaptador de salida `semantic`: el modelo devuelve logits [1, C, H, W] (C = fondo +
clases) y la máscara es el argmax por píxel. El pre/post-procesado sale del manifiesto.
Sin Qt ni onnxruntime: `infer` recibe una función `run(tensor) -> logits`."""

import cv2
import numpy as np

_FLIPS = {"hflip": (3,), "vflip": (2,), "hvflip": (2, 3)}   # ejes en NCHW


def preprocess(rgb, manifest):
    """RGB uint8 (h, w, 3) -> (tensor float32 [1, 3, H, W], h, w) con H, W múltiplos de
    `pad_multiple` (relleno negro abajo/derecha, antes de normalizar como el entrenamiento)."""
    h, w = rgb.shape[:2]
    m = manifest.pad_multiple
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x = rgb.astype(np.float32) / 255.0
    if manifest.mean is not None:
        x = (x - np.float32(manifest.mean)) / np.float32(manifest.std)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32), h, w


def infer(run, rgb, manifest):
    """Máscara uint8 (h, w) con los ids de clase del manifiesto (0 = fondo)."""
    tensor, h, w = preprocess(rgb, manifest)
    logits = run(tensor)
    for view in manifest.tta:
        axes = _FLIPS[view]
        logits = logits + np.flip(run(np.ascontiguousarray(np.flip(tensor, axes))), axes)
    # el promedio no cambia el argmax: no hace falta dividir por el nº de vistas
    index = logits[0].argmax(axis=0)[:h, :w]
    ids = np.zeros(logits.shape[1], np.uint8)
    for c in manifest.classes:   # el canal k de los logits es la clase con id k (0 = fondo)
        ids[c.id] = c.id
    return ids[index]
