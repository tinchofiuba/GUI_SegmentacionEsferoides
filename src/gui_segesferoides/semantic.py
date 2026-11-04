"""Adaptador de salida `semantic`: el modelo devuelve logits [1, C, H, W] (C = fondo +
clases) y la máscara es el argmax por píxel. El pre/post-procesado sale del manifiesto.
Sin Qt ni onnxruntime: `infer` recibe una función `run(tensor) -> logits`."""

from concurrent.futures import ThreadPoolExecutor

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


def infer(run, rgb, manifest, tile=1024, overlap=128, workers=1):
    """Máscara uint8 (h, w) con los ids de clase del manifiesto (0 = fondo).

    Si el lado mayor supera `tile`, procesa por mosaicos: ventanas de `tile` px con un
    margen de contexto de `overlap` px por lado que se descarta (solo se conserva el
    centro de `tile - 2 * overlap`). Así la RAM queda acotada sin importar el tamaño de
    la imagen. `tile=None` infiere la imagen entera de una vez. Con `workers` > 1 los mosaicos
    corren en paralelo (`run` debe ser seguro entre hilos, como `InferenceSession.run`)."""
    h, w = rgb.shape[:2]
    if not tile or max(h, w) <= tile:
        return _infer_whole(run, rgb, manifest)
    core = tile - 2 * overlap
    if core < 1:
        raise ValueError(f"tile ({tile}) debe ser mayor que 2 * overlap ({overlap})")
    out = np.zeros((h, w), np.uint8)

    def do(origin):
        cy, cx = origin
        y0, y1 = max(cy - overlap, 0), min(cy + core + overlap, h)
        x0, x1 = max(cx - overlap, 0), min(cx + core + overlap, w)
        window = _infer_whole(run, rgb[y0:y1, x0:x1], manifest)
        ch, cw = min(core, h - cy), min(core, w - cx)
        out[cy : cy + ch, cx : cx + cw] = window[cy - y0 : cy - y0 + ch, cx - x0 : cx - x0 + cw]   # regiones disjuntas

    origins = [(cy, cx) for cy in range(0, h, core) for cx in range(0, w, core)]
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(do, origins))     # list(): propaga las excepciones
    else:
        for origin in origins:
            do(origin)
    return out


def _infer_whole(run, rgb, manifest):
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
