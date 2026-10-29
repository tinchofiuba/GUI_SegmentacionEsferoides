"""Estadística de referencia del dataset (no del modelo): por magnificación y por
clase, `n` y media / p5 / p95 de área, diámetro equivalente, perímetro y
circularidad de los objetos de las máscaras anotadas (el ground truth). Las
reglas de post-procesado se refieren a estos valores (`c1.p5`...). Sin Qt.

Se guarda en ~/.config/gui_segesferoides/dataset_stats.json y se recalcula
sobre las imágenes que el usuario elija: si se taggean más, hay más estadística.

Convención de nombres (la del entrenamiento): la magnificación va al final del
nombre de la imagen (`muestra_4x.png`) y la máscara es `<nombre sin
magnificación>-Mask.png`, en la misma carpeta o en una `masks` hermana.
"""

import json
import os
import re
from datetime import datetime

import cv2
import numpy as np

from gui_segesferoides.calibration import default_path as calibration_path
from gui_segesferoides.editing import measure_objects

VERSION = 1
METRICS = [("area", "Área (px²)"), ("diameter", "Diámetro equiv. (px)"),
           ("perimeter", "Perímetro (px)"), ("circularity", "Circularidad")]
MAGNIFICATION = re.compile(r"(?<![A-Za-z0-9])(\d+)[xX]$")
MASK_DIRS = [".", "masks", "../masks"]


def default_path():
    return os.path.join(os.path.dirname(calibration_path()), "dataset_stats.json")


def magnification_of(image_path):
    """"4x" a partir de `muestra_4x.png`; None si el nombre no la trae."""
    m = MAGNIFICATION.search(os.path.splitext(os.path.basename(image_path))[0].rstrip("_- "))
    return f"{m.group(1)}x" if m else None


def mask_path_for(image_path):
    """Ruta de la máscara de una imagen (`<nombre sin magnificación>-Mask.png`) o None."""
    folder = os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    bare = re.sub(r"[_\-\s]*(?<![A-Za-z0-9])\d+[xX]$", "", stem) or stem
    for sub in MASK_DIRS:
        path = os.path.normpath(os.path.join(folder, sub, f"{bare}-Mask.png"))
        if os.path.isfile(path):
            return path
    return None


def read_mask(path, palette=None):
    """Máscara de ids (uint8, 0 = fondo). Acepta 1 canal con ids o a color
    (`palette` = {id: (r, g, b)}); None si no se puede leer o no se puede mapear."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        return raw.astype(np.uint8)
    if not palette:
        return None
    rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for class_id, color in palette.items():
        mask[np.all(rgb == color, axis=2)] = class_id
    return mask


def _summary(values):
    return {"mean": float(np.mean(values)), "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95))}


def summarize(objects):
    """`objects`: {magnificación: {id de clase: [filas de measure_objects]}} ->
    {magnificación: {"id": {"n", "area", "diameter", "perimeter", "circularity"}}}.
    Las clases sin objetos no aparecen (no se extrapola entre magnificaciones)."""
    out = {}
    for mag, per_class in objects.items():
        for class_id, rows in per_class.items():
            if not rows:
                continue
            column = {"area": "area", "diameter": "equivalent_diameter",
                      "perimeter": "perimeter", "circularity": "circularity"}
            out.setdefault(mag, {})[str(class_id)] = {
                "n": len(rows), **{k: _summary([r[v] for r in rows]) for k, v in column.items()}}
    return out


def compute(image_paths, palette=None, progress=None):
    """Mide los objetos de las máscaras de `image_paths`. `progress(i, total)` se
    llama antes de cada imagen. Devuelve `(datos, omitidas)`: `datos` con el
    formato del archivo y `omitidas` = [(ruta, motivo)] de las imágenes que no
    entraron."""
    objects, skipped, used = {}, [], 0
    for i, image in enumerate(image_paths):
        if progress:
            progress(i, len(image_paths))
        mag = magnification_of(image)
        mask_path = mask_path_for(image)
        if mag is None:
            skipped.append((image, "el nombre no trae la magnificación (_4x, _10x...)"))
        elif mask_path is None:
            skipped.append((image, "no se encontró <nombre>-Mask.png"))
        elif (mask := read_mask(mask_path, palette)) is None:
            skipped.append((image, "máscara ilegible o a color sin paleta"))
        else:
            used += 1
            for row in measure_objects(mask):
                objects.setdefault(mag, {}).setdefault(row["class_id"], []).append(row)
    data = {"version": VERSION, "updated": datetime.now().isoformat(timespec="seconds"),
            "n_images": used, "stats": summarize(objects)}
    return data, skipped


class DatasetStats:
    def __init__(self, path=None):
        self.path = path or default_path()
        self.data = None     # {"version", "updated", "n_images", "stats"}; None = sin calcular
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("stats"), dict):
                self.data = data
        except (OSError, ValueError):
            pass

    @property
    def stats(self):
        """{magnificación: {"id de clase": {...}}}, o {} si no hay."""
        return self.data["stats"] if self.data else {}

    def set(self, data):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.data = data
