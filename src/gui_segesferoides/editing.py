"""Operaciones puras de edición de máscaras (sin Qt), para usar la app
como herramienta de tagging: pintar, borrar/reasignar un objeto entero,
undo y guardado. No conoce ninguna clase concreta: los ids y colores vienen
de la paleta que declara el backend.
"""

import cv2
import numpy as np


def paint_circle(mask, x, y, radius, class_id):
    """Pinta un círculo de `class_id` (0 = borrar) en `mask`, in-place."""
    cv2.circle(mask, (int(x), int(y)), max(1, int(radius)), int(class_id), thickness=-1)


def paint_segment(mask, x0, y0, x1, y1, radius, class_id):
    """Trazo continuo (línea con extremos redondeados) de `class_id` (0 =
    borrar) entre dos puntos. Evita huecos cuando el mouse se mueve rápido,
    y con radio chico sirve para cortar entre objetos pegados."""
    p0 = (int(round(x0)), int(round(y0)))
    p1 = (int(round(x1)), int(round(y1)))
    r = max(1, int(radius))
    cv2.line(mask, p0, p1, int(class_id), thickness=2 * r)
    cv2.circle(mask, p0, r, int(class_id), thickness=-1)
    cv2.circle(mask, p1, r, int(class_id), thickness=-1)


def fill_polygon(mask, polygon, class_id):
    """Rellena con `class_id` el interior de `polygon` (puntos (N, 2) en coordenadas de
    imagen), pisando lo que hubiera. Se rasteriza con precisión de subpíxel para que
    un borde curvo no salga escalonado. Devuelve False si el polígono es demasiado
    chico para dejar algo (menos de 3 puntos o área menor a 1 px)."""
    pts = np.asarray(polygon, dtype=float)
    if len(pts) < 3:
        return False
    x, y = pts[:, 0], pts[:, 1]
    if 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) < 1.0:
        return False
    fixed = np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)      # 4 bits de subpíxel
    cv2.fillPoly(mask, [fixed], int(class_id), lineType=cv2.LINE_8, shift=4)
    return True


def erase_polygon(mask, polygon):
    """Borra (pone en 0), in-place, el interior de `polygon` (puntos (N, 2) en coordenadas de imagen).
    Devuelve True si algo cambió: un polígono sin área, o que solo cubre fondo, no hace nada."""
    pts = np.asarray(polygon, dtype=float)
    if len(pts) < 3:
        return False
    h, w = mask.shape
    x0, y0 = max(int(np.floor(pts[:, 0].min())) - 1, 0), max(int(np.floor(pts[:, 1].min())) - 1, 0)
    x1, y1 = min(int(np.ceil(pts[:, 0].max())) + 2, w), min(int(np.ceil(pts[:, 1].max())) + 2, h)
    if x1 <= x0 or y1 <= y0:
        return False
    sub = mask[y0:y1, x0:x1]
    before = sub.copy()
    fill_polygon(sub, pts - (x0, y0), 0)
    return not np.array_equal(before, sub)


def replace_object(mask, x, y, new_class):
    """Reasigna a `new_class` (0 = borrar) el objeto conexo bajo (x, y), es
    decir, la región 4-conexa con la misma clase. Devuelve True si cambió algo.
    Si (x, y) cae en fondo y new_class != 0, no hace nada (no hay objeto)."""
    h, w = mask.shape
    x, y = int(x), int(y)
    if not (0 <= x < w and 0 <= y < h):
        return False
    current = int(mask[y, x])
    if current == 0 or current == new_class:
        return False
    cv2.floodFill(mask, None, (x, y), int(new_class), loDiff=0, upDiff=0, flags=4)
    return True


def enclosed_background(binary):
    """Máscara booleana de los píxeles de fondo de `binary` que no tocan el
    borde de la imagen (4-conexos), o sea los huecos internos."""
    h, w = binary.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = binary == 0
    padded[0, :] = padded[-1, :] = padded[:, 0] = padded[:, -1] = 1
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2, flags=4)
    return padded[1:-1, 1:-1] == 1


def fill_holes(mask, class_id):
    """Rellena huecos internos de la clase `class_id`, in-place."""
    holes = enclosed_background(mask == class_id)
    mask[holes & (mask == 0)] = class_id


def object_region(mask, x, y):
    """Máscara booleana del objeto conexo (4-conexo, misma clase) bajo (x, y),
    o None si (x, y) cae en el fondo o fuera de la imagen."""
    h, w = mask.shape
    x, y = int(x), int(y)
    if not (0 <= x < w and 0 <= y < h) or mask[y, x] == 0:
        return None
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask, flood, (x, y), 0, loDiff=0, upDiff=0,
                  flags=4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
    return flood[1:-1, 1:-1] == 1


REFINE_MARGIN = 40   # px de imagen alrededor del objeto que ve el modelo al repasarlo


def padded_box(box, margin, shape):
    """`box` (x, y, w, h) ampliado `margin` px por lado y recortado a una imagen de `shape` (h, w)."""
    x, y, w, h = box
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, shape[1]), min(y + h + margin, shape[0])
    return x0, y0, x1 - x0, y1 - y0


def object_target_at(mask, x, y, margin=REFINE_MARGIN):
    """Objeto bajo (x, y) para repasar: {"class", "seed": (x, y) un píxel adentro, "box": (x, y, w, h)
    del objeto con `margin` px por lado}; None si ahí hay fondo."""
    region = object_region(mask, x, y)
    if region is None:
        return None
    box = padded_box(cv2.boundingRect(region.astype(np.uint8)), margin, mask.shape)
    return {"class": int(mask[int(y), int(x)]), "seed": (int(x), int(y)), "box": box}


def object_targets(mask, margin=REFINE_MARGIN):
    """Todos los objetos de `mask` (componentes 4-conexos de una misma clase), como `object_target_at`,
    ordenados por clase y luego por posición (y, x)."""
    targets = []
    for class_id in (int(c) for c in np.unique(mask) if c != 0):
        n, labels, boxes, _ = cv2.connectedComponentsWithStats((mask == class_id).astype(np.uint8), connectivity=4)
        for k in range(1, n):
            x, y, w, h, _ = (int(v) for v in boxes[k])
            ys, xs = np.nonzero(labels[y : y + h, x : x + w] == k)
            targets.append({"class": class_id, "seed": (x + int(xs[0]), y + int(ys[0])),
                            "box": padded_box((x, y, w, h), margin, mask.shape)})
    targets.sort(key=lambda t: (t["class"], t["box"][1], t["box"][0]))
    return targets


def merge_refined(mask, box, seed, pred, valid_ids):
    """Reemplaza, in-place, el objeto de `mask` que contiene a `seed` por lo que `pred` (máscara de ids
    del recorte `box`) segmenta sobre él: quedan los componentes de `pred` que se superponen con el objeto
    viejo, con la clase de `pred`. No toca a los vecinos (solo pisa el objeto viejo y fondo). Los ids de
    `pred` fuera de `valid_ids` se ignoran. Devuelve False (y no cambia nada) si el modelo no segmentó nada
    sobre el objeto."""
    x, y, w, h = box
    sub = mask[y : y + h, x : x + w]
    sx, sy = seed[0] - x, seed[1] - y
    if pred.shape != sub.shape or not (0 <= sx < w and 0 <= sy < h) or sub[sy, sx] == 0:
        return False
    _, labels = cv2.connectedComponents((sub == sub[sy, sx]).astype(np.uint8), connectivity=4)
    old = labels == labels[sy, sx]
    pred = np.where(np.isin(pred, list(valid_ids)), pred, 0)
    keep = np.zeros(old.shape, bool)
    for c in (int(v) for v in np.unique(pred[old]) if v != 0):
        _, comp = cv2.connectedComponents((pred == c).astype(np.uint8), connectivity=8)
        keep |= np.isin(comp, np.unique(comp[old & (pred == c)])) & (pred == c)
    if not keep.any():
        return False
    sub[old] = 0
    paint = keep & (sub == 0)
    sub[paint] = pred[paint]
    return True


def fill_object_holes(mask, region):
    """Rellena los huecos de fondo de un solo objeto (`region`, booleana) con
    su clase, in-place. Devuelve `region` ampliada con los píxeles agregados."""
    class_id = int(mask[region][0])
    holes = enclosed_background(region) & (mask == 0)
    mask[holes] = class_id
    return region | holes


def object_contours(region, holes=False):
    """Contornos de `region` como lista de arrays (N, 2) de (x, y) en píxeles
    de imagen: el externo y, con `holes=True`, también los de sus huecos."""
    x, y, w, h = cv2.boundingRect(region.astype(np.uint8))
    crop = np.zeros((h + 2, w + 2), np.uint8)
    crop[1:-1, 1:-1] = region[y : y + h, x : x + w]
    contours, _ = cv2.findContours(crop, cv2.RETR_LIST if holes else cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2).astype(np.float64) + (x - 1, y - 1) for c in contours]


def object_stats(region):
    """Medidas de un objeto, en píxeles: área, perímetro (largo del contorno
    externo), diámetro equivalente (el de un círculo de igual área) y
    circularidad 4πA/P² (1 = círculo; en píxeles se acota a 1)."""
    area = int(np.count_nonzero(region))
    perimeter = sum(cv2.arcLength(c.astype(np.float32), True) for c in object_contours(region))
    circularity = min(1.0, 4 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0
    return {
        "area": area,
        "perimeter": float(perimeter),
        "equivalent_diameter": float(np.sqrt(4 * area / np.pi)),
        "circularity": float(circularity),
    }


def count_objects(mask):
    """{id de clase: cantidad de objetos} de las clases presentes en `mask`. Un objeto
    es un componente 4-conexo de una misma clase, igual que en `measure_objects`."""
    counts = {}
    for class_id in (int(c) for c in np.unique(mask) if c != 0):
        n, _ = cv2.connectedComponents((mask == class_id).astype(np.uint8), connectivity=4)
        counts[class_id] = n - 1
    return counts


def measure_objects(mask):
    """Una fila por objeto (componente 4-conexo de una misma clase), con las
    mismas medidas que el análisis de partícula (`object_stats`) más la clase,
    el centroide y el bbox. Ordenadas por clase y luego por posición (y, x)."""
    rows = []
    for class_id in (int(c) for c in np.unique(mask) if c != 0):
        n, labels, boxes, centroids = cv2.connectedComponentsWithStats(
            (mask == class_id).astype(np.uint8), connectivity=4)
        for k in range(1, n):
            x, y, w, h, _ = boxes[k]
            region = labels[y : y + h, x : x + w] == k
            rows.append({
                "class_id": class_id,
                "centroid_x": float(centroids[k][0]),
                "centroid_y": float(centroids[k][1]),
                "bbox_x": int(x), "bbox_y": int(y), "bbox_w": int(w), "bbox_h": int(h),
                **object_stats(region),
            })
    rows.sort(key=lambda r: (r["class_id"], r["bbox_y"], r["bbox_x"]))
    return rows


def save_mask(path, mask, palette, mode="color"):
    """Guarda `mask` (ids de clase; 0 = fondo).

    mode="color": PNG a color, un color RGB por clase según `palette`
    ({id: (r, g, b)}), fondo negro — el formato de los datasets de color.
    mode="ids": PNG de 1 canal con el id de clase como valor de píxel.
    """
    if mode == "ids":
        out = mask.astype(np.uint8)
    elif mode == "color":
        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, (r, g, b) in palette.items():
            out[mask == class_id] = (b, g, r)  # cv2 escribe BGR
    else:
        raise ValueError(f"mode desconocido: {mode!r}")
    if not cv2.imwrite(path, out):
        raise OSError(f"No se pudo escribir la máscara en {path}")


class UndoStack:
    """Pila acotada de snapshots de máscara (cada una pesa HxW bytes)."""

    def __init__(self, max_size=8):
        self.max_size = max_size
        self._undo = []
        self._redo = []

    def push(self, mask):
        self._undo.append(mask.copy())
        if len(self._undo) > self.max_size:
            self._undo.pop(0)
        self._redo.clear()

    def discard_last(self):
        if self._undo:
            self._undo.pop()

    def undo(self, current):
        if not self._undo:
            return None
        self._redo.append(current.copy())
        return self._undo.pop()

    def redo(self, current):
        if not self._redo:
            return None
        self._undo.append(current.copy())
        return self._redo.pop()

    def clear(self):
        self._undo.clear()
        self._redo.clear()
