"""Matemática y render de la vista con zoom/pan (sin Qt). La imagen se
dibuja recortando SOLO la región visible a resolución nativa y reescalándola
al tamaño del widget, así el costo por frame depende del tamaño de la
ventana y no de los ~48 MP de la imagen. Para zooms bajos se usa una
pirámide (mitades sucesivas) de la imagen.
"""

import math

import cv2
import numpy as np

MAX_PYRAMID_LEVEL = 3
MIN_ZOOM_FACTOR = 0.9   # el zoom mínimo es "ajustar a ventana" * esto
MAX_ZOOM = 16.0
BACKGROUND_GRAY = 40


def build_pyramid(image_rgb, max_level=MAX_PYRAMID_LEVEL):
    """[nivel 0 (sin copiar), 1/2, 1/4, ...] — INTER_AREA para que no aliasee."""
    levels = [image_rgb]
    for _ in range(max_level):
        prev = levels[-1]
        h, w = prev.shape[:2]
        if min(h, w) < 64:
            break
        levels.append(cv2.resize(prev, (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    return levels


def fit_zoom(img_w, img_h, view_w, view_h):
    return min(view_w / img_w, view_h / img_h)


def fit_offset(img_w, img_h, view_w, view_h, zoom):
    """Offset (en coords de imagen) que centra la imagen en la vista."""
    return (img_w - view_w / zoom) / 2, (img_h - view_h / zoom) / 2


def zoom_at(offset, zoom, new_zoom, anchor_view):
    """Nuevo offset tras cambiar el zoom manteniendo fijo el punto de imagen
    que está bajo `anchor_view` (coords de widget)."""
    ax, ay = anchor_view
    ix = offset[0] + ax / zoom
    iy = offset[1] + ay / zoom
    return ix - ax / new_zoom, iy - ay / new_zoom


def view_to_image(offset, zoom, vx, vy):
    return offset[0] + vx / zoom, offset[1] + vy / zoom


def image_to_view(offset, zoom, ix, iy):
    return (ix - offset[0]) * zoom, (iy - offset[1]) * zoom


def render_view(levels, mask, palette, offset, zoom, view_w, view_h, alpha=0.5, show_mask=True):
    """Devuelve un array RGB (view_h, view_w, 3) con la región visible.
    `palette`: {id de clase: (r, g, b)}."""
    out = np.full((view_h, view_w, 3), BACKGROUND_GRAY, dtype=np.uint8)
    img_h, img_w = levels[0].shape[:2]

    sx0 = max(0, math.floor(offset[0]))
    sy0 = max(0, math.floor(offset[1]))
    sx1 = min(img_w, math.ceil(offset[0] + view_w / zoom))
    sy1 = min(img_h, math.ceil(offset[1] + view_h / zoom))
    if sx1 <= sx0 or sy1 <= sy0:
        return out

    dx0 = round((sx0 - offset[0]) * zoom)
    dy0 = round((sy0 - offset[1]) * zoom)
    dw = min(view_w - dx0, max(1, round((sx1 - sx0) * zoom)))
    dh = min(view_h - dy0, max(1, round((sy1 - sy0) * zoom)))
    if dw <= 0 or dh <= 0 or dx0 < 0 or dy0 < 0:
        dx0, dy0 = max(dx0, 0), max(dy0, 0)
        dw, dh = min(dw, view_w - dx0), min(dh, view_h - dy0)
        if dw <= 0 or dh <= 0:
            return out

    # Nivel más chico que aún tenga >= 1 píxel de fuente por píxel de pantalla
    level = 0
    while level < len(levels) - 1 and zoom <= 1 / 2 ** (level + 1):
        level += 1
    f = 2 ** level
    src = levels[level]
    crop = src[sy0 // f : max(sy0 // f + 1, -(-sy1 // f)), sx0 // f : max(sx0 // f + 1, -(-sx1 // f))]
    interp = cv2.INTER_LINEAR if zoom < 2 else cv2.INTER_NEAREST
    region = cv2.resize(crop, (dw, dh), interpolation=interp)

    if show_mask:
        mcrop = cv2.resize(mask[sy0:sy1, sx0:sx1], (dw, dh), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(region)
        for cid, rgb in palette.items():
            color[mcrop == cid] = rgb
        blended = cv2.addWeighted(region, 1 - alpha, color, alpha, 0)
        has_class = mcrop > 0
        region[has_class] = blended[has_class]

    out[dy0 : dy0 + dh, dx0 : dx0 + dw] = region
    return out
