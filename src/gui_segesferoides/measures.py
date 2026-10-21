"""Medidas de distancia trazadas sobre la imagen: modelo, geometría y dibujo.

Las medidas son anotaciones de pantalla del usuario: viven aparte de la máscara (no la
tocan ni entran en lo que se le manda al modelo) y solo llegan a un archivo si se
pide «Guardar imagen». Todas las coordenadas son de la imagen a resolución nativa.
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen

from gui_segesferoides.calibration import PX

# (nombre, (r, g, b)): los colores que se pueden elegir para una medida
MEASURE_COLORS = [
    ("Rojo", (255, 0, 0)),
    ("Verde", (0, 200, 0)),
    ("Azul", (0, 90, 255)),
    ("Amarillo", (255, 220, 0)),
    ("Cian", (0, 220, 220)),
    ("Magenta", (255, 0, 200)),
    ("Naranja", (255, 140, 0)),
    ("Blanco", (255, 255, 255)),
    ("Negro", (0, 0, 0)),
]
DEFAULT_MEASURE_COLOR = MEASURE_COLORS[0][1]
MIN_LENGTH_PX = 2.0     # un click casi sin arrastrar no deja una medida


@dataclass(frozen=True)
class Measure:
    p1: tuple          # (x, y) en píxeles de imagen
    p2: tuple
    color: tuple       # (r, g, b)
    label_offset: tuple = (0.0, 0.0)   # cuánto se corrió la etiqueta de su lugar de siempre (px de imagen)

    @property
    def length(self):
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

    def translated(self, dx, dy):
        """Misma medida corrida (dx, dy): conserva largo, orientación y color, y la etiqueta
        sigue en el mismo lugar respecto de la línea."""
        return replace(self, p1=(self.p1[0] + dx, self.p1[1] + dy), p2=(self.p2[0] + dx, self.p2[1] + dy))

    def with_color(self, color):
        return replace(self, color=tuple(color))

    def with_label_offset(self, offset):
        return replace(self, label_offset=(float(offset[0]), float(offset[1])))


def distance_to_segment(point, a, b):
    """Distancia del punto `point` al segmento a-b (mismas unidades que las entradas)."""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    squared = dx * dx + dy * dy
    if squared == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / squared))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def clamp_translation(measure, dx, dy, width, height):
    """(dx, dy) recortado para que la medida, corrida así, siga dentro de la imagen."""
    xs, ys = (measure.p1[0], measure.p2[0]), (measure.p1[1], measure.p2[1])
    dx = max(-min(xs), min(dx, (width - 1) - max(xs)))
    dy = max(-min(ys), min(dy, (height - 1) - max(ys)))
    return dx, dy


def clamp_label_shift(measure, dx, dy, width, height):
    """(dx, dy) recortado para que la etiqueta, corrida así, quede dentro de la imagen
    (se toma como referencia el punto medio de la línea más lo que ya se corrió)."""
    mx, my = (measure.p1[0] + measure.p2[0]) / 2, (measure.p1[1] + measure.p2[1]) / 2
    x = min(max(mx + measure.label_offset[0] + dx, 0.0), width - 1.0)
    y = min(max(my + measure.label_offset[1] + dy, 0.0), height - 1.0)
    return x - mx - measure.label_offset[0], y - my - measure.label_offset[1]


def clamp_point(point, width, height):
    return (min(max(point[0], 0.0), width - 1.0), min(max(point[1], 0.0), height - 1.0))


def label_text(measure, scale):
    """Distancia de la medida: en píxeles, o en la unidad de la calibración si la hay."""
    if scale.unit == PX:
        return f"{measure.length:.1f} px"
    return scale.length(measure.length)


def _cap(width):
    return max(4.0, width * 2.5)         # largo de cada tope de la línea


def _normal(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy) or 1.0
    return -dy / norm, dx / norm


def label_rect(a, b, width, label, font_px, offset=(0.0, 0.0)):
    """Rectángulo de la etiqueta de la medida a-b (mismas coordenadas que a y b). De
    fábrica queda a un costado de la línea, sin taparla; `offset` la corre de ahí."""
    font = QFont()
    font.setPixelSize(max(8, int(font_px)))
    metrics = QFontMetricsF(font)
    w, h = metrics.horizontalAdvance(label) + font_px * 0.7, metrics.height() + font_px * 0.2
    nx, ny = _normal(a, b)
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    ox, oy = (nx, ny) if ny < 0 or (ny == 0 and nx < 0) else (-nx, -ny)     # hacia arriba
    off = _cap(width) + width + 2 + (w / 2) * abs(ox) + (h / 2) * abs(oy)   # medio ancho de la caja sobre la normal
    return QRectF(cx + ox * off - w / 2 + offset[0], cy + oy * off - h / 2 + offset[1], w, h)


def draw_measure(painter, a, b, color, width, label, font_px, highlighted=False, label_offset=(0.0, 0.0)):
    """Dibuja una medida (línea con topes y etiqueta) con coordenadas de `painter`:
    en el canvas son de pantalla; al guardar la imagen, las de la imagen."""
    r, g, bl = color
    painter.setRenderHint(QPainter.Antialiasing)
    if highlighted:      # halo para que se note cuál se está editando
        painter.setPen(QPen(QColor(255, 255, 255, 200), width + 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(*a), QPointF(*b))
    painter.setPen(QPen(QColor(r, g, bl), width, Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(QPointF(*a), QPointF(*b))
    nx, ny = _normal(a, b)                          # normal a la línea: los topes de los extremos
    cap = _cap(width)
    for x, y in (a, b):
        painter.drawLine(QPointF(x - nx * cap, y - ny * cap), QPointF(x + nx * cap, y + ny * cap))
    if not label:
        return
    font = QFont()
    font.setPixelSize(max(8, int(font_px)))
    painter.setFont(font)
    box = label_rect(a, b, width, label, font_px, label_offset)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(0, 0, 0, 170))
    painter.drawRoundedRect(box, box.height() * 0.25, box.height() * 0.25)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(box, Qt.AlignCenter, label)


def render_measures(image_rgb, measures, scale, grayscale=False):
    """Copia de la imagen (RGB, uint8) con las medidas dibujadas, a resolución nativa.
    En escala de grises la imagen pasa a grises pero las medidas conservan su color,
    así que el resultado sigue teniendo 3 canales."""
    base = image_rgb
    if grayscale:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        base = np.stack([gray] * 3, axis=-1)
    out = np.ascontiguousarray(base.copy())
    if not measures:
        return out
    h, w = out.shape[:2]
    # Qt toma el buffer de numpy como de solo lectura y pinta sobre una copia propia: el
    # resultado se lee de vuelta de la QImage (sus filas pueden venir alineadas a 4 bytes).
    qimage = QImage(out.data, w, h, 3 * w, QImage.Format_RGB888)
    painter = QPainter(qimage)
    thickness = max(3.0, max(h, w) / 500)
    font_px = max(16.0, max(h, w) / 70)
    for m in measures:
        draw_measure(painter, m.p1, m.p2, m.color, thickness, label_text(m, scale), font_px,
                     label_offset=m.label_offset)
    painter.end()
    stride = qimage.bytesPerLine()
    pixels = qimage.constBits()
    pixels.setsize(h * stride)
    return np.frombuffer(pixels, np.uint8).reshape(h, stride)[:, : 3 * w].reshape(h, w, 3).copy()
