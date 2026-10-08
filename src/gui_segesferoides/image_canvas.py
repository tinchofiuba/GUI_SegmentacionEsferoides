"""Widget de imagen con zoom/pan, selección de zona y herramientas de
edición (pincel, borrador, click sobre objeto). Solo UI: no sabe nada del
modelo. Todas las coordenadas que emite son de la imagen a resolución nativa
(el render y la matemática de la vista viven en app/viewport.py).

Controles: rueda = zoom hacia el cursor; botón derecho o central arrastrando
= mover (pan); botón izquierdo = herramienta activa.
"""

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QSizePolicy, QWidget

from gui_segesferoides import viewport

TOOL_ZONE = "zone"
TOOL_BRUSH = "brush"
TOOL_ERASER = "eraser"
TOOL_DELETE = "delete"
TOOL_RECLASS = "reclass"
TOOL_ANALYZE = "analyze"
STROKE_TOOLS = (TOOL_BRUSH, TOOL_ERASER)
CLICK_TOOLS = (TOOL_DELETE, TOOL_RECLASS, TOOL_ANALYZE)
HIGHLIGHT_COLOR = (255, 0, 255)
HIGHLIGHT_WIDTH = 2        # px de pantalla: se ve igual con cualquier zoom


class ImageCanvas(QWidget):
    stroke_started = pyqtSignal()
    stroke_segment = pyqtSignal(float, float, float, float)  # x0, y0, x1, y1 (imagen)
    stroke_finished = pyqtSignal()
    object_clicked = pyqtSignal(float, float)                # x, y (imagen)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.tool = TOOL_ZONE
        self.brush_radius = 10     # px de imagen
        self.eraser_radius = 3
        self.alpha = 0.5
        self.show_mask = True

        self.levels = None
        self.mask = None
        self.palette = {}          # {id de clase: (r, g, b)}, la declara el backend
        self.zoom = 1.0
        self.offset = (0.0, 0.0)
        self.selection = None      # (x, y, w, h) en coords de imagen
        self.highlight = []        # contornos (N, 2) en coords de imagen a resaltar
        self._fit_pending = True

        self._mouse_view = None    # última posición del mouse (widget) para el cursor
        self._last_point = None    # último punto del trazo (imagen)
        self._stroking = False
        self._panning = False
        self._pan_last = None
        self._zone_origin = None   # (x, y) imagen
        self._qimage = None        # referencia viva al buffer del último frame

    # ---------- API ----------
    def set_image(self, levels, mask):
        self.levels, self.mask = levels, mask
        self.selection = None
        self.highlight = []
        self._fit_pending = True
        self.fit_view()

    def set_tool(self, tool):
        self.tool = tool
        if tool != TOOL_ZONE:
            self.selection = None
        self.setCursor(Qt.BlankCursor if tool in STROKE_TOOLS else
                       Qt.CrossCursor if tool != TOOL_ZONE else Qt.ArrowCursor)
        self.update()

    def clear_selection(self):
        self.selection = None
        self.update()

    def fit_view(self):
        if self.levels is None:
            return
        img_h, img_w = self.levels[0].shape[:2]
        self.zoom = viewport.fit_zoom(img_w, img_h, self.width(), self.height())
        self.offset = viewport.fit_offset(img_w, img_h, self.width(), self.height(), self.zoom)
        self.zoom_changed.emit(self.zoom)
        self.update()

    def _active_radius(self):
        return self.eraser_radius if self.tool == TOOL_ERASER else self.brush_radius

    # ---------- Render ----------
    def resizeEvent(self, event):
        if self._fit_pending:
            self.fit_view()
        super().resizeEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        if self.levels is None:
            gray = viewport.BACKGROUND_GRAY
            painter.fillRect(self.rect(), QColor(gray, gray, gray))
            return
        w, h = self.width(), self.height()
        frame = np.ascontiguousarray(viewport.render_view(
            self.levels, self.mask, self.palette, self.offset, self.zoom, w, h,
            self.alpha, self.show_mask))
        self._qimage = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        painter.drawImage(0, 0, self._qimage)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.selection is not None:
            x, y, sw, sh = self.selection
            vx, vy = viewport.image_to_view(self.offset, self.zoom, x, y)
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            painter.drawRect(QRectF(vx, vy, sw * self.zoom, sh * self.zoom))

        if self.highlight:
            path = QPainterPath()
            for contour in self.highlight:
                # +0.5: el contorno pasa por el centro de los píxeles del borde
                poly = QPolygonF([QPointF(*viewport.image_to_view(self.offset, self.zoom, x + 0.5, y + 0.5))
                                  for x, y in contour])
                path.addPolygon(poly)
                path.closeSubpath()
            pen = QPen(QColor(*HIGHLIGHT_COLOR), HIGHLIGHT_WIDTH)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

        if self.tool in STROKE_TOOLS and self._mouse_view is not None:
            r = self._active_radius() * self.zoom
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0), 3))
            painter.drawEllipse(QPointF(*self._mouse_view), r, r)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(QPointF(*self._mouse_view), r, r)

    # ---------- Mouse ----------
    def _to_image(self, pos):
        return viewport.view_to_image(self.offset, self.zoom, pos.x(), pos.y())

    def mousePressEvent(self, event):
        if self.levels is None:
            return
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = True
            self._pan_last = (event.x(), event.y())
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        ix, iy = self._to_image(event.pos())
        if self.tool == TOOL_ZONE:
            self._zone_origin = (ix, iy)
            self.selection = None
        elif self.tool in STROKE_TOOLS:
            self._stroking = True
            self._last_point = (ix, iy)
            self.stroke_started.emit()
            self.stroke_segment.emit(ix, iy, ix, iy)
            self.update()
        elif self.tool in CLICK_TOOLS:
            self.object_clicked.emit(ix, iy)

    def mouseMoveEvent(self, event):
        self._mouse_view = (event.x(), event.y())
        if self._panning:
            dx, dy = event.x() - self._pan_last[0], event.y() - self._pan_last[1]
            self.offset = (self.offset[0] - dx / self.zoom, self.offset[1] - dy / self.zoom)
            self._pan_last = (event.x(), event.y())
            self._fit_pending = False
        elif self._stroking:
            ix, iy = self._to_image(event.pos())
            self.stroke_segment.emit(self._last_point[0], self._last_point[1], ix, iy)
            self._last_point = (ix, iy)
        elif self._zone_origin is not None:
            ix, iy = self._to_image(event.pos())
            self.selection = self._clamped_rect(self._zone_origin, (ix, iy))
        self.update()

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = False
            self.set_tool(self.tool)  # restaura el cursor de la herramienta
        elif event.button() == Qt.LeftButton:
            if self._zone_origin is not None:
                ix, iy = self._to_image(event.pos())
                rect = self._clamped_rect(self._zone_origin, (ix, iy))
                self.selection = rect if rect and rect[2] > 2 and rect[3] > 2 else None
                self._zone_origin = None
            elif self._stroking:
                self._stroking = False
                self.stroke_finished.emit()
        self.update()

    def leaveEvent(self, _event):
        self._mouse_view = None
        self.update()

    def wheelEvent(self, event):
        if self.levels is None:
            return
        img_h, img_w = self.levels[0].shape[:2]
        fit = viewport.fit_zoom(img_w, img_h, self.width(), self.height())
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        new_zoom = min(viewport.MAX_ZOOM, max(fit * viewport.MIN_ZOOM_FACTOR, self.zoom * factor))
        self.offset = viewport.zoom_at(self.offset, self.zoom, new_zoom, (event.x(), event.y()))
        self.zoom = new_zoom
        self._fit_pending = False
        self._mouse_view = (event.x(), event.y())
        self.zoom_changed.emit(self.zoom)
        self.update()

    def _clamped_rect(self, a, b):
        img_h, img_w = self.levels[0].shape[:2]
        x0, x1 = sorted((min(max(a[0], 0), img_w), min(max(b[0], 0), img_w)))
        y0, y1 = sorted((min(max(a[1], 0), img_h), min(max(b[1], 0), img_h)))
        return (x0, y0, x1 - x0, y1 - y0)
