"""Widget de imagen con zoom/pan, selección de zona y herramientas de
edición (pincel, borrador, click sobre objeto). Solo UI: no sabe nada del
modelo. Todas las coordenadas que emite son de la imagen a resolución nativa
(el render y la matemática de la vista viven en app/viewport.py).

Controles: rueda = zoom hacia el cursor; botón derecho o central arrastrando
= mover (pan); botón izquierdo = herramienta activa.
"""

import math

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QLabel, QSizePolicy, QToolTip, QWidget

from gui_segesferoides import viewport
from gui_segesferoides.calibration import PX, Scale
from gui_segesferoides.measures import (
    DEFAULT_MEASURE_COLOR,
    MIN_LENGTH_PX,
    Measure,
    clamp_label_shift,
    clamp_point,
    clamp_translation,
    distance_to_segment,
    draw_measure,
    label_rect,
    label_text,
)

TOOL_ZONE = "zone"
TOOL_BRUSH = "brush"
TOOL_ERASER = "eraser"
TOOL_DELETE = "delete"
TOOL_RECLASS = "reclass"
TOOL_ANALYZE = "analyze"
TOOL_MEASURE = "measure"                 # trazar una línea de medida
TOOL_EDIT_MEASURE = "edit_measure"       # elegir una medida para eliminarla, recolorearla o moverla
STROKE_TOOLS = (TOOL_BRUSH, TOOL_ERASER)
CLICK_TOOLS = (TOOL_DELETE, TOOL_RECLASS, TOOL_ANALYZE)
MEASURE_HIT_PX = 8         # a cuántos px de pantalla de la línea cuenta como "click sobre la medida"
MEASURE_WIDTH = 2          # px de pantalla
MEASURE_FONT_PX = 12
HIGHLIGHT_COLOR = (255, 0, 255)
DISCARDED_COLOR = (255, 40, 40)   # contornos de lo que las reglas de post-procesado quitaron
HIGHLIGHT_WIDTH = 2        # px de pantalla: se ve igual con cualquier zoom


class ImageCanvas(QWidget):
    stroke_started = pyqtSignal()
    stroke_segment = pyqtSignal(float, float, float, float)  # x0, y0, x1, y1 (imagen)
    stroke_finished = pyqtSignal()
    object_clicked = pyqtSignal(float, float)                # x, y (imagen)
    zoom_changed = pyqtSignal(float)
    measures_changed = pyqtSignal()                          # se agregó, borró, recoloreó o movió una
    measure_clicked = pyqtSignal(int, QPoint)                # índice de la medida y posición global

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

        # Medidas: anotaciones de pantalla, aparte de la imagen y de la máscara.
        self.measures = []                        # [Measure], en coords de imagen
        self.measure_color = DEFAULT_MEASURE_COLOR  # color de la próxima que se trace
        self.selected_measure = None              # índice de la que se está editando
        self._measure_start = None                # (x, y) imagen: inicio del trazo en curso
        self._measure_end = None
        self._move_index = None                   # medida lista para moverse (tras «Mover»)
        self._moving = False
        self._move_part = "line"                  # qué se arrastra: "line" (toda la medida) o "label"
        self._move_last = None

        self.levels = None
        self.mask = None
        self.palette = {}          # {id de clase: (r, g, b)}, la declara el backend
        self.zoom = 1.0
        self.offset = (0.0, 0.0)
        self.selection = None      # (x, y, w, h) en coords de imagen
        self.highlight = []        # contornos (N, 2) en coords de imagen a resaltar
        self.discarded = []        # contornos de lo descartado por el post-procesado (en rojo)
        self.hover_text = None     # f(ix, iy) -> texto o None: tooltip de lo descartado
        self._fit_pending = True

        # Nombre de la imagen: un QLabel flotante sobre el canvas, no parte de la imagen.
        # No se dibuja en `levels` ni en la máscara, así que no llega al modelo ni a lo exportado.
        self._image_name = ""
        self.scale = Scale()       # px -> unidad de la calibración (px si no hay)
        self.name_label = QLabel(self)
        self.name_label.setAttribute(Qt.WA_TransparentForMouseEvents)   # no tapa clicks ni tooltips
        self.name_label.setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 150); color: white;"
                                      " padding: 3px 8px; border-radius: 4px; }")
        self.name_label.hide()

        self._mouse_view = None    # última posición del mouse (widget) para el cursor
        self._over_image = False   # el mouse está sobre la imagen (no sobre el fondo negro)
        self._last_point = None    # último punto del trazo (imagen)
        self._stroking = False
        self._panning = False
        self._pan_last = None
        self._zone_origin = None   # (x, y) imagen
        self._qimage = None        # referencia viva al buffer del último frame

    def _draw_contours(self, painter, contours, color, dashed=False):
        if not contours:
            return
        path = QPainterPath()
        for contour in contours:
            # +0.5: el contorno pasa por el centro de los píxeles del borde
            poly = QPolygonF([QPointF(*viewport.image_to_view(self.offset, self.zoom, x + 0.5, y + 0.5))
                              for x, y in contour])
            path.addPolygon(poly)
            path.closeSubpath()
        pen = QPen(QColor(*color), HIGHLIGHT_WIDTH)
        pen.setJoinStyle(Qt.RoundJoin)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    # ---------- API ----------
    def set_image(self, levels, mask):
        self.levels, self.mask = levels, mask
        self.selection = None
        self.highlight = []
        self.discarded = []
        self.measures = []                        # las medidas son de esa imagen
        self._reset_measure_state()
        self._fit_pending = True
        self.fit_view()

    # ---------- Medidas ----------
    def _reset_measure_state(self):
        self.selected_measure = None
        self._measure_start = self._measure_end = None
        self._move_index, self._moving, self._move_last = None, False, None
        self._move_part = "line"

    def measure_at(self, view_x, view_y):
        """Índice de la medida bajo el punto de pantalla (la más cercana), o None."""
        best, best_d = None, MEASURE_HIT_PX
        for i, m in enumerate(self.measures):
            a = viewport.image_to_view(self.offset, self.zoom, *m.p1)
            b = viewport.image_to_view(self.offset, self.zoom, *m.p2)
            d = distance_to_segment((view_x, view_y), a, b)
            if d <= best_d:
                best, best_d = i, d
        return best

    def _label_rect_view(self, index):
        """Rectángulo (de pantalla) de la etiqueta de la medida `index`."""
        m = self.measures[index]
        a = viewport.image_to_view(self.offset, self.zoom, *m.p1)
        b = viewport.image_to_view(self.offset, self.zoom, *m.p2)
        shift = (m.label_offset[0] * self.zoom, m.label_offset[1] * self.zoom)
        return label_rect(a, b, MEASURE_WIDTH, label_text(m, self.scale), MEASURE_FONT_PX, shift)

    def delete_measure(self, index):
        del self.measures[index]
        self._reset_measure_state()
        self.measures_changed.emit()
        self.update()

    def set_measure_color(self, index, color):
        self.measures[index] = self.measures[index].with_color(color)
        self.selected_measure = None
        self.measures_changed.emit()
        self.update()

    def begin_move(self, index):
        """Deja la medida lista para moverse. El próximo arrastre con el botón izquierdo mueve
        la etiqueta, si empieza sobre ella, o toda la medida (sin cambiar su largo, orientación
        ni color) si empieza en cualquier otro lado."""
        self._move_index, self.selected_measure = index, index
        self.setCursor(Qt.SizeAllCursor)
        self.update()

    def _draw_measures(self, painter):
        for i, m in enumerate(self.measures):
            a = viewport.image_to_view(self.offset, self.zoom, *m.p1)
            b = viewport.image_to_view(self.offset, self.zoom, *m.p2)
            draw_measure(painter, a, b, m.color, MEASURE_WIDTH, label_text(m, self.scale), MEASURE_FONT_PX,
                         highlighted=(i == self.selected_measure),
                         label_offset=(m.label_offset[0] * self.zoom, m.label_offset[1] * self.zoom))
        if self._measure_start is not None and self._measure_end is not None:     # trazo en curso
            preview = Measure(self._measure_start, self._measure_end, self.measure_color)
            a = viewport.image_to_view(self.offset, self.zoom, *preview.p1)
            b = viewport.image_to_view(self.offset, self.zoom, *preview.p2)
            draw_measure(painter, a, b, preview.color, MEASURE_WIDTH, label_text(preview, self.scale),
                         MEASURE_FONT_PX)

    def set_scale(self, scale):
        """Escala con la que la barra de referencia traduce píxeles a distancia."""
        self.scale = scale
        self.update()

    def scale_bar(self):
        """(largo en píxeles de imagen, texto) de la barra de escala, o None sin imagen.
        Con calibración dice a qué distancia equivale ("100 px = 200 µm")."""
        if self.levels is None:
            return None
        n = viewport.scale_bar_length(self.zoom)
        if self.scale.unit == PX:
            return n, f"{n} px"
        value = self.scale.convert_length(n)
        digits = max(0, 3 - math.floor(math.log10(value))) if value > 0 else 0
        number = f"{value:,.{digits}f}"
        number = number.rstrip("0").rstrip(".") if "." in number else number    # 200 µm, no 200.0 µm
        return n, f"{n} px = {number} {self.scale.unit}"

    def _draw_scale_bar(self, painter):
        bar = self.scale_bar()
        if bar is None:
            return
        n, text = bar
        length = n * self.zoom
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        box_w, box_h, margin = max(length, text_w) + 20, fm.height() + 26, 10
        box = QRectF(margin, self.height() - box_h - margin, box_w, box_h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(box.left() + 10, box.top() + 5, box_w - 20, fm.height()),
                         Qt.AlignLeft | Qt.AlignVCenter, text)
        y = box.bottom() - 10
        x0 = box.left() + 10
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(QPointF(x0, y), QPointF(x0 + length, y))
        painter.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        painter.drawLine(QPointF(x0 + length, y - 4), QPointF(x0 + length, y + 4))

    def set_image_name(self, name):
        """Etiqueta con el nombre de la imagen, abajo a la derecha ('' la oculta)."""
        self._image_name = name or ""
        self._place_name_label()

    def _place_name_label(self):
        if not self._image_name:
            self.name_label.hide()
            return
        room = max(80, int(self.width() * 0.6) - 20)           # nombres largos: se acortan al medio
        self.name_label.setText(
            self.name_label.fontMetrics().elidedText(self._image_name, Qt.ElideMiddle, room))
        self.name_label.adjustSize()
        self.name_label.move(self.width() - self.name_label.width() - 10,
                             self.height() - self.name_label.height() - 10)
        self.name_label.show()
        self.name_label.raise_()

    def set_tool(self, tool):
        if tool != self.tool:            # (el paneo vuelve a llamarla con la misma herramienta)
            self._reset_measure_state()
        self.tool = tool
        if tool != TOOL_ZONE:
            self.selection = None
        self.setCursor(self._tool_cursor())
        self.update()

    def _tool_cursor(self):
        """Cursor de la herramienta. Pincel y borrador dibujan su propio círculo y
        ocultan el del sistema, pero solo sobre la imagen: afuera, sobre el fondo, se
        vuelve a ver el cursor normal."""
        if self.tool in STROKE_TOOLS:
            return Qt.BlankCursor if self._over_image else Qt.ArrowCursor
        if self.tool == TOOL_EDIT_MEASURE and self._move_index is not None:
            return Qt.SizeAllCursor
        return Qt.CrossCursor if self.tool != TOOL_ZONE else Qt.ArrowCursor

    def _is_over_image(self, view_x, view_y):
        if self.levels is None:
            return False
        ix, iy = viewport.view_to_image(self.offset, self.zoom, view_x, view_y)
        img_h, img_w = self.levels[0].shape[:2]
        return 0 <= ix < img_w and 0 <= iy < img_h

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
        self._place_name_label()
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

        self._draw_contours(painter, self.discarded, DISCARDED_COLOR, dashed=True)
        self._draw_contours(painter, self.highlight, HIGHLIGHT_COLOR)
        self._draw_measures(painter)

        self._draw_scale_bar(painter)

        if self.tool in STROKE_TOOLS and self._mouse_view is not None and self._over_image:
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
        elif self.tool == TOOL_MEASURE:
            h, w = self.levels[0].shape[:2]
            self._measure_start = clamp_point((ix, iy), w, h)
            self._measure_end = self._measure_start
        elif self.tool == TOOL_EDIT_MEASURE:
            if self._move_index is not None:            # «Mover» elegido: este arrastre la corre
                on_label = self._label_rect_view(self._move_index).contains(event.x(), event.y())
                self._move_part = "label" if on_label else "line"
                self._moving, self._move_last = True, (ix, iy)
            else:
                index = self.measure_at(event.x(), event.y())
                if index is not None:
                    self.selected_measure = index
                    self.update()
                    self.measure_clicked.emit(index, event.globalPos())

    def mouseMoveEvent(self, event):
        self._mouse_view = (event.x(), event.y())
        over = self._is_over_image(event.x(), event.y())
        if over != self._over_image:
            self._over_image = over
            if not self._panning:          # al mover la vista, el cursor es la mano cerrada
                self.setCursor(self._tool_cursor())
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
        elif self._measure_start is not None:
            ix, iy = self._to_image(event.pos())
            h, w = self.levels[0].shape[:2]
            self._measure_end = clamp_point((ix, iy), w, h)
        elif self._moving:
            ix, iy = self._to_image(event.pos())
            h, w = self.levels[0].shape[:2]
            m = self.measures[self._move_index]
            want = (ix - self._move_last[0], iy - self._move_last[1])
            if self._move_part == "label":
                dx, dy = clamp_label_shift(m, want[0], want[1], w, h)
                self.measures[self._move_index] = m.with_label_offset(
                    (m.label_offset[0] + dx, m.label_offset[1] + dy))
            else:
                dx, dy = clamp_translation(m, want[0], want[1], w, h)
                self.measures[self._move_index] = m.translated(dx, dy)
            self._move_last = (self._move_last[0] + dx, self._move_last[1] + dy)
        elif self.discarded and self.hover_text is not None:
            ix, iy = self._to_image(event.pos())
            text = self.hover_text(int(ix), int(iy))
            if text:
                QToolTip.showText(event.globalPos(), text, self)
            else:
                QToolTip.hideText()
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
            elif self._measure_start is not None:
                self._finish_measure(event)
            elif self._moving:
                self._reset_measure_state()
                self.setCursor(self._tool_cursor())
                self.measures_changed.emit()
        self.update()

    def _finish_measure(self, event):
        ix, iy = self._to_image(event.pos())
        h, w = self.levels[0].shape[:2]
        measure = Measure(self._measure_start, clamp_point((ix, iy), w, h), self.measure_color)
        self._measure_start = self._measure_end = None
        if measure.length >= MIN_LENGTH_PX:                 # un click sin arrastrar no deja nada
            self.measures.append(measure)
            self.measures_changed.emit()

    def leaveEvent(self, _event):
        self._mouse_view = None
        self._over_image = False
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
