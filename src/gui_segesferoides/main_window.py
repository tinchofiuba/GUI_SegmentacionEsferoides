"""View de la app: ventana principal PyQt5. No sabe nada de ningún modelo:
todo lo que necesita (clases, colores, magnificaciones, modelos disponibles,
predicción) se lo pide a un backend por el protocolo de docs/protocol.md.
"""

import csv
import math
import os
import re

import cv2
import numpy as np
from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QShortcut,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui_segesferoides.calibration import PX, Calibration
from gui_segesferoides.calibration_dialog import CalibrationDialog
from gui_segesferoides.config import BackendSpec, create_backend
from gui_segesferoides.editing import (
    UndoStack,
    count_objects,
    fill_holes,
    fill_object_holes,
    fill_polygon,
    measure_objects,
    object_contours,
    object_region,
    object_stats,
    paint_segment,
    replace_object,
    save_mask,
)
from gui_segesferoides.image_canvas import (
    TOOL_ANALYZE,
    TOOL_BRUSH,
    TOOL_DELETE,
    TOOL_EDIT_MEASURE,
    TOOL_ERASER,
    TOOL_LASSO,
    TOOL_MEASURE,
    TOOL_RECLASS,
    TOOL_ZONE,
    ImageCanvas,
)
from gui_segesferoides.measures import MEASURE_COLORS, render_measures
from gui_segesferoides.installer import can_install, is_installed, plan, run_install
from gui_segesferoides.particle_dialog import SELECTED, ParticleDialog
from gui_segesferoides.postprocess_dialog import PostprocessDialog
from gui_segesferoides.postprocess_store import PostprocessStore
from gui_segesferoides.viewport import build_pyramid

TOOL_LABELS = [
    (TOOL_ZONE, "Zona (Z)"),
    (TOOL_BRUSH, "Pincel (B)"),
    (TOOL_LASSO, "Lazo (L)"),
    (TOOL_ERASER, "Borrador (E)"),
    (TOOL_DELETE, "Borrar objeto (D)"),
    (TOOL_RECLASS, "Cambiar clase de objeto (R)"),
    (TOOL_ANALYZE, "Analizar objeto (A)"),
    (TOOL_MEASURE, "Medir distancia"),
    (TOOL_EDIT_MEASURE, "Editar medida"),
]


# (clave en la estadística de referencia, clave en object_stats), en el orden del CSV
REFERENCE_METRICS = [("area", "area"), ("perimeter", "perimeter"),
                     ("diameter", "equivalent_diameter"), ("circularity", "circularity")]
REFERENCE_KIND = {"area": "area", "perimeter": "length", "diameter": "length", "circularity": None}
CSV_UNIT = {"µm": "um"}   # sufijo ASCII de las columnas: area_um2, perimeter_um...
REFERENCE_FIELDS = ["ref_n"] + [f"ref_{k}_{q}" for k, _ in REFERENCE_METRICS
                                for q in ("p5", "mean", "p95", "in_range")]


SAVE_FORMATS = [("PNG", ".png"), ("JPEG", ".jpg"), ("TIFF", ".tif"), ("BMP", ".bmp")]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _color_icon(rgb, size=14):
    """Cuadradito de color para los combos y menús de colores."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class CallWorker(QThread):
    """Corre `fn()` fuera del hilo de la UI (cargar/predecir bloquean)."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario, no se traga
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, specs, calibration_path=None, postprocess_path=None):
        super().__init__()
        self.calibration = Calibration(calibration_path)
        self.postprocess_store = PostprocessStore(postprocess_path)
        self.discarded = None       # uint8 (como la máscara): clase original de lo que descartaron las reglas
        self.discarded_reasons = []  # [{x, y, rule, conditions...}] en coords de imagen (ver protocol.md)
        self.discarded_labels = None  # int32: id de objeto descartado por píxel (0 = nada)
        self.discarded_texts = {}    # {id de objeto: texto del tooltip}
        self.specs = list(specs)             # [BackendSpec]
        self.backend = None                  # SubprocessBackend o CliBackend conectado
        self.model_loaded = False
        self.palette = {}                    # {id de clase: (r, g, b)}, del backend
        self.busy = False                    # una llamada al backend en curso
        self.worker = None
        self.settings = QSettings("GUI_SegmentacionEsferoides", "gui")
        self.full_res_image = None  # np.ndarray RGB, resolución nativa
        self.image_path = None
        self.mask = None            # np.ndarray uint8 con ids de clase, resolución nativa
        self.undo_stack = UndoStack()
        self.pending_bbox = None    # bbox (resolución nativa) de la predicción en curso
        self.pending_magnification = ""   # magnificación de la predicción en curso
        self.origin = "manual"      # manual | modelo | modelo_corregido
        self.dirty = False
        self.uninstalled = None     # BackendSpec elegido pero no instalado
        self.particle = None        # máscara booleana de la partícula analizada
        self.particle_dialog = None

        self.setWindowTitle("Tagging y predicción de segmentaciones")
        self._build_ui()
        self._build_shortcuts()
        self._update_controls()
        if self.specs:
            self._connect_backend(self.specs[0])
        else:
            self.status_label.setText("No hay backends configurados (ver config/backends.yaml).")

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addLayout(self._build_top_bar())

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)
        self.canvas = ImageCanvas()
        self.canvas.stroke_started.connect(self._on_stroke_started)
        self.canvas.stroke_segment.connect(self._on_stroke_segment)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.stroke_finished.connect(self._on_stroke_finished)
        self.canvas.object_clicked.connect(self._on_object_clicked)
        self.canvas.hover_text = self._discarded_text_at
        body.addWidget(self.canvas, stretch=1)
        body.addWidget(self._build_side_panel())

        bottom = QHBoxLayout()
        layout.addLayout(bottom)
        self.status_label = QLabel("Abrí una imagen para empezar.")
        bottom.addWidget(self.status_label, stretch=1)
        self.label_zoom = QLabel("Zoom: -")
        bottom.addWidget(self.label_zoom)
        fit_button = QPushButton("Ajustar (F)")
        fit_button.clicked.connect(self.canvas.fit_view)
        bottom.addWidget(fit_button)

        self.radio_full.setChecked(True)   # recién ahora existen el canvas y todas las herramientas
        self.tool_buttons[TOOL_BRUSH].setChecked(True)

    def _build_top_bar(self):
        """Archivo e historial a la izquierda; backend y modelo a la derecha."""
        top = QHBoxLayout()
        open_button = QPushButton("Abrir imagen...")
        open_button.clicked.connect(self._on_open_image)
        top.addWidget(open_button)
        self.accept_button = QPushButton("Aceptar y guardar máscara")
        self.accept_button.clicked.connect(self._on_accept)
        self.accept_button.setEnabled(False)
        top.addWidget(self.accept_button)
        self.export_button = QPushButton("Exportar resultados")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)
        top.addWidget(self.export_button)
        self.undo_button = QPushButton("Deshacer (Ctrl+Z)")
        self.undo_button.clicked.connect(self._undo)
        top.addWidget(self.undo_button)
        self.redo_button = QPushButton("Rehacer (Ctrl+Y)")
        self.redo_button.clicked.connect(self._redo)
        top.addWidget(self.redo_button)
        top.addStretch()
        top.addWidget(QLabel("Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for spec in self.specs:
            self.combo_backend.addItem(spec.name, spec)
        self.combo_backend.activated.connect(self._on_backend_selected)
        top.addWidget(self.combo_backend)
        top.addWidget(QLabel("Modelo:"))
        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(280)
        self.combo_model.activated.connect(self._on_model_selected)
        top.addWidget(self.combo_model)
        self.browse_button = QPushButton("Examinar...")
        self.browse_button.clicked.connect(self._on_browse_model)
        top.addWidget(self.browse_button)
        self.install_button = QPushButton("Instalar backend")
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        top.addWidget(self.install_button)
        return top

    def _build_side_panel(self):
        """Un grupo por función: predicción, edición, análisis y vista."""
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        # Un solo grupo exclusivo: la herramienta activa del canvas es una sola,
        # aunque sus botones estén repartidos entre los grupos del panel.
        tool_group = QButtonGroup(self)
        self.tool_buttons = {}

        def tool_button(tool, label):
            btn = QRadioButton(label)
            btn.toggled.connect(lambda checked, t=tool: checked and self._set_tool(t))
            tool_group.addButton(btn)
            self.tool_buttons[tool] = btn
            return btn

        def row(*widgets):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            for item in widgets:
                h.addWidget(item)
            return w

        def tint(group, name, background, border):
            """Fondo propio por grupo, solo para ese QGroupBox (el selector por nombre
            no se propaga a los botones y combos de adentro)."""
            group.setObjectName(name)
            self._group_colors[name] = (background, border)
            self._style_group(group, active=False)

        self._group_colors = {}      # {nombre del grupo: (fondo, borde)}
        self.group_boxes = {}        # {nombre del grupo: QGroupBox}

        # --- Predicción ---
        group = QGroupBox("Predicción")
        tint(group, "grupoPrediccion", "#e3f0fb", "#9cc3e6")
        self.group_boxes["Predicción"] = group
        g = QVBoxLayout(group)
        g.setSpacing(1)
        self.radio_full = QRadioButton("Imagen completa")
        self.radio_zone = QRadioButton("Zona seleccionada (Z)")
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.radio_full)
        mode_group.addButton(self.radio_zone)
        self.radio_zone.toggled.connect(self._on_prediction_mode)
        g.addWidget(self.radio_full)
        g.addWidget(self.radio_zone)
        # La herramienta Zona es la forma de dibujar el rectángulo del modo "zona":
        # no tiene botón propio, la activa el modo. Cuelga del panel, oculta y fuera
        # de todo layout, para que no sea un widget huérfano.
        zone_button = tool_button(TOOL_ZONE, "Zona")
        zone_button.setParent(panel)
        zone_button.hide()
        self.zone_hint = QLabel("Dibujá un rectángulo sobre la imagen.")
        self.zone_hint.setWordWrap(True)
        self.zone_hint.setVisible(False)
        g.addWidget(self.zone_hint)
        self.combo_magnification = QComboBox()
        self.combo_magnification.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_magnification.currentTextChanged.connect(self._update_scale)
        g.addWidget(row(QLabel("Magnificación:"), self.combo_magnification))
        self.predict_button = QPushButton("Predecir")
        self.predict_button.clicked.connect(self._on_predict_clicked)
        self.predict_button.setEnabled(False)
        g.addWidget(self.predict_button)
        self.postprocess_button = QPushButton("Post-proceso...")
        self.postprocess_button.setToolTip("Reglas con las que el backend filtra la predicción")
        self.postprocess_button.clicked.connect(self._on_postprocess)
        g.addWidget(self.postprocess_button)
        self.check_discarded = QCheckBox("Ver descartados (rojo)")
        self.check_discarded.setToolTip("Contornos rojos de lo que las reglas de post-procesado "
                                        "dejaron afuera en la última predicción (solo informativo).")
        self.check_discarded.toggled.connect(self._refresh_discarded)
        g.addWidget(self.check_discarded)
        box.addWidget(group)

        # --- Edición ---
        group = QGroupBox("Edición")
        tint(group, "grupoEdicion", "#fbe6e6", "#e0a3a3")
        self.group_boxes["Edición"] = group
        g = QVBoxLayout(group)
        g.setSpacing(1)
        for tool, label in TOOL_LABELS:
            if tool in (TOOL_BRUSH, TOOL_LASSO, TOOL_ERASER, TOOL_DELETE, TOOL_RECLASS):
                g.addWidget(tool_button(tool, label))
        self.lasso_hint = QLabel("Marcá puntos con clic (o arrastrá a mano alzada): la curva pasa por "
                                 "todos. Cerrala con clic sobre el primer punto, doble clic o Enter; "
                                 "Esc cancela. Se rellena con la clase elegida.")
        self.lasso_hint.setWordWrap(True)
        self.lasso_hint.setVisible(False)
        g.addWidget(self.lasso_hint)
        self.combo_class = QComboBox()
        self.combo_class.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_class.currentIndexChanged.connect(self._update_lasso_color)
        self.row_class = row(QLabel("Clase:"), self.combo_class)
        g.addWidget(self.row_class)
        # Cada tamaño aparece solo con su herramienta.
        self.row_brush = QWidget()
        self.slider_brush, self.label_brush = self._add_size_slider(
            QHBoxLayout(self.row_brush), "Pincel", 10, self._on_brush_size)
        self.row_eraser = QWidget()
        self.slider_eraser, self.label_eraser = self._add_size_slider(
            QHBoxLayout(self.row_eraser), "Borrador", 3, self._on_eraser_size)
        for r in (self.row_brush, self.row_eraser):
            r.layout().setContentsMargins(0, 0, 0, 0)
            g.addWidget(r)
        fill_button = QPushButton("Rellenar huecos (clase)")
        fill_button.clicked.connect(self._on_fill_holes)
        g.addWidget(fill_button)
        box.addWidget(group)

        # --- Análisis ---
        group = QGroupBox("Análisis")
        tint(group, "grupoAnalisis", "#e3f5e6", "#9fd0a8")
        self.group_boxes["Análisis"] = group
        g = QVBoxLayout(group)
        g.setSpacing(1)
        labels = dict(TOOL_LABELS)
        g.addWidget(tool_button(TOOL_ANALYZE, labels[TOOL_ANALYZE]))
        # Medir distancia: el color elegido es el de la PRÓXIMA medida; cada una conserva el suyo.
        self.combo_measure_color = QComboBox()
        for name, rgb in MEASURE_COLORS:
            self.combo_measure_color.addItem(_color_icon(rgb), name, rgb)
        self.combo_measure_color.setToolTip("Color de la próxima medida")
        self.combo_measure_color.currentIndexChanged.connect(self._on_measure_color)
        g.addWidget(row(tool_button(TOOL_MEASURE, labels[TOOL_MEASURE]), self.combo_measure_color))
        g.addWidget(tool_button(TOOL_EDIT_MEASURE, labels[TOOL_EDIT_MEASURE]))
        self.canvas.measure_color = self.combo_measure_color.currentData()
        self.canvas.measure_clicked.connect(self._on_measure_clicked)
        self.canvas.lasso_finished.connect(self._on_lasso_finished)
        self.calibrate_button = QPushButton("Calibrar...")
        self.calibrate_button.clicked.connect(self._on_calibrate)
        g.addWidget(self.calibrate_button)
        box.addWidget(group)

        # --- Vista ---
        group = QGroupBox("Vista")
        g = QVBoxLayout(group)
        g.setSpacing(1)
        self.check_show_mask = QCheckBox("Ver máscara (M)")
        self.check_show_mask.setChecked(True)
        self.check_show_mask.toggled.connect(self._refresh_canvas)
        g.addWidget(self.check_show_mask)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        self.slider_alpha.valueChanged.connect(self._refresh_canvas)
        g.addWidget(row(QLabel("Transparencia:"), self.slider_alpha))
        # Guardar la imagen de esta vista (solo esta imagen; la máscara se exporta aparte, en .png).
        self.save_image_button = QPushButton("Guardar imagen")
        self.save_image_button.setToolTip("Guarda la imagen como se ve, a resolución nativa: con la máscara "
                                          "(si «Ver máscara» está marcado) y las medidas")
        self.save_image_button.clicked.connect(self._on_save_image)
        self.combo_save_format = QComboBox()
        for name, ext in SAVE_FORMATS:
            self.combo_save_format.addItem(name, ext)
        self.combo_save_channels = QComboBox()
        self.combo_save_channels.addItem("Color", "color")
        self.combo_save_channels.addItem("Escala de grises", "gray")
        self.combo_save_channels.setToolTip("Escala de grises: la imagen pasa a grises, pero las medidas "
                                            "conservan sus colores")
        g.addWidget(row(self.save_image_button, self.combo_save_format))
        g.addWidget(row(QLabel("Canales:"), self.combo_save_channels))
        box.addWidget(group)

        box.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(320)
        return scroll

    def _on_measure_color(self, _index):
        self.canvas.measure_color = self.combo_measure_color.currentData()

    def _on_measure_clicked(self, index, global_pos):
        """Click con «Editar medida» sobre una medida: menú Eliminar / Cambiar color / Mover."""
        menu = QMenu(self)
        delete = menu.addAction("Eliminar")
        color_menu = menu.addMenu("Cambiar color")
        for name, rgb in MEASURE_COLORS:
            color_menu.addAction(_color_icon(rgb), name).setData(rgb)
        move = menu.addAction("Mover")
        chosen = menu.exec_(global_pos)
        if chosen is delete:
            self.canvas.delete_measure(index)
        elif chosen is move:
            self.canvas.begin_move(index)
        elif chosen is not None and chosen in color_menu.actions():
            self.canvas.set_measure_color(index, chosen.data())
        else:                                        # menú cerrado sin elegir: deja de resaltarla
            self.canvas.selected_measure = None
            self.canvas.update()

    def _on_save_image(self):
        if self.full_res_image is None:
            return
        ext = self.combo_save_format.currentData()
        name = self.combo_save_format.currentText()
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        suffix = "_medidas" if self.canvas.measures else "_copia"   # nunca pisa la imagen original
        default_dir = self._last_dir("save_dir") or os.path.dirname(self.image_path)
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", os.path.join(default_dir, f"{stem}{suffix}{ext}"), f"{name} (*{ext})")
        if not path:
            return
        root, old_ext = os.path.splitext(path)
        path = (root if old_ext.lower() in IMAGE_EXTENSIONS else path) + ext   # el formato lo manda el combo
        self.settings.setValue("save_dir", os.path.dirname(path))
        self.save_image(path)

    def save_image(self, path):
        """Guarda la imagen a resolución nativa, como se ve en pantalla: con la máscara (si «Ver
        máscara» está marcado, con la transparencia elegida) y las medidas dibujadas, con sus
        colores y distancias. En escala de grises la imagen pasa a grises, pero la máscara y
        las medidas conservan su color. El formato sale de la extensión de `path`."""
        scale = self.calibration.scale(self.combo_magnification.currentText())
        gray = self.combo_save_channels.currentData() == "gray"
        show_mask = self.check_show_mask.isChecked()
        out = render_measures(self.full_res_image, self.canvas.measures, scale, grayscale=gray,
                              mask=self.mask if show_mask else None, palette=self.palette,
                              alpha=self.slider_alpha.value() / 100)
        if not cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR)):
            QMessageBox.critical(self, "Error al guardar", f"No se pudo guardar la imagen:\n{path}")
            return False
        self.status_label.setText(f"Imagen guardada: {path}")
        return True

    def _style_group(self, group, active):
        """Estilo de un grupo del panel. El grupo que tiene la herramienta activa se enmarca
        con el color de selección (el mismo de los intervalos de confianza); el borde mide
        siempre lo mismo, así que el panel no se mueve al cambiar de herramienta."""
        name = group.objectName()
        background, border = self._group_colors[name]
        frame = SELECTED.name() if active else border
        group.setStyleSheet(
            f"QGroupBox#{name} {{ background-color: {background}; border: 2px solid {frame};"
            f" border-radius: 4px; margin-top: 10px; padding-top: 8px; }}"
            f"QGroupBox#{name}::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; }}"
            f"QGroupBox#{name}, QGroupBox#{name} QLabel, QGroupBox#{name} QRadioButton,"
            f" QGroupBox#{name} QCheckBox {{ color: #1f2933; }}")

    def _highlight_active_group(self, tool):
        """Enmarca el grupo al que pertenece la herramienta activa: así se nota con qué
        zona del panel se está trabajando (Predicción: zona; Edición; Análisis)."""
        owner = {TOOL_ZONE: "Predicción", TOOL_BRUSH: "Edición", TOOL_LASSO: "Edición", TOOL_ERASER: "Edición",
                 TOOL_DELETE: "Edición", TOOL_RECLASS: "Edición", TOOL_ANALYZE: "Análisis",
                 TOOL_MEASURE: "Análisis", TOOL_EDIT_MEASURE: "Análisis"}[tool]
        self.active_group = owner
        for title, group in self.group_boxes.items():
            self._style_group(group, active=(title == owner))

    def _on_prediction_mode(self, zone):
        """El modo "zona" activa la herramienta Zona (y desmarca las de edición y análisis);
        volver a "imagen completa" la suelta."""
        self.zone_hint.setVisible(zone)
        if zone:
            self.tool_buttons[TOOL_ZONE].setChecked(True)
        elif self.canvas.tool == TOOL_ZONE:
            self.tool_buttons[TOOL_BRUSH].setChecked(True)

    def _build_shortcuts(self):
        def sc(key, fn):
            QShortcut(QKeySequence(key), self, activated=fn)

        sc("Ctrl+Z", self._undo)
        sc("Ctrl+Y", self._redo)
        sc("Ctrl+S", self._on_accept)
        sc("Z", lambda: self.radio_zone.setChecked(True))
        sc("B", lambda: self.tool_buttons[TOOL_BRUSH].setChecked(True))
        sc("L", lambda: self.tool_buttons[TOOL_LASSO].setChecked(True))
        sc("E", lambda: self.tool_buttons[TOOL_ERASER].setChecked(True))
        sc("D", lambda: self.tool_buttons[TOOL_DELETE].setChecked(True))
        sc("R", lambda: self.tool_buttons[TOOL_RECLASS].setChecked(True))
        sc("A", lambda: self.tool_buttons[TOOL_ANALYZE].setChecked(True))
        sc("M", lambda: self.check_show_mask.toggle())
        sc("F", self.canvas.fit_view)
        sc("]", lambda: self._scale_active_size(1.25))
        sc("[", lambda: self._scale_active_size(0.8))
        for i in range(1, 10):  # teclas 1..9 = n-ésima clase del backend
            sc(str(i), lambda i=i: self.combo_class.setCurrentIndex(min(i - 1, self.combo_class.count() - 1)))

    def _add_size_slider(self, layout, name, value, on_change):
        layout.addWidget(QLabel(f"{name}:"))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, 150)
        slider.setValue(value)
        label = QLabel(f"{value}px")
        layout.addWidget(slider)
        layout.addWidget(label)
        # El callback usa self.label_*, que recién existen cuando esto retorna:
        # se conecta a una función que actualiza la etiqueta local.
        def changed(v):
            label.setText(f"{v}px")
            on_change(v)

        slider.valueChanged.connect(changed)
        on_change(value)
        return slider, label

    def _on_brush_size(self, value):
        self.canvas.brush_radius = value
        self.canvas.update()

    def _on_eraser_size(self, value):
        self.canvas.eraser_radius = value
        self.canvas.update()

    def _scale_active_size(self, factor):
        slider = self.slider_eraser if self.canvas.tool == TOOL_ERASER else self.slider_brush
        v = slider.value()
        scaled = round(v * factor)
        if scaled == v:  # a tamaños chicos, el factor no mueve el entero
            scaled += 1 if factor > 1 else -1
        slider.setValue(max(1, scaled))

    def _on_zoom_changed(self, zoom):
        self.label_zoom.setText(f"Zoom: {zoom * 100:.0f}%")

    def _set_mask(self, mask):
        self.mask = mask
        self.canvas.mask = mask
        self.canvas.update()

    def _set_tool(self, tool):
        if tool != TOOL_ANALYZE:
            self._clear_particle()
        self.canvas.set_tool(tool)
        self.row_class.setVisible(tool in (TOOL_BRUSH, TOOL_LASSO, TOOL_RECLASS))
        self.lasso_hint.setVisible(tool == TOOL_LASSO)
        self.row_brush.setVisible(tool == TOOL_BRUSH)
        self.row_eraser.setVisible(tool == TOOL_ERASER)
        self._highlight_active_group(tool)
        if tool != TOOL_ZONE and self.radio_zone.isChecked():
            # Una sola cosa activa a la vez: elegir pincel, borrador, etc. suelta el modo
            # "zona seleccionada" (y la predicción vuelve a ser sobre la imagen completa).
            self.radio_full.setChecked(True)

    # ---------- Backend / modelo ----------
    def _run_async(self, fn, on_done, on_fail=None, status=None):
        """Ejecuta una llamada al backend en un hilo; una a la vez."""
        self.busy = True
        self._update_controls()
        if status:
            self.status_label.setText(status)
        if self.worker is not None:
            self.worker.wait()  # el anterior ya emitió su resultado; que termine antes de soltarlo
        self.worker = CallWorker(fn)
        self.worker.done.connect(lambda r: self._finish_async(on_done, r))
        self.worker.failed.connect(lambda m: self._finish_async(on_fail or self._show_error, m))
        self.worker.start()

    def _finish_async(self, callback, arg):
        self.busy = False
        callback(arg)
        self._update_controls()  # después del callback: ya actualizó el estado

    def _show_error(self, message):
        self.status_label.setText("Error.")
        QMessageBox.critical(self, "Error", message)

    def _update_controls(self):
        has_backend = self.backend is not None
        free = not self.busy
        self.combo_backend.setEnabled(free and len(self.specs) > 1)
        self.combo_model.setEnabled(free and has_backend)
        self.browse_button.setVisible(has_backend and self.backend.info.accepts_model_path)
        self.browse_button.setEnabled(free)
        self.install_button.setVisible(self.uninstalled is not None)
        self.install_button.setEnabled(free and self.uninstalled is not None
                                       and can_install(self.uninstalled))
        self.export_button.setEnabled(free and has_backend and self.mask is not None)
        supports = has_backend and self.backend.info.supports_postprocess
        self.save_image_button.setEnabled(self.full_res_image is not None)
        self.postprocess_button.setVisible(bool(supports))
        self.postprocess_button.setEnabled(free and bool(supports) and self.model_loaded)
        self.check_discarded.setVisible(bool(supports))
        self.predict_button.setEnabled(
            free and has_backend and self.model_loaded and self.full_res_image is not None
        )

    def _connect_backend(self, spec: BackendSpec):
        def connect():
            backend = create_backend(spec)
            try:
                return backend, backend.list_models()
            except Exception:
                backend.close()
                raise

        old = self.backend
        self.backend, self.model_loaded = None, False
        if old is not None:
            old.close()
        self.uninstalled = None
        if not is_installed(spec):
            self._show_uninstalled(spec)
            return
        self._run_async(connect, self._on_backend_connected, self._on_backend_failed,
                        f"Iniciando backend '{spec.name}'...")

    def _show_uninstalled(self, spec):
        self.uninstalled = spec
        self.combo_model.clear()
        self.status_label.setText(
            f"El backend '{spec.name}' no está instalado."
            + ("" if can_install(spec) else " (Sin pasos `install` en config/backends.yaml.)"))
        self._update_controls()

    def _on_install_clicked(self):
        spec = self.uninstalled
        commands = "\n".join("  " + " ".join(cmd) for cmd, _ in plan(spec))
        reply = QMessageBox.question(
            self, "Instalar backend",
            f"Se va a instalar '{spec.name}' (descarga de la red, puede tardar varios minutos):\n\n"
            f"{commands}\n\n¿Continuar?")
        if reply != QMessageBox.Yes:
            return
        self._run_async(lambda: run_install(spec), lambda _: self._connect_backend(spec),
                        self._show_error, f"Instalando '{spec.name}'... (puede tardar varios minutos)")

    def _on_backend_failed(self, message):
        self.status_label.setText("No se pudo iniciar el backend.")
        QMessageBox.critical(self, "Error al iniciar el backend", message)

    def _on_backend_connected(self, result):
        self.backend, models = result
        info = self.backend.info
        self._clear_particle()
        self.particle_dialog = None  # se recrea con las clases del nuevo backend
        self.palette = info.palette
        self.canvas.palette = self.palette
        self.combo_class.clear()
        for c in info.classes:
            icon = _color_icon(info.palette[c.id]) if c.id in info.palette else QIcon()
            self.combo_class.addItem(icon, f"{c.id} - {c.name}", c.id)   # con el color de la clase
        self._update_lasso_color()
        self.combo_magnification.clear()
        self.combo_magnification.addItems(info.magnifications)
        self.combo_magnification.setEnabled(bool(info.magnifications))
        self._fill_models(models)
        self._update_controls()
        self.canvas.update()
        self.status_label.setText(f"Backend '{info.name}' listo.")

    def _fill_models(self, models):
        self.combo_model.clear()
        current = None
        for i, m in enumerate(models):
            note = f"  [{m.note}]" if m.note else ""
            self.combo_model.addItem(m.label + note, m.id)
            if m.current:
                current = i
        self.model_loaded = current is not None
        if current is not None:
            self.combo_model.setCurrentIndex(current)
        else:
            self.combo_model.setCurrentIndex(-1)
            self.status_label.setText("Elegí un modelo para poder predecir.")

    def _on_backend_selected(self, index):
        spec = self.combo_backend.itemData(index)
        if self.backend is None or spec.name != self.backend.info.name:
            self._connect_backend(spec)

    def _on_model_selected(self, index):
        self._load_model(self.combo_model.itemData(index))

    def _on_browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Elegir carpeta de un modelo")
        if path:
            self._load_model(path)

    def _load_model(self, model_id):
        backend = self.backend

        def refresh_models(_description):
            self.model_loaded = True
            self.status_label.setText(f"Modelo cargado: {_description}")
            if self.combo_model.findData(model_id) < 0:
                self.combo_model.addItem(os.path.basename(model_id.rstrip("/")), model_id)
            self.combo_model.setCurrentIndex(self.combo_model.findData(model_id))

        def failed(message):
            QMessageBox.critical(self, "Error al cargar el modelo", message)
            self._fill_models(backend.list_models())  # vuelve a reflejar cuál quedó cargado
            self._update_controls()
            self.status_label.setText("No se pudo cargar el modelo.")

        self._run_async(lambda: backend.load_model(model_id), refresh_models, failed,
                        f"Cargando modelo {os.path.basename(model_id.rstrip('/'))}...")

    # ---------- Imagen / render ----------
    def _on_open_image(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", self._last_dir("open_dir"),
            "Imágenes (*.png *.tif *.tiff *.jpg *.jpeg)",
        )
        if not path:
            return
        self.settings.setValue("open_dir", os.path.dirname(path))
        self.open_image(path)

    def open_image(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            QMessageBox.warning(self, "Error", f"No se pudo leer la imagen:\n{path}")
            return

        self._clear_particle()
        self.image_path = path
        self.full_res_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        full_h, full_w = self.full_res_image.shape[:2]
        self.mask = np.zeros((full_h, full_w), dtype=np.uint8)
        self.undo_stack.clear()
        self.origin = "manual"
        self.dirty = False

        self.discarded = None
        self.discarded_reasons, self.discarded_labels, self.discarded_texts = [], None, {}
        self.canvas.set_image(build_pyramid(self.full_res_image), self.mask)
        self.canvas.set_image_name(os.path.basename(path))

        self.accept_button.setEnabled(True)
        self._update_controls()
        self._select_magnification_from_name(path)
        self.status_label.setText(
            f"Imagen cargada: {full_w}x{full_h}px. Rueda = zoom, botón derecho = mover."
        )

    def _select_magnification_from_name(self, path):
        """Si el nombre del archivo trae una magnificación (`..._10x`, `-4X`),
        la elige en el combo, sin distinguir mayúsculas. Si no está entre las
        opciones del backend, no toca nada."""
        stem = os.path.splitext(os.path.basename(path))[0]
        for found in re.findall(r"(?<![A-Za-z0-9])(\d+)[xX](?![A-Za-z0-9])", stem):
            for i in range(self.combo_magnification.count()):
                if self.combo_magnification.itemText(i).lower() == f"{found}x":
                    self.combo_magnification.setCurrentIndex(i)
                    return

    def _mask_filename(self):
        """`<imagen>-Mask.png`: el nombre que espera el entrenamiento de
        u-resnet (`prepare_data`), que busca la máscara de `nombre_10x.png`
        como `nombre-Mask.png`. Por eso se quita del nombre una magnificación
        final (`_4x`, `-10X`); la magnificación no va en el nombre."""
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        stem = re.sub(r"[_\-\s]*(?<![A-Za-z0-9])\d+[xX]$", "", stem) or stem
        return f"{stem}-Mask.png"

    def _last_dir(self, key):
        value = self.settings.value(key, "")
        return value if value and os.path.isdir(value) else ""

    def _refresh_canvas(self, *_):
        self.canvas.alpha = self.slider_alpha.value() / 100
        self.canvas.show_mask = self.check_show_mask.isChecked()
        self.canvas.update()

    # ---------- Edición ----------
    def _selected_class(self):
        return self.combo_class.currentData()

    def _mark_edited(self):
        self.dirty = True
        if self.origin == "modelo":
            self.origin = "modelo_corregido"

    def _on_stroke_started(self):
        self.undo_stack.push(self.mask)

    def _on_stroke_segment(self, x0, y0, x1, y1):
        if self.mask is None:
            return
        if self.canvas.tool == TOOL_ERASER:
            class_id, radius = 0, self.canvas.eraser_radius
        else:
            class_id, radius = self._selected_class(), self.canvas.brush_radius
        paint_segment(self.mask, x0, y0, x1, y1, radius, class_id)
        self.canvas.update()

    def _on_stroke_finished(self):
        self._mark_edited()

    def _update_lasso_color(self, *_):
        """El lazo se dibuja con el color de la clase que está elegida."""
        rgb = self.palette.get(self._selected_class())
        if rgb is not None:
            self.canvas.lasso_color = tuple(rgb)
            self.canvas.update()

    def _on_lasso_finished(self, polygon):
        """El lazo se cerró: el interior pasa a ser un objeto de la clase elegida (pisa lo que
        hubiera), con un solo paso de deshacer."""
        if self.mask is None:
            return
        self.undo_stack.push(self.mask)
        if fill_polygon(self.mask, polygon, self._selected_class()):
            self._mark_edited()
            self.canvas.update()
            name = self.combo_class.currentText().split(" - ", 1)[-1]
            self.status_label.setText(f"Objeto agregado con el lazo: {name}.")
        else:
            self.undo_stack.discard_last()                     # no cambió nada: descartar el snapshot

    def _on_object_clicked(self, x, y):
        if self.mask is None:
            return
        if self.canvas.tool == TOOL_ANALYZE:
            self._select_particle(x, y)
            return
        new_class = 0 if self.canvas.tool == TOOL_DELETE else self._selected_class()
        self.undo_stack.push(self.mask)
        if replace_object(self.mask, x, y, new_class):
            self._mark_edited()
            self.canvas.update()
        else:
            self.undo_stack.discard_last()  # no cambió nada: descartar el snapshot
            self.status_label.setText("No hay ningún objeto bajo el cursor.")

    def _on_fill_holes(self):
        if self.mask is None:
            return
        self._clear_particle()
        self.undo_stack.push(self.mask)
        fill_holes(self.mask, self._selected_class())
        self._mark_edited()
        self.canvas.update()

    def _undo(self):
        self._restore(self.undo_stack.undo(self.mask) if self.mask is not None else None)

    def _redo(self):
        self._restore(self.undo_stack.redo(self.mask) if self.mask is not None else None)

    def _restore(self, mask):
        if mask is None:
            return
        self._clear_particle()
        self._set_mask(mask)
        self.dirty = True

    # ---------- Calibración ----------
    def _update_scale(self, *_):
        """La barra de escala del canvas sigue a la magnificación y a la calibración."""
        self.canvas.set_scale(self.calibration.scale(self.combo_magnification.currentText()))

    def _on_calibrate(self):
        magnifications = self.backend.info.magnifications if self.backend else ()
        dialog = CalibrationDialog(self.calibration, magnifications, self)
        if dialog.exec_() != dialog.Accepted:
            return
        try:
            self.calibration.save()
        except OSError as exc:
            QMessageBox.critical(self, "No se pudo guardar la calibración", str(exc))
        self._update_scale()
        if self.particle is not None:
            self._refresh_particle()   # se reexpresa en la unidad nueva

    # ---------- Análisis de partícula ----------
    def _select_particle(self, x, y):
        """Click con la herramienta Analizar: selecciona la partícula bajo el
        cursor; click otra vez sobre la misma (o en el fondo) la deselecciona."""
        region = object_region(self.mask, x, y)
        if region is None or (self.particle is not None and self.particle[int(y), int(x)]):
            self._clear_particle()
            return
        self.particle = region
        if self.particle_dialog is None:
            dialog = ParticleDialog([(c.id, c.name) for c in self.backend.info.classes], self)
            dialog.class_changed.connect(self._on_particle_class_changed)
            dialog.fill_holes_requested.connect(self._on_particle_fill_holes)
            dialog.delete_requested.connect(self._on_particle_delete)
            dialog.closed.connect(self._clear_particle)
            self.particle_dialog = dialog
        self._refresh_particle()
        self.particle_dialog.show()

    def _refresh_particle(self):
        class_id = int(self.mask[self.particle][0])
        reference, counts = self._reference_for(class_id)
        scale = self.calibration.scale(self.combo_magnification.currentText())
        self.particle_dialog.show_particle(class_id, object_stats(self.particle), reference, counts, scale)
        self.canvas.highlight = object_contours(self.particle, holes=True)
        self.canvas.update()

    def _reference_for(self, class_id):
        """Estadística de entrenamiento de la clase (a la magnificación elegida)
        y la cantidad de objetos de cada clase, si el backend la ofrece."""
        stats = None if self.busy else self.backend.reference_stats()
        magnification = self.combo_magnification.currentText()
        self.particle_dialog.magnification = magnification
        per_class = (stats or {}).get(magnification) or {}
        counts = [(c.name, per_class[str(c.id)]["n"])
                  for c in self.backend.info.classes if str(c.id) in per_class]
        return per_class.get(str(class_id)), counts

    def _clear_particle(self):
        self.particle = None
        self.canvas.highlight = []
        self.canvas.update()
        if self.particle_dialog is not None:
            self.particle_dialog.hide()

    def _on_particle_class_changed(self, class_id):
        if self.particle is None or self.mask[self.particle][0] == class_id:
            return
        self.undo_stack.push(self.mask)
        self.mask[self.particle] = class_id
        self._mark_edited()
        self._refresh_particle()

    def _on_particle_fill_holes(self):
        if self.particle is None:
            return
        self.undo_stack.push(self.mask)
        self.particle = fill_object_holes(self.mask, self.particle)
        self._mark_edited()
        self._refresh_particle()

    def _on_particle_delete(self):
        if self.particle is None:
            return
        reply = QMessageBox.question(
            self.particle_dialog, "Eliminar partícula",
            "¿Eliminar esta partícula de la máscara?")
        if reply != QMessageBox.Yes:
            return
        self.undo_stack.push(self.mask)
        self.mask[self.particle] = 0
        self._mark_edited()
        self._clear_particle()

    # ---------- Guardado ----------
    def _on_accept(self):
        if self.mask is None:
            return
        if not np.any(self.mask):
            reply = QMessageBox.question(
                self, "Máscara vacía", "La máscara no tiene ningún objeto. ¿Guardarla igual?"
            )
            if reply != QMessageBox.Yes:
                return
        default_dir = self._last_dir("save_dir") or os.path.dirname(self.image_path)
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar máscara", os.path.join(default_dir, self._mask_filename()),
            "PNG (*.png)",
        )
        if not path:
            return
        self.settings.setValue("save_dir", os.path.dirname(path))
        self.save_to(path)

    def save_to(self, path):
        try:
            save_mask(path, self.mask, self.palette, mode="ids")
        except OSError as exc:
            QMessageBox.critical(self, "Error al guardar", str(exc))
            return
        self.dirty = False
        self.status_label.setText(f"Máscara guardada en {path}")

    def _on_export(self):
        if self.mask is None:
            return
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        default_dir = self._last_dir("save_dir") or os.path.dirname(self.image_path)
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar resultados", os.path.join(default_dir, f"{stem}_resultados.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        self.settings.setValue("save_dir", os.path.dirname(path))
        self.export_results(path)

    def export_results(self, csv_path):
        """CSV con una fila por objeto + PNG de 1 canal con los ids de clase
        del backend (0 = fondo), junto al CSV, como `<imagen>-Mask.png`."""
        mask_path = os.path.join(os.path.dirname(csv_path), self._mask_filename())
        names = {c.id: c.name for c in self.backend.info.classes}
        backend = self.backend.info.name
        model = self.combo_model.currentData() if self.model_loaded else ""
        image = os.path.basename(self.image_path)
        magnification = self.combo_magnification.currentText()
        scale = self.calibration.scale(magnification)       # px si no hay calibración o se eligió px
        suffix = CSV_UNIT.get(scale.unit, scale.unit)
        area_col = "area_px" if scale.unit == PX else f"area_{suffix}2"
        fields = ["image", "backend", "model", "magnification", "unit",
                  "um_per_px_4x", "um_per_px_10x", "object", "class_id", "class_name",
                  "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                  area_col, f"perimeter_{suffix}", f"equivalent_diameter_{suffix}", "circularity"]
        # Calibración de cada magnificación, siempre en µm por píxel (vacío si no está calibrada).
        um_per_px = {m: "" if self.calibration.um_per_px(m) is None
                     else f"{self.calibration.um_per_px(m):.6g}" for m in ("4x", "10x")}
        reference = self._csv_reference()
        if reference is not None:
            fields += REFERENCE_FIELDS
        try:
            rows = measure_objects(self.mask)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for i, r in enumerate(rows, 1):
                    writer.writerow({
                        "image": image, "backend": backend, "model": model,
                        "magnification": magnification, "unit": suffix,
                        "um_per_px_4x": um_per_px["4x"], "um_per_px_10x": um_per_px["10x"],
                        "object": i,
                        "class_id": r["class_id"], "class_name": names.get(r["class_id"], ""),
                        "centroid_x": f"{r['centroid_x']:.1f}", "centroid_y": f"{r['centroid_y']:.1f}",
                        "bbox_x": r["bbox_x"], "bbox_y": r["bbox_y"],
                        "bbox_w": r["bbox_w"], "bbox_h": r["bbox_h"],
                        area_col: self._csv_number(scale, "area", r["area"]),
                        f"perimeter_{suffix}": self._csv_number(scale, "length", r["perimeter"]),
                        f"equivalent_diameter_{suffix}": self._csv_number(scale, "length", r["equivalent_diameter"]),
                        "circularity": f"{r['circularity']:.4f}",
                        **(self._reference_columns(reference, r, scale) if reference is not None else {}),
                    })
            save_mask(mask_path, self.mask, self.palette, mode="ids")
        except OSError as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))
            return
        self.status_label.setText(
            f"Exportados {len(rows)} objetos: {csv_path} y {os.path.basename(mask_path)}")

    def _csv_reference(self):
        """Estadística de entrenamiento de la magnificación elegida, para
        adjuntarla al CSV (solo si está activado y el backend la ofrece)."""
        if not self.calibration.csv_stats:
            return None
        stats = self.backend.reference_stats() or {}
        return stats.get(self.combo_magnification.currentText())

    @staticmethod
    def _csv_number(scale, kind, px):
        """Medida (en píxeles) en la unidad de `scale`, como texto del CSV. En píxeles
        conserva el formato de siempre (área entera, longitudes con 2 decimales);
        calibrada, 6 cifras significativas en notación decimal (sin exponente)."""
        if scale.unit == PX:
            return f"{px:.0f}" if kind == "area" else f"{px:.2f}"
        value = scale.convert_area(px) if kind == "area" else scale.convert_length(px)
        if value == 0:
            return "0"
        text = f"{value:.{max(0, 5 - math.floor(math.log10(abs(value))))}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    @staticmethod
    def _reference_columns(reference, row, scale):
        """Columnas ref_* de un objeto, en la misma unidad que el resto de las medidas del
        CSV; vacías si el modelo no tiene estadística para su clase. `in_range` se decide
        en píxeles (no depende de la escala)."""
        ref = reference.get(str(row["class_id"]), {})
        out = {name: "" for name in REFERENCE_FIELDS}
        if ref:
            out["ref_n"] = ref.get("n", "")
        for ref_key, stat_key in REFERENCE_METRICS:
            r = ref.get(ref_key)
            if not r:
                continue
            kind = REFERENCE_KIND[ref_key]
            for q in ("p5", "mean", "p95"):
                out[f"ref_{ref_key}_{q}"] = (f"{r[q]:.4g}" if kind is None or scale.unit == PX
                                             else MainWindow._csv_number(scale, kind, r[q]))
            out[f"ref_{ref_key}_in_range"] = "yes" if r["p5"] <= row[stat_key] <= r["p95"] else "no"
        return out

    def _confirm_discard(self):
        if not self.dirty:
            return True
        reply = QMessageBox.question(
            self, "Cambios sin guardar", "Hay cambios sin guardar en la máscara. ¿Descartarlos?"
        )
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        if self._confirm_discard():
            if self.backend is not None:
                self.backend.close()
            event.accept()
        else:
            event.ignore()

    # ---------- Predicción ----------
    def _on_predict_clicked(self):
        if self.full_res_image is None or self.backend is None:
            return

        bbox = None
        if self.radio_zone.isChecked():
            sel = self.canvas.selection
            if sel is None:
                QMessageBox.information(
                    self, "Zona no seleccionada",
                    "Dibujá un rectángulo sobre la imagen primero.",
                )
                return
            full_h, full_w = self.full_res_image.shape[:2]
            x, y = int(sel[0]), int(sel[1])
            bbox = (x, y, max(1, min(int(sel[2]), full_w - x)), max(1, min(int(sel[3]), full_h - y)))

        if self.dirty and np.any(self.mask):
            reply = QMessageBox.question(
                self, "Sobrescribir",
                "Predecir reemplaza la máscara " + ("de la zona" if bbox else "completa")
                + " (se puede deshacer). ¿Continuar?",
            )
            if reply != QMessageBox.Yes:
                return

        self.pending_bbox = bbox
        backend, path = self.backend, self.image_path
        magnification = self.pending_magnification = self.combo_magnification.currentText()
        user_rules = self.postprocess_store.get(backend.info.name)
        self._run_async(
            lambda: backend.predict_with_discarded(path, magnification, bbox, user_rules),
            self._on_predict_finished,
            self._on_predict_failed,
            "Procesando... (puede tardar varios segundos)",
        )

    def _on_predict_finished(self, result):
        pred, discarded, reasons = result
        bbox = self.pending_bbox
        full_h, full_w = self.mask.shape
        expected = (bbox[3], bbox[2]) if bbox else (full_h, full_w)
        unknown = set(np.unique(pred).tolist()) - {0} - set(self.palette)
        if pred.shape != expected or unknown:
            detail = (f"forma {pred.shape} en lugar de {expected}" if pred.shape != expected
                      else f"ids de clase no declarados: {sorted(unknown)}")
            self._on_predict_failed(f"El backend devolvió una máscara inválida: {detail}")
            return

        self._clear_particle()
        had_content = bool(np.any(self.mask))
        self.undo_stack.push(self.mask)
        if bbox is None:
            self._set_mask(pred)
        else:
            x, y, w, h = bbox
            self.mask[y : y + h, x : x + w] = pred
        # mezcla de tags previos y modelo => ya no es "puro modelo"
        self.origin = "modelo_corregido" if had_content else "modelo"
        self.dirty = True
        self._store_discarded(discarded, bbox, reasons)
        self.canvas.update()

        found = count_objects(pred)             # cantidad de detecciones por clase, no de píxeles
        counts = ", ".join(f"{c.name}: {found.get(c.id, 0)}" for c in self.backend.info.classes)
        self.status_label.setText(f"Listo. Corregí lo que haga falta y aceptá. Detecciones — {counts}")

    def _store_discarded(self, discarded, bbox, reasons=()):
        """Guarda lo que las reglas descartaron (a tamaño de imagen; con zona, solo
        se reemplaza esa zona), con el motivo de cada objeto, y actualiza los
        contornos rojos."""
        if discarded is None:
            if bbox is None:
                self.discarded, self.discarded_reasons = None, []
        else:
            ox, oy = (0, 0) if bbox is None else bbox[:2]
            if self.discarded is None or bbox is None:
                self.discarded = np.zeros(self.mask.shape, np.uint8)
                self.discarded_reasons = []
            if bbox is None:
                self.discarded[:] = discarded
            else:
                x, y, w, h = bbox
                self.discarded[y : y + h, x : x + w] = discarded
                self.discarded_reasons = [  # los de la zona se reemplazan por los nuevos
                    r for r in self.discarded_reasons
                    if not (x <= r["x"] < x + w and y <= r["y"] < y + h)]
            self.discarded_reasons += [{**r, "x": r["x"] + ox, "y": r["y"] + oy,
                                        "magnification": self.pending_magnification} for r in reasons]
        self._index_discarded()
        self._refresh_discarded()

    def _index_discarded(self):
        """Etiqueta cada objeto descartado (8-conexo, por clase original, como lo
        separa el backend) y le asocia el texto de su motivo."""
        self.discarded_labels, self.discarded_texts = None, {}
        if self.discarded is None:
            return
        labels = np.zeros(self.discarded.shape, np.int32)
        offset = 0
        for c in (int(v) for v in np.unique(self.discarded) if v != 0):
            n, lab = cv2.connectedComponents((self.discarded == c).astype(np.uint8), connectivity=8)
            labels[lab > 0] = lab[lab > 0] + offset
            offset += n - 1
        self.discarded_labels = labels
        h, w = labels.shape
        for reason in self.discarded_reasons:
            x, y = reason["x"], reason["y"]
            if 0 <= x < w and 0 <= y < h and labels[y, x]:
                self.discarded_texts[int(labels[y, x])] = self._discarded_text(reason)

    def _discarded_text_at(self, ix, iy):
        """Tooltip del objeto descartado bajo el cursor (None si no hay)."""
        labels = self.discarded_labels
        if labels is None or not (0 <= iy < labels.shape[0] and 0 <= ix < labels.shape[1]):
            return None
        label = int(labels[iy, ix])
        if not label:
            return None
        return self.discarded_texts.get(label, "Descartado (el backend no informó el motivo).")

    def _discarded_text(self, reason):
        names = {c.id: c.name for c in self.backend.info.classes} if self.backend else {}
        label = lambda cid: f"{cid} - {names[cid]}" if cid in names else str(cid)
        metrics = {"area": ("Área", "px²", 0), "diameter": ("Diámetro equivalente", "px", 1),
                   "circularity": ("Circularidad", "", 3), "perimeter": ("Perímetro", "px", 1)}
        ops = {"lt": "<", "le": "≤", "gt": ">", "ge": "≥"}
        title = reason.get("rule") or "(regla sin nombre)"
        lines = [f"Descartado por la regla «{title}»"]
        cls = f"Clase {label(reason['class'])}"
        as_class = reason.get("as_class", reason["class"])
        if as_class != reason["class"]:
            cls += f" (reclasificado antes como {label(as_class)})"
        lines.append(cls)
        if reason.get("magnification"):
            lines.append(f"Comparado con la estadística de {reason['magnification']}")
        for cond in reason.get("conditions", []):
            name, unit, digits = metrics.get(cond["metric"], (cond["metric"], "", 2))
            ref = f"  [{cond['reference']}]" if cond.get("reference") else ""
            lines.append(f"• {name} = {cond['actual']:.{digits}f} {ops.get(cond['op'], cond['op'])} "
                         f"{cond['value']:.{digits}f}{' ' + unit if unit else ''}{ref}")
        return "\n".join(lines)

    def _refresh_discarded(self):
        """Contornos externos de lo descartado, solo si el checkbox está marcado."""
        contours = []
        if self.check_discarded.isChecked() and self.discarded is not None:
            found, _ = cv2.findContours((self.discarded > 0).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [c.reshape(-1, 2).astype(np.float64) for c in found]
        self.canvas.discarded = contours
        self.canvas.update()

    def _on_postprocess(self):
        backend = self.backend
        names = [(c.id, c.name) for c in backend.info.classes]

        def open_editor(rules):
            config = self.postprocess_store.get(backend.info.name) or rules
            dialog = PostprocessDialog(config, rules["defaults"], names, self)
            if dialog.exec_() != dialog.Accepted:
                return
            edited = dialog.config()
            try:
                # lo igual a lo que trae el modelo no se guarda: así sigue las reglas del modelo
                same = edited == {"rules": rules["rules"], "fill_holes": rules["fill_holes"]}
                self.postprocess_store.set(backend.info.name, None if same else edited)
            except OSError as exc:
                QMessageBox.critical(self, "No se pudieron guardar las reglas", str(exc))
                return
            self.status_label.setText("Reglas de post-procesado guardadas: se aplican en la próxima predicción.")

        self._run_async(backend.get_postprocess, open_editor, self._show_error,
                        "Leyendo reglas del backend...")

    def _on_predict_failed(self, message):
        self.status_label.setText("Error al predecir.")
        QMessageBox.critical(self, "Error al predecir", message)
