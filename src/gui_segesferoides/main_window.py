"""View de la app: ventana principal PyQt5. No sabe nada de ningún modelo:
todo lo que necesita (clases, colores, magnificaciones, modelos disponibles,
predicción) se lo pide a un backend por el protocolo de docs/protocol.md.
"""

import csv
import os
import re

import cv2
import numpy as np
from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QShortcut,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui_segesferoides.config import BackendSpec, create_backend
from gui_segesferoides.editing import (
    UndoStack,
    fill_holes,
    fill_object_holes,
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
    TOOL_ERASER,
    TOOL_RECLASS,
    TOOL_ZONE,
    ImageCanvas,
)
from gui_segesferoides.installer import can_install, is_installed, plan, run_install
from gui_segesferoides.particle_dialog import ParticleDialog
from gui_segesferoides.viewport import build_pyramid

TOOL_LABELS = [
    (TOOL_ZONE, "Zona (Z)"),
    (TOOL_BRUSH, "Pincel (B)"),
    (TOOL_ERASER, "Borrador (E)"),
    (TOOL_DELETE, "Borrar objeto (D)"),
    (TOOL_RECLASS, "Cambiar clase de objeto (R)"),
    (TOOL_ANALYZE, "Analizar partícula (A)"),
]


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
    def __init__(self, specs):
        super().__init__()
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

        top = QHBoxLayout()
        layout.addLayout(top)
        open_button = QPushButton("Abrir imagen...")
        open_button.clicked.connect(self._on_open_image)
        top.addWidget(open_button)
        self.accept_button = QPushButton("Aceptar y guardar máscara")
        self.accept_button.clicked.connect(self._on_accept)
        self.accept_button.setEnabled(False)
        top.addWidget(self.accept_button)
        self.undo_button = QPushButton("Deshacer (Ctrl+Z)")
        self.undo_button.clicked.connect(self._undo)
        top.addWidget(self.undo_button)
        self.redo_button = QPushButton("Rehacer (Ctrl+Y)")
        self.redo_button.clicked.connect(self._redo)
        top.addWidget(self.redo_button)
        top.addStretch()
        top.addWidget(QLabel("Backend:"))
        self.combo_backend = QComboBox()
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

        self.canvas = ImageCanvas()
        self.canvas.stroke_started.connect(self._on_stroke_started)
        self.canvas.stroke_segment.connect(self._on_stroke_segment)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.stroke_finished.connect(self._on_stroke_finished)
        self.canvas.object_clicked.connect(self._on_object_clicked)
        layout.addWidget(self.canvas, stretch=1)

        tools = QHBoxLayout()
        layout.addLayout(tools)
        self.tool_buttons = {}
        tool_group = QButtonGroup(self)
        for tool, label in TOOL_LABELS:
            btn = QRadioButton(label)
            btn.toggled.connect(lambda checked, t=tool: checked and self._set_tool(t))
            tool_group.addButton(btn)
            tools.addWidget(btn)
            self.tool_buttons[tool] = btn
        self.tool_buttons[TOOL_ZONE].setChecked(True)

        brush = QHBoxLayout()
        layout.addLayout(brush)
        brush.addWidget(QLabel("Clase a pintar / asignar:"))
        self.combo_class = QComboBox()
        brush.addWidget(self.combo_class)
        self.slider_brush, self.label_brush = self._add_size_slider(brush, "Pincel", 10, self._on_brush_size)
        self.slider_eraser, self.label_eraser = self._add_size_slider(brush, "Borrador", 3, self._on_eraser_size)
        brush.addWidget(QLabel("Transparencia máscara:"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        self.slider_alpha.valueChanged.connect(self._refresh_canvas)
        brush.addWidget(self.slider_alpha)
        self.check_show_mask = QCheckBox("Ver máscara (M)")
        self.check_show_mask.setChecked(True)
        self.check_show_mask.toggled.connect(self._refresh_canvas)
        brush.addWidget(self.check_show_mask)
        self.label_zoom = QLabel("Zoom: -")
        brush.addWidget(self.label_zoom)
        fit_button = QPushButton("Ajustar (F)")
        fit_button.clicked.connect(self.canvas.fit_view)
        brush.addWidget(fit_button)
        fill_button = QPushButton("Rellenar huecos (clase)")
        fill_button.clicked.connect(self._on_fill_holes)
        brush.addWidget(fill_button)

        predict = QHBoxLayout()
        layout.addLayout(predict)
        self.radio_full = QRadioButton("Predecir toda la imagen")
        self.radio_zone = QRadioButton("Predecir zona seleccionada")
        self.radio_full.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.radio_full)
        mode_group.addButton(self.radio_zone)
        predict.addWidget(self.radio_full)
        predict.addWidget(self.radio_zone)
        predict.addWidget(QLabel("Magnificación:"))
        self.combo_magnification = QComboBox()
        predict.addWidget(self.combo_magnification)
        self.predict_button = QPushButton("Predecir")
        self.predict_button.clicked.connect(self._on_predict_clicked)
        self.predict_button.setEnabled(False)
        predict.addWidget(self.predict_button)
        predict.addStretch()
        self.export_button = QPushButton("Exportar resultados")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)
        predict.addWidget(self.export_button)

        self.status_label = QLabel("Abrí una imagen para empezar.")
        layout.addWidget(self.status_label)

    def _build_shortcuts(self):
        def sc(key, fn):
            QShortcut(QKeySequence(key), self, activated=fn)

        sc("Ctrl+Z", self._undo)
        sc("Ctrl+Y", self._redo)
        sc("Ctrl+S", self._on_accept)
        sc("Z", lambda: self.tool_buttons[TOOL_ZONE].setChecked(True))
        sc("B", lambda: self.tool_buttons[TOOL_BRUSH].setChecked(True))
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
            self.combo_class.addItem(f"{c.id} - {c.name}", c.id)
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

        self.canvas.set_image(build_pyramid(self.full_res_image), self.mask)

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
        self.particle_dialog.show_particle(class_id, object_stats(self.particle))
        self.canvas.highlight = object_contours(self.particle, holes=True)
        self.canvas.update()

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
        fields = ["image", "backend", "model", "object", "class_id", "class_name",
                  "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                  "area_px", "perimeter_px", "equivalent_diameter_px", "circularity"]
        try:
            rows = measure_objects(self.mask)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for i, r in enumerate(rows, 1):
                    writer.writerow({
                        "image": image, "backend": backend, "model": model, "object": i,
                        "class_id": r["class_id"], "class_name": names.get(r["class_id"], ""),
                        "centroid_x": f"{r['centroid_x']:.1f}", "centroid_y": f"{r['centroid_y']:.1f}",
                        "bbox_x": r["bbox_x"], "bbox_y": r["bbox_y"],
                        "bbox_w": r["bbox_w"], "bbox_h": r["bbox_h"],
                        "area_px": r["area"], "perimeter_px": f"{r['perimeter']:.2f}",
                        "equivalent_diameter_px": f"{r['equivalent_diameter']:.2f}",
                        "circularity": f"{r['circularity']:.4f}",
                    })
            save_mask(mask_path, self.mask, self.palette, mode="ids")
        except OSError as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))
            return
        self.status_label.setText(
            f"Exportados {len(rows)} objetos: {csv_path} y {os.path.basename(mask_path)}")

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
                    "Elegí la herramienta Zona y dibujá un rectángulo sobre la imagen primero.",
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
        magnification = self.combo_magnification.currentText()
        self._run_async(
            lambda: backend.predict(path, magnification, bbox),
            self._on_predict_finished,
            self._on_predict_failed,
            "Procesando... (puede tardar varios segundos)",
        )

    def _on_predict_finished(self, pred):
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
        self.canvas.update()

        counts = ", ".join(
            f"{c.name}: {int(np.sum(pred == c.id))}" for c in self.backend.info.classes
        )
        self.status_label.setText(f"Listo. Corregí lo que haga falta y aceptá. Píxeles — {counts}")

    def _on_predict_failed(self, message):
        self.status_label.setText("Error al predecir.")
        QMessageBox.critical(self, "Error al predecir", message)
