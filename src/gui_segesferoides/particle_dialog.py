"""Popup no modal con las medidas de la partícula seleccionada y acciones
sobre ella. Solo UI: la ventana principal le da los datos y escucha sus
señales; el diálogo no toca la máscara.
"""

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui_segesferoides.calibration import Scale

# (clave de object_stats, clave de la estadística de referencia, título, tipo)
METRICS = [
    ("area", "area", "Área", "area"),
    ("equivalent_diameter", "diameter", "Diámetro equivalente", "length"),
    ("circularity", "circularity", "Circularidad", "ratio"),
]


def format_value(kind, value, scale):
    """Texto de una medida (en píxeles) en la escala dada."""
    if kind == "area":
        return scale.area(value)
    if kind == "length":
        return scale.length(value)
    return f"{value:.2f}"
INSIDE, OUTSIDE = QColor("#2e9e5b"), QColor("#d9534f")


class RangeBar(QWidget):
    """Mini-plot: el intervalo [p5, p95] de referencia como una banda, la media
    como una marca fina y el valor actual como un punto (verde dentro del
    intervalo, rojo fuera; si se sale de la escala queda pegado al borde).
    Un click emite `clicked`."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(96, 18)
        self.setCursor(Qt.PointingHandCursor)
        self.value = self.low = self.high = self.mean = None

    def set_data(self, value, low, high, mean, interval_text):
        self.value, self.low, self.high, self.mean = value, low, high, mean
        self.setToolTip(f"Intervalo de referencia: {interval_text}\n(click para ver el detalle)")
        self.update()

    @property
    def inside(self):
        return self.low <= self.value <= self.high

    def _scale(self):
        """Rango visible: el intervalo con un margen a cada lado (sin pasar de
        0, ni de 1 si es una razón como la circularidad)."""
        margin = (self.high - self.low) * 0.6
        lo, hi = max(0.0, self.low - margin), self.high + margin
        if self.high <= 1.0:
            hi = min(hi, 1.0)
        return lo, hi

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def paintEvent(self, _event):
        if self.value is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        lo, hi = self._scale()
        pad, mid_y = 6.0, self.height() / 2
        span = self.width() - 2 * pad

        def x_of(v):
            return pad + span * min(1.0, max(0.0, (v - lo) / (hi - lo)))

        track = self.palette().mid().color()
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(pad, mid_y - 1.5, span, 3), 1.5, 1.5)
        band = QColor(self.palette().highlight().color())
        band.setAlpha(110)
        p.setBrush(band)
        x0, x1 = x_of(self.low), x_of(self.high)
        p.drawRoundedRect(QRectF(x0, mid_y - 4, x1 - x0, 8), 4, 4)
        p.setPen(QPen(self.palette().windowText().color(), 1.2))
        p.drawLine(int(x_of(self.mean)), int(mid_y - 5), int(x_of(self.mean)), int(mid_y + 5))
        p.setPen(QPen(self.palette().window().color(), 1.5))
        p.setBrush(INSIDE if self.inside else OUTSIDE)
        p.drawEllipse(QRectF(x_of(self.value) - 4.5, mid_y - 4.5, 9, 9))


class ReferenceDialog(QDialog):
    """Detalle de una métrica contra el set de entrenamiento: límites,
    media, valor actual y cuántos objetos hay de cada clase."""

    def __init__(self, title, value_text, ref, counts, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        fmt = ref["fmt"]
        rows = [
            ("Valor actual", value_text),
            ("Límite inferior (p5)", fmt(ref["p5"])),
            ("Media", fmt(ref["mean"])),
            ("Límite superior (p95)", fmt(ref["p95"])),
            (f"Objetos de {ref['class_label']} usados", str(ref["n"])),
        ]
        for label, text in rows:
            form.addRow(label + ":", QLabel(text))
        layout.addWidget(QLabel(
            f"<b>Objetos por clase del entrenamiento ({ref['magnification'] or 'todas'}):</b><br>"
            + "<br>".join(f"{name}: {n}" for name, n in counts)))
        close = QPushButton("Cerrar")
        close.clicked.connect(self.accept)
        layout.addWidget(close)


class ParticleDialog(QDialog):
    class_changed = pyqtSignal(int)
    fill_holes_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    closed = pyqtSignal()   # el usuario lo cerró (botón o X), no por código

    def __init__(self, classes, parent=None):
        """`classes`: [(id, nombre)] de las clases que declara el backend."""
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Partícula")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_class = QComboBox()
        for class_id, name in classes:
            self.combo_class.addItem(f"{class_id} - {name}", class_id)
        self.combo_class.activated.connect(
            lambda i: self.class_changed.emit(self.combo_class.itemData(i)))
        form.addRow("Clase:", self.combo_class)
        self._class_names = dict(classes)
        self.scale = Scale()
        self.magnification = ""   # la fija la ventana principal
        self.current_class = None
        self.reference = None   # estadística de referencia de la clase mostrada
        self._counts = []
        self._current = {}
        self.label_area = QLabel()
        self.label_perimeter = QLabel()
        self.label_diameter = QLabel()
        self.label_circularity = QLabel()
        self.bars = {key: RangeBar() for key, *_ in METRICS}
        for key, *_ in METRICS:
            self.bars[key].clicked.connect(lambda k=key: self._show_reference(k))
            self.bars[key].hide()
        form.addRow("Área:", self._with_bar(self.label_area, self.bars["area"]))
        form.addRow("Perímetro:", self.label_perimeter)   # sin estadística de referencia
        form.addRow("Diámetro equivalente:",
                    self._with_bar(self.label_diameter, self.bars["equivalent_diameter"]))
        form.addRow("Circularidad:", self._with_bar(self.label_circularity, self.bars["circularity"]))

        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.fill_button = QPushButton("Rellenar huecos")
        self.fill_button.clicked.connect(self.fill_holes_requested)
        self.delete_button = QPushButton("Eliminar")
        self.delete_button.clicked.connect(self.delete_requested)
        self.close_button = QPushButton("Cerrar")
        self.close_button.clicked.connect(self.close)
        for b in (self.fill_button, self.delete_button, self.close_button):
            buttons.addWidget(b)

    @staticmethod
    def _with_bar(label, bar):
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        box.addWidget(label)
        box.addWidget(bar)
        return row

    def show_particle(self, class_id, stats, reference=None, counts=(), scale=None):
        """`reference`: estadística de entrenamiento de esta clase y
        magnificación ({"n", "area": {mean,p5,p95}, ...}) o None; `counts`:
        [(nombre de clase, n)] para el popup de detalle."""
        self._current, self._counts = dict(stats), list(counts)
        self.reference = reference
        self.current_class = class_id
        self.scale = scale or Scale()
        self.combo_class.setCurrentIndex(self.combo_class.findData(class_id))
        self.label_area.setText(self.scale.area(stats["area"]))
        self.label_perimeter.setText(self.scale.length(stats["perimeter"]))
        self.label_diameter.setText(self.scale.length(stats["equivalent_diameter"]))
        self.label_circularity.setText(f"{stats['circularity']:.2f}")
        for key, ref_key, _title, kind in METRICS:
            r = (reference or {}).get(ref_key)
            self.bars[key].setVisible(bool(r))
            if r:
                fmt = lambda v, kind=kind: format_value(kind, v, self.scale)
                self.bars[key].set_data(stats[key], r["p5"], r["p95"], r["mean"],
                                        f"{fmt(r['p5'])} – {fmt(r['p95'])}")

    def _show_reference(self, key):
        _key, ref_key, title, kind = next(m for m in METRICS if m[0] == key)
        r = self.reference[ref_key]
        fmt = lambda v: format_value(kind, v, self.scale)
        ref = {**r, "fmt": fmt, "n": self.reference["n"], "magnification": self.magnification,
               "class_label": self._class_names.get(self.current_class, str(self.current_class))}
        ReferenceDialog(f"{title}: referencia del entrenamiento", fmt(self._current[key]),
                        ref, self._counts, self).exec_()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
