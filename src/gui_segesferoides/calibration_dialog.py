"""Diálogo para calibrar (1 px = X unidad) cada magnificación y elegir la
unidad en que se muestran las medidas. Solo UI: edita un `Calibration`."""

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from gui_segesferoides.calibration import DISPLAY_UNITS, MAGNIFICATIONS, UNITS_UM


class CalibrationDialog(QDialog):
    def __init__(self, calibration, magnifications=(), parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.setWindowTitle("Calibrar imagen")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Equivalencia de un píxel en longitud real, por magnificación\n"
                                "(0 = sin calibrar: se muestra en píxeles)."))
        form = QFormLayout()
        layout.addLayout(form)
        self.rows = {}   # {magnificación: (spin, combo de unidad)}
        names = list(dict.fromkeys([*MAGNIFICATIONS, *magnifications]))
        for mag in names:
            length, unit = calibration.entries.get(mag, (0.0, "µm"))
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(0, 1e9)
            spin.setValue(length)
            unit_combo = QComboBox()
            unit_combo.addItems(UNITS_UM)
            unit_combo.setCurrentText(unit)
            row = QHBoxLayout()
            row.addWidget(QLabel("1 px ="))
            row.addWidget(spin, stretch=1)
            row.addWidget(unit_combo)
            form.addRow(f"{mag}:", row)
            self.rows[mag] = (spin, unit_combo)
        self.combo_display = QComboBox()
        self.combo_display.addItems(DISPLAY_UNITS)
        self.combo_display.setCurrentText(calibration.display_unit)
        form.addRow("Mostrar medidas en:", self.combo_display)
        self.check_csv_stats = QCheckBox("Adjuntar datos estadísticos en el .csv")
        self.check_csv_stats.setToolTip("Agrega al CSV exportado, por objeto, el intervalo (p5, media, p95) "
                                        "del entrenamiento del modelo para su clase y magnificación.")
        self.check_csv_stats.setChecked(calibration.csv_stats)
        layout.addWidget(self.check_csv_stats)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self):
        for mag, (spin, unit_combo) in self.rows.items():
            self.calibration.set(mag, spin.value(), unit_combo.currentText())
        self.calibration.display_unit = self.combo_display.currentText()
        self.calibration.csv_stats = self.check_csv_stats.isChecked()
        self.accept()
