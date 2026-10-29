"""Ventana con la estadística de referencia del dataset (ver dataset_stats.py),
por magnificación y por clase, y el botón "Recalcular..." para rehacerla sobre
las imágenes (con máscara) que el usuario elija."""

import os

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from gui_segesferoides import dataset_stats

COLUMNS = ["Clase", "Métrica", "n", "p5", "Media", "p95"]


def _fmt(value):
    return f"{value:.3g}" if abs(value) < 10 else f"{value:,.1f}"


class StatsDialog(QDialog):
    def __init__(self, store, classes, palette=None, start_dir="", parent=None):
        """`store`: DatasetStats; `classes`: [(id, nombre)]; `palette`: {id: rgb} para
        leer máscaras a color; `start_dir`: carpeta inicial del selector."""
        super().__init__(parent)
        self.setWindowTitle("Estadística del dataset")
        self.resize(560, 480)
        self.store = store
        self._classes = list(classes)
        self._palette = palette
        self._start_dir = start_dir

        outer = QVBoxLayout(self)
        self.label_info = QLabel()
        self.label_info.setWordWrap(True)
        outer.addWidget(self.label_info)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, stretch=1)

        row = QHBoxLayout()
        outer.addLayout(row)
        self.button_recalc = QPushButton("Recalcular...")
        self.button_recalc.setToolTip("Elegí imágenes .png con su máscara (<nombre>-Mask.png): la estadística "
                                      "se rehace con los objetos de esas máscaras.")
        self.button_recalc.clicked.connect(self._recalculate)
        row.addWidget(self.button_recalc)
        row.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        self._refresh()

    def _name(self, class_id):
        return dict(self._classes).get(int(class_id), f"clase {class_id}")

    def _refresh(self):
        data = self.store.data
        self.tabs.clear()
        if not data or not data["stats"]:
            self.label_info.setText("Todavía no hay estadística. Usá \"Recalcular...\" para generarla desde "
                                    "imágenes con máscara.")
            return
        self.label_info.setText(f"Calculada con {data['n_images']} imágenes ({data['updated']}). "
                                "Unidades: píxeles nativos.")
        for mag in sorted(data["stats"], key=lambda m: int(m[:-1])):
            per_class = data["stats"][mag]
            table = QTableWidget(0, len(COLUMNS))
            table.setHorizontalHeaderLabels(COLUMNS)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            for class_id in sorted(per_class, key=int):
                stats = per_class[class_id]
                for key, label in dataset_stats.METRICS:
                    r = table.rowCount()
                    table.insertRow(r)
                    cells = [f"{class_id} - {self._name(class_id)}", label, str(stats["n"]),
                             *(_fmt(stats[key][q]) for q in ("p5", "mean", "p95"))]
                    for c, text in enumerate(cells):
                        table.setItem(r, c, QTableWidgetItem(text))
            table.resizeColumnsToContents()
            self.tabs.addTab(table, mag)

    def _recalculate(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Imágenes con máscara", self._start_dir,
                                                "Imágenes (*.png)")
        if not paths:
            return
        self._start_dir = os.path.dirname(paths[0])
        progress = QProgressDialog("Midiendo objetos...", "", 0, len(paths), self)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        try:
            data, skipped = dataset_stats.compute(
                paths, self._palette, lambda i, n: progress.setValue(i))
        finally:
            progress.close()
        if not data["stats"]:
            QMessageBox.warning(self, "Sin objetos", "No se encontró ningún objeto en las imágenes elegidas.\n\n"
                                + self._skipped_text(skipped))
            return
        try:
            self.store.set(data)
        except OSError as exc:
            QMessageBox.critical(self, "No se pudo guardar la estadística", str(exc))
            return
        self._refresh()
        if skipped:
            QMessageBox.information(self, "Imágenes omitidas", self._skipped_text(skipped))

    @staticmethod
    def _skipped_text(skipped):
        lines = [f"{os.path.basename(p)}: {why}" for p, why in skipped[:15]]
        if len(skipped) > 15:
            lines.append(f"... y {len(skipped) - 15} más")
        return "\n".join(lines) or "Ninguna imagen omitida."
