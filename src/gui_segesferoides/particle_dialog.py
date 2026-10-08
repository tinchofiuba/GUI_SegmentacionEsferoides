"""Popup no modal con las medidas de la partícula seleccionada y acciones
sobre ella. Solo UI: la ventana principal le da los datos y escucha sus
señales; el diálogo no toca la máscara.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


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
        self.label_area = QLabel()
        self.label_perimeter = QLabel()
        self.label_diameter = QLabel()
        self.label_circularity = QLabel()
        form.addRow("Área:", self.label_area)
        form.addRow("Perímetro:", self.label_perimeter)
        form.addRow("Diámetro equivalente:", self.label_diameter)
        form.addRow("Circularidad:", self.label_circularity)

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

    def show_particle(self, class_id, stats):
        self.combo_class.setCurrentIndex(self.combo_class.findData(class_id))
        self.label_area.setText(f"{stats['area']} px²")
        self.label_perimeter.setText(f"{stats['perimeter']:.1f} px")
        self.label_diameter.setText(f"{stats['equivalent_diameter']:.1f} px")
        self.label_circularity.setText(f"{stats['circularity']:.2f}")

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
