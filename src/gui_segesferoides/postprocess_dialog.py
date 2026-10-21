"""Editor de las reglas de post-procesado que expone el backend (ver
docs/protocol.md, `get_postprocess`). Solo UI: edita una copia del dict de
reglas y la devuelve con `config()`; la GUI no las evalúa.

Una regla: clase a la que aplica + condiciones (todas deben cumplirse) +
acción. Se evalúan en orden: reclasificar sigue con las reglas siguientes;
descartar y conservar terminan.
"""

import copy
import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)

METRICS = [("area", "Área (px²)"), ("diameter", "Diámetro equiv. (px)"),
           ("circularity", "Circularidad"), ("perimeter", "Perímetro (px)")]
OPS = [("lt", "<"), ("le", "≤"), ("gt", ">"), ("ge", "≥")]
ACTIONS = [("delete", "Descartar"), ("reclass", "Pasar a la clase"), ("keep", "Conservar (termina)")]
REFERENCE = re.compile(r"c\d+\.(p5|mean|p95)")


def _combo(options, current):
    box = QComboBox()
    for data, label in options:
        box.addItem(label, data)
    box.setCurrentIndex(max(0, box.findData(current)))
    return box


class PostprocessDialog(QDialog):
    def __init__(self, config, defaults, classes, parent=None):
        """`config`/`defaults`: {"rules": [...], "fill_holes": [...]}; `classes`: [(id, nombre)]."""
        super().__init__(parent)
        self.setWindowTitle("Reglas de post-procesado")
        self.resize(820, 460)
        self._defaults = copy.deepcopy(defaults)
        self._classes = list(classes)
        self._rules = copy.deepcopy(config["rules"])
        self._row = -1
        self._refs = [f"c{cid}.{q}" for cid, _ in self._classes for q in ("p5", "mean", "p95")]

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("Se evalúan en orden sobre cada objeto predicho. \"Pasar a la clase\" sigue con "
                               "las reglas siguientes (que ya ven la clase nueva); \"Descartar\" y "
                               "\"Conservar\" terminan. Los valores son números en píxeles o una referencia a la "
                               "estadística del entrenamiento (p. ej. c1.p5)."))
        body = QHBoxLayout()
        outer.addLayout(body, stretch=1)

        left = QVBoxLayout()
        body.addLayout(left)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)
        left.addWidget(self.list)
        row = QHBoxLayout()
        left.addLayout(row)
        for text, fn in (("▲", lambda: self._move(-1)), ("▼", lambda: self._move(1)),
                         ("Nueva", self._add), ("Duplicar", self._duplicate), ("Quitar", self._remove)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)

        self.group = QGroupBox("Regla")
        body.addWidget(self.group, stretch=1)
        form = QFormLayout(self.group)
        self.edit_name = QLineEdit()
        form.addRow("Nombre:", self.edit_name)
        self.combo_class = _combo([("any", "Cualquiera"), *[(cid, f"{cid} - {n}") for cid, n in self._classes]],
                                  "any")
        form.addRow("Clase predicha:", self.combo_class)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Métrica", "", "Valor"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 205)
        self.table.setColumnWidth(1, 50)
        form.addRow("Si se cumplen todas:", self.table)
        cond_row = QHBoxLayout()
        add_cond, del_cond = QPushButton("+ condición"), QPushButton("− condición")
        add_cond.clicked.connect(lambda: self._add_condition({"metric": "diameter", "op": "lt", "value": 0}))
        del_cond.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        cond_row.addWidget(add_cond)
        cond_row.addWidget(del_cond)
        cond_row.addStretch()
        form.addRow("", cond_row)
        action_row = QHBoxLayout()
        self.combo_action = _combo(ACTIONS, "delete")
        self.combo_to = _combo([(cid, f"{cid} - {n}") for cid, n in self._classes], None)
        self.combo_action.currentIndexChanged.connect(self._sync_action)
        action_row.addWidget(self.combo_action)
        action_row.addWidget(self.combo_to)
        action_row.addStretch()
        form.addRow("Entonces:", action_row)

        holes = QHBoxLayout()
        outer.addLayout(holes)
        holes.addWidget(QLabel("Al final, rellenar huecos de las clases:"))
        self.check_holes = {}
        for cid, name in self._classes:
            box = QCheckBox(f"{cid} - {name}")
            box.setChecked(cid in config.get("fill_holes", []))
            holes.addWidget(box)
            self.check_holes[cid] = box
        holes.addStretch()
        restore = QPushButton("Restaurar valores de fábrica")
        restore.clicked.connect(self._restore)
        holes.addWidget(restore)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._fill_list()

    # ---------- lista de reglas ----------
    def _fill_list(self, row=0):
        self.list.blockSignals(True)
        self.list.clear()
        for rule in self._rules:
            self._add_item(rule)
        self.list.blockSignals(False)
        self._row = -1
        self.list.setCurrentRow(min(row, len(self._rules) - 1))
        self._select(self.list.currentRow())

    def _add_item(self, rule):
        item = QListWidgetItem(rule.get("name") or "(sin nombre)")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if rule.get("enabled", True) else Qt.Unchecked)
        self.list.addItem(item)

    def _select(self, row):
        self._store_form()
        self._row = row
        self.group.setEnabled(row >= 0)
        if row < 0:
            return
        rule = self._rules[row]
        self.edit_name.setText(rule.get("name", ""))
        self.combo_class.setCurrentIndex(max(0, self.combo_class.findData(rule.get("class", "any"))))
        self.table.setRowCount(0)
        for cond in rule["conditions"]:
            self._add_condition(cond)
        self.combo_action.setCurrentIndex(max(0, self.combo_action.findData(rule["action"])))
        self.combo_to.setCurrentIndex(max(0, self.combo_to.findData(rule.get("to"))))
        self._sync_action()

    def _sync_action(self):
        self.combo_to.setEnabled(self.combo_action.currentData() == "reclass")

    def _add_condition(self, cond):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setCellWidget(r, 0, _combo(METRICS, cond["metric"]))
        self.table.setCellWidget(r, 1, _combo(OPS, cond["op"]))
        value = QComboBox()
        value.setEditable(True)
        value.addItems(self._refs)
        value.setEditText(str(cond["value"]))
        self.table.setCellWidget(r, 2, value)

    def _store_form(self):
        """Pasa el formulario a la regla que se estaba editando."""
        if not 0 <= self._row < len(self._rules):
            return
        rule = self._rules[self._row]
        rule["name"] = self.edit_name.text().strip()
        rule["class"] = self.combo_class.currentData()
        rule["conditions"] = []
        for r in range(self.table.rowCount()):
            text = self.table.cellWidget(r, 2).currentText().strip()
            try:
                value = float(text.replace(",", "."))
            except ValueError:
                value = text   # referencia (se valida al aceptar)
            rule["conditions"].append({"metric": self.table.cellWidget(r, 0).currentData(),
                                       "op": self.table.cellWidget(r, 1).currentData(), "value": value})
        rule["action"] = self.combo_action.currentData()
        if rule["action"] == "reclass":
            rule["to"] = self.combo_to.currentData()
        else:
            rule.pop("to", None)
        self.list.item(self._row).setText(rule["name"] or "(sin nombre)")

    def _commit(self):
        self._store_form()
        self._sync_enabled()

    def _add(self):
        self._commit()
        self._rules.append({"name": "Nueva regla", "enabled": True, "class": "any",
                            "conditions": [{"metric": "diameter", "op": "lt", "value": 0}],
                            "action": "delete"})
        self._fill_list(len(self._rules) - 1)

    def _duplicate(self):
        self._commit()
        if self._row >= 0:
            self._rules.insert(self._row + 1, copy.deepcopy(self._rules[self._row]))
            self._fill_list(self._row + 1)

    def _remove(self):
        self._commit()
        if self._row >= 0:
            del self._rules[self._row]
            self._row = -1
            self._fill_list(self.list.currentRow())

    def _move(self, delta):
        self._commit()
        row, new = self._row, self._row + delta
        if row < 0 or not 0 <= new < len(self._rules):
            return
        self._rules[row], self._rules[new] = self._rules[new], self._rules[row]
        self._fill_list(new)

    def _restore(self):
        if QMessageBox.question(self, "Restaurar", "Se pierden las reglas editadas. ¿Continuar?") != QMessageBox.Yes:
            return
        self._rules = copy.deepcopy(self._defaults["rules"])
        for cid, box in self.check_holes.items():
            box.setChecked(cid in self._defaults.get("fill_holes", []))
        self._fill_list()

    def _sync_enabled(self):
        for i, rule in enumerate(self._rules):
            rule["enabled"] = self.list.item(i).checkState() == Qt.Checked

    # ---------- resultado ----------
    def config(self):
        self._store_form()
        self._sync_enabled()
        return {"rules": copy.deepcopy(self._rules),
                "fill_holes": [cid for cid, box in self.check_holes.items() if box.isChecked()]}

    def _accept(self):
        config = self.config()
        for k, rule in enumerate(config["rules"], 1):
            for cond in rule["conditions"]:
                v = cond["value"]
                if isinstance(v, str) and not REFERENCE.fullmatch(v):
                    QMessageBox.warning(self, "Valor inválido",
                                        f"Regla {k}: '{v}' no es un número ni una referencia (c1.p5, c2.mean...).")
                    return
        self.accept()
