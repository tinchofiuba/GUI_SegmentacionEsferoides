import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPointF, QSettings, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QFileDialog, QGroupBox, QMenu, QMessageBox, QShortcut

from gui_segesferoides.config import BackendSpec
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
)
from gui_segesferoides.main_window import MainWindow

FAKE = str(Path(__file__).parent / "fake_backend.py")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_idle(qapp, window, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if not window.busy and (window.worker is None or not window.worker.isRunning()):
            break
        time.sleep(0.01)
    qapp.processEvents()
    assert not window.busy, "el backend no respondió a tiempo"


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path / "qs"))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    errors.clear()
    w = MainWindow([BackendSpec("fake", [sys.executable, FAKE])],
                   calibration_path=str(tmp_path / "calibration.json"),
                   postprocess_path=str(tmp_path / "postprocess.json"))
    wait_idle(qapp, w)
    yield w
    w.dirty = False
    w.close()


errors = []


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "cel_1.png"
    cv2.imwrite(str(p), np.full((300, 400, 3), 90, np.uint8))
    return str(p)


def test_window_takes_classes_and_magnifications_from_backend(window):
    assert window.palette == {1: (0, 255, 0), 2: (255, 255, 0), 5: (0, 255, 255)}
    assert [window.combo_class.itemData(i) for i in range(window.combo_class.count())] == [1, 2, 5]
    assert [window.combo_magnification.itemText(i) for i in range(window.combo_magnification.count())] == ["4x", "10x"]
    assert window.combo_model.count() == 2 and not window.model_loaded
    assert not window.predict_button.isEnabled()  # sin modelo cargado


def test_predict_needs_model_then_fills_mask_and_marks_origin(qapp, window, image_path):
    window.open_image(image_path)
    wait_idle(qapp, window)
    window._load_model("a")
    wait_idle(qapp, window)
    assert window.model_loaded and window.predict_button.isEnabled()

    window._on_predict_clicked()
    wait_idle(qapp, window)
    assert window.mask.shape == (300, 400) and set(np.unique(window.mask)) == {0, 2}
    assert window.origin == "modelo" and window.dirty

    window.canvas.tool = "reclass"  # el combo apunta a la clase 1
    window._on_object_clicked(200, 150)
    assert set(np.unique(window.mask)) == {0, 1}
    window.canvas.tool = "delete"
    window._on_object_clicked(200, 150)
    assert not window.mask.any()
    window._undo()
    window._undo()
    assert set(np.unique(window.mask)) == {0, 2}  # vuelve a lo que predijo el backend


def test_zone_prediction_only_touches_zone(qapp, window, image_path):
    window.open_image(image_path)
    window._load_model("a")
    wait_idle(qapp, window)
    window.radio_zone.setChecked(True)
    window.canvas.selection = (10.0, 20.0, 80.0, 60.0)
    window._on_predict_clicked()
    wait_idle(qapp, window)
    ys, xs = np.nonzero(window.mask)
    assert xs.min() >= 10 and xs.max() < 90 and ys.min() >= 20 and ys.max() < 80


def test_save_writes_id_mask(qapp, window, image_path, tmp_path):
    window.open_image(image_path)
    window._load_model("b")
    wait_idle(qapp, window)
    window._on_predict_clicked()
    wait_idle(qapp, window)
    out = str(tmp_path / "cel_1_Taggeada.png")
    window.save_to(out)
    saved = cv2.imread(out, cv2.IMREAD_UNCHANGED)
    assert saved.ndim == 2 and (saved == window.mask).all()
    assert not (tmp_path / "cel_1_Taggeada.origin.json").exists()
    assert not window.dirty


def test_failed_model_load_keeps_window_usable(qapp, window, image_path):
    window.open_image(image_path)
    window._load_model("a")
    wait_idle(qapp, window)
    window._load_model("malo")
    wait_idle(qapp, window)
    assert errors and "no existe" in errors[-1]
    assert not window.busy


def _window_with_two_particles(window, image_path):
    window.open_image(image_path)
    window.mask[20:60, 20:60] = 2
    window.mask[30:40, 30:40] = 0       # hueco
    window.mask[100:140, 100:140] = 1
    window.canvas.tool = "analyze"
    window.canvas.mask = window.mask


def test_analyze_selects_toggles_and_switches(window, image_path):
    _window_with_two_particles(window, image_path)
    window._on_object_clicked(25, 25)
    assert window.particle is not None and window.particle_dialog.isVisible()
    assert window.particle_dialog.label_area.text() == "1500 px²"
    assert window.particle_dialog.combo_class.currentData() == 2 and window.canvas.highlight
    window._on_object_clicked(120, 120)           # otra partícula: actualiza el popup
    assert window.particle_dialog.combo_class.currentData() == 1
    assert window.particle_dialog.label_area.text() == "1600 px²"
    window._on_object_clicked(120, 120)           # mismo objeto: deselecciona
    assert window.particle is None and not window.canvas.highlight
    assert not window.particle_dialog.isVisible()


def test_analyze_actions_change_only_selected_particle(window, image_path, monkeypatch):
    _window_with_two_particles(window, image_path)
    window._on_object_clicked(25, 25)
    window.particle_dialog.combo_class.setCurrentIndex(window.particle_dialog.combo_class.findData(5))
    window.particle_dialog.class_changed.emit(5)
    assert set(np.unique(window.mask[20:60, 20:60])) == {0, 5} and window.dirty
    assert np.all(window.mask[100:140, 100:140] == 1)

    window.particle_dialog.fill_button.click()
    assert np.all(window.mask[20:60, 20:60] == 5)
    assert window.particle_dialog.label_area.text() == "1600 px²"

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    window.particle_dialog.delete_button.click()
    assert window.particle is not None and window.mask[25, 25] == 5   # canceló
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    window.particle_dialog.delete_button.click()
    assert not window.mask[20:60, 20:60].any() and np.all(window.mask[100:140, 100:140] == 1)
    assert window.particle is None and not window.particle_dialog.isVisible()
    window._undo()
    assert np.all(window.mask[20:60, 20:60] == 5)


def test_closing_dialog_deselects(window, image_path):
    _window_with_two_particles(window, image_path)
    window._on_object_clicked(25, 25)
    window.particle_dialog.close_button.click()
    assert window.particle is None and not window.canvas.highlight


def test_uninstalled_backend_offers_install_then_connects(qapp, tmp_path, monkeypatch):
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path / "qs2"))
    repo = tmp_path / "repo"
    fake = str(Path(FAKE))
    # "instalar" = crear el ejecutable que el comando espera (un wrapper al backend falso)
    make = ("import os, sys; os.makedirs('bin'); open('bin/run', 'w').write("
            f"'#!/bin/sh\\nexec {sys.executable} {fake}\\n'); os.chmod('bin/run', 0o755)")
    spec = BackendSpec("inst", ["bin/run"], cwd=str(repo), install=[
        [sys.executable, "-c", f"import os; os.makedirs({str(repo)!r})"],
        [sys.executable, "-c", make]])
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(a[2]) or QMessageBox.Yes))
    w = MainWindow([spec], calibration_path=str(tmp_path / "calibration.json"),
                   postprocess_path=str(tmp_path / "postprocess.json"))
    qapp.processEvents()
    assert w.backend is None and w.uninstalled is spec
    assert not w.install_button.isHidden() and w.install_button.isEnabled()
    w._on_install_clicked()
    assert "bin/run" in asked[0] or "-c" in asked[0]
    wait_idle(qapp, w)
    assert w.uninstalled is None and w.backend is not None and w.install_button.isHidden()
    assert w.combo_model.count() == 2
    w.dirty = False
    w.close()


def test_uninstalled_without_install_steps_disables_button(qapp, tmp_path):
    spec = BackendSpec("sin-pasos", ["no/existe"], cwd=str(tmp_path / "nada"))
    w = MainWindow([spec], calibration_path=str(tmp_path / "calibration.json"),
                   postprocess_path=str(tmp_path / "postprocess.json"))
    assert w.uninstalled is spec and not w.install_button.isEnabled()
    w.close()


def test_export_results_writes_csv_and_id_png(qapp, window, image_path, tmp_path):
    import csv

    window.open_image(image_path)
    window._load_model("b")
    wait_idle(qapp, window)
    window.mask[:] = 0
    window.mask[10:20, 10:20] = 1
    window.mask[30:40, 30:50] = 5
    window.mask[60:70, 60:70] = 5
    assert window.export_button.isEnabled()
    out = str(tmp_path / "res.csv")
    window.export_results(out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [(r["class_id"], r["class_name"], r["area_px"]) for r in rows] == [
        ("1", "chicas", "100"), ("5", "raras", "200"), ("5", "raras", "100")]
    assert rows[0]["backend"] == "fake" and rows[0]["circularity"]
    ids = cv2.imread(str(tmp_path / "cel_1-Mask.png"), cv2.IMREAD_UNCHANGED)
    assert ids.ndim == 2 and set(np.unique(ids)) == {0, 1, 5}


def test_magnification_taken_from_file_name(qapp, window, image_path, tmp_path):
    import shutil

    def opened(name):
        p = tmp_path / name
        shutil.copy(image_path, p)
        window.combo_magnification.setCurrentIndex(0)
        window.open_image(str(p))
        return window.combo_magnification.currentText()

    assert opened("prueba_10X.png") == "10x"
    assert opened("prueba-10x.png") == "10x"
    assert opened("prueba_4x_b.png") == "4x"
    assert opened("prueba_8x.png") == "4x"       # no está en el combo: no toca nada
    assert opened("prueba_210x.png") == "4x"     # 210x no es 10x


def test_mask_filename_matches_what_training_expects(qapp, window, image_path, tmp_path):
    import shutil

    for name, expected in [("nombreimg_10x.png", "nombreimg-Mask.png"),
                           ("nombreimg-4X.png", "nombreimg-Mask.png"),
                           ("20 - 4d 3T3.tiff", "20 - 4d 3T3-Mask.png"),
                           ("a_10x_b.png", "a_10x_b-Mask.png")]:
        p = tmp_path / name
        shutil.copy(image_path, p)
        window.image_path = str(p)
        assert window._mask_filename() == expected


def test_analyze_shows_reference_bars_when_backend_has_stats(window, image_path):
    _window_with_two_particles(window, image_path)
    window.combo_magnification.setCurrentText("4x")
    window._on_object_clicked(25, 25)              # clase 2, área 1500: dentro de [1000, 2000]
    dialog = window.particle_dialog
    assert dialog.bars["area"].isVisibleTo(dialog) and dialog.bars["area"].inside
    assert dialog.bars["circularity"].isVisibleTo(dialog)
    assert not dialog.bars["equivalent_diameter"].isVisibleTo(dialog)   # el backend no la trae
    assert dialog._counts == [("chicas", 30), ("grandes", 10)]
    window._on_object_clicked(120, 120)            # clase 1: solo trae n, sin métricas
    assert not any(b.isVisibleTo(dialog) for b in dialog.bars.values())
    window.combo_magnification.setCurrentText("10x")   # sin estadística a 10x: nada que mostrar
    window._on_object_clicked(25, 25)
    assert not any(b.isVisibleTo(dialog) for b in dialog.bars.values())


def test_calibration_changes_displayed_units_and_persists(window, image_path, tmp_path):
    _window_with_two_particles(window, image_path)
    window.combo_magnification.setCurrentText("4x")
    window._on_object_clicked(25, 25)
    assert window.particle_dialog.label_area.text() == "1500 px²"
    window.calibration.set("4x", 2, "µm")          # 1 px = 2 µm
    window.calibration.display_unit = "mm"
    window.calibration.save()
    window._refresh_particle()
    assert window.particle_dialog.label_area.text() == "0.006000 mm²"
    window.combo_magnification.setCurrentText("10x")   # sin calibrar: píxeles
    window._refresh_particle()
    assert window.particle_dialog.label_area.text() == "1500 px²"
    from gui_segesferoides.calibration import Calibration
    again = Calibration(str(tmp_path / "calibration.json"))
    assert again.entries == {"4x": (2.0, "µm")} and again.display_unit == "mm"
    assert again.um_per_px("4x") == 2.0


def test_calibration_dialog_edits_calibration(qapp, tmp_path):
    from gui_segesferoides.calibration import Calibration
    from gui_segesferoides.calibration_dialog import CalibrationDialog
    cal = Calibration(str(tmp_path / "c.json"))
    d = CalibrationDialog(cal)
    spin, unit = d.rows["10x"]
    spin.setValue(1); unit.setCurrentText("mm")
    d.combo_display.setCurrentText("µm")
    d._accept()
    assert cal.um_per_px("10x") == 1000 and cal.um_per_px("4x") is None
    assert cal.scale("10x").factor == 1000 and cal.scale("4x").unit == "px"


def test_export_attaches_reference_stats_only_when_enabled(qapp, window, image_path, tmp_path):
    import csv

    window.open_image(image_path)
    window.combo_magnification.setCurrentText("4x")
    window.mask[:] = 0
    window.mask[10:50, 10:50] = 2          # área 1600, dentro de [1000, 2000]
    window.mask[100:110, 100:110] = 5      # el modelo no tiene estadística de esta clase
    off, on = str(tmp_path / "off.csv"), str(tmp_path / "on.csv")
    window.export_results(off)
    assert "ref_n" not in csv.DictReader(open(off, encoding="utf-8")).fieldnames
    window.calibration.csv_stats = True
    window.export_results(on)
    rows = list(csv.DictReader(open(on, encoding="utf-8")))
    assert (rows[0]["ref_n"], rows[0]["ref_area_p5"], rows[0]["ref_area_p95"],
            rows[0]["ref_area_in_range"]) == ("10", "1000", "2000", "yes")
    assert rows[0]["ref_diameter_p5"] == ""          # el backend no la trae
    # el perímetro de un cuadrado de 40 px es 156: dentro de [120, 200]
    assert (rows[0]["perimeter_px"], rows[0]["ref_perimeter_p5"], rows[0]["ref_perimeter_mean"],
            rows[0]["ref_perimeter_p95"], rows[0]["ref_perimeter_in_range"]) == (
        "156.00", "120", "160", "200", "yes")
    assert rows[1]["ref_perimeter_p5"] == "" and rows[1]["ref_perimeter_in_range"] == ""
    fields = csv.DictReader(open(on, encoding="utf-8")).fieldnames
    assert [f for f in fields if f.startswith("ref_perimeter")] == [
        "ref_perimeter_p5", "ref_perimeter_mean", "ref_perimeter_p95", "ref_perimeter_in_range"]
    assert rows[1]["ref_n"] == "" and rows[1]["ref_area_in_range"] == ""


def test_csv_stats_option_persists(qapp, tmp_path):
    from gui_segesferoides.calibration import Calibration
    from gui_segesferoides.calibration_dialog import CalibrationDialog
    cal = Calibration(str(tmp_path / "c.json"))
    d = CalibrationDialog(cal)
    d.check_csv_stats.setChecked(True)
    d._accept()
    cal.save()
    assert Calibration(str(tmp_path / "c.json")).csv_stats is True


def _predict_and_spy(qapp, window, image_path):
    sent = []
    original = window.backend._request

    def spy(cmd, **params):
        if cmd == "predict":
            sent.append(params)
        return original(cmd, **params)

    window.backend._request = spy
    window.open_image(image_path)
    window._load_model("a")
    wait_idle(qapp, window)
    window.radio_full.setChecked(True)
    window._on_predict_clicked()
    wait_idle(qapp, window)
    return sent


def test_predict_shows_discarded_contours_only_when_checked(qapp, window, image_path):
    sent = _predict_and_spy(qapp, window, image_path)
    assert "discarded_out" in sent[0] and "postprocess" not in sent[0]   # sin reglas del usuario
    assert window.discarded is not None and window.discarded[:6, :6].all()
    assert window.canvas.discarded == []                                  # checkbox apagado
    window.check_discarded.setChecked(True)
    assert len(window.canvas.discarded) == 1
    window.check_discarded.setChecked(False)
    assert window.canvas.discarded == []


def test_hovering_a_discarded_object_says_why(qapp, window, image_path):
    _predict_and_spy(qapp, window, image_path)
    text = window._discarded_text_at(2, 2)               # adentro del cuadrado descartado
    assert "regla de prueba" in text and "Diámetro equivalente = 6.8 < 10.0 px" in text
    assert "[c1.p5]" in text and "Clase 1" in text
    assert f"estadística de {window.combo_magnification.currentText()}" in text
    assert window._discarded_text_at(30, 30) is None     # fuera de lo descartado


def test_discarded_reasons_follow_a_zone_prediction(qapp, window, image_path):
    window.open_image(image_path)
    window._load_model("a")
    wait_idle(qapp, window)
    window.radio_zone.setChecked(True)
    window.canvas.selection = (10.0, 20.0, 80.0, 60.0)
    window._on_predict_clicked()
    wait_idle(qapp, window)
    assert [(r["x"], r["y"]) for r in window.discarded_reasons] == [(10, 20)]  # coords de la imagen
    assert "regla de prueba" in window._discarded_text_at(12, 22)
    assert window._discarded_text_at(2, 2) is None


def test_user_rules_are_saved_and_sent_with_predict(qapp, window, image_path, tmp_path):
    from gui_segesferoides.postprocess_store import PostprocessStore
    rules = {"rules": [{"name": "mía", "enabled": True, "class": 2, "action": "keep",
                        "conditions": [{"metric": "area", "op": "gt", "value": 10}]}], "fill_holes": []}
    window.postprocess_store.set("fake", rules)
    assert PostprocessStore(str(tmp_path / "postprocess.json")).get("fake") == rules
    sent = _predict_and_spy(qapp, window, image_path)
    assert sent[0]["postprocess"] == rules


def test_postprocess_button_follows_backend_flag(qapp, window):
    assert window.backend.info.supports_postprocess
    window._load_model("a")
    wait_idle(qapp, window)
    assert window.postprocess_button.isEnabled()
    assert window.backend.get_postprocess()["defaults"]["rules"][0]["name"] == "chicos"


def test_postprocess_dialog_edits_rules(qapp):
    from gui_segesferoides.postprocess_dialog import PostprocessDialog
    base = {"rules": [{"name": "a", "enabled": True, "class": "any", "action": "delete",
                       "conditions": [{"metric": "diameter", "op": "lt", "value": "c1.p5"}]},
                      {"name": "b", "enabled": False, "class": 1, "action": "reclass", "to": 2,
                       "conditions": [{"metric": "area", "op": "gt", "value": 500}]}],
            "fill_holes": [2]}
    d = PostprocessDialog(base, base, [(1, "chicas"), (2, "grandes")])
    assert d.config() == base                       # abrir y aceptar no cambia nada
    d.list.setCurrentRow(1)
    d.table.cellWidget(0, 2).setEditText("750")
    d.combo_to.setCurrentIndex(d.combo_to.findData(1))
    d.list.item(1).setCheckState(Qt.Checked)
    d._move(-1)                                     # b pasa a ser la primera
    d.check_holes[2].setChecked(False)
    cfg = d.config()
    assert [r["name"] for r in cfg["rules"]] == ["b", "a"]
    assert cfg["rules"][0]["conditions"][0]["value"] == 750.0 and cfg["rules"][0]["to"] == 1
    assert cfg["rules"][0]["enabled"] is True and cfg["fill_holes"] == []


# ---------- Panel lateral: grupos, herramientas y modo de predicción ----------

def _group_title(widget):
    """Título del QGroupBox que contiene a `widget`, o None si está fuera de los grupos."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent.title()
        parent = parent.parentWidget()
    return None


def test_side_panel_has_one_group_per_function(window):
    titles = sorted(g.title() for g in window.findChildren(QGroupBox))
    assert titles == ["Análisis", "Edición", "Predicción", "Vista"]


def test_each_control_lives_in_its_function_group(window):
    expected = {
        "Predicción": [window.radio_full, window.radio_zone, window.combo_magnification,
                       window.predict_button, window.postprocess_button, window.check_discarded],
        "Edición": [window.tool_buttons[t] for t in (TOOL_BRUSH, TOOL_ERASER, TOOL_DELETE, TOOL_RECLASS)]
                   + [window.combo_class, window.slider_brush, window.slider_eraser],
        "Análisis": [window.tool_buttons[TOOL_ANALYZE], window.calibrate_button],
        "Vista": [window.check_show_mask, window.slider_alpha],
    }
    for title, widgets in expected.items():
        for w in widgets:
            assert _group_title(w) == title, f"{w} debería estar en «{title}»"
    # archivo, historial, modelo y zoom no pertenecen a ningún grupo
    for w in (window.accept_button, window.export_button, window.undo_button, window.redo_button,
              window.combo_backend, window.combo_model, window.label_zoom):
        assert _group_title(w) is None


def test_window_starts_in_full_image_mode_with_the_brush(window):
    assert window.radio_full.isChecked() and not window.radio_zone.isChecked()
    assert window.canvas.tool == TOOL_BRUSH and window.tool_buttons[TOOL_BRUSH].isChecked()
    assert window.zone_hint.isHidden()


@pytest.mark.parametrize("tool, class_row, brush_row, eraser_row", [
    (TOOL_BRUSH, True, True, False),
    (TOOL_ERASER, False, False, True),
    (TOOL_DELETE, False, False, False),
    (TOOL_RECLASS, True, False, False),
    (TOOL_ANALYZE, False, False, False),
])
def test_each_tool_shows_only_its_own_options(window, tool, class_row, brush_row, eraser_row):
    window.tool_buttons[tool].setChecked(True)
    assert window.canvas.tool == tool
    assert (not window.row_class.isHidden()) == class_row
    assert (not window.row_brush.isHidden()) == brush_row
    assert (not window.row_eraser.isHidden()) == eraser_row


def test_tools_are_exclusive_across_groups(window):
    """Pincel (Edición) y Analizar (Análisis) están en grupos distintos pero comparten selección."""
    window.tool_buttons[TOOL_ANALYZE].setChecked(True)
    assert window.canvas.tool == TOOL_ANALYZE
    assert not window.tool_buttons[TOOL_BRUSH].isChecked()
    window.tool_buttons[TOOL_ERASER].setChecked(True)
    assert window.canvas.tool == TOOL_ERASER
    assert not window.tool_buttons[TOOL_ANALYZE].isChecked()


def test_zone_mode_activates_the_zone_tool_and_full_image_releases_it(window):
    window.radio_zone.setChecked(True)
    assert window.canvas.tool == TOOL_ZONE and not window.zone_hint.isHidden()
    for hidden in (window.row_class, window.row_brush, window.row_eraser):
        assert hidden.isHidden()            # la herramienta Zona no tiene opciones de edición
    window.radio_full.setChecked(True)
    assert window.canvas.tool == TOOL_BRUSH and window.zone_hint.isHidden()


def test_zone_tool_has_no_button_of_its_own(window):
    assert window.tool_buttons[TOOL_ZONE].parentWidget() is not None
    assert window.tool_buttons[TOOL_ZONE].isHidden()      # oculto aunque la ventana se muestre
    window.show()
    assert not window.tool_buttons[TOOL_ZONE].isVisible()
    window.hide()
    assert _group_title(window.tool_buttons[TOOL_ZONE]) is None


def test_z_shortcut_selects_zone_mode(window):
    z = [sc for sc in window.findChildren(QShortcut) if sc.key().toString() == "Z"]
    assert len(z) == 1
    z[0].activated.emit()
    assert window.radio_zone.isChecked() and window.canvas.tool == TOOL_ZONE


def test_leaving_the_zone_tool_clears_the_drawn_selection(window):
    window.radio_zone.setChecked(True)
    window.canvas.selection = (10.0, 20.0, 80.0, 60.0)
    window.tool_buttons[TOOL_BRUSH].setChecked(True)
    assert window.canvas.selection is None


def test_predict_in_zone_mode_without_a_drawn_zone_asks_for_it(qapp, window, image_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2])))
    window.open_image(image_path)
    window._load_model("a")
    wait_idle(qapp, window)
    window.radio_zone.setChecked(True)
    window._on_predict_clicked()
    assert shown and "rectángulo" in shown[0]
    assert window.canvas.tool == TOOL_ZONE and window.radio_zone.isChecked()
    assert not window.busy and not np.any(window.mask)    # no llegó a predecir


# ---------- Tooltip de descartados en el canvas ----------

def _move_mouse(canvas, x=50, y=50):
    canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton,
                                      Qt.NoButton, Qt.NoModifier))


def test_canvas_asks_for_hover_text_only_while_discards_are_shown(qapp, window, image_path):
    calls = []
    window.canvas.hover_text = lambda ix, iy: calls.append((ix, iy)) or None
    _predict_and_spy(qapp, window, image_path)
    _move_mouse(window.canvas)
    assert calls == []                                    # checkbox apagado: sin contornos rojos
    window.check_discarded.setChecked(True)
    _move_mouse(window.canvas)
    assert len(calls) == 1 and all(isinstance(v, int) for v in calls[0])
    window.check_discarded.setChecked(False)
    _move_mouse(window.canvas)
    assert len(calls) == 1


def test_hover_text_is_connected_to_the_window_lookup(window):
    assert window.canvas.hover_text == window._discarded_text_at


def test_discarded_text_without_reason_from_the_backend_says_so(qapp, window, image_path):
    _predict_and_spy(qapp, window, image_path)
    window.discarded_texts.clear()                        # backend que no informa motivos
    assert "no informó el motivo" in window._discarded_text_at(2, 2)


def test_discarded_text_mentions_a_reclassification_before_the_delete(qapp, window, image_path):
    _predict_and_spy(qapp, window, image_path)
    reason = {**window.discarded_reasons[0], "class": 1, "as_class": 2}
    text = window._discarded_text(reason)
    assert "reclasificado antes como 2" in text and "Clase 1" in text


def test_opening_an_image_shows_its_name_in_a_label_on_the_canvas(qapp, window, image_path):
    assert window.canvas.name_label.isHidden()
    window.open_image(image_path)
    assert not window.canvas.name_label.isHidden()
    assert window.canvas.name_label.text() == "cel_1.png"          # solo el nombre, no la ruta
    assert not np.any(window.mask)                                 # y no toca la máscara


def test_particle_popup_is_wide_and_its_range_bars_stretch(qapp, window):
    from PyQt5.QtWidgets import QSizePolicy

    from gui_segesferoides.particle_dialog import ParticleDialog, RangeBar
    d = ParticleDialog([(1, "células sueltas")])
    assert d.minimumWidth() >= 600
    bar = d.bars["area"]
    assert bar.minimumWidth() >= 240
    assert bar.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    narrow = RangeBar()
    narrow.set_data(15, 10, 20, 15, "10 – 20")
    lo, hi = narrow._scale()
    assert (20 - 10) / (hi - lo) >= 0.5          # el intervalo ocupa más de la mitad de la barra


# ---------- CSV en la unidad de calibración ----------

def _export_csv(qapp, window, image_path, tmp_path, name="u.csv"):
    import csv
    window.open_image(image_path)
    window.combo_magnification.setCurrentText("4x")
    window.mask[:] = 0
    window.mask[10:50, 10:50] = 2          # cuadrado de 40 px: área 1600, perímetro 156
    out = str(tmp_path / name)
    window.export_results(out)
    reader = csv.DictReader(open(out, encoding="utf-8"))
    return reader.fieldnames, list(reader)


def test_csv_without_calibration_is_in_pixels_and_says_so(qapp, window, image_path, tmp_path):
    fields, rows = _export_csv(qapp, window, image_path, tmp_path)
    assert {"area_px", "perimeter_px", "equivalent_diameter_px"} <= set(fields)
    r = rows[0]
    assert (r["unit"], r["magnification"], r["area_px"], r["perimeter_px"]) == ("px", "4x", "1600", "156.00")
    assert r["um_per_px_4x"] == "" and r["um_per_px_10x"] == ""      # sin calibrar


def test_csv_measures_use_the_display_unit_chosen_in_calibration(qapp, window, image_path, tmp_path):
    window.calibration.set("4x", 2.0, "µm")                          # 1 px = 2 µm
    window.calibration.display_unit = "mm"
    fields, rows = _export_csv(qapp, window, image_path, tmp_path)
    assert {"area_mm2", "perimeter_mm", "equivalent_diameter_mm"} <= set(fields)
    assert not {"area_px", "perimeter_px", "equivalent_diameter_px"} & set(fields)
    r = rows[0]
    assert r["unit"] == "mm"
    assert float(r["area_mm2"]) == pytest.approx(1600 * 0.002 ** 2)   # px² -> mm²
    assert float(r["perimeter_mm"]) == pytest.approx(156 * 0.002)
    assert float(r["equivalent_diameter_mm"]) == pytest.approx(2 * (1600 / np.pi) ** 0.5 * 0.002, rel=1e-4)
    assert "e" not in r["area_mm2"].lower()                          # nunca notación científica
    assert (r["bbox_x"], r["bbox_w"]) == ("10", "40")                # posiciones: siguen en píxeles


def test_csv_always_carries_both_calibrations_in_um_per_px(qapp, window, image_path, tmp_path):
    window.calibration.set("4x", 2.0, "µm")
    window.calibration.set("10x", 0.5, "mm")                         # 1 px = 0.5 mm = 500 µm
    window.calibration.display_unit = "nm"                           # la unidad de las medidas no cambia las columnas
    _fields, rows = _export_csv(qapp, window, image_path, tmp_path)
    assert (rows[0]["um_per_px_4x"], rows[0]["um_per_px_10x"]) == ("2", "500")
    assert rows[0]["unit"] == "nm" and float(rows[0]["perimeter_nm"]) == pytest.approx(156 * 2000)


def test_csv_is_in_pixels_when_that_magnification_has_no_calibration(qapp, window, image_path, tmp_path):
    window.calibration.set("10x", 1.0, "µm")                         # solo 10x calibrado; se exporta 4x
    window.calibration.display_unit = "µm"
    fields, rows = _export_csv(qapp, window, image_path, tmp_path)
    assert "area_px" in fields and rows[0]["unit"] == "px"
    assert rows[0]["um_per_px_10x"] == "1" and rows[0]["um_per_px_4x"] == ""


def test_csv_reference_columns_follow_the_same_unit(qapp, window, image_path, tmp_path):
    import csv
    window.calibration.set("4x", 2.0, "µm")
    window.calibration.display_unit = "µm"
    window.calibration.csv_stats = True
    window.open_image(image_path)
    window.combo_magnification.setCurrentText("4x")
    window.mask[:] = 0
    window.mask[10:50, 10:50] = 2
    out = str(tmp_path / "ref.csv")
    window.export_results(out)
    r = list(csv.DictReader(open(out, encoding="utf-8")))[0]
    assert float(r["ref_area_p5"]) == pytest.approx(1000 * 4)         # px² -> µm² (x2²)
    assert float(r["ref_perimeter_p95"]) == pytest.approx(200 * 2)    # px -> µm
    assert r["ref_area_in_range"] == "yes" and r["ref_perimeter_in_range"] == "yes"


def test_scale_bar_follows_magnification_and_calibration(qapp, window, image_path):
    window.calibration.set("4x", 2.0, "µm")
    window.calibration.display_unit = "µm"
    window.open_image(image_path)
    window.combo_magnification.setCurrentText("10x")      # sin calibrar: en píxeles
    assert window.canvas.scale.unit == "px"
    window.combo_magnification.setCurrentText("4x")
    assert window.canvas.scale.unit == "µm"
    n, text = window.canvas.scale_bar()
    assert text == f"{n} px = {n * 2:,} µm"
    window.calibration.display_unit = "mm"
    window._update_scale()                                # lo que hace el diálogo de calibración al aceptar
    assert window.canvas.scale.unit == "mm"
    assert window.canvas.scale_bar()[1].endswith("mm")


# ---------- Una sola cosa activa a la vez: zona o herramienta ----------

EDIT_AND_ANALYSIS = [TOOL_BRUSH, TOOL_ERASER, TOOL_DELETE, TOOL_RECLASS, TOOL_ANALYZE]


def _checked_tools(window):
    return [t for t, b in window.tool_buttons.items() if b.isChecked()]


def test_zone_mode_leaves_every_edit_and_analysis_tool_unchecked(window):
    for tool in EDIT_AND_ANALYSIS:
        window.tool_buttons[tool].setChecked(True)
        window.radio_zone.setChecked(True)
        assert _checked_tools(window) == [TOOL_ZONE]               # ninguna de edición ni análisis
        assert window.canvas.tool == TOOL_ZONE and window.radio_zone.isChecked()


@pytest.mark.parametrize("tool", EDIT_AND_ANALYSIS)
def test_choosing_a_tool_releases_the_zone_mode(window, tool):
    window.radio_zone.setChecked(True)
    window.tool_buttons[tool].setChecked(True)
    assert window.canvas.tool == tool
    assert not window.radio_zone.isChecked() and window.radio_full.isChecked()
    assert window.zone_hint.isHidden()
    assert _checked_tools(window) == [tool]


def test_the_zone_and_a_tool_are_never_checked_together(window):
    """El caso de la captura: «Zona seleccionada» junto con «Analizar objeto»."""
    window.radio_zone.setChecked(True)
    window.tool_buttons[TOOL_ANALYZE].setChecked(True)
    assert not (window.radio_zone.isChecked() and window.tool_buttons[TOOL_ANALYZE].isChecked())
    window.radio_zone.setChecked(True)                       # y para volver a la zona hay que elegirla
    assert not window.tool_buttons[TOOL_ANALYZE].isChecked() and window.canvas.tool == TOOL_ZONE


def test_cannot_paint_while_the_zone_is_selected(qapp, window, image_path):
    window.open_image(image_path)
    window.radio_zone.setChecked(True)
    canvas = window.canvas
    canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(50, 50), Qt.LeftButton,
                                       Qt.LeftButton, Qt.NoModifier))
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(50, 50), Qt.LeftButton,
                                         Qt.LeftButton, Qt.NoModifier))
    assert not np.any(window.mask)                           # no pintó nada: arrastrar dibuja una zona


def test_painting_works_again_after_choosing_the_brush(qapp, window, image_path):
    window.open_image(image_path)
    window.radio_zone.setChecked(True)
    window.tool_buttons[TOOL_BRUSH].setChecked(True)
    window.combo_class.setCurrentIndex(0)
    canvas = window.canvas
    canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(50, 50), Qt.LeftButton,
                                       Qt.LeftButton, Qt.NoModifier))
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(50, 50), Qt.LeftButton,
                                         Qt.LeftButton, Qt.NoModifier))
    assert np.any(window.mask)


def test_predicting_after_leaving_the_zone_mode_covers_the_whole_image(qapp, window, image_path,
                                                                       monkeypatch):
    # si la zona siguiera activa, predecir sin rectángulo abriría este aviso: que no cuelgue el test
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    sent = _predict_and_spy_after(qapp, window, image_path)
    assert sent[0]["bbox"] is None


def _predict_and_spy_after(qapp, window, image_path):
    """Modo zona -> elige el pincel -> predice: lo que se manda es la imagen completa."""
    sent = []
    original = window.backend._request

    def spy(cmd, **params):
        if cmd == "predict":
            sent.append(params)
        return original(cmd, **params)

    window.backend._request = spy
    window.open_image(image_path)
    window._load_model("a")
    wait_idle(qapp, window)
    window.radio_zone.setChecked(True)
    window.canvas.selection = (10.0, 20.0, 80.0, 60.0)
    window.tool_buttons[TOOL_BRUSH].setChecked(True)         # suelta la zona (y su rectángulo)
    window._on_predict_clicked()
    wait_idle(qapp, window)
    return sent


# ---------- Resaltado del grupo activo y nombre de la herramienta de análisis ----------

from gui_segesferoides.particle_dialog import SELECTED  # noqa: E402


def _framed_groups(window):
    """Títulos de los grupos del panel enmarcados con el color de selección."""
    return [title for title, g in window.group_boxes.items() if SELECTED.name() in g.styleSheet()]


@pytest.mark.parametrize("tool, group", [
    (TOOL_ZONE, "Predicción"), (TOOL_BRUSH, "Edición"), (TOOL_ERASER, "Edición"),
    (TOOL_DELETE, "Edición"), (TOOL_RECLASS, "Edición"), (TOOL_ANALYZE, "Análisis"),
])
def test_the_group_of_the_active_tool_is_framed(window, tool, group):
    if tool == TOOL_ZONE:
        window.radio_zone.setChecked(True)
    else:
        window.tool_buttons[tool].setChecked(True)
    assert window.active_group == group
    assert _framed_groups(window) == [group]                 # uno solo a la vez; Vista nunca


def test_the_frame_moves_with_the_selection(window):
    assert _framed_groups(window) == ["Edición"]             # al abrir: pincel
    window.radio_zone.setChecked(True)
    assert _framed_groups(window) == ["Predicción"]
    window.tool_buttons[TOOL_ANALYZE].setChecked(True)
    assert _framed_groups(window) == ["Análisis"]
    window.radio_full.setChecked(True)                       # «imagen completa» deja el pincel activo
    assert _framed_groups(window) == ["Análisis"]            # (no cambia la herramienta)


def test_framing_a_group_does_not_change_its_size(window):
    """El borde mide lo mismo con y sin resaltado: el panel no se mueve al cambiar de herramienta."""
    window.show()
    for group in window.group_boxes.values():
        window._style_group(group, active=False)
        plain = group.sizeHint()
        window._style_group(group, active=True)
        assert group.sizeHint() == plain
    window.hide()


def test_vista_group_is_never_framed(window):
    for tool in (TOOL_BRUSH, TOOL_ANALYZE, TOOL_DELETE):
        window.tool_buttons[tool].setChecked(True)
        assert "Vista" not in _framed_groups(window)


def test_analysis_tool_is_called_analizar_objeto(window):
    assert window.tool_buttons[TOOL_ANALYZE].text() == "Analizar objeto (A)"


# ---------- Medir distancia / Editar medida ----------

from gui_segesferoides.measures import MEASURE_COLORS, Measure  # noqa: E402

RED_RGB, GREEN_RGB, BLUE_RGB = (255, 0, 0), (0, 200, 0), (0, 90, 255)


def _drag_measure(window, start, end):
    c = window.canvas
    for kind, pos in ((QEvent.MouseButtonPress, start), (QEvent.MouseMove, end),
                      (QEvent.MouseButtonRelease, end)):
        button = Qt.NoButton if kind == QEvent.MouseMove else Qt.LeftButton
        event = QMouseEvent(kind, QPointF(*pos), button, button, Qt.NoModifier)
        {QEvent.MouseButtonPress: c.mousePressEvent, QEvent.MouseMove: c.mouseMoveEvent,
         QEvent.MouseButtonRelease: c.mouseReleaseEvent}[kind](event)


def _pick(menu, text):
    """La acción `text` del menú o de sus submenús."""
    for action in menu.actions():
        if action.text() == text:
            return action
        if action.menu() is not None:
            found = _pick(action.menu(), text)
            if found is not None:
                return found
    return None


def _choose_in_menu(monkeypatch, text):
    """Hace que el menú de la medida «elija» `text` (None = lo cierra sin elegir)."""
    monkeypatch.setattr(QMenu, "exec_", lambda self, *a: _pick(self, text) if text else None)


@pytest.fixture
def measured(qapp, window, image_path):
    window.open_image(image_path)
    window.tool_buttons[TOOL_MEASURE].setChecked(True)
    _drag_measure(window, (20, 40), (80, 40))
    window.tool_buttons[TOOL_EDIT_MEASURE].setChecked(True)
    return window


def test_measure_tools_live_in_analysis_with_the_color_combo_to_the_right(window):
    assert _group_title(window.tool_buttons[TOOL_MEASURE]) == "Análisis"
    assert _group_title(window.tool_buttons[TOOL_EDIT_MEASURE]) == "Análisis"
    assert window.tool_buttons[TOOL_MEASURE].text() == "Medir distancia"
    assert window.tool_buttons[TOOL_EDIT_MEASURE].text() == "Editar medida"
    # el combo de color comparte fila con «Medir distancia» y queda a su derecha
    assert window.combo_measure_color.parentWidget() is window.tool_buttons[TOOL_MEASURE].parentWidget()
    layout = window.combo_measure_color.parentWidget().layout()
    assert layout.indexOf(window.combo_measure_color) > layout.indexOf(window.tool_buttons[TOOL_MEASURE])


def test_color_combo_offers_the_palette_and_sets_the_next_measure_color(window):
    names = [window.combo_measure_color.itemText(i) for i in range(window.combo_measure_color.count())]
    assert names == [name for name, _rgb in MEASURE_COLORS]
    assert window.canvas.measure_color == window.combo_measure_color.itemData(0)
    window.combo_measure_color.setCurrentIndex(names.index("Azul"))
    assert window.canvas.measure_color == BLUE_RGB
    assert not window.combo_measure_color.itemIcon(0).isNull()      # cada color con su cuadradito


def test_each_measure_takes_the_color_selected_when_it_was_drawn(qapp, window, image_path):
    window.open_image(image_path)
    window.tool_buttons[TOOL_MEASURE].setChecked(True)
    _drag_measure(window, (10, 10), (60, 10))
    window.combo_measure_color.setCurrentIndex(1)                   # Verde
    _drag_measure(window, (10, 30), (60, 30))
    assert [m.color for m in window.canvas.measures] == [RED_RGB, GREEN_RGB]


@pytest.mark.parametrize("tool", [TOOL_MEASURE, TOOL_EDIT_MEASURE])
def test_measure_tools_are_exclusive_with_the_zone_and_frame_analysis(window, tool):
    window.radio_zone.setChecked(True)
    window.tool_buttons[tool].setChecked(True)
    assert window.canvas.tool == tool and not window.radio_zone.isChecked()
    assert _checked_tools(window) == [tool] and _framed_groups(window) == ["Análisis"]


def test_edit_menu_offers_delete_change_color_and_move(qapp, measured, monkeypatch):
    seen = []

    def fake_exec(menu, *_):
        seen.append(([a.text() for a in menu.actions()],
                     [a.text() for a in _pick(menu, "Cambiar color").menu().actions()]))
        return None

    monkeypatch.setattr(QMenu, "exec_", fake_exec)
    _drag_measure(measured, (50, 41), (50, 41))                        # un click sobre la medida
    options, colors = seen[0]
    assert options == ["Eliminar", "Cambiar color", "Mover"]
    assert colors == [name for name, _rgb in MEASURE_COLORS]


def test_edit_menu_delete(qapp, measured, monkeypatch):
    _choose_in_menu(monkeypatch, "Eliminar")
    _drag_measure(measured, (50, 41), (50, 41))
    assert measured.canvas.measures == []


def test_edit_menu_change_color_only_changes_that_measure(qapp, measured, monkeypatch):
    measured.tool_buttons[TOOL_MEASURE].setChecked(True)
    measured.combo_measure_color.setCurrentIndex(1)                     # una segunda medida, verde
    _drag_measure(measured, (20, 80), (80, 80))
    measured.tool_buttons[TOOL_EDIT_MEASURE].setChecked(True)
    first = measured.canvas.measures[0]
    _choose_in_menu(monkeypatch, "Azul")
    _drag_measure(measured, (50, 41), (50, 41))                         # la primera (roja)
    assert [m.color for m in measured.canvas.measures] == [BLUE_RGB, GREEN_RGB]
    assert (measured.canvas.measures[0].p1, measured.canvas.measures[0].p2) == (first.p1, first.p2)


def test_edit_menu_move_keeps_length_and_color(qapp, measured, monkeypatch):
    before = measured.canvas.measures[0]
    _choose_in_menu(monkeypatch, "Mover")
    _drag_measure(measured, (50, 41), (50, 41))
    assert measured.canvas.cursor().shape() == Qt.SizeAllCursor        # lista para moverse
    zoom = measured.canvas.zoom                                         # (la vista no está a escala 1:1)
    _drag_measure(measured, (50, 41), (70, 61))                         # +20 px de pantalla en x e y
    after = measured.canvas.measures[0]
    assert after.p1 == pytest.approx((before.p1[0] + 20 / zoom, before.p1[1] + 20 / zoom))
    assert after.p2 == pytest.approx((before.p2[0] + 20 / zoom, before.p2[1] + 20 / zoom))
    assert after.length == pytest.approx(before.length) and after.color == before.color
    assert after.p2[1] == after.p1[1]                                   # sigue horizontal


def test_closing_the_menu_without_choosing_changes_nothing(qapp, measured, monkeypatch):
    before = list(measured.canvas.measures)
    _choose_in_menu(monkeypatch, None)
    _drag_measure(measured, (50, 41), (50, 41))
    assert measured.canvas.measures == before and measured.canvas.selected_measure is None


def test_measures_do_not_touch_the_mask_or_mark_the_work_as_modified(qapp, window, image_path):
    window.open_image(image_path)
    mask_before = window.mask.copy()
    window.tool_buttons[TOOL_MEASURE].setChecked(True)
    _drag_measure(window, (20, 40), (80, 40))
    assert len(window.canvas.measures) == 1
    assert np.array_equal(window.mask, mask_before) and not window.dirty


def test_opening_another_image_clears_the_measures(qapp, window, image_path, tmp_path):
    window.open_image(image_path)
    window.tool_buttons[TOOL_MEASURE].setChecked(True)
    _drag_measure(window, (20, 40), (80, 40))
    other = tmp_path / "otra.png"
    cv2.imwrite(str(other), np.full((100, 100, 3), 50, np.uint8))
    window.dirty = False
    window.open_image(str(other))
    assert window.canvas.measures == []


# ---------- Guardar imagen ----------

@pytest.fixture
def colorful(qapp, window, tmp_path):
    """Imagen 200x300 de un color vivo, abierta en la ventana."""
    p = tmp_path / "color_4x.png"
    bgr = np.zeros((200, 300, 3), np.uint8)
    bgr[:] = (20, 60, 200)                                            # BGR: rojizo/naranja
    cv2.imwrite(str(p), bgr)
    window.open_image(str(p))
    return window


def test_save_controls_are_in_vista_and_offer_the_formats_and_channels(window):
    assert _group_title(window.save_image_button) == "Vista"
    assert _group_title(window.combo_save_format) == "Vista"
    assert _group_title(window.combo_save_channels) == "Vista"
    assert window.save_image_button.text() == "Guardar imagen"
    formats = [window.combo_save_format.itemText(i) for i in range(window.combo_save_format.count())]
    assert formats == ["PNG", "JPEG", "TIFF", "BMP"]
    channels = [window.combo_save_channels.itemText(i) for i in range(window.combo_save_channels.count())]
    assert channels == ["Color", "Escala de grises"]
    # el formato está a la derecha del botón, en la misma fila
    layout = window.combo_save_format.parentWidget().layout()
    assert window.combo_save_format.parentWidget() is window.save_image_button.parentWidget()
    assert layout.indexOf(window.combo_save_format) > layout.indexOf(window.save_image_button)


def test_save_image_button_needs_an_image(qapp, window, image_path):
    assert not window.save_image_button.isEnabled()
    window.open_image(image_path)
    assert window.save_image_button.isEnabled()


@pytest.mark.parametrize("ext", [".png", ".jpg", ".tif", ".bmp"])
def test_save_image_writes_each_format_at_native_resolution(colorful, tmp_path, ext):
    path = str(tmp_path / f"salida{ext}")
    assert colorful.save_image(path)
    saved = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert saved is not None and saved.shape == (200, 300, 3)           # misma resolución, 3 canales


def test_save_in_color_keeps_the_image_and_draws_the_measures(colorful, tmp_path):
    colorful.canvas.measures.append(Measure((20.0, 100.0), (280.0, 100.0), BLUE_RGB))   # coords de imagen
    path = str(tmp_path / "c.png")
    colorful.save_image(path)
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert tuple(rgb[10, 10]) == (200, 60, 20)                          # la imagen, sin tocar (en RGB)
    assert tuple(rgb[100, 150]) == BLUE_RGB                              # la medida, con su color


def test_save_in_grayscale_turns_the_image_gray_but_keeps_the_measure_colors(colorful, tmp_path):
    colorful.canvas.measures.append(Measure((20.0, 100.0), (280.0, 100.0), RED_RGB))
    colorful.combo_save_channels.setCurrentIndex(1)                     # Escala de grises
    path = str(tmp_path / "g.png")
    colorful.save_image(path)
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert rgb[10, 10, 0] == rgb[10, 10, 1] == rgb[10, 10, 2]           # fondo en grises
    assert tuple(rgb[100, 150]) == RED_RGB                               # la línea sigue roja


def test_save_in_grayscale_without_measures_is_all_gray(colorful, tmp_path):
    colorful.combo_save_channels.setCurrentIndex(1)
    path = str(tmp_path / "g2.png")
    colorful.save_image(path)
    img = cv2.imread(path)
    assert np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2])


def test_save_image_does_not_include_the_mask(colorful, tmp_path):
    colorful.mask[50:150, 50:150] = 2                                   # una máscara bien visible
    path = str(tmp_path / "sinmask.png")
    colorful.save_image(path)
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert tuple(rgb[100, 100]) == (200, 60, 20)


def test_save_image_dialog_uses_the_format_from_the_combo(qapp, colorful, tmp_path, monkeypatch):
    asked = []

    def fake_dialog(parent, title, default, filt):
        asked.append((default, filt))
        return str(tmp_path / "elegido.png"), ""                        # el usuario tipeó .png

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    colorful.combo_save_format.setCurrentIndex(1)                       # JPEG
    colorful._on_save_image()
    assert asked[0][0].endswith("color_4x_copia.jpg")                   # sin medidas: «_copia», nunca pisa el original
    assert (tmp_path / "elegido.jpg").exists() and not (tmp_path / "elegido.png").exists()


def test_save_image_default_name_mentions_the_measures(qapp, colorful, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a: asked.append(a[2]) or ("", "")))
    colorful.tool_buttons[TOOL_MEASURE].setChecked(True)
    _drag_measure(colorful, (20, 100), (280, 100))
    colorful._on_save_image()
    assert asked[0].endswith("color_4x_medidas.png")


def test_save_image_with_a_dotted_name_keeps_it(qapp, colorful, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a: (str(tmp_path / "muestra v2.1"), "")))
    colorful._on_save_image()
    assert (tmp_path / "muestra v2.1.png").exists()                     # no le comió «.1» como extensión


def test_save_image_labels_use_the_calibration(colorful, tmp_path):
    """Con calibración, la etiqueta dice la distancia en la unidad elegida (no en px)."""
    from gui_segesferoides.measures import label_text
    colorful.calibration.set("4x", 2.0, "µm")
    colorful.calibration.display_unit = "µm"
    colorful.combo_magnification.setCurrentText("4x")
    colorful.canvas.measures.append(Measure((20.0, 100.0), (120.0, 100.0), RED_RGB))     # 100 px
    scale = colorful.calibration.scale("4x")
    assert label_text(colorful.canvas.measures[0], scale) == scale.length(100.0) == "200.0 µm"
    assert colorful.save_image(str(tmp_path / "cal.png"))


def test_edit_menu_acts_on_the_clicked_measure_among_several(qapp, measured, monkeypatch):
    measured.tool_buttons[TOOL_MEASURE].setChecked(True)
    _drag_measure(measured, (20, 80), (80, 80))                         # segunda medida, más abajo
    measured.tool_buttons[TOOL_EDIT_MEASURE].setChecked(True)
    first, second = measured.canvas.measures
    _choose_in_menu(monkeypatch, "Eliminar")
    _drag_measure(measured, (50, 81), (50, 81))                         # click sobre la segunda
    assert measured.canvas.measures == [first]


def test_move_from_the_menu_lets_you_drag_the_label_or_the_line(qapp, measured, monkeypatch):
    """«Mover» sirve para las dos cosas: según dónde se empiece a arrastrar."""
    _choose_in_menu(monkeypatch, "Mover")
    canvas = measured.canvas
    before = canvas.measures[0]
    _drag_measure(measured, (50, 41), (50, 41))                         # elige «Mover»
    cx, cy = canvas._label_rect_view(0).center().x(), canvas._label_rect_view(0).center().y()
    _drag_measure(measured, (cx, cy), (cx + 30, cy + 30))               # arrastra la etiqueta
    labeled = canvas.measures[0]
    assert (labeled.p1, labeled.p2) == (before.p1, before.p2) and labeled.label_offset != (0.0, 0.0)
    _drag_measure(measured, (50, 41), (50, 41))                         # vuelve a elegir «Mover»
    _drag_measure(measured, (50, 41), (70, 61))                         # ahora arrastra la línea
    after = canvas.measures[0]
    assert after.p1 != before.p1 and after.length == pytest.approx(before.length)
    assert after.label_offset == labeled.label_offset                   # la etiqueta acompañó a la línea


def test_saved_image_draws_the_label_where_it_was_moved(colorful, tmp_path):
    from gui_segesferoides.measures import label_rect, label_text
    m = Measure((20.0, 100.0), (120.0, 100.0), RED_RGB)
    colorful.canvas.measures.append(m.with_label_offset((100.0, 60.0)))
    path = str(tmp_path / "etiqueta.png")
    colorful.save_image(path)
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    scale = colorful.calibration.scale("4x")
    h, w = rgb.shape[:2]
    rect = label_rect(m.p1, m.p2, max(3.0, max(h, w) / 500), label_text(m, scale),
                      max(16.0, max(h, w) / 70), (100.0, 60.0))
    x, y = int(rect.center().x()), int(rect.center().y())
    assert tuple(rgb[y, x]) != (200, 60, 20)                            # ahí está la caja oscura de la etiqueta


def test_status_after_predicting_counts_detections_not_pixels(qapp, window, image_path):
    window.open_image(image_path)
    window._load_model("a")
    wait_idle(qapp, window)
    window._on_predict_clicked()
    wait_idle(qapp, window)
    text = window.status_label.text()
    assert "Detecciones" in text and "Píxeles" not in text
    # el backend de prueba dibuja un círculo de clase 2: es UNA detección, aunque tenga miles de píxeles
    assert int(np.sum(window.mask == 2)) > 100
    cls2 = next(c.name for c in window.backend.info.classes if c.id == 2)
    assert re.search(rf"{re.escape(cls2)}: 1(,|$)", text)           # exacto: «1», no «17671»
    others = [c.name for c in window.backend.info.classes if c.id != 2]
    assert all(re.search(rf"{re.escape(n)}: 0(,|$)", text) for n in others)


# ---------- Lazo ----------

def _image_to_view(window, x, y):
    from gui_segesferoides import viewport
    return viewport.image_to_view(window.canvas.offset, window.canvas.zoom, x, y)


def _click_image(window, x, y):
    """Clic del mouse sobre el punto (x, y) de la imagen (pasando por las coordenadas de pantalla)."""
    vx, vy = _image_to_view(window, x, y)
    _drag_measure(window, (vx, vy), (vx, vy))


def _lasso_triangle(window, points=((60, 40), (160, 50), (110, 140))):
    for x, y in points:
        _click_image(window, x, y)
    _click_image(window, *points[0])                       # clic sobre el primero: cierra


def test_lasso_is_a_tool_in_edicion_right_after_the_brush(window):
    assert window.tool_buttons[TOOL_LASSO].text() == "Lazo (L)"
    assert _group_title(window.tool_buttons[TOOL_LASSO]) == "Edición"
    layout = window.tool_buttons[TOOL_BRUSH].parentWidget().layout()
    assert layout.indexOf(window.tool_buttons[TOOL_LASSO]) == layout.indexOf(window.tool_buttons[TOOL_BRUSH]) + 1


def test_lasso_shows_the_class_selector_and_a_hint_but_no_size_sliders(window):
    window.tool_buttons[TOOL_LASSO].setChecked(True)
    assert window.canvas.tool == TOOL_LASSO
    assert not window.row_class.isHidden() and not window.lasso_hint.isHidden()
    assert window.row_brush.isHidden() and window.row_eraser.isHidden()
    window.tool_buttons[TOOL_BRUSH].setChecked(True)
    assert window.lasso_hint.isHidden() and not window.row_class.isHidden()      # la pista es solo del lazo
    window.tool_buttons[TOOL_ERASER].setChecked(True)
    assert window.lasso_hint.isHidden() and window.row_class.isHidden()


def test_lasso_hint_explains_how_to_close_it(window):
    text = window.lasso_hint.text()
    assert "primer punto" in text and "doble clic" in text and "Enter" in text and "Esc" in text


def test_lasso_shortcut_and_group_frame(window):
    shortcuts = [sc for sc in window.findChildren(QShortcut) if sc.key().toString() == "L"]
    assert len(shortcuts) == 1
    shortcuts[0].activated.emit()
    assert window.canvas.tool == TOOL_LASSO and _framed_groups(window) == ["Edición"]
    assert _checked_tools(window) == [TOOL_LASSO]


def test_closing_a_lasso_adds_an_object_of_the_selected_class(qapp, window, image_path):
    window.open_image(image_path)
    window.tool_buttons[TOOL_LASSO].setChecked(True)
    window.combo_class.setCurrentIndex(1)
    class_id = window.combo_class.currentData()
    _lasso_triangle(window)
    inside = (110, 80)
    assert window.mask[inside[1], inside[0]] == class_id
    assert window.mask[5, 5] == 0 and window.mask[250, 350] == 0
    assert set(np.unique(window.mask)) == {0, class_id}
    assert window.dirty and "lazo" in window.status_label.text().lower()


def test_the_lasso_object_uses_the_class_chosen_in_the_combo(qapp, window, image_path):
    window.open_image(image_path)
    window.tool_buttons[TOOL_LASSO].setChecked(True)
    for index in range(window.combo_class.count()):
        window.mask[:] = 0
        window.combo_class.setCurrentIndex(index)
        _lasso_triangle(window)
        assert window.mask[80, 110] == window.combo_class.itemData(index)


def test_a_lasso_replaces_what_was_under_it(qapp, window, image_path):
    window.open_image(image_path)
    window.mask[:] = 0
    window.mask[30:150, 50:170] = window.combo_class.itemData(0)         # un objeto de la primera clase
    window.tool_buttons[TOOL_LASSO].setChecked(True)
    window.combo_class.setCurrentIndex(1)
    _lasso_triangle(window)
    assert window.mask[80, 110] == window.combo_class.itemData(1)        # lo de adentro, pisado
    assert window.mask[35, 55] == window.combo_class.itemData(0)         # lo de afuera del lazo, intacto


def test_lasso_is_one_undo_step(qapp, window, image_path):
    window.open_image(image_path)
    before = window.mask.copy()
    window.tool_buttons[TOOL_LASSO].setChecked(True)
    _lasso_triangle(window)
    assert np.any(window.mask)
    window._undo()
    assert np.array_equal(window.mask, before)                           # un solo Ctrl+Z lo deshace
    window._redo()
    assert np.any(window.mask)


def test_a_degenerate_lasso_changes_nothing(qapp, window, image_path):
    window.open_image(image_path)
    window.tool_buttons[TOOL_LASSO].setChecked(True)
    window._on_lasso_finished(np.array([(10.0, 10.0), (10.2, 10.0), (10.0, 10.2)]))   # menos de 1 px²
    assert not np.any(window.mask) and not window.dirty


def test_lasso_without_an_image_does_nothing(window):
    window._on_lasso_finished(np.array([(10.0, 10.0), (60.0, 10.0), (35.0, 60.0)]))
    assert window.mask is None or not np.any(window.mask)


def test_class_combo_shows_each_class_color(window):
    for i in range(window.combo_class.count()):
        class_id = window.combo_class.itemData(i)
        icon = window.combo_class.itemIcon(i)
        assert not icon.isNull()
        rgb = window.palette[class_id]
        pixel = icon.pixmap(14, 14).toImage().pixelColor(7, 7)
        assert (pixel.red(), pixel.green(), pixel.blue()) == tuple(rgb)


def test_lasso_preview_uses_the_selected_class_color(window):
    for i in range(window.combo_class.count()):
        window.combo_class.setCurrentIndex(i)
        assert window.canvas.lasso_color == tuple(window.palette[window.combo_class.currentData()])


def test_lasso_curve_is_curved_in_the_mask_not_a_straight_edge(qapp, window, image_path):
    """Marcar las 4 esquinas de un cuadrado deja un objeto redondeado: los lados abultan
    hacia afuera de la recta que uniría los puntos (con rectas, nada saldría de y=60)."""
    window.open_image(image_path)
    window.tool_buttons[TOOL_LASSO].setChecked(True)
    _lasso_triangle(window, points=((100, 60), (200, 60), (200, 160), (100, 160)))
    class_id = window.combo_class.currentData()
    above = np.flatnonzero(window.mask[:60, 150] == class_id)             # por encima del lado superior recto
    assert len(above) >= 3                                                # abulta al menos 3 px sobre la recta
    assert window.mask[80, 150] == class_id and window.mask[10, 10] == 0
