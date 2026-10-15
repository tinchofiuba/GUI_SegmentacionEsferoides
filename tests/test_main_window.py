import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QMessageBox

from gui_segesferoides.config import BackendSpec
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
    w = MainWindow([BackendSpec("fake", [sys.executable, FAKE])])
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
    w = MainWindow([spec])
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
    w = MainWindow([spec])
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
