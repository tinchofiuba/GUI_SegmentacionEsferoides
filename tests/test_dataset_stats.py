import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from gui_segesferoides import dataset_stats
from gui_segesferoides.postprocess_dialog import PostprocessDialog
from gui_segesferoides.stats_dialog import StatsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dataset(tmp_path):
    """Dos imágenes (4x y 10x) con máscara de ids, y una sin máscara."""
    mask = np.zeros((60, 60), np.uint8)
    mask[5:25, 5:25] = 1
    mask[35:55, 35:55] = 2
    for name in ("a_4x", "b_10x", "c_4x"):
        cv2.imwrite(str(tmp_path / f"{name}.png"), np.zeros((60, 60, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "a-Mask.png"), mask)
    cv2.imwrite(str(tmp_path / "b-Mask.png"), mask)
    return [str(tmp_path / f"{n}.png") for n in ("a_4x", "b_10x", "c_4x")]


def test_names():
    assert dataset_stats.magnification_of("/x/muestra_4x.png") == "4x"
    assert dataset_stats.magnification_of("/x/muestra-10X.png") == "10x"
    assert dataset_stats.magnification_of("/x/muestra.png") is None


def test_compute_groups_by_magnification_and_class(tmp_path):
    data, skipped = dataset_stats.compute(_dataset(tmp_path))
    assert data["n_images"] == 2
    assert [s[0].endswith("c_4x.png") for s in skipped] == [True]
    assert set(data["stats"]) == {"4x", "10x"}
    c1 = data["stats"]["4x"]["1"]
    assert c1["n"] == 1 and c1["area"]["mean"] == 400 and c1["area"]["p5"] == c1["area"]["p95"] == 400
    assert set(c1) == {"n", "area", "diameter", "perimeter", "circularity"}


def test_color_mask_needs_palette(tmp_path):
    color = np.zeros((20, 20, 3), np.uint8)
    color[2:12, 2:12] = (0, 255, 0)   # BGR: verde
    path = str(tmp_path / "m-Mask.png")
    cv2.imwrite(path, color)
    assert dataset_stats.read_mask(path) is None
    mask = dataset_stats.read_mask(path, {1: (0, 255, 0)})
    assert (mask == 1).sum() == 100


def test_store_roundtrip(tmp_path):
    path = str(tmp_path / "s.json")
    assert dataset_stats.DatasetStats(path).stats == {}
    data, _ = dataset_stats.compute(_dataset(tmp_path))
    dataset_stats.DatasetStats(path).set(data)
    assert dataset_stats.DatasetStats(path).stats == data["stats"]


def test_stats_dialog_shows_tabs(qapp, tmp_path):
    store = dataset_stats.DatasetStats(str(tmp_path / "s.json"))
    dialog = StatsDialog(store, [(1, "chicas"), (2, "grandes")])
    assert dialog.tabs.count() == 0
    store.set(dataset_stats.compute(_dataset(tmp_path))[0])
    dialog._refresh()
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == ["4x", "10x"]
    assert dialog.tabs.widget(0).rowCount() == 2 * 4   # 2 clases x 4 métricas


def test_postprocess_dialog_button_order(qapp, tmp_path):
    base = {"rules": [], "fill_holes": []}
    store = dataset_stats.DatasetStats(str(tmp_path / "s.json"))
    d = PostprocessDialog(base, base, [(1, "chicas")], stats_store=store)
    assert d.button_stats.isEnabled()
    d.show()
    qapp.processEvents()
    restore = next(b for b in d.findChildren(type(d.button_stats)) if b.text().startswith("Restaurar"))
    cancel = next(b for b in d.findChildren(type(d.button_stats)) if "ancel" in b.text())
    left = lambda b: b.mapTo(d, b.rect().topLeft()).x()  # noqa: E731 - el de Cancelar es hijo del QDialogButtonBox
    assert left(d.button_stats) < left(restore) < left(cancel)
    assert left(cancel) - (left(restore) + restore.width()) >= 12   # separado de Cancelar
    assert not PostprocessDialog(base, base, [(1, "chicas")]).button_stats.isEnabled()
