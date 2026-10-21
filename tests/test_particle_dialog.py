import pytest
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFormLayout

from gui_segesferoides.calibration import Scale
from gui_segesferoides.particle_dialog import ParticleDialog

STATS = {"area": 5000.0, "perimeter": 300.0, "equivalent_diameter": 80.0, "circularity": 0.8}
REFERENCE = {"n": 212,
             "area": {"mean": 4000.0, "p5": 2000.0, "p95": 9000.0},
             "perimeter": {"mean": 280.0, "p5": 150.0, "p95": 450.0},
             "diameter": {"mean": 75.0, "p5": 50.0, "p95": 110.0},
             "circularity": {"mean": 0.78, "p5": 0.5, "p95": 0.88}}
COUNTS = [("células sueltas", 212), ("esferoides", 193), ("atípicos", 13)]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp):
    d = ParticleDialog([(1, "células sueltas"), (2, "esferoides")])
    d.magnification = "10x"
    d.show_particle(1, STATS, REFERENCE, COUNTS, Scale())
    d.move(20, 40)
    d.show()
    qapp.processEvents()
    yield d
    d.close()


class FakeScreen:
    """Pantalla de 1600x900: la del modo offscreen es chica y no sirve para medir si entra."""
    def availableGeometry(self):
        return QRect(0, 0, 1600, 900)


def _open(d, key="equivalent_diameter"):
    d.bars[key].clicked.emit()
    return d.reference_panel


def test_clicking_a_bar_opens_a_non_modal_panel_to_the_right(qapp, dialog):
    panel = _open(dialog)
    frame = dialog.frameGeometry()
    assert panel.isVisible() and not panel.isModal()
    assert QApplication.activeModalWidget() is None            # se puede seguir usando la ventana
    assert panel.windowFlags() & Qt.FramelessWindowHint        # se ve como desplegable, sin barra de título
    assert panel.x() >= frame.x() + frame.width()              # a la derecha del popup
    assert panel.y() == frame.y()                              # alineado con su borde superior


def test_panel_shows_the_metric_that_was_clicked(qapp, dialog):
    panel = _open(dialog, "equivalent_diameter")
    assert "Diámetro equivalente" in panel.title_label.text()
    assert panel.form.itemAt(0, QFormLayout.FieldRole).widget().text() == Scale().length(80.0)
    panel = _open(dialog, "area")                              # otro intervalo: mismo desplegable, otro detalle
    assert panel is dialog.reference_panel and "Área" in panel.title_label.text()
    assert panel.form.itemAt(0, QFormLayout.FieldRole).widget().text() == Scale().area(5000.0)


def test_popup_moves_left_to_make_room_when_the_panel_does_not_fit(qapp, dialog, monkeypatch):
    monkeypatch.setattr(QApplication, "screenAt", staticmethod(lambda _p: FakeScreen()))
    dialog.move(1000, 100)                                     # el popup entra, pero no el desplegable a su lado
    qapp.processEvents()
    panel = _open(dialog)
    assert panel.frameGeometry().right() <= 1600               # el desplegable entra en la pantalla
    assert 0 <= dialog.x() < 1000                              # y para eso se corrió el popup a la izquierda
    assert panel.x() >= dialog.frameGeometry().x() + dialog.frameGeometry().width()


def test_popup_does_not_move_if_there_is_room(qapp, dialog, monkeypatch):
    monkeypatch.setattr(QApplication, "screenAt", staticmethod(lambda _p: FakeScreen()))
    dialog.move(30, 60)
    _open(dialog)
    assert (dialog.x(), dialog.y()) == (30, 60)


def test_only_the_clicked_interval_is_highlighted(qapp, dialog):
    assert not any(b.selected for b in dialog.bars.values())
    _open(dialog, "equivalent_diameter")
    assert [k for k, b in dialog.bars.items() if b.selected] == ["equivalent_diameter"]
    _open(dialog, "circularity")
    assert [k for k, b in dialog.bars.items() if b.selected] == ["circularity"]
    dialog.grab()                                              # pinta el resaltado sin errores


@pytest.mark.parametrize("how", ["x", "button", "escape"])
def test_closing_the_panel_clears_the_highlight(qapp, dialog, how):
    panel = _open(dialog)
    if how == "escape":
        QTest.keyClick(panel, Qt.Key_Escape)
    else:
        panel.close()
    assert not panel.isVisible()
    assert not any(b.selected for b in dialog.bars.values())


def test_closing_or_hiding_the_popup_closes_the_panel(qapp, dialog):
    panel = _open(dialog)
    dialog.hide()
    assert not panel.isVisible() and not any(b.selected for b in dialog.bars.values())
    dialog.show()
    panel = _open(dialog)
    dialog.close()
    assert not panel.isVisible()


def test_panel_follows_the_popup_when_it_moves(qapp, dialog):
    panel = _open(dialog)
    x0 = panel.x()
    dialog.move(dialog.x() + 100, dialog.y())
    qapp.processEvents()
    assert panel.x() == x0 + 100


def test_selecting_another_particle_refreshes_the_open_panel(qapp, dialog):
    panel = _open(dialog, "area")
    dialog.show_particle(1, {**STATS, "area": 7777.0}, REFERENCE, COUNTS, Scale())
    assert panel.isVisible() and dialog.bars["area"].selected
    assert panel.form.itemAt(0, QFormLayout.FieldRole).widget().text() == Scale().area(7777.0)


def test_panel_closes_if_the_new_particle_has_no_reference_for_that_metric(qapp, dialog):
    panel = _open(dialog, "area")
    dialog.show_particle(1, STATS, None, [], Scale())
    assert not panel.isVisible() and not any(b.selected for b in dialog.bars.values())
