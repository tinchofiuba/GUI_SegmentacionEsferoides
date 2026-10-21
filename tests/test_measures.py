import math

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from gui_segesferoides.calibration import Scale
from gui_segesferoides.measures import (
    DEFAULT_MEASURE_COLOR,
    MEASURE_COLORS,
    Measure,
    clamp_point,
    clamp_translation,
    distance_to_segment,
    label_text,
    render_measures,
)

RED, GREEN = (255, 0, 0), (0, 200, 0)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_length_orientation_and_color_survive_a_translation():
    m = Measure((10.0, 20.0), (40.0, 60.0), GREEN)
    moved = m.translated(5.0, -3.0)
    assert moved.length == pytest.approx(m.length) == pytest.approx(50.0)
    assert (moved.p2[0] - moved.p1[0], moved.p2[1] - moved.p1[1]) == (30.0, 40.0)   # misma orientación
    assert moved.color == GREEN and moved.p1 == (15.0, 17.0)
    assert m.p1 == (10.0, 20.0)                      # inmutable: la original no cambia


def test_with_color_changes_only_the_color():
    m = Measure((1.0, 2.0), (3.0, 4.0), RED).with_color((0, 90, 255))
    assert (m.p1, m.p2, m.color) == ((1.0, 2.0), (3.0, 4.0), (0, 90, 255))


def test_distance_to_segment():
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert distance_to_segment((5, 3), a, b) == pytest.approx(3)         # perpendicular
    assert distance_to_segment((14, 3), a, b) == pytest.approx(5)        # más allá del extremo: al extremo
    assert distance_to_segment((-3, 4), a, b) == pytest.approx(5)
    assert distance_to_segment((3, 4), (1, 1), (1, 1)) == pytest.approx(math.hypot(2, 3))   # segmento nulo


def test_clamp_translation_keeps_the_measure_inside_the_image():
    m = Measure((10.0, 10.0), (50.0, 30.0), RED)
    assert clamp_translation(m, 5, 5, 100, 100) == (5, 5)               # cabe: sin cambios
    assert clamp_translation(m, 999, 0, 100, 100) == (49, 0)            # tope a la derecha (x máx 99)
    assert clamp_translation(m, -999, -999, 100, 100) == (-10, -10)     # tope a la izquierda y arriba
    dx, dy = clamp_translation(m, 999, 999, 100, 100)
    assert m.translated(dx, dy).length == pytest.approx(m.length)


def test_clamp_point():
    assert clamp_point((-5.0, 500.0), 100, 80) == (0.0, 79.0)
    assert clamp_point((10.5, 20.5), 100, 80) == (10.5, 20.5)


def test_label_in_pixels_or_in_the_calibrated_unit():
    m = Measure((0.0, 0.0), (30.0, 40.0), RED)                          # 50 px
    assert label_text(m, Scale()) == "50.0 px"
    assert label_text(m, Scale(2.0, "µm")) == Scale(2.0, "µm").length(50.0)
    assert "µm" in label_text(m, Scale(2.0, "µm"))


def test_default_color_is_in_the_palette():
    assert DEFAULT_MEASURE_COLOR in [rgb for _name, rgb in MEASURE_COLORS]
    assert len({rgb for _n, rgb in MEASURE_COLORS}) == len(MEASURE_COLORS)   # sin repetidos


def _flat_image(h=100, w=120, rgb=(120, 80, 40)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = rgb
    return img


def test_render_without_measures_is_a_copy(qapp):
    img = _flat_image()
    out = render_measures(img, [], Scale())
    assert np.array_equal(out, img) and out is not img


def test_render_draws_the_measure_in_its_color_without_touching_the_input(qapp):
    img = _flat_image()
    before = img.copy()
    out = render_measures(img, [Measure((10.0, 50.0), (110.0, 50.0), RED)], Scale())
    assert out.shape == img.shape and tuple(out[50, 60]) == RED          # sobre la línea
    assert tuple(out[5, 5]) == (120, 80, 40)                             # lejos de la línea: intacto
    assert np.array_equal(img, before)


def test_render_in_grayscale_keeps_the_measure_colors(qapp):
    img = _flat_image(rgb=(200, 60, 20))                                 # imagen bien colorida
    out = render_measures(img, [Measure((10.0, 50.0), (110.0, 50.0), GREEN)], Scale(), grayscale=True)
    assert out.shape == (100, 120, 3)                                    # sigue teniendo 3 canales
    assert tuple(out[50, 60]) == GREEN                                   # la medida conserva su color
    corner = out[5, 5]
    assert corner[0] == corner[1] == corner[2]                           # el resto es gris
    assert not np.array_equal(out[5, 5], img[5, 5])


def test_render_grayscale_without_measures_is_pure_gray(qapp):
    out = render_measures(_flat_image(rgb=(200, 60, 20)), [], Scale(), grayscale=True)
    assert np.all(out[..., 0] == out[..., 1]) and np.all(out[..., 1] == out[..., 2])


def test_render_scales_the_line_with_the_image_size(qapp):
    big = _flat_image(h=3000, w=4000)
    out = render_measures(big, [Measure((200.0, 1500.0), (3800.0, 1500.0), RED)], Scale())
    thick = np.count_nonzero(np.all(out[:, 2000] == RED, axis=1))
    assert thick >= 5                                                    # visible en una imagen grande


# ---------- Etiqueta movible ----------

from gui_segesferoides.measures import clamp_label_shift, label_rect  # noqa: E402


def test_a_measure_starts_with_its_label_in_the_default_place():
    assert Measure((0.0, 0.0), (10.0, 0.0), RED).label_offset == (0.0, 0.0)


def test_with_label_offset_changes_only_the_label():
    m = Measure((1.0, 2.0), (30.0, 4.0), RED).with_label_offset((5, -7))
    assert m.label_offset == (5.0, -7.0) and (m.p1, m.p2, m.color) == ((1.0, 2.0), (30.0, 4.0), RED)


def test_translating_the_line_keeps_the_label_offset():
    m = Measure((10.0, 10.0), (50.0, 10.0), RED).with_label_offset((8.0, 20.0))
    assert m.translated(30, 5).label_offset == (8.0, 20.0)         # la etiqueta acompaña a la línea
    assert m.with_color(GREEN).label_offset == (8.0, 20.0)


def test_label_rect_moves_with_the_offset(qapp):
    a, b = (10.0, 50.0), (110.0, 50.0)
    home = label_rect(a, b, 3, "100 px", 16)
    moved = label_rect(a, b, 3, "100 px", 16, offset=(30.0, 45.0))
    assert (moved.x() - home.x(), moved.y() - home.y()) == pytest.approx((30.0, 45.0))
    assert (moved.width(), moved.height()) == (home.width(), home.height())
    assert home.center().y() < 50 and abs(home.center().x() - 60) < 1      # de fábrica: arriba y al medio


def test_clamp_label_shift_keeps_the_label_inside_the_image():
    m = Measure((10.0, 10.0), (50.0, 10.0), RED)                  # punto medio (30, 10)
    assert clamp_label_shift(m, 5, 5, 100, 100) == (5, 5)
    dx, dy = clamp_label_shift(m, 999, 999, 100, 100)
    assert (30 + dx, 10 + dy) == (99.0, 99.0)
    m2 = m.with_label_offset((40.0, 0.0))                         # ya corrida: el tope cuenta lo acumulado
    dx, dy = clamp_label_shift(m2, 999, 0, 100, 100)
    assert 30 + 40 + dx == 99.0


def test_render_draws_the_label_where_the_offset_puts_it(qapp):
    img = _flat_image(h=200, w=300, rgb=(230, 230, 230))
    m = Measure((20.0, 100.0), (120.0, 100.0), RED)
    moved = m.with_label_offset((120.0, 60.0))
    scale = Scale()
    home_rect = label_rect(m.p1, m.p2, 3.0, label_text(m, scale), 16.0)
    new_rect = label_rect(moved.p1, moved.p2, 3.0, label_text(moved, scale), 16.0, moved.label_offset)
    out_home = render_measures(img, [m], scale)
    out_moved = render_measures(img, [moved], scale)
    hx, hy = int(home_rect.center().x()), int(home_rect.center().y())
    nx, ny = int(new_rect.center().x()), int(new_rect.center().y())
    dark = lambda out, x, y: out[y, x].sum() < 3 * 150            # la caja de la etiqueta oscurece el fondo
    assert dark(out_home, hx, hy) and not dark(out_home, nx, ny)
    assert dark(out_moved, nx, ny) and not dark(out_moved, hx, hy)  # se fue de su lugar de siempre
