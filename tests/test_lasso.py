import numpy as np
import pytest

from gui_segesferoides.editing import fill_polygon
from gui_segesferoides.lasso import polygon_area, smooth_closed_curve

SQUARE = [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)]


def test_the_curve_passes_through_every_marked_point():
    curve = smooth_closed_curve(SQUARE)
    for p in SQUARE:
        assert np.min(np.hypot(*(curve - p).T)) < 1e-9


def test_the_curve_is_not_a_polygon_it_bends_between_points():
    """El lazo no une los puntos con rectas: entre dos puntos la curva se aparta de la cuerda."""
    curve = smooth_closed_curve(SQUARE)
    seg = curve[: len(curve) // 4]                       # el tramo entre el 1.º y el 2.º punto
    chord_y = 10.0                                       # la recta entre (10,10) y (60,10)
    assert np.max(np.abs(seg[:, 1] - chord_y)) > 2.0     # se curva (en un polígono sería 0)
    assert len(curve) > len(SQUARE) * 4                  # y tiene muchos puntos intermedios


def test_the_curve_is_closed_and_ordered():
    curve = smooth_closed_curve(SQUARE)
    assert tuple(curve[0]) == SQUARE[0]                  # arranca en el primer punto
    order = [int(np.argmin(np.hypot(*(curve - p).T))) for p in SQUARE]
    assert order == sorted(order)                        # recorre los puntos en el orden marcado
    assert np.hypot(*(curve[-1] - curve[0])) < 15        # el último tramo vuelve al primero


def test_the_curve_does_not_overshoot_wildly_with_uneven_points():
    """Puntos muy desparejos (dos casi juntos y otros lejos) no hacen lazos ni picos enormes."""
    pts = [(10.0, 10.0), (12.0, 10.0), (90.0, 15.0), (80.0, 80.0), (15.0, 70.0)]
    curve = smooth_closed_curve(pts)
    lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
    margin = 0.35 * np.max(hi - lo)
    assert np.all(curve >= lo - margin) and np.all(curve <= hi + margin)


def test_fewer_than_three_points_do_not_make_a_curve():
    assert smooth_closed_curve([(1.0, 2.0)]).shape == (1, 2)
    assert smooth_closed_curve([(1.0, 2.0), (5.0, 6.0)]).shape == (2, 2)
    assert smooth_closed_curve([]).shape == (0, 2)


def test_repeated_points_are_ignored():
    pts = [(10.0, 10.0), (10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0), (10.0, 10.0)]
    assert np.allclose(smooth_closed_curve(pts), smooth_closed_curve(SQUARE))


def test_polygon_area():
    assert polygon_area(SQUARE) == pytest.approx(2500.0)
    assert polygon_area([(0, 0), (10, 0)]) == 0.0
    assert polygon_area([(0, 0), (10, 0), (20, 0)]) == pytest.approx(0.0)        # colineales


def test_fill_polygon_fills_the_inside_with_the_class():
    mask = np.zeros((80, 80), np.uint8)
    assert fill_polygon(mask, smooth_closed_curve(SQUARE), 2)
    assert mask[35, 35] == 2 and mask[0, 0] == 0 and mask[75, 75] == 0
    assert set(np.unique(mask)) == {0, 2}


def test_fill_polygon_area_matches_the_curve_area():
    curve = smooth_closed_curve(SQUARE)
    mask = np.zeros((80, 80), np.uint8)
    fill_polygon(mask, curve, 3)
    # el relleno incluye los píxeles del borde: queda apenas por encima del área geométrica (~1 px de más)
    filled = int((mask == 3).sum())
    assert polygon_area(curve) <= filled <= polygon_area(curve) * 1.06


def test_fill_polygon_overwrites_what_was_there():
    mask = np.zeros((80, 80), np.uint8)
    mask[20:50, 20:50] = 1
    fill_polygon(mask, smooth_closed_curve(SQUARE), 2)
    assert mask[35, 35] == 2                              # lo de adentro, pisado
    assert mask[65, 65] == 0 and mask[0, 0] == 0


def test_fill_polygon_rejects_degenerate_polygons():
    mask = np.zeros((20, 20), np.uint8)
    assert not fill_polygon(mask, [(1, 1), (5, 5)], 1)                       # menos de 3 puntos
    assert not fill_polygon(mask, [(1, 1), (5, 5), (9, 9)], 1)               # colineales: sin área
    assert not fill_polygon(mask, [(1.0, 1.0), (1.4, 1.0), (1.0, 1.4)], 1)   # menos de 1 px²
    assert not mask.any()


def test_fill_polygon_clips_to_the_image():
    mask = np.zeros((40, 40), np.uint8)
    assert fill_polygon(mask, [(-20.0, -20.0), (60.0, -20.0), (60.0, 60.0), (-20.0, 60.0)], 4)
    assert np.all(mask == 4)
