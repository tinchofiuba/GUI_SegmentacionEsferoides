import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication

from gui_segesferoides.image_canvas import (
    TOOL_ANALYZE,
    TOOL_BRUSH,
    TOOL_DELETE,
    TOOL_ERASER,
    TOOL_RECLASS,
    TOOL_ZONE,
    ImageCanvas,
)
from gui_segesferoides.viewport import build_pyramid


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(qapp):
    """Canvas de 300x300 con una imagen de 100x100 en la esquina superior izquierda:
    la vista (0..99) es imagen y todo lo demás es fondo negro."""
    c = ImageCanvas()
    c.resize(300, 300)
    c.set_image(build_pyramid(np.zeros((100, 100, 3), np.uint8)), np.zeros((100, 100), np.uint8))
    c.zoom, c.offset = 1.0, (0.0, 0.0)
    return c


INSIDE, OUTSIDE = (50, 50), (200, 200)


def _mouse(canvas, kind, pos, button=Qt.NoButton):
    event = QMouseEvent(kind, QPointF(*pos), button, button, Qt.NoModifier)
    {QEvent.MouseMove: canvas.mouseMoveEvent, QEvent.MouseButtonPress: canvas.mousePressEvent,
     QEvent.MouseButtonRelease: canvas.mouseReleaseEvent}[kind](event)


def _move(canvas, pos):
    _mouse(canvas, QEvent.MouseMove, pos)


@pytest.mark.parametrize("tool", [TOOL_BRUSH, TOOL_ERASER])
def test_stroke_tools_hide_the_cursor_only_over_the_image(canvas, tool):
    canvas.set_tool(tool)
    _move(canvas, INSIDE)
    assert canvas.cursor().shape() == Qt.BlankCursor
    _move(canvas, OUTSIDE)                       # sobre el fondo negro: se ve el cursor normal
    assert canvas.cursor().shape() == Qt.ArrowCursor
    _move(canvas, INSIDE)
    assert canvas.cursor().shape() == Qt.BlankCursor


def test_brush_circle_is_not_drawn_over_the_background(canvas):
    canvas.set_tool(TOOL_BRUSH)
    _move(canvas, INSIDE)
    assert canvas._over_image
    _move(canvas, OUTSIDE)
    assert not canvas._over_image
    canvas.grab()                                # pinta sin círculo ni errores


def test_leaving_the_canvas_restores_the_normal_cursor(canvas):
    canvas.set_tool(TOOL_BRUSH)
    _move(canvas, INSIDE)
    canvas.leaveEvent(None)
    assert not canvas._over_image
    canvas.set_tool(TOOL_BRUSH)
    assert canvas.cursor().shape() == Qt.ArrowCursor


@pytest.mark.parametrize("tool, shape", [
    (TOOL_DELETE, Qt.CrossCursor), (TOOL_RECLASS, Qt.CrossCursor),
    (TOOL_ANALYZE, Qt.CrossCursor), (TOOL_ZONE, Qt.ArrowCursor),
])
def test_other_tools_keep_their_cursor_everywhere(canvas, tool, shape):
    canvas.set_tool(tool)
    for pos in (INSIDE, OUTSIDE):
        _move(canvas, pos)
        assert canvas.cursor().shape() == shape


def test_panning_keeps_the_closed_hand_while_crossing_the_image_edge(canvas):
    canvas.set_tool(TOOL_BRUSH)
    _move(canvas, INSIDE)
    _mouse(canvas, QEvent.MouseButtonPress, INSIDE, Qt.RightButton)
    assert canvas.cursor().shape() == Qt.ClosedHandCursor
    _move(canvas, OUTSIDE)                       # la vista se mueve; el cursor sigue siendo la mano
    assert canvas.cursor().shape() == Qt.ClosedHandCursor
    _mouse(canvas, QEvent.MouseButtonRelease, OUTSIDE, Qt.RightButton)
    assert canvas.cursor().shape() == Qt.ArrowCursor   # terminó el paneo estando sobre el fondo


# ---------- Nombre de la imagen ----------

def test_image_name_label_sits_bottom_right_and_ignores_the_mouse(canvas):
    assert canvas.name_label.isHidden()
    canvas.set_image_name("cel_1.png")
    label = canvas.name_label
    assert not label.isHidden() and label.text() == "cel_1.png"
    assert label.testAttribute(Qt.WA_TransparentForMouseEvents)       # no tapa clicks ni tooltips
    assert label.geometry().right() < canvas.width() and label.geometry().bottom() < canvas.height()
    assert label.x() > canvas.width() // 2 and label.y() > canvas.height() // 2


def test_image_name_follows_the_corner_when_resizing(canvas):
    canvas.set_image_name("cel_1.png")
    canvas.resize(500, 400)
    label = canvas.name_label
    assert label.geometry().right() > 450 and label.geometry().bottom() > 350


def test_long_image_name_is_elided_in_the_middle_keeping_the_extension(canvas):
    name = "una imagen con un nombre larguisimo " * 4 + "final_10x.png"
    canvas.set_image_name(name)
    text = canvas.name_label.text()
    assert "…" in text and text.endswith("10x.png") and len(text) < len(name)
    assert canvas.name_label.geometry().left() >= 0


def test_empty_image_name_hides_the_label(canvas):
    canvas.set_image_name("cel_1.png")
    canvas.set_image_name("")
    assert canvas.name_label.isHidden()


def test_image_name_is_a_label_not_part_of_the_image_data(canvas):
    """El nombre no se dibuja en la imagen ni en la máscara: lo que se manda al
    modelo y lo que se exporta sale de esos arrays, no de la pantalla."""
    before_img, before_mask = canvas.levels[0].copy(), canvas.mask.copy()
    canvas.set_image_name("cel_1.png")
    canvas.grab()
    assert np.array_equal(canvas.levels[0], before_img) and np.array_equal(canvas.mask, before_mask)


# ---------- Barra de escala ----------

def test_no_scale_bar_without_an_image(qapp):
    assert ImageCanvas().scale_bar() is None


def test_scale_bar_in_pixels_when_there_is_no_calibration(canvas):
    assert canvas.scale_bar() == (100, "100 px")


@pytest.mark.parametrize("factor, unit, expected", [
    (2.0, "µm", "100 px = 200 µm"),
    (0.002, "mm", "100 px = 0.2 mm"),
    (20.0, "µm", "100 px = 2,000 µm"),
    (0.5, "nm", "100 px = 50 nm"),
])
def test_scale_bar_says_what_distance_the_line_is(canvas, factor, unit, expected):
    from gui_segesferoides.calibration import Scale
    canvas.set_scale(Scale(factor, unit))
    assert canvas.scale_bar() == (100, expected)


def test_scale_bar_changes_value_with_the_zoom_and_stays_correct(canvas):
    from gui_segesferoides.calibration import Scale
    canvas.set_scale(Scale(2.0, "µm"))
    canvas.zoom = 0.3                                    # alejado: la barra pasa a 500 px
    assert canvas.scale_bar() == (500, "500 px = 1,000 µm")
    canvas.zoom = 4.0
    assert canvas.scale_bar() == (20, "20 px = 40 µm")


def test_scale_bar_is_only_drawn_on_screen_not_in_the_image_data(canvas):
    from gui_segesferoides.calibration import Scale
    before_img, before_mask = canvas.levels[0].copy(), canvas.mask.copy()
    canvas.set_scale(Scale(2.0, "µm"))
    pixmap = canvas.grab()                               # se pinta con la barra
    assert not pixmap.isNull()
    assert np.array_equal(canvas.levels[0], before_img) and np.array_equal(canvas.mask, before_mask)


# ---------- Medidas de distancia ----------

from gui_segesferoides.image_canvas import TOOL_EDIT_MEASURE, TOOL_MEASURE  # noqa: E402
from gui_segesferoides.measures import Measure  # noqa: E402

RED, GREEN, BLUE = (255, 0, 0), (0, 200, 0), (0, 90, 255)


def _press(canvas, pos):
    _mouse(canvas, QEvent.MouseButtonPress, pos, Qt.LeftButton)


def _release(canvas, pos):
    _mouse(canvas, QEvent.MouseButtonRelease, pos, Qt.LeftButton)


def _drag(canvas, start, end):
    _press(canvas, start)
    _move(canvas, end)
    _release(canvas, end)


@pytest.fixture
def changes(canvas):
    events = []
    canvas.measures_changed.connect(lambda: events.append(1))
    return events


def test_measure_tool_draws_a_line_from_press_to_release(canvas, changes):
    canvas.set_tool(TOOL_MEASURE)
    _drag(canvas, (10, 20), (60, 20))
    assert canvas.measures == [Measure((10.0, 20.0), (60.0, 20.0), RED)]
    assert canvas.measures[0].length == pytest.approx(50.0) and len(changes) == 1


def test_a_click_without_dragging_leaves_no_measure(canvas, changes):
    canvas.set_tool(TOOL_MEASURE)
    _press(canvas, (30, 30))
    _release(canvas, (30, 30))
    assert canvas.measures == [] and changes == []


def test_each_measure_keeps_the_color_it_was_drawn_with(canvas):
    canvas.set_tool(TOOL_MEASURE)
    _drag(canvas, (10, 10), (60, 10))
    canvas.measure_color = GREEN                      # el color elegido cambia solo la próxima
    _drag(canvas, (10, 30), (60, 30))
    canvas.measure_color = BLUE
    _drag(canvas, (10, 50), (60, 50))
    assert [m.color for m in canvas.measures] == [RED, GREEN, BLUE]


def test_measure_endpoints_are_clamped_to_the_image(canvas):
    canvas.set_tool(TOOL_MEASURE)
    _drag(canvas, (10, 20), (250, 250))               # suelta sobre el fondo negro
    m = canvas.measures[0]
    assert m.p2 == (99.0, 99.0) and m.p1 == (10.0, 20.0)


def test_measure_preview_follows_the_mouse_and_paints(canvas):
    canvas.set_tool(TOOL_MEASURE)
    _press(canvas, (10, 20))
    _move(canvas, (70, 20))
    assert canvas._measure_end == (70.0, 20.0) and canvas.measures == []
    assert not canvas.grab().isNull()


def test_measure_tool_uses_a_cross_cursor_everywhere(canvas):
    canvas.set_tool(TOOL_MEASURE)
    for pos in (INSIDE, OUTSIDE):
        _move(canvas, pos)
        assert canvas.cursor().shape() == Qt.CrossCursor


@pytest.fixture
def with_measure(canvas):
    canvas.measures.append(Measure((20.0, 40.0), (80.0, 40.0), RED))
    canvas.set_tool(TOOL_EDIT_MEASURE)
    return canvas


def test_measure_at_finds_the_line_within_the_tolerance(with_measure):
    assert with_measure.measure_at(50, 40) == 0       # sobre la línea
    assert with_measure.measure_at(50, 46) == 0       # a 6 px de pantalla: todavía cuenta
    assert with_measure.measure_at(50, 60) is None    # lejos
    assert with_measure.measure_at(95, 40) is None    # más allá del extremo


def test_clicking_a_measure_with_the_edit_tool_asks_what_to_do(with_measure):
    clicks = []
    with_measure.measure_clicked.connect(lambda i, pos: clicks.append(i))
    _press(with_measure, (50, 41))
    assert clicks == [0] and with_measure.selected_measure == 0
    _press(with_measure, (50, 90))                    # sobre nada
    assert clicks == [0]


def test_edit_tool_does_nothing_on_a_click_over_empty_space(with_measure):
    _press(with_measure, (50, 90))
    assert with_measure.selected_measure is None and with_measure.measures[0].color == RED


def test_delete_measure(with_measure, changes):
    with_measure.delete_measure(0)
    assert with_measure.measures == [] and with_measure.selected_measure is None and len(changes) == 1


def test_changing_a_measure_color_keeps_its_geometry(with_measure, changes):
    before = with_measure.measures[0]
    with_measure.set_measure_color(0, BLUE)
    after = with_measure.measures[0]
    assert after.color == BLUE and (after.p1, after.p2) == (before.p1, before.p2) and len(changes) == 1


def test_move_drags_the_measure_keeping_length_orientation_and_color(with_measure, changes):
    before = with_measure.measures[0]
    with_measure.begin_move(0)
    assert with_measure.cursor().shape() == Qt.SizeAllCursor
    _press(with_measure, (50, 41))
    _move(with_measure, (60, 51))
    _release(with_measure, (60, 51))
    after = with_measure.measures[0]
    assert (after.p1, after.p2) == ((30.0, 50.0), (90.0, 50.0))           # corrida (+10, +10)
    assert after.length == pytest.approx(before.length) and after.color == before.color
    assert after.p2[1] - after.p1[1] == before.p2[1] - before.p1[1]       # misma orientación
    assert len(changes) == 1
    assert with_measure.cursor().shape() == Qt.CrossCursor                # terminó el modo mover


def test_a_diagonal_measure_keeps_its_orientation_when_moved(canvas):
    canvas.measures.append(Measure((20.0, 20.0), (50.0, 60.0), GREEN))    # 3-4-5
    canvas.set_tool(TOOL_EDIT_MEASURE)
    canvas.begin_move(0)
    _press(canvas, (35, 40))
    _move(canvas, (45, 30))
    _release(canvas, (45, 30))
    m = canvas.measures[0]
    assert (m.p2[0] - m.p1[0], m.p2[1] - m.p1[1]) == (30.0, 40.0) and m.length == pytest.approx(50.0)


def test_moving_stops_at_the_image_edge_without_changing_the_length(with_measure):
    length = with_measure.measures[0].length
    with_measure.begin_move(0)
    _press(with_measure, (50, 41))
    _move(with_measure, (299, 299))                   # lejos, fuera de la imagen
    _release(with_measure, (299, 299))
    m = with_measure.measures[0]
    assert max(m.p1[0], m.p2[0]) <= 99 and max(m.p1[1], m.p2[1]) <= 99
    assert m.length == pytest.approx(length)


def test_after_moving_the_next_drag_does_not_move_it_again(with_measure):
    with_measure.begin_move(0)
    _drag(with_measure, (50, 41), (60, 41))
    moved = with_measure.measures[0]
    _drag(with_measure, (50, 41), (70, 70))           # sin volver a elegir «Mover»
    assert with_measure.measures[0] == moved


def test_changing_tool_cancels_a_pending_move(with_measure):
    with_measure.begin_move(0)
    with_measure.set_tool(TOOL_MEASURE)
    assert with_measure._move_index is None and with_measure.selected_measure is None


def test_opening_another_image_clears_the_measures(with_measure):
    with_measure.set_image(build_pyramid(np.zeros((50, 50, 3), np.uint8)), np.zeros((50, 50), np.uint8))
    assert with_measure.measures == []


def test_measures_are_screen_annotations_not_image_data(with_measure):
    before_img, before_mask = with_measure.levels[0].copy(), with_measure.mask.copy()
    assert not with_measure.grab().isNull()
    assert np.array_equal(with_measure.levels[0], before_img) and np.array_equal(with_measure.mask, before_mask)


def test_delete_removes_exactly_the_chosen_measure_among_several(canvas):
    a = Measure((10.0, 10.0), (60.0, 10.0), RED)
    b = Measure((10.0, 30.0), (60.0, 30.0), GREEN)
    c = Measure((10.0, 50.0), (60.0, 50.0), BLUE)
    canvas.measures[:] = [a, b, c]
    canvas.delete_measure(1)
    assert canvas.measures == [a, c]
    canvas.delete_measure(0)
    assert canvas.measures == [c]


def test_measure_at_picks_the_nearest_of_several(canvas):
    canvas.measures[:] = [Measure((10.0, 30.0), (80.0, 30.0), RED), Measure((10.0, 36.0), (80.0, 36.0), GREEN)]
    assert canvas.measure_at(40, 31) == 0 and canvas.measure_at(40, 35) == 1


# ---------- Mover la etiqueta o la línea ----------

def _label_center(canvas, index=0):
    c = canvas._label_rect_view(index).center()
    return (c.x(), c.y())


def test_move_can_drag_the_label_without_moving_the_line(with_measure, changes):
    before = with_measure.measures[0]
    with_measure.begin_move(0)
    start = _label_center(with_measure)
    _press(with_measure, start)                              # empieza sobre la etiqueta
    _move(with_measure, (start[0] + 15, start[1] + 25))
    _release(with_measure, (start[0] + 15, start[1] + 25))
    after = with_measure.measures[0]
    assert (after.p1, after.p2, after.color) == (before.p1, before.p2, before.color)   # la línea, quieta
    assert after.label_offset == pytest.approx((15.0, 25.0))
    assert len(changes) == 1 and with_measure._move_index is None


def test_move_still_drags_the_whole_measure_when_pressing_away_from_the_label(with_measure):
    before = with_measure.measures[0]
    with_measure.begin_move(0)
    _press(with_measure, (50, 41))                           # sobre la línea, no sobre la etiqueta
    _move(with_measure, (60, 51))
    _release(with_measure, (60, 51))
    after = with_measure.measures[0]
    assert (after.p1, after.p2) == ((before.p1[0] + 10, before.p1[1] + 10), (before.p2[0] + 10, before.p2[1] + 10))
    assert after.label_offset == before.label_offset == (0.0, 0.0)


def test_the_label_follows_the_line_when_the_whole_measure_moves(with_measure):
    with_measure.begin_move(0)
    start = _label_center(with_measure)
    _drag(with_measure, start, (start[0] + 10, start[1] + 30))          # primero, la etiqueta
    with_measure.begin_move(0)
    shifted = with_measure.measures[0].label_offset
    _press(with_measure, (50, 41))
    _move(with_measure, (60, 41))                                       # después, toda la medida
    _release(with_measure, (60, 41))
    assert with_measure.measures[0].label_offset == shifted             # conserva dónde dejó la etiqueta


def test_a_dragged_label_stays_inside_the_image(with_measure):
    with_measure.begin_move(0)
    start = _label_center(with_measure)
    _press(with_measure, start)
    _move(with_measure, (299, 299))
    _release(with_measure, (299, 299))
    m = with_measure.measures[0]
    mid = ((m.p1[0] + m.p2[0]) / 2, (m.p1[1] + m.p2[1]) / 2)
    assert mid[0] + m.label_offset[0] <= 99 and mid[1] + m.label_offset[1] <= 99
    assert (m.p1, m.p2) == ((20.0, 40.0), (80.0, 40.0))


def test_the_label_hit_area_follows_the_moved_label(with_measure):
    with_measure.begin_move(0)
    start = _label_center(with_measure)
    _drag(with_measure, start, (start[0] + 20, start[1] + 40))
    new_center = _label_center(with_measure)
    assert new_center == pytest.approx((start[0] + 20, start[1] + 40))
    with_measure.begin_move(0)
    _press(with_measure, new_center)                                    # ahora la etiqueta está acá
    assert with_measure._move_part == "label"
    _release(with_measure, new_center)
    with_measure.begin_move(0)
    _press(with_measure, start)                                         # y acá ya no hay etiqueta
    assert with_measure._move_part == "line"
    _release(with_measure, start)


def test_painting_with_a_moved_label_does_not_touch_the_image_data(with_measure):
    with_measure.measures[0] = with_measure.measures[0].with_label_offset((10.0, 30.0))
    before = with_measure.levels[0].copy()
    assert not with_measure.grab().isNull()
    assert np.array_equal(with_measure.levels[0], before)


def test_the_label_is_painted_on_screen_where_it_was_moved(qapp):
    c = ImageCanvas()
    c.resize(300, 300)
    c.set_image(build_pyramid(np.full((200, 200, 3), 235, np.uint8)), np.zeros((200, 200), np.uint8))
    c._fit_pending = False                                  # que grab() no reajuste el zoom: se mide a 1:1
    c.zoom, c.offset = 1.0, (0.0, 0.0)
    c.measures.append(Measure((20.0, 100.0), (120.0, 100.0), RED))
    home = c._label_rect_view(0).center()
    c.measures[0] = c.measures[0].with_label_offset((60.0, 70.0))
    moved = c._label_rect_view(0).center()
    image = c.grab().toImage()
    dark = lambda p: image.pixelColor(int(p.x()), int(p.y())).lightness() < 150   # la caja oscura de la etiqueta
    assert dark(moved) and not dark(home)


# ---------- Lazo ----------

from PyQt5.QtGui import QKeyEvent  # noqa: E402

from gui_segesferoides.image_canvas import TOOL_LASSO  # noqa: E402


@pytest.fixture
def lasso(canvas):
    canvas.set_tool(TOOL_LASSO)
    got = []
    canvas.lasso_finished.connect(lambda poly: got.append(poly))
    canvas.finished = got
    return canvas


def _click(canvas, pos):
    _press(canvas, pos)
    _release(canvas, pos)


def _key(canvas, key):
    canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))


TRIANGLE = [(20, 20), (80, 20), (50, 80)]


def test_clicks_add_points_and_clicking_the_first_one_closes_the_lasso(lasso):
    for p in TRIANGLE:
        _click(lasso, p)
    assert lasso._lasso == [(20.0, 20.0), (80.0, 20.0), (50.0, 80.0)] and lasso.finished == []
    _click(lasso, (22, 21))                               # cerca del primero (a menos de 12 px)
    assert len(lasso.finished) == 1 and lasso._lasso == []


def test_the_closed_lasso_is_a_smooth_curve_through_the_marked_points(lasso):
    for p in TRIANGLE:
        _click(lasso, p)
    _click(lasso, (20, 20))
    polygon = lasso.finished[0]
    assert polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) > 3 * 4     # muchos puntos: es una curva
    for p in TRIANGLE:
        assert np.min(np.hypot(*(polygon - p).T)) < 1e-6


def test_clicking_near_the_first_point_with_fewer_than_three_just_adds_a_point(lasso):
    _click(lasso, (20, 20))
    _click(lasso, (80, 20))
    _click(lasso, (21, 21))                               # cerca del primero, pero aún no hay 3 puntos
    assert lasso.finished == [] and len(lasso._lasso) == 3


def test_double_click_closes_the_lasso(lasso):
    for p in TRIANGLE:
        _click(lasso, p)
    lasso.mouseDoubleClickEvent(QMouseEvent(QEvent.MouseButtonDblClick, QPointF(50, 50), Qt.LeftButton,
                                            Qt.LeftButton, Qt.NoModifier))
    assert len(lasso.finished) == 1 and lasso._lasso == []


def test_enter_closes_and_escape_cancels(lasso):
    for p in TRIANGLE:
        _click(lasso, p)
    _key(lasso, Qt.Key_Escape)
    assert lasso.finished == [] and lasso._lasso == []
    for p in TRIANGLE:
        _click(lasso, p)
    _key(lasso, Qt.Key_Return)
    assert len(lasso.finished) == 1 and lasso._lasso == []


def test_a_lasso_with_fewer_than_three_points_leaves_nothing(lasso):
    _click(lasso, (20, 20))
    _click(lasso, (80, 20))
    _key(lasso, Qt.Key_Return)
    assert lasso.finished == [] and lasso._lasso == []


def test_a_microscopic_lasso_is_discarded(lasso):
    for p in [(20, 20), (21, 20), (20, 21)]:              # área de 0,5 px²
        _click(lasso, p)
    _key(lasso, Qt.Key_Return)
    assert lasso.finished == []


def test_dragging_freehand_adds_points_and_returning_to_the_start_closes(lasso):
    _press(lasso, (50, 20))
    for pos in [(70, 25), (85, 45), (80, 70), (55, 85), (30, 75), (20, 50), (30, 28), (49, 21)]:
        _move(lasso, pos)
    assert len(lasso._lasso) >= 5                         # sumó puntos mientras arrastraba
    _release(lasso, (49, 21))                             # y soltó sobre el punto de partida
    assert len(lasso.finished) == 1 and lasso._lasso == []


def test_a_freehand_drag_that_does_not_return_keeps_the_lasso_open(lasso):
    _press(lasso, (20, 20))
    for pos in [(40, 25), (60, 40), (80, 70)]:
        _move(lasso, pos)
    _release(lasso, (80, 70))
    assert lasso.finished == [] and len(lasso._lasso) >= 3


def test_lasso_points_are_clamped_to_the_image(lasso):
    _click(lasso, (250, 250))                             # sobre el fondo negro
    assert lasso._lasso == [(99.0, 99.0)]


def test_changing_the_tool_cancels_the_lasso(lasso):
    for p in TRIANGLE:
        _click(lasso, p)
    lasso.set_tool(TOOL_BRUSH)
    assert lasso._lasso == [] and lasso.finished == []


def test_opening_another_image_cancels_the_lasso(lasso):
    _click(lasso, (20, 20))
    lasso.set_image(build_pyramid(np.zeros((50, 50, 3), np.uint8)), np.zeros((50, 50), np.uint8))
    assert lasso._lasso == []


def test_lasso_preview_paints_without_touching_the_image_data(lasso):
    before_img, before_mask = lasso.levels[0].copy(), lasso.mask.copy()
    for p in TRIANGLE:
        _click(lasso, p)
    _move(lasso, (60, 60))                                # el punto provisorio de la vista previa
    assert not lasso.grab().isNull()
    assert np.array_equal(lasso.levels[0], before_img) and np.array_equal(lasso.mask, before_mask)


def test_lasso_uses_a_cross_cursor(lasso):
    for pos in (INSIDE, OUTSIDE):
        _move(lasso, pos)
        assert lasso.cursor().shape() == Qt.CrossCursor
