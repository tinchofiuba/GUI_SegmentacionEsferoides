import cv2
import numpy as np

from gui_segesferoides.editing import (
    count_objects,
    UndoStack,
    fill_holes,
    paint_circle,
    paint_segment,
    replace_object,
    save_mask,
)


def _mask_with_two_blobs():
    m = np.zeros((40, 40), np.uint8)
    m[2:10, 2:10] = 2
    m[20:30, 20:30] = 2
    return m


def test_replace_object_only_touches_clicked_blob():
    m = _mask_with_two_blobs()
    assert replace_object(m, 5, 5, 3)
    assert np.all(m[2:10, 2:10] == 3)
    assert np.all(m[20:30, 20:30] == 2)


def test_replace_object_delete_and_background_noop():
    m = _mask_with_two_blobs()
    assert replace_object(m, 25, 25, 0)
    assert m[20:30, 20:30].sum() == 0
    assert not replace_object(m, 15, 15, 1)  # fondo: nada que reasignar


def test_paint_and_erase():
    m = np.zeros((20, 20), np.uint8)
    paint_circle(m, 10, 10, 3, 1)
    assert m[10, 10] == 1
    paint_circle(m, 10, 10, 3, 0)
    assert m.sum() == 0


def test_fill_holes():
    m = np.zeros((20, 20), np.uint8)
    m[2:18, 2:18] = 2
    m[8:12, 8:12] = 0
    fill_holes(m, 2)
    assert np.all(m[2:18, 2:18] == 2)
    assert m[0, 0] == 0


PALETTE = {1: (0, 255, 0), 2: (255, 255, 0), 5: (0, 255, 255)}


def test_save_mask_color_roundtrip(tmp_path):
    m = np.zeros((10, 10), np.uint8)
    m[0:3, 0:3] = 1
    m[4:7, 4:7] = 2
    m[8:10, 8:10] = 5
    path = str(tmp_path / "x.png")
    save_mask(path, m, PALETTE)
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert tuple(rgb[1, 1]) == (0, 255, 0)
    assert tuple(rgb[5, 5]) == (255, 255, 0)
    assert tuple(rgb[9, 9]) == (0, 255, 255)
    assert tuple(rgb[3, 3]) == (0, 0, 0)  # fondo negro


def test_save_mask_ids(tmp_path):
    m = np.zeros((6, 6), np.uint8)
    m[1:3, 1:3] = 5
    path = str(tmp_path / "ids.png")
    save_mask(path, m, PALETTE, mode="ids")
    assert np.array_equal(cv2.imread(path, cv2.IMREAD_UNCHANGED), m)


def test_paint_segment_has_no_gaps_and_cuts():
    m = np.zeros((40, 100), np.uint8)
    m[10:30, 10:90] = 2
    paint_segment(m, 50, 0, 50, 39, 1, 0)  # corte vertical con borrador fino
    n, _ = cv2.connectedComponents((m == 2).astype(np.uint8), connectivity=4)
    assert n - 1 == 2


def test_undo_redo():
    s = UndoStack()
    m = np.zeros((4, 4), np.uint8)
    s.push(m)
    m2 = m.copy()
    m2[0, 0] = 1
    assert np.array_equal(s.undo(m2), m)
    assert np.array_equal(s.redo(m), m2)
    assert s.undo(np.zeros((4, 4), np.uint8)) is not None


def test_fill_holes_ignores_background_touching_border():
    m = np.zeros((12, 12), np.uint8)
    m[0:6, 0:6] = 2          # toca la esquina: el fondo de afuera no es hueco
    m[2:4, 2:4] = 0          # hueco real
    fill_holes(m, 2)
    assert np.all(m[2:4, 2:4] == 2) and m[8, 8] == 0 and m[0, 8] == 0


def test_object_region_and_stats_of_square():
    from gui_segesferoides.editing import object_region, object_stats

    m = _mask_with_two_blobs()
    region = object_region(m, 5, 5)
    assert region.sum() == 64 and region[5, 5] and not region[25, 25]
    assert object_region(m, 15, 15) is None and object_region(m, -1, 0) is None
    stats = object_stats(region)
    assert stats["area"] == 64 and stats["perimeter"] == 28.0
    assert abs(stats["equivalent_diameter"] - 9.026) < 0.01
    assert 0.8 < stats["circularity"] <= 1.0


def test_fill_object_holes_only_fills_that_object():
    from gui_segesferoides.editing import fill_object_holes, object_region

    m = _mask_with_two_blobs()
    m[4:6, 4:6] = 0
    m[24:26, 24:26] = 0
    region = fill_object_holes(m, object_region(m, 3, 3))
    assert np.all(m[2:10, 2:10] == 2) and region.sum() == 64
    assert np.all(m[24:26, 24:26] == 0)   # el otro objeto no se toca


def test_object_contours_include_holes_but_perimeter_is_external_only():
    from gui_segesferoides.editing import object_contours, object_region, object_stats

    m = np.zeros((30, 30), np.uint8)
    m[2:20, 2:20] = 2
    m[8:12, 8:12] = 0
    region = object_region(m, 3, 3)
    assert len(object_contours(region)) == 1
    assert len(object_contours(region, holes=True)) == 2
    assert object_stats(region)["perimeter"] == 68.0


def test_count_objects_counts_components_per_class_not_pixels():
    mask = np.zeros((40, 40), np.uint8)
    mask[2:12, 2:12] = 1            # una célula de 100 px
    mask[20:23, 2:5] = 1            # otra, de 9 px: cuenta igual
    mask[2:30, 20:38] = 2           # un esferoide enorme: sigue siendo 1
    assert count_objects(mask) == {1: 2, 2: 1}


def test_count_objects_uses_4_connectivity_like_the_rest_of_the_gui():
    mask = np.zeros((10, 10), np.uint8)
    mask[1, 1] = 3
    mask[2, 2] = 3                  # se tocan solo en diagonal: son dos objetos (igual que en el CSV)
    assert count_objects(mask) == {3: 2}
    from gui_segesferoides.editing import measure_objects
    assert len(measure_objects(mask)) == 2


def test_count_objects_of_an_empty_mask():
    assert count_objects(np.zeros((8, 8), np.uint8)) == {}
