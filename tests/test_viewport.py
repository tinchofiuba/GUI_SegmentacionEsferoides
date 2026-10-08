import numpy as np

from gui_segesferoides import viewport

PALETTE = {2: (255, 255, 0)}


def test_zoom_at_keeps_anchor_fixed():
    offset, zoom, new_zoom, anchor = (100.0, 50.0), 0.5, 2.0, (300, 200)
    before = viewport.view_to_image(offset, zoom, *anchor)
    new_offset = viewport.zoom_at(offset, zoom, new_zoom, anchor)
    after = viewport.view_to_image(new_offset, new_zoom, *anchor)
    assert np.allclose(before, after)


def test_view_image_roundtrip():
    off, z = (12.5, -3.0), 3.0
    assert np.allclose(viewport.image_to_view(off, z, *viewport.view_to_image(off, z, 40, 70)), (40, 70))


def test_fit_centers_image():
    z = viewport.fit_zoom(4000, 2000, 800, 600)
    assert z == 0.2
    ox, oy = viewport.fit_offset(4000, 2000, 800, 600, z)
    assert viewport.image_to_view((ox, oy), z, 0, 0)[0] == 0
    assert viewport.image_to_view((ox, oy), z, 0, 0)[1] > 0  # franja arriba


def test_render_shapes_and_mask_colors():
    img = np.full((800, 1200, 3), 100, np.uint8)
    mask = np.zeros((800, 1200), np.uint8)
    mask[200:400, 300:600] = 2
    levels = viewport.build_pyramid(img)
    assert len(levels) == 4 and levels[1].shape[:2] == (400, 600)
    for zoom, off in [(0.5, (0.0, 0.0)), (0.1, (-100.0, -50.0)), (4.0, (290.0, 190.0)), (16.0, (-5.0, -5.0))]:
        out = viewport.render_view(levels, mask, PALETTE, off, zoom, 640, 480, alpha=1.0)
        assert out.shape == (480, 640, 3)
    out = viewport.render_view(levels, mask, PALETTE, (250.0, 150.0), 2.0, 640, 480, alpha=1.0)
    assert tuple(out[100, 100]) == (255, 255, 0)   # (300,200) -> esferoide
    assert tuple(out[5, 5]) == (100, 100, 100)     # fuera de la máscara
    hidden = viewport.render_view(levels, mask, PALETTE, (250.0, 150.0), 2.0, 640, 480, show_mask=False)
    assert tuple(hidden[100, 100]) == (100, 100, 100)


def test_render_fully_outside_returns_background():
    img = np.zeros((100, 100, 3), np.uint8)
    levels = viewport.build_pyramid(img)
    mask = np.zeros((100, 100), np.uint8)
    out = viewport.render_view(levels, mask, PALETTE, (500.0, 500.0), 1.0, 200, 200)
    assert np.all(out == viewport.BACKGROUND_GRAY)
