import os

import numpy as np
import pytest

from gui_segesferoides import semantic
from gui_segesferoides.backend import BackendError
from gui_segesferoides.config import BackendSpec, create_backend
from gui_segesferoides.model_manifest import IMAGENET, load_manifest, parse_manifest

REAL = "/home/martin/repos/SegmentacionEsferoides/outputs/models/insitu_3t3_2026-10-06/UResNet_1006.onnx"

BASE = {"schema": 1, "name": "m", "output": "semantic",
        "input": {"normalize": "imagenet", "pad_multiple": 32, "tta": ["hflip"]},
        "classes": [{"id": 1, "name": "a", "color": [0, 255, 0]}, {"id": 2, "name": "b", "color": [255, 0, 0]}],
        "magnifications": ["4x"]}


def with_(**changes):
    d = {**BASE, **changes}
    return {k: v for k, v in d.items() if v is not None}


def test_parse_ok():
    m = parse_manifest(BASE)
    assert m.mean == IMAGENET["mean"] and m.pad_multiple == 32 and m.tta == ("hflip",)
    assert [c.id for c in m.classes] == [1, 2] and m.magnifications == ["4x"]


@pytest.mark.parametrize("bad", [
    with_(schema=2), with_(schema=None), with_(output="instance-seg"),
    with_(input={"tta": ["rot90"]}), with_(input={"normalize": "raro"}), with_(input={"pad_multiple": 0}),
    with_(classes=[]), with_(classes=[{"id": 0, "name": "x", "color": [0, 0, 0]}]),
    with_(classes=[{"id": 1, "name": "x", "color": [0, 0, 0]}, {"id": 1, "name": "y", "color": [0, 0, 0]}]),
    with_(classes=[{"id": 1, "name": "x"}]),
])
def test_parse_rejects(bad):
    with pytest.raises(BackendError):
        parse_manifest(bad)


def test_load_manifest_absent_is_none(tmp_path):
    assert load_manifest(str(tmp_path / "m.onnx")) is None


def test_preprocess_pads_black_then_normalizes_and_infer_crops():
    m = parse_manifest(with_(input={"normalize": "unit", "pad_multiple": 32}))
    rgb = np.full((40, 50, 3), 255, np.uint8)
    t, h, w = semantic.preprocess(rgb, m)
    assert t.shape == (1, 3, 64, 64) and (h, w) == (40, 50)
    assert t[0, 0, 0, 0] == 1.0 and t[0, 0, 63, 63] == 0.0          # relleno negro abajo/derecha
    mask = semantic.infer(lambda x: np.stack([np.zeros(x.shape[2:]), np.ones(x.shape[2:]) * 5,
                                              np.zeros(x.shape[2:])])[None].astype(np.float32), rgb, m)
    assert mask.shape == (40, 50) and mask.dtype == np.uint8 and (mask == 1).all()


def test_tta_averages_flipped_views():
    m = parse_manifest(with_(input={"normalize": "unit", "tta": ["hflip"]}))
    rgb = np.zeros((4, 4, 3), np.uint8)
    calls = []

    def run(x):                       # canal 1 gana siempre en la 1.ª vista; canal 2 en la 2.ª con más fuerza
        calls.append(x)
        out = np.zeros((1, 3) + x.shape[2:], np.float32)
        out[0, 1 if len(calls) == 1 else 2] = 1 if len(calls) == 1 else 3
        return out
    assert len(calls) == 0 and (semantic.infer(run, rgb, m) == 2).all() and len(calls) == 2


@pytest.mark.skipif(not os.path.exists(REAL), reason="no está el modelo real")
def test_real_u_resnet_model_with_manifest(tmp_path):
    spec = BackendSpec(name="u", type="onnx", options={"models_glob": [REAL]})
    backend = create_backend(spec)
    assert [c.id for c in backend.info.classes] == [1, 2, 3] and backend.info.magnifications == ["4x", "10x"]
    import cv2
    img = np.zeros((70, 90, 3), np.uint8)
    cv2.circle(img, (45, 35), 25, (200, 200, 200), -1)
    path = str(tmp_path / "x.png")
    cv2.imwrite(path, img)
    mask = backend.predict(path, "4x")
    assert mask.shape == (70, 90) and set(np.unique(mask)) <= {0, 1, 2, 3}
    assert backend.predict(path, "4x", bbox=(10, 10, 40, 30)).shape == (30, 40)


def test_tiled_matches_whole_for_a_local_model():
    """Con un modelo sin contexto (cada píxel depende solo de sí mismo) los mosaicos dan
    exactamente la misma máscara que la imagen entera, incluso con tamaños que no encajan."""
    m = parse_manifest(with_(input={"normalize": "unit", "pad_multiple": 32, "tta": ["hflip"]}))
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (437, 613, 3), dtype=np.uint8)

    def run(x):   # logits [1, 3, H, W]: el canal ganador depende del valor del píxel
        v = x[:, 0]
        return np.stack([np.zeros_like(v), v - 0.5, 0.5 - v], axis=1).astype(np.float32)
    whole = semantic.infer(run, rgb, m, tile=None)
    for tile, overlap in [(128, 16), (200, 0), (100, 30)]:
        assert (semantic.infer(run, rgb, m, tile, overlap) == whole).all()


def test_tiled_rejects_overlap_without_core():
    m = parse_manifest(BASE)
    with pytest.raises(ValueError):
        semantic.infer(lambda x: x, np.zeros((300, 300, 3), np.uint8), m, tile=100, overlap=50)


def test_parallel_tiles_equal_sequential():
    m = parse_manifest(with_(input={"normalize": "unit", "pad_multiple": 32, "tta": ["vflip"]}))
    rgb = np.random.default_rng(1).integers(0, 256, (300, 410, 3), dtype=np.uint8)

    def run(x):
        v = x[:, 1]
        return np.stack([np.zeros_like(v), v - 0.4, 0.6 - v], axis=1).astype(np.float32)
    assert (semantic.infer(run, rgb, m, 128, 16, workers=4) == semantic.infer(run, rgb, m, 128, 16, workers=1)).all()


def test_parallel_tile_error_propagates():
    m = parse_manifest(BASE)
    def boom(x):
        raise RuntimeError("falló")
    with pytest.raises(RuntimeError):
        semantic.infer(boom, np.zeros((300, 300, 3), np.uint8), m, 128, 16, workers=3)
