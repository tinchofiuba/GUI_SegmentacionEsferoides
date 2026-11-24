import os

import numpy as np
import pytest

from gui_segesferoides import semantic
from gui_segesferoides.backend import BackendError, OptionSpec
from gui_segesferoides.config import BackendSpec, create_backend
from gui_segesferoides.model_manifest import IMAGENET, load_manifest, manifest_path, parse_manifest

# Tests con modelos reales: opcionales. Apuntar a un .onnx (con su .yaml al lado) por variable de entorno;
# sin ella se saltan.
REAL = os.environ.get("GUI_TEST_UNET_ONNX", "")

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
    spec = BackendSpec(name="u", options={"models_glob": [REAL]})
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


YOLO = os.environ.get("GUI_TEST_YOLO_ONNX", "")
YOLO_BASE = {"schema": 1, "name": "y", "output": "instance-seg",
             "input": {"normalize": "unit", "pad_multiple": 32, "imgsz": 640},
             "decode": {"conf": 0.3, "iou": 0.6},
             "classes": [{"id": 1, "name": "a", "color": [0, 255, 0]}, {"id": 2, "name": "b", "color": [255, 0, 0]}]}


def test_parse_instance_seg():
    m = parse_manifest(YOLO_BASE)
    assert m.output == "instance-seg" and m.imgsz == 640 and (m.conf, m.iou) == (0.3, 0.6) and m.mean is None


@pytest.mark.parametrize("change", [
    {"input": {"normalize": "unit", "pad_multiple": 32}},                              # falta imgsz
    {"input": {"imgsz": 640, "tta": ["hflip"]}},                                       # tta no aplica
    {"input": {"imgsz": 0}}, {"decode": {"conf": 0}}, {"decode": {"conf": 1.5}}, {"decode": {"iou": 0}},
])
def test_parse_instance_seg_rejects(change):
    with pytest.raises(BackendError):
        parse_manifest({**YOLO_BASE, **change})


def test_yolo_adapter_maps_model_channel_to_manifest_class_order():
    """El canal i del modelo es classes[i]: la clase 1 del modelo se pinta con el id de classes[1]."""
    from gui_segesferoides import yolo_seg
    m = parse_manifest({**YOLO_BASE, "input": {"normalize": "unit", "pad_multiple": 32, "imgsz": 64},
                        "classes": [{"id": 5, "name": "a", "color": [0, 0, 0]},
                                    {"id": 9, "name": "b", "color": [0, 0, 0]}]})
    # 4 caja + 2 clases + 1 coeficiente; un objeto de la clase 1 del modelo en el centro del letterbox 64x64
    output0 = np.array([[[32.0], [32.0], [40.0], [40.0], [0.1], [0.9], [1.0]]], np.float32).transpose(0, 1, 2)
    protos = np.full((1, 1, 16, 16), 5.0, np.float32)
    seen = {}

    def run(x):
        seen["shape"] = x.shape
        return output0, protos
    mask = yolo_seg.infer(run, np.zeros((64, 64, 3), np.uint8), m)
    assert seen["shape"] == (1, 3, 64, 64) and mask[32, 32] == 9 and mask[0, 0] == 0 and set(np.unique(mask)) == {0, 9}


@pytest.mark.skipif(not os.path.exists(YOLO) or not os.path.exists(YOLO[:-5] + ".yaml"),
                    reason="no está el modelo YOLO real")
def test_real_yolo_model_with_manifest(tmp_path):
    import cv2
    backend = create_backend(BackendSpec(name="y", options={"models_glob": [YOLO]}))
    assert backend._manifest.output == "instance-seg"
    assert [(c.id, c.name) for c in backend.info.classes] == [(1, "Celula"), (2, "Esferoide"), (3, "Organoide")]
    img = np.zeros((300, 400, 3), np.uint8)
    path = str(tmp_path / "x.png")
    cv2.imwrite(path, img)
    mask = backend.predict(path, "10x")
    assert mask.shape == (300, 400) and mask.dtype == np.uint8
    assert backend.predict(path, "10x", bbox=(10, 10, 100, 80)).shape == (80, 100)


def test_instance_seg_manifest_with_wrong_model_is_rejected(tmp_path):
    """Un manifiesto instance-seg junto a un .onnx semántico: la prueba de humo lo rechaza al cargar."""
    import shutil
    src = REAL
    if not os.path.exists(src):
        pytest.skip("no está el modelo real")
    shutil.copy(src, tmp_path / "m.onnx")
    (tmp_path / "m.yaml").write_text(
        "schema: 1\noutput: instance-seg\ninput: {imgsz: 64, pad_multiple: 32}\n"
        "classes: [{id: 1, name: a, color: [0,255,0]}]\n")
    with pytest.raises(BackendError):
        create_backend(BackendSpec(name="y", options={"models_glob": [str(tmp_path / "m.onnx")]}))


def test_options_default_to_tta_on_and_take_custom_label():
    m = parse_manifest(with_(options=[{"id": "tta", "label": "Utilizar TTA"}]))
    assert m.options == (OptionSpec("tta", "checkbox", "Utilizar TTA", True),)


def test_no_options_key_means_no_controls():
    assert parse_manifest(BASE).options == ()


@pytest.mark.parametrize("bad", [
    with_(options=[{"id": "rot90"}]),                                     # id que el adaptador no conoce
    with_(options=[{"id": "tta"}, {"id": "tta"}]),                         # repetida
    with_(options=[{"label": "sin id"}]),
    with_(options=[{"id": "tta", "default": "si"}]),                       # checkbox sin bool
    with_(input={"normalize": "unit"}, options=[{"id": "tta"}]),           # tta sin vistas
    with_(options="tta"),
])
def test_options_reject(bad):
    with pytest.raises(BackendError):
        parse_manifest(bad)


IS = {"schema": 1, "output": "instance-seg", "input": {"imgsz": 640},
      "decode": {"conf": 0.3, "iou": 0.6}, "classes": [{"id": 1, "name": "a", "color": [0, 255, 0]}]}


def test_instance_seg_number_options_default_from_decode():
    m = parse_manifest({**IS, "options": [{"id": "conf"}, {"id": "iou", "label": "NMS"}]})
    assert m.options == (OptionSpec("conf", "number", "conf", 0.3, 0.01, 0.99),
                         OptionSpec("iou", "number", "NMS", 0.6, 0.01, 1.0))


def test_instance_seg_rejects_out_of_range_default():
    with pytest.raises(BackendError):
        parse_manifest({**IS, "options": [{"id": "conf", "default": 1.5}]})


@pytest.mark.skipif(not os.path.exists(REAL), reason="no está el modelo real")
def test_set_option_tta_changes_the_views_used_for_inference(tmp_path, monkeypatch):
    import shutil
    model = tmp_path / "m.onnx"                       # copia: el manifiesto real no declara controles
    shutil.copy(REAL, model)
    yaml_text = open(manifest_path(REAL), encoding="utf-8").read()
    (tmp_path / "m.yaml").write_text(yaml_text + "options:\n  - {id: tta, label: Utilizar TTA}\n", encoding="utf-8")
    backend = create_backend(BackendSpec(name="u", options={"models_glob": [str(model)]}))
    assert [o.id for o in backend.info.options] == ["tta"] and backend.info.options[0].default is True
    seen = []
    real_infer = semantic.infer

    def spy(run, rgb, manifest, *args, **kwargs):
        seen.append(manifest.tta)
        return real_infer(run, rgb, manifest, *args, **kwargs)
    monkeypatch.setattr(semantic, "infer", spy)
    import cv2
    img = np.zeros((70, 90, 3), np.uint8)
    cv2.circle(img, (45, 35), 25, (200, 200, 200), -1)
    path = str(tmp_path / "x.png")
    cv2.imwrite(path, img)
    backend.predict(path, "4x")
    backend.set_option("tta", False)
    backend.predict(path, "4x")
    backend.set_option("tta", True)
    backend.predict(path, "4x")
    assert seen == [("hflip", "vflip", "hvflip"), (), ("hflip", "vflip", "hvflip")]
    with pytest.raises(BackendError):
        backend.set_option("tta", "no")
    with pytest.raises(BackendError):
        backend.set_option("rot90", True)
