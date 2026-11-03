import numpy as np
import pytest

from gui_segesferoides.backend import BackendError
from gui_segesferoides.config import BackendSpec, create_backend, load_backends
from gui_segesferoides.onnx_backend import decode, letterbox, paint_instances


def test_letterbox_scales_longest_side_and_pads_to_stride():
    out, r, (left, top) = letterbox(np.zeros((100, 200, 3), np.uint8), 640, 32)
    assert r == 3.2 and out.shape == (320, 640, 3) and (left, top) == (0, 0)
    out, r, (left, top) = letterbox(np.zeros((100, 130, 3), np.uint8), 64, 32)   # 49x64 -> pad 15 en alto
    assert out.shape == (64, 64, 3) and top == 7 and left == 0


def test_decode_thresholds_and_nms_per_class():
    # 4 caja + 2 clases + 1 coeficiente; 3 anclas: dos casi iguales (clase 0) y una débil
    anchors = np.array([[50, 50, 20, 20, 0.9, 0.1, 1.0],
                        [52, 50, 20, 20, 0.8, 0.1, 1.0],
                        [200, 10, 20, 20, 0.1, 0.1, 1.0]], np.float32)
    out = anchors.T
    boxes, conf, cls, coefs = decode(out, 2, 0.25, 0.7)
    assert len(boxes) == 1 and cls[0] == 0 and conf[0] == pytest.approx(0.9)
    assert tuple(boxes[0][:2]) == (40.0, 40.0)   # cx,cy,w,h -> esquina


def test_paint_instances_higher_confidence_wins_and_maps_back():
    protos = np.full((1, 16, 16), 5.0, np.float32)           # todo "adentro"
    boxes = np.array([[0, 0, 32, 32], [16, 16, 32, 32]], np.float64)   # letterbox de 64x64, escala 1
    mask = paint_instances((64, 64), boxes, np.array([0.9, 0.5]), np.array([0, 1]),
                           np.ones((2, 1)), protos, 1.0, (0, 0), [1, 2], 64)
    assert mask[5, 5] == 1 and mask[40, 40] == 2 and mask[20, 20] == 1 and mask[60, 5] == 0


def test_config_onnx_needs_no_command(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text("backends:\n  - name: y\n    type: onnx\n    models_glob: [/no/existe/*.onnx]\n")
    (spec,) = load_backends(str(path))
    assert spec.type == "onnx" and spec.command == []
    with pytest.raises(BackendError):
        create_backend(spec)    # ni modelo ni classes


def test_missing_models_glob():
    with pytest.raises(BackendError):
        create_backend(BackendSpec(name="y", type="onnx"))
