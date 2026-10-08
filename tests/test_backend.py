import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from gui_segesferoides.backend import BackendError, SubprocessBackend

FAKE = str(Path(__file__).parent / "fake_backend.py")


@pytest.fixture
def backend():
    b = SubprocessBackend([sys.executable, FAKE])
    yield b
    b.close()


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.zeros((200, 300, 3), np.uint8))
    return str(p)


def test_info_declares_classes_and_palette(backend):
    assert backend.info.name == "fake"
    assert backend.info.palette == {1: (0, 255, 0), 2: (255, 255, 0), 5: (0, 255, 255)}
    assert backend.info.magnifications == ["4x", "10x"]


def test_models_and_load(backend):
    assert [m.id for m in backend.list_models()] == ["a", "b"]
    assert backend.load_model("a") == "fake:a"
    with pytest.raises(BackendError, match="no existe"):
        backend.load_model("malo")
    assert backend.load_model("b") == "fake:b"  # sigue vivo tras un error


def test_predict_full_and_bbox(backend, image_path):
    full = backend.predict(image_path, "10x")
    assert full.shape == (200, 300) and full.dtype == np.uint8 and full.max() == 2
    crop = backend.predict(image_path, "10x", bbox=(10, 20, 80, 60))
    assert crop.shape == (60, 80)


def test_backend_crash_is_reported(image_path):
    b = SubprocessBackend([sys.executable, FAKE, "--crash-on-predict"])
    with pytest.raises(BackendError, match="terminó"):
        b.predict(image_path, "10x")
    with pytest.raises(BackendError):
        b.list_models()
    b.close()


def test_missing_command():
    with pytest.raises(BackendError, match="No se pudo lanzar"):
        SubprocessBackend(["/no/existe/backend"])


def test_protocol_version_mismatch(tmp_path):
    script = tmp_path / "old.py"
    script.write_text(
        "import sys, json\n"
        "for l in sys.stdin:\n"
        "    r = json.loads(l)\n"
        "    res = {'protocol': 99, 'name': 'x', 'classes': []}\n"
        "    print(json.dumps({'id': r['id'], 'ok': True, 'result': res}), flush=True)\n"
    )
    with pytest.raises(BackendError, match="protocolo"):
        SubprocessBackend([sys.executable, str(script)])
