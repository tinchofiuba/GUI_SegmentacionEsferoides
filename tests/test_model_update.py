import hashlib
import json
import pathlib

import pytest

from gui_segesferoides.model_update import download_model, fetch_index, pending_updates


def make_release(tmp_path, onnx=b"weights-v1", name="m_2026"):
    rel = tmp_path / "release"
    rel.mkdir(exist_ok=True)
    (rel / f"{name}.onnx").write_bytes(onnx)
    (rel / f"{name}.yaml").write_text("schema: 1\n")
    index = {"schema": 1, "models": [{"name": name, "onnx": f"{name}.onnx", "yaml": f"{name}.yaml",
                                      "sha256": hashlib.sha256(onnx).hexdigest(), "notes": "mejor"}]}
    (rel / "models.json").write_text(json.dumps(index))
    return (rel / "models.json").resolve()


def test_new_model_is_pending_then_installed_then_up_to_date(tmp_path):
    url = pathlib.Path(make_release(tmp_path)).as_uri()
    models = fetch_index(url)
    folder = str(tmp_path / "models")
    assert pending_updates(models, folder) == models
    out = download_model(models[0], folder)
    assert out.endswith("m_2026.onnx") and (tmp_path / "models" / "m_2026" / "m_2026.yaml").exists()
    assert pending_updates(models, folder) == []


def test_changed_weights_under_same_name_are_pending(tmp_path):
    folder = str(tmp_path / "models")
    download_model(fetch_index(make_release(tmp_path).as_uri())[0], folder)
    new = fetch_index(make_release(tmp_path, onnx=b"weights-v2").as_uri())
    assert pending_updates(new, folder) == new


def test_corrupt_download_is_rejected_and_not_installed(tmp_path):
    models = fetch_index(make_release(tmp_path).as_uri())
    bad = models[0].__class__(**{**models[0].__dict__, "sha256": "0" * 64})
    with pytest.raises(ValueError):
        download_model(bad, str(tmp_path / "models"))
    assert not (tmp_path / "models").exists()


def test_index_rejects_path_traversal_names(tmp_path):
    from gui_segesferoides.model_update import parse_index
    with pytest.raises(ValueError):
        parse_index({"schema": 1, "models": [{"name": "../x", "onnx": "a", "yaml": "b", "sha256": "c"}]}, "file:///r/")
