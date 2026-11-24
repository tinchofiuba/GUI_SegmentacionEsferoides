import pytest

from gui_segesferoides.model_install import install_model, installed_models, models_dir_for


def test_models_dir_is_glob_prefix():
    assert models_dir_for(["models/*/*.onnx"]) == "models"
    assert models_dir_for(["models/*.onnx"]) == "models"
    assert models_dir_for(None) == "models"


def test_install_copies_onnx_and_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.onnx").write_bytes(b"x")
    (src / "m.yaml").write_text("schema: 1\n")
    out = install_model(str(src / "m.onnx"), str(tmp_path / "models"))
    assert out == str(tmp_path / "models" / "m" / "m.onnx")
    assert (tmp_path / "models" / "m" / "m.yaml").exists()
    assert installed_models(str(tmp_path / "models")) == [out]


def test_install_rejects_missing_manifest(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"x")
    with pytest.raises(ValueError):
        install_model(str(tmp_path / "m.onnx"), str(tmp_path / "models"))
