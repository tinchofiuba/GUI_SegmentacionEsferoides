import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from gui_segesferoides.backend import BackendError
from gui_segesferoides.config import BackendSpec, create_backend, load_backends

FAKE_CLI = str(Path(__file__).parent / "fake_cli.py")


@pytest.fixture
def repo(tmp_path):
    """Un 'repo de otro' con dos checkpoints; el segundo es el más reciente."""
    for i, name in enumerate(("run_a", "run_b")):
        ck = tmp_path / "outputs" / "runs" / name / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "best.msgpack").write_text("x")
        os.utime(ck / "best.msgpack", (time.time() + i, time.time() + i))
    return tmp_path


def make_spec(repo, **extra):
    options = {
        "classes": [{"id": 1, "name": "sueltas", "color": [0, 255, 0]},
                    {"id": 2, "name": "agregados", "color": [255, 255, 0]}],
        "models_glob": str(repo / "outputs/runs/*/checkpoints/best.msgpack"),
        "output_dir_regex": r"Output directory: (.+)",
        "mask": "{output_dir}/predictions/{stem}.png",
        **extra,
    }
    return BackendSpec("cli-fake", [sys.executable, FAKE_CLI, "--checkpoint", "{model}",
                                    "--input", "{image}"], cwd=str(repo), type="cli", options=options)


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.full((200, 300, 3), 80, np.uint8))
    return str(p)


def test_declares_classes_and_lists_models_latest_loaded(repo):
    b = create_backend(make_spec(repo))
    assert b.info.palette == {1: (0, 255, 0), 2: (255, 255, 0)} and b.info.magnifications == []
    models = b.list_models()
    assert [m.label for m in models] == ["run_b", "run_a"]
    assert [m.current for m in models] == [True, False]
    assert b.load_model(models[1].id) == "run_a"
    assert [m.current for m in b.list_models()] == [False, True]
    with pytest.raises(BackendError, match="No existe"):
        b.load_model("/no/existe")
    b.close()


def test_predict_full_and_crop_have_expected_shapes(repo, image_path):
    b = create_backend(make_spec(repo))
    full = b.predict(image_path, "", None)
    assert full.shape == (200, 300) and full.dtype == np.uint8 and set(np.unique(full)) == {0, 2}
    crop = b.predict(image_path, "", (10, 20, 80, 60))
    assert crop.shape == (60, 80)
    assert (repo / "outputs/infer/run_1").exists()      # sin cleanup, queda
    b.close()


def test_cleanup_removes_output_dir_only_inside_cwd(repo, image_path):
    b = create_backend(make_spec(repo, cleanup=True))
    b.predict(image_path, "", None)
    assert not (repo / "outputs/infer/run_1").exists()
    b.close()


def test_failing_command_reports_stderr(repo, image_path):
    spec = make_spec(repo)
    spec = BackendSpec(spec.name, spec.command + ["--fail"], spec.cwd, type="cli", options=spec.options)
    b = create_backend(spec)
    with pytest.raises(BackendError, match="Checkpoint not found"):
        b.predict(image_path, "", None)
    b.close()


def test_predict_without_models_and_missing_options(tmp_path, image_path):
    empty = make_spec(tmp_path, models_glob=str(tmp_path / "nada/*"))
    b = create_backend(empty)
    assert b.list_models() == []
    with pytest.raises(BackendError, match="ningún modelo"):
        b.predict(image_path, "", None)
    with pytest.raises(BackendError, match="falta la opción 'mask'"):
        create_backend(BackendSpec("x", ["true"], type="cli", options={"classes": []}))


def test_load_backends_reads_cli_options(tmp_path):
    cfg = tmp_path / "b.yaml"
    cfg.write_text(
        "backends:\n"
        "  - name: a\n    command: [py, -m, x]\n"
        "  - name: b\n    type: cli\n    command: [py, '{image}']\n    mask: '{output_dir}/m.png'\n"
        "    classes: []\n"
    )
    a, b = load_backends(cfg)
    assert a.type == "protocol" and a.options == {}
    assert b.type == "cli" and b.options["mask"] == "{output_dir}/m.png"
    cfg.write_text("backends:\n  - name: c\n    type: raro\n    command: [x]\n")
    with pytest.raises(ValueError, match="type desconocido"):
        load_backends(cfg)
