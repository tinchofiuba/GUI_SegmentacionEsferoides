import sys

import pytest

from gui_segesferoides.config import BackendSpec
from gui_segesferoides.installer import is_installed, plan, run_install


def spec(tmp_path, install, command=None):
    return BackendSpec("x", command or [".venv/bin/python", "-m", "m"], cwd=str(tmp_path / "repo"),
                       install=install)


def test_is_installed_needs_dir_and_executable(tmp_path):
    s = spec(tmp_path, [])
    assert not is_installed(s)
    (tmp_path / "repo/.venv/bin").mkdir(parents=True)
    assert not is_installed(s)                      # carpeta sin el ejecutable
    (tmp_path / "repo/.venv/bin/python").write_text("")
    assert is_installed(s)
    assert is_installed(BackendSpec("p", ["python3"])) and not is_installed(BackendSpec("p", ["no-existe-xyz"]))


def test_plan_expands_cwd_and_skip_if(tmp_path):
    s = spec(tmp_path, [["git", "clone", "URL", "{cwd}"], {"cmd": ["uv", "sync"], "skip_if": "{cwd}/.venv"}])
    assert plan(s) == [(["git", "clone", "URL", str(tmp_path / "repo")], None),
                       (["uv", "sync"], str(tmp_path / "repo/.venv"))]


def test_run_install_runs_in_order_creates_parent_and_skips(tmp_path):
    py = sys.executable
    s = spec(tmp_path, [
        [py, "-c", "import os; os.makedirs('repo'); open('repo/a', 'w')"],        # corre en la carpeta madre
        {"cmd": [py, "-c", "raise SystemExit(9)"], "skip_if": "{cwd}/a"},         # se salta: a existe
        [py, "-c", "open('b', 'w')"],                                             # ahora corre en repo/
    ])
    run_install(s)
    assert (tmp_path / "repo/a").exists() and (tmp_path / "repo/b").exists()


def test_run_install_failure_reports_command_and_output(tmp_path):
    s = spec(tmp_path, [[sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(2)"]])
    with pytest.raises(RuntimeError, match="boom"):
        run_install(s)
    with pytest.raises(RuntimeError, match="No se pudo ejecutar"):
        run_install(spec(tmp_path, [["/no/existe"]]))
