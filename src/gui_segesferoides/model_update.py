"""Actualizar modelos desde un índice publicado (p. ej. un asset de GitHub Releases).

El índice es un JSON que genera el repo del modelo (`u-resnet-make-index`):

    {"schema": 1, "models": [{"name": "insitu_3t3_2026-12-01", "onnx": "<url>", "yaml": "<url>",
                              "sha256": "<del .onnx>", "notes": "texto opcional"}]}

Una URL relativa se resuelve contra la del índice. Un modelo está al día si
`models/<name>/<name>.onnx` existe con el mismo sha256."""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass

TIMEOUT = 30


@dataclass(frozen=True)
class RemoteModel:
    name: str
    onnx: str
    yaml: str
    sha256: str
    notes: str = ""


def _read(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:   # noqa: S310 (http/https/file del config)
        return r.read()


def parse_index(data, base_url):
    if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("models"), list):
        raise ValueError("El índice de modelos no tiene el formato esperado (schema 1)")
    models = []
    for m in data["models"]:
        try:
            name = str(m["name"])
            if not name or os.path.basename(name) != name or name in (".", ".."):
                raise ValueError(f"nombre de modelo inválido: {name!r}")
            models.append(RemoteModel(name, urllib.parse.urljoin(base_url, m["onnx"]),
                                      urllib.parse.urljoin(base_url, m["yaml"]),
                                      str(m["sha256"]).lower(), str(m.get("notes", ""))))
        except KeyError as exc:
            raise ValueError(f"Entrada del índice sin el campo {exc}") from exc
    return models


def fetch_index(url):
    return parse_index(json.loads(_read(url).decode("utf-8")), url)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pending_updates(models, models_dir):
    """Modelos del índice que no están instalados o cuyo .onnx instalado es otro."""
    out = []
    for m in models:
        local = os.path.join(models_dir, m.name, m.name + ".onnx")
        if not os.path.isfile(local) or sha256_of(local) != m.sha256:
            out.append(m)
    return out


def download_model(model, models_dir):
    """Baja el .onnx y el .yaml, verifica el sha256 y recién entonces los instala.
    Devuelve la ruta del .onnx instalado."""
    dest = os.path.join(models_dir, model.name)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_tmp = os.path.join(tmp, model.name + ".onnx")
        yaml_tmp = os.path.join(tmp, model.name + ".yaml")
        for url, path in ((model.yaml, yaml_tmp), (model.onnx, onnx_tmp)):
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r, open(path, "wb") as f:   # noqa: S310
                shutil.copyfileobj(r, f)
        if sha256_of(onnx_tmp) != model.sha256:
            raise ValueError(f"{model.name}: la descarga está corrupta (sha256 no coincide)")
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(yaml_tmp, os.path.join(dest, model.name + ".yaml"))
        shutil.copy2(onnx_tmp, os.path.join(dest, model.name + ".onnx"))   # el .onnx al final: queda el más reciente
    return os.path.join(dest, model.name + ".onnx")
