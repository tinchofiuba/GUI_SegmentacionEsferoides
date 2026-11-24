"""Configuración de backends (config/backends.yaml)."""

from dataclasses import dataclass, field

import yaml

from gui_segesferoides.onnx_backend import OnnxBackend

# claves de versiones anteriores (backends por subproceso / programa externo) que ya no existen
REMOVED_KEYS = ("command", "cwd", "env", "install")


@dataclass(frozen=True)
class BackendSpec:
    name: str
    options: dict = field(default_factory=dict)   # models_glob, conf, tile... (ver onnx_backend.py)


def create_backend(spec):
    """Conecta el backend que describe `spec`: siempre ONNX, dentro de la GUI."""
    return OnnxBackend(spec)


def load_backends(path):
    """Lee la lista de backends. Formato:

        backends:
          - name: u-resnet
            models_glob: [~/modelos/uresnet/*.onnx]   # cada .onnx lleva al lado su .yaml (model_manifest.py)
            tile: 1024                                # opciones del motor ONNX, ver onnx_backend.py
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    specs = []
    for entry in data.get("backends", []):
        if "name" not in entry:
            raise ValueError(f"Backend inválido en {path} (falta 'name'): {entry}")
        old = [k for k in REMOVED_KEYS if k in entry] + (["type"] if entry.get("type", "onnx") != "onnx" else [])
        if old:
            raise ValueError(f"Backend '{entry['name']}' en {path}: {', '.join(old)} ya no existe. "
                             "La GUI solo corre modelos .onnx: usar `models_glob` (ver config/backends.example.yaml).")
        if not entry.get("models_glob"):
            raise ValueError(f"Backend '{entry['name']}' en {path}: falta 'models_glob'")
        specs.append(BackendSpec(name=entry["name"],
                                 options={k: v for k, v in entry.items() if k not in ("name", "type")}))
    return specs


def load_index_url(path):
    """URL del índice de modelos (clave `models_index_url` de nivel superior), o None."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("models_index_url") or None
