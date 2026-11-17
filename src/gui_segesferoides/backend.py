"""Tipos que comparten la GUI y los backends (hoy solo `OnnxBackend`, ver onnx_backend.py).

Un backend expone: `info` (BackendInfo), `list_models()`, `load_model(id)`,
`predict(image_path, magnification, bbox=None)` -> máscara uint8 con ids de clase, y `close()`.
Sin Qt — se testea con un backend falso (tests/fake_backend.py).
"""

from dataclasses import dataclass


class BackendError(RuntimeError):
    """El backend no pudo cargar el modelo o predecir."""


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str
    color: tuple  # RGB


@dataclass(frozen=True)
class ModelEntry:
    id: str
    label: str
    note: str = ""
    current: bool = False  # el backend lo tiene cargado ahora


@dataclass
class BackendInfo:
    name: str
    classes: list
    magnifications: list
    supports_bbox: bool = True

    @property
    def palette(self):
        return {c.id: c.color for c in self.classes}
