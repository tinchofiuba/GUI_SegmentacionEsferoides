"""Manifiesto de un modelo ONNX: el `.yaml` que va al lado del `.onnx` y le dice a la
GUI cómo usarlo (normalización, relleno, clases...). Solo datos, nunca código: así un
modelo nuevo o reentrenado se usa sin tocar la GUI mientras su salida sea de un tipo
conocido (`semantic`). Sin Qt ni onnxruntime."""

import os
from dataclasses import dataclass, field

import yaml

from gui_segesferoides.backend import BackendError, ClassInfo

SCHEMA = 1
OUTPUTS = ("semantic",)
TTA_VIEWS = {"hflip", "vflip", "hvflip"}
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


@dataclass(frozen=True)
class Manifest:
    name: str
    output: str
    classes: list
    mean: tuple = None            # None = sin normalizar (0-1)
    std: tuple = None
    pad_multiple: int = 1
    tta: tuple = ()
    magnifications: list = field(default_factory=list)


def manifest_path(onnx_path):
    return os.path.splitext(onnx_path)[0] + ".yaml"


def _normalization(value):
    if value in (None, "unit"):
        return None, None
    if value == "imagenet":
        return IMAGENET["mean"], IMAGENET["std"]
    if isinstance(value, dict) and len(value.get("mean", ())) == 3 and len(value.get("std", ())) == 3:
        return tuple(float(v) for v in value["mean"]), tuple(float(v) for v in value["std"])
    raise BackendError(f"input.normalize inválido: {value!r} (imagenet | unit | {{mean: [3], std: [3]}})")


def parse_manifest(data, source="manifiesto"):
    if not isinstance(data, dict):
        raise BackendError(f"{source}: no es un mapa YAML")
    if data.get("schema") != SCHEMA:
        raise BackendError(f"{source}: schema {data.get('schema')!r}; esta GUI entiende schema {SCHEMA}")
    if data.get("output") not in OUTPUTS:
        raise BackendError(f"{source}: output {data.get('output')!r} no soportado (usar {', '.join(OUTPUTS)})")
    inp = data.get("input") or {}
    if inp.get("color", "rgb") != "rgb":
        raise BackendError(f"{source}: input.color debe ser rgb")
    tta = tuple(inp.get("tta") or ())
    if set(tta) - TTA_VIEWS:
        raise BackendError(f"{source}: input.tta desconocido {sorted(set(tta) - TTA_VIEWS)} (usar {sorted(TTA_VIEWS)})")
    pad = int(inp.get("pad_multiple", 1))
    if pad < 1:
        raise BackendError(f"{source}: input.pad_multiple debe ser >= 1")
    try:
        classes = [ClassInfo(int(c["id"]), str(c["name"]), tuple(int(v) for v in c["color"]))
                   for c in data.get("classes") or []]
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendError(f"{source}: clases inválidas ({exc!r})") from exc
    if not classes or any(c.id <= 0 for c in classes) or len({c.id for c in classes}) != len(classes):
        raise BackendError(f"{source}: 'classes' vacío, con ids repetidos o <= 0 (0 es el fondo)")
    mean, std = _normalization(inp.get("normalize"))
    return Manifest(name=str(data.get("name", "")), output=data["output"], classes=classes, mean=mean, std=std,
                    pad_multiple=pad, tta=tta, magnifications=[str(m) for m in data.get("magnifications") or []])


def load_manifest(onnx_path):
    """Manifiesto al lado de `onnx_path`, o None si no hay `.yaml` (modelo sin manifiesto)."""
    path = manifest_path(onnx_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise BackendError(f"No se pudo leer {path}: {exc}") from exc
    return parse_manifest(data, path)
