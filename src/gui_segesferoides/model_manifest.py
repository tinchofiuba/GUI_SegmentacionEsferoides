"""Manifiesto de un modelo ONNX: el `.yaml` que va al lado del `.onnx` y le dice a la
GUI cómo usarlo (normalización, relleno, clases...). Solo datos, nunca código: así un
modelo nuevo o reentrenado se usa sin tocar la GUI mientras su salida sea de un tipo
conocido: `semantic` (logits por píxel, el canal k es la clase con id k) o `instance-seg`
(YOLOv8-seg; el canal i del modelo es `classes[i]`, en el orden de la lista). Sin Qt ni onnxruntime."""

import os
from dataclasses import dataclass, field

import yaml

from gui_segesferoides.backend import BackendError, ClassInfo, OptionSpec

SCHEMA = 1
OUTPUTS = ("semantic", "instance-seg")
TTA_VIEWS = {"hflip", "vflip", "hvflip"}
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
# Opciones que cada adaptador sabe aplicar: id -> (tipo de control, min, max). El id fija el tipo.
OPTIONS_BY_OUTPUT = {
    "semantic": {"tta": ("checkbox", None, None)},
    "instance-seg": {"conf": ("number", 0.01, 0.99), "iou": ("number", 0.01, 1.0)},
}


@dataclass(frozen=True)
class Manifest:
    name: str
    output: str
    classes: list
    mean: tuple = None            # None = sin normalizar (0-1)
    std: tuple = None
    pad_multiple: int = 1
    tta: tuple = ()
    imgsz: int = None             # instance-seg: lado mayor del letterbox
    conf: float = 0.25            # instance-seg: umbral de confianza
    iou: float = 0.7              # instance-seg: umbral de NMS
    magnifications: list = field(default_factory=list)
    options: tuple = ()           # [OptionSpec]: controles que la GUI muestra, ver set_option


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


def _parse_options(data, output, tta, extra, source):
    """Controles declarados en `options:`. Cada id es de los que sabe el adaptador del `output`;
    sin `options` no hay controles (el modelo funciona igual que antes)."""
    raw = data.get("options") or []
    if not isinstance(raw, list):
        raise BackendError(f"{source}: 'options' debe ser una lista")
    known = OPTIONS_BY_OUTPUT[output]
    fallback = {"tta": True, "conf": extra.get("conf"), "iou": extra.get("iou")}
    specs, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or "id" not in item:
            raise BackendError(f"{source}: cada opción necesita 'id'")
        oid = str(item["id"])
        if oid not in known:
            raise BackendError(f"{source}: opción {oid!r} no existe para output {output} (usar {sorted(known)})")
        if oid in seen:
            raise BackendError(f"{source}: opción {oid!r} repetida")
        if oid == "tta" and not tta:
            raise BackendError(f"{source}: la opción 'tta' necesita input.tta con vistas")
        seen.add(oid)
        kind, lo, hi = known[oid]
        label, default = str(item.get("label", oid)), item.get("default", fallback[oid])
        if kind == "checkbox":
            if not isinstance(default, bool):
                raise BackendError(f"{source}: la opción {oid!r} necesita default true/false")
            specs.append(OptionSpec(oid, kind, label, default))
        else:
            try:
                default = float(default)
            except (TypeError, ValueError) as exc:
                raise BackendError(f"{source}: default de {oid!r} no numérico ({exc!r})") from exc
            if not lo <= default <= hi:
                raise BackendError(f"{source}: default de {oid!r} fuera de [{lo}, {hi}]")
            specs.append(OptionSpec(oid, kind, label, default, lo, hi))
    return tuple(specs)


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
    extra = {}
    if data["output"] == "instance-seg":
        if tta:
            raise BackendError(f"{source}: input.tta no aplica a instance-seg")
        decode = data.get("decode") or {}
        try:
            extra = {"imgsz": int(inp["imgsz"]), "conf": float(decode.get("conf", 0.25)),
                     "iou": float(decode.get("iou", 0.7))}
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"{source}: instance-seg necesita input.imgsz numérico ({exc!r})") from exc
        if extra["imgsz"] < 1 or not (0 < extra["conf"] < 1) or not (0 < extra["iou"] <= 1):
            raise BackendError(f"{source}: imgsz >= 1, 0 < decode.conf < 1 y 0 < decode.iou <= 1")
    options = _parse_options(data, data["output"], tta, extra, source)
    return Manifest(name=str(data.get("name", "")), output=data["output"], classes=classes, mean=mean, std=std,
                    pad_multiple=pad, tta=tta, magnifications=[str(m) for m in data.get("magnifications") or []],
                    options=options, **extra)


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
