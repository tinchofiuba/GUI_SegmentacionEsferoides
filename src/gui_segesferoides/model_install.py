"""Instalar un modelo en la GUI: copiar su .onnx y su manifiesto .yaml a la carpeta de modelos."""

import glob
import os
import shutil

GLOB_CHARS = set("*?[")


def models_dir_for(models_globs, default="models"):
    """Carpeta donde instalar: el prefijo sin comodines del primer `models_glob` del backend."""
    for pattern in models_globs or ():
        parts = []
        for part in os.path.expanduser(pattern).split(os.sep):
            if GLOB_CHARS & set(part) or part.endswith(".onnx"):
                break
            parts.append(part)
        folder = os.sep.join(parts)
        if folder:
            return folder
    return default


def install_model(onnx_path, models_dir):
    """Copia `<m>.onnx` y `<m>.yaml` a `models_dir/<m>/`. Devuelve la ruta del .onnx instalado.

    Sin manifiesto el modelo no se entiende (ver docs/model_manifest.md): se rechaza."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    manifest = os.path.splitext(onnx_path)[0] + ".yaml"
    if not os.path.isfile(onnx_path) or not onnx_path.lower().endswith(".onnx"):
        raise ValueError(f"{onnx_path} no es un archivo .onnx")
    if not os.path.isfile(manifest):
        raise ValueError(f"Falta el manifiesto {os.path.basename(manifest)} al lado del .onnx")
    dest = os.path.join(models_dir, stem)
    os.makedirs(dest, exist_ok=True)
    for src in (manifest, onnx_path):
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))   # el .onnx al final: es el más reciente
    return os.path.join(dest, stem + ".onnx")


def installed_models(models_dir):
    return sorted(glob.glob(os.path.join(models_dir, "*", "*.onnx")))
