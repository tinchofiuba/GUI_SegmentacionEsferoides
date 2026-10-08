"""Backend `cli`: adapta un programa de línea de comandos que no habla el
protocolo (docs/protocol.md) a la misma interfaz que `SubprocessBackend`.

Cada predicción lanza el comando de nuevo (paga su arranque cada vez). El
comando no puede declarar clases ni modelos, así que los declara el YAML
(ver `config.load_backends`). La zona la recorta la GUI antes de llamar al
comando: la inferencia sigue siendo a escala nativa.
"""

import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from gui_segesferoides.backend import BackendError, BackendInfo, ClassInfo, ModelEntry

TOKENS = ("{image}", "{model}", "{stem}", "{output_dir}")


def _substitute(text, values):
    for token in TOKENS:
        text = text.replace(token, values.get(token, token))
    return text


class CliBackend:
    """Opciones (todas salvo `mask` y `classes` son opcionales):

    command           comando con {image} y {model}
    classes           [{id, name, color: [r, g, b]}]
    models_glob       patrón de archivos/carpetas de modelo; {model} es cada
                      coincidencia (la más reciente arranca cargada)
    mask              dónde queda la máscara: ruta con {output_dir} y {stem}
                      (stem = nombre de la imagen sin extensión)
    output_dir_regex  regex con un grupo sobre stdout que da {output_dir}
    cleanup           borrar {output_dir} después de leer la máscara (solo si
                      está dentro de `cwd`)
    """

    def __init__(self, spec):
        o = spec.options
        for key in ("classes", "mask"):
            if key not in o:
                raise BackendError(f"Backend cli '{spec.name}': falta la opción '{key}'")
        self.command, self.cwd, self.env = list(spec.command), spec.cwd, spec.env
        self.mask_template = o["mask"]
        self.output_dir_regex = re.compile(o["output_dir_regex"]) if o.get("output_dir_regex") else None
        self.cleanup = bool(o.get("cleanup", False))
        self.models_glob = os.path.expanduser(o["models_glob"]) if o.get("models_glob") else None
        self.model = None
        self.info = BackendInfo(
            name=spec.name,
            classes=[ClassInfo(int(c["id"]), c["name"], tuple(c["color"])) for c in o["classes"]],
            magnifications=[],
            supports_bbox=True,
        )
        self._tmp = tempfile.TemporaryDirectory(prefix="gui_seg_cli_")
        models = self._scan_models()
        if models:  # como el resto de backends, arranca con uno cargado: el más reciente
            self.model = models[0].id

    # ---------- API ----------
    def _scan_models(self):
        if not self.models_glob:
            return []
        paths = sorted(glob.glob(self.models_glob), key=os.path.getmtime, reverse=True)
        return [ModelEntry(p, self._label(p), current=(i == 0)) for i, p in enumerate(paths)]

    @staticmethod
    def _label(path):
        p = Path(path)
        # .../<run>/checkpoints/best.ext -> "<run>"; un directorio -> su nombre
        return p.parent.parent.name if p.parent.name == "checkpoints" else p.name

    def list_models(self):
        return [ModelEntry(m.id, m.label, m.note, m.id == self.model) for m in self._scan_models()]

    def load_model(self, model_id):
        if not os.path.exists(model_id):
            raise BackendError(f"No existe el modelo: {model_id}")
        self.model = model_id
        return self._label(model_id)

    def predict(self, image_path, magnification, bbox=None):
        if self.model is None:
            raise BackendError("No hay ningún modelo cargado")
        image_path = os.path.abspath(image_path)
        if bbox is None:
            source = image_path
        else:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise BackendError(f"No se pudo leer la imagen: {image_path}")
            x, y, w, h = bbox
            source = os.path.join(self._tmp.name, "crop.png")
            cv2.imwrite(source, img[y : y + h, x : x + w])
        values = {"{image}": source, "{model}": self.model,
                  "{stem}": os.path.splitext(os.path.basename(source))[0]}

        proc = self._run([_substitute(c, values) for c in self.command])
        output_dir = self._find_output_dir(proc.stdout)
        values["{output_dir}"] = output_dir
        mask_path = _substitute(self.mask_template, values)
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        finally:
            self._cleanup(output_dir)
        if mask is None or mask.ndim != 2:
            raise BackendError(f"El comando no dejó una máscara de 1 canal en {mask_path}")
        return mask.astype(np.uint8)

    def close(self):
        self._tmp.cleanup()

    # ---------- internos ----------
    def _run(self, command):
        try:
            proc = subprocess.run(
                command, cwd=self.cwd, env={**os.environ, **self.env},
                capture_output=True, text=True, encoding="utf-8",
            )
        except OSError as exc:
            raise BackendError(f"No se pudo lanzar {command}: {exc}") from exc
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
            raise BackendError(f"El comando terminó con código {proc.returncode}.\n{tail}")
        return proc

    def _find_output_dir(self, stdout):
        if self.output_dir_regex is None:
            return ""
        match = self.output_dir_regex.search(stdout)
        if match is None:
            raise BackendError("No se encontró la carpeta de salida en lo que imprimió el comando "
                               f"(regex {self.output_dir_regex.pattern!r}).")
        out = match.group(1).strip()
        return out if os.path.isabs(out) or not self.cwd else os.path.join(self.cwd, out)

    def _cleanup(self, output_dir):
        if not (self.cleanup and output_dir and self.cwd):
            return
        target, root = Path(output_dir).resolve(), Path(self.cwd).resolve()
        if root in target.parents:  # nunca borra fuera de la carpeta del backend
            shutil.rmtree(target, ignore_errors=True)
