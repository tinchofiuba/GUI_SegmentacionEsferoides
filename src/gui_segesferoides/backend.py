"""Cliente del protocolo de backends de segmentación (ver docs/protocol.md).

La GUI no importa ningún modelo: cada backend es un proceso aparte (con su
propio Python y dependencias) que habla JSON de una línea por stdin/stdout.
Sin Qt — se testea con un backend falso (tests/fake_backend.py).
"""

import json
import os
import subprocess
import tempfile
import threading
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

PROTOCOL_VERSION = 1


class BackendError(RuntimeError):
    """El backend respondió con error, murió o rompió el protocolo."""


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
    accepts_model_path: bool = False  # `load` acepta una ruta de carpeta arbitraria
    supports_reference_stats: bool = False  # responde `reference_stats`
    protocol: int = PROTOCOL_VERSION

    @property
    def palette(self):
        return {c.id: c.color for c in self.classes}

    @classmethod
    def from_dict(cls, d):
        if d.get("protocol") != PROTOCOL_VERSION:
            raise BackendError(
                f"El backend habla el protocolo {d.get('protocol')!r}, "
                f"esta GUI espera {PROTOCOL_VERSION}"
            )
        classes = [ClassInfo(int(c["id"]), c["name"], tuple(c["color"])) for c in d["classes"]]
        if any(c.id <= 0 for c in classes):
            raise BackendError("Los ids de clase deben ser > 0 (0 es el fondo)")
        return cls(
            name=d["name"],
            classes=classes,
            magnifications=list(d.get("magnifications", [])),
            supports_bbox=bool(d.get("supports_bbox", True)),
            accepts_model_path=bool(d.get("accepts_model_path", False)),
            supports_reference_stats=bool(d.get("supports_reference_stats", False)),
        )


class SubprocessBackend:
    """Lanza `command` como proceso hijo y le habla por stdin/stdout.
    Una sola petición a la vez (lock): el proceso mantiene el modelo cargado."""

    def __init__(self, command, cwd=None, env=None):
        self.command = list(command)
        self._lock = threading.Lock()
        self._next_id = 0
        self._stderr_tail = deque(maxlen=40)
        self._reference_stats = None   # caché del modelo cargado; None = sin pedir
        self._tmp = tempfile.TemporaryDirectory(prefix="gui_seg_")
        full_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"}
        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=cwd,
                env=full_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            raise BackendError(f"No se pudo lanzar el backend {self.command}: {exc}") from exc
        # Se drena stderr siempre: si se llena el pipe, el hijo se bloquea.
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self.info = BackendInfo.from_dict(self._request("info"))

    def _drain_stderr(self):
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip())

    def _request(self, cmd, **params):
        with self._lock:
            if self._proc.poll() is not None:
                raise BackendError(self._death_message())
            self._next_id += 1
            req_id = self._next_id
            try:
                self._proc.stdin.write(json.dumps({"id": req_id, "cmd": cmd, **params}) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise BackendError(self._death_message()) from exc
            if not line:
                raise BackendError(self._death_message())
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BackendError(f"Respuesta inválida del backend: {line[:200]!r}") from exc
            if resp.get("id") != req_id:
                raise BackendError(f"Respuesta desfasada (id {resp.get('id')} != {req_id})")
            if not resp.get("ok"):
                raise BackendError(resp.get("error", "error desconocido del backend"))
            return resp.get("result")

    def _death_message(self):
        tail = "\n".join(self._stderr_tail)
        return f"El backend terminó (código {self._proc.poll()}).\n{tail}".strip()

    # ---------- API ----------
    def list_models(self):
        return [ModelEntry(**m) for m in self._request("list_models")]

    def load_model(self, model_id):
        """Devuelve la descripción del modelo cargado (texto)."""
        description = self._request("load", model=model_id).get("description", model_id)
        self._reference_stats = None
        return description

    def reference_stats(self):
        """Estadística del set de entrenamiento del modelo cargado (ver
        docs/protocol.md) o None si el backend no la ofrece."""
        if not self.info.supports_reference_stats:
            return None
        if self._reference_stats is None:
            try:
                self._reference_stats = self._request("reference_stats")
            except BackendError:
                return None
        return self._reference_stats

    def predict(self, image_path, magnification, bbox=None):
        """Devuelve la máscara (uint8, ids de clase) del recorte `bbox`
        (x, y, w, h en píxeles nativos) o de la imagen completa."""
        out = os.path.join(self._tmp.name, "pred.png")
        result = self._request(
            "predict",
            image=os.path.abspath(image_path),
            magnification=magnification,
            bbox=list(bbox) if bbox is not None else None,
            out=out,
        )
        mask = cv2.imread(result["mask"], cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            raise BackendError("El backend devolvió una máscara ilegible (se espera PNG de 1 canal)")
        return mask.astype(np.uint8)

    def close(self):
        if self._proc.poll() is None:
            try:
                self._request("shutdown")
            except BackendError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._tmp.cleanup()
