"""Backend `onnx`: corre dentro de la GUI un modelo exportado a .onnx. Sin torch ni ultralytics:
solo onnxruntime, numpy y OpenCV.

Si junto al `.onnx` hay un manifiesto `.yaml` (ver model_manifest.py), cómo usar el modelo sale de
ahí y el adaptador de salida lo elige `output`: `semantic` (semantic.py) o `instance-seg` (yolo_seg.py).
Sin manifiesto solo se entiende un YOLOv8-seg de Ultralytics, con lo que diga su metadata (camino
heredado, abajo).

Camino heredado: es un parche para comparar YOLO con los otros modelos: el modelo da *instancias*
(caja + clase + máscara) y la GUI trabaja con una máscara de ids por píxel, así
que se pintan las instancias de menor a mayor confianza (gana la más segura).
El pre/post-procesado sigue a Ultralytics: letterbox al lado mayor = `imgsz`
(relleno 114, múltiplo del stride), NMS por clase, máscaras = coeficientes · protos
recortadas a su caja. La GUI recorta la zona antes de inferir.

Devuelve la máscara cruda: el post-procesado lo hace la GUI.
"""

import ast
import dataclasses
import glob
import os
import threading

import cv2
import numpy as np

from gui_segesferoides import semantic, yolo_seg
from gui_segesferoides.backend import BackendError, BackendInfo, ClassInfo, ModelEntry
from gui_segesferoides.model_manifest import load_manifest
from gui_segesferoides.yolo_seg import decode, letterbox, paint_instances  # noqa: F401 - re-exportados

DEFAULT_COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]
GENERIC_DIRS = {"exportados", "weights", "train", "outputs"}   # carpetas que no dicen qué modelo es


def _label(path):
    """.../v8_nano_X/outputs/train/weights/exportados/best.onnx -> "v8_nano_X"."""
    for folder in os.path.dirname(os.path.abspath(path)).split(os.sep)[::-1]:
        if folder and folder not in GENERIC_DIRS:
            return folder
    return os.path.basename(path)


class OnnxBackend:
    """Opciones del YAML (todas opcionales salvo `models_glob`):

    models_glob   patrón (o lista) de archivos .onnx; el más reciente arranca cargado
    classes       [{id, name, color: [r, g, b]}]; si falta, salen de la metadata del
                  .onnx (`names`), con ids 1..n (0 es el fondo) y colores por defecto
    conf, iou     umbral de confianza (0.25) y de NMS (0.7), como Ultralytics; con manifiesto
                  los toma de él, y estas opciones (si están) lo pisan
    imgsz         lado mayor del letterbox; por defecto el de la metadata (o 640)
    tile, tile_overlap   (modelos con manifiesto) mosaicos de `tile` px (1024) con `tile_overlap`
                  px de contexto por lado (128) cuando la imagen es más grande; tile: 0 = entera
    workers       mosaicos en paralelo (por defecto min(4, núcleos // 3)); cada sesión usa núcleos // workers hilos
    """

    def __init__(self, spec):
        o = spec.options
        if not o.get("models_glob"):
            raise BackendError(f"Backend onnx '{spec.name}': falta la opción 'models_glob'")
        globs = o["models_glob"]
        self.models_globs = [os.path.expanduser(g) for g in ([globs] if isinstance(globs, str) else globs)]
        self.conf, self.iou = float(o.get("conf", 0.25)), float(o.get("iou", 0.7))
        self._overrides = {k: float(o[k]) for k in ("conf", "iou") if k in o}   # pisan el default del manifiesto
        self.imgsz = int(o["imgsz"]) if o.get("imgsz") else None
        self.tile = int(o.get("tile", 1024))          # modelos con manifiesto: 0 = imagen entera
        self.tile_overlap = int(o.get("tile_overlap", 128))
        cpus = os.cpu_count() or 1
        # mosaicos en paralelo: 4 trabajadores x (núcleos/4) hilos cada uno. Medido en 12 núcleos con
        # 3072x3072: 40 s -> 31 s; un recorte de una sola ventana no empeora con menos hilos por sesión.
        self.workers = max(1, int(o.get("workers", min(4, cpus // 3))))
        self.threads = max(1, cpus // self.workers)
        self.model = None
        self._session = None
        self._meta = {}
        self._manifest = None
        self._lock = threading.Lock()     # protege _option_values: la GUI escribe en el hilo principal
        self._option_values = {}
        declared = o.get("classes")
        self._declared = [ClassInfo(int(c["id"]), c["name"], tuple(c["color"])) for c in declared or []]
        self.info = BackendInfo(
            name=spec.name,
            classes=list(self._declared),
            magnifications=[],
            supports_bbox=True,
        )
        models = self._scan_models()
        if models:
            self.load_model(models[0].id)
        elif not declared:
            raise BackendError(f"Backend onnx '{spec.name}': no hay ningún .onnx en {self.models_globs} "
                               "ni 'classes' declaradas")

    # ---------- API ----------
    def _scan_models(self):
        paths = sorted({p for pattern in self.models_globs for p in glob.glob(pattern)},
                       key=os.path.getmtime, reverse=True)
        return [ModelEntry(p, _label(p), current=(i == 0)) for i, p in enumerate(paths)]

    def list_models(self):
        return [ModelEntry(m.id, m.label, m.note, m.id == self.model) for m in self._scan_models()]

    def load_model(self, model_id):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendError("Falta onnxruntime (uv add onnxruntime)") from exc
        if not os.path.exists(model_id):
            raise BackendError(f"No existe el modelo: {model_id}")
        # Sin planificación previa de buffers ni arena: el pico de RAM baja ~3x (10 -> 3 GB en 3072x3072)
        # con la misma máscara y sin costo de tiempo. Importa con imágenes grandes sin recortar.
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False
        options.intra_op_num_threads = self.threads
        try:
            session = ort.InferenceSession(model_id, options, providers=ort.get_available_providers())
        except Exception as exc:  # noqa: BLE001 - onnxruntime lanza tipos propios
            raise BackendError(f"No se pudo abrir {model_id}: {exc}") from exc
        manifest = load_manifest(model_id)
        if manifest is not None:
            (self._check_semantic if manifest.output == "semantic" else self._check_instance_seg)(
                session, manifest, model_id)
            self.info.classes = list(manifest.classes)
            self.info.magnifications = list(manifest.magnifications)
            self._session, self._meta, self._manifest, self.model = session, {}, manifest, model_id
            self.info.options = list(manifest.options)
            with self._lock:
                self._option_values = {spec.id: self._overrides.get(spec.id, spec.default)
                                       for spec in manifest.options}
            return manifest.name or _label(model_id)
        meta = session.get_modelmeta().custom_metadata_map
        if meta.get("task") != "segment" or len(session.get_outputs()) != 2:
            raise BackendError(f"{model_id} no parece un YOLO-seg de Ultralytics (task=segment, 2 salidas)")
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.info.magnifications = []
        if not self._declared:
            self.info.classes = [ClassInfo(i + 1, str(names[i]), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                                 for i in sorted(names)]
        elif names and len(names) != len(self._declared):
            raise BackendError(f"El modelo tiene {len(names)} clases y el backend declara {len(self._declared)}")
        else:
            self.info.classes = list(self._declared)
        self._session, self._meta, self._manifest, self.model = session, meta, None, model_id
        self.info.options = []
        with self._lock:
            self._option_values = {}
        return _label(model_id)

    @staticmethod
    def _check_semantic(session, manifest, path):
        """Prueba de humo: el modelo tiene que producir [1, C, H, W] con un canal por clase."""
        probe = np.zeros((32, 32, 3), np.uint8)
        try:
            logits = session.run(None, {session.get_inputs()[0].name: semantic.preprocess(probe, manifest)[0]})[0]
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"{path} no corre con su manifiesto: {exc}") from exc
        if logits.ndim != 4 or logits.shape[1] <= max(c.id for c in manifest.classes):
            raise BackendError(f"{path}: salida {logits.shape} no encaja con las clases del manifiesto")

    @staticmethod
    def _check_instance_seg(session, manifest, path):
        """Prueba de humo: 2 salidas, y output0 con 4 (caja) + clases + coeficientes de máscara."""
        if len(session.get_outputs()) != 2:
            raise BackendError(f"{path}: instance-seg espera 2 salidas (detecciones y protos de máscara)")
        probe = np.zeros((1, 3, manifest.pad_multiple * 2, manifest.pad_multiple * 2), np.float32)
        try:
            output0, protos = session.run(None, {session.get_inputs()[0].name: probe})
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"{path} no corre con su manifiesto: {exc}") from exc
        if output0.ndim != 3 or protos.ndim != 4 or output0.shape[1] != 4 + len(manifest.classes) + protos.shape[1]:
            raise BackendError(f"{path}: salidas {output0.shape}/{protos.shape} no encajan con "
                               f"{len(manifest.classes)} clases del manifiesto")

    def set_option(self, option_id, value):
        """Cambia una opción declarada en el manifiesto (`info.options`). Se aplica a la siguiente
        predicción; una predicción en curso sigue con los valores que tenía al empezar."""
        spec = next((o for o in self.info.options if o.id == option_id), None)
        if spec is None:
            raise BackendError(f"El modelo no tiene la opción {option_id!r}")
        if spec.type == "checkbox":
            if not isinstance(value, bool):
                raise BackendError(f"La opción {option_id!r} acepta solo true/false")
        else:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise BackendError(f"La opción {option_id!r} necesita un número") from exc
            if not spec.min <= value <= spec.max:
                raise BackendError(f"La opción {option_id!r} debe estar entre {spec.min} y {spec.max}")
        with self._lock:
            self._option_values[option_id] = value

    def _effective_manifest(self, values):
        """Manifiesto con los valores de las opciones aplicados (copia; el original no cambia)."""
        m = self._manifest
        if m.output == "semantic":
            return m if values.get("tta", True) else dataclasses.replace(m, tta=())
        conf = values.get("conf", self._overrides.get("conf", m.conf))   # sin control: rige el YAML del backend
        iou = values.get("iou", self._overrides.get("iou", m.iou))
        return dataclasses.replace(m, conf=conf, iou=iou)

    def predict(self, image_path, magnification, bbox=None):
        if self._session is None:
            raise BackendError("No hay ningún modelo cargado")
        bgr = cv2.imread(os.path.abspath(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise BackendError(f"No se pudo leer la imagen: {image_path}")
        if bbox is not None:
            x, y, w, h = (int(v) for v in bbox)
            bgr = bgr[y : y + h, x : x + w]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self._manifest is not None and self._manifest.output == "instance-seg":
            name = self._session.get_inputs()[0].name
            with self._lock:
                manifest = self._effective_manifest(dict(self._option_values))
            try:
                return yolo_seg.infer(lambda t: self._session.run(None, {name: t}), rgb, manifest)
            except Exception as exc:  # noqa: BLE001
                raise BackendError(f"Falló la inferencia ONNX: {exc}") from exc
        if self._manifest is not None:
            name = self._session.get_inputs()[0].name
            with self._lock:
                manifest = self._effective_manifest(dict(self._option_values))
            try:
                return semantic.infer(lambda t: self._session.run(None, {name: t})[0], rgb, manifest,
                                      self.tile, self.tile_overlap, self.workers)
            except Exception as exc:  # noqa: BLE001
                raise BackendError(f"Falló la inferencia ONNX: {exc}") from exc
        imgsz = self.imgsz or int(ast.literal_eval(self._meta.get("imgsz", "[640, 640]"))[0])
        stride = int(self._meta.get("stride", 32))
        boxed, scale, pad = letterbox(rgb, imgsz, stride)
        tensor = (boxed.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        try:
            output0, protos = self._session.run(None, {self._session.get_inputs()[0].name: tensor})
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"Falló la inferencia ONNX: {exc}") from exc
        class_ids = [c.id for c in self.info.classes]
        boxes, confidence, cls, coefs = decode(output0[0], len(class_ids), self.conf, self.iou)
        if not len(boxes):
            return np.zeros(rgb.shape[:2], np.uint8)
        return paint_instances(rgb.shape[:2], boxes, confidence, cls, coefs, protos[0], scale, pad, class_ids,
                               boxed.shape[1])

    def close(self):
        self._session = None
