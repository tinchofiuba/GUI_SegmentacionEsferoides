"""Backend `onnx`: corre dentro de la GUI un modelo YOLOv8-seg exportado a .onnx
(Ultralytics, `format=onnx`, sin NMS) con la misma interfaz que `SubprocessBackend`
y `CliBackend`. Sin torch ni ultralytics: solo onnxruntime, numpy y OpenCV.

Es un parche para comparar YOLO con los otros modelos: el modelo da *instancias*
(caja + clase + máscara) y la GUI trabaja con una máscara de ids por píxel, así
que se pintan las instancias de menor a mayor confianza (gana la más segura).
El pre/post-procesado sigue a Ultralytics: letterbox al lado mayor = `imgsz`
(relleno 114, múltiplo del stride), NMS por clase, máscaras = coeficientes · protos
recortadas a su caja. La GUI recorta la zona antes de inferir.

Devuelve la máscara cruda: el post-procesado lo hace la GUI.
"""

import ast
import glob
import os

import cv2
import numpy as np

from gui_segesferoides import semantic
from gui_segesferoides.backend import BackendError, BackendInfo, ClassInfo, ModelEntry
from gui_segesferoides.model_manifest import load_manifest

DEFAULT_COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]
GENERIC_DIRS = {"exportados", "weights", "train", "outputs"}   # carpetas que no dicen qué modelo es


def _label(path):
    """.../v8_nano_X/outputs/train/weights/exportados/best.onnx -> "v8_nano_X"."""
    for folder in os.path.dirname(os.path.abspath(path)).split(os.sep)[::-1]:
        if folder and folder not in GENERIC_DIRS:
            return folder
    return os.path.basename(path)


def letterbox(image, imgsz, stride):
    """Escala para que el lado mayor sea `imgsz` y rellena (114) hasta múltiplos de
    `stride`, centrado. Devuelve (imagen, escala, (izq, arriba))."""
    h, w = image.shape[:2]
    r = imgsz / max(h, w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else image
    ph, pw = (-nh) % stride, (-nw) % stride
    top, left = ph // 2, pw // 2
    out = cv2.copyMakeBorder(resized, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


def decode(output0, num_classes, conf, iou):
    """`output0` [4 + nc + nm, N] -> (cajas xywh en el letterbox, confianza, clase, coeficientes),
    ya con umbral de confianza y NMS por clase."""
    pred = output0.T
    scores = pred[:, 4 : 4 + num_classes]
    cls = scores.argmax(axis=1)
    confidence = scores[np.arange(len(scores)), cls]
    keep = confidence > conf
    pred, cls, confidence = pred[keep], cls[keep], confidence[keep]
    if not len(pred):
        return np.zeros((0, 4)), confidence, cls, np.zeros((0, pred.shape[1] - 4 - num_classes))
    boxes = pred[:, :4].astype(np.float64).copy()
    boxes[:, 0] -= boxes[:, 2] / 2    # cx,cy,w,h -> x,y,w,h (esquina sup. izquierda)
    boxes[:, 1] -= boxes[:, 3] / 2
    idx = np.asarray(cv2.dnn.NMSBoxesBatched(boxes.tolist(), confidence.tolist(), cls.tolist(), conf, iou)).reshape(-1)
    return boxes[idx], confidence[idx], cls[idx], pred[idx, 4 + num_classes :]


def paint_instances(shape, boxes, confidence, cls, coefs, protos, scale, pad, class_ids, boxed_width):
    """Máscara de ids (uint8) de tamaño `shape` (h, w) de la imagen original.
    `boxes` (x, y, w, h) están en coordenadas del letterbox (ancho `boxed_width`);
    `protos` [nm, mh, mw]."""
    h, w = shape
    out = np.zeros(shape, np.uint8)
    nm, mh, mw = protos.shape
    step = boxed_width / mw   # los protos están a 1/4 del letterbox
    flat = protos.reshape(nm, -1)
    left, top = pad
    for i in np.argsort(confidence):   # de menos a más segura: la última pintada gana
        x, y, bw, bh = boxes[i]
        x0, y0 = max(int((x - left) / scale), 0), max(int((y - top) / scale), 0)
        x1, y1 = min(int(np.ceil((x + bw - left) / scale)), w), min(int(np.ceil((y + bh - top) / scale)), h)
        if x1 <= x0 or y1 <= y0:
            continue
        logits = (coefs[i] @ flat).reshape(mh, mw).astype(np.float32)
        # destino (píxel de la zona de la caja) -> píxel del proto, solo escala + traslación
        sx = scale / step
        m = np.float32([[sx, 0, (x0 + 0.5) * sx + left / step - 0.5],
                        [0, sx, (y0 + 0.5) * sx + top / step - 0.5]])
        region = cv2.warpAffine(logits, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=-20.0)
        out[y0:y1, x0:x1][region > 0] = class_ids[int(cls[i])]
    return out


class OnnxBackend:
    """Opciones del YAML (todas opcionales salvo `models_glob`):

    models_glob   patrón (o lista) de archivos .onnx; el más reciente arranca cargado
    classes       [{id, name, color: [r, g, b]}]; si falta, salen de la metadata del
                  .onnx (`names`), con ids 1..n (0 es el fondo) y colores por defecto
    conf, iou     umbral de confianza (0.25) y de NMS (0.7), como Ultralytics
    imgsz         lado mayor del letterbox; por defecto el de la metadata (o 640)
    """

    def __init__(self, spec):
        o = spec.options
        if not o.get("models_glob"):
            raise BackendError(f"Backend onnx '{spec.name}': falta la opción 'models_glob'")
        globs = o["models_glob"]
        self.models_globs = [os.path.expanduser(g) for g in ([globs] if isinstance(globs, str) else globs)]
        self.conf, self.iou = float(o.get("conf", 0.25)), float(o.get("iou", 0.7))
        self.imgsz = int(o["imgsz"]) if o.get("imgsz") else None
        self.model = None
        self._session = None
        self._meta = {}
        self._manifest = None
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
        try:
            session = ort.InferenceSession(model_id, providers=ort.get_available_providers())
        except Exception as exc:  # noqa: BLE001 - onnxruntime lanza tipos propios
            raise BackendError(f"No se pudo abrir {model_id}: {exc}") from exc
        manifest = load_manifest(model_id)
        if manifest is not None:
            self._check_semantic(session, manifest, model_id)
            self.info.classes = list(manifest.classes)
            self.info.magnifications = list(manifest.magnifications)
            self._session, self._meta, self._manifest, self.model = session, {}, manifest, model_id
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
        if self._manifest is not None:
            name = self._session.get_inputs()[0].name
            try:
                return semantic.infer(lambda t: self._session.run(None, {name: t})[0], rgb, self._manifest)
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
