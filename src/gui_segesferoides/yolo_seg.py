"""Adaptador de salida `instance-seg` (YOLOv8-seg de Ultralytics exportado a .onnx, sin NMS).

El modelo da *instancias* (caja + clase + máscara) y la GUI trabaja con una máscara de ids por
píxel, así que se pintan las instancias de menor a mayor confianza (gana la más segura).
Sigue a Ultralytics: letterbox al lado mayor = `imgsz` (relleno 114, múltiplo del stride),
NMS por clase, máscaras = coeficientes · protos recortadas a su caja. Todo lo que depende del
modelo (imgsz, stride, conf, iou, clases) viene del manifiesto. Sin Qt ni onnxruntime: `infer`
recibe una función `run(tensor) -> (output0, protos)`.
"""

import cv2
import numpy as np


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


def infer(run, rgb, manifest):
    """Máscara uint8 (h, w) con los ids de clase del manifiesto (0 = fondo). El canal `i` del
    modelo es `manifest.classes[i]` (el orden de la lista es el de las clases del modelo)."""
    boxed, scale, pad = letterbox(rgb, manifest.imgsz, manifest.pad_multiple)
    x = boxed.astype(np.float32) / 255.0
    if manifest.mean is not None:
        x = (x - np.float32(manifest.mean)) / np.float32(manifest.std)
    output0, protos = run(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
    class_ids = [c.id for c in manifest.classes]
    boxes, confidence, cls, coefs = decode(output0[0], len(class_ids), manifest.conf, manifest.iou)
    if not len(boxes):
        return np.zeros(rgb.shape[:2], np.uint8)
    return paint_instances(rgb.shape[:2], boxes, confidence, cls, coefs, protos[0], scale, pad, class_ids,
                           boxed.shape[1])
