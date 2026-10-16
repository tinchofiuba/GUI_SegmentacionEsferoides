"""Backend falso del protocolo (solo numpy/cv2), para testear la GUI sin
ningún modelo. Predice un círculo de clase 2 centrado en la región pedida.

    python tests/fake_backend.py [--crash-on-predict]
"""

import json
import sys

import cv2
import numpy as np

CRASH = "--crash-on-predict" in sys.argv
INFO = {
    "protocol": 1,
    "name": "fake",
    "classes": [
        {"id": 1, "name": "chicas", "color": [0, 255, 0]},
        {"id": 2, "name": "grandes", "color": [255, 255, 0]},
        {"id": 5, "name": "raras", "color": [0, 255, 255]},
    ],
    "magnifications": ["4x", "10x"],
    "supports_bbox": True,
    "supports_reference_stats": True,
}
REFERENCE = {"4x": {"2": {"n": 10, "area": {"mean": 1500, "p5": 1000, "p95": 2000},
                          "circularity": {"mean": 0.9, "p5": 0.8, "p95": 0.95}},
                    "1": {"n": 30}}}
loaded = None


def handle(req):
    global loaded
    cmd = req["cmd"]
    if cmd == "info":
        return INFO
    if cmd == "list_models":
        return [{"id": "a", "label": "Modelo A"}, {"id": "b", "label": "Modelo B", "note": "en curso"}]
    if cmd == "load":
        if req["model"] == "malo":
            raise ValueError("no existe ese modelo")
        loaded = req["model"]
        return {"description": f"fake:{loaded}"}
    if cmd == "reference_stats":
        return REFERENCE
    if cmd == "predict":
        if CRASH:
            sys.exit(3)
        img = cv2.imread(req["image"])
        if img is None:
            raise ValueError("no se pudo leer la imagen")
        h, w = img.shape[:2]
        x, y, bw, bh = req["bbox"] or (0, 0, w, h)
        mask = np.zeros((bh, bw), np.uint8)
        cv2.circle(mask, (bw // 2, bh // 2), max(1, min(bw, bh) // 4), 2, -1)
        cv2.imwrite(req["out"], mask)
        return {"mask": req["out"]}
    raise ValueError(f"comando desconocido: {cmd}")


def main():
    for line in sys.stdin:
        req = json.loads(line)
        if req["cmd"] == "shutdown":
            print(json.dumps({"id": req["id"], "ok": True, "result": None}), flush=True)
            return
        try:
            resp = {"id": req["id"], "ok": True, "result": handle(req)}
        except Exception as exc:  # noqa: BLE001
            resp = {"id": req["id"], "ok": False, "error": str(exc)}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
