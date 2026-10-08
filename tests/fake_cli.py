"""Programa de línea de comandos falso, con la forma de `spheroid-infer`: no
habla el protocolo; imprime la carpeta de salida y deja
<salida>/predictions/<stem>.png (círculo de clase 2) más un overlay.

    python tests/fake_cli.py --checkpoint CK --input IMG [--fail]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--input", required=True)
p.add_argument("--fail", action="store_true")
args = p.parse_args()

if args.fail:
    print("Error: Checkpoint not found", file=sys.stderr)
    sys.exit(1)
img = cv2.imread(args.input)
h, w = img.shape[:2]
out = Path("outputs/infer/run_1")           # relativa al cwd, como spheroid-seg
(out / "predictions").mkdir(parents=True, exist_ok=True)
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (w // 2, h // 2), max(1, min(w, h) // 4), 2, -1)
cv2.imwrite(str(out / "predictions" / f"{Path(args.input).stem}.png"), mask)
print(f"Using checkpoint: {args.checkpoint}\nInference summary\n  Output directory: {out}")
