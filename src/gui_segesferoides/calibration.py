"""Calibración píxel → longitud real, por magnificación, y unidad en que la
GUI muestra las medidas. Sin Qt. Se guarda en un JSON del usuario
(~/.config/gui_segesferoides/calibration.json) y se relee al abrir la app.

Cada magnificación guarda lo que el usuario escribió ("1 px = 1000 µm" o
"1 px = 1 mm"); internamente todo se pasa a µm/px.
"""

import json
import math
import os
from dataclasses import dataclass

UNITS_UM = {"nm": 1e-3, "µm": 1.0, "mm": 1e3, "cm": 1e4}   # unidad -> µm
PX = "px"
DISPLAY_UNITS = [PX, *UNITS_UM]
MAGNIFICATIONS = ["4x", "10x"]   # las que se pueden calibrar siempre


def default_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "gui_segesferoides", "calibration.json")


@dataclass(frozen=True)
class Scale:
    """Cómo mostrar una medida en píxeles: `factor` unidades por píxel."""
    factor: float = 1.0
    unit: str = PX

    def length(self, px):
        return _fmt(px * self.factor, self.unit)

    def area(self, px2):
        return _fmt(px2 * self.factor**2, f"{self.unit}²", digits=0 if self.unit == PX else None)

    def convert_length(self, px):
        return px * self.factor

    def convert_area(self, px2):
        return px2 * self.factor**2


def _fmt(value, unit, digits=None):
    """Número con ~4 cifras significativas (en píxeles, 1 decimal / entero)."""
    if digits is None:
        digits = 1 if unit == PX else max(0, 3 - math.floor(math.log10(abs(value)))) if value else 0
    sep = "" if unit.startswith(PX) else ","
    return f"{value:{sep}.{digits}f} {unit}"


class Calibration:
    def __init__(self, path=None):
        self.path = path or default_path()
        self.entries = {}          # {"4x": (longitud, unidad)}: 1 px = longitud unidad
        self.display_unit = PX
        self.csv_stats = False     # adjuntar la estadística del modelo al CSV exportado
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            entries = {m: (float(e["length"]), e["unit"]) for m, e in data["magnifications"].items()
                       if e["unit"] in UNITS_UM and float(e["length"]) > 0}
            display = data.get("display_unit", PX)
            csv_stats = bool(data.get("csv_stats", False))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return   # sin archivo o dañado: sin calibrar
        self.entries = entries
        self.display_unit = display if display in DISPLAY_UNITS else PX
        self.csv_stats = csv_stats

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {"display_unit": self.display_unit, "csv_stats": self.csv_stats,
                "magnifications": {m: {"length": l, "unit": u} for m, (l, u) in self.entries.items()}}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def set(self, magnification, length, unit):
        """`length` <= 0 borra la calibración de esa magnificación."""
        if length > 0:
            self.entries[magnification] = (float(length), unit)
        else:
            self.entries.pop(magnification, None)

    def um_per_px(self, magnification):
        entry = self.entries.get(magnification)
        return entry[0] * UNITS_UM[entry[1]] if entry else None

    def scale(self, magnification):
        """Escala para mostrar medidas a esa magnificación; en píxeles si no
        está calibrada o se eligió px."""
        um = self.um_per_px(magnification)
        if um is None or self.display_unit == PX:
            return Scale()
        return Scale(um / UNITS_UM[self.display_unit], self.display_unit)
