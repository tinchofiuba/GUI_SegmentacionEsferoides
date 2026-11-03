"""Post-procesado de la máscara predicha, por reglas. Lo hace la GUI, igual para
cualquier modelo (el backend devuelve la máscara cruda). Sin Qt.

Una regla: {"name", "enabled", "class": "any" | id, "conditions": [{"metric", "op",
"value"}] (todas deben cumplirse), "action": "delete" | "reclass" | "keep", "to": id}.
`value` es un número (píxeles) o una referencia "c<id>.<p5|mean|p95>" a la estadística
del dataset (dataset_stats.py) de esa clase, de la misma métrica y magnificación.
Se evalúan en orden sobre cada componente (8-conexo, por clase predicha), como una
cadena: "reclass" cambia la clase y sigue con las reglas siguientes (que ya ven la
clase nueva); "delete" y "keep" terminan. Si ninguna aplica, queda como está.
"""

import copy
import re

import cv2
import numpy as np

METRICS = ("area", "diameter", "circularity", "perimeter")
OPS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
       "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}
ACTIONS = ("delete", "reclass", "keep")
REFERENCE = re.compile(r"c(\d+)\.(p5|mean|p95)")

DEFAULT_POSTPROCESS = {
    "fill_holes": [2, 3],
    "rules": [
        {"name": "Descartar menores al p5 de la clase 1", "enabled": True, "class": "any",
         "conditions": [{"metric": "diameter", "op": "lt", "value": "c1.p5"}],
         "action": "delete"},
        {"name": "Clase 1 grande y poco circular pasa a clase 2", "enabled": True, "class": 1,
         "conditions": [{"metric": "diameter", "op": "gt", "value": "c1.p95"},
                        {"metric": "circularity", "op": "lt", "value": "c1.p5"}],
         "action": "reclass", "to": 2},
    ],
}


def defaults():
    return copy.deepcopy(DEFAULT_POSTPROCESS)


def validate(config):
    """Levanta ValueError con un mensaje legible si `config` ({"rules", "fill_holes"})
    no tiene el formato esperado."""
    if not isinstance(config, dict) or not isinstance(config.get("rules"), list):
        raise ValueError("postprocess debe ser {\"rules\": [...], \"fill_holes\": [...]}")
    for k, rule in enumerate(config["rules"], 1):
        where = f"regla {k}"
        if not isinstance(rule, dict) or not isinstance(rule.get("conditions"), list):
            raise ValueError(f"{where}: falta la lista 'conditions'")
        if rule.get("class", "any") != "any" and not isinstance(rule["class"], int):
            raise ValueError(f"{where}: 'class' debe ser 'any' o un id de clase")
        if rule.get("action") not in ACTIONS:
            raise ValueError(f"{where}: 'action' debe ser una de {ACTIONS}")
        if rule["action"] == "reclass" and not isinstance(rule.get("to"), int):
            raise ValueError(f"{where}: 'reclass' necesita 'to' (id de clase)")
        for cond in rule["conditions"]:
            if cond.get("metric") not in METRICS:
                raise ValueError(f"{where}: métrica inválida {cond.get('metric')!r}")
            if cond.get("op") not in OPS:
                raise ValueError(f"{where}: operador inválido {cond.get('op')!r}")
            value = cond.get("value")
            is_ref = isinstance(value, str) and REFERENCE.fullmatch(value)
            if not (is_ref or (isinstance(value, (int, float)) and not isinstance(value, bool))):
                raise ValueError(f"{where}: valor inválido {value!r} (número o 'c1.p5')")
    for c in config.get("fill_holes", []):
        if not isinstance(c, int):
            raise ValueError("fill_holes debe ser una lista de ids de clase")


def _resolve(value, metric, stats):
    """Número de `value`; None si es una referencia que la estadística no cubre."""
    if not isinstance(value, str):
        return float(value)
    class_id, stat = REFERENCE.fullmatch(value).groups()
    try:
        return float(stats[class_id][metric][stat])
    except (KeyError, TypeError):
        return None


def component_measures(labels, stats, i, needed=("area", "diameter", "circularity", "perimeter")):
    """Medidas (en píxeles) del componente `i` de `cv2.connectedComponentsWithStats`:
    área, diámetro equivalente y, si se piden, perímetro del contorno externo
    (recortando al bbox del componente) y circularidad 4πA/P²."""
    area = float(stats[i, cv2.CC_STAT_AREA])
    out = {"area": area, "diameter": 2 * np.sqrt(area / np.pi)}
    if set(needed) & {"circularity", "perimeter"}:
        x, y, w, h = (int(v) for v in stats[i, :4])
        crop = (labels[max(y - 1, 0): y + h + 1, max(x - 1, 0): x + w + 1] == i).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = max((cv2.arcLength(c, True) for c in contours), default=0.0)
        out["perimeter"] = float(perimeter)
        out["circularity"] = float(4 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0
    return out


def objects_by_class(mask):
    """{id de clase: [medidas de cada objeto]} con la misma definición de objeto y de
    medida que usan las reglas (componentes 8-conexos), para que la estadística del
    dataset y el filtrado sean comparables."""
    out = {}
    for c in (int(v) for v in np.unique(mask) if v != 0):
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == c).astype(np.uint8), connectivity=8)
        out[c] = [component_measures(labels, stats, i) for i in range(1, n)]
    return out


def apply(mask, config, stats=None, reasons=None, warnings=None):
    """Aplica las reglas de `config` a `mask`. `stats` es la estadística de la
    magnificación ({"1": {"diameter": {"p5": ...}}}). Devuelve (máscara limpia,
    descartados): `descartados` tiene, donde una regla `delete` quitó un componente,
    el id de clase que tenía. Una regla cuya referencia la estadística no cubre se
    omite y se anota en `warnings` (lista de textos). Si `reasons` es una lista, se le
    agrega un dict por cada componente descartado: {"x", "y" (un píxel de adentro),
    "class" (original), "as_class" (la que tenía al descartarse, si otra regla lo
    reclasificó antes), "rule" (nombre), "conditions": [{"metric", "op", "value",
    "reference", "actual"}]}."""
    validate(config)
    stats = stats or {}
    rules = []
    for rule in config["rules"]:
        if not rule.get("enabled", True):
            continue
        conds, conds_info = [], []
        for cond in rule["conditions"]:
            value = _resolve(cond["value"], cond["metric"], stats)
            if value is None:
                if warnings is not None:
                    warnings.append(f"regla {rule.get('name') or '?'!r} omitida: sin estadística para "
                                    f"{cond['value']!r}")
                break
            conds.append((cond["metric"], OPS[cond["op"]], value))
            conds_info.append({"metric": cond["metric"], "op": cond["op"], "value": value,
                               "reference": cond["value"] if isinstance(cond["value"], str) else None})
        else:
            rules.append((rule.get("class", "any"), conds, rule["action"], rule.get("to"),
                          rule.get("name", ""), conds_info))

    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    discarded = np.zeros(mask.shape, dtype=np.uint8)
    needed = {m for r in rules for m, _, _ in r[1]}
    for c in (int(v) for v in np.unique(mask) if v != 0):
        num_labels, labels, cstats, _ = cv2.connectedComponentsWithStats(
            (mask == c).astype(np.uint8), connectivity=8)
        for i in range(1, num_labels):
            measures = component_measures(labels, cstats, i, needed) if rules else {}
            final_class = c
            reason = None
            for rule_class, conds, action, to, name, conds_info in rules:
                if rule_class not in ("any", final_class):
                    continue
                if not all(op(measures[m], v) for m, op, v in conds):
                    continue
                if action == "reclass":
                    final_class = to        # sigue con las reglas siguientes
                    continue
                if action == "delete":
                    reason = {"rule": name, "as_class": final_class,
                              "conditions": [{**info, "actual": float(measures[info["metric"]])}
                                             for info in conds_info]}
                    final_class = 0
                break
            where = labels == i
            if final_class == 0:
                discarded[where] = c
                if reasons is not None:
                    ys, xs = np.nonzero(where)
                    reasons.append({"x": int(xs[0]), "y": int(ys[0]), "class": c, **reason})
            else:
                cleaned[where] = final_class

    # Consolidación física: rellenar agujeros dentro de las masas de las clases indicadas
    for c in config.get("fill_holes", []):
        class_mask = (cleaned == c).astype(np.uint8)
        if not class_mask.any():
            continue
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cleaned, contours, -1, c, thickness=cv2.FILLED)
    return cleaned, discarded
