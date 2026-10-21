# Protocolo de backends (versión 1)

Un backend es un ejecutable que la GUI lanza como proceso hijo. Lee una
petición JSON por línea en **stdin** y responde una línea JSON por petición en
**stdout**, en orden. Todo lo demás (logs, progreso, warnings de librerías) va
a **stderr**: *nada que no sea protocolo puede escribirse en stdout*.

La imagen y la máscara no viajan por el canal: se pasan rutas de archivo.

Petición: `{"id": <int>, "cmd": "<comando>", ...parámetros}`
Respuesta: `{"id": <mismo id>, "ok": true, "result": ...}` o
`{"id": <id>, "ok": false, "error": "mensaje legible"}`.
Un error no mata al backend; el proceso sigue atendiendo.

## Comandos

### `info`
Sin parámetros. Se pide una vez al conectar. Resultado:

```json
{
  "protocol": 1,
  "name": "u-resnet",
  "classes": [
    {"id": 1, "name": "células sueltas", "color": [0, 255, 0]},
    {"id": 2, "name": "esferoides",      "color": [255, 255, 0]}
  ],
  "magnifications": ["4x", "10x"],
  "supports_bbox": true,
  "accepts_model_path": false,
  "supports_reference_stats": true,
  "supports_postprocess": true
}
```

- `classes`: ids enteros **> 0** (0 es siempre el fondo) y color RGB para
  mostrar y guardar. La GUI no asume ninguna clase.
- `accepts_model_path`: si es `true`, `load` acepta una ruta de carpeta
  arbitraria como `model` (la GUI muestra "Examinar...").

### `list_models`
Resultado: lista de `{"id": str, "label": str, "note": str?, "current": bool?}`.
`current: true` marca el modelo que el backend ya tiene cargado (a lo sumo uno).
`note` es un aviso corto para mostrar junto al nombre (p. ej. "en curso").

### `load`
`{"cmd": "load", "model": "<id>"}`. Carga ese modelo y reemplaza al actual.
Resultado: `{"description": "texto para mostrar"}`.

### `predict`
```json
{"cmd": "predict", "image": "/ruta/img.png", "magnification": "10x",
 "bbox": [x, y, w, h], "out": "/tmp/pred.png"}
```
- `image`: ruta absoluta a la imagen original, a resolución nativa.
- `bbox`: `null` para la imagen completa; si no, zona en píxeles nativos.
  **El backend recorta antes de inferir** (nunca se reescala).
- `out`: dónde escribir la máscara. PNG de **1 canal, uint8**, con el id de
  clase en cada píxel y 0 en el fondo; de tamaño `h × w` del `bbox` (o de la
  imagen completa). La GUI rechaza la respuesta si la forma no coincide o
  aparecen ids no declarados en `info`.
- Resultado: `{"mask": "<ruta de la máscara escrita>"}`.
- Opcionales, solo si `info` trae `supports_postprocess: true`:
  - `postprocess`: `{"rules": [...], "fill_holes": [...]}` (formato de abajo);
    reemplaza las reglas del modelo **solo en esta petición**. Si falta, el
    backend usa las del modelo.
  - `discarded_out`: ruta donde escribir un PNG de 1 canal, del mismo tamaño que
    la máscara, con el **id de clase original** de lo que las reglas descartaron
    (0 = nada). El resultado suma `"discarded": "<ruta>"`. La GUI lo muestra
    como contornos rojos, solo informativo.
  - `discarded_reasons` (en el resultado, junto con `discarded`): por cada objeto
    descartado, `{"x", "y"` (un píxel dentro del objeto, en la grilla del PNG),
    `"class"` (original), `"as_class"` (la que tenía al descartarse, si otra regla
    lo reclasificó antes), `"rule"` (nombre), `"conditions": [{"metric", "op",
    "value"` (ya resuelto, en px), `"reference"` (p. ej. `"c1.p5"` o `null`),
    `"actual"` (lo medido)`}]}`. La GUI lo muestra al pasar el mouse sobre el
    contorno rojo.

Todo el procesamiento (padding, normalización, TTA, post-procesado) es
responsabilidad del backend.

### `reference_stats` (opcional)
Solo se pide si `info` trae `supports_reference_stats: true`. Sin parámetros.
Devuelve la estadística de los objetos del set de entrenamiento del modelo
cargado, por magnificación y por id de clase (como string), o `null` si ese
modelo no la tiene. La GUI la usa en el análisis de partícula para mostrar si
una medida cae dentro del intervalo `[p5, p95]`.

```json
{"10x": {"1": {"n": 212,
               "area":        {"mean": 40699.9, "p5": 12648.9, "p95": 85515.6},
               "diameter":    {"mean": 213.5,   "p5": 126.8,   "p95": 330.0},
               "perimeter":   {"mean": 778.8,   "p5": 434.5,   "p95": 1312.0},
               "circularity": {"mean": 0.83,    "p5": 0.53,    "p95": 0.88}}}}
```
Las métricas son opcionales (la GUI muestra solo las que vengan); unidades en
píxeles nativos.

### `get_postprocess` (opcional)
Solo si `info` trae `supports_postprocess: true`. Sin parámetros. Devuelve las
reglas de post-procesado del modelo cargado y las de fábrica:
`{"rules": [...], "fill_holes": [2, 3], "defaults": {"rules": [...], "fill_holes": [...]}}`.
La GUI las muestra en un editor y guarda las que el usuario cambie para
mandarlas en cada `predict`. El backend es el único que las evalúa.

Una regla (se evalúan en orden sobre cada objeto —componente conexo de una
clase predicha— como una cadena: `reclass` cambia la clase y sigue con las
reglas siguientes, que ya ven la clase nueva; `delete` y `keep` terminan):

```json
{"name": "Clase 1 grande y poco circular pasa a clase 2", "enabled": true,
 "class": 1,
 "conditions": [{"metric": "diameter",    "op": "gt", "value": "c1.p95"},
                {"metric": "circularity", "op": "lt", "value": "c1.p5"}],
 "action": "reclass", "to": 2}
```
- `class`: `"any"` o un id de clase.
- `conditions`: todas deben cumplirse. `metric`: `area`, `diameter`,
  `circularity` o `perimeter` (en píxeles nativos); `op`: `lt`, `le`, `gt`, `ge`;
  `value`: número o referencia `"c<id>.<p5|mean|p95>"` a la estadística de
  entrenamiento de esa clase, de la misma métrica y magnificación.
- `action`: `delete` (descarta y termina), `reclass` (con `"to"`; cambia la
  clase y sigue evaluando) o `keep` (lo deja como está y termina: lo protege de
  las reglas siguientes).
- Un error de formato se responde como error; una referencia sin estadística
  hace que esa regla se omita.

### `shutdown`
Responde `ok` y termina el proceso.

## Reglas
- El primer mensaje que recibe un backend recién lanzado es `info`.
- Si el backend arranca con un modelo por defecto, lo marca con `current` en
  `list_models`; si no hay ninguno cargado, la GUI deshabilita *Predecir* hasta
  que el usuario elija uno.
- Cambios incompatibles suben `protocol`; la GUI rechaza versiones que no conoce.
