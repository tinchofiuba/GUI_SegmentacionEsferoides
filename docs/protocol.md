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
  "supports_reference_stats": true
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
               "circularity": {"mean": 0.83,    "p5": 0.53,    "p95": 0.88}}}}
```
Las métricas son opcionales (la GUI muestra solo las que vengan); unidades en
píxeles nativos.

### `shutdown`
Responde `ok` y termina el proceso.

## Reglas
- El primer mensaje que recibe un backend recién lanzado es `info`.
- Si el backend arranca con un modelo por defecto, lo marca con `current` en
  `list_models`; si no hay ninguno cargado, la GUI deshabilita *Predecir* hasta
  que el usuario elija uno.
- Cambios incompatibles suben `protocol`; la GUI rechaza versiones que no conoce.
