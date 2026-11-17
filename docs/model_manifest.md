# Manifiesto de modelo

La GUI corre modelos `.onnx` y no conoce ningún repo de modelos. Para saber cómo usar un modelo
lee un **manifiesto**: un `.yaml` con el mismo nombre que el `.onnx`, en la misma carpeta
(`UResNet_1006.onnx` → `UResNet_1006.yaml`). Solo datos, nunca código: un modelo nuevo o reentrenado
se usa sin tocar la GUI mientras su salida sea de un tipo conocido.

Código: `model_manifest.py` (lectura y validación), `semantic.py` y `yolo_seg.py` (adaptadores de
salida), `onnx_backend.py` (el motor).

## Campos comunes

```yaml
schema: 1                      # la GUI rechaza versiones que no conoce
name: insitu_3t3               # opcional; se muestra como descripción del modelo
output: semantic               # semantic | instance-seg
input:
  color: rgb
  normalize: imagenet          # imagenet | unit (x/255, por defecto) | {mean: [3], std: [3]}
  pad_multiple: 32             # relleno negro abajo/derecha (semantic) o stride del letterbox (instance-seg)
classes:                       # ids > 0 (0 es el fondo), sin repetir; color RGB
  - {id: 1, name: células sueltas, color: [0, 255, 0]}
magnifications: [4x, 10x]      # opcional
```

## `output: semantic` (U-Net, U-ResNet...)

El modelo devuelve logits `[1, C, H, W]`; la máscara es el argmax por píxel. **El canal `k` de los
logits es la clase con id `k`** (el canal 0 es el fondo).

```yaml
input:
  tta: [hflip, vflip, hvflip]  # opcional: promedia los logits de estas vistas además de la normal
```

Imágenes grandes: la GUI infiere por mosaicos (opciones `tile`, `tile_overlap`, `workers` del backend
en `config/backends.yaml`), así la memoria no depende del tamaño de la imagen.

## `output: instance-seg` (YOLOv8-seg de Ultralytics, sin NMS en el grafo)

El modelo devuelve detecciones `[1, 4 + nc + nm, N]` y protos de máscara `[1, nm, mh, mw]`; la GUI
pinta las instancias de menor a mayor confianza. **El canal `i` del modelo es la `i`-ésima clase de
la lista `classes`** (el orden de la lista importa; los `id` pueden ser cualquiera).

```yaml
input:
  imgsz: 1920                  # obligatorio: lado mayor del letterbox
decode:
  conf: 0.25                   # umbral de confianza
  iou: 0.7                     # umbral de NMS por clase
```

## `options`: controles del modelo

Opcional. Cada entrada crea un control en el panel "Predicción" de la GUI. El `id` debe ser una
opción que el adaptador del `output` sabe aplicar; el tipo de control lo fija el id:

| output | id | control | default si no se pone |
|---|---|---|---|
| `semantic` | `tta` | checkbox (requiere `input.tta`) | `true` |
| `instance-seg` | `conf` | número en [0.01, 0.99] | `decode.conf` |
| `instance-seg` | `iou` | número en [0.01, 1.0] | `decode.iou` |

```yaml
options:
  - {id: tta, label: Utilizar TTA}
  - {id: conf, label: Confianza, default: 0.4}
```

La GUI no interpreta las opciones: al cambiar un control llama a `backend.set_option(id, valor)`, y el
adaptador aplica el valor en la siguiente predicción. Sin `options` el modelo no muestra controles.

## Al cargar

La GUI hace una prueba de humo (corre una imagen vacía) y rechaza con un mensaje claro un manifiesto
que no corresponde al modelo: salidas con otra forma, menos canales que clases, `schema` desconocido...

## Modelos sin manifiesto

Un `.onnx` sin `.yaml` solo se entiende si es un YOLOv8-seg de Ultralytics, y con lo que diga su
metadata. Es un camino heredado: conviene escribir el manifiesto.
