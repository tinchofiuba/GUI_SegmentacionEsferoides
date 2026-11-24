# GUI_SegmentacionEsferoides

GUI de escritorio (PyQt5) para **predecir, corregir y taggear** máscaras de
segmentación sobre imágenes de microscopía. Es independiente del modelo: solo
corre modelos **ONNX** (con `onnxruntime`, sin torch ni repos de modelos), y cada
`.onnx` lleva al lado un manifiesto `.yaml` que dice cómo usarlo — ver
[docs/model_manifest.md](docs/model_manifest.md).

## Instalación (Windows 10/11, Linux, macOS)

Solo hace falta Python (lo instala `uv`); no se necesita torch, GPU ni el repo del modelo.

1. Instalar [uv](https://docs.astral.sh/uv/getting-started/installation/). En Windows (PowerShell):
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Descargar/clonar este repo, abrir una terminal en su carpeta y correr `uv sync`.
3. Conseguir un modelo: una carpeta con `<modelo>.onnx` y `<modelo>.yaml` (el manifiesto). Lo genera
   el repo del modelo (`u-resnet-export-onnx` / al terminar un entrenamiento) o te lo pasan ya hecho.
4. `uv run gui-segesferoides`, y en la GUI el botón **Instalar modelo...** → elegir el `.onnx`.
   Se copia (con su `.yaml`) a `./models/<modelo>/` y queda cargado. Alternativa manual: copiar la
   carpeta a `models/`.

**Actualizar modelos:** si el YAML de backends tiene `models_index_url` (ver
`config/backends.example.yaml`), aparece **Buscar modelos nuevos**: baja el índice publicado por el repo
del modelo, compara el sha256 de cada `.onnx` con el instalado y descarga los nuevos o cambiados (verifica
la integridad antes de instalar). El más reciente queda cargado. Código: `model_update.py`.

Siempre se lanza desde la raíz del repo (las rutas de `models/` son relativas). Sin
`config/backends.yaml` se usa `config/backends.example.yaml` (modelos en `./models`); para otras
rutas copiarlo a `config/backends.yaml` y editarlo.

## Uso

```bash
uv sync --group dev
uv run gui-segesferoides
```

- Abrir imagen → elegir backend/modelo → *Predecir* (toda la imagen o una zona).
- Corregir: pincel (B), lazo (L), borrador (E), borrar objeto (D), cambiar clase de
  objeto (R), rellenar huecos; Ctrl+Z / Ctrl+Y.
- *Lazo* (L): se marcan puntos con clic (o arrastrando a mano alzada) y la curva
  pasa suavemente por todos, sin rectas. Se cierra con clic sobre el primer punto,
  doble clic o Enter (Esc cancela). El interior se rellena como un objeto de la
  clase elegida en *Clase* (pisa lo que hubiera) y se deshace con un solo Ctrl+Z.
- Analizar objeto (A): click sobre un objeto lo resalta y abre un popup con
  clase (editable), área, perímetro, diámetro equivalente y circularidad, más
  *Rellenar huecos* y *Eliminar* solo para esa partícula. Click de nuevo sobre
  la misma la deselecciona. Medidas en píxeles.
- *Medir distancia*: arrastrar sobre la imagen traza una línea y muestra su largo
  (en la unidad de *Calibrar*, o en px). El combo de la derecha elige el color de la
  próxima medida. *Editar medida*: click sobre una medida abre *Eliminar*, *Cambiar
  color* y *Mover*: arrastrar desde la etiqueta mueve solo la etiqueta; desde
  cualquier otro lado mueve toda la medida (conserva largo, orientación y color). Las medidas son
  anotaciones: no tocan la máscara ni lo que se le manda al modelo.
- *Guardar imagen* (en Vista): guarda la imagen a resolución nativa, como se ve:
  con la máscara (si *Ver máscara* está marcado, con la transparencia elegida) y las
  medidas dibujadas, en PNG, JPEG, TIFF o BMP, a color o en escala de grises (la
  máscara y las medidas conservan su color).
- Rueda = zoom, botón derecho/central = mover, `[` `]` = tamaño de la
  herramienta, F = ajustar, M = ver/ocultar máscara, 1-9 = clase.
- *Aceptar* (Ctrl+S) guarda `<imagen>-Mask.png`: PNG de 1 canal con el id de
  clase de cada píxel (0 = fondo), el formato que lee el entrenamiento de
  u-resnet. Si la imagen termina en `_4x` / `_10x`, ese sufijo se quita del
  nombre (u-resnet busca `nombre-Mask.png` para `nombre_10x.png`). La
  magnificación se detecta del nombre al abrir y solo se usa para predecir.
- *Exportar resultados* escribe un CSV con una fila por objeto (clase y las
  medidas del análisis, en la unidad de *Calibrar*; las calibraciones van en
  `um_per_px_4x` y `um_per_px_10x`) y la misma máscara de ids.

Las clases, colores y magnificaciones los declara el manifiesto del modelo: la GUI no tiene
ninguna clase escrita en el código.

## Tests

```bash
uv run pytest        # usa tests/fake_backend.py (en proceso); los tests con modelos reales se saltan si no están
uv run ruff check .
```
