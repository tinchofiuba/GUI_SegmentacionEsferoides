# GUI_SegmentacionEsferoides

GUI de escritorio (PyQt5) para **predecir, corregir y taggear** máscaras de
segmentación sobre imágenes de microscopía. No contiene ni importa ningún
modelo: habla con *backends* (procesos aparte, cada uno con su propio Python)
mediante un protocolo JSON de una línea por stdin/stdout — ver
[docs/protocol.md](docs/protocol.md).

## Uso

```bash
uv sync --group dev
cp config/backends.example.yaml config/backends.yaml   # ajustar rutas
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
- *Guardar imagen* (en Vista): guarda la imagen a resolución nativa con las medidas
  dibujadas, en PNG, JPEG, TIFF o BMP, a color o en escala de grises (las medidas
  conservan su color). No incluye la máscara.
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

Las clases, colores y magnificaciones los declara el backend: la GUI no tiene
ninguna clase escrita en el código.

## Tests

```bash
uv run pytest        # usa tests/fake_backend.py, no necesita ningún modelo
uv run ruff check .
```
