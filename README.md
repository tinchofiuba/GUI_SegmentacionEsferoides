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
- Corregir: pincel (B), borrador (E), borrar objeto (D), cambiar clase de
  objeto (R), rellenar huecos; Ctrl+Z / Ctrl+Y.
- Analizar partícula (A): click sobre un objeto lo resalta y abre un popup con
  clase (editable), área, perímetro, diámetro equivalente y circularidad, más
  *Rellenar huecos* y *Eliminar* solo para esa partícula. Click de nuevo sobre
  la misma la deselecciona. Medidas en píxeles.
- Rueda = zoom, botón derecho/central = mover, `[` `]` = tamaño de la
  herramienta, F = ajustar, M = ver/ocultar máscara, 1-9 = clase.
- *Aceptar* (Ctrl+S) guarda `<imagen>_Taggeada.png` (un color por clase, según
  la paleta del backend) y `<imagen>_Taggeada.origin.json` con el origen
  (`manual` / `modelo` / `modelo_corregido`).

Las clases, colores y magnificaciones los declara el backend: la GUI no tiene
ninguna clase escrita en el código.

## Tests

```bash
uv run pytest        # usa tests/fake_backend.py, no necesita ningún modelo
uv run ruff check .
```
