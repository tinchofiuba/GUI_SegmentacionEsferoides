# Traspaso: cómo conectar la GUI con varios modelos (2026-10-08)

> Primero leer la sección **"Estado al cierre de la sesión"** más abajo: lo de
> arriba es el análisis inicial y algunas partes ya se implementaron.

Resumen de la conversación sobre hacer que la GUI consuma modelos de otros
repos **sin saber que existen**, instalándolos con pip/uv. Escrito para quien
retome el trabajo desde este repo (GUI). Estado al 2026-10-08; **nada de esto
está commiteado** en ningún repo salvo lo que el usuario haya commiteado después.

## Los tres repos

| Repo | Ruta | Rol |
|---|---|---|
| GUI | `/home/martin/repos/GUI_SegmentacionEsferoides` | PyQt5, paquete `gui_segesferoides`. Sin torch ni jax. |
| Modelo propio | `/home/martin/repos/SegmentacionEsferoides` | U-Net PyTorch (`u_resnet_segesferoides`), Python >=3.10. Remoto: `tinchofiuba/SegmentacionEsferoides`. |
| spheroid-seg | `/home/martin/repos/spheroid-seg` | JAX/Flax, Python >=3.12,<3.15. **Repo de otra persona** (Edgardo Marchi, remoto `edgardomarchi/spheroid-seg`). |

Regla pedida por el usuario: la GUI no sabe nada de los modelos y los modelos
no saben nada de la GUI.

## Decisión de arquitectura: independencia por procesos

No se puede instalar todo en un mismo entorno (Python 3.10 vs >=3.12, y
PyTorch + JAX juntos son pesados y es probable que choquen en numpy/opencv).
Por eso cada modelo se instala en **su propio entorno aislado** y la GUI solo
conoce un comando:

```bash
uv tool install git+ssh://git@github.com/tinchofiuba/SegmentacionEsferoides
# → entorno propio + comando en el PATH (hoy falta registrar u-resnet-serve, ver pendientes)
```
```yaml
# config/backends.yaml de la GUI
- name: u-resnet
  command: [u-resnet-serve, --models-dir, ~/modelos/u-resnet]
```

`uv tool install` / `pipx` de verdad no se probó todavía; lo que sí se
verificó es que `uv build --wheel` del repo del modelo funciona y que la
rueda incluye `serve.py`.

## Protocolo v1 (ya implementado)

La GUI lanza el backend como proceso hijo y habla JSON de una línea por
stdin/stdout (imagen y máscara van por ruta de archivo). Comandos: `info`,
`list_models`, `load`, `predict`, `shutdown`. Especificación completa:
[protocol.md](protocol.md). Cliente: `src/gui_segesferoides/backend.py`
(`SubprocessBackend`). Backend falso para tests: `tests/fake_backend.py`.
La GUI no tiene clases escritas: ids, nombres y colores los declara el
backend en `info`.

Lado modelo propio: `src/u_resnet_segesferoides/serve.py`
(`python -m u_resnet_segesferoides.serve --models-dir outputs/models
[--model-name insitu_3t3 | --run-dir RUN]`). Verificado: da la **misma
máscara** que el código previo de la GUI en una zona con esferoides, y
funcionó de punta a punta con la GUI (zona 1024 px: 4,7 s; imagen
6128×8168: ~228 s en CPU).

## spheroid-seg: qué expone realmente

Leído de `src/spheroid_seg/infer.py` y su README:

- Instalable con pip (hatchling, `src/`), comandos `spheroid-train`,
  `spheroid-eval`, `spheroid-infer`.
- Entrada: `python -m spheroid_seg.infer --config X.yaml --input <archivo|carpeta>
  [--run-dir R | --checkpoint C]`. Necesita su config (`patch_size`,
  `num_classes`, `input_channels`).
- Salida: `<checkpoints_dir>/../infer/<config>_<timestamp>/predictions/<nombre>.png`,
  PNG de 1 canal `uint8` con id de clase, **del mismo tamaño que la entrada**;
  también `overlays/`. Por stdout solo imprime texto suelto.
- Procesa por mosaicos solapados (`_predict_stitched`), así que maneja
  imágenes grandes.
- **No** hay función pública `predict(imagen) -> máscara` (lo reutilizable
  tiene guion bajo), **no** acepta bbox ni magnificación, y cada llamada
  arranca un proceso nuevo.
- Semántica de clases distinta: entrena con 3 clases (fondo / célula suelta /
  agregado; esferoide y organoide fusionados). La GUI lo soporta porque cada
  backend declara las suyas, pero no son equivalentes a las del modelo propio.

### Medición de latencia (CPU, esta máquina, una corrida por caso)

| Entrada | Tiempo por llamada |
|---|---|
| Recorte 256×256 | 7,1 s |
| Recorte 1024×1024 | 9,5 s (2.ª vez: 9,4 s) |
| Imagen completa 6128×8168 | 85 s |

Hay ~7 s de costo fijo por llamada (arranque, JAX, compilación) que **no se
amortiza** entre llamadas. Son órdenes de magnitud, no números finos.

**Advertencia:** el único checkpoint disponible
(`spheroid-seg/outputs/runs/base_20260812_124531`) es una corrida de **una
sola época** (dice ≈ 0) y no se sabe si usó datos reales o sintéticos. Sirve
para medir tiempos, **no** para comparar calidad.

## Opciones consideradas y recomendación

1. **Backend `cli` genérico en la GUI (recomendado como primer paso).** Un
   segundo tipo de backend, configurado por YAML: comando con plantillas
   (`{image}`), patrón para encontrar la máscara (`mask_glob`) y clases
   declaradas a mano (un comando no puede declararlas). La GUI recorta ella
   misma la zona y pasa el recorte (se sigue infiriendo a escala nativa). No
   requiere tocar spheroid-seg. Costo: ~7–10 s por predicción.
2. **Adaptador persistente** (`spheroid-seg-serve`, paquete aparte que dependa
   de spheroid-seg y hable el protocolo). Paga el arranque una sola vez; no
   toca el repo ajeno (la alternativa es pedirle a su autor un
   `spheroid-serve` o mandarle un PR). Hacerlo solo si los 7–10 s molestan.
3. **Mantener el protocolo como vía principal** para el modelo propio (rápido,
   con zonas y selección de modelos).
4. Idea para más adelante: mover la especificación del protocolo y un test de
   conformidad a un tercer paquete neutral, porque hoy el protocolo está
   documentado en el repo de la GUI.

Se discutió también alinear el CLI del modelo propio con el de spheroid-seg
(mismos argumentos; salida `predictions/<nombre>.png` en ids de grises) para
que **un solo adaptador sirva a los dos**. Hoy `u-resnet-infer` guarda
máscaras a color en otra carpeta. Cambiar eso toca la salida de
`inference.py` (no la lógica del modelo), pero el CLAUDE.md del repo del
modelo exige que se pida explícitamente.

## Estado al cierre de la sesión (2026-10-08, noche)

**Hecho en esta sesión (todo sin commitear, 43 tests y ruff en verde):**

- **Herramienta "Analizar partícula (A)"**: click selecciona un objeto (contorno
  magenta de 2 px, también los huecos internos), segundo click lo deselecciona.
  Popup no modal con clase (editable), área, perímetro (solo contorno externo),
  diámetro equivalente y circularidad, en píxeles; botones *Rellenar huecos*,
  *Eliminar* (con confirmación) y *Cerrar*. `particle_dialog.py`, `editing.py`.
  Se corrigió de paso `fill_holes` (el fondo que toca el borde ya no cuenta como hueco).
- **Backend `cli`** (`src/gui_segesferoides/cli_backend.py`): adapta un programa
  sin protocolo (spheroid-seg). Opciones en el YAML: `command` con `{image}` /
  `{model}`, `classes`, `models_glob`, `mask`, `output_dir_regex`, `cleanup`.
  La GUI recorta la zona ella misma; no hay selector de magnificación.
  Verificado solo contra `tests/fake_cli.py`, **no contra `spheroid-infer` real**.
- **Botón "Instalar backend"** (`installer.py`): si el backend elegido no está
  instalado, aparece a la derecha del combo de modelos; corre los pasos
  `install` del YAML (`git clone` + `uv sync`, con `skip_if` para reanudar).
  Ya se usó con spheroid-seg: el repo quedó clonado en `~/repos/spheroid-seg`.
- `config/backends.yaml` (ignorado por git) y `backends.example.yaml` tienen
  los dos backends (`u-resnet` por protocolo, `spheroid-seg` por `cli`).
- Repo del modelo propio: se registró `u-resnet-serve` en `[project.scripts]`
  (una línea, sin commit, **sin `uv sync`** todavía).
- **No se tocó el repo spheroid-seg** (es de Edgardo Marchi): solo se clonó.
  Regla del usuario: nada de commits ni pushes ahí.

**Dónde quedamos:** spheroid-seg está instalado pero **no hay ningún checkpoint**
(el anterior se borró con el clon viejo), así que el selector de modelos
queda vacío y no se puede predecir con él.

## Pendientes, en orden sugerido

1. **Conseguir un checkpoint de spheroid-seg.** Caminos:
   a. Pedírselo a Edgardo (probablemente tampoco tenga uno real: su README dice
      que solo está validado con datos sintéticos).
   b. **Entrenarlo con las imágenes 3T3 propias** (recomendado: los splits ya
      son idénticos a los de spheroid-seg, 68/15/13). Falta un **script de
      conversión** de `~/repos/SegmentacionEsferoides/data/insitu_3t3/`:
      spheroid-seg espera `data/raw/<nombre>_10x.<ext>` y
      `data/masks/<nombre>_10x.png` en **grises con id de clase** (1 suelta,
      2 esferoide, 3 organoide; 2 y 3 los fusiona él), mientras que las
      máscaras propias son **a color** y se llaman `<nombre>-Mask.png`
      (BGR: verde (0,255,0)→1, amarillo (0,255,255)→2, cian (255,255,0)→3;
      ver `MASK_COLOR_TO_CLASS` en `dataset.py` del repo propio). Escribir
      máscaras convertidas y symlinks de las imágenes en
      `~/repos/spheroid-seg/data/{raw,masks}` (carpeta ignorada por su git;
      `data/splits/*.txt` y `metadata.csv` ya vienen en su repo). Después
      `spheroid-train --config configs/base.yaml`. Esta máquina no tiene GPU
      (12 CPUs): entrenar en Colab (`notebooks/colab_training.ipynb`).
   c. Camino de prueba rápida: `configs/tiny.yaml --epochs 5` con datos
      sintéticos (README, "Quickstart"). Sirve para validar la integración de
      punta a punta, no para comparar calidad.
   Pregunta pendiente al usuario: ¿entrenar local o en Colab?
2. **Probar a mano el backend `cli` contra `spheroid-infer` real** en la GUI
   (con el checkpoint del punto 1) y confirmar que la máscara vuelve bien,
   que `cleanup` borra `outputs/infer/<run>` y medir la latencia (~7–10 s).
3. **Modelo propio:** `uv sync` para que exista el comando `u-resnet-serve`
   (hoy registrado pero no instalado) y probar `uv tool install .` de punta a
   punta con la GUI. Se dejó sin hacer mientras corría un entrenamiento.
4. **Pesos del modelo propio:** un clon/`pip install` da el código, no los pesos
   (`outputs/` está gitignoreado). Decidir dónde alojarlos (p. ej. GitHub
   Releases) si se quiere que el botón "Instalar backend" deje todo listo. Hoy
   la URL de instalación del modelo propio es SSH; usar `https` en otras máquinas.
5. Decidir si hace falta el adaptador persistente (`spheroid-seg-serve`, opción 2
   de arriba) según cómo se sientan los 7–10 s por predicción.
6. Comparar resultados de los dos modelos en la GUI (objetivo final del usuario).
   Falta verificar que crop, augmentations y uso de val coincidan entre repos.
7. Traer del repo spheroid-seg al modelo propio: versiones acotadas de
   dependencias críticas, `ruff format` y CI en GitHub Actions.
8. Posible mejora si hace falta: perímetro del popup que sume también los bordes
   de los huecos (hoy solo el externo); medidas en µm (necesita escala).
9. **Commitear.** La GUI entera sigue sin commitear (incluye `uv.lock`);
   también el cambio de una línea en el `pyproject.toml` del modelo propio.

## Contexto relacionado (ya hecho, en el repo del modelo)

- El split train/val/test del modelo propio es ahora **idéntico al de
  spheroid-seg** (68/15/13); el anterior (seed=42, 58/28/10) está en
  `splits/insitu_3t3_seed42/`. Las métricas previas al 2026-10-06 no son
  comparables con las nuevas. Falta verificar que crop, augmentations y uso de
  val coincidan entre repos para una comparación realmente limpia.
- No hubo cambios en la lógica del modelo desde el commit `f648f0c`: solo
  selección de runs (`find_latest_run_dir`), el nuevo `serve.py` y los splits.

## Comandos útiles

```bash
# GUI
cd /home/martin/repos/GUI_SegmentacionEsferoides
uv run gui-segesferoides            # lee config/backends.yaml
uv run pytest -q && uv run ruff check .

# Backend del modelo propio, a mano
cd /home/martin/repos/SegmentacionEsferoides
uv run python -m u_resnet_segesferoides.serve --models-dir outputs/models --model-name insitu_3t3

# spheroid-seg (su propio venv, Python 3.12)
cd /home/martin/repos/spheroid-seg
.venv/bin/python -m spheroid_seg.infer --config configs/base.yaml \
    --checkpoint outputs/runs/base_20260812_124531/checkpoints/best_checkpoint.msgpack \
    --input <imagen o carpeta>
```
