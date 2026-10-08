"""Instalación de backends desde la GUI. Sin Qt.

Un backend declara en el YAML una lista `install` de pasos; la GUI los corre
en orden cuando el backend no está instalado. Cada paso es un comando
(`[git, clone, URL, "{cwd}"]`) o un mapa `{cmd: [...], skip_if: ruta}` que se
salta si `ruta` ya existe (así un intento cortado a mitad se puede reanudar).
`{cwd}` es la carpeta del backend; los pasos corren ahí, o en su carpeta
madre si todavía no existe.
"""

import os
import shutil
import subprocess


def is_installed(spec):
    """Hay carpeta (si el backend declara `cwd`) y existe el ejecutable del
    comando: relativo a `cwd` si lleva barra, o en el PATH si no."""
    if spec.cwd and not os.path.isdir(spec.cwd):
        return False
    exe = spec.command[0]
    if os.sep not in exe:
        return shutil.which(exe) is not None
    return os.path.exists(exe if os.path.isabs(exe) else os.path.join(spec.cwd or ".", exe))


def can_install(spec):
    return bool(spec.install)


def _normalize(step):
    if isinstance(step, dict):
        return list(step["cmd"]), step.get("skip_if")
    return list(step), None


def plan(spec):
    """[(comando, skip_if)] con `~` y `{cwd}` ya expandidos."""
    cwd = spec.cwd or ""

    def expand(text):
        return os.path.expanduser(str(text).replace("{cwd}", cwd))

    steps = []
    for raw in spec.install:
        cmd, skip_if = _normalize(raw)
        steps.append(([expand(c) for c in cmd], expand(skip_if) if skip_if else None))
    return steps


def run_install(spec):
    """Corre los pasos pendientes; levanta RuntimeError con el final de la
    salida si alguno falla. Bloquea: llamar fuera del hilo de la UI."""
    for cmd, skip_if in plan(spec):
        if skip_if and os.path.exists(skip_if):
            continue
        workdir = spec.cwd if spec.cwd and os.path.isdir(spec.cwd) else (
            os.path.dirname(spec.cwd.rstrip("/")) if spec.cwd else None)
        if workdir:
            os.makedirs(workdir, exist_ok=True)
        try:
            proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"No se pudo ejecutar {' '.join(cmd)}: {exc}") from exc
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
            raise RuntimeError(f"Falló '{' '.join(cmd)}' (código {proc.returncode}).\n{tail}")
