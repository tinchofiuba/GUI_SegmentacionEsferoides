"""Configuración de backends (config/backends.yaml)."""

import os
from dataclasses import dataclass, field

import yaml

from gui_segesferoides.backend import SubprocessBackend
from gui_segesferoides.cli_backend import CliBackend


@dataclass(frozen=True)
class BackendSpec:
    name: str
    command: list
    cwd: str = None
    env: dict = field(default_factory=dict)
    install: list = field(default_factory=list)  # pasos para instalarlo (ver installer.py)
    type: str = "protocol"   # "protocol" (docs/protocol.md) | "cli" (cli_backend.py)
    options: dict = field(default_factory=dict)  # resto de las claves (backends cli)


def create_backend(spec):
    """Conecta el backend que describe `spec`."""
    if spec.type == "cli":
        return CliBackend(spec)
    return SubprocessBackend(spec.command, cwd=spec.cwd, env=spec.env)


def load_backends(path):
    """Lee la lista de backends. Formato:

        backends:
          - name: u-resnet
            command: [/ruta/.venv/bin/python, -m, modulo.serve, --flag, valor]
            cwd: /ruta/al/repo        # opcional
            env: {CLAVE: valor}       # opcional
          - name: otro
            type: cli                 # programa sin protocolo; ver cli_backend.py
            command: [...]
            install: [[git, clone, URL, "{cwd}"], [uv, sync]]   # opcional, ver installer.py
            classes: [...]            # y demás opciones del backend cli
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    specs = []
    for entry in data.get("backends", []):
        if "name" not in entry or not entry.get("command"):
            raise ValueError(f"Backend inválido en {path} (faltan 'name' o 'command'): {entry}")
        cwd = entry.get("cwd")
        kind = entry.get("type", "protocol")
        if kind not in ("protocol", "cli"):
            raise ValueError(f"Backend '{entry['name']}': type desconocido {kind!r}")
        known = {"name", "command", "cwd", "env", "type", "install"}
        specs.append(BackendSpec(
            name=entry["name"],
            command=[os.path.expanduser(str(c)) for c in entry["command"]],
            cwd=os.path.expanduser(cwd) if cwd else None,
            env={k: str(v) for k, v in (entry.get("env") or {}).items()},
            install=list(entry.get("install") or []),
            type=kind,
            options={k: v for k, v in entry.items() if k not in known},
        ))
    return specs
