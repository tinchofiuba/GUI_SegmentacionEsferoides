"""Punto de entrada de la GUI.

    uv run gui-segesferoides [--backends config/backends.yaml]
"""

import argparse
import os
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

from gui_segesferoides.config import load_backends, load_index_url
from gui_segesferoides.main_window import MainWindow

DEFAULT_CONFIG = os.path.join("config", "backends.yaml")
FALLBACK_CONFIG = os.path.join("config", "backends.example.yaml")   # sin config local: modelos en ./models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default=DEFAULT_CONFIG,
                        help="YAML con los backends de segmentación disponibles")
    args = parser.parse_args()

    if args.backends == DEFAULT_CONFIG and not os.path.exists(DEFAULT_CONFIG):
        args.backends = FALLBACK_CONFIG

    app = QApplication(sys.argv)
    try:
        specs = load_backends(args.backends)
        index_url = load_index_url(args.backends)
    except (OSError, ValueError) as exc:
        QMessageBox.critical(None, "Configuración de backends", str(exc))
        sys.exit(1)

    window = MainWindow(specs, index_url=index_url)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
