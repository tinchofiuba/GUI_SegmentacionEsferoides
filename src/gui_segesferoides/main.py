"""Punto de entrada de la GUI.

    uv run gui-segesferoides [--backends config/backends.yaml]
"""

import argparse
import os
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

from gui_segesferoides.config import load_backends
from gui_segesferoides.main_window import MainWindow

DEFAULT_CONFIG = os.path.join("config", "backends.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default=DEFAULT_CONFIG,
                        help="YAML con los backends de segmentación disponibles")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    try:
        specs = load_backends(args.backends)
    except (OSError, ValueError) as exc:
        QMessageBox.critical(None, "Configuración de backends", str(exc))
        sys.exit(1)

    window = MainWindow(specs)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
