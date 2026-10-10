"""Application entry point.

Builds the QApplication + MainWindow and installs a global exception hook so
any bug that main_window.py hasn't already turned into a friendly
converter.ConversionError dialog still shows up as a dialog instead of a
silent crash. That matters most for a --windowed PyInstaller build (see
design.md): there is no console for a traceback to land in, so an uncaught
exception would otherwise make the whole app vanish with zero feedback --
exactly the failure mode this project promises its student users it avoids.
"""

import os
import sys
import traceback

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from mconvert.main_window import MainWindow


def _icon_path() -> str:
    """Locate resources/icon.png next to this module, whether running from
    source or from a PyInstaller --onefile build. A frozen build extracts
    its bundled data files under sys._MEIPASS instead of this file's real
    on-disk location, so os.path.dirname(__file__) alone would be wrong
    there; build.spec mirrors this package's own relative layout under
    _MEIPASS so the same "resources/icon.png" suffix works in both cases.
    """
    base_dir = getattr(sys, "_MEIPASS", None)
    if base_dir is not None:
        return os.path.join(base_dir, "mconvert", "resources", "icon.png")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "icon.png")


def install_exception_hook() -> None:
    """Route uncaught exceptions to a dialog instead of Qt's default handling
    (print to stderr, and on some binding/platform combinations terminate
    the process outright with no visible sign of what happened)."""

    def _hook(exc_type, exc_value, exc_traceback):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        sys.stderr.write(text)
        box = QMessageBox(
            QMessageBox.Icon.Critical, "未预期的错误",
            "程序遇到了一个未处理的错误：\n%s" % exc_value,
        )
        box.setDetailedText(text)
        box.exec()

    sys.excepthook = _hook


def main() -> int:
    install_exception_hook()
    app = QApplication(sys.argv)
    # Fusion renders identically across Windows/macOS/Linux and looks more
    # modern than each platform's native style (particularly Windows' dated
    # default widgets) -- without adding a theming dependency for PyInstaller
    # to bundle and for the .exe size to carry.
    app.setStyle("Fusion")

    icon_path = _icon_path()
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    return app.exec()
