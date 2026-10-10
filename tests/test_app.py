"""Tests for mconvert.app: the process entry point's global exception hook."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QPushButton
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

if PYSIDE6_AVAILABLE:
    from mconvert.app import install_exception_hook


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed in this environment")
class TestInstallExceptionHook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._original_excepthook = sys.excepthook
        install_exception_hook()

    def tearDown(self):
        sys.excepthook = self._original_excepthook

    def test_hook_shows_a_dialog_with_the_exception_and_traceback(self):
        with mock.patch("mconvert.app.QMessageBox") as message_box_cls:
            try:
                raise ValueError("boom")
            except ValueError:
                sys.excepthook(*sys.exc_info())

        message_box_cls.assert_called_once()
        self.assertIn("boom", message_box_cls.call_args[0][2])
        instance = message_box_cls.return_value
        instance.setDetailedText.assert_called_once()
        self.assertIn("ValueError: boom", instance.setDetailedText.call_args[0][0])
        instance.exec.assert_called_once()

    def test_hook_actually_fires_for_an_exception_raised_inside_a_qt_slot(self):
        # PySide6/PyQt exception propagation out of a slot is a known
        # cross-binding/version inconsistency (older bindings just print to
        # stderr and swallow it); verify it genuinely reaches sys.excepthook
        # here instead of assuming it does.
        button = QPushButton()

        def _boom():
            raise RuntimeError("slot boom")

        button.clicked.connect(_boom)

        with mock.patch("mconvert.app.QMessageBox") as message_box_cls:
            button.click()

        message_box_cls.assert_called_once()
        self.assertIn("slot boom", message_box_cls.call_args[0][2])


if __name__ == "__main__":
    unittest.main()
