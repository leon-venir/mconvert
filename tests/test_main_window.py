"""Headless functional tests for SingleFileTab.

Requires PySide6, which is not part of the base dev environment on this
machine (see design.md) -- these tests are skipped automatically if it is not
importable. Run with a Python that has PySide6 installed, e.g.:

    QT_QPA_PLATFORM=offscreen /path/to/env/bin/python -m unittest tests.test_main_window
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtWidgets import QApplication, QMessageBox
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

if PYSIDE6_AVAILABLE:
    from mconvert import formats
    from mconvert.main_window import MainWindow, SingleFileTab

    class _FakeDropEvent:
        """Duck-typed stand-in for QDropEvent/QDragEnterEvent: our handlers
        only call mimeData()/acceptProposedAction()/ignore(), so a real Qt
        drag session (which is finicky and platform-dependent to simulate
        headlessly) isn't needed to exercise the actual logic."""

        def __init__(self, local_paths):
            self._mime_data = QMimeData()
            self._mime_data.setUrls([QUrl.fromLocalFile(p) for p in local_paths])
            self.accepted = False
            self.ignored = False

        def mimeData(self):
            return self._mime_data

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.ignored = True

FIXTURE_POSCAR = (
    Path(__file__).resolve().parent.parent.parent
    / "mkits" / "test" / "phonopy4zpe" / "vasp" / "freq_h2o" / "POSCAR"
)


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed in this environment")
class TestSingleFileTab(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = SingleFileTab()

    def _select_target_format(self, key):
        index = list(formats.OUTPUT_FORMATS).index(key)
        self.tab.output_options.target_format_combo.setCurrentIndex(index)

    def test_controls_disabled_before_any_file_is_loaded(self):
        self.assertFalse(self.tab.preview_button.isEnabled())
        self.assertFalse(self.tab.convert_button.isEnabled())

    def test_loading_valid_file_populates_summary_and_enables_controls(self):
        self.tab._load(str(FIXTURE_POSCAR))

        self.assertTrue(self.tab.preview_button.isEnabled())
        self.assertTrue(self.tab.convert_button.isEnabled())
        self.assertIn("poscar", self.tab.detected_format_label.text().lower())
        self.assertEqual(self.tab.formula_label.text(), "H2O")
        self.assertEqual(self.tab.total_atoms_label.text(), "3")

    def test_loading_missing_file_shows_warning_and_keeps_controls_disabled(self):
        with mock.patch.object(QMessageBox, "warning") as warning:
            self.tab._load("/no/such/file/POSCAR")
        warning.assert_called_once()
        self.assertFalse(self.tab.preview_button.isEnabled())

    def test_switching_to_cif_disables_frac_and_dyn_options(self):
        self.tab._load(str(FIXTURE_POSCAR))
        self._select_target_format("cif")
        self.assertFalse(self.tab.output_options.frac_radio.isEnabled())
        self.assertFalse(self.tab.output_options.cart_radio.isEnabled())
        self.assertFalse(self.tab.output_options.dyn_checkbox.isEnabled())

    def test_switching_to_poscar_enables_frac_and_dyn_options(self):
        self.tab._load(str(FIXTURE_POSCAR))
        self._select_target_format("cif")
        self._select_target_format("poscar")
        self.assertTrue(self.tab.output_options.frac_radio.isEnabled())
        self.assertTrue(self.tab.output_options.dyn_checkbox.isEnabled())

    def test_default_output_path_tracks_target_format_until_user_edits_it(self):
        self.tab._load(str(FIXTURE_POSCAR))
        self._select_target_format("cif")
        self.assertTrue(self.tab.output_path_edit.text().endswith("structure.cif"))

        self.tab.output_path_edit.setText("/tmp/custom_name.cif")
        self.tab._on_output_path_edited("/tmp/custom_name.cif")
        self._select_target_format("xyz")
        # user-edited path must not be silently overwritten by the new default
        self.assertEqual(self.tab.output_path_edit.text(), "/tmp/custom_name.cif")

    def test_preview_shows_generated_content(self):
        self.tab._load(str(FIXTURE_POSCAR))
        self._select_target_format("cif")
        self.tab._on_preview()
        self.assertIn("_cell_length_a", self.tab.preview_text.toPlainText())

    def test_convert_writes_file_and_shows_success_message(self):
        self.tab._load(str(FIXTURE_POSCAR))
        self._select_target_format("xyz")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.xyz")
            self.tab.output_path_edit.setText(out_path)
            with mock.patch.object(QMessageBox, "information") as information:
                self.tab._on_convert()
            information.assert_called_once()
            self.assertTrue(Path(out_path).read_text().startswith("3\n"))

    def test_convert_asks_before_overwriting_and_respects_no(self):
        self.tab._load(str(FIXTURE_POSCAR))
        self._select_target_format("xyz")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.xyz")
            Path(out_path).write_text("original content")
            self.tab.output_path_edit.setText(out_path)
            with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
                self.tab._on_convert()
            self.assertEqual(Path(out_path).read_text(), "original content")

    def test_convert_without_output_path_warns(self):
        self.tab._load(str(FIXTURE_POSCAR))
        self.tab.output_path_edit.setText("")
        with mock.patch.object(QMessageBox, "warning") as warning:
            self.tab._on_convert()
        warning.assert_called_once()

    def test_drag_enter_accepts_a_single_local_file(self):
        event = _FakeDropEvent([str(FIXTURE_POSCAR)])
        self.tab.dragEnterEvent(event)
        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)

    def test_drag_enter_rejects_a_directory_or_multiple_files(self):
        for paths in ([str(FIXTURE_POSCAR.parent)], [str(FIXTURE_POSCAR), str(FIXTURE_POSCAR)]):
            event = _FakeDropEvent(paths)
            self.tab.dragEnterEvent(event)
            self.assertFalse(event.accepted)
            self.assertTrue(event.ignored)

    def test_dropping_a_file_loads_it(self):
        self.tab.dropEvent(_FakeDropEvent([str(FIXTURE_POSCAR)]))
        self.assertEqual(self.tab.formula_label.text(), "H2O")

    def test_dropping_a_directory_does_nothing(self):
        self.tab.dropEvent(_FakeDropEvent([str(FIXTURE_POSCAR.parent)]))
        self.assertFalse(self.tab.preview_button.isEnabled())


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed in this environment")
class TestMainWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_has_all_tabs(self):
        window = MainWindow()
        self.assertEqual(window.tabs.count(), 3)
        self.assertEqual(window.tabs.tabText(0), "单文件转换")
        self.assertEqual(window.tabs.tabText(1), "批量转换")
        self.assertEqual(window.tabs.tabText(2), "实/倒空间")

    def test_loading_in_the_single_file_tab_feeds_the_space_viewer(self):
        window = MainWindow()
        window.single_file_tab._load(str(FIXTURE_POSCAR))
        self.assertEqual(len(window.space_viewer_tab._pick_payload), 3)


if __name__ == "__main__":
    unittest.main()
