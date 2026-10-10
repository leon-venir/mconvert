"""Headless functional tests for BatchTab.

Same PySide6 requirement/skip behaviour as test_main_window.py.
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
    from mconvert.convert_worker import BatchConvertWorker
    from mconvert.main_window import BatchTab

    def _run_worker_synchronously(self):
        """Stand-in for QThread.start(): calling run() directly never fires
        QThread's own `finished` signal (that's wired to the real thread
        lifecycle started by start()), so emit it by hand to keep
        _on_worker_finished's wiring exercised deterministically."""
        self.run()
        self.finished.emit()

    class _FakeDropEvent:
        """See test_main_window.py's _FakeDropEvent for why a real QDropEvent
        is not needed to test this logic."""

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

FIXTURE_ADSORPT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "mkits" / "test" / "adsorpt"
)


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed in this environment")
class TestBatchTabScanning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = BatchTab()

    def _choose_folder(self, path):
        # bypass the QFileDialog, same pattern as SingleFileTab's tests calling _load() directly
        self.tab._input_root = path
        self.tab._refresh_output_dir_default()
        self.tab._scan()

    def test_start_disabled_until_a_folder_with_files_is_chosen(self):
        self.assertFalse(self.tab.start_button.isEnabled())

    def test_progress_bar_shows_0_percent_before_any_batch_runs(self):
        # QProgressBar's own default value is -1 ("no value"), which the
        # Fusion style renders as a blank bar with no "0%" text at all --
        # regression test for explicitly initializing it in _build_ui().
        self.assertEqual(self.tab.progress_bar.text(), "0%")

    def test_scanning_populates_table_with_pending_rows(self):
        self._choose_folder(str(FIXTURE_ADSORPT_DIR))
        self.assertEqual(self.tab.table.rowCount(), 3)
        for row in range(3):
            self.assertEqual(self.tab.table.item(row, BatchTab.COLUMN_FORMAT).text(), "-")
            self.assertEqual(self.tab.table.item(row, BatchTab.COLUMN_STATUS).text(), "待处理")
        self.assertTrue(self.tab.start_button.isEnabled())

    def test_output_dir_defaults_to_converted_under_input_root_until_edited(self):
        self._choose_folder(str(FIXTURE_ADSORPT_DIR))
        self.assertEqual(
            self.tab.output_dir_edit.text(), os.path.join(str(FIXTURE_ADSORPT_DIR), "converted"),
        )
        self.tab.output_dir_edit.setText("/tmp/custom_output")
        self.tab._on_output_dir_edited("/tmp/custom_output")
        self.tab.recursive_checkbox.setChecked(False)  # triggers a re-scan
        self.assertEqual(self.tab.output_dir_edit.text(), "/tmp/custom_output")

    def test_start_without_output_dir_warns(self):
        self._choose_folder(str(FIXTURE_ADSORPT_DIR))
        self.tab.output_dir_edit.setText("")
        with mock.patch.object(QMessageBox, "warning") as warning:
            self.tab._on_start()
        warning.assert_called_once()

    def test_start_with_nonempty_output_dir_asks_and_respects_no(self):
        self._choose_folder(str(FIXTURE_ADSORPT_DIR))
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "leftover.txt").write_text("from a previous run")
            self.tab.output_dir_edit.setText(tmpdir)
            with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
                self.tab._on_start()
            self.assertIsNone(self.tab._worker)

    def test_drag_enter_accepts_a_single_local_directory(self):
        event = _FakeDropEvent([str(FIXTURE_ADSORPT_DIR)])
        self.tab.dragEnterEvent(event)
        self.assertTrue(event.accepted)

    def test_drag_enter_rejects_a_single_file_or_multiple_paths(self):
        single_file = str(FIXTURE_ADSORPT_DIR / "tio2_h2o_ads_1.vasp")
        for paths in ([single_file], [str(FIXTURE_ADSORPT_DIR), str(FIXTURE_ADSORPT_DIR)]):
            event = _FakeDropEvent(paths)
            self.tab.dragEnterEvent(event)
            self.assertFalse(event.accepted)
            self.assertTrue(event.ignored)

    def test_dropping_a_folder_scans_it(self):
        self.tab.dropEvent(_FakeDropEvent([str(FIXTURE_ADSORPT_DIR)]))
        self.assertEqual(self.tab.table.rowCount(), 3)
        self.assertEqual(self.tab.folder_label.text(), str(FIXTURE_ADSORPT_DIR))


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed in this environment")
class TestBatchTabRun(unittest.TestCase):
    """Runs the worker synchronously (patching QThread.start to call run()
    directly) so the whole GUI -> worker -> GUI signal loop is deterministic
    without needing a live Qt event loop."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_full_batch_updates_table_progress_and_shows_summary(self):
        tab = BatchTab()
        tab._input_root = str(FIXTURE_ADSORPT_DIR)
        tab._scan()
        tab.output_options.target_format_combo.setCurrentIndex(list(formats.OUTPUT_FORMATS).index("cif"))

        with tempfile.TemporaryDirectory() as tmpdir:
            tab.output_dir_edit.setText(tmpdir)
            with mock.patch.object(BatchConvertWorker, "start", _run_worker_synchronously), \
                 mock.patch.object(QMessageBox, "information") as information:
                tab._on_start()

            information.assert_called_once()
            self.assertIn("成功 3", information.call_args[0][2])
            self.assertEqual(tab.progress_bar.value(), 3)
            for row in range(3):
                self.assertEqual(tab.table.item(row, BatchTab.COLUMN_STATUS).text(), "成功")
                self.assertEqual(tab.table.item(row, BatchTab.COLUMN_FORMAT).text(), "POSCAR (VASP)")
            self.assertEqual(len(list(Path(tmpdir).glob("*.cif"))), 3)
            self.assertTrue(tab.start_button.isEnabled())
            self.assertFalse(tab.cancel_button.isEnabled())

    def test_bad_file_in_batch_shows_error_in_its_row_without_stopping_others(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            (input_dir / "not_a_structure.txt").write_text("garbage")
            for name in ("tio2_h2o_ads_1.vasp", "tio2_h2o_ads_2.vasp"):
                (input_dir / name).write_bytes((FIXTURE_ADSORPT_DIR / name).read_bytes())

            tab = BatchTab()
            tab._input_root = str(input_dir)
            tab._scan()
            tab.output_options.target_format_combo.setCurrentIndex(list(formats.OUTPUT_FORMATS).index("cif"))
            tab.output_dir_edit.setText(str(Path(tmpdir) / "out"))

            with mock.patch.object(BatchConvertWorker, "start", _run_worker_synchronously), \
                 mock.patch.object(QMessageBox, "information"):
                tab._on_start()

            statuses = [tab.table.item(row, BatchTab.COLUMN_STATUS).text() for row in range(3)]
            self.assertEqual(statuses.count("成功"), 2)
            self.assertEqual(len(list((Path(tmpdir) / "out").glob("*.cif"))), 2)


if __name__ == "__main__":
    unittest.main()
