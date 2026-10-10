"""Tests for BatchConvertWorker (the QThread itself).

Path-planning logic (discover_input_files/resolve_output_paths) is tested
separately in test_batch.py without needing PySide6. These tests need it,
since convert_worker.py imports PySide6.QtCore.QThread; they are skipped
automatically if it is not importable.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

if PYSIDE6_AVAILABLE:
    from mconvert.batch import BatchJob
    from mconvert.convert_worker import BatchConvertWorker

FIXTURE_POSCAR = (
    Path(__file__).resolve().parent.parent.parent
    / "mkits" / "test" / "phonopy4zpe" / "vasp" / "freq_h2o" / "POSCAR"
)


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed in this environment")
class TestBatchConvertWorker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_converts_every_job_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "converted"
            jobs = [BatchJob(str(FIXTURE_POSCAR), str(out_dir / "out.cif"))]
            worker = BatchConvertWorker(jobs, "cif")

            results = []
            progress_calls = []
            worker.item_finished.connect(results.append)
            worker.progress.connect(lambda done, total: progress_calls.append((done, total)))
            worker.run()  # call directly: deterministic, no real thread needed for the test

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertEqual(results[0].detected_format, "poscar")
            self.assertEqual(progress_calls, [(1, 1)])
            self.assertTrue((out_dir / "out.cif").is_file())

    def test_one_bad_file_does_not_stop_the_rest_of_the_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_input = Path(tmpdir) / "not_a_structure.txt"
            bad_input.write_text("this is not a structure file")
            out_dir = Path(tmpdir) / "converted"

            jobs = [
                BatchJob(str(bad_input), str(out_dir / "bad.cif")),
                BatchJob(str(FIXTURE_POSCAR), str(out_dir / "good.cif")),
            ]
            worker = BatchConvertWorker(jobs, "cif")
            results = []
            worker.item_finished.connect(results.append)
            worker.run()

            self.assertEqual([r.ok for r in results], [False, True])
            self.assertTrue(results[0].message)
            self.assertTrue((out_dir / "good.cif").is_file())
            self.assertFalse((out_dir / "bad.cif").exists())

    def test_cancel_stops_before_next_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs = [
                BatchJob(str(FIXTURE_POSCAR), str(Path(tmpdir) / "out1.cif")),
                BatchJob(str(FIXTURE_POSCAR), str(Path(tmpdir) / "out2.cif")),
            ]
            worker = BatchConvertWorker(jobs, "cif")
            results = []

            def _cancel_after_first(_result):
                results.append(_result)
                worker.cancel()

            worker.item_finished.connect(_cancel_after_first)
            worker.run()

            self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
