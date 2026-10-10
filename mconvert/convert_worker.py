"""Background worker for batch structure-format conversion.

Runs on a QThread so scanning/parsing/writing many files never blocks the UI
thread. The batch tab (added to main_window.py once this exists, see
design.md M2) builds the job list with mconvert.batch.discover_input_files()/
resolve_output_paths(), then hands it to BatchConvertWorker and listens to
its Qt signals.
"""

import os
import traceback
from typing import List

from PySide6.QtCore import QThread, Signal

from mconvert import converter
from mconvert.batch import BatchItemResult, BatchJob


class BatchConvertWorker(QThread):
    """
    Converts a list of BatchJob items on a background thread.

    item_finished fires after every file (success or failure) so the GUI can
    update one table row at a time; progress reports (completed, total).
    Call cancel() to stop before the next file -- already-converted files
    are left in place.
    """

    item_finished = Signal(object)   # BatchItemResult
    progress = Signal(int, int)      # completed_count, total_count

    def __init__(self, jobs: List[BatchJob], target_format: str,
                 frac: bool = True, dyn: bool = False, parent=None):
        super().__init__(parent)
        self._jobs = jobs
        self._target_format = target_format
        self._frac = frac
        self._dyn = dyn
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        total = len(self._jobs)
        for completed, job in enumerate(self._jobs, start=1):
            if self._cancelled:
                break
            self.item_finished.emit(self._convert_one(job))
            self.progress.emit(completed, total)

    def _convert_one(self, job: BatchJob) -> BatchItemResult:
        try:
            loaded = converter.load_structure(job.input_path)
        except converter.ConversionError as err:
            return BatchItemResult(job.input_path, job.output_path, False, message=str(err))
        except Exception as err:
            # An unexpected engine bug on one file must not take down the
            # rest of a batch that might hold hundreds of files; record it
            # (with a traceback, so it isn't silently lost) and move on.
            return BatchItemResult(job.input_path, job.output_path, False,
                                    message="Unexpected error: %s" % err,
                                    traceback=traceback.format_exc())

        try:
            os.makedirs(os.path.dirname(job.output_path) or ".", exist_ok=True)
            converter.convert_to_file(
                loaded, self._target_format, job.output_path, frac=self._frac, dyn=self._dyn,
            )
        except converter.ConversionError as err:
            return BatchItemResult(job.input_path, job.output_path, False,
                                    detected_format=loaded.summary.detected_format, message=str(err))
        except OSError as err:
            return BatchItemResult(job.input_path, job.output_path, False,
                                    detected_format=loaded.summary.detected_format,
                                    message="Failed to create output directory: %s" % err)
        except Exception as err:
            return BatchItemResult(job.input_path, job.output_path, False,
                                    detected_format=loaded.summary.detected_format,
                                    message="Unexpected error: %s" % err,
                                    traceback=traceback.format_exc())

        return BatchItemResult(job.input_path, job.output_path, True,
                                detected_format=loaded.summary.detected_format)
