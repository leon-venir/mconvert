"""Batch-conversion planning: pure Python, no Qt dependency.

Kept separate from convert_worker.py (which needs PySide6 for QThread) so
this logic -- and its tests -- work in any Python environment, including one
without the GUI framework installed.
"""

import os
from dataclasses import dataclass
from typing import List

from mconvert import formats


@dataclass(frozen=True)
class BatchJob:
    """One file to convert and where its output should go."""
    input_path: str
    output_path: str


@dataclass(frozen=True)
class BatchItemResult:
    """The outcome of converting one BatchJob, reported after every file."""
    input_path: str
    output_path: str
    ok: bool
    detected_format: str = ""   # empty if the input file could not even be parsed
    message: str = ""           # empty on success, a short error otherwise
    traceback: str = ""         # only set for an unexpected (non-ConversionError) failure


def _is_clutter(name: str) -> bool:
    """Dotfiles/dirs (.git, .DS_Store, ...) and dunder dirs (__pycache__, ...)
    are never structure files; skip them instead of reporting them as failed
    conversion items."""
    return name.startswith(".") or name.startswith("__")


def discover_input_files(root: str, recursive: bool) -> List[str]:
    """
    List candidate structure files under root.

    Formats are identified by content, not extension (see formats.py), so
    every regular file is a candidate; anything that isn't actually a
    structure file simply surfaces as a failed item later.
    """
    paths = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not _is_clutter(d))
            for name in sorted(filenames):
                if not _is_clutter(name):
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in sorted(os.listdir(root)):
            if _is_clutter(name):
                continue
            full_path = os.path.join(root, name)
            if os.path.isfile(full_path):
                paths.append(full_path)
    return paths


def resolve_output_paths(input_paths: List[str], input_root: str, output_root: str,
                          target_format: str) -> List[BatchJob]:
    """
    Compute one output path per input file: mirror the input file's directory
    relative to input_root under output_root, and reuse its filename stem
    with the target format's extension.

    The stem is preserved (instead of always using the format's fixed
    default_filename) because a single flat folder commonly holds several
    structure files already -- e.g. mkits' own test/adsorpt/*.vasp -- and
    collapsing them all to one canonical name like "structure.cif" would
    make them overwrite each other. Any remaining collision (two different
    original names mapping to the same target name) gets a numeric suffix so
    nothing produced by this run overwrites another item from the same run.
    """
    extension = formats.batch_extension(target_format)
    used_paths = set()
    jobs = []
    for input_path in input_paths:
        relative_dir = os.path.dirname(os.path.relpath(input_path, input_root))
        stem = os.path.splitext(os.path.basename(input_path))[0] or os.path.basename(input_path)

        output_path = os.path.join(output_root, relative_dir, stem + extension)
        suffix = 2
        while output_path in used_paths:
            output_path = os.path.join(output_root, relative_dir, "%s_%d%s" % (stem, suffix, extension))
            suffix += 1

        used_paths.add(output_path)
        jobs.append(BatchJob(input_path=input_path, output_path=output_path))
    return jobs
