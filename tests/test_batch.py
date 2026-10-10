"""Tests for mconvert.batch: pure path-planning logic, no Qt dependency."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mconvert.batch import BatchJob, discover_input_files, resolve_output_paths

FIXTURE_ADSORPT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "mkits" / "test" / "adsorpt"
)


class TestDiscoverInputFiles(unittest.TestCase):

    def _make_tree(self, tmpdir):
        root = Path(tmpdir)
        (root / "a.poscar").write_text("a")
        (root / ".DS_Store").write_text("junk")
        (root / "sub").mkdir()
        (root / "sub" / "b.cif").write_text("b")
        (root / "sub" / "__pycache__").mkdir()
        (root / "sub" / "__pycache__" / "x.pyc").write_text("junk")
        return root

    def test_non_recursive_lists_only_top_level_visible_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_tree(tmpdir)
            found = discover_input_files(str(root), recursive=False)
        self.assertEqual(found, [str(root / "a.poscar")])

    def test_recursive_lists_nested_files_and_skips_dotted_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_tree(tmpdir)
            found = discover_input_files(str(root), recursive=True)
        self.assertEqual(
            sorted(found),
            sorted([str(root / "a.poscar"), str(root / "sub" / "b.cif")]),
        )

    def test_real_adsorpt_fixture_has_three_sibling_vasp_files(self):
        found = discover_input_files(str(FIXTURE_ADSORPT_DIR), recursive=False)
        self.assertEqual(len(found), 3)


class TestResolveOutputPaths(unittest.TestCase):

    def test_mirrors_relative_directory_and_swaps_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = [str(root / "sub" / "structure.poscar")]
            jobs = resolve_output_paths(inputs, str(root), str(root / "converted"), "cif")
        self.assertEqual(jobs, [
            BatchJob(
                input_path=str(root / "sub" / "structure.poscar"),
                output_path=str(root / "converted" / "sub" / "structure.cif"),
            )
        ])

    def test_same_directory_same_stem_gets_numeric_suffix_instead_of_colliding(self):
        # Mirrors mkits' own test/adsorpt/*.vasp: several structure files
        # sharing one flat directory. Two different sources ("a.poscar" and
        # "a.xyz") both stem to "a" and must not collapse onto one output.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = [str(root / "a.poscar"), str(root / "a.xyz")]
            jobs = resolve_output_paths(inputs, str(root), str(root / "out"), "cif")
        output_paths = [job.output_path for job in jobs]
        self.assertEqual(len(output_paths), len(set(output_paths)))
        self.assertEqual(output_paths[0], str(root / "out" / "a.cif"))
        self.assertEqual(output_paths[1], str(root / "out" / "a_2.cif"))

    def test_real_adsorpt_fixture_produces_three_distinct_outputs(self):
        inputs = discover_input_files(str(FIXTURE_ADSORPT_DIR), recursive=False)
        jobs = resolve_output_paths(inputs, str(FIXTURE_ADSORPT_DIR), "/tmp/converted", "cif")
        output_paths = {job.output_path for job in jobs}
        self.assertEqual(len(output_paths), 3)


if __name__ == "__main__":
    unittest.main()
