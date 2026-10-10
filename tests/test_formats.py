"""Tests for mconvert.formats: the format capability table stays internally
consistent and matches what mconvert.engine.structure.struct actually
supports (see structure.py's docstring / unsupported-format error message)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mconvert import formats

ENGINE_READABLE = {"poscar", "cif", "xsd", "xyz", "qein", "qeout", "lammps", "wien2k"}
ENGINE_WRITABLE = {"poscar", "qein", "cif", "xyz", "lammps", "wien2k"}


class TestFormatTable(unittest.TestCase):

    def test_matches_engine_readable_formats(self):
        self.assertEqual(set(formats.INPUT_FORMATS), ENGINE_READABLE)

    def test_matches_engine_writable_formats(self):
        self.assertEqual(set(formats.OUTPUT_FORMATS), ENGINE_WRITABLE)

    def test_no_duplicate_keys(self):
        keys = [spec.key for spec in formats.FORMATS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_writable_format_has_a_default_filename(self):
        for key in formats.OUTPUT_FORMATS:
            self.assertTrue(formats.default_filename(key))

    def test_non_writable_format_has_no_default_filename(self):
        with self.assertRaises(ValueError):
            formats.default_filename("xsd")

    def test_every_writable_format_has_a_batch_extension(self):
        for key in formats.OUTPUT_FORMATS:
            ext = formats.batch_extension(key)
            self.assertTrue(ext.startswith("."))

    def test_non_writable_format_has_no_batch_extension(self):
        with self.assertRaises(ValueError):
            formats.batch_extension("qeout")

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            formats.get("not-a-format")

    def test_frac_cart_and_dyn_only_apply_to_poscar_and_qein(self):
        for key in formats.INPUT_FORMATS:
            expected = key in ("poscar", "qein")
            self.assertEqual(formats.supports_frac_cart(key), expected)
            self.assertEqual(formats.supports_dyn(key), expected)

    def test_dialog_filter_contains_label_and_extensions(self):
        filt = formats.dialog_filter("cif")
        self.assertIn("CIF", filt)
        self.assertIn("*.cif", filt)

    def test_all_input_dialog_filter_lists_all_files_first(self):
        filt = formats.all_input_dialog_filter()
        self.assertTrue(filt.startswith("All files (*)"))
        for key in formats.INPUT_FORMATS:
            self.assertIn(formats.label(key), filt)


if __name__ == "__main__":
    unittest.main()
