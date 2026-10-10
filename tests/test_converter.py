"""Tests for mconvert.converter, the facade over mconvert.engine.structure."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mconvert import converter

FIXTURE_POSCAR = (
    Path(__file__).resolve().parent.parent.parent
    / "mkits" / "test" / "phonopy4zpe" / "vasp" / "freq_h2o" / "POSCAR"
)


class TestConverter(unittest.TestCase):

    def test_load_structure_detects_poscar_and_summarizes(self):
        loaded = converter.load_structure(str(FIXTURE_POSCAR))
        self.assertEqual(loaded.summary.detected_format, "poscar")
        self.assertEqual(loaded.summary.total_atoms, 3)
        self.assertEqual(loaded.summary.chemical_formula, "H2O")

    def test_load_structure_missing_file_raises_conversion_error(self):
        with self.assertRaises(converter.ConversionError):
            converter.load_structure("/no/such/file/POSCAR")

    def test_preview_cif_contains_cell_length(self):
        loaded = converter.load_structure(str(FIXTURE_POSCAR))
        text = converter.preview(loaded, "cif")
        self.assertIn("_cell_length_a", text)

    def test_preview_rejects_read_only_output_format(self):
        loaded = converter.load_structure(str(FIXTURE_POSCAR))
        with self.assertRaises(converter.ConversionError):
            converter.preview(loaded, "xsd")  # xsd is input-only

    def test_convert_to_file_writes_expected_content(self):
        loaded = converter.load_structure(str(FIXTURE_POSCAR))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "out.xyz")
            converter.convert_to_file(loaded, "xyz", out_path)
            content = Path(out_path).read_text()
        self.assertTrue(content.startswith("3\n"))


FCC_SI_POSCAR = """Si fcc primitive
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""

GRAPHENE_POSCAR = """graphene
1.0
2.46 0.0 0.0
-1.23 2.1304 0.0
0.0 0.0 20.0
C
2
Direct
0.3333333333 0.6666666667 0.5
0.6666666667 0.3333333333 0.5
"""


class TestSceneData(unittest.TestCase):

    def _load(self, text):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = Path(self._tmp.name) / "POSCAR"
        path.write_text(text)
        return converter.load_structure(str(path))

    def test_real_space_scene_lists_atoms_with_element_info(self):
        scene = converter.get_real_space_scene(converter.load_structure(str(FIXTURE_POSCAR)))
        self.assertEqual(sorted(a.symbol for a in scene.atoms), ["H", "H", "O"])
        oxygen = next(a for a in scene.atoms if a.symbol == "O")
        self.assertEqual(oxygen.atomic_number, 8)
        self.assertEqual(oxygen.name, "Oxygen")
        self.assertEqual([a.index for a in scene.atoms], [1, 2, 3])

    def test_cell_edges_are_the_twelve_lattice_vector_segments(self):
        loaded = self._load(FCC_SI_POSCAR)
        scene = converter.get_real_space_scene(loaded)
        self.assertEqual(scene.cell_edges.shape, (12, 2, 3))
        lengths = np.linalg.norm(scene.cell_edges[:, 1] - scene.cell_edges[:, 0], axis=1)
        np.testing.assert_allclose(lengths, 5.43 / np.sqrt(2))

    def test_atom_cart_matches_frac_times_lattice(self):
        scene = converter.get_real_space_scene(self._load(FCC_SI_POSCAR))
        for atom in scene.atoms:
            np.testing.assert_allclose(np.array(atom.frac) @ scene.lattice, atom.cart, atol=1e-8)

    def test_reciprocal_scene_for_a_3d_crystal(self):
        scene = converter.get_reciprocal_scene(self._load(FCC_SI_POSCAR))
        self.assertFalse(scene.is_2d)
        self.assertEqual(scene.lattice_type, "FCC")
        self.assertEqual(len(scene.zone_edges), 36)  # truncated octahedron
        gamma = next(k for k in scene.kpoints if k.key == "GAMMA")
        self.assertEqual(gamma.label, "\u0393")
        np.testing.assert_allclose(gamma.cart, 0.0, atol=1e-12)
        for k in scene.kpoints:
            np.testing.assert_allclose(np.array(k.frac) @ scene.rec_vectors, k.cart, atol=1e-10)

    def test_reciprocal_scene_for_a_2d_crystal(self):
        scene = converter.get_reciprocal_scene(self._load(GRAPHENE_POSCAR))
        self.assertTrue(scene.is_2d)
        self.assertEqual(scene.lattice_type, "hexagonal")
        self.assertEqual(len(scene.zone_edges), 6)
        k_point = next(k for k in scene.kpoints if k.key == "K")
        # |GK| = 4*pi / (3a) for a hexagonal lattice with lattice constant a
        self.assertAlmostEqual(np.linalg.norm(k_point.cart), 4 * np.pi / (3 * 2.46), places=3)
        # the K point is a corner of the first Brillouin zone
        corners = scene.zone_edges.reshape(-1, 3)
        self.assertLess(np.min(np.linalg.norm(corners - np.array(k_point.cart), axis=1)), 1e-3)

    def test_conventional_cell_is_flagged_as_having_points_outside_its_zone(self):
        conventional = """Si conventional
5.43
1 0 0
0 1 0
0 0 1
Si
8
Direct
0 0 0
0.5 0.5 0
0.5 0 0.5
0 0.5 0.5
0.25 0.25 0.25
0.75 0.75 0.25
0.75 0.25 0.75
0.25 0.75 0.75
"""
        folded = converter.get_reciprocal_scene(self._load(conventional))
        self.assertEqual(folded.lattice_type, "FCC")
        self.assertTrue(folded.points_outside_zone)
        primitive = converter.get_reciprocal_scene(self._load(FCC_SI_POSCAR))
        self.assertFalse(primitive.points_outside_zone)

    def test_high_symmetry_points_lie_inside_the_zone_for_every_bravais_variant(self):
        # one standard primitive cell per Setyawan-Curtarolo variant, built
        # from the paper's primitive vectors; the engine must pick the right
        # variant and every point must land in (or on) the first zone
        r3, cos, sin = np.sqrt(3), np.cos, np.sin
        def rhl(a, deg):
            al = np.radians(deg)
            return [[a * cos(al / 2), -a * sin(al / 2), 0], [a * cos(al / 2), a * sin(al / 2), 0],
                    [a * cos(al) / cos(al / 2), 0, a * np.sqrt(1 - cos(al) ** 2 / cos(al / 2) ** 2)]]
        def mcl(a, b, c, deg):
            al = np.radians(deg)
            return [[a, 0, 0], [0, b, 0], [0, c * cos(al), c * sin(al)]]
        def mclc(a, b, c, deg):
            al = np.radians(deg)
            return [[a / 2, b / 2, 0], [-a / 2, b / 2, 0], [0, c * cos(al), c * sin(al)]]
        cases = {
            "CUB": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            "FCC": [[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]],
            "BCC": [[-1.6, 1.6, 1.6], [1.6, -1.6, 1.6], [1.6, 1.6, -1.6]],
            "TET": [[3, 0, 0], [0, 3, 0], [0, 0, 5]],
            "BCT1": [[-2, 2, 1.5], [2, -2, 1.5], [2, 2, -1.5]],
            "BCT2": [[-1.5, 1.5, 2.5], [1.5, -1.5, 2.5], [1.5, 1.5, -2.5]],
            "ORC": [[3, 0, 0], [0, 4, 0], [0, 0, 5]],
            "ORCF1": [[0, 2, 2.5], [1.5, 0, 2.5], [1.5, 2, 0]],
            "ORCF2": [[0, 2, 2.5], [3, 0, 2.5], [3, 2, 0]],
            "ORCI": [[-1.5, 2, 2.5], [1.5, -2, 2.5], [1.5, 2, -2.5]],
            "ORCC": [[1.5, -2, 0], [1.5, 2, 0], [0, 0, 5]],
            # deliberately rounded (not exactly 120 degrees): a slightly distorted
            # cell must not be mistaken for a non-primitive one
            "HEX": [[1.23, -2.13, 0], [1.23, 2.13, 0], [0, 0, 6.7]],
            "RHL1": rhl(5, 70),
            "RHL2": rhl(5, 110),
            "MCL": mcl(4, 3, 3.5, 100),
            "MCLC3": mclc(5, 4, 4.5, 80),
            "MCLC1": mclc(5, 7, 6, 75),
        }
        for expected, lattice in cases.items():
            rows = "\n".join(" ".join("%.10f" % v for v in vec) for vec in lattice)
            scene = converter.get_reciprocal_scene(
                self._load("t\n1.0\n%s\nC\n1\nDirect\n0 0 0\n" % rows))
            self.assertEqual(scene.lattice_type, expected)
            self.assertFalse(scene.points_outside_zone, expected)

    def test_spelled_out_greek_labels_are_shown_as_greek_letters(self):
        tio2 = [[-1.5, 1.5, 2.5], [1.5, -1.5, 2.5], [1.5, 1.5, -2.5]]   # BCT2 has SIGMA and SIGMA1
        rows = "\n".join(" ".join(str(v) for v in vec) for vec in tio2)
        scene = converter.get_reciprocal_scene(self._load("t\n1.0\n%s\nC\n1\nDirect\n0 0 0\n" % rows))
        labels = {k.key: k.label for k in scene.kpoints}
        self.assertEqual(labels["SIGMA"], "\u03a3")
        self.assertEqual(labels["SIGMA1"], "\u03a31")

    def test_kpath_segments_only_reference_known_points(self):
        scene = converter.get_reciprocal_scene(self._load(FCC_SI_POSCAR))
        keys = {k.key for k in scene.kpoints}
        for segment in scene.kpath_segments:
            self.assertTrue(set(segment) <= keys)


if __name__ == "__main__":
    unittest.main()
