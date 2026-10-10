"""Tests for mconvert.brillouin: pure numpy, runs in any environment."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mconvert.brillouin import brillouin_zone_edges, points_outside_zone


def _rec(lattice):
    return np.linalg.inv(np.array(lattice, dtype=float)).T * 2 * np.pi


def _vertex_count(edges):
    return len({tuple(np.round(p, 6)) for edge in edges for p in edge})


class TestBrillouinZone(unittest.TestCase):

    def test_simple_cubic_is_a_cube(self):
        edges = brillouin_zone_edges(_rec(np.eye(3) * 4.0))
        self.assertEqual(len(edges), 12)
        self.assertEqual(_vertex_count(edges), 8)
        lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
        np.testing.assert_allclose(lengths, 2 * np.pi / 4.0)

    def test_fcc_is_a_truncated_octahedron(self):
        a = 4.0
        fcc = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        edges = brillouin_zone_edges(_rec(fcc))
        self.assertEqual(_vertex_count(edges), 24)
        self.assertEqual(len(edges), 36)

    def test_bcc_is_a_rhombic_dodecahedron(self):
        a = 4.0
        bcc = a / 2 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        edges = brillouin_zone_edges(_rec(bcc))
        self.assertEqual(_vertex_count(edges), 14)
        self.assertEqual(len(edges), 24)

    def test_every_vertex_lies_inside_every_bisecting_plane_of_a_skewed_cell(self):
        lattice = [[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.7, 6.0]]
        rec = _rec(lattice)
        edges = brillouin_zone_edges(rec)
        vertices = edges.reshape(-1, 3)
        for n in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]):
            g = np.array(n) @ rec
            self.assertLessEqual(np.max(vertices @ g), 0.5 * g @ g + 1e-6)

    def test_hexagonal_2d_is_a_hexagon_in_the_ab_plane(self):
        a = 3.0
        lattice = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20.0]]
        edges = brillouin_zone_edges(_rec(lattice), is_2d=True)
        self.assertEqual(len(edges), 6)
        self.assertEqual(_vertex_count(edges), 6)
        np.testing.assert_allclose(edges[..., 2], 0.0, atol=1e-9)
        lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
        np.testing.assert_allclose(lengths, lengths[0])

    def test_square_2d_is_a_square(self):
        lattice = [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 20.0]]
        edges = brillouin_zone_edges(_rec(lattice), is_2d=True)
        self.assertEqual(len(edges), 4)


class TestPointsOutsideZone(unittest.TestCase):

    def test_cubic_zone_boundary_counts_as_inside(self):
        rec = _rec(np.eye(3) * 4.0)
        half = np.pi / 4.0
        points = [[0, 0, 0], [half, 0, 0], [half, half, half], [half * 1.01, 0, 0], [0, 0, 2 * half]]
        np.testing.assert_array_equal(points_outside_zone(points, rec), [False, False, False, True, True])

    def test_2d_zone_only_looks_at_the_in_plane_planes(self):
        rec = _rec([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 20.0]])
        half = np.pi / 3.0
        # a large kz is irrelevant for a 2D zone, an in-plane overshoot is not
        np.testing.assert_array_equal(
            points_outside_zone([[half, half, 5.0], [half * 1.2, 0, 0]], rec, is_2d=True), [False, True])


if __name__ == "__main__":
    unittest.main()
