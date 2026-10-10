"""Tests for mconvert.picking: pure numpy, runs in any environment."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mconvert.picking import pick_nearest, project_points

WIDTH, HEIGHT = 800, 600


def _perspective(fov_deg=60.0, near=0.1, far=100.0):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    aspect = WIDTH / HEIGHT
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ])


def _camera_looking_down_minus_z(distance):
    view = np.eye(4)
    view[2, 3] = -distance
    return view


class TestProjectPoints(unittest.TestCase):

    def test_point_on_the_optical_axis_projects_to_the_screen_centre(self):
        screen, depth = project_points([[0, 0, 0]], _camera_looking_down_minus_z(10),
                                       _perspective(), WIDTH, HEIGHT)
        np.testing.assert_allclose(screen[0], [WIDTH / 2, HEIGHT / 2])
        self.assertAlmostEqual(depth[0], 10.0)

    def test_positive_x_goes_right_and_positive_y_goes_up(self):
        screen, _ = project_points([[1, 1, 0]], _camera_looking_down_minus_z(10),
                                   _perspective(), WIDTH, HEIGHT)
        self.assertGreater(screen[0, 0], WIDTH / 2)
        self.assertLess(screen[0, 1], HEIGHT / 2)   # screen y grows downward

    def test_point_behind_the_camera_has_non_positive_depth(self):
        _, depth = project_points([[0, 0, 20]], _camera_looking_down_minus_z(10),
                                  _perspective(), WIDTH, HEIGHT)
        self.assertLessEqual(depth[0], 0)


class TestPickNearest(unittest.TestCase):

    def setUp(self):
        self.view = _camera_looking_down_minus_z(10)
        self.projection = _perspective()
        self.size = (WIDTH, HEIGHT)

    def _screen(self, points):
        return project_points(points, self.view, self.projection, WIDTH, HEIGHT)[0]

    def test_click_on_a_point_selects_it(self):
        points = np.array([[0.0, 0, 0], [2.0, 0, 0]])
        click = self._screen(points)[1]
        self.assertEqual(pick_nearest(points, [0.3, 0.3], self.view, self.projection, click, self.size), 1)

    def test_click_on_empty_space_selects_nothing(self):
        points = np.array([[0.0, 0, 0]])
        self.assertIsNone(pick_nearest(points, [0.3], self.view, self.projection, (5, 5), self.size))

    def test_overlapping_points_pick_the_one_nearer_the_camera(self):
        points = np.array([[0.0, 0, -3.0], [0.0, 0, 3.0]])   # the second is closer to a camera at z=+10
        click = self._screen(points)[0]
        self.assertEqual(pick_nearest(points, [1.0, 1.0], self.view, self.projection, click, self.size), 1)

    def test_zero_radius_markers_are_still_clickable_within_the_minimum_radius(self):
        points = np.array([[1.0, 1.0, 0]])
        click = self._screen(points)[0] + np.array([5.0, -4.0])
        self.assertEqual(pick_nearest(points, [0.0], self.view, self.projection, click, self.size), 0)
        far_click = self._screen(points)[0] + np.array([40.0, 0.0])
        self.assertIsNone(pick_nearest(points, [0.0], self.view, self.projection, far_click, self.size))

    def test_larger_world_radius_gives_a_larger_hit_area(self):
        points = np.array([[0.0, 0, 0]])
        click = self._screen(points)[0] + np.array([30.0, 0.0])
        self.assertIsNone(pick_nearest(points, [0.1], self.view, self.projection, click, self.size))
        self.assertEqual(pick_nearest(points, [2.0], self.view, self.projection, click, self.size), 0)

    def test_points_behind_the_camera_are_ignored(self):
        points = np.array([[0.0, 0, 20.0]])
        self.assertIsNone(pick_nearest(points, [5.0], self.view, self.projection, (WIDTH / 2, HEIGHT / 2), self.size))

    def test_empty_scene_returns_none(self):
        self.assertIsNone(pick_nearest(np.zeros((0, 3)), [], self.view, self.projection, (1, 1), self.size))


if __name__ == "__main__":
    unittest.main()
