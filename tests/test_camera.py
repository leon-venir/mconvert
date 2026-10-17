"""Tests for mconvert.camera: pure numpy, runs in any environment."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mconvert.camera import MAX_ELEVATION, Camera
from mconvert.picking import project_points


class TestCamera(unittest.TestCase):

    def test_view_matrix_puts_the_centre_straight_ahead_at_the_orbit_distance(self):
        cam = Camera(center=(1.0, 2.0, 3.0), distance=7.0, azimuth=33.0, elevation=40.0)
        np.testing.assert_allclose(cam.view_matrix() @ [1.0, 2.0, 3.0, 1.0], [0, 0, -7.0, 1.0], atol=1e-9)

    def test_view_matrix_is_a_rigid_transform(self):
        cam = Camera(azimuth=-60, elevation=22)
        rotation = cam.view_matrix()[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_centre_projects_to_the_middle_of_the_screen(self):
        cam = Camera(center=(4.0, -1.0, 2.0), distance=12.0)
        screen, depth = project_points([cam.center], cam.view_matrix(), cam.projection_matrix(800 / 600), 800, 600)
        np.testing.assert_allclose(screen[0], [400, 300], atol=1e-6)
        self.assertAlmostEqual(depth[0], 12.0)

    def test_z_stays_up_on_screen(self):
        cam = Camera(azimuth=17, elevation=30)
        screen, _ = project_points([cam.center, cam.center + [0, 0, 1.0]], cam.view_matrix(),
                                   cam.projection_matrix(1.0), 600, 600)
        self.assertLess(screen[1, 1], screen[0, 1])   # screen y grows downward

    def test_dragging_right_turns_the_azimuth_and_dragging_down_raises_the_camera(self):
        cam = Camera(azimuth=0, elevation=0)
        cam.orbit(10, 0)
        self.assertLess(cam.azimuth, 0)
        cam.orbit(0, 10)
        self.assertGreater(cam.elevation, 0)

    def test_elevation_is_clamped_short_of_the_poles(self):
        cam = Camera()
        cam.orbit(0, 100000)
        self.assertEqual(cam.elevation, MAX_ELEVATION)
        cam.orbit(0, -1000000)
        self.assertEqual(cam.elevation, -MAX_ELEVATION)

    def test_pan_keeps_the_grabbed_point_under_the_cursor(self):
        cam = Camera(distance=10.0, azimuth=25, elevation=15)
        # a point on the focal plane moves on screen by exactly the pan distance
        plane_probe = cam.center
        before, _ = project_points([plane_probe], cam.view_matrix(), cam.projection_matrix(1.0), 600, 600)
        cam.pan(40, -25, 600)
        after, _ = project_points([plane_probe], cam.view_matrix(), cam.projection_matrix(1.0), 600, 600)
        np.testing.assert_allclose(after[0] - before[0], [40, -25], atol=1e-6)

    def test_zoom_in_and_out_are_inverse_and_distance_stays_positive(self):
        cam = Camera(distance=10.0)
        cam.zoom(3)
        cam.zoom(-3)
        self.assertAlmostEqual(cam.distance, 10.0)
        cam.zoom(10000)
        self.assertGreater(cam.distance, 0)

    def test_fit_frames_the_whole_box_inside_the_viewport(self):
        cam = Camera()
        low, high = np.array([0.0, 0, 0]), np.array([10.0, 6, 4])
        cam.fit(low, high)
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        screen, depth = project_points(corners, cam.view_matrix(), cam.projection_matrix(800 / 600), 800, 600)
        self.assertTrue(np.all(depth > 0))
        self.assertTrue(np.all((screen[:, 0] > 0) & (screen[:, 0] < 800) & (screen[:, 1] > 0) & (screen[:, 1] < 600)))


if __name__ == "__main__":
    unittest.main()
