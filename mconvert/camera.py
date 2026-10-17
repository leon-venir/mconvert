"""Orbit camera for the 3D viewer: pure numpy, no Qt.

Z is up. The camera sits on a sphere of radius `distance` around `center`,
described by azimuth/elevation, and produces OpenGL-convention view and
projection matrices (the same convention mconvert.picking expects).
"""

import numpy as np

ORBIT_DEGREES_PER_PIXEL = 0.5
MIN_DISTANCE = 1e-3
MAX_ELEVATION = 89.0


class Camera:
    FOV_DEGREES = 40.0   # vertical field of view

    def __init__(self, center=(0.0, 0.0, 0.0), distance=10.0, azimuth=-60.0, elevation=22.0):
        self.center = np.asarray(center, dtype=float)
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)

    def _direction_from_center(self):
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def eye(self):
        return self.center + self.distance * self._direction_from_center()

    def _axes(self):
        """Orthonormal camera axes (right, up, forward) in world coordinates."""
        forward = -self._direction_from_center()
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    def view_matrix(self) -> np.ndarray:
        right, up, forward = self._axes()
        eye = self.eye()
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
        view[:3, 3] = -view[:3, :3] @ eye
        return view

    def projection_matrix(self, aspect) -> np.ndarray:
        f = 1.0 / np.tan(np.radians(self.FOV_DEGREES) / 2.0)
        near, far = self.distance * 0.01, self.distance * 100.0
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ])

    def orbit(self, dx_pixels, dy_pixels):
        """Drag right/down by pixels: turn the scene with the cursor."""
        self.azimuth -= dx_pixels * ORBIT_DEGREES_PER_PIXEL
        self.elevation = float(np.clip(
            self.elevation + dy_pixels * ORBIT_DEGREES_PER_PIXEL, -MAX_ELEVATION, MAX_ELEVATION))

    def pan(self, dx_pixels, dy_pixels, viewport_height):
        """Slide the scene with the cursor, so the pixel under it stays under it."""
        right, up, _ = self._axes()
        world_per_pixel = 2.0 * self.distance * np.tan(np.radians(self.FOV_DEGREES) / 2.0) / viewport_height
        self.center = self.center - right * dx_pixels * world_per_pixel + up * dy_pixels * world_per_pixel

    def zoom(self, steps):
        """Positive steps move in (wheel up), negative move out."""
        self.distance = max(MIN_DISTANCE, self.distance * 0.9 ** steps)

    def fit(self, low, high):
        """Frame the axis-aligned box [low, high] and reset the view angles."""
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        self.center = (low + high) / 2.0
        extent = float(np.linalg.norm(high - low))
        self.distance = max(extent, MIN_DISTANCE) * 1.15 / np.tan(np.radians(self.FOV_DEGREES) / 2.0) / 2.0 * 1.35
        self.azimuth, self.elevation = -60.0, 22.0
