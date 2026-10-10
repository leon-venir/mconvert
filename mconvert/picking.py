"""Screen-space picking for the 3D viewer: pure numpy, no Qt or OpenGL.

The viewer hands over its current view/projection matrices; picking projects
every candidate point to the screen and selects the nearest hit, so it works
without a GPU readback and can be tested with synthetic matrices.
"""

from typing import Optional

import numpy as np


def project_points(points, view, projection, width, height):
    """
    Project (N, 3) world points to screen pixels (origin top-left).

    Returns (screen_xy of shape (N, 2), depth of shape (N,)). depth is the
    clip-space w, i.e. the distance in front of the camera for a perspective
    projection; it is <= 0 for points behind the camera.
    """
    points = np.asarray(points, dtype=float)
    homogeneous = np.c_[points, np.ones(len(points))]
    clip = homogeneous @ (projection @ view).T
    depth = clip[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / depth[:, None]
    screen = np.c_[(ndc[:, 0] + 1.0) / 2.0 * width, (1.0 - ndc[:, 1]) / 2.0 * height]
    return screen, depth


def pick_nearest(points, world_radii, view, projection, click_xy, size,
                 min_radius_px=8.0) -> Optional[int]:
    """
    Index of the point under a click, or None.

    A point is hit when the click lies inside its projected disc: the disc
    radius is world_radii scaled by the perspective factor, but never smaller
    than min_radius_px so small or fixed-pixel markers stay clickable. Of
    several hits, the one closest to the camera wins.
    """
    if len(points) == 0:
        return None
    width, height = size
    screen, depth = project_points(points, view, projection, width, height)
    in_front = depth > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        radius_px = np.asarray(world_radii, dtype=float) * projection[1, 1] * (height / 2.0) / depth
    radius_px = np.where(np.isfinite(radius_px), np.maximum(radius_px, min_radius_px), min_radius_px)

    distance = np.linalg.norm(screen - np.asarray(click_xy, dtype=float), axis=1)
    hits = np.flatnonzero(in_front & (distance <= radius_px))
    if len(hits) == 0:
        return None
    return int(hits[np.argmin(depth[hits])])
