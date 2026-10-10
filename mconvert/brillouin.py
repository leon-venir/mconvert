"""First Brillouin zone geometry: pure numpy, no Qt and no engine dependency.

The zone is the Wigner-Seitz cell of the reciprocal lattice: the set of points
closer to the origin than to any other reciprocal lattice point G, i.e. the
intersection of the half-spaces  G . k <= |G|^2 / 2.
"""

import itertools

import numpy as np

_NEIGHBOR_RANGE = 3
_MAX_PLANES_3D = 80
_MAX_PLANES_2D = 24


def _reciprocal_points(vectors, dim, max_planes):
    """The shortest reciprocal lattice points G != 0, as rows."""
    ranges = [range(-_NEIGHBOR_RANGE, _NEIGHBOR_RANGE + 1)] * dim
    indices = [n for n in itertools.product(*ranges) if any(n)]
    points = np.array(indices, dtype=float) @ vectors[:dim]
    order = np.argsort(np.linalg.norm(points, axis=1), kind="stable")
    return points[order[:max_planes]]


def _dedup(points, tol):
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in unique):
            unique.append(p)
    return np.array(unique)


def _zone_3d(vectors):
    planes = _reciprocal_points(vectors, 3, _MAX_PLANES_3D)
    rhs = 0.5 * np.einsum("ij,ij->i", planes, planes)
    scale = np.linalg.norm(planes[0])
    tol = 1e-7 * scale

    triples = np.array(list(itertools.combinations(range(len(planes)), 3)))
    matrices = planes[triples]
    solvable = np.abs(np.linalg.det(matrices)) > 1e-9 * scale**3
    triples, matrices = triples[solvable], matrices[solvable]
    candidates = np.linalg.solve(matrices, rhs[triples][..., None])[..., 0]

    inside = np.all(candidates @ planes.T <= rhs + tol, axis=1)
    vertices = _dedup(candidates[inside], 1e-6 * scale)

    on_plane = np.abs(vertices @ planes.T - rhs) < tol
    edges = []
    for i, j in itertools.combinations(range(len(vertices)), 2):
        if np.count_nonzero(on_plane[i] & on_plane[j]) >= 2:
            edges.append((vertices[i], vertices[j]))
    return np.array(edges)


def _zone_2d(vectors):
    basis, _ = np.linalg.qr(vectors[:2].T)        # orthonormal frame of the b1,b2 plane
    local = vectors[:2] @ basis                    # b1,b2 in that 2D frame
    planes = _reciprocal_points(local, 2, _MAX_PLANES_2D)
    rhs = 0.5 * np.einsum("ij,ij->i", planes, planes)
    scale = np.linalg.norm(planes[0])
    tol = 1e-7 * scale

    candidates = []
    for i, j in itertools.combinations(range(len(planes)), 2):
        matrix = planes[[i, j]]
        if abs(np.linalg.det(matrix)) > 1e-9 * scale**2:
            candidates.append(np.linalg.solve(matrix, rhs[[i, j]]))
    candidates = np.array(candidates)
    inside = np.all(candidates @ planes.T <= rhs + tol, axis=1)
    vertices = _dedup(candidates[inside], 1e-6 * scale)

    angles = np.arctan2(vertices[:, 1], vertices[:, 0])
    ordered = vertices[np.argsort(angles)]
    edges = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
    return np.array(edges) @ basis.T


def brillouin_zone_edges(rec_vectors, is_2d=False) -> np.ndarray:
    """
    Edges of the first Brillouin zone as an (N, 2, 3) array of line segments
    (start, end) in Cartesian reciprocal space.

    rec_vectors: (3, 3), rows are b1, b2, b3. With is_2d=True only b1, b2 are
    used and the zone is the polygon in their plane.
    """
    vectors = np.asarray(rec_vectors, dtype=float)
    return _zone_2d(vectors) if is_2d else _zone_3d(vectors)


def points_outside_zone(points, rec_vectors, is_2d=False, tol=1e-6) -> np.ndarray:
    """
    Boolean mask of the (N, 3) Cartesian reciprocal-space points that lie
    outside the first Brillouin zone of rec_vectors. A point is outside when
    it is farther along some G than the bisecting plane |G|^2 / 2; points on
    the zone boundary count as inside.
    """
    vectors = np.asarray(rec_vectors, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    dim = 2 if is_2d else 3
    planes = _reciprocal_points(vectors, dim, _MAX_PLANES_3D if dim == 3 else _MAX_PLANES_2D)
    rhs = 0.5 * np.einsum("ij,ij->i", planes, planes)
    excess = (points @ planes.T - rhs) / np.linalg.norm(planes[0])
    return np.any(excess > tol, axis=1)
