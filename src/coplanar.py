"""Find Union components whose meshes have coinciding faces.

Coincident surfaces between two Union volumes make the ray tracing in
Union_master ambiguous: a ray hitting the shared surface can end up in the
wrong volume.
"""

from collections import defaultdict

import numpy as np


def _plane_basis(normal):
    u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(normal, u)])


def _overlapping(a, b, eps):
    """For paired 2D triangles a[k], b[k] (k, 3, 2): do they overlap with positive
    area? Separating axis test over the six edge normals; touching along an edge
    is not overlap."""
    edges = np.concatenate([np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b], axis=1)
    axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
    pa, pb = np.einsum("kvd,kad->kav", a, axes), np.einsum("kvd,kad->kav", b, axes)
    gap = eps * np.linalg.norm(axes, axis=2)
    separated = (pa.max(2) <= pb.min(2) + gap) | (pb.max(2) <= pa.min(2) + gap)
    return ~separated.any(1)


def _any_overlap(a, b, eps):
    """Does any 2D triangle of a (n, 3, 2) overlap one of b (m, 3, 2)? Only pairs
    whose bounding boxes meet within eps are tested."""
    blo, bhi = b.min(1), b.max(1)
    # about 4M box pairs per pass
    chunk = max(1, (1 << 22) // len(b))
    for s in range(0, len(a), chunk):
        part = a[s:s + chunk]
        lo, hi = part.min(1), part.max(1)
        i, j = np.nonzero(np.all((lo[:, None] <= bhi[None] + eps) & (blo[None] <= hi[:, None] + eps), axis=2))
        if len(i) and _overlapping(part[i], b[j], eps).any():
            return True
    return False


def find_coincident_faces(meshes, tol=1e-9):
    """Return [(name_a, name_b, [(normal, offset), ...])], one entry per
    pair of meshes with overlapping triangles in a common plane, whichever
    way they face, listing every such plane.

    meshes: {component name: world-space trimesh or None}, as built by the
    meshers. tol is the distance (m) within which two planes count as the
    same plane, and also the tolerance on the components of their unit
    normals."""
    buckets = defaultdict(lambda: defaultdict(list))
    planes = {}
    for name, mesh in meshes.items():
        if mesh is None or len(mesh.faces) == 0:
            continue
        triangles = np.asarray(mesh.triangles)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        triangles, normals = triangles[lengths > 0], normals[lengths > 0] / lengths[lengths > 0, None]
        flip = normals[np.arange(len(normals)), np.argmax(np.abs(normals), axis=1)] < 0
        normals[flip] *= -1
        offsets = np.einsum("ij,ij->i", normals, triangles[:, 0])
        # ponytail: planes are bucketed by rounding, so two planes that
        # straddle a rounding boundary are missed; compare neighbouring
        # buckets if that ever matters.
        keys = np.round(np.column_stack([normals, offsets]) / tol).astype(np.int64)
        for key, tri, normal, offset in zip(map(tuple, keys), triangles, normals, offsets):
            buckets[key][name].append(tri)
            planes.setdefault(key, (normal, offset))

    found = defaultdict(list)
    for key, per_mesh in buckets.items():
        names = sorted(per_mesh)
        if len(names) < 2:
            continue
        normal, offset = planes[key]
        basis = _plane_basis(normal)
        flat = {n: np.array(per_mesh[n]) @ basis.T for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if _any_overlap(flat[a], flat[b], tol):
                    found[a, b].append((normal, offset))
    return [(a, b, shared) for (a, b), shared in sorted(found.items())]
