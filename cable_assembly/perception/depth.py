"""Pixels to 3D through the wrist camera's depth -- the base-frame points behind a Frame's pixels.

Frame.depth is aligned to the colour image (perception/camera.py), so a pixel's depth and its
colour are the same ray through Frame.K, and Frame.T_base_cam carries it into the base frame.
Everything here reads depth near a pixel rather than at it: the D415 leaves holes, and a thin
cable's own pixels are where its edges flicker. A MASK, where given, keeps the read on the object
(the cable's pixels, not the bench beside it). Like the rest of perception, K is used without
undistorting (the D415's colour distortion is ~0).
"""

import numpy as np


def rays(frame, uv):
    """Base-frame directions of pixel rays (N, 3), each scaled so its optical-axis component is
    1 m: a point at depth z along pixel i is T_base_cam[:3, 3] + z * rays[i]."""
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    K = frame.K
    d = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], axis=1)
    return d @ np.asarray(frame.T_base_cam, dtype=float)[:3, :3].T


def deproject(frame, uv, z):
    """Base-frame points at depths `z` (m, along the optical axis) behind pixels `uv` (N, 2)."""
    return np.asarray(frame.T_base_cam, dtype=float)[:3, 3] + np.asarray(z, dtype=float)[:, None] * rays(frame, uv)


def project(frame, points):
    """Base-frame points -> (uv (N, 2), depth along the optical axis (N,)); depth <= 0 is behind."""
    T = np.asarray(frame.T_base_cam, dtype=float)
    pc = (np.atleast_2d(points) - T[:3, 3]) @ T[:3, :3]
    z = pc[:, 2]
    zs = np.where(z > 1e-9, z, 1.0)
    K = frame.K
    return np.stack([K[0, 0] * pc[:, 0] / zs + K[0, 2], K[1, 1] * pc[:, 1] / zs + K[1, 2]], axis=1), z


def depth_near(frame, uv, radius_px=6, mask=None):
    """The median depth of the pixels within `radius_px` of `uv` that have a reading (and are in
    `mask`, if given); None if none do."""
    depth = frame.depth
    h, w = depth.shape
    u, v = round(float(uv[0])), round(float(uv[1]))
    r = int(radius_px)
    u0, u1, v0, v1 = max(0, u - r), min(w, u + r + 1), max(0, v - r), min(h, v + r + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    patch = depth[v0:v1, u0:u1]
    vv, uu = np.mgrid[v0:v1, u0:u1]
    keep = (patch > 0.0) & ((uu - u) ** 2 + (vv - v) ** 2 <= r * r)
    if mask is not None:
        keep &= np.asarray(mask)[v0:v1, u0:u1].astype(bool)
    return float(np.median(patch[keep])) if keep.any() else None


def lift(frame, uv, radius_px=6, mask=None):
    """The base-frame point behind pixel `uv`, at the depth read near it; None without one."""
    z = depth_near(frame, uv, radius_px, mask)
    return None if z is None else deproject(frame, [uv], [z])[0]


def mask_points(frame, mask, stride=2):
    """The base-frame points behind every `stride`-th pixel of `mask` that has a reading (N, 3)."""
    m = np.asarray(mask).astype(bool)[::stride, ::stride]
    v, u = np.nonzero(m)
    u, v = u * stride, v * stride
    z = frame.depth[v, u]
    ok = z > 0.0
    return deproject(frame, np.stack([u[ok], v[ok]], axis=1), z[ok])


def centreline_2d(mask, work_dim=1024):
    """A mask's centreline in pixels (N, 2) as (u, v), tip to tip, by perception.junction's tracer
    (it walks through crossings rather than cutting across them); None if it finds none."""
    from . import junction
    m = np.asarray(mask).astype(bool)
    if m.sum() < 50:
        return None
    res = junction.compute_junction(m, work_dim=work_dim, trace='graph')
    if res is None or res.get('_path') is None:
        return None
    return np.asarray(res['_path'], dtype=float)[:, ::-1] / float(res['_scale'])


def centreline_3d(frame, mask, step_px=8.0, radius_px=4, smooth=3):
    """A mask's centreline in the base frame (K, 3): the 2D centreline every `step_px` of arc,
    each sample lifted by the depth of the mask's own pixels near it, then a `smooth`-point moving
    average. Samples with no reading are dropped. None if fewer than 3 survive."""
    path = centreline_2d(mask)
    if path is None or len(path) < 2:
        return None
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    at = np.arange(0.0, s[-1] + 1e-9, step_px)
    uv = np.stack([np.interp(at, s, path[:, 0]), np.interp(at, s, path[:, 1])], axis=1)
    pts = [lift(frame, p, radius_px, mask) for p in uv]
    pts = np.array([p for p in pts if p is not None])
    if len(pts) < 3:
        return None
    if smooth > 1 and len(pts) >= smooth:
        k = np.ones(smooth) / smooth
        inner = np.stack([np.convolve(pts[:, i], k, mode='valid') for i in range(3)], axis=1)
        pts = np.vstack([pts[:1], inner, pts[-1:]])
    return pts


def spot_check(frame, point, bench_z, radius_px=6, bench_tol_m=0.0015, same_tol_m=0.003, cover_m=0.015):
    """What the camera sees where `point` (base frame) was: a verdict and the evidence.

        'gone'         the bench, where the point stood above it: what was there has been taken
        'still there'  a surface at the point's own height
        'covered'      something well in front of it (the fingers, what they hold)
        'unseen'       outside the image, behind the camera, or no depth there
        'unclear'      a surface, but neither the bench nor the point's height
    Returns (verdict, {'u', 'v', 'z_seen' (base z of the surface seen, m), 'expected_z'})."""
    point = np.asarray(point, dtype=float)
    uv, z_cam = project(frame, point[None])
    uv, z_cam = uv[0], float(z_cam[0])
    info = {'u': float(uv[0]), 'v': float(uv[1]), 'z_seen': None, 'expected_z': float(point[2])}
    h, w = frame.depth.shape
    if z_cam <= 0.0 or not (0.0 <= uv[0] < w and 0.0 <= uv[1] < h):
        return 'unseen', info
    d = depth_near(frame, uv, radius_px)
    if d is None:
        return 'unseen', info
    seen = deproject(frame, [uv], [d])[0]
    info['z_seen'] = float(seen[2])
    if d < z_cam - cover_m:
        return 'covered', info
    if seen[2] - bench_z < bench_tol_m and point[2] - bench_z > bench_tol_m:
        return 'gone', info
    if abs(seen[2] - point[2]) < same_tol_m:
        return 'still there', info
    return 'unclear', info
