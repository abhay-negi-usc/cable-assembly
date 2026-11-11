"""A captured RGB-D frame (cable_assembly/apps/capture_rgbd.py, frame.npz) as Gaussians in the
robot's base frame: the live view's camera layer. numpy only -- no Kit, no Warp -- so both
viewers use it: scripts/live_view_gl.py draws the Gaussians as spheres, and splat.write() takes
the same arrays for Isaac Sim.

ONE GAUSSIAN PER SAMPLED PIXEL that has a depth reading, at the point that pixel sees:
    position   the pixel's ray (through K) at its depth, moved to the base frame by T_base_cam
    size       the pixel's footprint at that depth (depth / focal length, times the stride)
    shape      by `splat` (SPLATS; DEFAULT_SPLAT):
                 2dgs  a surfel, as 2D Gaussian splatting draws a surface (Huang et al., SIGGRAPH
                       2024): a disc of no thickness lying ON the surface -- its normal from the
                       depth image (the surface's slope over SURFEL_NORMAL_M round it) -- and
                       stretched along the slant by 1/cos(angle to the ray), so it covers the
                       pixel's footprint there. Seen from another side it stays on the surface, so
                       views merge into one skin. Where there is no normal (a neighbour without
                       depth, or the surface nearly edge-on to the ray: a depth jump, not a
                       surface, or the edge of what was seen) the disc faces the camera, as in 3dgs.
                 3dgs  a disc facing the camera, a tenth as thick as wide: where a surface is,
                       not which way it faces (the camera layer before 2026-11-02)
    colour     the pixel's
then one per `voxel_m` cube, so a surface seen at a slant is not drawn denser than one seen square on.

Isaac Sim's RTX draws only ParticleField3DGaussianSplat (its hydra plugin has no surflet kernel,
though the schema names one), so a 2dgs surfel goes to it as an ellipsoid whose third scale is
SURFEL_FLAT of its width: flat to well under a micrometre.

LEFT OUT: pixels with no depth, nearer than `min_depth_m` or beyond `max_depth_m`, and everything
inside the gripper -- a cylinder about tool0's z axis (out of the flange) from the flange to just
past the fingertips. The 2F-85's fingers are in the bottom of every wrist-camera frame, and anything
it holds moves with it; kept, they would be drawn as part of the bench.

TOO NEAR (MIN_DEPTH_M). Nearer than the camera measures, a reading is not a measurement. The iiwa's
camera is a D405 since 2026-11-10, measuring from ~7 cm: the fingers, 0.13-0.19 m from it, read where
they are, inside the gripper cylinder. The D415 before it measured nothing nearer than ~0.3 m in its
848x480 depth mode and read them as 0.26-0.34 m (the 2026-10-30 scan), which put them in mid-air below
the fingertips, out of the gripper cylinder -- a wall under the gripper.
"""

import glob
import os
import threading
import time

import numpy as np

# The 2F-85, about tool0's z axis: its body and open fingers stay within 75 mm of it.
GRIPPER_RADIUS_M = 0.075
# Past the fingertips by this much, so the pads' edges and a held part's near end go too.
GRIPPER_MARGIN_M = 0.015
# How far out the fingertips are when the capture does not say (the iiwa bench's, 190 mm).
FINGERTIP_Z_M = 0.190


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def expand(given):
    """What --rgbd was given -> frame.npz paths: a file as it is; a run directory as its views
    (view_*.npz, a splat pick's splat_view_*.npz) or its frame.npz; `latest` as the newest splat
    scan or capture, under the repo's
    data/experiments or sim/'s (an app writes under the directory it was started in)."""
    paths = []
    for g in given:
        if g == "latest":
            runs = [d for root in (os.path.join(_REPO, "data", "experiments"), os.path.join(_REPO, "sim", "data", "experiments"))
                    for d in glob.glob(os.path.join(root, "splat_scan_*")) + glob.glob(os.path.join(root, "capture_rgbd_*"))]
            if not runs:
                raise SystemExit("[live] --rgbd latest: no splat scan or capture under data/experiments")
            g = max(runs, key=lambda d: os.path.basename(d).rsplit("_", 2)[-2:])
            print(f"[live] --rgbd latest: {g}", flush=True)
        if os.path.isdir(g):
            paths += sorted(glob.glob(os.path.join(g, "*view_*.npz"))) or sorted(glob.glob(os.path.join(g, "frame.npz")))
        else:
            paths.append(g)
    return paths


def load(path):
    """frame.npz as a dict: arrays as arrays, the strings and the stamp as Python values."""
    with np.load(path, allow_pickle=False) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def _quat_wxyz(R):
    """A rotation matrix's unit quaternion, (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quats_wxyz(R):
    """Rotation matrices (N, 3, 3) -> unit quaternions (N, 4), (w, x, y, z)."""
    d = np.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1)
    q = np.stack([np.sqrt(np.maximum(0.0, 1.0 + d[:, 0] + d[:, 1] + d[:, 2])),
                  np.copysign(np.sqrt(np.maximum(0.0, 1.0 + d[:, 0] - d[:, 1] - d[:, 2])), R[:, 2, 1] - R[:, 1, 2]),
                  np.copysign(np.sqrt(np.maximum(0.0, 1.0 - d[:, 0] + d[:, 1] - d[:, 2])), R[:, 0, 2] - R[:, 2, 0]),
                  np.copysign(np.sqrt(np.maximum(0.0, 1.0 - d[:, 0] - d[:, 1] + d[:, 2])), R[:, 1, 0] - R[:, 0, 1])],
                 axis=1) / 2.0
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# The camera's nearest reading (the D405's, ~7 cm at 1280x720; the D415's was 0.3 m); anything
# nearer is not a measurement. cable_assembly/perception/scene.py MIN_DEPTH_M is the same.
MIN_DEPTH_M = 0.07

# What a Gaussian is (the module docstring): 2dgs surfels on the surface, or 3dgs camera-facing discs.
SPLATS = ("2dgs", "3dgs")
DEFAULT_SPLAT = "2dgs"
# A surfel's thickness as a fraction of its width: no thickness, kept above zero for a renderer
# that inverts the covariance.
SURFEL_FLAT = 1e-3
# Nearer edge-on than this (cos of the surface's angle to the ray), a "surface" is a depth jump
# between two: no normal. It also caps a surfel's stretch along the slant at 1/0.2 = 5x.
SURFEL_MIN_COS = 0.2
# A neighbour this much nearer or farther (fraction of the depth) is across a jump: no normal.
SURFEL_MAX_STEP = 0.03


# A surfel's normal comes from the surface over about this far to either side, m. The D415's
# depth comes in 1 mm steps (and ~1 mm noise): neighbouring pixels, 2 mm apart at 0.5 m, read a
# staircase of camera-facing treads (median normal 90 deg off the bench's on the 2026-10-30
# capture); over 5 mm a step tilts it by ~6 deg, and the points are averaged over the patch first.
SURFEL_NORMAL_M = 0.005
# A patch with fewer of its pixels reading depth than this is at an edge of what was seen: no normal.
SURFEL_MIN_FILL = 0.7


def _box_mean(a, w, r):
    """The mean of `a` (h, w, ...) over each (2r+1)^2 window, weighted by `w` (h, w), and the
    fraction of the window's weight present."""
    def box(x):
        c = np.pad(x, [(r + 1, r), (r + 1, r)] + [(0, 0)] * (x.ndim - 2)).cumsum(0).cumsum(1)
        n = 2 * r + 1
        return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    cnt = box(w)
    mean = box(a * w[..., None]) / np.maximum(cnt, 1e-12)[..., None]
    return mean, cnt / (2 * r + 1) ** 2


def surfels(p_grid, ok_grid, k=1):
    """Surfel frames from a grid of camera-frame points (h, w, 3) and where they are valid (h, w):
    (R, stretch, has_normal), each over the grid. The normal is from the points averaged over a
    (2k+1)^2 patch, differenced k cells to either side. R's columns are, in the camera frame, the
    direction the surface slants away from the ray (stretched by `stretch` = 1/cos), the other
    in-plane axis, and the normal (toward the camera). Without a normal: R faces the camera."""
    h, w = ok_grid.shape
    R = np.broadcast_to(np.eye(3), (h, w, 3, 3)).copy()
    stretch = np.ones((h, w))
    has = np.zeros((h, w), bool)
    if h < 2 * k + 1 or w < 2 * k + 1:
        return R, stretch, has
    ps, fill = _box_mean(p_grid, ok_grid.astype(float), k)
    inner = (slice(k, h - k), slice(k, w - k))
    c, z = p_grid[inner], p_grid[inner][..., 2]
    l, r, up, dn = ps[k:h - k, :-2 * k], ps[k:h - k, 2 * k:], ps[:-2 * k, k:w - k], ps[2 * k:, k:w - k]
    near = ok_grid[inner] & (fill[inner] >= SURFEL_MIN_FILL)
    for nb, f in ((l, fill[k:h - k, :-2 * k]), (r, fill[k:h - k, 2 * k:]), (up, fill[:-2 * k, k:w - k]),
                  (dn, fill[2 * k:, k:w - k])):
        near &= (f >= SURFEL_MIN_FILL) & (np.abs(nb[..., 2] - z) < SURFEL_MAX_STEP * z)
    n = np.cross(r - l, dn - up)
    ray = c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    n *= -np.sign(np.sum(n * ray, axis=-1, keepdims=True) + 1e-12)  # toward the camera
    cos = -np.sum(n * ray, axis=-1)
    good = near & (cos >= SURFEL_MIN_COS)
    # In-plane, the way the ray slants across the surface; square on (no slant), any in-plane axis.
    t1 = ray + cos[..., None] * n
    flat = np.linalg.norm(t1, axis=-1) < 1e-6
    other = np.where((np.abs(n[..., 0]) < 0.9)[..., None], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    t1 = np.where(flat[..., None], np.cross(n, other), t1)
    t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    Rc = np.stack([t1, t2, n], axis=-1)
    R[inner] = np.where(good[..., None, None], Rc, R[inner])
    stretch[inner] = np.where(good, 1.0 / np.maximum(cos, SURFEL_MIN_COS), 1.0)
    has[inner] = good
    # The image's border, k cells deep, has no patch round it: it takes the nearest inner cell's
    # surfel. Left facing the camera, every view's border was a line of discs standing up off the
    # bench where views overlap (dark streaks in RTX).
    edge = [(k, k), (k, k)]
    R = np.pad(R[inner], edge + [(0, 0), (0, 0)], mode="edge")
    stretch = np.pad(stretch[inner], edge, mode="edge")
    has = np.pad(has[inner], edge, mode="edge") & ok_grid
    R[~has] = np.eye(3)
    stretch[~has] = 1.0
    return R, stretch, has


def gaussian_scales(sizes, stretch, splat):
    """(N, 3) standard deviations: neighbours overlap a little (a surface, not dots); a 2dgs surfel
    is stretched along the slant and flat, a 3dgs disc a tenth as thick as wide."""
    sigma = 0.6 * np.asarray(sizes, dtype=np.float32)
    if splat == "2dgs":
        return np.stack([sigma * stretch, sigma, SURFEL_FLAT * sigma], axis=1).astype(np.float32)
    return np.stack([sigma, sigma, 0.1 * sigma], axis=1).astype(np.float32)


def to_gaussians(frame, stride=2, max_depth_m=1.5, voxel_m=0.002, mask_gripper=True, min_depth_m=MIN_DEPTH_M,
                 splat=DEFAULT_SPLAT):
    """The frame's Gaussians: a dict of positions, scales, orientations (w, x, y, z), opacities and
    colors (the arrays splat.write takes), plus `radii` (a sphere per Gaussian, for ViewerGL),
    `sizes`, `stretch` (2dgs: along the slant) and `counts` (what was kept and why the rest was
    not). `splat`: one of SPLATS (the module docstring)."""
    if splat not in SPLATS:
        raise ValueError(f"splat must be one of {SPLATS}, not {splat!r}")
    depth, rgb, K = frame["depth_m"], frame["rgb"], frame["K"]
    T_base_cam = np.asarray(frame["T_base_cam"], dtype=float)
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[v, u].astype(float)
    counts = {"sampled": z.size}
    keep = (z >= max(min_depth_m, 1e-6)) & (z <= max_depth_m)
    counts["no_depth_or_far"] = int(z.size - keep.sum())  # or too near
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_grid = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)
    if splat == "2dgs":
        # SURFEL_NORMAL_M to either side, in grid cells at the depth most of the frame is at.
        k = max(1, int(round(SURFEL_NORMAL_M * 0.5 * (fx + fy) / (stride * np.median(z[keep]))))) if keep.any() else 1
        R_grid, stretch_grid, has_grid = surfels(p_grid, keep, k)
        R_cam, stretch, has = R_grid[keep], stretch_grid[keep], has_grid[keep]
    else:
        stretch, has = np.ones(int(keep.sum())), np.zeros(int(keep.sum()), bool)
    p_cam, u, v, z = p_grid[keep], u[keep], v[keep], z[keep]
    p = p_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    col = rgb[v, u].astype(np.float32) / 255.0
    size = z * stride / (0.5 * (fx + fy))
    idx = np.arange(len(p))

    if mask_gripper and "T_base_tool0" in frame:
        T = np.asarray(frame["T_base_tool0"], dtype=float)
        tip_z = float(np.asarray(frame["T_tool0_fingertip"])[2, 3]) if "T_tool0_fingertip" in frame else FINGERTIP_Z_M
        p_tool = (p - T[:3, 3]) @ T[:3, :3]
        inside = ((np.hypot(p_tool[:, 0], p_tool[:, 1]) < GRIPPER_RADIUS_M)
                  & (p_tool[:, 2] > -0.02) & (p_tool[:, 2] < tip_z + GRIPPER_MARGIN_M))
        counts["gripper"] = int(inside.sum())
        idx = idx[~inside]

    if voxel_m > 0.0 and len(idx):
        k = np.floor(p[idx] / voxel_m).astype(np.int64)
        k -= k.min(axis=0)
        span = k.max(axis=0) + 1
        first = np.unique((k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2], return_index=True)[1]
        counts["voxel_duplicates"] = int(len(idx) - len(first))
        idx = idx[first]
        size = size.copy()
        size[idx] = np.maximum(size[idx], voxel_m)
    p, col, size, stretch = p[idx], col[idx], size[idx], stretch[idx]
    counts["kept"] = len(p)
    if splat == "2dgs":
        counts["no_normal"] = int((~has[idx]).sum())  # of those kept: facing the camera

    n = len(p)
    if splat == "2dgs":
        quats = _quats_wxyz(T_base_cam[:3, :3] @ R_cam[idx]) if n else np.zeros((0, 4))
    else:
        quats = np.tile(_quat_wxyz(T_base_cam[:3, :3]), (n, 1))
    return {
        "positions": p.astype(np.float32),
        "scales": gaussian_scales(size, stretch, splat),
        "orientations": quats.astype(np.float32),
        "opacities": np.full(n, 0.95, np.float32),
        "colors": col,
        "radii": (0.5 * size).astype(np.float32),
        "sizes": size.astype(np.float32),
        "stretch": stretch.astype(np.float32),
        "counts": counts,
    }


def bench_plane(positions, band_m=0.02):
    """The flat surface most of the points lie on, as (height at their centre, m; tilt from
    horizontal, deg; points used), or None. A check of the camera pose: on a bench seen through a
    right hand-eye calibration this is the bench's height and ~0 deg."""
    p = np.asarray(positions, dtype=float)
    if len(p) < 100:
        return None
    near = p[np.abs(p[:, 2] - np.median(p[:, 2])) < band_m]
    if len(near) < 100:
        return None
    A = np.c_[near[:, 0], near[:, 1], np.ones(len(near))]
    (a, b, c), *_ = np.linalg.lstsq(A, near[:, 2], rcond=None)
    cx, cy = near[:, 0].mean(), near[:, 1].mean()
    return a * cx + b * cy + c, float(np.degrees(np.arctan(np.hypot(a, b)))), len(near)


def frustum(frame, depth_m=0.15):
    """The camera's view as line segments (starts, ends), base frame: its centre to the four image
    corners at `depth_m`, and the rectangle there."""
    K, T = frame["K"], np.asarray(frame["T_base_cam"], dtype=float)
    h, w = frame["depth_m"].shape
    corners = [((u - K[0, 2]) / K[0, 0] * depth_m, (v - K[1, 2]) / K[1, 1] * depth_m, depth_m)
               for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
    c = np.array(corners) @ T[:3, :3].T + T[:3, 3]
    o = np.tile(T[:3, 3], (4, 1))
    starts = np.vstack([o, c])
    ends = np.vstack([c, np.roll(c, -1, axis=0)])
    return starts.astype(np.float32), ends.astype(np.float32)


_KEY_OFFSET = 1 << 20  # voxel indices within +-2^20 (+-2 km at 2 mm) pack into 21 bits each


def voxel_keys(positions, voxel_m):
    """One int64 per point naming its voxel, on a grid fixed in the base frame."""
    k = np.floor(np.asarray(positions, dtype=float) / voxel_m).astype(np.int64) + _KEY_OFFSET
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


class LiveMap:
    """Frames merged into one set of Gaussians in the base frame: the live view's camera layer.

    EVERY FRAME ADDS TO IT, AND A POINT STAYS UNTIL A NEWER FRAME SUPERSEDES IT:
      * UPDATED -- the new frame has a point in the same `voxel_m` cube, or the cube just above or
        below it: the newest one is kept. (Views disagree on a surface's height by 0.5-1.5 mm; the
        same cube alone kept both where the bench lies across a cube boundary, two sheets of
        Gaussians a millimetre apart that RTX drew, flat as 2dgs surfels, with dark seams where
        they cross);
      * GONE -- the new frame saw through where it was: its pixel reads a surface more than
        `carve_m` behind it (the cable that lay there was picked up, so the bench shows instead).
    Anything else stays: seen again at the same surface between the new frame's own points, outside
    its view, behind something nearer (the fingers), or where it has no depth. So overlapping views
    combine -- a far overview does not thin out a close look, nor each view erase the last; only
    `max_points` drops points unasked, the oldest frames' first. `carve_m` sits above the D415's
    ~1 mm noise and the views' 0.5-1.5 mm disagreement, below a 4-5 mm cable.

    (It replaced everything a new frame saw at the surface before 2026-10-30: after the 15-view scan
    of that day it held 58% of what the views measured, 41% of it the last view's.)

    Its Gaussians are `splat` ones (SPLATS), each keeping the shape its frame gave it.

    add() runs in the frame receiver's thread, snapshot() and gaussians() in the render loop's."""

    def __init__(self, voxel_m=0.002, carve_m=0.003, max_points=400_000, stride=1, splat=DEFAULT_SPLAT):
        if splat not in SPLATS:
            raise ValueError(f"splat must be one of {SPLATS}, not {splat!r}")
        self.voxel_m, self.carve_m, self.max_points, self.stride = voxel_m, carve_m, max_points, stride
        self.splat = splat
        self.positions = np.zeros((0, 3), np.float32)
        self.colors = np.zeros((0, 3), np.float32)
        self.sizes = np.zeros(0, np.float32)
        self.orientations = np.zeros((0, 4), np.float32)
        self.scales = np.zeros((0, 3), np.float32)
        self._keys = np.zeros(0, np.int64)
        self._born = np.zeros(0, np.int64)  # the frame each point came from
        self.frames = 0
        self.version = 0
        self.last_frame = None  # the newest frame (for its view cone) and what it changed
        self.last_stats = None
        self._lock = threading.Lock()

    def seen_through(self, frame, positions=None):
        """Which points `frame` shows are gone: in its image, with a depth reading at their pixel
        (one it can measure, MIN_DEPTH_M or more), and more than carve_m in front of that reading."""
        p = self.positions if positions is None else np.asarray(positions, dtype=float)
        if not len(p):
            return np.zeros(0, bool)
        T = np.asarray(frame["T_base_cam"], dtype=float)
        pc = (p - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        K, depth = frame["K"], frame["depth_m"]
        h, w = depth.shape
        front = z > 1e-3
        zs = np.where(front, z, 1.0)
        u = np.rint(K[0, 0] * pc[:, 0] / zs + K[0, 2]).astype(np.int64)
        v = np.rint(K[1, 1] * pc[:, 1] / zs + K[1, 2]).astype(np.int64)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.zeros(len(p), np.float32)
        d[inside] = depth[v[inside], u[inside]]
        return inside & (d >= MIN_DEPTH_M) & (z < d - self.carve_m)

    def add(self, frame, stride=None):
        """Merge one frame in (every `stride`-th pixel, default the map's); what it changed, as a
        dict of counts."""
        g = to_gaussians(frame, stride=stride or self.stride, voxel_m=self.voxel_m, splat=self.splat)
        with self._lock:
            p, c, sz, q, sc, keys, born = (self.positions, self.colors, self.sizes, self.orientations,
                                           self.scales, self._keys, self._born)
        gone = self.seen_through(frame, p)
        new_keys = voxel_keys(g["positions"], self.voxel_m)
        updated = np.isin(keys, np.concatenate([new_keys - 1, new_keys, new_keys + 1])) & ~gone  # z: low bits
        keep = ~gone & ~updated
        n_new = len(g["positions"])
        p = np.vstack([p[keep], g["positions"]])
        c = np.vstack([c[keep], g["colors"]])
        sz = np.concatenate([sz[keep], g["sizes"]])
        q = np.vstack([q[keep], g["orientations"]])
        sc = np.vstack([sc[keep], g["scales"]])
        keys = np.concatenate([keys[keep], new_keys])
        born = np.concatenate([born[keep], np.full(n_new, self.frames, np.int64)])
        dropped = 0
        if len(p) > self.max_points:
            newest = np.sort(np.argsort(born, kind="stable")[-self.max_points:])
            dropped = len(p) - len(newest)
            p, c, sz, q, sc = p[newest], c[newest], sz[newest], q[newest], sc[newest]
            keys, born = keys[newest], born[newest]
        stats = {"added": n_new, "updated": int(updated.sum()), "gone": int(gone.sum()),
                 "dropped_oldest": dropped, "total": len(p), "gripper": g["counts"].get("gripper", 0)}
        with self._lock:
            self.positions, self.colors, self.sizes, self.orientations, self.scales = p, c, sz, q, sc
            self._keys, self._born = keys, born
            self.frames += 1
            self.version += 1
            self.last_frame, self.last_stats = frame, stats
        return stats

    def clear(self):
        """Empty the map (the viewer's reset); `version` still counts up, so a viewer redraws."""
        with self._lock:
            self.positions = np.zeros((0, 3), np.float32)
            self.colors = np.zeros((0, 3), np.float32)
            self.sizes = np.zeros(0, np.float32)
            self.orientations = np.zeros((0, 4), np.float32)
            self.scales = np.zeros((0, 3), np.float32)
            self._keys, self._born = np.zeros(0, np.int64), np.zeros(0, np.int64)
            self.frames = 0
            self.version += 1
            self.last_frame, self.last_stats = None, None

    def snapshot(self):
        """(positions, colors, sizes, orientations, version, the newest frame), consistent."""
        with self._lock:
            return (self.positions, self.colors, self.sizes, self.orientations, self.version,
                    self.last_frame)

    def gaussians(self):
        """The layer as splat.write() takes it, with its version: (dict, version), consistent."""
        with self._lock:
            return {"positions": self.positions, "scales": self.scales, "orientations": self.orientations,
                    "opacities": np.full(len(self.positions), 0.95, np.float32), "colors": self.colors}, self.version


class CameraLayer:
    """A viewer's camera layer, whatever draws it: a LiveMap, fed by --rgbd captures at start
    (seed) and by the task's frames as they come (add, from FrameReceiver's thread), and the
    status line its panel shows. The drawing is the viewer's: spheres in the OpenGL one, a
    Gaussian splat in Isaac Sim's."""

    def __init__(self, voxel_m=0.002, max_points=400_000, splat=DEFAULT_SPLAT):
        self.map = LiveMap(voxel_m=voxel_m, max_points=max_points, splat=splat)
        self.t_last = None

    def seed(self, paths, bench_z, stride=2):
        """Captured frames into the layer, in order; the joints the last was taken at (or None)."""
        joints = None
        for path in paths:
            frame = load(path)
            s = self.map.add(frame, stride=stride)
            cam = np.asarray(frame["T_base_cam"])[:3, 3]
            print(f"[live] {path}: {s['added']} points ({s['updated']} updating old ones, {s['gone']} old ones "
                  f"seen through), {s['total']} in the layer; "
                  f"camera at ({cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f}) m", flush=True)
            plane = bench_plane(self.map.positions)
            if plane is not None:
                print(f"[live]   the flat surface it saw: {1000 * plane[0]:.1f} mm high at its centre, tilted "
                      f"{plane[1]:.2f} deg ({plane[2]} points). The cell's bench: {1000 * bench_z:.1f} mm, level.", flush=True)
            if frame.get("robot") == "iiwa14":
                joints = np.asarray(frame["joints_rad"], dtype=float)
        return joints

    def add(self, frame):
        """A frame from the task."""
        s = self.map.add(frame)
        self.t_last = time.monotonic()
        print(f"[live] camera frame {frame.get('seq')} ({frame.get('phase') or '--'}): {s['added']} points "
              f"({s['updated']} updating old ones, {s['gone']} old ones seen through); {s['total']} in the layer",
              flush=True)

    def clear(self):
        """Nothing seen: the layer empty (the viewer's reset)."""
        self.map.clear()
        self.t_last = None

    def describe(self):
        n, frames = len(self.map.positions), self.map.frames
        if not frames:
            return "camera: no frames yet"
        ago = "" if self.t_last is None else f", the last {time.monotonic() - self.t_last:.0f} s ago"
        return f"camera: {n} {self.map.splat} Gaussians from {frames} frames{ago}"

    def splat_arrays(self):
        """The layer as splat.write() takes it, with its version: (dict, version)."""
        return self.map.gaussians()
