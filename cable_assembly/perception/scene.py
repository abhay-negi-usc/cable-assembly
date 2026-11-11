"""Scene -- a collision world for motion planning, from the wrist camera's RGB-D frames.

    scene = OccupancyScene.for_cell(cfg)            # the box in front of the arm, 1 cm voxels
    for f in frames: scene.add(f)                   # frame.npz dicts (capture_rgbd, splat_scan)
    world, handles = scene.to_cumotion(bench_z)     # an SDF obstacle + the bench, for cuMotion

THE GRID is a box of voxels in the base frame. A voxel is OCCUPIED when a frame's depth puts a
point in it. Each new frame REPLACES WHAT IT SEES, as the live view's camera layer does: every
occupied voxel in its view, with a depth reading at its pixel and no more than `margin_m` behind
that reading, is cleared before the frame's own points go in -- so a cable picked up between two
frames leaves no ghost obstacle where it lay. What no frame has seen counts as FREE: a wrist
camera sees the bench, not the room, and the planner has to be told about the rest (the bench
itself is added as an exact box below the grid; see to_cumotion).

SOLID BELOW WHAT IS SEEN (`fill_below`, on by default). A wrist camera looking down sees the tops of
things on the bench, not their sides or what is under them, and unseen space counts as free -- so
a box seen from above would be a lid over nothing. Everything below an occupied voxel, down to the
bench, is taken as solid too: right for anything resting on the bench (the arm cannot get under
it) -- EXCEPT where a camera has SEEN THROUGH: a voxel in front of a depth reading (a ray passed
through it to the surface) is known free and never filled. Without that, an overhang was a block to
the bench: on 2026-11-03 the socket box, 27 cm up, filled its column to the bench, and the stand-off
plan from a pick under it began "in collision" with nothing there (81 mm clear without the fill).
What the robot stands in at a plan's start (clear_spheres) is marked free the same way.

LEFT OUT of every frame: points with no depth, nearer than `min_depth_m` (the D415's nearest
reading: it reads the fingers, 0.13-0.19 m from it, as 0.26-0.34 m, which would stand a wall of
obstacle under the gripper) or beyond `max_depth_m`; the gripper (a cylinder
about tool0's z out past the fingertips, as the live view leaves it out) AND every reading whose
ray passes through the gripper on its way out (its spheres, robot/tool_geometry.py, fingers open)
-- the gripper hides what is behind it, so such a reading is the fingers read too far (0.26-0.34
m passes min_depth_m), never what is there; and, when the frame
carries its joints, a capsule round each arm link from the shoulder out. The planner checks the
arm against the scene, so the arm must not be in it.

THE SIGNED DISTANCE (sdf()) is what cuMotion's SDF obstacle takes: metres, positive outside,
negative inside, from a Euclidean distance transform of the occupancy (to the nearest occupied
voxel's centre, less half a voxel, so a voxel's face is ~0). Voxel (i, j, k)'s centre is at
lo + (i + 0.5, j + 0.5, k + 0.5) * voxel_m, cuMotion's convention with the grid placed at `lo`.
"""

import itertools

import numpy as np

# The gripper, about tool0's z: as sim/src/cable_assembly_sim/rgbd.py leaves it out of the view.
GRIPPER_RADIUS_M = 0.075
GRIPPER_MARGIN_M = 0.015
FINGERTIP_Z_M = 0.190
# A capsule round each arm link from the shoulder out (kinematics fk_links frames 2..8).
LINK_RADIUS_M = 0.09
# The D415's nearest reading in its 848x480 depth mode; anything nearer is not a measurement.
MIN_DEPTH_M = 0.3


class OccupancyScene:
    def __init__(self, lo, hi, voxel_m=0.01, margin_m=0.01, max_depth_m=1.5, fill_below=True,
                 min_depth_m=MIN_DEPTH_M):
        self.fill_below = bool(fill_below)
        self.lo = np.asarray(lo, dtype=float)
        self.voxel = float(voxel_m)
        self.shape = tuple(int(n) for n in np.ceil((np.asarray(hi, dtype=float) - self.lo) / self.voxel))
        self.hi = self.lo + np.asarray(self.shape) * self.voxel
        self.margin, self.max_depth = float(margin_m), float(max_depth_m)
        self.min_depth = max(float(min_depth_m), 1e-6)
        self.occ = np.zeros(self.shape, bool)
        self.free = np.zeros(self.shape, bool)    # SEEN free: in front of a reading, or the robot's own
        self.frames = 0

    @classmethod
    def for_cell(cls, cfg, voxel_m=None):
        """The box the cell's `scene:` block names (lo_mm, hi_mm, voxel_mm), else the default:
        in front of the arm, from the bench to 0.6 m above it."""
        s = (cfg.section('scene') if cfg.get('scene') else {}) or {}
        bench_z = float(cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0
        lo = np.asarray(s.get('lo_mm', [200.0, -500.0, 1000.0 * bench_z]), dtype=float) / 1000.0
        hi = np.asarray(s.get('hi_mm', [1100.0, 600.0, 1000.0 * bench_z + 600.0]), dtype=float) / 1000.0
        return cls(lo, hi, voxel_m or float(s.get('voxel_mm', 10.0)) / 1000.0)

    # ------------------------------------------------------------------ frames in
    def points(self, frame, stride=2):
        """A frame's base-frame points (N, 3), the robot left out (see the module docstring)."""
        return self._readings(frame, stride)[0]

    def _readings(self, frame, stride=2):
        """(the points kept (N, 3), their pixels' rows (N,), columns (N,), depths (N,))."""
        depth, K = frame['depth_m'], frame['K']
        T = np.asarray(frame['T_base_cam'], dtype=float)
        v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[v, u]
        ok = (z >= self.min_depth) & (z <= self.max_depth)
        u, v, z = u[ok], v[ok], z[ok].astype(float)
        p = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1) @ T[:3, :3].T + T[:3, 3]
        keep = np.ones(len(p), bool)
        if 'T_base_tool0' in frame:
            Tt = np.asarray(frame['T_base_tool0'], dtype=float)
            tip_z = float(np.asarray(frame['T_tool0_fingertip'])[2, 3]) if 'T_tool0_fingertip' in frame else FINGERTIP_Z_M
            pt = (p - Tt[:3, 3]) @ Tt[:3, :3]
            keep &= ~((np.hypot(pt[:, 0], pt[:, 1]) < GRIPPER_RADIUS_M) & (pt[:, 2] > -0.02)
                      & (pt[:, 2] < tip_z + GRIPPER_MARGIN_M))
            from ..robot.tool_geometry import gripper_view_spheres
            sph = gripper_view_spheres(tip_z)        # the V3's outer links too: the camera sees them
            keep &= ~_ray_through_spheres(T[:3, 3], p, [Tt[:3, :3] @ c + Tt[:3, 3] for c, _ in sph],
                                          [r for _, r in sph])
        if 'joints_rad' in frame and str(frame.get('robot', 'iiwa14')) == 'iiwa14':
            from ..robot.kinematics import iiwa14
            links = iiwa14.fk_links(np.asarray(frame['joints_rad'], dtype=float))[2:, :3, 3]
            for a, b in itertools.pairwise(links):
                keep &= _distance_to_segment(p, a, b) > LINK_RADIUS_M
        return p[keep], v[keep], u[keep], z[keep]

    def seen_free(self, frame, stride=2):
        """(grid-shaped) bool: the voxels `frame` saw THROUGH -- in its image, at a pixel with a kept
        reading (the robot's and out-of-range readings carve nothing), and more than margin_m + half
        a voxel in front of it."""
        _p, v, u, z = self._readings(frame, stride)
        depth = frame['depth_m']
        h, w = (depth.shape[0] + stride - 1) // stride, (depth.shape[1] + stride - 1) // stride
        img = np.zeros((h, w), np.float32)
        img[v // stride, u // stride] = z
        T = np.asarray(frame['T_base_cam'], dtype=float)
        K = frame['K']
        c = self.centres(np.indices(self.shape).reshape(3, -1).T)
        pc = (c - T[:3, 3]) @ T[:3, :3]
        zc = pc[:, 2]
        front = zc > 1e-3
        zs = np.where(front, zc, 1.0)
        uu = np.rint(K[0, 0] * pc[:, 0] / zs + K[0, 2]).astype(np.int64) // stride
        vv = np.rint(K[1, 1] * pc[:, 1] / zs + K[1, 2]).astype(np.int64) // stride
        inside = front & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        d = np.zeros(len(c), np.float32)
        d[inside] = img[vv[inside], uu[inside]]
        return (inside & (d > 0.0) & (zc < d - self.margin - 0.5 * self.voxel)).reshape(self.shape)

    def seen_by(self, frame, centres):
        """Which voxel centres `frame` sees: in its image, with a depth reading at their pixel,
        and no more than margin_m behind it."""
        if not len(centres):
            return np.zeros(0, bool)
        T = np.asarray(frame['T_base_cam'], dtype=float)
        pc = (centres - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        K, depth = frame['K'], frame['depth_m']
        h, w = depth.shape
        front = z > 1e-3
        zs = np.where(front, z, 1.0)
        u = np.rint(K[0, 0] * pc[:, 0] / zs + K[0, 2]).astype(np.int64)
        v = np.rint(K[1, 1] * pc[:, 1] / zs + K[1, 2]).astype(np.int64)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.zeros(len(centres), np.float32)
        d[inside] = depth[v[inside], u[inside]]
        return inside & (d > 0.0) & (z < d + self.margin)

    def add(self, frame, stride=2):
        """Merge one frame in; (voxels cleared, voxels it occupies)."""
        idx = np.argwhere(self.occ)
        cleared = 0
        if len(idx):
            seen = self.seen_by(frame, self.centres(idx))
            cleared = int(seen.sum())
            self.occ[tuple(idx[seen].T)] = False
        self.free |= self.seen_free(frame, stride)
        ijk = self.index(self.points(frame, stride))
        self.occ[tuple(ijk.T)] = True
        self.free[tuple(ijk.T)] = False           # something is there now
        self.frames += 1
        return cleared, len(np.unique(ijk, axis=0)) if len(ijk) else 0

    def clear_spheres(self, centres, radii):
        """Free every occupied voxel whose centre lies inside one of the spheres (base frame, m):
        what the robot holds, or stands in, at a plan's start is not an obstacle -- nor is the space
        below what it stands under, filled (fill_below) from something seen above: every voxel inside
        the spheres is marked free. Occupied voxels freed."""
        if not len(centres):
            return 0
        cen = np.asarray(centres, dtype=float).reshape(-1, 3)
        rad = np.ravel(radii).astype(float)
        lo = np.floor((cen.min(0) - rad.max() - self.lo) / self.voxel).astype(int).clip(0, np.asarray(self.shape) - 1)
        hi = np.ceil((cen.max(0) + rad.max() - self.lo) / self.voxel).astype(int).clip(0, np.asarray(self.shape) - 1)
        if np.any(hi < lo):
            return 0
        idx = np.indices(tuple(hi - lo + 1)).reshape(3, -1).T + lo
        c = self.centres(idx)
        hit = np.zeros(len(idx), bool)
        for p, r in zip(cen, rad):
            hit |= np.sum((c - p) ** 2, axis=1) <= float(r) ** 2
        sel = tuple(idx[hit].T)
        freed = int(self.occ[sel].sum())
        self.occ[sel] = False
        self.free[sel] = True
        return freed

    # ------------------------------------------------------------------ the grid
    def index(self, points):
        """The voxel (i, j, k) each point falls in, for those inside the box (M, 3)."""
        ijk = np.floor((np.asarray(points, dtype=float) - self.lo) / self.voxel).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.asarray(self.shape)), axis=1)
        return ijk[ok]

    def centres(self, ijk):
        return self.lo + (np.asarray(ijk, dtype=float) + 0.5) * self.voxel

    def solid(self):
        """The voxels taken as obstacle: the occupied ones and, with fill_below, every voxel below
        one in its column (z is the grid's last axis, up)."""
        if not self.fill_below:
            return self.occ
        below = np.flip(np.maximum.accumulate(np.flip(self.occ, axis=2), axis=2), axis=2)
        return self.occ | (below & ~self.free)

    def sdf(self):
        """Signed distance, m, per voxel: positive outside, negative inside (float32, the grid's
        shape). With nothing occupied, every voxel is a metre from anything."""
        solid = self.solid()
        if not solid.any():
            return np.ones(self.shape, np.float32)
        from scipy.ndimage import distance_transform_edt
        outside = distance_transform_edt(~solid) * self.voxel - 0.5 * self.voxel
        inside = distance_transform_edt(solid) * self.voxel - 0.5 * self.voxel
        return np.where(solid, -inside, outside).astype(np.float32)

    def save(self, path, **extra):
        """`extra`: more arrays kept alongside (robot/world.py's filter stamp); load() ignores them."""
        np.savez_compressed(path, occ=self.occ, free=self.free, lo=self.lo, voxel=self.voxel,
                            margin=self.margin, frames=self.frames, fill_below=self.fill_below, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        s = cls(z['lo'], z['lo'] + np.asarray(z['occ'].shape) * float(z['voxel']), float(z['voxel']), float(z['margin']),
                fill_below=bool(z['fill_below']))
        s.occ, s.frames = z['occ'].astype(bool), int(z['frames'])
        s.free = z['free'].astype(bool) if 'free' in z.files else np.zeros(s.occ.shape, bool)
        # The grid IS the saved array: rebuilt from lo + shape * voxel, ceil's rounding can add a voxel
        # (91 * 0.01 m came back as 92), and the grid would no longer match its occupancy.
        s.shape = tuple(int(n) for n in s.occ.shape)
        s.hi = s.lo + np.asarray(s.shape) * s.voxel
        return s

    # ------------------------------------------------------------------ cuMotion
    def to_cumotion(self, bench_z=None, world=None, bench_size_m=(3.0, 3.0, 0.2)):
        """A cuMotion World with the scene as an SDF obstacle and, if `bench_z` is given, the bench
        as a box under the grid's footprint (its top at bench_z; it stops at the grid's edge, so
        it does not cover the arm's base, which stands on the table below the bench top).
        Returns (world, {'sdf': handle, 'bench': handle or None})."""
        import cumotion
        world = world if world is not None else cumotion.create_world()
        sdf = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)
        sdf.set_attribute(cumotion.Obstacle.Attribute.GRID, cumotion.Obstacle.Grid(
            *self.shape, self.voxel, cumotion.Obstacle.GridPrecision.FLOAT, cumotion.Obstacle.GridPrecision.FLOAT))
        handles = {'sdf': world.add_obstacle(sdf, cumotion.Pose3.from_translation(self.lo)), 'bench': None}
        world.set_sdf_grid_values_from_host(handles['sdf'], np.ascontiguousarray(self.sdf()))
        if bench_z is not None:
            box = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            sides = np.array([self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], bench_size_m[2]])
            box.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, sides)
            centre = np.array([0.5 * (self.lo[0] + self.hi[0]), 0.5 * (self.lo[1] + self.hi[1]),
                               bench_z - 0.5 * bench_size_m[2]])
            handles['bench'] = world.add_obstacle(box, cumotion.Pose3.from_translation(centre))
        return world, handles


def _ray_through_spheres(o, p, centres, radii, before_m=0.01):
    """(N,) True where the ray from `o` to each point of `p` passes through one of the spheres at
    least `before_m` short of the point: something in front of it hid it."""
    d = np.asarray(p, dtype=float) - o
    length = np.linalg.norm(d, axis=1)
    if len(d) == 0 or len(centres) == 0:
        return np.zeros(len(d), bool)
    u = d / np.maximum(length, 1e-9)[:, None]
    oc = np.asarray(centres, dtype=float).reshape(-1, 3) - o
    r2 = np.asarray(radii, dtype=float) ** 2
    # All the spheres at once (N x M): 48 of them (the bench V3, tool_geometry.gripper_view_spheres)
    # one by one made a world rebuild of ~400 saved frames take many minutes.
    along = u @ oc.T                                            # where each ray passes nearest each centre
    near2 = np.sum(oc * oc, axis=1) - along * along
    inside = near2 < r2
    entry = along - np.sqrt(np.maximum(r2 - near2, 0.0))
    return np.any(inside & (along > 0.0) & (entry < (length - before_m)[:, None]), axis=1)


def _distance_to_segment(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
