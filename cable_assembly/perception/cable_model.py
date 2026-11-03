"""Cable model -- the BNC connector and its cable as the camera saw them, kept up to date.

    plug = ConnectorModel()                              # the plug's CAD (or a cylinder without it)
    T, stats = plug.fit(connector_points, T_start)       # rigid fit, from the junction pose
    rod = CableRod(start, direction, length_m=0.30)      # the cable: a deformable rod (Warp)
    rod.set_environment(height_field(env_points))        # what it rests on
    rod.trace(cable_points); rod.fit(cable_points)       # onto what the camera saw
    rod.pin(start, direction); rod.step(...)             # then as the connector is carried

THE CONNECTOR is rigid: the plug's outside (robot/description/bnc/bnc_plug.npz, made from McMaster's
CAD by sim/scripts/export_plug_mesh.py; gitignored) in the connector frame -- the origin on the axis
where the cable leaves the strain relief, x toward the mating face. Fitted by robust ICP from the
camera's points to the model's surface (the points see one side; the model is whole), from the
junction pose the scan found. Without the CAD, a cylinder of the connector's radius stands in.

THE CABLE is deformable: a chain of nodes `segment_m` apart (XPBD, perception/cable_rod_kernels.py,
in Warp) -- held to its length, resisting folding, under gravity, pinned at the connector (its first
two nodes leave the strain relief along the connector's axis), resting on what the camera saw
beneath it (a height field of the scene without the cable in it -- the bench, a fixture, anything:
no plane is assumed, only that a wrist camera looking down sees tops, not undersides), and drawn
toward the camera's points of the cable, each node to the nearest within `reach_m`. The camera sees
the cable's top: its points are taken `radius_m` lower, onto the axis.

(A signed distance from the occupancy was the first surface: where the voxels are finer than the
camera's samples it has holes, and the cable fell through them.)
"""

import os

import numpy as np
from scipy.spatial import cKDTree

PLUG_NPZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'robot', 'description',
                        'bnc', 'bnc_plug.npz')


def _surface_samples(points, faces, n, rng):
    tri = points[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


class ConnectorModel:
    """The plug's outside as surface samples in the connector frame; rigid ICP onto camera points."""

    def __init__(self, path=PLUG_NPZ, n_samples=6000, radius_m=0.0074, length_m=0.044, seed=0):
        rng = np.random.default_rng(seed)
        self.from_cad = os.path.isfile(path)
        if self.from_cad:
            with np.load(path) as z:
                self.samples = _surface_samples(z['points'], z['faces'], n_samples, rng)
        else:                                                    # a cylinder of the connector's size
            s, th = rng.random(n_samples) * length_m, rng.random(n_samples) * 2 * np.pi
            self.samples = np.stack([s, radius_m * np.cos(th), radius_m * np.sin(th)], axis=1)
        self.tree = cKDTree(self.samples)

    def fit(self, points, T_start, iters=40, huber_m=0.0015, reject_m=0.008):
        """(T_base_connector, stats): the rigid pose that lays the camera's `points` (base frame)
        on the model's surface, by ICP from `T_start`. stats: rms (m) and the fraction within
        huber_m, over the points within reject_m of the model."""
        P = np.asarray(points, dtype=float)
        T = np.asarray(T_start, dtype=float).copy()
        for _ in range(iters):
            q = (P - T[:3, 3]) @ T[:3, :3]                       # the points in the model frame
            d, idx = self.tree.query(q)
            keep = d < reject_m
            if keep.sum() < 10:
                break
            wgt = np.where(d[keep] < huber_m, 1.0, huber_m / np.maximum(d[keep], 1e-9))
            m, p = self.samples[idx[keep]], P[keep]              # model point (model frame) <-> camera point
            mc, pc = np.average(m, axis=0, weights=wgt), np.average(p, axis=0, weights=wgt)
            H = ((m - mc) * wgt[:, None]).T @ (p - pc)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            T_new = np.eye(4)
            T_new[:3, :3], T_new[:3, 3] = R, pc - R @ mc
            step = np.linalg.norm(T_new[:3, 3] - T[:3, 3])
            T = T_new
            if step < 1e-6:
                break
        q = (P - T[:3, 3]) @ T[:3, :3]
        d, _ = self.tree.query(q)
        near = d < reject_m
        return T, {'rms_m': float(np.sqrt(np.mean(d[near] ** 2))) if near.any() else np.inf,
                   'inliers': float(np.mean(d < huber_m)), 'points': len(P), 'from_cad': self.from_cad}


class HeightField:
    """The highest point seen in each `cell_m` square of the x-y plane: what something set down
    there would rest on. Cells nothing was seen in take their nearest seen neighbour's height."""

    def __init__(self, points, cell_m=0.004, pad_m=0.05):
        from scipy.ndimage import distance_transform_edt
        P = np.asarray(points, dtype=float)
        self.cell = float(cell_m)
        self.lo = P[:, :2].min(axis=0) - pad_m
        shape = tuple(int(n) for n in np.ceil((P[:, :2].max(axis=0) + pad_m - self.lo) / self.cell))
        ij = np.floor((P[:, :2] - self.lo) / self.cell).astype(np.int64)
        flat = np.ravel_multi_index(ij.T, shape)
        top = np.full(int(np.prod(shape)), -np.inf)
        np.maximum.at(top, flat, P[:, 2])
        top = top.reshape(shape)
        seen = np.isfinite(top)
        _, (ii, jj) = distance_transform_edt(~seen, return_indices=True)
        self.height = top[ii, jj]
        self.seen = seen

    def at(self, xy):
        """The surface's height under each x-y, interpolated between cell centres exactly as the
        rod's solver does (perception/cable_rod_kernels.py height_at)."""
        n = np.array(self.height.shape)
        q = np.clip((np.atleast_2d(xy) - self.lo) / self.cell - 0.5, 0.0, n - 1 - 1e-4)
        i, j = q[:, 0].astype(int), q[:, 1].astype(int)
        tx, ty = q[:, 0] - i, q[:, 1] - j
        h = self.height
        lo = h[i, j] + (h[i + 1, j] - h[i, j]) * tx
        hi = h[i, j + 1] + (h[i + 1, j + 1] - h[i, j + 1]) * tx
        return lo + (hi - lo) * ty


def downsample(points, voxel_m=0.003):
    """One point per `voxel_m` cube: the mean of those in it."""
    P = np.asarray(points, dtype=float)
    if not len(P):
        return P
    _, inv = np.unique(np.floor(P / voxel_m).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv).astype(float)
    return np.stack([np.bincount(inv, weights=P[:, i]) / n for i in range(3)], axis=1)


class CableRod:
    """The cable as an XPBD chain in Warp (see the module docstring). Positions in metres, base frame."""

    def __init__(self, start, direction, length_m=0.30, segment_m=0.01, radius_m=0.0025, device='cpu',
                 stretch_compliance=1e-9, bend_compliance=2e-6, pull_compliance=1e-6, damping=0.98, reach_m=0.03):
        import warp as wp

        from . import cable_rod_kernels as K
        wp.config.quiet = True
        wp.init()
        self._wp, self._K, self.device = wp, K, device
        self.segment, self.radius, self.reach = float(segment_m), float(radius_m), float(reach_m)
        self.n = round(float(length_m) / self.segment) + 1
        self.stretch_c, self.bend_c, self.pull_c, self.damping = stretch_compliance, bend_compliance, pull_compliance, damping
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        x0 = np.asarray(start, dtype=float) + np.outer(np.arange(self.n) * self.segment, d)
        self.x = wp.array(x0, dtype=wp.vec3, device=device)
        self.x_prev = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.v = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.target = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.has = wp.zeros(self.n, dtype=int, device=device)
        self.ground = None
        self.pin(start, direction)

    # ------------------------------------------------------------------ inputs
    def pin(self, start, direction):
        """Hold the first two nodes where the cable leaves the connector, along `direction`."""
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        x = self.x.numpy()
        x[0], x[1] = start, np.asarray(start, dtype=float) + self.segment * d
        w = np.ones(self.n)
        w[:2] = 0.0
        self.x = self._wp.array(x, dtype=self._wp.vec3, device=self.device)
        self.w = self._wp.array(w, dtype=float, device=self.device)

    def set_environment(self, field):
        """What the cable rests on: a HeightField of the scene without the cable in it."""
        wp = self._wp
        self.ground = wp.array(np.ascontiguousarray(field.height, dtype=np.float32), dtype=float, device=self.device)
        self.ground_lo, self.ground_cell = wp.vec2(float(field.lo[0]), float(field.lo[1])), float(field.cell)

    def set_targets(self, points):
        """Each node's pull toward the nearest of `points` (the cable's axis) within reach, or none."""
        x = self.x.numpy()
        tgt, has = np.zeros_like(x), np.zeros(self.n, dtype=np.int32)
        if points is not None and len(points):
            d, idx = cKDTree(points).query(x, distance_upper_bound=self.reach)
            ok = np.isfinite(d)
            tgt[ok], has[ok] = np.asarray(points)[idx[ok]], 1
        self.target = self._wp.array(tgt, dtype=self._wp.vec3, device=self.device)
        self.has = self._wp.array(has, dtype=int, device=self.device)
        return int(has.sum())

    def trace(self, points, max_turn_deg=60.0):
        """Lay the free nodes along `points` from the pinned start: each next node at the mean of
        the points about a segment ahead (within max_turn_deg of the way it was going); straight
        on past their end. A start for fit(), which only pulls toward points within reach."""
        P = np.asarray(points, dtype=float)
        x = self.x.numpy()
        tree = cKDTree(P) if len(P) else None
        cos_max = np.cos(np.radians(max_turn_deg))
        for i in range(2, self.n):
            d = x[i - 1] - x[i - 2]
            d /= np.linalg.norm(d)
            nxt = x[i - 1] + self.segment * d
            if tree is not None:
                near = tree.query_ball_point(x[i - 1], 1.5 * self.segment)
                cand = P[near] - x[i - 1] if near else np.zeros((0, 3))
                r = np.linalg.norm(cand, axis=1)
                ok = (r > 0.5 * self.segment) & ((cand @ d) > cos_max * r)
                if ok.any():
                    step = cand[ok].mean(axis=0)
                    nxt = x[i - 1] + self.segment * step / np.linalg.norm(step)
            x[i] = nxt
        self.x = self._wp.array(x, dtype=self._wp.vec3, device=self.device)
        self.v.zero_()

    # ------------------------------------------------------------------ the solver
    def step(self, dt=1.0 / 60.0, substeps=8, iterations=4, gravity=(0.0, 0.0, -9.81), pull=True):
        """Advance `dt` s: XPBD substeps of predict, constraints (iterations of each), velocities."""
        wp, K = self._wp, self._K
        h = dt / substeps
        g = wp.vec3(*gravity)
        n = self.n
        for _ in range(substeps):
            wp.launch(K.predict, n, inputs=[self.x, self.v, self.w, self.x_prev, g, self.damping, h], device=self.device)
            for _ in range(iterations):
                for parity in (0, 1):
                    wp.launch(K.stretch, (n + 1) // 2, inputs=[self.x, self.w, self.segment, self.stretch_c, h, parity],
                              device=self.device)
                for phase in (0, 1, 2):
                    wp.launch(K.bend, (n + 2) // 3, inputs=[self.x, self.w, self.segment, self.bend_c, h, phase],
                              device=self.device)
                if pull:
                    wp.launch(K.pull, n, inputs=[self.x, self.w, self.target, self.has, self.pull_c, h], device=self.device)
                if self.ground is not None:
                    wp.launch(K.rest_on, n, inputs=[self.x, self.w, self.ground, self.ground_lo, self.ground_cell,
                                                    self.radius], device=self.device)
            wp.launch(K.velocities, n, inputs=[self.x, self.x_prev, self.v, h], device=self.device)

    def fit(self, points, rounds=40, dt=1.0 / 60.0):
        """Settle onto `points` (the cable's axis): rounds of re-targeting and quasi-static steps
        (velocities cleared each round), gravity on. Returns the median node-to-point distance (m)
        of the nodes that found a point."""
        for _ in range(rounds):
            self.set_targets(points)
            self.step(dt)
            self.v.zero_()
        return self.residual(points)

    def residual(self, points):
        if points is None or not len(points):
            return np.inf
        d, _ = cKDTree(points).query(self.nodes(), distance_upper_bound=self.reach)
        return float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else np.inf

    def nodes(self):
        return self.x.numpy().astype(float)

    def length(self):
        return float(np.sum(np.linalg.norm(np.diff(self.nodes(), axis=0), axis=1)))
