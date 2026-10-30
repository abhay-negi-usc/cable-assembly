"""SplatScanner -- the connector found from many views' depth, the data the live splat is made of.

    scan.mode: splat          (bnc_assembly.yaml; the `splat_pick:` block tunes it)

The ground-plane scanner (skills/ground_pick.py) finds the connector from ONE image: it takes the
junction pixel's ray down to the bench plane. The connector does not lie ON the plane -- its top
is ~5-9 mm up -- so that point sits off along the ray by the height times the ray's slope (3 mm on
the first real frame), and one view's SAM3 junction and heading carry that view's errors whole.

This scanner starts the same way, so choosing the cable is unchanged: the first image, numbered,
the tag or the operator picks one. Then the camera ORBITS THE CHOSEN JUNCTION -- the splat scan's
rings (apps/splat_scan.py: orbit_views, check_plan) at `splat_pick.tilts_deg` x `pans`, every
line checked first, force-guarded -- and in every view:

  * SAM3 finds the cables again (detect_all) and the one whose junction is nearest the chosen
    one is taken (within `match_mm`);
  * the junction is LIFTED WITH DEPTH (perception/depth.lift, the cable's own pixels round it):
    a point on the connector's top, so its x-y is over the axis, with no plane assumed;
  * the heading is the detector's connector direction carried out to a second pixel and met at
    the junction's own height.

FUSED: the junction is the median of the views' x-y, the heading the mean of their directions
(each turned to agree with the first view's sign -- the detector's sign is not second-guessed;
see GroundPlaneScanner._project_all). The pose returned is the ground-plane scanner's kind --
origin ON the bench plane at the fused x-y, x = heading, z = up -- so everything after it (the
resting-height correction, the grasp) is unchanged. Logged: how far the views agreed (the spread)
and how far the fused junction is from the plane estimate.

IT FALLS BACK to the plane estimate, saying so, when the orbit cannot run (no depth -- a dry run;
too few views reachable) or fewer than `min_views` views find the cable. Every frame also goes
to the live view's camera layer (live_view.frames), so the splat builds as the orbit runs, and
each view is kept (`<image_dir>/splat_view_NN.npz`, capture_rgbd's format: scene_build and the
viewers' --rgbd take the directory).

RETRIES re-run the orbit about the re-matched junction: a missed grasp may have moved the cable.
"""

import os
import time

import numpy as np

from .. import log as urlog
from ..perception import depth as D
from ..transforms import frame_from_axis, inverse
from .ground_pick import GroundPlaneScanner

log = urlog.get('splat-pick')


class SplatScanner(GroundPlaneScanner):
    """GroundPlaneScanner's interface and selection; the pose from an orbit's fused depth."""

    def __init__(self, robot, camera, detector, estimator, cfg, data_root='data'):
        super().__init__(robot, camera, detector, estimator, cfg, data_root=data_root)
        self._cfg = cfg
        s = dict(cfg.section('splat_pick') or {}) if cfg.get('splat_pick') else {}
        self.tilts = np.radians(np.asarray(s.get('tilts_deg', [20.0]), dtype=float))
        self.pans = int(s.get('pans', 6))
        # The orbit's own caps, not a scale of the task's (bnc_assembly's are 60 mm/s, 120 deg/s).
        self.caps = dict(s.get('speed') or {'max_cartesian_translation_mm_s': 50.0,
                                            'max_cartesian_rotation_deg_s': 60.0,
                                            'max_joint_velocity_deg_s': 60.0})
        self.settle_s = float(s.get('settle_s', 0.5))
        self.n_frames = int(s.get('frames_per_view', 3))
        self.match_m = float(s.get('match_mm', 20.0)) / 1000.0
        self.match_px = float(s.get('match_px', 40.0))
        self.min_views = int(s.get('min_views', 3))
        self.max_spread_m = float(s.get('max_spread_mm', 6.0)) / 1000.0
        self.guard_force_n = float(s.get('guard_force_n', 15.0))
        self.clearance_m = float(s.get('min_clearance_mm', 60.0)) / 1000.0
        self.keep_views = bool(s.get('keep_views', True))
        self.bench_z = self.plane_z
        self.last = None                    # the last fusion: views, spread, gap to the plane estimate

    # ------------------------------------------------------------------ scan
    def scan(self, confirm=None):
        """The chosen cable's T_base_connector from the orbit (or the plane estimate, said why)."""
        T_plane = super().scan(confirm)
        if T_plane is None:
            return None
        if self.robot is None:
            return T_plane
        views = self._orbit(T_plane[:3, 3], confirm)
        fused = self._fuse(views, T_plane)
        if fused is None:
            return T_plane
        return fused

    # ------------------------------------------------------------------ the orbit
    def _orbit(self, junction, confirm=None):
        """[(frame, observation or None)] from the start view and every reachable orbit view, the
        arm back where it started. Empty if the orbit cannot run; the start view only if `confirm`
        (the app's per-step prompt) is declined.

        `junction` is the plane estimate: on the chosen junction's ray, wherever the cable really
        is. So the start view matches the chosen cable by PIXEL (that ray's), and the orbit turns
        about where the start view's depth puts the junction -- a cable lying off the bench stays
        in view -- and the other views match against that point."""
        from ..apps.capture_rgbd import save_frame, still_frames
        from ..apps.splat_scan import check_plan, orbit_views
        robot, arm = self.robot, self.robot.arm
        got = still_frames(self.camera, arm.q, self.n_frames)
        if got is None or got[1][-1].depth is None:
            log.warning('Splat pick: %s -- using the plane estimate.',
                        'the arm was moving' if got is None else 'no depth (a dry run, or camera.enable_depth off)')
            return []
        q0 = np.asarray(arm.q(), dtype=float)
        T0 = arm.fk(q0)
        cam0 = (T0 @ robot.T_tool0_cam)[:3, 3]
        first = self._observe(got[1], junction, by_pixel=True)
        if first[1] is not None:
            junction = first[1]['junction']
        plan = orbit_views(T0, inverse(T0) @ np.append(junction, 1.0), inverse(T0) @ np.append(cam0, 1.0),
                           self.tilts, self.pans)
        quiet = urlog.get('arm')
        level = quiet.level
        quiet.setLevel(50)                  # the IK's "no solution" for views being tried is not news
        try:
            plan = check_plan(arm, q0, T0, plan, robot.T_tool0_fingertip, self.bench_z, self.clearance_m)
        finally:
            quiet.setLevel(level)
        views = [first]
        if plan is None:
            log.warning('Splat pick: too few orbit views reachable from here -- the start view only.')
            return views
        log.info('Splat pick: orbiting the junction, %d views at %s deg off the line of sight, up to %.0f mm/s.',
                 len(plan), '/'.join(f'{np.degrees(t):.0f}' for t in self.tilts),
                 float(self.caps.get('max_cartesian_translation_mm_s', 0.0)))
        if confirm is not None and not confirm(f'splat pick: orbit the junction, {len(plan)} views'):
            log.info('Splat pick: orbit declined -- the scan view only.')
            return views
        out = self.image_dir if self.keep_views else None
        if out:
            os.makedirs(out, exist_ok=True)
            save_frame(os.path.join(out, 'splat_view_00.npz'), got[1], got[0], arm.model,
                       robot.T_tool0_cam, robot.T_tool0_fingertip, self._cfg)
        arm.zero_ft()
        arm.add_guard(lambda: arm.force() > self.guard_force_n)
        try:
            for k, (label, T) in enumerate(plan, start=1):
                if not arm.move_l(T, label=f'splat pick {label}', caps=self.caps):
                    log.error('Splat pick: the move to view %d stopped short; using what was seen.', k)
                    break
                time.sleep(self.settle_s)
                got = still_frames(self.camera, arm.q, self.n_frames)
                if got is None:
                    log.warning('  view %d: the arm moved through the exposure; skipped.', k)
                    continue
                views.append(self._observe(got[1], junction))
                if out:
                    save_frame(os.path.join(out, f'splat_view_{k:02d}.npz'), got[1], got[0], arm.model,
                               robot.T_tool0_cam, robot.T_tool0_fingertip, self._cfg)
            if not arm.move_l(T0, label='splat pick: back to the scan view', caps=self.caps):
                log.error('Splat pick: the move back to the scan view stopped short.')
        finally:
            arm.clear_guards()
        if out:
            log.info('Splat pick views in %s (scene_build --rgbd %s).', out, out)
        return views

    def _observe(self, frames, target, by_pixel=False):
        """(frame, the chosen cable in this view, or None): its junction lifted with depth, its
        heading (met at the junction's height), its pixel and its masks. The chosen cable is the
        one whose junction is nearest `target` -- in the image, `by_pixel` (target anywhere on the
        junction's ray), or in 3D."""
        from ..apps.capture_rgbd import median_depth
        frame = frames[-1]
        frame.depth = median_depth(frames)
        cables = self.detector.detect_all(frame, self.max_cables) or []
        best, best_d = None, self.match_px if by_pixel else self.match_m
        uv_t = D.project(frame, np.asarray(target, dtype=float)[None])[0][0] if by_pixel else None
        for cab in cables:
            ju, jv, jyaw = cab['junction']
            p = D.lift(frame, (ju, jv), mask=cab.get('cable_mask'))
            if p is None:
                p = D.lift(frame, (ju, jv))
            if p is None:
                continue
            d = (float(np.hypot(ju - uv_t[0], jv - uv_t[1])) if by_pixel
                 else float(np.linalg.norm(p - np.asarray(target, dtype=float))))
            if d < best_d:
                along = (ju + self._dir_px * np.cos(jyaw), jv + self._dir_px * np.sin(jyaw))
                ray = D.rays(frame, [along])[0]
                o = np.asarray(frame.T_base_cam, dtype=float)[:3, 3]
                if abs(ray[2]) < 1e-9:
                    continue
                q = o + (p[2] - o[2]) / ray[2] * ray       # the direction met at the junction's height
                heading = q - p
                heading[2] = 0.0
                if np.linalg.norm(heading) < 1e-6:
                    continue
                best, best_d = {'junction': p, 'heading': heading / np.linalg.norm(heading), 'uv': (ju, jv, jyaw),
                                'cable_mask': cab.get('cable_mask'), 'assembly_mask': cab.get('assembly_mask')}, d
        return frame, best

    # ------------------------------------------------------------------ fusion
    def _fuse(self, views, T_plane):
        """The fused T_base_connector, or None (too few views saw it; the plane estimate stands)."""
        seen = [obs for _, obs in views if obs is not None]
        if len(seen) < self.min_views:
            log.warning('Splat pick: the cable found in %d of %d views (min_views %d) -- using the plane '
                        'estimate.', len(seen), len(views), self.min_views)
            return None
        P = np.array([o['junction'] for o in seen])
        H = np.array([o['heading'] for o in seen])
        H[H @ H[0] < 0.0] *= -1.0                         # the first view's sign
        xy = np.median(P[:, :2], axis=0)
        heading = H.mean(axis=0)
        heading[2] = 0.0
        heading /= np.linalg.norm(heading)
        spread = float(np.median(np.linalg.norm(P[:, :2] - xy, axis=1)))
        turn = np.degrees(np.arccos(np.clip(H[:, :2] @ heading[:2] / np.linalg.norm(H[:, :2], axis=1), -1, 1)))
        gap = float(np.linalg.norm(xy - T_plane[:3, 3][:2]))
        self.last = {'views': len(views), 'seen': len(seen), 'spread_m': spread, 'gap_m': gap,
                     'top_m': float(np.median(P[:, 2])) - self.bench_z, 'heading_spread_deg': float(np.median(turn))}
        log.info('Splat pick: junction from %d of %d views, %.1f mm apart (median), heading within %.1f deg; '
                 'its top %.1f mm above the bench; %.1f mm from the one-image plane estimate.',
                 len(seen), len(views), 1000 * spread, self.last['heading_spread_deg'], 1000 * self.last['top_m'],
                 1000 * gap)
        if spread > self.max_spread_m:
            log.warning('Splat pick: the views disagree by %.1f mm (max_spread_mm %.0f) -- using the plane '
                        'estimate.', 1000 * spread, 1000 * self.max_spread_m)
            return None
        Pj = np.array([xy[0], xy[1], self.plane_z])
        self._selection = Pj.copy()                        # retries re-match against the fused junction
        return self._pose({'junction': Pj, 'heading': heading})


class Splat3DScanner(SplatScanner):
    """The splat pick with no plane anywhere in the estimate: the connector's AXIS, in 3D, from the
    orbit's views fused into one labelled map.

        scan.mode: splat_3d      (the `splat_pick:` block tunes the orbit and this)

    THE MAP. Every view the chosen cable is found in adds its depth-lifted pixels round the
    junction (within `radius_mm`), each labelled by that view's SAM3 masks: the cable side
    (cable_mask) or the connector (the rest of the cable's assembly_mask). They meet in `voxel_mm`
    cubes, and each cube is what most of the points in it said -- one view's mask edge is outvoted.

    THE POSE, from the connector's cubes alone (the cable droops off its axis; the connector is
    rigid):
      * the axis DIRECTION is their longest extent (PCA), turned to point from the cable's cubes
        toward the connector's -- tilted however the connector lies;
      * the axis LINE is the centre of the circle their cross-section fits (the visible surfaces
        of a cylinder, seen from round about), with no radius assumed -- the radius fitted is a
        check against the connector's (grasp_check);
      * the ORIGIN is on that line where the views' junctions (lifted with depth) fall along it.
    x = the axis, z = as near up as that allows. The pose is ON the axis, so the pick's
    resting-on-the-bench correction (pickup.rests_on_ground_plane) is skipped for it
    (`pose_on_axis`).

    Too few views, too few connector cubes, a fitted radius far from the connector's, or an axis
    steeper than `max_tilt_deg`: the splat pick's plane-fused estimate instead, said so. The
    labelled map and the axis go to the live view (the overlay 'splat_pick').
    """

    def __init__(self, robot, camera, detector, estimator, cfg, data_root='data'):
        super().__init__(robot, camera, detector, estimator, cfg, data_root=data_root)
        s = dict(cfg.section('splat_pick') or {}) if cfg.get('splat_pick') else {}
        self.voxel_m = float(s.get('voxel_mm', 2.0)) / 1000.0
        self.radius_m = float(s.get('radius_mm', 40.0)) / 1000.0
        self.min_connector_voxels = int(s.get('min_connector_voxels', 30))
        self.max_tilt = np.radians(float(s.get('max_tilt_deg', 45.0)))
        from .pick import connector_axis_height_m
        self.r_conn = connector_axis_height_m(cfg)        # 0: unknown, the radius goes unchecked
        self.pose_on_axis = False
        self.map = None                                   # the last labelled map: (points, is_connector)

    def scan(self, confirm=None):
        self.pose_on_axis = False
        return super().scan(confirm)

    def _fuse(self, views, T_plane):
        T = self._fuse_3d(views)
        if T is not None:
            self.pose_on_axis = True
            return T
        log.warning('Splat 3D: falling back to the splat pick\'s estimate on the bench plane.')
        return super()._fuse(views, T_plane)

    def labelled_map(self, views):
        """(voxel centres (N, 3), is_connector (N,), the junction (3,)) round the views' median
        junction, or None with too few views."""
        seen = [(f, o) for f, o in views if o is not None]
        if len(seen) < self.min_views:
            log.warning('Splat 3D: the cable found in %d of %d views (min_views %d).', len(seen), len(views),
                        self.min_views)
            return None
        j0 = np.median(np.array([o['junction'] for _, o in seen]), axis=0)
        pts, lab = [], []
        for frame, o in seen:
            cm, am = o.get('cable_mask'), o.get('assembly_mask')
            if cm is None or am is None:
                continue
            cm = np.asarray(cm, dtype=bool)
            for mask, label in ((cm, 0.0), (np.asarray(am, dtype=bool) & ~cm, 1.0)):
                p = D.mask_points(frame, mask, stride=2)
                p = p[np.linalg.norm(p - j0, axis=1) < self.radius_m]
                pts.append(p)
                lab.append(np.full(len(p), label))
        if not pts or not sum(len(p) for p in pts):
            return None
        P, L = np.vstack(pts), np.concatenate(lab)
        _, inv = np.unique(np.floor(P / self.voxel_m).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel()
        n = np.bincount(inv).astype(float)
        centres = np.stack([np.bincount(inv, weights=P[:, i]) / n for i in range(3)], axis=1)
        return centres, np.bincount(inv, weights=L) > 0.5 * n, j0

    def _fuse_3d(self, views):
        got = self.labelled_map(views)
        if got is None:
            return None
        C, is_con, j0 = got
        con, cab = C[is_con], C[~is_con]
        self.map = (C, is_con)
        if len(con) < self.min_connector_voxels:
            log.warning('Splat 3D: %d connector cubes round the junction (min_connector_voxels %d).',
                        len(con), self.min_connector_voxels)
            return None
        a, on_axis, r, inliers = fit_cylinder(con)
        toward = con.mean(axis=0) - (cab.mean(axis=0) if len(cab) else j0)
        if a @ toward < 0.0:
            a = -a
        origin = on_axis + ((j0 - on_axis) @ a) * a
        tilt = float(np.arcsin(min(abs(a[2]), 1.0)))
        self.last = {'views': len(views), 'seen': sum(o is not None for _, o in views), 'connector_voxels': len(con),
                     'cable_voxels': len(cab), 'radius_m': r, 'tilt_deg': float(np.degrees(tilt)),
                     'origin': origin, 'axis': a, 'junction': j0, 'on_cylinder': inliers}
        log.info('Splat 3D: %d connector and %d cable cubes from %d of %d views; axis %.1f deg off level, heading '
                 '%.1f deg; a cylinder of radius %.1f mm (the connector: %s) through %.0f%% of the connector cubes; '
                 'origin %s mm, %.1f mm above the bench.',
                 len(con), len(cab), self.last['seen'], len(views), np.degrees(tilt),
                 np.degrees(np.arctan2(a[1], a[0])), 1000 * r, f'{1000 * self.r_conn:.1f} mm' if self.r_conn else 'unknown',
                 100 * inliers,
                 (origin * 1000).round(1), 1000 * (origin[2] - self.bench_z))
        if self.r_conn and not 0.4 * self.r_conn < r < 2.0 * self.r_conn:
            log.warning('Splat 3D: the cross-section fits a %.1f mm radius, not the connector\'s %.1f mm.',
                        1000 * r, 1000 * self.r_conn)
            return None
        if tilt > self.max_tilt:
            log.warning('Splat 3D: the axis is %.0f deg off level (max_tilt_deg %.0f).', np.degrees(tilt),
                        np.degrees(self.max_tilt))
            return None
        self._draw(C, is_con, origin, a)
        T = np.eye(4)
        T[:3, :3] = frame_from_axis(a, [0.0, 0.0, 1.0])
        T[:3, 3] = origin
        self._selection = origin.copy()
        return T

    def _draw(self, C, is_con, origin, a):
        """The labelled map and the axis to the live view, if it is listening."""
        tx = getattr(self.robot, 'live_frames', None)
        if tx is None:
            return
        T = np.eye(4)
        T[:3, :3] = frame_from_axis(a, [0.0, 0.0, 1.0])
        T[:3, 3] = origin
        try:
            tx.send_overlay('splat_pick', [
                {'type': 'points', 'points': C[is_con].round(4).tolist(), 'color': [1.0, 0.55, 0.1], 'radius': 0.001},
                {'type': 'points', 'points': C[~is_con].round(4).tolist(), 'color': [0.2, 0.5, 1.0], 'radius': 0.001},
                {'type': 'line', 'points': [(origin - 0.03 * a).tolist(), (origin + 0.05 * a).tolist()],
                 'color': [1.0, 1.0, 0.2]},
                {'type': 'pose', 'T': T.tolist(), 'length': 0.03}],
                [f"splat 3D: axis {self.last['tilt_deg']:.1f} deg off level, radius {1000 * self.last['radius_m']:.1f} mm"])
        except Exception as exc:                                   # noqa: BLE001 -- drawing only
            log.debug('splat 3D overlay: %s', exc)


def fit_cylinder(points, tol_m=0.001):
    """The cylinder through `points` (N, 3): (axis direction, a point on the axis, radius, the
    fraction within `tol_m` of its surface). Started from their longest extent and a circle fitted
    across it, then direction, axis and radius solved together, robustly (soft-L1 at `tol_m`): an
    end face or a step in the radius is outvoted by the side, not averaged in. (Their longest
    extent alone tilts toward an end face the cameras see: 15 deg came out 8.6.)"""
    from scipy.optimize import least_squares
    P = np.asarray(points, dtype=float)
    m = P.mean(axis=0)
    a0 = np.linalg.svd(P - m, full_matrices=False)[2][0]
    e1 = np.cross(a0, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(a0, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a0, e1)
    x, y = (P - m) @ e1, (P - m) @ e2
    c = np.linalg.lstsq(np.stack([x, y, np.ones_like(x)], axis=1), -(x * x + y * y), rcond=None)[0]
    cx, cy = -c[0] / 2.0, -c[1] / 2.0
    r0 = float(np.sqrt(max(cx * cx + cy * cy - c[2], 1e-8)))

    def unpack(v):
        a = a0 + v[0] * e1 + v[1] * e2
        return a / np.linalg.norm(a), m + v[2] * e1 + v[3] * e2, v[4]

    def residual(v):
        a, o, r = unpack(v)
        w = P - o
        return np.linalg.norm(w - np.outer(w @ a, a), axis=1) - r

    sol = least_squares(residual, [0.0, 0.0, cx, cy, r0], loss='soft_l1', f_scale=tol_m)
    a, o, r = unpack(sol.x)
    return a, o, float(abs(r)), float(np.mean(np.abs(residual(sol.x)) < tol_m))
