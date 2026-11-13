"""Hand-eye calibration -- where the wrist camera sits on the flange (tool0 -> color optical),
measured from ArUco markers lying still on the bench.

    python -m cable_assembly.apps.hand_eye_calibration                       # the iiwa bench
    python -m cable_assembly.apps.hand_eye_calibration --cell ur10e_bench    # any other bench
    python -m cable_assembly.apps.hand_eye_calibration --dry-run             # rehearse

Talks to the ARM and the CAMERA only (no gripper, no Robot), so it runs on a cell that is not
calibrated yet -- which is when it is needed.

THE SETUP. Lay one or more markers of `dictionary` (the iiwa bench's: the AprilTag sheet, tag36h11,
36.2 mm, ids 0-15 -- OpenCV reads the AprilTag families as DICT_APRILTAG_*) FLAT and STILL on the bench,
each with its white border intact and none overlapping another. Their side lengths must be right (solvePnP scales
depth by it, and nothing in the image can tell): `marker_size_mm`, or per id in
`marker_sizes_mm`. Bigger is better -- an 80 mm marker pins the tilt far better than a 38.8 mm
one. Put the arm where the camera looks DOWN at them from about 25 cm (`start_joints_deg`, or
jog there on the pendant before starting the FRI app).

THE SWEEP. The camera tilts `tilt_deg` toward eight directions and turns `spin_deg` both ways
about the marker in front of it, then repeats two views from `raise_mm` further back -- turning
about two different axes is what the solve needs. The force guard is armed throughout; every
move is a straight line at `speed_scale` of the caps; the arm returns to the start at the end.

THE CLOSE-UPS (hand_eye_calibration.closeup; the user, 2026-11-12: "allow for visual servoing to get closer
views and more accurate estimates of each marker"). After the sweep, a first solve puts every marker on the
bench; the camera then visits each one in turn and SERVOS onto it -- face on, the marker on the principal point,
`distance_mm` away -- re-measuring it by PnP after every move and correcting (position-based: each step is the
measured marker against where it should be, carried through the hand-eye so far, so that hand-eye's error
shrinks with the step). From there a turn each way about two axes through the marker and about the line of
sight. Every view goes into the one joint solve with the sweep's: each marker is then seen large, in the image's
middle (least distortion) and from several sides, which is what its pose -- and the bench height and X through
it -- is limited by. THE GRIPPER COMES FIRST: the fingertips reach ~155 mm past the camera along tool z, so every
close-up pose (and the straight line to it) is checked against the bench -- the plane the markers lie on -- with
the tool's collision spheres (robot/tool_geometry.py), and backed off along the line of sight until they clear
it by `min_clearance_mm`. A marker that cannot be reached that way, or servoed onto, is skipped and said so.

THE RESULT goes to data/experiments/hand_eye_calibration_<stamp>/: result.yaml (the blocks to
paste into the cell and its frames file), views.csv, and every image annotated. The markers'
base height is printed too: on a marker lying on the bench, that IS the bench height
(ground_plane.z_mm), less the paper's thickness.
"""

import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .. import config as urconfig
from .. import log as urlog
from ..perception import hand_eye as he
from ..perception.aruco import get_dictionary
from ..transforms import (from_cfg, inverse, matrix_to_xyzrpy, pose_error, slerp_matrix,
                          translation_matrix, xyzrpy_to_matrix)
from ._common import experiment_dir

log = urlog.get('hand-eye')


def main():
    parser = urconfig.arg_parser('Hand-eye calibration: where the camera sits on the flange',
                                 'hand_eye_calibration')
    parser.add_argument('--resolve', metavar='RUN_DIR', default=None,
                        help='solve a run already on disk (its raw_NN.png and poses.jsonl -- a stopped one too), '
                             'with this config\'s dictionary and marker sizes; no arm, no camera')
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    urlog.setup()
    if args.resolve:
        sys.exit(0 if resolve(cfg, args.resolve) else 1)

    from ..robot import make_arm
    arm = make_arm(cfg)
    camera = None
    ok = False
    try:
        camera = _camera(cfg, arm)
        ok = calibrate(cfg, arm, camera, args.yes)
    except KeyboardInterrupt:
        log.warning('Interrupted.')
    finally:
        if camera is not None:
            camera.close()
        arm.disconnect()
    sys.exit(0 if ok else 1)


def _camera(cfg, arm):
    from ..perception import RealSenseCamera
    cam = RealSenseCamera(cfg)
    if not arm.dry_run:
        return cam
    # DRY RUN: the camera renders markers in front of where it starts, seen through a mount that
    # is the configured one put 4 mm / 1.5 deg off -- so the rehearsal shows the solve finding a
    # mount it was not told.
    X_true = from_cfg(cfg.section('hand_eye')) @ xyzrpy_to_matrix([0.003, -0.002, 0.002],
                                                                  np.radians([1.0, -0.8, 0.6]))
    return _RenderedMarkers(cam, arm, X_true, cfg.section('hand_eye_calibration'))


class _RenderedMarkers:
    """Dry-run camera: two markers on a plane 250 mm in front of the starting view."""

    def __init__(self, cam, arm, X_true, h):
        self.cam, self.arm, self.X_true = cam, arm, X_true
        self.K, self.D = cam.K, np.zeros(5)
        self.dictionary = get_dictionary(h.get('dictionary', 'DICT_4X4_50'))
        size = float(h.get('marker_size_mm', 38.8)) / 1000.0
        T_cam0 = arm.tcp_pose() @ X_true
        self.markers = [(mid, size, T_cam0 @ xyzrpy_to_matrix([dx, 0.0, 0.25],
                                                              [np.pi, 0.0, np.radians(yaw)]))
                        for mid, dx, yaw in ((7, 0.05, 15.0), (3, -0.05, -20.0))]
        log.warning('DRY RUN: rendering markers %s, %.1f mm, 250 mm below the start view.',
                    [m[0] for m in self.markers], size * 1000.0)

    def capture(self):
        T_base_cam = self.arm.tcp_pose() @ self.X_true
        img = he.render_markers(self.K, self.cam.width, self.cam.height,
                                [(mid, s, inverse(T_base_cam) @ T) for mid, s, T in self.markers],
                                self.dictionary)
        frame = self.cam.capture()
        frame.color, frame.D = img, self.D
        return frame

    def close(self):
        self.cam.close()


def view_poses(T0, pivot, tilt, spin, raise_m):
    """tool0 targets for the sweep, about `pivot` (tool0 frame, m): eight tilts toward the
    compass points, a turn each way about the line of sight, and two turns from further back."""
    a = pivot / np.linalg.norm(pivot)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    P = translation_matrix(pivot)

    def about_pivot(axis, angle, back=0.0):
        R = np.eye(4)
        R[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
        return T0 @ P @ R @ inverse(P) @ translation_matrix(-back * a)

    views = []
    for k in range(8):
        th = k * np.pi / 4.0
        views.append((f'tilt {np.degrees(tilt):.0f} deg toward {45 * k} deg',
                      about_pivot(np.cos(th) * u + np.sin(th) * w, tilt)))
    for sgn in (1.0, -1.0):
        views.append((f'spin {np.degrees(sgn * spin):+.0f} deg', about_pivot(a, sgn * spin)))
    for sgn in (1.0, -1.0):
        views.append((f'back {raise_m * 1000:.0f} mm, spin {np.degrees(sgn * spin / 2):+.0f} deg',
                      about_pivot(a, sgn * spin / 2.0, back=raise_m)))
    return views


def calibrate(cfg, arm, camera, assume_yes=False):
    """The sweep and the solve. Returns hand_eye.solve's result dict, or False."""
    h = cfg.section('hand_eye_calibration')
    dictionary = get_dictionary(h.get('dictionary', 'DICT_4X4_50'))
    detector = he.make_detector(dictionary)
    default_size = float(h.get('marker_size_mm', 38.8)) / 1000.0
    per_id = {int(k): float(v) / 1000.0 for k, v in (h.get('marker_sizes_mm') or {}).items()}
    only = h.get('marker_ids')
    scale = float(h.get('speed_scale', 0.3))
    settle = float(h.get('settle_s', 0.5))
    f_max = float(h.get('guard_force_n', 15.0))
    X_cfg = from_cfg(cfg.section('hand_eye'))

    def size_of(mid):
        return per_id.get(mid, default_size)

    q_start = h.get('start_joints_deg')
    if q_start is not None:
        if not _ask(assume_yes, f'Move to start_joints_deg {list(q_start)}?'):
            return False
        if not arm.move_j(np.radians(np.asarray(q_start, dtype=float)), label='start view',
                          caps=scale):
            return False

    q_now = np.asarray(arm.q())
    if not arm.joints_ok(q_now):
        lo, hi = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
        bad = [f'{n} {np.degrees(v):.1f} deg (limits {np.degrees(a):.0f}..{np.degrees(b):.0f})'
               for n, v, a, b in zip(arm.joint_names, q_now, lo, hi) if not a <= v <= b]
        log.error('The arm starts outside its joint limits: %s. Jog it back inside on the '
                  'pendant (the wrist un-flipped), with the markers in view.', ', '.join(bad))
        return False

    out = experiment_dir(cfg, 'hand_eye_calibration')
    T0, frame, corners = _capture(arm, camera, detector, settle, only)
    if not corners:
        log.error('No marker in view. Put the markers under the camera (the start view must '
                  'see them) and try again.')
        return False
    # THE PIVOT: the marker nearest the image centre, carried into tool0 by the CONFIGURED
    # mount. That mount may be a placeholder, but the line of sight's depth dominates the pivot,
    # and an error of a few cm only shifts where the markers sit in the tilted views.
    centre = np.array([frame.color.shape[1] / 2.0, frame.color.shape[0] / 2.0])
    mid = min(corners, key=lambda m: np.linalg.norm(corners[m].mean(axis=0) - centre))
    T_cam_m = he.marker_poses(corners[mid], size_of(mid), frame.K, frame.D)[0]
    pivot = (X_cfg @ np.append(T_cam_m[:3, 3], 1.0))[:3]
    log.info('Start view: markers %s; pivoting about marker %d, %.0f mm from the camera.',
             sorted(corners), mid, 1000.0 * np.linalg.norm(T_cam_m[:3, 3]))

    plan = view_poses(T0, pivot, np.radians(float(h.get('tilt_deg', 15.0))),
                      np.radians(float(h.get('spin_deg', 30.0))),
                      float(h.get('raise_mm', 40.0)) / 1000.0)
    # Each view must be reachable on the SAME branch: a solution that needs a wrist flip is one
    # a straight-line move cannot get to, and would stop the sweep half way.
    # THE STEP ALLOWED is 60 deg of any joint PAST the turn the view itself commands: the two spins are 2 x spin_deg
    # apart about the line of sight (~A7), which a fixed 60 deg refused outright at spin_deg 30 (2026-11-12); what
    # this is for -- a wrist flip, A5/A7 swinging through tens of degrees for a small tilt near A6 = 0 -- still shows.
    q = np.asarray(arm.q())
    T_prev = T0
    for label, T in plan:
        sol = arm.ik(T, qnear=q)
        turn = pose_error(T_prev, T)[1]
        T_prev = T
        if sol is None or not arm.joints_ok(sol) or np.max(np.abs(np.asarray(sol) - q)) > np.radians(60.0) + turn:
            why = ('no IK solution inside the joint limits' if sol is None or not arm.joints_ok(sol) else
                   'A%d would swing %.0f deg' % (int(np.argmax(np.abs(np.asarray(sol) - q))) + 1,
                                                 np.degrees(np.max(np.abs(np.asarray(sol) - q)))))
            log.error('View "%s" is out of reach from here without flipping the wrist (%s). Start '
                      'with A6 well away from 0 (45-90 deg; near 0 the wrist is singular and '
                      'a tilt swings A5/A7 through tens of degrees), A4 well inside its -120 deg limit '
                      '(-70..-95: the sheet further out from the base, not right under the elbow) and A5, '
                      'A7 nearer the middle of their ranges, or lower tilt_deg / spin_deg.', label, why)
            return False
        q = np.asarray(sol)
    if not _ask(assume_yes, f'The arm will make {len(plan)} moves of up to '
                f'{float(h.get("tilt_deg", 15.0)):.0f} deg about a point '
                f'{1000.0 * np.linalg.norm(pivot):.0f} mm out from the flange, at {scale:.0%} '
                'speed, guarded, then return here. Hand on the e-stop?'):
        return False

    views = [_record(out, 0, 'start', T0, frame, corners, detector)]
    # THE GUARD, RE-TARED BEFORE EVERY MOVE (2026-11-12 18:49: one tare at the start, and the guard tripped in free
    # space 560 mm over the bench on the third tilt -- the iiwa's force is estimated from its joint torques, and the
    # estimate drifts by several N as the arm reconfigures, past 10 N over large moves; more with the tool's load
    # data in Sunrise not matching the tool). Each move starts from a tare where the arm stands still, so the guard
    # sees what that move adds; a trip says what it read.
    peak = [0.0]

    def guard():
        f = arm.force()
        peak[0] = max(peak[0], f)
        return f > f_max

    def move(T, label):
        """True when it arrived; 'refused' when the arm would not take the line and did not move (a singularity on
        it, a joint jump: said by the arm); False when it stopped part way -- the guard, or the link."""
        arm.zero_ft()
        peak[0] = 0.0
        A0 = arm.fk(np.asarray(arm.q()))
        if arm.move_l(T, label=label, caps=scale):
            return True
        moved_m, moved_rad = pose_error(A0, arm.fk(np.asarray(arm.q())))
        if peak[0] <= f_max and moved_m < 1e-3 and moved_rad < np.radians(0.2):
            return 'refused'
        if peak[0] > f_max:
            log.error('Move "%s" stopped by the guard: the contact force read %.1f N (guard_force_n %.0f). Nothing '
                      'should touch anything in this calibration -- if nothing did, it is the force estimate '
                      'drifting over the move (the tool\'s load data in Sunrise off, e.g. after a mount change: '
                      'apps/payload_calibration); raise guard_force_n with care, or lower tilt_deg.', label,
                      peak[0], f_max)
        else:
            log.error('Move "%s" stopped part way (%.1f mm / %.2f deg from where it began; the force %.1f N): see '
                      'the arm\'s line above.', label, 1000 * moved_m, np.degrees(moved_rad), peak[0])
        return False

    arm.add_guard(guard)
    try:
        for k, (label, T) in enumerate(plan, start=1):
            if move(T, label) is not True:
                return False
            T_meas, frame, corners = _capture(arm, camera, detector, settle, only)
            lin, ang = pose_error(T_meas, T)
            log.info('  view %2d  %-34s markers %-10s (arrived within %.1f mm, %.2f deg)', k,
                     label, sorted(corners), 1000.0 * lin, np.degrees(ang))
            views.append(_record(out, k, label, T_meas, frame, corners, detector))
        sizes = {m: size_of(m) for v in views for m in v['corners']}
        cu = dict(h.get('closeup') or {})
        if bool(cu.get('enabled', False)):
            stopped = closeups(cfg, arm, camera, detector, views, sizes, frame.K, frame.D, X_cfg, out, only, settle,
                               scale, cu, assume_yes, h, move) == 'stopped'
            if stopped:
                # THE VIEWS SO FAR ARE SOLVED, the arm left where it stopped (moving it on after a stop is the
                # operator's call)
                log.warning('CLOSE-UPS STOPPED -- solving the %d views taken; the arm is LEFT WHERE IT STOPPED.',
                            len(views))
                return _solve_and_report(cfg, out, views, sizes, frame.K, frame.D, X_cfg, h)
        if move(T0, 'back to start') is not True:
            return False
    finally:
        arm.clear_guards()

    sizes = {m: size_of(m) for v in views for m in v['corners']}
    return _solve_and_report(cfg, out, views, sizes, frame.K, frame.D, X_cfg, h)


def _solve_and_report(cfg, out, views, sizes, K, D, X_cfg, h):
    try:
        r = he.solve(views, K, D, sizes, max_view_rms_px=float(h.get('max_view_rms_px', 1.5)))
    except ValueError as exc:
        log.error('Cannot solve: %s', exc)
        return False
    _report(cfg, out, views, r, X_cfg)
    return r


def resolve(cfg, run_dir):
    """Solve a run on disk (--resolve): its poses.jsonl and raw_NN.png, detected again with this config's
    dictionary and ids, sized by its marker sizes; written back into the run's folder."""
    h = cfg.section('hand_eye_calibration')
    path = os.path.join(run_dir, 'poses.jsonl')
    if not os.path.isfile(path):
        log.error('%s has no poses.jsonl -- only runs since 2026-11-12 19:50 record their poses as they go.', run_dir)
        return False
    detector = he.make_detector(get_dictionary(h.get('dictionary', 'DICT_4X4_50')))
    only = h.get('marker_ids')
    default_size = float(h.get('marker_size_mm', 38.8)) / 1000.0
    per_id = {int(k): float(v) / 1000.0 for k, v in (h.get('marker_sizes_mm') or {}).items()}
    views, K, D = [], None, None
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for r in rows:
        img = cv2.imread(os.path.join(run_dir, f"raw_{r['k']:02d}.png"))
        if img is None:
            log.warning('  view %d: no raw_%02d.png -- left out.', r['k'], r['k'])
            continue
        corners = he.detect_corners(detector, img)
        if only:
            corners = {m: c for m, c in corners.items() if m in set(int(i) for i in only)}
        views.append({'T_base_tool0': np.asarray(r['T_base_tool0'], dtype=float), 'corners': corners,
                      'label': r['label']})
        K, D = np.asarray(r['K'], dtype=float), np.asarray(r['D'], dtype=float)
    log.info('RE-SOLVING %s: %d views.', run_dir, len(views))
    sizes = {m: per_id.get(m, default_size) for v in views for m in v['corners']}
    return _solve_and_report(cfg, run_dir, views, sizes, K, D, from_cfg(cfg.section('hand_eye')), h)


def _bench(markers, T_base_cam):
    """(a point, the unit normal toward the camera) of the plane the markers lie on: their mean normal, through
    the lowest of them along it -- the bench, as the gripper must clear it."""
    n = np.mean([T[:3, 2] for T in markers.values()], axis=0)
    n /= np.linalg.norm(n)
    P = np.array([T[:3, 3] for T in markers.values()])
    if n @ (T_base_cam[:3, 3] - P.mean(axis=0)) < 0.0:
        n = -n
    return P[int(np.argmin(P @ n))], n


def _clearance(T_base_tool0, spheres, bench):
    """The tool's least distance above the bench (m): every collision sphere's centre less its radius."""
    p0, n = bench
    R, t = T_base_tool0[:3, :3], T_base_tool0[:3, 3]
    return min(float((R @ c + t - p0) @ n) - r for c, r in spheres)


def _path_clearance(Ta, Tb, spheres, bench, n=12):
    return min(_clearance(slerp_matrix(Ta, Tb, a), spheres, bench) for a in np.linspace(0.0, 1.0, n))


def _face_on(T_base_marker, T_base_cam, d):
    """The camera pose `d` from the marker along its normal, looking straight at its centre, turned about the
    line of sight as the camera is now (no wrist spin for nothing)."""
    nm = T_base_marker[:3, 2] * (1.0 if T_base_marker[:3, 2] @ (T_base_cam[:3, 3] - T_base_marker[:3, 3]) > 0 else -1.0)
    z = -nm
    x = T_base_cam[:3, 0] - (T_base_cam[:3, 0] @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(T_base_cam[:3, 1], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = T_base_marker[:3, 3] + d * nm
    return T


def _about(T_base_tool0, point, axis, angle):
    """T_base_tool0 turned by `angle` about a base-frame `axis` through a base-frame `point`."""
    R = np.eye(4)
    R[:3, :3] = Rotation.from_rotvec(angle * np.asarray(axis) / np.linalg.norm(axis)).as_matrix()
    return translation_matrix(point) @ R @ translation_matrix(-np.asarray(point)) @ T_base_tool0


def closeups(cfg, arm, camera, detector, views, sizes, K, D, X_cfg, out, only, settle, scale, cu, assume_yes, h, move):
    """The close-up stage (module docstring, THE CLOSE-UPS): appends its views to `views`. False only when the
    arm should stop (a move refused or failed); a marker it cannot do is skipped."""
    from ..robot import tool_geometry as G
    d_near = float(cu.get('distance_mm', 150.0)) / 1000.0
    d_far = float(cu.get('max_distance_mm', 300.0)) / 1000.0
    clear = float(cu.get('min_clearance_mm', 40.0)) / 1000.0
    tilt = np.radians(float(cu.get('tilt_deg', 10.0)))
    spin = np.radians(float(cu.get('spin_deg', 20.0)))
    iters = int(cu.get('servo_iterations', 4))
    tol_px = float(cu.get('centre_tol_px', 10.0))
    tol_m = float(cu.get('distance_tol_mm', 3.0)) / 1000.0
    try:
        r0 = he.solve(views, K, D, sizes, max_view_rms_px=float(h.get('max_view_rms_px', 1.5)))
    except ValueError as exc:
        log.warning('CLOSE-UPS skipped: the sweep alone does not solve (%s).', exc)
        return True
    X = r0['X']
    markers = dict(r0['markers'])
    cap = int(cu.get('max_markers', 0) or 0)
    spheres = [(np.asarray(c, dtype=float), float(r)) for c, r in G.tool_spheres(cfg)]
    A_now = arm.fk(np.asarray(arm.q()))
    bench = _bench(markers, A_now @ X)
    log.info('CLOSE-UPS: the sweep\'s first solve puts %d markers %s on the bench (its normal %s in base); X so far '
             '%.1f mm / %.2f deg from the configured one.', len(markers), sorted(markers), np.round(bench[1], 3),
             1000.0 * pose_error(X, X_cfg)[0], np.degrees(pose_error(X, X_cfg)[1]))
    if not _ask(assume_yes, f'The camera will visit {len(markers) if not cap else min(cap, len(markers))} markers, '
                f'servoing to {1000 * d_near:.0f} mm from each (further where the gripper would come within '
                f'{1000 * clear:.0f} mm of the bench), at {scale:.0%} speed, guarded. Hand on the e-stop?'):
        return False
    cx, cy = float(K[0, 2]), float(K[1, 2])
    done, skipped = [], []
    todo = sorted(markers)
    k = len(views)
    while todo and (not cap or len(done) < cap):
        cam = A_now @ X
        m = min(todo, key=lambda i: np.linalg.norm(markers[i][:3, 3] - cam[:3, 3]))     # the nearest next
        todo.remove(m)
        Mm = markers[m]

        def target(Mm_, A_):
            """The nearest face-on tool0 pose that clears the bench, its distance; (None, d) without one."""
            d = d_near
            while d <= d_far + 1e-9:
                T = _face_on(Mm_, A_ @ X, d) @ inverse(X)
                if (_clearance(T, spheres, bench) >= clear
                        and _path_clearance(A_, T, spheres, bench) >= min(clear, _clearance(A_, spheres, bench))):
                    return T, d
                d += 0.010
            return None, d

        def reach(T):
            q = np.asarray(arm.q())
            sol = arm.ik(T, qnear=q)
            turn = pose_error(arm.fk(q), T)[1]
            return (sol is not None and arm.joints_ok(sol)
                    and np.max(np.abs(np.asarray(sol) - q)) <= np.radians(60.0) + turn)

        T, d = target(Mm, A_now)
        if T is None or not reach(T):
            skipped.append(m)
            log.warning('  marker %2d: %s -- skipped.', m, 'the gripper would not clear the bench within %.0f mm'
                        % (1000 * d_far) if T is None else 'its face-on pose is out of reach on this wrist branch')
            continue
        res = move(T, f'close-up {m}')
        if res == 'refused':
            skipped.append(m)
            log.warning('  marker %2d: the arm will not take the line to it (said above) -- skipped.', m)
            continue
        if res is not True:
            return 'stopped'
        # ---- SERVO: re-measure, correct, until it is centred at its distance ----
        ok, why = False, 'not seen'
        for it in range(iters + 1):
            A_now, frame, corners = _capture(arm, camera, detector, settle, only)
            if m not in corners:
                why = 'lost from view'
                break
            pred = inverse(A_now @ X) @ Mm
            T_cm = min(he.marker_poses(corners[m], sizes[m], K, D),
                       key=lambda Tc: np.linalg.norm(Tc[:3, 3] - pred[:3, 3]) + 0.05 * np.degrees(
                           Rotation.from_matrix(Tc[:3, :3].T @ pred[:3, :3]).magnitude()))
            c = corners[m].mean(axis=0)
            off_px = float(np.hypot(c[0] - cx, c[1] - cy))
            dist = float(T_cm[2, 3])
            Mm = A_now @ X @ T_cm                                  # the marker, re-measured from here
            if off_px <= tol_px and abs(dist - d) <= tol_m:
                ok = True
                break
            if it == iters:
                break                                              # out of steps: this frame is where it is
            T, d = target(Mm, A_now)
            if T is None or not reach(T):
                why = 'its corrected pose is out of reach'
                break
            log.info('    servo %d: marker %d %.0f px off the centre, %.1f mm away (want %.0f) -> correcting',
                     it + 1, m, off_px, 1000 * dist, 1000 * d)
            res = move(T, f'close-up {m} servo {it + 1}')
            if res == 'refused':
                why = 'a correction the arm will not take'
                break
            if res is not True:
                return 'stopped'
        if not ok:
            if why == 'not seen':
                why = 'still %.0f px / %.1f mm off after %d steps' % (off_px, 1000 * abs(dist - d), iters)
            if why.startswith('still'):
                log.info('  marker %2d: %s -- its views taken anyway.', m, why)
            else:
                skipped.append(m)
                log.warning('  marker %2d: %s -- skipped.', m, why)
                continue
        side = float(np.mean(np.linalg.norm(np.roll(corners[m], -1, axis=0) - corners[m], axis=1)))
        log.info('  marker %2d: centred %.0f px off at %.1f mm, %.0f px across', m, off_px, 1000 * dist, side)
        views.append(_record(out, k, f'close {m}', A_now, frame, corners, detector))
        k += 1
        # ---- AROUND IT: two axes across the line of sight, and the line of sight ----
        C = A_now @ X
        A_c = A_now
        for label, axis, ang in ((f'close {m} tilt +x', C[:3, 0], tilt), (f'close {m} tilt -x', C[:3, 0], -tilt),
                                 (f'close {m} tilt +y', C[:3, 1], tilt), (f'close {m} tilt -y', C[:3, 1], -tilt),
                                 (f'close {m} spin +', C[:3, 2], spin), (f'close {m} spin -', C[:3, 2], -spin)):
            Tv = _about(A_c, Mm[:3, 3], axis, ang)
            if (_clearance(Tv, spheres, bench) < clear or _path_clearance(arm.fk(np.asarray(arm.q())), Tv, spheres,
                                                                          bench) < clear or not reach(Tv)):
                log.info('    %s: skipped (the bench or the reach)', label)
                continue
            res = move(Tv, label)
            if res == 'refused':
                log.info('    %s: skipped (the arm will not take the line, said above)', label)
                continue
            if res is not True:
                return 'stopped'
            A_now, frame, corners = _capture(arm, camera, detector, settle, only)
            views.append(_record(out, k, label, A_now, frame, corners, detector))
            k += 1
        done.append(m)
    # ---- UP OFF THE BENCH before the straight line back to the start: along the bench's normal to the start's
    # height, so the line back cannot sweep the gripper across it
    T0 = np.asarray(views[0]['T_base_tool0'], dtype=float)
    A_now = arm.fk(np.asarray(arm.q()))
    rise = float((T0[:3, 3] - A_now[:3, 3]) @ bench[1])
    if rise > 0.002:
        if move(translation_matrix(rise * bench[1]) @ A_now, 'close-ups: up off the bench') is not True:
            return 'stopped'
    log.info('CLOSE-UPS: %d markers %s seen close%s.', len(done), done,
             '; skipped %s' % skipped if skipped else '')
    return True


def _ask(assume_yes, question):
    if assume_yes:
        return True
    try:
        answer = input(question + ' [yes/N] ')
    except EOFError:
        answer = ''
    if answer.strip().lower() != 'yes':
        log.info('Stopped; nothing moved.')
        return False
    return True


def _capture(arm, camera, detector, settle, only):
    """(T_base_tool0, frame, corners) with the arm STILL: the pose is read either side of the
    exposure, and a view taken while it moved is retaken."""
    for _ in range(5):
        time.sleep(settle)
        q0 = np.asarray(arm.q())
        frame = camera.capture()
        q1 = np.asarray(arm.q())
        if np.max(np.abs(q1 - q0)) < np.radians(0.01):
            corners = he.detect_corners(detector, frame.color)
            if only:
                corners = {m: c for m, c in corners.items() if m in set(int(i) for i in only)}
            return arm.fk(q1), frame, corners
    raise RuntimeError('the arm kept moving through the exposure')


def _record(out, k, label, T, frame, corners, detector):
    # The raw frame too, lossless: a run can then be re-solved (other sizes, other markers)
    # without moving the arm again -- the annotated jpg's outlines sit on the corners. AND ITS POSE, NOW
    # (2026-11-12 19:43: a run stopped at view 70 lost all 70 -- the poses were only in memory until the solve):
    # poses.jsonl, one line a view, with the intrinsics the frame was taken under; --resolve solves from them.
    cv2.imwrite(os.path.join(out, f'raw_{k:02d}.png'), frame.color)
    with open(os.path.join(out, 'poses.jsonl'), 'a') as fh:
        fh.write(json.dumps({'k': k, 'label': label, 'T_base_tool0': np.asarray(T, dtype=float).tolist(),
                             'K': np.asarray(frame.K, dtype=float).tolist(),
                             'D': np.asarray(frame.D, dtype=float).ravel().tolist()}) + '\n')
    img = frame.color.copy()
    for m, c in corners.items():
        cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(img, str(m), tuple(int(v) for v in c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
    cv2.putText(img, f'{k:02d} {label}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out, f'view_{k:02d}.jpg'), img)
    return {'T_base_tool0': T, 'corners': corners, 'label': label}


def _fmt(T):
    xyz, rpy = matrix_to_xyzrpy(T)
    return (f'xyz_mm: [{xyz[0] * 1000:.2f}, {xyz[1] * 1000:.2f}, {xyz[2] * 1000:.2f}]',
            'rpy_deg: [{:.3f}, {:.3f}, {:.3f}]'.format(*np.degrees(rpy)))


def _report(cfg, out, views, r, X_cfg):
    X = r['X']
    log.info('=' * 78)
    log.info('HAND-EYE (tool0 -> camera color optical), %d views, %s:', len(views),
             f'dropped {r["dropped"]}' if r['dropped'] else 'none dropped')
    xyz_s, rpy_s = _fmt(X)
    log.info('  %s   %s', xyz_s, rpy_s)
    lin, ang = pose_error(X, X_cfg)
    log.info('  differs from the configured hand_eye by %.1f mm, %.2f deg.', lin * 1000.0,
             np.degrees(ang))
    log.info('  reprojection %.2f px RMS (per view: %s)', r['rms_px'],
             ', '.join('-' if v is None else f'{v:.2f}' for v in r['view_rms_px']))
    for m, (rms_mm, max_mm, rms_deg, max_deg) in sorted(r['scatter'].items()):
        Tm = r['markers'][m]
        log.info('  marker %2d at z = %+.1f mm: single-view scatter %.2f mm RMS (max %.2f), '
                 '%.2f deg RMS (max %.2f)', m, Tm[2, 3] * 1000.0, rms_mm, max_mm, rms_deg,
                 max_deg)
    log.info('  views turned up to %.1f deg; about a second axis %.1f deg.', r['span_deg'],
             r['second_axis_deg'])
    worst_mm = max(s[0] for s in r['scatter'].values())
    # A single view's PnP of a small marker is good to ~1% of its range, so a few mm of scatter
    # is normal; one marker far above the rest is not -- it is the signature of a WRONG SIZE
    # (PnP puts it at the wrong depth in every view). A size wrong on EVERY marker shows nowhere:
    # it slides the camera along its line of sight, and the fit is as good either way.
    med_mm = float(np.median([s[0] for s in r['scatter'].values()]))
    odd = sorted(m for m, s in r['scatter'].items() if s[0] > max(3.0 * med_mm, 3.0))
    if odd:
        log.warning('Marker(s) %s scatter far more than the rest (%.1f mm median): most likely '
                    'printed at another size. Measure them and set marker_sizes_mm, or leave '
                    'them out with marker_ids, and solve again.', odd, med_mm)
    good = r['rms_px'] < 1.5 and not odd and r['second_axis_deg'] > 3.0
    (log.info if good else log.warning)(
        'Quality: %s.', 'GOOD' if good else
        'CHECK -- want reprojection < 1.5 px, no marker scattering far above the rest and a '
        'second axis > 3 deg. Look at the worst views\' images (blur, a marker cut by the frame '
        'edge, a lifted corner)')
    z = np.median([T[2, 3] for T in r['markers'].values()]) * 1000.0
    cell = cfg.get('_cell_path', 'the cell file')
    with open(os.path.join(out, 'result.yaml'), 'w') as fh:
        fh.write(f'# hand_eye_calibration, {len(views)} views, {r["rms_px"]:.2f} px RMS, '
                 f'scatter <= {worst_mm:.2f} mm\n')
        fh.write(f'# -> {cell}\nhand_eye:\n  {xyz_s}\n  {rpy_s}\n')
        fh.write(f'# The markers lie on the bench, so this is its height (less the paper):\n'
                 f'ground_plane:\n  z_mm: {z:.1f}\n')
        fh.write(f'# -> the cell\'s frames file, frames: section\ncamera:\n  {xyz_s}\n'
                 f'  {rpy_s}\n')
    with open(os.path.join(out, 'views.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['view', 'label', 'x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg',
                     'yaw_deg', 'markers', 'rms_px'])
        for k, v in enumerate(views):
            xyz, rpy = matrix_to_xyzrpy(v['T_base_tool0'])
            rms = r['view_rms_px'][k]
            wr.writerow([k, v['label'], *np.round(xyz * 1000.0, 2), *np.round(np.degrees(rpy), 3),
                         ' '.join(map(str, sorted(v['corners']))),
                         '' if rms is None else f'{rms:.3f}'])
    log.info('Bench height from the markers: ground_plane.z_mm = %.1f (less the paper).', z)
    log.info('Written: %s', os.path.join(out, 'result.yaml'))


if __name__ == '__main__':
    main()
