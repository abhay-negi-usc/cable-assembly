"""Hand-eye calibration -- where the wrist camera sits on the flange (tool0 -> color optical),
measured from ArUco markers lying still on the bench.

    python -m cable_assembly.apps.hand_eye_calibration                       # the iiwa bench
    python -m cable_assembly.apps.hand_eye_calibration --cell ur10e_bench    # any other bench
    python -m cable_assembly.apps.hand_eye_calibration --dry-run             # rehearse

Talks to the ARM and the CAMERA only (no gripper, no Robot), so it runs on a cell that is not
calibrated yet -- which is when it is needed.

THE SETUP. Lay one or more DICT_4X4_50 markers FLAT and STILL on the bench, each with its white
border intact and none overlapping another. Their side lengths must be right (solvePnP scales
depth by it, and nothing in the image can tell): `marker_size_mm`, or per id in
`marker_sizes_mm`. Bigger is better -- an 80 mm marker pins the tilt far better than a 38.8 mm
one. Put the arm where the camera looks DOWN at them from about 25 cm (`start_joints_deg`, or
jog there on the pendant before starting the FRI app).

THE SWEEP. The camera tilts `tilt_deg` toward eight directions and turns `spin_deg` both ways
about the marker in front of it, then repeats two views from `raise_mm` further back -- turning
about two different axes is what the solve needs. The force guard is armed throughout; every
move is a straight line at `speed_scale` of the caps; the arm returns to the start at the end.

THE RESULT goes to data/experiments/hand_eye_calibration_<stamp>/: result.yaml (the blocks to
paste into the cell and its frames file), views.csv, and every image annotated. The markers'
base height is printed too: on a marker lying on the bench, that IS the bench height
(ground_plane.z_mm), less the paper's thickness.
"""

import csv
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
from ..transforms import (from_cfg, inverse, matrix_to_xyzrpy, pose_error, translation_matrix,
                          xyzrpy_to_matrix)
from ._common import experiment_dir

log = urlog.get('hand-eye')


def main():
    parser = urconfig.arg_parser('Hand-eye calibration: where the camera sits on the flange',
                                 'hand_eye_calibration')
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    urlog.setup()

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
    q = np.asarray(arm.q())
    for label, T in plan:
        sol = arm.ik(T, qnear=q)
        if sol is None or not arm.joints_ok(sol) or np.max(np.abs(np.asarray(sol) - q)) > np.radians(60.0):
            log.error('View "%s" is out of reach from here without flipping the wrist. Start '
                      'with A6 well away from 0 (60-100 deg; near 0 the wrist is singular and '
                      'a tilt swings A5/A7 through tens of degrees) and A5, A7 nearer the '
                      'middle of their ranges, or lower tilt_deg / spin_deg.', label)
            return False
        q = np.asarray(sol)
    if not _ask(assume_yes, f'The arm will make {len(plan)} moves of up to '
                f'{float(h.get("tilt_deg", 15.0)):.0f} deg about a point '
                f'{1000.0 * np.linalg.norm(pivot):.0f} mm out from the flange, at {scale:.0%} '
                'speed, guarded, then return here. Hand on the e-stop?'):
        return False

    views = [_record(out, 0, 'start', T0, frame, corners, detector)]
    arm.zero_ft()
    arm.add_guard(lambda: arm.force() > f_max)
    try:
        for k, (label, T) in enumerate(plan, start=1):
            if not arm.move_l(T, label=label, caps=scale):
                return False
            T_meas, frame, corners = _capture(arm, camera, detector, settle, only)
            lin, ang = pose_error(T_meas, T)
            log.info('  view %2d  %-34s markers %-10s (arrived within %.1f mm, %.2f deg)', k,
                     label, sorted(corners), 1000.0 * lin, np.degrees(ang))
            views.append(_record(out, k, label, T_meas, frame, corners, detector))
        if not arm.move_l(T0, label='back to start', caps=scale):
            return False
    finally:
        arm.clear_guards()

    sizes = {m: size_of(m) for v in views for m in v['corners']}
    try:
        r = he.solve(views, frame.K, frame.D, sizes,
                     max_view_rms_px=float(h.get('max_view_rms_px', 1.5)))
    except ValueError as exc:
        log.error('Cannot solve: %s', exc)
        return False
    _report(cfg, out, views, r, X_cfg)
    return r


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
    # without moving the arm again -- the annotated jpg's outlines sit on the corners.
    cv2.imwrite(os.path.join(out, f'raw_{k:02d}.png'), frame.color)
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
