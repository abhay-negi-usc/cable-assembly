"""Splat scan -- from where the arm is, the wrist camera tilts and pans about the point it looks at
on the bench, so the live view's camera layer fills in from every side.

    cd sim && ./run.sh python scripts/live_view_gl.py                 # watch it build (any order)
    python -m cable_assembly.apps.splat_scan                          # the iiwa bench, from here
    python -m cable_assembly.apps.splat_scan --set splat_scan.from_pick=true   # from pick_joints_deg
    python -m cable_assembly.apps.splat_scan --dry-run --set splat_scan.from_pick=true   # rehearse
    python -m cable_assembly.apps.splat_scan --check             # the latest scan's agreement (or --check DIR)

THE PATH. The pivot is the point the camera looks at: along its line of sight, at the depth the
D415 reads at the image centre (where the line meets the bench, in a dry run). The camera then
tilts `tilts_deg` off that line about the pivot, toward `pans` directions evenly round it -- a
ring of views per tilt, each ring walked in order and the next one back the other way, so every
move is a short step and the pivot stays in the middle of every view.

CHECKED BEFORE ANYTHING MOVES, for every view and the middle of every move: reachable on the same
IK branch as the one before (a wrist flip is a move a straight line cannot make), within the joint
limits, and every link frame from the shoulder out and the fingertips at least
`min_clearance_mm` above the bench. A view that fails is left out, said which and why; the scan
does not start if more than half are, or if there is no safe line back to the start.

THE MOVES, as the hand-eye sweep makes them: it asks first (--yes skips), then straight lines at
`speed_scale` of the caps with the force guard armed (`guard_force_n`), `settle_s` before each
exposure and the joints read either side of it (a view taken while the arm moved is retaken), and
back to the start at the end. If a move stops short (the guard, a fault) or the scan is
interrupted, the arm stays where it stopped: nothing moves it again unasked.

THE OUTPUT. With live_view.frames on (the iiwa bench), every frame goes to the viewer as it is
taken, and the camera layer builds as you watch. Each view is also kept, in
data/experiments/splat_scan_<stamp>/:
    view_NN.npz        capture_rgbd's frame.npz format: live_view_gl.py --rgbd takes them
    view_NN.png, view_NN_depth.png (uint16, mm)
    transforms.json    the views for a splat trainer: nerfstudio's format (OpenGL camera axes:
                       x right, y up, looking down -z), depth in mm

HOW WELL THE VIEWS AGREE is logged at the end (and --check works it out for a scan already
taken, no robot): each view's depth, placed by its own pose, read against the start view's along
the start view's rays -- the median difference per view. Pose error (hand-eye, kinematics) shows
here, and so would a view taken while the camera still shook. The first scans (2026-10-29/30,
at 30% speed) agreed to 0.5-1.5 mm per view, the same in both runs to 0.2 mm: the pattern follows
the pan, so it is calibration residual, and a faster scan that agrees as well is as accurate.
"""

import concurrent.futures
import glob
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .. import log as urlog
from ..transforms import inverse, pose_error, slerp_matrix, translation_matrix
from ._common import experiment_dir, latest_experiment
from ._runner import run_app
from .capture_rgbd import save_frame, still_frames

log = urlog.get('splat-scan')

# The plan walks every line as the iiwa's move_l will (backends/iiwa_fri.py): an IK waypoint at
# least every millimetre and half degree, and a joint jump past 5 deg between two is a singularity
# or a branch change, which move_l refuses. Checked only at its middle and end, a line move_l
# refuses at 96% passed (tilts [15, 30, 45] from the pick pose), and the scan would stop there.
_LINE_STEP_M = 0.001
_LINE_STEP_RAD = np.radians(0.5)
_LINE_MAX_JOINT_STEP = np.radians(5.0)
# OpenCV camera axes (x right, y down, z forward) -> OpenGL's (x right, y up, z back).
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def main():
    if len(sys.argv) in (2, 3) and sys.argv[1] == '--check':     # a scan already taken: no robot
        from .. import config as urconfig
        urlog.setup()
        cfg = urconfig.load('splat_scan', [])
        out = sys.argv[2] if len(sys.argv) == 3 else latest_experiment(cfg, 'splat_scan')
        paths = sorted(glob.glob(os.path.join(out, 'view_*.npz'))) if out else []
        if not paths:
            sys.exit(f'splat_scan --check: no views in {out or "any splat scan"}.')
        log.info('%s: %d views.', out, len(paths))
        sys.exit(0 if report_agreement(paths) else 1)
    run_app('Splat scan: tilt and pan the wrist camera about what it looks at', 'splat_scan', scan,
            with_gripper=False, needs_camera=True, needs=('hand_eye', 'fingertip', 'ground_plane'))


def scan(cfg, robot, camera, args):
    s = cfg.section('splat_scan')
    arm = robot.arm
    scale = float(s.get('speed_scale', 0.3))
    settle = float(s.get('settle_s', 0.5))
    n_frames = int(s.get('frames_per_view', 5))
    bench_z = float(cfg.get_path('ground_plane.z_mm')) / 1000.0
    clearance = float(s.get('min_clearance_mm', 60.0)) / 1000.0
    tilts = np.radians(np.asarray(s.get('tilts_deg', [15.0, 25.0, 35.0]), dtype=float))
    pans = int(s.get('pans', 8))
    arm.set_speed_scale(scale, 'splat scan')

    if s.get('from_pick'):
        q_pick = np.radians(np.asarray(cfg.get('pick_joints_deg'), dtype=float))
        if not _ask(args.yes, f'Move to pick_joints_deg {list(np.round(np.degrees(q_pick), 1))} first?'):
            return False
        if not arm.move_j(q_pick, label='pick pose', caps=scale):
            return False
    q0 = np.asarray(arm.q(), dtype=float)
    if not arm.joints_ok(q0):
        log.error('The arm starts outside its joint limits; jog it back inside first.')
        return False
    T0 = arm.fk(q0)

    # The start view, which also finds the pivot.
    time.sleep(settle)
    got = still_frames(camera, arm.q, n_frames)
    if got is None:
        log.error('The arm is moving; start the scan with it still.')
        return False
    frames0 = got[1]
    pivot = look_point(frames0, robot.T_tool0_cam, T0, bench_z)
    if pivot is None:
        log.error('The camera does not look down at the bench from here: no point to scan about.')
        return False
    cam0 = (T0 @ robot.T_tool0_cam)[:3, 3]
    log.info('Pivot: (%.3f, %.3f, %.3f) m, %.0f mm from the camera.', *pivot, 1000 * np.linalg.norm(pivot - cam0))

    plan = orbit_views(T0, inverse(T0) @ np.append(pivot, 1.0), (inverse(T0) @ np.append(cam0, 1.0)),
                       tilts, pans)
    n_planned = len(plan)
    arm_log = urlog.get('arm')
    level = arm_log.level
    arm_log.setLevel(logging.CRITICAL)     # the IK's "no solution" for views being tried is not news
    try:
        plan = check_plan(arm, q0, T0, plan, robot.T_tool0_fingertip, bench_z, clearance)
    finally:
        arm_log.setLevel(level)
    if plan is None:
        return False
    if not _ask(args.yes, f'The camera will tilt up to {np.degrees(tilts.max()):.0f} deg about a point '
                f'{1000 * np.linalg.norm(pivot - cam0):.0f} mm in front of it: {len(plan)} views '
                f'({n_planned - len(plan)} of {n_planned} left out), at {scale:.0%} speed, guarded, '
                f'then return here. Hand on the e-stop?'):
        return False

    # A dry run's camera has no depth and keeps nothing: no output directory for it.
    out = experiment_dir(cfg, 'splat_scan') if frames0[-1].depth is not None else None
    model, tip = arm.model, robot.T_tool0_fingertip
    # Each view is written (median depth, npz, pngs: ~0.4 s) by a thread of its own while the arm
    # goes on to the next; move_l sends its line ahead, so the thread cannot hold it up.
    writer = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='splat-keep')
    views = [writer.submit(_keep, out, 0, 'start', frames0, got[0], model, robot.T_tool0_cam, tip, cfg)]
    arm.zero_ft()
    f_max = float(s.get('guard_force_n', 15.0))
    arm.add_guard(lambda: arm.force() > f_max)
    ok = back = False
    try:
        for k, (label, T) in enumerate(plan, start=1):
            if not arm.move_l(T, label=label, caps=scale):
                log.error('The move to view %d (%s) stopped short; the arm stays there.', k, label)
                break
            time.sleep(settle)
            got = None
            for _ in range(5):
                got = still_frames(camera, arm.q, n_frames)
                if got is not None:
                    break
                time.sleep(settle)
            if got is None:
                log.error('View %d: the arm kept moving through the exposure; the arm stays there.', k)
                break
            views.append(writer.submit(_keep, out, k, label, got[1], got[0], model, robot.T_tool0_cam, tip, cfg))
        else:
            ok = True
            back = arm.move_l(T0, label='back to the start', caps=scale)
            if not back:
                log.error('The move back to the start stopped short: check the arm.')
    finally:
        arm.clear_guards()
        writer.shutdown(wait=True)
    views = [v for v in (f.result() for f in views) if v is not None]
    if views:
        write_transforms(out, views)
        report_agreement(sorted(glob.glob(os.path.join(out, 'view_*.npz'))))
        log.info('%d views in %s. To see them again: cd sim && ./run.sh python scripts/live_view_gl.py '
                 '--rgbd latest (or --rgbd %s); their collision scene: python -m '
                 'cable_assembly.apps.scene_build --show', len(views), out, os.path.abspath(out))
    return ok and back


def look_point(frames, T_tool0_cam, T_base_tool0, bench_z):
    """The point the camera looks at, base frame: along its line of sight, at the median depth of
    the image's central patch -- or, with no depth (a dry run), where the line meets the bench.
    None if it does not look down at the bench."""
    T = T_base_tool0 @ T_tool0_cam
    o, d = T[:3, 3], T[:3, 2]
    depth = frames[-1].depth
    if depth is not None:
        h, w = depth.shape
        patch = np.stack([f.depth[h // 2 - h // 20:h // 2 + h // 20, w // 2 - w // 20:w // 2 + w // 20]
                          for f in frames])
        seen = patch[patch > 0.0]
        if seen.size > 0.2 * patch.size:
            return o + float(np.median(seen)) * d
    if d[2] > -0.3:
        return None
    return o + (bench_z - o[2]) / d[2] * d


def orbit_views(T0, pivot_tool0, cam_tool0, tilts, pans):
    """tool0 targets: the camera tilted by each of `tilts` about the pivot, toward `pans`
    directions round its line of sight; ring by ring, alternate rings walked back the other way.
    `pivot_tool0` and `cam_tool0` are points in tool0's frame (homogeneous or not)."""
    p, c = np.asarray(pivot_tool0, dtype=float)[:3], np.asarray(cam_tool0, dtype=float)[:3]
    a = (p - c) / np.linalg.norm(p - c)                       # the line of sight
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    P = translation_matrix(p)
    views = []
    for i, tilt in enumerate(tilts):
        order = range(pans) if i % 2 == 0 else range(pans - 1, -1, -1)
        for j in order:
            th = 2.0 * np.pi * j / pans
            R = np.eye(4)
            R[:3, :3] = Rotation.from_rotvec(tilt * (np.cos(th) * u + np.sin(th) * w)).as_matrix()
            views.append((f'tilt {np.degrees(tilt):.0f} deg, pan {np.degrees(th):.0f} deg',
                          T0 @ P @ R @ inverse(P)))
    return views


def check_plan(arm, q0, T0, plan, T_tool0_tip, bench_z, clearance):
    """Every view and every straight move to it, walked as move_l will: on one IK branch, inside
    the limits, and clear of the bench all the way. A view that fails is left out (said which, and
    why) and the next is checked from the last one kept. Returns the views kept, or None -- nothing
    to do, more than half left out, or no way back to the start."""
    q, T_prev, kept = np.asarray(q0, dtype=float), T0, []
    for label, T in [*plan, ('back to the start', T0)]:
        q_try, why = q, None
        lin, ang = pose_error(T_prev, T)
        n = max(1, int(np.ceil(lin / _LINE_STEP_M)), int(np.ceil(ang / _LINE_STEP_RAD)))
        for k in range(1, n + 1):
            what = 'there' if k == n else f'{100.0 * k / n:.0f}% of the way there'
            sol = arm.ik(slerp_matrix(T_prev, T, k / n), qnear=q_try)
            if sol is None or not arm.joints_ok(sol) or np.max(np.abs(np.asarray(sol) - q_try)) > _LINE_MAX_JOINT_STEP:
                why = f'out of reach {what} without a wrist flip, or past a joint limit'
                break
            q_try = np.asarray(sol, dtype=float)
            # From the shoulder out: the base and link 1 sit on the table, below the bench top.
            low = min(float(np.min(arm.model.fk_links(q_try)[2:, 2, 3])),
                      float((arm.fk(q_try) @ T_tool0_tip)[2, 3]))
            if low - bench_z < clearance:
                why = (f'{what}, the arm comes within {1000 * (low - bench_z):.0f} mm of the bench '
                       f'(min_clearance_mm {1000 * clearance:.0f})')
                break
        if why is not None:
            if label == 'back to the start':
                log.error('No safe straight line back to the start from "%s": %s.', kept[-1][0] if kept else 'start', why)
                return None
            log.warning('  leaving out "%s": %s.', label, why)
            continue
        if label != 'back to the start':
            kept.append((label, T))
        q, T_prev = q_try, T
    if len(kept) < max(1, len(plan) // 2):
        log.error('Only %d of %d views are reachable from here. Start with A5, A6 and A7 nearer '
                  'mid-range, or lower tilts_deg.', len(kept), len(plan))
        return None
    return kept


def agreement(views, stride=4, window_m=0.03):
    """How well views (frame.npz dicts, the start view first) agree: each later view's depth,
    placed by its own pose, read against the start view's depth along the start view's rays.
    Per later view: the differences, m, where both read (those past `window_m` -- an edge,
    something one of them sees round -- left out)."""
    ref = views[0]
    Tr, Kr, dr = np.asarray(ref['T_base_cam'], dtype=float), ref['K'], ref['depth_m']
    out = []
    for f in views[1:]:
        d, K, T = f['depth_m'], f['K'], np.asarray(f['T_base_cam'], dtype=float)
        v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[v, u]
        ok = z > 0.0
        u, v, z = u[ok], v[ok], z[ok].astype(float)
        p = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1) @ T[:3, :3].T + T[:3, 3]
        pc = (p - Tr[:3, 3]) @ Tr[:3, :3]
        pc = pc[pc[:, 2] > 0.05]
        ur = np.rint(Kr[0, 0] * pc[:, 0] / pc[:, 2] + Kr[0, 2]).astype(np.int64)
        vr = np.rint(Kr[1, 1] * pc[:, 1] / pc[:, 2] + Kr[1, 2]).astype(np.int64)
        inside = (ur >= 0) & (ur < dr.shape[1]) & (vr >= 0) & (vr < dr.shape[0])
        seen = dr[vr[inside], ur[inside]]
        diff = seen - pc[inside, 2]
        diff = diff[(seen > 0.0) & (np.abs(diff) < window_m)]
        out.append(diff)
    return out


def report_agreement(paths, warn_mm=3.0):
    """Log how well a scan's views agree (agreement); False if any view is off by more than
    `warn_mm` (median) or shares nothing with the start view."""
    views = []
    for p in paths:
        with np.load(p) as z:
            views.append({k: z[k] for k in ('depth_m', 'K', 'T_base_cam')})
    if len(views) < 2:
        return True
    diffs = agreement(views)
    med = np.array([1000.0 * np.median(np.abs(d)) if len(d) else np.inf for d in diffs])
    worst = int(np.argmax(med))
    log.info('The views against the start view: median depth difference %.1f-%.1f mm per view '
             '(%s), worst view %d.', med.min(), med.max(), ' '.join(f'{m:.1f}' for m in med), worst + 1)
    if med[worst] > warn_mm:
        log.warning('View %d disagrees with the start view by %.1f mm: was the arm still at the '
                    'exposure (settle_s), and is the hand-eye current?', worst + 1, med[worst])
        return False
    return True


def _keep(out, k, label, frames, q, model, T_tool0_cam, T_tool0_tip, cfg):
    """One view to view_NN.npz/.png/_depth.png; its transforms.json entry, or None without depth."""
    if frames[-1].depth is None:
        log.info('  view %2d  %-28s (no depth: a dry run keeps nothing)', k, label)
        return None
    name = f'view_{k:02d}'
    depth, T_base_cam = save_frame(os.path.join(out, f'{name}.npz'), frames, q, model, T_tool0_cam,
                                   T_tool0_tip, cfg)
    cv2.imwrite(os.path.join(out, f'{name}.png'), frames[-1].color)
    cv2.imwrite(os.path.join(out, f'{name}_depth.png'), np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16))
    log.info('  view %2d  %-28s depth on %.0f%% of the pixels', k, label, 100.0 * (depth > 0).mean())
    return {'file_path': f'{name}.png', 'depth_file_path': f'{name}_depth.png',
            'transform_matrix': (T_base_cam @ _CV_TO_GL).tolist(), 'K': np.asarray(frames[-1].K).tolist(),
            'w': depth.shape[1], 'h': depth.shape[0]}


def write_transforms(out, views):
    """transforms.json, nerfstudio's format: one camera (the D415's colour), the views' poses."""
    K = views[0]['K']
    doc = {'camera_model': 'OPENCV', 'fl_x': K[0][0], 'fl_y': K[1][1], 'cx': K[0][2], 'cy': K[1][2],
           'w': views[0]['w'], 'h': views[0]['h'], 'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
           'depth_unit_scale_factor': 0.001,
           'frames': [{k: v[k] for k in ('file_path', 'depth_file_path', 'transform_matrix')} for v in views]}
    with open(os.path.join(out, 'transforms.json'), 'w') as f:
        json.dump(doc, f, indent=1)


def _ask(assume_yes, question):
    if assume_yes:
        return True
    try:
        answer = input(question + ' [yes/N] ')
    except EOFError:
        answer = ''
    if answer.strip().lower() != 'yes':
        log.info('Stopped.')
        return False
    return True


if __name__ == '__main__':
    main()
