"""The connector (the BNC's or an RJ45, pose6d.plug) and its socket found by the 6D pose
estimator, from several views.

    pose_method: pose6d      (bnc_assembly.yaml; the `pose6d:` block tunes it)

`estimate_part` is the shared procedure, for the connector (Pose6DScanner, the pick's scanner)
and the socket (bnc_assembly's target, assembly.target_source pose6d):
  1. A VIEW FROM WHERE THE ARM IS: the estimator's full search on it (perception/pose6d.py).
  2. AN ORBIT ABOUT WHAT IT FOUND: the camera tilts `pose6d.orbit.tilts_deg` off its line of sight
     to the part's middle, toward `pans` directions round it (apps/splat_scan.py's orbit_views;
     every line checked first as move_l will walk it, force-guarded, the app's step prompt
     first) and comes back.
  3. ALL THE VIEWS TOGETHER: the estimator again, the first estimate as the prior -- each view
     refined from it, the ones that disagree dropped, one pose refined against all the rest.
Not found in the first view, or the joint estimate failing the gate: None, said why. The views
are kept (`views`) for a collision scene.

Pose6DScanner gives the pick the junction frame every scanner gives (perception/pose6d.py
T_cord_junction): on the plug's axis where the cable leaves the strain relief, x toward the
mating face, z as near up as that allows -- for an RJ45 too, whose latch side is measured: that turn is
kept apart (`turn`, P6.grasp_junction), so a plug lying latch down is still grasped from above and the
belief carries the latch. It is ON the axis (pose_on_axis): no resting-height
correction. It never loads SAM3 (detector None).

AFTER THE SCAN, THROUGH THE PICK AND IN THE HAND (ConnectorTracker, pose6d.in_hand): the plug
keeps being estimated, one frame at a time, each from the last pose -- in base while it lies
still, in tool0 once the gripper holds it. bnc_assembly reads the in-hand pose before the
stand-off and, if enough estimates agree, takes it as the belief (measured_belief).

IN THE LIVE VIEW: the plug's CAD where it is estimated (overlay 'pose6d_connector', following the
gripper once held), its junction frame, each view's axis; the socket's CAD, the seated plug and
the target (overlay 'pose6d_socket', bnc_assembly).
"""

import atexit
import threading
import time

import numpy as np

from .. import log as urlog
from ..perception import pose6d as P6
from scipy.spatial.transform import Rotation

from ..transforms import inverse, pose_error, translation_matrix

log = urlog.get('pose6d')

OVERLAY_CONNECTOR, OVERLAY_SOCKET = 'pose6d_connector', 'pose6d_socket'
GOLD, ORANGE, GREY, BLUE = [1.0, 0.85, 0.2], [1.0, 0.55, 0.1], [0.55, 0.55, 0.55], [0.3, 0.8, 1.0]


# ---------------------------------------------------------------------------- the live view

def part_items(cfg, part, T_base_obj, color, radius=0.0007):
    """`part`'s CAD at `T_base_obj` as overlay points ([] if the service cannot give them)."""
    try:
        p = P6.service(cfg).model_points(part)
    except P6.Pose6DError as exc:
        log.debug('no CAD points for the %s: %s', part, exc)
        return []
    T = np.asarray(T_base_obj, dtype=float)
    return [{'type': 'points', 'points': (p @ T[:3, :3].T + T[:3, 3]).round(4).tolist(), 'color': color,
             'radius': radius}]


def axis_item(T_base_obj, part, color, half_m=0.025):
    """The part's axis through its middle, as a line."""
    T = np.asarray(T_base_obj, dtype=float)
    c, a = T[:3, :3] @ P6.CENTRE_M[part] + T[:3, 3], T[:3, 2]
    return {'type': 'line', 'points': [(c - half_m * a).round(4).tolist(), (c + half_m * a).round(4).tolist()],
            'color': color}


def send_overlay(robot, name, items, text=()):
    tx = getattr(robot, 'live_frames', None)
    if tx is None:
        return
    try:
        tx.send_overlay(name, items, text)
    except Exception as exc:                                       # noqa: BLE001 -- drawing only
        log.debug('%s overlay: %s', name, exc)


def mesh_item(cfg, part, T_base_obj, color, name=None):
    """`part`'s CAD (or objects.yaml's `name`) at `T_base_obj` as an overlay mesh (the file's path: the
    viewer reads it, as it does the held part's), or None without one."""
    mesh = P6.cad_mesh(cfg, part, name)
    if mesh is None:
        return None
    return {'type': 'mesh', 'mesh': str(mesh[0]), 'scale': float(mesh[1]),
            'T': np.asarray(T_base_obj, dtype=float).round(6).tolist(), 'color': color}


MOUNT_GREY = [0.28, 0.28, 0.32]


def draw_socket(cfg, robot, reply, T_target, socket=P6.SOCKET):
    """The socket's CAD where the views put it (a mesh; its points without one), the mount it sits in
    (P6.MOUNTS, dark grey), and its frame -- only those (the user: no plug drawn seated in it; the
    mount shown too, 2026-11-03). `T_target` is not drawn."""
    T_s = reply['T_base_obj']
    m = mesh_item(cfg, socket, T_s, BLUE)
    items = [m] if m is not None else part_items(cfg, socket, T_s, BLUE)
    if socket in P6.MOUNTS:
        mount = mesh_item(cfg, socket, T_s, MOUNT_GREY, name=P6.MOUNTS[socket])
        if mount is not None:
            items.append(mount)
    items.append({'type': 'pose', 'T': np.asarray(T_s, dtype=float).tolist(), 'length': 0.04})
    send_overlay(robot, OVERLAY_SOCKET, items,
                 [(f"6D pose: socket from {reply.get('n_used', 1)} views, score {reply.get('score', 0.0):.2f} "
                   f"({_terms(reply)})")])


def _orbit_cfg(cfg, override=None):
    """pose6d.orbit, with `override` (a dict of the same keys, e.g. assembly.visual_target.orbit for
    the socket) on top."""
    c = dict(cfg.section('pose6d') or {}) if cfg.get('pose6d') else {}
    o = {**dict(c.get('orbit') or {}), **dict(override or {})}
    return {'tilts': np.radians(np.asarray(o.get('tilts_deg', [20.0]), dtype=float)), 'pans': int(o.get('pans', 6)),
            'caps': dict(o.get('speed') or {'max_cartesian_translation_mm_s': 50.0, 'max_cartesian_rotation_deg_s': 60.0,
                                            'max_joint_velocity_deg_s': 60.0}),
            'settle_s': float(o.get('settle_s', 0.5)), 'frames': int(o.get('frames_per_view', 3)),
            'clearance_m': float(o.get('min_clearance_mm', 60.0)) / 1000.0,
            'guard_n': float(o.get('guard_force_n', 15.0)), 'enabled': bool(o.get('enabled', True)),
            'max_views': None if o.get('max_views') is None else int(o['max_views']),
            'low': [dict(v) for v in (o.get('low_views') or [])],
            'bench_z': float(cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0}


# STILL ENOUGH TO TAKE A VIEW (the user, 2026-11-03: less waiting, no image on the move, no blur).
STILL_POLL_S = 0.02           # how often the joints are read while waiting
STILL_WINDOW_S = 0.06         # still = no joint moved more than STILL_DEG over this long
STILL_DEG = 0.02
MAX_IMAGE_SHIFT_PX = 0.5      # first vs last frame of a view: more = the camera was still moving (blur)
VIEW_TRIES = 3


def wait_still(arm, max_s):
    """Until the arm is still (no joint more than STILL_DEG over STILL_WINDOW_S), at most `max_s`;
    the time waited. Replaces a fixed settle: a view is taken as soon as the arm has stopped."""
    t0 = time.monotonic()
    tol = np.radians(STILL_DEG)
    hist = [(t0, np.asarray(arm.q(), dtype=float))]
    while time.monotonic() - t0 < max_s:
        time.sleep(STILL_POLL_S)
        now = time.monotonic()
        hist.append((now, np.asarray(arm.q(), dtype=float)))
        hist = [h for h in hist if now - h[0] <= STILL_WINDOW_S + 1e-3]
        if now - t0 >= STILL_WINDOW_S and len(hist) >= 2:
            qs = np.array([q for _, q in hist])
            if np.max(np.ptp(qs, axis=0)) <= tol:
                break
    return time.monotonic() - t0


def image_shift_px(a, b):
    """How far the image moved between two colour frames (BGR), px: phase correlation on the central
    512 x 512 crop. 0 for frames with nothing in them (a dry run's)."""
    import cv2
    ga, gb = (cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (a, b))
    h, w = ga.shape
    c = (slice(max(0, h // 2 - 256), h // 2 + 256), slice(max(0, w // 2 - 256), w // 2 + 256))
    ga, gb = ga[c], gb[c]
    if ga.std() < 1.0 or gb.std() < 1.0:
        return 0.0
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    return float(np.hypot(dx, dy))


def still_view(camera, arm, n_frames, settle_s=0.5):
    """(joints, one Frame with the median depth of `n_frames`), or None if no still view came in
    VIEW_TRIES. Waits until the arm IS still (wait_still, at most `settle_s`), then takes the frames
    with the joints read either side; a view whose first and last frames differ by more than
    MAX_IMAGE_SHIFT_PX (the camera still moving -- blur) is taken again."""
    from ..apps.capture_rgbd import median_depth, still_frames
    for attempt in range(VIEW_TRIES):
        waited = wait_still(arm, settle_s)
        got = still_frames(camera, arm.q, n_frames)
        if got is None:
            log.debug('view: the joints moved through the exposure (try %d)', attempt + 1)
            continue
        frames = got[1]
        shift = image_shift_px(frames[0].color, frames[-1].color) if len(frames) > 1 else 0.0
        if shift > MAX_IMAGE_SHIFT_PX:
            log.debug('view: the image moved %.2f px across the exposure (try %d)', shift, attempt + 1)
            continue
        log.debug('view: still after %.2f s, image steady to %.2f px', waited, shift)
        frame = frames[-1]
        if frame.depth is not None:
            frame.depth = median_depth(frames)
        return np.asarray(got[0], dtype=float), frame
    return None


def _spread(plan, T_tool0_cam, cam0, n):
    """`n` of the planned views [(name, T_base_tool0)], spread as far apart as they go (their camera
    positions' closest pair farthest apart) -- the 2026-11-03
    socket data: 4 SPREAD views agree with 12 to 0.8 mm / 1.3 deg, 4 neighbouring ones do not."""
    import itertools
    cams = np.array([(T @ T_tool0_cam)[:3, 3] for _, T in plan])
    n = min(n, len(plan))
    if n <= 1:
        return [plan[int(np.argmax(np.linalg.norm(cams - np.asarray(cam0, dtype=float), axis=1)))]][:n]
    D = np.linalg.norm(cams[:, None, :] - cams[None], axis=2)
    # every set (a ring of 16 has 560 of 3): the one whose CLOSEST pair is farthest apart -- greedy
    # picking takes two opposite views first and crowds the third
    chosen = max(itertools.combinations(range(len(plan)), n),
                 key=lambda c: (min(D[a, b] for a, b in itertools.combinations(c, 2)),
                                sum(D[a, b] for a, b in itertools.combinations(c, 2))))
    return [plan[k] for k in sorted(chosen)]


def low_views(T0, T_tool0_cam, pivot, specs):
    """tool0 targets LOOKING UP at `pivot` (the user, 2026-11-06: "the socket estimate is biased pitched down,
    add a view of the socket from a lower angle that looks upwards towards the socket"). Each spec
    {elevation_deg, azimuth_deg}: the camera swung rigidly about the pivot -- first `azimuth_deg` about the
    vertical through it, then in that vertical plane -- until its line of sight to the pivot rises
    `elevation_deg` above horizontal. The distance is kept, and the pivot stays where it was in the image (a
    rigid turn about it), so a view framed past a held part stays framed."""
    cam = np.asarray(T0, dtype=float) @ np.asarray(T_tool0_cam, dtype=float)
    p = np.asarray(pivot, dtype=float)[:3]
    a = p - cam[:3, 3]
    a /= np.linalg.norm(a)
    P = translation_matrix(p)
    out = []
    for spec in specs:
        az = np.radians(float(spec.get('azimuth_deg', 0.0) or 0.0))
        el = np.radians(float(spec.get('elevation_deg', 15.0)))
        Rz = Rotation.from_rotvec([0.0, 0.0, az])
        a1 = Rz.apply(a)
        hz = np.array([a1[0], a1[1], 0.0])
        if np.linalg.norm(hz) < 1e-6:
            continue                                   # looking straight down: no vertical plane to swing in
        hz /= np.linalg.norm(hz)
        a2 = np.cos(el) * hz + np.array([0.0, 0.0, np.sin(el)])
        axis = np.cross(a1, a2)
        ang = float(np.arctan2(np.linalg.norm(axis), np.dot(a1, a2)))
        R = Rz if ang < 1e-9 else Rotation.from_rotvec(axis / np.linalg.norm(axis) * ang) * Rz
        M = np.eye(4)
        M[:3, :3] = R.as_matrix()
        T_cam = P @ M @ inverse(P) @ cam
        out.append((f'low view: looking up {np.degrees(el):+.0f} deg, azimuth {np.degrees(az):+.0f} deg',
                    T_cam @ inverse(np.asarray(T_tool0_cam, dtype=float))))
    return out


def orbit_views(robot, camera, pivot, oc, confirm=None, label='orbit', on_view=None):
    """[(joints, Frame)] from an orbit of the camera about `pivot` (base frame), the arm back where
    it started; [] when it cannot run (too few views reachable, declined). `on_view(view)` is called
    with each view as it is taken (estimate_part sends it to the estimator then, while the arm moves on)."""
    from ..apps.splat_scan import check_plan
    from ..apps.splat_scan import orbit_views as plan_orbit
    arm = robot.arm
    q0 = np.asarray(arm.q(), dtype=float)
    T0 = arm.fk(q0)
    cam0 = (T0 @ robot.T_tool0_cam)[:3, 3]
    plan = plan_orbit(T0, inverse(T0) @ np.append(pivot, 1.0), inverse(T0) @ np.append(cam0, 1.0), oc['tilts'], oc['pans'])
    quiet = urlog.get('arm')
    level = quiet.level
    quiet.setLevel(50)
    try:
        plan = check_plan(arm, q0, T0, plan, robot.T_tool0_fingertip, oc['bench_z'], oc['clearance_m'])
    finally:
        quiet.setLevel(level)
    if not plan:
        log.warning('%s: too few views reachable from here -- this view only.', label)
        return []
    if oc.get('max_views') is not None and len(plan) > oc['max_views']:
        plan = _spread(plan, robot.T_tool0_cam, cam0, oc['max_views'])
        log.info('%s: the %d most spread of the reachable views (orbit max_views): %s.', label, len(plan),
                 ', '.join(n for n, _ in plan))
    # THE LOW VIEWS, after the ring and always kept (not thinned by max_views): the whole walk checked again
    # with them on the end; one that cannot be reached is left out, said why.
    if oc.get('low'):
        low = low_views(T0, robot.T_tool0_cam, pivot, oc['low'])
        quiet.setLevel(50)
        try:
            full = check_plan(arm, q0, T0, list(plan) + low, robot.T_tool0_fingertip, oc['bench_z'], oc['clearance_m'])
        finally:
            quiet.setLevel(level)
        kept = [n for n, _ in (full or []) if n.startswith('low view')]
        if full is not None and kept:
            plan = full
            log.info('%s: + %s.', label, ', '.join(kept))
        else:
            log.warning('%s: no low view reachable from here (%s) -- the ring views only.', label,
                        ', '.join(n for n, _ in low) or 'none planned')
    if confirm is not None and not confirm(f'{label}: orbit for {len(plan)} more views'):
        log.info('%s: orbit declined -- this view only.', label)
        return []
    views = []
    arm.zero_ft()
    arm.add_guard(lambda: arm.force() > oc['guard_n'])
    try:
        for k, (name, T) in enumerate(plan, start=1):
            if not arm.move_l(T, label=f'{label} {name}', caps=oc['caps']):
                log.error('%s: the move to view %d stopped short; using what was seen.', label, k)
                break
            got = still_view(camera, arm, oc['frames'], oc['settle_s'])     # waits only until still
            if got is None:
                log.warning('  %s view %d: the arm moved through the exposure; skipped.', label, k)
                continue
            views.append(got)
            if on_view is not None:
                on_view(got)
        if not arm.move_l(T0, label=f'{label}: back', caps=oc['caps']):
            log.error('%s: the move back stopped short.', label)
    finally:
        arm.clear_guards()
    return views


def estimate_part(cfg, robot, camera, part, confirm=None, label=None, prior=None, orbit=None, hints=None):
    """(the service's reply, [(joints, Frame)] used) for `part` (perception/pose6d.py PLUG, SOCKET,
    RJ45 or RJ45_SOCKET) from where the arm is plus an orbit about it; (None, views) if not found.

    `prior` (T_base_obj, e.g. the socket from a recorded mate): the first view refines from it
    (its projection boxes the part) before searching the whole image; found in neither, the orbit
    still runs about it and the views together decide. Every view and reply is kept (save_views).
    `orbit`: keys over pose6d.orbit for this part (_orbit_cfg)."""
    label = label or f'6D pose: the {part}'
    oc = _orbit_cfg(cfg, orbit)
    svc = P6.service(cfg)
    replies = []
    first = still_view(camera, robot.arm, oc['frames'], oc['settle_s'])
    if first is None:
        log.error('%s: the arm was moving.', label)
        return None, []
    views = [first]
    try:
        r0 = {'found': False}
        if prior is not None:
            r0 = svc.estimate(part, [first[1]], prior=prior)
            replies.append(('first view, from the prior', r0))
        if prior is not None and r0.get('found'):
            log.info('%s: found in the first view from the recorded pose, score %.2f (%s); %.1f mm from it.', label,
                     r0['score'], _terms(r0), 1000 * np.linalg.norm(r0['T_base_obj'][:3, 3] - prior[:3, 3]))
        elif prior is not None:
            log.warning('%s: not found from the recorded pose in the first view (%s) -- searching the image.',
                        label, r0.get('why') or _terms(r0))
        if not r0.get('found'):
            r0 = svc.estimate(part, [first[1]])
            replies.append(('first view, searched', r0))
        if r0.get('found') and part == P6.plug_part(cfg):
            r0 = _lying_flat(cfg, svc, part, first[1], r0, replies, label)
        if r0.get('found'):
            T0 = r0['T_base_obj']
            log.info('%s: found in the first view, score %.2f (%s).', label, r0['score'], _terms(r0))
        elif prior is not None and oc['enabled']:
            T0 = np.asarray(prior, dtype=float)
            log.warning('%s: not found in the first view (%s) -- orbiting about the recorded pose anyway; the '
                        'views together decide.', label, r0.get('why') or _terms(r0))
        else:
            log.error('%s: not found in the first view (%s).', label, r0.get('why') or _terms(r0))
            # A CANDIDATE THAT FITTED BUT MISSED THE GATE, for the caller to look at closer (`hints`): on
            # 2026-11-04 13:53 the plug lay against the finger, its mask took the cable along, and the
            # fit -- in the right place -- scored IoU 0.69.
            if hints is not None:
                best = max((v for v in r0.get('views', []) if v.get('T_base_obj') is not None),
                           key=lambda v: v.get('score', 0.0), default=None)
                if best is not None:
                    hints.append((np.asarray(best['T_base_obj'], dtype=float), best))
            return None, views
        if not oc['enabled']:
            _passed(label, r0, svc, 'the first view')
            return r0, views
        pivot = T0[:3, :3] @ P6.CENTRE_M[part] + T0[:3, 3]
        # EACH VIEW TO THE ESTIMATOR AS IT IS TAKEN (the user, 2026-11-04): refined on the server while
        # the arm drives to the next, the joint pose asked for once the last is in -- the same estimate
        # as all of them sent at the end (pose6d_server session ops), without the wait. The session
        # opens with the first orbit view, so an orbit that takes none costs nothing.
        ses = []

        def on_view(got):
            if not ses:
                ses.append(svc.session(part, prior=T0))
                ses[0].add(first[1])
            ses[0].add(got[1])
        t_orbit = time.monotonic()
        views = [first] + orbit_views(robot, camera, pivot, oc, confirm, label, on_view=on_view)
        if len(views) == 1:
            if r0.get('found'):
                _passed(label, r0, svc, 'the first view (no orbit view reached)')
            return (r0 if r0.get('found') else None), views
        t_wait = time.monotonic()
        r = ses[0].finish()
        log.info('%s: the orbit took %.1f s; the estimate was ready %.1f s after the arm was back.', label,
                 t_wait - t_orbit, time.monotonic() - t_wait)
        replies.append(('all views', r))
        if not r.get('found'):
            log.error('%s: the %d views together fail the gate (%s) -- not trusted.', label, len(views), _terms(r))
            return None, views
        gap = np.linalg.norm(r['T_base_obj'][:3, 3] - T0[:3, 3])
        log.info('%s: %d of %d views together, score %.2f (%s); %.1f mm from the first view\'s estimate.',
                 label, r['n_used'], len(views), r['score'], _terms(r), 1000 * gap)
        _passed(label, r, svc, f"{r['n_used']} of {len(views)} views fused")
        return r, views
    finally:
        save_views(cfg, label, views, replies, prior)


# ------------------------------------------------- looking past the part in the hand
# THE HELD PLUG IN THE SOCKET VIEW (the user, 2026-11-04): with the connector in the hand, the
# socket views put the socket mid-image and the plug -- rigid on the camera, ~0.2 m in front of
# it -- stood up from the bottom edge into it (data/pose6d/20261104_071018_*: its tip at v ~400 px,
# the socket's lower edge at ~410). The camera is pitched about its own centre until the socket's
# CAD, projected from its last estimate, sits `margin_px` above the plug's; then looked at again and
# re-aimed from what that view measures (servoing on the estimate) until the correction is small.
# An orbit about the socket from there keeps it at that place in the image.

def _px(K, T_cam_x, points):
    """(u, v) of `points` (N, 3, in frame x) seen from a camera at T_cam_x; those in front only."""
    p = points @ T_cam_x[:3, :3].T + T_cam_x[:3, 3]
    p = p[p[:, 2] > 0.02]
    uv = p @ np.asarray(K, dtype=float).T
    return uv[:, :2] / uv[:, 2:3]


def _rot_x(a):
    T = np.eye(4)
    c, s = np.cos(a), np.sin(a)
    T[1:3, 1:3] = [[c, -s], [s, c]]
    return T


def held_points(held):
    """The held part's CAD vertices in tool0 (robot/held_part.HeldPart)."""
    from ..robot import tool_geometry
    p = tool_geometry.read_ply(held.mesh_path)[0] * held.scale
    return p @ held.T_tool0_obj[:3, :3].T + held.T_tool0_obj[:3, 3]


def held_clearance_px(K, T_base_cam, socket_pts, held_pts_cam, margin_px):
    """(clearance, top) in px for the camera at T_base_cam: how far the socket's CAD points (base) sit
    above the held part's (camera frame) in the columns they share (inf if none), and how far below
    the image's top edge the socket's highest point is."""
    s = _px(K, inverse(T_base_cam), socket_pts)
    if len(s) == 0:
        return -np.inf, -np.inf
    h = held_pts_cam
    near = h[(h[:, 0] > s[:, 0].min() - margin_px) & (h[:, 0] < s[:, 0].max() + margin_px)]
    clear = np.inf if len(near) == 0 else float(near[:, 1].min() - s[:, 1].max())
    return clear, float(s[:, 1].min())


def aim_cfg(cfg, aim):
    """assembly.visual_target.aim_past_held, with the orbit's speed, frames and bench clearance."""
    a = dict(aim or {})
    oc = _orbit_cfg(cfg)
    return {'enabled': bool(a.get('enabled', False)), 'margin_px': float(a.get('margin_px', 40.0)),
            'top_margin_px': float(a.get('top_margin_px', 30.0)), 'max_pitch_deg': float(a.get('max_pitch_deg', 25.0)),
            'servo_iters': int(a.get('servo_iters', 3)), 'tol_deg': float(a.get('tol_deg', 1.0)),
            **{k: oc[k] for k in ('caps', 'frames', 'settle_s', 'bench_z', 'clearance_m')}}


def aim_past_held(robot, K,T_base_socket, socket_pts, ac):
    """(pitch rad, T_base_tool0) -- the least pitch of the camera about its own centre that puts the
    socket `margin_px` clear above the held part and `top_margin_px` below the image's top edge,
    within +-max_pitch_deg and reachable on a straight line (splat_scan.check_plan); (0, None)
    when no pitch is needed, (None, None) when none does it."""
    from ..apps.splat_scan import check_plan
    arm = robot.arm
    q0 = np.asarray(arm.q(), dtype=float)
    T0 = arm.fk(q0)
    T_cam0 = T0 @ robot.T_tool0_cam
    held_px = _px(K, inverse(robot.T_tool0_cam), held_points(robot.held))
    S = (np.asarray(T_base_socket)[:3, :3] @ socket_pts.T).T + np.asarray(T_base_socket)[:3, 3]
    m, top = float(ac['margin_px']), float(ac['top_margin_px'])
    ok = lambda a: (lambda c: c[0] >= m and c[1] >= top)(held_clearance_px(K, T_cam0 @ _rot_x(a), S, held_px, m))
    if ok(0.0):
        return 0.0, None
    step = np.radians(0.25)
    lim = np.radians(float(ac['max_pitch_deg']))
    cands = sorted([a for a in np.arange(-lim, lim + 1e-9, step) if ok(a)], key=abs)
    for a in cands:
        T = T_cam0 @ _rot_x(a) @ inverse(robot.T_tool0_cam)
        if check_plan(arm, q0, T0, [('aim past the held part', T)], robot.T_tool0_fingertip,
                      ac['bench_z'], ac['clearance_m']):
            return float(a), T
    return None, None


def view_past_held(cfg, robot, camera, part, prior, ac, label):
    """The arm turned, from where it is, so the held part does not hide `part` (see above): aimed
    from `prior` (its last estimate, base), then re-aimed from each new view's estimate -- at most
    `servo_iters` moves, until the correction is under `tol_deg`. The latest estimate of `part`
    (T_base_obj, `prior` if none was had); the arm stays at the last aim."""
    from ..robot import tool_geometry
    mesh = P6.cad_mesh(cfg, part, name=P6.OBJECTS[part])
    if robot.held is None or mesh is None or camera is None:
        return prior
    socket_pts = tool_geometry.read_ply(mesh[0])[0] * mesh[1]
    svc = P6.service(cfg)
    K = np.asarray(camera.K, dtype=float)
    T_obj = np.asarray(prior, dtype=float)
    quiet = urlog.get('arm')
    for it in range(1, int(ac['servo_iters']) + 1):
        level = quiet.level
        quiet.setLevel(50)
        try:
            a, T = aim_past_held(robot, K, T_obj, socket_pts, ac)
        finally:
            quiet.setLevel(level)
        if a is None:
            log.warning('%s: no reachable pitch within %.0f deg clears the %s in the hand -- viewing from here.',
                        label, ac['max_pitch_deg'], robot.held.name)
            return T_obj
        if T is None or abs(a) < np.radians(ac['tol_deg']):
            log.info('%s: the %s is %s the %s in the hand%s.', label, part, 'clear of' if T is None else
                     'nearly clear of', robot.held.name, '' if it == 1 else f' after {it - 1} re-aim(s)')
            return T_obj
        log.info('%s: pitching the camera %.1f deg so the %s in the hand does not hide the %s (aim %d, from %s).',
                 label, np.degrees(a), robot.held.name, part, it, 'its last estimate' if it == 1 else 'this view')
        if not robot.arm.move_l(T, label=f'{label}: aim past the held part', caps=ac['caps']):
            log.error('%s: the aim move stopped short.', label)
            return T_obj
        got = still_view(camera, robot.arm, ac['frames'], ac['settle_s'])
        if got is None:
            return T_obj
        r = svc.estimate(part, [got[1]], prior=T_obj)
        if not r.get('found'):
            log.warning('%s: not found after the aim (%s) -- keeping the aim from the last estimate.', label,
                        r.get('why') or _terms(r))
            return T_obj
        log.info('%s: after aim %d, %.1f mm from the estimate it was aimed from (%s).', label, it,
                 1000.0 * np.linalg.norm(r['T_base_obj'][:3, 3] - T_obj[:3, 3]), _terms(r))
        T_obj = np.asarray(r['T_base_obj'], dtype=float)
    return T_obj


# ------------------------------------------------- a closer look at the part, servoed
# THE CONNECTOR CLOSER, BEFORE THE PICK (the user, 2026-11-04): the scan sees the plug from ~0.43 m,
# near the top of the image (v ~124 px on 2026-11-04 08:11) -- a few hundred pixels of it. The camera
# is brought `distance_mm` from the plug's middle along its present line of sight to it, the plug
# centred, and looks; the estimate from that view re-aims it (servoing on the estimate), until a view
# agrees with the estimate it was aimed from to tol_mm / tol_deg, or servo_iters looks are spent.
# Depth is used: 0.30 m was still inside what the D415 read (the D405 reads from ~7 cm).

def close_cfg(cfg):
    """pose6d.close_view, with the orbit's speed, frames and bench clearance."""
    c = dict(cfg.section('pose6d') or {}) if cfg.get('pose6d') else {}
    a = dict(c.get('close_view') or {})
    oc = _orbit_cfg(cfg)
    return {'enabled': bool(a.get('enabled', False)), 'distance_m': float(a.get('distance_mm', 300.0)) / 1000.0,
            'max_distance_m': float(a.get('max_distance_mm', 380.0)) / 1000.0,
            'servo_iters': int(a.get('servo_iters', 3)), 'tol_m': float(a.get('tol_mm', 3.0)) / 1000.0,
            'tol_rad': np.radians(float(a.get('tol_deg', 3.0))),
            'max_shift_m': float(a.get('max_shift_mm', 15.0)) / 1000.0,
            'max_shift_rad': np.radians(float(a.get('max_shift_deg', 20.0))),
            **{k: oc[k] for k in ('caps', 'frames', 'settle_s', 'bench_z', 'clearance_m')}}


def close_view_pose(robot, centre, dist):
    """tool0 for the camera `dist` from `centre` (base), on its present line of sight to it, aimed at it
    (look_at: its roll kept as near as the new direction allows)."""
    from ..transforms import look_at
    T_cam = robot.arm.fk(robot.arm.q()) @ robot.T_tool0_cam
    ray = np.asarray(centre, dtype=float) - T_cam[:3, 3]
    ray /= np.linalg.norm(ray)
    return look_at(np.asarray(centre, dtype=float) - dist * ray, centre, T_cam) @ inverse(robot.T_tool0_cam)


def servo_close_view(cfg, robot, camera, part, prior, sc, label, max_shift=None):
    """(reply, [(joints, Frame)]): `part` looked at from close up (above), servoing on its estimate,
    starting from `prior` (T_base_obj). The reply is the last view's that passed the gate and stayed
    within max_shift of `prior`; None (said why) when none did -- the caller keeps its own. The arm
    stays at the last view."""
    from ..apps.splat_scan import check_plan
    arm = robot.arm
    svc = P6.service(cfg)
    T_obj, best, views, replies = np.asarray(prior, dtype=float), None, [], []
    try:
        for it in range(1, sc['servo_iters'] + 1):
            centre = T_obj[:3, :3] @ P6.CENTRE_M[part] + T_obj[:3, 3]
            T = None
            quiet = urlog.get('arm')
            level = quiet.level
            quiet.setLevel(50)
            try:
                q0 = np.asarray(arm.q(), dtype=float)
                for d in np.arange(sc['distance_m'], sc['max_distance_m'] + 1e-9, 0.02):
                    cand = close_view_pose(robot, centre, d)
                    if check_plan(arm, q0, arm.fk(q0), [('close view', cand)], robot.T_tool0_fingertip,
                                  sc['bench_z'], sc['clearance_m']):
                        T = cand
                        break
            finally:
                quiet.setLevel(level)
            if T is None:
                log.warning('%s: no reachable close view within %.0f mm of the %s -- %s.', label,
                            1000 * sc['max_distance_m'], part, 'keeping the last' if best else 'the scan stands')
                break
            cam_d = np.linalg.norm((T @ robot.T_tool0_cam)[:3, 3] - centre)
            log.info('%s: close view %d, the camera %.0f mm from the %s, aimed at %s.', label, it, 1000 * cam_d,
                     part, 'the scan\'s estimate' if it == 1 else 'the last view\'s')
            if not arm.move_l(T, label=f'{label}: close view {it}', caps=sc['caps']):
                log.error('%s: the move to the close view stopped short.', label)
                break
            got = still_view(camera, arm, sc['frames'], sc['settle_s'])
            if got is None:
                break
            views.append(got)
            r = svc.estimate(part, [got[1]], prior=T_obj)
            replies.append((f'close view {it}', r))
            if not r.get('found'):
                log.warning('%s: not found in close view %d (%s).', label, it, r.get('why') or _terms(r))
                break
            d_lin, d_ang = pose_error(np.asarray(prior, dtype=float), r['T_base_obj'])
            if P6.SYMMETRY[part] == 'axis_z':        # round: its turn about the axis is a convention
                d_ang = float(np.arccos(np.clip(np.asarray(prior)[:3, 2] @ r['T_base_obj'][:3, 2], -1.0, 1.0)))
            lim_m, lim_rad = max_shift if max_shift is not None else (sc['max_shift_m'], sc['max_shift_rad'])
            if d_lin > lim_m or d_ang > lim_rad:
                log.warning('%s: close view %d puts it %.1f mm / %.1f deg from the scan -- over pose6d.close_view.'
                            'max_shift_mm/_deg, not trusted.', label, it, 1000 * d_lin, np.degrees(d_ang))
                break
            log.info('%s: close view %d, score %.2f (%s); %.1f mm / %.1f deg from the scan.', label, it,
                     r['score'], _terms(r), 1000 * d_lin, np.degrees(d_ang))
            best = r
            # CONVERGED when the view agrees with the estimate it was aimed from: aiming again would
            # look from (nearly) the same place
            c_lin, c_ang = pose_error(T_obj, r['T_base_obj'])
            if P6.SYMMETRY[part] == 'axis_z':
                c_ang = float(np.arccos(np.clip(T_obj[:3, 2] @ r['T_base_obj'][:3, 2], -1.0, 1.0)))
            T_obj = np.asarray(r['T_base_obj'], dtype=float)
            if c_lin < sc['tol_m'] and c_ang < sc['tol_rad']:
                log.info('%s: close view %d agrees with the estimate it was aimed from (%.1f mm / %.1f deg) -- '
                         'converged.', label, it, 1000 * c_lin, np.degrees(c_ang))
                break
        if best is not None:
            _passed(label, best, svc, 'the close view')
        return best, views
    finally:
        save_views(cfg, label, views, replies, prior)


def plug_tilt_deg(T_base_obj):
    """How far the plug's axis (its model z) stands off the base plane, deg."""
    return float(np.degrees(np.arcsin(min(1.0, abs(float(np.asarray(T_base_obj, dtype=float)[2, 2]))))))


def _lying_flat(cfg, svc, part, frame, r, replies, label, headings=12):
    """A plug on the table lies on it: an estimate standing more than pose6d.max_lying_tilt_deg off the
    base plane (null: no check) is re-fitted from FLAT starts -- its centre kept, the axis level at
    `headings` headings round -- and the best of those that pass the gate wins; none passing, the
    estimate is not trusted. 2026-11-03: the plug pointing almost straight away from the wrist camera
    showed only its round head, and the search stood it up on end, nose up."""
    c = dict(cfg.section('pose6d') or {}) if cfg.get('pose6d') else {}
    limit = c.get('max_lying_tilt_deg', 25.0)
    tilt = plug_tilt_deg(r['T_base_obj'])
    if limit is None or tilt <= float(limit):
        return r
    T = np.asarray(r['T_base_obj'], dtype=float)
    centre = T[:3, :3] @ P6.CENTRE_M[part] + T[:3, 3]
    log.warning('%s: the estimate stands %.0f deg off the table (pose6d.max_lying_tilt_deg %.0f) -- a plug lying '
                'there cannot; re-fitting from %d level headings.', label, tilt, float(limit), headings)
    best = None
    for a in np.linspace(0.0, 2.0 * np.pi, int(headings), endpoint=False):
        z = np.array([np.cos(a), np.sin(a), 0.0])
        x = np.array([0.0, 0.0, 1.0])
        R = np.column_stack([x, np.cross(z, x), z])
        prior = np.eye(4)
        prior[:3, :3] = R
        prior[:3, 3] = centre - R @ P6.CENTRE_M[part]
        rr = svc.estimate(part, [frame], prior=prior)
        if rr.get('found') and plug_tilt_deg(rr['T_base_obj']) <= float(limit) and (
                best is None or rr['score'] > best['score']):
            best = rr
    replies.append(('first view, level re-fit', best or {'found': False}))
    if best is None:
        log.error('%s: no level fit passes the gate -- not trusted.', label)
        return {'found': False, 'why': f'stood {tilt:.0f} deg off the table; no level fit passed the gate'}
    log.info('%s: level re-fit, score %.2f (%s), %.0f deg off the table.', label, best['score'], _terms(best),
             plug_tilt_deg(best['T_base_obj']))
    return best


def save_views(cfg, label, views, replies, prior=None):
    """One estimate's views (as the splat scan writes them: rgb, depth_m, K, T_base_cam, joints_rad)
    and the service's replies, under <data_dir>/pose6d/<time>_<label>/ -- to look at a failure, and
    re-run the estimator on it, without the robot. Best-effort: it never fails the run."""
    import json
    import os
    import re
    try:
        d = os.path.join(cfg.get('data_dir', 'data'), 'pose6d',
                         time.strftime('%Y%m%d_%H%M%S_') + re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_'))
        os.makedirs(d, exist_ok=True)
        for k, (q, f) in enumerate(views):
            np.savez_compressed(os.path.join(d, f'view_{k:02d}.npz'), rgb=np.ascontiguousarray(f.color[..., ::-1]),
                                depth_m=np.asarray(f.depth, dtype=np.float32), K=np.asarray(f.K, float),
                                T_base_cam=np.asarray(f.T_base_cam, float), joints_rad=np.asarray(q, float))

        def plain(x):
            return x.tolist() if isinstance(x, np.ndarray) else x

        out = {'label': label, 'prior': plain(prior),
               'replies': [{'what': w, **{k: plain(v) for k, v in r.items() if k != 'views'},
                            'views': [{k: plain(v) for k, v in vv.items()} for vv in r.get('views', [])]}
                           for w, r in replies]}
        with open(os.path.join(d, 'replies.json'), 'w') as fh:
            json.dump(out, fh, indent=1, default=str)
        log.info('%s: views and replies kept in %s', label, d)
    except Exception as exc:                                       # noqa: BLE001 -- a record, never the run
        log.warning('%s: the views were not kept (%s).', label, exc)


def _terms(r):
    t = r.get('terms') or {}
    return ', '.join(f'{k} {v:.2f}' for k, v in t.items()) or 'no terms'


def _passed(label, r, svc, how):
    """The one PASSED line (as apps/bnc_assembly.py prints for each phase): the estimate's terms
    against the gate the server tested them on (perception/pose6d_server.py defaults)."""
    q = getattr(svc, 'request', {}) or {}
    t = r.get('terms') or {}
    gate = [('iou', 'min_iou', 0.75), ('edge_fwd', 'min_edge', 0.5), ('edge_cover', 'min_cover', 0.7)]
    log.info('PASSED %s: %s -- %s', label, how, ', '.join(
        f'{k} {t[k]:.2f} >= {float(q.get(m, d)):.2f}' for k, m, d in gate if k in t) or _terms(r))



def _free_sam3(det):
    """SAM3's GPU memory back (the model dropped, the cache emptied) -- also after a load that failed
    part way, which left 2.3 GB held in this process on 2026-11-04."""
    import gc
    if det is not None:
        for attr in ('core', 'detector'):
            if hasattr(det, attr):
                setattr(det, attr, None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                      # noqa: BLE001
        pass



class Pose6DArmError(Exception):
    """The arm could not return from a look -- the scan stops where it is."""

class Pose6DScanner:
    """The pick's scanner (scan()/reset()/reselect(), .camera/.estimator/.listeners) for
    pose_method pose6d: the plug's junction frame from estimate_part."""

    def __init__(self, robot, camera, estimator, cfg, data_root='data'):
        self.robot, self.camera, self.estimator, self.cfg = robot, camera, estimator, cfg
        self.detector = None                    # no SAM3 in this mode
        self.listeners, self.tracker = [], None
        self.part = P6.plug_part(cfg)
        self.in_hand = None                     # a ConnectorTracker, started from each scan's pose
        self.pose_on_axis = True
        self.scene = None
        self.last = None
        self._selection = None
        self.data_root = data_root
        self.used_fallback = False
        # THE PLUG'S TURN IN THE GRASP (P6.grasp_junction): scan() returns the junction frame z up, the one
        # every grasp is written in; the plug as the jaws will hold it is that @ self.turn. The identity for
        # a round plug; an RJ45's latch side, squared to pose6d.grasp_square_deg.
        self.turn = np.eye(4)

    def _grasp_frame(self, T_junction, label):
        """(the levelled junction frame the pick grasps, the turn kept in self.turn) -- see P6.grasp_junction."""
        c = dict(self.cfg.section('pose6d') or {}) if self.cfg.get('pose6d') else {}
        sq = c.get('grasp_square_deg', 90.0)
        T, self.turn = P6.grasp_junction(T_junction, self.part, None if sq is None else float(sq))
        if P6.SYMMETRY[self.part] == 'none':
            R = T[:3, :3].T @ np.asarray(T_junction, dtype=float)[:3, :3]
            log.info('%s: its latch side %.0f deg round from up (about its axis); the jaws will hold it at %+.0f '
                     '(pose6d.grasp_square_deg %s) -- the grasp is written in the levelled frame, the belief '
                     'carries the turn.', label, np.degrees(np.arctan2(R[2, 1], R[1, 1])), P6.turn_deg(self.turn),
                     sq)
        return T

    def reset(self):
        pass

    def reselect(self):
        self._selection = None

    def fallback_scan(self, confirm=None):
        """THE 6D ESTIMATOR DID NOT FIND THE CONNECTOR (the user, 2026-11-04): SAM3, its cable centreline
        traced to the junction, the junction's pose on the bench plane fitted from the depth
        (skills/ground_pick.py DepthPlaneScanner) -- pose6d.fallback. T_base_junction, or None.

        GPU MEMORY (the 13:48 run: "CUDA out of memory" loading SAM3 beside the 6D server's 4 GB and a
        2.2 GB training run, on a 9.6 GB card): with fallback.free_6d (default) the 6D server is stopped
        first, SAM3 loaded, the scan made, SAM3 unloaded, and the 6D server started again in the
        background for what comes next (the socket). SAM3 is not kept between scans."""
        c = dict(self.cfg.section('pose6d') or {}) if self.cfg.get('pose6d') else {}
        fb = dict(c.get('fallback') or {})
        if not fb.get('enabled', False):
            return None
        from ..perception.sam3 import make_detector
        from .ground_pick import DepthPlaneScanner
        free = bool(fb.get('free_6d', True))
        log.warning('6D pose: FALLING BACK to SAM3 + the centreline trace + the junction on the depth-fitted '
                    'bench plane (pose6d.fallback)%s. Loading SAM3...',
                    ' -- the 6D server stopped first, for its GPU memory' if free else '')
        if free:
            P6.shutdown()
        det, T = None, None
        try:
            try:
                det = make_detector(self.cfg)
            except Exception as exc:                       # noqa: BLE001 -- e.g. CUDA out of memory
                log.error('6D pose: the fallback could not load SAM3 (%s: %s) -- no pose.', type(exc).__name__,
                          str(exc).split('.')[0])
                return None
            scanner = DepthPlaneScanner(self.robot, self.camera, det, self.estimator, self.cfg,
                                        data_root=self.data_root, plane_cfg=fb.get('plane'))
            T = scanner.scan(confirm)
        finally:
            _free_sam3(det)
            if free:
                P6.preload(self.cfg)                    # back in the background for the socket
        if T is None:
            log.error('6D pose: the SAM3 fallback found no connector either.')
            return None
        self.used_fallback, self.last = True, None       # no 6D reply: nothing downstream refines from one
        self.turn = np.eye(4)                              # SAM3 sees the axis, not a latch
        if P6.SYMMETRY[self.part] == 'none':
            log.warning('6D pose: the SAM3 fallback cannot tell which way round the %s lies -- its turn about the '
                        'axis is taken as z up (the latch up), which may be the wrong way round.', self.part)
        self._selection = T[:3, 3].copy()
        log.info('PASSED 6D pose: the connector (SAM3 fallback): junction at %s mm, heading %.1f deg.',
                 (T[:3, 3] * 1000).round(1), np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        return T

    def _rest_on_plane(self, r):
        """`pose6d.rest_on_plane_mm` (base frame; null = off): the estimate moved straight up or down
        so the plug's lowest CAD point is on that plane -- its heading and tilt as estimated. For a
        plug known to lie on a known surface (0: the plane the robot stands on), when the camera
        chain's height is in question (the 2026-11-02 estimates put the plug's underside 4.3-5.2 mm
        above the robot's plane). A copy of the reply; the estimate as made is kept as T_base_obj_raw."""
        c = dict(self.cfg.section('pose6d') or {}) if self.cfg.get('pose6d') else {}
        plane_mm = c.get('rest_on_plane_mm')
        if plane_mm is None:
            return r
        from ..robot import tool_geometry
        mesh = P6.cad_mesh(self.cfg, self.part)
        if mesh is None:
            log.warning('pose6d.rest_on_plane_mm: no CAD for %s -- the estimate as made.', self.part)
            return r
        verts = np.asarray(tool_geometry.read_ply(mesh[0])[0], dtype=float) * mesh[1]
        T = np.array(r['T_base_obj'], dtype=float)
        bottom = float(np.min(verts @ T[2, :3] + T[2, 3]))
        dz = float(plane_mm) / 1000.0 - bottom
        T[2, 3] += dz
        log.info('6D pose: rested on the plane at z %.1f mm (pose6d.rest_on_plane_mm): the plug\'s underside was '
                 'estimated at %.1f mm, moved %+.1f mm.', float(plane_mm), 1000.0 * bottom, 1000.0 * dz)
        return {**r, 'T_base_obj': T, 'T_base_obj_raw': np.array(r['T_base_obj'], dtype=float), 'rested_dz': dz}

    def _plausible(self, r, label):
        """`r` if the plug could be there, else None: its junction within pose6d.max_off_bench_mm (40) of the
        bench (ground_plane.z_mm) -- the plug lies on it. On 2026-11-06 01:02 an estimate put the junction at
        z -451 mm, 457 mm UNDER the bench: the coaxial grasp had no IK (read as the cable's heading), the square
        pick was refused against the bench, and the run asked for a person over a cable it had never touched."""
        if r is None:
            return None
        c = dict(self.cfg.section('pose6d') or {}) if self.cfg.get('pose6d') else {}
        lim = c.get('max_off_bench_mm', 40.0)
        if lim is None:
            return r
        bench = float(self.cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0
        z = float(P6.junction_from_plug(r['T_base_obj'], part=self.part)[2, 3])
        if abs(z - bench) * 1000.0 <= float(lim):
            return r
        log.warning('%s: REJECTED -- it puts the junction at z %.0f mm, %.0f mm %s the bench (z %.1f mm; '
                    'pose6d.max_off_bench_mm %.0f). Not the plug lying on the bench.', label, 1000.0 * z,
                    abs(1000.0 * (z - bench)), 'under' if z < bench else 'over', 1000.0 * bench, float(lim))
        return None

    def _attempt(self, confirm, label, sc):
        """One 6D estimate of the connector from where the camera is: (reply or None, views, looked_close).
        Not found, but a candidate fitted and missed the gate: that candidate looked at from close up
        (centred, clear of the gripper -- the 13:53 plug lay against the finger), the close view's own
        estimate passing the gate; the arm back where the attempt started."""
        hints = []
        r, views = estimate_part(self.cfg, self.robot, self.camera, self.part, confirm, label, hints=hints)
        if r is not None or not hints or not sc['enabled']:
            return r, list(views), False
        T_hint, v = hints[0]
        log.warning('%s: the best candidate fitted but missed the gate (%s) -- looking at it from close up.',
                    label, _terms(v))
        T_here = self.robot.arm.fk(self.robot.arm.q())
        rc, close = servo_close_view(self.cfg, self.robot, self.camera, self.part, T_hint, sc,
                                     label + ' (close, from a near miss)', max_shift=(0.04, np.radians(35.0)))
        if close and not self.robot.arm.move_l(T_here, label='back from the close view', caps=sc['caps']):
            raise Pose6DArmError('could not go back from the close view')
        return rc, list(views) + list(close), rc is not None

    def scan(self, confirm=None):
        """The connector by the 6D estimator ONLY (the user, 2026-11-04): from the scan pose; not found
        there, from the camera moved by each of pose6d.retry_views in the base x-y plane (+/-50 mm in x, then
        y), one after another, until one finds it. Then the close view (pose6d.close_view), and the arm
        back at the scan pose -- the pick starts there. The SAM3 fallback only when pose6d.fallback is on."""
        self.scene = None
        if self.in_hand is not None:
            self.in_hand.stop()                 # the service is this scan's now
        sc = close_cfg(self.cfg)
        c = dict(self.cfg.section('pose6d') or {}) if self.cfg.get('pose6d') else {}
        offsets = [(0.0, 0.0)] + [(float(o.get('x_mm', 0.0)) / 1000.0, float(o.get('y_mm', 0.0)) / 1000.0)
                                  for o in (c.get('retry_views') or [])]
        arm = self.robot.arm
        T_home = arm.fk(arm.q())
        r, views, looked_close = None, [], False
        try:
            for k, (dx, dy) in enumerate(offsets):
                label = '6D pose: the connector' + ('' if k == 0 else ' (camera %+.0f, %+.0f mm)' % (1000 * dx, 1000 * dy))
                if k:
                    log.warning('6D pose: the connector not found -- moving the camera %+.0f mm in x, %+.0f mm in y '
                                '(base) and looking again (try %d of %d).', 1000 * dx, 1000 * dy, k, len(offsets) - 1)
                    T = translation_matrix([dx, dy, 0.0]) @ T_home
                    if not arm.move_l(T, label=f'{label}: the view', caps=sc['caps']):
                        log.warning('%s: that view is out of reach -- the next.', label)
                        continue
                r, v, looked_close = self._attempt(confirm, label, sc)
                views += v
                r = self._plausible(r, label)
                if r is not None:
                    if k:
                        log.info('%s: FOUND from the moved camera.', label)
                    break
            if r is not None and sc['enabled'] and not looked_close:
                # CLOSER, SERVOED on the estimate (the user, 2026-11-04): it replaces the scan's if trusted
                rc, close = servo_close_view(self.cfg, self.robot, self.camera, self.part, r['T_base_obj'], sc,
                                             '6D pose: the connector (close)')
                views += list(close)
                rc = self._plausible(rc, '6D pose: the connector (close)')
                if rc is not None:
                    r = rc
        except Pose6DArmError as exc:
            log.error('6D pose: %s.', exc)
            return None
        # BACK TO THE SCAN POSE, down a straight line: the pick's move to the pre-grasp starts from there.
        # From the close view (A5 115, A6 -33) that joint move swung A6 96 deg and dipped the fingertip to
        # 30 mm over the bench (2026-11-04 08:51 and 08:56); from the scan pose it keeps >= 108 mm.
        if pose_error(arm.fk(arm.q()), T_home)[0] > 1e-3 or pose_error(arm.fk(arm.q()), T_home)[1] > 1e-3:
            if not arm.move_l(T_home, label='back to the scan pose', caps=sc['caps']):
                log.error('6D pose: could not go back to the scan pose.')
                return None
        self.scene = self._scene(views)
        self.used_fallback = False
        if r is None:
            if c.get('fallback', {}).get('enabled', False):
                return self.fallback_scan(confirm)
            log.error('6D pose: the connector not found from the scan pose or any of the %d moved views.',
                      len(offsets) - 1)
            return None
        r = self._rest_on_plane(r)
        T_measured = P6.junction_from_plug(r['T_base_obj'], part=self.part)
        T = self._grasp_frame(T_measured, '6D pose: the connector')
        self.last = r
        self._selection = T[:3, 3].copy()
        log.info('6D pose: the connector\'s junction at %s mm, heading %.1f deg, %.1f deg off level.',
                 (T[:3, 3] * 1000).round(1), np.degrees(np.arctan2(T[1, 0], T[0, 0])),
                 np.degrees(np.arcsin(np.clip(T[2, 0], -1, 1))))
        self._draw(r, T_measured)
        if self.in_hand is not None:
            self.in_hand.start(T_base_plug=r['T_base_obj'])
        return T

    def refine_before_close(self, robot, geom):
        """ONE MORE LOOK, from the grasp pose with the fingers still open round the plug, just before
        they close (the user, 2026-11-03): the in-hand pose is the estimate carried rigidly from the
        close (pose6d.rigid_grasp), so the nearer the estimate is made to the close, the less the
        plug can have been mis-seen or nudged. Refined from the scan's estimate in one view, its
        depth ignored (the plug is ~0.18 m from the camera, nearer than the D415 reads): the
        silhouette's size sets the range. Taken only if it passes the gate and lies within
        pose6d.refine_before_close.max_shift_mm / _deg of the scan's (a jump that size is a misfit,
        not a correction) -- then it REPLACES geom.T_base_detection, which the in-hand belief is
        held from. The grasp itself does not move: the fingers are already round the plug.
        True when it replaced the estimate."""
        c = dict(self.cfg.section('pose6d') or {}) if self.cfg.get('pose6d') else {}
        rc = c.get('refine_before_close') or {}
        if not rc.get('enabled', False) or self.last is None or self.camera is None or robot.arm.dry_run:
            return False
        label = '6D pose: the connector before the close'
        svc = P6.service(self.cfg)
        oc = _orbit_cfg(self.cfg, None)
        view = still_view(self.camera, robot.arm, oc['frames'], oc['settle_s'])
        if view is None:
            log.warning('%s: the arm was moving -- the scan\'s estimate stays.', label)
            return False
        T_prior = np.asarray(self.last['T_base_obj'], dtype=float)
        r = {'found': False}
        try:
            r = svc.estimate(self.part, [view[1]], prior=T_prior,
                             use_depth=bool(rc.get('use_depth', False)))
        except P6.Pose6DError as exc:
            r = {'found': False, 'why': f'the estimator failed: {exc}'}
        save_views(self.cfg, label, [view], [('before the close', r)], T_prior)
        if not r.get('found'):
            log.warning('%s: not found (%s) -- the scan\'s estimate stays.', label, r.get('why') or _terms(r))
            return False
        r = self._rest_on_plane(r)
        from ..transforms import pose_error
        d_lin, d_ang = pose_error(T_prior, r['T_base_obj'])
        if P6.SYMMETRY.get(self.part) == 'axis_z':        # round: compare the axis, not the turn about it
            d_ang = float(np.arccos(np.clip(T_prior[:3, 2] @ np.asarray(r['T_base_obj'])[:3, 2], -1.0, 1.0)))
        lim_mm, lim_deg = float(rc.get('max_shift_mm', 10.0)), float(rc.get('max_shift_deg', 15.0))
        if 1000.0 * d_lin > lim_mm or np.degrees(d_ang) > lim_deg:
            log.warning('%s: %.1f mm / %.1f deg from the scan\'s estimate, over the %.0f mm / %.0f deg a correction '
                        'may be -- taken for a misfit; the scan\'s estimate stays.', label, 1000.0 * d_lin,
                        np.degrees(d_ang), lim_mm, lim_deg)
            return False
        T_measured = P6.junction_from_plug(r['T_base_obj'], part=self.part)
        T = self._grasp_frame(T_measured, label)
        geom.T_base_detection = T
        self.last = r
        _passed(label, r, svc, 'one view at the grasp, %.1f mm / %.1f deg from the scan\'s estimate (<= %.0f mm / '
                '%.0f deg)' % (1000.0 * d_lin, np.degrees(d_ang), lim_mm, lim_deg))
        log.info('%s: the in-hand pose will be held from THIS estimate (score %.2f, %s).', label,
                 r.get('score', 0.0), _terms(r))
        self._draw(r, T_measured)
        return True

    def _draw(self, r, T_junction):
        items = part_items(self.cfg, self.part, r['T_base_obj'], GOLD)
        items += [axis_item(v['T_base_obj'], self.part, ORANGE if v.get('consistent', True) else GREY)
                  for v in r.get('views', []) if v.get('found') and v.get('T_base_obj') is not None]
        items.append({'type': 'pose', 'T': T_junction.tolist(), 'length': 0.03})
        send_overlay(self.robot, OVERLAY_CONNECTOR, items,
                     [(f"6D pose: connector from {r.get('n_used', 1)} views, score {r.get('score', 0.0):.2f} "
                       f"({_terms(r)}); lines: each view's axis")])

    def _scene(self, views):
        """The views as a collision scene (perception/scene.py), for planning the approach."""
        return scene_from_views(self.cfg, self.robot, views)


def load_saved_views(cfg, label):
    """(directory, [(joints, Frame)]) of the newest views save_views kept under `label` -- the views
    a run can reuse instead of looking again (a fixture that has not moved) -- or (None, [])."""
    import glob
    import os
    import re
    from ..perception.camera import Frame
    tag = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
    dirs = sorted(d for d in glob.glob(os.path.join(cfg.get('data_dir', 'data'), 'pose6d', '*_' + tag))
                  if glob.glob(os.path.join(d, 'view_*.npz')))
    if not dirs:
        return None, []
    views = []
    for p in sorted(glob.glob(os.path.join(dirs[-1], 'view_*.npz'))):
        with np.load(p) as z:
            views.append((np.asarray(z['joints_rad'], dtype=float),
                          Frame(np.ascontiguousarray(z['rgb'][..., ::-1]), np.asarray(z['K'], dtype=float), None,
                                0.0, np.asarray(z['T_base_cam'], dtype=float), np.asarray(z['depth_m'], dtype=np.float32))))
    return dirs[-1], views


def scene_from_views(cfg, robot, views):
    """[(joints, Frame)] as a collision scene (perception/scene.py OccupancyScene, the robot left out
    of every frame), or None without a usable view."""
    good = [(q, f) for q, f in views if f.depth is not None and f.T_base_cam is not None]
    if not good:
        return None
    from ..perception.scene import OccupancyScene
    arm = robot.arm
    scene = OccupancyScene.for_cell(cfg)
    for q, f in good:
        scene.add({'depth_m': f.depth, 'K': f.K, 'T_base_cam': np.asarray(f.T_base_cam, dtype=float),
                   'T_base_tool0': arm.fk(q), 'T_tool0_fingertip': robot.T_tool0_fingertip,
                   'joints_rad': q, 'robot': arm.model.NAME})
    return scene


# ---------------------------------------------------------------------------- after the scan

def in_hand_tracker(cfg, robot, camera, scanner):
    """A ConnectorTracker for a Pose6DScanner (pose6d.in_hand.enabled, a real arm), attached to it
    so each scan starts it -- or None."""
    if not isinstance(scanner, Pose6DScanner) or robot.arm.dry_run or camera is None:
        return None
    tracker = ConnectorTracker(cfg, robot, camera)
    if not tracker.enabled:
        return None
    scanner.in_hand = tracker
    return tracker


class ConnectorTracker:
    """The plug kept estimated from the scan on -- through the approach, in the gripper, after the
    lift -- one frame at a time, each estimate starting from the last; drawn in the live view.

    pose6d.in_hand (bnc_assembly.yaml). LYING STILL, before the grasp: the estimate is in base,
    and the next starts from it. HELD -- the gripper closed (past `closed_counts`) with the
    fingertip within `attach_mm` of the plug's axis, as the cable model decides it -- it moves
    with the flange: the estimate is kept in tool0, from the camera's own frame (the camera is on
    the flange: no arm timing enters it), and the next starts from it carried to where the arm
    is. Let go (a missed grasp, the put-back, seated in the socket), it lies where it was let go
    and is tracked there, until the next scan or stop(). Each request only refines (refine_only):
    a plug out of view is no pose, not a search of the image; after a miss it waits `miss_wait_s`.

    IN THE HAND the fingers hide some of the plug, so the gate is looser (in_hand.min_iou,
    min_edge, min_cover) and agreement over time makes up for it: in_hand() is the medoid of the
    last `history` estimates and how many agree with it (agree_mm, agree_deg). The camera is on
    the flange too, so the held plug is seen from ONE viewpoint whatever the arm does: more frames
    average the noise (the lighting, what is behind it), they add no viewpoint.

    It observes. bnc_assembly reads it once, before the stand-off (measured_belief)."""

    def __init__(self, cfg, robot, camera):
        c = dict((cfg.section('pose6d') or {}).get('in_hand') or {}) if cfg.get('pose6d') else {}
        self.cfg, self.robot, self.camera = cfg, robot, camera
        self.part = P6.plug_part(cfg)
        self.enabled = bool(c.get('enabled', True))
        self.gap = float(c.get('gap_s', 1.0))
        self.overlay_hz = float(c.get('overlay_hz', 5.0))
        self.attach_m = float(c.get('attach_mm', 25.0)) / 1000.0
        self.closed_counts = int(c.get('closed_counts', 150))
        self.request = {k: c[k] for k in ('min_iou', 'min_edge', 'min_cover') if k in c}
        # DEPTH, per estimate: `auto` uses it while the plug is at least depth_min_mm from the camera
        # and not nearer, where the camera reads nothing true (the D405: ~7 cm; the iiwa's D415 before
        # it: 0.3 m, so nothing in the hand).
        ud = c.get('use_depth', 'auto')
        self.use_depth = None if str(ud).lower() == 'auto' else bool(ud)
        self.depth_min_m = float(c.get('depth_min_mm', 70.0)) / 1000.0
        self.request['joint_iters'] = int(c.get('joint_iters', 0))       # one view: its refinement is it
        self.request['refine_only'] = True
        self.request['quiet'] = True             # one after another: its lines would bury the prompts
        self.miss_wait = float(c.get('miss_wait_s', 1.0))
        self.history = max(1, int(c.get('history', 9)))
        self.agree_m = float(c.get('agree_mm', 3.0)) / 1000.0
        self.agree_rad = np.radians(float(c.get('agree_deg', 5.0)))
        self.update_belief = bool(c.get('update_belief', True))
        self.min_estimates = max(1, int(c.get('min_estimates', 3)))
        self.wait_s = float(c.get('wait_s', 30.0))
        self.max_corr_m = float(c.get('max_correction_mm', 15.0)) / 1000.0
        self.max_corr_rad = np.radians(float(c.get('max_correction_deg', 20.0)))
        self.T_base_plug = None                  # lying still: the last estimate
        self.T_tool0_plug = None                 # held: where it is in the hand
        self.held = []                           # [(monotonic time, T_tool0_plug)] accepted, held
        self.last = None                         # the last reply
        self.stats = {'estimates': 0, 'accepted': 0}
        self._epoch = 0                          # +1 each time it is taken or let go
        self._was_closed = None
        self._version = self._drawn = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._exit_hook = False

    # ------------------------------------------------------------------ running
    def begin(self, T_base_plug=None, T_tool0_plug=None):
        """The state to track from: the plug lying at `T_base_plug`, or held at `T_tool0_plug`."""
        with self._lock:
            self.T_base_plug = None if T_base_plug is None else np.asarray(T_base_plug, dtype=float)
            self.T_tool0_plug = None if T_tool0_plug is None else np.asarray(T_tool0_plug, dtype=float)
            self.held, self.last = [], None
            self.stats = {'estimates': 0, 'accepted': 0}
            self._was_closed = None              # held only when it is seen to CLOSE on it
            self._epoch += 1
            self._version += 1

    def start(self, T_base_plug=None, T_tool0_plug=None):
        """Track, in threads of its own, from the plug lying at `T_base_plug` or held at `T_tool0_plug`."""
        if not self.enabled:
            return
        self.stop()
        self.begin(T_base_plug, T_tool0_plug)
        self._stop = stop = threading.Event()        # this run's: one still finishing must not run on
        self._threads = [threading.Thread(target=self._estimate_loop, args=(stop,), name='pose6d-track', daemon=True),
                         threading.Thread(target=self._draw_loop, args=(stop,), name='pose6d-draw', daemon=True)]
        for t in self._threads:
            t.start()
        if not self._exit_hook:
            atexit.register(self.stop)           # before the service's own shutdown (registered first)
            self._exit_hook = True
        log.info('6D tracking: the connector, %s.', 'in the hand' if T_tool0_plug is not None else 'from the scan')

    def stop(self):
        self._stop.set()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=1.0)              # a request in flight finishes on its own
        self._threads = []

    def running(self):
        return any(t.is_alive() for t in self._threads)

    def _estimate_loop(self, stop):
        try:
            while not stop.is_set():
                if not self.step():
                    break
                found = self.last is not None and self.last.get('found')
                stop.wait(self.gap if found else max(self.gap, self.miss_wait))
        except Exception as exc:                 # noqa: BLE001 -- an observer: it stops, the run goes on
            log.warning('6D tracking stopped: %s', exc)

    def _draw_loop(self, stop):
        arm = self.robot.arm
        try:
            while not stop.is_set():
                tool = arm.fk(arm.q())
                self.hold(tool, self._counts())
                self.draw(tool)
                stop.wait(1.0 / max(self.overlay_hz, 0.1))
        except Exception as exc:                 # noqa: BLE001 -- drawing only
            log.debug('6D tracking overlay stopped: %s', exc)

    def _counts(self):
        return getattr(getattr(self.robot, 'gripper', None), 'last_counts', None)

    # ------------------------------------------------------------------ one estimate
    def step(self, frame=None):
        """One frame, one estimate from the last pose. False with nothing to track from."""
        if frame is None:
            try:
                frame = self.camera.capture(reconnect=False)
            except TypeError:                    # a camera without that option
                frame = self.camera.capture()
        if frame is None or frame.T_base_cam is None or frame.depth is None:
            return True
        tool = np.asarray(frame.T_base_cam, dtype=float) @ inverse(self.robot.T_tool0_cam)   # at the exposure
        with self._lock:
            epoch, held = self._epoch, self.T_tool0_plug is not None
            prior = tool @ self.T_tool0_plug if held else self.T_base_plug
        if prior is None:
            return False
        use_depth = self.use_depth
        if use_depth is None:
            centre = prior[:3, :3] @ P6.CENTRE_M[self.part] + prior[:3, 3]
            use_depth = bool(np.linalg.norm(centre - np.asarray(frame.T_base_cam, dtype=float)[:3, 3]) >= self.depth_min_m)
        r = P6.service(self.cfg).estimate(self.part, [frame], prior=prior, use_depth=use_depth, hide_gripper=not held,
                                          **self.request)
        with self._lock:
            self.stats['estimates'] += 1
            self.last = r
            if not r.get('found') or epoch != self._epoch:      # rejected, or taken / let go meanwhile
                return True
            self.stats['accepted'] += 1
            if held:
                self.held = (self.held + [(time.monotonic(), inverse(tool) @ r['T_base_obj'])])[-self.history:]
                self.T_tool0_plug = P6.robust_pose([T for _, T in self.held], self.part, self.agree_m,
                                                   self.agree_rad)[0]
            else:
                self.T_base_plug = r['T_base_obj']
            self._version += 1
        return True

    def hold(self, tool, counts):
        """Take the plug into the hand when the gripper closes on it, leave it where it is let go.
        True while it is held."""
        closed = counts is not None and int(counts) >= self.closed_counts
        with self._lock:
            closing, self._was_closed = closed and self._was_closed is False, closed if counts is not None else None
            if self.T_tool0_plug is None and closing and self.T_base_plug is not None:
                tip = inverse(self.T_base_plug) @ np.append((tool @ self.robot.T_tool0_fingertip)[:3, 3], 1.0)
                s = float(np.clip(tip[2], P6.JUNCTION_Z[self.part], P6.MATING_Z[self.part]))
                if np.linalg.norm(tip[:3] - [0.0, 0.0, s]) < self.attach_m:
                    self.T_tool0_plug = inverse(tool) @ self.T_base_plug
                    self.held = []
                    self._epoch += 1
                    self._version += 1
                    log.info('6D tracking: the gripper holds the connector; tracking it in the hand.')
            elif self.T_tool0_plug is not None and counts is not None and not closed:
                self.T_base_plug = tool @ self.T_tool0_plug
                self.T_tool0_plug = None
                self._epoch += 1
                self._version += 1
                log.info('6D tracking: the gripper let go of the connector; tracking it where it was let go.')
            return self.T_tool0_plug is not None

    # ------------------------------------------------------------------ results
    def in_hand(self, since=None):
        """(T_tool0_plug -- the medoid --, how many agree with it, how many) from the estimates in
        the hand (since `since`, monotonic); None without one."""
        with self._lock:
            got = [T for t, T in self.held if since is None or t >= since]
        if not got:
            return None
        T, n_agree = P6.robust_pose(got, self.part, self.agree_m, self.agree_rad)
        return T, n_agree, len(got)

    def measured_belief(self, T_ftip_conn, T_tool0_fingertip, T_junction_connector, since=None):
        """The in-hand belief from the estimates made since `since` (monotonic; now if None), once
        `min_estimates` agree (waiting up to `wait_s`) and it is within max_correction_mm/_deg of
        `T_ftip_conn` -- else `T_ftip_conn` unchanged, said why."""
        if not self.update_belief:
            return T_ftip_conn
        with self._lock:
            held = self.T_tool0_plug is not None
        if not held:
            log.warning('6D IN HAND: the connector is not being tracked in the hand -- the pick\'s belief stays.')
            return T_ftip_conn
        since = time.monotonic() if since is None else since
        deadline = time.monotonic() + self.wait_s
        log.info('6D IN HAND: measuring the connector in the gripper (%d agreeing estimates, up to %.0f s).',
                 self.min_estimates, self.wait_s)
        while True:
            got = self.in_hand(since)
            if (got is not None and got[1] >= self.min_estimates) or time.monotonic() >= deadline \
                    or not self.running():
                break
            time.sleep(0.2)
        if got is None or got[1] < self.min_estimates:
            log.warning('6D IN HAND: %s in %.0f s -- the pick\'s belief stays.',
                        'no estimate passed the gate' if got is None
                        else f'{got[1]} of {got[2]} estimates agree (need {self.min_estimates})', self.wait_s)
            return T_ftip_conn
        T_ftip_plug = inverse(T_tool0_fingertip) @ got[0]
        T_new = P6.belief_from_plug(T_ftip_plug, T_junction_connector, T_ftip_conn, self.part)
        d_lin, d_ang = pose_error(T_ftip_conn, T_new)
        if d_lin > self.max_corr_m or d_ang > self.max_corr_rad:
            log.error('6D IN HAND: the measured connector is %.1f mm / %.1f deg from the pick\'s belief -- over '
                      'pose6d.in_hand.max_correction_mm/_deg; the pick\'s belief stays.', 1000 * d_lin,
                      np.degrees(d_ang))
            return T_ftip_conn
        log.info('6D IN HAND: %d of %d estimates agree; the belief moves %.1f mm / %.1f deg to the measured '
                 'connector (%s).', got[1], got[2], 1000 * d_lin, np.degrees(d_ang),
                 'its whole pose' if P6.SYMMETRY[self.part] == 'none' else 'its axis and position; its turn about the axis kept')
        return T_new

    def report(self, T_tool0_fingertip, T_ftip_junction_commanded):
        """Log where the junction is in the fingertip's frame, measured in the hand, against where
        the grasp meant it to be (the inverse of fingertip_in_connector)."""
        got = self.in_hand()
        if got is None:
            log.info('6D IN HAND: no estimate in the hand passed the gate (%d/%d accepted overall).',
                     self.stats['accepted'], self.stats['estimates'])
            return None
        J = inverse(T_tool0_fingertip) @ got[0] @ P6.T_cord_junction(self.part)
        C = np.asarray(T_ftip_junction_commanded, dtype=float)
        d = inverse(C) @ J
        tilt = np.degrees(np.arccos(np.clip(J[:3, 0] @ C[:3, 0], -1.0, 1.0)))
        log.info('6D IN HAND (%d of %d estimates agree): the junction sits %s mm from where the grasp meant it '
                 '(its frame: x along the connector), its axis %.1f deg off.', got[1], got[2],
                 (d[:3, 3] * 1000).round(1).tolist(), tilt)
        return d

    # ------------------------------------------------------------------ the live view
    def draw(self, tool):
        with self._lock:
            held = self.T_tool0_plug is not None
            if not held and self._version == self._drawn:
                return
            self._drawn = self._version
            T = tool @ self.T_tool0_plug if held else self.T_base_plug
            r, n = self.last, len(self.held)
        if T is None:
            return
        items = part_items(self.cfg, self.part, T, GOLD if r is None or r.get('found') else GREY)
        items.append({'type': 'pose', 'T': P6.junction_from_plug(T, part=self.part).tolist(), 'length': 0.03})
        where = 'in the gripper' if held else 'lying'
        last = 'no estimate yet' if r is None else (
            f"last {'accepted' if r.get('found') else 'rejected'} ({_terms(r)})")
        agree = ''
        if held and n:
            got = self.in_hand()
            agree = f'; {got[1]} of {got[2]} in-hand estimates agree' if got else ''
        send_overlay(self.robot, OVERLAY_CONNECTOR, items,
                     [(f"6D tracking: connector {where}; {last}{agree}; "
                       f"{self.stats['accepted']}/{self.stats['estimates']} accepted")])


# ================================================================================ ONLINE THROUGH CONTACT
def online_cfg(cfg):
    """pose6d.online (bnc_assembly.yaml), as a dict ({} without one)."""
    return dict((cfg.section('pose6d') or {}).get('online') or {}) if cfg.get('pose6d') else {}


def socket_tracker(cfg, robot, camera):
    """A SocketTracker (pose6d.online.enabled, a real arm, a camera), or None."""
    if robot.arm.dry_run or camera is None or not bool(online_cfg(cfg).get('enabled', False)):
        return None
    return SocketTracker(cfg, robot, camera)


class SocketTracker(ConnectorTracker):
    """THE SOCKET KEPT ESTIMATED THROUGH CONTACT (the user, 2026-11-11: "update both the in-hand connector
    pose and the socket pose via the RGBD data online during all contact phases"). The ConnectorTracker's
    loop -- one wrist frame, one estimate refined from the last, in a thread of its own -- for the socket,
    which never moves: every accepted estimate is kept in base, and the pose is the medoid of the last
    `history` (pose6d.online.socket). The held plug hides part of it, so the gate is looser than the
    localization's and agreement over time makes up for it. Drawn as the socket's overlay (pose6d_socket),
    so the live view shows the socket where the newest agreed estimate puts it. It observes; OnlinePoses
    decides what the run does with it."""

    def __init__(self, cfg, robot, camera):
        super().__init__(cfg, robot, camera)
        c = dict(online_cfg(cfg).get('socket') or {})
        self.part = P6.socket_part(cfg)
        self.enabled = bool(online_cfg(cfg).get('enabled', False))
        self.gap = float(c.get('gap_s', 0.5))
        self.miss_wait = float(c.get('miss_wait_s', 1.0))
        self.request = {k: c[k] for k in ('min_iou', 'min_edge', 'min_cover') if k in c}
        self.request.update(joint_iters=0, refine_only=True, quiet=True)
        ud = c.get('use_depth', 'auto')
        self.use_depth = None if str(ud).lower() == 'auto' else bool(ud)
        self.depth_min_m = float(c.get('depth_min_mm', 70.0)) / 1000.0
        self.history = max(1, int(c.get('history', 7)))
        self.agree_m = float(c.get('agree_mm', 3.0)) / 1000.0
        self.agree_rad = np.radians(float(c.get('agree_deg', 5.0)))
        self.base = []                           # [(monotonic time, T_base_socket)] accepted

    def start(self, T_base_plug=None, T_tool0_plug=None):
        """Track from the socket at `T_base_plug` (its last 6D pose)."""
        if not self.enabled or T_base_plug is None:
            return
        super().start(T_base_plug=T_base_plug)
        with self._lock:
            self.base = []

    def hold(self, tool, counts):
        return False                             # the socket is never in the hand

    def step(self, frame=None):
        if frame is None:
            try:
                frame = self.camera.capture(reconnect=False)
            except TypeError:
                frame = self.camera.capture()
        if frame is None or frame.T_base_cam is None or frame.depth is None:
            return True
        with self._lock:
            epoch, prior = self._epoch, self.T_base_plug
        if prior is None:
            return False
        use_depth = self.use_depth
        if use_depth is None:
            centre = prior[:3, :3] @ P6.CENTRE_M[self.part] + prior[:3, 3]
            use_depth = bool(np.linalg.norm(centre - np.asarray(frame.T_base_cam, dtype=float)[:3, 3]) >= self.depth_min_m)
        r = P6.service(self.cfg).estimate(self.part, [frame], prior=prior, use_depth=use_depth, hide_gripper=True,
                                          **self.request)
        with self._lock:
            self.stats['estimates'] += 1
            self.last = r
            if not r.get('found') or epoch != self._epoch:
                return True
            self.stats['accepted'] += 1
            self.base = (self.base + [(time.monotonic(), np.asarray(r['T_base_obj'], dtype=float))])[-self.history:]
            self.T_base_plug = P6.robust_pose([T for _, T in self.base], self.part, self.agree_m, self.agree_rad)[0]
            self._version += 1
        return True

    def measured(self, since=None):
        """(T_base_socket -- the medoid --, how many agree with it, how many) from the estimates since `since`
        (monotonic), or None without one."""
        with self._lock:
            got = [T for t, T in self.base if since is None or t >= since]
        if not got:
            return None
        T, n_agree = P6.robust_pose(got, self.part, self.agree_m, self.agree_rad)
        return T, n_agree, len(got)

    def draw(self, tool):
        with self._lock:
            if self._version == self._drawn:
                return
            self._drawn = self._version
            T, r, n = self.T_base_plug, self.last, len(self.base)
        if T is None:
            return
        m = mesh_item(self.cfg, self.part, T, BLUE)
        items = [m] if m is not None else part_items(self.cfg, self.part, T, BLUE)
        items.append({'type': 'pose', 'T': np.asarray(T, dtype=float).tolist(), 'length': 0.04})
        last = 'no estimate yet' if r is None else f"last {'accepted' if r.get('found') else 'rejected'} ({_terms(r)})"
        got = self.measured() if n else None
        agree = f'; {got[1]} of {got[2]} agree' if got else ''
        send_overlay(self.robot, OVERLAY_SOCKET, items,
                     [f"6D online: socket; {last}{agree}; {self.stats['accepted']}/{self.stats['estimates']} accepted"])


def _limited_step(A, G, max_m, max_rad):
    """`A` moved toward `G` (4x4, in the same frame) by at most `max_m` of translation and `max_rad` of turn."""
    d = G[:3, 3] - A[:3, 3]
    n = float(np.linalg.norm(d))
    out = np.array(A, dtype=float)
    out[:3, 3] = A[:3, 3] + (d if n <= max_m else d * (max_m / n))
    rv = Rotation.from_matrix(G[:3, :3] @ A[:3, :3].T).as_rotvec()
    a = float(np.linalg.norm(rv))
    if a > max_rad:
        rv = rv * (max_rad / a)
    out[:3, :3] = Rotation.from_rotvec(rv).as_matrix() @ A[:3, :3]
    return out


def _drop(D, axis, axial, round_part):
    """A delta in a part's own frame with what is not to be taken from vision dropped: the translation along
    `axis` (0 = x, 2 = z) unless `axial`, and -- for a round part -- the turn about it (no view tells it): only
    the axis's tilt is kept."""
    D = np.array(D, dtype=float)
    if not axial:
        D[axis, 3] = 0.0
    if round_part:
        e = np.zeros(3)
        e[axis] = 1.0
        a = D[:3, :3] @ e
        rv = np.cross(e, a)
        s = float(np.linalg.norm(rv))
        ang = float(np.arctan2(s, float(e @ a)))
        D[:3, :3] = np.eye(3) if s < 1e-12 else Rotation.from_rotvec(rv / s * ang).as_matrix()
    return D


class OnlinePoses:
    """THE SOCKET AND THE HELD CONNECTOR, UPDATED ONLINE THROUGH CONTACT (pose6d.online; the user, 2026-11-11).

    Every contact reference the insertion builds is `T_target @ row @ inv(T_tool0_conn)` (bnc_assembly
    traj_ref). A re-measured target T' and in-hand pose C' turn ANY such reference into

        ref' = T' @ row @ inv(C') = L @ ref @ R,      L = T' T^-1 (base),  R = C C'^-1 (tool0)

    so the correction is applied in one place -- the compliance controller's command (robot/admittance.py:
    the arm's `reference_correction`) -- and reaches every loop of the phase without touching them.

    WHAT IS TAKEN FROM THE ESTIMATES: the socket tracker's medoid and the in-hand tracker's, once
    `min_estimates` agree since the phase began, each as a delta in the part's OWN frame from the beliefs the
    phase started on -- without the translation along the insertion axis (`axial: false`: the contact measures
    depth; a single view's range is its weakest direction) and, for a round part, without the turn about it.
    A delta past max_socket_mm/_deg (max_hand_mm/_deg) is refused (said once). APPLIED GRADUALLY: the applied
    deltas move toward those at `rate_mm_s` / `rate_deg_s`, so a new estimate is never a step in the reference
    while the plug is touching the socket.

    applied() gives the corrected socket pose and in-hand belief, for the run to adopt when the phase ends."""

    def __init__(self, cfg, robot, sock, conn, T_socket0, T_ftip_conn0, T_junction_conn, target_fn):
        c = online_cfg(cfg)
        self.robot, self.sock, self.conn = robot, sock, conn
        self.T_s0 = np.asarray(T_socket0, dtype=float)
        self.F0 = np.asarray(T_ftip_conn0, dtype=float)
        self.C0 = robot.T_tool0_fingertip @ self.F0
        self.T_jc = np.asarray(T_junction_conn, dtype=float)
        self.target_fn = target_fn               # socket pose -> the run's target frame (P6.target_from_socket)
        self.Tt0 = target_fn(self.T_s0)
        self.min_est = max(1, int(c.get('min_estimates', 2)))
        self.rate_m = float(c.get('rate_mm_s', 1.0)) / 1000.0
        self.rate_rad = np.radians(float(c.get('rate_deg_s', 1.0)))
        self.max_s = (float(c.get('max_socket_mm', 5.0)) / 1000.0, np.radians(float(c.get('max_socket_deg', 5.0))))
        self.max_h = (float(c.get('max_hand_mm', 5.0)) / 1000.0, np.radians(float(c.get('max_hand_deg', 5.0))))
        self.axial = bool(c.get('axial', False))
        self.since = time.monotonic()
        self.D = np.eye(4)                       # applied socket delta, in the socket's frame
        self.E = np.eye(4)                       # applied in-hand delta, in the connector's frame
        self.D_goal, self.E_goal = np.eye(4), np.eye(4)
        self._L, self._R = np.eye(4), np.eye(4)
        self._t_goal, self._t_last = 0.0, None
        self._refused = set()
        self.used = {'socket': 0, 'hand': 0}

    # -------------------------------------------------------------- the goals, from the trackers
    def _goals(self):
        if self.sock is not None:
            got = self.sock.measured(self.since)
            if got is not None and got[1] >= self.min_est:
                D = _drop(inverse(self.T_s0) @ got[0], 2, self.axial, P6.SYMMETRY[self.sock.part] != 'none')
                lin, ang = float(np.linalg.norm(D[:3, 3])), float(Rotation.from_matrix(D[:3, :3]).magnitude())
                if lin <= self.max_s[0] and ang <= self.max_s[1]:
                    self.D_goal, self.used['socket'] = D, got[2]
                elif ('socket', got[2]) not in self._refused:
                    self._refused.add(('socket', got[2]))
                    log.warning('6D ONLINE: the socket estimate is %.1f mm / %.1f deg from the one the contact began on '
                                '-- over pose6d.online.max_socket_mm/_deg; not followed.', 1000 * lin, np.degrees(ang))
        if self.conn is not None:
            got = self.conn.in_hand(self.since)
            if got is not None and got[1] >= self.min_est:
                F = P6.belief_from_plug(inverse(self.robot.T_tool0_fingertip) @ got[0], self.T_jc, self.F0,
                                        self.conn.part)
                E = _drop(inverse(self.C0) @ (self.robot.T_tool0_fingertip @ F), 0, self.axial,
                          P6.SYMMETRY[self.conn.part] != 'none')
                lin, ang = float(np.linalg.norm(E[:3, 3])), float(Rotation.from_matrix(E[:3, :3]).magnitude())
                if lin <= self.max_h[0] and ang <= self.max_h[1]:
                    self.E_goal, self.used['hand'] = E, got[2]
                elif ('hand', got[2]) not in self._refused:
                    self._refused.add(('hand', got[2]))
                    log.warning('6D ONLINE: the in-hand estimate is %.1f mm / %.1f deg from the belief the contact began '
                                'on -- over pose6d.online.max_hand_mm/_deg; not followed.', 1000 * lin, np.degrees(ang))

    def _advance(self):
        now = time.monotonic()
        if now - self._t_goal >= 0.1:            # the trackers at most 10 times a second, not every servo cycle
            self._t_goal = now
            self._goals()
        dt = 0.0 if self._t_last is None else min(now - self._t_last, 0.1)
        self._t_last = now
        if dt > 0.0:
            self.D = _limited_step(self.D, self.D_goal, self.rate_m * dt, self.rate_rad * dt)
            self.E = _limited_step(self.E, self.E_goal, self.rate_m * dt, self.rate_rad * dt)
            self._L = self.target_fn(self.T_s0 @ self.D) @ inverse(self.Tt0)
            self._R = self.C0 @ inverse(self.E) @ inverse(self.C0)

    # -------------------------------------------------------------- the controller's hook
    def correct(self, T_ref):
        """The reference as the online estimates put it (each servo cycle)."""
        self._advance()
        return self._L @ np.asarray(T_ref, dtype=float) @ self._R

    def uncorrect(self, T):
        """A pose as the phase's own (uncorrected) references describe it: what correct() takes to `T`."""
        return inverse(self._L) @ np.asarray(T, dtype=float) @ inverse(self._R)

    # -------------------------------------------------------------- the result
    def applied(self):
        """(socket pose, in-hand belief T_ftip_conn, socket delta (mm, deg), in-hand delta (mm, deg)) as applied."""
        T_s = self.T_s0 @ self.D
        F = inverse(self.robot.T_tool0_fingertip) @ (self.C0 @ self.E)
        mm_deg = lambda M: (1000.0 * float(np.linalg.norm(M[:3, 3])),                 # noqa: E731
                            float(np.degrees(Rotation.from_matrix(M[:3, :3]).magnitude())))
        return T_s, F, mm_deg(self.D), mm_deg(self.E)
