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
mating face, z as near up as that allows (an RJ45's: along its latch side, as measured). It is ON the axis (pose_on_axis): no resting-height
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
from ..transforms import inverse, pose_error

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


def draw_socket(cfg, robot, reply, T_target, socket=P6.SOCKET):
    """The socket's CAD where the views put it, the plug seated in it, and the target frame."""
    T_s = reply['T_base_obj']
    items = part_items(cfg, socket, T_s, BLUE)
    items += part_items(cfg, P6.PLUG_FOR[socket], T_s @ P6.T_socket_cord_seated(socket), GREY, radius=0.0005)
    items += [axis_item(v['T_base_obj'], socket, ORANGE if v.get('consistent', True) else GREY)
              for v in reply.get('views', []) if v.get('found') and v.get('T_base_obj') is not None]
    items.append({'type': 'pose', 'T': np.asarray(T_target, dtype=float).tolist(), 'length': 0.03})
    send_overlay(robot, OVERLAY_SOCKET, items,
                 [(f"6D pose: socket from {reply.get('n_used', 1)} views, score {reply.get('score', 0.0):.2f} "
                   f"({_terms(reply)}); target = the seated connector frame")])


def _orbit_cfg(cfg):
    c = dict(cfg.section('pose6d') or {}) if cfg.get('pose6d') else {}
    o = dict(c.get('orbit') or {})
    return {'tilts': np.radians(np.asarray(o.get('tilts_deg', [20.0]), dtype=float)), 'pans': int(o.get('pans', 6)),
            'caps': dict(o.get('speed') or {'max_cartesian_translation_mm_s': 50.0, 'max_cartesian_rotation_deg_s': 60.0,
                                            'max_joint_velocity_deg_s': 60.0}),
            'settle_s': float(o.get('settle_s', 0.5)), 'frames': int(o.get('frames_per_view', 3)),
            'clearance_m': float(o.get('min_clearance_mm', 60.0)) / 1000.0,
            'guard_n': float(o.get('guard_force_n', 15.0)), 'enabled': bool(o.get('enabled', True)),
            'bench_z': float(cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0}


def still_view(camera, arm, n_frames):
    """(joints, one Frame with the median depth of `n_frames`), or None if the arm moved."""
    from ..apps.capture_rgbd import median_depth, still_frames
    got = still_frames(camera, arm.q, n_frames)
    if got is None:
        return None
    frame = got[1][-1]
    if frame.depth is not None:
        frame.depth = median_depth(got[1])
    return np.asarray(got[0], dtype=float), frame


def orbit_views(robot, camera, pivot, oc, confirm=None, label='orbit'):
    """[(joints, Frame)] from an orbit of the camera about `pivot` (base frame), the arm back where
    it started; [] when it cannot run (too few views reachable, declined)."""
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
            time.sleep(oc['settle_s'])
            got = still_view(camera, arm, oc['frames'])
            if got is None:
                log.warning('  %s view %d: the arm moved through the exposure; skipped.', label, k)
                continue
            views.append(got)
        if not arm.move_l(T0, label=f'{label}: back', caps=oc['caps']):
            log.error('%s: the move back stopped short.', label)
    finally:
        arm.clear_guards()
    return views


def estimate_part(cfg, robot, camera, part, confirm=None, label=None, prior=None):
    """(the service's reply, [(joints, Frame)] used) for `part` (perception/pose6d.py PLUG, SOCKET,
    RJ45 or RJ45_SOCKET) from where the arm is plus an orbit about it; (None, views) if not found.

    `prior` (T_base_obj, e.g. the socket from a recorded mate): the first view refines from it
    (its projection boxes the part) before searching the whole image; found in neither, the orbit
    still runs about it and the views together decide. Every view and reply is kept (save_views)."""
    label = label or f'6D pose: the {part}'
    oc = _orbit_cfg(cfg)
    svc = P6.service(cfg)
    replies = []
    first = still_view(camera, robot.arm, oc['frames'])
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
        if r0.get('found'):
            T0 = r0['T_base_obj']
            log.info('%s: found in the first view, score %.2f (%s).', label, r0['score'], _terms(r0))
        elif prior is not None and oc['enabled']:
            T0 = np.asarray(prior, dtype=float)
            log.warning('%s: not found in the first view (%s) -- orbiting about the recorded pose anyway; the '
                        'views together decide.', label, r0.get('why') or _terms(r0))
        else:
            log.error('%s: not found in the first view (%s).', label, r0.get('why') or _terms(r0))
            return None, views
        if not oc['enabled']:
            return r0, views
        pivot = T0[:3, :3] @ P6.CENTRE_M[part] + T0[:3, 3]
        views = [first] + orbit_views(robot, camera, pivot, oc, confirm, label)
        if len(views) == 1:
            return (r0 if r0.get('found') else None), views
        r = svc.estimate(part, [f for _, f in views], prior=T0)
        replies.append(('all views', r))
        if not r.get('found'):
            log.error('%s: the %d views together fail the gate (%s) -- not trusted.', label, len(views), _terms(r))
            return None, views
        gap = np.linalg.norm(r['T_base_obj'][:3, 3] - T0[:3, 3])
        log.info('%s: %d of %d views together, score %.2f (%s); %.1f mm from the first view\'s estimate.',
                 label, r['n_used'], len(views), r['score'], _terms(r), 1000 * gap)
        return r, views
    finally:
        save_views(cfg, label, views, replies, prior)


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

    def reset(self):
        pass

    def reselect(self):
        self._selection = None

    def scan(self, confirm=None):
        self.scene = None
        if self.in_hand is not None:
            self.in_hand.stop()                 # the service is this scan's now
        r, views = estimate_part(self.cfg, self.robot, self.camera, self.part, confirm, '6D pose: the connector')
        self.scene = self._scene(views)
        if r is None:
            return None
        T = P6.junction_from_plug(r['T_base_obj'], part=self.part)
        self.last = r
        self._selection = T[:3, 3].copy()
        log.info('6D pose: the connector\'s junction at %s mm, heading %.1f deg, %.1f deg off level.',
                 (T[:3, 3] * 1000).round(1), np.degrees(np.arctan2(T[1, 0], T[0, 0])),
                 np.degrees(np.arcsin(np.clip(T[2, 0], -1, 1))))
        self._draw(r, T)
        if self.in_hand is not None:
            self.in_hand.start(T_base_plug=r['T_base_obj'])
        return T

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
        # (lying, on the approach) and not nearer (in the hand: what the D415 reads there is wrong).
        ud = c.get('use_depth', 'auto')
        self.use_depth = None if str(ud).lower() == 'auto' else bool(ud)
        self.depth_min_m = float(c.get('depth_min_mm', 300.0)) / 1000.0
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
        r = P6.service(self.cfg).estimate(self.part, [frame], prior=prior, use_depth=use_depth, **self.request)
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
