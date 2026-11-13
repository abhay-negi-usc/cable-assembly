"""6D pose of the connector and its socket (the BNC's or the RJ45's) from their CAD models, from
several views.

    pose_method: pose6d          (bnc_assembly.yaml: the connector AND the socket come from here,
                                  and SAM3 is never loaded -- one method or the other, not both)

THE ESTIMATOR is the CAD-based pipeline in ~/abhay_ws/RAG-6DPose-code-abhay (pose_pipeline:
Grounding DINO + SAM2 for the part's mask, render-and-compare against its mesh, an acceptance
gate). It runs in its own env (`rag6dpose`) as a long-lived process -- pose6d_server.py, next to
this file -- started on first use and kept for the run (`service(cfg)`, `shutdown()`); the apps
start it in the background as they set up (`preload(cfg)`), the plug's and socket's models loaded.

SEVERAL VIEWS PER ESTIMATE (`estimate`): every view's frame goes over with the camera's pose at
capture, and the service estimates each, drops the ones that disagree with the best, and refines
one base-frame pose against all the rest at once (see pose6d_server.py). The caller drives the
views (skills/pose6d_pick.py: a first view, then an orbit about what it found).

THE PARTS' FRAMES are their STEP files' (the meshes are those, in mm; here metres):
  * the PLUG ('cord_end_pos_z', McMaster 6641T31's +z end): the cord's axis is z, the mating face
    at z 152.40 mm, the strain relief from 108.65 (where the cable leaves it: the JUNCTION) to
    137.29. The pick's connector frame (the junction frame every scanner returns) is the junction
    on the axis, x along +z toward the mating face (T_cord_junction).
  * the SOCKET ('adapter', 4441N11): its barrel's axis is z, the barrel's top at 12.14 mm; only the
    axis is measured, not the turn about it (SYMMETRY). Seated
    (sim/scripts/build_bnc.py), the plug's insulator face (151.38 along the cord) rests on the
    socket's dielectric (3.73 up it), facing down: z_socket = 155.11 - z_cord, y_socket = -y_cord
    (T_socket_cord_seated). So the junction frame of a seated plug, in the socket's frame, is
    T_socket_cord_seated @ T_cord_junction -- which is what turns a socket pose into a target.
  * the ETHERNET PLUG ('rj45_plug'; pose6d.plug: rj45_plug): McMaster 9953K212's clear body on the
    lab plug's boot, measured (prepare_models.py RJ45_BOOT -- the lab's is not McMaster's), turned
    so the cord's axis is z too ((x, y, z) = (-z, y, x) of the STEP's): the mating face at z 153.74
    mm, the strain relief from 119.30 (the JUNCTION), the latch side +y. It has
    no symmetry: the whole pose is measured, the turn about the axis too, and its junction frame
    keeps it -- z along the latch side (T_cord_junction(RJ45)).
  * the ETHERNET SOCKET ('rj45_adapter', 1422N31, a two-sided coupler on the BNC adapter's 26 x 31
    flange): its axis is z, the front jack opening at z 18.30 mm (the flange's face), its latch
    slot toward -y, its floor at 6.18; the rear jack opens at -11.70. No symmetry either: the
    latch slot sets which way up the plug goes in, so its whole pose is measured. Seated, the
    plug points down its -z, latch toward -y, its tip on the floor (an assumption: the CAD has no
    stop but the floor): z_socket = 159.92 - z_cord, y_socket = -y_cord (T_socket_cord_seated).
  The socket is the plug's: pose6d.plug picks both (SOCKET_FOR).
"""

import atexit
import json
import os
import shutil
import subprocess
import tempfile
import threading

import numpy as np

from .. import log as urlog

log = urlog.get('pose6d')

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose6d_server.py')
PLUG, SOCKET, RJ45, RJ45_SOCKET = 'plug', 'socket', 'rj45_plug', 'rj45_socket'
USB, USB_SOCKET = 'usb_plug', 'usb_socket'             # the USB-A cord 4974T135 and the adapter 4574N11 (2026-11-12)
PLUGS = (PLUG, RJ45, USB)            # what pose6d.plug may name: the part the pick picks
SOCKET_FOR = {PLUG: SOCKET, RJ45: RJ45_SOCKET, USB: USB_SOCKET}          # ... and the socket it goes into
PLUG_FOR = {v: k for k, v in SOCKET_FOR.items()}
# The estimator's model of each part. The BNC socket's is the 4441N11 AS MOUNTED on the iiwa bench
# (adapter_mount, obj 8): its flange and nose in front of the panel (adapter_front, obj 7 -- the body
# behind the panel is hidden, and with it in the silhouette no view passed the gate: iou 0.48-0.74)
# plus the lab's mounting box it is bolted into, in the adapter's frame. 2026-11-02, 3 views: iou
# 0.88-0.91 with the box, 0.78-0.86 without, 0.48-0.74 the full adapter.
# EVERY SOCKET'S MODEL CARRIES ITS TWO M3 CAP SCREWS (the user, 2026-11-12: "the sockets have M3 socket head cap
# screws holding them into the mount"): heads only, in the D-flange's two diagonal holes (the estimator repo's
# prepare_models.py FLANGE_SCREWS; the meshes without them in its models' no_screws/). On this bench's five-view
# BNC socket estimates they change the pose by under 0.7 mm (the run-to-run spread) and fit a little better
# (score 2.27 -> 2.28); at the insertion's range they are the flange's clearest landmarks.
OBJECTS = {PLUG: 'cord_end_pos_z', SOCKET: 'adapter_mount', RJ45: 'rj45_plug', RJ45_SOCKET: 'rj45_adapter',
           USB: 'usb_plug', USB_SOCKET: 'usb_adapter_mount'}
# The USB socket's too is AS MOUNTED (usb_adapter_mount, obj 13: the adapter from its flange's back face forward +
# the same mounting box, 2026-11-12): the whole adapter (usb_adapter) passed no gate on the bench -- the search took
# the BNC socket for it (iou 0.69), a refine from the right place got iou 0.56 -- and this one is found (0.91).
# ... and the model DRAWN where one is wanted whole (cad_mesh: the live view, the held part).
DRAWN = {SOCKET: 'adapter', USB_SOCKET: 'usb_adapter'}
# ... and what a part is mounted in, drawn beside it where the estimate puts it (the live view): the
# BNC socket's mounting box and its two button heads (socket_mount, obj 9), in the adapter's frame.
MOUNTS = {SOCKET: 'socket_mount', USB_SOCKET: 'usb_socket_mount'}
# What a view can tell apart: neither part's turn about its own axis. The plug is round; the
# socket's round barrel hides its flange's turn -- on the 2026-10-30 scan (pose_pipeline.eval_scan)
# fourteen views put its axis within 0.6-6.8 deg of each other and its turn about it anywhere.
# The RJ45 parts have none: the plug's latch side and flat body, the jack's latch slot and
# uneven housings tell every turn apart.
# The USB-A parts have none either: the plug is flat and goes in one way up (the receptacle's tongue into its
# cavity), though its two wide faces look much alike -- see usb_assembly.yaml on which way up it lies.
SYMMETRY = {PLUG: 'axis_z', SOCKET: 'axis_z', RJ45: 'none', RJ45_SOCKET: 'none', USB: 'none', USB_SOCKET: 'none'}
# The parts' middles in their CAD frames (models_info.json's boxes): what an orbit turns about.
CENTRE_M = {PLUG: np.array([0.0, 0.0, 0.1305]), SOCKET: np.array([0.0, 0.0, -0.0034]),
            RJ45: np.array([0.0, 0.00175, 0.13652]), RJ45_SOCKET: np.array([0.0, 0.0, 0.0033]),
            USB: np.array([0.0, 0.0, 0.4452]), USB_SOCKET: np.array([0.0, -0.0005, -0.0084])}
# Along each plug's axis: where the cable leaves the boot, and the mating face. The USB's: the overmold's rear
# (421.2 mm along the 0.94 m cord's STEP axis) and its shell's front (469.2).
JUNCTION_Z = {PLUG: 0.10865, RJ45: 0.11930, USB: 0.42120}
MATING_Z = {PLUG: 0.15240, RJ45: 0.15374, USB: 0.46920}
JUNCTION_Z_M, MATING_Z_M = JUNCTION_Z[PLUG], MATING_Z[PLUG]
SEAT_Z_M = 0.15138 + 0.00373       # the insulator face along the cord + the dielectric up the socket
# z_socket = SEAT_Z - z_cord, seated: the BNC's above; the RJ45's tip (153.74) on the jack's floor (6.18)
# the USB's: its shell's front 3.7 mm below the adapter's origin, home -- 10.35 mm in from the receptacle's face
# (6.65), where the receptacle's tongue meets the back of the plug's cavity (the two CADs fitted, 2026-11-12;
# the overmold's front then 1.65 mm off the face). The push home finds the stop itself.
SEAT_Z = {SOCKET: SEAT_Z_M, RJ45_SOCKET: 0.15374 + 0.00618, USB_SOCKET: 0.46920 - 0.00370}


def plug_part(cfg):
    """The plug the pick estimates: pose6d.plug (PLUG, the BNC's, by default; RJ45)."""
    c = dict(cfg.section('pose6d') or {}) if cfg.get('pose6d') else {}
    part = str(c.get('plug', PLUG))
    if part not in PLUGS:
        raise Pose6DError(f'pose6d.plug {part!r}: one of {", ".join(PLUGS)}')
    return part


def socket_part(cfg):
    """The socket the target is estimated from: the one pose6d.plug goes into."""
    return SOCKET_FOR[plug_part(cfg)]


def T_cord_junction(part=PLUG):
    """The junction frame in the plug's CAD frame: on the axis where the cable leaves the strain
    relief, x along +z (toward the mating face). The BNC's: y along the cord's y (its turn is
    a choice anyway: SYMMETRY). The RJ45's: z along the latch side (+y), y = z x x along +x. The USB's, as the
    RJ45's: z along its model's +y -- the wide face AWAY from the cavity the receptacle's tongue enters."""
    T = np.eye(4)
    if SYMMETRY[part] == 'none':
        T[:3, :3] = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    else:
        T[:3, :3] = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    T[2, 3] = JUNCTION_Z[part]
    return T


def T_socket_cord_seated(socket=SOCKET):
    """The plug's CAD frame in the socket's, the plug seated: x along x, y and z reversed (an
    RJ45's latch, +y, toward the jack's latch slot, -y; a USB plug's cavity, -y, over the receptacle's tongue, +y)."""
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[2, 3] = SEAT_Z[socket]
    return T


def seated_junction_in_socket(socket=SOCKET):
    return T_socket_cord_seated(socket) @ T_cord_junction(PLUG_FOR[socket])


def junction_from_plug(T_base_plug, up=(0.0, 0.0, 1.0), part=PLUG):
    """The pick's connector (junction) frame from the plug's CAD pose: on the axis where the cable
    leaves the strain relief, x toward the mating face. A round plug's turn about its axis is a
    choice, made as every scanner makes it: z as near `up` as the axis allows. One with no
    symmetry (the RJ45) keeps the turn measured: z along its latch side."""
    from ..transforms import frame_from_axis
    T = np.asarray(T_base_plug, dtype=float) @ T_cord_junction(part)
    if SYMMETRY[part] != 'none':
        T[:3, :3] = frame_from_axis(T[:3, 0], up)
    return T


def grasp_junction(T_junction, part=PLUG, square_deg=90.0, up=(0.0, 0.0, 1.0)):
    """(T_grasp, T_turn) for a pick from above: `T_grasp` the junction frame `T_junction` (as
    junction_from_plug gives it) with its z as near `up` as the axis allows -- the frame every
    grasp is written in (pickup.fingertip_in_connector) -- and `T_turn` the turn about its own x
    from there to the plug as the jaws will HOLD it, so the held junction is T_grasp @ T_turn.

    A round plug: the identity (its turn is a convention, already z up). One with no symmetry (the
    RJ45, z along its latch side): lying latch DOWN its measured frame points z at the bench, and a
    grasp written in it would come from under the bench; so the grasp is written in the levelled
    frame and the latch's turn is carried in the belief instead. `square_deg`: flat pads square a
    flat-sided bite to themselves, so the turn the plug ends at is the measured one rounded to this
    (90: latch up, down or to either side); None or 0 keeps it as measured (a round bite)."""
    from ..transforms import frame_from_axis, xyzrpy_to_matrix
    T = np.asarray(T_junction, dtype=float)
    G = T.copy()
    G[:3, :3] = frame_from_axis(T[:3, 0], up)
    if SYMMETRY[part] != 'none':
        return G, np.eye(4)
    R = G[:3, :3].T @ T[:3, :3]                      # a rotation about x: the two share their axis
    phi = float(np.arctan2(R[2, 1], R[1, 1]))
    if square_deg:
        step = np.radians(float(square_deg))
        phi = step * round(phi / step)
    return G, xyzrpy_to_matrix([0.0, 0.0, 0.0], [phi, 0.0, 0.0])


def turn_deg(T_turn):
    """The turn about x of grasp_junction's T_turn, deg in (-180, 180]."""
    T = np.asarray(T_turn, dtype=float)
    return float(np.degrees(np.arctan2(T[2, 1], T[1, 1])))


def target_from_socket(T_base_socket, T_junction_connector, up=(0.0, 0.0, 1.0), socket=SOCKET):
    """The assembly target -- where the plug's CONNECTOR frame sits when seated -- from the
    socket's CAD pose and the part's junction -> connector transform (the one the in-hand belief
    is built with, so target and belief describe the same datum).

    The seated junction frame's turn about the insertion axis is not measured (the estimate's is
    arbitrary: SYMMETRY), so it is set as the pick sets the connector's: z as near `up` as the
    axis allows. The arm then turns the held connector as little as it can to meet it, and the
    bayonet's clocking is found as it always is, by the clocking search. A socket with no symmetry
    (the RJ45's) has its turn measured: the target keeps it -- the plug goes in latch to slot."""
    from ..transforms import frame_from_axis
    J = np.asarray(T_base_socket, dtype=float) @ seated_junction_in_socket(socket)
    if SYMMETRY[socket] != 'none':
        J[:3, :3] = frame_from_axis(J[:3, 0], up)
    return J @ np.asarray(T_junction_connector, dtype=float)


def belief_from_plug(T_ftip_plug, T_junction_connector, T_ftip_conn_belief, part=PLUG):
    """The in-hand belief -- the connector frame in the fingertip's -- from the plug's pose measured
    in the fingertip's frame: its axis and position as measured; its turn about the axis, which a
    view cannot tell of a round plug (SYMMETRY), kept from `T_ftip_conn_belief` -- measured too for
    one with no symmetry."""
    from ..transforms import frame_from_axis, inverse
    J = np.asarray(T_ftip_plug, dtype=float) @ T_cord_junction(part)
    if SYMMETRY[part] != 'none':
        J_belief = np.asarray(T_ftip_conn_belief, dtype=float) @ inverse(T_junction_connector)
        J[:3, :3] = frame_from_axis(J[:3, 0], J_belief[:3, 2])
    return J @ np.asarray(T_junction_connector, dtype=float)


def plug_from_belief(T_ftip_conn, T_junction_connector, part=PLUG):
    """The plug's CAD pose in the fingertip's frame that the in-hand belief describes."""
    from ..transforms import inverse
    return (np.asarray(T_ftip_conn, dtype=float) @ inverse(T_junction_connector)
            @ inverse(T_cord_junction(part)))


def robust_pose(Ts, part, agree_m, agree_rad):
    """(the medoid of the poses `Ts`, how many agree with it) -- agreeing: the part's middle within
    `agree_m` and its orientation within `agree_rad` (only its axis's, for a round part: SYMMETRY)."""
    Ts = [np.asarray(T, dtype=float) for T in Ts]
    c = np.array([T[:3, :3] @ CENTRE_M[part] + T[:3, 3] for T in Ts])
    dc = np.linalg.norm(c[:, None] - c[None], axis=2)
    if SYMMETRY[part] == 'none':
        R = np.array([T[:3, :3] for T in Ts])
        tr = np.einsum('iab,jab->ij', R, R)                    # trace(R_i^T R_j)
        da = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    else:
        a = np.array([T[:3, 2] for T in Ts])
        da = np.arccos(np.clip(a @ a.T, -1.0, 1.0))
    k = int(np.argmin((dc / agree_m + da / agree_rad).sum(axis=1)))
    return Ts[k], int(((dc[k] <= agree_m) & (da[k] <= agree_rad)).sum())


class Pose6DError(RuntimeError):
    pass


class Pose6DService:
    """The estimator process (pose6d_server.py in the rag6dpose env), and requests to it."""

    def __init__(self, cfg):
        c = dict(cfg.section('pose6d') or {}) if cfg.get('pose6d') else {}
        self.python = os.path.expanduser(str(c.get('python', '~/miniconda3/envs/rag6dpose/bin/python')))
        self.repo = os.path.expanduser(str(c.get('repo', '~/abhay_ws/RAG-6DPose-code-abhay')))
        self.timeout = float(c.get('timeout_s', 300.0))
        self.server = os.path.expanduser(str(c.get('server') or SERVER))       # a stand-in, in tests
        self.device = str(c.get('device', 'cuda'))
        # The robot's own gripper in the wrist view (perception/gripper_mask.py): sent with each view of a
        # part not in the hand, for the server to hide from its detector. None: nothing hidden.
        self.gripper_mask = c.get('gripper_mask')
        self._masks = {}
        self.request = {k: c[k] for k in ('min_iou', 'min_edge', 'min_cover', 'max_candidates', 'refine_top',
                                          'consistent_deg', 'consistent_mm', 'joint_iters', 'size_tol') if k in c}
        # Loaded (and run once) before it reports ready: the plug pose6d.plug names, and its socket
        # unless the target is the recorded mate (assembly.target_source kinematic) -- or the parts
        # pose6d.preload lists.
        pre = c.get('preload', 'auto')
        if str(pre).lower() == 'auto':
            kinematic = str(cfg.get_path('assembly.target_source') or '').strip().lower() == 'kinematic'
            self.preload = [plug_part(cfg)] + ([] if kinematic else [socket_part(cfg)])
        else:
            self.preload = list(pre or [])
        if not os.path.isfile(self.python):
            raise Pose6DError(f'no Python at {self.python} (pose6d.python: the rag6dpose env)')
        if not os.path.isdir(os.path.join(self.repo, 'pose_pipeline')):
            raise Pose6DError(f'no pose_pipeline in {self.repo} (pose6d.repo)')
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'VIRTUAL_ENV')}
        env['PYTHONNOUSERSITE'] = '1'                 # ROS's PYTHONPATH and ~/.local stay out of it
        self._dir = tempfile.mkdtemp(prefix='pose6d_', dir='/dev/shm' if os.path.isdir('/dev/shm') else None)
        self._proc = subprocess.Popen([self.python, '-u', self.server, '--repo', self.repo, '--device', self.device,
                                       '--preload', ','.join(OBJECTS[p] for p in self.preload)],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self._lock, self._n, self._models = threading.Lock(), 0, {}
        threading.Thread(target=self._relay, name='pose6d-log', daemon=True).start()
        log.info('Starting the 6D pose estimator (pid %d, %s) -- its models load in ~10 s%s.', self._proc.pid, self.repo,
                 f", then the {' and '.join(self.preload)} models" if self.preload else '')
        ready = self._read(self.timeout)
        if not ready.get('ready'):
            self.close()
            raise Pose6DError(f"the estimator did not start: {ready.get('error', ready)}")
        self.ready = ready
        log.info('6D pose estimator ready: %s.', ', '.join(ready.get('objects', [])))

    def _relay(self):
        for line in self._proc.stderr:
            log.info('  %s', line.rstrip())

    def _read(self, timeout):
        out = {}

        def read():
            out['line'] = self._proc.stdout.readline()

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise Pose6DError(f'the estimator did not answer in {timeout:.0f} s')
        if not out.get('line'):
            raise Pose6DError(f'the estimator exited (code {self._proc.poll()})')
        return json.loads(out['line'])

    def _ask(self, req):
        """Send one request, read its reply (the caller holds the lock)."""
        self._n += 1
        req = dict(req, id=self._n)
        self._proc.stdin.write(json.dumps(req) + '\n')
        self._proc.stdin.flush()
        return self._read(self.timeout)

    def estimate(self, part, views, prior=None, **overrides):
        """One estimate of `part` (PLUG, SOCKET or RJ45) from `views` -- [(Frame, ...)] with depth and
        T_base_cam -- given an optional `prior` T_base_obj. `overrides` replace the config's
        request keys for this one (the in-hand gate, use_depth). `hide_gripper` (default true): send the
        open gripper's pixels (pose6d.gripper_mask) to be hidden -- false for a part IN the hand, where
        the fingers are not where the open-gripper mask has them. Returns the service's reply, with
        'T_base_obj' (and each view's) as arrays."""
        hide = bool(overrides.pop('hide_gripper', True)) and bool(self.gripper_mask)
        with self._lock:
            paths, sent = [], []
            for k, frame in enumerate(views):
                path, view = self._pack(frame, f'r{self._n + 1}_v{k}', hide)
                paths.append(path)
                sent.append(view)
            req = dict(self.request, **overrides, op='estimate', object=OBJECTS[part], symmetry=SYMMETRY[part],
                       views=sent, prior=None if prior is None else np.asarray(prior, float).tolist())
            try:
                out = self._ask(req)
            finally:
                for p in paths:
                    if os.path.exists(p):
                        os.remove(p)
        return self._reply(out)

    def _pack(self, frame, stem, hide):
        """(npz path, the view as a request carries it): the frame written for the server."""
        if frame.depth is None or frame.T_base_cam is None:
            raise Pose6DError('a view without depth or a camera pose (camera.enable_depth?)')
        path = os.path.join(self._dir, f'{stem}.npz')
        extra = {}
        if hide:
            shape = tuple(np.asarray(frame.color).shape[:2])
            if shape not in self._masks:
                from . import gripper_mask
                self._masks[shape] = gripper_mask.load(self.gripper_mask, shape)
                if self._masks[shape] is None:
                    log.warning('6D pose: no gripper mask at %s (pose6d.gripper_mask) -- nothing hidden.',
                                self.gripper_mask)
            if self._masks[shape] is not None:
                extra['robot_mask'] = self._masks[shape]
        np.savez(path, color=np.ascontiguousarray(frame.color),
                 depth_mm=(np.asarray(frame.depth, dtype=np.float32) * 1000.0), K=np.asarray(frame.K, float), **extra)
        return path, {'npz': path, 'T_base_cam': np.asarray(frame.T_base_cam, float).tolist()}

    def session(self, part, prior=None, **overrides):
        """An estimate whose views are sent AS THEY ARE TAKEN (the user, 2026-11-04: the orbit's moves and the
        estimator's work overlapped): `.add(frame)` returns at once -- the view is refined on the server
        while the arm drives to the next -- and `.finish()` waits for them and returns what estimate()
        would have from all of them."""
        return _Session(self, part, prior, overrides)

    def _reply(self, out):
        if out.get('error'):
            raise Pose6DError(out['error'])
        # RIGID, each of them (transforms.rigid): the service's poses are float32 through JSON -- rotations to ~1e-6
        # only -- and the socket's anchors the whole run (the target, the stand-off, the trajectory)
        from ..transforms import rigid
        if out.get('T_base_obj') is not None:
            out['T_base_obj'] = rigid(out['T_base_obj'])
        for v in out.get('views', []):
            if v.get('T_base_obj') is not None:
                v['T_base_obj'] = rigid(v['T_base_obj'])
        return out

    def model_points(self, part, n=800):
        """(n, 3) points on `part`'s CAD surface, its frame, m -- for drawing it. Asked once."""
        if part not in self._models:
            with self._lock:
                out = self._ask({'op': 'model', 'object': OBJECTS[part], 'n': int(n)})
            if out.get('error'):
                raise Pose6DError(out['error'])
            self._models[part] = np.asarray(out['points'], float).reshape(-1, 3)
        return self._models[part]

    def close(self):
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({'op': 'quit'}) + '\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                self._proc.kill()
        shutil.rmtree(self._dir, ignore_errors=True)



class _Session:
    """Service.session(): the views of one estimate sent one by one, from a worker thread, as they come."""

    def __init__(self, svc, part, prior, overrides):
        import concurrent.futures
        import uuid
        self.svc, self.part = svc, part
        self.hide = bool(overrides.pop('hide_gripper', True)) and bool(svc.gripper_mask)
        self.req = dict(svc.request, **overrides, object=OBJECTS[part], symmetry=SYMMETRY[part],
                        prior=None if prior is None else np.asarray(prior, float).tolist())
        self.id = uuid.uuid4().hex[:12]
        self.k = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='pose6d-view')
        self.futures = []

    def _send(self, frame, k):
        with self.svc._lock:
            path, view = self.svc._pack(frame, f's{self.id}_v{k}', self.hide)
            try:
                out = self.svc._ask(dict(self.req, op='session_view', session=self.id, view=view))
            finally:
                if os.path.exists(path):
                    os.remove(path)
        if out.get('error'):
            raise Pose6DError(out['error'])
        return out.get('view', {})

    def add(self, frame):
        """Queue one view (returns at once; the views keep their order)."""
        self.futures.append(self.pool.submit(self._send, frame, self.k))
        self.k += 1
        return self.futures[-1]

    def finish(self):
        """Every view in, then the joint pose: the reply estimate() gives."""
        try:
            for f in self.futures:
                f.result()
            with self.svc._lock:
                out = self.svc._ask({'op': 'session_joint', 'session': self.id})
        finally:
            self.pool.shutdown(wait=False)
        return self.svc._reply(out)

_SERVICE = None
_STARTING = threading.Lock()             # one start, however many threads ask during it


def service(cfg):
    """The run's one estimator process, started on first use (or by preload)."""
    global _SERVICE
    with _STARTING:
        if _SERVICE is None:
            _SERVICE = Pose6DService(cfg)
            atexit.register(shutdown)
        return _SERVICE


_PRELOAD = None


def preload(cfg):
    """Start the estimator NOW, in the background, its parts' models loaded and run once and their
    points fetched -- so it is ready by the time the run first needs it (it takes ~20 s), not
    started then. A failure here is only logged: the first use starts it, and says why not. A preload
    already under way is returned, not started twice."""
    global _PRELOAD
    if _PRELOAD is not None and _PRELOAD.is_alive():
        return _PRELOAD

    def run():
        try:
            svc = service(cfg)
            for part in svc.preload:
                svc.model_points(part)
        except Exception as exc:          # noqa: BLE001 -- the first use tries again
            log.warning('6D pose estimator: preloading failed (%s); it starts on first use instead.', exc)

    t = threading.Thread(target=run, name='pose6d-preload', daemon=True)
    t.start()
    _PRELOAD = t
    return t


def wait_ready(timeout_s):
    """Wait up to `timeout_s` for a preload under way to finish. True when none is running (any more)."""
    t = _PRELOAD
    if t is None:
        return True
    t.join(max(0.0, float(timeout_s)))
    return not t.is_alive()


def shutdown():
    global _SERVICE
    if _SERVICE is not None:
        _SERVICE.close()
        _SERVICE = None


def cad_mesh(cfg, part, name=None):
    """(path, scale to m) of `part`'s CAD model, whole (DRAWN; else the very mesh the estimator fits) --
    for modelling it where no estimate is wanted (the plug in the gripper: robot/held_part.py) and
    drawing it (the live view); `name`: another of objects.yaml's models (MOUNTS: what it is mounted in). Read from the repo's
    objects.yaml (pose6d.repo); the models are in mm. None, said why, if there is none to be had."""
    import yaml
    c = dict(cfg.section('pose6d') or {}) if cfg.get('pose6d') else {}
    repo = os.path.expanduser(str(c.get('repo', '~/abhay_ws/RAG-6DPose-code-abhay')))
    try:
        with open(os.path.join(repo, 'pose_pipeline', 'objects.yaml')) as f:
            objects = yaml.safe_load(f)
        name = name or DRAWN.get(part, OBJECTS[part])
        path = os.path.join(repo, objects['models_dir'], f"obj_{int(objects['objects'][name]['obj_id']):06d}.ply")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        log.warning('no CAD model for %s in %s (%s).', part, repo, exc)
        return None
    if not os.path.isfile(path):
        log.warning('no CAD model for %s: %s is missing.', part, path)
        return None
    return path, 0.001
