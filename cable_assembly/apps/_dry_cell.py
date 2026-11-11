"""THE DRY CELL -- what a --dry-run stands in for, so a 6D-pose task runs END TO END with no camera, no part and
no contact (`dry_cell:` in the task config; on only in a dry run, and only where the config enables it --
ethernet_assembly.yaml does).

A dry run's camera returns black frames with no depth, so the 6D estimator cannot see the plug or the socket
(the 2026-11-06 11:26 run: "a view without depth" at the first socket view); its gripper shuts on nothing, so
every grasp is EMPTY; and its arm has no contact, so no engage ever meets a socket. Here:

  THE 6D ESTIMATES   known poses instead (the estimator is never started): the SOCKET from dry_cell.socket, else
                     its own last measurement (data/pose6d/last_socket.json), else the cell's recorded mate; the
                     PLUG lying where the reset pose puts a cable (bnc_assembly aligned_place_pose -- where the
                     coaxial grasp reaches), turned dry_cell.plug.latch_deg about its axis (180: latch down).
                     DryScanner gives the pick what Pose6DScanner gives it: the levelled junction and the turn.
  THE PART IN THE JAWS  a dry-run close stalls in the grasp-check band (Robotiq2F85.dry_run_object_counts).
  CONTACT            a VirtualJack on the sim arm's contact hook: an RJ45 jack at the target, its face
                     dry_cell.jack.depth_mm in front of the seat, clearance_mm round the plug, insertion_n of
                     resistance inside, a stiff floor at the seat, and a latch that catches within latch_mm of
                     it (dry_cell.latches false: never) and then holds the plug in. The TRUE plug is the belief
                     offset by dry_cell.in_hand_error (zero: exactly where believed). It acts on the tool only
                     while the gripper is closed. On the KUKA's impedance (robot.compliance_controller
                     kuka_impedance) the sim arm also DEFLECTS by force / stiffness (SimArm.deflect), as the
                     controller's spring would -- a rigid arm would follow every pull and fail every latch.

NOT PHYSICS: piecewise springs along and across the jack's axis, no friction cone, no rotation -- enough to run
the task's logic through contact, the push home, the latch tug and the recoveries, not to tune them.
"""

import json
import os

import numpy as np

from .. import log as urlog
from ..perception import pose6d as P6
from ..transforms import from_cfg, inverse, translation_matrix, xyzrpy_to_matrix

log = urlog.get('dry-cell')


def enabled(cfg):
    """A dry run, and dry_cell.enabled."""
    from ..robot.backends import simulated_perception
    return bool(simulated_perception(cfg) and (cfg.get_path('dry_cell.enabled', False)))


class DryScanner:
    """The pick's scanner in a dry run: the plug at `plug_pose_fn()` (its CONNECTOR frame lying flat, z up) turned
    latch_deg about its axis -- returned as Pose6DScanner returns it, the levelled junction with the turn kept."""

    pose_on_axis = True

    def __init__(self, cfg, T_junction_conn, plug_pose_fn=None):
        self.cfg, self.T_junction_conn, self.plug_pose_fn = cfg, np.asarray(T_junction_conn, dtype=float), plug_pose_fn
        self.part = P6.plug_part(cfg)
        self.latch_deg = float(cfg.get_path('dry_cell.plug.latch_deg', 0.0) or 0.0)
        self.turn = np.eye(4)
        self.scene, self.camera, self.detector, self.listeners = None, None, None, []
        self.estimator = type('NoEstimator', (), {'reset': staticmethod(lambda: None)})()
        self.scans = 0

    def reset(self):
        pass

    def reselect(self):
        pass

    def scan(self, confirm=None):
        if self.plug_pose_fn is None:
            log.error('DRY CELL: no plug pose to stand in for the scan.')
            return None
        T_conn = np.asarray(self.plug_pose_fn(), dtype=float)
        M = T_conn @ inverse(self.T_junction_conn) @ xyzrpy_to_matrix([0, 0, 0], [np.radians(self.latch_deg), 0, 0])
        sq = self.cfg.get_path('pose6d.grasp_square_deg', 90.0)
        T, self.turn = P6.grasp_junction(M, self.part, None if sq is None else float(sq))
        self.scans += 1
        log.info('DRY CELL: the %s stands in for the 6D scan at %s mm, heading %.1f deg, latch %+.0f deg (scan %d).',
                 self.part, np.round(T[:3, 3] * 1000.0, 1).tolist(), np.degrees(np.arctan2(T[1, 0], T[0, 0])),
                 self.latch_deg, self.scans)
        return T


def socket_pose(cfg, socket, targets, tname, T_junction_conn):
    """(T_base_socket, where it came from) for a dry run, or (None, why): dry_cell.socket, else the part's own last
    measurement, else the recorded mate (`tname`'s, else the first) as the seated plug's connector frame."""
    c = cfg.get_path('dry_cell.socket')
    if c:
        return from_cfg(c), 'dry_cell.socket'
    path = os.path.join(cfg.get('data_dir', 'data'), 'pose6d', 'last_socket.json')
    try:
        with open(path) as fh:
            last = json.load(fh)
        if last.get('part') == socket:
            return np.asarray(last['T_base_obj'], dtype=float), f'its last measurement ({last.get("time", "?")})'
    except (OSError, ValueError, KeyError):
        pass
    name = tname if tname in targets else next(iter(targets), None)
    if name is None:
        return None, 'no dry_cell.socket, no measurement of it and no recorded mate in the frames file'
    T = targets[name] @ inverse(T_junction_conn) @ inverse(P6.seated_junction_in_socket(socket))
    return T, f'the recorded mate {name!r}'


class VirtualJack:
    """The contact a dry run's arm feels (SimArm.contact): the held plug against a jack at the target. See the
    module docstring. `target(T)` with the target connector frame (x into the jack, 0 at the seat), `hold(T)` with
    the plug's datum on tool0 at each grasp; it pushes on the tool only while `gripper` is closed past `closed_counts`."""

    def __init__(self, cfg, gripper=None):
        j = dict(cfg.get_path('dry_cell.jack') or {})
        self.depth = float(j.get('depth_mm', 12.1)) / 1000.0
        self.clear = float(j.get('clearance_mm', 0.3)) / 1000.0
        self.entry = self.clear + float(j.get('chamfer_mm', 0.4)) / 1000.0
        self.f_ins = float(j.get('insertion_n', 4.0))
        self.k_floor = float(j.get('floor_n_per_mm', 20.0)) * 1000.0
        self.k_wall = float(j.get('wall_n_per_mm', 20.0)) * 1000.0
        self.k_side = float(j.get('side_n_per_mm', 5.0)) * 1000.0
        self.k_latch = float(j.get('latch_n_per_mm', 20.0)) * 1000.0
        self.latch_at = -float(j.get('latch_mm', 0.5)) / 1000.0
        self.latches = bool(cfg.get_path('dry_cell.latches', True))
        e = cfg.get_path('dry_cell.in_hand_error')
        self.T_err = from_cfg(e) if e else np.eye(4)
        self.gripper = gripper
        self.closed_counts = int(cfg.get_path('gripper.open_counts', 3) or 3) + 20
        self.T_target = self.T_tool0_plug = None
        self.entered = self.latched = False
        self.K = None                                 # the KUKA's stiffness when the arm deflects (deflect)

    # ---- the task's side
    def target(self, T_base_target):
        self.T_target = np.asarray(T_base_target, dtype=float)
        self.entered = self.latched = False

    def hold(self, T_tool0_conn_belief):
        """A new grasp: the true plug is the belief offset by dry_cell.in_hand_error; out of the jack, unlatched."""
        self.T_tool0_plug = np.asarray(T_tool0_conn_belief, dtype=float) @ self.T_err
        self.entered = self.latched = False

    def _active(self):
        if self.T_target is None or self.T_tool0_plug is None:
            return False
        if self.gripper is None:
            return True
        try:
            return int(self.gripper.position()) > self.closed_counts
        except Exception:                                # noqa: BLE001
            return True

    # ---- the law, in the target frame: p = the plug's datum, x into the jack (0 seated)
    def _update(self, p):
        x, lat = p[0], max(abs(p[1]), abs(p[2]))
        if x < -self.depth:
            self.entered = False                         # out of the jack (and out of its latch: pulled past it)
            self.latched = False
        elif not self.entered and lat <= self.entry:
            self.entered = True                          # through the opening, square to it
        if self.entered and self.latches and x >= self.latch_at:
            if not self.latched:
                log.info('DRY CELL: the latch CLICKS (the plug %.2f mm from the floor).', -1000.0 * x)
            self.latched = True

    def _axial(self, x):
        """The jack's push on the plug along its axis, + OUT of the jack."""
        if x < -self.depth:
            return 0.0
        if not self.entered:
            return self.k_wall * (x + self.depth)        # on the face, beside the opening
        f = self.k_floor * max(0.0, x)
        if self.latched:
            f -= self.k_latch * max(0.0, self.latch_at - x)   # the latch holds it in
        else:
            f += self.f_ins
        return f

    def _lateral(self, v):
        if not self.entered:
            return 0.0
        return self.k_side * np.sign(v) * max(0.0, abs(v) - self.clear)

    def _p(self, T_base_tool0):
        return (inverse(self.T_target) @ np.asarray(T_base_tool0, dtype=float) @ self.T_tool0_plug)[:3, 3]

    def __call__(self, arm):
        """The raw wrench [f; tau] on the tool in base_link, the moment about the flange (SimArm.contact)."""
        if not self._active():
            return np.zeros(6)
        T = arm.tcp_pose()
        p = self._p(T)
        self._update(p)
        f_t = -np.array([self._axial(p[0]), self._lateral(p[1]), self._lateral(p[2])])
        R = self.T_target[:3, :3]
        F = R @ f_t
        r = (T @ self.T_tool0_plug)[:3, 3] - T[:3, 3]
        return np.concatenate([F, np.cross(r, F)])

    def deflect(self, arm, T_cmd):
        """On the KUKA's impedance: the tool where the controller's spring (K, isotropic) and the jack balance --
        per axis, K (commanded - actual) = the jack's push at actual (solved by bisection; each law rises with
        depth). The commanded pose itself out of contact."""
        if not self._active() or self.K is None:
            return T_cmd
        pc = self._p(T_cmd)
        self._update(pc)
        laws = (self._axial, self._lateral, self._lateral)
        pt = pc.copy()
        for i in range(3):
            g = (lambda v, i=i: self.K * (pc[i] - v) - laws[i](v))
            if abs(laws[i](pc[i])) < 1e-9:
                continue
            lo, hi = pc[i] - 0.05, pc[i] + 0.05            # g(lo) > 0 > g(hi) for a law that rises with v
            if g(lo) < 0.0 or g(hi) > 0.0:
                continue
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if g(mid) > 0.0 else (lo, mid)
            pt[i] = 0.5 * (lo + hi)
        return translation_matrix(self.T_target[:3, :3] @ (pt - pc)) @ np.asarray(T_cmd, dtype=float)


class DryCell:
    """The run's stand-ins (see the module docstring), made by `setup` -- or None when the run is not a dry run
    with dry_cell.enabled."""

    def __init__(self, cfg, robot, T_junction_conn):
        self.cfg = cfg
        self.scanner = DryScanner(cfg, T_junction_conn)
        self.jack = VirtualJack(cfg, getattr(robot, 'gripper', None))
        arm = robot.arm
        if hasattr(arm, 'contact'):
            arm.contact = self.jack
        if str(getattr(arm, 'compliance', 'admittance')).lower() == 'kuka_impedance':
            from ..robot.admittance import KUKA_DEFAULT_STIFFNESS
            K = getattr(arm, 'kuka_stiffness', None) or cfg.get_path('robot.fri.cartesian_impedance.stiffness') \
                or KUKA_DEFAULT_STIFFNESS
            self.jack.K = min(float(v) for v in list(K)[:3])
            arm.deflect = self.jack.deflect
        band = cfg.get_path('grasp_check.connector_counts')
        g = getattr(robot, 'gripper', None)
        if band and g is not None:
            g.dry_run_object_counts = int(round(0.5 * (min(band) + max(band))))
        log.warning('DRY CELL: the 6D estimates, the part in the jaws and the contact are STOOD IN FOR -- the plug '
                    'and the socket at known poses (the estimator not started), a close stalls at %s counts, a '
                    'virtual jack %.1f mm deep at the target%s%s.',
                    None if g is None else getattr(g, 'dry_run_object_counts', None), 1000.0 * self.jack.depth,
                    '' if self.jack.latches else ', whose latch NEVER catches (dry_cell.latches false)',
                    ', the arm deflecting on the KUKA\'s %.0f N/m' % self.jack.K if self.jack.K else '')

    @classmethod
    def setup(cls, cfg, robot, T_junction_conn):
        return cls(cfg, robot, T_junction_conn) if enabled(cfg) else None
