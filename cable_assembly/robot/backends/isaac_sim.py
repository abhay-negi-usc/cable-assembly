"""IsaacSimArm -- the iiwa 14 BNC cell simulated in Isaac Sim, behind the same Arm interface as the
hardware, so the bench's own apps (bnc_assembly, ...) run on it unchanged.

The simulation runs in its own process and Python environment (sim/, Isaac Lab 3.0):

    cd sim && ./run.sh python scripts/bnc_cell_server.py     # the cell; --viz kit to watch it

and this backend connects to it on localhost (robot.isaac_sim in the cell config).

LOCKSTEP, AND WHY. The simulated cell runs tens of times slower than real time (0.1 mm signed-
distance contacts on a 13-joint arm). A real-time client would then send its 125 Hz admittance
setpoints into a robot that had barely moved between them, and every timeout and force-guard
persistence would expire in a fraction of the simulated time it was tuned for. So nothing here
waits on the wall clock: each servo setpoint advances the simulation by exactly its period, a
move advances it setpoint by setpoint, a settle or a sleep advances it by its duration, and
`clock()` is the simulated time -- which is what the guards and the apps' timeouts read
(Arm.clock). The bench's control loops therefore run in the time they were written for.

WHAT DIFFERS FROM THE iiwa OVER FRI:
  - moves: the same plans as IiwaFRIArm -- a phase-synchronised joint profile under the speed caps
    for move_j, IK every millimetre along the line with the same time scaling for move_l --
    streamed at the servo rate with the guards polled every cycle. No jerk limit (ruckig).
  - the wrench: the cell's wrist F/T (the gripper mount's joint wrench), low-passed like
    IiwaFRIArm's external torques, tared here. The real iiwa ESTIMATES it from joint torques; this
    is that estimate without the model error.
  - the gripper: IsaacSimGripper answers the Robotiq's registers with the simulated fingers.
  - THE SIMULATOR KNOWS THE TRUTH: `truth()` returns the plug's pose in the socket, the collar's
    angle and its spring -- what a run can be checked against, and never what an app plans from.
"""

import csv
import os
from datetime import datetime
from multiprocessing.connection import Client

import numpy as np

from ... import log as urlog
from ...transforms import pose_error, slerp_matrix
from ..arm import Arm, ArmError
from ..gripper import OBJ_MOVING, Robotiq2F85
from .iiwa_fri import _LINE_MAX_JOINT_STEP, _LINE_STEP_M, _LINE_STEP_RAD, _TimeScaling

log = urlog.get('arm')


class IsaacSimArm(Arm):
    """`model` (kinematics/iiwa14.py) in the simulated cell; cfg robot.isaac_sim configures it."""

    supports_force_mode = False

    def __init__(self, cfg, model, frames=None):
        super().__init__(cfg, model)
        c = cfg.get_path('robot.isaac_sim', {}) or {}
        self.stream_rate = float(c.get('stream_rate_hz', 125.0))
        self.filter_hz = float(c.get('wrench_filter_hz', 10.0))
        self.rest_speed = float(np.radians(c.get('rest_speed_deg_s', 0.5)))
        margin = np.radians(float(c.get('joint_limit_margin_deg', 2.0)))
        self.joint_limits = np.array(model.JOINT_LIMITS, dtype=float) + np.array([margin, -margin])
        host, port = c.get('host', 'localhost'), int(c.get('port', 6007))
        try:
            self._conn = Client((host, port), authkey=str(c.get('authkey', 'cable-assembly-sim')).encode())
        except OSError as exc:
            raise ArmError(f'No Isaac Sim cell on {host}:{port} ({exc}). Start it first: '
                           f'cd sim && ./run.sh python scripts/bnc_cell_server.py') from exc
        self.info = self._call('info')
        self.dt_physics = float(self.info['dt'])
        self._tare = np.zeros(6)
        self._wrench_f = None
        self._state = None
        # THE TRUTH TRACE: what the plug really did, beside what the app believed -- the point of
        # running the methodology on a simulation. A row every truth_log_every simulated seconds.
        self._truth_every = float(c.get('truth_log_every_s', 0.04))
        self._truth_next = 0.0
        self._truth_csv = None
        path = c.get('truth_csv', os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                                               datetime.now().strftime('isaac_sim_truth_%Y%m%d_%H%M%S.csv')))
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._truth_file = open(path, 'w', newline='')
            self._truth_csv = csv.writer(self._truth_file)
            self._truth_csv.writerow(['t_s', 'phase', 'plug_height_mm', 'plug_lateral_mm', 'plug_tilt_deg',
                                      'collar_deg', 'body_yaw_deg', 'collar_in_socket_deg', 'collar_slide_mm',
                                      'plug_in_tool_z_mm', 'gripper_counts', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
            log.info('Simulation truth trace: %s', path)
        self._phase = 'start'
        self.cfg_grasp_depth_mm = float(c.get('grasp_depth_mm', 178.5))
        if c.get('reset', True):
            q0 = np.radians(np.asarray(c.get('start_joints_deg', np.degrees(model.HOME)), dtype=float))
            st = self._call('reset', q=q0.tolist(), collar_deg=float(c.get('collar_deg', 0.0)),
                            grasp_depth_m=float(c.get('grasp_depth_mm', 178.5)) / 1000.0)
            log.info('SIMULATED CELL reset: arm at %s deg, the plug held at its strain relief, '
                     'collar at %.0f deg.', np.round(np.degrees(q0), 1).tolist(), c.get('collar_deg', 0.0))
        else:
            st = self._call('state')
        self._absorb(st)
        self._q_cmd = self._q.copy()
        log.warning('SIMULATED CELL (Isaac Sim, lockstep): every timing below is simulated time.')
        if frames is not None:
            self.publish_frames(frames, cfg)

    # ------------------------------------------------------------------ the link
    def _call(self, cmd, **kwargs):
        try:
            self._conn.send((cmd, kwargs))
            reply = self._conn.recv()
        except (EOFError, OSError) as exc:
            raise ArmError(f'the Isaac Sim cell went away during {cmd!r} ({exc})') from exc
        if not reply.get('ok'):
            raise ArmError(f'Isaac Sim cell: {cmd} failed: {reply.get("error")}')
        return reply

    def _absorb(self, st):
        """Take a state reply: joints, time, and the wrench through the low-pass."""
        dt = st['t'] - self._state['t'] if self._state is not None else 0.0
        self._state = st
        self._q = np.asarray(st['q'], dtype=float)
        w = np.asarray(st['wrench'], dtype=float)
        if self._wrench_f is None or self.filter_hz <= 0.0:
            self._wrench_f = w
        elif dt > 0.0:
            a = 1.0 - np.exp(-2.0 * np.pi * self.filter_hz * dt)
            self._wrench_f = self._wrench_f + a * (w - self._wrench_f)
        if self._truth_csv is not None and st['t'] >= self._truth_next:
            tr = st['truth']
            self._truth_csv.writerow([f"{st['t']:.4f}", self._phase] + [
                f'{v:.3f}' for v in (tr['height_mm'], tr['lateral_mm'], tr['tilt_deg'], tr['collar_deg'],
                                     tr['body_yaw_deg'], tr['collar_abs_deg'], tr['slide_mm'],
                                     tr['plug_in_tool_mm'][2], st['gripper']['counts'], *self._wrench_f)])
            self._truth_next = st['t'] + self._truth_every
        return st

    def _say_truth(self, where):
        tr = self._state['truth']
        log.info('SIM TRUTH [%s] t=%.2f s: plug %.2f mm above home, %.2f mm off the axis, tilt %.2f deg; '
                 'collar %+.1f deg in the socket (locked at -96 or +84), its spring %.2f mm; in the hand at '
                 '%.2f mm (placed %.1f).', where, self._state['t'], tr['height_mm'], tr['lateral_mm'],
                 tr['tilt_deg'], tr['collar_abs_deg'], tr['slide_mm'], tr['plug_in_tool_mm'][2],
                 float(self.cfg_grasp_depth_mm))

    def _advance(self, seconds, q=None):
        """Track joint target `q` (default: hold the last command) for `seconds` of simulated time."""
        if q is not None:
            self._q_cmd = np.asarray(q, dtype=float)
        return self._absorb(self._call('step', q=self._q_cmd.tolist(), seconds=float(seconds)))

    def set_speed_scale(self, scale, phase=''):
        """As Arm's -- and, since every phase of an app announces itself here, say where the plug
        really is at each one."""
        super().set_speed_scale(scale, phase)
        if phase:
            self._say_truth(phase)
            self._phase = phase
            self._call('note', text=phase)

    # ------------------------------------------------------------------ time
    def clock(self):
        return float(self._state['t'])

    def sleep(self, seconds):
        """Hold the last command while `seconds` of simulated time pass, in servo-period steps so
        the wrench filter sees them."""
        remaining, dt = float(seconds), 1.0 / self.stream_rate
        while remaining > 1e-9:
            self._advance(min(dt, remaining))
            remaining -= dt

    # ------------------------------------------------------------------ state
    def q(self):
        return list(self._q)

    def tcp_pose(self):
        return self.model.fk(self._q)

    def fk(self, q):
        return self.model.fk(np.asarray(q, dtype=float))

    def ik(self, T_base_tool0, qnear=None):
        seed = self._q if qnear is None else np.asarray(qnear, dtype=float)
        q = self.model.ik(T_base_tool0, seed, self.joint_limits)
        if q is None:
            log.error('No IK solution (unreachable, or no arm angle keeps the joints inside the '
                      'limits less the margin).')
            return None
        return list(q)

    def joints_ok(self, q):
        q = np.asarray(q, dtype=float)
        return q.shape == (self.dof,) and self.model.within(q, self.joint_limits)

    def truth(self):
        """The simulator's ground truth about the plug (see the module docstring)."""
        return dict(self._state['truth'])

    # ------------------------------------------------------------------ motion
    def _stream_path(self, sample, duration, label):
        """Stream q = sample(t) for t in [0, duration] at the servo rate, polling the guards every
        setpoint. False (holding where the arm is) if one trips."""
        dt = 1.0 / self.stream_rate
        n = max(1, int(np.ceil(duration / dt)))
        for k in range(1, n + 1):
            self._advance(dt, sample(min(k * dt, duration)))
            if self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                self.stop()
                return False
        return True

    def _await_rest(self, timeout=1.0):
        """Let the arm come to rest on its last command (joint speeds under rest_speed)."""
        t_end = self.clock() + timeout
        prev = self._q.copy()
        while self.clock() < t_end:
            self._advance(1.0 / self.stream_rate)
            if np.max(np.abs(self._q - prev)) * self.stream_rate < self.rest_speed:
                return True
            prev = self._q.copy()
        return False

    def move_j(self, q_target, speed=None, accel=None, label='move', caps=None):
        """Joint-space move: phase-synchronised (a straight line in joint space), the leading joint
        at `speed` under a trapezoidal profile, none above its own cap."""
        q_target = np.asarray(q_target, dtype=float)
        if not self.joints_ok(q_target):
            log.error('[%s] move rejected: target outside the iiwa joint limits less the margin '
                      '(deg %s).', label, np.round(np.degrees(q_target), 1).tolist())
            return False
        auto_speed, auto_accel, which = self._speeds(q_target, caps)
        speed = auto_speed if speed is None else float(speed)
        accel = auto_accel if accel is None else float(accel)
        q0 = self._q_cmd.copy()
        d = float(np.max(np.abs(q_target - q0)))
        log.info('[%s] move_j %.2f rad/s (%.0f deg/s leading joint, %s cap), accel %.2f rad/s^2',
                 label, speed, np.degrees(speed), which, accel)
        if d < 1e-9:
            return True
        scaling = _TimeScaling(d / max(speed, 1e-6), speed / max(accel, 1e-6))
        ok = self._stream_path(lambda t: q0 + (q_target - q0) * (scaling(t) / scaling.cruise),
                               scaling.duration, label)
        return ok and (self._await_rest() or True)

    def move_l(self, T_base_tool0, speed=None, accel=None, label='move', caps=None):
        """Straight-line move of tool0: IK every millimetre, time-scaled so the tool stays under the
        Cartesian caps and every joint under its own (IiwaFRIArm's plan), then streamed."""
        jv, _, cv, cr = self._limits(caps)
        cv = max(float(speed) if speed is not None else cv, 1e-4)
        accel = self.cart_accel if accel is None else float(accel)
        q0 = self._q_cmd.copy()
        T0, T1 = self.model.fk(q0), np.asarray(T_base_tool0, dtype=float)
        lin, ang = pose_error(T0, T1)
        n = max(1, int(np.ceil(lin / _LINE_STEP_M)), int(np.ceil(ang / _LINE_STEP_RAD)))
        path, q = [q0], q0
        for k in range(1, n + 1):
            q = self.model.ik(slerp_matrix(T0, T1, k / n), q, self.joint_limits)
            if q is None:
                log.error('[%s] linear move rejected: the line leaves the reachable workspace or '
                          'the joint limits at %.0f%%.', label, 100.0 * k / n)
                return False
            jump = float(np.max(np.abs(q - path[-1])))
            if jump > _LINE_MAX_JOINT_STEP:
                log.error('[%s] linear move rejected: a %.1f deg joint jump at %.0f%% -- the line '
                          'passes a singularity.', label, np.degrees(jump), 100.0 * k / n)
                return False
            path.append(q)
        path = np.array(path)
        dq = np.abs(np.diff(path, axis=0))
        h = np.max(dq / jv, axis=1)
        h = np.maximum(h, lin / n / cv)
        if cr > 0.0:
            h = np.maximum(h, ang / n / cr)
        t_way = np.concatenate([[0.0], np.cumsum(h)])
        scaling = _TimeScaling(t_way[-1], cv / max(accel, 1e-6))
        log.info('[%s] move_l %.3f m/s, accel %.2f m/s^2: %.1f mm in %.2f s', label, cv, accel,
                 1000.0 * lin, scaling.duration)

        def sample(t):
            tau = scaling(t)
            i = int(np.clip(np.searchsorted(t_way, tau, side='right') - 1, 0, n - 1))
            w = (tau - t_way[i]) / max(t_way[i + 1] - t_way[i], 1e-12)
            return path[i] + (path[i + 1] - path[i]) * min(max(w, 0.0), 1.0)

        ok = self._stream_path(sample, scaling.duration, label)
        return ok and (self._await_rest() or True)

    def servo_j(self, q, dt, lookahead=0.1, gain=300):
        q = np.asarray(q, dtype=float)
        if not self.joints_ok(q):
            log.error('servo_j target outside the joint limits -- holding.')
            q = None
        self._advance(dt, q)

    def servo_l(self, T_base_tool0, dt, lookahead=0.1, gain=300):
        """One Cartesian setpoint, IK from the previous one so the stream stays on its branch."""
        q = self.model.ik(T_base_tool0, self._q_cmd, self.joint_limits)
        if q is None:
            log.error('servo_l target unreachable -- holding.')
        self._advance(dt, q)

    def servo_stop(self):
        """The arm settles on the last setpoint and holds it, as the real one does."""
        self._await_rest(timeout=0.5)

    def stop(self):
        """Hold where the arm IS (not where it was going)."""
        self._q_cmd = self._q.copy()
        self._advance(1.0 / self.stream_rate)

    def _settle(self):
        self.sleep(self.settle_s)

    # ------------------------------------------------------------------ force / torque
    def _raw_wrench(self):
        return np.array(self._wrench_f, dtype=float)

    def zero_ft(self, settle=True):
        """Tare like an F/T: optionally let the arm settle first, then average 0.1 s of readings."""
        if settle:
            self.sleep(0.3)
        samples = []
        for _ in range(max(1, int(0.1 * self.stream_rate))):
            self._advance(1.0 / self.stream_rate)
            samples.append(self._raw_wrench())
        self._tare = np.mean(samples, axis=0)
        return True

    def wrench(self):
        return self._raw_wrench() - self._tare

    # ------------------------------------------------------------------ the gripper
    def make_gripper(self, cfg):
        return IsaacSimGripper(cfg, self)

    # ------------------------------------------------------------------ lifecycle
    def disconnect(self):
        if self._state is not None:
            self._say_truth('end')
        if self._truth_csv is not None:
            self._truth_file.close()
            self._truth_csv = None
        try:
            self._conn.close()
        except OSError:
            pass


class IsaacSimGripper(Robotiq2F85):
    """The Robotiq's register interface answered by the simulated fingers. Every method of
    Robotiq2F85 above the registers (grasp checks, widths) runs unchanged; the waits run in
    simulated time instead of against the wall clock."""

    def __init__(self, cfg, arm):
        g = cfg.section('gripper')
        self.port = 'isaac_sim'
        self.open_counts = int(g.get('open_counts', 0))
        self.closed_counts = int(g.get('closed_counts', 255))
        self.speed = int(g.get('speed_counts', 255))
        self.force = int(g.get('force_counts', 150))
        self.timeout_s = float(g.get('timeout_s', 5.0))
        from ..gripper_kinematics import GROOVE_DEPTH_M
        self.groove_depth_m = (float(g['groove_depth_mm']) / 1000.0
                               if g.get('groove_depth_mm') is not None else GROOVE_DEPTH_M)
        self.dry_run = False
        self.client = None
        self.arm = arm
        self._req = 0

    def _write(self, action, position, speed, force):
        if action & 0x08:                                   # rGTO: go to the requested position
            self._req = int(np.clip(position, 0, 255))
            self.arm._absorb(self.arm._call('gripper', counts=self._req, speed=int(speed), force=int(force)))

    def _read(self):
        g = self.arm._state['gripper']
        return {'activated': True, 'status': 3, 'obj': int(g['obj']), 'fault': 0,
                'pos_req': self._req, 'pos': int(round(g['counts'])), 'current': 0}

    def activate(self):
        return True

    def clear_fault(self):
        return True

    def disconnect(self):
        pass

    def go_to(self, counts, label='', wait=True):
        """As Robotiq2F85.go_to -- a stall on an object is success -- in simulated time."""
        counts = int(np.clip(counts, 0, 255))
        log.info('--> gripper %s (%d counts)', label or 'move', counts)
        self._write(0x09, counts, self.speed, self.force)
        if not wait:
            return True
        t_end = self.arm.clock() + self.timeout_s
        self.arm.sleep(0.05)
        while self.arm.clock() < t_end:
            state = self._read()
            if state['obj'] != OBJ_MOVING:
                log.info('    settled at %d counts (obj=%d).', state['pos'], state['obj'])
                return True
            self.arm.sleep(0.02)
        log.error('Gripper did not settle within %.1f s (simulated).', self.timeout_s)
        return False
