"""IiwaFRIArm -- the KUKA LBR iiwa 14 over FRI (Fast Robot Interface), through pyFRI.

FRI is not a motion controller the way the UR's is. It takes ONE joint position per controller
cycle and nothing else: no moveJ, no moveL, no IK, no force mode, no tared F/T sensor. So this
backend supplies, client-side, what the UR controller does for URArm:

    IK                 kinematics/iiwa14.py (closed form, arm angle kept from the seed)
    moveJ              a phase-synchronised, jerk-limited joint trajectory (ruckig)
    moveL              IK every millimetre along the line, time-scaled under the Cartesian and
                       joint caps, streamed
    servoJ / servoL    a stream of setpoints the driver joins and tracks
    the wrench         pinv(J^T) . tau_ext from KUKA's external joint torques, tared here

THE CYCLE ITSELF RUNS IN ANOTHER PROCESS (fri_driver.py, which explains why). This class plans,
checks and waits; the driver turns each command into a smooth position per cycle and owns every
real-time guarantee. They share a small file in /dev/shm; neither ever waits on the other.

BEFORE IT CAN MOVE, on the smartPAD: an FRI application (the lab's lbr-stack LBRServer, or any
app that opens an FRI session) must be running in POSITION client command mode with its client
IP pointing at this PC. The constructor waits `robot.fri.connect_timeout_s` for that session to
reach COMMANDING_ACTIVE and explains what it is still waiting for.

IF THE SESSION LEAVES COMMANDING (e-stop, a safety stop, the app stopped on the pendant, the
link degrading), the move in progress returns False, and the driver holds the arm where it is
when the session comes back -- it never resumes the abandoned motion.

THE WRENCH is the contact wrench in base_link, moment about the flange (tool0), like every
backend's. KUKA's estimate carries model error, so it is tared like an F/T sensor -- and if the
Sunrise app does not carry the tool's load data, the tool's own weight shows up as contact force
(set robot.fri.compensate_payload with robot.payload, see the cell).
"""

import atexit
import json
import subprocess
import sys
import threading
import time

import numpy as np

from ... import log as urlog
from ...transforms import pose_error, slerp_matrix
from ..arm import Arm, ArmError
from ..kinematics import iiwa14
from . import fri_driver as drv

log = urlog.get('arm')

_G = 9.81
# move_l places an IK waypoint at least this often along the line.
_LINE_STEP_M = 0.001
_LINE_STEP_RAD = np.radians(0.5)
# A larger joint change between neighbouring waypoints means the line crosses a singularity (or
# the arm-angle branch flips); the move is refused rather than whipped through it.
_LINE_MAX_JOINT_STEP = np.radians(5.0)


def _sleep_until(t):
    dt = t - time.monotonic()
    if dt > 0.0:
        time.sleep(dt)


class _TimeScaling:
    """Real time -> progress along a path that takes `cruise` seconds at full speed, the speed
    ramping 0 -> 1 over `ramp` seconds at the start and back to 0 at the end (a lower, shorter
    ramp when the path is too short to reach full speed). The ramp is a half cosine, so the
    acceleration is continuous -- zero at either end of a ramp instead of stepping there -- and a
    move starts and stops without a knock."""

    def __init__(self, cruise, ramp):
        self.cruise = float(cruise)
        ramp = max(float(ramp), 1e-6)
        # A ramp covers peak * r / 2 whatever its shape, so a path too short for full speed has
        # the same peak and ramp time as a linear ramp's triangle would.
        if self.cruise >= ramp:
            self.peak, self.r = 1.0, ramp
            self.duration = self.cruise + ramp
        else:
            self.peak = np.sqrt(self.cruise / ramp)
            self.r = ramp * self.peak
            self.duration = 2.0 * self.r

    def _ramp(self, t):
        """Progress `t` seconds into a ramp: speed peak * (1 - cos(pi t / r)) / 2."""
        return 0.5 * self.peak * (t - self.r / np.pi * np.sin(np.pi * t / self.r))

    def __call__(self, t):
        if t <= 0.0:
            return 0.0
        if t >= self.duration:
            return self.cruise
        if t < self.r:
            return self._ramp(t)
        if t <= self.duration - self.r:
            return 0.5 * self.peak * self.r + self.peak * (t - self.r)
        return self.cruise - self._ramp(self.duration - t)


def wrench_from_torques(model, q, tau_ext, damping=0.005, payload=None):
    """[f; tau about the flange] in base_link from external joint torques -- the force the
    environment applies to the tool, untared.

    tau_ext = J^T F, with J the geometric Jacobian at tool0 ([v; w], v at the flange). A 7-joint
    arm has more torques than the wrench has numbers, so F is the least-squares fit, damped:
    F = (J J^T + l^2 I)^-1 J tau_ext. At the bench poses the damping costs ~3 mN; it only matters
    near a straight-arm singularity, where the undamped fit would explode.

    `payload` = (mass kg, CoG in the flange frame, m): the tool's own weight, subtracted when the
    controller's torque model does not already carry it."""
    q = np.asarray(q, dtype=float)
    J = model.jacobian(q)
    F = np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), J @ np.asarray(tau_ext, dtype=float))
    if payload is not None:
        mass, cog = payload
        f_g = np.array([0.0, 0.0, -mass * _G])              # the tool's weight, base_link
        r = model.fk(q)[:3, :3] @ np.asarray(cog, dtype=float)
        F = F - np.concatenate([f_g, np.cross(r, f_g)])
    return F


class IiwaFRIArm(Arm):
    """The iiwa 14 over FRI. Blocking; every move returns True/False."""

    supports_force_mode = False              # compliance is the AdmittanceController over servo_l

    def __init__(self, cfg, frames=None):
        super().__init__(cfg, iiwa14)
        r = cfg.section('robot')
        f = dict(r.get('fri') or {})

        margin = np.radians(float(f.get('joint_limit_margin_deg', 2.0)))
        self.joint_limits = self.model.JOINT_LIMITS + np.array([margin, -margin])
        self.v_cap = self.model.VELOCITY_LIMITS * float(f.get('max_velocity_fraction', 0.5))
        self.JOINT_VELOCITY_CEILING = float(np.max(self.v_cap))
        self.accel_ramp = float(f.get('accel_ramp_s', 0.1))
        self.stream_accel = float(f.get('stream_accel_rad_s2', 4.0))
        self.stream_accel_ramp = float(f.get('stream_accel_ramp_s', 0.02))
        self.stream_rate = float(f.get('stream_rate_hz', 125.0))
        self.line_lookahead = float(f.get('line_lookahead_s', 0.1))
        self.line_ramp = float(f.get('line_ramp_s', 0.25))
        self.wrench_damping = float(f.get('wrench_damping', 0.005))
        stop_accel = float(f.get('stop_accel_rad_s2', 4.0))
        stop_jerk = stop_accel / float(f.get('stop_accel_ramp_s', 0.05))

        payload = r.get('payload') or {}
        self._payload = None
        if f.get('compensate_payload', False) and payload.get('mass_kg'):
            self._payload = (float(payload['mass_kg']),
                             np.asarray(payload.get('cog_m', [0.0, 0.0, 0.0]), dtype=float))
            log.info('Subtracting the tool weight from the torque estimate: %.2f kg, CoG %s m '
                     '(flange frame).', *self._payload)

        self._tare = np.zeros(6)
        self._seq = 0
        self._lock = threading.Lock()
        self._stream_q = None                # the last streamed setpoint, while a stream runs
        self._fault_seen = 0

        port = int(f.get('port', 30200))
        options = {
            'port': port, 'host': f.get('host'), 'sdk': f.get('sdk', 'pyfri'),
            'sdk_options': f.get('sdk_options') or {},
            'rt_priority': int(f.get('rt_priority', 80)),
            'torque_filter_hz': float(f.get('torque_filter_hz', 10.0)),
            'q_min': list(self.joint_limits[:, 0]), 'q_max': list(self.joint_limits[:, 1]),
            'hard_min': list(self.model.JOINT_LIMITS[:, 0]),
            'hard_max': list(self.model.JOINT_LIMITS[:, 1]),
            'v_cap': list(self.v_cap), 'stop_accel': stop_accel, 'stop_jerk': stop_jerk,
        }
        self._chan = drv.Channel.create()
        # -P: the script's own directory (the backends package) must not shadow anything.
        self._proc = subprocess.Popen(
            [sys.executable, '-P', '-u', drv.__file__, '--channel', self._chan.path,
             '--options', json.dumps(options)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._relay_thread = threading.Thread(target=self._relay, name='fri-log', daemon=True)
        self._relay_thread.start()
        atexit.register(self.disconnect)
        log.info('FRI driver started (pid %d); joint speed caps %s deg/s (%.0f%% of rated).',
                 self._proc.pid, np.round(np.degrees(self.v_cap)).astype(int).tolist(),
                 100.0 * float(f.get('max_velocity_fraction', 0.5)))
        try:
            self._await_commanding(float(f.get('connect_timeout_s', 60.0)), port)
        except BaseException:
            self.disconnect()
            raise

        if frames is not None:
            self.publish_frames(frames, cfg)

    # ------------------------------------------------------------------ the driver
    def _relay(self):
        """The driver's lines, into this process's log."""
        for line in self._proc.stdout:
            log.info('[fri] %s', line.rstrip())

    def _state(self):
        s = self._chan.state.read() if self._chan.state is not None else None
        if s is None:
            raise ArmError('no state from the FRI driver')
        return s

    def _send(self, mode, epoch, q=None, v=None, a=None, j=None, dt=0.0, stamp=None):
        """Write one command. `stamp` (CLOCK_MONOTONIC) is when a stream setpoint is FOR -- the
        driver places it on the reference by this, not by when it happens to read it."""
        zero = np.zeros(self.dof)
        with self._lock:
            self._seq += 1
            self._chan.command.write(drv.COMMAND.pack(
                seq=self._seq, epoch=epoch, mode=mode, dt=dt,
                stamp=time.monotonic() if stamp is None else stamp,
                q=zero if q is None else q, v=zero if v is None else v,
                a=zero if a is None else a, j=zero if j is None else j))
            return self._seq

    def _await_commanding(self, timeout, port):
        hints = {
            None: 'starting the FRI driver',
            drv.NO_PACKETS: (f'no packet from the controller yet. On the smartPAD, start the FRI app '
                             f'(client command mode POSITION) with its client IP set to this PC; '
                             f'UDP port {port} must be open here'),
            drv.IDLE: 'the controller ended the FRI session',
            drv.MONITORING_WAIT: 'connected; FRI is monitoring and waiting for the link quality',
            drv.MONITORING_READY: 'connected; waiting for the app to start commanding',
            drv.COMMANDING_WAIT: 'commanding requested; synchronising',
        }
        deadline = time.monotonic() + timeout
        said = object()
        while True:
            if self._proc.poll() is not None:
                raise ArmError(f'the FRI driver exited (code {self._proc.returncode}) -- see the '
                               '[fri] lines above')
            s = self._chan.state.read()
            session = None if s is None else int(s.session)
            if s is not None and int(s.fault) == drv.FAULT_COMMAND_MODE:
                raise ArmError('The FRI app on the controller is not in POSITION client command '
                               'mode. Restart it with POSITION (the lbr-stack LBRServer asks on '
                               'the smartPAD).')
            if s is not None and int(s.fault) == drv.FAULT_CONNECT:
                raise ArmError(f'The FRI driver could not open UDP port {port}.')
            if session == drv.COMMANDING_ACTIVE:
                self._fault_seen = int(s.fault_count)
                log.info('FRI commanding: %.0f ms cycle, link %s, at q = %s deg.',
                         1000.0 * s.sample_time,
                         ('POOR', 'FAIR', 'GOOD', 'EXCELLENT')[int(s.quality)],
                         np.round(np.degrees(s.q), 1).tolist())
                return
            if session != said:
                log.info('FRI: %s.', hints.get(session, drv.SESSION_NAMES.get(session)))
                said = session
            if time.monotonic() > deadline:
                raise ArmError(f'No FRI commanding session after {timeout:.0f} s: '
                               f'{hints.get(session, session)}.')
            time.sleep(0.05)

    def _commanding(self, label):
        """The state, if the arm can be commanded now; else log why not and return None."""
        s = self._state()
        if self._proc.poll() is not None:
            log.error('[%s] the FRI driver has exited.', label)
            return None
        if int(s.session) != drv.COMMANDING_ACTIVE:
            log.error('[%s] the iiwa is not commanding (FRI session %s). Restart the FRI app on '
                      'the smartPAD.', label, drv.SESSION_NAMES.get(int(s.session)))
            return None
        return s

    def _alive(self, s, epoch, label):
        """False (logged) if the session the motion started in is gone or has stalled."""
        if self._proc.poll() is not None:
            log.error('[%s] the FRI driver exited mid-motion.', label)
            return False
        if int(s.session) != drv.COMMANDING_ACTIVE or int(s.epoch) != epoch:
            log.error('[%s] the controller LEFT COMMANDING mid-motion (session %s, safety state %d, '
                      'drive state %d) -- the move is abandoned. E-stop, a safety stop, the app '
                      'paused on the smartPAD, or the link degrading.', label,
                      drv.SESSION_NAMES.get(int(s.session)), int(s.safety), int(s.drive))
            return False
        stale = time.monotonic() - s.time
        if stale > max(0.25, 20.0 * s.sample_time):
            log.error('[%s] no FRI cycle for %.2f s -- the driver or the network has stalled.',
                      label, stale)
            return False
        if int(s.fault_count) != self._fault_seen:
            self._fault_seen = int(s.fault_count)
            log.error('[%s] the FRI driver reports: %s.', label,
                      drv.FAULT_TEXT.get(int(s.fault), int(s.fault)))
            return False
        return True

    def _await(self, epoch, label, finished):
        """Poll at 100 Hz until `finished(state)`, a guard trips, the session goes, or timeout."""
        deadline = time.monotonic() + self.move_timeout
        last_log = time.monotonic()
        while True:
            s = self._state()
            if not self._alive(s, epoch, label):
                return False
            if finished(s):
                self._wait_cycles(s, 2)      # let the measured joints catch the last command
                return True
            if self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                self._brake(epoch)
                return False
            now = time.monotonic()
            if now > deadline:
                log.error('[%s] not finished after %.0f s -- stopping.', label, self.move_timeout)
                self._brake(epoch)
                return False
            if now - last_log > 5.0:
                log.info('[%s] executing...', label)
                last_log = now
            time.sleep(0.01)

    def _wait_cycles(self, s, n):
        target, deadline = s.cycle + n, time.monotonic() + 0.2
        while self._state().cycle < target and time.monotonic() < deadline:
            time.sleep(0.001)

    def _brake(self, epoch, timeout=3.0):
        """STOP, and wait for the arm to come to rest."""
        self._stream_q = None
        seq = self._send(drv.STOP, epoch)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self._state()
            if int(s.session) != drv.COMMANDING_ACTIVE or (s.applied_seq >= seq and s.at_rest):
                return
            time.sleep(0.005)
        log.error('The arm did not come to rest within %.1f s of a stop.', timeout)

    # ------------------------------------------------------------------ state
    def q(self):
        return list(self._state().q)

    def tcp_pose(self):
        return self.model.fk(self._state().q)

    def fk(self, q):
        return self.model.fk(np.asarray(q, dtype=float))

    def ik(self, T_base_tool0, qnear=None):
        seed = self._state().q if qnear is None else np.asarray(qnear, dtype=float)
        q = self.model.ik(T_base_tool0, seed, self.joint_limits)
        if q is None:
            log.error('No IK solution (unreachable, or no arm angle keeps the joints inside the '
                      'limits less the margin).')
            return None
        return list(q)

    def joints_ok(self, q):
        q = np.asarray(q, dtype=float)
        return q.shape == (self.dof,) and self.model.within(q, self.joint_limits)

    # ------------------------------------------------------------------ motion
    def move_j(self, q_target, speed=None, accel=None, label='move', caps=None):
        """Joint-space move: phase-synchronised (a straight line in joint space, like a UR moveJ),
        the leading joint at `speed` and none above its own cap."""
        q_target = np.asarray(q_target, dtype=float)
        if not self.joints_ok(q_target):
            log.error('[%s] move rejected: target outside the iiwa joint limits less the margin '
                      '(deg %s).', label, np.round(np.degrees(q_target), 1).tolist())
            return False
        s = self._commanding(label)
        if s is None:
            return False
        auto_speed, auto_accel, which = self._speeds(q_target, caps)
        if speed is None:
            speed = auto_speed
        else:
            which = 'explicit'
        accel = auto_accel if accel is None else float(accel)
        # Logged as SENT: the task's speed (a UR-sized number, phase-scaled) clipped joint by
        # joint to the iiwa's caps -- the asked figure alone reads several times too fast.
        v = np.minimum(speed, self.v_cap)
        lead = int(np.argmax(np.abs(q_target - np.asarray(s.q, dtype=float)) / v))
        log.info('[%s] move_j A%d sets the pace at %.0f deg/s (%.0f asked, %s cap; every joint '
                 'held to its own cap), accel %.2f rad/s^2', label, lead + 1, np.degrees(v[lead]),
                 np.degrees(speed), which, accel)
        self._stream_q = None
        a = np.full(self.dof, accel)
        seq = self._send(drv.MOVE, int(s.epoch), q_target, v, a, a / self.accel_ramp)
        return self._await(int(s.epoch), label, lambda st: st.done_seq >= seq)

    def move_l(self, T_base_tool0, speed=None, accel=None, label='move', caps=None):
        """Straight-line move of tool0: IK along the line, time-scaled so the tool stays under the
        Cartesian caps and every joint under its own, then streamed. `speed` is the tool's linear
        speed (m/s) and `accel` its acceleration (m/s^2), as for a UR moveL."""
        s = self._commanding(label)
        if s is None:
            return False
        jv, _, cv, cr = self._limits(caps)
        cv = max(float(speed) if speed is not None else cv, 1e-4)
        accel = self.cart_accel if accel is None else float(accel)
        q0 = np.asarray(s.q_cmd, dtype=float)
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

        # Seconds per segment at full speed: whichever cap binds, translation, rotation or joint.
        dq = np.abs(np.diff(path, axis=0))
        h = np.max(dq / np.minimum(jv, self.v_cap), axis=1)
        h = np.maximum(h, lin / n / cv)
        if cr > 0.0:
            h = np.maximum(h, ang / n / cr)
        t_way = np.concatenate([[0.0], np.cumsum(h)])
        # The speed ramps over at least line_ramp_s: at a slow scan speed cv/accel alone is a
        # few milliseconds, a velocity step that knocks the arm at every start and stop.
        scaling = _TimeScaling(t_way[-1], max(cv / max(accel, 1e-6), self.line_ramp))
        log.info('[%s] move_l %.3f m/s, accel %.2f m/s^2: %.1f mm in %.2f s', label, cv, accel,
                 1000.0 * lin, scaling.duration)

        # SENT AHEAD. The path is known in full, so each setpoint goes out `line_lookahead_s`
        # before the time it is for, stamped with that time (the lead builds up over the first
        # line_lookahead_s, sending the path at double speed, so the stream starts exactly where
        # the arm is). The driver's reference then always reaches past its next cycle, and a
        # hiccup in this process -- the GIL held by another thread, a slow log write -- shorter
        # than the lead changes nothing. Stamped when sent instead, a setpoint ~3 ms late let
        # the reference run out, and the driver braked and caught up at its stream limits: a
        # 7.5 mm/s line swung between 1 and 23 mm/s (offline, at 2 ms mean jitter).
        dt = 1.0 / self.stream_rate
        lead = self.line_lookahead
        epoch = int(s.epoch)
        a = np.full(self.dof, self.stream_accel)
        j = a / self.stream_accel_ramp
        t_start = time.monotonic()
        t_last, worst_gap = t_start, 0.0
        while True:
            t0 = time.monotonic()
            worst_gap, t_last = max(worst_gap, t0 - t_last), t0
            t = t0 - t_start
            t_for = min(t + min(t, lead), scaling.duration)
            tau = scaling(t_for)
            i = int(np.clip(np.searchsorted(t_way, tau, side='right') - 1, 0, n - 1))
            w = (tau - t_way[i]) / max(t_way[i + 1] - t_way[i], 1e-12)
            q = path[i] + (path[i + 1] - path[i]) * min(max(w, 0.0), 1.0)
            seq = self._send(drv.STREAM, epoch, q, self.v_cap, a, j, dt, stamp=t_start + t_for)
            if not self._alive(self._state(), epoch, label):
                return False
            if self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                self._brake(epoch)
                return False
            if t_for >= scaling.duration:
                break
            _sleep_until(t0 + dt)
        if worst_gap > max(lead, 2.0 * dt):
            log.warning('[%s] the line stream stalled for %.0f ms, more than its %.0f ms lead: the '
                        'arm slowed and caught up (something in this process held it up).',
                        label, 1000.0 * worst_gap, 1000.0 * lead)
        return self._await(epoch, label, lambda st: st.applied_seq >= seq and st.at_rest)

    def _stream(self, q, dt, label):
        s = self._state()
        if int(s.session) != drv.COMMANDING_ACTIVE or self._proc.poll() is not None:
            raise ArmError(f'{label}: the iiwa is not commanding (FRI session '
                           f'{drv.SESSION_NAMES.get(int(s.session))}).')
        a = np.full(self.dof, self.stream_accel)
        self._send(drv.STREAM, int(s.epoch), q, self.v_cap, a, a / self.stream_accel_ramp, dt)
        self._stream_q = np.asarray(q, dtype=float)

    def servo_j(self, q, dt, lookahead=0.1, gain=300):
        """One setpoint of a joint stream, then wait out the period. `lookahead` and `gain` are
        the UR's servo tuning; the driver's tracking limits play that part here."""
        t0 = time.monotonic()
        q = np.asarray(q, dtype=float)
        if not self.joints_ok(q):
            log.error('servo_j target outside the joint limits -- holding.')
        else:
            self._stream(q, dt, 'servo_j')
        _sleep_until(t0 + dt)

    def servo_l(self, T_base_tool0, dt, lookahead=0.1, gain=300):
        """One CARTESIAN setpoint (IK from the previous one, so the stream stays on its branch),
        then wait out the period."""
        t0 = time.monotonic()
        seed = self._stream_q if self._stream_q is not None else self._state().q_cmd
        q = self.model.ik(T_base_tool0, seed, self.joint_limits)
        if q is None:
            log.error('servo_l target unreachable -- holding.')
        else:
            self._stream(q, dt, 'servo_l')
        _sleep_until(t0 + dt)

    def servo_stop(self):
        """End the stream. The arm settles on the LAST setpoint and holds it -- as the UR holds the
        last servoL pose -- so a connector pressed home stays home."""
        self._stream_q = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            s = self._state()
            if int(s.session) != drv.COMMANDING_ACTIVE or s.at_rest:
                return
            time.sleep(0.005)

    def stop(self):
        s = self._state()
        if int(s.session) == drv.COMMANDING_ACTIVE:
            self._brake(int(s.epoch))

    # ------------------------------------------------------------------ force / torque
    def _raw_wrench(self, s=None):
        s = self._state() if s is None else s
        return wrench_from_torques(self.model, s.q, s.tau_ext, self.wrench_damping, self._payload)

    def zero_ft(self, settle=True):
        """Tare. With `settle`, wait for the filtered torques to settle and average 0.2 s."""
        if not settle:
            self._tare = self._raw_wrench()
            return True
        time.sleep(0.3)
        samples = []
        for _ in range(20):
            samples.append(self._raw_wrench())
            time.sleep(0.01)
        self._tare = np.mean(samples, axis=0)
        # 5 N, not more: the Robotiq + D415 weighs ~12 N, which is what a missing load reads as.
        if float(np.linalg.norm(self._tare[:3])) > 5.0:
            log.warning('The tare absorbed %.1f N -- the tool weight, if the Sunrise app has no '
                        'load data for it. A base-frame tare cancels the weight\'s force at every '
                        'orientation, but not its moment about the flange, which turns with the '
                        'tool (~1 Nm at 60 deg of tilt). For the moments, set '
                        'robot.fri.compensate_payload.', float(np.linalg.norm(self._tare[:3])))
        return True

    def wrench(self):
        return self._raw_wrench() - self._tare

    def external_torques(self):
        """(q, tau_ext) from ONE state: KUKA's external joint torques (Nm, low-passed at
        torque_filter_hz) and the joints they were read at -- untared, no payload subtracted.
        What apps/payload_calibration fits the tool's mass and CoG to."""
        s = self._state()
        return np.asarray(s.q, dtype=float), np.asarray(s.tau_ext, dtype=float)

    # ------------------------------------------------------------------ lifecycle
    def disconnect(self):
        """Brake, end the FRI session, stop the driver. Idempotent; also runs at exit."""
        proc = getattr(self, '_proc', None)
        if proc is None:
            return
        self._proc = None
        try:
            if proc.poll() is None:
                s = self._chan.state.read()
                if s is not None and int(s.session) == drv.COMMANDING_ACTIVE:
                    self._send(drv.QUIT, int(s.epoch))       # brakes, then leaves
                    try:
                        proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        pass
                if proc.poll() is None:
                    proc.terminate()                         # blocked on a silent socket
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        finally:
            self._relay_thread.join(timeout=1.0)            # the driver's last lines, to EOF
            proc.stdout.close()
            self._chan.close(unlink=True)
            atexit.unregister(self.disconnect)
            log.info('FRI session closed.')
