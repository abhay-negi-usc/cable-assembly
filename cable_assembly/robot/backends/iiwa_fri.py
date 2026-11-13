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

KEPT OPEN BETWEEN RUNS (robot.fri.keep_alive; the user, 2026-11-11: "can ctrl+c still keep the connection
open so i don't have to restart it on pendant?"). The driver is started under fixed names for its port
(fri_driver.paths) and outlives the run: on Ctrl+C -- or any exit -- the run brakes the arm to rest and lets
go, and the driver HOLDS it there with the session open. The next run ATTACHES to that driver instead of
starting one, so the pendant's app never sees its client go. Started with other driver options (limits, the
CPU, the SDK), the kept session is ended and a new one started -- then the pendant's app may need its
restart after all. End one by hand: python -m cable_assembly.robot.backends.fri_driver end --port 30200.

IF THE SESSION LEAVES COMMANDING (e-stop, a safety stop, the app stopped on the pendant, the
link degrading), the move in progress returns False, and the driver holds the arm where it is
when the session comes back -- it never resumes the abandoned motion.

THE WRENCH is the contact wrench in base_link, moment about the flange (tool0), like every
backend's. KUKA's estimate carries model error, so it is tared like an F/T sensor -- and if the
Sunrise app does not carry the tool's load data, the tool's own weight shows up as contact force
(set robot.fri.compensate_payload with robot.payload, see the cell).
"""

import atexit
import itertools
import json
import os
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


def _is_driver(pid, channel):
    """Whether `pid` is a live FRI driver serving `channel` (not a reused pid)."""
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as fh:
            cmd = fh.read().split(b'\0')
        with open(f'/proc/{pid}/stat') as fh:
            if fh.read().split(') ')[1][0] == 'Z':
                return False
    except (OSError, IndexError):
        return False
    return any(c.endswith(b'fri_driver.py') for c in cmd) and channel.encode() in cmd


class _KeptDriver:
    """A driver this run attached to (not its child): the bits of Popen the backend uses."""

    def __init__(self, pid):
        self.pid, self.returncode, self.stdout = int(pid), None, None

    def poll(self):
        if self.returncode is None:
            try:
                with open(f'/proc/{self.pid}/stat') as fh:
                    if fh.read().split(') ')[1][0] == 'Z':
                        self.returncode = 0
            except (OSError, IndexError):
                self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                raise subprocess.TimeoutExpired('fri_driver', timeout)
            time.sleep(0.02)
        return self.returncode

    def terminate(self):
        try:
            os.kill(self.pid, 15)
        except ProcessLookupError:
            pass

    def kill(self):
        try:
            os.kill(self.pid, 9)
        except ProcessLookupError:
            pass


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


IMPEDANCE_MAX_SETTLE_DEG = 6.0   # how far off its command the arm may stand at rest under the KUKA's impedance


def reserve_fri_cpu(cfg):
    """robot.fri.cpu: the core the FRI driver gets to itself -- this process (and so every process it starts after
    this, the 6D server's included) is kept off it. None (unset, or no such core): nothing reserved. Idempotent."""
    cpu = (cfg.get_path('robot.fri') or {}).get('cpu') if hasattr(cfg, 'get_path') else None
    if cpu is None:
        return None
    try:
        cpu = int(cpu)
        allc = set(range(os.cpu_count() or 1))
        if cpu not in allc or len(allc) < 2:
            log.warning('robot.fri.cpu %s: no such core here (0..%d) -- not reserved.', cpu, len(allc) - 1)
            return None
        mine = os.sched_getaffinity(0)
        if cpu in mine:
            os.sched_setaffinity(0, (mine - {cpu}) or (allc - {cpu}))
            log.info('CPU %d kept for the FRI driver: this process and those it starts run on the other %d.', cpu,
                     len(os.sched_getaffinity(0)))
        return cpu
    except (OSError, AttributeError, ValueError) as exc:
        log.warning('robot.fri.cpu not reserved (%s).', exc)
        return None


def _check_kuka_impedance(mode, name, stiffness):
    """robot.compliance_controller kuka_impedance: the controller MUST be in Cartesian impedance -- in POSITION the
    streamed reference would be followed rigidly into every contact, with no compliance anywhere. Raises ArmError
    (before anything moves) unless it is."""
    if mode != 1:
        raise ArmError('robot.compliance_controller is kuka_impedance (the KUKA-impedance version of the app), but '
                       'the FRI app on the controller reports %s control. Nothing here yields to contact in that '
                       'mode -- restart the FRI app on the smartPAD with FRI control mode CARTESIAN_IMPEDANCE_CONTROL '
                       '(client command mode POSITION), or run the admittance version.'
                       % (name or 'an UNREPORTED'))
    log.info('FRI: the controller is in CARTESIAN IMPEDANCE control -- it, not this PC, yields to contact. Its '
             'spring is taken as %s (N/m, Nm/rad: robot.fri.cartesian_impedance.stiffness -- check it matches the FRI '
             "app's), and the tool's load data must be set in Sunrise, or the arm sags off every checked path.",
             [round(v) for v in stiffness])


def _check_load_data(raw_wrench, mass_kg):
    """Cartesian impedance holds the tool up only if Sunrise knows its weight. At rest, the external force KUKA
    estimates with nothing subtracted is then ~0; about the tool's weight means the load data is missing (2026-10-29:
    ~12 N), and the spring would carry that weight as a sag off every checked path. Raises ArmError if so."""
    F = float(np.linalg.norm(np.asarray(raw_wrench, dtype=float)[:3]))
    if mass_kg > 0 and F > 0.5 * mass_kg * 9.81:
        raise ArmError('The arm reads %.1f N of external force at rest, about the tool\'s weight (%.2f kg = %.1f N): the '
                       'FRI app has no LOAD DATA for it. Under Cartesian impedance the spring would carry that weight '
                       'as a sag off every checked path. Enter the tool\'s load data in Sunrise (robot.payload: mass and '
                       'CoG), or run the admittance version.' % (F, mass_kg, mass_kg * 9.81))
    log.info('FRI: %.1f N of external force at rest with nothing subtracted -- the tool\'s load data is in Sunrise.', F)


class IiwaFRIArm(Arm):
    """The iiwa 14 over FRI. Blocking; every move returns True/False."""

    supports_force_mode = False              # compliance is the AdmittanceController over servo_l ...
    supports_kuka_impedance = True           # ... or the controller's own (robot.compliance_controller)

    def __init__(self, cfg, frames=None):
        super().__init__(cfg, iiwa14)
        r = cfg.section('robot')
        f = dict(r.get('fri') or {})
        # THE CONTROLLER'S CARTESIAN SPRING, as the FRI app on the pendant sets it -- what the KUKA-impedance
        # version sizes its pushes and pulls by. Not readable over FRI: it must be written here to match.
        ci = dict(f.get('cartesian_impedance') or {})
        from ..admittance import KUKA_DEFAULT_STIFFNESS
        self.kuka_stiffness = [float(v) for v in (ci.get('stiffness') or KUKA_DEFAULT_STIFFNESS)]

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
        self._payload_mass = float(payload.get('mass_kg') or 0.0)
        if self.compliance == 'kuka_impedance':
            # The impedance carries the tool only with its load data IN SUNRISE (checked at connect) -- and then
            # KUKA's estimate has the weight out already: subtracting it here too would take it off twice.
            if f.get('compensate_payload', False):
                log.info('robot.fri.compensate_payload ignored: under the KUKA\'s impedance the tool\'s load data is '
                         'Sunrise\'s (checked at connect), so its weight is not subtracted here as well.')
        elif f.get('compensate_payload', False) and payload.get('mass_kg'):
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
            'cpu': reserve_fri_cpu(cfg),
            'torque_filter_hz': float(f.get('torque_filter_hz', 10.0)),
            'q_min': list(self.joint_limits[:, 0]), 'q_max': list(self.joint_limits[:, 1]),
            'hard_min': list(self.model.JOINT_LIMITS[:, 0]),
            'hard_max': list(self.model.JOINT_LIMITS[:, 1]),
            'v_cap': list(self.v_cap), 'stop_accel': stop_accel, 'stop_jerk': stop_jerk,
        }
        self.keep_alive = bool(f.get('keep_alive', False))
        self._port = port
        self.attached = False
        self._paths = None
        self._relay_stop = threading.Event()
        if self.keep_alive:
            self._paths = drv.paths(port)
            options.update(keep_alive=True, keep_alive_max_s=f.get('keep_alive_max_s', 3600.0),
                           owner_file=self._paths['owner'], info_file=self._paths['info'])
            self._attach(port, options)
        if not self.attached:
            self._spawn(options)
        atexit.register(self.disconnect)
        if not self.attached:
            log.info('FRI driver started (pid %d)%s; joint speed caps %s deg/s (%.0f%% of rated).',
                     self._proc.pid, ' -- kept open after the run (robot.fri.keep_alive)' if self.keep_alive else '',
                     np.round(np.degrees(self.v_cap)).astype(int).tolist(),
                     100.0 * float(f.get('max_velocity_fraction', 0.5)))
        try:
            self._await_commanding(float(f.get('connect_timeout_s', 60.0)), port)
        except BaseException:
            self.disconnect(release=True)        # nothing to keep: a session that never came, or a wrong one
            raise

        if frames is not None:
            self.publish_frames(frames, cfg)

    # ------------------------------------------------------------------ the driver
    def _spawn(self, options):
        """Start the driver process (under the port's fixed names with keep_alive, else a fresh channel)."""
        pp = self._paths
        if pp is not None:
            for k in ('channel', 'info', 'owner'):
                try:
                    os.unlink(pp[k])
                except FileNotFoundError:
                    pass
            fd = os.open(pp['channel'], os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.ftruncate(fd, drv.Channel.SIZE)
            finally:
                os.close(fd)
            self._chan = drv.Channel(pp['channel'])
            with open(pp['owner'], 'w') as fh:
                fh.write(str(os.getpid()))
            out = open(pp['log'], 'w')           # the driver's lines: this run and the next follow the file
        else:
            self._chan = drv.Channel.create()
            out = subprocess.PIPE
        # -P: the script's own directory (the backends package) must not shadow anything.
        self._proc = subprocess.Popen(
            [sys.executable, '-P', '-u', drv.__file__, '--channel', self._chan.path,
             '--options', json.dumps(options)],
            stdout=out, stderr=subprocess.STDOUT, text=True, bufsize=1,
            # ITS OWN SESSION: Ctrl+C in the terminal goes to the task, which brakes the arm to rest and
            # then ends the session (disconnect). In the terminal's process group the driver got the
            # SIGINT too and dropped the FRI link at once -- the controller stopped the arm on the lost
            # connection, abruptly (the user, 2026-11-04). It still brakes by itself if the task dies.
            start_new_session=True)
        if pp is not None:
            out.close()
            with open(pp['info'], 'w') as fh:
                json.dump({'pid': self._proc.pid, 'options': options}, fh)
            self._relay_thread = threading.Thread(target=self._relay_file, args=(pp['log'], False),
                                                  name='fri-log', daemon=True)
        else:
            self._relay_thread = threading.Thread(target=self._relay, name='fri-log', daemon=True)
        self._relay_thread.start()

    def _attach(self, port, options):
        """KEEP ALIVE: attach to the driver an earlier run kept open on `port`, if there is one started with these
        options. True if attached."""
        pp = self._paths
        try:
            with open(pp['info']) as fh:
                info = json.load(fh)
            pid = int(info['pid'])
        except (OSError, ValueError, KeyError):
            return False
        if not _is_driver(pid, pp['channel']):
            return False
        have = info.get('options') or {}
        diff = sorted(k for k in set(options) | set(have) if json.dumps(options.get(k)) != json.dumps(have.get(k)))
        if diff:
            log.warning('FRI: a session is kept open (driver pid %d), but with other driver options (%s) -- ending it '
                        'and starting a new one. The FRI app on the pendant may need restarting this once.',
                        pid, ', '.join(diff))
            drv.end_session(port)
            return False
        self._chan = drv.Channel(pp['channel'])
        with open(pp['owner'], 'w') as fh:
            fh.write(str(os.getpid()))
        self._proc = _KeptDriver(pid)
        s = self._chan.state.read()
        self._seq = int(s.applied_seq) if s is not None else 0       # the driver ignores a repeated seq
        self._relay_thread = threading.Thread(target=self._relay_file, args=(pp['log'], True), name='fri-log',
                                              daemon=True)
        self._relay_thread.start()
        self.attached = True
        log.info('FRI: ATTACHED to the session the last run kept open (driver pid %d, session %s) -- no restart '
                 'on the pendant.', pid, drv.SESSION_NAMES.get(int(s.session)) if s is not None else '?')
        return True

    def _relay(self):
        """The driver's lines, into this process's log."""
        for line in self._proc.stdout:
            log.info('[fri] %s', line.rstrip())

    def _relay_file(self, path, from_end):
        """The kept driver's lines (its log file), into this process's log, until disconnect."""
        try:
            fh = open(path)
        except OSError:
            return
        with fh:
            if from_end:
                fh.seek(0, os.SEEK_END)
            while True:
                line = fh.readline()
                if line:
                    log.info('[fri] %s', line.rstrip())
                elif self._relay_stop.is_set():
                    return
                else:
                    time.sleep(0.05)

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
                # THE CONTROLLER'S CONTROL MODE (the LBRServer asks for it at every launch). This code
                # plans, checks paths and streams assuming the arm IS where it is commanded. Under
                # joint/Cartesian impedance it is a spring about the command -- and Sunrise carries no
                # load data for this tool (~1.3 kg), so the arm lags, stops short and sags off the
                # checked path (2026-11-04 09:17: 6 deg behind in a move, 2.6 deg short at rest).
                mode = int(s.get('control_mode', -1)) if hasattr(s, 'get') else -1
                name = {0: 'POSITION', 1: 'CARTESIAN IMPEDANCE', 2: 'JOINT IMPEDANCE', 3: 'NO CONTROL'}.get(mode)
                if self.compliance == 'kuka_impedance':
                    _check_kuka_impedance(mode, name, self.kuka_stiffness)
                    _check_load_data(wrench_from_torques(self.model, s.q, s.tau_ext, self.wrench_damping, None),
                                     self._payload_mass)
                    return
                if name is None:
                    log.warning('FRI: the controller\'s control mode is not reported.')
                elif mode != 0:
                    # REFUSED, not just said (2026-11-11: after a reboot the pendant's app came up in CARTESIAN
                    # IMPEDANCE; the error went by at connect, the arm settled 1.4-2.5 deg off its command at rest,
                    # and every planned move was refused "starts 1.9 deg from where the arm is" at the pick).
                    raise ArmError(f'The FRI app on the pendant is in {name} control, but this is the ADMITTANCE '
                                   'version: every path check and plan assumes the arm follows its command rigidly. '
                                   'Restart the FRI app on the pendant choosing POSITION_CONTROL -- or, for the '
                                   'controller\'s impedance, run python -m cable_assembly.apps.bnc_assembly_impedance.')
                else:
                    log.info('FRI: the controller is in POSITION control.')
                return
            if session != said:
                log.info('FRI: %s.', hints.get(session, drv.SESSION_NAMES.get(session)))
                said = session
            if time.monotonic() > deadline:
                raise ArmError(f'No FRI commanding session after {timeout:.0f} s: '
                               f'{hints.get(session, session)}.')
            time.sleep(0.05)

    def link_down(self):
        """None while the FRI session is COMMANDING; else why not (2026-11-06: the session went IDLE mid-run and
        the campaign kept offering retries that failed at once)."""
        if self.dry_run:
            return None
        try:
            if self._proc.poll() is not None:
                return 'the FRI driver has exited'
            sess = int(self._state().session)
        except Exception as exc:                       # noqa: BLE001
            return f'the FRI state cannot be read ({exc})'
        if sess != drv.COMMANDING_ACTIVE:
            return f'the iiwa is not commanding (FRI session {drv.SESSION_NAMES.get(sess, sess)})'
        return None

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
        # FROM WHERE THE ARM IS: under the KUKA's impedance that is the MEASURED joints, which settle off the command
        # at rest (2026-11-12 22:10: 1.3 deg) -- a detour planned from the command was then refused by
        # move_joint_path, which starts a path from the measured joints (and slides the command onto them)
        q_from = np.asarray(s.q if self.compliance == 'kuka_impedance' else s.q_cmd, dtype=float)
        went = self._go_round(q_from, q_target, label, caps)
        if went is not None:                 # the straight line is blocked (Arm.path_check)
            return went
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
            q = self.path_ik(slerp_matrix(T0, T1, k / n), q)
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
        if self._line_refused(path, label):
            return False

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

        def sample(tau):
            i = int(np.clip(np.searchsorted(t_way, tau, side='right') - 1, 0, n - 1))
            w = (tau - t_way[i]) / max(t_way[i + 1] - t_way[i], 1e-12)
            return path[i] + (path[i + 1] - path[i]) * min(max(w, 0.0), 1.0)

        return self._stream_timed(lambda t: sample(scaling(t)), scaling.duration, int(s.epoch), label)

    def move_joint_path(self, times, q, label='path', caps=None):
        """Follow a planned joint path (times (N,) from 0, positions (N, dof)) -- a cuMotion plan
        (robot/planner.py) -- streamed as move_l streams its line, its timing stretched until no
        joint passes the move's caps or its own velocity cap."""
        s = self._commanding(label)
        if s is None:
            return False
        times, q = np.asarray(times, dtype=float), np.asarray(q, dtype=float)
        if not all(self.joints_ok(x) for x in q):
            log.error('[%s] path rejected: it leaves the iiwa joint limits less the margin.', label)
            return False
        # WHERE THE ARM IS: its command under position control (the two agree to ~0.05 deg). Under the KUKA's
        # impedance the arm settles OFF its command -- the tool's pose held, the wrist and elbow free in the
        # null space (2026-11-06 11:38, at rest: A5 -2.9, A7 +2.9 deg, every external torque under 1 Nm) -- and a
        # plan starts from where it IS (the measured joints). The command then slides onto the arm before the path
        # (the spring relaxes; nothing jumps), up to IMPEDANCE_MAX_SETTLE_DEG -- beyond that something holds it.
        q_cmd = np.asarray(s.q_cmd, dtype=float)
        here = np.asarray(s.q, dtype=float) if self.compliance == 'kuka_impedance' else q_cmd
        off = float(np.max(np.abs(q[0] - here)))
        if off > np.radians(1.0):
            log.error('[%s] path rejected: it starts %.1f deg from where the arm %s.', label, np.degrees(off),
                      'stands (measured)' if here is not q_cmd else 'is')
            return False
        if here is not q_cmd:
            gap = float(np.max(np.abs(q[0] - q_cmd)))
            if gap > np.radians(IMPEDANCE_MAX_SETTLE_DEG):
                log.error('[%s] path rejected: the arm stands %.1f deg off its command -- more than the impedance '
                          'lets it settle at rest (%.0f deg): something is holding or pushing it.', label,
                          np.degrees(gap), IMPEDANCE_MAX_SETTLE_DEG)
                return False
            if gap > np.radians(0.2):
                log.info('[%s] the command slides %.1f deg onto where the arm stands (the impedance let it settle '
                         'there) as the path starts.', label, np.degrees(gap))
        # SMOOTH, AND SLOW ENOUGH FOR EVERY LIMIT. A spline through the samples (at rest at both
        # ends), its time stretched -- uniformly, so the path stays the path -- until its joint
        # velocity, acceleration and jerk, and the tool's speed and turn rate, are all under this
        # move's caps; the acceleration also under HALF the driver's stream limit, so the driver only
        # follows it. Held to velocity alone, a cuMotion plan kept its own accelerations, past
        # that limit: the driver fell behind and caught up braking and accelerating flat out
        # (+-4 rad/s^2 back and forth, offline) -- the jerk.
        from scipy.interpolate import CubicSpline
        from scipy.ndimage import uniform_filter1d
        jv, ja, cv, cr = self._limits(caps)
        spl = CubicSpline(times, q, axis=0, bc_type='clamped')
        tt = np.linspace(0.0, float(times[-1]), max(50, int(np.ceil(float(times[-1]) / 0.002)) + 1))
        # acceleration and jerk over 20 ms, as a controller cycle or two sees them: not the spline's
        # own second and third derivatives at each knot, which a few microradians of noise inflate
        a_s = uniform_filter1d(spl(tt, 2), 10, axis=0)
        v = np.max(np.abs(spl(tt, 1)), axis=0)
        acc = np.max(np.abs(a_s), axis=0)
        jerk = np.max(np.abs(uniform_filter1d(np.gradient(a_s, tt, axis=0), 10, axis=0)), axis=0)
        a_cap = min(float(ja), 0.5 * self.stream_accel)
        pairs = list(itertools.pairwise(self.fk(x) for x in spl(tt[::5])))     # the tool, every 10 ms
        step = 5.0 * float(tt[1] - tt[0])
        tool_v = max((float(np.linalg.norm(b[:3, 3] - a_[:3, 3])) for a_, b in pairs), default=0.0) / step
        tool_w = max((pose_error(a_, b)[1] for a_, b in pairs), default=0.0) / step
        ratios = {'joint velocity': float(np.max(v / np.minimum(jv, self.v_cap))),
                  'joint acceleration': float(np.sqrt(np.max(acc) / a_cap)),
                  'jerk': float(np.cbrt(np.max(jerk) / (a_cap / self.accel_ramp))),
                  'tool speed': tool_v / cv if cv > 0.0 else 0.0,
                  'tool rotation': tool_w / cr if cr > 0.0 else 0.0}
        binding = max(ratios, key=ratios.get)
        stretch = max(1.0, ratios[binding])
        duration = float(times[-1]) * stretch
        log.info('[%s] joint path: %d samples, %.2f s (x%.2f: %s binds)', label, len(q), duration, stretch, binding)

        def sample(t):
            return np.asarray(spl(min(max(t / stretch, 0.0), float(times[-1]))), dtype=float)

        return self._stream_timed(sample, duration, int(s.epoch), label)

    def _stream_timed(self, sample, duration, epoch, label):
        """Stream `sample(t)` (joint positions at t s into the move) for `duration` s, guarded,
        then wait for the arm to come to rest on the last of it.

        SENT AHEAD. The path is known in full, so each setpoint goes out `line_lookahead_s`
        before the time it is for, stamped with that time (the lead builds up over the first
        line_lookahead_s, sending the path at double speed, so the stream starts exactly where
        the arm is). The driver's reference then always reaches past its next cycle, and a
        hiccup in this process -- the GIL held by another thread, a slow log write -- shorter
        than the lead changes nothing. Stamped when sent instead, a setpoint ~3 ms late let
        the reference run out, and the driver braked and caught up at its stream limits: a
        7.5 mm/s line swung between 1 and 23 mm/s (offline, at 2 ms mean jitter).
        """
        dt = 1.0 / self.stream_rate
        lead = self.line_lookahead
        a = np.full(self.dof, self.stream_accel)
        j = a / self.stream_accel_ramp
        t_start = time.monotonic()
        t_last, worst_gap = t_start, 0.0
        while True:
            t0 = time.monotonic()
            worst_gap, t_last = max(worst_gap, t0 - t_last), t0
            t = t0 - t_start
            t_for = min(t + min(t, lead), duration)
            q = sample(t_for)
            seq = self._send(drv.STREAM, epoch, q, self.v_cap, a, j, dt, stamp=t_start + t_for)
            if not self._alive(self._state(), epoch, label):
                return False
            if self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                self._brake(epoch)
                return False
            if t_for >= duration:
                break
            _sleep_until(t0 + dt)
        if worst_gap > max(lead, 2.0 * dt):
            log.warning('[%s] the stream stalled for %.0f ms, more than its %.0f ms lead: the '
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
        q = self.servo_ik(T_base_tool0, seed, dt)
        if q is not None:
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
        # 5 N, not more: the Robotiq + camera weighs ~12 N, which is what a missing load reads as.
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
    def disconnect(self, release=False):
        """Brake, end the FRI session, stop the driver. Idempotent; also runs at exit.

        KEEP ALIVE (robot.fri.keep_alive), unless `release`: brake to rest and LET GO -- the driver holds the arm
        with the session open for the next run (fri_driver: the owner gone)."""
        proc = getattr(self, '_proc', None)
        if proc is None:
            return
        if getattr(self, 'keep_alive', False) and not release and proc.poll() is None:
            self._proc = None
            try:
                s = self._chan.state.read()
                if s is not None and int(s.session) == drv.COMMANDING_ACTIVE:
                    self._brake(int(s.epoch))
            finally:
                try:
                    with open(self._paths['owner']) as fh:
                        mine = fh.read().strip() == str(os.getpid())
                    if mine:
                        os.unlink(self._paths['owner'])
                except OSError:
                    pass
                self._relay_stop.set()
                self._relay_thread.join(timeout=1.0)
                self._chan.close(unlink=False)
                atexit.unregister(self.disconnect)
                log.info('FRI session KEPT OPEN: the arm at rest, held by the driver (pid %d) -- the next run '
                         'attaches, no restart on the pendant. To end it: python -m '
                         'cable_assembly.robot.backends.fri_driver end --port %d', proc.pid, self._port)
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
            self._relay_stop.set()
            self._relay_thread.join(timeout=1.0)            # the driver's last lines, to EOF
            if getattr(proc, 'stdout', None) is not None:
                proc.stdout.close()
            self._chan.close(unlink=True)
            if self._paths is not None:
                for k in ('info', 'owner'):
                    try:
                        os.unlink(self._paths[k])
                    except FileNotFoundError:
                        pass
            atexit.unregister(self.disconnect)
            log.info('FRI session closed.')
