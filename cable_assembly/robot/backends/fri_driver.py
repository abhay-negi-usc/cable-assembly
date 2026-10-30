"""The FRI process: the only code that talks to the KUKA controller, and the only code that has to
keep the controller's clock.

WHY A SEPARATE PROCESS. FRI is a hard real-time contract: the controller sends its state every
sample period (1-10 ms, set in the Sunrise app) and drops the commanding session if a reply is
late or the commanded joints jump. pyFRI's ClientApplication.step() blocks in C++ on the UDP
socket WITHOUT releasing the GIL, so inside the task's process it would freeze the camera, SAM3
and the logger every cycle -- and they in turn would make the replies late. Here the loop owns an
interpreter (and, where the user may, a real-time priority) and nothing the task does can delay
it. backends/iiwa_fri.py launches this file as a script and talks to it through shared memory.

EACH CYCLE (Client.command):
    1. read the latest command from shared memory -- never blocks
    2. advance a jerk-limited trajectory (ruckig) one sample toward it
    3. send that joint position; publish the state (measured joints, filtered external torques,
       session, progress) back to shared memory

THE COMMANDS (written by IiwaFRIArm):
    MOVE    go to q and stop there, phase-synchronised: the joints move together on a straight
            line in joint space, as a UR moveJ does, under the command's vel/accel/jerk limits
    STREAM  follow setpoints sent every `dt` (servo_j, servo_l, move_l), each stamped with
            the time it is for -- when it was sent, for a servo stream; move_l, which knows its
            whole line, sends ahead: they make a reference two periods behind, which the arm tracks
            exactly when it can and catches up within the limits when it cannot (Follower);
            when the stream ends the arm settles on the last setpoint and holds it
    STOP    brake to rest as hard as the stop limits allow, then hold
    QUIT    stop, then end the session and exit

WHATEVER ARRIVES FROM THE TASK:
  * every position sent to the robot is continuous in position, velocity and acceleration, with
    each joint under its velocity cap -- a jump never reaches the controller;
  * a target outside the joint limits is clamped inside them;
  * (re)entering COMMANDING_ACTIVE starts by HOLDING where the robot is and bumps `epoch`; a
    command stamped with an older epoch is ignored, so a session that drops and comes back never
    resumes a motion the task already gave up on;
  * if the task process dies, the arm is braked to rest and the session ended.

SELF-CONTAINED: numpy, ruckig and the FRI SDK only, so the child never imports the task's
packages. The SDK is pyFRI (`sdk: pyfri`) or a module with the same names standing in for it,
loaded by path -- tests/fake_pyfri.py.

SHARED MEMORY is a plain file under /dev/shm holding two records of float64s, each with ONE
writer (command: the task; state: this process) and read lock-free under a sequence counter (a
seqlock). That is sound on x86-64, whose stores and loads are not reordered with their own kind;
a port to a weakly ordered CPU (ARM) would need real fences.
"""

import argparse
import importlib
import importlib.util
import json
import mmap
import os
import sys
import tempfile
import time

import numpy as np

DOF = 7

# Session states in KUKA's numbering (ESessionState); NO_PACKETS = nothing heard yet.
NO_PACKETS, IDLE, MONITORING_WAIT, MONITORING_READY, COMMANDING_WAIT, COMMANDING_ACTIVE = range(-1, 5)
SESSION_NAMES = {NO_PACKETS: 'NO_PACKETS', IDLE: 'IDLE', MONITORING_WAIT: 'MONITORING_WAIT',
                 MONITORING_READY: 'MONITORING_READY', COMMANDING_WAIT: 'COMMANDING_WAIT',
                 COMMANDING_ACTIVE: 'COMMANDING_ACTIVE'}
POSITION_MODE = 1                      # EClientCommandMode.POSITION, FRI 1.x

HOLD, MOVE, STREAM, STOP, QUIT = range(5)

FAULT_NONE, FAULT_COMMAND_MODE, FAULT_TRAJECTORY, FAULT_CONNECT, FAULT_CALLBACK = range(5)
FAULT_TEXT = {
    FAULT_NONE: 'none',
    FAULT_COMMAND_MODE: 'the Sunrise FRI app is not in POSITION client command mode',
    FAULT_TRAJECTORY: 'the trajectory generator rejected a command (held position)',
    FAULT_CONNECT: 'could not open the FRI UDP port',
    FAULT_CALLBACK: 'an exception in the FRI callback (held position)',
}


class Layout:
    """Named fields of a flat float64 record: scalars and DOF-vectors."""

    def __init__(self, *fields):
        self.index, i = {}, 0
        for name, n in fields:
            self.index[name] = i if n == 1 else slice(i, i + n)
            i += n
        self.size = i

    def pack(self, **values):
        v = np.zeros(self.size)
        for name, value in values.items():
            v[self.index[name]] = value
        return v

    def unpack(self, v):
        return _Record({name: (float(v[ix]) if isinstance(ix, int) else v[ix].copy())
                        for name, ix in self.index.items()})


class _Record(dict):
    __getattr__ = dict.__getitem__


STATE = Layout(('cycle', 1), ('time', 1), ('session', 1), ('quality', 1), ('safety', 1),
               ('drive', 1), ('command_mode', 1), ('sample_time', 1), ('epoch', 1),
               ('applied_seq', 1), ('done_seq', 1), ('at_rest', 1), ('fault', 1),
               ('fault_count', 1), ('q', DOF), ('q_cmd', DOF), ('qd_cmd', DOF), ('tau_ext', DOF))
COMMAND = Layout(('seq', 1), ('epoch', 1), ('mode', 1), ('dt', 1), ('stamp', 1), ('q', DOF),
                 ('v', DOF), ('a', DOF), ('j', DOF))


class Slot:
    """One single-writer record in shared memory, read lock-free under a sequence counter."""

    def __init__(self, buf, offset, layout):
        self.layout = layout
        self._seq = np.ndarray((1,), np.uint64, buf, offset)
        self._data = np.ndarray((layout.size,), np.float64, buf, offset + 8)

    def write(self, vec):
        self._seq[0] += 1                    # odd: a write is in progress
        self._data[:] = vec
        self._seq[0] += 1                    # even: stable

    def read(self):
        """The latest complete record, or None if nothing was ever written."""
        v = None
        for _ in range(2000):
            s = int(self._seq[0])
            if s == 0:
                return None
            if not s & 1:
                v = self._data.copy()
                if int(self._seq[0]) == s:
                    return self.layout.unpack(v)
            time.sleep(0)                    # the writer is mid-record (microseconds): yield
        # A writer that died mid-record. Return what is there; the caller sees the process gone.
        return None if v is None else self.layout.unpack(v)


class Channel:
    """The shared-memory file both processes map: the command slot (the task writes) and the state
    slot (this process writes). A plain file -- no multiprocessing resource tracker to fight."""

    SIZE = 16 + 8 * (COMMAND.size + STATE.size)

    def __init__(self, path):
        self.path = path
        fd = os.open(path, os.O_RDWR)
        try:
            self._mm = mmap.mmap(fd, self.SIZE)
        finally:
            os.close(fd)
        self.command = Slot(self._mm, 0, COMMAND)
        self.state = Slot(self._mm, 8 + 8 * COMMAND.size, STATE)

    @classmethod
    def create(cls):
        d = '/dev/shm' if os.path.isdir('/dev/shm') else None
        fd, path = tempfile.mkstemp(prefix='cable_assembly_fri_', dir=d)
        try:
            os.ftruncate(fd, cls.SIZE)
        finally:
            os.close(fd)
        return cls(path)

    def close(self, unlink=False):
        self.command = self.state = None
        try:
            self._mm.close()
        except BufferError:                  # a numpy view still alive somewhere; the OS frees it
            pass
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class Follower:
    """Turns commands into one joint position per controller cycle.

    MOVE and STOP are ruckig trajectories (jerk-limited, time-optimal) in joint coordinates.

    A STREAM is TRACKED, which ruckig does not do by itself: re-planning toward a moving target
    each cycle treats it as standing still, so the arm trails it for good (measured: 0.3 s behind
    a 0.25 m/s line, and 3 mm off it). Instead:

      * The setpoints make a reference r(t): straight segments between them, each placed at the
        time it is for (`stamp`, CLOCK_MONOTONIC, shared by both processes: when the task sent
        it, or later for a stream sent ahead) plus a lag of STREAM_LAG periods. Stamping at
        arrival instead would quantise the times to this loop's period -- setpoints sent every
        8 ms arrive 5 or 10 ms apart at a 5 ms cycle -- and turn a smooth stream into velocity
        steps of tens of percent.
      * ruckig runs on the ERROR e = q - r, in a frame moving with the reference over the cycle:
        from (e, de/dt) to (0, 0) under the stream limits, with the velocity cap shared between
        the reference's speed and the correction's. A stream already tracked has nothing to
        correct and goes out exactly; a jump or a late setpoint is caught up within the limits;
        when the stream ends the arm comes to rest exactly on its last setpoint.

    The clock is the robot's cycle (sample period per step), locked gently to CLOCK_MONOTONIC so
    the callback's own jitter does not show up as tracking error.

    `q_lo`/`q_hi` are the limits targets are clamped into; `hard_lo`/`hard_hi` the mechanical
    ones no output may cross; `v_cap` the per-joint velocity ceiling every command is held under.
    """

    # A setpoint is due this many stream periods after it was sent, so the reference trails the
    # stream by about that much -- and a setpoint up to a period late still arrives before it is
    # due, without the reference stalling.
    STREAM_LAG = 2.0

    def __init__(self, dt, q_lo, q_hi, hard_lo, hard_hi, v_cap, stop_accel, stop_jerk):
        from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig
        from ruckig import Synchronization
        self._CI, self._Result, self._Sync = ControlInterface, Result, Synchronization
        self.dt = float(dt)
        self.otg = Ruckig(DOF, self.dt)
        self.inp, self.out = InputParameter(DOF), OutputParameter(DOF)
        self.q_lo, self.q_hi = np.asarray(q_lo, float), np.asarray(q_hi, float)
        self.hard_lo, self.hard_hi = np.asarray(hard_lo, float), np.asarray(hard_hi, float)
        self.v_cap = np.asarray(v_cap, float)
        self.stop_accel, self.stop_jerk = float(stop_accel), float(stop_jerk)
        self.t = None                        # the cycle clock, s on CLOCK_MONOTONIC
        self.q = np.zeros(DOF)
        self.qd = np.zeros(DOF)
        self.qdd = np.zeros(DOF)
        self.mode = HOLD
        self.seq = self.done_seq = 0
        self.faulted = False                 # set when a command was refused; the caller clears it
        self._pts = None                     # the stream reference: [(t, q), ...], not yet passed
        self._stream_limits = (np.full(DOF, 1e-3), np.full(DOF, 1e-3))

    def reset(self, q):
        """Hold `q`, at rest. Called on entering COMMANDING_ACTIVE."""
        self.q = np.asarray(q, float).copy()
        self._hold()

    def _hold(self):
        self.mode, self._pts = HOLD, None
        self.qd = np.zeros(DOF)
        self.qdd = np.zeros(DOF)

    def _absolute_input(self):
        """Ruckig plans in joint coordinates from what was actually sent."""
        self.inp.current_position = list(self.q)
        self.inp.current_velocity = list(self.qd)
        self.inp.current_acceleration = list(self.qdd)
        self.inp.control_interface = self._CI.Position
        self.inp.target_velocity = [0.0] * DOF
        self.inp.target_acceleration = [0.0] * DOF

    def _clock(self, now):
        """Advance the cycle clock one period, pulled gently toward `now`."""
        if self.t is None or abs(now - self.t - self.dt) > 0.05:
            self.t = now                     # first cycle, or a stall: re-lock
        else:
            self.t += self.dt
            self.t += 0.05 * (now - self.t)

    def command(self, mode, seq, q, v, a, j, dt_stream, stamp, now):
        if mode == STOP:
            self.stop()
        elif mode == MOVE:
            self._absolute_input()
            self.inp.synchronization = self._Sync.Phase
            self.inp.target_position = list(np.clip(np.asarray(q, float), self.q_lo, self.q_hi))
            self.inp.max_velocity = list(np.minimum(np.abs(v), self.v_cap))
            self.inp.max_acceleration = list(np.maximum(np.abs(a), 1e-3))
            self.inp.max_jerk = list(np.maximum(np.abs(j), 1e-3))
            self.mode, self._pts = MOVE, None
        elif mode == STREAM:
            t_now = now if self.t is None else self.t
            due = max(float(stamp) + self.STREAM_LAG * float(dt_stream), t_now + self.dt)
            end = np.clip(np.asarray(q, float), self.q_lo, self.q_hi)
            if self.mode != STREAM:
                self._pts = [(t_now, self.q.copy())]     # the stream starts where the arm is
            self._pts.append((max(due, self._pts[-1][0] + 1e-6), end))
            self._stream_limits = (np.maximum(np.abs(np.asarray(a, float)), 1e-3),
                                   np.maximum(np.abs(np.asarray(j, float)), 1e-3))
            self.mode = STREAM
        self.seq = seq

    def stop(self):
        """Brake to rest, all joints together, under the stop limits."""
        if self.mode == HOLD:
            return
        self._absolute_input()
        self.inp.control_interface = self._CI.Velocity
        self.inp.synchronization = self._Sync.Time
        self.inp.max_acceleration = [self.stop_accel] * DOF
        self.inp.max_jerk = [self.stop_jerk] * DOF
        self.mode, self._pts = STOP, None

    def _reference(self, t):
        """The reference at time t: on the polyline through the pending setpoints."""
        pts = self._pts
        if t <= pts[0][0]:
            return pts[0][1]
        for (t0, q0), (t1, q1) in zip(pts, pts[1:]):
            if t <= t1:
                return q0 + (q1 - q0) * ((t - t0) / (t1 - t0))
        return pts[-1][1]

    @property
    def at_rest(self):
        if self.mode == HOLD:
            return True
        if self.mode != STREAM:
            return False
        t1, end = self._pts[-1]
        return (self.t >= t1 and np.max(np.abs(self.qd)) < 1e-6
                and np.max(np.abs(self.q - end)) < 1e-9)

    def _update(self):
        """One ruckig step; an error result or a raised RuckigError both come back as None."""
        try:
            res = self.otg.update(self.inp, self.out)
        except Exception:                    # noqa: BLE001 -- ruckig raises on some inputs
            return None
        return res if res in (self._Result.Working, self._Result.Finished) else None

    def _stream_step(self):
        """One tracking step, or None if ruckig could not plan it."""
        while len(self._pts) > 2 and self._pts[1][0] <= self.t:
            self._pts.pop(0)                 # passed
        r0 = self._reference(self.t)
        r1 = self._reference(self.t + self.dt)
        v_ref = (r1 - r0) / self.dt          # the frame's velocity over this cycle
        e, de = self.q - r0, self.qd - v_ref
        if np.max(np.abs(e)) < 1e-10 and np.max(np.abs(de)) < 1e-8 \
                and np.max(np.abs(self.qdd)) < 1e-6:
            return r1, v_ref, np.zeros(DOF)  # on the reference: nothing to correct
        a, j = self._stream_limits
        inp = self.inp
        if np.any(np.abs(v_ref) > self.v_cap):
            # The reference itself outruns the caps (a jump): no correction on top of it can
            # stay under them, so chase it in joint coordinates instead, as fast as allowed.
            self._absolute_input()
            inp.synchronization = self._Sync.Time
            inp.target_position = list(r1)
            v = np.clip(v_ref, -self.v_cap, self.v_cap)
            v[np.abs(v) < 1e-6] = 0.0        # rounding noise ruckig can fail to synchronise
            inp.target_velocity = list(v)
            inp.max_velocity = list(self.v_cap)
            inp.max_acceleration, inp.max_jerk = list(a), list(j)
            if self._update() is None:
                inp.synchronization = self._Sync.No
                if self._update() is None:
                    return None
            o = self.out
            return (np.asarray(o.new_position), np.asarray(o.new_velocity),
                    np.asarray(o.new_acceleration))
        inp.control_interface = self._CI.Position
        # Phase-synchronised, the correction keeps the error's direction, so a stream along a
        # line stays on it (per joint, each correction would leave it by milliradians).
        inp.synchronization = self._Sync.Phase
        inp.current_position, inp.current_velocity = list(e), list(de)
        inp.current_acceleration = list(self.qdd)
        inp.target_position = [0.0] * DOF
        inp.target_velocity = [0.0] * DOF
        inp.target_acceleration = [0.0] * DOF
        inp.max_velocity = list(np.maximum(self.v_cap - np.abs(v_ref), 1e-3))
        inp.max_acceleration, inp.max_jerk = list(a), list(j)
        if self._update() is None:
            inp.synchronization = self._Sync.No
            if self._update() is None:
                return None
        o = self.out
        return (r1 + np.asarray(o.new_position), v_ref + np.asarray(o.new_velocity),
                np.asarray(o.new_acceleration))

    def step(self, now):
        """The joint position to send this cycle; `now` is CLOCK_MONOTONIC at the callback."""
        self._clock(now)
        if self.mode == HOLD:
            return self.q
        if self.mode == STREAM:
            nxt = self._stream_step()
            if nxt is not None:
                q, self.qd, self.qdd = nxt
                self.q = np.clip(q, self.hard_lo, self.hard_hi)
                return self.q
            self.faulted = True
            self.stop()
        res = self._update()
        if res is None and self.mode == MOVE:
            # Numerically awkward inputs (near-zero moves) can defeat the phase synchronisation;
            # each joint on its own still respects every limit.
            self.inp.synchronization = self._Sync.No
            res = self._update()
        if res is None:
            self.faulted = True
            if self.mode == STOP:            # not even a brake is possible: hold the last output
                self.reset(self.q)
                return self.q
            self.stop()
            res = self._update()
            if res is None:
                self.reset(self.q)
                return self.q
        self.out.pass_to_input(self.inp)
        self.q = np.clip(np.asarray(self.out.new_position), self.hard_lo, self.hard_hi)
        self.qd = np.asarray(self.out.new_velocity)
        self.qdd = np.asarray(self.out.new_acceleration)
        if res == self._Result.Finished:
            if self.mode == MOVE:
                self.done_seq = self.seq
            self._hold()
        return self.q


def make_client(sdk, chan, opts):
    """The FRI client: an `sdk.LBRClient` whose callbacks run the Follower and publish state."""

    q_lo, q_hi = np.asarray(opts['q_min'], float), np.asarray(opts['q_max'], float)
    hard_lo, hard_hi = np.asarray(opts['hard_min'], float), np.asarray(opts['hard_max'], float)
    v_cap = np.asarray(opts['v_cap'], float)
    tau_fc = float(opts.get('torque_filter_hz', 10.0))

    class Client(sdk.LBRClient):
        def __init__(self):
            super().__init__()
            self.session = NO_PACKETS
            self.epoch = 0
            self.fault = FAULT_NONE
            self.fault_count = 0
            self.follower = None
            self.cmd_seq = 0
            self.applied_seq = 0
            self.tau = None
            self.cycle = 0
            self.quitting = False
            self.done = False
            self._mode_warned = False

        # ---------------------------------------------------------------- FRI callbacks
        def onStateChange(self, old_state, new_state):
            self.session = int(new_state)
            say(f'session {SESSION_NAMES.get(int(old_state))} -> {SESSION_NAMES.get(self.session)}')
            if self.session == COMMANDING_ACTIVE:
                self.epoch += 1
                self.follower = None         # built on the first command() from the live state
            elif int(old_state) == COMMANDING_ACTIVE:
                self.follower = None
            self.publish(filter_torque=False)    # the cycle's own callback filters

        def monitor(self):
            if self.quitting:
                self.done = True             # nothing is moving: leave at once
            self.publish()

        def waitForCommand(self):
            st = self.robotState()
            if self._position_mode(st):
                self.robotCommand().setJointPosition(np.asarray(st.getIpoJointPosition(), float))
            if self.quitting:
                self.done = True
            self.publish()

        def command(self):
            st = self.robotState()
            now = time.monotonic()
            try:
                if not self._position_mode(st):
                    self.publish()
                    return
                if self.follower is None or abs(self.follower.dt - st.getSampleTime()) > 1e-9:
                    self.follower = Follower(st.getSampleTime(), q_lo, q_hi, hard_lo, hard_hi,
                                             v_cap, opts['stop_accel'], opts['stop_jerk'])
                    self.follower.reset(np.asarray(st.getIpoJointPosition(), float))
                self._read_command(now)
                q = self.follower.step(now)
                if self.follower.faulted:
                    self.follower.faulted = False
                    self._fault(FAULT_TRAJECTORY)
                if self.quitting and self.follower.at_rest:
                    self.done = True
            except Exception as exc:         # noqa: BLE001 -- a raise here would end the session
                say(f'callback error: {exc!r}')
                self._fault(FAULT_CALLBACK)
                q = (self.follower.q if self.follower is not None
                     else np.asarray(st.getIpoJointPosition(), float))
                if self.follower is not None:
                    self.follower.reset(q)
            self.robotCommand().setJointPosition(np.asarray(q, float))
            self.publish(now)

        # ---------------------------------------------------------------- helpers
        def _position_mode(self, st):
            if int(st.getClientCommandMode()) == POSITION_MODE:
                return True
            if not self._mode_warned:
                say('the FRI app is not in POSITION client command mode -- not commanding')
                self._mode_warned = True
                self._fault(FAULT_COMMAND_MODE)
            return False

        def _fault(self, code):
            self.fault = code
            self.fault_count += 1
            say(f'fault: {FAULT_TEXT[code]}')

        def _read_command(self, now):
            c = chan.command.read()
            if c is None or int(c.seq) == self.cmd_seq:
                return
            self.cmd_seq = int(c.seq)
            mode = int(c.mode)
            if mode == QUIT:
                self.quitting = True
                self.follower.stop()
            elif int(c.epoch) == self.epoch:     # an older epoch's command is a stale one
                self.follower.command(mode, self.cmd_seq, c.q, c.v, c.a, c.j, c.dt, c.stamp, now)
            self.applied_seq = self.cmd_seq

        def orphaned(self):
            """The task process is gone: brake, then leave."""
            self.quitting = True
            if self.follower is not None:
                self.follower.stop()
            else:
                self.done = True

        def publish(self, now=None, filter_torque=True):
            now = time.monotonic() if now is None else now
            self.cycle += 1
            if self.session == NO_PACKETS:
                # Before the first packet the SDK's state message may not exist yet -- reading it
                # is a crash in C++, not an exception.
                q, tau, dt, extra = np.zeros(DOF), np.zeros(DOF), 0.0, {}
            else:
                st = self.robotState()
                q = np.asarray(st.getMeasuredJointPosition(), float)
                tau = np.asarray(st.getExternalTorque(), float)
                dt = float(st.getSampleTime())
                extra = dict(quality=int(st.getConnectionQuality()),
                             safety=int(st.getSafetyState()), drive=int(st.getDriveState()),
                             command_mode=int(st.getClientCommandMode()), sample_time=dt)
            if self.tau is None or tau_fc <= 0.0 or dt <= 0.0:
                self.tau = tau
            elif filter_torque:              # first-order low-pass on the torque estimate
                self.tau = self.tau + (1.0 - np.exp(-2.0 * np.pi * tau_fc * dt)) * (tau - self.tau)
            f = self.follower
            chan.state.write(STATE.pack(
                cycle=self.cycle, time=now, session=self.session, epoch=self.epoch,
                applied_seq=self.applied_seq, done_seq=f.done_seq if f else 0,
                at_rest=1.0 if (f is None or f.at_rest) else 0.0,
                fault=self.fault, fault_count=self.fault_count, q=q,
                q_cmd=f.q if f else q, qd_cmd=f.qd if f else np.zeros(DOF), tau_ext=self.tau,
                **extra))

    return Client()


def say(msg):
    """One line to the task's log (IiwaFRIArm relays this process's stdout). Never fatal: with
    the task gone the pipe is broken, and the loop still has an arm to brake."""
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        sys.stdout = open(os.devnull, 'w')


def _load_sdk(name):
    if name.endswith('.py'):
        spec = importlib.util.spec_from_file_location('fri_sdk', name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(name)


def _realtime(priority):
    if priority <= 0:
        return
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(priority)))
        say(f'real-time priority {priority} (SCHED_FIFO)')
    except (PermissionError, OSError, AttributeError):
        say('no real-time priority (needs an rtprio limit for this user); running at normal '
            'priority -- fine at a 5-10 ms send period, marginal at 1 ms')


def serve(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--channel', required=True, help='the shared-memory file IiwaFRIArm created')
    p.add_argument('--options', required=True, help='JSON: limits, port, host, sdk, ...')
    a = p.parse_args(argv)
    opts = json.loads(a.options)

    sdk = _load_sdk(opts.get('sdk', 'pyfri'))
    if hasattr(sdk, 'configure'):
        sdk.configure(opts.get('sdk_options') or {})
    chan = Channel(a.channel)
    _realtime(int(opts.get('rt_priority', 0)))
    client = make_client(sdk, chan, opts)
    app = sdk.ClientApplication(client)
    client.publish()                         # "driver up, nothing heard yet"
    port, host = int(opts.get('port', 30200)), opts.get('host')
    if not app.connect(port, host):
        client._fault(FAULT_CONNECT)
        client.publish()
        return 2
    say(f'FRI client {getattr(sdk, "FRI_CLIENT_VERSION", "?")} listening on UDP {port}'
        + (f' for {host}' if host else ''))
    ppid = os.getppid()
    try:
        while not client.done:
            if os.getppid() != ppid and not client.quitting:
                say('the task process is gone -- braking to rest and ending the session')
                client.orphaned()
            if not app.step():
                say('FRI step failed: the connection to the controller is lost')
                break
    finally:
        app.disconnect()
        chan.close(unlink=os.getppid() != ppid)   # an orphan's file has no one left to remove it
    say('session ended')
    return 0


if __name__ == '__main__':
    sys.exit(serve())
