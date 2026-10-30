"""A stand-in for pyFRI: the names the FRI driver uses, with a simulated KUKA controller behind
them. The driver loads it by path (robot.fri.sdk), in its own process, exactly as it loads pyFRI,
so everything above the SDK -- the driver loop, the shared memory, IiwaFRIArm -- runs for real.

The controller keeps real time (one step per sample period) and holds the client to what the
real one does:

  * the session runs MONITORING_WAIT -> MONITORING_READY -> COMMANDING_WAIT -> COMMANDING_ACTIVE,
    and COMMANDING_WAIT only advances once the client echoes the interpolator position;
  * a commanded joint faster than its rated speed ABORTS the session (to IDLE), as KUKA does --
    so a jump that slipped through the driver fails the test instead of passing silently;
  * the measured joints are the last command, reported as float32 like pyFRI's.

configure() options (robot.fri.sdk_options):
    q0              start joints (rad), default the iiwa14 model's HOME
    sample_time     s, default 0.005
    command_mode    EClientCommandMode name, default POSITION
    floor           {'z': m, 'k': N/m}: a floor under tool0 pushing +z in base_link
    tau_bias        (7,) Nm added to every external torque (a model error to tare away)
    drop            {'at_s': s, 'for_s': s}: leave commanding (MONITORING_READY) for a while,
                    as an e-stop or a paused app does, then come back through COMMANDING_WAIT
    silent          True: never send a packet (no controller on the network)
    record          a path: every position commanded while COMMANDING_ACTIVE, as rows of
                    (time, q1..q7), saved there (.npy) when the client disconnects
"""

import enum
import time

import numpy as np

from cable_assembly.robot.kinematics import iiwa14

FRI_CLIENT_VERSION = 'fake'


class ESessionState(enum.IntEnum):
    IDLE = 0
    MONITORING_WAIT = 1
    MONITORING_READY = 2
    COMMANDING_WAIT = 3
    COMMANDING_ACTIVE = 4


class EClientCommandMode(enum.IntEnum):
    NO_COMMAND_MODE = 0
    POSITION = 1
    WRENCH = 2
    TORQUE = 3


class EConnectionQuality(enum.IntEnum):
    POOR = 0
    FAIR = 1
    GOOD = 2
    EXCELLENT = 3


class ESafetyState(enum.IntEnum):
    NORMAL_OPERATION = 0
    SAFETY_STOP_LEVEL_0 = 1
    SAFETY_STOP_LEVEL_1 = 2
    SAFETY_STOP_LEVEL_2 = 3


class EDriveState(enum.IntEnum):
    OFF = 0
    TRANSITIONING = 1
    ACTIVE = 2


_OPTS = {}


def configure(opts):
    _OPTS.clear()
    _OPTS.update(opts)


class _LBRState:
    def __init__(self, ctl):
        self._c = ctl

    def getSampleTime(self):
        return self._c.dt

    def getSessionState(self):
        return self._c.session

    def getConnectionQuality(self):
        return EConnectionQuality.EXCELLENT

    def getSafetyState(self):
        return (ESafetyState.SAFETY_STOP_LEVEL_1 if self._c.dropped
                else ESafetyState.NORMAL_OPERATION)

    def getDriveState(self):
        return EDriveState.ACTIVE

    def getClientCommandMode(self):
        return self._c.command_mode

    def getMeasuredJointPosition(self):
        return self._c.q.astype(np.float32)

    def getIpoJointPosition(self):
        return self._c.q.astype(np.float32)

    def getExternalTorque(self):
        return self._c.tau_ext().astype(np.float32)


class _LBRCommand:
    def __init__(self):
        self.q = None

    def setJointPosition(self, values):
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (7,):
            raise RuntimeError('Input array must have shape (7,)!')
        self.q = v.copy()


class LBRClient:
    def __init__(self):
        self._ctl = None
        self._command = _LBRCommand()

    def robotState(self):
        return self._ctl.state

    def robotCommand(self):
        return self._command


class ClientApplication:
    """The controller, simulated."""

    def __init__(self, client):
        o = dict(_OPTS)
        self.client = client
        client._ctl = self
        self.state = _LBRState(self)
        self.dt = float(o.get('sample_time', 0.005))
        self.q = np.asarray(o.get('q0', iiwa14.HOME), dtype=float).copy()
        self.command_mode = EClientCommandMode[o.get('command_mode', 'POSITION')]
        self.floor = o.get('floor')
        self.tau_bias = np.asarray(o.get('tau_bias', np.zeros(7)), dtype=float)
        self.drop = o.get('drop')
        self.silent = bool(o.get('silent', False))
        self.record, self._log = o.get('record'), []
        self.session = ESessionState.IDLE
        self.dropped = False
        self.aborted = False
        self._synced = 0
        self._cycle = 0

    def tau_ext(self):
        tau = self.tau_bias.copy()
        if self.floor:
            pen = float(self.floor['z']) - iiwa14.fk(self.q)[2, 3]
            if pen > 0.0:
                F = np.array([0.0, 0.0, float(self.floor['k']) * pen, 0.0, 0.0, 0.0])
                tau += iiwa14.jacobian(self.q).T @ F
        return tau

    def connect(self, port, host=None):
        self._t0 = self._next = time.monotonic()
        return True

    def disconnect(self):
        if self.record and self._log:
            np.save(self.record, np.array(self._log))

    def _session_now(self):
        if self.aborted:
            return ESessionState.IDLE
        t = time.monotonic() - self._t0
        if self.drop and self.drop['at_s'] <= t < self.drop['at_s'] + self.drop['for_s']:
            self.dropped = True
            self._synced = 0
            return ESessionState.MONITORING_READY
        self.dropped = False
        if self._cycle < 5:
            return ESessionState.MONITORING_WAIT
        if self.session in (ESessionState.MONITORING_WAIT, ESessionState.MONITORING_READY) \
                and self._cycle < 10:
            return ESessionState.MONITORING_READY
        if self.session == ESessionState.COMMANDING_ACTIVE:
            return self.session
        return ESessionState.COMMANDING_ACTIVE if self._synced >= 5 \
            else ESessionState.COMMANDING_WAIT

    def step(self):
        self._next += self.dt
        delay = self._next - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        elif delay < -10 * self.dt:
            self._next = time.monotonic()            # fell far behind: do not burst to catch up
        if self.silent:
            return True
        self._cycle += 1
        new = self._session_now()
        if new != self.session:
            old, self.session = self.session, new
            self.client.onStateChange(old, new)
        cmd = self.client.robotCommand()
        cmd.q = None
        if self.session in (ESessionState.MONITORING_WAIT, ESessionState.MONITORING_READY):
            self.client.monitor()
        elif self.session == ESessionState.COMMANDING_WAIT:
            self.client.waitForCommand()
            in_sync = cmd.q is not None and np.allclose(cmd.q, self.q.astype(np.float32),
                                                        atol=1e-6)
            self._synced = self._synced + 1 if in_sync else 0
        elif self.session == ESessionState.COMMANDING_ACTIVE:
            self.client.command()
            if cmd.q is None:
                self._abort('no command in COMMANDING_ACTIVE')
            else:
                v = np.abs(cmd.q - self.q) / self.dt
                if np.any(v > iiwa14.VELOCITY_LIMITS * 1.0001):
                    self._abort(f'commanded joint velocity {np.round(np.degrees(v), 1)} deg/s '
                                f'over the rated limits')
                else:
                    self.q = cmd.q.copy()
                    if self.record:
                        self._log.append(np.concatenate([[time.monotonic()], self.q]))
        return True

    def _abort(self, why):
        print(f'FAKE CONTROLLER ABORT: {why}', flush=True)
        self.aborted = True
