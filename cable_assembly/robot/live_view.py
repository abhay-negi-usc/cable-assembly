"""LIVE VIEW -- the robot's state, streamed for a viewer to mirror while a task runs on it.

    python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench     # the cell has live_view on
    cd sim && ./run.sh python scripts/live_view.py                   # the iiwa 14 in Isaac Sim, live

A daemon thread samples the arm's joints, the gripper's position and the task's phase `rate_hz`
times a second, and sends each sample as one UDP datagram to `host:port`. Nothing answers and
nothing waits: with no viewer listening the datagrams are dropped, so a viewer can be started,
closed and restarted at any point of a run without the task noticing.

IT READS ONLY WHAT IS ALREADY IN MEMORY, so it cannot slow or disturb the control:
  * the joints are arm.q() -- for the iiwa a lock-free read of the FRI driver's shared-memory
    state (backends/fri_driver.py), never a message to the controller;
  * the gripper is the position the TASK last read over Modbus (Robotiq2F85.last_counts). The
    serial bus belongs to the task's thread; a second reader on it would garble both. The task
    polls it every 20 ms while the fingers move, so the motion still shows;
  * the phase is the last one the task announced (Arm.set_speed_scale).

THE DATAGRAM, one JSON object (version 1):
    v        1
    seq      sample counter; a restarted task starts again at 0
    t        sender wall time, s (time.time())
    robot    the model's NAME: iiwa14 | ur10e
    joints   the model's JOINT_NAMES
    q        measured joints, rad, in that order
    gripper  Robotiq counts, 0 open .. 255 closed, as last read; null when unknown
    phase    the task's current phase; '' before the first
    dry_run  true when the arm is the kinematic simulator

CONFIG (a `live_view:` block; configs/cells/iiwa14_bench.yaml turns it on):
    enabled   false
    host      127.0.0.1    where the viewer runs -- another PC's address works too
    port      6008
    rate_hz   60
"""

import json
import socket
import threading
import time

from .. import log as urlog

log = urlog.get('live-view')

DEFAULT_PORT = 6008


class LivePublisher:
    """Streams `arm` (and `gripper`, if any) to a viewer from a daemon thread until close()."""

    def __init__(self, arm, gripper=None, host='127.0.0.1', port=DEFAULT_PORT, rate_hz=60.0):
        self.arm = arm
        self.gripper = gripper               # Robot.register_gripper swaps it
        # Resolved once: a name looked up on every send would put DNS in the loop.
        self.address = socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_DGRAM)[0][4]
        self.period = 1.0 / max(float(rate_hz), 1.0)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='live-view', daemon=True)
        self._thread.start()

    def sample(self):
        """One datagram's content (see the module docstring)."""
        g = getattr(self.gripper, 'last_counts', None) if self.gripper is not None else None
        return {
            'v': 1, 'seq': self._seq, 't': time.time(),
            'robot': self.arm.model.NAME, 'joints': list(self.arm.joint_names),
            'q': [float(x) for x in self.arm.q()],
            'gripper': None if g is None else int(g),
            'phase': getattr(self.arm, 'phase', ''), 'dry_run': bool(self.arm.dry_run),
        }

    def _send(self):
        """Sample and send once. Raises if the state cannot be read; a lost datagram is silent."""
        msg = json.dumps(self.sample()).encode()
        self._seq += 1
        try:
            self._sock.sendto(msg, self.address)
        except OSError:                      # a full buffer, no route: the viewer's loss only
            pass

    def _run(self):
        failing = False
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self._send()
                failing = False
            except Exception as exc:         # noqa: BLE001 -- e.g. the arm mid-teardown
                if not failing:
                    log.debug('live view: no sample (%s)', exc)
                failing = True
            next_t += self.period
            delay = next_t - time.monotonic()
            if delay < 0.0:                  # fell behind (a stalled process): don't burst
                next_t, delay = time.monotonic(), 0.0
            self._stop.wait(delay)

    def close(self):
        """Stop streaming, after one last sample: the viewer is left showing where the arm ended
        (a dry run's instant moves can finish between two samples). Idempotent; call it before
        the arm disconnects."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._send()
        except Exception:                    # noqa: BLE001 -- the arm already gone
            pass
        self._sock.close()


def start(cfg, arm, gripper=None):
    """A running LivePublisher if the config's live_view.enabled, else None."""
    c = cfg.get_path('live_view', {}) or {}
    if not c.get('enabled', False):
        return None
    host, port = str(c.get('host', '127.0.0.1')), int(c.get('port', DEFAULT_PORT))
    pub = LivePublisher(arm, gripper, host, port, float(c.get('rate_hz', 60.0)))
    log.info('Live view: streaming the robot state to udp://%s:%d at %.0f Hz. To watch it: '
             'cd sim && ./run.sh python scripts/live_view.py', host, port, 1.0 / pub.period)
    return pub
