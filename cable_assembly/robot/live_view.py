"""LIVE VIEW -- the robot's state, streamed for a viewer to mirror while a task runs on it.

    python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench     # the cell has live_view on
    cd sim && ./run.sh python scripts/live_view_gl.py                # the iiwa 14, live (OpenGL, light)
    cd sim && ./run.sh python scripts/live_view.py                   # ... or in Isaac Sim (2 GiB of GPU)

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
    enabled      false
    host         127.0.0.1    where the viewer runs -- another PC's address works too
    port         6008
    rate_hz      60
    frames       false        also send every frame the camera captures (needs camera.enable_depth)
    frames_port  6009

THE CAMERA'S FRAMES (FrameSender), for the viewer's camera layer: each frame the task captures,
colour + depth + the camera pose at capture, at half resolution, over TCP -- a frame is ~1 MB, too
big for a datagram. The camera hands it over in the task's thread (a strided copy, ~1 ms); a
thread of its own sends it. It keeps only the newest: a frame the viewer has not taken by the next
capture is dropped, not queued. With no viewer listening, a refused connect costs nothing, and is
not retried for FRAMES_RETRY_S.

A FRAME on the wire: 4 bytes (big-endian) of header length, the header (JSON), the RGB bytes
(h, w, 3 uint8), the depth bytes (h, w uint16, `depth_unit` m each, 0 = no reading). The header:
    v 1, kind "rgbd", seq, t (sender wall time), robot, phase, w, h,
    K (3x3, for this resolution), T_base_cam, T_base_tool0, T_tool0_fingertip (4x4 or null),
    depth_unit, rgb_bytes, depth_bytes

AN OVERLAY (FrameSender.send_overlay) -- something the task worked out, for the viewer to draw: the
header alone, kind "overlay", with `name` (a newer one of the same name replaces it), `text` (lines
for the viewer's panel) and `items`, each one of
    {"type": "points", "points": [[x, y, z], ...], "color": [r, g, b], "radius": m}
    {"type": "line",   "points": [[x, y, z], ...], "color": [r, g, b]}          (a polyline)
    {"type": "pose",   "T": 4x4, "length": m}                                   (RGB axes)
in the base frame, m. Unlike frames, overlays queue (up to 64) rather than replace each other.
"""

import collections
import json
import socket
import struct
import threading
import time

import numpy as np

from .. import log as urlog

log = urlog.get('live-view')

DEFAULT_PORT = 6008
FRAMES_PORT = 6009
# Every FRAME_STRIDE-th pixel each way: 640x360 from the D415's 1280x720, ~1.2 MB a frame. The
# viewer keeps one point per 2 mm, which this still out-samples on the bench.
FRAME_STRIDE = 2
# After a refused or failed connect, frames are dropped for this long before the next try.
FRAMES_RETRY_S = 2.0


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
             'cd sim && ./run.sh python scripts/live_view_gl.py', host, port, 1.0 / pub.period)
    return pub


class FrameSender:
    """Sends each frame the camera captures to the viewer (see the module docstring). Register it
    as a camera listener: `camera.listeners.append(sender)`."""

    def __init__(self, robot_name, T_tool0_cam, T_tool0_tip=None, host='127.0.0.1', port=FRAMES_PORT,
                 arm=None, stride=FRAME_STRIDE):
        self.robot_name, self.arm, self.stride = robot_name, arm, int(stride)
        self.T_cam_tool0 = np.linalg.inv(np.asarray(T_tool0_cam, dtype=float))
        self.T_tool0_tip = None if T_tool0_tip is None else np.asarray(T_tool0_tip, dtype=float)
        self.address = (host, int(port))
        self._seq = 0
        self._pending = None                 # the newest frame not yet sent
        self._overlays = collections.deque(maxlen=64)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._sock = None
        self._next_try = 0.0
        self._warned = False
        self._thread = threading.Thread(target=self._run, name='live-frames', daemon=True)
        self._thread.start()

    def __call__(self, frame):
        """The camera listener: in the task's thread, so only a strided copy and a hand-over."""
        if frame.depth is None or frame.T_base_cam is None:
            if not self._warned:
                self._warned = True
                log.warning('live view: the camera frames carry no %s, so none are sent%s.',
                            'depth' if frame.depth is None else 'pose',
                            ' (set camera.enable_depth)' if frame.depth is None else '')
            return
        s = self.stride
        rgb = np.ascontiguousarray(frame.color[::s, ::s, ::-1])          # BGR -> RGB
        depth = np.clip(np.rint(frame.depth[::s, ::s] * 1000.0), 0, 65535).astype(np.uint16)
        K = np.array(frame.K, dtype=float)
        K[:2] /= s                                 # every s-th pixel from 0: fx, fy, cx, cy / s
        T_base_cam = np.asarray(frame.T_base_cam, dtype=float)
        header = {
            'v': 1, 'kind': 'rgbd', 'seq': self._seq, 't': time.time(), 'robot': self.robot_name,
            'phase': getattr(self.arm, 'phase', '') or '', 'w': rgb.shape[1], 'h': rgb.shape[0],
            'K': K.tolist(), 'T_base_cam': T_base_cam.tolist(),
            'T_base_tool0': (T_base_cam @ self.T_cam_tool0).tolist(),
            'T_tool0_fingertip': None if self.T_tool0_tip is None else self.T_tool0_tip.tolist(),
            'depth_unit': 0.001, 'rgb_bytes': rgb.nbytes, 'depth_bytes': depth.nbytes,
        }
        self._seq += 1
        with self._lock:
            self._pending = (header, rgb, depth)
        self._wake.set()

    def send_overlay(self, name, items, text=()):
        """Queue an overlay for the viewer (see the module docstring); returns at once."""
        header = {'v': 1, 'kind': 'overlay', 'name': str(name), 't': time.time(),
                  'text': [str(t) for t in text], 'items': items}
        with self._lock:
            self._overlays.append((header, None, None))
        self._wake.set()

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                item, self._pending = self._pending, None
                queued = list(self._overlays)
                self._overlays.clear()
            for overlay in queued:
                self._send(*overlay)
            if item is not None:
                self._send(*item)

    def _send(self, header, rgb, depth):
        if self._sock is None:
            if time.monotonic() < self._next_try:
                return                               # no viewer a moment ago: drop this one
            try:
                self._sock = socket.create_connection(self.address, timeout=0.5)
                self._sock.settimeout(2.0)
            except OSError:
                self._next_try = time.monotonic() + FRAMES_RETRY_S
                return
        h = json.dumps(header).encode()
        payload = b'' if rgb is None else rgb.tobytes() + depth.tobytes()
        try:
            self._sock.sendall(struct.pack('>I', len(h)) + h + payload)
        except OSError:                              # the viewer went away mid-frame
            self._sock.close()
            self._sock = None
            self._next_try = time.monotonic() + FRAMES_RETRY_S

    def close(self):
        """Stop; a frame not sent yet is dropped. Idempotent."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def start_frames(cfg, arm, T_tool0_cam, T_tool0_tip=None):
    """A FrameSender if the config's live_view.enabled and live_view.frames, else None."""
    c = cfg.get_path('live_view', {}) or {}
    if not (c.get('enabled', False) and c.get('frames', False)):
        return None
    host, port = str(c.get('host', '127.0.0.1')), int(c.get('frames_port', FRAMES_PORT))
    log.info('Live view: sending the camera frames to tcp://%s:%d.', host, port)
    return FrameSender(arm.model.NAME, T_tool0_cam, T_tool0_tip, host, port, arm=arm)
