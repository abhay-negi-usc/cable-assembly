"""RealSense capture (the bench's D415; the UR cell's D405) -- replaces realsense2_camera + cv_bridge + the CameraInfo topic.

Also fixes, by construction, the pose/image time-association problem that the ROS stack solved
with a tf2 lookup at `msg.header.stamp`. SAM3 takes 1-2 s per frame, so by the time a detection
exists the arm may have moved; fusing it against the arm's CURRENT pose would attribute the
detection to the wrong viewpoint and corrupt the triangulation.

`capture()` therefore returns a Frame that carries the camera pose SAMPLED AT CAPTURE. There is
no later lookup to get wrong, and no tf cache to expire. The pose travels WITH the pixels.
"""

import contextlib
import threading
import time

import numpy as np

from .. import log as urlog

log = urlog.get('camera')


class Frame:
    """One captured image plus everything needed to interpret it geometrically."""

    def __init__(self, color, K, D, stamp, T_base_cam=None, depth=None):
        self.color = color              # HxWx3 uint8, BGR
        self.K = K                      # 3x3 intrinsics
        self.D = D                      # distortion coefficients
        self.stamp = stamp              # time.monotonic() at capture
        self.T_base_cam = T_base_cam    # camera pose in base_link AT CAPTURE -- the whole point
        # HxW float32, metres along the optical axis, 0 where the camera has no reading. ALIGNED TO
        # THE COLOR IMAGE, so K is its intrinsics too. None unless camera.enable_depth.
        self.depth = depth

    @property
    def rgb(self):
        return self.color[:, :, ::-1]

    @property
    def age(self):
        return time.monotonic() - self.stamp


def _realsense_holders():
    """[(pid, command)] of OTHER processes that have one of the RealSense's /dev/video nodes open."""
    import glob
    import os
    nodes = set()
    for name in glob.glob('/sys/class/video4linux/video*/name'):
        try:
            with open(name) as f:
                if 'realsense' in f.read().lower():
                    nodes.add('/dev/' + name.split('/')[-2])
        except OSError:
            pass
    out, me = [], os.getpid()
    for fd_dir in glob.glob('/proc/[0-9]*/fd') if nodes else ():
        pid = int(fd_dir.split('/')[2])
        if pid == me:
            continue
        try:
            if not any(os.readlink(f'{fd_dir}/{fd}') in nodes for fd in os.listdir(fd_dir)):
                continue
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = ' '.join(f.read().replace(b'\0', b' ').decode(errors='replace').split())
        except OSError:                                 # gone, or not ours to look at
            continue
        out.append((pid, cmd[:120]))
    return out


class RealSenseCamera:
    """Blocking capture from a RealSense. Pass `pose_fn` (usually robot.camera) to stamp each frame
    with the camera pose at the moment of capture."""

    def __init__(self, cfg, pose_fn=None):
        c = cfg.section('camera')
        self.serial = str(c.get('serial_no', '')) or None
        self.cell = cfg.get('cell')                     # named when the camera is not there
        self.width = int(c.get('width', 1280))
        self.height = int(c.get('height', 720))
        self.fps = int(c.get('fps', 30))
        self.enable_depth = bool(c.get('enable_depth', False))
        # The DEPTH stream's own resolution (it is aligned to the color image's afterwards). Not the
        # color's: a D415 at 1280x720 depth cannot see closer than ~0.45 m. Measured 2026-10-29 from
        # the iiwa's pick view, the bench 0.35 m away: 43% of the pixels read, ~0.1 m of noise at
        # 1280x720; 98% and 0.7 mm at 848x480 (Intel's recommended D4xx depth mode).
        self.depth_size = (int(c.get('depth_width', 848)), int(c.get('depth_height', 480)))
        self.pose_fn = pose_fn
        # Called with every Frame capture() returns, in the caller's thread (the live view's frame
        # sender, robot/live_view.py). One that raises is logged, never the capture's failure.
        self.listeners = []
        # One capture at a time: the grasp recorder and the cable tracker capture from threads of
        # their own, and the pipeline's wait_for_frames is not for two callers at once.
        self._capture_lock = threading.Lock()
        from ..robot.backends import simulated_perception
        self.dry_run = simulated_perception(cfg)      # a dry run, or a simulated cell (no camera)
        # RECONNECT. A USB camera that drops out mid-run used to end the run: wait_for_frames
        # raises, and nothing caught it. Losing an hour of a cycle test to a nudged cable is a
        # bad trade for a fault that fixes itself the moment the plug goes back in -- so the
        # default is to WAIT, indefinitely, and carry on where it left off.
        rc = c.get('reconnect', {}) or {}
        self.reconnect_enabled = bool(rc.get('enabled', True))
        self.reconnect_interval_s = float(rc.get('retry_interval_s', 2.0))
        self.reconnect_max_wait_s = float(rc.get('max_wait_s', 0.0))   # 0 = forever

        self._align = None
        if self.dry_run:
            log.warning('DRY RUN: the camera returns a black frame.')
            self.pipeline = None
            self.K = np.array([[600.0, 0, self.width / 2],
                               [0, 600.0, self.height / 2],
                               [0, 0, 1.0]])
            self.D = np.zeros(5)
            return

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError('pyrealsense2 is not installed. `pip install pyrealsense2`') from exc
        self._rs = rs

        self.pipeline = rs.pipeline()
        profile = self._start(rs)

        cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = cp.get_intrinsics()
        # Read the ACTUAL resolution back from the device -- after a fallback it may differ from
        # what the config asked for, and K/width/height must describe what we're really getting.
        self.width, self.height, self.fps = cp.width(), cp.height(), cp.fps()
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]])
        self.D = np.asarray(intr.coeffs, dtype=float)
        log.info('RealSense %s streaming %dx%d @ %d fps (fx=%.1f, fy=%.1f).',
                 self.serial or '(first found)', self.width, self.height, self.fps,
                 intr.fx, intr.fy)

        if self.enable_depth:
            # The depth stream comes from the left IR imager, ~15 mm from the color one and with its
            # own intrinsics. Reprojected onto the color image, a depth pixel and a color pixel are
            # the same ray, and Frame's one K serves both.
            self._align = rs.align(rs.stream.color)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            log.info('Depth on: %dx%d, aligned to color (%.4g m a unit).', *self.depth_size, self.depth_scale)

        try:
            for _ in range(5):              # auto-exposure settle
                self.pipeline.wait_for_frames()
        except RuntimeError as exc:
            # STARTED BUT NO FRAMES. The D415 dropped off USB and came back (2026-11-05, 15:15) able
            # to stream color alone and depth alone but never the pair: every run's start timed out
            # here, after the arm had connected, until the camera was hardware-reset.
            log.warning('Camera: started, but no frame within 5 s (%s) -- resetting the camera and '
                        'starting it again.', exc)
            self._restart(rs, exc)

    def _restart(self, rs, first_exc, tries=4):
        """Reset the camera and start it again, up to `tries` times (one reset was not always
        enough: 2026-11-05 17:51, the start right after it failed too). Refuses a camera that comes
        back in another mode -- every pose is measured through K."""
        K0, last = self.K.copy(), first_exc
        for k in range(1, tries + 1):
            self._hardware_reset(rs)
            try:
                self.pipeline = rs.pipeline()
                cp = self._start_color(rs).get_stream(rs.stream.color).as_video_stream_profile()
                intr = cp.get_intrinsics()
                same = (cp.width(), cp.height()) == (self.width, self.height) and np.allclose(
                    [intr.fx, intr.fy, intr.ppx, intr.ppy], [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]], atol=1e-6)
                if same:
                    for _ in range(5):
                        self.pipeline.wait_for_frames()
            except Exception as exc:                    # noqa: BLE001 -- any device fault: reset again
                last = exc
                log.warning('Camera: not streaming after reset %d of %d (%s: %s).', k, tries,
                            type(exc).__name__, exc)
                continue
            if not same:
                raise RuntimeError('the camera came back from its reset in another mode') from first_exc
            log.info('Camera: streaming after %d reset%s.', k, '' if k == 1 else 's')
            return
        raise RuntimeError(f'the camera would not stream after {tries} resets (last: {last}). Unplug and '
                           'replug its USB cable.') from first_exc

    def _device(self, rs):
        """The configured camera (camera.serial_no) -- or the first one, if none is pinned -- or None
        when it is not on the bus."""
        try:
            for d in rs.context().query_devices():
                if not self.serial or d.get_info(rs.camera_info.serial_number) == self.serial:
                    return d
        except Exception:                               # noqa: BLE001 -- mid-enumeration
            pass
        return None

    def _wait_for_device(self, rs, wait_s):
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            if self._device(rs) is not None:
                time.sleep(2.0)                         # enumerated; let its sensors come up
                return True
            time.sleep(0.5)
        return False

    def _hardware_reset(self, rs, wait_s=20.0):
        """Reset the device (as replugging it would) and wait until it is on the bus again."""
        try:
            self.pipeline.stop()
        except Exception:                               # noqa: BLE001 -- not running
            pass
        dev = self._device(rs)
        if dev is None:
            return False
        dev.hardware_reset()
        time.sleep(3.0)                                 # it is still listed until it drops off
        return self._wait_for_device(rs, wait_s)

    def _start(self, rs, tries=4):
        """Start the stream at start-up, resetting a camera that is there and will not start.

        2026-11-10, after a replug: the D415 refused its 1280x720 @ 30 -- and the fallback's
        1920x1080 @ 30 -- and the run ended in the camera's start; a minute later both started. A
        start that fails is now reset and retried like one that starts but sends no frames (__init__,
        _restart). Not reset: a camera that is not on the bus (nothing to reset), and one another
        process has open (a reset would pull it out from under that process)."""
        try:
            return self._start_color(rs)
        except RuntimeError as exc:
            first = exc
        holders = _realsense_holders()
        if holders:
            raise RuntimeError(
                'the camera would not start (%s): another process has it open -- %s. Stop that '
                'process (an earlier run still closing?) and start again.'
                % (first, '; '.join(f'pid {p}: {cmd}' for p, cmd in holders))) from first
        if self._device(rs) is None and not self._wait_for_device(rs, 20.0):
            # 2026-11-10 17:04: a run started without --cell iiwa14_bench got bnc_assembly's default
            # cell, the UR's -- its D405's serial, on the iiwa's PC. Naming the cell says so.
            raise RuntimeError(
                'camera %s (cell %s) is not on the USB bus (found: %s). Is that the right cell -- '
                'e.g. --cell iiwa14_bench? If it is, check the camera\'s cable: a USB 3 port, '
                'straight into the PC.'
                % (self.serial or '(any)', self.cell or '(none)',
                   ', '.join(self._serials(rs)) or 'no RealSense')) from first
        log.warning('Camera: would not start (%s) -- resetting it and trying again.', first)
        last = first
        for k in range(1, tries + 1):
            self._hardware_reset(rs)
            try:
                self.pipeline = rs.pipeline()
                profile = self._start_color(rs)
            except RuntimeError as exc:
                last = exc
                log.warning('Camera: not starting after reset %d of %d (%s).', k, tries, exc)
                continue
            log.info('Camera: started after %d reset%s.', k, '' if k == 1 else 's')
            return profile
        raise RuntimeError(f'the camera would not start after {tries} resets (last: {last}). Unplug and '
                           'replug its USB cable (a USB 3 port, straight into the PC).') from first

    def _serials(self, rs):
        try:
            return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
        except Exception:                               # noqa: BLE001
            return []

    def _start_color(self, rs):
        """Start the pipeline in the configured color mode -- or, if the camera does not OFFER that
        mode, in the best one it does.

        The fallback is for a config written for another camera: the D405's color sensor exposes
        its own set of (resolution, fps, format) profiles, and an unsupported combination makes
        pipeline.start() raise 'Couldn't resolve requests'. A mode the camera offers that will not
        start is a device fault, not a config one: that raises, for the caller to reset the camera
        (2026-11-10: falling back from it only started a second mode that failed the same way)."""
        def start(w, h, f):
            cfg = rs.config()
            if self.serial:
                cfg.enable_device(self.serial)
            if self.enable_depth:
                cfg.enable_stream(rs.stream.depth, *self.depth_size, rs.format.z16, self.fps)
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            return self.pipeline.start(cfg)

        # 1. Exactly what the config asked for.
        try:
            return start(self.width, self.height, self.fps)
        except RuntimeError as exc:
            first = exc
        modes = self._color_modes(rs)
        if not modes or (self.width, self.height, self.fps) in modes:
            raise first                     # not on the bus, or a mode it has: a device fault
        log.warning('Color %dx%d @ %d fps is not a mode this camera offers (%s); auto-selecting one '
                    'it does.', self.width, self.height, self.fps, first)

        # 2. Auto-select from the modes the device ACTUALLY reports -- not librealsense's "default"
        #    profile, which for the D405 is 30 fps and does not actually start. Highest resolution
        #    first, then highest fps at that resolution: more pixels sharpen the cable fit, and the
        #    scan holds each view still so fps barely matters. (The D405 tops out at 1280x720 @ 15.)
        w, h, f = max(modes, key=lambda m: (m[0] * m[1], m[2]))
        log.info('Auto-selected color mode %dx%d @ %d fps.', w, h, f)
        try:
            return start(w, h, f)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not start the camera's color stream in its own reported {w}x{h} @ {f} fps "
                f"either. Supported color modes:\n{self._fmt_modes(modes)}\nOriginal error: {exc}") from exc

    def _color_modes(self, rs):
        """Sorted [(w, h, fps)] color modes the configured camera supports ([] if it is not there)."""
        try:
            dev = self._device(rs)
            if dev is None:
                return []
            return sorted({(v.width(), v.height(), p.fps())
                           for s in dev.query_sensors()
                           for p in s.get_stream_profiles()
                           if p.stream_type() == rs.stream.color and p.is_video_stream_profile()
                           for v in [p.as_video_stream_profile()]})
        except Exception:                           # noqa: BLE001
            return []

    @staticmethod
    def _fmt_modes(modes):
        return '\n'.join(f'  {w}x{h} @ {f} fps' for w, h, f in modes) or '  (none)'

    def capture(self, timeout_ms=5000, reconnect=None):
        """One frame, stamped with the camera pose at capture.

        SURVIVES A DISCONNECT. If the device drops out, this waits for it to come back and
        returns the frame it was asked for, rather than raising and taking the run with it.
        `reconnect=False` restores the old raise-immediately behaviour, which is what a
        best-effort BACKGROUND capture wants: a recorder thread that blocks forever on a
        vanished camera would sit there holding its `with` block open long after the grasp it
        was recording finished."""
        with getattr(self, '_capture_lock', None) or contextlib.nullcontext():   # none without __init__
            frame = self._capture(timeout_ms, reconnect)
        for listener in getattr(self, 'listeners', ()):     # () on one built without __init__
            try:
                listener(frame)
            except Exception as exc:                    # noqa: BLE001 -- the listener's fault only
                log.debug('camera listener %r failed: %s', listener, exc)
        return frame

    def _capture(self, timeout_ms, reconnect):
        if self.dry_run:
            color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return Frame(color, self.K, self.D, time.monotonic(),
                         self.pose_fn() if self.pose_fn else None)
        wait = self.reconnect_enabled if reconnect is None else bool(reconnect)
        try:
            return self._capture_once(timeout_ms)
        except Exception as exc:                        # noqa: BLE001 -- any device fault
            if not wait:
                raise
            log.error('CAMERA LOST (%s). Waiting for it to come back -- the run is PAUSED here, '
                      'the arm is holding position, and nothing is retried until there is a '
                      'frame again.', exc)
        self._await_device()
        return self._capture_once(timeout_ms)

    def _await_device(self):
        """Block until the pipeline yields a frame again, restarting it each attempt.

        RE-READS THE INTRINSICS AND REFUSES A DIFFERENT ONE. A device that comes back at another
        resolution -- or a DIFFERENT CAMERA appearing on the bus when no serial_no is pinned --
        has a different K, and every pose this run has produced or will produce is measured
        through K. Carrying on with the wrong one would not fail; it would quietly return wrong
        answers, which is worse than the disconnect."""
        rs = self._rs
        K0, size0 = np.array(self.K, dtype=float), (self.width, self.height)
        t0, n = time.monotonic(), 0
        while True:
            n += 1
            waited = time.monotonic() - t0
            if self.reconnect_max_wait_s > 0.0 and waited > self.reconnect_max_wait_s:
                raise RuntimeError(
                    'the camera did not come back within camera.reconnect.max_wait_s '
                    f'({self.reconnect_max_wait_s:.0f} s)')
            try:
                self.pipeline.stop()
            except Exception:                           # noqa: BLE001 -- already down
                pass
            time.sleep(self.reconnect_interval_s)
            started = False
            try:
                self.pipeline = rs.pipeline()
                profile = self._start_color(rs)
                started = True
                cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = cp.get_intrinsics()
                self.width, self.height, self.fps = cp.width(), cp.height(), cp.fps()
                self.K = np.array([[intr.fx, 0.0, intr.ppx],
                                   [0.0, intr.fy, intr.ppy],
                                   [0.0, 0.0, 1.0]])
                self.D = np.asarray(intr.coeffs, dtype=float)
                for _ in range(5):                      # let auto-exposure settle again
                    self.pipeline.wait_for_frames()
            except Exception as exc:                    # noqa: BLE001 -- still gone
                if n == 1 or n % 15 == 0:
                    log.warning('  camera still down after %.0f s (%s) -- still waiting.',
                                waited, exc)
                if started:
                    # On the bus and started, yet no frames: restarting the stream does not clear
                    # that (see __init__); a hardware reset does.
                    log.warning('  the camera started but sent no frames -- resetting it.')
                    self._hardware_reset(rs)
                continue
            if (self.width, self.height) != size0 or not np.allclose(self.K, K0, atol=1e-6):
                raise RuntimeError(
                    'the camera came back DIFFERENT: %dx%d fx=%.1f fy=%.1f, was %dx%d fx=%.1f '
                    'fy=%.1f. Every pose is measured through K, so continuing would return '
                    'wrong answers rather than fail. Pin camera.serial_no if more than one '
                    'device is on the bus, and re-check the mode.'
                    % (self.width, self.height, self.K[0, 0], self.K[1, 1],
                       size0[0], size0[1], K0[0, 0], K0[1, 1]))
            log.info('CAMERA BACK after %.0f s (%dx%d, fx=%.1f) -- resuming.',
                     time.monotonic() - t0, self.width, self.height, self.K[0, 0])
            return

    def _capture_once(self, timeout_ms):
        """One frame, no reconnect handling. Raises if the device is not there."""
        # DRAIN the pipeline's buffered frames first, THEN wait for a genuinely new one. The
        # pipeline keeps producing frames at 15 fps while the arm moves and settles, so a plain
        # wait_for_frames() can hand back a STALE frame captured mid-move -- which we would then
        # label with the current (settled) pose, corrupting the ray geometry the fusion depends on.
        # (The ROS stack avoided this differently: it looked the pose up at the image's OWN
        # timestamp via tf2. Here we instead guarantee the frame is fresh, so "pose now" == "pose of
        # this frame".) Draining discards the mid-move backlog; the wait then returns a frame
        # exposed after the arm has come to rest, matching the pose we sample right after.
        while self.pipeline.poll_for_frames():
            pass
        frames = self.pipeline.wait_for_frames(timeout_ms)
        T_base_cam = self.pose_fn() if self.pose_fn else None
        if self._align is not None:
            frames = self._align.process(frames)
        stamp = time.monotonic()

        cf = frames.get_color_frame()
        if not cf:
            raise RuntimeError('the pipeline returned no color frame')
        color = np.asanyarray(cf.get_data())
        depth = None
        if self.enable_depth:
            d = frames.get_depth_frame()
            if d:
                depth = np.asanyarray(d.get_data()).astype(np.float32) * np.float32(self.depth_scale)
        return Frame(color, self.K, self.D, stamp, T_base_cam, depth)

    def preview(self):
        """The newest colour image (BGR, a copy), or None -- for a live preview (robot/live_view.py
        CameraPreview): no drain-and-wait, no depth, no alignment, no listeners, no pose. Taken only
        when no capture is under way, so a capture never waits on it; None in a dry run, closed, or
        if the device does not answer."""
        lock = getattr(self, '_capture_lock', None)
        if getattr(self, 'dry_run', True) or getattr(self, 'pipeline', None) is None or lock is None:
            return None
        if not lock.acquire(blocking=False):
            return None
        try:
            if getattr(self, '_closed', False):
                return None
            newest = None
            while True:
                fs = self.pipeline.poll_for_frames()
                if not fs:
                    break
                newest = fs
            if newest is None:
                newest = self.pipeline.wait_for_frames(200)
            cf = newest.get_color_frame()
            return None if not cf else np.asanyarray(cf.get_data()).copy()
        except Exception:                            # noqa: BLE001 -- a preview, never an error
            return None
        finally:
            lock.release()

    def close(self):
        with getattr(self, '_capture_lock', None) or contextlib.nullcontext():
            self._closed = True                      # a preview after this takes nothing
            if self.pipeline is not None:
                self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
