"""CableTracker -- the BNC connector and its cable, modelled once the junction is found and kept up
to date as the cable is picked up.

    tracker = CableTracker(cfg, robot, camera, robot.live_frames)
    scanner.tracker = tracker          # Splat3DScanner calls tracker.fit(...) when its pose holds
    ...                                # the pick: the tracker runs in a thread of its own
    tracker.stop()

THE MODEL (perception/cable_model.py): the plug's CAD fitted rigidly to the orbit's connector
points from the junction pose, and the cable as a Warp rod pinned where it leaves the strain
relief, resting on what the camera saw (the scene without the cable), fitted to the orbit's cable
points.

THROUGH THE PICK, `rate_hz` times a second (update()):
  * the CONNECTOR is carried by the gripper once it closes near it (the fingertip within
    `attach_mm` of the connector, the Robotiq past `closed_counts`), and left where it is when it
    opens again; the rod's start follows it;
  * `capture_hz` times a second the camera takes a frame -- which also goes to the live view's
    camera layer, so the splat updates as the cable moves -- and the depth points near the rod that
    stand clear of the surface it lay on are the cable now: the rod is pulled onto them. (The
    D415 sees nothing nearer than 0.3 m: the cable just below the fingers is the rod's alone.)
  * the rod steps: gravity, its length and stiffness, the surface, those points;
  * `overlay_hz` times a second the model goes to the live view (overlay 'cable_model').

It observes; nothing the pick decides waits on it, and a fault in it only stops it.
"""

import threading
import time

import numpy as np

from .. import log as urlog

log = urlog.get('cable-model')


class CableTracker:
    def __init__(self, cfg, robot, camera, sender=None):
        c = dict(cfg.section('cable_model') or {}) if cfg.get('cable_model') else {}
        self.enabled = bool(c.get('enabled', True))
        self.robot, self.camera, self.sender = robot, camera, sender
        self.length = float(c.get('length_mm', 290.0)) / 1000.0
        self.segment = float(c.get('segment_mm', 10.0)) / 1000.0
        self.radius = float(c.get('radius_mm', 2.5)) / 1000.0
        self.rate_hz = float(c.get('rate_hz', 30.0))
        self.capture_hz = float(c.get('capture_hz', 4.0))
        self.overlay_hz = float(c.get('overlay_hz', 5.0))
        self.attach_m = float(c.get('attach_mm', 25.0)) / 1000.0
        self.closed_counts = int(c.get('closed_counts', 150))
        self.reach_m = float(c.get('reach_mm', 30.0)) / 1000.0
        self.max_shift_m = float(c.get('max_icp_shift_mm', 10.0)) / 1000.0
        self.plug = self.rod = self.T_conn = None
        self.T_tool0_conn = None               # set while the gripper holds the connector
        self.stats = {}
        self._thread, self._stop = None, threading.Event()
        self._t_capture = self._t_overlay = 0.0

    # ------------------------------------------------------------------ the fit at the junction
    def fit(self, T_conn, connector_points, cable_points, surface_points):
        """Model the connector and cable from the orbit: the plug's CAD onto `connector_points`
        from `T_conn`, the rod onto `cable_points` resting on `surface_points` (base frame, m).
        Returns the fitted T_base_connector (T_conn if the CAD fit is not trusted)."""
        from ..perception.cable_model import (
            CableRod,
            ConnectorModel,
            HeightField,
            downsample,
        )
        self.plug = ConnectorModel()
        T, st = self.plug.fit(connector_points, T_conn)
        shift = float(np.linalg.norm(T[:3, 3] - T_conn[:3, 3]))
        turn = float(np.degrees(np.arccos(np.clip(T[:3, 0] @ T_conn[:3, 0], -1.0, 1.0))))
        if not np.isfinite(st['rms_m']) or st['rms_m'] > 0.003 or shift > self.max_shift_m:
            log.warning('Cable model: the plug %s fit is off (rms %.1f mm, %.1f mm / %.1f deg from the junction '
                        'pose) -- keeping the junction pose.', 'CAD' if st['from_cad'] else 'cylinder',
                        1000 * st['rms_m'], 1000 * shift, turn)
            T = np.asarray(T_conn, dtype=float)
        self.T_conn = T
        axis_points = downsample(cable_points, 0.003) - [0.0, 0.0, self.radius]
        self.rod = CableRod(T[:3, 3], -T[:3, 0], length_m=self.length, segment_m=self.segment,
                            radius_m=self.radius, reach_m=self.reach_m)
        self.surface = HeightField(downsample(surface_points, 0.004))
        self.rod.set_environment(self.surface)
        self.rod.trace(axis_points)
        res = self.rod.fit(axis_points)
        self.stats = {'plug_rms_m': st['rms_m'], 'plug_shift_m': shift, 'plug_turn_deg': turn,
                      'plug_from_cad': st['from_cad'], 'rod_residual_m': res, 'cable_points': len(axis_points)}
        log.info('Cable model: the plug (%s) fitted to %d points, %.1f mm rms, %.1f mm / %.1f deg from the junction '
                 'pose; the cable as %d nodes %.0f mm apart on %d points, %.1f mm off them (median).',
                 'CAD' if st['from_cad'] else 'a cylinder', st['points'], 1000 * st['rms_m'], 1000 * shift, turn,
                 self.rod.n, 1000 * self.segment, len(axis_points), 1000 * res)
        self.send(force=True)
        return T

    # ------------------------------------------------------------------ through the pick
    def start(self):
        if self._thread is not None or self.rod is None or not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='cable-model', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self):
        period = 1.0 / self.rate_hz
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                self.update(t0)
                self._stop.wait(max(0.0, period - (time.monotonic() - t0)))
        except Exception as exc:                     # noqa: BLE001 -- an observer: it stops, the pick goes on
            log.warning('Cable model stopped: %s', exc)

    def update(self, now=None, tool=None, counts=None, frame=None):
        """One step (see the module docstring). `tool` (T_base_tool0), `counts` (the Robotiq's)
        and `frame` are read from the robot and camera when not given (tests give them)."""
        now = time.monotonic() if now is None else now
        arm = self.robot.arm
        if tool is None:
            tool = arm.fk(arm.q())
        if counts is None:
            counts = getattr(getattr(self.robot, 'gripper', None), 'last_counts', None)
        self._hold(tool, counts)
        T = self.T_conn
        self.rod.pin(T[:3, 3], -T[:3, 0])
        if frame is None and self.camera is not None and now - self._t_capture >= 1.0 / self.capture_hz:
            self._t_capture = now
            try:
                frame = self.camera.capture(reconnect=False)     # a vanished camera must not block it
            except TypeError:                                   # a camera without that option
                frame = self.camera.capture()
        if frame is not None:
            self.stats['on_points'] = self.rod.set_targets(self._cable_now(frame, tool))
        self.rod.step(1.0 / self.rate_hz)
        if now - self._t_overlay >= 1.0 / self.overlay_hz:
            self._t_overlay = now
            self.send()

    def _hold(self, tool, counts):
        """Attach the connector to the tool when the gripper closes near it; let go when it opens."""
        closed = counts is not None and int(counts) >= self.closed_counts
        if self.T_tool0_conn is None and closed:
            tip = (tool @ self.robot.T_tool0_fingertip)[:3, 3]
            a, o = self.T_conn[:3, 0], self.T_conn[:3, 3]
            s = float(np.clip((tip - o) @ a, 0.0, 0.05))         # the nearest point on the plug's axis
            if np.linalg.norm(tip - (o + s * a)) < self.attach_m:
                self.T_tool0_conn = np.linalg.inv(tool) @ self.T_conn
                log.info('Cable model: the gripper holds the connector; carrying it.')
        elif self.T_tool0_conn is not None and not closed and counts is not None:
            self.T_tool0_conn = None
            log.info('Cable model: the gripper let go of the connector.')
        if self.T_tool0_conn is not None:
            self.T_conn = tool @ self.T_tool0_conn

    def _cable_now(self, frame, tool):
        """The frame's depth points that are the cable now: near the rod, clear of the surface it
        lay on (bench, fixture), the robot left out -- taken `radius` down, onto its axis."""
        from ..perception.scene import OccupancyScene
        if getattr(frame, 'depth', None) is None or frame.T_base_cam is None:
            return None
        f = {'depth_m': frame.depth, 'K': frame.K, 'T_base_cam': np.asarray(frame.T_base_cam, dtype=float),
             'T_base_tool0': tool, 'T_tool0_fingertip': self.robot.T_tool0_fingertip,
             'joints_rad': np.asarray(self.robot.arm.q(), dtype=float), 'robot': self.robot.arm.model.NAME}
        p = OccupancyScene([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], voxel_m=0.5).points(f, stride=2)
        if not len(p):
            return None
        from scipy.spatial import cKDTree
        d, _ = cKDTree(self.rod.nodes()).query(p, distance_upper_bound=self.reach_m)
        p = p[np.isfinite(d)]
        p = p[p[:, 2] > self.surface.at(p[:, :2]) + 1.2 * self.radius]
        return p - [0.0, 0.0, self.radius] if len(p) else None

    # ------------------------------------------------------------------ to the live view
    def send(self, force=False):
        if self.sender is None or self.rod is None:
            return
        T = self.T_conn
        plug = self.plug.samples[:: max(1, len(self.plug.samples) // 400)] @ T[:3, :3].T + T[:3, 3]
        held = 'carried by the gripper' if self.T_tool0_conn is not None else 'where it lies'
        try:
            self.sender.send_overlay('cable_model', [
                {'type': 'line', 'points': self.rod.nodes().round(4).tolist(), 'color': [1.0, 0.45, 0.1]},
                {'type': 'points', 'points': plug.round(4).tolist(), 'color': [1.0, 0.85, 0.2], 'radius': 0.0008},
                {'type': 'pose', 'T': T.tolist(), 'length': 0.02}],
                [(f"cable model: connector {held}; cable {self.rod.n} nodes, "
                  f"{self.stats.get('on_points', '-')} on points now")])
        except Exception as exc:                                     # noqa: BLE001 -- drawing only
            log.debug('cable model overlay: %s', exc)
