"""skills/splat_pick.py: the connector found from an orbit's depth. A simulated arm (dry run)
carries a camera that renders, with exact depth, the bench and a connector lying on it; a stand-in
SAM3 reports the junction pixel and the connector's direction as the real one does. Known answers:
the fused junction lands over the true one where the one-image plane estimate is off along its
ray, the heading is the connector's, the arm ends where it started -- and with too few views or no
depth the plane estimate is what comes back."""

import builtins
import os
import time

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.perception import depth as D
from cable_assembly.perception.camera import Frame
from cable_assembly.robot import Robot
from cable_assembly.skills.splat_pick import SplatScanner

W, H = 320, 180
K = np.array([[232.0, 0.0, 160.0], [0.0, 232.0, 90.0], [0.0, 0.0, 1.0]])
TOP = 0.012                                  # the connector's top above the bench
HEADING = np.radians(80.0)                   # the connector's direction, from base +x
J_XY = np.array([0.60, 0.11])                # the junction, ~60 mm off the scan view's centre


def _box(bench_z):
    """The cable and connector as one 10 mm wide bar on the bench, the junction in its middle."""
    h = np.array([np.cos(HEADING), np.sin(HEADING), 0.0])
    R = np.stack([h, np.cross([0.0, 0.0, 1.0], h), [0.0, 0.0, 1.0]], axis=1)   # box axes in base
    centre = np.array([*J_XY, bench_z + TOP / 2])
    return R, centre, np.array([0.06, 0.005, TOP / 2])


def _render(T, bench_z):
    """Depth (along the optical axis) of the bench and the bar, and which pixels are the bar."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3) @ T[:3, :3].T
    o = T[:3, 3]
    t = np.where(d[:, 2] < 0, (bench_z - o[2]) / d[:, 2], np.inf)
    R, c, half = _box(bench_z)
    ol, dl = (o - c) @ R, d @ R
    with np.errstate(divide='ignore', invalid='ignore'):
        t0, t1 = (-half - ol) / dl, (half - ol) / dl
    tn, tf = np.nanmax(np.minimum(t0, t1), axis=1), np.nanmin(np.maximum(t0, t1), axis=1)
    hit = (tn <= tf) & (tf > 0) & (tn < t)
    t = np.where(hit, tn, t)
    return np.where(np.isfinite(t), t, 0.0).reshape(H, W).astype(np.float32), hit.reshape(H, W)


class BenchCamera:
    def __init__(self, robot, bench_z, depth=True):
        self.robot, self.bench_z, self.depth, self.listeners = robot, bench_z, depth, []

    def capture(self):
        T = self.robot.camera()
        depth, bar = _render(T, self.bench_z)
        color = np.full((H, W, 3), 200, np.uint8)
        color[bar] = 30
        return Frame(color, K, np.zeros(5), time.monotonic(), T, depth if self.depth else None)


class StandInSAM3:
    """detect_all as junction mode returns it: the junction pixel (the top of the junction), the
    connector's direction in the image, the bar as the cable mask. `seen` limits how many calls
    find it."""

    last_debug = None

    def __init__(self, bench_z, seen=None):
        self.top = np.array([*J_XY, bench_z + TOP])
        self.seen, self.calls = seen, 0

    def detect_all(self, frame, n):
        self.calls += 1
        if self.seen is not None and self.calls > self.seen:
            return []
        h = np.array([np.cos(HEADING), np.sin(HEADING), 0.0])
        (uv, uv2), _ = D.project(frame, np.stack([self.top, self.top + 0.02 * h]))
        yaw = float(np.arctan2(uv2[1] - uv[1], uv2[0] - uv[0]))
        mask = frame.color[..., 0] < 100
        return [{'junction': (float(uv[0]), float(uv[1]), yaw), 'cable_mask': mask, 'ends': [],
                 'size': int(mask.sum())}]

    def label_cables(self, frame, cables):
        pass


@pytest.fixture
def cell(tmp_path, monkeypatch):
    monkeypatch.setattr(builtins, 'input', lambda prompt='': '1')     # "which cable": the only one
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'scan.mode=splat',
                                  'live_view.enabled=false', f'data_dir={tmp_path}',
                                  'splat_pick.settle_s=0', f'last_camera_image_path={tmp_path}/last.png'])
    robot = Robot(cfg, with_gripper=False, needs=())
    q_pick = np.radians(np.asarray(cfg.get('pick_joints_deg'), dtype=float))
    assert robot.arm.move_j(q_pick, label='pick pose')
    yield cfg, robot, q_pick, float(cfg.get_path('ground_plane.z_m')), str(tmp_path)
    robot.close()


def _plane_estimate(scanner, robot, bench_z):
    """Where the one-image scanner puts the junction: its pixel's ray met with the bench."""
    frame = BenchCamera(robot, bench_z).capture()
    u, v, _ = StandInSAM3(bench_z).detect_all(frame, 1)[0]['junction']
    return scanner._project((u, v), frame)


def test_the_orbit_finds_the_junction_over_the_connector(cell):
    cfg, robot, q_pick, bench_z, data = cell
    sam3 = StandInSAM3(bench_z)
    scanner = SplatScanner(robot, BenchCamera(robot, bench_z), sam3, None, cfg, data_root=data)
    plane = _plane_estimate(scanner, robot, bench_z)
    T = scanner.scan()
    assert T is not None
    fused_err = np.linalg.norm(T[:2, 3] - J_XY)
    plane_err = np.linalg.norm(plane[:2] - J_XY)
    assert fused_err < 0.3e-3, fused_err                                  # on the connector's axis
    assert plane_err > 1.0e-3 and plane_err > 5 * fused_err, plane_err    # off along its ray
    assert T[2, 3] == pytest.approx(bench_z)                              # the scanner's kind of pose
    assert np.degrees(np.arctan2(T[1, 0], T[0, 0])) == pytest.approx(np.degrees(HEADING), abs=0.5)
    assert scanner.last['seen'] == scanner.last['views'] == 6             # the scan view and 5 of 6 round
                                                                          # it (pan 180 is out of reach)
    assert scanner.last['top_m'] == pytest.approx(TOP, abs=0.5e-3)
    np.testing.assert_allclose(robot.arm.q(), q_pick, atol=1e-6)          # back at the scan view
    views = sorted(f for f in os.listdir(scanner.image_dir) if f.startswith('splat_view_'))
    assert len(views) == 6                                                # kept, for scene_build


def test_too_few_views_finding_it_leave_the_plane_estimate(cell):
    cfg, robot, _, bench_z, data = cell
    scanner = SplatScanner(robot, BenchCamera(robot, bench_z), StandInSAM3(bench_z, seen=3), None, cfg,
                           data_root=data)
    plane = _plane_estimate(scanner, robot, bench_z)
    T = scanner.scan()                     # 1 call to choose, then the scan view and one orbit view
    np.testing.assert_allclose(T[:3, 3], plane, atol=1e-9)
    assert scanner.last is None


def test_without_depth_it_is_the_ground_plane_scanner(cell):
    cfg, robot, q_pick, bench_z, data = cell
    sam3 = StandInSAM3(bench_z)
    scanner = SplatScanner(robot, BenchCamera(robot, bench_z, depth=False), sam3, None, cfg, data_root=data)
    plane = _plane_estimate(scanner, robot, bench_z)
    T = scanner.scan()
    np.testing.assert_allclose(T[:3, 3], plane, atol=1e-9)
    np.testing.assert_allclose(robot.arm.q(), q_pick, atol=1e-9)          # nothing moved
    assert sam3.calls == 1


# ---------------------------------------------------------------------------- splat_3d: no plane

R_CONN, L_CONN, R_CABLE = 0.008, 0.035, 0.0025
TILT = np.radians(15.0)                     # the connector's axis, rising toward its tip
RAISED = 0.040                              # its junction this far above the bench (on a fixture)


def _axis(bench_z):
    a = np.array([np.cos(TILT) * np.cos(HEADING), np.cos(TILT) * np.sin(HEADING), np.sin(TILT)])
    return np.array([*J_XY, bench_z + RAISED]), a


def _cylinder(o, d, O, a, r, s0, s1):
    """Ray parameter t of the first hit on a closed cylinder (side and end discs); inf where none."""
    w = o - O
    dp, wp = d - np.outer(d @ a, a), w - (w @ a) * a
    A, B, C = np.sum(dp * dp, axis=1), 2.0 * dp @ wp, wp @ wp - r * r
    disc = B * B - 4.0 * A * C
    with np.errstate(invalid='ignore', divide='ignore'):
        t = (-B - np.sqrt(disc)) / (2.0 * A)
        s = w @ a + t * (d @ a)
        side = np.where((disc >= 0) & (t > 0) & (s >= s0) & (s <= s1), t, np.inf)
        best = side
        for sc in (s0, s1):
            tc = ((O + sc * a - o) @ a) / (d @ a)
            p = o + tc[:, None] * d - (O + sc * a)
            best = np.minimum(best, np.where((tc > 0) & (np.sum(p * p, axis=1) <= r * r), tc, np.inf))
    return best


class RaisedCamera:
    """The bench, and a connector + cable on one tilted axis, raised off it: depth, and which
    pixels are the cable and which the connector."""

    def __init__(self, robot, bench_z):
        self.robot, self.bench_z, self.listeners = robot, bench_z, []

    def render(self, T):
        v, u = np.mgrid[0:H, 0:W]
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3) @ T[:3, :3].T
        o = T[:3, 3]
        O, a = _axis(self.bench_z)
        bench = np.where(d[:, 2] < 0, (self.bench_z - o[2]) / d[:, 2], np.inf)
        conn = _cylinder(o, d, O, a, R_CONN, 0.0, L_CONN)
        cable = _cylinder(o, d, O, a, R_CABLE, -0.06, 0.0)
        t = np.minimum(np.minimum(bench, conn), cable)
        is_cable = (cable <= t) & np.isfinite(cable)
        is_conn = (conn <= t) & np.isfinite(conn) & ~is_cable
        return (np.where(np.isfinite(t), t, 0.0).reshape(H, W).astype(np.float32),
                is_cable.reshape(H, W), is_conn.reshape(H, W))

    def capture(self):
        T = self.robot.camera()
        depth, cab, con = self.render(T)
        color = np.full((H, W, 3), 200, np.uint8)
        color[cab] = 30
        color[con] = 60
        return Frame(color, K, np.zeros(5), time.monotonic(), T, depth)


class RaisedSAM3:
    """detect_all for RaisedCamera: the junction pixel where the axis leaves the connector, the
    connector's direction in the image, the cable side and the whole assembly as masks."""

    last_debug = None

    def __init__(self, bench_z, connector=True):
        self.bench_z, self.connector, self.calls = bench_z, connector, 0

    def detect_all(self, frame, n):
        self.calls += 1
        O, a = _axis(self.bench_z)
        (uv, uv2), _ = D.project(frame, np.stack([O, O + 0.02 * a]))
        cable = frame.color[..., 0] == 30
        assembly = cable | (frame.color[..., 0] == 60) if self.connector else cable
        return [{'junction': (float(uv[0]), float(uv[1]), float(np.arctan2(uv2[1] - uv[1], uv2[0] - uv[0]))),
                 'cable_mask': cable, 'assembly_mask': assembly, 'ends': [], 'size': int(assembly.sum())}]

    def label_cables(self, frame, cables):
        pass


def test_splat_3d_finds_a_raised_tilted_connector_where_the_plane_methods_cannot(cell):
    from cable_assembly.skills.splat_pick import Splat3DScanner
    cfg, robot, q_pick, bench_z, data = cell
    O, a = _axis(bench_z)
    s3 = Splat3DScanner(robot, RaisedCamera(robot, bench_z), RaisedSAM3(bench_z), None, cfg, data_root=data)
    T = s3.scan()
    assert s3.pose_on_axis
    assert np.linalg.norm(T[:3, 3] - O) < 1.5e-3, T[:3, 3] - O                # on the axis, at the junction
    assert np.degrees(np.arccos(np.clip(T[:3, 0] @ a, -1, 1))) < 2.0           # tilted as it lies
    assert s3.last['radius_m'] == pytest.approx(R_CONN, abs=1.5e-3)
    np.testing.assert_allclose(robot.arm.q(), q_pick, atol=1e-6)
    # The splat pick on the plane puts it on the bench, level: 40 mm low and 15 deg off.
    robot.arm.move_j(q_pick, label='again')
    sp = SplatScanner(robot, RaisedCamera(robot, bench_z), RaisedSAM3(bench_z), None, cfg, data_root=data)
    Tp = sp.scan()
    assert not getattr(sp, 'pose_on_axis', False)
    assert abs(Tp[2, 3] - O[2]) > 0.035 and abs(T[2, 3] - O[2]) < 1.5e-3
    assert np.degrees(np.arccos(np.clip(Tp[:3, 0] @ a, -1, 1))) > 10.0


def test_splat_3d_without_connector_pixels_falls_back_to_the_plane_fusion(cell):
    from cable_assembly.skills.splat_pick import Splat3DScanner
    cfg, robot, _, bench_z, data = cell
    s3 = Splat3DScanner(robot, RaisedCamera(robot, bench_z), RaisedSAM3(bench_z, connector=False), None, cfg,
                        data_root=data)
    T = s3.scan()
    assert T is not None and not s3.pose_on_axis
    assert T[2, 3] == pytest.approx(bench_z)                                  # the plane fusion's pose
