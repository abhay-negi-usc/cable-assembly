"""The live view's camera layer: every frame the task captures goes to the viewer
(robot/live_view.py FrameSender -> sim/src/cable_assembly_sim/live.py FrameReceiver), which merges
them (sim/src/cable_assembly_sim/rgbd.py LiveMap). The sim-side modules are loaded from their files:
the sim package is not installed in this env, and neither needs Isaac.

What must hold: the camera hands over every frame and a failing listener never fails a capture; a
frame arrives as it was sent; with no viewer the task never waits; and in the merged layer, a new
view replaces what it sees -- the cable's old spot clears -- while what it cannot see stays."""

import importlib.util
import os
import socket
import time

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.config import Config
from cable_assembly.perception.camera import Frame, RealSenseCamera
from cable_assembly.robot import Robot, live_view

SIM_SRC = os.path.join(os.path.dirname(__file__), '..', 'sim', 'src', 'cable_assembly_sim')
BENCH_Z = 0.0062
# A camera 0.35 m above the bench at (0.6, 0.15), looking straight down: optical x along base x,
# y along -y, z along -z.
T_BASE_CAM = np.array([[1.0, 0.0, 0.0, 0.6], [0.0, -1.0, 0.0, 0.15], [0.0, 0.0, -1.0, 0.35], [0, 0, 0, 1.0]])
K = np.array([[116.0, 0.0, 80.0], [0.0, 116.0, 45.0], [0.0, 0.0, 1.0]])
W, H = 160, 90
CAM_TO_BENCH = 0.35 - BENCH_Z


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SIM_SRC, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def rgbd():
    return _load('rgbd')


@pytest.fixture(scope='module')
def live():
    return _load('live')


def _frame(depth, T=T_BASE_CAM, rgb=None):
    """A frame as the receiver hands it over (no T_base_tool0: nothing masked as the gripper)."""
    return {'rgb': np.full((H, W, 3), 128, np.uint8) if rgb is None else rgb,
            'depth_m': depth.astype(np.float32), 'K': K, 'T_base_cam': T}


def _bench(cable=False):
    d = np.full((H, W), CAM_TO_BENCH, np.float32)
    if cable:
        d[40:50, :] -= 0.004          # a 4 mm cable across the view
    return d


# ---------------------------------------------------------------------------- the camera's hook

def test_the_camera_hands_every_frame_to_its_listeners():
    cam = RealSenseCamera(Config({'robot': {'dry_run': True}, 'camera': {}}))
    got = []
    cam.listeners += [got.append, lambda f: 1 / 0]            # the second always raises
    frames = [cam.capture() for _ in range(3)]
    assert got == frames                                     # and the raise reached no one


def test_a_robot_with_frames_on_listens_to_its_camera(tmp_path):
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.frames=true',
                                  'live_view.port=0', 'live_view.frames_port=1'])
    robot = Robot(cfg)
    try:
        assert robot.live_frames is not None
        cam = robot.register_camera(RealSenseCamera(cfg, pose_fn=robot.camera))
        assert cam.listeners == [robot.live_frames]
        t0 = time.monotonic()
        cam.capture()                                        # a dry run's frame has no depth: not sent
        assert time.monotonic() - t0 < 0.5
    finally:
        robot.close()
    assert not robot.live_frames._thread.is_alive()


def test_frames_stay_off_unless_asked_for():
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.port=0',
                                  'live_view.frames=false'])
    robot = Robot(cfg)
    try:
        assert robot.live_frames is None
    finally:
        robot.close()


# ---------------------------------------------------------------------------- the wire

def test_a_frame_arrives_as_it_was_sent(live):
    got = []
    rx = live.FrameReceiver('127.0.0.1', 0, got.append)
    T_tool0_cam = np.array([[1, 0, 0, -0.07], [0, 1, 0, 0], [0, 0, 1, 0.0175], [0, 0, 0, 1.0]])
    T_tool0_tip = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.19], [0, 0, 0, 1.0]])
    tx = live_view.FrameSender('iiwa14', T_tool0_cam, T_tool0_tip, '127.0.0.1', rx.port)
    try:
        rng = np.random.default_rng(1)
        color = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)            # BGR, as captured
        depth = rng.uniform(0.3, 0.6, (180, 320)).astype(np.float32)
        Kf = np.array([[232.5, 0, 160.0], [0, 232.3, 90.0], [0, 0, 1.0]])
        tx(Frame(color, Kf, np.zeros(5), 0.0, T_BASE_CAM, depth))
        t0 = time.monotonic()
        while not got and time.monotonic() - t0 < 3.0:
            time.sleep(0.01)
        assert len(got) == 1
        f = got[0]
        np.testing.assert_array_equal(f['rgb'], color[::2, ::2, ::-1])
        np.testing.assert_allclose(f['depth_m'], depth[::2, ::2], atol=0.0005)   # 1 mm units
        np.testing.assert_allclose(f['K'], [[116.25, 0, 80.0], [0, 116.15, 45.0], [0, 0, 1.0]])
        np.testing.assert_allclose(f['T_base_cam'], T_BASE_CAM)
        np.testing.assert_allclose(f['T_base_tool0'] @ T_tool0_cam, T_BASE_CAM, atol=1e-12)
        np.testing.assert_allclose(f['T_tool0_fingertip'], T_tool0_tip)
        assert f['robot'] == 'iiwa14' and f['seq'] == 0
    finally:
        tx.close()
        rx.close()


def test_with_no_viewer_the_task_never_waits():
    with socket.socket() as s:                               # a port nobody listens on
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    tx = live_view.FrameSender('iiwa14', np.eye(4), None, '127.0.0.1', port)
    frame = Frame(np.zeros((720, 1280, 3), np.uint8), K, np.zeros(5), 0.0, T_BASE_CAM,
                  np.full((720, 1280), 0.4, np.float32))
    t0 = time.monotonic()
    for _ in range(10):
        tx(frame)
    assert (time.monotonic() - t0) / 10 < 0.02               # a strided copy, nothing more
    tx.close()
    assert not tx._thread.is_alive()


# ---------------------------------------------------------------------------- the merged layer

def test_a_new_view_clears_the_cables_old_spot(rgbd):
    m = rgbd.LiveMap(voxel_m=0.002)
    m.add(_frame(_bench(cable=True)))
    above = lambda: (m.positions[:, 2] > BENCH_Z + 0.002).sum()          # noqa: E731
    assert above() > 0                                                  # the cable is in
    stats = m.add(_frame(_bench()))                                     # picked up: bench only
    assert above() == 0
    assert stats['gone'] > 0                                            # seen through: the bench behind
    np.testing.assert_allclose(m.positions[:, 2], BENCH_Z, atol=1e-4)
    n_bench = len(m.positions)
    m.add(_frame(_bench()))                                             # the same view again
    assert abs(len(m.positions) - n_bench) <= 0.01 * n_bench            # updated, not doubled


def test_views_combine_and_a_far_one_does_not_thin_out_a_near_one(rgbd):
    """A point stays until a newer frame updates it (the same voxel) or sees through it. Seen
    again from three times as far -- a third the pixels per millimetre -- the close look's points
    between the far view's stay: the layer is both views, not the last."""
    far = T_BASE_CAM.copy()
    far[2, 3] = BENCH_Z + 3 * CAM_TO_BENCH
    near_only = rgbd.LiveMap(voxel_m=0.002)
    near_only.add(_frame(_bench()))
    far_only = rgbd.LiveMap(voxel_m=0.002)
    far_only.add(_frame(np.full((H, W), 3 * CAM_TO_BENCH, np.float32), T=far))
    m = rgbd.LiveMap(voxel_m=0.002)
    m.add(_frame(_bench()))
    stats = m.add(_frame(np.full((H, W), 3 * CAM_TO_BENCH, np.float32), T=far))
    lo, hi = near_only.positions[:, :2].min(axis=0), near_only.positions[:, :2].max(axis=0)

    def in_near(pts):                                    # inside the close look's footprint
        return int(np.all((pts[:, :2] >= lo) & (pts[:, :2] <= hi), axis=1).sum())

    assert stats['gone'] == 0
    assert in_near(far_only.positions) < 0.2 * len(near_only.positions)  # the far view alone: ~1/9
    assert in_near(m.positions) >= 0.95 * len(near_only.positions)       # both: the near detail stays
    assert len(m.positions) > len(far_only.positions)


def test_what_the_new_view_cannot_see_stays(rgbd):
    m = rgbd.LiveMap(voxel_m=0.002)
    m.add(_frame(_bench()))
    n0 = len(m.positions)
    near = _bench()
    near[:, :40] = 0.15                      # a finger in front of the left quarter
    near[:, 120:] = 0.0                      # no reading on the right quarter
    m.add(_frame(near))
    on_bench = np.abs(m.positions[:, 2] - BENCH_Z) < 1e-3
    # Behind the finger and where there was no reading, the bench stays: about half of it.
    x = m.positions[on_bench, 0]
    assert (x < 0.6 - 0.05).sum() > 0.2 * n0 and (x > 0.6 + 0.05).sum() > 0.2 * n0
    assert not (m.positions[:, 2] > 0.15).any()      # the finger itself is nearer than the D415 reads

    moved = T_BASE_CAM.copy()
    moved[0, 3] += 1.0                       # a view a metre away sees none of it
    m2 = rgbd.LiveMap(voxel_m=0.002)
    m2.add(_frame(_bench()))
    before = m2.positions.copy()
    m2.add(_frame(_bench(), T=moved))
    assert len(m2.positions) == 2 * len(before)


def test_past_the_cap_the_oldest_frame_goes_first(rgbd):
    m = rgbd.LiveMap(voxel_m=0.002, max_points=5000)
    first = T_BASE_CAM.copy()
    second = T_BASE_CAM.copy()
    second[0, 3] += 1.0
    m.add(_frame(_bench(), T=first))
    m.add(_frame(_bench(), T=second))
    assert len(m.positions) == 5000
    assert np.all(m.positions[:, 0] > 1.0)                               # all from the second view
    assert m.last_stats['dropped_oldest'] > 0 and m.version == 2


# ---------------------------------------------------------------------------- one sender at a time

class _HeldArm:
    """An arm reporting fixed joints."""

    def __init__(self, q, dry_run):
        from cable_assembly.robot.kinematics import iiwa14
        self.model, self.joint_names, self.dry_run, self.phase = iiwa14, list(iiwa14.JOINT_NAMES), dry_run, ''
        self._q = list(q)

    def q(self):
        return self._q


def _watch(stream, seconds):
    """Every sample the viewer would show over `seconds`: (sender, q) each poll."""
    shown, t_end = [], time.monotonic() + seconds
    while time.monotonic() < t_end:
        if stream.poll():
            shown.append((stream.sender, tuple(stream.latest['q'])))
        time.sleep(0.005)
    return shown


def test_a_dry_run_never_flashes_into_the_live_arm(live):
    """The glitch the user saw: a dry-run Robot (a rehearsal, the test suite) streaming to the
    viewer's port while the real arm did. Its samples, and its last one on close, must not show."""
    from cable_assembly.robot.kinematics import iiwa14
    stream = live.Stream('127.0.0.1', 0, iiwa14.JOINT_NAMES, hold_s=0.3)
    port = stream.sock.getsockname()[1]
    q_real, q_dry = [0.1] * 7, [-0.5] * 7
    real = live_view.LivePublisher(_HeldArm(q_real, False), None, '127.0.0.1', port, 100.0)
    try:
        assert _watch(stream, 0.2)                                   # the real arm holds the view
        dry = live_view.LivePublisher(_HeldArm(q_dry, True), None, '127.0.0.1', port, 100.0)
        shown = _watch(stream, 0.3)
        dry.close()                                                  # and its last sample on close
        shown += _watch(stream, 0.2)
        assert shown and all(q == tuple(q_real) for _, q in shown)
    finally:
        real.close()
    # Once the real arm has been silent for hold_s, a new sender takes over. (Polling meanwhile, as
    # a viewer does every frame: an unread backlog would be stamped fresh when finally read.)
    _watch(stream, 0.4)
    late = live_view.LivePublisher(_HeldArm(q_dry, True), None, '127.0.0.1', port, 100.0)
    try:
        assert _watch(stream, 0.2)[-1][1] == tuple(q_dry)
    finally:
        late.close()
        stream.close()


def test_a_real_arm_takes_over_from_a_dry_run_at_once(live):
    from cable_assembly.robot.kinematics import iiwa14
    stream = live.Stream('127.0.0.1', 0, iiwa14.JOINT_NAMES, hold_s=5.0)
    port = stream.sock.getsockname()[1]
    dry = live_view.LivePublisher(_HeldArm([-0.5] * 7, True), None, '127.0.0.1', port, 100.0)
    real = None
    try:
        assert _watch(stream, 0.2)[-1][1] == tuple([-0.5] * 7)
        real = live_view.LivePublisher(_HeldArm([0.1] * 7, False), None, '127.0.0.1', port, 100.0)
        shown = _watch(stream, 0.3)
        assert shown[-1][1] == tuple([0.1] * 7)                       # well within hold_s
    finally:
        dry.close()
        if real is not None:
            real.close()
        stream.close()
