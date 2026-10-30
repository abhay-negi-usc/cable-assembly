"""robot/live_view.py: the robot's state streamed to a viewer (sim/scripts/live_view.py) while a
task runs -- what a datagram carries, that it follows the REAL FRI arm without disturbing it, and
that a missing viewer or a failing read never reaches the task."""

import json
import os
import socket
import time

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.config import Config
from cable_assembly.robot import Robot, make_arm
from cable_assembly.robot import live_view
from cable_assembly.robot.gripper import Robotiq2F85
from cable_assembly.robot.kinematics import iiwa14

FAKE_PYFRI = os.path.join(os.path.dirname(__file__), 'fake_pyfri.py')


@pytest.fixture
def viewer():
    """A UDP socket standing in for the viewer, on a free port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    s.settimeout(2.0)
    yield s
    s.close()


def _live(port, rate_hz=100.0):
    return {'live_view': {'enabled': True, 'host': '127.0.0.1', 'port': port, 'rate_hz': rate_hz}}


def _recv(sock):
    return json.loads(sock.recv(65536))


def _latest(sock, settle_s=0.1):
    """The newest datagram after `settle_s`, the older ones drained."""
    time.sleep(settle_s)
    sock.setblocking(False)
    msg = None
    try:
        while True:
            msg = json.loads(sock.recv(65536))
    except BlockingIOError:
        pass
    finally:
        sock.settimeout(2.0)
    return msg if msg is not None else _recv(sock)


def test_off_unless_the_config_turns_it_on():
    arm = make_arm(Config({'robot': {'type': 'iiwa14', 'dry_run': True}}))
    assert live_view.start(Config({}), arm) is None
    assert live_view.start(Config({'live_view': {'enabled': False}}), arm) is None


def test_a_datagram_carries_the_joints_the_gripper_and_the_phase(viewer):
    port = viewer.getsockname()[1]
    cfg = Config({'robot': {'type': 'iiwa14', 'dry_run': True}, 'settle_s': 0.0, **_live(port)})
    arm = make_arm(cfg)
    gripper = Robotiq2F85(cfg)                         # the dry-run gripper
    pub = live_view.start(cfg, arm, gripper)
    try:
        msg = _recv(viewer)
        assert msg['v'] == 1 and msg['robot'] == 'iiwa14' and msg['dry_run'] is True
        assert msg['joints'] == list(iiwa14.JOINT_NAMES)
        assert np.allclose(msg['q'], arm.q())
        assert msg['gripper'] == gripper.open_counts and msg['phase'] == ''

        q = np.radians([-27, 50, 66, -96, 120, -57, -123])
        assert arm.move_j(q)
        gripper.go_to(140)
        arm.set_speed_scale(0.5, 'scan')
        msg = _latest(viewer)
        assert np.allclose(msg['q'], q) and msg['gripper'] == 140 and msg['phase'] == 'scan'
        arm.set_speed_scale(1.0)                       # an unnamed rescale keeps the phase
        assert _latest(viewer)['phase'] == 'scan'
    finally:
        pub.close()


def test_the_viewer_is_left_where_the_arm_ended(viewer):
    """A dry run's moves are instant: a whole task can end between two samples. close() sends a
    last one, so the viewer does not keep showing the start."""
    port = viewer.getsockname()[1]
    cfg = Config({'robot': {'type': 'iiwa14', 'dry_run': True}, 'settle_s': 0.0, **_live(port, 1.0)})
    arm = make_arm(cfg)
    pub = live_view.start(cfg, arm)
    q = np.radians([-27, 50, 66, -96, 120, -57, -123])
    assert arm.move_j(q)
    arm.set_speed_scale(1.8, 'reset')
    pub.close()
    msg = _latest(viewer, settle_s=0.0)
    assert np.allclose(msg['q'], q) and msg['phase'] == 'reset'


def test_the_gripper_position_is_the_one_the_task_last_read():
    """The publisher never touches the serial bus: it shows what the task's own reads saw."""
    g = Robotiq2F85.__new__(Robotiq2F85)
    assert g.last_counts is None                       # nothing read yet

    class _Regs:
        registers = [0x31 << 8 | 0, 0 << 8 | 0xFF, 187 << 8 | 0x22]   # gPO 187, current 0x22

        @staticmethod
        def isError():
            return False

    g._call = lambda *a, **k: _Regs()
    assert g._read()['pos'] == 187
    assert g.last_counts == 187


def test_no_viewer_and_a_failing_read_never_reach_the_task():
    class FlakyArm:
        model, joint_names, dry_run, phase = iiwa14, list(iiwa14.JOINT_NAMES), False, ''
        calls = 0

        def q(self):
            FlakyArm.calls += 1
            if FlakyArm.calls % 2:
                raise RuntimeError('no state from the FRI driver')
            return list(iiwa14.HOME)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:   # a port nobody listens on
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    pub = live_view.LivePublisher(FlakyArm(), None, '127.0.0.1', port, rate_hz=200.0)
    time.sleep(0.2)
    assert pub._thread.is_alive() and FlakyArm.calls > 10
    pub.close()
    pub.close()                                        # idempotent
    assert not pub._thread.is_alive()


def test_it_follows_the_real_fri_arm_through_a_move(viewer):
    """IiwaFRIArm, its driver process and shared memory, against the fake KUKA controller in real
    time: the stream tracks the move as it happens, and the move is unaffected by it."""
    port = viewer.getsockname()[1]
    cfg = Config({'robot': {'type': 'iiwa14', 'fri': {
        'sdk': FAKE_PYFRI, 'rt_priority': 0, 'connect_timeout_s': 10.0}},
        'settle_s': 0.0, **_live(port, rate_hz=200.0)})
    arm = make_arm(cfg)
    pub = live_view.start(cfg, arm)
    try:
        q0 = np.asarray(arm.q())
        q1 = q0 + np.radians([10, -5, 8, 6, -10, 12, 15])
        assert arm.move_j(q1, speed=0.5, accel=2.0)
        assert np.allclose(arm.q(), q1, atol=1e-3)
        seen = []
        viewer.setblocking(False)
        try:
            while True:
                seen.append(json.loads(viewer.recv(65536)))
        except BlockingIOError:
            pass
        seqs = [m['seq'] for m in seen]
        assert seqs == sorted(seqs) and len(seen) > 50
        progress = [np.dot(np.asarray(m['q']) - q0, q1 - q0) / np.dot(q1 - q0, q1 - q0) for m in seen]
        assert min(progress) < 0.05 and max(progress) > 0.99      # start to end ...
        assert any(0.3 < p < 0.7 for p in progress)                # ... and in between
        assert np.all(np.diff(progress) > -1e-3)                   # never backwards
    finally:
        pub.close()
        arm.disconnect()


def test_the_iiwa_bench_streams_and_stops_before_the_arm_goes(viewer):
    port = viewer.getsockname()[1]
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', f'live_view.port={port}'])
    robot = Robot(cfg)
    try:
        assert robot.live is not None
        msg = _recv(viewer)
        assert msg['robot'] == 'iiwa14' and msg['gripper'] is not None
    finally:
        robot.close()
    assert not robot.live._thread.is_alive()
