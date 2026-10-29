"""The iiwa FRI backend's own guarantees -- the ones the shared Arm contract
(test_arm_backends.py) cannot see.

The Follower is the code that decides every joint position the KUKA controller receives, so it
is held directly: whatever it is commanded, its output stays continuous and under the caps. The
rest runs the real IiwaFRIArm and driver process against tests/fake_pyfri.py, which keeps real
time and aborts the session on a joint-speed violation the way KUKA does.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from cable_assembly.config import Config
from cable_assembly.robot import ArmError, make_arm
from cable_assembly.robot.backends import fri_driver as drv
from cable_assembly.robot.backends.iiwa_fri import wrench_from_torques
from cable_assembly.robot.kinematics import iiwa14

FAKE_PYFRI = os.path.join(os.path.dirname(__file__), 'fake_pyfri.py')
DT = 0.005
LIM = iiwa14.JOINT_LIMITS
V_CAP = iiwa14.VELOCITY_LIMITS * 0.5
Q0 = iiwa14.HOME.copy()


def _follower():
    f = drv.Follower(DT, LIM[:, 0], LIM[:, 1], LIM[:, 0], LIM[:, 1], V_CAP, 4.0, 80.0)
    f.reset(Q0)
    return f


def _run(f, t0, seconds, feed=None):
    """Step `f` for `seconds` of robot time from `t0`; `feed(t)` may command it first each cycle.
    Returns the times and the positions sent."""
    ts, qs = [], []
    t = t0
    for _ in range(int(round(seconds / DT))):
        if feed is not None:
            feed(t)
        qs.append(f.step(t).copy())
        ts.append(t)
        t += DT
    return np.array(ts), np.array(qs)


def _vel_acc(qs):
    v = np.diff(qs, axis=0) / DT
    return v, np.diff(v, axis=0) / DT


# ------------------------------------------------------------------ the Follower
def test_a_move_is_a_straight_joint_line_under_the_limits():
    f = _follower()
    target = Q0 + np.radians([20.0, -10.0, 5.0, 15.0, 0.0, -8.0, 30.0])
    a = np.full(7, 1.2)
    f.command(drv.MOVE, 1, target, V_CAP, a, a / 0.1, 0.0, 0.0, 0.0)
    _, qs = _run(f, 0.0, 4.0)
    assert f.done_seq == 1 and f.at_rest
    assert np.allclose(qs[-1], target, atol=1e-12), 'ends exactly on the target'
    v, acc = _vel_acc(np.vstack([Q0, qs]))
    assert np.all(np.abs(v) <= V_CAP + 1e-9)
    assert np.all(np.abs(acc) <= 1.2 + 1e-6)
    # Phase-synchronised: every sample lies on the line from Q0 to the target.
    d = target - Q0
    s = (qs - Q0) @ d / (d @ d)
    assert np.max(np.abs((qs - Q0) - np.outer(s, d))) < 1e-9


def test_a_jump_in_a_stream_never_reaches_the_robot():
    """A setpoint 20 deg away arriving in one stream period is a jump no arm can make. The
    reference has it; the robot must not."""
    f = _follower()
    a = np.full(7, 4.0)
    far = Q0 + np.radians([20.0, 0, 0, 0, 0, 0, 0])
    f.command(drv.STREAM, 1, far, V_CAP, a, a / 0.02, 0.008, 0.0, 0.0)
    _, qs = _run(f, 0.0, 3.0)
    v, acc = _vel_acc(np.vstack([Q0, qs]))
    assert np.all(np.abs(v) <= V_CAP + 1e-9), 'under the velocity cap'
    assert np.all(np.abs(acc) <= 4.0 + 1e-6), 'under the stream acceleration'
    assert f.at_rest and np.allclose(qs[-1], far, atol=1e-9), 'and it still gets there, and stops'


def test_a_smooth_stream_is_followed_exactly_despite_late_setpoints():
    """Setpoints along a joint-space line, sent every 8 ms with Python-sized jitter and arriving
    at a 5 ms cycle: the arm stays on the line, no faster and no harder than the stream."""
    f = _follower()
    a = np.full(7, 4.0)
    d = np.radians([10.0, -5.0, 0.0, 8.0, 0.0, 3.0, -6.0])

    def path(t):                                   # 0 -> 1 over 1 s, smooth start and stop
        u = min(max(t, 0.0), 1.0)
        return Q0 + d * (3 * u * u - 2 * u ** 3)

    rng = np.random.default_rng(0)
    pending = []
    t_send, seq = 0.0, 0
    while t_send < 1.05:
        pending.append((t_send, path(t_send)))
        t_send += 0.008 + rng.uniform(-0.001, 0.004)

    def feed(t):
        nonlocal seq
        while pending and pending[0][0] <= t:      # read at the next cycle after it was sent
            stamp, q = pending.pop(0)
            seq += 1
            f.command(drv.STREAM, seq, q, V_CAP, a, a / 0.02, 0.008, stamp, t)

    _, qs = _run(f, 0.0, 1.5, feed)
    s = (qs - Q0) @ d / (d @ d)
    assert np.max(np.abs((qs - Q0) - np.outer(s, d))) < 2e-5, 'on the line (rad)'
    v, acc = _vel_acc(np.vstack([Q0, qs]))
    assert np.max(np.abs(v @ d / (d @ d))) < 1.6, 'no faster than the stream (peak 1.5/s)'
    assert np.all(np.abs(acc) <= 4.0 + 1e-6)
    assert f.at_rest and np.allclose(qs[-1], Q0 + d, atol=1e-9)


def test_stop_brakes_within_the_stop_limits_and_holds():
    f = _follower()
    a = np.full(7, 1.2)
    f.command(drv.MOVE, 1, Q0 + np.radians([40.0, 0, 0, 0, 0, 0, 0]), V_CAP, a, a / 0.1,
              0.0, 0.0, 0.0)
    _, q1 = _run(f, 0.0, 0.8)
    assert abs(f.qd[0]) > 0.3, 'moving when the stop comes'
    f.command(drv.STOP, 2, None, None, None, None, 0.0, 0.0, 0.8)
    _, q2 = _run(f, 0.8, 1.0)
    assert f.mode == drv.HOLD and f.at_rest
    v, acc = _vel_acc(np.vstack([q1[-2:], q2]))
    assert np.all(np.abs(acc) <= 4.0 + 1e-6)
    assert np.allclose(q2[-1], q2[-20]), 'held'
    assert f.done_seq == 0, 'a stopped move is not a finished one'


def test_targets_outside_the_limits_are_clamped():
    lo, hi = LIM[:, 0] + 0.05, LIM[:, 1] - 0.05
    f = drv.Follower(DT, lo, hi, LIM[:, 0], LIM[:, 1], V_CAP, 4.0, 80.0)
    f.reset(Q0)
    target = Q0.copy()
    target[6] = 4.0                                      # past A7's 175 deg
    a = np.full(7, 2.0)
    f.command(drv.MOVE, 1, target, V_CAP, a, a / 0.1, 0.0, 0.0, 0.0)
    _, qs = _run(f, 0.0, 6.0)
    assert f.at_rest and abs(qs[-1][6] - hi[6]) < 1e-12 and np.all(qs[:, 6] <= hi[6] + 1e-12)


# ------------------------------------------------------------------ the wrench
@pytest.mark.parametrize('q_deg', [[0, 30, 0, -90, 0, 60, 0], [20, 20, -10, -80, 15, 80, 40]])
def test_the_wrench_comes_back_out_of_the_joint_torques(q_deg):
    q = np.radians(q_deg)
    F = np.array([3.0, -2.0, 15.0, 0.2, -0.1, 0.05])
    tau = iiwa14.jacobian(q).T @ F
    assert np.allclose(wrench_from_torques(iiwa14, q, tau), F, atol=0.01)
    # A tool weighing 1.3 kg with its CoG 30 mm out along the flange x axis, NOT in the
    # controller's load data: subtracting it leaves exactly the contact.
    m, cog = 1.3, np.array([0.03, 0.0, 0.05])
    f_g = np.array([0.0, 0.0, -m * 9.81])
    r = iiwa14.fk(q)[:3, :3] @ cog
    tau_w = iiwa14.jacobian(q).T @ (F + np.concatenate([f_g, np.cross(r, f_g)]))
    assert np.allclose(wrench_from_torques(iiwa14, q, tau_w, payload=(m, cog)), F, atol=0.01)


# ------------------------------------------------------------------ the process, end to end
def _cfg(**sdk_options):
    return Config({'robot': {'type': 'iiwa14', 'fri': {
        'sdk': FAKE_PYFRI, 'sdk_options': sdk_options, 'rt_priority': 0,
        'connect_timeout_s': 10.0}}, 'settle_s': 0.0})


def _gone(pid):
    try:
        with open(f'/proc/{pid}/stat') as fh:
            return fh.read().split(') ')[1][0] == 'Z'      # a zombie has exited
    except FileNotFoundError:
        return True


def test_no_controller_is_a_clear_error_and_leaves_nothing_running():
    cfg = _cfg(silent=True)
    cfg['robot']['fri']['connect_timeout_s'] = 1.0
    t = time.monotonic()
    with pytest.raises(ArmError, match='no packet from the controller'):
        make_arm(cfg)
    assert time.monotonic() - t < 5.0
    assert not [n for n in os.listdir('/dev/shm') if n.startswith('cable_assembly_fri_')
                and os.stat(os.path.join('/dev/shm', n)).st_mtime > t - 1.0], 'no stale channel'


def test_the_wrong_client_command_mode_is_refused():
    with pytest.raises(ArmError, match='POSITION'):
        make_arm(_cfg(command_mode='TORQUE'))


def test_the_tare_removes_the_torque_model_bias():
    arm = make_arm(_cfg(tau_bias=[0.4, -1.1, 0.2, 0.9, 0.0, 0.3, -0.05]))
    try:
        assert np.linalg.norm(arm.wrench()[:3]) > 1.0, 'the bias reads as force before the tare'
        arm.zero_ft()
        assert np.linalg.norm(arm.wrench()) < 1e-6
    finally:
        arm.disconnect()


def test_a_session_that_drops_mid_move_is_not_resumed():
    """An e-stop or a paused app takes the controller out of commanding mid-move. The move must
    fail, and when the session comes back the arm must stay where it stopped: resuming a motion
    the task already gave up on is exactly the surprise an operator cannot see coming."""
    arm = make_arm(_cfg(drop={'at_s': 0.6, 'for_s': 0.4}))
    try:
        q_start = np.asarray(arm.q())
        target = q_start + np.radians([40.0, 0, 0, 0, 0, 0, 0])
        assert arm.move_j(target, speed=0.2, label='long') is False
        q_drop = np.asarray(arm.q())
        assert 0.01 < q_drop[0] - q_start[0] < np.radians(39.0), 'it was under way'
        deadline = time.monotonic() + 3.0
        while arm._state().session != drv.COMMANDING_ACTIVE and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.3)
        assert np.allclose(arm.q(), q_drop, atol=1e-6), 'held where it stopped'
        assert arm.move_j(q_start, label='back') is True, 'and commandable again'
        assert np.allclose(arm.q(), q_start, atol=1e-5)
    finally:
        arm.disconnect()


def test_the_driver_brakes_and_exits_when_the_task_dies():
    script = textwrap.dedent(f'''
        import os, signal, sys
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from test_fri_backend import _cfg
        from cable_assembly.robot import make_arm
        arm = make_arm(_cfg())
        print(arm._proc.pid, arm._chan.path, flush=True)
        arm._send(2, int(arm._state().epoch), arm._state().q, arm.v_cap, [4.0] * 7, [200.0] * 7,
                  0.008)
        os.kill(os.getpid(), signal.SIGKILL)
    ''')
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                         timeout=30)
    pid_s, path = out.stdout.split()[:2]
    pid = int(pid_s)
    deadline = time.monotonic() + 5.0
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert _gone(pid), 'the orphaned driver must end its session'
        assert not os.path.exists(path), 'and remove its channel'
    finally:
        if not _gone(pid):
            os.kill(pid, signal.SIGKILL)
