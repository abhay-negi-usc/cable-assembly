"""The Arm contract (cable_assembly/robot/arm.py), held against every backend.

The task code is written once against Arm, so every backend must behave the same through it:
the simulated UR10e, the simulated iiwa 14, and the REAL URArm code talking to a fake controller
(tests/fake_rtde.py). A behaviour only one backend has is a bug in the layering.
"""

import numpy as np
import pytest

from cable_assembly.config import Config
from cable_assembly.robot import Arm, ArmError, SimArm, URArm, make_arm
from cable_assembly.robot.kinematics import iiwa14, ur10e
from cable_assembly.transforms import pose_error, translation_matrix

from fake_rtde import FakeRTDE, install

# A UR10e start that is clear of every singularity, tool pointing down.
Q_UR = np.radians([-90.0, -100.0, -110.0, -60.0, 90.0, 0.0])


def _cfg(robot, **top):
    return Config({'robot': robot, 'settle_s': 0.0, **top})


def _sim_ur():
    return make_arm(_cfg({'type': 'ur10e', 'dry_run': True, 'dry_run_joints': list(Q_UR)}))


def _sim_iiwa():
    return make_arm(_cfg({'type': 'iiwa14', 'dry_run': True}))


def _rtde_ur(monkeypatch, **kw):
    ctl = FakeRTDE(q0=Q_UR, **kw)
    install(monkeypatch, ctl)
    arm = make_arm(_cfg({'type': 'ur10e', 'ip': 'fake'}))
    return arm, ctl


@pytest.fixture(params=['sim-ur10e', 'sim-iiwa14', 'rtde-ur10e'])
def arm(request, monkeypatch):
    if request.param == 'sim-ur10e':
        return _sim_ur()
    if request.param == 'sim-iiwa14':
        return _sim_iiwa()
    return _rtde_ur(monkeypatch)[0]


# ------------------------------------------------------------------ selection
def test_the_config_picks_the_backend(monkeypatch):
    assert isinstance(_sim_ur(), SimArm) and _sim_ur().model is ur10e
    assert isinstance(_sim_iiwa(), SimArm) and _sim_iiwa().model is iiwa14
    assert make_arm(_cfg({'dry_run': True})).model is ur10e, 'robot.type defaults to the UR10e'
    arm, _ = _rtde_ur(monkeypatch)
    assert isinstance(arm, URArm) and isinstance(arm, Arm)

    with pytest.raises(ValueError, match='unknown robot type'):
        make_arm(_cfg({'type': 'ur5', 'dry_run': True}))
    with pytest.raises(ArmError, match='cannot drive'):
        make_arm(_cfg({'type': 'iiwa14', 'backend': 'ur_rtde'}))
    # Not written yet -- and the error says how to run the iiwa anyway.
    with pytest.raises(ArmError, match='--dry-run'):
        make_arm(_cfg({'type': 'iiwa14'}))


def test_joint_roles_name_the_same_job_on_both_arms():
    ur, kuka = _sim_ur(), _sim_iiwa()
    assert ur.joint_names[ur.joint_index('tool_roll')] == 'wrist_3_joint'
    assert ur.joint_names[ur.joint_index('wrist_nod')] == 'wrist_1_joint'
    assert kuka.joint_names[kuka.joint_index('tool_roll')] == 'A7'
    assert kuka.joint_names[kuka.joint_index('wrist_nod')] == 'A6'
    assert ur.tool_roll_joint == 5 and kuka.tool_roll_joint == 6
    with pytest.raises(ValueError, match='no joint role'):
        ur.joint_index('elbow_flip')


# ------------------------------------------------------------------ the contract, every backend
def test_state_and_kinematics_agree(arm):
    q = arm.q()
    assert len(q) == arm.dof == len(arm.joint_names)
    assert pose_error(arm.tcp_pose(), arm.fk(q))[0] < 1e-6
    assert arm.joints_ok(q)


def test_move_to_puts_tool0_on_the_target_and_ik_stays_on_the_branch(arm):
    T0 = arm.tcp_pose()
    T = translation_matrix([0.03, -0.02, 0.05]) @ T0
    q_before = np.asarray(arm.q())
    assert arm.move_to(T, label='test') is True
    lin, ang = pose_error(arm.tcp_pose(), T)
    assert lin < 1e-6 and ang < 1e-6
    # A 6 cm move must not swing any joint far: the nearest branch, not merely a valid one.
    assert np.max(np.abs(np.asarray(arm.q()) - q_before)) < np.radians(20.0)


def test_move_l_travels_a_straight_line(arm):
    T0 = arm.tcp_pose()
    T1 = translation_matrix([0.0, 0.0, -0.08]) @ T0
    seen = []
    arm.add_guard(lambda: seen.append(arm.tcp_pose()[:3, 3].copy()) or False)
    try:
        assert arm.move_l(T1, label='line') is True
    finally:
        arm.clear_guards()
    assert pose_error(arm.tcp_pose(), T1)[0] < 1e-6
    assert len(seen) > 5, 'a guarded move is polled along the way, not only at the end'
    p0, d = T0[:3, 3], T1[:3, 3] - T0[:3, 3]
    for p in seen:
        off = (p - p0) - np.dot(p - p0, d) / np.dot(d, d) * d
        assert np.linalg.norm(off) < 1e-4, 'every intermediate point lies on the line'


def test_a_target_outside_the_joint_limits_is_refused_not_clipped(arm):
    q = list(arm.q())
    q[0] = 7.0                                       # past +2pi (UR) and +170 deg (iiwa)
    assert arm.joints_ok(q) is False
    before = arm.q()
    assert arm.move_j(q, label='too far') is False
    assert np.allclose(arm.q(), before), 'a refused move leaves the arm where it was'


def test_an_unreachable_pose_has_no_ik(arm):
    T = arm.tcp_pose().copy()
    T[:3, 3] = [3.0, 0.0, 0.5]                       # past both arms' reach
    assert arm.ik(T) is None
    assert arm.move_to(T, label='unreachable') is False


def test_the_tool_roll_window_is_the_rolls_own_limit(arm):
    lo, hi = arm.tool_roll_window(arm.q())
    j, (jlo, jhi) = arm.tool_roll_joint, arm.model.JOINT_LIMITS[arm.tool_roll_joint]
    q0 = arm.q()[j]
    tol = np.radians(0.5)
    assert lo <= q0 <= hi
    # Capped at a full turn each way, otherwise ending at the joint's limit.
    assert abs(lo - max(jlo, q0 - 2 * np.pi)) < tol and abs(hi - min(jhi, q0 + 2 * np.pi)) < tol


def test_servo_l_follows_a_streamed_reference(arm):
    T0 = arm.tcp_pose()
    for k in range(1, 26):
        arm.servo_l(translation_matrix([0.0, 0.0, -0.001 * k]) @ T0, 0.008)
    arm.servo_stop()
    assert pose_error(arm.tcp_pose(), translation_matrix([0.0, 0.0, -0.025]) @ T0)[0] < 1e-6


# ------------------------------------------------------------------ guards and the wrench
def _wall(z_wall, k=2000.0):
    """A floor at base z = z_wall under tool0: pushes back in +z, proportional to penetration."""
    def wrench(q, fk):
        pen = z_wall - fk(q)[2, 3]
        return np.array([0.0, 0.0, k * pen if pen > 0 else 0.0, 0.0, 0.0, 0.0])
    return wrench


@pytest.mark.parametrize('backend', ['sim-ur10e', 'sim-iiwa14', 'rtde-ur10e'])
def test_a_guard_stops_a_move_at_contact(backend, monkeypatch):
    if backend == 'rtde-ur10e':
        z_wall = ur10e.fk(Q_UR)[2, 3] - 0.03
        arm, _ = _rtde_ur(monkeypatch, contact=lambda q: _wall(z_wall)(q, ur10e.fk))
    else:
        arm = _sim_ur() if backend == 'sim-ur10e' else _sim_iiwa()
        z_wall = arm.tcp_pose()[2, 3] - 0.03
        arm.contact = lambda a: _wall(z_wall)(a.q(), a.fk)
    assert arm.zero_ft() is True
    assert arm.force() == 0.0, 'free space reads zero after the tare'
    T_down = translation_matrix([0.0, 0.0, -0.10]) @ arm.tcp_pose()
    arm.add_guard(lambda: arm.force() > 10.0)
    try:
        ok = arm.move_l(T_down, label='into the wall')
    finally:
        arm.clear_guards()
    assert ok is False, 'the guard must cancel the move'
    z = arm.tcp_pose()[2, 3]
    assert z_wall - 0.01 < z < z_wall, f'stopped at the wall, not past it ({z - z_wall:+.4f} m)'
    w = arm.wrench()
    assert w[2] > 10.0 and abs(w[0]) < 1e-9 and abs(w[1]) < 1e-9, \
        'the wrench comes back in base_link: +z stays +z, and nothing leaks into x/y'


def test_the_ur_tool_roll_window_is_asked_of_the_controller(monkeypatch):
    """An installation safety setting can narrow wrist_3 below the model's +/-360. The window
    must come from the controller, so the narrower range is what the clocking plans with."""
    arm, _ = _rtde_ur(monkeypatch, wrist_3_limits=np.radians([-170.0, 190.0]))
    lo, hi = arm.tool_roll_window(arm.q())
    assert abs(np.degrees(lo) + 170.0) < 0.5 and abs(np.degrees(hi) - 190.0) < 0.5


def test_controller_compliance_is_a_capability_not_an_assumption(monkeypatch):
    arm, ctl = _rtde_ur(monkeypatch)
    assert arm.supports_force_mode
    arm.force_mode(np.eye(4), [0, 0, 1, 0, 0, 0], [0.0] * 6, [0.1] * 6)
    arm.end_force_mode()
    assert [c[0] for c in ctl.calls if c[0].startswith('forceMode')][-1] == 'forceModeStop'

    class _NoCompliance(SimArm):
        supports_force_mode = False
        force_mode = Arm.force_mode

    bare = _NoCompliance(_cfg({'type': 'iiwa14', 'dry_run': True}), iiwa14)
    with pytest.raises(ArmError, match='AdmittanceController'):
        bare.force_mode(np.eye(4), [1] * 6, [0.0] * 6, [0.1] * 6)
