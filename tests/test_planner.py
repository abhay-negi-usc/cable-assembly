"""robot/planner.py and robot/tool_geometry.py: cuMotion moves round what the camera saw, the tool
on the arm included. Scenes are synthetic depth frames (tests/test_scene.py's renderer) of the
bench and boxes on it."""

import os
import sys

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.perception.scene import OccupancyScene
from cable_assembly.robot.kinematics import iiwa14 as k

sys.path.insert(0, os.path.dirname(__file__))
from test_scene import BENCH_Z, _down, _frame

cumotion = pytest.importorskip('cumotion')
DESC = os.path.join(os.path.dirname(k.__file__), '..', 'description', 'iiwa14')
if not os.path.isfile(os.path.join(DESC, 'iiwa14.xrdf')):
    pytest.skip('no iiwa14.xrdf (robot/description/make_iiwa14_xrdf.py writes it)', allow_module_level=True)


@pytest.fixture(scope='module')
def cfg():
    return C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true'])


@pytest.fixture(scope='module')
def planner(cfg):
    from cable_assembly.robot.planner import Planner
    return Planner.for_cell(cfg)


def _scene(*boxes, eye=None):
    s = OccupancyScene([0.2, -0.5, BENCH_Z], [1.1, 0.6, BENCH_Z + 0.6], voxel_m=0.01)
    centre = eye if eye is not None else np.mean([0.5 * (lo + hi) for lo, hi in boxes], axis=0) if boxes else [0.6, 0, 0]
    s.add(_frame(_down(centre[0], centre[1], 1.2), list(boxes)))
    return s


def test_the_tool_is_on_the_robot_and_counts(planner, cfg):
    """An obstacle where only the fingers are, at a pose: clear for the arm alone, not with the tool."""
    assert planner.n_tool_spheres > 20
    q = k.HOME.copy()
    tip = (k.fk(q) @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.17], [0, 0, 0, 1.0]]))[:3, 3]
    block = (tip - [0.03, 0.03, tip[2] - BENCH_Z], tip + [0.03, 0.03, 0.0])
    s = _scene(block)
    assert planner.clearance(q, s) < 0.0
    arm_only = cumotion.load_robot_from_file(os.path.join(DESC, 'iiwa14.xrdf'), os.path.join(DESC, 'iiwa14.urdf'))
    world, _ = s.to_cumotion(bench_z=planner.bench_z)
    assert cumotion.create_robot_world_inspector(arm_only, world.add_world_view()).min_distance_to_obstacle(q) > 0.0


def test_a_plan_goes_round_a_pillar_the_camera_saw(planner):
    q0, q1 = k.HOME.copy(), k.HOME.copy()
    q0[0], q1[0] = np.radians(45), np.radians(-45)
    mid = k.fk(0.5 * (q0 + q1))[:3, 3]
    pillar = (np.array([mid[0] - 0.06, mid[1] - 0.06, BENCH_Z]), np.array([mid[0] + 0.06, mid[1] + 0.06, mid[2] + 0.05]))
    s = _scene(pillar)
    assert planner.clearance(0.5 * (q0 + q1), s) < 0.0                  # the straight swing hits it
    path = planner.plan(q0, q1, s)
    assert path is not None, planner.why
    np.testing.assert_allclose(path.q[0], q0, atol=1e-6)
    np.testing.assert_allclose(path.q[-1], q1, atol=1e-9)
    assert path.clearance > 0.0
    assert all(planner.clearance(q, s) > -1e-3 for q in path.q[::10])


def test_a_goal_inside_what_was_seen_is_refused(planner):
    q0, q1 = k.HOME.copy(), k.HOME.copy()
    q1[0] = np.radians(-45)
    tip = k.fk(q1)[:3, 3]
    s = _scene((tip - [0.08, 0.08, tip[2] - BENCH_Z], tip + [0.08, 0.08, 0.05]))
    assert planner.plan(q0, q1, s) is None
    assert 'goal' in planner.why


def test_a_path_is_streamed_on_the_iiwa(tmp_path, planner):
    """IiwaFRIArm.move_joint_path against the fake controller: the commanded joints run the path,
    end on it, and stay under the caps; a path not starting where the arm is is refused."""
    from test_fri_backend import DT, _cfg

    from cable_assembly.robot import make_arm
    rec = tmp_path / 'commanded.npy'
    arm = make_arm(_cfg(record=str(rec)))
    try:
        q0 = np.asarray(arm.q(), dtype=float)
        q1 = q0 + np.radians([20.0, 5.0, 0.0, -10.0, 0.0, 10.0, 0.0])
        t = np.linspace(0.0, 1.0, 51)
        u = 3 * t ** 2 - 2 * t ** 3                                       # smooth, from rest to rest
        path = q0 + np.outer(u, q1 - q0)
        assert arm.move_joint_path(t, path, label='planned') is True
        np.testing.assert_allclose(arm.q(), q1, atol=1e-5)
        assert arm.move_joint_path(t, path + np.radians(5.0), label='elsewhere') is False
    finally:
        arm.disconnect()
    q = np.load(rec)[:, 1:]
    v = np.abs(np.diff(q, axis=0)) / DT
    assert np.all(v <= arm.v_cap * 1.01)


def test_the_grasp_align_plans_round_the_scene_when_it_has_one(cfg, planner):
    """GraspController._go with a scanned scene: planned and driven (a dry run: joint moves along
    the plan) when it clears, refused with the arm parked when the goal is in what was seen."""
    import types

    from cable_assembly.robot import Robot
    from cable_assembly.skills.pick import GraspController
    robot = Robot(C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false']),
                  with_gripper=False, needs=())
    try:
        ctl = GraspController(cfg)
        ctl._planner = planner
        q0, q1 = k.HOME.copy(), k.HOME.copy()
        q1[0] = np.radians(-30)
        assert robot.arm.move_j(q0, label='start')
        clear = types.SimpleNamespace(scene=_scene(eye=[0.6, 0.0, 0.0]))
        assert ctl._go(robot, clear, k.fk(q1), q1, 'grasp-align') is True
        np.testing.assert_allclose(robot.arm.q(), q1, atol=1e-6)
        assert robot.arm.move_j(q0, label='back')
        tip = k.fk(q1)[:3, 3]
        blocked = types.SimpleNamespace(scene=_scene((tip - [0.08, 0.08, tip[2] - BENCH_Z], tip + [0.08, 0.08, 0.05])))
        assert ctl._go(robot, blocked, k.fk(q1), q1, 'grasp-align') is False
        assert ctl.last_refusal == 'unreachable'
        np.testing.assert_allclose(robot.arm.q(), q0, atol=1e-9)        # never moved
    finally:
        robot.close()


def test_a_held_part_rides_on_the_tool_and_counts(cfg, planner):
    """A rod held 60 mm past the fingertip: a block where only the rod reaches is clear for the
    planner without it and not for the one holding it."""
    from cable_assembly.robot import tool_geometry
    from cable_assembly.robot.planner import Planner
    T_tool0_rod = np.eye(4)
    T_tool0_rod[2, 3] = 0.19                                                   # at the fingertip, along tool z
    held = tool_geometry.rod_spheres(T_tool0_rod, 0.0, 0.08, 0.009)
    assert len(held) >= 10 and all(r == 0.009 for _, r in held)
    np.testing.assert_allclose(held[-1][0], [0.0, 0.0, 0.27])
    holding = Planner.for_cell(cfg, held=held)
    assert holding.n_tool_spheres == planner.n_tool_spheres + len(held)
    q = k.HOME.copy()
    end = (k.fk(q) @ np.array([0.0, 0.0, 0.26, 1.0]))[:3]                    # where only the rod is
    s = _scene((end - [0.02, 0.02, end[2] - BENCH_Z], end + [0.02, 0.02, -0.015]))
    assert planner.clearance(q, s) > 0.0 > holding.clearance(q, s)


def test_a_plan_to_a_pose_chooses_its_goal_and_ends_there(planner):
    q0 = k.HOME.copy()
    q_aim = k.HOME.copy()
    q_aim[0], q_aim[3] = np.radians(-40), q_aim[3] + np.radians(10)
    T = k.fk(q_aim)
    path, q_goal = planner.plan_to_pose(q0, T, _scene())
    assert path is not None, planner.why
    np.testing.assert_allclose(k.fk(q_goal), T, atol=1e-6)
    np.testing.assert_allclose(path.q[-1], q_goal, atol=1e-9)
    # A goal the scene fills, at every arm angle: refused, said why.
    tip = T[:3, 3]
    s = _scene((tip - [0.10, 0.10, tip[2] - BENCH_Z], tip + [0.10, 0.10, 0.10]))
    assert planner.plan_to_pose(q0, T, s) == (None, None)
    assert 'collision' in planner.why


def test_what_the_robot_stands_in_can_be_cleared_from_the_scene():
    s = _scene((np.array([0.55, -0.05, BENCH_Z]), np.array([0.65, 0.05, BENCH_Z + 0.10])))
    n = int(s.occ.sum())
    assert n > 0
    assert s.clear_spheres([[0.60, 0.0, BENCH_Z + 0.10]], [0.03]) > 0
    assert 0 < int(s.occ.sum()) < n
    assert s.clear_spheres([], []) == 0


def test_where_the_arm_stands_is_freed_before_planning(planner):
    """Older views can put something where the arm now is (the cable it picked up): freed, the
    start is clear, and nothing far from the arm is touched."""
    q = k.HOME.copy()
    tip = (k.fk(q) @ np.array([0.0, 0.0, 0.2, 1.0]))[:3]
    far = (np.array([0.9, -0.4, BENCH_Z]), np.array([1.0, -0.3, BENCH_Z + 0.1]))
    s = _scene((tip - [0.04, 0.04, tip[2] - BENCH_Z], tip + [0.04, 0.04, 0.0]), far, eye=[0.7, -0.1, 0.0])
    assert planner.clearance(q, s) < 0.0
    far_before = int(s.occ[tuple(s.index(np.array([[0.95, -0.35, BENCH_Z + 0.05]]))[0])])
    assert planner.free_start(s, q) > 0
    assert planner.clearance(q, s) >= -0.005
    assert int(s.occ[tuple(s.index(np.array([[0.95, -0.35, BENCH_Z + 0.05]]))[0])]) == far_before


def test_a_plan_streams_smoothly_within_the_arms_acceleration(tmp_path, cfg):
    """A cuMotion plan, planned to the arm's acceleration and jerk (arm_limits) and streamed: the
    commanded joints accelerate no harder than the arm's cap allows -- not the driver's flat-out
    +-4 rad/s^2 catch-up that a plan to the description's own limits, or its noisy sampled
    positions, drove the arm into (the jerk, 2026-10-30)."""
    from test_fri_backend import DT, _cfg

    from cable_assembly.robot import make_arm
    from cable_assembly.robot.planner import Planner, arm_limits
    planner = Planner.for_cell(cfg)
    accel = arm_limits(cfg)[0]
    rec = tmp_path / 'commanded.npy'
    arm = make_arm(_cfg(record=str(rec)))
    try:
        q0 = np.asarray(arm.q(), dtype=float)
        q1 = q0 + np.radians([40.0, 0.0, 0.0, 15.0, 0.0, -10.0, 20.0])
        path = planner.plan(q0, q1, _scene())
        assert path is not None, planner.why
        # the samples come from cuMotion's velocity: their second differences stay near its limit
        a_samples = np.max(np.abs(np.diff(path.q, 2, axis=0))) / np.mean(np.diff(path.times)) ** 2
        assert a_samples < 1.5 * accel
        assert arm.move_joint_path(path.times, path.q, label='planned', caps={'max_joint_acceleration_deg_s2':
                                                                              float(np.degrees(accel))}) is True
        np.testing.assert_allclose(arm.q(), q1, atol=1e-5)
    finally:
        arm.disconnect()
    q = np.load(rec)[:, 1:]
    a = np.diff(q, 2, axis=0) / DT ** 2
    smooth = np.apply_along_axis(lambda x: np.convolve(x, np.ones(9) / 9, mode='same'), 0, a)
    assert np.max(np.abs(smooth)) < 1.5 * accel                       # the motion itself
    assert np.max(np.abs(a)) < arm.stream_accel                       # nowhere the driver's flat-out


def _through_the_plane():
    """tool0 pointing down 80 mm over the base plane: the gripper (~0.2 m) goes through it."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.diag([1.0, -1.0, -1.0]), [0.6, 0.0, 0.08]
    return T


def test_the_plane_the_base_stands_on_is_always_an_obstacle(planner):
    """With no scene at all, the plane under the base: the arm at home clear of it (its own base,
    bolted to it, left out), a goal with the gripper through it refused and the plane named, a plan
    above it found."""
    q0 = k.HOME.copy()
    assert planner.clearance(q0) > 0.0
    assert planner.plan_to_pose(q0, _through_the_plane()) == (None, None)
    assert 'the base plane' in planner.why
    q1 = k.HOME.copy()
    q1[0] = np.radians(-30)
    path = planner.plan(q0, q1)
    assert path is not None, planner.why
    assert path.clearance > 0.0


def test_the_grasp_align_plans_round_the_base_plane_without_a_scene(cfg, planner):
    import types

    from cable_assembly.robot import Robot
    from cable_assembly.skills.pick import GraspController
    robot = Robot(C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false']),
                  with_gripper=False, needs=())
    try:
        ctl = GraspController(cfg)
        ctl._planner = planner
        q0 = k.HOME.copy()
        assert robot.arm.move_j(q0, label='start')
        no_scan = types.SimpleNamespace(scene=None)
        T = _through_the_plane()
        q_low = next(q for psi in np.radians(np.arange(-180, 180, 10)) for q in k.ik_all(T, psi, q0)
                     if np.all((q >= k.JOINT_LIMITS[:, 0]) & (q <= k.JOINT_LIMITS[:, 1])))
        assert ctl._go(robot, no_scan, T, q_low, 'grasp-align') is False
        assert ctl.last_refusal == 'unreachable'
        np.testing.assert_allclose(robot.arm.q(), q0, atol=1e-9)        # never moved
        q1 = k.HOME.copy()
        q1[0] = np.radians(-30)
        assert ctl._go(robot, no_scan, k.fk(q1), q1, 'grasp-align') is True
        np.testing.assert_allclose(robot.arm.q(), q1, atol=1e-6)
    finally:
        robot.close()


# The reset's move home on 2026-10-31: from the engage pose, its straight joint line drove the
# gripper ~200 mm into the bench.
ENGAGE_Q = np.radians([80.0, 102.2, -73.5, -109.4, 72.0, -28.4, -95.0])
RESET_HOME_Q = np.radians([-27.0, 50.0, 66.0, -96.0, 120.0, -57.0, -123.0])


def _dry_iiwa():
    from cable_assembly.robot import Robot
    return Robot(C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false']),
                 with_gripper=False, needs=())


def test_a_plain_joint_move_through_the_base_plane_goes_round_it():
    """move_j checks its straight line first: the 2026-10-31 move home is blocked, planned round,
    and every position the arm passes is clear of the plane; a clear move is driven as asked."""
    from cable_assembly.robot.arm import joint_line
    robot = _dry_iiwa()
    try:
        check = robot.arm.path_check
        assert check is not None
        assert 'into the base plane' in check.refuse(joint_line(ENGAGE_Q, RESET_HOME_Q))
        robot.arm._q = ENGAGE_Q.copy()
        seen = []
        robot.arm.add_guard(lambda: seen.append(robot.arm.q()) and False)
        assert robot.arm.move_j(RESET_HOME_Q, label='home')
        np.testing.assert_allclose(robot.arm.q(), RESET_HOME_Q, atol=1e-6)
        clear = [check.planner.clearance(q) for q in seen]
        assert min(clear) > 0.0, 1000 * min(clear)
        robot.arm.clear_guards()
        q1 = RESET_HOME_Q.copy()
        q1[0] += np.radians(10)
        assert check.refuse(joint_line(RESET_HOME_Q, q1)) is None
        assert robot.arm.move_j(q1, label='clear')
    finally:
        robot.close()


def test_a_line_into_the_plane_is_refused_and_a_pick_onto_it_is_not():
    """move_l: a line ending with the gripper deep through the plane is refused and the arm stays;
    down to the fingertips just touching it (cuMotion's spheres a little into it, as on every real
    pick) and back up is driven."""
    robot = _dry_iiwa()
    try:
        q0 = k.HOME.copy()
        robot.arm._q = q0.copy()
        T = robot.arm.fk(q0)
        tip = (T @ robot.T_tool0_fingertip)[:3, 3]
        deep = T.copy()
        deep[2, 3] -= tip[2] + 0.10                      # fingertips 100 mm under the plane
        assert robot.arm.move_l(deep, label='through') is False
        np.testing.assert_allclose(robot.arm.q(), q0, atol=1e-9)
        touch = T.copy()
        touch[2, 3] -= tip[2] - 0.004                    # fingertips 4 mm over it
        assert robot.arm.move_l(touch, label='pick'), robot.arm.path_check.refuse([robot.arm.q()])
        assert robot.arm.move_l(T, label='lift')
    finally:
        robot.close()
