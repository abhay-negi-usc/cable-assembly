"""The cuMotion description of the iiwa 14 (description/iiwa14/iiwa14.xrdf + iiwa14.urdf).

Skipped without cuMotion, which is an optional install (setup-env.sh step 5). What is held here:
the description loads with the arm's joints and limits, cuMotion's FK agrees with the numpy
kinematics the arm actually moves by, the sphere model does not call the cell's own poses
self-colliding, and the trajectory optimizer plans around an obstacle a straight joint move hits.
"""
import os

import numpy as np
import pytest

from cable_assembly.robot.kinematics import iiwa14 as k

cumotion = pytest.importorskip('cumotion')

DESC = os.path.join(os.path.dirname(k.__file__), '..', 'description', 'iiwa14')
XRDF = os.path.join(DESC, 'iiwa14.xrdf')
URDF = os.path.join(DESC, 'iiwa14.urdf')
PICK = np.radians([-27, 50, 66, -96, 120, -57, -123])      # configs/cells/iiwa14_bench.yaml


@pytest.fixture(scope='module')
def robot():
    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    return cumotion.load_robot_from_file(XRDF, URDF)


def _world(*boxes):
    world = cumotion.create_world()
    for center, sides in boxes:
        ob = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        ob.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.array(sides, float))
        world.add_obstacle(ob, cumotion.Pose3.from_translation(np.array(center, float)))
    return world, world.add_world_view()


def test_joints_and_limits_are_the_arms(robot):
    kin = robot.kinematics()
    assert kin.num_cspace_coords() == k.DOF
    assert [kin.cspace_coord_name(i) for i in range(k.DOF)] == list(k.JOINT_NAMES)
    lim = np.array([[kin.cspace_coord_limits(i).lower, kin.cspace_coord_limits(i).upper]
                    for i in range(k.DOF)])
    np.testing.assert_allclose(lim, k.JOINT_LIMITS, atol=1e-9)
    vel = np.array([kin.cspace_coord_velocity_limit(i) for i in range(k.DOF)])
    np.testing.assert_allclose(vel, k.VELOCITY_LIMITS, atol=1e-9)


def test_fk_matches_the_numpy_kinematics(robot):
    """tool0 from cuMotion vs kinematics/iiwa14.fk -- the model FRI's joint commands come from."""
    kin = robot.kinematics()
    rng = np.random.default_rng(0)
    for q in rng.uniform(k.JOINT_LIMITS[:, 0], k.JOINT_LIMITS[:, 1], (200, k.DOF)):
        np.testing.assert_allclose(kin.pose(q, 'tool0').matrix(), k.fk(q), atol=1e-9)


@pytest.mark.parametrize('name, q', [('home', k.HOME), ('pick', PICK)])
def test_cell_poses_are_not_self_colliding(robot, name, q):
    _world_, view = _world()
    insp = cumotion.create_robot_world_inspector(robot, view)
    assert not insp.in_self_collision(q), insp.frames_in_self_collision(q)


def test_a_folded_arm_is_self_colliding(robot):
    """The spheres must still catch a real self-collision: the wrist folded back into the base."""
    _world_, view = _world()
    insp = cumotion.create_robot_world_inspector(robot, view)
    folded = np.radians([0, 115, 0, -120, 0, 120, 0])
    assert insp.in_self_collision(folded)


def test_plans_around_an_obstacle_a_straight_move_hits(robot):
    # A swing of A1 from +60 to -60 deg; the obstacle sits where the straight joint-space move
    # puts tool0 half way, so the plan has to go over or around it.
    q0, q1 = k.HOME.copy(), k.HOME.copy()
    q0[0], q1[0] = np.radians(60), np.radians(-60)
    mid = k.fk(0.5 * (q0 + q1))[:3, 3]
    world, view = _world(([0, 0, -0.10], [3, 3, 0.1]),           # floor, below the base spheres
                         (mid, [0.12, 0.12, 0.12]))
    insp = cumotion.create_robot_world_inspector(robot, view)
    assert insp.min_distance_to_obstacle(q0) > 0 and insp.min_distance_to_obstacle(q1) > 0
    assert insp.in_collision_with_obstacle(0.5 * (q0 + q1))

    opt = cumotion.create_trajectory_optimizer(cumotion.create_default_trajectory_optimizer_config(
        robot_description=robot, tool_frame_name='tool0', world_view=view))
    res = opt.plan_to_cspace_target(q0, cumotion.TrajectoryOptimizer.CSpaceTarget(q1))
    assert res.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS, res.status()

    traj = res.trajectory()
    d = traj.domain()
    kin = robot.kinematics()
    for t in np.linspace(d.lower, d.upper, 400):
        q = traj.eval(t).ravel()
        assert kin.within_cspace_limits(q, True)
        assert not insp.in_collision_with_obstacle(q)
        assert not insp.in_self_collision(q)
    np.testing.assert_allclose(traj.eval(d.upper).ravel(), q1, atol=1e-6)
    assert np.all(np.abs(np.array([traj.eval(t, 1).ravel() for t in np.linspace(
        d.lower, d.upper, 400)])) <= k.VELOCITY_LIMITS + 1e-6)
