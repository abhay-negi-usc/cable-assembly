"""robot/world.py: what the wrist camera has seen, in this run or any before, as the world every plain
move is checked against.

What must hold (the 2026-11-02 collision: a reconfiguring move_j swept into the sockets' structure,
checked against the base plane only): a thing the camera has seen blocks a plain move through it --
and the same move is clear without the world; a frame taken while the arm moved is left out; and
what one run saw -- its world, saved, or view sets it kept -- the next run avoids from its first move."""

import os

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.perception.camera import Frame

cumotion = pytest.importorskip('cumotion')
if not os.path.isfile(os.path.join(os.path.dirname(__file__), '..', 'cable_assembly', 'robot', 'description',
                                   'iiwa14', 'iiwa14.xrdf')):
    pytest.skip('no iiwa14.xrdf (robot/description/make_iiwa14_xrdf.py writes it)', allow_module_level=True)

def _cfg(tmp_path, *extra):
    """A dry run whose world reads and writes only under tmp_path (not the repo's data/)."""
    return C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false',
                                   f'data_dir={tmp_path}', f'robot.world.roots=[{tmp_path}]',
                                   f'robot.world.file={tmp_path}/world/world_scene.npz', 'robot.world.prior=true'] + list(extra))


Q_VIEW = np.radians([20.1, 23.2, 5.7, -109.8, 72.7, -36.9, -58.9])   # the bench's socket view pose
K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
W, H = 320, 240


@pytest.fixture
def robot(tmp_path):
    from cable_assembly.robot import Robot
    r = Robot(_cfg(tmp_path), with_gripper=False, needs=())
    r.arm.move_j(Q_VIEW, label='view')
    return r


def _wall(robot, dist=0.40, half_w=0.12, half_h=0.10):
    """A frame from where the arm is: a wall square to the camera `dist` ahead, `half_w` x `half_h` m."""
    T = robot.arm.fk(robot.arm.q()) @ robot.T_tool0_cam
    depth = np.zeros((H, W), np.float32)
    u0, u1 = int(K[0, 2] - half_w / dist * K[0, 0]), int(K[0, 2] + half_w / dist * K[0, 0])
    v0, v1 = int(K[1, 2] - half_h / dist * K[1, 1]), int(K[1, 2] + half_h / dist * K[1, 1])
    depth[v0:v1, u0:u1] = dist
    return Frame(np.zeros((H, W, 3), np.uint8), K, None, 0.0, T, depth)


def _forward(robot, metres, steps=40):
    """Joint vectors driving the camera straight ahead (along its line of sight) by `metres`."""
    from cable_assembly.transforms import translation_matrix
    q = np.asarray(robot.arm.q(), dtype=float)
    T0 = robot.arm.fk(q)
    a = (T0 @ robot.T_tool0_cam)[:3, 2]
    path = [q]
    for s in np.linspace(0.0, metres, steps)[1:]:
        q = np.asarray(robot.arm.ik(translation_matrix(a * s) @ T0, qnear=q), dtype=float)
        path.append(q)
    return path


def test_a_seen_wall_blocks_a_move_through_it(robot):
    pc, world = robot.arm.path_check, robot.world
    path = _forward(robot, 0.35)             # the fingertips (~0.19 m out) go through a wall 0.40 m ahead
    pc.world = None
    assert pc.refuse(path) is None, 'without the world, only the base plane: clear'
    pc.world = world
    world(_wall(robot))
    assert world.scene.occ.sum() > 50
    why = pc.refuse(path)
    assert why is not None and 'scene' in why
    short = _forward(robot, 0.05)            # a short step that stays clear of it
    assert pc.refuse(short) is None


def test_a_frame_from_a_moving_arm_is_left_out(robot):
    world = robot.world
    f = _wall(robot)
    f.T_base_cam = f.T_base_cam.copy()
    f.T_base_cam[:3, 3] += [0.01, 0.0, 0.0]  # stamped 10 mm from where the arm is now
    world(f)
    assert world.skipped_moving == 1 and world.scene.occ.sum() == 0


def test_what_one_run_saw_the_next_run_avoids(robot, tmp_path):
    """The world is saved as it changes and a new run starts from it -- as the live view's splat keeps
    what it was shown -- without any view set kept on disk."""
    from cable_assembly.robot import Robot
    robot.world(_wall(robot))
    robot.world.save(force=True)
    assert (tmp_path / 'world' / 'world_scene.npz').is_file()
    nxt = Robot(_cfg(tmp_path), with_gripper=False, needs=())
    assert nxt.world.scene.occ.sum() == robot.world.scene.occ.sum() > 50
    nxt.arm.move_j(Q_VIEW, label='view')
    assert nxt.arm.path_check.refuse(_forward(nxt, 0.35)) is not None


def test_saved_views_seed_a_new_runs_world(robot, tmp_path):
    from cable_assembly.robot import Robot
    d = tmp_path / 'pose6d' / '20261102_230000_6d_pose_the_socket'
    d.mkdir(parents=True)
    f = _wall(robot)
    np.savez_compressed(d / 'view_00.npz', rgb=f.color[..., ::-1], depth_m=f.depth, K=f.K, T_base_cam=f.T_base_cam,
                        joints_rad=np.asarray(robot.arm.q(), dtype=float))
    fresh = Robot(_cfg(tmp_path), with_gripper=False, needs=())
    assert fresh.world.scene.occ.sum() > 50, 'seeded from the saved views'
    fresh.arm.move_j(Q_VIEW, label='view')
    assert fresh.arm.path_check.refuse(_forward(fresh, 0.35)) is not None


def test_a_world_filtered_the_old_way_is_rebuilt(robot, tmp_path):
    """A world saved before the camera filter changed holds what it no longer lets in (the V3's outer
    finger, 2026-11-03): the next run starts from the view sets instead."""
    from cable_assembly.perception.scene import OccupancyScene
    from cable_assembly.robot import Robot
    robot.world(_wall(robot))
    robot.world.save(force=True)
    path = tmp_path / 'world' / 'world_scene.npz'
    old = OccupancyScene.load(str(path))
    old.save(str(path))                       # no filter stamp: as saved before it existed
    nxt = Robot(_cfg(tmp_path), with_gripper=False, needs=())
    assert nxt.world.scene.occ.sum() == 0, 'not loaded: rebuilt (from no view sets here)'


def test_without_the_prior_a_run_starts_empty(robot, tmp_path):
    """robot.world.prior false (the bench, the user 2026-11-03): neither the saved world nor a saved view
    set is read -- only the robot and its base plane -- and what the run's camera sees still counts."""
    from cable_assembly.robot import Robot
    robot.world(_wall(robot))
    robot.world.save(force=True)
    d = tmp_path / 'pose6d' / '20261102_230000_6d_pose_the_socket'
    d.mkdir(parents=True)
    f = _wall(robot)
    np.savez_compressed(d / 'view_00.npz', rgb=f.color[..., ::-1], depth_m=f.depth, K=f.K, T_base_cam=f.T_base_cam,
                        joints_rad=np.asarray(robot.arm.q(), dtype=float))
    fresh = Robot(_cfg(tmp_path, 'robot.world.prior=false'), with_gripper=False, needs=())
    assert fresh.world.scene.occ.sum() == 0
    fresh.arm.move_j(Q_VIEW, label='view')
    fresh.world(_wall(fresh))
    assert fresh.world.scene.occ.sum() > 50


def test_the_start_is_freed_wherever_the_arm_now_is_not_where_the_world_last_changed(robot):
    # 2026-11-04: after the clocking, the move home refused -- "the start is in collision (-7 mm):
    # tool0 against the scene". The world had not changed since an earlier move, so the planner
    # reused a scene copy freed at THAT move's start, not where the gripper now stood at the socket.
    pc, world = robot.arm.path_check, robot.world
    world(_wall(robot))
    assert pc.refuse(_forward(robot, 0.05)) is None       # synced (and freed) at the view pose
    q_in = _forward(robot, 0.22)[-1]                       # the fingertips just into the wall
    path = pc.detour(q_in, Q_VIEW)
    assert path is not None, pc.why
