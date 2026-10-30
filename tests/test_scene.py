"""perception/scene.py: a collision world for cuMotion from the wrist camera's RGB-D frames.
Synthetic depth frames of a bench with a block on it, rendered by ray casting, so the answers are
known: where the occupancy and the signed distance put the block, that a frame which no longer sees
it clears it, that the robot is left out, and -- with cuMotion and the cell's iiwa description --
that a plan through where the camera saw the block goes round it."""

import os

import numpy as np
import pytest

from cable_assembly.perception.scene import OccupancyScene
from cable_assembly.robot.kinematics import iiwa14 as k

BENCH_Z = 0.0062
W, H = 160, 120
K = np.array([[120.0, 0, 80.0], [0, 120.0, 60.0], [0, 0, 1.0]])


def _down(x, y, z):
    """A camera at (x, y, z) looking straight down (optical y along base -y)."""
    return np.array([[1.0, 0, 0, x], [0, -1.0, 0, y], [0, 0, -1.0, z], [0, 0, 0, 1.0]])


def _render(T, boxes):
    """Depth (along the optical axis) of the bench plane and axis-aligned `boxes` (lo, hi)."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3) @ T[:3, :3].T
    o = T[:3, 3]
    t = np.where(d[:, 2] < 0, (BENCH_Z - o[2]) / d[:, 2], np.inf)
    for lo, hi in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            t0, t1 = (np.asarray(lo) - o) / d, (np.asarray(hi) - o) / d
        tn = np.nanmax(np.minimum(t0, t1), axis=1)
        tf = np.nanmin(np.maximum(t0, t1), axis=1)
        hit = (tn <= tf) & (tf > 0)
        t = np.where(hit & (tn < t), tn, t)
    return np.where(np.isfinite(t), t, 0.0).reshape(H, W).astype(np.float32)   # depth: t, as d_z(cam) = 1


def _frame(T, boxes=(), **extra):
    return {'depth_m': _render(T, boxes), 'K': K, 'T_base_cam': T, **extra}


BOX = (np.array([0.55, -0.05, BENCH_Z]), np.array([0.65, 0.05, 0.10]))      # 10 cm square, ~9 cm tall


def _scene(**kw):
    return OccupancyScene([0.3, -0.4, BENCH_Z], [1.0, 0.4, 0.6], voxel_m=0.01, **kw)


def test_the_block_is_occupied_and_inside_is_negative():
    s = _scene()
    s.add(_frame(_down(0.6, 0.0, 0.6), [BOX]))
    sdf = s.sdf()
    centre = s.index([[0.60, 0.0, 0.05]])[0]                  # inside the block, unseen but below its top
    above = s.index([[0.60, 0.0, 0.20]])[0]                   # 10 cm above its top
    aside = s.index([[0.45, 0.0, 0.05]])[0]                   # beside it, 10 cm from its side
    assert sdf[tuple(centre)] < 0
    assert sdf[tuple(above)] == pytest.approx(0.10, abs=0.015)
    assert sdf[tuple(aside)] == pytest.approx(0.10, abs=0.015)
    lid = _scene(fill_below=False)
    lid.add(_frame(_down(0.6, 0.0, 0.6), [BOX]))
    assert lid.sdf()[tuple(centre)] > 0                       # without fill_below: a lid over nothing


def test_a_frame_that_no_longer_sees_the_block_clears_it():
    s = _scene()
    s.add(_frame(_down(0.6, 0.0, 0.6), [BOX]))
    top = tuple(s.index([[0.60, 0.0, 0.10]])[0])          # the voxel the block's top surface is in
    assert s.occ[top]
    cleared, _ = s.add(_frame(_down(0.6, 0.0, 0.6)))         # the block taken away
    assert cleared > 0 and not s.occ[top]
    assert s.sdf()[tuple(s.index([[0.60, 0.0, 0.05]])[0])] > 0.03
    # A view that sees only the bench beside it does not clear the block.
    s2 = _scene()
    s2.add(_frame(_down(0.6, 0.0, 0.6), [BOX]))
    s2.add(_frame(_down(0.9, 0.0, 0.4)))
    assert s2.occ[top]


def test_what_the_new_view_cannot_see_stays():
    s = _scene()
    s.add(_frame(_down(0.6, 0.0, 0.6), [BOX]))
    top = tuple(s.index([[0.60, 0.0, 0.10]])[0])          # the voxel the block's top surface is in
    occluder = (np.array([0.5, -0.1, 0.30]), np.array([0.7, 0.1, 0.31]))     # a plate above the block
    s.add(_frame(_down(0.6, 0.0, 0.6), [BOX, occluder]))
    assert s.occ[top]


def test_the_gripper_and_the_arm_are_left_out():
    s = _scene()
    T_tool0 = _down(0.6, 0.0, 0.45)                                  # tool0 above the bench, pointing down
    finger = (np.array([0.58, -0.02, 0.30]), np.array([0.62, 0.02, 0.32]))  # 13-15 cm out along tool z
    f = _frame(_down(0.6, 0.0, 0.75), [finger], T_base_tool0=T_tool0)       # the finger 0.43 m off
    p = s.points(f)
    assert len(p) and np.all(p[:, 2] < 0.29)                         # the finger's points are gone
    q = np.radians([0.0, 30.0, 0.0, -90.0, 0.0, 60.0, 0.0])
    links = k.fk_links(q)[2:, :3, 3]
    near_link = (links[3] - 0.02, links[3] + 0.02)
    f2 = _frame(_down(*(links[3][:2]), 1.2), [near_link], joints_rad=q, robot='iiwa14')
    assert not np.any(np.linalg.norm(s.points(f2) - links[3], axis=1) < 0.05)


def test_a_reading_nearer_than_the_camera_can_measure_is_left_out():
    """The D415 reads the fingers, nearer than its ~0.3 m minimum, as ~0.27 m: kept, they stood a
    wall of obstacle in mid-air under the gripper."""
    plate = (np.array([0.55, -0.05, 0.33]), np.array([0.65, 0.05, 0.34]))     # 0.26 m from the camera
    s = _scene()
    s.add(_frame(_down(0.6, 0.0, 0.6), [plate]))
    assert not s.occ[tuple(s.index([[0.60, 0.0, 0.341]])[0])]
    assert s.solid()[tuple(s.index([[0.60, 0.0, 0.05]])[0])] == 0             # nothing under it either
    keep = _scene(min_depth_m=0.0)
    keep.add(_frame(_down(0.6, 0.0, 0.6), [plate]))
    assert keep.occ[tuple(keep.index([[0.60, 0.0, 0.341]])[0])]


def test_save_and_load_keep_the_scene(tmp_path):
    s = _scene()
    s.add(_frame(_down(0.6, 0.0, 0.6), [BOX]))
    s.save(tmp_path / 'scene.npz')
    s2 = OccupancyScene.load(tmp_path / 'scene.npz')
    np.testing.assert_array_equal(s2.occ, s.occ)
    np.testing.assert_allclose(s2.sdf(), s.sdf())


# ---------------------------------------------------------------------------- with cuMotion

DESC = os.path.join(os.path.dirname(k.__file__), '..', 'description', 'iiwa14')


@pytest.fixture(scope='module')
def robot():
    cumotion = pytest.importorskip('cumotion')
    if not os.path.isfile(os.path.join(DESC, 'iiwa14.xrdf')):
        pytest.skip('no iiwa14.xrdf (robot/description/make_iiwa14_xrdf.py writes it)')
    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    return cumotion.load_robot_from_file(os.path.join(DESC, 'iiwa14.xrdf'), os.path.join(DESC, 'iiwa14.urdf'))


def test_cumotion_plans_round_a_block_the_camera_saw(robot):
    import cumotion
    q0, q1 = k.HOME.copy(), k.HOME.copy()
    q0[0], q1[0] = np.radians(50), np.radians(-50)
    mid = k.fk(0.5 * (q0 + q1))[:3, 3]
    block = (np.array([mid[0] - 0.06, mid[1] - 0.06, BENCH_Z]), np.array([mid[0] + 0.06, mid[1] + 0.06, mid[2] + 0.05]))
    s = _scene()
    s.add(_frame(_down(mid[0], mid[1], 1.2), [block]))               # seen only from above
    world, _ = s.to_cumotion(bench_z=BENCH_Z)
    view = world.add_world_view()
    insp = cumotion.create_robot_world_inspector(robot, view)
    assert insp.min_distance_to_obstacle(q0) > 0 and insp.min_distance_to_obstacle(q1) > 0
    assert insp.in_collision_with_obstacle(0.5 * (q0 + q1))         # the straight move hits it
    opt = cumotion.create_trajectory_optimizer(cumotion.create_default_trajectory_optimizer_config(
        robot_description=robot, tool_frame_name='tool0', world_view=view))
    res = opt.plan_to_cspace_target(q0, cumotion.TrajectoryOptimizer.CSpaceTarget(q1))
    assert res.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS, res.status()
    traj = res.trajectory()
    d = traj.domain()
    for t in np.linspace(d.lower, d.upper, 60):
        assert not insp.in_collision_with_obstacle(traj.eval(t))
    # Taken away and seen again: the straight move is clear.
    s.add(_frame(_down(mid[0], mid[1], 1.2)))
    world2, _ = s.to_cumotion(bench_z=BENCH_Z)
    insp2 = cumotion.create_robot_world_inspector(robot, world2.add_world_view())
    assert not insp2.in_collision_with_obstacle(0.5 * (q0 + q1))


def test_scene_build_checks_the_moving_arm_not_its_base(robot):
    """scene_build's check: clear at a pose away from the scene, in collision with a block where an
    arm link is -- and the base beside the bench never counts (it was all a single minimum said)."""
    from cable_assembly.apps import scene_build
    q = k.HOME.copy()
    s = _scene()
    s.add(_frame(_down(0.6, 0.0, 0.6), [BOX]))
    frames = [('x/view_00.npz', {'joints_rad': q, 'robot': 'iiwa14'})]
    assert scene_build.check(s, frames, BENCH_Z)
    wrist = k.fk_links(q)[6, :3, 3]                                  # ~0.5 m up, inside the grid
    pillar = (np.array([wrist[0] - 0.05, wrist[1] - 0.05, BENCH_Z]), wrist + 0.05)
    s.add(_frame(_down(wrist[0], wrist[1], wrist[2] + 0.45), [pillar]))
    assert not scene_build.check(s, frames, BENCH_Z)


def test_the_latest_scan_is_found_and_a_run_directory_is_its_views(tmp_path):
    from cable_assembly.apps._common import frame_files, latest_experiment
    from cable_assembly.config import Config
    exp = tmp_path / 'experiments'
    for stamp in ('29991231_235958', '29991231_235959'):        # newer than any real run
        d = exp / f'splat_scan_{stamp}'
        d.mkdir(parents=True)
        for n in (1, 0):
            (d / f'view_{n:02d}.npz').touch()
    (exp / 'capture_rgbd_29991231_235959').mkdir()
    (exp / 'capture_rgbd_29991231_235959' / 'frame.npz').touch()
    cfg = Config({'data_dir': str(tmp_path)})
    newest = str(exp / 'splat_scan_29991231_235959')
    assert latest_experiment(cfg, 'splat_scan') == newest
    paths, what = frame_files(cfg, [])
    assert paths == [f'{newest}/view_00.npz', f'{newest}/view_01.npz'] and newest in what
    paths, _ = frame_files(cfg, [str(exp / 'capture_rgbd_29991231_235959'), f'{newest}/view_01.npz'])
    assert paths == [str(exp / 'capture_rgbd_29991231_235959' / 'frame.npz'), f'{newest}/view_01.npz']
