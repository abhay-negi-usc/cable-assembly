"""apps/splat_scan.py: the wrist camera tilts and pans about the point it looks at, for the live
view's camera layer. Planned and checked before anything moves, so these pin the plan: the
geometry of the views, what the check leaves out from the bench's pick pose, and a whole scan on
the simulated arm with a camera that sees the bench -- every view kept, in a format a splat trainer
reads, placing the bench where it is."""

import json
import os
import types

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.apps import splat_scan as S
from cable_assembly.perception.camera import Frame
from cable_assembly.robot import Robot, make_arm
from cable_assembly.transforms import from_cfg, inverse

K = np.array([[116.0, 0.0, 80.0], [0.0, 116.0, 45.0], [0.0, 0.0, 1.0]])
W, H = 160, 90


@pytest.fixture(scope='module')
def cfg():
    return C.load('splat_scan', ['robot.dry_run=true'])


@pytest.fixture(scope='module')
def bench(cfg):
    return {'T_tool0_cam': from_cfg(cfg.section('hand_eye')), 'tip': from_cfg(cfg.section('fingertip_grasp')),
            'z': cfg.get_path('ground_plane.z_mm') / 1000.0,
            'q_pick': np.radians(np.asarray(cfg.get('pick_joints_deg'), dtype=float))}


def _pivot_on_bench(T0, bench):
    T = T0 @ bench['T_tool0_cam']
    o, d = T[:3, 3], T[:3, 2]
    return o + (bench['z'] - o[2]) / d[2] * d, o


def test_every_view_looks_at_the_pivot_from_the_tilt_asked(cfg, bench):
    arm = make_arm(cfg)
    T0 = arm.fk(bench['q_pick'])
    pivot, cam0 = _pivot_on_bench(T0, bench)
    tilts = np.radians([12.0, 24.0])
    plan = S.orbit_views(T0, inverse(T0) @ np.append(pivot, 1), inverse(T0) @ np.append(cam0, 1), tilts, 8)
    assert len(plan) == 16
    sight0 = (pivot - cam0) / np.linalg.norm(pivot - cam0)
    for i, (label, T) in enumerate(plan):
        Tc = T @ bench['T_tool0_cam']
        to_pivot = pivot - Tc[:3, 3]
        assert np.linalg.norm(to_pivot) == pytest.approx(np.linalg.norm(pivot - cam0), abs=1e-9)   # same distance
        sight = to_pivot / np.linalg.norm(to_pivot)
        # The camera's own axes turn with it: the pivot stays where it was in the image.
        np.testing.assert_allclose(inverse(Tc)[:3, :3] @ sight, inverse(T0 @ bench['T_tool0_cam'])[:3, :3] @ sight0, atol=1e-9)
        assert np.degrees(np.arccos(np.clip(sight @ sight0, -1, 1))) == pytest.approx(12.0 if i < 8 else 24.0, abs=1e-6)
    pans = [lbl.split('pan ')[1] for lbl, _ in plan]
    assert pans[:8] == pans[8:][::-1]                        # the second ring walked back


def test_from_the_pick_pose_only_the_blocked_views_are_left_out(cfg, bench):
    arm = make_arm(cfg)
    q0 = bench['q_pick']
    T0 = arm.fk(q0)
    pivot, cam0 = _pivot_on_bench(T0, bench)
    plan = S.orbit_views(T0, inverse(T0) @ np.append(pivot, 1), inverse(T0) @ np.append(cam0, 1),
                         np.radians([12.0, 24.0]), 8)
    kept = S.check_plan(arm, q0, T0, plan, bench['tip'], bench['z'], 0.06)
    assert [lbl for lbl, _ in plan if lbl not in [k for k, _ in kept]] == \
        ['tilt 24 deg, pan 180 deg', 'tilt 24 deg, pan 135 deg']


@pytest.mark.parametrize('tilts', [[15.0, 30.0, 45.0], [20.0, 40.0]])
def test_every_line_kept_is_one_move_l_can_drive(cfg, bench, tilts):
    """The plan walks every line as move_l does, not only its middle and end. Checked that way,
    both these ring sets kept a line that has no IK, or a 5 deg joint jump, part way along -- the
    scan would have stopped there, mid-orbit. Walked independently here, every millimetre."""
    from cable_assembly.robot.backends.iiwa_fri import (
        _LINE_MAX_JOINT_STEP,
        _LINE_STEP_M,
        _LINE_STEP_RAD,
    )
    from cable_assembly.transforms import pose_error, slerp_matrix

    arm = make_arm(cfg)
    q0 = bench['q_pick']
    T0 = arm.fk(q0)
    pivot, cam0 = _pivot_on_bench(T0, bench)
    plan = S.orbit_views(T0, inverse(T0) @ np.append(pivot, 1), inverse(T0) @ np.append(cam0, 1),
                         np.radians(tilts), 8)
    kept = S.check_plan(arm, q0, T0, plan, bench['tip'], bench['z'], 0.06)
    assert kept is not None and len(kept) >= len(plan) // 2
    q, T_prev = q0, T0
    for label, T in [*kept, ('back', T0)]:
        lin, ang = pose_error(T_prev, T)
        n = max(1, int(np.ceil(lin / _LINE_STEP_M)), int(np.ceil(ang / _LINE_STEP_RAD)))
        for k in range(1, n + 1):
            nxt = arm.model.ik(slerp_matrix(T_prev, T, k / n), q, arm.joint_limits)
            assert nxt is not None and np.max(np.abs(nxt - q)) <= _LINE_MAX_JOINT_STEP, f'{label} at {k}/{n}'
            q = nxt
        T_prev = T


def test_a_start_too_near_the_bench_is_refused(cfg, bench):
    arm = make_arm(cfg)
    q0 = bench['q_pick']
    T0 = arm.fk(q0)
    pivot, cam0 = _pivot_on_bench(T0, bench)
    plan = S.orbit_views(T0, inverse(T0) @ np.append(pivot, 1), inverse(T0) @ np.append(cam0, 1),
                         np.radians([12.0]), 8)
    tip_z = (T0 @ bench['tip'])[2, 3] - bench['z']
    assert S.check_plan(arm, q0, T0, plan, bench['tip'], bench['z'], tip_z + 0.01) is None


def test_the_pivot_comes_from_the_depth_or_else_the_bench(bench):
    T0 = np.eye(4)
    T0[:3, :3] = np.diag([1.0, -1.0, -1.0])                  # tool0 straight down
    T0[:3, 3] = [0.6, 0.0, 0.4]
    T_tool0_cam = np.eye(4)
    depth = np.full((H, W), 0.30, np.float32)
    frames = [Frame(np.zeros((H, W, 3), np.uint8), K, np.zeros(5), 0.0, None, depth)] * 3
    np.testing.assert_allclose(S.look_point(frames, T_tool0_cam, T0, 0.0), [0.6, 0.0, 0.10], atol=1e-6)
    blind = [Frame(np.zeros((H, W, 3), np.uint8), K, np.zeros(5), 0.0, None, None)]
    np.testing.assert_allclose(S.look_point(blind, T_tool0_cam, T0, 0.0), [0.6, 0.0, 0.0])
    up = T0.copy()
    up[:3, :3] = np.eye(3)                                  # looking up: nothing to scan about
    assert S.look_point(blind, T_tool0_cam, up, 0.0) is None


class BenchCamera:
    """Sees the bench plane from wherever the arm is (its pose_fn), with depth."""

    def __init__(self, robot, bench_z):
        self.robot, self.bench_z, self.K, self.listeners = robot, bench_z, K, []

    def capture(self):
        T = self.robot.camera()
        v, u = np.mgrid[0:H, 0:W]
        ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1) @ T[:3, :3].T
        t = (self.bench_z - T[2, 3]) / ray[..., 2]
        depth = np.where(t > 0, t, 0.0).astype(np.float32)
        color = np.full((H, W, 3), 90, np.uint8)
        return Frame(color, K, np.zeros(5), 0.0, T, depth)

    def close(self):
        pass


def test_a_scan_keeps_every_view_where_the_bench_is(tmp_path, bench):
    cfg = C.load('splat_scan', ['robot.dry_run=true', 'splat_scan.from_pick=true', 'splat_scan.settle_s=0',
                                'splat_scan.frames_per_view=2', f'data_dir={tmp_path}', 'live_view.enabled=false'])
    robot = Robot(cfg, with_gripper=False, needs=())
    try:
        camera = BenchCamera(robot, bench['z'])
        assert S.scan(cfg, robot, camera, types.SimpleNamespace(yes=True))
        np.testing.assert_allclose(robot.arm.q(), bench['q_pick'], atol=1e-6)          # back at the start
    finally:
        robot.close()
    (out,) = [os.path.join(tmp_path, 'experiments', d) for d in os.listdir(os.path.join(tmp_path, 'experiments'))]
    views = sorted(f for f in os.listdir(out) if f.endswith('.npz'))
    assert len(views) == 19                                  # the start and 18 of the 24 planned
    with open(os.path.join(out, 'transforms.json')) as f:
        doc = json.load(f)
    assert len(doc['frames']) == 19 and doc['fl_x'] == K[0, 0] and doc['depth_unit_scale_factor'] == 0.001
    for entry, name in zip(doc['frames'], views):
        z = np.load(os.path.join(out, name))
        np.testing.assert_allclose(np.array(entry['transform_matrix']), z['T_base_cam'] @ np.diag([1, -1, -1, 1]))
        # The view's centre pixel, back through its pose, lands on the bench.
        d = float(z['depth_m'][H // 2, W // 2])
        T = z['T_base_cam']
        p = T[:3, 3] + d * (T[:3, :3] @ np.array([(W // 2 - K[0, 2]) / K[0, 0], (H // 2 - K[1, 2]) / K[1, 1], 1.0]))
        assert p[2] == pytest.approx(bench['z'], abs=1e-6)
        assert os.path.isfile(os.path.join(out, entry['file_path']))
        assert os.path.isfile(os.path.join(out, entry['depth_file_path']))
    # The views agree: exact depth from exact poses. A pose 4 mm off shows as 4 mm.
    paths = [os.path.join(out, v) for v in views]
    assert S.report_agreement(paths)
    loaded = [dict(np.load(p)) for p in paths]
    assert max(np.median(np.abs(d)) for d in S.agreement(loaded)) < 1e-4
    off = dict(loaded[5], T_base_cam=loaded[5]['T_base_cam'].copy())
    off['T_base_cam'][2, 3] += 0.004
    (d,) = S.agreement([loaded[0], off])
    assert np.median(np.abs(d)) == pytest.approx(0.004, abs=0.0005)
