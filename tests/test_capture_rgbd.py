"""apps/capture_rgbd.py and the live view's reader of what it writes (sim/src/cable_assembly_sim/
rgbd.py, loaded from its file: the sim package is not installed in this env). The two halves meet
only at frame.npz, so this pins it: a capture of the bench, seen through the bench's real hand-eye
from the pick pose, must come back as Gaussians lying on the bench -- with the gripper left out."""

import importlib.util
import os

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.apps import capture_rgbd
from cable_assembly.config import Config
from cable_assembly.perception.camera import Frame
from cable_assembly.robot.kinematics import iiwa14
from cable_assembly.transforms import from_cfg, inverse

RGBD_PY = os.path.join(os.path.dirname(__file__), '..', 'sim', 'src', 'cable_assembly_sim', 'rgbd.py')
PICK = np.radians([-27, 50, 66, -96, 120, -57, -123])
# A 320x180 D415: its 1280x720 intrinsics scaled by 1/4.
K = np.array([[232.5, 0.0, 160.0], [0.0, 232.3, 90.0], [0.0, 0.0, 1.0]])
W, H = 320, 180


@pytest.fixture(scope='module')
def rgbd():
    spec = importlib.util.spec_from_file_location('rgbd', RGBD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def bench():
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    return {'T_tool0_cam': from_cfg(cfg.section('hand_eye')),
            'T_tool0_tip': from_cfg(cfg.section('fingertip_grasp')),
            'bench_z': float(cfg.get_path('ground_plane.z_mm')) / 1000.0}


def _depth_to_plane(T_base_cam, n_base, d):
    """Each pixel's depth (along the optical axis) to the plane n . p = d, 0 where it misses."""
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    ray_b = ray @ T_base_cam[:3, :3].T
    t = (d - n_base @ T_base_cam[:3, 3]) / (ray_b @ n_base)
    return np.where(t > 0.0, t, 0.0).astype(np.float32)


class FakeCamera:
    """The bench plane, and in front of it a disc square to the tool 150 mm out, within 30 mm of its
    axis: a finger. A few pixels drop out in every frame, different ones each time."""

    def __init__(self, T_base_tool0, bench):
        self.K = K
        T_base_cam = T_base_tool0 @ bench['T_tool0_cam']
        self.depth = _depth_to_plane(T_base_cam, np.array([0.0, 0.0, 1.0]), bench['bench_z'])
        z_tool = T_base_tool0[:3, 2]
        finger = _depth_to_plane(T_base_cam, z_tool, z_tool @ T_base_tool0[:3, 3] + 0.150)
        v, u = np.mgrid[0:H, 0:W]
        p_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1) * finger[..., None]
        T_tool_cam = bench['T_tool0_cam']
        p_tool = p_cam @ T_tool_cam[:3, :3].T + T_tool_cam[:3, 3]
        on_disc = (finger > 0) & (np.hypot(p_tool[..., 0], p_tool[..., 1]) < 0.030)
        self.depth = np.where(on_disc, finger, self.depth).astype(np.float32)
        self.color = np.zeros((H, W, 3), np.uint8)
        self.color[..., 2] = 200                                  # red, in the camera's BGR
        self.rng = np.random.default_rng(0)

    def capture(self):
        d = self.depth.copy()
        d[self.rng.random(d.shape) < 0.05] = 0.0                  # dropouts
        return Frame(self.color.copy(), self.K, np.zeros(5), 0.0, None, d)


def _capture(tmp_path, bench, read_q):
    cfg = Config({'data_dir': str(tmp_path), 'cell': 'iiwa14_bench'})
    ok = capture_rgbd.capture(cfg, FakeCamera(iiwa14.fk(PICK), bench), iiwa14, bench['T_tool0_cam'],
                              bench['T_tool0_tip'], read_q, n_frames=5)
    found = [os.path.join(r, f) for r, _, fs in os.walk(tmp_path) for f in fs if f == 'frame.npz']
    return ok, found


def test_a_capture_comes_back_as_gaussians_on_the_bench(tmp_path, bench, rgbd):
    ok, found = _capture(tmp_path, bench, lambda: PICK)
    assert ok and len(found) == 1
    frame = rgbd.load(found[0])
    assert frame['robot'] == 'iiwa14' and frame['cell'] == 'iiwa14_bench'
    assert frame['rgb'].shape == (H, W, 3) and np.all(frame['rgb'][..., 0] == 200)   # stored as RGB
    np.testing.assert_allclose(frame['T_base_cam'], iiwa14.fk(PICK) @ bench['T_tool0_cam'])
    # The median of 5 frames fills nearly every dropout (a pixel is empty only if all 5 lost it).
    assert (frame['depth_m'] > 0).mean() > 0.99

    # The synthetic finger is nearer than the D415 reads (MIN_DEPTH_M): min_depth_m=0 so the
    # gripper cylinder is what leaves it out.
    g = rgbd.to_gaussians(frame, stride=1, voxel_m=0.0, min_depth_m=0.0)
    p = g['positions']
    assert len(p) > 0.8 * W * H
    np.testing.assert_allclose(p[:, 2], bench['bench_z'], atol=1e-4)             # all on the bench
    assert g['counts']['gripper'] > 0                                           # the finger went
    np.testing.assert_allclose(g['colors'], np.tile([200 / 255, 0, 0], (len(p), 1)), atol=1e-6)
    for k in ('scales', 'orientations', 'opacities', 'colors', 'radii'):
        assert len(g[k]) == len(p)
    np.testing.assert_allclose(np.linalg.norm(g['orientations'], axis=1), 1.0, atol=1e-6)


def test_the_finger_is_left_out_and_nothing_else(tmp_path, bench, rgbd):
    _, found = _capture(tmp_path, bench, lambda: PICK)
    frame = rgbd.load(found[0])
    kept = rgbd.to_gaussians(frame, stride=1, voxel_m=0.0, min_depth_m=0.0)
    every = rgbd.to_gaussians(frame, stride=1, voxel_m=0.0, mask_gripper=False, min_depth_m=0.0)
    assert len(rgbd.to_gaussians(frame, stride=1, voxel_m=0.0, mask_gripper=False)['positions']) \
        < len(every['positions']), 'too near to be a reading, even unmasked'
    on_finger = np.abs(every['positions'][:, 2] - bench['bench_z']) > 1e-3
    assert on_finger.sum() > 0
    T_tool_base = inverse(iiwa14.fk(PICK))
    p_tool = every['positions'] @ T_tool_base[:3, :3].T + T_tool_base[:3, 3]
    np.testing.assert_allclose(p_tool[on_finger, 2], 0.150, atol=1e-4)          # the plate it was
    assert len(kept['positions']) == len(every['positions']) - kept['counts']['gripper']
    assert kept['counts']['gripper'] >= on_finger.sum()


def test_a_voxel_keeps_one_gaussian(tmp_path, bench, rgbd):
    _, found = _capture(tmp_path, bench, lambda: PICK)
    frame = rgbd.load(found[0])
    g = rgbd.to_gaussians(frame, stride=1, voxel_m=0.01)
    keys = np.floor(g['positions'] / 0.01).astype(int)
    assert len(np.unique(keys, axis=0)) == len(keys)
    assert np.all(g['radii'] >= 0.005 - 1e-9)                                   # at least the voxel


def test_an_arm_that_moves_spoils_the_capture(tmp_path, bench):
    qs = iter([PICK, PICK + np.radians([0, 0, 0, 0, 0, 0, 0.5])])
    ok, found = _capture(tmp_path, bench, lambda: next(qs))
    assert not ok and found == []
