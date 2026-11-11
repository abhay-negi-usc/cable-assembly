"""perception/gripper_mask.py: the wrist camera's view of the robot's own gripper, learned from saved
views -- the pixels that never change (2026-11-03: the 6D detector took the gripper's knuckle for the
BNC plug)."""

import numpy as np

from cable_assembly.perception import gripper_mask


def test_the_pixels_that_never_change_are_the_gripper(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for k in range(30):
        rgb = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)       # a different scene each view
        rgb[45:, 10:30] = 40                                          # the gripper, bottom edge, still
        p = tmp_path / f'view_{k:02d}.npz'
        np.savez(p, rgb=rgb)
        paths.append(str(p))
    m = gripper_mask.learn(paths, grow_px=0)
    assert m[50, 20] and m[59, 15]
    assert not m[10, 40] and not m[50, 60]
    import cv2
    png = tmp_path / 'mask.png'
    cv2.imwrite(str(png), m.astype(np.uint8) * 255)
    big = gripper_mask.load(str(png), (120, 160))
    assert big.shape == (120, 160) and big[100, 40] and not big[20, 80]


def test_the_d415s_bench_mask_leaves_the_middle_of_the_view_free():
    """The D415's mask (unused since the D405 took its place, 2026-11-10: pose6d.gripper_mask null)."""
    m = gripper_mask.load('configs/cells/iiwa14_bench.gripper_mask.png', (720, 1280))
    assert m is not None
    assert 0.05 < m.mean() < 0.2
    assert not m[:400].any(), 'the gripper is in the bottom of the wrist view'
    assert not m[470, 990], 'where the plug lay against the right finger pad (2026-11-03)'


def test_only_the_newest_cameras_views_are_learned_from(tmp_path):
    """2026-11-10: the iiwa's D405 took the D415's place. The D415's hundreds of saved views show the
    gripper elsewhere; learned from with the D405's, the mask is neither camera's."""
    rng = np.random.default_rng(1)
    paths = []
    for k in range(50):
        old = k < 30                                                  # the D415's views first
        rgb = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        if old:
            rgb[45:, 50:70] = 40                                      # its gripper, bottom right
        else:
            rgb[45:, 10:30] = 40                                      # the new camera's, bottom left
        K = np.array([[930.0 if old else 650.0, 0.0, 40.0], [0.0, 930.0 if old else 650.0, 30.0], [0.0, 0.0, 1.0]])
        p = tmp_path / f'view_{k:02d}.npz'
        np.savez(p, rgb=rgb, K=K)
        paths.append(str(p))
    m = gripper_mask.learn(paths, grow_px=0)
    assert m[50, 20] and not m[50, 60]
    try:
        gripper_mask.learn(paths[:40], grow_px=0)                     # 10 of the new camera's: too few
    except ValueError as exc:
        assert 'newest camera' in str(exc)
    else:
        raise AssertionError('too few views of the newest camera must refuse, not fall back to the old')
