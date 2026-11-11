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


def test_the_bench_mask_leaves_the_middle_of_the_view_free():
    m = gripper_mask.load('configs/cells/iiwa14_bench.gripper_mask.png', (720, 1280))
    assert m is not None
    assert 0.05 < m.mean() < 0.2
    assert not m[:400].any(), 'the gripper is in the bottom of the wrist view'
    assert not m[470, 990], 'where the plug lay against the right finger pad (2026-11-03)'
