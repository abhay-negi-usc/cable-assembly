"""robot/held_part.py and Robot.hold / release: the part in the hand, its CAD on the tool -- read from
a PLY, carried by the plain-move check, drawn by the live view, and let go of when the gripper opens.
The CAD here is a synthetic one (a long box), so nothing outside the repo is needed."""

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.robot import tool_geometry
from cable_assembly.robot.held_part import HeldPart
from cable_assembly.robot.kinematics import iiwa14 as k


def _box_ply(path, lo, hi):
    """A closed box as a binary PLY in mm, the 6D models' format (float x, y, z; uchar-int faces)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype='<f4')
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    faces = np.array([(3, t) for t in f], dtype=np.dtype([('n', 'u1'), ('i', '<i4', 3)]))
    with open(path, 'wb') as out:
        out.write(b'ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty float x\n'
                  b'property float y\nproperty float z\nelement face 12\n'
                  b'property list uchar int vertex_indices\nend_header\n')
        out.write(v.tobytes())
        out.write(faces.tobytes())
    return str(path)


def test_a_ply_is_read_and_every_vertex_is_inside_a_sphere(tmp_path):
    path = _box_ply(tmp_path / 'box.ply', (-7, -7, 100), (7, 7, 150))
    points, tris = tool_geometry.read_ply(path)
    assert points.shape == (8, 3) and tris.shape == (12, 3)
    part = HeldPart('box', path, 0.001, np.eye(4), 0.007)
    c = np.array([s[0] for s in part.spheres])
    r = np.array([s[1] for s in part.spheres])
    d = np.linalg.norm(points[:, None] * 0.001 - c[None], axis=2) - r[None]
    assert np.all(d.min(axis=1) <= 0.0)
    assert part.live()['mesh'] == path and np.allclose(part.live()['T_tool0'], np.eye(4))


def _dry_iiwa(with_gripper=True):
    from cable_assembly.robot import Robot
    return Robot(C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false']),
                 with_gripper=with_gripper, needs=())


def test_the_gripper_opening_lets_go_of_the_held_part(tmp_path):
    """hold() puts the part on the move check; opening the fingers past where they held it takes it
    off again, whoever opens them; squeezing harder does not."""
    pytest.importorskip('cumotion')
    robot = _dry_iiwa()
    try:
        assert robot.gripper.close('grip')
        part = HeldPart('box', _box_ply(tmp_path / 'box.ply', (-7, -7, 100), (7, 7, 150)), 0.001,
                        robot.T_tool0_fingertip, 0.007)
        robot.hold(part)
        assert robot.held is part and robot.arm.path_check._held is not None
        robot.gripper.go_to(255, 'squeeze')
        assert robot.held is part
        assert robot.gripper.open('put down')
        assert robot.held is None and robot.arm.path_check._held is None
    finally:
        robot.close()


def test_the_move_check_carries_the_held_part():
    """The bare tool may go down to 40 mm over the plane; holding a part that reaches 150 mm past the
    fingertips, the same line is refused -- the part would go through the bench."""
    pytest.importorskip('cumotion')
    import tempfile
    robot = _dry_iiwa(with_gripper=False)
    try:
        q0 = k.HOME.copy()
        robot.arm._q = q0.copy()
        T = robot.arm.fk(q0)
        tip = (T @ robot.T_tool0_fingertip)[:3, 3]
        down = T.copy()
        down[2, 3] -= tip[2] - 0.04                      # the bare fingertips 40 mm over the plane
        T_tool0_rod = np.eye(4)                          # along tool0's z, from the fingertips on
        T_tool0_rod[:3, 3] = robot.T_tool0_fingertip[:3, 3]
        with tempfile.TemporaryDirectory() as d:
            part = HeldPart('rod', _box_ply(f'{d}/rod.ply', (-7, -7, 0), (7, 7, 150)), 0.001, T_tool0_rod, 0.007)
        robot.hold(part)
        assert robot.arm.move_l(down, label='down') is False
        np.testing.assert_allclose(robot.arm.q(), q0, atol=1e-9)
        robot.release()
        assert robot.arm.move_l(down, label='down')
    finally:
        robot.close()
