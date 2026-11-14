"""The camera mount and D435i meshes (cable_assembly/robot/description/tool/*.stl) as the sim
places them on tool0 (sim/scripts/build_iiwa14.py: the CAD frame, the flange face at 72.4 mm, turned
-90 deg about z) agree with the camera's optics and with the cell's hand-eye
(configs/cells/iiwa14_bench.yaml): the D435i (2026-11-13, on the lab's mount by its 1/4-20 screw, as the
D415 was) sits on the mount's bracket, its tripod screw on the bracket hole's axis, its front plate
looking along the optical axis 4.3 mm in front of the optical origin, which is its colour imager."""

import importlib.util
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cable_assembly import config as C
from cable_assembly.transforms import from_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def build():
    spec = importlib.util.spec_from_file_location('build_iiwa14', os.path.join(ROOT, 'sim', 'scripts', 'build_iiwa14.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _in_tool0(build, points):
    R = Rotation.from_euler('z', build.CAD_TO_TOOL0_YAW_DEG, degrees=True).as_matrix()
    return (points - [0.0, 0.0, build.CAD_FLANGE_FACE_Z_M]) @ R.T


def _in_camera(points, T_tool0_cam):
    return (points - T_tool0_cam[:3, 3]) @ T_tool0_cam[:3, :3]


def test_the_d435i_sits_on_the_bracket_where_its_optics_are(build):
    from cable_assembly.robot import tool_geometry as G
    points, faces = build.read_stl(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['camera'][0]))
    c = _in_camera(_in_tool0(build, points), G.T_TOOL0_CAMERA_CAD)     # x right, y down, z forward
    tri = c[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * area, 1e-15)[:, None]
    front = n[:, 2] > 0.98
    assert area[front].sum() > 1200e-6                         # the front plate, 90 x 25 less its rounded ends
    assert c[:, 2].max() == pytest.approx(0.0043, abs=2e-4)    # the plate 4.3 mm in front of the optics
    lo, hi = c.min(axis=0), c.max(axis=0)
    assert hi - lo == pytest.approx([0.0892, 0.025, 0.025], abs=2e-4)   # Intel's model: 89.2 wide (spec 90)
    assert (lo + hi)[0] / 2 == pytest.approx(0.0325, abs=4e-4)  # the colour imager, 32.5 mm left of the middle
    assert (lo + hi)[1] / 2 == pytest.approx(0.0, abs=2e-4)     # at mid-height
    # its base on the bracket's seat, its screw on the bracket hole's axis
    m = _in_camera(_in_tool0(build, build.read_stl(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['flange_mount'][0]))[0]),
                   G.T_TOOL0_CAMERA_CAD)
    assert not np.all((m > lo + 1e-4) & (m < hi - 1e-4), axis=1).any(), 'the D435i runs into the mount'
    screw = np.subtract(G.SCREW_IN_D415_M, G.T_D415_CAMERA[:3, 3])        # in the D435i's frame
    assert hi[1] == pytest.approx(screw[1], abs=2e-4)
    wall = m[(np.hypot(m[:, 0] - screw[0], m[:, 2] - screw[2]) < 0.004) & (m[:, 1] > screw[1] - 1e-4)
             & (m[:, 1] < screw[1] + 0.0041)]
    A = np.c_[2 * wall[:, 0], 2 * wall[:, 2], np.ones(len(wall))]
    cx, cz, k = np.linalg.lstsq(A, wall[:, 0] ** 2 + wall[:, 2] ** 2, rcond=None)[0]
    assert np.sqrt(k + cx ** 2 + cz ** 2) * 2 == pytest.approx(0.0065, abs=1e-4)   # 1/4-20 clearance
    assert np.hypot(cx - screw[0], cz - screw[2]) < 1e-4


@pytest.mark.parametrize('cell, mm, deg', [('iiwa14_bench', 6.0, 1.5), ('iiwa14_sim', 1e-1, 0.05)])
def test_the_cells_hand_eye_is_near_the_cad(build, cell, mm, deg):
    """The cell's hand-eye is the D415's MEASURED one carried over to the D435i about the shared screw (a prior
    until the D435i's own calibration, 2026-11-13); the D415's measured one agreed with the CAD to 4.9 mm / 1.0 deg."""
    from cable_assembly.robot import tool_geometry as G
    T_cam = from_cfg(C.load('bnc_assembly', [f'cell={cell}']).section('hand_eye'))
    d = np.linalg.inv(G.T_TOOL0_CAMERA_CAD) @ T_cam
    assert 1000.0 * np.linalg.norm(d[:3, 3]) < mm
    assert np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude()) < deg


def test_the_mount_is_centred_on_the_flange_and_sits_on_it(build):
    points, faces = build.read_stl(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['flange_mount'][0]))
    p = _in_tool0(build, points)
    assert p[:, 2].min() == pytest.approx(-0.0025, abs=2e-4)   # the 2.5 mm rim round the flange
    tri = p[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flat_at_0 = (n[:, 2] < 0) & (np.abs(n[:, 2]) > 0.99 * np.linalg.norm(n, axis=1)) & np.all(np.abs(tri[..., 2]) < 1e-4, axis=1)
    seat = tri[flat_at_0].reshape(-1, 3)                       # the face the flange sits on, facing it
    r = np.linalg.norm(seat[:, :2], axis=1)
    assert r.min() == pytest.approx(0.0218, abs=5e-4) and r.max() == pytest.approx(0.0325, abs=5e-4)
    assert np.linalg.norm(seat[:, :2].mean(axis=0)) < 1e-3     # round the axis


def test_the_sim_and_the_planner_place_the_tool_alike(build):
    """sim/scripts/build_iiwa14.py (no cable_assembly import) and robot/tool_geometry.py carry the
    same two placement numbers and read the same STLs."""
    from cable_assembly.robot import tool_geometry as G
    assert build.CAD_FLANGE_FACE_Z_M == G.CAD_FLANGE_FACE_Z_M
    assert build.CAD_TO_TOOL0_YAW_DEG == G.CAD_TO_TOOL0_YAW_DEG
    assert os.path.samefile(build.TOOL_DIR, G.TOOL_DIR)
    assert os.path.samefile(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['camera'][0]), G.CAMERA_STL)
    assert os.path.samefile(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['flange_mount'][0]), G.MOUNT_STL)
    assert build.BENCH_COUPLER_MM == pytest.approx(1000.0 * G.PLATE_M + 11.14)
    p, _ = G.mesh_in_tool0(G.CAMERA_STL)
    q = _in_tool0(build, build.read_stl(G.CAMERA_STL)[0])
    np.testing.assert_allclose(p, q, atol=1e-9)
