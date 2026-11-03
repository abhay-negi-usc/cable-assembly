"""The camera mount and D415 meshes (cable_assembly/robot/description/tool/*.stl) as the sim
places them on tool0 (sim/scripts/build_iiwa14.py: the CAD frame, the flange face at 72.4 mm, turned
-90 deg about z) agree with the MEASURED hand-eye (configs/cells/iiwa14_bench.yaml): the D415's front
face looks along the calibrated optical axis, and the calibrated optical origin sits just behind
that face, in the middle of the module's height."""

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


def test_the_d415_sits_where_the_hand_eye_says(build):
    points, faces = build.read_stl(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['d415'][0]))
    p = _in_tool0(build, points)
    tri = p[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * area, 1e-15)[:, None]
    T_cam = from_cfg(C.load('bnc_assembly', ['cell=iiwa14_bench']).section('hand_eye'))
    axis, origin = T_cam[:3, 2], T_cam[:3, 3]
    front = n @ axis > 0.98                                    # the face looking along the optical axis
    assert area[front].sum() > 1000e-6                         # the front cover, ~1460 mm^2
    face_normal = np.average(n[front], axis=0, weights=area[front])
    assert np.degrees(np.arccos(face_normal @ axis / np.linalg.norm(face_normal))) < 1.5
    centre = np.average(tri[front].mean(axis=1), axis=0, weights=area[front])
    behind = (centre - origin) @ axis                          # the lens centre behind the cover
    assert 0.0 < behind < 0.015, behind
    across = (centre - origin) - behind * axis
    assert abs(across @ T_cam[:3, 1]) < 0.006                  # mid-height of the module


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
    p, _ = G.mesh_in_tool0(G.D415_STL)
    q = _in_tool0(build, build.read_stl(G.D415_STL)[0])
    np.testing.assert_allclose(p, q, atol=1e-9)
