"""The camera mount and D405 meshes (cable_assembly/robot/description/tool/*.stl) as the sim
places them on tool0 (sim/scripts/build_iiwa14.py: the mount's CAD frame, the flange face at z 0, turned
a half turn about z) agree with the camera's optics and with the cell's hand-eye
(configs/cells/iiwa14_bench.yaml): the D405 on its own mount (d405_iiwa_mount.step, 2026-11-12), its bottom
on the tab with its tripod screw on the tab's 1/4-20 hole, its back on the plate with its M3's on the plate's
two holes, its front plate looking along the optical axis just in front of the optical origin, which is its
left imager."""

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


def test_the_d405_sits_on_the_bracket_where_its_optics_are(build):
    from cable_assembly.robot import tool_geometry as G
    points, faces = build.read_stl(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['d405'][0]))
    c = _in_camera(_in_tool0(build, points), G.T_TOOL0_CAMERA_CAD)     # x right, y down, z forward
    tri = c[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * area, 1e-15)[:, None]
    front = n[:, 2] > 0.98
    assert area[front].sum() > 1500e-6                         # the front plate, 42 x 42 less its corners
    assert c[:, 2].max() == pytest.approx(0.0038, abs=2e-4)    # the plate 3.8 mm in front of the optics
    lo, hi = c.min(axis=0), c.max(axis=0)
    assert hi - lo == pytest.approx([0.042, 0.042, 0.023], abs=2e-4)
    assert (lo + hi)[0] / 2 == pytest.approx(0.009, abs=2e-4)  # the left imager, 9 mm left of the middle
    assert (lo + hi)[1] / 2 == pytest.approx(0.0, abs=2e-4)    # at mid-height
    # its bottom on the tab, its screw on the 1/4-20 hole's axis; its back on the plate, its M3's on its holes
    m = _in_camera(_in_tool0(build, build.read_stl(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['flange_mount'][0]))[0]),
                   G.T_TOOL0_CAMERA_CAD)
    assert not np.all((m > lo + 1e-4) & (m < hi - 1e-4), axis=1).any(), 'the D405 runs into the mount'
    T_cam_cad = np.linalg.inv(G.T_CAD_CAMERA)
    screw = T_cam_cad[:3, :3] @ G.SCREW_ON_SEAT_M + T_cam_cad[:3, 3]
    assert hi[1] == pytest.approx(screw[1], abs=2e-4)                  # the bottom on the seat
    assert screw[0] == pytest.approx(0.009, abs=1e-4)                  # the screw centred across the camera
    assert screw[2] - lo[2] == pytest.approx(0.00835, abs=2e-4)        # 8.35 mm in front of its back

    def hole(points, axis, centre, r_max, span):
        """The diameter of the mount's hole round `centre` along camera `axis` (1 = y, 2 = z) over `span`."""
        k = [i for i in range(3) if i != axis]
        d = np.hypot(points[:, k[0]] - centre[k[0]], points[:, k[1]] - centre[k[1]])
        w = points[(d < r_max) & (points[:, axis] > span[0]) & (points[:, axis] < span[1])]
        A = np.c_[2 * w[:, k[0]], 2 * w[:, k[1]], np.ones(len(w))]
        c0, c1, q = np.linalg.lstsq(A, w[:, k[0]] ** 2 + w[:, k[1]] ** 2, rcond=None)[0]
        return 2 * np.sqrt(q + c0 ** 2 + c1 ** 2), np.hypot(c0 - centre[k[0]], c1 - centre[k[1]])

    dia, off = hole(m, 1, screw, 0.0045, (screw[1] + 1e-4, screw[1] + 0.0034))
    assert dia == pytest.approx(0.0070, abs=1e-4) and off < 1e-4      # 1/4-20 clearance, centred on the screw
    for h in G.BACK_HOLES_M:
        p = T_cam_cad[:3, :3] @ h + T_cam_cad[:3, 3]
        assert p[2] == pytest.approx(lo[2], abs=2e-4) and p[1] == pytest.approx(0.0, abs=1e-4)   # mid-height
        assert abs(p[0] - 0.009) == pytest.approx(0.010, abs=1e-4)    # 20 mm apart, either side of the middle
        dia, off = hole(m, 2, p, 0.0025, (lo[2] - 0.0029, lo[2] - 1e-4))
        assert dia == pytest.approx(0.00335, abs=1e-4) and off < 1e-4   # M3 clearance, centred on the thread


@pytest.mark.parametrize('cell, mm, deg', [('iiwa14_bench', 6.0, 1.5),     # MEASURED 2026-11-12: 1.9 mm / 0.98 deg
                                            ('iiwa14_sim', 1e-1, 0.05)])    # the CAD itself
def test_the_cells_hand_eye_is_near_the_cad(build, cell, mm, deg):
    """The bench's hand-eye is MEASURED on the new mount (2026-11-12) and agrees with its CAD to a few mm and
    about a degree, as the old mount's did (4.9 mm / 1.0 deg); the sim cell carries the CAD."""
    from cable_assembly.robot import tool_geometry as G
    T_cam = from_cfg(C.load('bnc_assembly', [f'cell={cell}']).section('hand_eye'))
    d = np.linalg.inv(G.T_TOOL0_CAMERA_CAD) @ T_cam
    assert 1000.0 * np.linalg.norm(d[:3, 3]) < mm
    assert np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude()) < deg


def test_the_mount_is_centred_on_the_flange_and_sits_on_it(build):
    from cable_assembly.robot import tool_geometry as G
    points, faces = build.read_stl(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['flange_mount'][0]))
    p = _in_tool0(build, points)
    assert p[:, 2].min() == pytest.approx(0.0, abs=1e-6)       # nothing behind the flange face
    tri = p[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flat = np.abs(n[:, 2]) > 0.99 * np.linalg.norm(n, axis=1)
    ring = np.all(np.linalg.norm(tri[..., :2], axis=2) < 0.0376, axis=1)
    seat = tri[flat & ring & (n[:, 2] < 0) & np.all(np.abs(tri[..., 2]) < 1e-6, axis=1)].reshape(-1, 3)
    r = np.linalg.norm(seat[:, :2], axis=1)                     # the face on the flange: the 32 mm bore to the rim
    assert r.min() == pytest.approx(0.016, abs=5e-4) and r.max() == pytest.approx(0.0375, abs=5e-4)
    assert np.linalg.norm(seat[:, :2].mean(axis=0)) < 1e-3     # round the axis
    shoulder = tri[flat & ring & (n[:, 2] > 0) & np.all(np.abs(tri[..., 2] - G.PLATE_M) < 1e-6, axis=1)].reshape(-1, 3)
    r = np.linalg.norm(shoulder[:, :2], axis=1)                 # the coupling's seat, round the 62.9 mm boss
    assert r.min() == pytest.approx(0.03145, abs=5e-4) and r.max() == pytest.approx(0.0375, abs=5e-4)
    assert p[np.linalg.norm(p[:, :2], axis=1) < 0.031, 2].max() == pytest.approx(G.PLATE_M + 0.0025, abs=1e-6)
    # the bracket and the camera on tool -X
    assert p[:, 0].min() == pytest.approx(-0.09441, abs=1e-4) and p[:, 0].max() == pytest.approx(0.0375, abs=1e-4)


def test_the_sim_and_the_planner_place_the_tool_alike(build):
    """sim/scripts/build_iiwa14.py (no cable_assembly import) and robot/tool_geometry.py carry the
    same two placement numbers and read the same STLs."""
    from cable_assembly.robot import tool_geometry as G
    assert build.CAD_FLANGE_FACE_Z_M == G.CAD_FLANGE_FACE_Z_M
    assert build.CAD_TO_TOOL0_YAW_DEG == G.CAD_TO_TOOL0_YAW_DEG
    assert os.path.samefile(build.TOOL_DIR, G.TOOL_DIR)
    assert os.path.samefile(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['d405'][0]), G.CAMERA_STL)
    assert os.path.samefile(os.path.join(build.TOOL_DIR, build.TOOL_PARTS['flange_mount'][0]), G.MOUNT_STL)
    assert build.BENCH_COUPLER_MM == pytest.approx(1000.0 * G.PLATE_M + 11.14)
    p, _ = G.mesh_in_tool0(G.CAMERA_STL)
    q = _in_tool0(build, build.read_stl(G.CAMERA_STL)[0])
    np.testing.assert_allclose(p, q, atol=1e-9)
