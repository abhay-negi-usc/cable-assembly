"""The iiwa tool's D405 mesh, cable_assembly/robot/description/tool/d405.stl, from Intel's own.

Intel's d405.stl (realsense-ros, realsense2_description/meshes, Apache-2.0: millimetres, the front plate at
z = 0, the module along -z, centred on its tripod screw across) put in the CAD frame of the D405's flange mount
(d405_iiwa_mount.step, 2026-11-12; metres), the way _d405.urdf.xacro hangs it off its screw: the mesh in the
camera_link frame at [3.8, -9, 0] mm, rpy [90, 0, 90] deg; the optical frame rpy [-90, 0, -90] from
camera_link -- so the mesh lands where tool_geometry.T_CAD_CAMERA puts the D405's optics: its bottom on the
mount's tab, its screw on the tab's 1/4-20 hole, its back on the plate over its two M3 holes
(tests/test_tool_meshes.py checks it).

    python analysis/d405_tool_mesh.py \\
        ~/jeon_team_ws/aruco_pose/src/realsense-ros/realsense2_description/meshes/d405.stl
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cable_assembly.robot import tool_geometry as G  # noqa: E402

# _d405.urdf.xacro: camera_link -> the mesh, and camera_link -> the optical frame.
MESH_IN_LINK_XYZ_M = (0.0038, -0.009, 0.0)
MESH_IN_LINK_RPY = (np.pi / 2, 0.0, np.pi / 2)
OPTICAL_IN_LINK_RPY = (-np.pi / 2, 0.0, -np.pi / 2)


def _t(rpy=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def write_stl(path, points, tris):
    """A binary STL of `tris` (T, 3) into `points` (V, 3), normals from the winding."""
    v = np.asarray(points, dtype=np.float64)[np.asarray(tris)]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    rec = np.zeros(len(v), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = n, v
    with open(path, 'wb') as f:
        f.write(b'D405 (Intel realsense2_description d405.stl) in the d405_iiwa_mount CAD frame, m'.ljust(80, b' '))
        f.write(np.uint32(len(v)).tobytes())
        f.write(rec.tobytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('intel_stl', help="realsense2_description/meshes/d405.stl (millimetres)")
    ap.add_argument('out', nargs='?', default=G.CAMERA_STL)
    a = ap.parse_args(argv)
    points, tris = G.read_stl(os.path.expanduser(a.intel_stl))
    T_link_mesh = _t(MESH_IN_LINK_RPY, MESH_IN_LINK_XYZ_M)
    T_link_opt = _t(OPTICAL_IN_LINK_RPY)
    T_cad_mesh = G.T_CAD_CAMERA @ np.linalg.inv(T_link_opt) @ T_link_mesh
    p = (points * 1e-3) @ T_cad_mesh[:3, :3].T + T_cad_mesh[:3, 3]
    write_stl(a.out, p, tris)
    print(f'{a.out}: {len(tris)} triangles, CAD bbox {np.round(p.min(0) * 1e3, 2)} .. {np.round(p.max(0) * 1e3, 2)} mm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
