"""The iiwa tool's D435i mesh, cable_assembly/robot/description/tool/d435i.stl, from Intel's own.

Intel's D435 model (librealsense common/res/sources/D435.obj, Apache-2.0 -- the D435i's housing is the D435's:
millimetres, x across to the camera's LEFT, y up, z forward, the front plate at z = 0, centred across and up and
down on the module) put in the CAD frame of the lab's flange mount (realsense_flangemount.stl; metres), the way
_d435.urdf.xacro hangs it off its tripod screw: the screw centred across and at the bottom, the front plate
14.9 mm in front of it. In the colour optical frame (x right, y down, z forward) tool_geometry.T_CAD_CAMERA puts
the screw at [+32.5, +12.5, -10.6] mm, so a model point (x, y, z) lands at [32.5 - x, -y, z + 4.3] mm.
(tests/test_tool_meshes.py checks it.)

    python analysis/d435i_tool_mesh.py ~/librealsense/common/res/sources/D435.obj
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cable_assembly.robot import tool_geometry as G  # noqa: E402


def read_obj(path):
    """A triangle OBJ's (points (V, 3), triangles (T, 3)), 0-based."""
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                v.append([float(t) for t in line.split()[1:4]])
            elif line.startswith('f '):
                f.append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    return np.array(v), np.array(f)


def write_stl(path, points, tris):
    """A binary STL of `tris` (T, 3) into `points` (V, 3), normals from the winding."""
    v = np.asarray(points, dtype=np.float64)[np.asarray(tris)]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    rec = np.zeros(len(v), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = n, v
    with open(path, 'wb') as f:
        f.write(b'D435i (Intel librealsense D435.obj) in the realsense_flangemount CAD frame, m'.ljust(80, b' '))
        f.write(np.uint32(len(v)).tobytes())
        f.write(rec.tobytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('intel_obj', help='librealsense common/res/sources/D435.obj (millimetres)')
    ap.add_argument('out', nargs='?', default=G.CAMERA_STL)
    a = ap.parse_args(argv)
    points, tris = read_obj(os.path.expanduser(a.intel_obj))
    screw = -np.asarray(G.D435I_FROM_SCREW_M)                      # the screw in the optical frame
    lo, hi = points.min(0), points.max(0)
    centre = (lo + hi) / 2
    # model -> optical: x = screw_x - (x - centre_x), y = -(y - centre_y), z = z + (front - colour)
    T_opt_model = np.diag([-1.0, -1.0, 1.0, 1.0])
    T_opt_model[:3, 3] = [screw[0] + centre[0] * 1e-3, centre[1] * 1e-3, G.D435I_FRONT_FROM_SCREW_M - G.D435I_FROM_SCREW_M[2]]
    T_cad_model = G.T_CAD_CAMERA @ T_opt_model
    p = (points * 1e-3) @ T_cad_model[:3, :3].T + T_cad_model[:3, 3]
    write_stl(a.out, p, tris)
    print(f'{a.out}: {len(tris)} triangles, CAD bbox {np.round(p.min(0) * 1e3, 2)} .. {np.round(p.max(0) * 1e3, 2)} mm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
