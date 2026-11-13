"""A closer D405 mount for the iiwa (v2, 2026-11-13): the camera ~110 mm from the fingertip, 55 deg off the tool
axis, under the 2F-85's housing -- d405_iiwa_mount_v2.step / .stl next to the v1 (d405_iiwa_mount.step, kept).

WHY: the v1 holds the D405 168 mm from the fingertip, 15 deg off the axis, over the housing: the gripper fills ~25%
of its image. v2 puts it beside the fingers, below the housing (whose bottom is tool z 111 mm), where only the
fingers' edges cross its view. The placement came from a ray-cast search against Robotiq's V3 CAD (open, closed on
50 mm and on 10 mm; the D405's 87 x 58 deg field, its 70 mm minimum range):

                         fingertip   gripper in image   pads seen   50 mm object (near side) seen   10 mm object
    v1 (as built)          168 mm         25%              70%               100%                         81%
    v2 (this)              110 mm         16%              71%                94%                         90%

At 100 mm the near face of a 50 mm part is closer than the D405 can range (70 mm): 110 is the closest that sees it.

THE MOUNT'S FRAME is v1's (tool_geometry.py): z along the flange's axis, z = 0 the flange face, the camera on +x
(tool -X after the half turn). The flange disc and the coupling boss are v1's, cut out of its STEP unchanged. A
6 mm plate runs from the disc's rim out and down at 55 deg -- the camera's back sits on it, its two M3s through it
-- with two 10 mm ribs on its outer side; a 6 mm tab square to it carries the camera's bottom, the 1/4-20 through a
7 mm hole, the head in a 12 mm x 3 mm counterbore on the tab's outer face. Nothing comes within 1 mm of the
coupling (r 37.5) or the housing (x +/-37.5, z 10..115). The camera is mounted image-up-away as v1 (the gripper at
the top of the image).

    cd sim && ./run.sh --extra cad python ../analysis/d405_iiwa_mount_v2.py
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(HERE, '..', 'cable_assembly', 'robot', 'description', 'tool')
V1_STEP = os.path.join(TOOL_DIR, 'd405_iiwa_mount.step')
OUT = os.path.join(TOOL_DIR, 'd405_iiwa_mount_v2')

FINGERTIP_Z = 184.1            # tool z of the pad tips, closed (fingertip_grasp)
DIST, PHI = 110.0, 55.0        # optical origin -> fingertip, and its angle off the tool axis
T_PLATE, T_TAB, RIB = 6.0, 6.0, 10.0
CB_D, CB_DEPTH, Q20_D, M3_D = 12.0, 3.0, 7.0, 3.35
# the D405 in its colour optical frame (x right, y down, z ahead), mm: Intel's 42 x 42 x 23 box, the tripod screw
# on its bottom 8.35 mm in front of its back, the back's M3s 21 mm up and +/-10 across (tool_geometry.D405_FROM_SCREW_M)
BOX = np.array([[-12.0, -21.0, -19.2], [30.0, 21.0, 3.8]])
SCREW = np.array([9.0, 21.0, -10.85])
M3 = [np.array([9.0 + s, 0.0, -19.2]) for s in (-10.0, 10.0)]


def camera_in_cad():
    """(R, o): the D405's optical frame in the mount's CAD frame."""
    p = np.radians(PHI)
    z = np.array([-np.sin(p), 0.0, np.cos(p)])            # CAD +x is tool -x: the camera looks back in at the tip
    o = np.array([DIST * np.sin(p), 0.0, FINGERTIP_Z - DIST * np.cos(p)])
    y = np.array([np.cos(p), 0.0, np.sin(p)])             # image down: away from the axis
    return np.c_[np.cross(y, z), y, z], o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args(argv)
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    occ = gmsh.model.occ
    R, o = camera_in_cad()
    A = [*R[0], o[0], *R[1], o[1], *R[2], o[2]]

    def local_box(lo, hi):
        t = occ.addBox(*lo, *(np.asarray(hi) - lo)); occ.affineTransform([(3, t)], A); return (3, t)

    def local_cyl(p, axis, length, r):
        t = occ.addCylinder(*p, *(np.asarray(axis) * length), r); occ.affineTransform([(3, t)], A); return (3, t)

    # v1's disc + boss: its solid inside r 37.5, z <= 12.5
    v1 = occ.importShapes(V1_STEP)
    keep = occ.addCylinder(0, 0, -1, 0, 0, 14.0, 37.5)
    disc, _ = occ.intersect(v1, [(3, keep)])

    xlo, xhi = BOX[0, 0] - 3.0, BOX[1, 0] + 3.0
    zb = BOX[0, 2]
    plate = local_box([xlo, -200.0, zb - T_PLATE], [xhi, BOX[1, 1] + T_TAB, zb])
    ribs = [local_box([x0, -200.0, zb - T_PLATE - RIB], [x0 + 5.0, BOX[1, 1] + T_TAB, zb - T_PLATE])
            for x0 in (xlo, xhi - 5.0)]
    tab = local_box([xlo, BOX[1, 1], zb - T_PLATE], [xhi, BOX[1, 1] + T_TAB, -1.0])
    pad = occ.addBox(30.0, -(xhi - xlo) / 2 - 2, 0.0, 40.0, xhi - xlo + 4, 10.0)
    body, _ = occ.fuse(disc, [plate, tab, (3, pad)] + ribs)
    # keep-outs: below the flange face, the coupling, the housing, the camera
    cuts = [(3, occ.addBox(-300, -300, -300, 600, 600, 300)),
            (3, occ.addCylinder(0, 0, 10.0, 0, 0, 20.0, 38.5)),
            (3, occ.addBox(-38.5, -43.5, 10.0, 77.0, 87.0, 105.0)),
            (3, occ.addCylinder(0, 0, 10.0, 0, 0, 200.0, 0.1))]
    cuts.append(local_box(BOX[0] - 0.2, BOX[1] + 0.2))
    # holes: the 1/4-20 + its counterbore (tab's outer face), the two M3s
    yo = BOX[1, 1] + T_TAB
    cuts.append(local_cyl(SCREW * [1, 0, 1] + [0, BOX[1, 1] - 1, 0], [0, 1, 0], T_TAB + 2, Q20_D / 2))
    cuts.append(local_cyl(SCREW * [1, 0, 1] + [0, yo - CB_DEPTH, 0], [0, 1, 0], CB_DEPTH + 1, CB_D / 2))
    for m in M3:
        cuts.append(local_cyl(m + [0, 0, 1], [0, 0, -1], T_PLATE + RIB + 2, M3_D / 2))
    body, _ = occ.cut(body, cuts)
    # the plate beyond the pad: trim what runs past the flange's radius region on the -x side
    occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if len(vols) > 1:   # keep the biggest piece (anything cut loose is debris)
        big = max(vols, key=lambda v: occ.getMass(*v))
        occ.remove([v for v in vols if v != big], recursive=True)
        occ.synchronize()
    gmsh.write(a.out + '.step')
    gmsh.option.setNumber('Mesh.MeshSizeMax', 2.0)
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.2)
    gmsh.option.setNumber('Mesh.StlOneSolidPerSurface', 0)
    gmsh.option.setNumber('Mesh.ScalingFactor', 1e-3)
    gmsh.model.mesh.generate(2)
    gmsh.write(a.out + '.stl')
    v = gmsh.model.getEntities(3)
    bb = occ.getBoundingBox(*v[0])
    print(f'{len(v)} solid(s), {occ.getMass(*v[0]) / 1000:.1f} cm3, bbox {np.round(bb[:3], 1)} .. {np.round(bb[3:], 1)} mm')
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, o
    Tt = np.diag([-1.0, -1.0, 1.0, 1.0]) @ T
    rpy = np.degrees([np.arctan2(Tt[2, 1], Tt[2, 2]), -np.arcsin(Tt[2, 0]), np.arctan2(Tt[1, 0], Tt[0, 0])])
    print('CAD hand-eye (tool0 -> D405 colour optical): xyz', np.round(Tt[:3, 3], 2), 'mm, rpy', np.round(rpy, 2), 'deg')
    print('1/4-20 screw on the camera, CAD:', np.round(R @ SCREW + o, 2))
    gmsh.finalize()
    return 0


if __name__ == '__main__':
    sys.exit(main())
