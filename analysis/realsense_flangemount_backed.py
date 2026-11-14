"""Two new versions of the lab's RealSense flange mount (realsense_flangemount.stl, the D415 mount used until
2026-11-12), each with a back plate the camera's rear M3s bolt through (2026-11-13):

    realsense_flangemount_d415_backed.stl   the D415 as before (same pose), plus the back plate
    realsense_flangemount_d435i.stl         a D435i on the same 1/4-20 hole, plus the back plate

THE ORIGINAL: a 7.94 mm plate on the iiwa flange (flange face CAD z 72.4, a 2.5 mm pilot lip down to 69.9), an arm
out to -y ending in a tab tilted 15 deg. The camera's bottom sits on the tab (1/4-20 through it, along the camera's
up), looking along F = (0, sin15, cos15): forward, toward the gripper and 15 deg in toward the axis. Here everything
is placed in the camera frame (X width = CAD x, U up = (0, -cos15, sin15), F forward), mm:

    D415 (lab d415.stl, in place):  X -14.7..84.7, U 78.7..103.0, F 51.1..71.7 (back at F 51.1)
    tab:                            X -14.4..84.0, U 69.25..78.5, F 52.3..70.4; the 1/4-20 at X 35, F 61.4
                                    (the D415's screw is 9.87 mm behind its glass: Intel _d415.urdf.xacro)

THE BACK PLATE: 5 mm, 0.2 mm behind the camera's back, from the tab's underside to the camera's top, the tab's
width, fused to the tab's back edge (the gap between them filled). The rear M3s are 45 mm apart, centred, at
mid-height (Intel's d435i mesh: holes at +/-22.5 mm, centre height). The D415's I could not confirm from any
mesh here, so its holes are horizontal SLOTS, 45 +/- 6 mm centre to centre. Measure / check before printing.

THE D435i (90 x 25 x 25.05): its 1/4-20 is 14.9 mm behind its front plate (Intel _d435.urdf.xacro:
d435_cam_mount_from_center_offset), i.e. 10.15 mm in front of its back -- as the D415's (10.2). Kept on the same
tab hole, its back lands where the D415's did (F 51.25) and its top 0.7 mm higher.

CLEARANCE: the iiwa's link 7 is r 51.9 mm right behind the flange face; nothing new is added behind the face
(z < 72.4) within r 54. The camera's back already sits 2.6 mm behind the face (as in the original).

    cd sim && uv run --no-project --with trimesh --with manifold3d --with rtree python ../analysis/realsense_flangemount_backed.py
"""
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', 'cable_assembly', 'robot', 'description', 'tool')

C15, S15 = np.cos(np.radians(15)), np.sin(np.radians(15))
B = np.c_[[1, 0, 0], [0, -C15, S15], [0, S15, C15]]        # columns: X, U, F in CAD
T_CAM = np.eye(4); T_CAM[:3, :3] = B                         # camera-frame (X, U, F) -> CAD, mm

TAB_U0, TAB_U1, TAB_F0 = 69.25, 78.5, 52.3
TAB_X = (-14.4, 84.0)
SEAT_U = 78.71                                               # camera bottom (on the tab)
SCREW_X, SCREW_F = 35.0, 61.4
PLATE, GAP, M3 = 5.0, 0.2, 3.4
FLANGE_Z, KEEP_R = 72.4, 54.0

CAMS = {
    #           width,  height, depth, screw->back, holes (half-spacing, slot half-length)
    'd415':  dict(w=99.4, h=24.3, d=20.55, back_from_screw=SCREW_F - 51.11, half=22.5, slot=3.0),
    'd435i': dict(w=90.0, h=25.0, d=25.05, back_from_screw=25.05 - 14.9, half=22.5, slot=0.0),
}


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m.apply_transform(T_CAM)


def hole_along_f(x, u, f0, f1, r, slot):
    """A through hole (or slot along X) along F, in camera coordinates."""
    c = trimesh.creation.cylinder(radius=r, height=f1 - f0, sections=48)
    c.apply_translation([0, 0, (f0 + f1) / 2])
    if slot > 0:
        parts = [c.copy().apply_translation([x + s, u, 0]) for s in (-slot, slot)]
        parts.append(trimesh.creation.box(extents=[2 * slot, 2 * r, f1 - f0]).apply_translation([x, u, (f0 + f1) / 2]))
        h = trimesh.boolean.union(parts, engine='manifold')
    else:
        h = c.apply_translation([x, u, 0])
    return h.apply_transform(T_CAM)


def build(name, mount):
    c = CAMS[name]
    back = SCREW_F - c['back_from_screw']                    # camera back, F
    top = SEAT_U + c['h']
    f1 = back - GAP; f0 = f1 - PLATE
    x0, x1 = TAB_X
    add = [box([x0, TAB_U0, f0], [x1, top, f1]),             # the back plate
           box([x0, TAB_U0, f0 + 0.5], [x1, TAB_U1, TAB_F0 + 0.5])]   # tab extended back to meet it
    # keep-out: nothing NEW behind the flange face within r KEEP_R of the axis (link 7); the original is untouched
    keep = trimesh.creation.cylinder(radius=KEEP_R, height=100, sections=128).apply_translation([0, 0, FLANGE_Z - 50])
    add = trimesh.boolean.difference([trimesh.boolean.union(add, engine='manifold'), keep], engine='manifold')
    body = trimesh.boolean.union([mount, add], engine='manifold')
    cuts = []
    for s in (-1, 1):
        cuts.append(hole_along_f(SCREW_X + s * c['half'], SEAT_U + c['h'] / 2, f0 - 1, f1 + 1, M3 / 2, c['slot']))
    body = trimesh.boolean.difference([body] + cuts, engine='manifold')
    # the camera's box must not hit the mount (0.1 mm slack for the original's seat)
    cam = box([SCREW_X - c['w'] / 2, SEAT_U + 0.1, back], [SCREW_X + c['w'] / 2, top, back + c['d']])
    clash = trimesh.boolean.intersection([body, cam], engine='manifold')
    return body, cam, clash.volume if not clash.is_empty else 0.0


def main():
    mount = trimesh.load(os.path.join(TOOL, 'realsense_flangemount.stl')).apply_scale(1000.0)
    for name, out in (('d415', 'realsense_flangemount_d415_backed'), ('d435i', 'realsense_flangemount_d435i')):
        body, cam, clash = build(name, mount)
        c = CAMS[name]
        back = SCREW_F - c['back_from_screw']
        print(f'{out}: watertight {body.is_watertight}, {body.volume / 1000:.1f} cm3 (original {mount.volume / 1000:.1f}),'
              f' camera-box overlap {clash:.2f} mm3, camera back at F {back:.2f}')
        pts = (T_CAM[:3, :3] @ np.array([[SCREW_X + s * c['half'], SEAT_U + c['h'] / 2, back] for s in (-1, 1)]).T).T
        print('  rear M3 centres on the camera back, CAD mm:', np.round(pts, 2).tolist())
        body.copy().apply_scale(1e-3).export(os.path.join(TOOL, out + '.stl'))
        if len(sys.argv) > 1:
            cam.copy().apply_scale(1e-3).export(os.path.join(sys.argv[1], name + '_box.stl'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
