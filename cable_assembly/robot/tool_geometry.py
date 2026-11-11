"""Tool geometry -- what is bolted to the iiwa's flange, for collision checking: the lab's camera
mount and the D415 (robot/description/tool/*.stl), and the Robotiq 2F-85 on its coupling.

THE STLs are in the CAD assembly frame the camera pose was given in (2026-10-29: the D415 at
[0, -68.79, 12.65] mm, rpy [-15, 0, 0] from the flange face), in metres. Its z is tool0's, the
flange face is at z = 72.400 mm in it (the floor of the mount's recess, which the flange's rim sits
in), and it is turned -90 deg about z from tool0's axes (the camera on tool -X). T_TOOL0_CAD takes
CAD points to tool0. sim/scripts/build_iiwa14.py places the same meshes in the sim with the same
two numbers (tests/test_tool_meshes.py holds them equal, and the D415 to the measured hand-eye).

THE GRIPPER has no mesh here: it is an envelope of spheres along tool0's z, from the coupling
(7.94 mm plate + ~20 mm coupling) to the fingertips (`fingertip_grasp`, 190 mm on the iiwa bench) --
a column of spheres round the housing and two rows at the fingers, open (85 mm), which is wider
than closed. Conservative by a few millimetres everywhere, on purpose.
"""

import os

import numpy as np

TOOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'description', 'tool')
MOUNT_STL = os.path.join(TOOL_DIR, 'realsense_flangemount.stl')
D415_STL = os.path.join(TOOL_DIR, 'd415.stl')
CAD_FLANGE_FACE_Z_M = 0.0724
CAD_TO_TOOL0_YAW_DEG = -90.0
PLATE_M = 0.00794                      # flange face -> the coupling's seat on the mount


def _t_tool0_cad():
    c, s = np.cos(np.radians(CAD_TO_TOOL0_YAW_DEG)), np.sin(np.radians(CAD_TO_TOOL0_YAW_DEG))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = T[:3, :3] @ [0.0, 0.0, -CAD_FLANGE_FACE_Z_M]
    return T


T_TOOL0_CAD = _t_tool0_cad()


def read_stl(path):
    """A binary STL as (points (V, 3) float64, triangles (T, 3) int), duplicate vertices merged."""
    with open(path, 'rb') as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), np.uint32)[0])
        tri = np.frombuffer(f.read(50 * n), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))['v']
    points, index = np.unique(np.round(tri.reshape(-1, 3).astype(np.float64), 6), axis=0, return_inverse=True)
    return points, index.reshape(-1, 3)


def read_ply(path):
    """A binary little-endian PLY of float x, y, z vertices and triangle faces -- the 6D estimator's
    CAD models (trimesh writes them) -- as (points (V, 3) float64, triangles (T, 3) int), as stored."""
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'{path}: not a PLY')
        n_v = n_f = 0
        vprops, element = [], None
        while True:
            line = f.readline().decode('ascii').split()
            if not line:
                continue
            if line[0] == 'format' and line[1] != 'binary_little_endian':
                raise ValueError(f'{path}: {line[1]} PLY (only binary_little_endian is read)')
            if line[0] == 'element':
                element = line[1]
                n_v, n_f = (int(line[2]), n_f) if element == 'vertex' else (n_v, int(line[2]) if element == 'face' else n_f)
            elif line[0] == 'property' and element == 'vertex':
                vprops.append(line[-1])
            elif line[0] == 'end_header':
                break
        if vprops[:3] != ['x', 'y', 'z'] or len(vprops) != 3:
            raise ValueError(f'{path}: vertices {vprops} (only float x, y, z are read)')
        points = np.frombuffer(f.read(12 * n_v), '<f4').reshape(n_v, 3).astype(np.float64)
        faces = np.frombuffer(f.read(13 * n_f), dtype=np.dtype([('n', 'u1'), ('i', '<i4', 3)]))
    if np.any(faces['n'] != 3):
        raise ValueError(f'{path}: faces that are not triangles')
    return points, faces['i'].astype(int)


def part_spheres(points, grid_m=0.008):
    """[(centre (3,), radius)] round a part held in the gripper, from its CAD's vertices, in the frame
    they are given in: surface_spheres on an 8 mm grid, every vertex inside a sphere (~5 mm over the
    BNC plug's 7.4 mm radius). Not cuMotion's generator: it fills a CLOSED mesh, and the 6D models'
    are not -- it left a third of the BNC plug's outside."""
    return surface_spheres(np.asarray(points, dtype=float), grid_m)


def mesh_in_tool0(path):
    """An STL's (points, triangles), the points in tool0's frame."""
    points, tris = read_stl(path)
    return points @ T_TOOL0_CAD[:3, :3].T + T_TOOL0_CAD[:3, 3], tris


def surface_spheres(points, grid_m=0.015):
    """[(centre, radius)]: one sphere per occupied `grid_m` cube of `points`, at the mean of its
    points, reaching the farthest of them -- for a thin part (the mount's 8 mm plate), which
    cuMotion's generator, filling the inside, covers with a few millimetre spheres (15-43%)."""
    keys, inv = np.unique(np.floor(points / grid_m).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    out = []
    for k in range(len(keys)):
        p = points[inv == k]
        c = p.mean(axis=0)
        out.append((c, float(np.linalg.norm(p - c, axis=1).max()) + 1e-4))
    return out


def camera_spheres(max_overshoot_m=0.004, max_spheres=24):
    """[(centre in tool0 (3,), radius)] round the mount (surface_spheres) and the D415 (cuMotion's
    sphere generator: inside the mesh, overshooting by at most `max_overshoot_m`)."""
    import cumotion
    out = surface_spheres(mesh_in_tool0(MOUNT_STL)[0])
    points, tris = mesh_in_tool0(D415_STL)
    for s in cumotion.generate_collision_spheres(list(points), list(tris.astype(np.int32)),
                                                 max_overshoot=max_overshoot_m, max_num_spheres=max_spheres):
        out.append((np.asarray(s.center, dtype=float).ravel(), float(s.radius)))
    return out


def gripper_spheres(fingertip_z_m=0.190, open_half_m=0.0425):
    """[(centre in tool0, radius)] enveloping the 2F-85 and its coupling, fingers open."""
    base = PLATE_M                                  # the coupling starts on the mount's plate
    tip = float(fingertip_z_m)
    housing = [(np.array([0.0, 0.0, z]), 0.045) for z in np.linspace(base + 0.035, tip - 0.065, 4)]
    fingers = [(np.array([0.0, side * (open_half_m + 0.008), z]), 0.013)
               for side in (-1.0, 1.0) for z in np.linspace(tip - 0.055, tip - 0.006, 4)]
    return housing + fingers


# THE BENCH GRIPPER AS THE CAMERA SEES IT (Robotiq 2F-85 V3 on the 19.1 mm coupler; its CAD meshed in
# sim/scripts/build_iiwa14.py --stage bench), fully open -- its widest: tool0 (y, z) mm of the links
# either side of the axis. The OUTER FINGER links reach +/-74 mm off the axis, past gripper_spheres'
# +/-50: the wrist camera sees the right one in the bottom-right corner of every frame, and the D415,
# 0.17 m from it -- nearer than it can measure -- reads it 0.34-0.50 m away. Those readings stood in
# the socket views' scene where tool0 goes after the lift (2026-11-03: the stand-off "starts in
# collision", -17 mm). For leaving the gripper out of camera readings only, not for planning.
GRIPPER_VIEW_LINKS_MM = (
    ((30.6, 74.0), (74.0, 125.0)),      # the outer knuckle and finger: its pivot out to the far end, open
    ((12.7, 80.0), (56.0, 130.0)),      # the inner knuckle
    ((44.0, 105.0), (56.0, 170.0)),     # the fingertip and its pad, open
)
GRIPPER_VIEW_RADIUS_M = 0.016


def gripper_view_spheres(fingertip_z_m=0.190):
    """[(centre in tool0, radius)]: gripper_spheres (fingers open) plus the V3's knuckles and outer
    finger links, either side -- every reading whose ray passes one is the gripper (scene.py)."""
    out = list(gripper_spheres(fingertip_z_m))
    for (y0, z0), (y1, z1) in GRIPPER_VIEW_LINKS_MM:
        for t in np.linspace(0.0, 1.0, 6):
            y, z = (y0 + t * (y1 - y0)) / 1000.0, (z0 + t * (z1 - z0)) / 1000.0
            out += [(np.array([0.0, side * y, z]), GRIPPER_VIEW_RADIUS_M) for side in (-1.0, 1.0)]
    return out


def rod_spheres(T_tool0_obj, z0_m, z1_m, radius_m, step_m=None):
    """[(centre in tool0, radius)] along the z axis of a frame held on the tool (T_tool0_obj), from
    z0 to z1 -- a held connector, or the cable leaving it. Spheres half a radius apart."""
    T = np.asarray(T_tool0_obj, dtype=float)
    step = float(step_m or 0.5 * radius_m)
    n = max(2, int(np.ceil(abs(z1_m - z0_m) / step)) + 1)
    return [(T[:3, :3] @ np.array([0.0, 0.0, z]) + T[:3, 3], float(radius_m)) for z in np.linspace(z0_m, z1_m, n)]


def tool_spheres(cfg=None, grip_half_m=None):
    """The whole tool's spheres in tool0: the camera mount and D415 (from the STLs) and the
    gripper (from `fingertip_grasp`'s z when a config is given) -- its fingers open, or closed on
    something `grip_half_m` across half (a held part: the fingers are that far from the middle)."""
    tip = 0.190
    if cfg is not None and cfg.get('fingertip_grasp'):
        tip = float(cfg.section('fingertip_grasp').get('xyz_mm', [0.0, 0.0, 190.0])[2]) / 1000.0
    if grip_half_m is None:
        return camera_spheres() + gripper_spheres(tip)
    return camera_spheres() + gripper_spheres(tip, open_half_m=float(grip_half_m))
