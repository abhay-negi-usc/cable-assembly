"""A CLOSER D435i view of the USB plug held by its cable (2026-11-14, the user: "a closer view of the connector
without being too obstructive on the gripper") -- on top of analysis/realsense_flangemount_d435i_usbview.py (its
scene, renderer and scores; mount A there is the 'farther, RGB-D at 848x480' answer).

THE IDEA. Held by the cable, the plug sits out on the tool's -x side, past the fingertips (tool z 184-232 mm), its
wide face's normal pointing back at -x/-z. Everything that moves in the 2F-85 -- knuckles, fingers, the pads -- stays
within tool x +/-19.5 mm, open (85 mm) or shut; its base and coupling are a +/-37.5 mm block up to z 108. So the
space at tool x < -45 beside the gripper, level with its fingers, is free: a camera on a longer arm there, looking
up the tool and in at the plug from 100-140 mm, sees its wide face nearly square on -- the face the cavity side is
told from -- with nothing between. The arm is two 6 mm webs from the flange plate down the gripper's side.

WHAT IS CHECKED beyond the first script's scores: the camera and mount against the gripper's sweep (the convex hull
of each moving link open and closed on the cable, +3 mm), its base, the held plug, cable and the lab's longer relief
(+3 mm), the USB socket's model as mounted and the board behind it at the mate, the stand-off and the way-up probe
(+/-3 mm across the axis), the bench at the stand-off, link 7's keep-out, the wrist (A6). Depth: the share of the
visible plug past the min-Z at 424x240.

    cd sim && ./run.sh --with trimesh --with manifold3d --with rtree --with opencv-python-headless --with pyyaml \\
        --with scipy python ../analysis/realsense_flangemount_d435i_usb_close.py <out_dir>
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realsense_flangemount_d435i_usbview as V                       # noqa: E402

TOOL, REPO = V.TOOL, V.REPO
GRASPS = V.GRASPS
MARGIN = 3.0


# ======================================================================== the gripper open, and its sweep
def open_gripper(cache):
    """{link: triangles tool0 mm}: Robotiq's OPEN V3 STEP + the bench pads (sim/scripts/build_iiwa14.py's frame)."""
    if not os.path.exists(cache):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call([sys.executable, os.path.join(here, '_mesh_v3_open.py'), cache],
                              cwd=os.path.join(REPO, 'sim'))
    z = np.load(cache)
    return {k: z[k] for k in z.files}


def sweep_hulls(open_links, closed_grip_tris):
    """[Delaunay]: per moving link (knuckles, fingers, pads, each side), the hull of it open and shut on the cable,
    grown by MARGIN -- the region the closing sweeps (the linkage's arcs bow < 2 mm off these chords)."""
    closed = closed_grip_tris.reshape(-1, 3)
    hulls = []
    for k, T in open_links.items():
        if k in ('1', '6', '7'):
            continue
        P = T.reshape(-1, 3)
        lo, hi = P.min(0) - 2, P.max(0) + 2
        side = np.sign(P[:, 1].mean())
        # the same link shut: closed-model points in the link's x/z band on its side, near the axis
        C = closed[(np.sign(closed[:, 1]) == side) & (closed[:, 0] >= lo[0]) & (closed[:, 0] <= hi[0])
                   & (closed[:, 2] >= lo[2] - 20) & (closed[:, 2] <= hi[2] + 20)]
        Q = np.r_[P, C]
        c = Q.mean(0)
        Q = Q + MARGIN * (Q - c) / np.maximum(np.linalg.norm(Q - c, axis=1, keepdims=True), 1e-9)
        hulls.append(Delaunay(trimesh.convex.convex_hull(Q).vertices))
    return hulls


def inside_any(hulls, P):
    return any(bool((h.find_simplex(P) >= 0).any()) for h in hulls)


# ======================================================================== camera poses in tool0
def cam_tool0(optics, phi_deg, roll_deg=0.0):
    """T_tool0_cam: colour optics at `optics` (tool0 mm), looking along (sin phi, 0, cos phi) -- phi toward +x, the
    flange axis side, as the lab mount (15) -- image x along tool -y as the lab mount, turned `roll_deg` about the
    line of sight."""
    p = np.radians(phi_deg)
    T = np.eye(4)
    T[:3, 0] = [0.0, -1.0, 0.0]
    T[:3, 1] = [np.cos(p), 0.0, -np.sin(p)]
    T[:3, 2] = [np.sin(p), 0.0, np.cos(p)]
    T[:3, 3] = optics
    return T @ V.rot('z', roll_deg)


def aim(optics, target):
    """phi so the line of sight from `optics` passes through `target` (both in the tool's xz plane)."""
    d = np.subtract(target, optics)
    return float(np.degrees(np.arctan2(d[0], d[2])))


# ======================================================================== the mount: plate + two webs + bracket
def build_close_mount(T_cad_cam, original, plate=6.0, tab=8.0, gap=0.2, web=6.0):
    """The lab mount's disc, a plate-thick shelf from its rim out past link 7's r 54, and from the shelf two 6 mm webs
    (at the bracket's two ends) down to the camera's tab + back plate (1/4-20 sunk, rear M3s) -- not a solid wedge,
    which on a 100 mm drop weighs 200+ g. Minus the coupling, the gripper's base block and the camera's box."""
    lo, hi = V.BODY_LO, V.BODY_HI
    x0, x1 = V.SCREW_O[0] - 36.0, V.SCREW_O[0] + 36.0
    back = lo[2] - gap
    tabb = trimesh.creation.box(bounds=[[x0, hi[1] + gap, back - plate], [x1, hi[1] + gap + tab, V.SCREW_O[2] + 12.0]])
    backp = trimesh.creation.box(bounds=[[x0, lo[1] - 2.0, back - plate], [x1, hi[1] + gap + tab, back]])
    bracket = trimesh.boolean.union([tabb, backp], engine='manifold').apply_transform(T_cad_cam)
    B = np.asarray(bracket.vertices)
    c = V.apply(T_cad_cam, [V.SCREW_O])[0]
    ang = np.arctan2(c[1], c[0])
    u = np.array([np.cos(ang), np.sin(ang), 0.0])            # out toward the camera
    t = np.array([-np.sin(ang), np.cos(ang), 0.0])           # along the rim
    r_out = max(V.KEEP_R + 8.0, float(np.hypot(c[0], c[1])) - 10.0)
    # the shelf: across the bracket's width, from inside the disc out to r_out, plate-thick (CAD z 72.4..80.4)
    span = [float((B @ t).min()), float((B @ t).max())]
    S = np.array([r * u + s * t + [0, 0, z] for r in (28.0, r_out) for s in span
                  for z in (V.FLANGE_Z + 0.4, V.FLANGE_Z + 8.0)])
    shelf = trimesh.convex.convex_hull(S)
    webs = []
    for e in (span[0], span[1] - web):
        band = B[((B @ t) >= e - 0.01) & ((B @ t) <= e + web + 0.01)]
        Sb = S[((S @ t) >= e - 0.01) & ((S @ t) <= e + web + 0.01)]
        Sb = np.r_[Sb, [q + web * t if abs((q @ t) - e) < 0.02 else q - web * t for q in Sb]]
        webs.append(trimesh.convex.convex_hull(np.r_[band, Sb, [r_out * u + (e + k) * t + [0, 0, V.FLANGE_Z + 0.4]
                                                                   for k in (0, web)]]))
    disc = trimesh.boolean.intersection([original, trimesh.creation.cylinder(radius=37.6, height=60, sections=128)
                                         .apply_translation([0, 0, V.FLANGE_Z])], engine='manifold')
    keep = [trimesh.creation.cylinder(radius=V.KEEP_R, height=200, sections=128).apply_translation([0, 0, V.FLANGE_Z - 100]),
            trimesh.creation.cylinder(radius=V.COUPLING_R, height=40, sections=128).apply_translation([0, 0, V.FLANGE_Z + 28.0]),
            V.gripper_keepout(),
            trimesh.creation.box(bounds=[lo - [0.2, 0.2, gap], hi + [0.2, 0.2, 80.0]]).apply_transform(T_cad_cam)]
    add = trimesh.boolean.union([shelf, bracket] + webs, engine='manifold')
    add = trimesh.boolean.difference([add] + keep, engine='manifold')
    body = trimesh.boolean.union([disc, add], engine='manifold')
    holes = [V.hole(T_cad_cam, [V.SCREW_O[0], hi[1], V.SCREW_O[2]], [0, 1, 0], 3.3, 0.0, 40.0),
             V.hole(T_cad_cam, [V.SCREW_O[0], hi[1] + gap + tab - 3.5, V.SCREW_O[2]], [0, 1, 0], 5.75, 0.0, 10.0)]
    holes += [V.hole(T_cad_cam, [V.SCREW_O[0] + s * 22.5, (lo[1] + hi[1]) / 2.0, back], [0, 0, -1], 1.7, -5, plate + 30)
              for s in (-1, 1)]
    return trimesh.boolean.difference([body] + holes, engine='manifold')


# ======================================================================== clearances
def held_items_points(scene, d, a):
    items, T_p, T_j = scene.items(d, a, 70.0)
    held = np.concatenate([t.reshape(-1, 3) for t, l in items if l in (1, 2, 3)])
    return held, T_p


def socket_clearance(P_tool0, scene, d, a):
    """Least distance (mm) from the tool points to the socket as mounted (its model, the box included) and to the
    board behind it (the model's back face, extended), at the mate, the stand-off and +/-3 mm either way across the
    axis at the mate (the way-up probe), via the IK of usbview."""
    standoff, mate = V.insertion_poses(d, a)
    T_s = mate @ V.plug_pose(d, a)[0] @ V.inverse(V.Tmm(V.P6.T_socket_cord_seated(V.P6.USB_SOCKET)))
    S = V.apply(T_s, scene.socket_model.reshape(-1, 3))
    board_x = float(S[:, 0].max())
    from scipy.spatial import cKDTree
    tree = cKDTree(S)
    out = {}
    probes = [('mate', mate), ('standoff', standoff)] + [
        (f'probe {ax}{s:+d}', V.trans(*(s * 3.0 * np.eye(3)[k])) @ mate) for k, ax in ((1, 'y'), (2, 'z')) for s in (-1, 1)]
    for name, T in probes:
        P = V.apply(T, P_tool0)
        out[name] = {'socket_mm': float(tree.query(P)[0].min()), 'board_mm': float(board_x - P[:, 0].max())}
    return out


def evaluate(name, T_t0c, scene, mount_tool0, hulls, spheres, out_dir, open_links):
    body = trimesh.creation.box(extents=V.BODY_HI - V.BODY_LO).apply_translation(
        (V.BODY_LO + V.BODY_HI) / 2).apply_transform(T_t0c)
    Pc = body.sample(3000)
    Pm = mount_tool0.sample(8000) if mount_tool0 is not None else np.zeros((0, 3))
    rr = np.hypot(Pm[:, 0], Pm[:, 1]) if len(Pm) else np.zeros(0)
    Pnew = np.r_[Pc, Pm[rr > 37.7]]                         # the camera + what is new of the mount
    r = {'hand_eye_xyz_mm': T_t0c[:3, 3].round(2).tolist()}
    from cable_assembly.transforms import matrix_to_xyzrpy
    r['hand_eye_rpy_deg'] = np.degrees(matrix_to_xyzrpy(T_t0c)[1]).round(2).tolist()
    r['in_finger_sweep'] = inside_any(hulls, Pnew)
    base_box = (np.abs(Pnew[:, 0]) < 37.5 + MARGIN) & (np.abs(Pnew[:, 1]) < 42.4 + MARGIN) & (Pnew[:, 2] > 7.9) & (Pnew[:, 2] < 108.5 + MARGIN)
    r['in_gripper_base'] = bool(base_box.any())
    # the camera alone against the 2F-85's base block (|x| 37.5, |y| 42.4, tool z 7.9..108.5): its gap, mm
    gx = np.maximum(np.abs(Pc[:, 0]) - 37.5, 0); gy = np.maximum(np.abs(Pc[:, 1]) - 42.4, 0)
    gz = np.maximum(np.maximum(7.9 - Pc[:, 2], Pc[:, 2] - 108.5), 0)
    r['camera_to_gripper_base_mm'] = float(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).min())
    behind = Pnew[:, 2] < 0
    r['keepout_mm'] = float(max(0.0, (V.KEEP_R - np.hypot(Pnew[behind, 0], Pnew[behind, 1])).max())) if behind.any() else 0.0
    r['a6_clear_to_deg'], r['wrist_clearance_mm'], r['link7_clearance_mm'] = V.wrist_clearance(Pnew[::5], spheres)
    from scipy.spatial import cKDTree
    r['views'] = {}
    extra = [(np.asarray(mount_tool0.vertices)[mount_tool0.faces], V.LABELS['mount'])] if mount_tool0 is not None else []
    for d, a in GRASPS:
        held, _ = held_items_points(scene, d, a)
        v = V.score_view(T_t0c, scene, d, a, f'{name}_d{d}_a{a}', out_dir, extra)
        v['held_clearance_mm'] = float(cKDTree(held).query(Pnew)[0].min())
        v['arm'] = V.arm_clearance(Pnew, d, a)
        v['socket'] = socket_clearance(Pnew, scene, d, a)
        r['views'][f'd{d}_a{a}'] = v
    # the OPEN gripper in this camera's view (the plug not drawn: nothing held)
    items = [(T, V.LABELS['pad'] if k.startswith('pad') else V.LABELS['gripper']) for k, T in open_links.items()] + extra
    lab, _ = V.render(T_t0c, items)
    r['open_gripper_in_image_pct'] = float(100.0 * np.isin(lab, [4, 5]).mean())
    img = np.zeros((V.H, V.W, 3), np.uint8)
    for k, c in V.COLORS.items():
        img[lab == k] = c
    big = cv2.resize(img, (2 * V.W, 2 * V.H), interpolation=cv2.INTER_NEAREST)
    cv2.putText(big, f'{name}: gripper OPEN (85 mm), {r["open_gripper_in_image_pct"]:.0f}% of the image', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(out_dir, f'{name}_open.png'), big)
    return r


def candidates():
    """name -> T_tool0_cam. The plug's middle over the three grasps is about (-33, 0, 207) mm in tool0."""
    mid = np.array([-33.0, 0.0, 212.0])
    out = {'current': V.T_TOOL0_CAD @ V.Tmm(V.G.T_CAD_CAMERA),
           'A_back40_tilt7': V.T_TOOL0_CAD @ V.candidates()['A_back40_tilt7']}
    # CLOSE-1: beside the gripper, level with its knuckles (optics tool x -82, z 105), aimed at the plug's middle:
    # ~120 mm from it. Landscape, as the lab mount.
    o = np.array([-82.0, -2.5, 105.0])
    out['close1_x82_z105'] = cam_tool0(o, aim(o, mid))
    # CLOSE-2: CLOSE-1 drawn back 15 mm along the gripper's side (x -86, z 91): ~130 mm, so the nearest of the plug
    # stays past the 424x240 min-Z (105) for every grasp scored -- RGB-D held, a little smaller.
    o = np.array([-86.0, -2.5, 91.0])
    out['close2_x86_z91'] = cam_tool0(o, aim(o, mid))
    # (2026-11-14, rejected: optics x -78, z 125, ~100 mm -- the plug 31% out of frame for the distance-5 bite and
    # nearer than any depth mode reads.)
    # CLOSE-3: as CLOSE-1 turned 90 deg about its line of sight (portrait): the 69 deg width along the plug and the
    # socket beyond it -- the socket kept in view at the stand-off; the body hangs out along -x.
    o = np.array([-82.0, -2.5, 105.0])
    out['close3_portrait'] = cam_tool0(o, aim(o, mid + [-12.0, 0, 12.0]), roll_deg=-90.0)
    return out


def main():
    out_dir = sys.argv[1]
    scene = V.Scene(out_dir)
    open_links = open_gripper(os.path.join(out_dir, 'open_tool0.npz'))
    hulls = sweep_hulls(open_links, scene.grip)
    original = trimesh.load(os.path.join(TOOL, 'realsense_flangemount.stl')).apply_scale(1000.0)
    spheres = V.wrist_spheres()
    report = {}
    for name, T_t0c in candidates().items():
        T_cad_cam = V.T_CAD_TOOL0 @ T_t0c
        if name == 'current':
            mount = original
        elif name.startswith('A_'):
            mount = trimesh.load(os.path.join(TOOL, 'realsense_flangemount_d435i_usb_a.stl')).apply_scale(1000.0)
        else:
            mount = build_close_mount(T_cad_cam, original)
            stl = os.path.join(TOOL, f'realsense_flangemount_d435i_usb_{name.split("_")[0]}.stl')
            mount.copy().apply_scale(1e-3).export(stl)
        m_tool0 = mount.copy().apply_transform(V.T_TOOL0_CAD)
        r = evaluate(name, T_t0c, scene, m_tool0, hulls, spheres, out_dir, open_links)
        if not name.startswith(('current', 'A_')):
            r['stl'] = os.path.relpath(stl, REPO)
            r['watertight'] = bool(mount.is_watertight)
            r['pieces_mm3'] = sorted(round(float(p.volume)) for p in mount.split(only_watertight=False))
            clash = trimesh.boolean.intersection([mount, V.camera_box_mesh(T_cad_cam)], engine='manifold')
            r['mount_camera_overlap_mm3'] = 0.0 if clash.is_empty else float(clash.volume)
        mass, cog = V.payload(mount, T_cad_cam)
        r['mount_g_solid'] = round(mass, 1)
        r['camera_mount_cog_tool0_mm'] = cog.round(1).tolist()
        report[name] = r
        print(name, 'done', flush=True)
    with open(os.path.join(out_dir, 'report_close.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
