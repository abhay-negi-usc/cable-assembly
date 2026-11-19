"""A D435i flange mount for the USB-A plug HELD BY ITS CABLE (2026-11-14), against the lab mount it is on now
(realsense_flangemount.stl + the D435i on its 1/4-20, robot/tool_geometry.py T_TOOL0_CAMERA_CAD).

THE HELD PLUG (configs/usb_assembly.yaml pickup.cable_grasp, compiled by config.cable_grasp_pose): the pads' tips
`distance_mm` behind the CAD junction (where the cable leaves the model's overmold + 15 mm of relief), the gripper
`angle_deg` off the cable's axis, `z_mm` over the plug's axis. 45 deg is the configured angle. The lab cord's ribbed
relief runs ~21 mm past the CAD's and the constant-diameter cable starts ~25 mm back: a bite "right behind the
junction" on the real cord is distance ~25; the configured (and the 2026-11-13 18:04 run's) is 5. Both are scored.
Held so, the plug points down the tool (+z) and out toward tool -x -- toward the camera's side -- 45 deg off the
flange axis; its wide faces' normal in the tool's xz plane.

WHAT IS SCORED (a z-buffered render of the CAD plug, its cable + the lab's longer relief, the 2F-85 V3 CLOSED on the
5.5 mm cable with the bench's pads, and the USB adapter at the stand-off -- 70 mm short of home -- and at the mouth):
range to the plug's middle / its nearest visible point (the depth sensor's min-Z: 280 mm at 1280x720, 195 at
848x480, 105 at 424x240 -- Intel's D435 datasheet), its size in pixels, the share of it the gripper/cable hide,
the angle its wide face is seen at (0 = square on), the socket's share in frame and seen, the plug's offset from
the image centre. And for the mount: nothing new within r 54 mm behind the flange face (link 7), nothing in the
coupling's r 37.5 / the gripper's body, clearance from the bench and the socket panel at the stand-off and the mate
(the USB socket's measured pose, IK from the 2026-11-13 stand-off solution), and from links 5-7 (the XRDF's
collision spheres) over A6 +/-120, A7 +/-175.

    cd sim && ./run.sh --with trimesh --with manifold3d --with rtree --with opencv-python-headless \\
        python ../analysis/realsense_flangemount_d435i_usbview.py [out_dir]

out_dir (default: a temp dir) gets the wrist renders; the candidate mounts go to robot/description/tool/.
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'sim', 'scripts'))

from cable_assembly.config import cable_grasp_pose                     # noqa: E402
from cable_assembly.perception import pose6d as P6                      # noqa: E402
from cable_assembly.robot import tool_geometry as G                     # noqa: E402
from cable_assembly.robot.kinematics import iiwa14 as K                 # noqa: E402
from cable_assembly.transforms import inverse, xyzrpy_to_matrix         # noqa: E402

TOOL = G.TOOL_DIR
MODELS = os.path.expanduser('~/abhay_ws/RAG-6DPose-code-abhay/data/bnc/models')
FLANGE_Z = G.CAD_FLANGE_FACE_Z_M * 1000.0          # 72.4: the flange face in the mount's CAD frame
COUPLER_MM = 7.94 + 11.0 + 0.14                    # tool0 -> the 2F-85 V3 base (sim/scripts/build_iiwa14.py)
FINGERTIP_Z = 182.0                                # configs/cells/iiwa14_bench.yaml fingertip_grasp (the pad tips)
CABLE_R = 2.75                                     # the cord, 5.5 mm across
PAD = (37.8, 22.4, 11.6)                           # along the finger, wide, thick (closing)
KEEP_R, COUPLING_R = 54.0, 38.5                    # link 7 behind the face; the C-062 coupling + 1 mm
BENCH_Z, PANEL_X = 6.2, 795.0                      # base mm: the bench; the socket panel (the adapter's face ~810)
W, H, FX = 640, 360, 913.0 / 2.0                   # rendered at half of 1280x720 (fx 913: HFOV 69.4 deg)
MIN_Z = {'1280x720': 280.0, '848x480': 195.0, '424x240': 105.0}
DENSITY = 1.24e-3                                  # g/mm^3, PLA/PETG solid -- the mass is an upper bound
CAMERA_G = 72.0                                    # Intel's figure for the D435i
LABELS = {'none': 0, 'plug': 1, 'shell': 2, 'cable': 3, 'gripper': 4, 'pad': 5, 'socket': 6, 'mount': 7}
COLORS = {0: (235, 235, 235), 1: (40, 170, 40), 2: (60, 220, 220), 3: (60, 60, 60), 4: (100, 90, 90),
          5: (170, 120, 230), 6: (40, 120, 220), 7: (200, 60, 60)}           # BGR: plug green, shell yellow, pads pink, socket orange

# The D435i's body in its colour optical frame (x right, y down, z out), mm: tool_geometry's D435I_FROM_SCREW
# (-32.5, -12.5, 10.6) puts the 1/4-20 at (32.5, 12.5, -10.6); the body 90 wide centred on it, 25 tall with the
# screw on its bottom face, 25.05 deep -- 14.9 to the front plate, 10.15 to the back.
SCREW_O = np.array([32.5, 12.5, -10.6])
BODY_LO = SCREW_O + [-45.0, -25.0, -10.15]
BODY_HI = SCREW_O + [45.0, 0.0, 14.9]


def Tmm(T):
    T = np.array(T, dtype=float)
    T[:3, 3] *= 1000.0
    return T


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = {'x': [[1, 0, 0], [0, c, -s], [0, s, c]], 'y': [[c, 0, s], [0, 1, 0], [-s, 0, c]],
         'z': [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
    T = np.eye(4)
    T[:3, :3] = R
    return T


def trans(*v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def apply(T, P):
    P = np.asarray(P, float)
    return P @ T[:3, :3].T + T[:3, 3]


T_TOOL0_CAD = Tmm(G.T_TOOL0_CAD)
T_CAD_TOOL0 = inverse(T_TOOL0_CAD)


# ======================================================================== the scene in tool0 (mm)
def plug_pose(d, a, z=-2.0):
    """T_tool0_plug (the plug's CAD frame, mm) and T_tool0_junction for the cable grasp (d, a, z)."""
    T_t0f = Tmm(xyzrpy_to_matrix([0, 0, FINGERTIP_Z / 1000.0], np.radians([180.0, 0.0, 180.0])))
    g = cable_grasp_pose(d, a, z)
    T_jf = Tmm(xyzrpy_to_matrix(np.array(g['xyz_mm']) / 1000.0, np.radians(g['rpy_deg'])))
    T_t0j = T_t0f @ inverse(T_jf)
    return T_t0j @ inverse(Tmm(P6.T_cord_junction(P6.USB))), T_t0j


def load_tris(path, scale=1.0):
    m = trimesh.load(path, force='mesh')
    return np.asarray(m.vertices, float)[m.faces] * scale


def cylinder_tris(p0, p1, r0, r1, n=24):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(ax, [0, 0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    ring = lambda p, r: p + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    A, B = ring(p0, r0), ring(p1, r1)
    out = []
    for i in range(n):
        j = (i + 1) % n
        out += [[A[i], A[j], B[j]], [A[i], B[j], B[i]]]
    return np.array(out)


def box_tris(lo, hi):
    m = trimesh.creation.box(extents=np.subtract(hi, lo))
    m.apply_translation(np.add(lo, hi) / 2.0)
    return np.asarray(m.vertices)[m.faces]


def gripper_tris(cache):
    """The 2F-85 V3 closed on the 5.5 mm cable, tool0 mm: Robotiq's CLOSED STEP (sim/scripts/build_iiwa14.py's
    meshing and its base frame), its fingers + knuckles moved out so the bench's pads (11.6 mm, on the fingertip
    mounts' faces -- |y| 5.0 closed) grip at |y| 2.75; the pads' tips at the fingertip (z 182). Cached."""
    if os.path.exists(cache):
        z = np.load(cache)
        return z['grip'], z['pads']
    import build_iiwa14 as B
    parts = B._step_triangles(os.path.join(B.V3_DIR, '2-Finger_85_Adaptive_Gripper-closed.STEP'))
    shift = CABLE_R + PAD[2] - 5.0
    out = []
    for name, T in parts.items():
        Tb = np.stack([T[..., 2], T[..., 0], T[..., 1] + B.V3_Z_OFF_MM], axis=-1)
        g = name.split('Group')[1][0] if 'Group' in name else '1'
        if g in '2345':
            Tb = Tb + [0.0, shift * np.sign(Tb[..., 1].mean()), 0.0]
        out.append(Tb)
    cp = np.concatenate(list(B._step_triangles(os.path.join(B.V3_DIR, B.V3_COUPLING_STEP), 1.0).values()))
    out.append(cp + [0.0, 0.0, B.V3_COUPLING_Z_MM])
    grip = np.concatenate(out) + [0.0, 0.0, COUPLER_MM]
    pads = np.concatenate([box_tris((-PAD[1] / 2, s * CABLE_R if s > 0 else -CABLE_R - PAD[2], FINGERTIP_Z - PAD[0]),
                                    (PAD[1] / 2, CABLE_R + PAD[2] if s > 0 else -CABLE_R, FINGERTIP_Z))
                           for s in (1, -1)])
    np.savez_compressed(cache, grip=grip, pads=pads)
    return grip, pads


class Scene:
    def __init__(self, cache_dir):
        self.plug_model = load_tris(os.path.join(MODELS, 'obj_000011.ply'))          # mm, z along the plug
        self.socket_model = load_tris(os.path.join(MODELS, 'obj_000013.ply'))        # usb_adapter_mount, mm
        self.grip, self.pads = gripper_tris(os.path.join(cache_dir, 'gripper_closed_cable.npz'))

    def items(self, d, a, socket_ahead_mm=70.0):
        """[(triangles in tool0 mm, label)] for grasp (d, a), the socket `socket_ahead_mm` short of home."""
        T_p, T_j = plug_pose(d, a)
        shell = self.plug_model[..., 2].mean(axis=1) >= 456.7
        out = [(apply(T_p, self.plug_model[~shell]), LABELS['plug']), (apply(T_p, self.plug_model[shell]), LABELS['shell'])]
        j, back = T_j[:3, 3], -T_j[:3, 0]
        relief_end = j + 21.0 * back                       # the lab cord's longer ribbed relief, ~5 -> 3 mm radius
        out.append((cylinder_tris(j, relief_end, 4.5, 3.0), LABELS['cable']))
        out.append((cylinder_tris(relief_end, relief_end + 120.0 * back, CABLE_R, CABLE_R), LABELS['cable']))
        out.append((self.grip, LABELS['gripper']))
        out.append((self.pads, LABELS['pad']))
        T_s = T_p @ trans(0, 0, socket_ahead_mm) @ inverse(Tmm(P6.T_socket_cord_seated(P6.USB_SOCKET)))
        out.append((apply(T_s, self.socket_model), LABELS['socket']))
        return out, T_p, T_j


# ======================================================================== rendering
def render(T_tool0_cam, items, only=None):
    """(labels HxW uint8, depth HxW mm) -- painter's order on triangle mean depth."""
    T_c = inverse(T_tool0_cam)
    lab = np.zeros((H, W), np.uint8)
    dep = np.full((H, W), np.inf, np.float32)
    allt, alll = [], []
    for tris, l in items:
        if only is not None and l not in only:
            continue
        allt.append(apply(T_c, tris.reshape(-1, 3)).reshape(-1, 3, 3))
        alll.append(np.full(len(tris), l, np.uint8))
    if not allt:
        return lab, dep
    T = np.concatenate(allt)
    L = np.concatenate(alll)
    keep = (T[..., 2] > 5.0).all(axis=1)
    T, L = T[keep], L[keep]
    uv = np.empty(T.shape[:2] + (2,))
    uv[..., 0] = FX * T[..., 0] / T[..., 2] + (W - 1) / 2.0
    uv[..., 1] = FX * T[..., 1] / T[..., 2] + (H - 1) / 2.0
    inside = ((uv[..., 0].max(1) >= 0) & (uv[..., 0].min(1) < W) & (uv[..., 1].max(1) >= 0) & (uv[..., 1].min(1) < H))
    T, L, uv = T[inside], L[inside], uv[inside]
    zm = T[..., 2].mean(axis=1)
    order = np.argsort(-zm)
    pts = np.round(uv * 16).astype(np.int32)
    for i in order:
        cv2.fillConvexPoly(lab, pts[i], int(L[i]), lineType=cv2.LINE_8, shift=4)
        cv2.fillConvexPoly(dep, pts[i], float(zm[i]), lineType=cv2.LINE_8, shift=4)
    return lab, dep


def project(T_tool0_cam, P):
    Q = apply(inverse(T_tool0_cam), P)
    return np.c_[FX * Q[:, 0] / Q[:, 2] + (W - 1) / 2.0, FX * Q[:, 1] / Q[:, 2] + (H - 1) / 2.0], Q[:, 2]


def score_view(T_tool0_cam, scene, d, a, tag=None, out_dir=None, extra=()):
    """`extra`: [(triangles tool0 mm, label)] drawn too -- the mount itself, in its own camera's view."""
    items, T_p, T_j = scene.items(d, a, 70.0)
    items = items + list(extra)
    lab, dep = render(T_tool0_cam, items)
    alone, _ = render(T_tool0_cam, items, only={1, 2})
    s_alone, _ = render(T_tool0_cam, items, only={6})
    items_m, _, _ = scene.items(d, a, 3.0)                       # at the mouth (engage)
    items_m = items_m + list(extra)
    lab_m, _ = render(T_tool0_cam, items_m)
    s_alone_m, _ = render(T_tool0_cam, items_m, only={6})
    plug_px = (alone > 0).sum()
    vis = (lab == 1) | (lab == 2)
    cam = T_tool0_cam[:3, 3]
    centre = apply(T_p, [P6.CENTRE_M[P6.USB] * 1000.0])[0]
    face = apply(T_p, [[0, 0, P6.MATING_Z[P6.USB] * 1000.0]])[0]
    ray = (centre - cam) / np.linalg.norm(centre - cam)
    n = T_p[:3, 1]
    pv = apply(T_p, scene.plug_model.reshape(-1, 3))
    uv, z = project(T_tool0_cam, pv)
    in_frame = ((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)).mean()
    ys, xs = np.nonzero(alone)
    res = {
        'range_centre_mm': float(np.linalg.norm(centre - cam)),
        'range_face_mm': float(np.linalg.norm(face - cam)),
        'nearest_visible_mm': float(dep[vis].min()) if vis.any() else float('nan'),
        'plug_in_frame': float(in_frame),
        'plug_px_1280': [int(2 * (xs.max() - xs.min())), int(2 * (ys.max() - ys.min()))] if len(xs) else [0, 0],
        'plug_area_px_1280': int(4 * plug_px),
        'occluded_pct': float(100.0 * (1.0 - vis.sum() / max(plug_px, 1))),
        'mount_in_image_pct': float(100.0 * (lab == LABELS['mount']).mean()),
        'shell_visible_pct': float(100.0 * (lab == 2).sum() / max((alone == 2).sum(), 1)),
        'depth_ok_pct': {k: float(100.0 * (dep[vis] >= v).mean()) if vis.any() else 0.0 for k, v in MIN_Z.items()},
        'face_angle_deg': float(np.degrees(np.arccos(min(1.0, abs(n @ ray))))),
        'centre_offset_px_1280': (2 * (project(T_tool0_cam, [centre])[0][0] - [(W - 1) / 2, (H - 1) / 2])).round(0).tolist(),
        'socket_standoff_visible_pct': float(100.0 * (lab == 6).sum() / max((s_alone == 6).sum(), 1)),
        'socket_standoff_in_frame_pct': float(100.0 * frame_share(T_tool0_cam, items, 6)),
        'socket_mouth_visible_pct': float(100.0 * (lab_m == 6).sum() / max((s_alone_m == 6).sum(), 1)),
        'socket_mouth_in_frame_pct': float(100.0 * frame_share(T_tool0_cam, items_m, 6)),
    }
    if out_dir is not None and tag is not None:
        img = np.zeros((H, W, 3), np.uint8)
        for k, c in COLORS.items():
            img[lab == k] = c
        big = cv2.resize(img, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
        txt = (f"{tag}: plug {res['range_centre_mm']:.0f} mm (nearest {res['nearest_visible_mm']:.0f}), "
               f"{res['occluded_pct']:.0f}% hidden, face {res['face_angle_deg']:.0f} deg, "
               f"socket {res['socket_standoff_in_frame_pct']:.0f}% in frame")
        cv2.putText(big, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(out_dir, f'{tag}.png'), big)
    return res


def frame_share(T_tool0_cam, items, label):
    P = np.concatenate([t.reshape(-1, 3) for t, l in items if l == label])
    uv, z = project(T_tool0_cam, P)
    return float(((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)).mean())


# ======================================================================== the mounts
def body_corners(T_tool0_cam, pad=0.0):
    lo, hi = BODY_LO - pad, BODY_HI + pad
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return apply(T_tool0_cam, c)


def camera_box_mesh(T_cad_cam):
    m = trimesh.creation.box(extents=BODY_HI - BODY_LO)
    m.apply_translation((BODY_LO + BODY_HI) / 2.0)
    return m.apply_transform(T_cad_cam)


def cam_pose_cad(optics_cad, tilt_deg, roll_deg=0.0):
    """T_cad_cam (mm): the colour optical frame at `optics_cad`, looking along F = (0, sin t, cos t) -- tilted
    `tilt_deg` from the flange axis toward it, as the lab mount (15) -- image x along CAD +x; `roll_deg` turns it
    about its line of sight (180: upside down, its bottom -- the 1/4-20 -- away from the axis)."""
    T = trans(*optics_cad) @ rot('x', -tilt_deg) @ rot('z', roll_deg)
    return T


def build_mount(T_cad_cam, original, plate=6.0, tab=8.0, gap=0.2):
    """The lab mount's flange plate (its 75 mm disc, holes and pilot lip as they are), and a new bracket: a tab
    under the camera's bottom (the 1/4-20) and a back plate (the rear M3s, 45 mm apart), joined to the disc by the
    hull of the two -- minus link 7's keep-out (r 54 behind the face), the coupling and gripper body, the camera."""
    disc = trimesh.boolean.intersection([original, trimesh.creation.cylinder(radius=37.6, height=60, sections=128)
                                         .apply_translation([0, 0, FLANGE_Z])], engine='manifold')
    lo, hi = BODY_LO, BODY_HI
    x0, x1 = SCREW_O[0] - 36.0, SCREW_O[0] + 36.0
    back = lo[2] - gap
    tabb = trimesh.creation.box(bounds=[[x0, hi[1] + gap, back - plate], [x1, hi[1] + gap + tab, SCREW_O[2] + 12.0]])
    backp = trimesh.creation.box(bounds=[[x0, lo[1] - 2.0, back - plate], [x1, hi[1] + gap + tab, back]])
    bracket = trimesh.boolean.union([tabb, backp], engine='manifold').apply_transform(T_cad_cam)
    # where the arm meets the disc: its rim facing the camera, the plate's full 8 mm (z 72.4..80.4)
    c = apply(T_cad_cam, [SCREW_O])[0]
    ang = np.arctan2(c[1], c[0])
    rim = []
    r_out = max(KEEP_R + 8.0, float(np.hypot(c[0], c[1])) - 12.0)   # the shelf over link 7's keep-out, plate-thick
    for da in np.radians([-35, -12, 12, 35]):
        for r in (31.0, 37.5, r_out):
            for z in (FLANGE_Z + 0.5, FLANGE_Z + 7.9):
                rim.append([r * np.cos(ang + da), r * np.sin(ang + da), z])
    arm = trimesh.convex.convex_hull(np.r_[np.asarray(bracket.vertices), rim])
    keep = [trimesh.creation.cylinder(radius=KEEP_R, height=200, sections=128).apply_translation([0, 0, FLANGE_Z - 100]),
            trimesh.creation.cylinder(radius=COUPLING_R, height=40, sections=128).apply_translation([0, 0, FLANGE_Z + 8.0 + 20]),
            gripper_keepout(),
            camera_box_mesh(T_cad_cam @ trans(0, 0, 0)).apply_scale(1.0)]
    cam_clear = trimesh.creation.box(bounds=[lo - [0.2, 0.2, 0.0] + [0, 0, -gap], hi + [0.2, 0.2, 60.0]]).apply_transform(T_cad_cam)
    keep[-1] = cam_clear
    add = trimesh.boolean.union([arm, bracket], engine='manifold')
    cut = trimesh.boolean.intersection([bracket, trimesh.boolean.union(keep[:3], engine='manifold')], engine='manifold')
    build_mount.bracket_cut_mm3 = 0.0 if cut.is_empty else float(cut.volume)   # the tab/back plate lost to keep-outs
    add = trimesh.boolean.difference([add] + keep, engine='manifold')
    body = trimesh.boolean.union([disc, add], engine='manifold')
    holes = [hole(T_cad_cam, [SCREW_O[0], hi[1], SCREW_O[2]], [0, 1, 0], 3.3, 0.0, 40.0)]       # 1/4-20 clearance
    # its head sunk into the tab (11.5 x 3.5 mm: a 1/4-20 x 3/8" button head): on A the tab's underside is ~2 mm
    # off link 7 behind the flange face
    holes += [hole(T_cad_cam, [SCREW_O[0], hi[1] + gap + tab - 3.5, SCREW_O[2]], [0, 1, 0], 5.75, 0.0, 10.0)]
    holes += [hole(T_cad_cam, [SCREW_O[0] + s * 22.5, (lo[1] + hi[1]) / 2.0, back], [0, 0, -1], 1.7, -5, plate + 30)
              for s in (-1, 1)]                                                                    # M3 clearance
    body = trimesh.boolean.difference([body] + holes, engine='manifold')
    return body


def gripper_keepout():
    """The 2F-85 V3 base + coupling, as a box in the mount's CAD frame: tool0 x +/-37.5 (CAD y), y +/-42.4 (CAD x),
    from the plate's face to the base's top (tool0 z 8..~110) + 1.5 mm."""
    b = trimesh.creation.box(bounds=[[-44.0, -39.0, FLANGE_Z + 8.0], [44.0, 39.0, FLANGE_Z + 112.0]])
    return b


def hole(T_cad_cam, p_o, axis_o, r, t0, t1):
    axis_o = np.asarray(axis_o, float)
    c = trimesh.creation.cylinder(radius=r, height=t1 - t0, sections=40)
    c.apply_translation([0, 0, (t0 + t1) / 2.0])
    z = np.array([0, 0, 1.0])
    R = trimesh.geometry.align_vectors(z, axis_o)
    c.apply_transform(R)
    c.apply_translation(p_o)
    return c.apply_transform(T_cad_cam)


# ======================================================================== checks of a mount on the arm
URDF_WRIST = {'A6': (np.array([0, -70.7, 192.5]), 'y'), 'A7': (np.array([0, 70.7, 91.0]), 'z')}


def wrist_spheres():
    import yaml
    with open(os.path.join(REPO, 'cable_assembly', 'robot', 'description', 'iiwa14', 'iiwa14.xrdf')) as f:
        x = yaml.safe_load(f)
    s = x['geometry']['iiwa14_collision_spheres']['spheres']
    return {k: [(np.array(e['center']) * 1000.0, e['radius'] * 1000.0) for e in s[k]] for k in ('link_5', 'link_6', 'link_7')}


def wrist_clearance(P_tool0, spheres):
    """(the |A6| up to which the points stay clear of links 5 and 6's XRDF spheres over every A7, deg; the least
    clearance within |A6| <= 90, mm; the clearance from link 7's, mm). The spheres are cuMotion's -- inflated: the
    lab mount, which has never touched the wrist, reads 38 mm to |A6| 90 and 0 at ~110."""
    by = {}
    for q6 in range(0, 121, 5):
        best = np.inf
        for sgn in (-1, 1):
            for q7 in np.arange(-175, 176, 5):
                T56 = trans(0, -70.7, 192.5) @ rot('y', sgn * q6)
                T67 = trans(0, 70.7, 91.0) @ rot('z', q7)
                for T, key in ((T56 @ T67 @ trans(0, 0, 35.0), 'link_5'), (T67 @ trans(0, 0, 35.0), 'link_6')):
                    Q = apply(T, P_tool0)
                    for c, r in spheres[key]:
                        best = min(best, float((np.linalg.norm(Q - c, axis=1) - r).min()))
        by[q6] = best
    ok = [q for q in sorted(by) if all(by[k] >= 0 for k in by if k <= q)]
    P7 = apply(trans(0, 0, 35.0), P_tool0)
    l7 = min(float((np.linalg.norm(P7 - c, axis=1) - r).min()) for c, r in spheres['link_7'])
    return (ok[-1] if ok else None), min(v for k, v in by.items() if k <= 90), l7


def insertion_poses(d, a):
    """(T_base_tool0 at the stand-off, at the mate) mm, the USB socket's measured target (2026-11-13 18:27:
    [810.54, 235.04, 272.94] mm, rpy ~0) -- the connector datum 48 mm along the plug from the junction."""
    T_tgt = Tmm(xyzrpy_to_matrix([0.81054, 0.23504, 0.27294], np.radians([0.74, -0.68, -1.5])))
    _, T_j = plug_pose(d, a)
    T_t0c = T_j @ trans(48.0, 0, 0)
    mate = T_tgt @ inverse(T_t0c)
    return T_tgt @ trans(-70.0, 0, 0) @ inverse(T_t0c), mate


def arm_clearance(P_tool0, d, a):
    seed = np.radians([0.8, 34.2, 36.4, -97.6, 96.9, -37.1, -89.0])
    out = {}
    for name, T in zip(('standoff', 'mate'), insertion_poses(d, a)):
        Tm = T.copy()
        Tm[:3, 3] /= 1000.0
        q = K.ik(Tm, seed)
        if q is None:
            out[name] = None
            continue
        P = apply(Tmm(K.fk(q)), P_tool0)
        out[name] = {'bench_mm': float(P[:, 2].min() - BENCH_Z), 'panel_mm': float(PANEL_X - P[:, 0].max()),
                     'q_deg': np.degrees(q).round(1).tolist()}
    return out


def mount_points(T_tool0_cam, mount_tool0=None):
    P = body_corners(T_tool0_cam)
    if mount_tool0 is not None:
        P = np.r_[P, mount_tool0.sample(3000)]
    return P


def keepout_violation(T_tool0_cam):
    """mm the camera body reaches inside r 54 behind the flange face (>0: a clash)."""
    P = trimesh.creation.box(extents=BODY_HI - BODY_LO).apply_translation((BODY_LO + BODY_HI) / 2).apply_transform(
        T_tool0_cam).sample(4000)
    behind = P[:, 2] < 0.0
    r = np.hypot(P[behind, 0], P[behind, 1])
    return float(max(0.0, (KEEP_R - r).max())) if behind.any() else 0.0


def payload(mount_cad, T_cad_cam):
    """(mount g, camera g, total CoG tool0 mm) -- solid print, the camera's body centre."""
    mass = mount_cad.volume * DENSITY
    c_m = apply(T_TOOL0_CAD, [mount_cad.center_mass])[0]
    c_c = apply(T_TOOL0_CAD @ T_cad_cam, [(BODY_LO + BODY_HI) / 2.0])[0]
    return mass, (mass * c_m + CAMERA_G * c_c) / (mass + CAMERA_G)


# ======================================================================== main
GRASPS = [(5, 45), (25, 45), (25, 60)]


def back_off(o, back, out, tilt):
    """T_cad_cam: the lab mount's optics `o` moved `out` mm away from the axis (CAD -y) and `back` mm back along
    the new line of sight, tilted `tilt` deg in."""
    return cam_pose_cad([o[0], o[1] - out - back * np.sin(np.radians(tilt)), o[2] - back * np.cos(np.radians(tilt))],
                        tilt)


def candidates():
    """name -> T_cad_cam (mm). CURRENT: the lab mount (tool_geometry). The rest are placed in the CAD frame."""
    cur = Tmm(G.T_CAD_CAMERA)
    out = {'current': cur}
    o = cur[:3, 3]
    # A: re-tilted 7 deg in (was 15) so the socket stays in frame at the stand-off, backed 40 mm along its line of
    # sight and 7 mm out (optics r 80) so its body and tab clear link 7's r 54 behind the face: the plug's nearest
    # point 205+ mm away -- past the 848x480 depth min-Z (195) for every grasp scored.
    out['A_back40_tilt7'] = back_off(o, 40.0, 7.0, 7.0)
    # (B, 55 mm back: rejected 2026-11-14 -- its shelf over link 7 shows in the bottom 11% of its own view and its
    # body reaches the wrist's spheres at |A6| < 90.)
    # C: the camera where it is, re-tilted only (the socket in frame); RGB-D only at 424x240 (min-Z 105).
    out['C_tilt7_only'] = cam_pose_cad(o, 7.0)
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='usbview_')
    os.makedirs(out_dir, exist_ok=True)
    scene = Scene(out_dir)
    original = trimesh.load(os.path.join(TOOL, 'realsense_flangemount.stl')).apply_scale(1000.0)
    spheres = wrist_spheres()
    report = {}
    for name, T_cad_cam in candidates().items():
        T_t0c = T_TOOL0_CAD @ T_cad_cam
        r = {'hand_eye_xyz_mm': T_t0c[:3, 3].round(2).tolist()}
        from cable_assembly.transforms import matrix_to_xyzrpy
        _x, _r = matrix_to_xyzrpy(T_t0c)
        r['hand_eye_rpy_deg'] = np.degrees(_r).round(2).tolist()
        r['keepout_clash_mm'] = keepout_violation(T_t0c)
        if name == 'current':
            mount = original
        else:
            mount = build_mount(T_cad_cam, original)
            stl = os.path.join(TOOL, f'realsense_flangemount_d435i_usb_{name.split("_")[0].lower()}.stl')
            mount.copy().apply_scale(1e-3).export(stl)
            r['stl'] = os.path.relpath(stl, REPO)
            clash = trimesh.boolean.intersection([mount, camera_box_mesh(T_cad_cam)], engine='manifold')
            r['mount_camera_overlap_mm3'] = float(clash.volume) if not clash.is_empty else 0.0
            r['watertight'] = bool(mount.is_watertight)
            r['pieces_mm3'] = sorted(round(float(p.volume)) for p in mount.split(only_watertight=False))
            r['bracket_cut_by_keepouts_mm3'] = build_mount.bracket_cut_mm3
        m_tool0 = mount.copy().apply_transform(T_TOOL0_CAD)
        P = mount_points(T_t0c, m_tool0)
        Pnew = mount_points(T_t0c, None if name == 'current' else m_tool0)
        behind = m_tool0.sample(6000)
        bm = behind[:, 2] < 0
        rr = np.hypot(behind[:, 0], behind[:, 1])
        bm = bm & (rr > 37.7)                      # the disc's pilot lip (r 32.5, 2.5 mm deep) is the flange's own fit
        r['mount_inside_keepout_mm'] = float(max(0.0, (KEEP_R - rr[bm]).max())) if bm.any() else 0.0
        r['a6_clear_to_deg'], r['wrist_clearance_mm'], r['link7_clearance_mm'] = wrist_clearance(
            np.r_[body_corners(T_t0c), behind[rr > 37.7][::6]], spheres)
        mass, cog = payload(mount, T_cad_cam)
        r['mount_g_solid'] = round(mass, 1)
        r['camera_mount_cog_tool0_mm'] = cog.round(1).tolist()
        r['lever_mm'] = float(np.hypot(*cog[:2]))
        r['views'] = {}
        for d, a in GRASPS:
            tag = f'{name}_d{d}_a{a}'
            v = score_view(T_t0c, scene, d, a, tag, out_dir,
                           [(np.asarray(m_tool0.vertices)[m_tool0.faces], LABELS['mount'])])
            v['arm'] = arm_clearance(P, d, a)
            r['views'][f'd{d}_a{a}'] = v
        report[name] = r
        print(json.dumps({name: r}, indent=1, default=float), flush=True)
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    print('renders and report.json in', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
