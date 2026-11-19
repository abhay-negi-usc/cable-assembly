"""The 2F-85 V3 OPEN (Robotiq STEP via sim/scripts/build_iiwa14.py) + the bench pads, per link, tool0 mm -> an npz.
Run from sim/ with ./run.sh (gmsh). Used by realsense_flangemount_d435i_usb_close.py."""
import sys, os, numpy as np
sys.path.insert(0, 'scripts'); import build_iiwa14 as B
out = sys.argv[1]
parts = B._step_triangles(os.path.join(B.V3_DIR, B.V3_STEP))
res = {}
for name, T in parts.items():
    Tb = np.stack([T[..., 2], T[..., 0], T[..., 1] + B.V3_Z_OFF_MM], axis=-1)
    g = name.split("Group")[1][0] if "Group" in name else '1'
    side = 'L' if Tb[..., 1].mean() < 0 else 'R'
    res.setdefault(g + side if g in '2345' else g, []).append(Tb)
L, Wd, Hh = B.PAD_MM
for side, s in (('L', -1.0), ('R', 1.0)):
    x0, x1 = sorted((s * B.PAD_MOUNT_FACE_MM, s * (B.PAD_MOUNT_FACE_MM - Hh)))
    pad = B._box_triangles((x0, B.PAD_CENTRE_ALONG_MM - L / 2, -Wd / 2), (x1, B.PAD_CENTRE_ALONG_MM + L / 2, Wd / 2))
    res['pad' + side] = [np.stack([pad[..., 2], pad[..., 0], pad[..., 1] + B.V3_Z_OFF_MM], axis=-1)]
out_d = {k: np.concatenate(v) + [0, 0, 7.94 + 11.0 + 0.14] for k, v in res.items()}
for k, v in sorted(out_d.items()):
    A = v.reshape(-1, 3); print(k, A.min(0).round(1), A.max(0).round(1))
np.savez_compressed(out, **out_d)
