"""THE HELD PLUG'S TIP AND AXIS FROM ONE COLOUR IMAGE -- no depth, no CAD render (pose6d.in_hand.rgb2d).

For a plug HELD BY ITS CABLE in front of a wrist camera too near for depth (the D435i: the plug ~0.17 m away,
inside its ~0.2 m minimum; the user, 2026-11-14: "implement the new pose estimator" in usb_assembly). Studied in
analysis/in_hand_rgb_usb.py on the USB runs of 2026-11-13, scored against the depth-PCA estimate of the same
D405 views: tip 3.7 mm median (p90 5.1), axis 5.9 deg (p90 10.7), 14 of 18 views -- the server's RGB jaw fit on the
same views 12.8 mm (p90 24.1) / 33.7 deg. NOT YET SHOWN ON THE D435i: on run 180404 it kept 2 of 5 views and has
no truth there.

  1. THE MASK. SAM2 (the estimator's server, op 'segment') on the box the belief's plug fills grown 60%, with
     positive points along the belief's plug past the pads and negatives either side of the pads (prompts()).
  2. TRIMMED to the unsaturated pixels (black overmold and cord, silver shell; HSV saturation < sat_max): the
     pink pads and the coloured fixture behind the plug go -- 2026-11-14, untrimmed, the mask ran onto the blue
     fixture and the "tip" sat on it in three D435i views that agreed with each other to 5 mm. The component
     nearest the fingertip's pixel is kept (the cord leaves the pads there).
  3. THE 2D AXIS AND TIP: the mask's principal axis from its part beyond the pads, the tip at its far end.
  4. THE RANGE FROM THE WIDTH: the mask's width a few mm behind the tip (back_mm) is the plug's effective width
     width_mm (16.1 mm, calibrated leave-one-run-out against the depth: it is the plug as SAM2 outlines it, seen
     from that side, not the CAD's 16 x 8) -- Z = width_mm * fx / width_px; the tip is its pixel's ray at Z.
  5. THE AXIS FROM THE TIP TO THE GRIP: the cord is held between the pads, on their mid-plane (tool0 y = 0) at the
     fingertip's height (tool0 z): the point there on the plane of sight through the 2D axis is where the plug's
     axis starts. Back-projecting the tip onto the pads' plane instead is ill-posed: the camera sits nearly IN it.
     SEEN END-ON (the pads' line in the plane of sight: the camera nearly in the pads' mid-plane AND the plug
     hanging in it -- the D435i's nominal case on its flange mount, which sits 2.5 mm off that plane) there is no
     grip point to take: end_on 'belief' keeps the measured tip and takes the axis from the belief, turned the
     least into the plane of sight (the 2D axis still fixes that plane); 'refuse' gives no estimate.
  6. GATES -- a view that fails one is no estimate, not a guess: the 2D direction within max_dir_deg of the
     belief's, the range in z_range_mm, the mask elongated (min_elong).

WHAT IT DOES NOT GIVE: the turn about the plug's own axis -- which way up the plug is. That is kept from the belief
(pose(): the least rotation that brings the belief's axis onto the measured one) and left to the contact
(assembly.engage.way_up). Pure numpy/OpenCV: the mask comes from the caller.
"""

import cv2
import numpy as np

DEFAULTS = dict(width_mm=16.1, sat_max=120, back_mm=(3.0, 13.0), max_dir_deg=60.0, z_range_mm=(120.0, 300.0),
                min_elong=2.0, min_end_on=0.05, min_px=200, end_on='belief')


def settings(block):
    """DEFAULTS updated from a `pose6d.in_hand.rgb2d:` block (mm / deg)."""
    s = dict(DEFAULTS)
    for k, v in dict(block or {}).items():
        if k == 'end_on':
            s[k] = str(v).strip().lower()
        elif k in s:
            s[k] = tuple(float(x) for x in v) if isinstance(s[k], tuple) else float(v)
    return s


def project(K, T_cam_x, p):
    """The pixel of point `p` (its frame x) seen by the camera; None behind it."""
    c = np.asarray(T_cam_x, float)[:3, :3] @ np.asarray(p, float) + np.asarray(T_cam_x, float)[:3, 3]
    if c[2] <= 1e-6:
        return None
    return (np.asarray(K, float) @ (c / c[2]))[:2]


def prompts(K, T_cam_tool0, T_tool0_plug, model_pts, ftip_z, shape):
    """SAM2's prompts for the held plug: (box [x0, y0, x1, y1] -- the belief's plug grown 60% --, positive points
    along it past the pads, negative points either side of the pads, the fingertip's pixel); None when the belief
    is not in the image. `model_pts`: points on the plug's CAD (its frame, m); `ftip_z`: the fingertip's tool0 z."""
    T = np.asarray(T_cam_tool0, float) @ np.asarray(T_tool0_plug, float)
    P = np.asarray(model_pts, float) @ T[:3, :3].T + T[:3, 3]
    P = P[P[:, 2] > 1e-3]
    f = project(K, T_cam_tool0, [0.0, 0.0, ftip_z])
    if len(P) < 3 or f is None:
        return None
    uv = (np.asarray(K, float) @ (P / P[:, 2:3]).T).T[:, :2]
    lo, hi = uv.min(0), uv.max(0)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    H, W = shape[:2]
    box = np.r_[np.clip(c - 1.6 * h, 0, [W - 1, H - 1]), np.clip(c + 1.6 * h, 0, [W - 1, H - 1])]
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None
    L = float(np.linalg.norm(c - f))
    d = (c - f) / max(L, 1.0)
    n = np.array([-d[1], d[0]])
    pos = [f + d * L * t for t in (0.6, 1.0, 1.4)]
    neg = [f - d * L * 0.4 + n * L * 0.5, f - d * L * 0.4 - n * L * 0.5]
    inside = lambda p: 0 <= p[0] < W and 0 <= p[1] < H        # noqa: E731
    return box, [p for p in pos if inside(p)], [p for p in neg if inside(p)], f


def trim(mask, color_bgr, ftip_uv, sat_max):
    """`mask`'s unsaturated pixels, opened, the component nearest `ftip_uv`; None if nothing is left."""
    hsv = cv2.cvtColor(np.ascontiguousarray(color_bgr), cv2.COLOR_BGR2HSV)
    m = np.asarray(mask, bool) & (hsv[..., 1] < sat_max)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    ys, xs = np.nonzero(lab)
    i = int(np.argmin(np.hypot(xs - ftip_uv[0], ys - ftip_uv[1])))
    return lab == lab[ys[i], xs[i]]


def axis_tip_2d(m, ftip_uv, pctl=99.5):
    """(a point on the 2D axis, its unit direction away from the pads, the tip pixel) of mask `m`: the axis from
    the part beyond the pads (the thin cord between them would tilt it), the tip on that axis at the far end."""
    ys, xs = np.nonzero(m)
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0] * np.sign(Vt[0] @ (c - ftip_uv))
    far = P[(P - c) @ d > np.percentile((P - c) @ d, 30)]
    c2 = far.mean(0)
    _, _, Vt = np.linalg.svd(far - c2, full_matrices=False)
    d2 = Vt[0] * np.sign(Vt[0] @ d)
    t2 = (P - c2) @ d2
    nrm = np.array([-d2[1], d2[0]])
    tip = c2 + d2 * np.percentile(t2, pctl)
    end = t2 > np.percentile(t2, 95)
    return c2, d2, tip + nrm * np.median(((P - c2) @ nrm)[end])


def widths(m, c2, d2, tip_uv, s0_px, s1_px, nbin=6):
    """(distance back from the tip px, the mask's width across the axis px) in `nbin` slices s0_px..s1_px back."""
    ys, xs = np.nonzero(m)
    P = np.c_[xs, ys].astype(float)
    t = (tip_uv - P) @ d2
    s = (P - c2) @ np.array([-d2[1], d2[0]])
    e = np.linspace(s0_px, s1_px, nbin)
    out = []
    for a, b in zip(e[:-1], e[1:]):
        sel = (t >= a) & (t < b)
        if sel.sum() > 10:
            out.append(((a + b) / 2, np.percentile(s[sel], 95) - np.percentile(s[sel], 5)))
    return np.array(out)


def estimate(color_bgr, mask, K, T_cam_tool0, ftip_z, T_tool0_plug_prior, mating_z, s=None):
    """{'tip', 'axis' (tool0, m / unit, the axis toward the tip), 'Z' (m), 'tip_uv', 'mask'} -- or
    {'why': ...} when a gate refuses the view. `mating_z`: the tip's z in the plug's CAD frame (P6.MATING_Z)."""
    s = dict(DEFAULTS) if s is None else s
    K, T_ct = np.asarray(K, float), np.asarray(T_cam_tool0, float)
    f = project(K, T_ct, [0.0, 0.0, ftip_z])
    if f is None:
        return {'why': 'the fingertip is behind the camera'}
    m = trim(mask, color_bgr, f, s['sat_max'])
    if m is None or m.sum() < s['min_px']:
        return {'why': 'too little of the mask is the plug (after the colour trim)'}
    c2, d2, tip = axis_tip_2d(m, f)
    fx = K[0, 0]
    Z = 0.2
    for _ in range(4):                                   # the slices are in mm, so they need the range: iterate
        w = widths(m, c2, d2, tip, s['back_mm'][0] * fx / (Z * 1000.0), s['back_mm'][1] * fx / (Z * 1000.0))
        if len(w) == 0:
            return {'why': 'no width behind the tip'}
        Z = s['width_mm'] / 1000.0 * fx / float(np.median(w[:, 1]))
    # THE GATES
    pt = project(K, T_ct @ np.asarray(T_tool0_plug_prior, float), [0.0, 0.0, mating_z])
    if pt is not None:
        d0 = (pt - f) / max(np.linalg.norm(pt - f), 1e-9)
        ang = float(np.degrees(np.arccos(np.clip(d2 @ d0, -1.0, 1.0))))
        if ang > s['max_dir_deg']:
            return {'why': f'the plug points {ang:.0f} deg from the belief in the image'}
    if not (s['z_range_mm'][0] <= 1000.0 * Z <= s['z_range_mm'][1]):
        return {'why': f'range {1000.0 * Z:.0f} mm out of z_range_mm'}
    ys, xs = np.nonzero(m)
    sv = np.linalg.svd(np.c_[xs, ys] - [xs.mean(), ys.mean()], full_matrices=False, compute_uv=False)
    if sv[0] / max(sv[1], 1e-6) < s['min_elong']:
        return {'why': f'the mask is not elongated ({sv[0] / max(sv[1], 1e-6):.1f})'}
    r1 = np.linalg.solve(K, [tip[0], tip[1], 1.0])
    r2 = np.linalg.solve(K, [*(tip - d2 * 100.0), 1.0])
    n = np.cross(r1, r2)                                 # the plane of sight through the 2D axis, camera frame
    q0 = T_ct[:3, :3] @ [0.0, 0.0, ftip_z] + T_ct[:3, 3]
    qx = T_ct[:3, 0]                                     # the pads' centre line: tool0 x through the grip
    T_tc = np.linalg.inv(T_ct)
    X = r1 * Z / r1[2]
    tip3 = T_tc[:3, :3] @ X + T_tc[:3, 3]
    out = {'tip': tip3, 'Z': float(Z), 'tip_uv': tip, 'mask': m}
    if abs(n @ qx) / np.linalg.norm(n) < s['min_end_on']:
        if s.get('end_on', 'belief') != 'belief':
            return {'why': "the pads' line is seen end-on: no grip point on it"}
        nt = T_tc[:3, :3] @ (n / np.linalg.norm(n))     # the plane of sight's normal, tool0
        b = np.asarray(T_tool0_plug_prior, float)[:3, 2]
        ax = b - (b @ nt) * nt
        if np.linalg.norm(ax) < 1e-6:
            return {'why': "the pads' line is seen end-on and the belief's axis is across the plane of sight"}
        return dict(out, axis=ax / np.linalg.norm(ax), axis_from='belief')
    A = np.array([-(n @ q0) / (n @ qx), 0.0, ftip_z])    # the grip, tool0
    ax = tip3 - A
    if np.linalg.norm(ax) < 1e-3:
        return {'why': 'the tip is at the grip'}
    return dict(out, axis=ax / np.linalg.norm(ax), axis_from='grip', grip=A)


def pose(tip, axis, T_tool0_plug_prior, mating_z):
    """The plug's T_tool0_plug with its tip at `tip` and its axis (CAD +z) along `axis`: the belief's turned the
    least that does it -- its turn about its own axis is the belief's (no image here measures it)."""
    Rp = np.asarray(T_tool0_plug_prior, float)[:3, :3]
    a, b = Rp[:, 2], np.asarray(axis, float) / np.linalg.norm(axis)
    w = np.cross(a, b)
    sn, cs = float(np.linalg.norm(w)), float(a @ b)
    if sn < 1e-12:
        R = np.eye(3)
    else:
        k = w / sn
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        th = np.arctan2(sn, cs)
        R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
    T = np.eye(4)
    T[:3, :3] = R @ Rp
    T[:3, 3] = np.asarray(tip, float) - T[:3, 2] * mating_z
    return T
