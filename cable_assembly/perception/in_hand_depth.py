"""THE HELD PLUG'S TIP AND AXIS FROM THE WRIST CAMERA'S DEPTH ALONE -- no CAD render, no mask (pose6d.in_hand.depth_axis).

The user, 2026-11-18 ("do option 1, the depth-based measure"): the RGB-D CAD fit of the USB plug held by its cable in
front of the D435i slid the plug ~150 mm back along the line of sight (run usb_assembly_20261118_192913: its outline
still covered the plug, the depth there read the pegboard, depth score 0.08) and the tracker then refined from that.
The depth itself is good there: 93-98% of the plug's pixels read, and its points' long axis and far end, frame by
frame, repeat to ~1-1.5 mm and ~2 deg over 60 frames (tip [-39.3, 6.9, 222.7] mm in tool0, 7 deg / ~8 mm from the
hand-load nominal).

  1. THE POINTS: the depth pixels in the image box the belief's plug fills (grown), back-projected, into tool0.
  2. THE REGION: those within radius_mm of the belief's axis, from behind_mm behind the plug's junction (the pads hold
     the cable further back) to ahead_mm past its mating end. Then refit_passes times: the same about the fitted line.
  3. THE AXIS: first their principal direction, signed along the belief's (toward the mating end); then the
     CENTRELINE: per fit_bin_mm slice, the middle of the points' extent across the axis and their nearest face pushed
     surface_to_axis_mm (half the plug's thickness seen) away from the camera; a line through those, over the body
     (fit_back_mm behind the overmold's front). The principal direction alone tilts where a side face shows.
  4. THE TIP (tip_from 'front'): the metal shell reads little or no depth, so the points' far end wanders between the
     overmold's front and part-way up the shell. The overmold's front -- where the points' width across the axis
     falls below front_width_frac of the body's -- is steady, and the tip is front_to_tip_mm past it on the line:
     15.0, read against the colour image on the 2026-11-18 hand loads (the shell's end 14 / 14 / 15 mm past the
     found front on three, ~17 on a fourth held 14 deg off the nominal: +-1-2 mm; the CAD's 9.5 is short: the found
     front is ~5 mm behind the overmold's visible edge, where it tapers). 'end': the tip_pctl percentile along the axis.
  5. GATES -- a view that fails one is no estimate: >= min_points points, the points elongated (min_elong), at least
     min_extent_mm of plug along the axis, the axis within max_axis_deg and the tip within max_tip_mm of the belief's.

WHAT IT DOES NOT GIVE: the turn about the plug's own axis -- which way up it is. That stays the belief's
(in_hand_plug2d.pose: the least rotation that brings the belief's axis onto the measured one), and is the contact's
to tell (assembly.engage.way_up). Pure numpy.
"""

import numpy as np

DEFAULTS = dict(radius_mm=20.0, behind_mm=8.0, ahead_mm=20.0, roi_grow=1.6, stride=2, depth_range_mm=(100.0, 600.0),
                refit_passes=2, surface_to_axis_mm=4.0, tip_pctl=99.0, min_points=300, min_elong=2.0,
                min_extent_mm=25.0, max_axis_deg=30.0, max_tip_mm=30.0,
                tip_from='front', front_to_tip_mm=15.0, front_width_frac=0.75, bin_mm=1.0, min_bin_points=8,
                fit_bin_mm=2.0, fit_back_mm=28.0, centreline=True)


def settings(block):
    """DEFAULTS updated from a `pose6d.in_hand.depth_axis:` block (mm / deg)."""
    s = dict(DEFAULTS)
    for k, v in dict(block or {}).items():
        if k in s:
            s[k] = (tuple(float(x) for x in v) if isinstance(s[k], tuple)
                    else str(v).strip().lower() in ('1', 'true', 'yes', 'on') if isinstance(s[k], bool) else type(s[k])(v))
    s['tip_from'] = str(s['tip_from']).strip().lower()
    return s


def overmold_front(t, w, s):
    """Where along the axis (m, in `t`'s origin) the overmold's full width ends, toward the mating end: the last
    bin_mm slice whose width (5-95 percentile of `w`, the points across the axis) is still front_width_frac of the
    overmold's (the 90th percentile of the slices' widths), its 3-slice running median; None without one.
    THE METAL SHELL READS LITTLE OR NO DEPTH on the D435i (2026-11-18): the points' far end sits anywhere from the
    overmold's front to part-way up the shell, so the tip is placed from the front (front_to_tip_mm, the CAD's: its
    16 mm body ends 9.5 mm short of the mating face; on the lab plug, read against the colour image: ~15 past the
    front this finds)."""
    b = s['bin_mm'] / 1000.0
    edges = np.arange(t.min(), t.max() + b, b)
    mids, widths = [], []
    for e0 in edges[:-1]:
        m = (t >= e0) & (t < e0 + b)
        if m.sum() >= s['min_bin_points']:
            lo, hi = np.percentile(w[m], [5, 95])
            mids.append(e0 + b / 2.0)
            widths.append(hi - lo)
        else:
            mids.append(e0 + b / 2.0)
            widths.append(0.0)
    widths = np.asarray(widths)
    if len(widths) < 5 or not np.any(widths > 0):
        return None
    sm = np.array([np.median(widths[max(0, i - 1):i + 2]) for i in range(len(widths))])
    # the body's width: the 90th percentile of the slices -- the overmold need not be most of them when the region
    # runs along the cable (a hand load placed further out: 2026-11-18 20:18, 36 mm past the nominal)
    full = np.percentile(sm[sm > 0], 90)
    ok = np.nonzero(sm >= s['front_width_frac'] * full)[0]
    if len(ok) == 0:
        return None
    return float(mids[ok[-1]] + b / 2.0)


def _box(K, T_cam_tool0, pts_tool0, grow, shape):
    """The pixel box [u0, v0, u1, v1] that `pts_tool0` fill, grown by `grow` about its middle; None if behind."""
    T = np.asarray(T_cam_tool0, float)
    P = pts_tool0 @ T[:3, :3].T + T[:3, 3]
    if np.any(P[:, 2] <= 1e-3):
        return None
    uv = (np.asarray(K, float) @ (P / P[:, 2:3]).T).T[:, :2]
    lo, hi = uv.min(0), uv.max(0)
    c, h = (lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, 8.0)
    H, W = shape[:2]
    u0, v0 = np.floor(np.clip(c - grow * h, 0, [W - 1, H - 1])).astype(int)
    u1, v1 = np.ceil(np.clip(c + grow * h, 0, [W - 1, H - 1])).astype(int)
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None
    return u0, v0, u1, v1


def points(depth_m, K, T_cam_tool0, box, stride, depth_range_m):
    """(n, 3) tool0 points (m) of the valid depth pixels in `box`, every `stride`-th pixel."""
    u0, v0, u1, v1 = box
    D = np.asarray(depth_m, float)[v0:v1 + 1:stride, u0:u1 + 1:stride]
    vs, us = np.mgrid[v0:v1 + 1:stride, u0:u1 + 1:stride]
    ok = np.isfinite(D) & (D > depth_range_m[0]) & (D < depth_range_m[1])
    Z, u, v = D[ok], us[ok].astype(float), vs[ok].astype(float)
    K = np.asarray(K, float)
    X = np.c_[(u - K[0, 2]) * Z / K[0, 0], (v - K[1, 2]) * Z / K[1, 1], Z]
    T_tc = np.linalg.inv(np.asarray(T_cam_tool0, float))
    return X @ T_tc[:3, :3].T + T_tc[:3, 3]


def _view_side(c, a, cam):
    """(the unit direction from the camera to `c` across the axis `a`, the unit direction across both)."""
    view = c - cam
    view = view - (view @ a) * a
    view = view / max(np.linalg.norm(view), 1e-9)
    return view, np.cross(a, view)


def _centres(sel, c, a, side, view, t, lo, hi, s):
    """(m, 3) centreline points: per fit_bin_mm slice of `t` in [lo, hi] with >= min_bin_points, the middle of the
    points' 5-95 percentile extent along `side`, and their nearest face (10th percentile along `view`) pushed
    surface_to_axis_mm away from the camera."""
    b = s['fit_bin_mm'] / 1000.0
    t0, t1 = max(t.min(), lo), min(t.max(), hi)
    d = sel - c
    out = []
    for e0 in np.arange(t0, t1, b):
        m = (t >= e0) & (t < e0 + b)
        if m.sum() < s['min_bin_points']:
            continue
        sw = np.percentile(d[m] @ side, [5, 95]).mean()
        dv = np.percentile(d[m] @ view, 10) + s['surface_to_axis_mm'] / 1000.0
        out.append(c + a * (e0 + b / 2.0) + side * sw + view * dv)
    return np.asarray(out)


def _select(P, origin, axis, t0, t1, radius):
    """The points within `radius` of the line origin + t * axis, t0 <= t <= t1."""
    d = P - origin
    t = d @ axis
    r = np.linalg.norm(d - np.outer(t, axis), axis=1)
    return P[(t >= t0) & (t <= t1) & (r <= radius)]


def estimate(depth_m, K, T_cam_tool0, T_tool0_plug_prior, junction_z, mating_z, s=None):
    """{'tip', 'axis' (tool0, m / unit, toward the mating end), 'n', 'elong', 'extent_mm', 'rms_mm', 'axis_deg',
    'tip_mm' (from the belief's)} -- or {'why': ...} when a gate refuses the view. `junction_z` / `mating_z`: the
    plug's rear and mating end along its CAD z (P6.JUNCTION_Z / MATING_Z)."""
    s = dict(DEFAULTS) if s is None else s
    Tp = np.asarray(T_tool0_plug_prior, float)
    a0 = Tp[:3, 2] / np.linalg.norm(Tp[:3, 2])
    j0 = Tp[:3, :3] @ [0.0, 0.0, junction_z] + Tp[:3, 3]
    tip0 = Tp[:3, :3] @ [0.0, 0.0, mating_z] + Tp[:3, 3]
    L = float(mating_z - junction_z)
    behind, ahead, radius = s['behind_mm'] / 1000.0, s['ahead_mm'] / 1000.0, s['radius_mm'] / 1000.0
    seg = np.array([j0 - a0 * behind, tip0 + a0 * ahead])
    ring = np.array([np.cross(a0, e) for e in np.eye(3)])
    ring = ring[np.linalg.norm(ring, axis=1) > 1e-6]
    ring = radius * ring / np.linalg.norm(ring, axis=1)[:, None]
    hull = np.vstack([seg] + [seg + r for r in ring] + [seg - r for r in ring])
    box = _box(K, T_cam_tool0, hull, s['roi_grow'], np.shape(depth_m))
    if box is None:
        return {'why': 'the belief is not in the image'}
    P = points(depth_m, K, T_cam_tool0, box, max(1, int(s['stride'])),
               (s['depth_range_mm'][0] / 1000.0, s['depth_range_mm'][1] / 1000.0))
    sel = _select(P, j0, a0, -behind, L + ahead, radius)
    if len(sel) < s['min_points']:
        return {'why': f'{len(sel)} depth points about the belief (min_points {s["min_points"]:.0f})', 'box': box}
    c, a = sel.mean(0), a0
    for _ in range(max(1, int(s['refit_passes'])) + 1):
        c = sel.mean(0)
        _, sv, Vt = np.linalg.svd(sel - c, full_matrices=False)
        a = Vt[0] * np.sign(Vt[0] @ a0)
        t = (sel - c) @ a
        nxt = _select(P, c, a, t.min() - 0.005, t.max() + 0.005, radius)
        if len(nxt) < s['min_points']:
            break
        sel = nxt
    c = sel.mean(0)
    _, sv, Vt = np.linalg.svd(sel - c, full_matrices=False)
    a = Vt[0] * np.sign(Vt[0] @ a0)
    t = (sel - c) @ a
    elong = float(sv[0] / max(sv[1], 1e-9))
    extent = float(np.percentile(t, s['tip_pctl']) - np.percentile(t, 100.0 - s['tip_pctl']))
    # THE CENTRELINE, NOT THE POINTS' PRINCIPAL AXIS: on the wide body the camera also sees a side face, which pulls
    # the points' mean sideways where it does not on the narrower strain relief -- a plain fit tilts ~7 deg (the
    # synthetic box of tests/test_in_hand_depth.py). Per fit_bin_mm slice: the middle of the points' extent across the
    # axis, and the face nearest the camera pushed surface_to_axis_mm back; a line through those, over the body only
    # (fit_back_mm behind the overmold's front) when the front is found.
    cam = np.linalg.inv(np.asarray(T_cam_tool0, float))[:3, 3]
    front, rms = None, float('nan')
    for _ in range(max(1, int(s['refit_passes'])) if s['centreline'] else 0):
        view, side = _view_side(c, a, cam)
        t = (sel - c) @ a
        if s['tip_from'] == 'front':
            front = overmold_front(t, (sel - c) @ side, s)
        hi = np.inf if front is None else front
        lo = -np.inf if front is None else front - s['fit_back_mm'] / 1000.0
        cen = _centres(sel, c, a, side, view, t, lo, hi, s)
        if len(cen) < 4:
            break
        cc = cen.mean(0)
        _, _, Vt = np.linalg.svd(cen - cc, full_matrices=False)
        a = Vt[0] * np.sign(Vt[0] @ a0)
        d = cen - cc
        rms = float(1000.0 * np.sqrt(np.mean(np.sum((d - np.outer(d @ a, a)) ** 2, axis=1))))
        c = cc
    view, side = _view_side(c, a, cam)
    t = (sel - c) @ a
    r = np.linalg.norm((sel - c) - np.outer(t, a), axis=1)
    t_end = float(np.percentile(t, s['tip_pctl']))
    front = None
    if s['tip_from'] == 'front':
        front = overmold_front(t, (sel - c) @ side, s)
        if front is None:
            return {'why': 'no overmold front in the depth points (its width never settles)', 'box': box,
                    'axis': a, 'n': int(len(sel))}
        tip = c + a * (front + s['front_to_tip_mm'] / 1000.0)
    else:
        tip = c + a * t_end
    axis_deg = float(np.degrees(np.arccos(np.clip(a @ a0, -1.0, 1.0))))
    tip_mm = float(1000.0 * np.linalg.norm(tip - tip0))
    out = {'tip': tip, 'axis': a, 'n': int(len(sel)), 'elong': elong, 'extent_mm': 1000.0 * extent,
           'rms_mm': rms, 'spread_mm': float(1000.0 * np.sqrt(np.mean(r ** 2))), 'axis_deg': axis_deg, 'tip_mm': tip_mm, 'box': box,
           'front_to_end_mm': None if front is None else 1000.0 * (t_end - front)}
    if len(sel) < s['min_points']:
        return dict(out, why=f'{len(sel)} depth points on the plug (min_points {s["min_points"]:.0f})')
    if elong < s['min_elong']:
        return dict(out, why=f'the depth points are not elongated ({elong:.1f} < {s["min_elong"]:.1f})')
    if 1000.0 * extent < s['min_extent_mm']:
        return dict(out, why=f'{1000.0 * extent:.0f} mm of plug along its axis (min_extent_mm {s["min_extent_mm"]:.0f})')
    if axis_deg > s['max_axis_deg']:
        return dict(out, why=f'the axis {axis_deg:.0f} deg from the belief\'s (max_axis_deg {s["max_axis_deg"]:.0f})')
    if tip_mm > s['max_tip_mm']:
        return dict(out, why=f'the tip {tip_mm:.0f} mm from the belief\'s (max_tip_mm {s["max_tip_mm"]:.0f})')
    return out
