"""RGB-ONLY in-hand pose of the held USB plug, model-free: its 2D mask (analysis/in_hand_rgb_segment.py) trimmed to
the unsaturated pixels (black overmold/cord, silver/white shell -- not the pink pads or the coloured fixture), the
axis fitted in the image, the far end found, both back-projected onto the PADS' MID-PLANE (tool y = 0: the jaws centre
the cord there). Gives the mating tip and the axis in the pads' plane (slide x/z + pivot); no turn, no CAD.
Scored against the depth-PCA estimate of the same views on the D405 runs (proxy truth, 1.3 mm repeatable), with the
server's RGB jaw fit (analysis/in_hand_usb_methods.py) for comparison.
    ~/miniconda3/envs/cable/bin/python analysis/in_hand_rgb_usb.py
"""
import cv2
import numpy as np

V = dict(np.load('data/in_hand_rgb_usb/views.npz'))         # read once: npz members decompress on every access
MK = dict(np.load('data/in_hand_rgb_usb/masks.npz'))
KEYS = sorted({k.split('__')[0] for k in V})


def trim(mask, rgb, ftip_uv, sat_max=70):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = mask & (hsv[..., 1] < sat_max)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    # the component nearest the fingertip pixel (the cord leaves the pads there), else the largest
    ys, xs = np.nonzero(lab)
    d = np.hypot(xs - ftip_uv[0], ys - ftip_uv[1])
    keep = lab[ys[np.argmin(d)], xs[np.argmin(d)]]
    return lab == keep


def axis_tip_2d(m, ftip_uv, pctl=99.5):
    ys, xs = np.nonzero(m)
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0] * np.sign(Vt[0] @ (c - ftip_uv))           # away from the pads
    t = (P - c) @ d
    # the axis from the part BEYOND the pads (the cord between them is thin; the plug sets the axis)
    far = P[t > np.percentile(t, 30)]
    c2 = far.mean(0)
    _, _, Vt = np.linalg.svd(far - c2, full_matrices=False)
    d2 = Vt[0] * np.sign(Vt[0] @ d)
    t2 = (P - c2) @ d2
    tip = c2 + d2 * np.percentile(t2, pctl)
    # the tip ON the axis: the end of the mask's spine (median lateral offset of the last 5%)
    lat = (P - c2) @ np.array([-d2[1], d2[0]])
    end = t2 > np.percentile(t2, 95)
    tip = tip + np.array([-d2[1], d2[0]]) * np.median(lat[end])
    return c2, d2, tip


def to_plane(uv, K, T_ct):
    """The pixel's ray onto the pads' mid-plane (tool y = 0), in tool0 coordinates."""
    r = np.linalg.solve(K, [uv[0], uv[1], 1.0])
    n, p0 = T_ct[:3, 1], T_ct[:3, 3]
    X = r * (n @ p0) / (n @ r)
    T_tc = np.linalg.inv(T_ct)
    return T_tc[:3, :3] @ X + T_tc[:3, 3]


def estimate(k, variant='pts', sat_max=70):
    if f'{k}__{variant}' not in MK:
        return None
    rgb, K, T_ct, f = V[f'{k}__rgb'], V[f'{k}__K'], V[f'{k}__T_ct'], V[f'{k}__ftip_uv']
    m = trim(MK[f'{k}__{variant}'], rgb, f, sat_max)
    if m is None or m.sum() < 200:
        return None
    c, d, tip = axis_tip_2d(m, f)
    a = to_plane(tip, K, T_ct)
    b = to_plane(tip - 60 * d, K, T_ct)
    ax = (a - b) / np.linalg.norm(a - b)
    return dict(tip=a, axis=ax, mask=m, tip_uv=tip, d_uv=d)


def score(est, k):
    gt_tip, gt_ax = V[f'{k}__pca_tip'], V[f'{k}__pca_axis']
    if est is None or not np.all(np.isfinite(gt_tip)):
        return None
    e = est['tip'] - gt_tip
    al = e @ gt_ax
    return dict(err=np.linalg.norm(e) * 1000, along=al * 1000, lat=np.linalg.norm(e - al * gt_ax) * 1000,
                ang=np.degrees(np.arccos(np.clip(est['axis'] @ gt_ax, -1, 1))))


def jaw_baseline():
    rows = list(np.load('data/in_hand_methods_usb/rows.npy', allow_pickle=True))
    P6TIP = None
    from cable_assembly.perception import pose6d as P6
    P6TIP = np.array([0, 0, P6.MATING_Z[P6.USB]])
    out = {}
    for r in rows:
        if r['init'] == 0 and r['T'] is not None and r['method'] in ('jaw', 'refine6', 'detect') and not r['depth']:
            out[(r['method'], f"{r['run']}_{r['view']}")] = dict(tip=r['T'][:3, :3] @ P6TIP + r['T'][:3, 3],
                                                                axis=r['T'][:3, 2])
    return out


def report():
    import sys
    sys.path.insert(0, '.')
    base = jaw_baseline()
    d405 = [k for k in KEYS if not k.startswith('180404')]
    print(f"{'method':28s} {'n':>3s} | {'tip err':>8s} {'along':>7s} {'lat':>6s} {'axis':>6s}   (vs depth-PCA, D405 runs: median [90th pct]) | D435i repeat tip/axis")
    variants = [(f'2D-plane {v} sat<{s}', (lambda k, v=v, s=s: estimate(k, v, s))) for v in ('pts', 'box', 'dino')
                for s in (50, 70, 255)]
    variants += [(f'server {m} (RGB)', (lambda k, m=m: base.get((m, k)))) for m in ('jaw', 'refine6', 'detect')]
    res = {}
    for name, fn in variants:
        S = [score(fn(k), k) for k in d405]
        S = [s for s in S if s is not None]
        E = [fn(k) for k in KEYS if k.startswith('180404')]
        E = [e for e in E if e is not None]
        rep = ''
        if len(E) >= 2:
            T = np.array([e['tip'] for e in E])
            A = np.array([e['axis'] for e in E])
            am = A.mean(0) / np.linalg.norm(A.mean(0))
            rep = f"{np.sqrt(np.mean(np.sum((T - T.mean(0)) ** 2, 1))) * 1000:5.1f} mm {np.sqrt(np.mean(np.degrees(np.arccos(np.clip(A @ am, -1, 1))) ** 2)):5.1f} deg  tip {np.round(T.mean(0) * 1000, 1).tolist()}"
        if not S:
            print(f'{name:28s}   0'); continue
        q = lambda key: (np.median([abs(s[key]) for s in S]), np.percentile([abs(s[key]) for s in S], 90))  # noqa: E731
        print(f"{name:28s} {len(S):3d} | " + ' '.join(f'{q(kk)[0]:5.1f}[{q(kk)[1]:4.1f}]' for kk in ('err', 'along', 'lat', 'ang'))
              + f' | {rep}')
        res[name] = S
    return res


if __name__ == '__main__':
    report()


# ------------------------------------------------------------------ SCALE: range from the plug's apparent width
def widths(m, c2, d2, tip_uv, s0_px, s1_px, nbin=6):
    """The mask's width (px, across the axis) in `nbin` slices from s0_px to s1_px back from the tip: (t back, width)."""
    ys, xs = np.nonzero(m)
    P = np.c_[xs, ys].astype(float)
    t = (tip_uv - P) @ d2                      # distance back from the tip, px
    s = (P - c2) @ np.array([-d2[1], d2[0]])
    out = []
    for a, b in zip(np.linspace(s0_px, s1_px, nbin)[:-1], np.linspace(s0_px, s1_px, nbin)[1:]):
        sel = (t >= a) & (t < b)
        if sel.sum() > 10:
            out.append(((a + b) / 2, np.percentile(s[sel], 95) - np.percentile(s[sel], 5)))
    return np.array(out)


def scale_estimate(k, W_mm, variant='pts', sat=255, back=(3.0, 13.0)):
    """Tip and axis (tool0) from the 2D mask and the plug's known effective width W_mm (from `back` mm behind the
    tip): the tip's range from the width there; the axis from the 2D axis and the range a slice 25 mm back."""
    rgb, K, T_ct, f = V[f'{k}__rgb'], V[f'{k}__K'], V[f'{k}__T_ct'], V[f'{k}__ftip_uv']
    if f'{k}__{variant}' not in MK:
        return None
    m = trim(MK[f'{k}__{variant}'], rgb, f, sat)
    if m is None or m.sum() < 200:
        return None
    c2, d2, tip = axis_tip_2d(m, f)
    fx = K[0, 0]
    Z = 0.2
    for _ in range(4):                          # the slices in mm need the range; iterate
        w = widths(m, c2, d2, tip, back[0] * fx / (Z * 1000), back[1] * fx / (Z * 1000))
        if len(w) == 0:
            return None
        Z = W_mm / 1000.0 * fx / np.median(w[:, 1])
    ray = np.linalg.solve(K, [tip[0], tip[1], 1.0])
    X_tip = ray * Z / ray[2]
    # a second station 25 mm back along the image axis: the same width-to-range (the overmold is near-prismatic
    # there? no -- use the tip's range and the 2D direction, the axis assumed square to the line of sight's change)
    p2 = tip - d2 * 25.0 * fx / (Z * 1000)
    r2 = np.linalg.solve(K, [p2[0], p2[1], 1.0])
    # the axis: in the plane through the camera and the 2D axis line, PERPENDICULAR to the tool's y (the jaws hold
    # it in the pads' plane: its direction has no tool-y component)
    n_img = np.cross(ray, r2)                   # normal of the plane of sight through the 2D axis
    ty = T_ct[:3, 1]
    ax = np.cross(n_img, ty)
    ax /= np.linalg.norm(ax)
    if ax @ (X_tip - r2 * Z / r2[2]) < 0:
        ax = -ax
    T_tc = np.linalg.inv(T_ct)
    return dict(tip=T_tc[:3, :3] @ X_tip + T_tc[:3, 3], axis=T_tc[:3, :3] @ ax, Z=Z, w=np.median(w[:, 1]),
                mask=m, tip_uv=tip, d_uv=d2)


def calibrate_and_score(variant='pts', sat=255, back=(3.0, 13.0)):
    d405 = [k for k in KEYS if not k.startswith('180404')]
    runs = sorted({k.split('_')[0] for k in d405})
    # effective width per view from the proxy truth: W = w_px * Z_gt / fx
    Wv = {}
    for k in d405:
        e = scale_estimate(k, 12.0, variant, sat, back)
        gt = V[f'{k}__pca_tip']
        if e is None or not np.all(np.isfinite(gt)):
            continue
        T_ct = V[f'{k}__T_ct']
        Zgt = (T_ct[:3, :3] @ gt + T_ct[:3, 3])[2]
        Wv[k] = e['w'] * Zgt / V[f'{k}__K'][0, 0] * 1000
    S = []
    for run in runs:                            # leave one run out
        W = np.median([w for kk, w in Wv.items() if not kk.startswith(run)])
        for k in d405:
            if k.startswith(run):
                s = score(scale_estimate(k, W, variant, sat, back), k)
                if s:
                    S.append(s)
    return S, np.median(list(Wv.values())), np.std(list(Wv.values()))


# ------------------------------------------------------------------ THE IMPROVED RGB ESTIMATE
def rgb_inhand(k, W_mm, variant='pts', sat=255, back=(3.0, 13.0), max_dir_deg=60.0, z_range=(0.12, 0.30),
               min_elong=2.0):
    """scale_estimate's tip, the axis ANCHORED at the cord's grip (where the 2D axis's plane of sight meets the
    pads' centre line, tool y = 0 and z = the fingertip's), and gates: the 2D direction within max_dir_deg of the
    grasp's, the range in z_range, the mask elongated. None when a gate fails (why in e['why'])."""
    e = scale_estimate(k, W_mm, variant, sat, back)
    if e is None:
        return None
    K, T_ct, f = V[f'{k}__K'], V[f'{k}__T_ct'], V[f'{k}__ftip_uv']
    Tp = T_ct @ V[f'{k}__prior_T_t0p']                     # the grasp's plug, its tip projected
    from cable_assembly.perception import pose6d as P6
    pt = Tp[:3, :3] @ [0.0, 0.0, P6.MATING_Z[P6.USB]] + Tp[:3, 3]
    d0 = (K @ (pt / pt[2]))[:2] - f
    d0 /= np.linalg.norm(d0)
    if np.degrees(np.arccos(np.clip(e['d_uv'] @ d0, -1, 1))) > max_dir_deg:
        return None
    if not (z_range[0] <= e['Z'] <= z_range[1]):
        return None
    ys, xs = np.nonzero(e['mask'])
    P = np.c_[xs, ys] - [xs.mean(), ys.mean()]
    sv = np.linalg.svd(P, full_matrices=False, compute_uv=False)
    if sv[0] / max(sv[1], 1e-6) < min_elong:
        return None
    r1 = np.linalg.solve(K, [*e['tip_uv'], 1.0])
    r2 = np.linalg.solve(K, [*(e['tip_uv'] - e['d_uv'] * 100.0), 1.0])
    n = np.cross(r1, r2)
    zf = V[f'{k}__T_tool0_ftip'][2, 3]
    q0 = T_ct[:3, :3] @ [0.0, 0.0, zf] + T_ct[:3, 3]
    qx = T_ct[:3, 0]
    if abs(n @ qx) / np.linalg.norm(n) < 0.05:                  # the pads' line seen end-on: no anchor
        return None
    A = np.array([-(n @ q0) / (n @ qx), 0.0, zf])
    ax = e['tip'] - A
    e['axis'] = ax / np.linalg.norm(ax)
    e['anchor'] = A
    return e


def fuse(ests):
    """The views of one hold: the median tip, the mean axis of those within 5 mm of it."""
    T = np.array([e['tip'] for e in ests])
    m = np.median(T, 0)
    keep = [e for e in ests if np.linalg.norm(e['tip'] - m) < 0.005] or ests
    a = np.mean([e['axis'] for e in keep], 0)
    return dict(tip=np.median([e['tip'] for e in keep], 0), axis=a / np.linalg.norm(a), n=len(keep))


def evaluate_improved(**kw):
    d405 = [k for k in KEYS if not k.startswith('180404')]
    runs = sorted({k.split('_')[0] for k in d405})
    Wv = {}
    for k in d405:
        e = scale_estimate(k, 12.0, kw.get('variant', 'pts'), kw.get('sat', 255), kw.get('back', (3.0, 13.0)))
        gt = V[f'{k}__pca_tip']
        if e is None or not np.all(np.isfinite(gt)):
            continue
        Zgt = (V[f'{k}__T_ct'][:3, :3] @ gt + V[f'{k}__T_ct'][:3, 3])[2]
        Wv[k] = e['w'] * Zgt / V[f'{k}__K'][0, 0] * 1000
    per_view, per_run, kept = [], [], 0
    for run in runs:
        W = np.median([w for kk, w in Wv.items() if not kk.startswith(run)])
        ests = [(k, rgb_inhand(k, W, **kw)) for k in d405 if k.startswith(run)]
        good = [(k, e) for k, e in ests if e is not None]
        kept += len(good)
        per_view += [score(e, k) for k, e in good]
        if good:
            F = fuse([e for _, e in good])
            per_run.append((run, score(F, good[0][0]), len(good), len(ests)))
    Wall = np.median(list(Wv.values()))
    d435 = [rgb_inhand(k, Wall, **kw) for k in KEYS if k.startswith('180404')]
    d435 = [e for e in d435 if e is not None]
    return per_view, per_run, kept, len(d405), Wall, d435
