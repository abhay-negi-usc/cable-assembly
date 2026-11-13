"""THE SOCKET, RE-MEASURED AT THE STAND-OFF BY ITS LANDMARKS (the user, 2026-11-13: "the offline pose estimation of
the socket appears to be more accurate than the online ... add that in as an optional feature, default on").

WHY. The socket's 6D pose is estimated from ~250 mm, from the front, and carried to the stand-off through the arm
and the hand-eye. On the BNC run of 2026-11-12 (bnc_assembly_20261112_230717) that pose, drawn into the wrist
camera's frames at the socket, sat 3.2-3.9 mm from where the screws show the socket to be, in every frame of
four attempts -- ~2.9 mm along the socket's axis (the direction a frontal view fixes least) and ~1.7 mm across
it -- while it fitted the views it was made from. The plug then met the barrel's rim 2-3 mm off its axis. What matters to the
insertion is where the socket is IN THE VIEW IT IS ENTERED FROM, and there the flange shows three things a pixel
can be put on: the two M3 socket head cap screws that hold it in its mount (their hex sockets: dark, ringed by the
head's bright top) and -- the BNC's -- the barrel's rim.

WHAT. fit(): the screw heads are looked for near where the 6D pose puts them (a dark hex socket with the brighter
head top round it on nearly every side), the pair whose places agree with the model is taken, and the socket's
TRANSLATION is solved from them -- three unknowns, the 6D pose's orientation HELD (several views fix that far
better than one close one; a single close view's own 6D estimates turned the axis 7-9 deg that day). The barrel's
rim, where it shows and agrees, joins them. On that run's 249 recorded frames: 248 fitted (0.6 px), the socket's
place the same within ~0.3 mm frame to frame and ~1 mm across the four attempts' different arm poses. No GPU,
under 0.1 s a frame.

The range comes from the screws' spacing (30.6 mm, ~120 px at the stand-off: ~1.5 mm of range a pixel), so along
the line of sight it is the weakest -- still several times better than what it replaces.

WHAT IT IS NOT. Not an estimate of the socket's orientation, and not of the plug in the hand: a plug held 10 mm
from where the belief has it misses a perfectly placed socket all the same.

The landmarks are in each socket's CAD frame, mm -- the estimator repo's prepare_models.py FLANGE_SCREWS (the head
tops 3.0 mm over where the heads sit) and, the USB's, its SHIFT. Only the BNC's have been seen in images.
"""

import numpy as np

HEAD_D_MM, HEX_AF_MM = 5.5, 2.5                     # an M3 socket head cap screw's head (ISO 4762)

# part (pose6d.SOCKET, RJ45_SOCKET, USB_SOCKET) -> the screw heads' TOP centres, and the BNC barrel's rim
# `verified`: the landmarks have been found in real frames where the model puts them (the BNC's: 248 of 249
# frames of the 2026-11-12 run, four attempts, the socket's place the same within ~1 mm across them). The others'
# are their CAD's: measured and drawn, and applied only when the config says so (standoff_landmarks.apply).
LANDMARKS = {
    'socket': {'screws': [(-9.5, 12.001, 8.37), (9.5, -12.001, 8.37)],
               'rim': {'z': 12.14, 'r_out': 4.82, 'r_in': 4.13}, 'verified': True},
    'rj45_socket': {'screws': [(-9.5, 12.014, 18.0), (9.5, -11.989, 18.0)], 'verified': False},
    'usb_socket': {'screws': [(-9.5, 11.502, 3.11), (9.5, -12.502, 3.11)], 'verified': False},
}


def landmarks_for(part):
    """The landmarks of socket `part`, or None when it has none."""
    return LANDMARKS.get(str(part))


# ------------------------------------------------------------------------------------------------ geometry
def _project(K, T, pts_m):
    p = np.asarray(pts_m, dtype=float) @ T[:3, :3].T + T[:3, 3]
    return np.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], axis=1)


def _circle_ellipse(K, T, centre_mm, radius_mm, n=72):
    """The image ellipse of a circle about the part's z axis direction: (centre px, a px, b px, angle rad)."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c = np.asarray(centre_mm, dtype=float)
    pts = (c[None, :] + radius_mm * np.stack([np.cos(a), np.sin(a), np.zeros(n)], axis=1)) / 1000.0
    uv = _project(K, T, pts)
    mu = _project(K, T, c[None, :] / 1000.0)[0]
    w, v = np.linalg.eigh(np.cov((uv - uv.mean(axis=0)).T))
    return mu, float(np.sqrt(2.0 * w[1])), float(np.sqrt(2.0 * w[0])), float(np.arctan2(v[1, 1], v[0, 1]))


def _ellipse_grid(a, b, ang, margin=1.45):
    """Kernel pixel offsets and their place in the ellipse's own frame: (h, rho, theta)."""
    h = int(np.ceil(margin * a)) + 1
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    ca, sa = np.cos(ang), np.sin(ang)
    x, y = (ca * xx + sa * yy) / a, (-sa * xx + ca * yy) / max(b, 1e-6)
    return h, np.hypot(x, y), np.arctan2(y, x)


def _solve(K, T0, pts_m, meas_px):
    """The translation (m, the camera frame) that puts the model's `pts_m` on `meas_px`, T0's rotation held
    -> (t, rms residual px)."""
    from scipy.optimize import least_squares
    pts_m, meas_px = np.asarray(pts_m, dtype=float), np.asarray(meas_px, dtype=float)

    def res(t):
        T = T0.copy()
        T[:3, 3] = T0[:3, 3] + t / 1000.0
        return (_project(K, T, pts_m) - meas_px).ravel()
    sol = least_squares(res, np.zeros(3), x_scale=1.0)
    r = res(sol.x).reshape(-1, 2)
    return sol.x / 1000.0, float(np.sqrt(np.mean(np.sum(r * r, axis=1))))


# ------------------------------------------------------------------------------------------------ the screws
def _head_response(gray, K, T, top_mm, search_px, sectors=8, keep=2, lit=4, outside_min=-5.0):
    """Where in a window about a screw head's predicted place there is a dark hex socket with the brighter head top
    round it, itself brighter than what lies round the head. -> (response map over the window, its (x0, y0),
    predicted centre px). The response at a pixel: the head-top ring's mean brightness less the socket's, sector by
    sector, the `keep`-th lowest of them -- so the ring must be brighter than the socket on all but `keep` of its
    `sectors` (a highlight or a shadow may take those), which an edge or a lone dark spot beside something bright
    is not. AND THE RING MUST STAND OUT FROM ITS SURROUNDINGS on at least `lit` sectors (ring less the band just
    outside the head >= `outside_min` there), or the response is 0: a dark spot on a bright face -- a shadow on the
    fixture, a gap beside a highlight -- has a brighter ring too, and brighter still beyond it (2026-11-12 run,
    first attempt: two such, 25 px off, sat as far apart as the screws and beat them on contrast; ring over
    surroundings on 0 and 1 sectors of 8 there, on 4-7 at every real head)."""
    import cv2
    mu, a, b, ang = _circle_ellipse(K, T, top_mm, HEAD_D_MM / 2.0)
    h, rho, th = _ellipse_grid(a, b, ang, margin=1.5)
    s = int(np.ceil(search_px))
    x0, y0 = int(round(mu[0])) - s - h, int(round(mu[1])) - s - h
    x1, y1 = x0 + 2 * (s + h) + 1, y0 + 2 * (s + h) + 1
    H, W = gray.shape
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None, (x0, y0), mu
    roi = gray[y0:y1, x0:x1]
    hole = (rho <= 0.40).astype(np.float32)
    if hole.sum() < 4:
        return None, (x0, y0), mu

    def mean(mask):
        return cv2.filter2D(roi, -1, mask / mask.sum(), borderType=cv2.BORDER_REPLICATE)
    m_hole = mean(hole)
    ring, outside = (rho >= 0.60) & (rho <= 0.92), (rho >= 1.12) & (rho <= 1.45)
    inner, outer = [], []
    for k in range(sectors):
        lo = -np.pi + 2.0 * np.pi * k / sectors
        sec = (th >= lo) & (th < lo + 2.0 * np.pi / sectors)
        mr, mo = (ring & sec).astype(np.float32), (outside & sec).astype(np.float32)
        if mr.sum() < 2 or mo.sum() < 2:
            continue
        m_ring = mean(mr)
        inner.append(m_ring - m_hole)
        outer.append(m_ring - mean(mo))
    if len(inner) <= max(keep, lit):
        return None, (x0, y0), mu
    resp = np.sort(np.stack(inner), axis=0)[keep]
    stands_out = np.sort(np.stack(outer), axis=0)[len(outer) - lit] >= outside_min
    resp = np.where(stands_out, resp, 0.0).astype(np.float32)
    return resp[h:-h, h:-h], (x0 + h, y0 + h), mu


def _peaks(resp, origin, min_contrast, n=6, nms=4):
    """The window's `n` strongest local maxima over `min_contrast` -> [(u, v, response)], each refined to a
    fraction of a pixel by the response's centroid about it."""
    import cv2
    out = []
    if resp is None:
        return out
    k = 2 * nms + 1
    peak = (resp >= cv2.dilate(resp, np.ones((k, k), np.uint8))) & (resp >= min_contrast)
    ys, xs = np.nonzero(peak)
    for i in np.argsort(-resp[ys, xs])[:n]:
        y, x = int(ys[i]), int(xs[i])
        ya, yb, xa, xb = max(0, y - 2), min(resp.shape[0], y + 3), max(0, x - 2), min(resp.shape[1], x + 3)
        w = np.clip(resp[ya:yb, xa:xb] - 0.5 * resp[y, x], 0.0, None)
        gy, gx = np.mgrid[ya:yb, xa:xb]
        sw = float(w.sum())
        cx, cy = (float((w * gx).sum() / sw), float((w * gy).sum() / sw)) if sw > 0 else (float(x), float(y))
        out.append((origin[0] + cx, origin[1] + cy, float(resp[y, x])))
    return out


# ------------------------------------------------------------------------------------------------ the rim
def _rim(gray, K, T, rim, hidden=None, search_px=5, sectors=16, min_arc=0.35):
    """The barrel's rim near where `T` puts it: the shift (px) of its ellipse at which the rim's bright band stands
    out most from what lies just outside it, over the sectors that are in view -> (centre px, contrast, the
    fraction of the rim used) or None. The far side is the plug's to hide and the bezel's to light up; a sector
    counts only if the rim is brighter than its outside there."""
    centre = (0.0, 0.0, rim['z'])
    mu, a, b, ang = _circle_ellipse(K, T, centre, rim['r_out'])
    if b < 3.0:
        return None
    h, rho, th = _ellipse_grid(a, b, ang, margin=1.45)
    s = int(search_px)
    H, W = gray.shape
    cx, cy = int(round(mu[0])), int(round(mu[1]))
    if cx - s - h < 0 or cy - s - h < 0 or cx + s + h + 1 > W or cy + s + h + 1 > H:
        return None
    band = (rho >= rim['r_in'] / rim['r_out'] - 0.03) & (rho <= 1.02)
    outside = (rho >= 1.14) & (rho <= 1.42)
    sec = np.floor((th + np.pi) / (2.0 * np.pi) * sectors).astype(int) % sectors
    best = None
    scores = np.full((2 * s + 1, 2 * s + 1), -np.inf)
    for dv in range(-s, s + 1):
        for du in range(-s, s + 1):
            win = gray[cy + dv - h:cy + dv + h + 1, cx + du - h:cx + du + h + 1]
            hid = None if hidden is None else hidden[cy + dv - h:cy + dv + h + 1, cx + du - h:cx + du + h + 1]
            con = []
            for k in range(sectors):
                mb, mo = band & (sec == k), outside & (sec == k)
                if hid is not None and (hid[mb].any() or hid[mo].any()):
                    continue
                if mb.sum() < 2 or mo.sum() < 2:
                    continue
                con.append(float(win[mb].mean() - win[mo].mean()))
            pos = [c for c in con if c > 0.0]
            if len(pos) < min_arc * sectors:
                continue
            sc = float(np.sum(pos)) / sectors                 # brighter-than-outside, over the whole rim
            scores[dv + s, du + s] = sc
            if best is None or sc > best[0]:
                best = (sc, du, dv, len(pos) / sectors)
    if best is None:
        return None
    sc, du, dv, arc = best
    if abs(du) == s or abs(dv) == s:                          # at the window's edge: not a peak
        return None
    # a fraction of a pixel from the scores about the peak
    w = scores[dv + s - 1:dv + s + 2, du + s - 1:du + s + 2]
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w - 0.6 * sc, 0.0, None)
    gy, gx = np.mgrid[-1:2, -1:2]
    fu, fv = (float((w * gx).sum() / w.sum()), float((w * gy).sum() / w.sum())) if w.sum() > 0 else (0.0, 0.0)
    return (cx + du + fu, cy + dv + fv), sc, arc             # the masks were laid about the whole pixel (cx, cy)


# ------------------------------------------------------------------------------------------------ one frame
def fit(img, K, T_cam_obj, part='socket', search_mm=8.0, min_contrast=10.0, max_residual_px=2.0, hidden=None,
        use_rim=True, rim_agree_px=2.5, ambiguous_mm=2.0, ambiguous_ratio=0.8, outside_min=-5.0):
    """The socket's translation from its landmarks in one colour image (BGR or grey), T_cam_obj's rotation held.

    `T_cam_obj` (m): the socket where the 6D estimate puts it, in this camera. `hidden`: HxW bool, True where
    something stands in front (the held plug): a landmark predicted there is not looked for. `search_mm`: how far
    from the 6D pose's the socket may be -- each head is looked for within that of its predicted place in the
    image, and the correction (range included) may be no longer. Kept tight on purpose: on the 2026-11-12 frames a
    shadow on the fixture and the plug's own slot, 35 px (9 mm) from the two heads and as far apart as they are,
    out-scored them wherever they were let in.
    -> {'ok', 'why', 't_cam_mm' (the correction, the camera frame), 'T_cam_obj' (corrected), 'residual_px',
        'screws': [{'pred', 'px', 'contrast'} ...], 'rim': {...} or None, 'range_mm'}"""
    import cv2
    spec = landmarks_for(part)
    out = {'ok': False, 'why': '', 't_cam_mm': None, 'T_cam_obj': None, 'residual_px': None, 'screws': [],
           'rim': None, 'part': part}
    if spec is None:
        out['why'] = f'no landmarks for {part}'
        return out
    K = np.asarray(K, dtype=float)
    T0 = np.asarray(T_cam_obj, dtype=float)
    g = np.asarray(img)
    gray = (cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) if g.ndim == 3 else g).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    tops = np.asarray(spec['screws'], dtype=float)
    z = float((T0[:3, :3] @ (tops.mean(axis=0) / 1000.0) + T0[:3, 3])[2])
    if not np.isfinite(z) or z <= 0.03:
        out['why'] = 'the socket is not in front of the camera'
        return out
    search_px = float(search_mm) * K[0, 0] / (1000.0 * z)
    cands = []
    for top in tops:
        resp, origin, mu = _head_response(gray, K, T0, top, search_px, outside_min=outside_min)
        hid = hidden is not None and 0 <= int(mu[1]) < hidden.shape[0] and 0 <= int(mu[0]) < hidden.shape[1] \
            and bool(hidden[int(mu[1]), int(mu[0])])
        pk = [] if hid else _peaks(resp, origin, float(min_contrast))
        pk = [q for q in pk if np.hypot(q[0] - mu[0], q[1] - mu[1]) <= search_px]       # a disc, not its square
        if hidden is not None:
            pk = [p for p in pk if not hidden[int(np.clip(p[1], 0, hidden.shape[0] - 1)),
                                              int(np.clip(p[0], 0, hidden.shape[1] - 1))]]
        cands.append(pk)
        out['screws'].append({'pred': [float(mu[0]), float(mu[1])], 'px': None, 'contrast': None,
                              'hidden': bool(hid), 'candidates': len(pk),
                              'in_image': resp is not None})
    if any(not c for c in cands):
        k = [i for i, c in enumerate(cands) if not c]
        s = out['screws'][k[0]]
        out['why'] = ('screw head %d %s' % (k[0], 'is behind the held plug' if s['hidden'] else
                                            'is out of the image' if not s['in_image'] else
                                            'was not found (no dark socket ringed by a brighter head within '
                                            '%.0f mm of where the 6D pose puts it)' % search_mm))
        return out
    # THE PAIR: every two candidates that sit as the model's two heads do under a translation alone; of those, the
    # one that looks most like two heads (the contrasts' sum). A head gives a second, weaker candidate ~8 px off
    # its socket (its lit far wall and the shadow at its foot), and the two heads' second candidates pair up as
    # well as their first -- ~3 mm away, at 45-70% of the contrast on the 2026-11-12 frames. Another pair nearly as
    # good somewhere else makes the frame ambiguous: no answer is better than a choice between them.
    pairs = []
    for p0 in cands[0]:
        for p1 in cands[1]:
            t, rms = _solve(K, T0, tops / 1000.0, [p0[:2], p1[:2]])
            if rms > max_residual_px or np.linalg.norm(t) * 1000.0 > search_mm:
                continue
            pairs.append((p0[2] + p1[2], p0, p1, t, rms))
    if not pairs:
        out['why'] = ('the screw heads found do not sit as the model\'s do (no pair within %.1f px of the held '
                      'orientation\'s)' % max_residual_px)
        return out
    pairs.sort(key=lambda p: -p[0])
    c_best, p0, p1, t, rms = pairs[0]
    for c, _q0, _q1, tq, _r in pairs[1:]:
        if np.linalg.norm(tq - t) * 1000.0 > ambiguous_mm and c >= ambiguous_ratio * c_best:
            out['why'] = ('two places fit the screw heads, %.1f mm apart (contrast %.0f and %.0f)'
                          % (np.linalg.norm(tq - t) * 1000.0, c_best, c))
            return out
    pts, meas = [tops[0] / 1000.0, tops[1] / 1000.0], [p0[:2], p1[:2]]
    for s, p in zip(out['screws'], (p0, p1)):
        s.update(px=[float(p[0]), float(p[1])], contrast=float(p[2]))
    T1 = T0.copy()
    T1[:3, 3] = T0[:3, 3] + t
    if use_rim and spec.get('rim'):
        rim = spec['rim']
        got = _rim(gray, K, T1, rim, hidden=hidden)
        if got is not None:
            (ru, rv), rc, arc = got
            pred = _project(K, T1, np.array([[0.0, 0.0, rim['z'] / 1000.0]]))[0]
            off = float(np.hypot(ru - pred[0], rv - pred[1]))
            used = rc >= min_contrast and off <= rim_agree_px
            out['rim'] = {'px': [float(ru), float(rv)], 'contrast': float(rc), 'arc': float(arc),
                          'from_screws_px': off, 'used': bool(used)}
            if used:
                t3, rms3 = _solve(K, T0, pts + [np.array([0.0, 0.0, rim['z'] / 1000.0])], meas + [(ru, rv)])
                if rms3 <= max_residual_px:
                    t, rms = t3, rms3
                    T1[:3, 3] = T0[:3, 3] + t
                else:
                    out['rim']['used'] = False
    out.update(ok=True, t_cam_mm=(t * 1000.0).tolist(), T_cam_obj=T1, residual_px=rms,
               range_mm=float(1000.0 * (T1[:3, :3] @ (tops.mean(axis=0) / 1000.0) + T1[:3, 3])[2]))
    return out


# ------------------------------------------------------------------------------------------------ several frames
def fuse(results, agree_mm=1.0, min_ok=None):
    """One correction from several frames' (the arm still): the median of those that fitted, if enough did and
    they agree -> {'ok', 'why', 't_cam_mm', 'spread_mm', 'n', 'of', 'residual_px'}."""
    ok = [r for r in results if r and r.get('ok')]
    need = max(1, (len(results) + 1) // 2) if min_ok is None else int(min_ok)
    out = {'ok': False, 'why': '', 't_cam_mm': None, 'spread_mm': None, 'n': len(ok), 'of': len(results),
           'residual_px': None, 'rim_used': sum(1 for r in ok if (r.get('rim') or {}).get('used'))}
    if len(ok) < need:
        why = next((r.get('why') for r in results if r and not r.get('ok') and r.get('why')), 'no frame')
        out['why'] = '%d of %d frames fitted (need %d): %s' % (len(ok), len(results), need, why)
        return out
    t = np.array([r['t_cam_mm'] for r in ok], dtype=float)
    med = np.median(t, axis=0)
    spread = float(np.max(np.linalg.norm(t - med, axis=1)))
    out.update(t_cam_mm=med.tolist(), spread_mm=spread,
               residual_px=float(np.median([r['residual_px'] for r in ok])))
    if spread > agree_mm:
        out['why'] = 'the frames disagree by %.1f mm (over %.1f)' % (spread, agree_mm)
        return out
    out['ok'] = True
    return out


def measure(images, K, T_base_cams, T_base_obj, part='socket', hidden=None, agree_mm=1.0, **fit_kw):
    """THE STAND-OFF MEASUREMENT: `images` taken with the arm still (their camera poses `T_base_cams`), each fitted
    from where `T_base_obj` (the 6D pose, m) puts the socket, and fused.

    -> {'ok', 'why', 'shift_m' (the socket's translation correction, the base frame), 'along_m' (its part along
        the socket's own axis, + out of the socket toward the plug), 'across_m', 'T_base_obj' (corrected: the
        orientation unchanged), 'n', 'of', 'spread_mm', 'residual_px', 'rim_used', 'frames': each image's fit}"""
    T_base_obj = np.asarray(T_base_obj, dtype=float)
    fits = []
    for k, img in enumerate(images):
        if img is None:
            fits.append(None)
            continue
        T_bc = np.asarray(T_base_cams[k], dtype=float)
        r = fit(img, K, np.linalg.inv(T_bc) @ T_base_obj, part=part,
                hidden=None if hidden is None else hidden[k], **fit_kw)
        r['T_base_cam'] = T_bc
        fits.append(r)
    f = fuse(fits, agree_mm=agree_mm)
    out = dict(f, frames=fits, shift_m=None, along_m=None, across_m=None, T_base_obj=None, part=part)
    if not f['ok']:
        return out
    # each frame's correction carried to the base frame by its own camera pose, then the median
    d = np.median([r['T_base_cam'][:3, :3] @ (np.asarray(r['t_cam_mm']) / 1000.0) for r in fits if r and r['ok']],
                  axis=0)
    axis = T_base_obj[:3, 2]
    along = float(d @ axis)
    T_new = T_base_obj.copy()
    T_new[:3, 3] = T_base_obj[:3, 3] + d
    out.update(shift_m=d, along_m=along, across_m=float(np.linalg.norm(d - along * axis)), T_base_obj=T_new)
    return out


def plug_mask(shape, K, T_cam_plug, vertices_m, grow_px=6):
    """HxW bool: where the held plug (its CAD's `vertices_m` at `T_cam_plug`) stands in the image, grown a little."""
    import cv2
    m = np.zeros(shape[:2], np.uint8)
    V = np.asarray(vertices_m, dtype=float)
    V = V[::max(1, len(V) // 4000)]
    p = V @ T_cam_plug[:3, :3].T + T_cam_plug[:3, 3]
    p = p[p[:, 2] > 0.02]
    if len(p) < 3:
        return m.astype(bool)
    uv = np.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], axis=1)
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    if grow_px > 0:
        m = cv2.dilate(m, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8))
    return m.astype(bool)


def draw(img, K, T_cam_obj, res, part='socket', zoom=4, pad_px=70):
    """`img` about the socket, enlarged: the landmarks where the 6D pose puts them (red), where they were found
    (green crosses) and where the corrected pose puts them (yellow). None when nothing can be drawn."""
    import cv2
    spec = landmarks_for(part)
    if spec is None or img is None:
        return None
    K = np.asarray(K, dtype=float)
    T0 = np.asarray(T_cam_obj, dtype=float)
    tops = np.asarray(spec['screws'], dtype=float)
    c = _project(K, T0, np.array([[0.0, 0.0, tops[:, 2].mean() / 1000.0]]))[0]
    x0, y0 = int(c[0]) - pad_px, int(c[1]) - pad_px
    x0, y0 = max(0, min(img.shape[1] - 2 * pad_px, x0)), max(0, min(img.shape[0] - 2 * pad_px, y0))
    crop = np.ascontiguousarray(img[y0:y0 + 2 * pad_px, x0:x0 + 2 * pad_px])
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)

    def pt(uv):
        return int(round((uv[0] - x0) * zoom)), int(round((uv[1] - y0) * zoom))

    def model(T, colour):
        shapes = [(top, HEAD_D_MM / 2.0) for top in tops] + [(top, HEX_AF_MM / 2.0) for top in tops]
        if spec.get('rim'):
            shapes.append(((0.0, 0.0, spec['rim']['z']), spec['rim']['r_out']))
        for centre, r in shapes:
            a = np.linspace(0.0, 2.0 * np.pi, 60)
            pts = (np.asarray(centre)[None, :] + r * np.stack([np.cos(a), np.sin(a), 0 * a], axis=1)) / 1000.0
            uv = _project(K, T, pts)
            cv2.polylines(vis, [np.array([pt(p) for p in uv], np.int32)], True, colour, 1, cv2.LINE_AA)
    model(T0, (0, 0, 255))
    if res and res.get('ok'):
        model(np.asarray(res['T_cam_obj'], dtype=float), (0, 255, 255))
    for s in (res or {}).get('screws', []):
        if s.get('px'):
            cv2.drawMarker(vis, pt(s['px']), (0, 255, 0), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
    rim = (res or {}).get('rim')
    if rim and rim.get('px'):
        cv2.drawMarker(vis, pt(rim['px']), (0, 255, 0) if rim.get('used') else (160, 160, 160),
                       cv2.MARKER_TILTED_CROSS, 14, 1, cv2.LINE_AA)
    if res and res.get('ok'):
        t = res['t_cam_mm']
        txt = 'moved %.1f mm (%+.1f %+.1f %+.1f, camera), %.1f px' % (np.linalg.norm(t), t[0], t[1], t[2],
                                                                       res['residual_px'])
    else:
        txt = 'not fitted: ' + str((res or {}).get('why', ''))[:70]
    cv2.putText(vis, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
