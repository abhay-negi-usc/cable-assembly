"""THE BNC PLUG'S SLOT ROLL, FROM THE WRIST CAMERA (the user, 2026-11-06: "see if you can determine the connector's
roll orientation from in-hand images"; then "using the slot estimate the connector roll orientation, but still do
connector clocking to account for errors (with the idea that it requires less clocking)").

WHAT IS SEEN. The bayonet sleeve carries two J-slots 180 deg apart: an axial ENTRY open toward the mating face and a
helix running back from it about 110 deg round. The wrist camera sees the held plug from behind at ~35 deg off its
axis, so the slotted band of the sleeve -- the neck, between the front lip and the knurl -- is a narrow crescent,
~700 px at 1280x720. With the socket behind the plug (the stand-off, contact) a slot reads DARK. The entry is the
clearest of it: a dark stripe down the whole neck. The camera rides with the plug, so the view never changes while
it is held -- one roll per grasp, measurable whenever the arm stands still.

THE MODEL (SlotModel). The plug's CAD unwrapped onto (z, phi): the outer radius at each height and angle. Its
per-height maximum is the SOLID -- the plug as a body of revolution, the slots filled. The contiguous rows nearest
the mating face where some angles fall well under the solid are the slotted NECK; those angles there are the SLOT
MAP. The NIST CAD draws the helix of ONE slot only (the user, 2026-11-06: "it is actually symmetric on both
sides"), so with `symmetric` the largest cut -- the full J -- is copied 180 deg round. `entry_deg` is where that
J's entry is, in the CAD's own angle (90).

THE MEASUREMENT (estimate). Each pixel near the plug is ray-marched against the solid (march): where it lands --
height, angle, radius. The landing-height outline (lip, neck, knurl) and the silhouette are aligned to the image's
edges by a shift (gradient across each edge; the silhouette weighted -- the rings alone repeat ~5 px apart), which
absorbs the in-hand belief's error (~8-18 px on the bench). On the neck, what is the same all round at one height
(the shadow under the lip) is taken off; then for every roll the entries' angles are scored by how much darker a
strip there is than the strips beside it. -> the roll (mod 180: the slots are symmetric), the score curve, and a
margin: the peak against the best roll more than 25 deg away, in the curve's standard deviations.

HOW GOOD (2026-11-06, the contact frames of the 12 runs that recorded them, against the entry read by eye): median
9, max 16 deg; the three frames of a run agree to 3 deg. Matching the CAD's whole J instead did NOT work -- the
real entry runs the neck's full height where the CAD's covers the top half, and every run came out 144-156 deg.
Engage/settle frames: up to 70 deg off -- pushed home, the plug tilts in the pads and the 2D alignment no longer
registers the model (the slot itself had not moved). Hence bnc_assembly aims by the stand-off capture, and
connector clocking still SEARCHES, nearest first.

    python -m cable_assembly.perception.slot_roll <trial folder>     # the recorded engage frames, re-estimated
"""

import json
import os
from datetime import datetime

import numpy as np

from .. import log as urlog

log = urlog.get('slot-roll')

DPHI_DEG = 0.5              # the unwrap's angle step
DZ_M = 1e-4                 # its height step


# ---------------------------------------------------------------------------- the model
class SlotModel:
    """The plug as a body of revolution + its slot map (module docstring). Build with from_cad()."""

    _cache = {}

    def __init__(self, V, F, mating_z=None, search_mm=12.0, cut_mm=0.3, symmetric=True):
        import cv2
        V = np.asarray(V, dtype=float)
        F = np.asarray(F, dtype=np.int64)
        self.z_min, self.z_max = float(V[:, 2].min()), float(V[:, 2].max())
        self.mating_z = float(self.z_max if mating_z is None else mating_z)
        nz = int(np.ceil((self.z_max - self.z_min) / DZ_M)) + 2
        nphi = int(round(360.0 / DPHI_DEG))
        # THE UNWRAP: every triangle drawn into (phi, z) with its mean radius, smallest first -> the outermost wins
        tri = V[F]
        r = np.hypot(tri[:, :, 0], tri[:, :, 1]).mean(axis=1)
        ph = np.degrees(np.arctan2(tri[:, :, 1], tri[:, :, 0])) % 360.0
        wrap = ph.max(axis=1) - ph.min(axis=1) > 180.0
        ph[wrap] = np.where(ph[wrap] < 180.0, ph[wrap] + 360.0, ph[wrap])
        u = ph / DPHI_DEG
        v = (tri[:, :, 2] - self.z_min) / DZ_M
        pts = np.round(np.stack([u, v], axis=2) * 16).astype(np.int32)
        U = np.zeros((nz, 2 * nphi), np.float32)
        q = np.round(r / 1e-5).astype(np.int64)                       # 0.01 mm radius layers
        order = np.argsort(q, kind='stable')
        pts, q = pts[order], q[order]
        cuts = np.r_[0, np.nonzero(np.diff(q))[0] + 1, len(q)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            cv2.fillPoly(U, list(pts[a:b]), float(q[a] * 1e-5), lineType=cv2.LINE_8, shift=4)
        U = np.maximum(U[:, :nphi], U[:, nphi:])                      # what crossed 360 folded back
        self.U = U
        self.R = U.max(axis=1)                                        # the SOLID: radius per height
        # THE NECK: the rows where some angles are cut well under the solid -- the CONTIGUOUS run of them nearest
        # the mating face (the knurl behind it is cut too, by its grooves, past an uncut step)
        zs = self.z_min + np.arange(nz) * DZ_M
        cut = U < (self.R[:, None] - cut_mm / 1000.0)
        frac = cut.mean(axis=1)
        cand = ((zs >= self.mating_z - search_mm / 1000.0) & (zs <= self.mating_z)
                & (frac > 0.005) & (frac < 0.7) & (self.R > 0))
        if not cand.any():
            raise ValueError('no slotted band within %.0f mm of the mating face in this CAD' % search_mm)
        hi = int(np.nonzero(cand)[0].max())
        lo = hi
        while lo - 1 >= 0 and cand[lo - 1]:
            lo -= 1
        rows = np.arange(lo, hi + 1)
        self.neck = (float(zs[rows.min()]), float(zs[rows.max()]))
        self.r_neck = float(np.median(self.R[rows]))
        S = np.zeros_like(cut)
        S[rows.min():rows.max() + 1] = cut[rows.min():rows.max() + 1]
        # THE FULL J, and (symmetric) its copy 180 deg round. Components found with the map rolled so that an
        # uncut column is the seam -- nothing then crosses 0/360.
        free = np.nonzero(~S.any(axis=0))[0]
        seam = int(free[0]) if len(free) else 0
        Sr = np.roll(S, -seam, axis=1).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(Sr, connectivity=8)
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
        J = np.roll(lab == big, seam, axis=1) if big else S
        self.slot = (J | np.roll(J, nphi // 2, axis=1)) if symmetric else S
        # THE ENTRY: where the J reaches the top of the neck (toward the mating face)
        top = J[max(rows.max() - 5, 0):rows.max() + 1].any(axis=0)
        a = np.radians(np.nonzero(top)[0] * DPHI_DEG)
        self.entry_deg = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0) if len(a) else 0.0
        self.symmetric = bool(symmetric)
        # the solid's constant-radius runs, for the outline the alignment matches
        self.seg = np.cumsum(np.r_[0, np.abs(np.diff(self.R)) > 3e-4]).astype(np.int32)

    @classmethod
    def from_cad(cls, path, scale, mating_z=None, symmetric=True):
        key = (path, float(scale), mating_z, bool(symmetric))
        if key not in cls._cache:
            from .engage_vision import load_mesh
            V, F = load_mesh(path, scale)
            cls._cache[key] = cls(V, F, mating_z=mating_z, symmetric=symmetric)
        return cls._cache[key]

    def iz(self, z):
        return np.clip(np.round((np.asarray(z) - self.z_min) / DZ_M).astype(int), 0, len(self.R) - 1)

    def window(self, z, phi_deg, roll_deg):
        """Is (z, phi) -- the plug's own angle -- a slot, the CAD turned by `roll_deg` about the plug's z?"""
        j = np.round(((np.asarray(phi_deg) - roll_deg) % 360.0) / DPHI_DEG).astype(int) % self.slot.shape[1]
        return self.slot[self.iz(z), j]

    def summary(self):
        return {'neck_z_mm': [round(1000 * self.neck[0], 2), round(1000 * self.neck[1], 2)],
                'r_neck_mm': round(1000 * self.r_neck, 3), 'entry_deg': round(self.entry_deg, 2),
                'symmetric': self.symmetric, 'slot_cells': int(self.slot.sum())}


# ---------------------------------------------------------------------------- the view
def _box(model, K, T_cam_plug, shape, pad):
    """The pixels the front of the plug (knurl to mating face) can cover, padded."""
    zs = np.linspace(model.neck[0] - 0.006, model.mating_z, 25)
    ph = np.radians(np.arange(0, 360, 10))
    r = np.maximum(model.R[model.iz(zs)], model.r_neck)
    P = np.stack([np.outer(r, np.cos(ph)), np.outer(r, np.sin(ph)), np.repeat(zs[:, None], len(ph), 1)], -1)
    P = P.reshape(-1, 3) @ T_cam_plug[:3, :3].T + T_cam_plug[:3, 3]
    P = P[P[:, 2] > 0.02]
    if len(P) == 0:
        return None
    u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
    v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
    x0, y0 = int(np.floor(u.min())) - pad, int(np.floor(v.min())) - pad
    x1, y1 = int(np.ceil(u.max())) + pad, int(np.ceil(v.max())) + pad
    h, w = shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None


def march(model, K, T_cam_plug, box, n=256, r_max=None):
    """Each pixel of `box` ray-marched against the solid: (z, phi deg, r) where it lands, NaN where it misses."""
    x0, y0, x1, y1 = box
    us, vs = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    d_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    R, t = np.asarray(T_cam_plug)[:3, :3], np.asarray(T_cam_plug)[:3, 3]
    o = -R.T @ t                                           # the camera centre, plug frame
    d = d_cam @ R
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    rm = float(model.R.max() + 2e-4) if r_max is None else float(r_max)
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2.0 * (o[0] * d[..., 0] + o[1] * d[..., 1])
    c = o[0] ** 2 + o[1] ** 2 - rm ** 2
    disc = b * b - 4.0 * a * c
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t0 = (-b - sq) / (2.0 * a + 1e-18)
    t1 = (-b + sq) / (2.0 * a + 1e-18)
    s = np.linspace(0.0, 1.0, n)
    ts = t0[..., None] + (t1 - t0)[..., None] * s
    P = o + ts[..., None] * d[..., None, :]
    rr = np.hypot(P[..., 0], P[..., 1])
    zz = P[..., 2]
    inside = ((rr <= model.R[model.iz(zz)] + 1e-5) & (zz >= model.z_min) & (zz <= model.z_max)
              & ok[..., None] & (ts > 0))
    hit = inside.any(-1)
    k = np.argmax(inside, -1)
    Ph = np.take_along_axis(P, k[..., None, None].repeat(3, -1), -2)[..., 0, :]
    nan = np.full(hit.shape, np.nan)
    return (np.where(hit, Ph[..., 2], nan), np.where(hit, np.degrees(np.arctan2(Ph[..., 1], Ph[..., 0])) % 360, nan),
            np.where(hit, np.hypot(Ph[..., 0], Ph[..., 1]), nan))


FRONT_MM = 6.0             # the outline aligned: the knurl and forward (behind it the real plug is boot and fingers)
ZBAND_M = 3e-4             # the neck's height bands, each with its own median taken off before matching
SILHOUETTE_W = 6.0         # the silhouette's weight in it against the inner outlines (2026-11-06: the rings -- rim,
                           # lip/neck, neck/knurl -- repeat ~5 px apart along the axis and alone gave peaks a ring
                           # out; the plug against what is behind it is the one edge that does not repeat)


def _edges(model, zh):
    """The outline the alignment matches: where the landing height changes run (lip / neck / knurl), and the
    silhouette of the front -- what lands further back counts as background."""
    z = np.nan_to_num(zh, nan=model.z_min)
    front = np.isfinite(zh) & (z >= model.neck[0] - FRONT_MM / 1000.0)
    lab = np.where(front, model.seg[model.iz(z)] + 1, 0)
    e = np.zeros(lab.shape, bool)
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    e[1:, :] |= lab[1:, :] != lab[:-1, :]
    # each edge's normal, from the labels smoothed: the alignment scores only the image gradient ACROSS the
    # outline -- the knurl's stripes run along it and otherwise pull the crescent down onto the knurl
    import cv2
    f = cv2.GaussianBlur(lab.astype(np.float32), (5, 5), 0)
    nx, ny = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
    nn = np.hypot(nx, ny) + 1e-9
    hit = np.isfinite(zh).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    w = np.where((cv2.dilate(hit, k) != cv2.erode(hit, k)) & e, SILHOUETTE_W, 1.0)
    return e, nx / nn, ny / nn, w


def estimate(img, K, T_cam_plug, model, roll_step_deg=3.0, shift_px=26, half_width_deg=9.0, ring_deg=28.0,
             min_height=0.6):
    """The slot roll in one BGR image (module docstring) -> dict, or None when the plug's front is not in view.

    roll_deg: the CAD turned by this about the plug's z puts its entries on the darkest axial stripes (mod 180 when
    symmetric); entry_deg: the entry's angle in the plug's frame then; margin: peak vs the best roll >25 deg away,
    in the curve's standard deviations; curve: the score per roll (z-scored; rolls whose entries are out of view
    score the curve's floor); shift_px: px the outline was moved by to meet the image."""
    import cv2
    K = np.asarray(K, dtype=float)
    T_cam_plug = np.asarray(T_cam_plug, dtype=float)
    box = _box(model, K, T_cam_plug, img.shape, int(shift_px + 4))
    if box is None:
        return None
    x0, y0, x1, y1 = box
    zh, phh, rh = march(model, K, T_cam_plug, box)
    neck = (np.isfinite(zh) & (np.nan_to_num(zh) >= model.neck[0]) & (np.nan_to_num(zh) <= model.neck[1])
            & (np.abs(np.nan_to_num(rh) - model.r_neck) < 3e-4))
    if neck.sum() < 80:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    H, W = gray.shape
    # THE OUTLINE ALIGNED: the shift whose model edges sit on the most image gradient ACROSS them
    e, enx, eny, ew = _edges(model, zh)
    ey, ex = np.nonzero(e)
    nx_, ny_, w_ = enx[ey, ex], eny[ey, ex], ew[ey, ex]
    ey, ex = ey + y0, ex + x0
    best = (-1.0, 0, 0)
    for dy in range(-shift_px, shift_px + 1):
        yy = ey + dy
        for dx in range(-shift_px, shift_px + 1):
            xx = ex + dx
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if ok.sum() < 0.8 * len(ey):
                continue
            s = float((w_[ok] * np.abs(gx[yy[ok], xx[ok]] * nx_[ok] + gy[yy[ok], xx[ok]] * ny_[ok])).sum()
                      / w_[ok].sum())
            if s > best[0]:
                best = (s, dx, dy)
    _, sdx, sdy = best
    ny, nx = np.nonzero(neck)
    yy, xx = ny + y0 + sdy, nx + x0 + sdx
    keep = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    ny, nx, yy, xx = ny[keep], nx[keep], yy[keep], xx[keep]
    z, ph, g = zh[ny, nx], phh[ny, nx], gray[yy, xx].astype(float)
    # WHAT IS THE SAME ALL ROUND AT ONE HEIGHT taken off -- the shadow and the reflection under the lip
    zb = ((z - model.neck[0]) / ZBAND_M).astype(int)
    for bnd in np.unique(zb):
        g[zb == bnd] -= np.median(g[zb == bnd])
    sd = float(np.std(g)) + 1e-6
    # THE ENTRY: a DARK AXIAL STRIPE down the neck (2026-11-06: on the bench the entry is the one feature that reads
    # the same at contact as at the stand-off -- the helix as drawn in the CAD does not match the real one well
    # enough to match against, and the bright specular streak at the limb fooled a deviation-of-either-sign cue).
    # For each roll, each entry (the J's and its twin 180 round) is scored by how much darker a +-half_width strip
    # at its angle is than the strips beside it -- if it is seen down min_height of the neck.
    rolls = np.arange(0.0, 180.0 if model.symmetric else 360.0, float(roll_step_deg))
    span = model.neck[1] - model.neck[0]
    entries = (0.0, 180.0) if model.symmetric else (0.0,)
    sc = np.full(len(rolls), np.nan)
    for i, r in enumerate(rolls):
        vals = []
        for off in entries:
            d = np.abs((ph - (model.entry_deg + r + off) + 180.0) % 360.0 - 180.0)
            inn = d < half_width_deg
            ring = (d >= half_width_deg + 2.0) & (d < ring_deg)
            if inn.sum() < 25 or ring.sum() < 25 or np.ptp(z[inn]) < min_height * span:
                continue
            vals.append((g[ring].mean() - g[inn].mean()) / sd)
        if vals:
            sc[i] = max(vals)
    if not np.isfinite(sc).any():
        return None
    sc = np.where(np.isfinite(sc), sc, np.nanmin(sc))
    zc = (sc - sc.mean()) / (sc.std() + 1e-9)
    res = _summarise(rolls, zc, model)
    res.update({'shift_px': [int(sdx), int(sdy)], 'box': [int(x0), int(y0), int(x1), int(y1)],
                'neck_px': int(neck.sum()), 'edge_score': round(float(best[0]), 2),
                'contrast': round(float(np.nanmax(sc)), 3), 'half_width_deg': float(half_width_deg)})
    res['_view'] = (zh, phh, neck)
    return res


def _summarise(rolls, z, model):
    i = int(np.argmax(z))
    period = 180.0 if model.symmetric else 360.0
    d = np.abs(rolls - rolls[i]) % period
    far = np.minimum(d, period - d) > 25.0
    margin = float(z[i] - z[far].max()) if far.any() else 0.0
    roll = float(rolls[i])
    return {'roll_deg': roll, 'entry_deg': float((model.entry_deg + roll) % period), 'margin': round(margin, 3),
            'peak_z': round(float(z[i]), 3), 'rolls_deg': rolls.tolist(), 'curve': np.round(z, 4).tolist(),
            'period_deg': period}


def fuse(results, model):
    """Several frames' estimates of the SAME roll (the camera rides with the plug) -> one: the z-scored curves
    summed. None in, None out."""
    rs = [r for r in results if r is not None]
    if not rs:
        return None
    rolls = np.asarray(rs[0]['rolls_deg'])
    z = np.sum([np.asarray(r['curve']) for r in rs], axis=0) / np.sqrt(len(rs))
    z = (z - z.mean()) / (z.std() + 1e-9)
    out = _summarise(rolls, z, model)
    out['n_frames'] = len(rs)
    out['frames_deg'] = [r['roll_deg'] for r in rs]
    return out


def draw(img, res, model, scale=4):
    """The plug's front, enlarged, with the neck (yellow), the entry stripes at the estimated roll (green) and the
    CAD's slots there (thin magenta) drawn where the alignment put them, and the score curve under it."""
    import cv2
    if res is None or '_view' not in res:
        return None
    zh, phh, neck = res['_view']
    x0, y0, x1, y1 = res['box']
    dx, dy = res['shift_px']
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    crop = img[y0:y1, x0:x1]
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    win = np.zeros(zh.shape, np.uint8)
    stripe = np.zeros(zh.shape, np.uint8)
    ny, nx = np.nonzero(neck)
    win[ny, nx] = model.window(zh[ny, nx], phh[ny, nx], res['roll_deg'])
    hw = float(res.get('half_width_deg', 9.0))
    for off in ((0.0, 180.0) if model.symmetric else (0.0,)):
        d = np.abs((phh[ny, nx] - (model.entry_deg + res['roll_deg'] + off) + 180.0) % 360.0 - 180.0)
        stripe[ny[d < hw], nx[d < hw]] = 1
    for m, col in ((neck.astype(np.uint8), (0, 220, 255)), (win, (255, 0, 255)), (stripe, (0, 255, 0))):
        m = cv2.warpAffine(m, M, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                                 cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(up, cs, -1, col, 1)
    h, w = 120, up.shape[1]
    plot = np.full((h, w, 3), 30, np.uint8)
    z = np.asarray(res['curve'])
    xs = np.linspace(8, w - 8, len(z))
    ys = h - 10 - (z - z.min()) / (np.ptp(z) + 1e-9) * (h - 30)
    cv2.polylines(plot, [np.stack([xs, ys], 1).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    i = int(np.argmax(z))
    cv2.circle(plot, (int(xs[i]), int(ys[i])), 4, (0, 255, 0), -1)
    cv2.putText(plot, 'roll %.0f deg  margin %.2f' % (res['roll_deg'], res['margin']), (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return np.vstack([up, plot])


def public(res):
    """The estimate without its arrays (for JSON)."""
    return None if res is None else {k: v for k, v in res.items() if not k.startswith('_')}


# ---------------------------------------------------------------------------- the angle about the socket
def entry_dir(T_frame_plug, entry_deg):
    """The slot entry's direction (unit, radial) in the frame T_frame_plug is given in. With the tool0 -> plug
    belief the camera measured against, it is a FIXED direction on the tool -- the camera rides with the plug --
    and survives any later change to the belief (connector clocking's reset)."""
    ph = np.radians(entry_deg)
    return np.asarray(T_frame_plug)[:3, :3] @ np.array([np.cos(ph), np.sin(ph), 0.0])


def angle_about_deg(v, axis, up=(0.0, 0.0, 1.0)):
    """The direction `v` (base frame) as an angle about the socket's insertion axis `axis` (+X of the target: INTO
    the socket), from straight up (base +z projected; base +x if the axis is vertical), RIGHT-HANDED about the
    axis -- i.e. CLOCKWISE seen looking INTO the socket. The same sense as connector clocking's roll, so turning
    the wrist by theta there adds theta here. Mod 180 for symmetric slots is the caller's."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    u = np.asarray(up, dtype=float)
    u = u - a * (u @ a)
    if np.linalg.norm(u) < 0.2:
        u = np.array([1.0, 0.0, 0.0]) - a * a[0]
    u /= np.linalg.norm(u)
    e = np.asarray(v, dtype=float)
    e = e - a * (e @ a)
    return float(np.degrees(np.arctan2(np.cross(u, e) @ a, u @ e)))


def entry_angle_deg(T_base_plug, entry_deg, axis, up=(0.0, 0.0, 1.0)):
    """angle_about_deg of the entry of a plug at T_base_plug."""
    return angle_about_deg(entry_dir(T_base_plug, entry_deg), axis, up)


def wrap(x, period=180.0):
    """-> (-period/2, period/2]."""
    return float(-((-(x + period / 2.0)) % period) + period / 2.0)


def plan_legs(theta_star_deg, step_deg, max_deg, lo_deg, hi_deg, period=180.0):
    """THE SEARCH, NEAREST FIRST (the user, 2026-11-06: "try both angles in increasing increments so for example,
    +15, -15, +30, -30, etc"): roll targets (deg, from the engaged roll) at theta*, theta* + step, theta* - step,
    theta* + 2 step, ... out to +-max_deg. A target outside the reachable band [lo, hi] takes its equivalent a
    `period` round (the other slot) if that is inside, else the band's edge; repeats dropped."""
    out = [theta_star_deg]
    k = 1
    while k * step_deg <= max_deg + 1e-9:
        out += [theta_star_deg + k * step_deg, theta_star_deg - k * step_deg]
        k += 1
    legs = []
    for t in out:
        if not lo_deg - 1e-9 <= t <= hi_deg + 1e-9:
            for alt in (t - period, t + period):
                if lo_deg - 1e-9 <= alt <= hi_deg + 1e-9:
                    t = alt
                    break
            else:
                t = float(np.clip(t, lo_deg, hi_deg))
        if not legs or abs(legs[-1] - t) > 1e-6:
            legs.append(float(t))
    return legs


def nearest_turn(alpha0_deg, pin_deg, lo_deg, hi_deg, period=180.0):
    """theta*: the turn from the engaged roll that brings the entry onto the pins -- the one of theta + k period
    nearest zero inside [lo, hi] (None if none is)."""
    base = wrap(pin_deg - alpha0_deg, period)
    cands = [base + k * period for k in range(-3, 4)]
    cands = [c for c in cands if lo_deg - 1e-9 <= c <= hi_deg + 1e-9]
    return min(cands, key=abs) if cands else None


# ---------------------------------------------------------------------------- the pins, learned
class PinStore:
    """Where the socket's bayonet pins are (entry_angle_deg's convention, mod 180), learned from drop-ins: the
    entry's angle each time the plug dropped onto the pins. Kept with the socket's position -- a socket moved more
    than `max_move_mm` is another fixture, and its pins unknown."""

    def __init__(self, path):
        self.path = path

    def _load(self):
        try:
            with open(self.path) as fh:
                return list(json.load(fh).get('drop_ins', []))
        except (OSError, ValueError):
            return []

    def get(self, socket_xyz_m, max_move_mm=10.0, last=5, period=180.0):
        rows = [r for r in self._load()
                if np.linalg.norm(np.asarray(r['socket_xyz_m']) - np.asarray(socket_xyz_m)) * 1000.0 <= max_move_mm]
        if not rows:
            return None
        rows = rows[-last:]
        a = np.radians(np.asarray([r['pin_deg'] for r in rows]) * 360.0 / period)
        mean = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) * period / 360.0
        spread = float(np.degrees(np.arccos(min(1.0, np.hypot(np.sin(a).mean(), np.cos(a).mean())))) * period / 360.0)
        return {'pin_deg': wrap(mean, period), 'n': len(rows), 'spread_deg': spread}

    def add(self, pin_deg, socket_xyz_m, **meta):
        rows = self._load()
        rows.append(dict(meta, pin_deg=float(pin_deg), socket_xyz_m=[float(v) for v in socket_xyz_m],
                         time=datetime.now().isoformat(timespec='seconds')))
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'w') as fh:
            json.dump({'about': 'bnc_assembly connector clocking: the socket pins\' angle about the insertion axis '
                                '(deg from up, clockwise looking into the socket, mod 180), one row per drop-in',
                       'drop_ins': rows}, fh, indent=1)


# ---------------------------------------------------------------------------- re-estimating a recorded trial
def replay(trial_dir, every=1, stage=None, symmetric=True, cad=None, out=None):
    """The recorded engage frames of a trial, re-estimated -> prints each and the fusion; writes a drawing per frame
    to `out` (default engage_frames/slot_roll/). `cad`: a plug mesh (mm) in the 6D plug CAD's frame -- e.g.
    data/cad/bnc_plug_two_j_slots.ply -- else the recorded plug CAD."""
    import csv
    import cv2
    d = os.path.join(trial_dir, 'engage_frames')
    meta = json.load(open(os.path.join(d, 'meta.json')))
    model = SlotModel.from_cad(*((cad, 0.001) if cad else meta['plug_cad']), mating_z=0.1524 if cad else None,
                               symmetric=symmetric)
    K = np.asarray(meta['K'], dtype=float)
    T = np.linalg.inv(np.asarray(meta['T_tool0_cam'])) @ np.asarray(meta['T_tool0_plug'])
    rows = list(csv.DictReader(open(os.path.join(d, 'frames.csv'))))
    if stage:
        rows = [r for r in rows if r['stage'] == stage]
    out = out or os.path.join(d, 'slot_roll')
    os.makedirs(out, exist_ok=True)
    res = []
    for r in rows[::max(1, int(every))]:
        img = cv2.imread(os.path.join(d, r['file']))
        if img is None:
            continue
        e = estimate(img, K, T, model)
        res.append(e)
        if e is not None:
            print(f"{r['file']}: roll {e['roll_deg']:.0f} deg, margin {e['margin']:.2f}, shift {e['shift_px']}")
            vis = draw(img, e, model)
            if vis is not None:
                cv2.imwrite(os.path.join(out, r['file']), vis)
    f = fuse(res, model)
    if f is not None:
        print(f"FUSED over {f['n_frames']} frames: roll {f['roll_deg']:.0f} deg (entry at {f['entry_deg']:.0f} in "
              f"the plug frame), margin {f['margin']:.2f}")
    return f


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Re-estimate the BNC plug\'s slot roll on a trial\'s engage frames.')
    ap.add_argument('trial', help='a trial folder (with engage_frames/)')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--stage', default=None, help='only frames of this stage (e.g. contact)')
    ap.add_argument('--cad', default=None, help='the plug mesh (mm, the 6D plug CAD\'s frame), e.g. '
                                                'data/cad/bnc_plug_two_j_slots.ply')
    ap.add_argument('--out', default=None, help='where the drawings go (default <trial>/engage_frames/slot_roll)')
    a = ap.parse_args(argv)
    replay(a.trial, a.every, a.stage, cad=a.cad, out=a.out)


if __name__ == '__main__':
    main()
