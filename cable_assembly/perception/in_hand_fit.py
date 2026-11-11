"""THE CONNECTOR IN THE HAND, FITTED WITH THE GRIPPER'S CAD AND THE PLUG'S -- 3 DOF, in the jaws' plane.

A pinch between two parallel flat pads leaves a round plug free in three ways, all in the plane
midway between the pads (the fingertip frame's x-z, tool0's): SLIDE along its own axis, SHIFT
across it in that plane, and PIVOT about the closing direction (y). Its place along the closing
direction is the pads' midplane, and its axis lies in that plane (parallel pads square a cylinder
up), so the belief is put there first. Its ROLL about its own axis is NOT estimated: the plug is
drawn as a body of revolution (its CAD's radius profile), so the bayonet's channels never enter
the fit -- the clocking is the clocking search's (the user, 2026-11-05: SE(2) only).

THE GRIPPER FIRST. The bench's pads are pink blocks (`pad_mm`: 37.8 along the finger, 22.4 wide,
11.6 thick -- the boxes the sim's V3 model carries, sim/scripts/build_iiwa14.py PAD_MM), their tips
at the fingertip frame. Their CAD is fitted to the edges of the image's pink that nothing hides --
the tips (the vertical image shift) and the inner edges either side of the plug (the horizontal
one) -- giving the SHIFT: the hand-eye's and the gripper model's residual at the pads, applied to
the plug's drawing too, so the plug is fitted against the pads the camera sees.

THE GAP between the pads -- what they closed on, so which part of the plug is between them -- comes
from where the gripper stopped (`counts`: the calibrated finger model, robot/gripper_kinematics.py,
+ `counts_gap_offset_mm` for the bench's pads: 6.3 puts the 204-count grasps on the boot, 11-11.8
mm, and the 197-count ones within 0.3 mm of the collar, 14.5-14.7). Not from the image, unless no
counts are given: the plug, nearer the camera than the far half of each pad's inner face, hides
that half, so the pink's inner edge is partly the plug's own outline and the gap reads wide
(`gap_offset_mm` 1.5 on the bench's grasps, 2.2 on a synthetic one).

THEN THE PLUG, AGAINST THE GRIPPER. Its pixels -- GrabCut seeded from the belief's silhouettes over
the search, with the pink and the far depth as background -- above AND between the pads (the pads
are pink and the boot black, so the jaws no longer hide it), down to `between_pads_mm` from the
pad tips (below: the black finger links). Its CAD is drawn with a stub of cable behind it (the
cable runs on between the pads; without it the boot is pulled down over the cable's pixels).
Hypotheses the pads rule out are never scored:

    nothing between the pads may be wider than the gap (+ `penetration_mm`): the jaws closed on the
        plug, so a wider section -- the collar -- is outside them;
    what is between them is as wide as the gap (- `contact_mm`) for `min_grip_mm` along the axis:
        the jaws closed ON something.

THE EDGE-ON VIEW. The camera sits in the jaws' plane, so the plane projects to one image column:
every move fitted here moves the plug's drawing up or down that column and scales it, nothing
sideways. Slide and shift trade against each other along the line of sight (a pixel of the nose
is 0.2-0.4 mm). The pads' constraint is what pins that trade, and a weak pull toward the belief
(`prior_weight` at `prior_sigma_*`) settles what both leave -- it decides between equal fits only.

TRUSTED only when the pads were found (their CAD on their edges to `max_pad_residual_px`), the gap is one the plug has, the plug's
silhouette matches (`min_iou`), the best fit is not at the edge of the search, and the frames
agree (`max_spread_*`).
"""

import numpy as np

from .in_hand_rgb import Silhouette, radius_profile

# The bench's pads (the user, 2026-11-02), mm: along the finger (tool z), wide (tool x), thick (closing).
PAD_MM = (37.8, 22.4, 11.6)
SEARCH_STEP_M = 0.001          # the plug's profile, sampled this finely for the search's drawings


def pink_mask(bgr, hue=(150, 8), min_sat=40, min_val=80):
    """uint8 0/1: the pads' colour. `hue` is OpenCV's (0-179), (lo, hi) wrapping past 179 when lo > hi."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = int(hue[0]), int(hue[1])
    in_hue = ((h >= lo) | (h <= hi)) if lo > hi else ((h >= lo) & (h <= hi))
    return (in_hue & (s >= min_sat) & (v >= min_val)).astype(np.uint8)


def _project(P_cam, K):
    return P_cam[..., :2] / np.maximum(P_cam[..., 2:3], 1e-3) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def _rot(axis, a):
    """Rotation by `a` about the unit `axis` (Rodrigues)."""
    k = np.asarray(axis, dtype=float)
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * Kx + (1.0 - np.cos(a)) * Kx @ Kx


class Pads:
    """The two pads as boxes, in the fingertip frame: x across the finger (+-width/2), y the closing
    direction (gripping faces at +-gap/2), z from the tips (0) back toward the flange (length)."""

    def __init__(self, T_tool0_fingertip, pad_mm=PAD_MM):
        self.T = np.asarray(T_tool0_fingertip, dtype=float)
        self.T_inv = np.linalg.inv(self.T)
        self.length, self.width, self.thick = (float(v) / 1000.0 for v in pad_mm)

    @property
    def centre(self):
        """tool0, m: midway between the pads' faces, half-way along them."""
        return (self.T @ [0.0, 0.0, self.length / 2.0, 1.0])[:3]

    @property
    def closing(self):
        """tool0: the closing direction (the fingertip frame's y)."""
        return self.T[:3, 1].copy()

    def corners(self, gap):
        """Two (8, 3) arrays, tool0, m: each pad's corners at a gap of `gap` between the faces."""
        out = []
        for s in (-1.0, 1.0):
            y0, y1 = sorted((s * gap / 2.0, s * (gap / 2.0 + self.thick)))
            c = np.array([[x, y, z, 1.0] for x in (-self.width / 2.0, self.width / 2.0)
                          for y in (y0, y1) for z in (0.0, self.length)])
            out.append((self.T @ c.T).T[:, :3])
        return out

    def mask(self, gap, T_cam_tool0, K, shape, offset=(0, 0)):
        """uint8 0/1 of `shape` whose (0, 0) is image pixel `offset` (x, y)."""
        import cv2
        m = np.zeros(shape, np.uint8)
        for c in self.corners(gap):
            P = c @ T_cam_tool0[:3, :3].T + T_cam_tool0[:3, 3]
            if np.any(P[:, 2] <= 1e-3):
                continue
            uv = _project(P, K) - np.asarray(offset, dtype=float)
            cv2.fillConvexPoly(m, np.rint(cv2.convexHull(uv.astype(np.float32))).astype(np.int32), 1)
        return m

    def between(self, points):
        """(N, 3) tool0 points -> bool: those within the pads' faces' extent (x across, z along)."""
        q = np.asarray(points, dtype=float) @ self.T_inv[:3, :3].T + self.T_inv[:3, 3]
        return (np.abs(q[:, 0]) <= self.width / 2.0) & (q[:, 2] >= 0.0) & (q[:, 2] <= self.length)

    def point(self, along):
        """tool0: on the midplane's centre line, `along` m back from the tips."""
        return (self.T @ [0.0, 0.0, along, 1.0])[:3]


class InHandCadFit:
    """Fits the held plug's slide, shift and pivot in the jaws' plane. See the module docstring.

    vertices           the plug's CAD (N, 3), m, its axis along z, the cable leaving at its lowest z
    T_tool0_cam        the camera on the flange (hand_eye)
    T_tool0_fingertip  the pads' tips, closed (fingertip_grasp)"""

    def __init__(self, vertices, T_tool0_cam, T_tool0_fingertip, pad_mm=PAD_MM, cable_diameter_m=0.005,
                 cable_stub_m=0.03, slide_range_m=0.02, shift_range_m=0.006, pivot_range_rad=np.radians(15.0),
                 gap_range_m=(0.004, 0.02), gap_offset_m=0.0015, counts_gap_offset_m=0.0063, shift_px=20, between_pads_m=0.015, penetration_m=0.001,
                 contact_m=0.001, min_grip_m=0.003, prior_sigma=(0.010, 0.004, np.radians(8.0)),
                 prior_weight=0.01, min_iou=0.75, max_pad_residual_px=3.0, min_pad_rows=8, max_spread_m=0.002,
                 max_spread_rad=np.radians(2.0), pink=None, use_depth=True, unlike_bench=60.0, like_bench=6.0,
                 texture_sd=6.0, texture_px=4):
        v = np.asarray(vertices, dtype=float)
        z, r = radius_profile(v)
        zs, rs = radius_profile(v, SEARCH_STEP_M)
        self.r_min, self.r_max = float(r.min()), float(r.max())
        stub = np.arange(zs[0] - float(cable_stub_m), zs[0], SEARCH_STEP_M)
        rc = float(cable_diameter_m) / 2.0
        self.z_fine, self.r_fine = np.r_[stub, z], np.r_[np.full(len(stub), rc), r]
        self.z_st, self.r_st = np.r_[stub, zs], np.r_[np.full(len(stub), rc), rs]
        self.sil = Silhouette(self.z_fine, self.r_fine)          # drawn with the cable stub
        self.sil_search = Silhouette(self.z_st, self.r_st)
        self.sil_plug = Silhouette(z, r)                        # the plug alone, for the overlay
        self.T_tool0_cam = np.asarray(T_tool0_cam, dtype=float)
        self.T_cam_tool0 = np.linalg.inv(self.T_tool0_cam)
        self.pads = Pads(T_tool0_fingertip, pad_mm)
        self.slide_range, self.shift_range = float(slide_range_m), float(shift_range_m)
        self.pivot_range = float(pivot_range_rad)
        self.gap_range = (float(gap_range_m[0]), float(gap_range_m[1]))
        self.gap_offset = float(gap_offset_m)
        self.counts_gap_offset = float(counts_gap_offset_m)
        self.shift_px = int(shift_px)
        self.between_pads = float(between_pads_m)
        self.penetration, self.contact, self.min_grip = float(penetration_m), float(contact_m), float(min_grip_m)
        self.prior_sigma = tuple(float(s) for s in prior_sigma)
        self.prior_weight = float(prior_weight)
        self.min_iou = float(min_iou)
        self.max_pad_residual_px, self.min_pad_rows = float(max_pad_residual_px), int(min_pad_rows)
        self.max_spread_m, self.max_spread_rad = float(max_spread_m), float(max_spread_rad)
        self.pink = dict(pink or {})
        self.use_depth = bool(use_depth)
        self.unlike_bench, self.like_bench = float(unlike_bench), float(like_bench)
        self.texture_sd, self.texture_px = float(texture_sd), int(texture_px)

    @classmethod
    def from_cfg(cls, c, vertices, T_tool0_cam, T_tool0_fingertip):
        """From an `in_hand_fit:` block (mm / deg)."""
        mm = lambda k, d: float(c.get(k, d)) / 1000.0        # noqa: E731
        pk = dict(c.get('pink') or {})
        return cls(vertices, T_tool0_cam, T_tool0_fingertip,
                   pad_mm=tuple(c.get('pad_mm', PAD_MM)),
                   cable_diameter_m=mm('cable_diameter_mm', 5.0), cable_stub_m=mm('cable_stub_mm', 30.0),
                   slide_range_m=mm('slide_range_mm', 20.0), shift_range_m=mm('shift_range_mm', 6.0),
                   pivot_range_rad=np.radians(float(c.get('pivot_range_deg', 15.0))),
                   gap_range_m=tuple(float(g) / 1000.0 for g in c.get('gap_range_mm', (4.0, 20.0))),
                   gap_offset_m=mm('gap_offset_mm', 1.5), counts_gap_offset_m=mm('counts_gap_offset_mm', 6.3),
                   shift_px=int(c.get('register_shift_px', 20)),
                   between_pads_m=mm('between_pads_mm', 15.0), penetration_m=mm('penetration_mm', 1.0),
                   contact_m=mm('contact_mm', 1.0), min_grip_m=mm('min_grip_mm', 3.0),
                   prior_sigma=(mm('prior_sigma_slide_mm', 10.0), mm('prior_sigma_shift_mm', 4.0),
                                np.radians(float(c.get('prior_sigma_pivot_deg', 8.0)))),
                   prior_weight=float(c.get('prior_weight', 0.01)),
                   min_iou=float(c.get('min_iou', 0.75)),
                   max_pad_residual_px=float(c.get('max_pad_residual_px', 3.0)),
                   min_pad_rows=int(c.get('min_pad_rows', 8)),
                   max_spread_m=mm('max_spread_mm', 2.0),
                   max_spread_rad=np.radians(float(c.get('max_spread_deg', 2.0))),
                   pink={'hue': tuple(pk.get('hue', (150, 8))), 'min_sat': int(pk.get('min_sat', 40)),
                         'min_val': int(pk.get('min_val', 80))},
                   use_depth=bool(c.get('use_depth', True)),
                   unlike_bench=float(c.get('unlike_bench', 60.0)), like_bench=float(c.get('like_bench', 6.0)),
                   texture_sd=float(c.get('texture_sd', 6.0)), texture_px=int(c.get('texture_px', 4)))

    # ------------------------------------------------------------------ the three moves
    def in_plane(self, T_tool0_plug):
        """The belief put where a pinch puts it: its axis turned (least) into the plane parallel to the
        pads, and moved along the closing direction onto their midplane. (pose, mm moved, deg turned)"""
        T = np.array(T_tool0_plug, dtype=float)
        y, c = self.pads.closing, self.pads.centre
        a0 = T[:3, 2]
        a = a0 - (a0 @ y) * y
        a /= np.linalg.norm(a)
        ang = float(np.arccos(np.clip(a0 @ a, -1.0, 1.0)))
        if ang > 1e-9:
            k = np.cross(a0, a)
            T[:3, :3] = _rot(k / np.linalg.norm(k), ang) @ T[:3, :3]
        d = float((T[:3, 3] - c) @ y)
        T[:3, 3] -= d * y
        return T, abs(d), float(np.degrees(ang))

    def moved(self, T0, slide, shift, pivot):
        """T0 slid `slide` along its axis, shifted `shift` across it in the jaws' plane, then turned
        `pivot` about the closing direction through the pads' centre."""
        y, c = self.pads.closing, self.pads.centre
        ax = T0[:3, 2]
        T = np.array(T0, dtype=float)
        T[:3, 3] += slide * ax + shift * np.cross(y, ax)
        R = _rot(y, pivot)
        out = np.eye(4)
        out[:3, :3] = R @ T[:3, :3]
        out[:3, 3] = R @ (T[:3, 3] - c) + c
        return out

    def held(self, T_tool0_plug, gap):
        """(possible, why): whether the pads at `gap` can be holding the plug at this pose."""
        P = T_tool0_plug[:3, :3] @ np.stack([np.zeros_like(self.z_st), np.zeros_like(self.z_st), self.z_st]) \
            + T_tool0_plug[:3, 3:4]
        inside = self.pads.between(P.T)
        if not inside.any():
            return False, 'not between the pads'
        r = self.r_st[inside]
        if r.max() > gap / 2.0 + self.penetration:
            return False, f'{2000 * r.max():.1f} mm of it between pads {1000 * gap:.1f} mm apart'
        if np.count_nonzero(r >= gap / 2.0 - self.contact) * SEARCH_STEP_M < self.min_grip:
            return False, 'the pads touch too little of it'
        return True, ''

    # ------------------------------------------------------------------ the gripper
    def midline(self, K):
        """(a, b): the jaws' midplane in the image, the column u = a v + b (the camera sits in that plane,
        so the whole plane is this one line)."""
        p = [self.T_cam_tool0 @ np.append(self.pads.point(d), 1.0) for d in (0.0, self.pads.length)]
        (u0, v0), (u1, v1) = (_project(q[:3], K) for q in p)
        a = (u1 - u0) / (v1 - v0) if abs(v1 - v0) > 1e-6 else 0.0
        return a, u0 - a * v0

    @staticmethod
    def _inner_edges(mask, mid_u, reach):
        """Per row of `mask` (0/1): the columns of the pink nearest the midline on either side, within
        `reach` px of it (NaN where none): the pads' gripping faces."""
        h, w = mask.shape
        cols = np.arange(w, dtype=float)
        left = np.where((mask > 0) & (cols[None, :] < mid_u[:, None]) & (cols[None, :] > mid_u[:, None] - reach),
                        cols[None, :], -np.inf).max(axis=1)
        right = np.where((mask > 0) & (cols[None, :] >= mid_u[:, None]) & (cols[None, :] < mid_u[:, None] + reach),
                         cols[None, :], np.inf).min(axis=1)
        return np.where(np.isfinite(left), left, np.nan), np.where(np.isfinite(right), right, np.nan)

    def register(self, bgr, K):
        """The pads' CAD fitted to the edges of the image's pink that nothing hides -- {'gap', 'shift' (px),
        'residual_px', 'rows', 'box', 'pink_box'}, or None. Their TOP edge (the tips) gives the vertical
        image shift; their INNER edges (the gripping faces, either side of the plug), the horizontal one
        and the gap. Not their area: the finger mounts hide the pads' outer sides and the fingers shade
        their lower half, and an area fit pulled the gap 3 mm short (2026-11-05 18:43 grasp)."""
        K = np.asarray(K, dtype=float)
        h, w = bgr.shape[:2]
        uv = np.vstack([_project(c @ self.T_cam_tool0[:3, :3].T + self.T_cam_tool0[:3, 3], K)
                        for c in self.pads.corners(self.gap_range[1])])
        S = self.shift_px
        x0, y0 = np.maximum(np.floor(uv.min(axis=0)).astype(int) - S, 0)
        x1, y1 = np.minimum(np.ceil(uv.max(axis=0)).astype(int) + S, [w, h])
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        pink = pink_mask(bgr[y0:y1, x0:x1], **self.pink)
        bh, bw = pink.shape
        a, b = self.midline(K)
        rows = np.arange(y0, y1, dtype=float)
        reach = float(np.ptp(uv[:, 0])) / 2.0 + S

        # THE TOP EDGE -> dv: the first pink row of each column vs the CAD's, over the columns both have.
        def tops(m):
            have = m.any(axis=0)
            return np.where(have, m.argmax(axis=0), -1)
        m_mid = self.pads.mask(0.5 * (self.gap_range[0] + self.gap_range[1]), self.T_cam_tool0, K, (bh, bw), (x0, y0))
        t_obs, t_cad = tops(pink), tops(m_mid)
        both = (t_obs >= 0) & (t_cad >= 0)
        if np.count_nonzero(both) < 10:
            return None
        dv = int(np.clip(np.rint(np.median(t_obs[both] - t_cad[both])), -S, S))

        # THE INNER EDGES -> du and the gap, over the rows from the tips back `between_pads` (the lit part).
        r_lo = int(np.median(t_cad[t_cad >= 0])) + dv + 3
        r_hi = self.bottom_row(K, (0, dv)) - y0 - 3
        if r_hi - r_lo < 5:
            return None
        band = slice(max(r_lo, 0), min(r_hi, bh))
        mid = a * (rows + 0.0) + b - x0                              # the midline, ROI columns, each row
        oL, oR = self._inner_edges(pink[band], mid[band], reach)
        ok_rows = np.isfinite(oL) & np.isfinite(oR)
        if np.count_nonzero(ok_rows) < 5:
            return None

        def residual(gap):
            m = self.pads.mask(gap, self.T_cam_tool0, K, (bh, bw), (x0, y0 - dv))
            cL, cR = self._inner_edges(m[band], mid[band] - a * dv, reach)
            k = ok_rows & np.isfinite(cL) & np.isfinite(cR)
            if np.count_nonzero(k) < 5:
                return np.inf, 0, 0
            du = float(np.median((oL[k] + oR[k]) / 2.0 - (cL[k] + cR[k]) / 2.0))
            return float(np.median(np.abs((oR[k] - oL[k]) - (cR[k] - cL[k])))), du, int(np.count_nonzero(k))

        best = (np.inf, 0.0, 0.0, 0)
        for gap in np.arange(self.gap_range[0], self.gap_range[1] + 1e-9, 0.0005):
            res, du, n = residual(gap)
            best = min(best, (res, gap, du, n))
        g0 = best[1]
        for gap in g0 + np.arange(-0.0005, 0.00051, 0.0001):
            if self.gap_range[0] <= gap <= self.gap_range[1]:
                res, du, n = residual(gap)
                best = min(best, (res, gap, du, n))
        res, gap, du, n = best
        if not np.isfinite(res):
            return None
        # THE FEATURES MATCHED, in image pixels, for the live view: the tips (top edge) and gripping faces.
        cols = np.flatnonzero(both)
        brow = np.arange(band.start, band.start + len(oL))[ok_rows] + y0
        edges = {'top': np.stack([cols + x0, t_obs[cols] + y0], axis=1).astype(int),
                 'inner': np.concatenate([np.stack([oL[ok_rows] + x0, brow], axis=1),
                                          np.stack([oR[ok_rows] + x0, brow], axis=1)]).astype(int)}
        return {'gap': float(gap) - self.gap_offset, 'gap_seen': float(gap),
                'shift': (int(np.clip(np.rint(du), -S, S)), dv), 'residual_px': float(res),
                'rows': int(n), 'box': (x0, y0, x1, y1), 'pink_box': pink, 'edges': edges}

    # ------------------------------------------------------------------ the plug
    def _grid(self, centre, half, step):
        return [centre + d for d in np.arange(-half, half + 1e-9, step)]

    def _roi(self, T0, K, shape, shift):
        pts = []
        for s in (-self.slide_range, 0.0, self.slide_range):
            for t in (-self.shift_range, 0.0, self.shift_range):
                for a in (-self.pivot_range, 0.0, self.pivot_range):
                    uv, front = self.sil.project(self.T_cam_tool0 @ self.moved(T0, s, t, a), K)
                    if front:
                        pts.append(uv.reshape(-1, 2) + shift)
        if not pts:
            return None
        pts = np.vstack(pts)
        h, w = shape[:2]
        x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 15
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 15
        y1 = min(y1, self.bottom_row(K, shift))
        return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)

    def bottom_row(self, K, shift=(0, 0)):
        """The image row `between_pads` back from the pad tips, on the midplane: rows below it unused."""
        p = self.T_cam_tool0 @ np.append(self.pads.point(self.between_pads), 1.0)
        return int(np.floor(K[1, 1] * p[1] / p[2] + K[1, 2] + shift[1]))

    def segment(self, bgr, depth, T0, K, box, shift, pink):
        """The plug's pixels in `box` (uint8 0/1): GrabCut seeded from the silhouettes over the search
        (inside all: the plug; inside some: probably; inside none: probably not), the pads' pink and
        anything the depth puts well beyond the plug: not."""
        import cv2
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(bgr[y0:y1, x0:x1])
        shape = crop.shape[:2]
        off = (x0 - shift[0], y0 - shift[1])
        union = np.zeros(shape, np.uint8)
        core = np.ones(shape, np.uint8)
        far = 0.0
        for s in np.linspace(-self.slide_range, self.slide_range, 5):
            for t in (-self.shift_range, 0.0, self.shift_range):
                for a in (-self.pivot_range, 0.0, self.pivot_range):
                    T = self.T_cam_tool0 @ self.moved(T0, s, t, a)
                    m = self.sil_search.mask(T, K, shape, off)
                    union |= m
                    core &= m
                    far = max(far, float((self.sil_search.rings[..., 2].max() + 0.01) + T[2, 3]))
        gc = np.where(union > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        gc[core > 0] = cv2.GC_FGD if core.sum() > 50 else cv2.GC_PR_FGD
        pk = pink[y0:y1, x0:x1] > 0
        # THE BENCH'S COLOUR, from what no hypothesis reaches: inside the search, what is far from it
        # is the plug, what is near it probably not. Seeded from the core alone, the plug's colour
        # model learned only the black boot, and the silver nose went to the bench (2026-11-02 23:46).
        outside = (union == 0) & ~pk
        if np.count_nonzero(outside) > 200:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
            bg = lab[outside]
            mu = bg.mean(axis=0)
            Ci = np.linalg.inv(np.cov(bg.T) + np.eye(3) * 4.0)
            dl = lab - mu
            d2 = np.einsum('...i,ij,...j->...', dl, Ci, dl)
            inside = (union > 0) & ~pk
            gc[inside & (d2 < self.like_bench) & (core == 0)] = cv2.GC_PR_BGD
            gc[inside & (d2 > self.unlike_bench)] = cv2.GC_FGD
            # ... and the metal is the bench's colour (Lab 121/135/134 against 125/133/131) but not its
            # smoothness: the INSIDE of what is textured (eroded, so not the halo round every edge),
            # away from the pads' edges, is the plug too.
            k = 2 * self.texture_px + 1
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mean = cv2.blur(g, (k, k))
            sd = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mean * mean, 0.0))
            tex = cv2.erode((sd > self.texture_sd).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            near_pink = cv2.dilate(pk.astype(np.uint8), np.ones((k + 4, k + 4), np.uint8)) > 0
            gc[inside & tex & ~near_pink] = cv2.GC_FGD
        gc[pk] = cv2.GC_BGD
        if self.use_depth and depth is not None:
            d = np.asarray(depth)[y0:y1, x0:x1]
            gc[(d > far + 0.03)] = cv2.GC_BGD                   # read, and well beyond any plug pose
        gc[:2, :], gc[:, :2], gc[:, -2:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD   # not the bottom: the pads' rows
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(crop, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return np.zeros(shape, np.uint8)
        fg = (((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & ~pk).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 1:
            fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        # SOLID: the metal's highlights read as bench; a hole inside the plug is the plug.
        hole = np.pad(1 - fg, 1, constant_values=1)
        cv2.floodFill(hole, None, (0, 0), 0)
        return fg | hole[1:-1, 1:-1]

    def _score(self, T0, s, t, a, K, target, off, n_target):
        m = self.sil_search.mask(self.T_cam_tool0 @ self.moved(T0, s, t, a), K, target.shape, off)
        inter = np.count_nonzero(m & target)
        union = np.count_nonzero(m) + n_target - inter
        iou = inter / union if union else 0.0
        ss, st, sa = self.prior_sigma
        return iou - self.prior_weight * ((s / ss) ** 2 + (t / st) ** 2 + (a / sa) ** 2), iou

    def _pads_found(self, reg):
        return reg is not None and reg['residual_px'] <= self.max_pad_residual_px and reg['rows'] >= self.min_pad_rows

    def fit_frame(self, frame, K, T0, reg, gap=None, progress=None):
        """One frame, the pads registered (`reg`): {'slide', 'shift', 'pivot', 'iou', 'at_edge', 'box',
        'mask', 'held', 'n_held'} or None (nothing in view). `progress(stage, **info)`, if given, hears the
        segmentation and every better candidate of the search (the live view)."""
        tell = progress or (lambda *a, **k: None)
        K = np.asarray(K, dtype=float)
        bgr = frame.color
        shift = reg['shift']
        box = self._roi(T0, K, bgr.shape, shift)
        if box is None or box[2] - box[0] < 10 or box[3] - box[1] < 10:
            return None
        pink = np.zeros(bgr.shape[:2], np.uint8)
        bx0, by0, bx1, by1 = reg['box']
        pink[by0:by1, bx0:bx1] = reg['pink_box']
        target = self.segment(bgr, getattr(frame, 'depth', None), T0, K, box, shift, pink)
        off = (box[0] - shift[0], box[1] - shift[1])
        n_target = np.count_nonzero(target)
        gap = reg['gap'] if gap is None else gap
        tell('segment', box=box, mask=target, gap=gap)
        best, n_held = None, 0
        n_grid = len(self._grid(0.0, self.slide_range, 0.002)) * len(self._grid(0.0, self.shift_range, 0.0015)) \
            * len(self._grid(0.0, self.pivot_range, np.radians(2.0)))
        n_eval = 0
        for s in self._grid(0.0, self.slide_range, 0.002):
            for t in self._grid(0.0, self.shift_range, 0.0015):
                for a in self._grid(0.0, self.pivot_range, np.radians(2.0)):
                    if not self.held(self.moved(T0, s, t, a), gap)[0]:
                        continue
                    n_held += 1
                    sc, iou = self._score(T0, s, t, a, K, target, off, n_target)
                    n_eval += 1
                    if best is None or sc > best[0]:
                        best = (sc, iou, s, t, a)
                        tell('coarse', box=box, mask=target, gap=gap, best=best, n=n_eval, of=n_grid)
        if best is None:
            return {'slide': 0.0, 'shift': 0.0, 'pivot': 0.0, 'iou': 0.0, 'at_edge': False, 'box': box,
                    'mask': target, 'held': False, 'n_held': 0,
                    'why_not_held': self.held(T0, gap)[1]}
        _, _, s0, t0, a0 = best
        for s in self._grid(s0, 0.002, 0.0005):
            for t in self._grid(t0, 0.0015, 0.0005):
                for a in self._grid(a0, np.radians(2.0), np.radians(0.5)):
                    if (abs(s) > self.slide_range + 1e-9 or abs(t) > self.shift_range + 1e-9
                            or abs(a) > self.pivot_range + 1e-9 or not self.held(self.moved(T0, s, t, a), gap)[0]):
                        continue
                    sc, iou = self._score(T0, s, t, a, K, target, off, n_target)
                    if sc > best[0]:
                        best = (sc, iou, s, t, a)
                        tell('fine', box=box, mask=target, gap=gap, best=best)
        _, iou, s, t, a = best
        at_edge = (abs(s) > self.slide_range - 0.001 or abs(t) > self.shift_range - 0.00075
                   or (self.pivot_range > 0.0 and abs(a) > self.pivot_range - np.radians(1.0)))
        return {'slide': float(s), 'shift': float(t), 'pivot': float(a), 'iou': float(iou), 'at_edge': bool(at_edge),
                'box': box, 'mask': target, 'held': True, 'n_held': n_held}

    def gap_from_counts(self, counts):
        """The pads' gap, m, at a Robotiq position: the calibrated finger model (robot/gripper_kinematics.py,
        the UR's grooved tips) + `counts_gap_offset` for the bench's pads."""
        from ..robot import gripper_kinematics
        return float(gripper_kinematics.gap_from_counts(float(counts))) + self.counts_gap_offset

    def refine(self, frames, T_tool0_plug, counts=None, progress=None):
        """Frames (each with .color BGR, .K and optionally .depth) -> a result dict. `counts`: where the
        gripper stopped on the plug -- its gap then comes from the finger model; without, from the image.

            ok, why          trusted, and what failed when not
            slide, shift, pivot, iou, pad_residual_px, image_shift   the median over the frames that fitted
            gap, gap_from     the gap the pads were taken to have, and from what ('counts' / 'image')
            gap_image         the gap read from the image's pink edges (see the module docstring)
            T_tool0_plug     the fitted pose (the in-plane belief's when not ok; the roll is the belief's)
            in_plane_mm, in_plane_deg   how far the belief had to move to lie between the pads
            per_frame        each frame's fit (without masks)

        `progress(state)`, if given, is called as the fit goes -- the pads registered, the plug segmented, each
        better candidate, the verdict -- with a dict for live_image() (the in-hand fit's online view)."""
        T0, d_mm, d_deg = self.in_plane(T_tool0_plug)
        n_frames = len(frames)

        def tell_for(i, f, reg):
            if progress is None:
                return None

            def tell(stage, **info):
                try:
                    progress({'stage': stage, 'frame': i, 'n_frames': n_frames, 'bgr': f.color, 'K': f.K,
                              'T0': T0, 'T_prior': T_tool0_plug, 'reg': reg, **info})
                except Exception:          # noqa: BLE001 -- a view, never a reason for the fit to fail
                    pass
            return tell
        gap_counts = None if counts is None else self.gap_from_counts(counts)
        out = {'ok': False, 'why': '', 'slide': 0.0, 'shift': 0.0, 'pivot': 0.0, 'iou': 0.0, 'gap': 0.0,
               'gap_from': 'image' if gap_counts is None else 'counts', 'gap_image': 0.0,
               'pad_residual_px': float('inf'), 'image_shift': (0, 0), 'T_tool0_plug': T0, 'T_in_plane': T0,
               'in_plane_mm': 1000.0 * d_mm, 'in_plane_deg': d_deg, 'per_frame': [], 'fits': [], 'regs': []}
        fits = []
        for i, f in enumerate(frames):
            reg = self.register(f.color, f.K)
            tell = tell_for(i, f, reg)
            if tell is not None:
                tell('pads')
            if reg is None:
                continue
            out['regs'].append(reg)
            if not self._pads_found(reg):
                continue
            gap = reg['gap'] if gap_counts is None else gap_counts
            r = self.fit_frame(f, f.K, T0, reg, gap, progress=tell)
            if r is not None:
                r.update(gap=gap, gap_image=reg['gap'], pad_residual_px=reg['residual_px'], image_shift=reg['shift'])
                fits.append(r)
        out['fits'] = fits
        out['per_frame'] = [{k: v for k, v in r.items() if k != 'mask'} for r in fits]
        if out['regs']:
            best_reg = min(out['regs'], key=lambda g: g['residual_px'])
            out.update(gap=best_reg['gap'] if gap_counts is None else gap_counts, gap_image=best_reg['gap'],
                       pad_residual_px=best_reg['residual_px'], image_shift=best_reg['shift'])
        if not any(self._pads_found(g) for g in out['regs']):
            out['why'] = ('the pads were not found in the image' if not out['regs'] else
                          f"the pads' CAD does not fit their edges ({out['pad_residual_px']:.1f} px > "
                          f'{self.max_pad_residual_px:.1f}, or too few rows)') + ' -- check in_hand_fit.pink'
            return self._told(out, frames, progress, T0, T_tool0_plug)
        held = [r for r in fits if r['held']]
        if not held:
            why = fits[0].get('why_not_held', '') if fits else 'the plug is not in view'
            out['why'] = f'no pose between the pads fits ({why}) -- the gap or the belief is off'
            return self._told(out, frames, progress, T0, T_tool0_plug)
        med = {k: float(np.median([r[k] for r in held])) for k in ('slide', 'shift', 'pivot', 'iou', 'gap', 'gap_image', 'pad_residual_px')}
        out.update(med)
        out['image_shift'] = tuple(int(np.median([r['image_shift'][i] for r in held])) for i in (0, 1))
        spread_s = max(float(np.ptp([r['slide'] for r in held])), float(np.ptp([r['shift'] for r in held])))
        spread_a = float(np.ptp([r['pivot'] for r in held]))
        if not 2.0 * self.r_min - 0.002 <= out['gap'] <= 2.0 * self.r_max + 0.001:
            out['why'] = (f"the pads are {1000 * out['gap']:.1f} mm apart -- no part of the plug is that wide "
                          f'({2000 * self.r_min:.1f}-{2000 * self.r_max:.1f} mm)')
        elif out['iou'] < self.min_iou:
            out['why'] = f"the CAD matches the image poorly (IoU {out['iou']:.2f} < {self.min_iou:.2f})"
        elif sum(r['at_edge'] for r in held) * 2 > len(held):
            out['why'] = 'the best fit is at the edge of the search: further off than searched'
        elif spread_s > self.max_spread_m or spread_a > self.max_spread_rad:
            out['why'] = (f'the frames disagree ({1000 * spread_s:.1f} mm, {np.degrees(spread_a):.1f} deg apart): '
                          'the plug moved, or the image is ambiguous')
        else:
            out['ok'] = True
            out['T_tool0_plug'] = self.moved(T0, out['slide'], out['shift'], out['pivot'])
        return self._told(out, frames, progress, T0, T_tool0_plug)

    @staticmethod
    def _told(out, frames, progress, T0, T_prior):
        """`out`, after telling the live view the verdict."""
        if progress is not None and frames:
            fits = out.get('fits') or []
            reg = min(out['regs'], key=lambda g: g['residual_px']) if out.get('regs') else None
            try:
                progress({'stage': 'done', 'frame': len(frames) - 1, 'n_frames': len(frames), 'bgr': frames[0].color,
                          'K': frames[0].K, 'T0': T0, 'T_prior': T_prior, 'reg': reg, 'result': out,
                          'box': fits[0]['box'] if fits else None, 'mask': fits[0]['mask'] if fits else None,
                          'gap': out.get('gap'),
                          'best': (0.0, out['iou'], out['slide'], out['shift'], out['pivot']) if fits else None})
            except Exception:              # noqa: BLE001
                pass
        return out

    # ------------------------------------------------------------------ the online view
    STAGES = {'pads': 'the pads registered on their edges', 'segment': 'the plug segmented',
              'coarse': 'coarse search', 'fine': 'fine search', 'done': 'done'}

    def live_image(self, st, width=1100, body_h=560):
        """THE CAD MATCHING AS IT HAPPENS (the user, 2026-11-05: "add an online visualization of the CAD based
        feature matching for the in hand pose estimation"), drawn from a refine(progress=) state: the region
        round the pads and the plug, enlarged, with
            the pads' edges found in the image -- their tips (red) and gripping faces (orange) --
            the pads' CAD where those edges put it (yellow),
            the plug's pixels as segmented (cyan),
            the belief's silhouette (magenta) and the best candidate so far, then the fit (green),
            the jaws' midplane (dashed) and the last row the fit uses (grey),
        and above it the stage, the numbers and, at the end, the verdict. BGR, `width` px wide."""
        import cv2
        bgr, K = st['bgr'], np.asarray(st['K'], dtype=float)
        H, W = bgr.shape[:2]
        reg = st.get('reg')
        du, dv = (reg['shift'] if reg else (0, 0))
        boxes = [b for b in ((reg or {}).get('box'), st.get('box')) if b is not None]
        if boxes:
            x0 = max(min(b[0] for b in boxes) - 30, 0)
            y0 = max(min(b[1] for b in boxes) - 30, 0)
            x1 = min(max(b[2] for b in boxes) + 30, W)
            y1 = min(max(b[3] for b in boxes) + 30, H)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        crop = bgr[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        sc = min(width / cw, body_h / ch)
        ow, oh = max(int(cw * sc), 1), max(int(ch * sc), 1)
        img = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
        RED, ORANGE, YELLOW, CYAN, MAGENTA, GREEN, GREY = ((40, 40, 235), (0, 140, 255), (0, 230, 255),
                                                          (230, 210, 0), (230, 60, 230), (60, 220, 60), (150, 150, 150))

        def P(u, v):
            return int(round((u - x0 + 0.5) * sc)), int(round((v - y0 + 0.5) * sc))

        def big(mask):
            return cv2.resize(mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)

        def outline(mask, col, thick):
            cs, _ = cv2.findContours(big(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, col, thick, cv2.LINE_AA)

        off = (x0 - du, y0 - dv)                   # the crop's corner in the CAD's (unshifted) pixels
        if st.get('mask') is not None and st.get('box') is not None:
            bx0, by0, bx1, by1 = st['box']
            m = np.zeros((ch, cw), np.uint8)
            sx0, sy0 = max(bx0, x0), max(by0, y0)
            sx1, sy1 = min(bx1, x1), min(by1, y1)
            if sx1 > sx0 and sy1 > sy0:
                m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = st['mask'][sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0]
            sel = big(m) > 0
            img[sel] = (0.55 * img[sel] + 0.45 * np.array(CYAN)).astype(np.uint8)
            outline(m, CYAN, 1)
        if reg:
            outline(self.pads.mask(reg['gap_seen'], self.T_cam_tool0, K, (ch, cw), off), YELLOW, 2)
            for u, v in reg['edges']['top'][::2]:
                cv2.circle(img, P(u, v), 2, RED, -1, cv2.LINE_AA)
            for u, v in reg['edges']['inner'][::2]:
                cv2.circle(img, P(u, v), 2, ORANGE, -1, cv2.LINE_AA)
            a_, b_ = self.midline(K)
            for v in range(y0, y1, 12):
                cv2.line(img, P(a_ * v + b_ + du, v), P(a_ * (v + 6) + b_ + du, v + 6), GREY, 1, cv2.LINE_AA)
            yb = self.bottom_row(K, (du, dv))
            if y0 <= yb < y1:
                cv2.line(img, P(x0, yb), P(x1, yb), GREY, 1, cv2.LINE_AA)
        outline(self.sil_plug.mask(self.T_cam_tool0 @ st['T_prior'], K, (ch, cw), off), MAGENTA, 1)
        best = st.get('best')
        if best is not None:
            outline(self.sil_plug.mask(self.T_cam_tool0 @ self.moved(st['T0'], best[2], best[3], best[4]), K,
                                       (ch, cw), off), GREEN, 2)

        # ---- the words: stage, numbers, verdict; and the legend
        res = st.get('result')
        stage = st['stage']
        head = f"IN-HAND CAD FIT   frame {st['frame'] + 1}/{st['n_frames']}   "
        if stage == 'pads':
            head += ('the pads registered on their edges' if reg and self._pads_found(reg) else
                     'the pads NOT found on their edges' if reg else 'no pads in view')
        elif stage == 'coarse':
            head += f"coarse search {st.get('n', 0)}/{st.get('of', 0)}"
        elif stage == 'done' and res is not None:
            head = f"IN-HAND CAD FIT   {st['n_frames']} frame{'s' if st['n_frames'] != 1 else ''}   " + \
                ('TRUSTED' if res['ok'] else 'NOT USED')
        else:
            head += self.STAGES.get(stage, stage)
        nums = []
        if best is not None:
            nums.append(f'slide {1000 * best[2]:+.1f} mm   shift {1000 * best[3]:+.1f} mm   '
                        f'pivot {np.degrees(best[4]):+.1f} deg   IoU {best[1]:.2f}')
        if reg:
            gap = st.get('gap')
            nums.append((f'gap {1000 * gap:.1f} mm   ' if gap else '') +
                        f"pads on their edges to {reg['residual_px']:.1f} px ({reg['rows']} rows), "
                        f'image {du:+d}/{dv:+d} px off the CAD')
        verdict = None
        if res is not None:
            verdict = ('the belief moves to the fit' if res['ok'] else 'NOT USED: ' + res['why'])[:130]
        top_h = 30 + 24 * len(nums) + (26 if verdict else 0) + 8
        out = np.full((top_h + oh + 34, max(ow, width), 3), 28, np.uint8)
        xo = (out.shape[1] - ow) // 2
        out[top_h:top_h + oh, xo:xo + ow] = img
        hcol = (GREEN if res['ok'] else RED) if res is not None else (235, 235, 235)
        cv2.putText(out, head, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, hcol, 2, cv2.LINE_AA)
        for i, t in enumerate(nums):
            cv2.putText(out, t, (12, 52 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)
        if verdict:
            cv2.putText(out, verdict, (12, 52 + 24 * len(nums)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, hcol, 1, cv2.LINE_AA)
        x, y = 12, top_h + oh + 23
        for label, col in (('pad tips found', RED), ('pad faces found', ORANGE), ('pads CAD', YELLOW),
                           ('plug pixels', CYAN), ('belief', MAGENTA), ('fit / best so far', GREEN)):
            cv2.rectangle(out, (x, y - 11), (x + 14, y + 1), col, -1)
            cv2.putText(out, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            x += 30 + int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0])
        return out

    def overlay(self, bgr, K, T_prior, result):
        """A diagnostic image: the pads as fitted (yellow), the belief (magenta), the fit (green), the
        plug's pixels (cyan tint), the bottom of the fitted rows (grey)."""
        import cv2
        img = bgr.copy()
        K = np.asarray(K, dtype=float)
        sh = np.asarray(result.get('image_shift') or (0, 0), dtype=float)
        fits = result.get('fits') or []
        if fits:
            x0, y0, x1, y1 = fits[0]['box']
            tint = img[y0:y1, x0:x1]
            sel = fits[0]['mask'] > 0
            tint[sel] = (0.5 * tint[sel] + [128, 128, 0]).astype(np.uint8)
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
        if result.get('gap_image') or result.get('gap'):
            m = self.pads.mask((result.get('gap_image') or result['gap']) + self.gap_offset, self.T_cam_tool0, K,
                               img.shape[:2], tuple(-sh))
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (0, 255, 255), 1)
        T_fit = self.moved(result['T_in_plane'], result['slide'], result['shift'], result['pivot'])
        for T, col in ((T_prior, (255, 0, 255)), (T_fit, (0, 255, 0))):
            m = self.sil_plug.mask(self.T_cam_tool0 @ T, K, img.shape[:2], tuple(-sh))
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, col, 2)
        y = self.bottom_row(K, sh)
        cv2.line(img, (0, y), (img.shape[1], y), (128, 128, 128), 1)
        text = (f"in-hand CAD fit: slide {1000 * result['slide']:+.1f} mm  shift {1000 * result['shift']:+.1f} mm  "
                f"pivot {np.degrees(result['pivot']):+.1f} deg  IoU {result['iou']:.2f}  gap {1000 * result['gap']:.1f} mm "
                f"({result.get('gap_from', 'image')})  "
                f"pads +-{result['pad_residual_px']:.1f} px")
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if result['ok'] else (0, 0, 255), 2)
        cv2.putText(img, 'OK' if result['ok'] else 'NOT USED: ' + result['why'][:110], (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if result['ok'] else (0, 0, 255), 2)
        return img


# ---------------------------------------------------------------------- replay
def save_case(path, frames, T_prior, T_tool0_cam, T_tool0_fingertip, counts=None):
    """One in-hand fit's inputs, for `python -m cable_assembly.perception.in_hand_fit` to replay exactly."""
    depth = [f.depth for f in frames if getattr(f, 'depth', None) is not None]
    np.savez_compressed(path, color=np.stack([f.color for f in frames]), K=np.asarray(frames[0].K, dtype=float),
                        depth=np.stack(depth) if len(depth) == len(frames) else np.zeros(0),
                        T_prior=np.asarray(T_prior, dtype=float), T_tool0_cam=np.asarray(T_tool0_cam, dtype=float),
                        T_tool0_fingertip=np.asarray(T_tool0_fingertip, dtype=float),
                        counts=np.array(-1 if counts is None else int(counts)))


def load_case(path):
    """(frames, T_prior, T_tool0_cam, T_tool0_fingertip, counts or None) from save_case's file."""
    import types
    with np.load(path) as z:
        depth = z['depth'] if z['depth'].size else [None] * len(z['color'])
        frames = [types.SimpleNamespace(color=c, K=z['K'], depth=d) for c, d in zip(z['color'], depth)]
        counts = int(z['counts']) if 'counts' in z.files and int(z['counts']) >= 0 else None
        return frames, z['T_prior'], z['T_tool0_cam'], z['T_tool0_fingertip'], counts


def prior_from_cfg(cfg, T_tool0_fingertip):
    """The in-hand belief's plug pose (tool0) the way apps/bnc_assembly.py derives it from the config --
    for replaying images saved without one (data/grasp_images). The app's own, saved with each fit, is
    what a replay of its .npz uses."""
    from .. import tool_frames
    from ..skills.pick import belief_offset, fingertip_in_connector, held_belief, offset_belief
    from ..transforms import from_cfg, inverse
    from . import pose6d as P6
    init = cfg.get_path('estimation.initial_connector_in_fingertip')
    init_name = cfg.get_path('estimation.initial_connector_frame')
    if init_name:
        T_ftip_conn = inverse(T_tool0_fingertip) @ tool_frames.load_frames(cfg)[init_name]
    else:
        T_ftip_conn = from_cfg(init) if init else from_cfg(cfg.section('junction_in_fingertip'))
    grip = fingertip_in_connector(cfg)
    catalogue = np.array(T_ftip_conn, dtype=float)
    T_ftip_conn = held_belief(T_ftip_conn, from_cfg(cfg.section('junction_in_fingertip')), grip)
    T_jc = inverse(from_cfg(cfg.section('junction_in_fingertip'))) @ catalogue
    part = P6.plug_part(cfg)
    if str(cfg.get('pose_method', 'sam3')) == 'pose6d' and P6.SYMMETRY[part] == 'axis_z':
        T_jc[1:3, 3] = 0.0
        T_ftip_conn = inverse(grip) @ T_jc
    T_ftip_conn = offset_belief(T_ftip_conn, belief_offset(cfg))
    return np.asarray(T_tool0_fingertip, dtype=float) @ P6.plug_from_belief(T_ftip_conn, T_jc, part)


def _latest_K(repo):
    """The wrist camera's intrinsics from the newest saved 6D view (the same camera and mode)."""
    import glob
    import os
    views = sorted(glob.glob(os.path.join(repo, 'data', 'pose6d', '*', 'view_*.npz')), key=os.path.getmtime)
    if not views:
        return None
    with np.load(views[-1]) as z:
        return z['K'] if 'K' in z.files else None


def main(argv=None):
    """Replay the in-hand CAD fit on saved inputs, with the config's `in_hand_fit:` settings:

        python -m cable_assembly.perception.in_hand_fit [--cell iiwa14_bench] [--set k=v ...] FILE...

    FILE: a .npz the app saved with a fit (data/in_hand_fit/*.npz: its frames, belief and frames of
    reference, replayed exactly), or a wrist image (data/grasp_images/*/grasp_*.png: the belief from the
    config, the camera's K from the newest saved 6D view, the gripper's counts from its name). Prints each fit; draws it beside the input
    (<name>_fit.png, or in --out)."""
    import argparse
    import os
    import re
    import types

    import cv2

    from .. import config as urconfig
    from ..robot import tool_geometry
    from . import pose6d as P6
    p = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+')
    p.add_argument('--cell', default='iiwa14_bench')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--out', default=None, help='a folder for the drawings (default: beside each input)')
    p.add_argument('--no-counts', action='store_true', help='the gap from the image, not the gripper\'s counts')
    a = p.parse_args(argv)
    cfg = urconfig.load('bnc_assembly', [f'cell={a.cell}'] + a.set)
    c = dict(cfg.section('in_hand_fit') or {}) if cfg.get('in_hand_fit') else {}
    part = P6.plug_part(cfg)
    mesh = P6.cad_mesh(cfg, part)
    if mesh is None:
        print(f'no CAD for {part}')
        return 2
    verts = np.asarray(tool_geometry.read_ply(mesh[0])[0], dtype=float) * mesh[1]
    robot = None
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    n_ok = 0
    for path in a.files:
        if path.endswith('.npz'):
            frames, T_prior, T_tool0_cam, T_tool0_ftip, counts = load_case(path)
        else:
            if robot is None:                    # the config's frames, without the arm
                from ..robot import Robot
                rc = urconfig.load('bnc_assembly', [f'cell={a.cell}', 'robot.dry_run=true', 'live_view.enabled=false',
                                                    'robot.world.enabled=false'] + a.set)
                robot = Robot(rc, with_gripper=False, needs=())
                K = _latest_K(repo)
                if K is None:
                    print('no saved 6D view to take the camera\'s K from')
                    return 2
            img = cv2.imread(path)
            if img is None:
                print(f'{path}: not an image')
                continue
            frames = [types.SimpleNamespace(color=img, K=K, depth=None)]
            m = re.search(r'(\d+)counts', os.path.basename(path))       # the grasp recorder's names
            counts = int(m.group(1)) if m else None
            T_tool0_cam, T_tool0_ftip = robot.T_tool0_cam, robot.T_tool0_fingertip
            T_prior = prior_from_cfg(cfg, T_tool0_ftip)
        fitter = InHandCadFit.from_cfg(c, verts, T_tool0_cam, T_tool0_ftip)
        if a.no_counts:
            counts = None
        r = fitter.refine(frames, T_prior, counts)
        n_ok += bool(r['ok'])
        print(f"{path}: {'OK' if r['ok'] else 'NOT USED (' + r['why'] + ')'} -- slide {1000 * r['slide']:+.1f} mm, "
              f"shift {1000 * r['shift']:+.1f} mm, pivot {np.degrees(r['pivot']):+.1f} deg, IoU {r['iou']:.2f}; "
              f"pads {1000 * r['gap']:.1f} mm apart (from the {r['gap_from']}; the image's edges {1000 * r['gap_image']:.1f}), "
              f"their CAD +-{r['pad_residual_px']:.1f} px, image {list(r['image_shift'])} px")
        stem = os.path.splitext(os.path.basename(path))[0]
        folder = a.out or os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f'{stem}_fit.png'), fitter.overlay(frames[0].color, frames[0].K, T_prior, r))
    print(f'{n_ok}/{len(a.files)} trusted')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
