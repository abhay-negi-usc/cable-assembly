"""THE CONNECTOR IN THE HAND, FROM THE WRIST CAMERA'S COLOUR IMAGE -- no depth, no 6D search.

The plug between the fingers is ~0.2 m from the camera, nearer than the D415's depth read (0.3 m at
848x480; the iiwa's camera until 2026-11-10 -- the D405 reads it), and RAG-6D's full search is slow and blind to what the grasp already fixes. But the
camera rides on the flange, so the plug's pose in the image IS its pose in the hand, with no
arm or hand-eye error in between; its CAD gives its size, so its size in pixels gives its range;
and a two-finger pinch leaves it free in only a few ways. This fits the two that matter, starting
from the belief (where the commanded grasp puts it):

    slide   along the plug's own axis (CAD +z), m -- the jaws closing at a pitch push it back or
            forward; ~2.5 px per mm of the tip at the iiwa bench's 70 deg pickup
    pivot   about the pinch axis (tool0 y, the fingers' closing direction) through the fingertip
            point, rad -- the one turn a pinch does not resist

Its position across the closing direction is the jaws', and its spin about its own axis does not
show on a round plug (nor matter to the fit).

THE IMAGE SIDE: the plug's pixels by GrabCut (OpenCV), seeded from the belief's silhouette grown
by the search range, in a box round it -- only ABOVE THE JAWS: rows past the jaws' top edge
(`jaw_clear_m` beyond the fingertip, projected) are left out of both sides, since the black boot
between the black fingers cannot be told apart. THE MODEL SIDE: the CAD as a body of revolution --
its radius profile along z -- drawn as one projected frustum per step: exact for a round plug, and
~1 ms, so a coarse grid then a fine one over (slide, pivot) costs well under a second a frame.
The score is the two masks' IoU.

Several frames, each fitted alone; the median is the answer, and it is TRUSTED only if the best
IoU is high enough (`min_iou`), the frames agree (`max_spread_*`), and it is not at the edge of
the search (that is "further than searched", not a measurement).

PIVOT IS NOT FITTED BY DEFAULT (`pivot_range_rad` 0). The camera sees the plane the plug pivots
in edge-on, so a pivot and a slide move what shows above the jaws -- the collar and the nose --
alike: on a clean synthetic frame, slide -3 mm + pivot -9.5 deg fitted a plug placed exactly at
IoU 0.98. Fitting both trades one for the other; fitting the slide alone puts the NOSE -- what
has to find the socket -- where the image shows it, and leaves the axis the belief's. A pivot
needs a second view (the D415's IR imager, 55 mm across) to tell apart.

THE PRIOR'S AXIS MUST PASS THROUGH THE PINCH POINT (`max_axis_offset_m`). The camera sees the
plane of the plug's axis nearly edge-on, so a slide along the axis and a shift across it in that
plane look alike (on the 2026-11-02 grasp: +10 mm along vs 7.5 mm across, the same IoU). The
pick aims the fingertip at the axis, so the axis runs through the pinch, and slide + pivot about
it keep it there; a prior off it (the UR10e's belief_offset, 10 mm across) fits as well but
somewhere else. From an on-axis prior the slide is sharp: IoU 0.955 at the best, ~0.9 a
millimetre either side.
"""

import numpy as np

# The plug's radius profile is sampled every this along its axis, m.
PROFILE_STEP_M = 0.0005
# Points round each circle of the profile, projected.
CIRCLE_POINTS = 32


def radius_profile(vertices, step=PROFILE_STEP_M):
    """A round part's CAD vertices (N, 3), m, axis along z -> (z (K,), radius (K,)): the largest
    distance from the axis in each `step`-long slice, empty slices filled from their neighbours."""
    v = np.asarray(vertices, dtype=float)
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    edges = np.arange(z0, z1 + step, step)
    k = np.clip(np.digitize(v[:, 2], edges) - 1, 0, len(edges) - 2)
    r = np.zeros(len(edges) - 1)
    np.maximum.at(r, k, np.hypot(v[:, 0], v[:, 1]))
    z = 0.5 * (edges[:-1] + edges[1:])
    have = r > 0
    r = np.interp(z, z[have], r[have])
    return z, r


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def moved(T_tool0_plug, slide, pivot, pivot_point):
    """The plug's tool0 pose slid `slide` along its own z and turned `pivot` about tool0 y through
    `pivot_point` (tool0, m)."""
    T = np.array(T_tool0_plug, dtype=float)
    T[:3, 3] += T[:3, 2] * slide
    R = _rot_y(pivot)
    p = np.asarray(pivot_point, dtype=float)
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ (T[:3, 3] - p) + p
    return out


class Silhouette:
    """A round part's silhouette in an image, from its radius profile."""

    def __init__(self, z, r, n=CIRCLE_POINTS):
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        self.rings = np.stack([np.outer(r, np.cos(a)), np.outer(r, np.sin(a)),
                               np.repeat(np.asarray(z)[:, None], n, axis=1)], axis=-1)  # (K, n, 3)

    def project(self, T_cam_obj, K):
        """(K, n, 2) pixel coordinates of every ring, and whether all of it is in front."""
        P = self.rings @ T_cam_obj[:3, :3].T + T_cam_obj[:3, 3]
        front = bool(np.all(P[..., 2] > 1e-3))
        uv = P[..., :2] / np.maximum(P[..., 2:3], 1e-3) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        return uv, front

    def mask(self, T_cam_obj, K, shape, offset=(0, 0)):
        """A uint8 mask of `shape` (rows, cols) whose (0, 0) is image pixel `offset` (x, y)."""
        import cv2
        uv, front = self.project(T_cam_obj, K)
        m = np.zeros(shape, np.uint8)
        if not front:
            return m
        uv = uv - np.asarray(offset, dtype=float)
        for i in range(len(uv) - 1):
            hull = cv2.convexHull(np.vstack([uv[i], uv[i + 1]]).astype(np.float32))
            cv2.fillConvexPoly(m, np.rint(hull).astype(np.int32), 1)
        return m


class InHandRefiner:
    """Fits (slide, pivot) of the held plug from wrist-camera colour frames. See the module docstring.

    T_tool0_cam   the camera on the flange (hand_eye)
    pivot_point   tool0, m: where the fingers pinch it (the fingertip point)
    jaw_clear_m   the jaws' top edge, this far beyond the pivot point along tool0 z
    """

    def __init__(self, vertices, T_tool0_cam, pivot_point, slide_range_m=0.025, pivot_range_rad=0.0,
                 jaw_clear_m=0.015, min_iou=0.75, max_spread_m=0.002, max_spread_rad=np.radians(2.0),
                 max_axis_offset_m=0.004):
        self.sil = Silhouette(*radius_profile(vertices))
        self.T_tool0_cam = np.asarray(T_tool0_cam, dtype=float)
        self.T_cam_tool0 = np.linalg.inv(self.T_tool0_cam)
        self.pivot_point = np.asarray(pivot_point, dtype=float)
        self.slide_range, self.pivot_range = float(slide_range_m), float(pivot_range_rad)
        self.jaw_clear = float(jaw_clear_m)
        self.min_iou, self.max_spread_m, self.max_spread_rad = float(min_iou), float(max_spread_m), float(max_spread_rad)
        self.max_axis_offset = float(max_axis_offset_m)

    @classmethod
    def from_cfg(cls, c, vertices, T_tool0_cam, T_tool0_fingertip):
        """From an `in_hand_rgb:` block (mm / deg)."""
        return cls(vertices, T_tool0_cam, np.asarray(T_tool0_fingertip, dtype=float)[:3, 3],
                   slide_range_m=float(c.get('slide_range_mm', 25.0)) / 1000.0,
                   pivot_range_rad=np.radians(float(c.get('pivot_range_deg', 0.0))),
                   jaw_clear_m=float(c.get('jaw_clear_mm', 15.0)) / 1000.0,
                   min_iou=float(c.get('min_iou', 0.75)),
                   max_spread_m=float(c.get('max_spread_mm', 2.0)) / 1000.0,
                   max_spread_rad=np.radians(float(c.get('max_spread_deg', 2.0))),
                   max_axis_offset_m=float(c.get('max_axis_offset_mm', 4.0)) / 1000.0)

    # ------------------------------------------------------------------ geometry
    def axis_offset(self, T_tool0_plug):
        """How far the pinch point is from the plug's axis, m."""
        d = self.pivot_point - T_tool0_plug[:3, 3]
        return float(np.linalg.norm(d - (d @ T_tool0_plug[:3, 2]) * T_tool0_plug[:3, 2]))

    def _cam(self, T_tool0_plug):
        return self.T_cam_tool0 @ T_tool0_plug

    def jaw_row(self, K):
        """The image row of the jaws' top edge: rows below it (larger) are left out."""
        p = self.T_cam_tool0 @ np.append(self.pivot_point + [0.0, 0.0, self.jaw_clear], 1.0)
        return int(np.floor(K[1, 1] * p[1] / p[2] + K[1, 2]))

    def roi(self, T_tool0_plug, K, shape):
        """(x0, y0, x1, y1): a box round every silhouette the search can reach, above the jaws."""
        pts = []
        for s in (-self.slide_range, 0.0, self.slide_range):
            for a in (-self.pivot_range, 0.0, self.pivot_range):
                uv, front = self.sil.project(self._cam(moved(T_tool0_plug, s, a, self.pivot_point)), K)
                if front:
                    pts.append(uv.reshape(-1, 2))
        pts = np.vstack(pts)
        pad = 20
        x0, y0 = np.floor(pts.min(axis=0)).astype(int) - pad
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + pad
        h, w = shape[:2]
        y1 = min(y1, self.jaw_row(K))
        return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)

    # ------------------------------------------------------------------ image side
    def segment(self, bgr, T_tool0_plug, K, box):
        """The plug's pixels in `box` (uint8 0/1), by GrabCut seeded from the belief's silhouettes
        over the whole search: inside every one of them, probably the plug; inside none, probably
        not; the box's border, not."""
        import cv2
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(bgr[y0:y1, x0:x1])
        shape = crop.shape[:2]
        union = np.zeros(shape, np.uint8)
        core = np.ones(shape, np.uint8)
        for s in np.linspace(-self.slide_range, self.slide_range, 5):
            for a in np.linspace(-self.pivot_range, self.pivot_range, 3):
                m = self.sil.mask(self._cam(moved(T_tool0_plug, s, a, self.pivot_point)), K, shape, (x0, y0))
                union |= m
                core &= m
        gc = np.where(union > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        gc[core > 0] = cv2.GC_FGD if core.sum() > 50 else cv2.GC_PR_FGD
        gc[:2, :], gc[:, :2], gc[:, -2:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD   # not the bottom: the jaws' edge
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        # The largest piece only: a stray dark speck on the bench is not the plug.
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 1:
            fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        return fg

    # ------------------------------------------------------------------ the fit
    def _iou(self, T_tool0_plug, s, a, K, target, box):
        x0, y0, x1, y1 = box
        m = self.sil.mask(self._cam(moved(T_tool0_plug, s, a, self.pivot_point)), K, target.shape, (x0, y0))
        union = np.count_nonzero(m | target)
        return np.count_nonzero(m & target) / union if union else 0.0

    def fit_frame(self, bgr, K, T_tool0_plug):
        """One frame: {'slide', 'pivot', 'iou', 'at_edge', 'box', 'mask'}."""
        K = np.asarray(K, dtype=float)
        box = self.roi(T_tool0_plug, K, bgr.shape)
        if box[2] - box[0] < 10 or box[3] - box[1] < 10:
            return None
        target = self.segment(bgr, T_tool0_plug, K, box)
        best = (-1.0, 0.0, 0.0)
        for s in np.arange(-self.slide_range, self.slide_range + 1e-9, 0.002):
            for a in np.arange(-self.pivot_range, self.pivot_range + 1e-9, np.radians(2.0)):
                best = max(best, (self._iou(T_tool0_plug, s, a, K, target, box), s, a))
        _, s0, a0 = best
        for s in s0 + np.arange(-0.002, 0.00201, 0.0005):
            for a in a0 + np.radians(np.arange(-2.0, 2.01, 0.5)):
                if abs(s) <= self.slide_range + 1e-9 and abs(a) <= self.pivot_range + 1e-9:
                    best = max(best, (self._iou(T_tool0_plug, s, a, K, target, box), s, a))
        iou, s, a = best
        at_edge = abs(s) > self.slide_range - 0.001 or (self.pivot_range > 0.0
                                                       and abs(a) > self.pivot_range - np.radians(1.0))
        return {'slide': float(s), 'pivot': float(a), 'iou': float(iou), 'at_edge': bool(at_edge),
                'box': box, 'mask': target}

    def refine(self, frames, T_tool0_plug):
        """Frames (each with .color BGR and .K) -> a result dict:
            ok          trusted (the module docstring's tests)
            why         what failed, when not ok
            slide, pivot, iou    the median over the frames that fitted
            T_tool0_plug         the belief moved by them (the prior's, when not ok)
            per_frame            each frame's fit (without masks)"""
        T_tool0_plug = np.asarray(T_tool0_plug, dtype=float)
        off = self.axis_offset(T_tool0_plug)
        if off > self.max_axis_offset:
            return {'ok': False, 'slide': 0.0, 'pivot': 0.0, 'iou': 0.0, 'T_tool0_plug': T_tool0_plug,
                    'per_frame': [], 'fits': [], 'axis_offset': off,
                    'why': (f'the belief\'s axis misses the pinch point by {1000 * off:.1f} mm (> '
                            f'{1000 * self.max_axis_offset:.1f}): a slide fitted from there is not the plug\'s -- '
                            'check pickup.belief_offset')}
        fits = []
        for f in frames:
            r = self.fit_frame(f.color, f.K, T_tool0_plug)
            if r is not None:
                fits.append(r)
        out = {'ok': False, 'why': '', 'slide': 0.0, 'pivot': 0.0, 'iou': 0.0, 'T_tool0_plug': T_tool0_plug,
               'per_frame': [{k: v for k, v in r.items() if k != 'mask'} for r in fits], 'fits': fits,
               'axis_offset': off}
        if not fits:
            out['why'] = 'the plug is not in view above the jaws'
            return out
        s = float(np.median([r['slide'] for r in fits]))
        a = float(np.median([r['pivot'] for r in fits]))
        iou = float(np.median([r['iou'] for r in fits]))
        out.update(slide=s, pivot=a, iou=iou)
        spread_s = float(np.ptp([r['slide'] for r in fits]))
        spread_a = float(np.ptp([r['pivot'] for r in fits]))
        if iou < self.min_iou:
            out['why'] = f'the CAD matches the image poorly (IoU {iou:.2f} < {self.min_iou:.2f})'
        elif sum(r['at_edge'] for r in fits) * 2 > len(fits):
            out['why'] = 'the best fit is at the edge of the search: further off than searched'
        elif spread_s > self.max_spread_m or spread_a > self.max_spread_rad:
            out['why'] = (f'the frames disagree ({1000 * spread_s:.1f} mm, {np.degrees(spread_a):.1f} deg apart): '
                          'the plug moved, or the image is ambiguous')
        else:
            out['ok'] = True
            out['T_tool0_plug'] = moved(T_tool0_plug, s, a, self.pivot_point)
        return out

    def overlay(self, bgr, K, T_prior, result):
        """A diagnostic image: the belief (magenta), the fit (green), the segmented plug (cyan tint),
        the jaws' cutoff (grey line)."""
        import cv2
        img = bgr.copy()
        K = np.asarray(K, dtype=float)
        fits = result.get('fits') or []
        if fits:
            x0, y0, x1, y1 = fits[0]['box']
            tint = img[y0:y1, x0:x1]
            tint[fits[0]['mask'] > 0] = (0.5 * tint[fits[0]['mask'] > 0] + [128, 128, 0]).astype(np.uint8)
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
        for T, col in ((T_prior, (255, 0, 255)),
                       (moved(T_prior, result['slide'], result['pivot'], self.pivot_point), (0, 255, 0))):
            m = self.sil.mask(self._cam(T), K, img.shape[:2])
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, col, 2)
        y = self.jaw_row(K)
        cv2.line(img, (0, y), (img.shape[1], y), (128, 128, 128), 1)
        text = (f"in-hand RGB: slide {1000 * result['slide']:+.1f} mm  pivot {np.degrees(result['pivot']):+.1f} deg  "
                f"IoU {result['iou']:.2f}  {'OK' if result['ok'] else 'NOT USED: ' + result['why']}")
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if result['ok'] else (0, 0, 255), 2)
        return img
