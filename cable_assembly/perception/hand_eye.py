"""HAND-EYE -- where the camera sits on the flange (tool0 -> color optical frame), solved from
views of ArUco markers lying still on the bench.

THE CONSTRAINT. A marker that does not move has the same base pose in every view:

    T_base_tool0[i] @ X @ T_cam_marker[i] = T_base_marker          for every view i

X is the unknown; the arm's forward kinematics gives T_base_tool0, the image gives T_cam_marker.
Rotating the camera about two different axes pins X down -- a set of views that only ever turns
about ONE axis leaves X free to slide along it, which `solve` reports rather than hides.

TWO STAGES.
  1. cv2.calibrateHandEye (Park) on the per-view marker poses: closed form, close but noisy --
     a single small marker's PnP is weakest in exactly its tilt.
  2. A JOINT FIT IN PIXELS: X and every marker's base pose together, minimising the reprojection
     of every detected corner in every view. Each corner counts once, so the per-view tilt noise
     averages down instead of being baked in by stage 1.

WHAT IT REPORTS besides X:
  * per-view reprojection RMS (px) -- a view far above the rest is a blurred or misdetected one;
  * the SCATTER of each marker's base pose when every view is solved ALONE through X, in mm and
    deg -- how well one X explains all the views, in the units a grasp misses by;
  * each marker's base pose -- a marker lying on the bench gives the BENCH HEIGHT for free.

`render_markers` draws markers into a synthetic image, so a dry run (and the tests) exercise the
whole chain -- detection, PnP, the solve -- against a known X.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..transforms import inverse


def make_detector(dictionary):
    """An ArUco detector with SUBPIXEL corner refinement -- the default detector stops at the
    contour's pixel corners, and the hand-eye is only as good as its corners."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_corners(detector, image):
    """{marker_id: (4, 2) float64 pixel corners, ArUco order TL, TR, BR, BL}."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for c, i in zip(corners, ids.flatten())}


def object_points(size_m):
    """The four corners in the marker frame: TL, TR, BR, BL, +Z out of the printed face (the
    same convention as perception/aruco.py)."""
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def marker_poses(corners, size_m, K, D):
    """BOTH T_cam_marker solutions of a square's four corners, best reprojection first.

    A small marker seen nearly face-on has two poses that reproject almost equally well -- the
    true tilt and its mirror, several degrees apart -- and the better-reprojecting one is the
    wrong one often enough to matter. `solve` keeps both and lets the other views decide."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        object_points(size_m), np.asarray(corners, dtype=float), K, D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for k in np.argsort(np.ravel(errs))[:n]:
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvecs[k])[0]
        T[:3, 3] = np.ravel(tvecs[k])
        out.append(T)
    return out


def _angle(Ta, Tb):
    return Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).magnitude()


def _vec(T):
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def _mat(v):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    T[:3, 3] = v[3:6]
    return T


def _mean_pose(Ts):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_matrix([t[:3, :3] for t in Ts]).mean().as_matrix()
    T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
    return T


def _project(K, D, T_cam_obj, pts):
    rvec = cv2.Rodrigues(T_cam_obj[:3, :3])[0]
    img, _ = cv2.projectPoints(pts, rvec, T_cam_obj[:3, 3], K, D)
    return img.reshape(-1, 2)


def rotation_spread(T_base_tool0s):
    """(span_deg, second_axis_deg): the largest rotation between any view and the first, and the
    second singular value of the views' rotations relative to the first (deg). The second number
    is the one that matters -- near 0 means every view turned about ONE axis."""
    R0 = Rotation.from_matrix(T_base_tool0s[0][:3, :3])
    rv = np.array([(R0.inv() * Rotation.from_matrix(T[:3, :3])).as_rotvec()
                   for T in T_base_tool0s[1:]])
    if len(rv) == 0:
        return 0.0, 0.0
    span = float(np.degrees(np.max(np.linalg.norm(rv, axis=1))))
    sv = np.linalg.svd(rv, compute_uv=False) / np.sqrt(len(rv))
    return span, float(np.degrees(sv[1])) if len(sv) > 1 else 0.0


def solve(views, K, D, sizes, max_view_rms_px=1.5):
    """X = T_tool0_cam from `views` = [{'T_base_tool0': 4x4, 'corners': {id: (4, 2)}}, ...].

    `sizes` = {marker_id: side m}; markers without a size are ignored. A view whose RMS is above
    `max_view_rms_px` (and 3x the median) after the first fit is dropped and the fit repeated once.

    Returns a dict: X, X_closed_form, markers {id: T_base_marker}, view_rms_px (None for dropped
    views), dropped, scatter {id: (rms_mm, max_mm, rms_deg, max_deg)}, rms_px, span_deg,
    second_axis_deg."""
    K, D = np.asarray(K, dtype=float), np.asarray(D, dtype=float)
    obs = [{m: np.asarray(c, dtype=float) for m, c in v['corners'].items() if m in sizes}
           for v in views]
    A = [np.asarray(v['T_base_tool0'], dtype=float) for v in views]
    cands = [{m: marker_poses(c, sizes[m], K, D) for m, c in o.items()} for o in obs]
    obs = [{m: c for m, c in o.items() if cands[i][m]} for i, o in enumerate(obs)]
    counts = {}
    for o in obs:
        for m in o:
            counts[m] = counts.get(m, 0) + 1
    if not counts:
        raise ValueError('no marker of a known size was detected in any view')
    ref = max(counts, key=counts.get)
    use = [i for i, o in enumerate(obs) if ref in o]
    if len(use) < 3:
        raise ValueError(f'marker {ref} is in only {len(use)} views; the hand-eye needs >= 3')
    # A marker in one or two views -- one at the frame's edge, on a sheet -- pins nothing and
    # fits loosely: it would add noise, and its scatter would read as a bad calibration.
    counts = {m: n for m, n in counts.items() if n >= 3}
    obs = [{m: c for m, c in o.items() if m in counts} for o in obs]
    # Degenerate, not merely weak: the closed form returns a mirror image, not a worse answer.
    _span, second = rotation_spread([A[i] for i in use])
    if second < 1.0:
        raise ValueError(f'the views of marker {ref} turn about one axis only (second axis '
                         f'{second:.2f} deg), which leaves the mount free along it -- tilt the '
                         'camera in two directions')

    # ---- 1. closed form on the best-seen marker, then the PnP mirror ambiguity resolved ----
    pnp = [{m: cands[i][m][0] for m in o} for i, o in enumerate(obs)]

    def closed_form():
        R, t = cv2.calibrateHandEye([A[i][:3, :3] for i in use], [A[i][:3, 3] for i in use],
                                    [pnp[i][ref][:3, :3] for i in use],
                                    [pnp[i][ref][:3, 3] for i in use],
                                    method=cv2.CALIB_HAND_EYE_PARK)
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = R, t.ravel()
        return X

    def medoid(Ts):
        return Ts[int(np.argmin([sum(_angle(Ta, Tb) for Tb in Ts) for Ta in Ts]))]

    # Twice: a mirrored view skews X, X picks the medoid, the medoid re-picks every view's
    # solution, and the closed form is re-run on the corrected set.
    X0 = closed_form()
    for _ in range(2):
        for m in counts:
            seen = [i for i, o in enumerate(obs) if m in o]
            mid = medoid([A[i] @ X0 @ pnp[i][m] for i in seen])
            for i in seen:
                pnp[i][m] = min(cands[i][m], key=lambda T, i=i: _angle(A[i] @ X0 @ T, mid))
        X0 = closed_form()

    def fit(active, X_init):
        markers = sorted({m for i in active for m in obs[i]})
        M_init = {m: _mean_pose([A[i] @ X_init @ pnp[i][m] for i in active if m in obs[i]])
                  for m in markers}
        pts = {m: object_points(sizes[m]) for m in markers}

        def residuals(p):
            X = _mat(p[:6])
            Xinv = inverse(X)
            out = []
            for i in active:
                Ainv = inverse(A[i])
                for m, c in obs[i].items():
                    M = _mat(p[6 + 6 * markers.index(m):12 + 6 * markers.index(m)])
                    out.append((_project(K, D, Xinv @ Ainv @ M, pts[m]) - c).ravel())
            return np.concatenate(out)

        p0 = np.concatenate([_vec(X_init)] + [_vec(M_init[m]) for m in markers])
        sol = least_squares(residuals, p0, loss='soft_l1', f_scale=1.0, x_scale='jac')
        X = _mat(sol.x[:6])
        M = {m: _mat(sol.x[6 + 6 * k:12 + 6 * k]) for k, m in enumerate(markers)}
        rms = {}
        for i in active:
            r = [(_project(K, D, inverse(X) @ inverse(A[i]) @ M[m], pts[m]) - c).ravel()
                 for m, c in obs[i].items()]
            rms[i] = float(np.sqrt(np.mean(np.concatenate(r) ** 2)))
        return X, M, rms

    active = [i for i, o in enumerate(obs) if o]
    X, M, rms = fit(active, X0)
    med = float(np.median(list(rms.values())))
    dropped = [i for i in active if rms[i] > max(max_view_rms_px, 3.0 * med)]
    if dropped and len(active) - len(dropped) >= 3:
        active = [i for i in active if i not in dropped]
        X, M, rms = fit(active, X)
    else:
        dropped = []

    scatter = {}
    for m, Tm in M.items():
        d_mm, d_deg = [], []
        for i in active:
            if m in obs[i]:
                Ti = A[i] @ X @ pnp[i][m]
                d_mm.append(1000.0 * np.linalg.norm(Ti[:3, 3] - Tm[:3, 3]))
                d_deg.append(np.degrees((Rotation.from_matrix(Tm[:3, :3]).inv()
                                         * Rotation.from_matrix(Ti[:3, :3])).magnitude()))
        scatter[m] = (float(np.sqrt(np.mean(np.square(d_mm)))), float(np.max(d_mm)),
                      float(np.sqrt(np.mean(np.square(d_deg)))), float(np.max(d_deg)))
    span, second = rotation_spread([A[i] for i in active])
    all_rms = [rms[i] for i in active]
    return dict(X=X, X_closed_form=X0, markers=M, scatter=scatter, dropped=dropped,
                view_rms_px=[rms.get(i) for i in range(len(views))],
                rms_px=float(np.sqrt(np.mean(np.square(all_rms)))),
                span_deg=span, second_axis_deg=second)


def render_markers(K, width, height, markers, dictionary, background=170):
    """A synthetic BGR image of `markers` = [(marker_id, side_m, T_cam_marker), ...], each with a
    white quiet zone, on a grey background. Pinhole, no distortion; markers behind the camera
    are skipped."""
    img = np.full((height, width), background, dtype=np.uint8)
    side = 240
    quiet = side // 4
    full = side + 2 * quiet
    for mid, size_m, T in markers:
        tag = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(dictionary, int(mid), side),
                                 quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
        h = float(size_m) / 2.0 * full / side
        P = (T[:3, :3] @ object_points(2.0 * h).T).T + T[:3, 3]
        if np.any(P[:, 2] <= 0.0):
            continue
        uv = (K @ P.T).T
        dst = (uv[:, :2] / uv[:, 2:3]).astype(np.float32)
        src = np.float32([[0, 0], [full, 0], [full, full], [0, full]])
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(tag, H, (width, height), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full_like(tag, 255), H, (width, height))
        img[mask > 127] = warped[mask > 127]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
