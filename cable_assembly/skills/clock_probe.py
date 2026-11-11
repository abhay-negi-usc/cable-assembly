"""THE SOCKET'S AXIS AND THE IN-HAND ERROR FROM A CLOCKING TURN, by proprioception AND the wrench (the user,
2026-11-06: "a proprioception and wrench based method for estimating misalignment between the connector and
socket via connector clocking").

THE IDEA. Engaged, the plug's sleeve is on the socket's barrel: whatever the arm does, the plug's axis stays
on the socket's (to its 0.1 mm clearance). Turn the wrist about the axis the ESTIMATE gives, under a light
push, and the misalignment cannot show up as motion of the plug -- it shows up as

  1. motion of the HAND off the commanded screw: the controller's give (the force null), read by the joints;
  2. deflection of the GRASP: the plug turned and shifted in the pads, which the joints cannot see -- but the
     wrench can: deflection = compliance x the contact wrench.

The plug's true pose is the hand (measured) . the in-hand transform (unknown, constant) . the grasp deflection
(compliance x measured wrench). Holding it on the socket's axis (unknown, fixed in the base) at every sample
gives equations LINEAR in the unknowns, once the measured roll turns the in-hand terms:

  lateral   q_k + R_k (d + s tau) + [R_k u_k]_xy = b + beta z_k
  tilt      t_k + R_k tau + [R_k (phi_k x z)]_xy  = beta

  q_k, z_k  where the belief puts the constraint point (the pivot, s along the connector axis), in the
            estimated axis frame; t_k the believed connector axis's lateral slope there; R_k its turn
  b, beta   the socket's TRUE axis: lateral offset at the estimated axis frame's origin, and slope
  d, tau    the in-hand error: the true connector origin, and its axis's slope, in the believed connector's
            lateral axes
  u_k, phi_k the grasp's deflection at the pivot -- c_lin f + c_rot (tau_g x (pivot - g)), c_rot tau_g, with
            tau_g the moment about the grasp point g. Known compliances move it to the left; unknown ones
            are two more columns (they are identifiable when the force varies along the turn).

WHY BOTH. Proprioception alone (fit_axis_offset, the 02:33 axis probe) assumes all the give is the hand's:
whatever the grasp absorbed -- (contact force / grasp stiffness), ~0.3-1 mm at 5-10 N on the strain relief,
and 30 mm x (moment / ~10 Nm/rad) of turn -- is fitted as geometry, and it rotates with the wrist, so it
biases the IN-HAND estimate and the socket axis both. The wrench alone needs the grasp's compliance and a
stiff hand. Together, any split between the controller's give and the grasp's is handled, and the compliance
itself can be fitted.

WHAT IT CANNOT SEE: the axial offset and the roll -- turning about the axis leaves both unchanged. And it needs
the plug ENGAGED throughout (the constraint is the whole method): a trace that fits badly is the sign it was
not, and the gate (`ProbeFit.trusted`) refuses it.

NOR, ON THE BNC, THE TILT (validated in the Newton sim, sim/scripts/newton_bnc_clock_probe.py, 2026-11-06): with
the collar on the pins the plug has ~2 deg of play on the barrel, the grasp turns at 5-20 Nm/rad on the strain
relief, so moments under the null's 0.5 Nm deadband (~3 deg of twist) never reach the hand, and the arm's
moment at the connector is its force error x the 0.2 m flange lever. Fitting beta / tau there came out WORSE
than the prior, even with a perfect sensor. Use use_tilt=False: the lateral terms are then the offsets at the
pivot, which is what the socket constrains. With the errors the user gave (+/-1 mm, +/-5 deg in every DOF, both
socket and grasp), a 60 deg there-and-back turn, the force null on, an iiwa-grade wrench (2 N bias, 1.5 N
flipping with the turn, 1 N noise), 256 worlds, median / p99:
  the socket axis where the plug was held   0.90 / 1.89 mm prior -> 0.33 / 0.74 mm
  the in-hand offset at the pivot           0.78 / 1.44 mm prior -> 0.21 / 0.62 mm (proprioception alone 0.24 / 1.11)
and a 120 deg follow-up turn (stiff in the plane) about the corrected axis carried half the side force of one
about the uncorrected axis (6 vs 13 N median). APPLY IT AS A TRANSLATION ONLY: re-aligning the hand onto an axis
direction the probe does not know tilts the engaged plug by the tilt error (that read worse than no correction).
The wrench's gain over proprioception alone grows with the sensor's accuracy (ideal: 3-4x) and with a stiff
hand (no null: proprioception alone loses most of it). Recommended on the iiwa: use_tilt False, the force only
(compliance (c_lin, 0): no moment term), c_lin calibrated, NOT fitted (the turn-direction bias of the arm's
estimate pulls a fitted compliance stiff), smooth ~0.5 s (keeps the residual gate meaningful), pivot at the
middle of the sleeve-on-barrel engagement.

FRAMES. 4x4 `T_a_b` as everywhere in cable_assembly. The axis is `axis` (0 = x, the app's connector +X;
2 = z, the sim's) of BOTH the estimated socket axis frame and the connector frame; lateral = the other two,
in cyclic order (axis 0: y, z; axis 2: x, y) -- so the app's (y, z) and fit_axis_offset's agree.
Numpy only: it runs on the robot PC (bnc_assembly's axis probe) and in the sim's validation
(sim/scripts/newton_bnc_clock_probe.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _perm(axis):
    """P with columns = the frame's directions of the internal x, y, z (z = the axis): v_frame = P v_internal."""
    i = int(axis) % 3
    cols = [(i + 1) % 3, (i + 2) % 3, i]
    P = np.zeros((3, 3))
    P[cols, [0, 1, 2]] = 1.0
    return P


def _twist(R):
    """The turn of R about its z (swing-twist), rad."""
    return np.arctan2(R[..., 1, 0] - R[..., 0, 1], R[..., 0, 0] + R[..., 1, 1])


@dataclass
class ProbeFit:
    b: np.ndarray            # (2,) m: the socket's true axis off the estimated one, at the axis frame's origin
    beta: np.ndarray         # (2,) the true axis's lateral slope (d lateral / d axial) in the estimated axis frame
    d: np.ndarray            # (2,) m: the true connector origin in the believed connector's lateral axes
    tau: np.ndarray          # (2,) the true connector axis's lateral slope in the believed connector frame
    compliance: np.ndarray   # (2,) (c_lin m/N, c_rot rad/(N m)) used or fitted; nan when no wrench
    std: dict                # 1-sigma of each of the above (same shapes), from the weighted normal equations
    residual_mm: np.ndarray  # (n,) lateral residual per sample
    residual_deg: np.ndarray  # (n,) tilt residual per sample (nan without tilt)
    span_deg: float          # the measured turn's range
    n: int
    cond: float
    weights: np.ndarray      # (n,) the robust weights at the end
    pivot: float
    axis: int
    gates: dict = field(default_factory=dict)

    @property
    def trusted(self):
        return not self.why

    @property
    def why(self):
        """'' when every gate passed, else the first failed one."""
        for k, v in self.gates.items():
            if not v[0]:
                return f'{k}: {v[1]}'
        return ''

    def axis_frame(self, T_axis):
        """The socket's axis as fitted: the estimated frame moved by b at its origin and turned by beta (its origin
        stays at the same height along the axis, its roll stays)."""
        P = _perm(self.axis)
        u = np.array([self.beta[0], self.beta[1], 1.0])
        u /= np.linalg.norm(u)
        z = np.array([0.0, 0.0, 1.0])
        R_tilt = _rot_between(z, u)
        T = np.eye(4)
        T[:3, :3] = P @ R_tilt @ P.T
        T[:3, 3] = P @ np.array([self.b[0], self.b[1], 0.0])
        return np.asarray(T_axis, dtype=float) @ T

    def hand_conn(self, T_hand_conn):
        """The in-hand belief corrected: the connector moved by d and its axis turned onto tau (roll and axial kept)."""
        P = _perm(self.axis)
        u = np.array([self.tau[0], self.tau[1], 1.0])
        u /= np.linalg.norm(u)
        T = np.eye(4)
        T[:3, :3] = P @ _rot_between(np.array([0.0, 0.0, 1.0]), u) @ P.T
        T[:3, 3] = P @ np.array([self.d[0], self.d[1], 0.0])
        return np.asarray(T_hand_conn, dtype=float) @ T

    def summary(self):
        s = (f'socket axis {1e3 * np.linalg.norm(self.b):.2f} mm off (b {np.round(1e3 * self.b, 2).tolist()} mm), '
             f'tilted {np.degrees(np.arctan(np.linalg.norm(self.beta))):.2f} deg; in-hand {1e3 * np.linalg.norm(self.d):.2f} mm '
             f'(d {np.round(1e3 * self.d, 2).tolist()} mm), {np.degrees(np.arctan(np.linalg.norm(self.tau))):.2f} deg; '
             f'residual {np.sqrt(np.mean(self.residual_mm ** 2)):.3f} mm rms over {self.n} samples, {self.span_deg:.0f} deg')
        if np.all(np.isfinite(self.compliance)):
            c_lin, c_rot = self.compliance
            s += f'; grasp {1 / c_lin:.0f} N/m, ' + (f'{1 / c_rot:.1f} Nm/rad' if c_rot > 0 else 'no moment term')
        return s


def _rot_between(a, b):
    """The smallest rotation taking unit a onto unit b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def fit_clock_probe(T_axis, T_hand, T_hand_conn, wrench=None, *, axis=0, wrench_about='hand', grasp_point=None,
                    pivot=0.0, compliance=None, fit_compliance=False, compliance_rel_sigma=1.0,
                    compliance_bounds=(0.25, 4.0), smooth=0, use_tilt=True,
                    sigma=(1e-4, np.radians(0.5)), prior_sigma=(5e-3, np.radians(10.0)), huber=2.0, iterations=5,
                    gates=None):
    """Fit the socket's true axis and the in-hand error to a clocking turn (see the module's docstring).

    T_axis        (4, 4) the estimated socket axis frame, in base (the screw axis the turn went about)
    T_hand        (n, 4, 4) the measured hand (tool0) poses, in base
    T_hand_conn   (4, 4) the in-hand belief, hand -> connector
    wrench        (n, 6) the EXTERNAL wrench on the tool (arm.wrench(): what the contact applies), base axes, the
                  moment about the hand's origin (`wrench_about` 'hand') or the believed connector's ('connector').
                  None: proprioception only.
    grasp_point   (3,) the grasp's compliance centre in the connector frame [m] (the pads' middle)
    pivot         where along the connector axis the lateral constraint holds [m, connector frame]: the middle of
                  the engagement (sleeve on barrel)
    compliance    (c_lin m/N, c_rot rad/(N m)) of the grasp: the nominal; None with a wrench = not compensated
    fit_compliance True / 'lin' / 'rot': fit both / only c_lin / only c_rot (the other held at `compliance`; c_rot 0 =
                  no moment term, for an arm whose moment estimate at the connector is poor -- the iiwa's: its force
                  error x the 0.2 m flange-connector lever) with `compliance` as a prior of relative sigma
                  `compliance_rel_sigma`, within
                  `compliance_bounds` x the nominal (one out of them is pinned at the bound and the rest re-solved:
                  a short turn identifies it poorly, and a negative compliance is nonsense)
    smooth        centred moving average of the wrench over this many samples (0: none). The turn is slow (the
                  quasi-static deflection varies over seconds) and the arm's estimate is noisy: unsmoothed, the noise
                  in the regressor biases a fitted compliance toward zero (regression dilution)
    use_tilt      fit the tilt equations (beta, tau); False: lateral only (beta = tau = 0 assumed)
    sigma         (m, rad) of a sample's lateral and tilt residual: the weighting between them
    prior_sigma   (m, rad) of the weak zero-mean prior on b, d and beta, tau (keeps a short turn well posed)
    huber         robust threshold, in sigmas of the per-sample residual
    gates         dict of limits for `trusted`: min_points, min_span_deg, max_rms_mm, max_b_mm, max_d_mm,
                  max_tilt_deg, max_std_mm (defaults below)
    """
    P = _perm(axis)
    Pt = np.eye(4)
    Pt[:3, :3] = P
    T_axis = np.asarray(T_axis, dtype=float) @ Pt              # internal: z = the axis
    T_hc = np.asarray(T_hand_conn, dtype=float) @ Pt
    T_hand = np.asarray(T_hand, dtype=float).reshape(-1, 4, 4)
    n = len(T_hand)
    inv_A = np.linalg.inv(T_axis)
    C = inv_A[None] @ T_hand @ T_hc[None]                       # believed connector in the axis frame
    R, p = C[:, :3, :3], C[:, :3, 3]
    v = np.array([0.0, 0.0, float(pivot)])
    q = p + R @ v                                              # the believed pivot
    z = q[:, 2]
    t = R[:, 0:2, 2] / R[:, 2:3, 2]                            # believed axis slope
    R2 = R[:, 0:2, 0:2]
    theta = np.unwrap(_twist(R))

    use_w = wrench is not None and compliance is not None
    if use_w:
        W = np.asarray(wrench, dtype=float).reshape(n, 6)
        if smooth and smooth > 1 and n > smooth:
            k = int(smooth)
            pad = np.pad(W, ((k // 2, k - 1 - k // 2), (0, 0)), mode='edge')
            W = np.cumsum(np.vstack([np.zeros((1, 6)), pad]), axis=0)
            W = (W[k:] - W[:-k]) / k
        f_b, m_b = W[:, :3], W[:, 3:]
        R_bc = T_hand[:, :3, :3] @ T_hc[:3, :3]                 # connector axes in base
        o_c = (T_hand @ T_hc)[:, :3, 3]
        if wrench_about == 'hand':
            m_b = m_b + np.cross(T_hand[:, :3, 3] - o_c, f_b)  # -> about the connector origin
        elif wrench_about != 'connector':
            raise ValueError(f'wrench_about {wrench_about!r}: hand or connector')
        f_c = np.einsum('nji,nj->ni', R_bc, f_b)               # connector axes
        m_c = np.einsum('nji,nj->ni', R_bc, m_b)
        g = P.T @ np.asarray(grasp_point if grasp_point is not None else np.zeros(3), dtype=float)
        m_g = m_c - np.cross(g[None], f_c)                     # the moment about the grasp point
        # the deflection at the pivot per unit compliance (connector frame), then in the axis frame
        u_lin = f_c
        u_rot = np.cross(m_g, (v - g)[None])
        L_lin = np.einsum('nij,nj->ni', R, u_lin)[:, 0:2]
        L_rot = np.einsum('nij,nj->ni', R, u_rot)[:, 0:2]
        T_rot = np.einsum('nij,nj->ni', R, np.cross(m_g, np.array([0.0, 0.0, 1.0])[None]))[:, 0:2]
        c_nom = np.asarray(compliance, dtype=float)

    # unknowns: b(2) beta(2) d(2) tau(2) [c_lin c_rot]
    nx = 8 + (2 if use_w and fit_compliance else 0)
    I2 = np.eye(2)
    A_lat = np.zeros((n, 2, nx))
    A_lat[:, :, 0:2] = I2
    A_lat[:, :, 2:4] = z[:, None, None] * I2
    A_lat[:, :, 4:6] = -R2
    A_lat[:, :, 6:8] = -float(pivot) * R2
    y_lat = q[:, 0:2].copy()
    A_tl = np.zeros((n, 2, nx))
    A_tl[:, :, 2:4] = I2
    A_tl[:, :, 6:8] = -R2
    y_tl = t.copy()
    if use_w:
        if fit_compliance:
            A_lat[:, :, 8], A_lat[:, :, 9] = -L_lin, -L_rot
            A_tl[:, :, 9] = -T_rot
        else:
            y_lat += c_nom[0] * L_lin + c_nom[1] * L_rot
            y_tl += c_nom[1] * T_rot
    free = np.ones(nx, dtype=bool)
    if not use_tilt:
        free[2:4] = free[6:8] = False
    x = np.zeros(nx)                                           # fixed entries keep their value here
    if use_w and fit_compliance:
        which = {True: (8, 9), 'lin': (8,), 'rot': (9,)}[fit_compliance]
        for j in (8, 9):
            if j not in which:
                free[j], x[j] = False, c_nom[j - 8]

    s_lat, s_tl = sigma
    pr_lin, pr_ang = prior_sigma
    prior_sd = np.r_[[pr_lin] * 2, [pr_ang] * 2, [pr_lin] * 2, [pr_ang] * 2]
    prior_mu = np.zeros(8)
    if use_w and fit_compliance:
        prior_sd = np.r_[prior_sd, np.where(c_nom > 0, c_nom * compliance_rel_sigma, 1.0)]
        prior_mu = np.r_[prior_mu, c_nom]
    w = np.ones(n)
    lo_c = hi_c = None
    if use_w and fit_compliance:
        lo_c, hi_c = c_nom * compliance_bounds[0], c_nom * compliance_bounds[1]
    for _ in range(max(1, iterations)):
        rows, rhs = [A_lat * (np.sqrt(w)[:, None, None] / s_lat)], [y_lat * (np.sqrt(w)[:, None] / s_lat)]
        if use_tilt:
            rows.append(A_tl * (np.sqrt(w)[:, None, None] / s_tl))
            rhs.append(y_tl * (np.sqrt(w)[:, None] / s_tl))
        A_full = np.concatenate([r.reshape(-1, nx) for r in rows])
        y_full = np.concatenate([r.reshape(-1) for r in rhs])
        fr = free.copy()
        for _k in range(3):                                    # a compliance out of its bounds: pinned there, re-solved
            A = A_full[:, fr]
            y = y_full - A_full[:, ~fr] @ x[~fr]
            Ap = np.diag(1.0 / prior_sd)[fr][:, fr]
            A_all = np.vstack([A, Ap])
            y_all = np.r_[y, (prior_mu / prior_sd)[fr]]
            sol, *_ = np.linalg.lstsq(A_all, y_all, rcond=None)
            x[fr] = sol
            if lo_c is None:
                break
            out = [j for j in (8, 9) if fr[j] and not lo_c[j - 8] <= x[j] <= hi_c[j - 8]]
            if not out:
                break
            for j in out:
                x[j] = float(np.clip(x[j], lo_c[j - 8], hi_c[j - 8]))
                fr[j] = False
        e_lat = y_lat - np.einsum('nij,j->ni', A_lat, x)
        e_tl = y_tl - np.einsum('nij,j->ni', A_tl, x)
        r = np.hypot(*e_lat.T) / s_lat
        if use_tilt:
            r = np.sqrt(r ** 2 + (np.hypot(*e_tl.T) / s_tl) ** 2)
        if huber is None:
            break
        w = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-12))
    free = fr

    N = A_all.T @ A_all
    cov = np.full((nx, nx), np.nan)
    cov[np.ix_(free, free)] = np.linalg.pinv(N)
    # scale by the fit's own residual where it is worse than the stated sigma
    dof = max(1, A.shape[0] - int(free.sum()))
    s2 = max(1.0, float(np.sum((A @ sol - y) ** 2)) / dof)
    sd = np.sqrt(np.diag(cov) * s2)
    comp = (x[8:10] if (use_w and fit_compliance) else (c_nom if use_w else np.full(2, np.nan)))
    fit = ProbeFit(b=x[0:2], beta=x[2:4], d=x[4:6], tau=x[6:8], compliance=np.asarray(comp, dtype=float),
                   std={'b': sd[0:2], 'beta': sd[2:4], 'd': sd[4:6], 'tau': sd[6:8],
                        'compliance': sd[8:10] if nx > 8 else np.full(2, np.nan)},
                   residual_mm=np.hypot(*e_lat.T) * 1e3,
                   residual_deg=np.degrees(np.hypot(*e_tl.T)) if use_tilt else np.full(n, np.nan),
                   span_deg=float(np.degrees(np.ptp(theta))) if n else 0.0, n=n,
                   cond=float(np.linalg.cond(A_all)), weights=w, pivot=float(pivot), axis=int(axis))
    g_ = {'min_points': 40, 'min_span_deg': 30.0, 'max_rms_mm': 0.5, 'max_b_mm': 6.0, 'max_d_mm': 6.0,
          'max_tilt_deg': 8.0, 'max_std_mm': 0.5}
    g_.update(gates or {})
    rms = float(np.sqrt(np.mean(fit.residual_mm ** 2))) if n else np.inf
    tilt = float(np.degrees(np.arctan(max(np.linalg.norm(fit.beta), np.linalg.norm(fit.tau)))))
    std_mm = 1e3 * float(max(np.linalg.norm(fit.std['b']), np.linalg.norm(fit.std['d'])))
    fit.gates = {
        'points': (n >= g_['min_points'], f'{n} samples, under {g_["min_points"]}'),
        'span': (fit.span_deg >= g_['min_span_deg'], f'the turn spanned {fit.span_deg:.0f} deg, under {g_["min_span_deg"]:.0f}'),
        'fit': (rms <= g_['max_rms_mm'], f'the trace fits poorly ({rms:.2f} mm rms, over {g_["max_rms_mm"]:.2f}): not engaged throughout?'),
        'axis': (1e3 * np.linalg.norm(fit.b) <= g_['max_b_mm'], f'the axis would move {1e3 * np.linalg.norm(fit.b):.1f} mm, over {g_["max_b_mm"]:.1f}'),
        'hand': (1e3 * np.linalg.norm(fit.d) <= g_['max_d_mm'], f'the in-hand error {1e3 * np.linalg.norm(fit.d):.1f} mm is over {g_["max_d_mm"]:.1f}'),
        'tilt': (tilt <= g_['max_tilt_deg'], f'a fitted tilt of {tilt:.1f} deg is over {g_["max_tilt_deg"]:.1f}'),
        'std': (std_mm <= g_['max_std_mm'], f'too uncertain ({std_mm:.2f} mm 1-sigma, over {g_["max_std_mm"]:.2f}): turn further'),
    }
    return fit
