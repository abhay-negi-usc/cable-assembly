"""skills/clock_probe.py: the socket's axis and the in-hand error from a clocking turn, by proprioception and the
wrench -- checked on exact synthetic traces (the plug held on the true axis, the grasp deflected by a known
compliance x the wrench), in the app's convention (axis +X) and the sim's (+z)."""

import numpy as np
import pytest

from cable_assembly.apps.bnc_assembly import fit_axis_offset
from cable_assembly.skills.clock_probe import _perm, _rot_between, fit_clock_probe
from cable_assembly.transforms import inverse, translation_matrix, xyzrpy_to_matrix

C_LIN, C_ROT = 1.0 / 8000.0, 1.0 / 12.0
G = 0.030          # the pads' middle, up the connector axis


def _rot(axis_vec, ang):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(axis_vec, dtype=float) * ang).as_matrix()


def _T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _deflection(axis, f_c, m_c, g):
    """The grasp's deflection (connector frame) under the external wrench at the connector origin."""
    m_g = m_c - np.cross(g, f_c)
    phi = C_ROT * m_g
    u_o = C_LIN * f_c + np.cross(phi, -g)        # at the origin: u_g + phi x (0 - g)
    return _T(_rot(phi / max(np.linalg.norm(phi), 1e-15), np.linalg.norm(phi)), u_o)


def make_trace(axis, b, beta, d, tau, turn_deg=(0, 60, 0), n=240, wrench_scale=1.0, seed=0, push=-5.0,
               noise_mm=0.0, height=0.004):
    """A turn about the ESTIMATED axis with the plug on the TRUE one. Returns (T_axis, T_hand, T_hand_conn,
    wrench about the hand, base axes)."""
    rng = np.random.default_rng(seed)
    P = _perm(axis)
    lat = lambda v2: P @ np.array([v2[0], v2[1], 0.0])           # noqa: E731
    ax = P[:, 2]
    T_axis = xyzrpy_to_matrix([0.80, 0.11, 0.27], np.radians([-135.0, -2.0, -4.0]))
    T_hc_bel = xyzrpy_to_matrix([0.012, -0.004, 0.17], np.radians([5.0, -70.0, 3.0]))
    # the truths: the socket axis in the estimated axis frame; the true connector in the believed one
    u = P @ np.array([beta[0], beta[1], 1.0])
    T_true_axis = T_axis @ _T(_rot_between(ax, u / np.linalg.norm(u)), lat(b))
    tv = P @ np.array([tau[0], tau[1], 1.0])
    E = _T(_rot_between(ax, tv / np.linalg.norm(tv)), lat(d))
    T_hc_true = T_hc_bel @ E
    knots = np.radians(np.asarray(turn_deg, dtype=float))
    th = np.interp(np.linspace(0, len(knots) - 1, n), np.arange(len(knots)), knots)
    g = P @ np.array([0.0, 0.0, G])
    T_hand, W = [], []
    for i, a in enumerate(th):
        T_conn = T_true_axis @ _T(_rot(ax, a), ax * height)       # on the true axis, turned
        # a wrench that varies along the turn, as the contact's would (connector frame, at its origin)
        f_c = wrench_scale * (P @ np.array([3.0 * np.cos(2 * a + 0.3) + 1.0, 2.0 * np.sin(a) - 0.5, push]))
        m_c = wrench_scale * (P @ np.array([0.08 * np.sin(a + 0.4), -0.05 + 0.06 * np.cos(a), 0.01]))
        D = _deflection(axis, f_c, m_c, g)
        T_bel_conn_true_frame = T_conn @ inverse(D)               # where the plug would be undeflected
        Th = T_bel_conn_true_frame @ inverse(T_hc_true)
        if noise_mm:
            Th = translation_matrix(rng.normal(0, noise_mm / 1e3, 3)) @ Th
        T_hand.append(Th)
        # -> the wrench about the hand origin, base axes (as arm.wrench() gives it)
        R_bc = (Th @ T_hc_bel)[:3, :3]
        f_b, m_b = R_bc @ f_c, R_bc @ m_c
        o_c = (Th @ T_hc_bel)[:3, 3]
        m_h = m_b + np.cross(o_c - Th[:3, 3], f_b)
        W.append(np.r_[f_b, m_h])
    return T_axis, np.array(T_hand), T_hc_bel, np.array(W)


B, BETA = np.array([0.0012, -0.0021]), np.array([0.025, -0.015])
D, TAU = np.array([-0.0018, 0.0009]), np.array([-0.04, 0.03])


@pytest.mark.parametrize('axis', [0, 2])
def test_exact_recovery_with_the_grasp_compliance_known(axis):
    Ta, Th, Thc, W = make_trace(axis, B, BETA, D, TAU)
    fit = fit_clock_probe(Ta, Th, Thc, W, axis=axis, grasp_point=_perm(axis) @ [0, 0, G], pivot=0.0,
                          compliance=(C_LIN, C_ROT), huber=None)
    # first-order model: errors are products of mm and deg-scale quantities
    assert np.allclose(fit.b, B, atol=6e-5) and np.allclose(fit.d, D, atol=6e-5), (fit.b, fit.d)
    assert np.allclose(fit.beta, BETA, atol=2e-3) and np.allclose(fit.tau, TAU, atol=2e-3), (fit.beta, fit.tau)
    assert fit.residual_mm.max() < 0.06 and fit.trusted, fit.why


@pytest.mark.parametrize('axis', [0, 2])
def test_the_compliance_is_fitted_when_the_wrench_varies(axis):
    """Only the part of the wrench that is neither fixed in the hand (looks like an in-hand offset) nor fixed in
    the base (looks like the axis) identifies it -- so it takes a longer turn than the axis does (60 deg: ~25%)."""
    Ta, Th, Thc, W = make_trace(axis, B, BETA, D, TAU, turn_deg=(0, 90, -90, 0))
    fit = fit_clock_probe(Ta, Th, Thc, W, axis=axis, grasp_point=_perm(axis) @ [0, 0, G],
                          compliance=(2 * C_LIN, 0.5 * C_ROT), fit_compliance=True, huber=None)
    assert np.allclose(fit.compliance, (C_LIN, C_ROT), rtol=0.1), 1 / fit.compliance
    assert np.allclose(fit.b, B, atol=1e-4) and np.allclose(fit.d, D, atol=1e-4)


def test_proprioception_alone_is_biased_by_the_grasp_deflection():
    """The point of the wrench: the same trace without it fits the deflection as geometry."""
    Ta, Th, Thc, W = make_trace(0, B, BETA, D, TAU, wrench_scale=2.0)
    kin = fit_clock_probe(Ta, Th, Thc, None, axis=0, huber=None)
    wr = fit_clock_probe(Ta, Th, Thc, W, axis=0, grasp_point=[G, 0, 0], compliance=(C_LIN, C_ROT), huber=None)
    e_kin = np.linalg.norm(kin.b - B) + np.linalg.norm(kin.d - D)
    e_wr = np.linalg.norm(wr.b - B) + np.linalg.norm(wr.d - D)
    assert e_wr < 0.15e-3 < e_kin and e_wr < e_kin / 4, (e_kin, e_wr)


def test_lateral_only_without_wrench_is_the_axis_probe_fit():
    """No tilt, no wrench, no deflection: the same numbers as bnc_assembly.fit_axis_offset (its y, z in mm)."""
    Ta, Th, Thc, W = make_trace(0, B, (0, 0), D, (0, 0), wrench_scale=0.0)
    fit = fit_clock_probe(Ta, Th, Thc, None, axis=0, use_tilt=False, huber=None, prior_sigma=(1e3, 1e3))
    rel = np.linalg.inv(Ta)[None] @ Th @ Thc[None]
    th = np.arctan2(rel[:, 2, 1], rel[:, 1, 1])
    B0, d0, _, _ = fit_axis_offset(th, rel[:, 1, 3] * 1e3, rel[:, 2, 3] * 1e3)
    assert np.allclose(fit.b * 1e3, B0, atol=1e-6) and np.allclose(fit.d * 1e3, d0, atol=1e-6)
    assert np.allclose(fit.b, B, atol=1e-6) and np.allclose(fit.d, D, atol=1e-6)


def test_corrected_frames_put_the_trace_on_the_axis():
    Ta, Th, Thc, W = make_trace(2, B, BETA, D, TAU, wrench_scale=0.0)
    fit = fit_clock_probe(Ta, Th, Thc, None, axis=2, huber=None)
    Ta2, Thc2 = fit.axis_frame(Ta), fit.hand_conn(Thc)
    rel = np.linalg.inv(Ta2)[None] @ Th @ Thc2[None]
    assert np.abs(rel[:, 0:2, 3]).max() < 3e-5, 'the corrected connector origin on the corrected axis'
    assert np.degrees(np.arccos(np.clip(rel[:, 2, 2], -1, 1))).max() < 0.1, 'and coaxial with it'


def test_gates_refuse_a_short_or_bad_trace():
    Ta, Th, Thc, W = make_trace(0, B, BETA, D, TAU, turn_deg=(0, 15, 0), wrench_scale=0.0)
    fit = fit_clock_probe(Ta, Th, Thc, None, axis=0)
    assert not fit.trusted and fit.why.startswith('span'), fit.why
    Ta, Th, Thc, W = make_trace(0, B, BETA, D, TAU, wrench_scale=0.0, noise_mm=1.5)
    fit = fit_clock_probe(Ta, Th, Thc, None, axis=0)
    assert not fit.trusted and fit.why.startswith('fit'), fit.why


def test_robust_weights_shrug_off_a_slip():
    """A burst of samples off the model (the plug jumping in the pads, a drop transient) is down-weighted."""
    Ta, Th, Thc, W = make_trace(2, B, BETA, D, TAU, wrench_scale=0.0)
    Th = Th.copy()
    Th[100:112, :3, 3] += np.array([0.002, -0.001, 0.0])
    plain = fit_clock_probe(Ta, Th, Thc, None, axis=2, huber=None)
    rob = fit_clock_probe(Ta, Th, Thc, None, axis=2, huber=2.0, iterations=10)
    assert np.linalg.norm(rob.b - B) < np.linalg.norm(plain.b - B) and np.linalg.norm(rob.b - B) < 1e-4


def test_only_the_lateral_compliance_fitted_the_moment_term_held():
    """fit_compliance='lin': c_rot held at what is given (0 drops the moment term -- the iiwa's), c_lin fitted."""
    Ta, Th, Thc, W = make_trace(2, B, BETA, D, TAU, turn_deg=(0, 90, -90, 0))
    fit = fit_clock_probe(Ta, Th, Thc, W, axis=2, grasp_point=[0, 0, G], compliance=(2 * C_LIN, C_ROT),
                          fit_compliance='lin', huber=None)
    assert np.isclose(fit.compliance[0], C_LIN, rtol=0.1) and fit.compliance[1] == C_ROT, 1 / fit.compliance
    assert np.allclose(fit.b, B, atol=1e-4) and np.allclose(fit.d, D, atol=1e-4)
    # bounded: a nominal 10x off cannot leave [1/4, 4] x itself
    fit = fit_clock_probe(Ta, Th, Thc, W, axis=2, grasp_point=[0, 0, G], compliance=(10 * C_LIN, C_ROT),
                          fit_compliance='lin', huber=None)
    assert np.isclose(fit.compliance[0], 2.5 * C_LIN), fit.compliance
