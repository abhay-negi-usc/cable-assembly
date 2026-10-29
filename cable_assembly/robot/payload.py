"""PAYLOAD -- the tool's mass and centre of mass, fitted from the joint torques it causes.

THE MODEL. A tool the controller does not know about shows up in KUKA's external joint torques as
its own weight -- a force f = [0, 0, -m g] at its centre of mass c (flange frame):

    tau_ext(q) = J(q)^T [f; (R(q) c) x f] + b

J the geometric Jacobian at tool0, R tool0's rotation -- the convention of
backends/iiwa_fri.wrench_from_torques, so a payload fitted here is exactly the one its
`compensate_payload` subtracts. It is LINEAR in theta = (m, m c), so the fit is linear least
squares. b is a constant offset on the joints in `bias_joints`: KUKA's own model error, a few
tenths of a Nm, which would otherwise be read as payload.

THE MASS AND THE OFFSETS. The iiwa's wrist is spherical: A5-A7 turn the tool about ONE point, so
with the elbow still every torque the turning changes depends on m (flange offset + c) alone --
well determined -- and the mass by itself shows only in the LEVEL of the torques on A2-A4, the
weight hanging at the wrist point. Offsets on all three absorb it (`fit` refuses); offsets on two
make the mass rest on the third's level, an assumption the data cannot check (on the bench,
2026-10-29: 1.11 or 1.39 kg, with the same residual). So TWO STEPS:
  1. the mass from those levels, no offset on A2-A4 (or a mass weighed or summed from the parts);
  2. the CoG with that mass GIVEN (`mass_kg`) and an offset on every joint -- which takes up the
     model error the levels left behind, and lets the CoG come from how the torques turn.

WHAT THE POSES MUST DO. Gravity has to point different ways IN THE TOOL'S FRAME: a tool that only
turns about a vertical axis never shows its CoG along that axis. Tilting it two ways -- the
wrist, A5-A7, about a fixed elbow -- pins all three. `fit` refuses a set that does not.

NOT THE INERTIA: a static pose shows weight, not inertia. For slow, guarded motion the inertia's
torques are far below the model error; Sunrise's load data can take it from a CAD estimate.
"""

import numpy as np

G = 9.81


def gravity_torques(model, q, mass, cog):
    """tau_ext a payload of `mass` kg with its CoG at `cog` (flange frame, m) causes at `q`."""
    return regressor(model, q) @ np.concatenate([[mass], mass * np.asarray(cog, dtype=float)])


def regressor(model, q):
    """(DOF, 4) A with tau_ext = A @ [m, m cx, m cy, m cz] (no joint bias)."""
    q = np.asarray(q, dtype=float)
    J = model.jacobian(q)
    R = model.fk(q)[:3, :3]
    ez = np.array([0.0, 0.0, 1.0])
    # f = -m g ez;  (R c) x f = -g (R (m c)) x ez = g [ez]x R (m c)
    ez_x = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return np.column_stack([J[:3].T @ (-G * ez), J[3:].T @ (G * ez_x @ R)])


def tilt_span_deg(model, qs):
    """The largest angle between gravity's directions in the TOOL frame over `qs` -- near 0, the
    tool never tilted differently and the CoG cannot be seen along the axis it kept."""
    g = np.array([model.fk(np.asarray(q, dtype=float))[:3, :3].T @ [0.0, 0.0, -1.0] for q in qs])
    cos = np.clip(g @ g.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(np.min(cos))))


def fit(model, samples, bias_joints=(), mass_kg=None, min_tilt_deg=20.0):
    """Mass and CoG from `samples` = [(q, tau_ext), ...], the arm still at each; a constant
    offset is fitted on each joint in `bias_joints` (0-based). With `mass_kg` the mass is GIVEN
    and only the CoG (and the offsets) are fitted -- then an offset may go on every joint.

    Returns a dict: mass_kg, cog_m (flange frame), sigma_mass_kg (0 when given), sigma_cog_m
    (1-sigma, from the residual), bias_nm (per joint, 0 where not fitted), residual_nm (RMS per
    sample), rms_nm, tilt_span_deg. Raises ValueError when the poses cannot separate the
    unknowns."""
    qs = [np.asarray(q, dtype=float) for q, _ in samples]
    taus = [np.asarray(t, dtype=float) for _, t in samples]
    dof = taus[0].size
    bias_joints = sorted(set(int(j) for j in bias_joints))
    E = np.eye(dof)[:, bias_joints]
    k0 = 0 if mass_kg is None else 1                 # the first regressor column fitted
    n_par = 4 - k0 + len(bias_joints)
    if len(samples) * dof <= n_par + dof:
        raise ValueError(f'{len(samples)} poses is too few for {n_par} unknowns')
    span = tilt_span_deg(model, qs)
    if span < min_tilt_deg:
        raise ValueError(f'gravity turned only {span:.1f} deg in the tool frame across the poses '
                         f'(want >= {min_tilt_deg:.0f}): tilt the tool more, two ways')

    A = [regressor(model, q) for q in qs]
    M = np.vstack([np.hstack([a[:, k0:], E]) for a in A])
    y = np.concatenate(taus)
    if mass_kg is not None:
        y = y - float(mass_kg) * np.concatenate([a[:, 0] for a in A])
    # Columns scaled to unit norm, so the conditioning says what the POSES pin down, not the units.
    scale = np.linalg.norm(M, axis=0)
    scale[scale == 0.0] = 1.0
    sv = np.linalg.svd(M / scale, compute_uv=False)
    if sv[-1] < 1e-3 * sv[0]:
        raise ValueError(f'the payload is undetermined (conditioning {sv[0] / sv[-1]:.0f}): tilt '
                         'the tool further, in more directions -- or an offset is fitted on a '
                         'joint that carries the mass (A2-A4 on the iiwa)')
    theta = np.linalg.lstsq(M, y, rcond=None)[0]
    r = y - M @ theta
    dof_free = max(1, y.size - n_par)
    cov = (r @ r / dof_free) * np.linalg.inv(M.T @ M)

    if mass_kg is None:
        m, p = float(theta[0]), theta[1:4]
        if m <= 0.0:
            raise ValueError(f'the fitted mass is {m:.3f} kg: no payload in the torques (does '
                             'the Sunrise app already carry its load data?), or a sign problem')
        Jc = np.hstack([-p[:, None] / m ** 2, np.eye(3) / m])      # d c / d (m, p)
        cov_c, sigma_m = Jc @ cov[:4, :4] @ Jc.T, float(np.sqrt(cov[0, 0]))
    else:
        m, p = float(mass_kg), theta[:3]
        cov_c, sigma_m = cov[:3, :3] / m ** 2, 0.0
    res = r.reshape(len(samples), dof)
    return dict(mass_kg=m, cog_m=p / m, sigma_mass_kg=sigma_m,
                sigma_cog_m=np.sqrt(np.diag(cov_c)), bias_nm=E @ theta[4 - k0:],
                residual_nm=np.sqrt(np.mean(res ** 2, axis=1)),
                rms_nm=float(np.sqrt(np.mean(r ** 2))), tilt_span_deg=span)
