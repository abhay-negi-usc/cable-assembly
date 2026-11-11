"""VALIDATION OF THE CLOCKING PROBE (cable_assembly/skills/clock_probe.py) in the Newton BNC sim: the socket's true
axis and the in-hand error, from a turn under the clocking push, by proprioception and the wrench -- against the
ground truth, at the full estimation error, randomised friction / grasp stiffness / washer, and the iiwa's wrench
error.

THE CELL, AS THE APP RUNS IT (bnc_assembly.connector_clocking + its axis probe):
  - the plug ENGAGED: coaxial with the TRUE socket, the collar just over the pins at a random clocking (when it
    happens to sit on the slots the plug drops in under the push -- it still turns, the body inside the collar)
  - the HAND: the sim's D6 target, tracked perfectly (the iiwa in position control); the D6's spring is the GRASP's
    compliance (the strain relief in the pads), which proprioception cannot see. The wrench the arm reports is the
    spring's, plus the sensor's noise and bias.
  - the CONTROLLER: the app's admittance at the believed connector, in its axes (AdmittanceController, mass-spring-
    damper + deadband + clamp + anti-windup): axial spring 1500 N/m under a 3 mm virtual push, roll rigid, and the
    force NULL on the lateral axes and the tilts (D 2000 N s/m / 100 Nm s/rad, deadbands 1.5 N / 0.5 Nm, caps 5 mm /
    3 deg) -- or springs there (--null off)
  - the PROBE: the reference turned about the ESTIMATED socket axis, 0 -> turn -> 0 at the clocking speed (9 deg/s)

  Each world's trace (hand poses, measured wrench) is then fitted by every method below, on the CPU, and scored
  against the truth (the socket and in-hand errors the env drew):
    kin2     proprioception only, lateral only: fit_axis_offset, the app's axis probe as of 2026-11-06
    kin4     proprioception only, lateral + tilt
    wr_nom   + the wrench, the grasp's compliance at its nominal (10 kN/m, 10 Nm/rad; the sim draws 5-20 k, 5-20)
    wr_fit   + the wrench, the compliance fitted (the nominal as a prior, 50%)
    wr_true  + the wrench, the TRUE compliance (an oracle: the upper bound)

  --follow: then a FOLLOW-UP turn (the bayonet search's first leg, 120 deg) with the side load NOT nulled (stiff in
  the plane, as a position-controlled wrist would be), half the worlds about the axis + belief the probe corrected
  (--follow_method), half about the uncorrected ones: the peak side force and moment at the connector are what the
  correction is worth to the search and to collar clocking.

    ./run.sh python scripts/newton_bnc_clock_probe.py --num_envs 256 --turn_deg 60 --follow
    ./run.sh python scripts/newton_bnc_clock_probe.py --null off --sensor iiwa
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from cable_assembly_sim.newton_bnc.env import (BncNewtonEnv, EnvCfg, euler_xyz_to_rot, rot_to_euler_xyz,  # noqa: E402
                                               rot_to_rotvec, rot_x, rot_y, rot_z, rotvec_to_rot, sym, uniform, wrap)
from cable_assembly_sim.newton_bnc.model import COLLAR_ON_PINS_HEIGHT, GRASP_Z  # noqa: E402

import newton  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SENSORS = {
    # white noise (N, Nm), bias fixed in the base (N, Nm), bias fixed in the hand, a part that flips with the turn's
    # direction (joint friction in the torque estimate), all per-episode uniform half-widths except the noise
    "ideal": dict(noise=(0.0, 0.0), base=(0.0, 0.0), hand=(0.0, 0.0), flip=(0.0, 0.0)),
    "sim": dict(noise=(0.5, 0.02), base=(1.0, 0.03), hand=(0.0, 0.0), flip=(0.0, 0.0)),
    "iiwa": dict(noise=(1.0, 0.05), base=(2.0, 0.08), hand=(1.0, 0.05), flip=(1.5, 0.05)),
}

PIVOT = 0.0045
"""Where along the plug axis the socket holds it laterally while the collar is on the pins (plug frame): between the
collar's front lip (3 mm) and the barrel's rim (5.9 mm up the plug at 6.2 mm high)."""


def _T(R, p):
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3], T[..., :3, 3], T[..., 3, 3] = R, p, 1.0
    return T


class Admittance:
    """bnc_assembly's AdmittanceController._step, batched: M a = w - D v - S delta in the believed connector's axes
    (x, y lateral, z the axis, rx ry tilt, rz roll), deadband on w, clamp with anti-windup."""

    def __init__(self, n, dev, null=True, lat_k=1500.0, rot_k=10.0):
        self.M = torch.tensor([50.0, 50.0, 50.0, 5.0, 5.0, 5.0], device=dev)
        S = torch.tensor([lat_k, lat_k, 1500.0, rot_k, rot_k, 0.0], device=dev)
        D = 2.0 * torch.sqrt(S * self.M)
        self.sel = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], device=dev)       # roll: commanded (rigid)
        self.dead = torch.tensor([3.0, 3.0, 3.0, 0.5, 0.5, 0.0], device=dev)       # the bench's compliance deadband
        self.clamp = torch.tensor([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], device=dev)
        if null:
            for i in (0, 1):
                S[i], D[i], self.dead[i], self.clamp[i] = 0.0, 2000.0, 1.5, 0.005
            for i in (3, 4):
                S[i], D[i], self.dead[i], self.clamp[i] = 0.0, 100.0, 0.5, math.radians(3.0)
        self.S, self.D = S, D
        self.delta = torch.zeros(n, 6, device=dev)
        self.vel = torch.zeros(n, 6, device=dev)

    def step(self, w, dt, mask=None):
        w = w * self.sel
        w = torch.sign(w) * (w.abs() - self.dead).clamp_min(0.0)
        acc = (w - self.D * self.vel - self.S * self.delta) / self.M
        vel = self.vel + acc * dt
        delta = self.delta + vel * dt
        at = delta.abs() >= self.clamp
        delta = torch.maximum(torch.minimum(delta, self.clamp), -self.clamp)
        vel = torch.where(at & (torch.sign(vel) == torch.sign(delta)), 0.0, vel)
        if mask is None:
            self.vel, self.delta = vel, delta
        else:
            self.vel, self.delta = torch.where(mask[:, None], vel, self.vel), torch.where(mask[:, None], delta, self.delta)

    def pose(self):
        return rotvec_to_rot(self.delta[:, 3:]), self.delta[:, :3]


def axis_rot(u, a):
    """Rotation by angle a (n,) about unit axis u (n, 3)."""
    return rotvec_to_rot(u * a[:, None])


class Probe:
    def __init__(self, a):
        self.a = a
        cfg = EnvCfg(num_envs=a.num_envs, physics_hz=1000.0, policy_hz=a.control_hz, noise_scale=a.noise_scale,
                     void_artifacts=False, max_force=math.inf, max_torque=math.inf, randomize_domain=not a.nominal)
        torch.manual_seed(a.seed)
        self.env = env = BncNewtonEnv(cfg)
        self.n, self.dev = env.num_envs, env.device
        self.dt = env.step_dt
        self.sensor = SENSORS[a.sensor]
        self._engaged_start()

    # ---- the start: engaged, relaxed ----------------------------------------------------------------------------
    def _engaged_start(self):
        env, n, dev, a = self.env, self.n, self.dev, self.a
        ids = torch.arange(n, device=dev)
        env._reset(ids)                     # errors, domain, noise level (and a start we replace)
        if a.errors == "uniform":
            # the user's spec (2026-11-06): within +/-err_mm and +/-err_deg in every dimension, for the socket (its
            # frame) and the plug in the hand (the connector frame), independently
            mm, rad = a.err_mm / 1000.0, math.radians(a.err_deg)
            rpy = sym((n, 3), rad, dev)
            env.R_obs = rot_z(rpy[:, 2]) @ rot_y(rpy[:, 1]) @ rot_x(rpy[:, 0])
            env.p_obs = sym((n, 3), mm, dev)
            rpy = sym((n, 3), rad, dev)
            R_c, p_c = rot_z(rpy[:, 2]) @ rot_y(rpy[:, 1]) @ rot_x(rpy[:, 0]), sym((n, 3), mm, dev)
            # the env's in-hand error is at the fingertips: E = G^-1 . E_c . G
            env.R_E = R_c
            env.p_E = -env.G_p + p_c + (R_c @ env.G_p[:, None])[..., 0]
        yaw = uniform(n, -math.pi, math.pi, dev)
        R_true = euler_xyz_to_rot(torch.stack((torch.zeros(n, device=dev), torch.zeros(n, device=dev), yaw), -1))
        p_true = torch.zeros(n, 3, device=dev)
        p_true[:, 2] = COLLAR_ON_PINS_HEIGHT + a.start_gap_mm / 1000.0
        G = env.G_p
        R_X, p_X = R_true, p_true + (R_true @ G[:, None])[..., 0]
        q, qd = env._q(), env._qd()
        bm = env.bm
        q[:, 0:3], q[:, 3:6] = p_X, rot_to_euler_xyz(R_X)
        q[:, bm.slide_q] = 0.0
        q[:, bm.turn_q] = uniform(n, -math.pi, math.pi, dev)       # the collar's clocking on the body
        qd[:] = 0.0
        tq = env._target_q
        tq[:, 0:6] = q[:, 0:6]
        tq[:, bm.slide_q] = 0.001
        env._clear_solver(torch.ones(n, dtype=torch.bool, device=dev))
        newton.eval_fk(env.model, env.s0.joint_q, env.s0.joint_qd, env.s0)
        env._read()
        # the hand (P_bel's target): X = P . G . E
        env.R_tgt = R_X @ env.R_E.mT
        env.p_tgt = p_X - (env.R_tgt @ (G + env.p_E)[..., None])[..., 0]
        env._drive()
        self.R_h0, self.p_h0 = env.R_tgt.clone(), env.p_tgt.clone()
        # the sensor: per-episode biases
        s = self.sensor
        self.b_base = torch.cat((sym((n, 3), s["base"][0], dev), sym((n, 3), s["base"][1], dev)), -1)
        self.b_hand = torch.cat((sym((n, 3), s["hand"][0], dev), sym((n, 3), s["hand"][1], dev)), -1)
        self.b_flip = torch.cat((sym((n, 3), s["flip"][0], dev), sym((n, 3), s["flip"][1], dev)), -1)
        self.noise = torch.tensor([s["noise"][0]] * 3 + [s["noise"][1]] * 3, device=dev)

    # ---- measurement ---------------------------------------------------------------------------------------------
    def wrench_meas(self, R_h, p_h, turn_sign):
        """The external wrench on the tool (what the contact applies), base axes, the moment about the hand origin --
        as arm.wrench() reports it (moved to the hand) -- with the sensor's errors. Also the true one."""
        env = self.env
        w = env._wrench()                                  # the hand's spring on the plug, at X, world axes
        f, m = -w[:, :3], -w[:, 3:]
        m_h = m + torch.linalg.cross(env.p_X - p_h, f)
        true = torch.cat((f, m_h), -1)
        bh = torch.cat(((R_h @ self.b_hand[:, :3, None])[..., 0], (R_h @ self.b_hand[:, 3:, None])[..., 0]), -1)
        meas = true + self.b_base + bh + turn_sign[:, None] * self.b_flip
        meas = meas + torch.randn_like(meas) * self.noise * env.noise_lvl[:, None]
        return meas, true

    # ---- one controlled turn ---------------------------------------------------------------------------------------
    def run_turn(self, knots_deg, axis_p, axis_u, R_start, p_start, adm, push_m, rec, phase, settle_s=0.0,
                 E_hat=None):
        """Turn the reference about the line (axis_p, axis_u) through the knots (deg, from the start), at the clocking
        speed (cosine-eased per segment), under the push (along -axis_u), the admittance at the believed connector
        (hand . E_hat). Records into rec[phase]."""
        env, n, dev, a = self.env, self.n, self.dev, self.a
        if E_hat is None:
            E_hat = (torch.eye(3, device=dev).repeat(n, 1, 1), torch.zeros(n, 3, device=dev))
        R_e, p_e = E_hat
        w_rad = math.radians(a.omega_deg_s)
        segs = [(math.radians(k0), math.radians(k1)) for k0, k1 in zip(knots_deg[:-1], knots_deg[1:])]
        schedule = []
        for _ in range(int(round(settle_s / self.dt))):
            schedule.append((0.0, 0.0))
        for k0, k1 in segs:
            T = max(abs(k1 - k0) / w_rad * 1.5, self.dt)          # cosine ease: peak speed 1.5x the mean -> mean ~omega/1.5
            m = int(math.ceil(T / self.dt))
            for i in range(1, m + 1):
                s = 0.5 - 0.5 * math.cos(math.pi * i / m)
                schedule.append((k0 + (k1 - k0) * s, math.copysign(1.0, k1 - k0)))
        out = {k: [] for k in ("R_h", "p_h", "w", "w_true", "R_true", "p_true", "delta", "theta_cmd", "collar", "slide")}
        ramp = max(1, int(0.5 / self.dt))
        t_wall = time.time()
        for k, (th, sgn) in enumerate(schedule):
            if k and k % int(round(2.0 / self.dt)) == 0:
                print(f"  {phase}: {k * self.dt:.0f} / {len(schedule) * self.dt:.0f} s sim, {time.time() - t_wall:.0f} s wall",
                      flush=True)
            push = push_m * min(1.0, (k + 1) / ramp) if phase == "probe" else push_m
            th_t = torch.full((n,), th, device=dev)
            Rl = axis_rot(axis_u, th_t)
            R_ref = Rl @ R_start
            p_ref = axis_p + (Rl @ (p_start - axis_p)[..., None])[..., 0] - axis_u * push
            # the law at the believed connector C = H . E_hat: T_cmd = T_ref . E . Delta . E^-1
            R_c_ref = R_ref @ R_e
            p_c_ref = p_ref + (R_ref @ p_e[..., None])[..., 0]
            Rd, pd = adm.pose()
            R_c = R_c_ref @ Rd
            p_c = p_c_ref + (R_c_ref @ pd[..., None])[..., 0]
            R_h = R_c @ R_e.mT
            p_h = p_c - (R_h @ p_e[..., None])[..., 0]
            env.R_tgt, env.p_tgt = R_h, p_h
            env._drive()
            env._simulate()
            env._guard()
            env._read()
            sgn_t = torch.full((n,), sgn, device=dev)
            meas, true = self.wrench_meas(R_h, p_h, sgn_t)
            # to the believed connector origin, in its (reference) axes
            f, m = meas[:, :3], meas[:, 3:]
            m_c = m + torch.linalg.cross(p_h - p_c, f)
            w_c = torch.cat(((R_c_ref.mT @ f[..., None])[..., 0], (R_c_ref.mT @ m_c[..., None])[..., 0]), -1)
            adm.step(w_c, self.dt)
            ph = p_h + torch.randn_like(p_h) * a.proprio_noise_mm / 1000.0
            for key, v in (("R_h", R_h), ("p_h", ph), ("w", meas), ("w_true", true), ("R_true", env.R_true),
                           ("p_true", env.p_true), ("delta", adm.delta), ("theta_cmd", th_t),
                           ("collar", env.collar_mis()), ("slide", env.slide)):
                out[key].append(v.detach().clone())
        rec[phase] = {k: torch.stack(v, 1).cpu().numpy() for k, v in out.items()}   # (n, steps, ...)
        return R_c_ref @ R_e.mT, p_c_ref - ((R_c_ref @ R_e.mT) @ p_e[..., None])[..., 0]


# ---- truth ---------------------------------------------------------------------------------------------------------

def truths(env):
    """Per world, in the estimator's terms: the socket's true axis in the observed socket frame (b at its origin,
    beta), the true connector in the believed one (d, tau), and the grasp's true compliance."""
    R_o, p_o = env.R_obs.cpu().numpy(), env.p_obs.cpu().numpy()
    q0 = np.einsum("nji,nj->ni", R_o, -p_o)                     # the world origin in S_obs
    u = R_o[:, 2, :]                                             # world z in S_obs (rows of R_o^T ... R^T e_z)
    beta = u[:, :2] / u[:, 2:3]
    b = q0[:, :2] - beta * q0[:, 2:3]
    R_E, p_E = env.R_E.cpu().numpy(), env.p_E.cpu().numpy()
    Gp = env.G_p.cpu().numpy()
    # E_c = G E G^-1
    p_c = Gp + p_E - np.einsum("nij,j->ni", R_E, Gp)
    a_c = R_E[:, :, 2]
    d = p_c[:, :2]
    tau = a_c[:, :2] / a_c[:, 2:3]
    comp = np.stack((1.0 / env.k_lin.cpu().numpy(), 1.0 / env.k_rot.cpu().numpy()), -1)
    return dict(b=b, beta=beta, d=d, tau=tau, comp=comp, mu=env.mu.cpu().numpy())


METHODS = {
    "kin2": dict(wrench=False, use_tilt=False, pivot=0.0),
    "kin2p": dict(wrench=False, use_tilt=False),
    "kin4": dict(wrench=False, use_tilt=True),
    "wr_lat": dict(wrench=True, comp="nominal", use_tilt=False),
    "wr_nom": dict(wrench=True, comp="nominal"),
    "wr_fit": dict(wrench=True, comp="nominal", fit=True),
    "wr_true": dict(wrench=True, comp="true"),
    "wr_true_lat": dict(wrench=True, comp="true", use_tilt=False),
    # the force only (no moment term: the iiwa's moment estimate at the connector is its force error x 0.2 m), the
    # lateral compliance fitted -- what the bench's 03:04 trace supports
    "wrF_fit": dict(wrench=True, comp="force", fit="lin", use_tilt=False),
    "wrF_fit4": dict(wrench=True, comp="force", fit="lin"),
    # the wrench smoothed over ~0.5 s (31 fitted samples at 125 Hz / 2)
    "wr_fit_s": dict(wrench=True, comp="nominal", fit=True, use_tilt=False, smooth=31),
    "wr_nom_s": dict(wrench=True, comp="nominal", use_tilt=False, smooth=31),
    "wrF_fit_s": dict(wrench=True, comp="force", fit="lin", use_tilt=False, smooth=31),
    "wr_true_s": dict(wrench=True, comp="true", use_tilt=False, smooth=31),
}
NOMINAL = (1.0 / 1.0e4, 1.0 / 10.0)
DEFAULT_METHODS = ["kin2", "kin4", "wr_nom", "wr_fit", "wr_true"]


def fit_all(D, methods, fit_every=2, sig_t_deg=2.0, overrides=None):
    """Every method on every world's probe trace. D: the arrays (as saved: probe_R_h, probe_p_h, probe_w, R_obs, p_obs,
    truth_comp)."""
    from cable_assembly.skills.clock_probe import fit_clock_probe

    T_axis = _T(D["R_obs"], D["p_obs"])
    T_hand = _T(D["probe_R_h"], D["probe_p_h"])
    out = {m: [] for m in methods}
    s = slice(None, None, fit_every)
    for i in range(T_hand.shape[0]):
        for m in methods:
            spec = dict(METHODS[m], **(overrides or {}))
            kw = dict(axis=2, pivot=spec.get("pivot", PIVOT), use_tilt=spec.get("use_tilt", True),
                      sigma=(1e-4, math.radians(spec.get("sig_t", sig_t_deg))))
            if spec["wrench"]:
                comp = {"nominal": NOMINAL, "force": (NOMINAL[0], 0.0)}.get(spec["comp"]) or tuple(D["truth_comp"][i])
                kw.update(wrench=D["probe_w"][i][s], wrench_about="hand", grasp_point=(0.0, 0.0, GRASP_Z),
                          compliance=comp, fit_compliance=spec.get("fit", False), compliance_rel_sigma=0.5,
                          smooth=spec.get("smooth", 0))
            out[m].append(fit_clock_probe(T_axis[i], T_hand[i][s], np.eye(4), **kw))
    return out


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "n/a"
    return f"{np.median(x):6.3f} / {np.percentile(x, 90):6.3f} / {np.percentile(x, 99):6.3f}"


def premise(D):
    """The method's premise and what happened: the TRUE plug's pivot on the TRUE axis all along, its tilt, the drop."""
    pt, Rt = D["probe_p_true"], D["probe_R_true"]
    r_true = np.linalg.norm(pt[..., :2] + Rt[..., :2, 2] * PIVOT, axis=-1)
    tilt_true = np.degrees(np.arccos(np.clip(Rt[..., 2, 2], -1, 1)))
    engaged = r_true.max(1) < 0.5e-3
    dropped = pt[:, -1, 2] < 0.004
    return engaged, dropped, r_true, tilt_true


def analyze(D, methods, fit_every=2, sig_t_deg=2.0, overrides=None, quiet=False):
    """Fit and score. -> (results dict, fits, errors per method)."""
    engaged, dropped, r_true, tilt_true = premise(D)
    tr = {k[6:]: D[k] for k in D if k.startswith("truth_")}
    # the height (in the observed socket frame) where the probe held the plug: the believed pivot's mean
    R_o, p_o = D["R_obs"], D["p_obs"]
    piv = D["probe_p_h"] + D["probe_R_h"][..., :, 2] * PIVOT
    zbar = np.einsum("nj,nkj->nk", R_o[:, :, 2], piv - p_o[:, None]).mean(1)
    t1 = time.time()
    fits = fit_all(D, methods, fit_every, sig_t_deg, overrides)

    def errs(fl):
        b = np.array([f.b for f in fl])
        be = np.array([f.beta for f in fl])
        d = np.array([f.d for f in fl])
        ta = np.array([f.tau for f in fl])
        return dict(b=1e3 * np.linalg.norm(b - tr["b"], axis=1),
                    b_eng=1e3 * np.linalg.norm(b + be * zbar[:, None] - tr["b"] - tr["beta"] * zbar[:, None], axis=1),
                    beta=np.degrees(np.arctan(np.linalg.norm(be - tr["beta"], axis=1))),
                    d=1e3 * np.linalg.norm(d - tr["d"], axis=1),
                    d_eng=1e3 * np.linalg.norm(d + PIVOT * ta - tr["d"] - PIVOT * tr["tau"], axis=1),
                    tau=np.degrees(np.arctan(np.linalg.norm(ta - tr["tau"], axis=1))),
                    rms=np.array([np.sqrt(np.mean(f.residual_mm ** 2)) for f in fl]),
                    trusted=np.array([f.trusted for f in fl]))

    prior = dict(b=1e3 * np.linalg.norm(tr["b"], axis=1),
                 b_eng=1e3 * np.linalg.norm(tr["b"] + tr["beta"] * zbar[:, None], axis=1), beta=np.degrees(np.arctan(np.linalg.norm(tr["beta"], axis=1))),
                 d=1e3 * np.linalg.norm(tr["d"], axis=1),
                 d_eng=1e3 * np.linalg.norm(tr["d"] + PIVOT * tr["tau"], axis=1), tau=np.degrees(np.arctan(np.linalg.norm(tr["tau"], axis=1))))
    E = {m: errs(fits[m]) for m in methods}
    if not quiet:
        print(f"fitted {len(methods)} methods x {len(engaged)} worlds in {time.time() - t1:.0f} s")
        print(f"\nerrors over the {engaged.sum()} worlds engaged throughout (median / p90 / p99); socket axis b at the "
              f"socket frame's origin [mm], tilt beta [deg]; in-hand d at the connector origin [mm], tau [deg]")
        print(f"  b_eng: the axis where the probe held the plug (median {1e3 * np.median(zbar):.1f} mm up the socket frame)")
        print(f"  d_eng: the in-hand offset of the pivot ({1e3 * PIVOT:.1f} mm up the plug), where the socket holds it")
        cols = ("b_eng", "d_eng", "b", "beta", "d", "tau")
        units = ("mm", "mm", "mm", "deg", "mm", "deg")
        print(f"{'':12s} " + " ".join(f"{c + ' ' + u:>22s}" for c, u in zip(cols, units)) + f" {'rms mm':>8s} {'trusted':>8s}")
        print(f"{'prior':12s} " + " ".join(f"{stats(prior[c][engaged]):>22s}" for c in cols))
        for m in methods:
            e = E[m]
            print(f"{m:12s} " + " ".join(f"{stats(e[c][engaged]):>22s}" for c in cols)
                  + f" {np.median(e['rms'][engaged]):8.3f} {e['trusted'][engaged].mean():8.1%}")
        for m in methods:
            why = [f.why.split(":")[0] for f in fits[m] if not f.trusted]
            if why:
                u, c = np.unique(why, return_counts=True)
                print(f"  {m} refused: " + ", ".join(f"{k} {v}" for k, v in zip(u, c)))
        for m in methods:
            e = E[m]
            tm = engaged & e["trusted"]
            if tm.any() and tm.sum() < engaged.sum():
                print(f"  {m} trusted only ({tm.sum()}): b {stats(e['b'][tm])}  d {stats(e['d'][tm])}")
        for m in methods:
            if METHODS[m].get("fit"):
                c = np.array([f.compliance for f in fits[m]])
                rel = c / tr["comp"]
                print(f"{m} compliance / true: lin {stats(rel[engaged, 0])}, rot {stats(rel[engaged, 1])}")
    res = {"engaged": float(engaged.mean()), "dropped": float(dropped.mean()),
           "prior": {k: stats(v[engaged]) for k, v in prior.items()},
           "methods": {m: {k: stats(v[engaged]) for k, v in E[m].items() if k != "trusted"} | {
               "trusted": float(E[m]["trusted"][engaged].mean())} for m in methods}}
    return res, fits, E


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--errors", choices=("uniform", "env"), default="uniform",
                   help="uniform: +/-err_mm, +/-err_deg in all 6 DOF of the socket and of the in-hand pose (the user's "
                        "spec); env: the RL env's ErrorCfg (in-hand SE(2) 2-3 mm / 5 deg, socket 2 mm, 3 deg tilt)")
    p.add_argument("--err_mm", type=float, default=1.0)
    p.add_argument("--err_deg", type=float, default=5.0)
    p.add_argument("--noise_scale", type=float, default=1.0, help="x the env's full-spec estimation errors (--errors env)")
    p.add_argument("--nominal", action="store_true", help="no domain randomisation")
    p.add_argument("--turn_deg", type=float, nargs="+", default=[60.0, 0.0], help="probe knots from the start [deg]")
    p.add_argument("--omega_deg_s", type=float, default=9.0, help="the app's clocking speed (90 x 0.10)")
    p.add_argument("--control_hz", type=float, default=125.0)
    p.add_argument("--push_mm", type=float, default=3.0)
    p.add_argument("--null", choices=("on", "off"), default="on")
    p.add_argument("--sensor", choices=tuple(SENSORS), default="iiwa")
    p.add_argument("--proprio_noise_mm", type=float, default=0.02)
    p.add_argument("--start_gap_mm", type=float, default=0.25, help="the collar above the pins at the start")
    p.add_argument("--settle_s", type=float, default=1.0)
    p.add_argument("--fit_every", type=int, default=2, help="fit every k-th control sample")
    p.add_argument("--sig_t_deg", type=float, default=2.0, help="a sample's tilt sigma: the engagement's play")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=tuple(METHODS))
    p.add_argument("--follow", action="store_true", help="the A/B follow-up turn (stiff), corrected vs not")
    p.add_argument("--follow_method", default="wr_nom_s", help="the correction applied (lateral only: tilt is not identified)")
    p.add_argument("--follow_deg", type=float, default=120.0)
    p.add_argument("--out", default=None, help="results .json (and traces .npz beside it)")
    p.add_argument("--refit", default=None, help="a saved .npz: fit and score it again, no simulation")
    a = p.parse_args()

    if a.refit:
        D = dict(np.load(a.refit))
        analyze(D, a.methods, a.fit_every, a.sig_t_deg)
        return

    t0 = time.time()
    pr = Probe(a)
    env, n, dev = pr.env, pr.n, pr.dev
    tr = truths(env)
    axis_p, axis_u = env.p_obs.clone(), env.R_obs[:, :, 2].clone()
    adm = Admittance(n, dev, null=(a.null == "on"))
    rec = {}
    R_ref, p_ref = pr.run_turn([0.0] + list(a.turn_deg), axis_p, axis_u, pr.R_h0, pr.p_h0, adm, a.push_mm / 1000.0,
                               rec, "probe", settle_s=a.settle_s)
    t_sim = time.time() - t0
    D = {f"probe_{k}": v for k, v in rec["probe"].items()}
    D.update({f"truth_{k}": v for k, v in tr.items()})
    D["R_obs"], D["p_obs"] = env.R_obs.cpu().numpy(), env.p_obs.cpu().numpy()
    P = rec["probe"]
    steps = P["p_true"].shape[1]
    engaged, dropped, r_true, tilt_true = premise(D)
    f_side = np.linalg.norm(P["w_true"][..., :2], axis=-1)
    print(f"{n} worlds, {steps} control steps ({steps * pr.dt:.1f} s sim) in {t_sim:.0f} s; engaged throughout "
          f"{engaged.mean():.1%} (pivot off-axis max p50 {np.median(r_true.max(1)) * 1e3:.2f} mm, tilt max p50 "
          f"{np.median(tilt_true.max(1)):.2f} deg); dropped onto the slots {dropped.mean():.1%}; true side force at "
          f"the hand p50/p95 of each world's peak {np.median(f_side.max(1)):.1f} / {np.percentile(f_side.max(1), 95):.1f} N;"
          f" null give max p50 {1e3 * np.median(np.abs(P['delta'][..., :2]).max((1, 2))):.2f} mm, "
          f"{np.degrees(np.median(np.abs(P['delta'][..., 3:5]).max((1, 2)))):.2f} deg", flush=True)
    methods = list(dict.fromkeys(a.methods + ([a.follow_method] if a.follow else [])))
    res, fits, E = analyze(D, methods, a.fit_every, a.sig_t_deg)
    res["args"] = vars(a)
    res["side_force_peak_p50"] = float(np.median(f_side.max(1)))
    if a.follow:
        res["follow"] = follow(pr, rec, fits[a.follow_method], tr, a, R_ref, p_ref, adm)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        arrays = {f"{ph}_{k}": v.astype(np.float64 if k in ("R_h", "p_h") else np.float32)
                  for ph, d in rec.items() for k, v in d.items()}
        arrays["R_obs"], arrays["p_obs"] = D["R_obs"], D["p_obs"]
        arrays.update({f"truth_{k}": v for k, v in tr.items()})
        arrays.update({f"err_{m}_{k}": v for m in E for k, v in E[m].items()})
        arrays["engaged"], arrays["dropped"] = engaged, dropped
        np.savez_compressed(os.path.splitext(a.out)[0] + ".npz", **arrays)
        print("wrote", a.out)


def follow(pr, rec, fl, tr, a, R_ref, p_ref, adm):
    """The bayonet search's first leg, stiff in the plane (tilt nulled as the app does), A/B: even worlds about the corrected axis with the
    corrected belief, odd worlds about the uncorrected ones. -> peak side force / moment at the connector."""
    env, n, dev = pr.env, pr.n, pr.dev
    corr = torch.zeros(n, dtype=torch.bool, device=dev)
    corr[0::2] = True
    R_o, p_o = env.R_obs.cpu().numpy(), env.p_obs.cpu().numpy()
    u0 = env.R_obs[:, :, 2].clone()
    A_p, A_u = env.p_obs.clone(), u0.clone()
    R_e = torch.eye(3, device=dev).repeat(n, 1, 1)
    p_e = torch.zeros(n, 3, device=dev)
    for i in range(0, n, 2):
        f = fl[i]
        Ta = f.axis_frame(_T(R_o[i], p_o[i]))
        Ehat = f.hand_conn(np.eye(4))
        A_p[i] = torch.tensor(Ta[:3, 3], device=dev, dtype=torch.float32)
        A_u[i] = torch.tensor(Ta[:3, 2], device=dev, dtype=torch.float32)
        R_e[i] = torch.tensor(Ehat[:3, :3], device=dev, dtype=torch.float32)
        p_e[i] = torch.tensor(Ehat[:3, 3], device=dev, dtype=torch.float32)
    # The give the null held across the axis folded into the reference (the app's zero_axes), the push taken back out
    # (the turn re-applies it along its own axis); the axial admittance's state carried on.
    dl = adm.delta.clone()
    dl[:, 2] = 0.0
    R_h = R_ref @ rotvec_to_rot(dl[:, 3:])
    p_h = p_ref + (R_ref @ dl[:, :3, None])[..., 0] + u0 * a.push_mm / 1000.0
    # The corrected half: the hand TRANSLATED so the corrected belief's pivot sits on the corrected axis -- its
    # orientation kept (the plug is coaxial with the socket now, and the tilt is not what the probe identifies:
    # turning the hand onto an axis direction it does not know would tilt an engaged plug by up to the tilt error).
    R_c = R_h @ R_e
    p_piv = p_h + (R_h @ p_e[..., None])[..., 0] + R_c[:, :, 2] * PIVOT
    rel = p_piv - A_p
    perp = rel - A_u * (rel * A_u).sum(-1, keepdim=True)
    R_s = R_h
    p_s = torch.where(corr[:, None], p_h - perp, p_h)
    # Stiff in the plane (no lateral give: what is left of the misalignment becomes side force), the tilt null and
    # the axial spring + push as the app's clocking has them.
    stiff = Admittance(n, dev, null=True)
    stiff.sel = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], device=dev)
    stiff.delta[:, 2], stiff.vel[:, 2] = adm.delta[:, 2], adm.vel[:, 2]
    pr.run_turn([0.0, a.follow_deg], A_p, A_u, R_s, p_s, stiff, a.push_mm / 1000.0, rec, "follow",
                E_hat=(R_e, p_e))
    F = rec["follow"]
    fs = np.linalg.norm(F["w_true"][..., :2], axis=-1).max(1)
    ms = np.linalg.norm(F["w_true"][..., 3:5], axis=-1).max(1)
    r_true = np.linalg.norm(F["p_true"][..., :2] + F["R_true"][..., :2, 2] * PIVOT, axis=-1).max(1)
    c = corr.cpu().numpy()
    out = {}
    print(f"\nFOLLOW-UP {a.follow_deg:.0f} deg turn, stiff in the plane, tilt nulled (peak over the turn; median / p90 / p99):")
    print(f"  corrected: {int(c.sum())} worlds with the {a.follow_method} correction, shift applied p50 "
          f"{1e3 * np.median(np.linalg.norm(perp.cpu().numpy()[c], axis=-1)):.2f} mm")
    for name, m in (("corrected", c), ("uncorrected", ~c)):
        print(f"  {name:12s} side force {stats(fs[m])} N   moment {stats(ms[m])} Nm   plug off the true axis "
              f"{stats(1e3 * r_true[m])} mm")
        out[name] = {"side_force": stats(fs[m]), "moment": stats(ms[m]), "off_axis_mm": stats(1e3 * r_true[m])}
    return out


if __name__ == "__main__":
    with torch.inference_mode():
        main()
