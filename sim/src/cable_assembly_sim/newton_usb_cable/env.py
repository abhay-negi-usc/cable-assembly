"""USB-A insertion with the gripper holding the CABLE (newton_usb_cable/model.py), as an RSL-RL VecEnv.

The same task, observations, actions, reward and curriculum as newton_usb/env.py (UsbNewtonEnv, which this
subclasses): the policy moves the BELIEVED plug's target in delta-pose steps, in the observed socket's frame,
and sees proprioception and the wrench only. What changes is the grasp:

  - THE GRASP is on the cable, rigid in the pads. Pads on a cylinder fix 2 of its 6 DOFs: the position
    along the jaws' closing axis (centred) and the tilt out of the pads' plane. The other 4 are the grasp's
    pose: across the cable in the pads' plane (y), along it (z), turned in that plane (about the closing
    axis, x) and rolled about the cable's own axis (z).
  - GRASP CONTROL ERROR (GraspControlCfg), every episode, not scaled by the curriculum: the cable lands in
    the pads off the commanded grasp. It is KNOWN to the robot (in the believed plug pose), so at zero
    uncertainty the plug is where the robot believes, but the grasp still varies from episode to episode.
  - GRASP KNOWLEDGE ERROR (CableErrorCfg.grasp_*), scaled by the curriculum: the believed grasp is off the
    actual one in the same 4 DOFs. Plus the socket's 6-DOF pose error, as before.
  - THE CABLE between the pads and the overmold bends and twists (EI, GJ randomised per episode). The robot
    believes it straight at rest; under load it bends, which only the wrench shows.

The latch (detent) is Coulomb friction on the plug's own world-z DOF, as before. There is no slip model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import warp as wp

from cable_assembly_sim.newton_usb.env import (DomainCfg, EnvCfg, UsbNewtonEnv, quat_to_rot, rot_to_quat)  # first
from cable_assembly_sim.newton_usb.model import PAD_HALF, PLUG_DY
from cable_assembly_sim.newton_bnc.env import (euler_xyz_to_rot, rot_to_euler_xyz, rot_x, rot_y, rot_z, sym, uniform,
                                               wrap)

from .model import CABLE_RADIUS, OVERMOLD_BACK_Z, CableModel, CableModelCfg, build, nominal_hand

import newton  # noqa: E402
from newton import ModelFlags  # noqa: E402


@dataclass
class CableErrorCfg:
    """Half-widths of the uniform estimation errors at noise scale 1 (the curriculum scales them)."""

    grasp_y: float = 0.003
    """Across the cable in the pads' plane [m]."""
    grasp_z: float = 0.003
    """Along the cable [m]."""
    grasp_rot_deg: float = 3.0
    """About the jaws' closing axis, in the pads' plane. Over the ~88 mm from the grasp to the plug's tip
    this moves the tip ~1.5 mm a degree (the grasp on the overmold: 0.65)."""
    grasp_roll_deg: float = 3.0
    """About the cable's axis (the plug's turn about its own axis)."""
    target_xy: float = 0.007
    target_z: float = 0.010
    target_tilt_deg: float = 7.0
    target_yaw_deg: float = 7.0


@dataclass
class GraspControlCfg:
    """Half-widths of the grasp's control error, every episode, known to the robot, never scaled."""

    y: float = 0.0015
    z: float = 0.003
    rot_deg: float = 3.0
    roll_deg: float = 5.0


@dataclass
class CableDomainCfg(DomainCfg):
    bend_stiffness: tuple[float, float] = (3.0e-4, 3.0e-3)
    """EI [N m^2], drawn log-uniform: a 5.5 mm USB cord, from limp to stiff."""
    twist_ratio: tuple[float, float] = (0.5, 1.0)
    """GJ / EI."""
    detent: tuple[float, float] = (1.0, 3.0)
    """The latch's breakaway force [N]: a USB-A receptacle mates under 3 N."""


@dataclass
class CableEnvCfg(EnvCfg):
    model: CableModelCfg = field(default_factory=CableModelCfg)
    errors: CableErrorCfg = field(default_factory=CableErrorCfg)
    domain: CableDomainCfg = field(default_factory=CableDomainCfg)
    grasp_control: GraspControlCfg = field(default_factory=GraspControlCfg)
    randomize_grasp: bool = True
    """Draw the grasp control error (False: the nominal grasp, for debugging)."""
    kp_upright: bool = True
    kp_free_radius: float = 0.010
    curriculum_on: str = "pose"
    """What the curriculum's level widens. "pose": the estimation errors (CableErrorCfg x level), the cable's
    stiffness over its whole range. "stiffness": the cable's EI range, from the stiffest (level 0: EI = the
    range's top) down log-uniformly toward the softest (level 1: the whole range), with the pose errors fixed at
    `pose_scale` x CableErrorCfg (0: none; the grasp's control error is still drawn, and known)."""
    pose_scale: float = 0.0
    history: int = 8
    """Frames of observation history (0.4 s at 20 Hz), on top of the actor's GRU."""
    obs_action: bool = True
    """With obs_mode "poses": each frame also carries the policy's last action, so commanded motion can be set
    against observed motion (state estimation: contact, the cable's give, a stuck plug)."""
    obs_mode: str = "wrench"
    """The actor's observation. "wrench": UsbNewtonEnv's (the believed plug, the measured wrench, velocities,
    the last action, the target's lead). "poses": only the in-hand connector pose (the plug in the hand frame,
    live: the cable's bend shows in it; with pose error, biased by the grasp's knowledge error) and the target
    socket's pose (the observed socket, in the hand frame), as a wrist camera sees them, over `history` frames.
    The critic keeps its privileged inputs either way."""
    # Low force: the latch needs < 3 N, and the cable is held 25 mm above the plug in a 20 N grip (below).
    max_force: float = 5.0
    force_soft: float = 2.5
    force_scale: float = 2.5
    hold_force: float = 1.0
    contact_force: float = 0.75
    launch_factor: float = 2.0
    grasp_slip: bool = True
    """End the episode as a failure when the load on the grasp reaches `slip_margin` of what the pads hold
    (`grasp_util` >= 1), and penalise it from `slip_soft` of that. The policy is never to make the cable slip.
    The holding capacity, for a Robotiq 2F-85 closing with force N (domain grip_force, 18-22 N) on a 5.5 mm
    jacket through its pads (friction mu, domain pad_mu 0.3-0.8; contact `slip_pad_len` along the cable):
    friction -- the in-plane force (along the cable and across it) 2 mu N (~22 N at mu 0.55), the twist about
    the cable 2 mu N r (~0.06 N m), the moment turning the cable in the pads' plane mu N l / 2 (~0.12 N m);
    pressure -- the force pushing the cable out between the jaws N, the moment tipping it into the pads
    N l / 3 (~0.15 N m). The friction terms combine (elliptically), the worst of the rest counts."""
    slip_pad_len: float = 0.022
    """[m] The 2F-85 fingertip pad's extent along a cable held across the fingers."""
    slip_margin: float = 0.7
    slip_soft: float = 0.5
    w_slip: float = 2.0
    slip_penalty: float = 20.0
    weld_solref: tuple[float, float] = (0.002, 1.0)
    weld_solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 0.001, 0.5, 2.0)
    """The cable-to-plug weld. MuJoCo's default (0.02 s; 0.9-0.95) stretched it 1.1 mm under 2 N; this holds
    it to 0.09 mm and 0.1 deg under 14 N."""


def grasp_tf(y, z, rx, rz):
    """(R, p) of a 4-DOF grasp offset in the cable frame (plug-aligned axes, z up the cable)."""
    zero = torch.zeros_like(y)
    return rot_x(rx) @ rot_z(rz), torch.stack((zero, y, z), -1)


class UsbCableEnv(UsbNewtonEnv):
    um: CableModel

    # ---- hooks -------------------------------------------------------------------------------------

    def _build_model(self, cfg, n, device):
        from dataclasses import replace

        return build(replace(cfg.model, num_worlds=n), device=device)

    def _detent_dof(self) -> int:
        return self.um.plug_dof0 + 2

    def _init_extra(self):
        cfg, dev, n, um = self.cfg, self.device, self.num_envs, self.um
        R_PF, p_PF = nominal_hand(cfg.model)
        self.R_PF0 = torch.tensor(R_PF, dtype=torch.float32, device=dev)
        self.R_FP0 = self.R_PF0.T.contiguous()
        self.p_anchor = torch.tensor(p_PF, dtype=torch.float32, device=dev)  # the arm D6's zero: the nominal hand
        self.G_p = torch.zeros(3, device=dev)  # base class: X is the hand, (R_E, p_E) the hand in the believed plug
        self.T_CP = torch.tensor([0.0, -PLUG_DY, -cfg.model.grasp_z], device=dev)  # the plug's origin from the grasp
        self.R_C = torch.eye(3, device=dev).repeat(n, 1, 1)  # the grasp control error
        self.p_C = torch.zeros(n, 3, device=dev)
        self.R_FP_true = self.R_FP0.repeat(n, 1, 1)
        self.p_FP_true = torch.zeros(n, 3, device=dev)
        self.EI = torch.full((n,), cfg.model.bend_stiffness, device=dev)
        self.GJ = torch.full((n,), cfg.model.twist_stiffness, device=dev)
        self.err_ctrl = torch.zeros(n, 4, device=dev)
        self.err = torch.zeros(n, 10, device=dev)
        self.R_D = torch.eye(3, device=dev).repeat(n, 1, 1)  # the in-hand measurement's bias (plug frame)
        self.p_D = torch.zeros(n, 3, device=dev)
        self.hist_pose = torch.zeros(n, cfg.history, self.pose_frame_dim, device=dev)
        self.ei_lo = torch.full((n,), cfg.domain.bend_stiffness[1], device=dev)
        self._hand_inertia0 = self._body_inertia[0, um.hand_body].clone()
        self._joint_X_p = wp.to_torch(self.model.joint_X_p).view(n, um.joints_per_world, 7)
        self._bq = lambda: wp.to_torch(self.s0.body_q).view(n, um.bodies_per_world, 7)  # noqa: E731
        es = wp.to_torch(self.solver.mjw_model.eq_solref)
        es[..., 0], es[..., 1] = cfg.weld_solref
        ei = wp.to_torch(self.solver.mjw_model.eq_solimp)
        ei[...] = torch.tensor(cfg.weld_solimp, device=dev)

    # ---- state -------------------------------------------------------------------------------------

    def _read(self):
        q, P = self._q(), self.um.plug_dof0
        self.R_X = euler_xyz_to_rot(q[:, 3:6]) @ self.R_PF0
        self.p_X = q[:, 0:3] + self.p_anchor
        R_F = self.R_X @ self.R_E.mT  # the believed plug
        self.R_bel, self.p_bel = R_F, self.p_X - (R_F @ self.p_E[..., None])[..., 0]
        self.R_true, self.p_true = euler_xyz_to_rot(q[:, P + 3:P + 6]), q[:, P:P + 3].clone()

    def _drive(self):
        R_xt = self.R_tgt @ self.R_E
        p_xt = self.p_tgt + (self.R_tgt @ self.p_E[..., None])[..., 0]
        q = self._q()
        e = rot_to_euler_xyz(R_xt @ self.R_FP0)
        e = q[:, 3:6] + wrap(e - q[:, 3:6])
        self._target_q[:, 0:3] = p_xt - self.p_anchor
        self._target_q[:, 3:6] = e
        self.R_xt, self.p_xt = R_xt, p_xt

    def _guard(self):
        if not hasattr(self, "_healthy"):
            self._healthy = self._per_world_arrays()  # first step: every world healthy
        q, qd = self._q(), self._qd()
        bad = ~(torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1))
        if bad.any():
            self._heal(bad)
            self._reset(bad.nonzero(as_tuple=False).squeeze(-1))
        return bad

    def _per_world_arrays(self):
        """{name: (view, healthy copy)} for the solver's per-world float arrays (MuJoCo-Warp's model, e.g. the
        invweight0 / acc0 constants it derives once, and Newton's model.mujoco attributes), as (num_envs, -1)."""
        n, out = self.num_envs, {}

        def add(obj, name):
            for k in dir(obj):
                if k.startswith("_"):
                    continue
                try:
                    v = getattr(obj, k)
                except Exception:
                    continue
                if isinstance(v, wp.array) and v.size and v.shape[0] % n == 0 and v.shape[0] >= n:
                    try:
                        t = wp.to_torch(v)
                    except Exception:
                        continue
                    if t.is_floating_point() and t.is_cuda:
                        t = t.view(n, -1)
                        out[f"{name}.{k}"] = (t, t.clone())

        add(self.solver.mjw_model, "mjw_model")
        if hasattr(self.model, "mujoco"):
            add(self.model.mujoco, "model.mujoco")
        return out

    def _heal(self, bad):
        """Recover worlds that went non-finite. MuJoCo-Warp keeps per-world solver data across steps (the factored
        mass matrix among it) and derives some per-world constants (body / dof invweight0, actuator_acc0) from it;
        after a divergence, resetting the state alone left the world NaN on every later step, through every reset
        (A3 from iteration ~423, B from ~170: one world at a time, its reset each step disturbing the batch).
        Zeroing the data did not recover it; a healthy world's data does (the state is rewritten by the reset)."""
        self._restore_constants(bad)
        self._copy_data(bad)

    def _restore_constants(self, worlds):
        for t, h in self._healthy.values():
            row = worlds & ~torch.isfinite(t).all(-1)
            if row.any():
                t[row] = h[row]
        # notify_model_changed (every reset) re-derives from the world's data: NaN again if that data was.
        qld = wp.to_torch(self.solver.mjw_data.qLD).view(self.num_envs, -1)
        row = worlds & ~torch.isfinite(qld).all(-1)
        if row.any():
            self._copy_data(row)

    def _copy_data(self, worlds, obj=None, depth=0, donor=None):
        """Copy the solver's per-world data rows of `worlds` from a healthy world."""
        n = self.num_envs
        if obj is None:
            ok = (~worlds).nonzero()
            if not len(ok):
                return
            obj, donor = self.solver.mjw_data, int(ok[0, 0])
        for k in dir(obj):
            if k.startswith("_"):
                continue
            try:
                v = getattr(obj, k)
            except Exception:
                continue
            if isinstance(v, wp.array):
                if v.size and v.shape[0] == n:
                    try:
                        t = wp.to_torch(v)
                    except Exception:
                        continue
                    if t.is_floating_point() and t.stride(0) != 0:
                        t[worlds] = t[donor].clone()
            elif depth < 2 and type(v).__module__.startswith("mujoco_warp") and not callable(v):
                self._copy_data(worlds, v, depth + 1, donor)

    def _slip(self):
        pass

    # ---- the grasp's holding limit ------------------------------------------------------------------

    @property
    def _failure_penalty(self):
        return self.cfg.slip_penalty

    def grasp_load(self):
        """The load on the grasp as a fraction of its slip margin (num_envs,): >= 1 slips (see CableEnvCfg)."""
        cfg = self.cfg
        w = self._wrench()
        f = (self.R_X.mT @ w[:, 0:3, None])[..., 0]  # hand frame, at the grasp point
        t = (self.R_X.mT @ w[:, 3:6, None])[..., 0]
        a = self.R_FP_true[:, :, 2]  # the cable's axis at the grasp, hand frame
        y = self.R_FP_true[:, :, 0]  # the jaws' closing axis
        y = y - (y * a).sum(-1, keepdim=True) * a
        y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-9)
        b = torch.linalg.cross(a, y)  # across the cable, in the pads' plane
        mu, N, r, ln = self.pad_mu, self.grip, CABLE_RADIUS, cfg.slip_pad_len
        f_plane = torch.hypot((f * a).sum(-1), (f * b).sum(-1))
        fric = torch.sqrt((f_plane / (2 * mu * N)) ** 2 + ((t * y).sum(-1) / (mu * N * ln / 2)) ** 2
                          + ((t * a).sum(-1) / (2 * mu * N * r)) ** 2)
        press = torch.maximum((f * y).sum(-1).abs() / N, (t * b).sum(-1).abs() / (N * ln / 3))
        return (torch.maximum(fric, press) / cfg.slip_margin).nan_to_num(0.0)

    def _failure(self):
        if not self.cfg.grasp_slip:
            return super()._failure()
        return self.grasp_util >= 1.0

    def _reward(self, a):
        reward, terms, seated, engaged, force = super()._reward(a)
        cfg = self.cfg
        self.grasp_util = self.grasp_load()
        if cfg.grasp_slip:
            terms["grasp"] = -cfg.w_slip * ((self.grasp_util - cfg.slip_soft).clamp_min(0.0) / (1.0 - cfg.slip_soft)) ** 2
            reward = reward + terms["grasp"]
        return reward, terms, seated, engaged, force

    def step(self, actions):
        out = super().step(actions)
        if self.cfg.grasp_slip:
            slipped = (self.grasp_util >= 1.0) & ~self.voided
            self.slip_events = getattr(self, "slip_events", 0) + int(slipped.sum())
            log = self.extras["log"]
            log["/metrics/grasp_slip_total"] = torch.tensor(float(self.slip_events))
            log["/metrics/grasp_load_mean"] = self.grasp_util.mean()
            log["/metrics/grasp_load_p99"] = torch.quantile(self.grasp_util.float(), 0.99)
        return out

    # ---- reset -------------------------------------------------------------------------------------

    def _reset(self, ids):
        cfg, dev, k, um = self.cfg, self.device, len(ids), self.um
        e = cfg.errors
        s = torch.full((k,), float(cfg.noise_scale), device=dev)
        if cfg.curriculum != "off" and cfg.curriculum_mix > 0:
            mix = torch.rand(k, device=dev) < cfg.curriculum_mix
            s[mix] = torch.rand(int(mix.sum()), device=dev) * cfg.noise_scale
        self.ep_scale[ids] = s
        if cfg.curriculum_on == "stiffness":
            s_stiff, s = s, torch.full((k,), float(cfg.pose_scale), device=dev)
        else:
            s_stiff = torch.ones(k, device=dev)

        # The socket's observed pose (6 DOF), as UsbNewtonEnv.
        t_rpy = torch.stack((sym(k, math.radians(e.target_tilt_deg) * s, dev), sym(k, math.radians(e.target_tilt_deg) * s, dev),
                             sym(k, math.radians(e.target_yaw_deg) * s, dev)), -1)
        t_xyz = torch.stack((sym(k, e.target_xy * s, dev), sym(k, e.target_xy * s, dev), sym(k, e.target_z * s, dev)), -1)
        self.R_obs[ids] = rot_z(t_rpy[:, 2]) @ rot_y(t_rpy[:, 1]) @ rot_x(t_rpy[:, 0])
        self.p_obs[ids] = t_xyz

        # The grasp: control error C (actual vs commanded, known) and knowledge error K (believed vs actual).
        gc = cfg.grasp_control
        on = 1.0 if cfg.randomize_grasp else 0.0
        c = (sym(k, gc.y * on, dev), sym(k, gc.z * on, dev), sym(k, math.radians(gc.rot_deg) * on, dev),
             sym(k, math.radians(gc.roll_deg) * on, dev))
        kk = (sym(k, e.grasp_y * s, dev), sym(k, e.grasp_z * s, dev), sym(k, math.radians(e.grasp_rot_deg) * s, dev),
              sym(k, math.radians(e.grasp_roll_deg) * s, dev))
        R_C, p_C = grasp_tf(*c)
        R_K, p_K = grasp_tf(*kk)
        self.R_C[ids], self.p_C[ids] = R_C, p_C
        # Hand -> cable at the grasp, actual and believed; -> plug through the straight cable (T_CP).
        R_FC, p_FC = self.R_FP0 @ R_C, (self.R_FP0 @ p_C[..., None])[..., 0]
        R_FCb, p_FCb = R_FC @ R_K, p_FC + (R_FC @ p_K[..., None])[..., 0]
        R_FP, p_FP = R_FC, p_FC + (R_FC @ self.T_CP)
        R_FPb, p_FPb = R_FCb, p_FCb + (R_FCb @ self.T_CP)
        self.R_FP_true[ids], self.p_FP_true[ids] = R_FP, p_FP
        # Base-class convention: (R_E, p_E) = the hand in the believed plug frame.
        self.R_E[ids] = R_FPb.mT
        self.p_E[ids] = -(R_FPb.mT @ p_FPb[..., None])[..., 0]
        # The in-hand pose measurement's bias: the believed plug in the actual one's frame (identity without error).
        self.R_D[ids] = R_FP.mT @ R_FPb
        self.p_D[ids] = (R_FP.mT @ (p_FPb - p_FP)[..., None])[..., 0]
        self.err_ctrl[ids] = torch.stack((c[0] / 0.0015, c[1] / 0.003, c[2] / 0.052, c[3] / 0.087), -1)
        self.err[ids] = torch.cat((torch.stack((kk[0] / 0.003, kk[1] / 0.003, kk[2] / 0.052, kk[3] / 0.052), -1),
                                   t_xyz / 0.007, t_rpy / 0.12), -1)
        # The grasp's actual pose: the hand -> link 1 joint's anchor in the hand, this world's.
        hold = torch.tensor([0.0, 0.0, -cfg.model.grasp_hold], device=dev)
        p_J = p_FC + (R_FC @ hold)
        Xp = self._joint_X_p[ids]
        Xp[:, um.grasp_joint, 0:3] = p_J
        Xp[:, um.grasp_joint, 3:7] = rot_to_quat(R_FC)
        self._joint_X_p[ids] = Xp

        # Domain (UsbNewtonEnv's, the hand's mass on the hand, plus the cable's stiffness).
        d = cfg.domain
        if cfg.randomize_domain:
            mu = uniform(k, *d.mu_parts, dev)
            k_lin, k_rot = uniform(k, *d.lin_stiffness, dev), uniform(k, *d.rot_stiffness, dev)
            det = uniform(k, *d.detent, dev)
            mass = uniform(k, *d.hand_mass, dev)
            self.delay[ids] = torch.randint(d.action_delay[0], d.action_delay[1] + 1, (k,), device=dev) > 0
            # Log-uniform over [lo, hi]; the stiffness curriculum raises lo from hi toward the range's bottom.
            l0, l1 = math.log(d.bend_stiffness[0]), math.log(d.bend_stiffness[1])
            lo = l1 + (l0 - l1) * s_stiff
            self.ei_lo[ids] = torch.exp(lo)
            EI = torch.exp(lo + (l1 - lo) * torch.rand(k, device=dev))
            GJ = EI * uniform(k, *d.twist_ratio, dev)
        else:
            m = cfg.model
            mu = torch.full((k,), 0.3, device=dev)
            k_lin, k_rot = torch.full((k,), m.lin_ke, device=dev), torch.full((k,), m.rot_ke, device=dev)
            det = torch.full((k,), 0.5 * sum(d.detent), device=dev)
            mass = torch.full((k,), m.hand_mass, device=dev)
            self.delay[ids] = False
            EI, GJ = torch.full((k,), m.bend_stiffness, device=dev), torch.full((k,), m.twist_stiffness, device=dev)
        self.mu[ids], self.detent_f[ids] = mu, det
        sm = self._shape_mu[ids]
        sm[:, um.plug_shapes + um.socket_shapes] = mu[:, None]
        self._shape_mu[ids] = sm
        self.k_lin[ids], self.k_rot[ids], self.EI[ids], self.GJ[ids] = k_lin, k_rot, EI, GJ
        m = cfg.model
        ke, kd = self._dof_ke[ids], self._dof_kd[ids]
        ke[:, 0:3], kd[:, 0:3] = k_lin[:, None], (m.lin_kd * torch.sqrt(k_lin / m.lin_ke))[:, None]
        ke[:, 3:6], kd[:, 3:6] = k_rot[:, None], (m.rot_kd * torch.sqrt(k_rot / m.rot_ke))[:, None]
        L, c0 = m.link_len, um.cable_dof0
        for i in range(m.cable_segments):
            for a, kk_ in ((0, EI / L), (1, EI / L), (2, GJ / L)):
                ke[:, c0 + 3 * i + a] = kk_
                kd[:, c0 + 3 * i + a] = m.cable_damping_ratio * 2.0 * torch.sqrt(kk_ * m.cable_armature)
        self._dof_ke[ids], self._dof_kd[ids] = ke, kd
        hb, r = um.hand_body, (mass / m.hand_mass)[:, None, None]
        self.mass[ids] = mass
        bm_, bi = self._body_mass[ids], self._body_inv_mass[ids]
        bm_[:, hb], bi[:, hb] = mass, 1.0 / mass
        self._body_mass[ids], self._body_inv_mass[ids] = bm_, bi
        I, Ii = self._body_inertia[ids], self._body_inv_inertia[ids]
        I[:, hb] = self._hand_inertia0 * r
        Ii[:, hb] = torch.linalg.inv(self._hand_inertia0) / r
        self._body_inertia[ids], self._body_inv_inertia[ids] = I, Ii
        self.a_prev[ids] = 0.0

        # The start: the believed plug above the observed socket; the hand through the believed grasp; the
        # actual plug through the actual grasp and the straight cable.
        start = torch.stack((sym(k, cfg.start_xy, dev), sym(k, cfg.start_xy, dev),
                             uniform(k, *cfg.start_height, dev) + cfg.start_clear * s), -1)
        yaw = sym(k, math.radians(cfg.start_yaw_deg), dev)
        R_b = self.R_obs[ids] @ rot_z(yaw)
        p_b = self.p_obs[ids] + (self.R_obs[ids] @ start[..., None])[..., 0]
        self.R_tgt[ids], self.p_tgt[ids] = R_b, p_b
        R_h = R_b @ self.R_E[ids]
        p_h = p_b + (R_b @ self.p_E[ids][..., None])[..., 0]
        R_p, p_p = R_h @ R_FP, p_h + (R_h @ p_FP[..., None])[..., 0]
        q, qd = self._q(), self._qd()
        q_new = torch.zeros(k, self.C, device=dev)
        q_new[:, 0:3] = p_h - self.p_anchor
        q_new[:, 3:6] = rot_to_euler_xyz(R_h @ self.R_FP0)
        P = um.plug_dof0
        q_new[:, P:P + 3] = p_p
        q_new[:, P + 3:P + 6] = rot_to_euler_xyz(R_p)
        q[ids] = q_new
        qd[ids] = 0.0
        tq = self._target_q[ids]
        tq[:] = 0.0
        tq[:, 0:6] = q_new[:, 0:6]
        self._target_q[ids] = tq
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        mask[ids] = True
        self.solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES
                                         | ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.JOINT_PROPERTIES)
        self._clear_solver(mask)
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        if hasattr(self, "_healthy"):  # notify_model_changed re-derives a diverged world's constants as NaN
            self._restore_constants(mask)

        self.wrench_bias[ids] = torch.cat((sym((k, 3), cfg.wrench_bias[0], dev), sym((k, 3), cfg.wrench_bias[1], dev)), -1)
        self.actions[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.first_seated[ids] = -1
        self.peak_force[ids] = 0.0
        self.peak_speed[ids] = 0.0
        self.slip_total[ids] = 0.0
        self._read()
        if hasattr(self, "R_xt"):
            self.R_xt[ids], self.p_xt[ids] = self.R_X[ids], self.p_X[ids]
        else:
            self.R_xt, self.p_xt = self.R_X.clone(), self.p_X.clone()
        # The commanded target persists across steps (_apply_action adds deltas to it); start each episode at the
        # believed pose. Without this a world whose target once went NaN stayed NaN through every later reset
        # (the lead clamps pass NaN through) and blew up on every step for the rest of training (A3, iter ~423).
        self.R_tgt[ids], self.p_tgt[ids] = self.R_bel[ids], self.p_bel[ids]

    # ---- observations ------------------------------------------------------------------------------

    def _observe(self, reset_ids=None):
        out = super()._observe(reset_ids)
        if self.cfg.obs_mode == "poses":
            out["policy"] = self._observe_poses(reset_ids)
        extra = torch.cat((self.err_ctrl, torch.stack((self.EI / 1e-3, self.GJ / self.EI), -1),
                           self.cable_bend()[:, None] / 0.2), -1)
        out["critic"] = torch.cat((out["critic"], torch.nan_to_num(extra, 0.0, 0.0, 0.0)), -1)
        return out

    @property
    def pose_frame_dim(self) -> int:
        return 18 + (6 if self.cfg.obs_action else 0)

    def actor_obs_map(self, old_dim: int):
        """Where an older poses-only actor's inputs sit in this one's (for --init_from across a wider observation):
        the old (history_old x 18) frames, oldest first, map onto this one's newest frames' first 18 entries."""
        F = self.pose_frame_dim
        if self.cfg.obs_mode != "poses" or old_dim % 18 or old_dim // 18 > self.cfg.history:
            return None
        h_old = old_dim // 18
        idx = [(self.cfg.history - h_old + k) * F + j for k in range(h_old) for j in range(18)]
        return torch.tensor(idx, dtype=torch.long)

    def _observe_poses(self, reset_ids=None):
        """The in-hand connector pose and the observed socket's, both in the hand frame (position / 0.1 m and the
        rotation's first two columns), over the last `history` steps."""
        R_Xt = self.R_X.mT
        R_hp = R_Xt @ self.R_true  # the plug in the hand, live
        p_hp = (R_Xt @ (self.p_true - self.p_X)[..., None])[..., 0]
        R_m = R_hp @ self.R_D  # as measured (biased by the grasp's knowledge error)
        p_m = p_hp + (R_hp @ self.p_D[..., None])[..., 0]
        R_hs = R_Xt @ self.R_obs  # the observed socket in the hand
        p_hs = (R_Xt @ (self.p_obs - self.p_X)[..., None])[..., 0]
        parts = [p_m / 0.1, R_m[:, :, 0], R_m[:, :, 1], p_hs / 0.1, R_hs[:, :, 0], R_hs[:, :, 1]]
        if self.cfg.obs_action:
            parts.append(self.actions)  # what it commanded, against what moved: contact, the cable's give
        frame = torch.cat(parts, -1)
        self.hist_pose = torch.cat((self.hist_pose[:, 1:], frame[:, None]), 1)
        if reset_ids is not None:
            self.hist_pose[reset_ids] = frame[reset_ids, None, :].expand(-1, self.cfg.history, -1)
        return torch.nan_to_num(self.hist_pose.flatten(1), 0.0, 0.0, 0.0)

    def cable_bend(self):
        """The cable's total bend [rad]: the angle between the plug's axis and the cable's at the grasp."""
        R_c = self.R_X @ self.R_FP_true  # the plug's orientation if the cable were straight
        z = (R_c[:, :, 2] * self.R_true[:, :, 2]).sum(-1).clamp(-1.0, 1.0)
        return torch.acos(z)

    # ---- drawing -----------------------------------------------------------------------------------

    def cable_points(self, ids):
        """The cable's centreline, hand to overmold, in the socket frame (len(ids), N + 3, 3) [m]: a stub above
        the grasp, the grasp point, each link's top, the overmold's back."""
        um, m = self.um, self.cfg.model
        bq = self._bq()[ids]
        R_X, p_X = self.R_X[ids], self.p_X[ids]
        R_FC = self.R_FP0 @ self.R_C[ids]
        p_FC = (self.R_FP0 @ self.p_C[ids][..., None])[..., 0]
        g = p_X + (R_X @ p_FC[..., None])[..., 0]
        up = (R_X @ R_FC)[:, :, 2]
        pts = [g + 0.03 * up, g] + [bq[:, b, 0:3] for b in um.link_bodies]
        end = torch.tensor([0.0, PLUG_DY, OVERMOLD_BACK_Z], device=self.device)
        pts.append(self.p_true[ids] + (self.R_true[ids] @ end))
        return torch.stack(pts, 1)

    def snapshot(self, ids) -> dict:
        h, r, tilt, seated, engaged = self.plug_state()
        across = CABLE_RADIUS + PAD_HALF[1]
        R_X, p_X = self.R_X[ids], self.p_X[ids]
        p_pad = torch.stack([p_X + R_X[:, :, 1] * sgn * across for sgn in (-1.0, 1.0)], 1)
        R_pad = torch.stack([R_X, R_X], 1)
        out = {
            "p_true": self.p_true[ids], "R_true": self.R_true[ids], "R_pad": R_pad, "p_pad": p_pad,
            "cable": self.cable_points(ids), "bend": self.cable_bend()[ids], "EI": self.EI[ids],
            "h": h[ids], "seated": seated[ids], "engaged": engaged[ids], "voided": self.voided[ids],
            "force": torch.linalg.vector_norm(self._wrench()[ids, 0:3], dim=-1), "push": self.meas_push[ids],
            "detent": self.detent_now[ids], "err": self.err[ids], "scale": self.ep_scale[ids], "speed": self.speed[ids],
            "ctrl": self.err_ctrl[ids],
        }
        return {k_: v.detach().cpu().numpy() for k_, v in out.items()}


__all__ = ["CableDomainCfg", "CableEnvCfg", "CableErrorCfg", "GraspControlCfg", "UsbCableEnv", "quat_to_rot"]
