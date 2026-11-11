"""BNC insertion with the bench plug, in Newton (MuJoCo-Warp, SDF mesh contact), as an RSL-RL VecEnv.

THE TASK (the bench's methodology, bnc_assembly.yaml): the gripper holds the plug at its strain
relief above the socket; push it on and rock it about its axis until the collar -- which turns on the
body, dragged only by the friction of its seat -- finds the bayonet pins, and the pins run down the
straight slots to the slot floor. Success is that SEATED state (the bench's "connector clocking"
success, a 5 mm advance). The locking quarter-turn after it is the bench's scripted collar regrasp
and twist, not part of this policy.

ROBUSTNESS. Two estimation errors, drawn per episode, which the policy never observes:
  - IN-HAND: the plug sits in the fingers off the pose the robot believes, by an SE(2) error in the
    fingertips' midplane (the plane between the pads, normal along the closing axis = tool y): a shift
    along the pads (x), along the plug axis (z), and a turn about the closing axis (y).
  - TARGET: the socket's pose is observed with a 6-DOF error (position and orientation).
and domain randomisation: the friction of the plug on the socket (`mu_parts`) and of the collar on
its seat (`mu_seat`, which is what makes the collar follow the body), the arm + grasp stiffness, and
the collar's wave-washer stiffness. The critic sees all of it (asymmetric actor-critic).

FRAMES (all per world; the world frame IS the true socket frame):
  S_obs  the observed socket                        = true socket (+) target error
  P_bel  the plug where the robot believes it is      = tool pose (+) nominal grasp
  F      the fingertip frame = P_bel . G, G = (0, 0, GRASP_Z): the pads' middle on the strain relief
  X      what the D6 drive moves = F . E, E the in-hand error   (the true plug is X . G^-1)
The policy acts on P_bel in S_obs: its action moves P_bel's TARGET, the drive follows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import warp as wp

from .model import COLLAR_SLIDE_RANGE, GRASP_Z, BncModel, ModelCfg, build, make_solver  # model first: it configures newton

import newton  # noqa: E402
from newton import ModelFlags  # noqa: E402


@dataclass
class ErrorCfg:
    """Half-widths of the uniform per-episode estimation errors (scaled by `noise_scale`)."""

    in_hand_x: float = 0.002
    """Along the pads, in the midplane [m]."""
    in_hand_z: float = 0.003
    """Along the plug axis [m]."""
    in_hand_rot_deg: float = 5.0
    """About the jaw-closing axis (the midplane's normal)."""
    target_xy: float = 0.002
    target_z: float = 0.0015
    target_tilt_deg: float = 3.0
    """Roll and pitch of the socket's axis, each."""
    target_yaw_deg: float = 10.0


@dataclass
class DomainCfg:
    mu_parts: tuple[float, float] = (0.15, 0.45)
    """Plug (collar, contact fingers, insulator) on the socket. Nominal 0.3 (build_bnc.py)."""
    mu_seat: tuple[float, float] = (0.25, 0.55)
    """Collar on its seat ring. Nominal 0.35."""
    lin_stiffness: tuple[float, float] = (5.0e3, 2.0e4)
    """Arm + grasp, at the fingertips [N/m]."""
    rot_stiffness: tuple[float, float] = (5.0, 20.0)
    """Arm + grasp about the fingertips [N m/rad]: the rubber strain relief in the pads."""
    washer_stiffness: tuple[float, float] = (3500.0, 6500.0)
    hand_mass: tuple[float, float] = (0.25, 0.25)
    """Reflected mass of arm + gripper at the fingertips [kg] (ModelCfg.hand_mass nominal)."""
    action_delay: tuple[int, int] = (0, 0)
    """Policy steps (inclusive range) between the policy's action and the drive receiving it: the
    controller's latency (one step = 50 ms at 20 Hz)."""


WIDE_DOMAIN = DomainCfg(mu_parts=(0.1, 0.6), mu_seat=(0.2, 0.65), lin_stiffness=(4.0e3, 2.5e4), rot_stiffness=(4.0, 25.0),
                        washer_stiffness=(3000.0, 7500.0), hand_mass=(0.15, 0.5), action_delay=(0, 1))
"""Wider ranges for the pose + wrench policy (train --wide_dr)."""


@dataclass
class EnvCfg:
    num_envs: int = 1024
    physics_hz: float = 1000.0
    """1 kHz: with the stiff contacts (ModelCfg.contact_ke) it keeps interference ~0.1 mm; 2 kHz is no better."""
    policy_hz: float = 20.0
    collide_every: int = 1
    """Collision every Nth substep. At 2, the stale contacts of the skipped substep let the bayonet pins
    pass through a misclocked collar at ~25 N (38-55% of pushes), at no saving in speed."""
    episode_s: float = 8.0
    model: ModelCfg = field(default_factory=ModelCfg)
    errors: ErrorCfg = field(default_factory=ErrorCfg)
    domain: DomainCfg = field(default_factory=DomainCfg)
    noise_scale: float = 1.0
    """Multiplies every estimation error."""
    curriculum_steps: int = 0
    """If > 0: noise_scale ramps from curriculum_start to 1 over this many policy steps (training)."""
    curriculum_start: float = 0.25
    randomize_domain: bool = True

    # The start: the believed plug above the observed socket.
    start_height: tuple[float, float] = (0.015, 0.020)
    start_xy: float = 0.0015
    start_yaw_deg: float = 30.0

    # The action: displacement of P_bel's target per policy step at |a| = 1, in S_obs.
    pos_scale: float = 0.001
    rot_scale_deg: float = 3.0
    max_lead: tuple[float, float] = (0.002, math.radians(6.0))
    """How far the target may run ahead of the measured pose (caps the contact force at ~K x lead)."""
    pos_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.006, -0.006, -0.004), (0.006, 0.006, 0.030))
    """Of the target in S_obs [m]. 6 mm keeps the plug off the 11.7 mm beside-the-barrel exploit."""
    max_tilt_deg: float = 12.0
    max_force: float = 30.0
    max_torque: float = 1.0
    """Cap on the grasp's rotational spring [N m], applied like max_force (0.5 was too weak to hold the plug
    against the contact moments: 19% vs 52% at full error for push + base twist). A 2F-85 holding the strain
    relief lets the plug turn in its pads well below the ~2.8 N m that in-hand turn plus target tilt
    errors produced uncapped; there, the soft contacts let the sleeve tilt 4-6 deg into the barrel
    (v5: every lost seat and most collars stuck on the pins). inf disables it."""
    base_yaw: float = 0.0
    """A base twist added to the policy's yaw action (x rot_scale_deg per step) while the MEASURED push
    exceeds base_yaw_push: the collar angle is unobservable, and turning one way while pressing always
    finds a slot within 180 deg. Policies without it dithered (v3: net ~0 twist on misclocked collars).
    Sensor-based, so it deploys as part of the controller. 0 disables it."""
    base_yaw_push: float = 5.0
    base_push: float = 0.0
    turn_band_deg: tuple[float, float] | None = None
    """The wrist's turn budget: the believed plug's turn about the observed socket axis since the start
    (unwrapped, + = base_yaw's direction) is held inside this band; the target is clamped at its edges,
    as the bench's wrist can roll only so far ([-120, 120] deg, started at one end: (0, 230)). The base
    twist reverses near each edge (turn_margin_deg), so it rocks instead of winding up, and the policy
    observes the turn and the twist's direction. None: no budget (and no extra observations)."""
    turn_margin_deg: float = 20.0
    """A base downward action (x pos_scale per step, along the observed socket axis) always added to the
    policy's: with base_yaw it makes a push-and-twist controller (52% seated at full error alone), and
    the policy learns a residual on it. 0 disables it."""
    """Cap on the arm spring's force [N]: the target is pulled back toward the measured pose so that
    K x stretch stays under it. Above ~30 N the contacts start to give (pins through the collar), and
    a BNC seats by hand with less. inf disables it."""

    # What the robot measures.
    wrench_noise: tuple[float, float] = (0.5, 0.02)
    """Per-step Gaussian std of force [N] and torque [N m] (the iiwa's torque-based estimate)."""
    wrench_bias: tuple[float, float] = (1.0, 0.03)
    """Per-episode uniform bias half-width."""
    history: int = 3
    pose_noise: tuple[float, float] = (0.0, 0.0)
    """Per-step Gaussian std of the connector pose the "pose_wrench" policy observes: position [m] and
    rotation [deg] (a tracker's jitter on top of the per-episode in-hand / target errors)."""
    noise_level: tuple[float, float] = (1.0, 1.0)
    """Per-episode uniform multiplier on pose_noise and wrench_noise (sensor quality varies)."""
    pw_history: int = 8
    """Frames (pose + wrench) in the "pose_wrench" observation."""
    observe_twice: bool = False
    """Reproduce a bug the 2026-10-30 policy (model_700) was trained with: on every step with a reset
    anywhere -- nearly every step, episodes being staggered -- ALL worlds were observed a second time
    right after the first, so the observed velocities (finite differences over the step) read zero.
    The policy never saw a velocity. Set it to run that policy."""

    # Outcome thresholds, on the TRUE plug in the TRUE socket.
    seated_height: float = 0.0016
    """Pins at the slot floor read 1.08 mm (0.6 in this model: the collar slide sits 0.24 mm past
    its stop under load) and lower once the helix starts; resting on the pins reads 5.8."""
    engaged_height: float = 0.0045
    center_tol: float = 0.0008
    tilt_tol_deg: float = 3.0
    force_limit: float = 30.0
    clock_shaping: float = 3.0
    """Weight of a potential-based shaping term on the TRUE collar angle off its slot entries, while
    the collar can be on the pins: potential -w x angle / 90 deg. Policy-invariant (Ng et al. 1999);
    it only speeds up learning to turn the collar in, which was v3's main failure (collars left on
    the pins 54 deg off). The policy never observes the angle."""
    clock_band: tuple[float, float] = (0.0045, 0.0080)
    """Plug heights where the collar is at the pins (the shaping applies here)."""
    jam_shaping: float = 0.0
    """Weight of a potential-based term on the TRUE jam depth: while the collar lip is beside the barrel
    (> 9 mm off-axis), potential -w x (rim height - lip height) / 6 mm. v12 jammed there in 4.6% of
    full-error episodes and almost never lifted out (median lift 0.03 mm in the 12 s left): lifting first
    costs keypoint reward. Policy-invariant; 0 disables it."""
    shaping_gamma: float = 0.99
    """PPO's discount, for the shaping term."""

    # Simulator artifacts, on the TRUE plug: an episode that reaches a pose real parts cannot is voided
    # (ends as a time-out, no reward, not a success), so a policy cannot learn to exploit lost contacts.
    void_artifacts: bool = True
    artifacts_terminal: bool = False
    """End an artifact episode as a FAILURE (terminal, no value bootstrap) instead of a neutral time-out:
    the policy then learns to avoid what provokes them -- mostly pushing the collar onto the barrel rim
    off-centre, where it slides through the 0.7 mm wall. Non-finite physics stays a time-out."""
    pass_through_band: tuple[float, float] = (0.0028, 0.0046)
    """Plug heights where the pins must be in the slot's straight part: above, the collar can rest on
    the pins; below, the washer's give and the lock turn allow any angle."""
    pass_through_deg: float = 15.0
    """Collar angle off the slot entries, in that band, that means the pins went through the collar
    (clean runs: p99 3 deg)."""
    lip_z: float = 0.003
    """The collar's front lip on the plug axis, in the plug frame [m]."""
    off_axis_check: bool = False
    """Void plugs below 5.6 mm with the origin more than 1.5 mm off-axis (the sleeve through the barrel).
    Off since 2026-10-31: with the fixed contacts what it caught were real jams -- the collar's lip BESIDE
    the barrel in the socket's well (checked: no penetration) -- which it ended (terminal) before the
    policy could back out. Not needed for honest scores: an off-axis plug can never count as seated."""
    slide_overrun: float = 0.0005
    """Past the collar's rear stop (the body pushed through the collar)."""


# ---- rotation helpers (torch, batched) ------------------------------------------------------------

def rot_x(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack((o, z, z, z, c, -s, z, s, c), -1).view(*a.shape, 3, 3)


def rot_y(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack((c, z, s, z, o, z, -s, z, c), -1).view(*a.shape, 3, 3)


def rot_z(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).view(*a.shape, 3, 3)


def euler_xyz_to_rot(e):
    """R = Rx(a) Ry(b) Rz(c): Newton's D6 with angular axes x, y, z (checked against eval_fk)."""
    return rot_x(e[..., 0]) @ rot_y(e[..., 1]) @ rot_z(e[..., 2])


def rot_to_euler_xyz(R):
    b = torch.asin(R[..., 0, 2].clamp(-1.0, 1.0))
    a = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    c = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack((a, b, c), -1)


def rotvec_to_rot(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    k = v / angle.clamp_min(1e-9)
    K = torch.zeros(*v.shape[:-1], 3, 3, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.mT
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return torch.eye(3, device=v.device) + s * K + (1 - c) * (K @ K)


def rot_to_rotvec(R):
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1.0, 1.0)
    angle = torch.acos(cos)
    w = torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1)
    # Small rotations only here (leads, rates): angle / (2 sin angle), its series near 0.
    scale = torch.where(angle > 1e-4, angle / (2 * torch.sin(angle)).clamp_min(1e-9), 0.5 + angle**2 / 12)
    return w * scale[..., None]


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def uniform(shape, lo, hi, device):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def sym(shape, half, device):
    return (2 * torch.rand(shape, device=device) - 1) * half


class BncNewtonEnv:
    """RSL-RL VecEnv (duck-typed: rsl_rl.env.VecEnv's attributes and methods)."""

    def __init__(self, cfg: EnvCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = n = cfg.num_envs
        self.num_actions = 6
        self.substeps = int(round(cfg.physics_hz / cfg.policy_hz))
        self.sim_dt = 1.0 / cfg.physics_hz
        self.step_dt = self.substeps * self.sim_dt
        self.max_episode_length = int(round(cfg.episode_s / self.step_dt))

        wp.init()
        self.bm: BncModel = build(replace(cfg.model, num_worlds=n), device=device)
        self.model = self.bm.model
        self.pipeline, self.solver = make_solver(self.bm, cfg.model)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        bm, dev = self.bm, self.device
        self.C, self.D = bm.coords_per_world, bm.dofs_per_world

        # Per-world layout checks: shapes and DOFs are replicated world by world.
        shape_world = self.model.shape_world.numpy()
        assert np.all(shape_world == np.repeat(np.arange(n), bm.shapes_per_world)), "unexpected shape layout"
        self._shape_mu = wp.to_torch(self.model.shape_material_mu).view(n, bm.shapes_per_world)
        self._dof_ke = wp.to_torch(self.model.joint_target_ke).view(n, self.D)
        self._dof_kd = wp.to_torch(self.model.joint_target_kd).view(n, self.D)
        self._target_q = wp.to_torch(self.control.joint_target_q).view(n, self.C)
        self._shape_mu[:, bm.collar_shapes] = 0.0

        # Constant frames.
        self.G_p = torch.tensor([0.0, 0.0, GRASP_Z], device=dev)
        I3 = torch.eye(3, device=dev)

        # Per-episode quantities.
        self.R_obs = I3.repeat(n, 1, 1)
        self.p_obs = torch.zeros(n, 3, device=dev)
        self.R_E = I3.repeat(n, 1, 1)
        self.p_E = torch.zeros(n, 3, device=dev)
        self.R_tgt = I3.repeat(n, 1, 1)  # P_bel's target
        self.p_tgt = torch.zeros(n, 3, device=dev)
        self.k_lin = torch.full((n,), cfg.model.lin_ke, device=dev)
        self.k_rot = torch.full((n,), cfg.model.rot_ke, device=dev)
        self.mu = torch.zeros(n, 2, device=dev)
        self.err = torch.zeros(n, 9, device=dev)  # in-hand (x, z, rot), target (xyz, rpy) -- for the critic
        self.wrench_bias = torch.zeros(n, 6, device=dev)
        self.prev_bel = (I3.repeat(n, 1, 1), torch.zeros(n, 3, device=dev))
        self.actions = torch.zeros(n, 6, device=dev)
        self.hist = torch.zeros(n, cfg.history, 15, device=dev)
        self.hist_pw = torch.zeros(n, cfg.pw_history, 15, device=dev)
        self.noise_lvl = torch.ones(n, device=dev)
        self.delay = torch.zeros(n, dtype=torch.bool, device=dev)
        self.a_prev = torch.zeros(n, 6, device=dev)
        self.mass = torch.full((n,), cfg.model.hand_mass, device=dev)
        self.turn = torch.zeros(n, device=dev)  # unwrapped turn of the believed plug since the start [rad]
        self.yaw_prev = torch.zeros(n, device=dev)
        self.yaw_dir = torch.ones(n, device=dev)  # the base twist's direction (+1 / -1)
        B = bm.bodies_per_world
        self._body_mass = wp.to_torch(self.model.body_mass).view(n, B)
        self._body_inv_mass = wp.to_torch(self.model.body_inv_mass).view(n, B)
        self._body_inertia = wp.to_torch(self.model.body_inertia).view(n, B, 3, 3)
        self._body_inv_inertia = wp.to_torch(self.model.body_inv_inertia).view(n, B, 3, 3)
        self._inertia0 = self._body_inertia[0, bm.plug_body].clone()
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.first_seated = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.peak_force = torch.zeros(n, device=dev)
        self.ep_return = torch.zeros(n, device=dev)
        self.prev_phi = torch.zeros(n, device=dev)  # the collar-clocking potential at the last step
        self.prev_jam = torch.zeros(n, device=dev)  # the jam-escape potential at the last step
        self.touched = torch.zeros(n, dtype=torch.bool, device=dev)  # measured push > 5 N since the reset
        self.nonfinite_count = 0
        self.overflow_steps = torch.zeros((), device=dev)  # steps whose contacts overflowed the buffer
        self.artifact_counts = torch.zeros(3, device=dev)  # voided episodes: pass-through, off-axis, overrun
        self.voided = torch.zeros(n, dtype=torch.bool, device=dev)  # this step's voided worlds
        self.step_count = 0
        self._outcomes: list[torch.Tensor] = []
        self.extras: dict = {}

        self._reset(torch.arange(n, device=dev))
        self._read()
        self._obs = self._observe()

    # ---- simulation --------------------------------------------------------------------------------

    def _q(self):
        return wp.to_torch(self.s0.joint_q).view(self.num_envs, self.C)

    def _qd(self):
        return wp.to_torch(self.s0.joint_qd).view(self.num_envs, self.D)

    def _simulate(self):
        for i in range(self.substeps):
            if i % self.cfg.collide_every == 0:
                self.pipeline.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.sim_dt)
            self.s0, self.s1 = self.s1, self.s0

    def _guard(self):
        """Worlds whose state went non-finite (a rare contact blow-up): put them back at a finite pose and
        clear MuJoCo's warm start, so the reset that follows starts clean. Returns the mask."""
        q, qd = self._q(), self._qd()
        bad = ~(torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1))
        if bad.any():
            q[bad] = 0.0
            q[bad, 2] = 0.05
            qd[bad] = 0.0
            self._clear_solver(bad)
            newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        return bad

    def _clear_solver(self, mask):
        """Zero MuJoCo's per-world carry-over (warm start, applied forces) in the masked worlds."""
        m = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        m[:-1] = mask
        self.solver.reset(self.s0, world_mask=wp.from_torch(m), flags=0)

    def _read(self):
        """Measured state: the D6 pose X -> fingertip F -> believed plug; and the true plug."""
        q = self._q()
        self.R_X = euler_xyz_to_rot(q[:, 3:6])
        self.p_X = q[:, 0:3].clone()
        R_F = self.R_X @ self.R_E.mT
        p_F = self.p_X - (R_F @ self.p_E[..., None])[..., 0]
        self.R_bel, self.p_bel = R_F, p_F - (R_F @ self.G_p)
        self.R_true, self.p_true = self.R_X, self.p_X - (self.R_X @ self.G_p)
        self.collar_turn = q[:, self.bm.turn_q].clone()
        self.slide = q[:, self.bm.slide_q].clone()

    def _yaw_in_obs(self, R):
        """The yaw of R's x axis about the observed socket's axis (as rl_insertion.py measures it)."""
        x = (self.R_obs.mT @ R[:, :, 0:1])[..., 0]
        return torch.atan2(x[:, 1], x[:, 0])

    def _track_turn(self, reset_ids=None):
        yaw = self._yaw_in_obs(self.R_bel)
        if reset_ids is None:
            self.turn = self.turn + wrap(yaw - self.yaw_prev)
        else:
            self.turn[reset_ids] = 0.0
            self.yaw_dir[reset_ids] = 1.0
        self.yaw_prev = yaw

    def _drive(self):
        """Send P_bel's target to the D6: X_t = P_tgt . G . E, as euler angles unwrapped onto q's."""
        R_xt = self.R_tgt @ self.R_E
        p_xt = self.p_tgt + (self.R_tgt @ (self.G_p + self.p_E)[..., None])[..., 0]
        e = rot_to_euler_xyz(R_xt)
        q = self._q()
        e = q[:, 3:6] + wrap(e - q[:, 3:6])
        self._target_q[:, 0:3] = p_xt
        self._target_q[:, 3:6] = e
        self.R_xt, self.p_xt = R_xt, p_xt

    # ---- VecEnv ------------------------------------------------------------------------------------

    def get_observations(self):
        from tensordict import TensorDict

        return TensorDict(self._obs, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        if cfg.curriculum_steps > 0:
            self.step_count += 1
            cfg.noise_scale = min(1.0, cfg.curriculum_start + (1 - cfg.curriculum_start) * self.step_count / cfg.curriculum_steps)
        a = actions.clamp(-1.0, 1.0)
        self.actions = a
        if cfg.turn_band_deg is not None:
            lo, hi = (math.radians(x) for x in cfg.turn_band_deg)
            m = math.radians(cfg.turn_margin_deg)
            self.yaw_dir = torch.where(self.turn > hi - m, -1.0, torch.where(self.turn < lo + m, 1.0, self.yaw_dir))
        if cfg.base_yaw or cfg.base_push:
            pushing = (self.meas_push > cfg.base_yaw_push).float()
            a = a.clone()
            a[:, 5] = (a[:, 5] + cfg.base_yaw * pushing * self.yaw_dir).clamp(-1.0, 1.0)
            a[:, 2] = (a[:, 2] - cfg.base_push).clamp(-1.0, 1.0)
        if cfg.domain.action_delay[1] > 0:
            a, self.a_prev = torch.where(self.delay[:, None], self.a_prev, a), a

        self._apply_action(a)
        self._simulate()
        # A full contact buffer drops contacts in random worlds, silently: count it (no host sync).
        self.overflow_steps += (wp.to_torch(self.contacts.rigid_contact_count)[0] > self.contacts.rigid_contact_max).float()
        bad = self._guard()
        self._read()
        self._track_turn()
        self.episode_length_buf += 1

        reward, terms, seated, engaged, force = self._reward(a)
        if cfg.clock_shaping:
            self.prev_phi = self._phi  # reset worlds get their new start's potential in _reset
        if cfg.jam_shaping:
            self.prev_jam = self._jam
        self.nonfinite_count += int(bad.sum())
        terminal = torch.zeros_like(bad)
        if cfg.void_artifacts:
            art = self._artifacts() & ~bad[:, None]
            self.artifact_counts += art.sum(0)
            if cfg.artifacts_terminal:
                terminal = art.any(-1)
            bad = bad | art.any(-1)
        self.voided = bad
        reward = torch.where(bad, 0.0, reward)
        force = torch.where(bad, 0.0, force)
        seated, engaged = seated & ~bad, engaged & ~bad
        self.ep_return += reward
        just = seated & (self.first_seated < 0)
        self.first_seated[just] = self.episode_length_buf[just]
        self.peak_force = torch.maximum(self.peak_force, force)

        # A world whose physics went non-finite, or showed an artifact, ends as a time-out: PPO bootstraps
        # it from the value, so the artifact is neither rewarded nor punished. It is not a success.
        self.first_seated[bad] = -1
        time_out = ((self.episode_length_buf >= self.max_episode_length) | bad) & ~terminal
        dones = time_out | terminal
        self.extras = {"time_outs": time_out, "log": {f"/reward/{k}": v.mean() for k, v in terms.items()}}
        if cfg.turn_band_deg is not None:
            self.extras["log"]["/metrics/turn_deg_min"] = torch.rad2deg(self.turn.min())
            self.extras["log"]["/metrics/turn_deg_max"] = torch.rad2deg(self.turn.max())
            self.extras["log"]["/metrics/twist_reversed"] = (self.yaw_dir < 0).float().mean()
        self.extras["log"]["/metrics/nonfinite_total"] = torch.tensor(float(self.nonfinite_count))
        self.extras["log"]["/metrics/contact_overflow_steps"] = self.overflow_steps
        for i, k in enumerate(("pass_through", "off_axis", "slide_overrun")):
            self.extras["log"][f"/metrics/artifact_{k}"] = self.artifact_counts[i]
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids):
            self._record_outcomes(done_ids, seated, engaged)
        if len(done_ids):
            self._reset(done_ids)
            self._read()
        if cfg.observe_twice:
            self._observe()
            self._obs = self._observe(reset_ids=done_ids)
        else:
            self._obs = self._observe(reset_ids=done_ids if len(done_ids) else None)
        return self.get_observations(), reward, dones, self.extras

    def _apply_action(self, a):
        cfg = self.cfg
        rot_scale = math.radians(cfg.rot_scale_deg)
        d_pos = (self.R_obs @ (a[:, 0:3] * cfg.pos_scale)[..., None])[..., 0]
        d_rot = (self.R_obs @ (a[:, 3:6] * rot_scale)[..., None])[..., 0]
        lead_p, lead_r = cfg.max_lead
        p = self.p_bel + (self.p_tgt + d_pos - self.p_bel).clamp(-lead_p, lead_p)
        rel = (self.R_obs.mT @ (p - self.p_obs)[..., None])[..., 0]
        lo, hi = (torch.tensor(b, device=self.device) for b in cfg.pos_bounds)
        rel = torch.maximum(torch.minimum(rel, hi), lo)
        self.p_tgt = self.p_obs + (self.R_obs @ rel[..., None])[..., 0]

        R = rotvec_to_rot(d_rot) @ self.R_tgt
        lead = rot_to_rotvec(R @ self.R_bel.mT)
        ang = torch.linalg.vector_norm(lead, dim=-1, keepdim=True)
        R = rotvec_to_rot(lead * (lead_r / ang.clamp_min(1e-9)).clamp(max=1.0)) @ self.R_bel
        # Tilt bound: the plug axis within max_tilt of the observed socket axis.
        R_rel = self.R_obs.mT @ R
        z = R_rel[:, :, 2]
        tilt = torch.acos(z[:, 2].clamp(-1.0, 1.0))
        excess = (tilt - math.radians(cfg.max_tilt_deg)).clamp_min(0.0)
        axis = torch.linalg.cross(z, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(z))
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-9)
        R_rel = rotvec_to_rot(axis * excess[:, None]) @ R_rel
        self.R_tgt = self.R_obs @ R_rel
        if cfg.turn_band_deg is not None:
            # Hold the target's turn inside the wrist's band.
            lo, hi = (math.radians(x) for x in cfg.turn_band_deg)
            t = self.turn + wrap(self._yaw_in_obs(self.R_tgt) - self._yaw_in_obs(self.R_bel))
            excess = t - t.clamp(lo, hi)
            self.R_tgt = self.R_obs @ rot_z(-excess) @ self.R_obs.mT @ self.R_tgt
        if math.isfinite(cfg.max_torque):
            # The rotational spring's stretch at this target (X's target against X), capped at max_torque / K.
            R_xt = self.R_tgt @ self.R_E
            st = rot_to_rotvec(R_xt @ self.R_X.mT)
            ang = torch.linalg.vector_norm(st, dim=-1, keepdim=True)
            cap = (cfg.max_torque / self.k_rot)[:, None]
            scale = (cap / ang.clamp_min(1e-9)).clamp(max=1.0)
            self.R_tgt = rotvec_to_rot(st * scale) @ self.R_X @ self.R_E.mT
        if math.isfinite(cfg.max_force):
            # The arm spring's stretch at this target (X's target minus X), capped at max_force / K.
            s = self.p_tgt + (self.R_tgt @ (self.G_p + self.p_E)[..., None])[..., 0] - self.p_X
            cap = (cfg.max_force / self.k_lin)[:, None]
            norm = torch.linalg.vector_norm(s, dim=-1, keepdim=True)
            self.p_tgt = self.p_tgt - s * (1.0 - (cap / norm.clamp_min(1e-9)).clamp(max=1.0))
        self._drive()

    # ---- observations ------------------------------------------------------------------------------

    def _wrench(self):
        """What the hand applies to the plug at the fingertips, from the drive's spring (world frame)."""
        f = self.k_lin[:, None] * (self.p_xt - self.p_X)
        t = self.k_rot[:, None] * rot_to_rotvec(self.R_xt @ self.R_X.mT)
        return torch.cat((f, t), -1)

    def _observe(self, reset_ids=None):
        cfg, n, dev = self.cfg, self.num_envs, self.device
        Ro = self.R_obs.mT
        p_rel = (Ro @ (self.p_bel - self.p_obs)[..., None])[..., 0]
        R_rel = Ro @ self.R_bel
        rot6 = torch.cat((R_rel[:, :, 0], R_rel[:, :, 1]), -1)
        w = self._wrench()
        self.true_wrench = w
        f_obs = (Ro @ w[:, 0:3, None])[..., 0]
        t_obs = (Ro @ w[:, 3:6, None])[..., 0]
        noise = torch.randn(n, 6, device=dev) * torch.tensor(
            [cfg.wrench_noise[0]] * 3 + [cfg.wrench_noise[1]] * 3, device=dev) * self.noise_lvl[:, None]
        meas = torch.cat((f_obs, t_obs), -1) + noise + self.wrench_bias
        self.meas_push = -meas[:, 2]  # the measured push along the observed socket axis [N]
        frame = torch.cat((p_rel / 0.01, rot6, meas[:, 0:3] / 10.0, meas[:, 3:6] / 0.2), -1)
        # Every world's history advances; a world just reset starts over, filled with its first frame.
        # (Until 2026-10-30 a reset anywhere skipped the advance everywhere: with staggered episodes the
        # other worlds' histories stayed frozen near their reset frame. model_700 and v2 trained on that.)
        self.hist = torch.cat((self.hist[:, 1:], frame[:, None]), 1)
        if reset_ids is not None:
            self.hist[reset_ids] = frame[reset_ids, None, :].expand(-1, cfg.history, -1)
        # Pose + wrench only: the connector's pose in the observed socket with tracker jitter, and the
        # measured wrench -- no velocities, no own actions, no controller state.
        sp, sr = cfg.pose_noise
        lvl = self.noise_lvl[:, None]
        p_n = p_rel + torch.randn(n, 3, device=dev) * sp * lvl
        R_n = rotvec_to_rot(torch.randn(n, 3, device=dev) * math.radians(sr) * lvl) @ R_rel
        frame_pw = torch.cat((p_n / 0.01, R_n[:, :, 0], R_n[:, :, 1], meas[:, 0:3] / 10.0, meas[:, 3:6] / 0.2), -1)
        self.hist_pw = torch.cat((self.hist_pw[:, 1:], frame_pw[:, None]), 1)
        if reset_ids is not None:
            self.hist_pw[reset_ids] = frame_pw[reset_ids, None, :].expand(-1, cfg.pw_history, -1)
        pose_wrench = self.hist_pw.flatten(1)
        Rp, pp = self.prev_bel
        vel = (Ro @ ((self.p_bel - pp) / self.step_dt)[..., None])[..., 0] / 0.02
        ang = (Ro @ rot_to_rotvec(self.R_bel @ Rp.mT)[..., None])[..., 0] / self.step_dt
        if reset_ids is not None:
            vel[reset_ids] = 0.0
            ang[reset_ids] = 0.0
        self.prev_bel = (self.R_bel.clone(), self.p_bel.clone())
        lead_p = (Ro @ (self.p_tgt - self.p_bel)[..., None])[..., 0] / cfg.max_lead[0]
        lead_r = (Ro @ rot_to_rotvec(self.R_tgt @ self.R_bel.mT)[..., None])[..., 0] / cfg.max_lead[1]
        policy = torch.cat((self.hist.flatten(1), vel, ang, self.actions, lead_p, lead_r), -1)

        # Privileged: the true plug in the true socket, the collar, the errors, the domain.
        rot6_t = torch.cat((self.R_true[:, :, 0], self.R_true[:, :, 1]), -1)
        collar = self.collar_turn + torch.atan2(self.R_true[:, 1, 0], self.R_true[:, 0, 0])
        dom = torch.stack((self.mu[:, 0], self.mu[:, 1], self.k_lin / 1e4, self.k_rot / 10.0), -1)
        critic = torch.cat((policy, self.p_true / 0.01, rot6_t, torch.sin(2 * collar)[:, None],
                            torch.cos(2 * collar)[:, None], self.slide[:, None] / 0.001, self.err, dom,
                            w[:, 0:3] / 10.0, w[:, 3:6] / 0.2), -1)
        # A distillation teacher's: privileged, but only what touch could reveal -- not the collar angle or
        # slide, which the student cannot know before the plug drops in (a teacher that knows which way to
        # twist cannot be imitated; it teaches the average).
        priv = torch.cat((self.p_true / 0.01, rot6_t, self.err, dom, w[:, 0:3] / 10.0, w[:, 3:6] / 0.2), -1)
        teacher = torch.cat((policy, priv), -1)
        # Blind until contact: the privileged part is zero until the measured push first exceeds 5 N (and a
        # flag says so). Before contact the student cannot know the errors either, so a teacher that used
        # them there taught it an average (tilted landings; distillation fell from 96% to 92%).
        self.touched = self.touched | (self.meas_push > 5.0)
        if reset_ids is not None:
            self.touched[reset_ids] = False
        t = self.touched.float()[:, None]
        teacher_gated = torch.cat((policy, t, priv * t), -1)
        if cfg.turn_band_deg is not None:
            lo, hi = (math.radians(x) for x in cfg.turn_band_deg)
            ex = torch.stack(((2 * self.turn - lo - hi) / (hi - lo), self.yaw_dir), -1)
            policy, pose_wrench, critic = (torch.cat((g, ex), -1) for g in (policy, pose_wrench, critic))
            teacher, teacher_gated = (torch.cat((g, ex), -1) for g in (teacher, teacher_gated))
        return {"policy": torch.nan_to_num(policy, 0.0, 0.0, 0.0), "pose_wrench": torch.nan_to_num(pose_wrench, 0.0, 0.0, 0.0),
                "critic_pw": torch.nan_to_num(torch.cat((critic, pose_wrench, self.mass[:, None] / 0.25,
                                                         self.delay.float()[:, None], self.noise_lvl[:, None]), -1), 0.0, 0.0, 0.0), "critic": torch.nan_to_num(critic, 0.0, 0.0, 0.0),
                "teacher": torch.nan_to_num(teacher, 0.0, 0.0, 0.0),
                "teacher_gated": torch.nan_to_num(teacher_gated, 0.0, 0.0, 0.0)}

    # ---- reward ------------------------------------------------------------------------------------

    def plug_state(self):
        cfg = self.cfg
        h = self.p_true[:, 2]
        r = torch.linalg.vector_norm(self.p_true[:, 0:2], dim=-1)
        tilt = torch.acos(self.R_true[:, 2, 2].clamp(-1.0, 1.0))
        centered = (r < cfg.center_tol) & (tilt < math.radians(cfg.tilt_tol_deg))
        return h, r, tilt, centered & (h < cfg.seated_height), centered & (h < cfg.engaged_height)

    def collar_mis(self):
        """The true collar angle off its slot entries (at 0 and pi), in [0, pi/2]."""
        col = self.collar_turn + torch.atan2(self.R_true[:, 1, 0], self.R_true[:, 0, 0])
        return wrap(2 * col).abs() / 2

    def _clock_potential(self):
        h = self.p_true[:, 2]
        lo, hi = self.cfg.clock_band
        on_pins = (h > lo) & (h < hi)
        return torch.where(on_pins, -self.cfg.clock_shaping * self.collar_mis() / (math.pi / 2), 0.0)

    def _jam_potential(self):
        lip = self.p_true + self.R_true[:, :, 2] * self.cfg.lip_z
        beside = torch.linalg.vector_norm(lip[:, 0:2], dim=-1) > 0.009
        depth = (0.01214 - lip[:, 2]).clamp_min(0.0)
        return torch.where(beside, -self.cfg.jam_shaping * depth / 0.006, 0.0)

    def _artifacts(self):
        """(n, 3) bool: the pins through the collar, the sleeve through the barrel, the body through the
        collar's stop. Thresholds calibrated on a clean run (0.7% of worlds flagged; 72% before the
        contact fix)."""
        cfg = self.cfg
        h, r, _, _, _ = self.plug_state()
        mis = self.collar_mis()
        lo, hi = cfg.pass_through_band
        lip = self.p_true + self.R_true[:, :, 2] * cfg.lip_z
        over = torch.linalg.vector_norm(lip[:, 0:2], dim=-1) < 0.001  # the sleeve is on the barrel
        through = over & (h > lo) & (h < hi) & (mis > math.radians(cfg.pass_through_deg))
        off = (h < 0.0056) & (r > 0.0015) & cfg.off_axis_check
        overrun = self.slide < COLLAR_SLIDE_RANGE[0] - cfg.slide_overrun
        return torch.stack((through, off, overrun), -1)

    def _reward(self, a):
        cfg = self.cfg
        h, r, tilt, seated, engaged = self.plug_state()
        # Keypoints on the plug axis (0, 15, 30 mm up) against the same points seated.
        offs = torch.tensor([0.0, 0.015, 0.030], device=self.device)
        kp = self.p_true[:, None, :] + self.R_true[:, None, :, 2] * offs[None, :, None]
        kp_goal = torch.zeros_like(kp)
        kp_goal[..., 2] = offs
        d = torch.linalg.vector_norm(kp - kp_goal, dim=-1).mean(-1)

        def squash(x, a_, b_):
            return 1.0 / (torch.exp(a_ * x) + b_ + torch.exp(-a_ * x))

        force = torch.linalg.vector_norm(self.true_wrench[:, 0:3] if hasattr(self, "true_wrench") else self._wrench()[:, 0:3], dim=-1)
        terms = {
            "kp_coarse": squash(d, 50.0, 2.0),
            "kp_fine": squash(d, 300.0, 0.0),
            "engaged": engaged.float(),
            "seated": 3.0 * seated.float(),
            "force": -0.05 * ((force - cfg.force_limit).clamp_min(0.0) / 10.0),
            "action": -0.01 * (a**2).sum(-1),
        }
        if cfg.clock_shaping:
            phi = self._clock_potential()
            terms["clock"] = cfg.shaping_gamma * phi - self.prev_phi
            self._phi = phi
        if cfg.jam_shaping:
            jam = self._jam_potential()
            terms["jam"] = cfg.shaping_gamma * jam - self.prev_jam
            self._jam = jam
        return sum(terms.values()), terms, seated, engaged, force

    def _record_outcomes(self, ids, seated, engaged):
        self._outcomes.append(torch.stack((seated[ids].float(), engaged[ids].float(),
                                           (self.first_seated[ids] >= 0).float(), self.peak_force[ids]), -1))
        rec = torch.cat(self._outcomes)[-2048:]
        self._outcomes = [rec]
        self.extras["log"].update({"/metrics/seated_at_end": rec[:, 0].mean(), "/metrics/engaged_at_end": rec[:, 1].mean(),
                                   "/metrics/ever_seated": rec[:, 2].mean(), "/metrics/peak_force": rec[:, 3].mean(),
                                   "/metrics/noise_scale": torch.tensor(self.cfg.noise_scale)})

    # ---- reset -------------------------------------------------------------------------------------

    def _reset(self, ids):
        cfg, dev, k = self.cfg, self.device, len(ids)
        e, s = cfg.errors, cfg.noise_scale
        bm = self.bm

        # Target error: the observed socket.
        t_rpy = torch.stack((sym(k, math.radians(e.target_tilt_deg) * s, dev),
                             sym(k, math.radians(e.target_tilt_deg) * s, dev),
                             sym(k, math.radians(e.target_yaw_deg) * s, dev)), -1)
        t_xyz = torch.stack((sym(k, e.target_xy * s, dev), sym(k, e.target_xy * s, dev), sym(k, e.target_z * s, dev)), -1)
        self.R_obs[ids] = rot_z(t_rpy[:, 2]) @ rot_y(t_rpy[:, 1]) @ rot_x(t_rpy[:, 0])
        self.p_obs[ids] = t_xyz
        # In-hand error: SE(2) in the fingertips' midplane (x-z), turned about the closing axis y.
        h_x, h_z = sym(k, e.in_hand_x * s, dev), sym(k, e.in_hand_z * s, dev)
        h_r = sym(k, math.radians(e.in_hand_rot_deg) * s, dev)
        self.R_E[ids] = rot_y(h_r)
        self.p_E[ids] = torch.stack((h_x, torch.zeros_like(h_x), h_z), -1)
        self.err[ids] = torch.cat((torch.stack((h_x / 0.002, h_z / 0.003, h_r / 0.087), -1),
                                   t_xyz / 0.002, t_rpy / 0.05), -1)

        # Domain.
        d = cfg.domain
        if cfg.randomize_domain:
            mu_p, mu_s = uniform(k, *d.mu_parts, dev), uniform(k, *d.mu_seat, dev)
            k_lin, k_rot = uniform(k, *d.lin_stiffness, dev), uniform(k, *d.rot_stiffness, dev)
            k_w = uniform(k, *d.washer_stiffness, dev)
            mass = uniform(k, *d.hand_mass, dev)
            self.delay[ids] = torch.randint(d.action_delay[0], d.action_delay[1] + 1, (k,), device=dev) > 0
        else:
            mass = torch.full((k,), cfg.model.hand_mass, device=dev)
            self.delay[ids] = False
            mu_p, mu_s = torch.full((k,), 0.3, device=dev), torch.full((k,), 0.35, device=dev)
            k_lin, k_rot = torch.full((k,), cfg.model.lin_ke, device=dev), torch.full((k,), cfg.model.rot_ke, device=dev)
            k_w = torch.full((k,), 5000.0, device=dev)
        self.mu[ids] = torch.stack((mu_p, mu_s), -1)
        mu = self._shape_mu[ids]
        mu[:, bm.mu_parts_shapes] = mu_p[:, None]
        mu[:, bm.mu_seat_shapes] = mu_s[:, None]
        self._shape_mu[ids] = mu
        self.k_lin[ids], self.k_rot[ids] = k_lin, k_rot
        m = cfg.model
        ke = self._dof_ke[ids]
        kd = self._dof_kd[ids]
        ke[:, 0:3], kd[:, 0:3] = k_lin[:, None], (m.lin_kd * torch.sqrt(k_lin / m.lin_ke))[:, None]
        ke[:, 3:6], kd[:, 3:6] = k_rot[:, None], (m.rot_kd * torch.sqrt(k_rot / m.rot_ke))[:, None]
        ke[:, bm.slide_q] = k_w
        self._dof_ke[ids], self._dof_kd[ids] = ke, kd
        self._dirty = True
        if d.hand_mass[0] != d.hand_mass[1] or (self.mass[ids] != mass).any():
            pb, r = self.bm.plug_body, (mass / cfg.model.hand_mass)[:, None, None]
            self.mass[ids] = mass
            bm_, bi = self._body_mass[ids], self._body_inv_mass[ids]
            bm_[:, pb], bi[:, pb] = mass, 1.0 / mass
            self._body_mass[ids], self._body_inv_mass[ids] = bm_, bi
            I, Ii = self._body_inertia[ids], self._body_inv_inertia[ids]
            I[:, pb] = self._inertia0 * r
            Ii[:, pb] = torch.linalg.inv(self._inertia0) / r
            self._body_inertia[ids], self._body_inv_inertia[ids] = I, Ii
            self._mass_dirty = True
        lo, hi = cfg.noise_level
        self.noise_lvl[ids] = uniform(k, lo, hi, dev)
        self.a_prev[ids] = 0.0

        # The start: the believed plug above the observed socket, axis on the observed axis.
        start = torch.stack((sym(k, cfg.start_xy, dev), sym(k, cfg.start_xy, dev),
                             uniform(k, *cfg.start_height, dev)), -1)
        yaw = sym(k, math.radians(cfg.start_yaw_deg), dev)
        R_b = self.R_obs[ids] @ rot_z(yaw)
        p_b = self.p_obs[ids] + (self.R_obs[ids] @ start[..., None])[..., 0]
        self.R_tgt[ids], self.p_tgt[ids] = R_b, p_b
        R_x = R_b @ self.R_E[ids]
        p_x = p_b + (R_b @ (self.G_p + self.p_E[ids])[..., None])[..., 0]
        q = self._q()
        qd = self._qd()
        q_new = torch.zeros(k, self.C, device=dev)
        q_new[:, 0:3] = p_x
        q_new[:, 3:6] = rot_to_euler_xyz(R_x)
        q_new[:, bm.slide_q] = 0.0
        collar_world = uniform(k, -math.pi, math.pi, dev)
        q_new[:, bm.turn_q] = wrap(collar_world - torch.atan2(R_x[:, 1, 0], R_x[:, 0, 0]))
        q[ids] = q_new
        qd[ids] = 0.0
        tq = self._target_q
        tq[ids, 0:6] = q_new[:, 0:6]
        tq[ids, bm.slide_q] = q_new[:, bm.slide_q] * 0 + 0.001  # the washer's preload target
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        mask[ids] = True
        self._clear_solver(mask)
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)

        self.wrench_bias[ids] = torch.cat((sym((k, 3), cfg.wrench_bias[0], dev), sym((k, 3), cfg.wrench_bias[1], dev)), -1)
        self.actions[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.first_seated[ids] = -1
        self.peak_force[ids] = 0.0
        self.ep_return[ids] = 0.0
        self._read()
        self._track_turn(ids)
        self.prev_phi[ids] = self._clock_potential()[ids]
        self.prev_jam[ids] = self._jam_potential()[ids]
        if hasattr(self, "R_xt"):
            self.R_xt[ids], self.p_xt[ids] = self.R_X[ids], self.p_X[ids]
        else:
            self.R_xt, self.p_xt = self.R_X.clone(), self.p_X.clone()
        if self._dirty:
            flags = ModelFlags.SHAPE_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES
            if getattr(self, "_mass_dirty", False):
                flags |= ModelFlags.BODY_INERTIAL_PROPERTIES
                self._mass_dirty = False
            self.solver.notify_model_changed(flags)
            self._dirty = False

    def close(self):
        pass
