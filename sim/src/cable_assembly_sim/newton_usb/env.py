"""USB-A insertion with the bench's tilted grasp, in Newton (MuJoCo-Warp, SDF mesh contact), as an RSL-RL VecEnv.

THE TASK: the 2F-85 holds the plug by its overmold, the fingers at 20 deg to the plug axis (the bench's
pick, as for the BNC), above the panel-mount receptacle. Push it in until it is SEATED: the tip at the
pocket floor, the shell past the detent springs. The plug must find the 12.5 x 5.12 mm shell (0.3 mm of
clearance a side) and its tongue cavity the 1.84 mm tongue (0.04-0.07 mm), behind a 0.7-0.8 mm lead-in.

WHAT THE POLICY OBSERVES: proprioception and the wrench only, with history. Each frame is the BELIEVED plug
pose -- the arm's forward kinematics through the nominal grasp, which is what proprioception gives --
expressed in the frame of the socket estimate the task started from (a fixed transform of the joint
readings, not a sensor), and the wrench the arm measures at the hand (noisy, biased). Plus velocities,
the last action and how far the target leads the arm. No vision during the insertion, no true pose.

ROBUSTNESS. Two estimation errors, drawn per episode, which the policy never observes:
  - IN-HAND: the plug sits in the fingers off the pose the robot believes, by an SE(2) error in the jaws'
    midplane (normal = the closing axis = plug x): a shift across the plug (y), along its axis (z), and a
    turn about the closing axis. The turn tilts the plug in its tight direction and, over the 37 mm from
    the fingertips to the tip, moves the tip ~0.65 mm a degree.
  - TARGET: the socket's pose is observed with a 6-DOF error.
A CURRICULUM scales both: it starts easy (`curriculum_start`) and raises the scale by `curriculum_step`
whenever the success rate of episodes run at the current scale reaches `promote_at` (adaptive), or on a
fixed schedule (linear). Domain randomisation, per episode: plug-on-socket friction, arm + grasp stiffness,
the detent's resistance, the hand's reflected mass. The critic sees all of it (asymmetric actor-critic).

FRAMES (all per world; the world frame IS the true socket frame), as newton_bnc/env.py:
  S_obs  the observed socket                        = true socket (+) target error
  P_bel  the plug where the robot believes it is      = tool pose (+) nominal grasp
  F      the fingertip point = P_bel . G, G = (0, 0, GRASP_Z), plug-aligned
  X      what the D6 drive moves = F . E, E the in-hand error   (the true plug is X . G^-1)
The pads are rigid with F (the hand), so in the plug body they sit at G . E^-1 . pad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import warp as wp

from .model import (DETENT_TOP_Z, GRASP_Z, PAD_HALF, PANEL_TOP_Z, SHELL_TOP_Z, TIP_DEPTH, ModelCfg, UsbModel, build,  # model first
                    make_solver, pad_offsets)
from cable_assembly_sim.newton_bnc.env import (euler_xyz_to_rot, rot_to_euler_xyz, rot_to_rotvec, rot_x, rot_y,
                                               rot_z, rotvec_to_rot, sym, uniform, wrap)

import newton  # noqa: E402
from newton import ModelFlags  # noqa: E402


@dataclass
class ErrorCfg:
    """Half-widths of the uniform per-episode estimation errors (at noise scale 1). Sized so the worst case at
    the plug's tip is ~10 mm and ~10 deg: socket 7 mm / 7 deg, in hand 3 mm / 3 deg (the socket's height,
    which costs travel rather than search, to 10 mm). The first run's full range (in hand 1-2 mm / 3 deg,
    socket 1.5 mm / 2-3 deg) is noise scale ~0.2-0.3 here."""

    in_hand_y: float = 0.003
    """Across the plug in the jaws' midplane (its thin direction) [m]."""
    in_hand_z: float = 0.003
    """Along the plug axis [m]: how far up the pads the plug sits. At +3 mm and 3 deg the pads' lowest corner
    is still 2.2 mm over the panel with the plug seated."""
    in_hand_rot_deg: float = 3.0
    """About the jaws' closing axis (plug x)."""
    target_xy: float = 0.007
    target_z: float = 0.010
    target_tilt_deg: float = 7.0
    """Roll and pitch of the receptacle's axis, each."""
    target_yaw_deg: float = 7.0


@dataclass
class DomainCfg:
    mu_parts: tuple[float, float] = (0.15, 0.45)
    """Plug shell on the receptacle and its tongue. Nominal 0.3."""
    lin_stiffness: tuple[float, float] = (5.0e3, 2.0e4)
    """Arm + grasp, at the fingertips [N/m]."""
    rot_stiffness: tuple[float, float] = (5.0, 20.0)
    """Arm + grasp about the fingertips [N m/rad]."""
    detent: tuple[float, float] = (3.0, 12.0)
    """The receptacle springs' resistance to insertion [N]: Coulomb friction along the socket axis once the
    shell is past their tops (env._set_detent)."""
    hand_mass: tuple[float, float] = (0.2, 0.35)
    action_delay: tuple[int, int] = (0, 1)
    """Policy steps between the action and the drive (the controller's latency; one step = 50 ms)."""
    pad_mu: tuple[float, float] = (0.3, 0.8)
    """The 2F-85's pads on the cable's PVC overmold (with EnvCfg.slip): rubber-like pads on PVC, dry, from
    worn/dusty (0.3) to clean (0.8)."""
    grip_force: tuple[float, float] = (18.0, 22.0)
    """The 2F-85's grip at its minimum setting (20 N nominal) [N]."""


@dataclass
class EnvCfg:
    num_envs: int = 1024
    physics_hz: float = 1000.0
    policy_hz: float = 20.0
    episode_s: float = 25.0
    model: ModelCfg = field(default_factory=ModelCfg)
    errors: ErrorCfg = field(default_factory=ErrorCfg)
    domain: DomainCfg = field(default_factory=DomainCfg)
    randomize_domain: bool = True

    # The uncertainty curriculum.
    noise_scale: float = 1.0
    """Multiplies every estimation error. The curriculum sets it; evaluations fix it."""
    curriculum: str = "off"
    """"adaptive": raise the scale when episodes at the current scale succeed often enough; "linear": ramp
    from curriculum_start to curriculum_max over curriculum_steps policy steps; "off": noise_scale fixed."""
    curriculum_start: float = 0.05
    curriculum_step: float = 0.05
    curriculum_max: float = 1.0
    promote_at: float = 0.80
    """Success rate (seated at the end) over the episodes run at the current scale that moves it up."""
    promote_window: int = 1024
    """Episodes at the current scale before the rate is judged (and between judgements)."""
    demote_at: float = 0.0
    """Below this rate the scale steps back down (0: never)."""
    curriculum_mix: float = 0.0
    """With a curriculum: this fraction of episodes draws its scale uniformly from [0, the current scale] instead
    of at it, so easier cases stay in the batch (they do not count towards promotion)."""
    promote_stall: int = 0
    """Promote anyway after this many judgements in a row at one scale without promotion, if the rate is at
    least promote_floor (0: never). Keeps a scale that cannot reach promote_at from ending the curriculum."""
    promote_floor: float = 0.6
    curriculum_steps: int = 0

    # The start: the believed plug above the observed socket.
    start_height: tuple[float, float] = (0.014, 0.018)
    """Plug origin above the observed socket: the tip 3-7 mm above the housing's face, with no error. Raised by
    start_clear x the noise scale, so a socket truly higher than observed (or a plug lower in the hand)
    never starts the plug inside it."""
    start_clear: float = 0.015
    """At scale 1: target_z + in_hand_z + 2 mm for the face's rise at the plate's rim under a 7 deg tilt."""
    start_xy: float = 0.0005
    start_yaw_deg: float = 2.0

    # The action: displacement of P_bel's target per policy step at |a| = 1, in S_obs.
    pos_scale: float = 0.001
    rot_scale_deg: float = 1.5
    max_lead: tuple[float, float, float] = (0.003, 0.007, math.radians(4.0))
    """How far the target may run ahead of the measured pose, in the observed socket's plane, along its axis,
    and in rotation. With the arm's stiffness this is the force the policy can apply (max_force caps it): the
    axial 7 mm lets the softest arm (5 kN/m) push past the stiffest detent (12 N). At 4 mm / -2 mm, 4% of
    zero-uncertainty plugs stalled in the springs."""
    pos_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.016, -0.016, -0.020), (0.016, 0.016, 0.045))
    """The believed plug's target, in the observed socket: the socket may be 7 mm off and 10 mm deeper, the
    plug 3 mm and 3 deg (2 mm at the tip) off in the hand, and a search needs room past that."""
    max_tilt_deg: float = 14.0
    """The plug axis within this of the observed socket axis: 7 deg of socket tilt + 3 in the hand + margin."""
    max_force: float = 15.0
    """Hard cap on the arm spring's force [N]. The bench's 2F-85 holds the overmold at its minimum grip
    (~20 N): with pad friction 0.3-0.8 the plug slips in the fingers at 12-32 N axial. The detent needs up to
    12 N; 15 N leaves 3 N for wall friction under a tilt. At a 12 N cap (stage 1d) a policy seated only 56% of
    11-12 N detents even with no estimation error, and 22-30% at noise scale 0.75. Earlier runs: 20, 12 N."""
    force_soft: float = 6.0
    """Above this force [N] the reward penalises it (quadratically): the detent needs up to 12 N, so seating
    may cost a little, but leaning on the plug or the face does not pay."""
    hold_force: float = 2.0
    """Once seated, force above this [N] is penalised too: nothing more to push for."""
    contact_force: float = 1.5
    """True force [N] above which the plug counts as in contact (for the contact-speed penalty)."""
    contact_speed: float = 0.005
    """Speed along the plug's axis in contact [m/s] above which the reward penalises it (quadratically):
    impacts. Sliding sideways across the face (searching) is free; it does not load the grasp along the
    pads. A 20 Hz policy at 1 mm a step moves 20 mm/s."""
    w_force: float = 1.0
    force_scale: float = 6.0
    """The force penalty is -w_force ((F - force_soft) / force_scale)^2 a step: -1 at 12 N, -2.25 at the 15 N cap."""
    w_hold: float = 0.9
    w_rate: float = 0.1
    """Smoothness: -w_rate |a_t - a_(t-1)|^2 a step (the action is the target's step in mm / 1.5 deg, so a
    reversal of 1 mm is -0.4). Penalises jitter in the commanded motion."""
    w_speed: float = 0.5
    w_align: float = 0.0
    """Once the plug's tip is in the opening: -w_align (tilt / 3 deg) a step, the tilt of the TRUE plug axis off
    the TRUE socket axis. Stage 1d's plugs stuck part-way in sat at a median 3 deg against 1.3 deg for seated
    ones. Paid only inside the opening, so it cannot pay for hovering at the estimate."""
    max_torque: float = 1.0
    """Cap on the grasp's rotational spring [N m] (a 2F-85 on a PVC overmold turns in its pads beyond it)."""
    base_push: float = 0.4
    """A base downward action (x pos_scale per step, along the observed socket axis) always added to the
    policy's: the policy learns a residual on a constant push. 0 disables it."""

    # What the robot measures.
    wrench_noise: tuple[float, float] = (0.5, 0.02)
    """Per-step Gaussian std of force [N] and torque [N m] (the iiwa's torque-based estimate)."""
    wrench_bias: tuple[float, float] = (1.0, 0.03)
    """Per-episode uniform bias half-width."""
    history: int = 4
    """Frames (believed pose + measured wrench) in the policy's observation."""

    # Outcome thresholds, on the TRUE plug in the TRUE socket.
    seated_height: float = 0.0008
    """Plug origin above seated: the tip within 0.8 mm of the floor (11.2 of 12 mm in, past the springs)."""
    engaged_height: float = 0.0070
    """The tip 2.9 mm into the shell (the tongue in its cavity)."""
    center_tol: float = 0.0008
    tilt_tol_deg: float = 3.0
    void_artifacts: bool = True
    """End (as a time-out, not a success) episodes that reach a pose real parts cannot: the plug below the
    shell's top while off the pocket's axis by more than its clearance allows (through a wall), below the
    floor, or kicked (the hand's spring stretched past launch_factor x max_force, which the force cap rules
    out unless a contact fired the plug back: the 0.33 mm walls round the tongue cavity are thin for an SDF,
    and a tongue edge driven past their middle flips its normal). ~0.5% of episodes at full error."""
    artifacts_terminal: bool = False
    """End artifact episodes as FAILURES (terminal, no bootstrap) rather than neutral time-outs, so the policy
    learns to avoid what provokes them (newton_bnc's choice)."""
    launch_factor: float = 1.6
    detent: bool = True

    # The grasp slipping (off: the plug is fixed in the hand, the in-hand error constant through an episode).
    slip: bool = False
    """The plug slips in the pads when the hand's load exceeds their friction (env._slip): the force in the
    jaws' plane (across the plug, along it) and the twist about the closing axis share an elliptic limit,
    2 x pad_mu x grip for the force and that times slip_radius for the twist. A slip changes the in-hand
    error E, which the policy does not observe (only the hand appearing to jump, and the wrench relaxing)."""
    slip_radius: float = 0.006
    """Effective radius of the pads' friction against a twist about the closing axis [m] (the patch on the
    overmold's side, ~8 x 30 mm)."""
    slip_rate: tuple[float, float] = (0.02, 0.4)
    """Fastest slide [m/s] and turn [rad/s] in the pads: the drive's target jumps at each action, and a slip
    taken at once would jump the pads (rigid with the hand) into whatever they touch."""
    slip_max: tuple[float, float] = (0.006, 0.012)
    """Bounds on the in-hand offset |y|, |z| [m]: past them the plug would leave the pads."""


class UsbNewtonEnv:
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
        if cfg.curriculum != "off":
            cfg.noise_scale = cfg.curriculum_start

        wp.init()
        self.um: UsbModel = build(replace(cfg.model, num_worlds=n), device=device)
        self.model = self.um.model
        self.pipeline, self.solver = make_solver(self.um, cfg.model)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        um, dev = self.um, self.device
        self.C, self.D = um.coords_per_world, um.dofs_per_world
        B = um.bodies_per_world

        shape_world = self.model.shape_world.numpy()
        assert np.all(shape_world == np.repeat(np.arange(n), um.shapes_per_world)), "unexpected shape layout"
        self._shape_mu = wp.to_torch(self.model.shape_material_mu).view(n, um.shapes_per_world)
        self._shape_tf = wp.to_torch(self.model.shape_transform).view(n, um.shapes_per_world, 7)
        self._dof_ke = wp.to_torch(self.model.joint_target_ke).view(n, self.D)
        self._dof_kd = wp.to_torch(self.model.joint_target_kd).view(n, self.D)
        self._target_q = wp.to_torch(self.control.joint_target_q).view(n, self.C)
        self._body_mass = wp.to_torch(self.model.body_mass).view(n, B)
        self._body_inv_mass = wp.to_torch(self.model.body_inv_mass).view(n, B)
        self._body_inertia = wp.to_torch(self.model.body_inertia).view(n, B, 3, 3)
        self._body_inv_inertia = wp.to_torch(self.model.body_inv_inertia).view(n, B, 3, 3)
        self._inertia0 = self._body_inertia[0, um.plug_body].clone()

        I3 = torch.eye(3, device=dev)
        self.G_p = torch.tensor([0.0, 0.0, GRASP_Z], device=dev)
        pads = pad_offsets()
        self.pad_R = torch.tensor(np.stack([R for R, _ in pads]), dtype=torch.float32, device=dev)
        self.pad_c = torch.tensor(np.stack([c for _, c in pads]), dtype=torch.float32, device=dev)
        # The pads' corners in the hand's frame F (plug-aligned at the fingertip point; G and the pad offsets).
        h = np.asarray(PAD_HALF)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        corners = [R @ (signs * h).T + c[:, None] for R, c in pads]
        self._pad_corners = torch.tensor(np.concatenate(corners, 1).T, dtype=torch.float32, device=dev)
        # MuJoCo-Warp's per-world Coulomb friction of the D6's linear z DOF (the detent, _set_detent).
        self._frictionloss = wp.to_torch(self.solver.mjw_model.dof_frictionloss)
        mjc_to_newton = self.solver.mjc_dof_to_newton_dof.numpy()[0]
        self._z_dof = int(np.nonzero(mjc_to_newton == 2)[0][0])
        self._dof_solref = wp.to_torch(self.solver.mjw_model.dof_solref)
        self._dof_solimp = wp.to_torch(self.solver.mjw_model.dof_solimp)
        self._detent_solref = torch.tensor([2.0 * self.sim_dt, 1.0], device=dev)
        self._detent_solimp = torch.tensor([0.99, 0.999, 0.001, 0.5, 2.0], device=dev)
        self.detent_now = torch.zeros(n, device=dev)

        self.R_obs = I3.repeat(n, 1, 1)
        self.p_obs = torch.zeros(n, 3, device=dev)
        self.R_E = I3.repeat(n, 1, 1)
        self.p_E = torch.zeros(n, 3, device=dev)
        self.R_tgt = I3.repeat(n, 1, 1)
        self.p_tgt = torch.zeros(n, 3, device=dev)
        self.k_lin = torch.full((n,), cfg.model.lin_ke, device=dev)
        self.k_rot = torch.full((n,), cfg.model.rot_ke, device=dev)
        self.mu = torch.full((n,), 0.3, device=dev)
        self.detent_f = torch.zeros(n, device=dev)
        self.mass = torch.full((n,), cfg.model.hand_mass, device=dev)
        self.err = torch.zeros(n, 9, device=dev)
        self.ep_scale = torch.zeros(n, device=dev)
        self.wrench_bias = torch.zeros(n, 6, device=dev)
        self.prev_bel = (I3.repeat(n, 1, 1), torch.zeros(n, 3, device=dev))
        self.actions = torch.zeros(n, 6, device=dev)
        self.d_actions = torch.zeros(n, 6, device=dev)
        self.hist = torch.zeros(n, cfg.history, 15, device=dev)
        self.delay = torch.zeros(n, dtype=torch.bool, device=dev)
        self.a_prev = torch.zeros(n, 6, device=dev)
        self.meas_push = torch.zeros(n, device=dev)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.first_seated = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.peak_force = torch.zeros(n, device=dev)
        self.peak_speed = torch.zeros(n, device=dev)
        self.e_rot = torch.zeros(n, device=dev)
        self.pad_mu = torch.full((n,), 0.55, device=dev)
        self.grip = torch.full((n,), 20.0, device=dev)
        self.slip_total = torch.zeros(n, device=dev)
        self.speed = torch.zeros(n, device=dev)
        self.nonfinite_count = 0
        self.overflow_steps = torch.zeros((), device=dev)
        self.artifact_counts = torch.zeros(3, device=dev)
        self.voided = torch.zeros(n, dtype=torch.bool, device=dev)
        self.step_count = 0
        self._outcomes: list[torch.Tensor] = []
        self._cur_n = 0  # episodes finished at the current curriculum scale
        self._cur_success = 0.0
        self.extras: dict = {}

        self._reset(torch.arange(n, device=dev))
        self._read()
        self._obs = self._observe(reset_ids=torch.arange(n, device=dev))

    # ---- simulation --------------------------------------------------------------------------------

    def _q(self):
        return wp.to_torch(self.s0.joint_q).view(self.num_envs, self.C)

    def _qd(self):
        return wp.to_torch(self.s0.joint_qd).view(self.num_envs, self.D)

    def _set_detent(self):
        """The receptacle springs as Coulomb friction on the D6's axial DOF (world z = the true socket axis):
        `detent_f`, ramped in over the first mm of the shell's travel past the springs' tops. MuJoCo's
        frictionloss is a constraint (implicit): it holds a plug pushed with less and lets it slide past.
        Written straight into MuJoCo-Warp's per-world dof_frictionloss, every policy step (notify would
        recompute the model's constants)."""
        tip_z = self.p_true[:, 2] - TIP_DEPTH * self.R_true[:, 2, 2]
        ramp = ((DETENT_TOP_Z - tip_z) / 0.001).clamp(0.0, 1.0)
        on = self.detent_f * ramp if self.cfg.detent else torch.zeros_like(ramp)
        self._frictionloss[:, self._z_dof] = on.clamp_min(1.0e-4)
        # MuJoCo's frictionloss is a soft constraint: at its default impedance a plug pushed with 8 N crept
        # through a 12 N detent in 30% of tries. Near-hard (Newton writes neither back: no custom attribute).
        self._dof_solref[:, self._z_dof] = self._detent_solref
        self._dof_solimp[:, self._z_dof] = self._detent_solimp
        self.detent_now = on

    def _simulate(self):
        for _ in range(self.substeps):
            self.pipeline.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.sim_dt)
            self.s0, self.s1 = self.s1, self.s0

    def _slip(self):
        """The plug slips in the pads: where the hand's load on it (the D6 spring: force in the jaws' plane,
        twist about the closing axis = plug x) lies outside the pads' elliptic friction limit, shift the
        in-hand error E so the spring sheds the excess. The drive's target X = F . E moves back by what slid;
        the hand F = X . E^-1 (what proprioception reports) moves on by it, and the pads with it."""
        q = self._q()
        R_X, p_X = euler_xyz_to_rot(q[:, 3:6]), q[:, 0:3]
        f = (R_X.mT @ (self.k_lin[:, None] * (self.p_xt - p_X))[..., None])[..., 0]  # plug frame
        t = (R_X.mT @ (self.k_rot[:, None] * rot_to_rotvec(self.R_xt @ R_X.mT))[..., None])[..., 0]
        F_s = 2.0 * self.pad_mu * self.grip
        T_s = F_s * self.cfg.slip_radius
        ft, tx = f[:, 1:3], t[:, 0]
        u = torch.sqrt((torch.linalg.vector_norm(ft, dim=-1) / F_s) ** 2 + (tx / T_s) ** 2).nan_to_num(0.0)
        ex = (1.0 - 1.0 / u.clamp_min(1.0))  # the fraction of the load past the limit
        if not bool((ex > 0).any()):
            return
        cfg = self.cfg
        dt = self.step_dt
        # The slide, in the plug frame (its y, z: the jaws' plane), rate-limited.
        d = torch.zeros_like(f)
        d[:, 1:3] = ex[:, None] * ft / self.k_lin[:, None]
        dn = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        d = d * (cfg.slip_rate[0] * dt / dn.clamp_min(1e-12)).clamp(max=1.0)
        # The target X = P_tgt . G . E comes back toward the plug by d: E's offset (hand frame F, = R_E d in it,
        # still in the jaws' plane) gives up d, and its turn about the closing axis gives up dth. E is kept as
        # (0, y, z) and rot_x(angle) -- composing rotation matrices here instead (conjugating by the target's)
        # drifted them off orthonormal, the drift fed back through the targets, and the plugs flew off.
        p_E = self.p_E - (self.R_E @ d[..., None])[..., 0]
        lim = torch.tensor([0.0, cfg.slip_max[0], cfg.slip_max[1]], device=self.device)
        p_E = torch.maximum(torch.minimum(p_E, lim), -lim)
        dth = (ex * tx / self.k_rot).clamp(-cfg.slip_rate[1] * dt, cfg.slip_rate[1] * dt)
        e_rot = self.e_rot - dth
        R_E = rot_x(e_rot)
        # The hand cannot slide on into the adapter or the panel: a slip that would take the pads (rigid
        # with the hand) below the surface under them is refused (their contact carries the load instead).
        blocked = self._pads_below(R_X, p_X, R_E, p_E) & ~self._pads_below(R_X, p_X, self.R_E, self.p_E)
        moved = (ex > 0) & ~blocked
        p_E = torch.where(moved[:, None], p_E, self.p_E)
        e_rot = torch.where(moved, e_rot, self.e_rot)
        dth = torch.where(moved, dth, 0.0)
        self.slip_total += torch.linalg.vector_norm(p_E - self.p_E, dim=-1) + GRASP_Z * dth.abs()
        self.p_E, self.e_rot, self.R_E = p_E, e_rot, rot_x(e_rot)
        self._drive()
        self._place_pads(torch.arange(self.num_envs, device=self.device))
        self.err[:, 0:3] = torch.stack((self.p_E[:, 1] / 0.003, self.p_E[:, 2] / 0.003, self.e_rot / 0.052), -1)

    def _guard(self):
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
        m = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        m[:-1] = mask
        self.solver.reset(self.s0, world_mask=wp.from_torch(m), flags=0)

    def _read(self):
        q = self._q()
        self.R_X = euler_xyz_to_rot(q[:, 3:6])
        self.p_X = q[:, 0:3].clone()
        R_F = self.R_X @ self.R_E.mT
        p_F = self.p_X - (R_F @ self.p_E[..., None])[..., 0]
        self.R_bel, self.p_bel = R_F, p_F - (R_F @ self.G_p)
        self.R_true, self.p_true = self.R_X, self.p_X - (self.R_X @ self.G_p)

    def _drive(self):
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
        self.step_count += 1
        if cfg.curriculum == "linear" and cfg.curriculum_steps > 0:
            f = min(1.0, self.step_count / cfg.curriculum_steps)
            cfg.noise_scale = cfg.curriculum_start + (cfg.curriculum_max - cfg.curriculum_start) * f
        a = actions.clamp(-1.0, 1.0)
        self.d_actions = a - self.actions  # this step's change of the policy's action (for the smoothness penalty)
        self.actions = a
        if cfg.base_push:
            a = a.clone()
            a[:, 2] = (a[:, 2] - cfg.base_push).clamp(-1.0, 1.0)
        if cfg.domain.action_delay[1] > 0:
            a, self.a_prev = torch.where(self.delay[:, None], self.a_prev, a), a

        self._apply_action(a)
        self._set_detent()
        self._simulate()
        self.overflow_steps += (wp.to_torch(self.contacts.rigid_contact_count)[0] > self.contacts.rigid_contact_max).float()
        bad = self._guard()
        if cfg.slip:
            # Once a policy step, on the load the step ended with (written mid-step, the drive targets and the
            # pads' poses raced the physics kernels: 98% of seated plugs spun up).
            self._slip()
        self._read()
        self.episode_length_buf += 1

        reward, terms, seated, engaged, force = self._reward(a)
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
        just = seated & (self.first_seated < 0)
        self.first_seated[just] = self.episode_length_buf[just]
        self.peak_force = torch.maximum(self.peak_force, force)
        contact_speed = torch.where(bad | (force < cfg.contact_force), 0.0, self.speed)
        self.peak_speed = torch.maximum(self.peak_speed, contact_speed)
        self.first_seated[bad] = -1

        time_out = ((self.episode_length_buf >= self.max_episode_length) | bad) & ~terminal
        dones = time_out | terminal
        self.extras = {"time_outs": time_out, "log": {f"/reward/{k}": v.mean() for k, v in terms.items()}}
        self.extras["log"]["/metrics/nonfinite_total"] = torch.tensor(float(self.nonfinite_count))
        self.extras["log"]["/metrics/contact_overflow_steps"] = self.overflow_steps
        self.extras["log"]["/metrics/artifact_through_wall"] = self.artifact_counts[0]
        self.extras["log"]["/metrics/artifact_through_floor"] = self.artifact_counts[1]
        self.extras["log"]["/metrics/artifact_kicked"] = self.artifact_counts[2]
        self.extras["log"]["/metrics/noise_scale"] = torch.tensor(float(cfg.noise_scale))
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids):
            self._record_outcomes(done_ids, seated, engaged, force)
            self._reset(done_ids)
            self._read()
        self._obs = self._observe(reset_ids=done_ids if len(done_ids) else None)
        return self.get_observations(), reward, dones, self.extras

    def _apply_action(self, a):
        cfg = self.cfg
        d_pos = (self.R_obs @ (a[:, 0:3] * cfg.pos_scale)[..., None])[..., 0]
        d_rot = (self.R_obs @ (a[:, 3:6] * math.radians(cfg.rot_scale_deg))[..., None])[..., 0]
        lead_xy, lead_z, lead_r = cfg.max_lead
        lead = (self.R_obs.mT @ (self.p_tgt + d_pos - self.p_bel)[..., None])[..., 0]
        lim = torch.tensor([lead_xy, lead_xy, lead_z], device=self.device)
        lead = torch.maximum(torch.minimum(lead, lim), -lim)
        p = self.p_bel + (self.R_obs @ lead[..., None])[..., 0]
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
        if math.isfinite(cfg.max_torque):
            R_xt = self.R_tgt @ self.R_E
            st = rot_to_rotvec(R_xt @ self.R_X.mT)
            ang = torch.linalg.vector_norm(st, dim=-1, keepdim=True)
            scale = ((cfg.max_torque / self.k_rot)[:, None] / ang.clamp_min(1e-9)).clamp(max=1.0)
            self.R_tgt = rotvec_to_rot(st * scale) @ self.R_X @ self.R_E.mT
        if math.isfinite(cfg.max_force):
            # The arm spring's stretch at this target, capped at max_force / K. Isotropic: under a capped push the
            # hand gives laterally too, which is what lets the plug centre itself on the lead-in (capping the axial
            # push first and holding the lateral target seated 19% at full error with the base push, against 48%).
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
        noise = torch.randn(n, 6, device=dev) * torch.tensor([cfg.wrench_noise[0]] * 3 + [cfg.wrench_noise[1]] * 3, device=dev)
        meas = torch.cat((f_obs, t_obs), -1) + noise + self.wrench_bias
        self.meas_push = -meas[:, 2]
        frame = torch.cat((p_rel / 0.01, rot6, meas[:, 0:3] / 10.0, meas[:, 3:6] / 0.2), -1)
        self.hist = torch.cat((self.hist[:, 1:], frame[:, None]), 1)
        if reset_ids is not None:
            self.hist[reset_ids] = frame[reset_ids, None, :].expand(-1, cfg.history, -1)
        Rp, pp = self.prev_bel
        vel = (Ro @ ((self.p_bel - pp) / self.step_dt)[..., None])[..., 0] / 0.02
        ang = (Ro @ rot_to_rotvec(self.R_bel @ Rp.mT)[..., None])[..., 0] / self.step_dt
        if reset_ids is not None:
            vel[reset_ids] = 0.0
            ang[reset_ids] = 0.0
        self.prev_bel = (self.R_bel.clone(), self.p_bel.clone())
        lead_p = (Ro @ (self.p_tgt - self.p_bel)[..., None])[..., 0] / torch.tensor(
            [cfg.max_lead[0], cfg.max_lead[0], cfg.max_lead[1]], device=dev)
        lead_r = (Ro @ rot_to_rotvec(self.R_tgt @ self.R_bel.mT)[..., None])[..., 0] / cfg.max_lead[2]
        policy = torch.cat((self.hist.flatten(1), vel, ang, self.actions, lead_p, lead_r), -1)

        # Privileged (critic only): the true plug in the true socket, the errors, the domain, the true wrench.
        rot6_t = torch.cat((self.R_true[:, :, 0], self.R_true[:, :, 1]), -1)
        dom = torch.stack((self.mu, self.k_lin / 1e4, self.k_rot / 10.0, self.detent_f / 10.0, self.mass / 0.25,
                           self.delay.float(), self.ep_scale), -1)
        critic = torch.cat((policy, self.p_true / 0.01, rot6_t, self.err, dom, w[:, 0:3] / 10.0, w[:, 3:6] / 0.2), -1)
        if cfg.slip:  # the grasp's friction limit (force, twist), and how far the plug has slid
            F_s = 2.0 * self.pad_mu * self.grip
            critic = torch.cat((critic, torch.stack((F_s / 20.0, F_s * cfg.slip_radius / 0.08,
                                                     self.slip_total / 0.003), -1)), -1)
        return {"policy": torch.nan_to_num(policy, 0.0, 0.0, 0.0), "critic": torch.nan_to_num(critic, 0.0, 0.0, 0.0)}

    # ---- outcome and reward ------------------------------------------------------------------------

    def plug_state(self):
        cfg = self.cfg
        h = self.p_true[:, 2]
        r = torch.linalg.vector_norm(self.p_true[:, 0:2], dim=-1)
        tilt = torch.acos(self.R_true[:, 2, 2].clamp(-1.0, 1.0))
        centered = (r < cfg.center_tol) & (tilt < math.radians(cfg.tilt_tol_deg))
        return h, r, tilt, centered & (h < cfg.seated_height), centered & (h < cfg.engaged_height)

    def _artifacts(self):
        """(n, 3) bool: the plug through the pocket's wall (its tip below the shell's top, its axis off by more
        than the shell's clearance and a tilt's allow), through the floor, kicked back by a contact."""
        h, r, _, _, _ = self.plug_state()
        tip = self.p_true - self.R_true[:, :, 2] * TIP_DEPTH
        tip_r = torch.linalg.vector_norm(tip[:, 0:2], dim=-1)
        through_wall = (tip[:, 2] < 0.0045) & (tip_r > 0.0012)
        through_floor = h < -0.0005
        kicked = torch.linalg.vector_norm(self._wrench()[:, 0:3], dim=-1) > self.cfg.launch_factor * self.cfg.max_force
        return torch.stack((through_wall, through_floor, kicked), -1)

    def _reward(self, a):
        cfg = self.cfg
        h, r, tilt, seated, engaged = self.plug_state()
        offs = torch.tensor([-TIP_DEPTH, 0.010, 0.025], device=self.device)
        kp = self.p_true[:, None, :] + self.R_true[:, None, :, 2] * offs[None, :, None]
        kp_goal = torch.zeros_like(kp)
        kp_goal[..., 2] = offs
        # Height only: the actor cannot see where the socket truly is, so paying for lateral closeness to it pays
        # for hovering at the estimate (its expected position) rather than searching. The keypoints' heights
        # above their seated heights only fall once the plug is in the opening.
        d = (kp[..., 2] - kp_goal[..., 2]).abs().mean(-1)

        def squash(x, a_, b_):
            return 1.0 / (torch.exp(a_ * x) + b_ + torch.exp(-a_ * x))

        force = torch.linalg.vector_norm(self._wrench()[:, 0:3], dim=-1)
        # The hand's speed along the plug's axis (the D6's linear rates: the fingertip point, world frame).
        self.speed = (self._qd()[:, 0:3] * self.R_true[:, :, 2]).sum(-1).abs().nan_to_num(0.0)
        contact = force > cfg.contact_force
        terms = {
            "kp_coarse": squash(d, 50.0, 2.0),
            "kp_fine": squash(d, 300.0, 0.0),
            "engaged": engaged.float(),
            "seated": 3.0 * seated.float(),
            # Low force (the plug slips in a 20 N grip) and gentle contact.
            "force": -cfg.w_force * ((force - cfg.force_soft).clamp_min(0.0) / cfg.force_scale) ** 2,
            "hold": -cfg.w_hold * seated.float() * (force - cfg.hold_force).clamp_min(0.0) / 10.0,
            "speed": -cfg.w_speed * contact.float() * ((self.speed - cfg.contact_speed).clamp_min(0.0) / 0.01) ** 2,
            "action": -0.01 * (a**2).sum(-1),
            "smooth": -cfg.w_rate * (self.d_actions**2).sum(-1),
        }
        if cfg.w_align > 0:
            tip = self.p_true - self.R_true[:, :, 2] * TIP_DEPTH
            in_open = (tip[:, 2] < SHELL_TOP_Z) & (torch.linalg.vector_norm(tip[:, 0:2], dim=-1) < 0.0015)
            terms["align"] = -cfg.w_align * in_open.float() * torch.rad2deg(tilt).nan_to_num(0.0) / 3.0
        return sum(terms.values()), terms, seated, engaged, force

    def _record_outcomes(self, ids, seated, engaged, force):
        cfg = self.cfg
        rec_new = torch.stack((seated[ids].float(), engaged[ids].float(), (self.first_seated[ids] >= 0).float(),
                               self.peak_force[ids], self.ep_scale[ids], force[ids], self.peak_speed[ids],
                               self.slip_total[ids]), -1)
        self._outcomes.append(rec_new)
        rec = torch.cat(self._outcomes)[-2048:]
        self._outcomes = [rec]
        self.extras["log"].update({"/metrics/seated_at_end": rec[:, 0].mean(), "/metrics/engaged_at_end": rec[:, 1].mean(),
                                   "/metrics/ever_seated": rec[:, 2].mean(), "/metrics/peak_force": rec[:, 3].mean(),
                                   "/metrics/end_force": rec[:, 5].mean(),
                                   "/metrics/peak_contact_speed_mm_s": 1e3 * rec[:, 6].mean()})
        if cfg.slip:
            self.extras["log"].update({"/metrics/slip_mm": 1e3 * rec[:, 7].mean(),
                                       "/metrics/slipped_1mm": (rec[:, 7] > 0.001).float().mean()})
        if cfg.curriculum == "adaptive":
            at = (rec_new[:, 4] - cfg.noise_scale).abs() < 1e-6  # run at the current scale
            self._cur_n += int(at.sum())
            self._cur_success += float(rec_new[at, 0].sum())
            if self._cur_n >= cfg.promote_window:
                rate = self._cur_success / self._cur_n
                old = cfg.noise_scale
                self._stalled = getattr(self, "_stalled", 0) + 1
                stalled = cfg.promote_stall > 0 and self._stalled >= cfg.promote_stall and rate >= cfg.promote_floor
                if (rate >= cfg.promote_at or stalled) and old < cfg.curriculum_max:
                    cfg.noise_scale = min(cfg.curriculum_max, round(old + cfg.curriculum_step, 6))
                    if stalled:
                        print(f"[curriculum] stalled {self._stalled} windows at {old:.2f}: promoting anyway", flush=True)
                elif cfg.demote_at > 0 and rate < cfg.demote_at:
                    cfg.noise_scale = max(cfg.curriculum_start, round(old - cfg.curriculum_step, 6))
                if cfg.noise_scale != old:
                    self._stalled = 0
                    print(f"[curriculum] step {self.step_count}: success {rate:.3f} at scale {old:.2f} "
                          f"-> scale {cfg.noise_scale:.2f}", flush=True)
                self.extras["log"]["/metrics/curriculum_rate"] = torch.tensor(rate)
                self._cur_n, self._cur_success = 0, 0.0

    # ---- reset -------------------------------------------------------------------------------------

    def _reset(self, ids):
        cfg, dev, k = self.cfg, self.device, len(ids)
        e = cfg.errors
        s = torch.full((k,), float(cfg.noise_scale), device=dev)
        if cfg.curriculum != "off" and cfg.curriculum_mix > 0:
            mix = torch.rand(k, device=dev) < cfg.curriculum_mix
            s[mix] = torch.rand(int(mix.sum()), device=dev) * cfg.noise_scale
        um = self.um
        self.ep_scale[ids] = s

        # Target error: the observed socket.
        t_rpy = torch.stack((sym(k, math.radians(e.target_tilt_deg) * s, dev), sym(k, math.radians(e.target_tilt_deg) * s, dev),
                             sym(k, math.radians(e.target_yaw_deg) * s, dev)), -1)
        t_xyz = torch.stack((sym(k, e.target_xy * s, dev), sym(k, e.target_xy * s, dev), sym(k, e.target_z * s, dev)), -1)
        self.R_obs[ids] = rot_z(t_rpy[:, 2]) @ rot_y(t_rpy[:, 1]) @ rot_x(t_rpy[:, 0])
        self.p_obs[ids] = t_xyz
        # In-hand error: SE(2) in the jaws' midplane (plug y-z), turned about the closing axis (plug x).
        h_y, h_z = sym(k, e.in_hand_y * s, dev), sym(k, e.in_hand_z * s, dev)
        h_r = sym(k, math.radians(e.in_hand_rot_deg) * s, dev)
        self.R_E[ids] = rot_x(h_r)
        self.e_rot[ids] = h_r
        self.p_E[ids] = torch.stack((torch.zeros_like(h_y), h_y, h_z), -1)
        self.err[ids] = torch.cat((torch.stack((h_y / 0.003, h_z / 0.003, h_r / 0.052), -1),
                                   t_xyz / 0.007, t_rpy / 0.12), -1)

        # Domain.
        d = cfg.domain
        if cfg.randomize_domain:
            mu = uniform(k, *d.mu_parts, dev)
            k_lin, k_rot = uniform(k, *d.lin_stiffness, dev), uniform(k, *d.rot_stiffness, dev)
            det = uniform(k, *d.detent, dev)
            mass = uniform(k, *d.hand_mass, dev)
            self.delay[ids] = torch.randint(d.action_delay[0], d.action_delay[1] + 1, (k,), device=dev) > 0
            self.pad_mu[ids], self.grip[ids] = uniform(k, *d.pad_mu, dev), uniform(k, *d.grip_force, dev)
        else:
            mu = torch.full((k,), 0.3, device=dev)
            k_lin, k_rot = torch.full((k,), cfg.model.lin_ke, device=dev), torch.full((k,), cfg.model.rot_ke, device=dev)
            det = torch.full((k,), 0.5 * sum(d.detent), device=dev)
            mass = torch.full((k,), cfg.model.hand_mass, device=dev)
            self.delay[ids] = False
            self.pad_mu[ids], self.grip[ids] = 0.5 * sum(d.pad_mu), 0.5 * sum(d.grip_force)
        self.mu[ids], self.detent_f[ids] = mu, det
        sm = self._shape_mu[ids]
        sm[:, um.plug_shapes + um.socket_shapes] = mu[:, None]  # MuJoCo takes the larger of the pair's
        self._shape_mu[ids] = sm
        self.k_lin[ids], self.k_rot[ids] = k_lin, k_rot
        m = cfg.model
        ke, kd = self._dof_ke[ids], self._dof_kd[ids]
        ke[:, 0:3], kd[:, 0:3] = k_lin[:, None], (m.lin_kd * torch.sqrt(k_lin / m.lin_ke))[:, None]
        ke[:, 3:6], kd[:, 3:6] = k_rot[:, None], (m.rot_kd * torch.sqrt(k_rot / m.rot_ke))[:, None]
        self._dof_ke[ids], self._dof_kd[ids] = ke, kd
        pb, r = um.plug_body, (mass / cfg.model.hand_mass)[:, None, None]
        self.mass[ids] = mass
        bm_, bi = self._body_mass[ids], self._body_inv_mass[ids]
        bm_[:, pb], bi[:, pb] = mass, 1.0 / mass
        self._body_mass[ids], self._body_inv_mass[ids] = bm_, bi
        I, Ii = self._body_inertia[ids], self._body_inv_inertia[ids]
        I[:, pb] = self._inertia0 * r
        Ii[:, pb] = torch.linalg.inv(self._inertia0) / r
        self._body_inertia[ids], self._body_inv_inertia[ids] = I, Ii
        self.a_prev[ids] = 0.0

        self._place_pads(ids)

        # The start: the believed plug above the observed socket, axis on the observed axis.
        start = torch.stack((sym(k, cfg.start_xy, dev), sym(k, cfg.start_xy, dev),
                             uniform(k, *cfg.start_height, dev) + cfg.start_clear * s), -1)
        yaw = sym(k, math.radians(cfg.start_yaw_deg), dev)
        R_b = self.R_obs[ids] @ rot_z(yaw)
        p_b = self.p_obs[ids] + (self.R_obs[ids] @ start[..., None])[..., 0]
        self.R_tgt[ids], self.p_tgt[ids] = R_b, p_b
        R_x = R_b @ self.R_E[ids]
        p_x = p_b + (R_b @ (self.G_p + self.p_E[ids])[..., None])[..., 0]
        q, qd = self._q(), self._qd()
        q_new = torch.zeros(k, self.C, device=dev)
        q_new[:, 0:3] = p_x
        q_new[:, 3:6] = rot_to_euler_xyz(R_x)
        q[ids] = q_new
        qd[ids] = 0.0
        self._target_q[ids, 0:6] = q_new[:, 0:6]
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        mask[ids] = True
        self._clear_solver(mask)
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)

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
        self.solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES
                                         | ModelFlags.BODY_INERTIAL_PROPERTIES)

    def _pads_below(self, R_X, p_X, R_E, p_E, margin=0.0002):
        """Whether a pad corner of the hand F = X . E^-1 is within `margin` of (or under) the adapter's top
        (r < 11.7 mm) or the panel's face, in the true socket frame."""
        if not self.um.pad_shapes:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        R_F = R_X @ R_E.mT
        p_F = p_X - (R_F @ p_E[..., None])[..., 0]
        z = (R_F @ self._pad_corners.T)[..., :].transpose(1, 2) + p_F[:, None]  # (n, corners, 3)
        r = torch.linalg.vector_norm(z[..., 0:2], dim=-1)
        floor = torch.where(r < 0.0117, 0.00665, PANEL_TOP_Z)
        return ((z[..., 2] - floor) < margin).any(-1)

    def _place_pads(self, ids):
        """The pads ride with the hand: in the plug body at G . E^-1 . pad."""
        if not self.um.pad_shapes:
            return
        # Only the pads' entries: writing back whole rows of the shape transforms mid-episode (as a gather and a
        # scatter) set seated plugs spinning in 95% of worlds, though the values were the same.
        RE_t = self.R_E[ids].mT
        base = self.G_p - (RE_t @ self.p_E[ids][..., None])[..., 0]
        for j, sh in enumerate(self.um.pad_shapes):
            self._shape_tf[ids, sh, 0:3] = base + (RE_t @ self.pad_c[j][:, None])[..., 0]
            self._shape_tf[ids, sh, 3:7] = rot_to_quat(RE_t @ self.pad_R[j])

    def snapshot(self, ids) -> dict:
        """What render.py draws, for worlds `ids` (numpy, SI): the true plug, the pads, the outcome."""
        h, r, tilt, seated, engaged = self.plug_state()
        R_pad, p_pad = [], []
        for sh in self.um.pad_shapes:
            tf = self._shape_tf[ids, sh]
            Rb = quat_to_rot(tf[:, 3:7])
            R_pad.append(self.R_true[ids] @ Rb)
            p_pad.append(self.p_true[ids] + (self.R_true[ids] @ tf[:, 0:3, None])[..., 0])
        out = {
            "p_true": self.p_true[ids], "R_true": self.R_true[ids], "R_pad": torch.stack(R_pad, 1), "p_pad": torch.stack(p_pad, 1),
            "h": h[ids], "seated": seated[ids], "engaged": engaged[ids], "voided": self.voided[ids],
            "force": torch.linalg.vector_norm(self._wrench()[ids, 0:3], dim=-1), "push": self.meas_push[ids],
            "detent": self.detent_now[ids], "err": self.err[ids], "scale": self.ep_scale[ids], "speed": self.speed[ids],
            "slip": self.slip_total[ids],
        }
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def close(self):
        pass


def quat_to_rot(q):
    """xyzw -> rotation matrix (batched)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).view(*q.shape[:-1], 3, 3)


def rot_to_quat(R):
    """Rotation matrix -> xyzw (batched; Shepperd's method)."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cands = torch.stack((tr, m00, m11, m22), -1).argmax(-1)
    out = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)
    s0 = torch.sqrt((1 + tr).clamp_min(1e-12)) * 2
    q0 = torch.stack(((R[..., 2, 1] - R[..., 1, 2]) / s0, (R[..., 0, 2] - R[..., 2, 0]) / s0,
                      (R[..., 1, 0] - R[..., 0, 1]) / s0, 0.25 * s0), -1)
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) * 2
    q1 = torch.stack((0.25 * s1, (R[..., 0, 1] + R[..., 1, 0]) / s1, (R[..., 0, 2] + R[..., 2, 0]) / s1,
                      (R[..., 2, 1] - R[..., 1, 2]) / s1), -1)
    s2 = torch.sqrt((1 + m11 - m00 - m22).clamp_min(1e-12)) * 2
    q2 = torch.stack(((R[..., 0, 1] + R[..., 1, 0]) / s2, 0.25 * s2, (R[..., 1, 2] + R[..., 2, 1]) / s2,
                      (R[..., 0, 2] - R[..., 2, 0]) / s2), -1)
    s3 = torch.sqrt((1 + m22 - m00 - m11).clamp_min(1e-12)) * 2
    q3 = torch.stack(((R[..., 0, 2] + R[..., 2, 0]) / s3, (R[..., 1, 2] + R[..., 2, 1]) / s3, 0.25 * s3,
                      (R[..., 1, 0] - R[..., 0, 1]) / s3), -1)
    for i, qi in enumerate((q0, q1, q2, q3)):
        out = torch.where((cands == i)[..., None], qi, out)
    return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True)
