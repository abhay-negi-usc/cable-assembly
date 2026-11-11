"""BNC INSERTION BY THE TRAINED POLICY -- from the stand-off to the plug seated (the bayonet's pins at
the slot floor), robust to the in-hand and socket pose errors (assembly.insertion_mode: rl).

THE POLICY was trained in simulation (sim/src/cable_assembly_sim/newton_bnc/env.py, Newton/MuJoCo-Warp,
the bench plug and socket's CAD): a GRU actor that sees only TASK-SPACE quantities -- the believed plug
in the observed socket, the wrench the hand applies, their history, its own last action and how far its
target leads the plug -- and moves the plug's target by up to 1 mm / 3 deg a step at 20 Hz. Exported by
sim/scripts/newton_bnc_export.py to plain arrays (an .npz: the weights, the observation normaliser and
every constant of the env it was trained in), run here in numpy: no torch, no rsl_rl on the robot PC.

FRAMES. The sim's are the CAD's, so they map exactly (perception/pose6d.py):
  sim plug    the plug's CAD frame (the 6D model's, cord_end_pos_z) turned to look the other way: z up
              the cable, origin on the axis SEAT_Z_M from the CAD's -- where it sits when seated
  sim socket  that plug frame, seated: the target connector frame times the same transform
  S_obs       the socket the robot believes in (the target, recorded or measured)
  P_bel       the plug where the robot believes it is: tool0 (measured joints) . tool0->connector belief
  G           the sim's compliance centre, 30 mm up the plug from its origin (GRASP_Z): the arm's
              admittance runs about it, and the wrench is referred to it, as the sim's spring was
The sim's socket is round but for the pins, which only the collar meets -- and the collar turns freely
-- so the turn of S_obs about its axis is a convention: it is set to the believed plug's at the start.

THE CONTROLLER. The sim's arm was a spring (1e4 N/m, 10 N m/rad at G) pulling the plug toward the
policy's target; here the same spring is the software admittance (robot/admittance.py, about G), whose
reference IS the policy's target. The target is clamped as in training: no more than max_lead ahead of
the plug, inside pos_bounds and max_tilt of the socket axis, and the spring's pull under max_force /
max_torque. The sim's base controller is part of the policy's action (base_push down, base_yaw turning
while the measured push exceeds 5 N): the policy is a residual on it.

THE WRIST'S TURN BUDGET (policies trained with env turn_band_deg, e.g. (0, 230); older ones have none):
the believed plug's turn about S_obs's axis since the start, unwrapped (+ = the base twist's first
direction), is held inside the band -- the target's turn is clamped at its edges, as in training, so
the wrist never winds past it -- and the base twist reverses within turn_margin_deg of an edge, so the
plug rocks. The policy observes the turn and the twist's direction (two inputs after the rest).

SEATED, WITHOUT GROUND TRUTH: the believed height cannot tell seated (1.1 mm) from jammed beside the
barrel (~6 mm) with +-4.5 mm of pose error, but the ADVANCE after first touch can -- seated_advance_mm
below where the push first exceeded touch_n, held hold_s (calibrated on the sim: sim/scripts/
newton_bnc_deploy_check.py).
"""

import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .. import log as urlog

log = urlog.get('rl-insert')


# ---------------------------------------------------------------------------- rotation helpers
def rotvec(R):
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_rotvec()


def from_rotvec(v):
    return Rotation.from_rotvec(np.asarray(v, dtype=float)).as_matrix()


def _elu(x):
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


# ---------------------------------------------------------------------------- the network
class Policy:
    """The exported actor: normaliser -> GRU -> MLP (elu) -> the mean action (deterministic)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.spec = json.loads(str(z['spec']))
        self.path = path
        self.mean, self.std = z['obs_mean'].astype(np.float64), z['obs_std'].astype(np.float64)
        self.eps = float(self.spec.get('obs_eps', 0.01))
        self.w_ih, self.w_hh = z['gru_w_ih'].astype(np.float64), z['gru_w_hh'].astype(np.float64)
        self.b_ih, self.b_hh = z['gru_b_ih'].astype(np.float64), z['gru_b_hh'].astype(np.float64)
        n = len(self.spec['mlp'])
        self.layers = [(z[f'mlp_w{i}'].astype(np.float64), z[f'mlp_b{i}'].astype(np.float64)) for i in range(n)]
        self.h = np.zeros(self.w_hh.shape[1])

    @property
    def obs_dim(self):
        return self.mean.shape[0]

    def reset(self):
        self.h = np.zeros_like(self.h)

    def __call__(self, obs):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / (self.std + self.eps)
        H = self.h.shape[0]
        gi, gh = self.w_ih @ x + self.b_ih, self.w_hh @ self.h + self.b_hh
        r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
        u = 1.0 / (1.0 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
        c = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1.0 - u) * c + u * self.h
        y = self.h
        for i, (w, b) in enumerate(self.layers):
            y = w @ y + b
            if i < len(self.layers) - 1:
                y = _elu(y)
        return y


# ---------------------------------------------------------------------------- the env, one world
class TaskSpace:
    """The sim env's observation and action, for one arm (env.py _observe 'policy' and _apply_action),
    in any frame (base here, the true socket's there): S_obs, P_bel and the targets are 4x4 poses in
    it. `k_lin` / `k_rot`: the spring the force and torque caps assume (the sim's nominal)."""

    def __init__(self, spec):
        e = spec['env']
        self.e = e
        self.G = np.array([0.0, 0.0, float(spec['grasp_z'])])
        self.k_lin, self.k_rot = float(e['model']['lin_ke']), float(e['model']['rot_ke'])
        self.step_dt = 1.0 / float(e['policy_hz'])
        self.hist_n = int(e['history'])
        band = e.get('turn_band_deg')
        self.band = None if band is None else (np.radians(float(band[0])), np.radians(float(band[1])))
        self.margin = np.radians(float(e.get('turn_margin_deg', 20.0)))
        self.S = None

    def _yaw(self, R):
        """The yaw of R's x axis about S_obs's axis (env.py _yaw_in_obs)."""
        x = self.S[:3, :3].T @ R[:3, 0]
        return float(np.arctan2(x[1], x[0]))

    def reset(self, S_obs, P_bel):
        self.S = np.asarray(S_obs, dtype=float)
        self.P_tgt = np.asarray(P_bel, dtype=float).copy()
        self.P_bel = self.P_tgt.copy()
        self.prev = None
        self.actions = np.zeros(6)
        self.hist = None
        self.meas_push = 0.0
        self.turn, self.yaw_dir = 0.0, 1.0
        self.yaw_prev = self._yaw(self.P_bel[:3, :3])

    # ---- observe ---------------------------------------------------------------------------------
    def observe(self, P_bel, wrench_S, dt=None):
        """The 'policy' observation from the believed plug `P_bel` and the wrench the hand applies to
        the plug at G, `wrench_S` (force, torque) in S_obs's axes. `dt`: since the last observation
        (None: the policy's step)."""
        e = self.e
        P_bel = np.asarray(P_bel, dtype=float)
        self.P_bel = P_bel
        yaw = self._yaw(P_bel[:3, :3])
        self.turn += float(np.angle(np.exp(1j * (yaw - self.yaw_prev))))
        self.yaw_prev = yaw
        Ro = self.S[:3, :3].T
        p_rel = Ro @ (P_bel[:3, 3] - self.S[:3, 3])
        R_rel = Ro @ P_bel[:3, :3]
        w = np.asarray(wrench_S, dtype=float)
        self.meas_push = -float(w[2])
        frame = np.concatenate([p_rel / 0.01, R_rel[:, 0], R_rel[:, 1], w[:3] / 10.0, w[3:] / 0.2])
        if self.hist is None:
            self.hist = np.tile(frame, (self.hist_n, 1))
            vel, ang = np.zeros(3), np.zeros(3)
        else:
            self.hist = np.vstack([self.hist[1:], frame])
            dt = self.step_dt if dt is None else float(dt)
            vel = Ro @ ((P_bel[:3, 3] - self.prev[:3, 3]) / dt) / 0.02
            ang = Ro @ rotvec(P_bel[:3, :3] @ self.prev[:3, :3].T) / dt
        self.prev = P_bel.copy()
        lead_p = Ro @ (self.P_tgt[:3, 3] - P_bel[:3, 3]) / e['max_lead'][0]
        lead_r = Ro @ rotvec(self.P_tgt[:3, :3] @ P_bel[:3, :3].T) / e['max_lead'][1]
        self.p_rel = p_rel
        obs = [self.hist.ravel(), vel, ang, self.actions, lead_p, lead_r]
        if self.band is not None:
            lo, hi = self.band
            obs.append([(2.0 * self.turn - lo - hi) / (hi - lo), self.yaw_dir])
        return np.concatenate(obs)

    # ---- act -------------------------------------------------------------------------------------
    def apply(self, action):
        """The policy's raw action -> P_bel's new target (4x4), clamped as in training. The base
        controller (base_push, base_yaw) is added here, as the env did."""
        e = self.e
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.actions = a.copy()
        if self.band is not None:
            lo, hi = self.band
            self.yaw_dir = -1.0 if self.turn > hi - self.margin else (1.0 if self.turn < lo + self.margin else self.yaw_dir)
        if e.get('base_yaw') or e.get('base_push'):
            a = a.copy()
            pushing = 1.0 if self.meas_push > float(e.get('base_yaw_push', 5.0)) else 0.0
            a[5] = np.clip(a[5] + float(e.get('base_yaw', 0.0)) * pushing * self.yaw_dir, -1.0, 1.0)
            a[2] = np.clip(a[2] - float(e.get('base_push', 0.0)), -1.0, 1.0)
        R_o, p_o = self.S[:3, :3], self.S[:3, 3]
        R_bel, p_bel = self.P_bel[:3, :3], self.P_bel[:3, 3]
        R_tgt, p_tgt = self.P_tgt[:3, :3], self.P_tgt[:3, 3]
        d_pos = R_o @ (a[:3] * float(e['pos_scale']))
        d_rot = R_o @ (a[3:] * np.radians(float(e['rot_scale_deg'])))
        lead_p, lead_r = e['max_lead']
        # (the sim clamps the lead per axis of ITS world, the TRUE socket; S_obs's axes are the stand-in)
        lead = R_o.T @ (p_tgt + d_pos - p_bel)
        p = p_bel + R_o @ np.clip(lead, -lead_p, lead_p)
        rel = np.clip(R_o.T @ (p - p_o), e['pos_bounds'][0], e['pos_bounds'][1])
        p_tgt = p_o + R_o @ rel

        R = from_rotvec(d_rot) @ R_tgt
        lr = rotvec(R @ R_bel.T)
        ang = np.linalg.norm(lr)
        R = from_rotvec(lr * min(1.0, lead_r / max(ang, 1e-9))) @ R_bel
        R_rel = R_o.T @ R
        z = R_rel[:, 2]
        tilt = np.arccos(np.clip(z[2], -1.0, 1.0))
        excess = max(0.0, tilt - np.radians(float(e['max_tilt_deg'])))
        axis = np.cross(z, [0.0, 0.0, 1.0])
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        R_tgt = R_o @ (from_rotvec(axis * excess) @ R_rel)
        if self.band is not None:
            # The target's turn held inside the wrist's band.
            t = self.turn + float(np.angle(np.exp(1j * (self._yaw(R_tgt) - self._yaw(R_bel)))))
            over = t - float(np.clip(t, *self.band))
            R_tgt = R_o @ Rotation.from_euler('z', -over).as_matrix() @ R_o.T @ R_tgt
        if np.isfinite(float(e['max_torque'])):
            st = rotvec(R_tgt @ R_bel.T)
            ang = np.linalg.norm(st)
            R_tgt = from_rotvec(st * min(1.0, float(e['max_torque']) / self.k_rot / max(ang, 1e-9))) @ R_bel
        if np.isfinite(float(e['max_force'])):
            s = p_tgt + R_tgt @ self.G - (p_bel + R_bel @ self.G)
            n = np.linalg.norm(s)
            p_tgt = p_tgt - s * (1.0 - min(1.0, float(e['max_force']) / self.k_lin / max(n, 1e-9)))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_tgt, p_tgt
        self.P_tgt = T
        return T


# ---------------------------------------------------------------------------- frames
def T_cad_simplug():
    """The sim's plug frame in the 6D CAD's (perception/pose6d.py): origin SEAT_Z_M up the CAD's axis
    (the insulator face + 3.73 mm: the sim's insulator face is at its z +3.726 mm), z reversed (the
    sim's runs up the cable), x kept."""
    from ..perception import pose6d as P6
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[2, 3] = P6.SEAT_Z_M
    return T


def T_conn_simplug(T_junction_conn, part):
    """The sim's plug frame in the app's connector frame: connector -> junction -> CAD -> sim plug."""
    from ..perception import pose6d as P6
    from ..transforms import inverse
    return inverse(T_junction_conn) @ inverse(P6.T_cord_junction(part)) @ T_cad_simplug()


def roll_matched(S, P):
    """S turned about its own z so the plug's x, projected on S's plane, is S's x: the socket's turn is
    a convention (round but for the pins, which only the freely turning collar meets)."""
    x = S[:3, :3].T @ P[:3, 0]
    yaw = np.arctan2(x[1], x[0])
    Rz = np.eye(4)
    Rz[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    return S @ Rz


class SeatDetector:
    """Seated from the arm's own measurements (module doc): advance since the first touch, held."""

    def __init__(self, touch_n, advance_m, hold_s):
        self.touch_n, self.advance_m, self.hold_s = float(touch_n), float(advance_m), float(hold_s)
        self.h_touch, self.since = None, None

    def __call__(self, h_bel, push, t):
        if self.h_touch is None:
            if push > self.touch_n:
                self.h_touch = float(h_bel)
            return False
        if self.h_touch - h_bel >= self.advance_m:
            self.since = t if self.since is None else self.since
            return t - self.since >= self.hold_s - 1e-9
        self.since = None
        return False


# ---------------------------------------------------------------------------- the run
def run(robot, policy, T_base_target, T_tool0_conn, T_conn_sp, rc, guard_n=40.0, on_step=None):
    """The policy's insertion from where the arm is (the believed plug above the target, its axis on
    the target's). Returns (status, info): status 'seated' (SeatDetector), 'timeout', 'turn' (turned
    past rc['max_turn_deg'] -- a safety stop; a policy with a turn band clamps itself inside it, so set it a
    little past the band's width), 'force' (the measured force over guard_n), 'servo' (a command refused).
    info: {'steps', 'advance_mm', 'turn_deg', 'peak_n', 'h_touch_mm'}."""
    from ..robot.admittance import AdmittanceController
    from ..transforms import inverse, slerp_matrix
    spec = policy.spec
    ts = TaskSpace(spec)
    arm = robot.arm
    T_tool0_sp = T_tool0_conn @ T_conn_sp
    G = np.eye(4)
    G[:3, 3] = ts.G
    T_tool0_G = T_tool0_sp @ G
    S0 = T_base_target @ T_conn_sp
    P0 = arm.tcp_pose() @ T_tool0_sp
    S = roll_matched(S0, P0)
    ts.reset(S, P0)
    policy.reset()
    adm = AdmittanceController(arm, {**rc['admittance'], 'frame': T_tool0_G})
    det = SeatDetector(rc['touch_n'], rc['seated_advance_mm'] / 1000.0, rc['hold_s'])
    n_steps = int(round(float(spec['env']['episode_s']) * float(spec['env']['policy_hz'])))
    cycles = max(1, int(round(adm.rate / float(spec['env']['policy_hz']))))
    arm.zero_ft()
    T_ref = arm.tcp_pose()
    T_ref_prev = T_ref
    adm.warmup(T_ref)
    info = {'steps': 0, 'advance_mm': 0.0, 'turn_deg': 0.0, 'peak_n': 0.0, 'h_touch_mm': None}
    yaw0, status, t0, t_prev = None, 'timeout', time.monotonic(), None
    log.info('RL INSERTION: %s, %d steps at %.0f Hz max (%.0f s); the believed plug %.1f mm above the socket, '
             '%.2f mm off its axis.', policy.path, n_steps, float(spec['env']['policy_hz']),
             float(spec['env']['episode_s']), 1000 * (inverse(S) @ P0)[2, 3],
             1000 * np.linalg.norm((inverse(S) @ P0)[:2, 3]))
    try:
        for k in range(n_steps):
            T_tool0 = arm.tcp_pose()
            P_bel = T_tool0 @ T_tool0_sp
            T_G = T_tool0 @ T_tool0_G
            w = arm.wrench_in(np.block([[S[:3, :3], T_G[:3, 3:4]], [np.zeros((1, 3)), np.ones((1, 1))]]), T_tool0)
            w_hand = -np.asarray(w, dtype=float)              # the hand on the plug = minus the plug on the hand
            now = time.monotonic()
            obs = ts.observe(P_bel, w_hand, None if t_prev is None else now - t_prev)
            t_prev = now
            a = policy(obs)
            P_tgt = ts.apply(a)
            T_ref = P_tgt @ inverse(T_tool0_sp)
            f = float(np.linalg.norm(w[:3]))
            info['peak_n'] = max(info['peak_n'], f)
            yaw = float(np.arctan2(*(inverse(S)[:3, :3] @ P_bel[:3, 0])[[1, 0]]))
            yaw0 = yaw if yaw0 is None else yaw0
            info['turn_deg'] += 0.0 if k == 0 else float(np.degrees(np.angle(np.exp(1j * (yaw - info['_yaw'])))))
            info['_yaw'] = yaw
            info['steps'] = k + 1
            h = float(ts.p_rel[2])
            if det(h, ts.meas_push, now - t0):
                status = 'seated'
                break
            info['h_touch_mm'] = None if det.h_touch is None else 1000.0 * det.h_touch
            info['advance_mm'] = 0.0 if det.h_touch is None else 1000.0 * (det.h_touch - h)
            if f > guard_n:
                status = 'force'
                break
            if abs(info['turn_deg']) > float(rc['max_turn_deg']):
                status = 'turn'
                break
            # THE NEW TARGET RAMPED IN over the step's servo cycles, not jumped to: a 1 mm / 3 deg step at
            # once needed 5-11 deg of A4 in one 8 ms cycle (dry run, 2026-11-04) -- the stream's branch-jump
            # guard held it. (The sim's spring took the step; the arm's servo should not have to.)
            for c in range(cycles):
                adm._step(slerp_matrix(T_ref_prev, T_ref, (c + 1) / cycles), 1.0 / adm.rate)
            T_ref_prev = T_ref
            if on_step is not None:
                on_step(k, obs, a, P_bel, P_tgt, w_hand)
    finally:
        adm.stop()
        info.pop('_yaw', None)
    if status == 'seated':
        info['advance_mm'] = 1000.0 * (det.h_touch - float(ts.p_rel[2]))
        info['h_touch_mm'] = 1000.0 * det.h_touch
    return status, info
