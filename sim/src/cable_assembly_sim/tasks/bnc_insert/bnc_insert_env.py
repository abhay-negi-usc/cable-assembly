"""The BNC insertion environment (configuration and conventions: bnc_insert_env_cfg.py).

Frames. The SOCKET frame is the socket's own (z up its axis, the pins at +/-90 deg about it). The
policy acts on and observes the HELD frame: where the plug would be if it sat in the gripper exactly
as grasped, i.e. tool0 composed with the nominal grasp. The robot can know that one; the true plug
pose, which differs by the in-hand slip, goes to the critic only.
"""

import math

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import math as math_utils

from cable_assembly_sim.assets.bnc import ENTRY_YAW_DEG
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_JOINT_SIGNS, TOOL0

from .bnc_insert_env_cfg import BncInsertEnvCfg


OUTCOME_WINDOW = 1024


def squashing_fn(x, a, b):
    """Factory's bounded keypoint reward, 1 / (exp(ax) + b + exp(-ax))."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def wrap_to_pi(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class BncInsertEnv(DirectRLEnv):
    cfg: BncInsertEnvCfg

    def __init__(self, cfg: BncInsertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._robot = self.scene["robot"]
        self._socket = self.scene["socket"]
        self._plug = self.scene["plug"]

        self._arm_ids, names = self._robot.find_joints(ARM_JOINTS, preserve_order=True)
        assert names == ARM_JOINTS, names
        self._gripper_ids, names = self._robot.find_joints(list(GRIPPER_JOINT_SIGNS), preserve_order=True)
        self._gripper_signs = torch.tensor([GRIPPER_JOINT_SIGNS[n] for n in names], device=self.device)
        self._tool0_id = self._robot.find_bodies(TOOL0)[0][0]
        # A fixed base has no Jacobian for its root body; a floating one prepends its 6 DOFs.
        self._jacobian_id = self._tool0_id - 1 if self._robot.is_fixed_base else self._tool0_id
        self._jacobian_cols = [i + self._robot.num_base_dofs for i in self._arm_ids]

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": cfg.ik_damping}
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

        n, dev = self.num_envs, self.device
        # The nominal grasp: the plug's frame in tool0's, pointing back up the tool axis.
        self._grasp_pos = torch.tensor([0.0, 0.0, cfg.grasp_depth], device=dev).repeat(n, 1)
        self._grasp_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
        self._identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        self._ik_seed = torch.tensor([math.radians(q) for q in cfg.ik_seed_deg], device=dev)
        self._keypoint_offsets = torch.zeros((cfg.num_keypoints, 3), device=dev)
        self._keypoint_offsets[:, 2] = (torch.linspace(0.0, 1.0, cfg.num_keypoints, device=dev) - 0.5) * cfg.keypoint_scale
        self._pos_bounds = torch.tensor(cfg.pos_action_bounds, device=dev)

        # Per-episode state, filled by the reset.
        self._socket_obs_pos = torch.zeros((n, 3), device=dev)  # what the policy believes, env frame
        self._socket_obs_yaw = torch.zeros(n, device=dev)
        self._ik_target_pos = torch.zeros((n, 3), device=dev)
        self._ik_target_quat = self._identity_quat.clone()
        self._prev_held_pos = torch.zeros((n, 3), device=dev)
        self._prev_held_quat = self._identity_quat.clone()
        self.prev_actions = torch.zeros((n, cfg.action_space), device=dev)
        self.ep_succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        # Outcomes of the last OUTCOME_WINDOW finished episodes (locked, engaged, ever locked): a
        # step ends only a few, so a rate over just those would be noise.
        self._outcomes = torch.full((OUTCOME_WINDOW, 3), float("nan"), device=dev)
        self._outcome_next = 0

        for asset in (self._robot, self._plug, self._socket):
            set_friction(asset, cfg.friction, n)

        self._compute_intermediate_values()

    # ---- State ------------------------------------------------------------------------------------

    def _compute_intermediate_values(self):
        """Read the simulation into the frames the task is written in."""
        origins = self.scene.env_origins
        self.socket_pos = self._socket.data.root_pos_w.torch - origins
        self.socket_quat = self._socket.data.root_quat_w.torch
        self.plug_pos = self._plug.data.root_pos_w.torch - origins
        self.plug_quat = self._plug.data.root_quat_w.torch
        self._compute_arm_state()

        # The plug against its seat, true and as observed.
        self.plug_in_socket_pos, self.plug_in_socket_quat = math_utils.subtract_frame_transforms(
            self.socket_pos, self.socket_quat, self.plug_pos, self.plug_quat
        )
        obs_socket_quat = yaw_quat(self._socket_obs_yaw)
        self.held_in_obs_socket_pos, self.held_in_obs_socket_quat = math_utils.subtract_frame_transforms(
            self._socket_obs_pos, obs_socket_quat, self.held_pos, self.held_quat
        )
        self.plug_yaw = math_utils.euler_xyz_from_quat(self.plug_in_socket_quat)[2]

    def _compute_arm_state(self):
        """What the IK needs: the held frame, its Jacobian and the arm's joints."""
        origins = self.scene.env_origins
        self.tool0_pos = self._robot.data.body_pos_w.torch[:, self._tool0_id] - origins
        self.tool0_quat = self._robot.data.body_quat_w.torch[:, self._tool0_id]
        self.held_pos, self.held_quat = math_utils.combine_frame_transforms(
            self.tool0_pos, self.tool0_quat, self._grasp_pos, self._grasp_quat
        )
        # The held frame's Jacobian: tool0's, its linear rows moved to the held frame's origin.
        jac = self._robot.data.body_link_jacobian_w.torch[:, self._jacobian_id, :, self._jacobian_cols]
        r = self.held_pos - self.tool0_pos
        jac_lin = jac[:, 0:3] - torch.linalg.cross(r.unsqueeze(-1).expand(-1, -1, jac.shape[-1]), jac[:, 3:6], dim=1)
        self.held_jacobian = torch.cat((jac_lin, jac[:, 3:6]), dim=1)
        self.arm_joint_pos = self._robot.data.joint_pos.torch[:, self._arm_ids]
        self.arm_joint_vel = self._robot.data.joint_vel.torch[:, self._arm_ids]

    def _get_dones(self):
        """Episodes end only on time-out."""
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    # ---- Actions ----------------------------------------------------------------------------------

    def _pre_physics_step(self, action):
        self.actions = self.cfg.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ema_factor) * self.actions
        obs_socket_quat = yaw_quat(self._socket_obs_yaw)

        # The displacement, in the observed socket frame, moves the TARGET, which then may lead the
        # measured pose by at most max_target_lead. Re-targeting from the measured pose instead
        # (Factory's way) has no memory: the plug's own weight sagged the arm 0.25 mm a step, without
        # end. The lead bounds the contact force all the same, at about the arm stiffness times it.
        d_pos = math_utils.quat_apply(obs_socket_quat, self.actions[:, 0:3] * self.cfg.pos_action_scale)
        d_rot = math_utils.quat_apply(obs_socket_quat, self.actions[:, 3:6] * self.cfg.rot_action_scale)
        lead_pos, lead_rot = self.cfg.max_target_lead
        lead = torch.clamp(self._ik_target_pos + d_pos - self.held_pos, -lead_pos, lead_pos)
        # Keep the target near the socket axis, as Factory does.
        rel = math_utils.quat_apply_inverse(obs_socket_quat, self.held_pos + lead - self._socket_obs_pos)
        rel = torch.clamp(rel, self._pos_bounds[0], self._pos_bounds[1])
        target_pos = self._socket_obs_pos + math_utils.quat_apply(obs_socket_quat, rel)

        target_quat = math_utils.quat_mul(quat_from_rotvec(d_rot), self._ik_target_quat)
        lead = math_utils.axis_angle_from_quat(math_utils.quat_mul(target_quat, math_utils.quat_conjugate(self.held_quat)))
        angle = torch.linalg.vector_norm(lead, dim=-1, keepdim=True)
        lead = lead * torch.clamp(lead_rot / angle.clamp_min(1e-9), max=1.0)
        target_quat = math_utils.quat_mul(quat_from_rotvec(lead), self.held_quat)
        if self.cfg.lock_tilt:
            target_quat = yaw_quat(math_utils.euler_xyz_from_quat(target_quat)[2])
        self._ik_target_pos[:] = target_pos
        self._ik_target_quat[:] = target_quat
        self._ik.set_command(torch.cat((target_pos, target_quat), dim=-1))

    def _apply_action(self):
        """Every physics step: IK toward the held-frame target from the current state. (The gripper's
        squeeze target is set once, at reset.)"""
        self._compute_arm_state()
        q = self._ik.compute(self.held_pos, self.held_quat, self.held_jacobian, self.arm_joint_pos)
        self._robot.set_joint_position_target_index(target=q, joint_ids=self._arm_ids)

    def _set_gripper_target(self, deg, env_ids=None):
        target = (self._gripper_signs * math.radians(deg)).repeat(self.num_envs if env_ids is None else len(env_ids), 1)
        self._robot.set_joint_position_target_index(target=target, joint_ids=self._gripper_ids, env_ids=env_ids)

    # ---- Observations -----------------------------------------------------------------------------

    def _get_observations(self):
        held_yaw = math_utils.euler_xyz_from_quat(self.held_in_obs_socket_quat)[2]
        obs_socket_quat = yaw_quat(self._socket_obs_yaw)
        # Finite differences over the policy step: the arm's contact-time velocities are noisy.
        lin_vel = math_utils.quat_apply_inverse(obs_socket_quat, (self.held_pos - self._prev_held_pos) / self.step_dt)
        d_quat = math_utils.quat_mul(self.held_quat, math_utils.quat_conjugate(self._prev_held_quat))
        ang_vel = math_utils.quat_apply_inverse(obs_socket_quat, math_utils.axis_angle_from_quat(d_quat) / self.step_dt)
        self._prev_held_pos[:] = self.held_pos
        self._prev_held_quat[:] = self.held_quat

        policy = torch.cat(
            (self.held_in_obs_socket_pos, torch.sin(held_yaw).unsqueeze(-1), torch.cos(held_yaw).unsqueeze(-1),
             lin_vel, ang_vel, self.actions),
            dim=-1,
        )
        # The critic sees the truth: the held frame against the real socket, the real plug, the arm.
        held_in_socket_pos, held_in_socket_quat = math_utils.subtract_frame_transforms(
            self.socket_pos, self.socket_quat, self.held_pos, self.held_quat
        )
        true_yaw = math_utils.euler_xyz_from_quat(held_in_socket_quat)[2]
        plug_in_held_pos, plug_in_held_quat = math_utils.subtract_frame_transforms(
            self.held_pos, self.held_quat, self.plug_pos, self.plug_quat
        )
        critic = torch.cat(
            (held_in_socket_pos, torch.sin(true_yaw).unsqueeze(-1), torch.cos(true_yaw).unsqueeze(-1),
             lin_vel, ang_vel, self.actions,
             self.plug_in_socket_pos, self.plug_in_socket_quat, plug_in_held_pos, plug_in_held_quat,
             self.arm_joint_pos, self.arm_joint_vel),
            dim=-1,
        )
        return {"policy": policy, "critic": critic}

    # ---- Rewards ----------------------------------------------------------------------------------

    def _plug_state(self):
        """(engaged, locked) of the true plug, against the true socket."""
        p = self.plug_in_socket_pos
        centered = torch.linalg.vector_norm(p[:, 0:2], dim=-1) < self.cfg.center_tol
        engaged = centered & (p[:, 2] < self.cfg.engage_height)
        # Locked at 0 or 180 deg: the two pins are alike.
        clock_err = torch.abs(wrap_to_pi(2.0 * self.plug_yaw)) / 2.0
        locked = engaged & (p[:, 2] < self.cfg.success_height) & (clock_err < math.radians(self.cfg.success_yaw_deg))
        return engaged, locked

    def _get_rewards(self):
        cfg = self.cfg
        n = self.num_envs
        # Keypoints along the plug axis, against the same points on the seat (the socket frame).
        kp_plug = math_utils.combine_frame_transforms(
            self.plug_pos.unsqueeze(1).expand(-1, cfg.num_keypoints, -1).reshape(-1, 3),
            self.plug_quat.unsqueeze(1).expand(-1, cfg.num_keypoints, -1).reshape(-1, 4),
            self._keypoint_offsets.repeat(n, 1),
        )[0].view(n, cfg.num_keypoints, 3)
        kp_seat = math_utils.combine_frame_transforms(
            self.socket_pos.unsqueeze(1).expand(-1, cfg.num_keypoints, -1).reshape(-1, 3),
            self.socket_quat.unsqueeze(1).expand(-1, cfg.num_keypoints, -1).reshape(-1, 4),
            self._keypoint_offsets.repeat(n, 1),
        )[0].view(n, cfg.num_keypoints, 3)
        kp_dist = torch.linalg.vector_norm(kp_plug - kp_seat, dim=-1).mean(-1)

        engaged, locked = self._plug_state()
        # Once the pins are in the straight slots the only way on is the -90 deg turn; pay for it.
        twist = engaged.float() * torch.clamp(1.0 - torch.abs(wrap_to_pi(self.plug_yaw)) / (math.pi / 2), 0.0, 1.0)
        terms = {
            "kp_baseline": squashing_fn(kp_dist, *cfg.keypoint_coef_baseline),
            "kp_coarse": squashing_fn(kp_dist, *cfg.keypoint_coef_coarse),
            "kp_fine": squashing_fn(kp_dist, *cfg.keypoint_coef_fine),
            "engaged": engaged.float(),
            "twist": twist,
            "locked": locked.float(),
        }
        reward = sum(terms.values())

        self.ep_succeeded |= locked
        # A new dict every step: RSL-RL keeps each step's and averages them over the iteration.
        log = self.extras["log"] = {}
        for name, value in terms.items():
            log[f"Episode_Reward/{name}"] = value.mean()
        done = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # (locked, engaged, ever locked) of each episode ending now.
        self.ended_outcomes = torch.stack((locked, engaged, self.ep_succeeded), dim=-1)[done].float()
        recent = self.ended_outcomes[-OUTCOME_WINDOW:]
        slots = (self._outcome_next + torch.arange(len(recent), device=self.device)) % OUTCOME_WINDOW
        self._outcomes[slots] = recent
        self._outcome_next = (self._outcome_next + len(recent)) % OUTCOME_WINDOW
        if not torch.isnan(self._outcomes[0, 0]):
            # Over the last OUTCOME_WINDOW episodes to finish.
            rates = torch.nanmean(self._outcomes, dim=0)
            log["Metrics/success_rate"] = rates[0]
            log["Metrics/engaged_rate"] = rates[1]
            log["Metrics/ever_locked_rate"] = rates[2]
        self.prev_actions = self.actions.clone()
        return reward

    # ---- Reset ------------------------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        """Place the socket and put the arm over it holding the plug -- with state writes alone, no
        physics step, so any subset of environments can reset (the RSL-RL runner starts them at
        random points of their episodes). Isaac Lab's UR10e gear-assembly reset works the same way."""
        super()._reset_idx(env_ids)
        cfg, dev = self.cfg, self.device
        ids = torch.as_tensor(env_ids, device=dev).long()
        n = len(ids)
        origins = self.scene.env_origins[ids]

        # The socket, and what the policy will be told about it.
        socket_pos = torch.tensor(cfg.socket_pos, device=dev).repeat(n, 1)
        socket_pos[:, 0:2] += uniform((n, 2), torch.tensor(cfg.socket_pos_noise, device=dev))
        socket_yaw = math.radians(cfg.socket_yaw_deg) + uniform(n, math.radians(cfg.socket_yaw_noise_deg), dev)
        pose = torch.cat((socket_pos + origins, yaw_quat(socket_yaw)), dim=-1)
        self._socket.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self._socket.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=dev), env_ids=ids)
        self._socket_obs_pos[ids] = socket_pos + torch.randn((n, 3), device=dev) * cfg.socket_obs_pos_noise
        self._socket_obs_pos[ids, 2] = socket_pos[:, 2]
        self._socket_obs_yaw[ids] = socket_yaw + torch.randn(n, device=dev) * math.radians(cfg.socket_obs_yaw_noise_deg)

        # Where the held frame starts: above the seat, clocked to the bayonet entry.
        start = torch.zeros((n, 3), device=dev)
        start[:, 0:2] = uniform((n, 2), cfg.start_xy_noise, dev)
        start[:, 2] = cfg.start_height[0] + torch.rand(n, device=dev) * (cfg.start_height[1] - cfg.start_height[0])
        start_yaw = math.radians(ENTRY_YAW_DEG) + uniform(n, math.radians(cfg.start_yaw_noise_deg), dev)
        target_pos, target_quat = math_utils.combine_frame_transforms(
            socket_pos, yaw_quat(socket_yaw), start, yaw_quat(start_yaw)
        )

        # The arm: DLS IK from the seed, each iteration written straight into the joints. The fingers
        # go to just wider than the plug.
        joint_pos = self._robot.data.default_joint_pos.torch[ids].clone()
        joint_pos[:, self._arm_ids] = self._ik_seed
        joint_pos[:, self._gripper_ids] = self._gripper_signs * math.radians(cfg.gripper_grasp_deg)
        for _ in range(cfg.ik_iterations):
            self._write_joints(joint_pos, ids)
            self._compute_arm_state()
            pos_err, rot_err = math_utils.compute_pose_error(
                self.held_pos[ids], self.held_quat[ids], target_pos, target_quat, rot_error_type="axis_angle"
            )
            jac = self.held_jacobian[ids]
            lam = (cfg.ik_damping**2) * torch.eye(6, device=dev)
            err = torch.cat((pos_err, rot_err), dim=-1).unsqueeze(-1)
            joint_pos[:, self._arm_ids] += (jac.mT @ torch.linalg.solve(jac @ jac.mT + lam, err)).squeeze(-1)
        self._write_joints(joint_pos, ids)
        self._compute_intermediate_values()
        self._log_ik_error(ids, target_pos, target_quat)

        # The plug between the pads, off its nominal grasp by the in-hand noise. The gripper's target
        # is past contact, so its drive closes the last few tenths and squeezes from the first step.
        noise = uniform((n, 3), torch.tensor(cfg.in_hand_noise, device=dev))
        plug_pos, plug_quat = math_utils.combine_frame_transforms(self.held_pos[ids], self.held_quat[ids], noise)
        self._plug.write_root_pose_to_sim_index(root_pose=torch.cat((plug_pos + origins, plug_quat), dim=-1), env_ids=ids)
        self._plug.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=dev), env_ids=ids)
        self._plug.reset(ids)
        self._set_gripper_target(cfg.gripper_close_deg, ids)

        self.actions[ids] = 0.0
        self.prev_actions[ids] = 0.0
        self.ep_succeeded[ids] = False
        self._prev_held_pos[ids] = self.held_pos[ids]
        self._prev_held_quat[ids] = self.held_quat[ids]
        # The episode's target starts where the plug is held.
        self._ik_target_pos[ids] = self.held_pos[ids]
        self._ik_target_quat[ids] = yaw_quat(math_utils.euler_xyz_from_quat(self.held_quat[ids])[2])
        self._ik.set_command(torch.cat((self._ik_target_pos, self._ik_target_quat), dim=-1))

    def _write_joints(self, joint_pos, ids):
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=ids)
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=ids)

    def _log_ik_error(self, ids, target_pos, target_quat):
        pos_err, rot_err = math_utils.compute_pose_error(
            self.held_pos[ids], self.held_quat[ids], target_pos, target_quat, rot_error_type="axis_angle"
        )
        log = self.extras.setdefault("log", {})
        log["Reset/ik_pos_err_max_mm"] = torch.linalg.vector_norm(pos_err, dim=-1).max().item() * 1000
        log["Reset/ik_rot_err_max_deg"] = math.degrees(torch.linalg.vector_norm(rot_err, dim=-1).max().item())


def quat_from_rotvec(rotvec):
    """Quaternions (x, y, z, w) of rotation vectors (axis times angle)."""
    angle = torch.linalg.vector_norm(rotvec, dim=-1)
    quat = math_utils.quat_from_angle_axis(angle, rotvec / angle.clamp_min(1e-9).unsqueeze(-1))
    return torch.where((angle > 1e-6).unsqueeze(-1), quat, torch.tensor([0.0, 0.0, 0.0, 1.0], device=rotvec.device))


def yaw_quat(yaw):
    """Quaternions (x, y, z, w) of rotations by `yaw` about z."""
    zero = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(zero, zero, yaw)


def uniform(shape, half_range, device=None):
    """Uniform in [-half_range, half_range]; `half_range` a float or a tensor broadcast over `shape`."""
    if isinstance(half_range, torch.Tensor):
        device = half_range.device
    return (2.0 * torch.rand(shape, device=device) - 1.0) * half_range


def set_friction(asset, value, num_envs):
    """Static and dynamic friction of every shape of `asset` (Factory's factory_utils.set_friction)."""
    import warp as wp

    materials = wp.to_torch(asset.root_view.get_material_properties())
    materials[..., 0] = value
    materials[..., 1] = value
    env_ids = torch.arange(num_envs, device="cpu", dtype=torch.int32)
    asset.root_view.set_material_properties(wp.from_torch(materials.contiguous()), wp.from_torch(env_ids))
