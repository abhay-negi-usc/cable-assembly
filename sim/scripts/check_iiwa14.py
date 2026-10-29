"""First run of the iiwa 14 model in the simulator: is it the arm cable_assembly moves?

Spawns IIWA14_2F85_CFG, drives it to HOME and to random joint vectors inside the limits, and after
each settle compares the simulated tool0 pose with cable_assembly/robot/kinematics/iiwa14.py -- the
IK the real arm moves by -- at the joint angles the simulator actually reached. Then closes and opens
the gripper. Exits non-zero past 1 mm / 0.1 deg, so it doubles as the smoke test after an Isaac Lab
or USD rebuild.

    cd sim
    uv run python scripts/check_iiwa14.py                      # headless, Newton
    uv run --extra isaacsim python scripts/check_iiwa14.py --viz kit physics=isaacsim_physx
"""

import argparse
import importlib.util
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--poses", type=int, default=5, help="random joint vectors after HOME")
parser.add_argument("--settle_s", type=float, default=2.0, help="sim time to settle on each target")
add_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from cable_assembly_sim.assets import IIWA14_2F85_CFG  # noqa: E402
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_DRIVE, HOME_DEG, TOOL0  # noqa: E402

# The real arm's kinematics, loaded by path: cable_assembly itself is not installed in this
# environment (its torch and numpy pins differ), and this module only needs numpy.
_KIN = os.path.join(os.path.dirname(__file__), "..", "..", "cable_assembly", "robot", "kinematics", "iiwa14.py")
_spec = importlib.util.spec_from_file_location("iiwa14_kinematics", _KIN)
kin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kin)

POS_TOL_M = 1e-3
ROT_TOL_DEG = 0.1


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = replace(IIWA14_2F85_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def settle(sim, scene, robot, arm_ids, q_arm, finger):
    targets = robot.data.joint_pos.torch.clone()
    targets[:, arm_ids] = q_arm
    targets[:, robot.find_joints(GRIPPER_DRIVE)[0]] = finger
    for _ in range(int(args_cli.settle_s / sim.get_physics_dt())):
        robot.set_joint_position_target_index(target=targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def tool0_error(robot, arm_ids, tool0_id, origin):
    """Simulated tool0 vs kinematics.iiwa14.fk at the joint angles the simulator reached."""
    q = robot.data.joint_pos.torch[0, arm_ids].cpu().numpy().astype(float)
    pose = robot.data.body_pose_w.torch[0, tool0_id]
    p_sim = (pose[:3] - origin).cpu().numpy()
    R_sim = matrix_from_quat(pose[3:7].unsqueeze(0))[0].cpu().numpy()
    T = kin.fk(q)
    dp = np.linalg.norm(p_sim - T[:3, 3])
    dR = math.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ R_sim) - 1) / 2, -1, 1)))
    return q, dp, dR


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([2.0, 2.0, 1.8], [0.0, 0.0, 0.8])
        scene = instantiate(CheckSceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot = scene["robot"]
        arm_ids, names = robot.find_joints(ARM_JOINTS, preserve_order=True)
        assert names == ARM_JOINTS, names
        tool0_id = robot.find_bodies(TOOL0)[0][0]
        origin = scene.env_origins[0]
        print("joints:", robot.joint_names)
        print("bodies:", robot.body_names)

        rng = np.random.default_rng(0)
        lim = kin.JOINT_LIMITS * 0.8
        targets = [np.radians(HOME_DEG)] + [rng.uniform(lim[:, 0], lim[:, 1]) for _ in range(args_cli.poses)]
        worst = (0.0, 0.0)
        for i, q_cmd in enumerate(targets):
            settle(sim, scene, robot, arm_ids, torch.tensor(q_cmd, device=sim.device, dtype=torch.float32), 0.0)
            q, dp, dR = tool0_error(robot, arm_ids, tool0_id, origin)
            tracking = np.degrees(np.abs(q - q_cmd)).max()
            print(f"pose {i}: tool0 vs fk {dp * 1000:6.3f} mm {dR:6.3f} deg | joint tracking {tracking:5.2f} deg")
            worst = (max(worst[0], dp), max(worst[1], dR))

        finger = robot.find_joints(GRIPPER_DRIVE)[0]
        for name, cmd in (("closed", math.radians(45.0)), ("open", 0.0)):
            settle(sim, scene, robot, arm_ids, torch.tensor(targets[0], device=sim.device, dtype=torch.float32), cmd)
            print(f"gripper {name}: finger_joint {math.degrees(robot.data.joint_pos.torch[0, finger].item()):5.1f} deg")

        ok = worst[0] < POS_TOL_M and worst[1] < ROT_TOL_DEG
        print(f"{'PASS' if ok else 'FAIL'}: worst tool0 error {worst[0] * 1000:.3f} mm, {worst[1]:.3f} deg")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
