"""First run of the iiwa 14 model in the simulator: is it the arm cable_assembly moves?

Spawns IIWA14_2F85_CFG, drives it to HOME and to random joint vectors inside the limits, and after
each settle compares the simulated tool0 pose with cable_assembly/robot/kinematics/iiwa14.py -- the
IK the real arm moves by -- at the joint angles the simulator actually reached. Then closes and opens
the gripper. Exits non-zero past 1 mm / 0.1 deg, so it doubles as the smoke test after an Isaac Lab
or USD rebuild.

    cd sim
    ./run.sh python scripts/check_iiwa14.py --physics isaacsim_physx            # headless
    ./run.sh python scripts/check_iiwa14.py --physics isaacsim_physx --viz kit  # with the viewport
"""

import argparse
import importlib.util
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--poses", type=int, default=5, help="random joint vectors after HOME")
parser.add_argument("--settle_s", type=float, default=2.0, help="sim time to settle, on top of the travel time")
# Not one of the launcher's own flags: launch_simulation reads `physics` from the args it is given,
# as Isaac Lab's convert_urdf.py sets it.
parser.add_argument("--physics", choices=["physx", "isaacsim_physx", "ovphysx", "newton_mjwarp"], default=None)
add_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from cable_assembly_sim.assets import IIWA14_2F85_CFG  # noqa: E402
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_DRIVE, GRIPPER_JOINT_SIGNS, HOME_DEG, TOOL0  # noqa: E402

# The real arm's kinematics, loaded by path: cable_assembly itself is not installed in this
# environment (its torch and numpy pins differ), and this module only needs numpy.
_KIN = os.path.join(os.path.dirname(__file__), "..", "..", "cable_assembly", "robot", "kinematics", "iiwa14.py")
_spec = importlib.util.spec_from_file_location("iiwa14_kinematics", _KIN)
kin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kin)

POS_TOL_M = 1e-3
ROT_TOL_DEG = 0.1
GRIPPER_TOL_DEG = 2.0


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = replace(IIWA14_2F85_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def settle(sim, scene, robot, arm_ids, gripper_ids, gripper_signs, q_arm, finger, duration):
    """Hold arm and gripper targets for `duration` s of sim time. `finger` is finger_joint's angle;
    the rest of the 2F-85 linkage gets the matching targets (GRIPPER_JOINT_SIGNS)."""
    q_arm = torch.tensor(q_arm, device=sim.device, dtype=torch.float32).unsqueeze(0)
    q_gripper = (gripper_signs * finger).unsqueeze(0)
    for _ in range(int(duration / sim.get_physics_dt())):
        robot.set_joint_position_target_index(target=q_arm, joint_ids=arm_ids)
        robot.set_joint_position_target_index(target=q_gripper, joint_ids=gripper_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def travel_s(q_from, q_to):
    """The slowest joint's time at its rated speed; the arm cannot arrive sooner."""
    return float(np.max(np.abs(np.asarray(q_to) - np.asarray(q_from)) / kin.VELOCITY_LIMITS))


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
    # A bare PhysicsCfg placeholder, as launch_simulation's docstring has it: --physics replaces the
    # physics configs it finds in its argument, and a default SimulationCfg holds none, so passing
    # one ran PhysX whatever --physics said.
    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg)
        sim = SimulationContext(sim_cfg)
        print("physics:", type(sim_cfg.physics).__name__)
        sim.set_camera_view([2.0, 2.0, 1.8], [0.0, 0.0, 0.8])
        scene = instantiate(CheckSceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot = scene["robot"]
        arm_ids, names = robot.find_joints(ARM_JOINTS, preserve_order=True)
        assert names == ARM_JOINTS, names
        gripper_ids, gripper_names = robot.find_joints(list(GRIPPER_JOINT_SIGNS), preserve_order=True)
        gripper_signs = torch.tensor([GRIPPER_JOINT_SIGNS[n] for n in gripper_names], device=sim.device)
        finger_id = gripper_ids[gripper_names.index(GRIPPER_DRIVE)]
        tool0_id = robot.find_bodies(TOOL0)[0][0]
        origin = scene.env_origins[0]
        print("fixed base:", robot.is_fixed_base, "base DOFs:", robot.num_base_dofs)
        print("joints:", robot.joint_names)
        print("bodies:", robot.body_names)

        rng = np.random.default_rng(0)
        lim = kin.JOINT_LIMITS * 0.8
        home = np.radians(HOME_DEG)
        targets = [home] + [rng.uniform(lim[:, 0], lim[:, 1]) for _ in range(args_cli.poses)] + [home]
        q_prev = robot.data.joint_pos.torch[0, arm_ids].cpu().numpy()
        errors = []
        for i, q_cmd in enumerate(targets):
            duration = args_cli.settle_s + travel_s(q_prev, q_cmd)
            settle(sim, scene, robot, arm_ids, gripper_ids, gripper_signs, q_cmd, 0.0, duration)
            q, dp, dR = tool0_error(robot, arm_ids, tool0_id, origin)
            tracking = np.degrees(np.abs(q - q_cmd)).max()
            print(f"pose {i}: tool0 vs fk {dp * 1000:6.3f} mm {dR:6.3f} deg | joint tracking {tracking:5.2f} deg")
            errors.append((dp, dR))
            q_prev = q_cmd

        gripper_ok = True
        for name, cmd_deg in (("closed", 45.0), ("open", 0.0)):
            settle(sim, scene, robot, arm_ids, gripper_ids, gripper_signs, home, math.radians(cmd_deg), args_cli.settle_s)
            got = math.degrees(robot.data.joint_pos.torch[0, finger_id].item())
            print(f"gripper {name}: finger_joint {got:5.1f} deg (commanded {cmd_deg:.1f})")
            gripper_ok &= abs(got - cmd_deg) < GRIPPER_TOL_DEG

        # NaN compares false against everything, so a simulation that blew up must fail explicitly.
        errors = np.array(errors)
        finite = bool(np.isfinite(errors).all())
        worst = errors.max(axis=0) if finite else (math.nan, math.nan)
        ok = finite and worst[0] < POS_TOL_M and worst[1] < ROT_TOL_DEG and gripper_ok
        print(
            f"{'PASS' if ok else 'FAIL'}: worst tool0 error {worst[0] * 1000:.3f} mm, {worst[1]:.3f} deg"
            f"{'' if finite else ' (the simulation went non-finite)'};"
            f" gripper {'ok' if gripper_ok else 'did not reach its targets'}"
        )
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
