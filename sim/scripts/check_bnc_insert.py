"""Can the BNC insertion be done at all, through the task's own action interface?

Runs CableAssembly-BNC-Insert-iiwa14-v0 with a scripted controller instead of a policy: centre the
plug over the socket and hold it at the bayonet entry clocking, push it down the straight slots,
then turn it -90 deg while pressing until it locks. The controller cheats -- it reads the TRUE plug
pose, which the policy never sees -- because the question here is whether the physics, the grasp
and the pose-delta -> IK -> joint-target chain let the connector lock, not whether it can be
learnt. Exits non-zero below --min_success, so it doubles as the smoke test after a change.

    cd sim
    ./run.sh python scripts/check_bnc_insert.py                  # headless, 16 envs (~3 min)
    ./run.sh python scripts/check_bnc_insert.py --viz kit --num_envs 4
"""

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=1)
# The controller has no search: a plug that meets the inner fit (bnc.py) more than ~0.05 mm off
# centre jams there. It locked 88% of 32 (2026-10-29); under half means something broke.
parser.add_argument("--min_success", type=float, default=0.5, help="fraction of envs that must lock")
parser.add_argument("--per_env", action="store_true", help="print every env's plug state at the episode end")
parser.add_argument("--friction", type=float, default=None, help="override the task's contact friction")
parser.add_argument("--arm_stiffness_scale", type=float, default=None, help="scale A1..A7 stiffness (damping by its root)")
parser.add_argument("--position_iterations", type=int, default=None, help="override PhysX's position iteration cap")
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cable_assembly_sim.tasks  # noqa: E402, F401
from cable_assembly_sim.assets.bnc import (  # noqa: E402
    ENTRY_YAW_DEG,
    PLUG_MOUTH_Z,
    SOCKET_RIM_Z,
    TWIST_START_HEIGHT,
)

TASK = "CableAssembly-BNC-Insert-iiwa14-v0"


def scripted_action(env):
    """One action from the true plug pose (bnc_insert_env.BncInsertEnv frames)."""
    cfg = env.cfg
    p = env.plug_in_socket_pos
    yaw = torch.atan2(torch.sin(env.plug_yaw), torch.cos(env.plug_yaw))
    a = torch.zeros((env.num_envs, 6), device=env.device)
    # Half the error a step: through the action EMA (0.2) a gain of 1 would put the loop's poles on
    # the unit circle and the plug would orbit the axis.
    a[:, 0:2] = torch.clamp(-0.5 * p[:, 0:2] / cfg.pos_action_scale, -1.0, 1.0)
    off_axis = torch.linalg.vector_norm(p[:, 0:2], dim=-1)

    height = p[:, 2]
    entry_err = math.radians(ENTRY_YAW_DEG) - yaw
    clocked = torch.abs(entry_err) < math.radians(3.0)
    # The sleeve clears the barrel by ~0.2 mm; past the rim the contact has to do the rest (the
    # plug's inner contact then enters the barrel bore with under 0.1 mm).
    clear_of_rim = height > SOCKET_RIM_Z - PLUG_MOUTH_Z + 0.001
    go = clocked & (off_axis < 0.00015)
    in_slots = height < TWIST_START_HEIGHT + 0.0005
    # Landed on the rim off axis: lift off it, so friction does not pin the plug there.
    a[:, 2] = torch.where(
        in_slots | go, -0.5, torch.where(clear_of_rim, torch.where(clocked, -0.25, 0.0), 0.25)
    )
    # Hold the entry clocking down the straight slots, then turn to lock -- and stop there.
    a[:, 5] = torch.where(
        in_slots, torch.where(yaw > 0.0, -1.0, 0.0), torch.clamp(0.5 * entry_err / cfg.rot_action_scale, -1.0, 1.0)
    )
    return a


def print_envs(env):
    """Per env: the plug mouth's offset from the socket axis, its tilt, height and clocking."""
    from isaaclab.utils import math as math_utils

    mouth = math_utils.quat_apply(env.plug_in_socket_quat, torch.tensor([0.0, 0.0, PLUG_MOUTH_Z], device=env.device))
    mouth = mouth + env.plug_in_socket_pos
    axis = math_utils.quat_apply(env.plug_in_socket_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device))
    tilt = torch.rad2deg(torch.acos(axis[:, 2].clamp(-1, 1)))
    for i in range(env.num_envs):
        print(f"    env {i}: height {env.plug_in_socket_pos[i, 2] * 1000:6.2f} mm, mouth off axis "
              f"{mouth[i, 0:2].norm() * 1000:5.2f} mm, origin off axis {env.plug_in_socket_pos[i, 0:2].norm() * 1000:5.2f} mm,"
              f" tilt {tilt[i]:5.2f} deg, yaw {math.degrees(env.plug_yaw[i]):6.1f} deg", flush=True)


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.friction is not None:
        env_cfg.friction = args_cli.friction
    if args_cli.position_iterations is not None:
        env_cfg.sim.physics.max_position_iteration_count = args_cli.position_iterations
    if args_cli.arm_stiffness_scale is not None:
        arm = env_cfg.scene.robot.actuators["arm"]
        arm.stiffness = arm.stiffness * args_cli.arm_stiffness_scale
        arm.damping = arm.damping * math.sqrt(args_cli.arm_stiffness_scale)
    print(f"friction {env_cfg.friction}, arm stiffness {env_cfg.scene.robot.actuators['arm'].stiffness} "
          f"damping {env_cfg.scene.robot.actuators['arm'].damping}", flush=True)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        print("physics:", type(u.cfg.sim.physics).__name__, flush=True)
        env.reset()
        locked_total, n_total = 0.0, 0
        for episode in range(args_cli.episodes):
            log = u.extras.get("log", {})
            print(f"episode {episode}: reset IK error max {log.get('Reset/ik_pos_err_max_mm', float('nan')):.3f} mm "
                  f"{log.get('Reset/ik_rot_err_max_deg', float('nan')):.3f} deg", flush=True)
            plug_in_held_0 = None
            step = 0
            while True:
                with torch.inference_mode():
                    if step == 1:
                        # The plug is read back from the simulator only after the reset's first step.
                        plug_in_held_0 = (u.plug_pos - u.held_pos).norm(dim=-1)
                    if step % 15 == 1:
                        p, yaw = u.plug_in_socket_pos, torch.rad2deg(u.plug_yaw)
                        engaged, locked = u._plug_state()
                        slip = ((u.plug_pos - u.held_pos).norm(dim=-1) - plug_in_held_0).abs().max() * 1000
                        print(f"  t {step * u.step_dt:4.1f} s: height mm min/mean/max "
                              f"{p[:, 2].min() * 1000:6.2f}/{p[:, 2].mean() * 1000:6.2f}/{p[:, 2].max() * 1000:6.2f}"
                              f"  |xy| max {p[:, 0:2].norm(dim=-1).max() * 1000:5.2f} mm"
                              f"  yaw deg {yaw.min():6.1f}..{yaw.max():6.1f}"
                              f"  engaged {engaged.float().mean():.2f} locked {locked.float().mean():.2f}"
                              f"  slip max {slip:5.2f} mm", flush=True)
                    env.step(scripted_action(u))
                    step += 1
                # The time-out step resets every env; the env logs the outcome just before.
                if u.reset_buf.any():
                    break
                if args_cli.per_env and step == u.max_episode_length - 2:
                    print_envs(u)
            # (locked, engaged, ever locked) of the episodes the time-out just ended: all of them.
            locked, engaged, ever = u.ended_outcomes.mean(dim=0).tolist()
            print(f"episode {episode}: locked {locked:.2f} at the end, {ever:.2f} at some point, engaged {engaged:.2f}",
                  flush=True)
            locked_total += locked * u.num_envs
            n_total += u.num_envs
        rate = locked_total / n_total
        ok = rate >= args_cli.min_success and math.isfinite(rate)
        print(f"{'PASS' if ok else 'FAIL'}: {rate:.2f} of {n_total} insertions locked (need {args_cli.min_success:.2f})",
              flush=True)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
