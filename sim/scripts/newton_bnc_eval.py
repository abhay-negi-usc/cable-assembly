"""Test a trained BNC policy (scripts/newton_bnc_train.py) for robustness, against the scripted baseline
(scripts/newton_bnc_check.py), in the Newton env with friction and stiffness randomised.

Conditions, one episode per world each (errors are uniform with the training half-widths times the
scale; `--scales` sets the "both" sweep):
    none              no estimation error
    in_hand           only the in-hand SE(2) error (x, z along the plug, turn about the closing axis)
    target            only the socket's 6-DOF error
    both x s          both, at each scale s (1.0 = the training range; above it is out of distribution)
    one component     each of the 9 error components alone at its full half-width (--components)
and, over the `both x 1.0` episodes, success by friction bin (plug on socket, collar on its seat).

    ./run.sh python scripts/newton_bnc_eval.py --checkpoint logs/rsl_rl/bnc_newton/<run>/model_<it>.pt
"""

import argparse
import dataclasses
import json
import os
import sys
import time

from cable_assembly_sim.newton_bnc.env import BncNewtonEnv, EnvCfg, ErrorCfg  # first: configures newton

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_bnc_check import scripted  # noqa: E402

ZERO = ErrorCfg(**{f.name: 0.0 for f in dataclasses.fields(ErrorCfg)})
IN_HAND = ("in_hand_x", "in_hand_z", "in_hand_rot_deg")
TARGET = ("target_xy", "target_z", "target_tilt_deg", "target_yaw_deg")


def only(*names):
    full = ErrorCfg()
    return dataclasses.replace(ZERO, **{n: getattr(full, n) for n in names})


def load_policy(env, checkpoint):
    from rsl_rl.runners import OnPolicyRunner
    from newton_bnc_train import train_cfg

    # The architecture the run was trained with (its config.json; model_700's predates it: an MLP).
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "config.json")
    run_args = json.load(open(cfg_path))["args"] if os.path.exists(cfg_path) else {}
    args = argparse.Namespace(seed=0, max_iters=1, save_interval=1, rnn=run_args.get("rnn", ""),
                              privileged=run_args.get("privileged", False))
    runner = OnPolicyRunner(env, train_cfg(args), log_dir=None, device=str(env.device))
    runner.load(checkpoint)
    return runner.get_inference_policy(device=str(env.device))


@torch.inference_mode()
def episode(env, errors, scale, act):
    """Reset every world with these errors, run one episode, return per-world outcomes."""
    n, dev = env.num_envs, env.device
    env.cfg.errors, env.cfg.noise_scale, env.cfg.curriculum_steps = errors, scale, 0
    ids = torch.arange(n, device=dev)
    if hasattr(act, "reset"):
        act.reset()  # a recurrent policy's memory
    env._reset(ids)
    env._read()
    env._obs = env._observe(reset_ids=ids)
    obs = env.get_observations()
    first = torch.full((n,), -1, dtype=torch.long, device=dev)
    peak = torch.zeros(n, device=dev)
    nonfinite0 = env.nonfinite_count
    void = torch.zeros(n, dtype=torch.bool, device=dev)  # non-finite or a simulator artifact: a failure
    seated = engaged = None
    T = env.max_episode_length - 1  # stop before the env's own time-out reset
    for t in range(T):
        obs = env.step(act(obs, t))[0]
        void |= env.voided
        h, r, tilt, seated, engaged = env.plug_state()
        seated, engaged = seated & ~void, engaged & ~void
        first[(first < 0) & seated] = t + 1
        peak = torch.maximum(peak, torch.linalg.vector_norm(env.true_wrench[:, 0:3], dim=-1).nan_to_num(0.0))
    return {
        "seated_end": seated.float(), "ever_seated": ((first >= 0) & ~void).float(), "engaged_end": engaged.float(),
        "voided": void.float(),
        "t_seat_s": torch.where(first >= 0, first.float() * env.step_dt, torch.nan),
        "peak_force": peak, "mu": env.mu.clone(), "nonfinite": env.nonfinite_count - nonfinite0,
    }


def summary(o):
    t = o["t_seat_s"][torch.isfinite(o["t_seat_s"])]
    return {
        "seated_end": round(o["seated_end"].mean().item(), 3),
        "ever_seated": round(o["ever_seated"].mean().item(), 3),
        "engaged_end": round(o["engaged_end"].mean().item(), 3),
        "t_seat_median_s": round(t.median().item(), 2) if len(t) else None,
        "peak_force_p95_N": round(o["peak_force"].quantile(0.95).item(), 1),
        "nonfinite": o["nonfinite"],
        "voided": round(o["voided"].mean().item(), 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--components", action="store_true", help="also each error component alone")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--stochastic", action="store_true", help="sample the policy's actions (as in training)")
    p.add_argument("--observe_twice", action="store_true", help="the observation bug model_700 was trained with (env.py)")
    p.add_argument("--components_only", action="store_true")
    p.add_argument("--quick", action="store_true", help="only none, in_hand, target and both x1")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_force", type=float, default=EnvCfg.max_force, help="cap on the arm spring [N] (inf: none)")
    p.add_argument("--no_void", action="store_true", help="keep episodes that show a simulator artifact")
    p.add_argument("--episode_s", type=float, default=None, help="default: the run's (config.json)")
    p.add_argument("--base_yaw", type=float, default=None, help="default: the run's (config.json)")
    p.add_argument("--contact_ke", type=float, default=None, help="default: the current (accurate) model's")
    p.add_argument("--physics_hz", type=float, default=None, help="default: the current (accurate) env's")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    # The controller settings the run trained with, unless overridden.
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "config.json")
    run_args = json.load(open(cfg_path))["args"] if os.path.exists(cfg_path) else {}
    a.episode_s = a.episode_s if a.episode_s is not None else run_args.get("episode_s", EnvCfg.episode_s)
    a.base_yaw = a.base_yaw if a.base_yaw is not None else run_args.get("base_yaw", 0.0)
    max_torque = run_args.get("max_torque", float("inf"))  # runs before the torque cap had none
    base_push = run_args.get("base_push", 0.0)
    from cable_assembly_sim.newton_bnc.model import ModelCfg

    # Every checkpoint is judged in the current, accurate physics unless asked otherwise.
    ke = a.contact_ke if a.contact_ke is not None else ModelCfg.contact_ke
    model = ModelCfg(contact_ke=ke, contact_kd=ModelCfg.contact_kd * (ke / ModelCfg.contact_ke) ** 0.5)
    hz = a.physics_hz if a.physics_hz is not None else EnvCfg.physics_hz
    env = BncNewtonEnv(EnvCfg(num_envs=a.num_envs, observe_twice=a.observe_twice, max_force=a.max_force, episode_s=a.episode_s, base_yaw=a.base_yaw, max_torque=max_torque, base_push=base_push, model=model, physics_hz=hz,
                              void_artifacts=not a.no_void))
    policy = load_policy(env, a.checkpoint)
    class Act:
        def __call__(self, obs, t):
            return policy(obs, stochastic_output=True) if a.stochastic else policy(obs)

        def reset(self):
            if getattr(policy, "is_recurrent", False):
                policy.reset()

    agents = {"policy": Act()}
    if not a.no_baseline:
        agents["scripted"] = lambda obs, t: scripted(env, obs["policy"], t)

    conditions = [("none", ZERO, 1.0), ("in_hand", only(*IN_HAND), 1.0), ("target", only(*TARGET), 1.0)]
    conditions += [(f"both x{s:g}", ErrorCfg(), s) for s in a.scales]
    if a.quick:
        conditions = conditions[:3] + [("both x1", ErrorCfg(), 1.0)]
    if a.components_only:
        conditions = []
    if a.components or a.components_only:
        conditions += [(f"only {n}", only(n), 1.0) for n in IN_HAND + TARGET]

    results = {}
    t0 = time.time()
    for name, errors, scale in conditions:
        for agent, act in agents.items():
            if agent == "scripted" and name.startswith("only "):
                continue
            o = episode(env, errors, scale, act)
            results.setdefault(name, {})[agent] = summary(o)
            print(f"{name:24s} {agent:9s} {json.dumps(results[name][agent])}", flush=True)
            if name == "both x1" and agent == "policy":
                # Success by friction bin over the training distribution.
                bins = {}
                for i, (lo, hi) in enumerate(((0.15, 0.25), (0.25, 0.35), (0.35, 0.45))):
                    m = (o["mu"][:, 0] >= lo) & (o["mu"][:, 0] < hi)
                    bins[f"mu_parts {lo}-{hi}"] = round(o["ever_seated"][m].mean().item(), 3)
                for lo, hi in ((0.25, 0.35), (0.35, 0.45), (0.45, 0.55)):
                    m = (o["mu"][:, 1] >= lo) & (o["mu"][:, 1] < hi)
                    bins[f"mu_seat {lo}-{hi}"] = round(o["ever_seated"][m].mean().item(), 3)
                results[name]["policy_by_friction"] = bins
                print(f"{'':24s} by friction (ever seated): {json.dumps(bins)}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), f"eval_{os.path.basename(a.checkpoint)[:-3]}" + ("_stochastic" if a.stochastic else "") + ("_twice" if a.observe_twice else "") + ("_components" if a.components_only else "") + (f"_f{a.max_force:g}" if a.max_force != EnvCfg.max_force else "") + (f"_ke{a.contact_ke:g}" if a.contact_ke else "") + ".json")
    with open(out, "w") as f:
        json.dump({"checkpoint": a.checkpoint, "num_envs": a.num_envs, "errors_full": dataclasses.asdict(ErrorCfg()),
                   "results": results}, f, indent=1)
    print(f"wrote {out} ({time.time() - t0:.0f} s); contact-buffer overflow steps: {int(env.overflow_steps)}; "
          f"artifacts (pass-through, off-axis, overrun): {env.artifact_counts.int().tolist()}", flush=True)


if __name__ == "__main__":
    main()
