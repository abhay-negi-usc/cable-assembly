"""Evaluate a trained USB-A insertion policy (scripts/newton_usb_train.py) on many worlds, deterministically, with
friction, stiffness, detent, mass and latency randomised as in training. One episode per world per condition:

    none        no estimation error
    in_hand     only the in-hand error (at full range)
    target      only the socket-pose error (at full range)
    both x s    both, at each scale s (1.0 = the training range; above it is out of distribution)

Voided episodes (simulator artifacts) count as failures. Writes <run>/eval_<checkpoint>.json, and with --video a
low-res video of the first worlds at full uncertainty.

    ./run_usb.sh python scripts/newton_usb_eval.py --checkpoint logs/rsl_rl/usb_newton/<run>/model_<it>.pt
"""

import argparse
import dataclasses
import json
import os
import sys
import time

from cable_assembly_sim.newton_usb.env import DomainCfg, EnvCfg, ErrorCfg, UsbNewtonEnv  # first: configures newton

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_usb_train import EvalPolicy, train_cfg  # noqa: E402

from cable_assembly_sim.newton_usb import render  # noqa: E402

ZERO = ErrorCfg(**{f.name: 0.0 for f in dataclasses.fields(ErrorCfg)})
IN_HAND = ("in_hand_y", "in_hand_z", "in_hand_rot_deg")
TARGET = ("target_xy", "target_z", "target_tilt_deg", "target_yaw_deg")


def only(full, *names):
    return dataclasses.replace(ZERO, **{n: getattr(full, n) for n in names})


def load(env, checkpoint):
    from rsl_rl.runners import OnPolicyRunner

    run_args = json.load(open(os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "config.json")))["args"]
    args = argparse.Namespace(seed=0, max_iters=1, save_interval=1, rnn=run_args.get("rnn", "gru"))
    runner = OnPolicyRunner(env, train_cfg(args), log_dir=None, device=str(env.device))
    runner.load(checkpoint)
    return EvalPolicy(runner.alg.get_policy())


@torch.inference_mode()
def episode(env, policy, errors, scale, video=None, worlds=2, title=""):
    n, dev = env.num_envs, env.device
    env.cfg.errors, env.cfg.noise_scale = errors, scale
    ids = torch.arange(n, device=dev)
    env._reset(ids)
    env._read()
    env._obs = env._observe(reset_ids=ids)
    policy.reset()
    obs = env.get_observations()
    first = torch.full((n,), -1, device=dev)
    void = torch.zeros(n, dtype=torch.bool, device=dev)
    peak = torch.zeros(n, device=dev)
    show = torch.arange(min(worlds, n), device=dev)
    snaps = [env.snapshot(show)] if video else None
    for t in range(env.max_episode_length - 1):
        obs = env.step(policy(obs))[0]
        void |= env.voided
        seated = env.plug_state()[3] & ~void
        first[(first < 0) & seated] = t + 1
        peak = torch.maximum(peak, torch.linalg.vector_norm(env.true_wrench[:, 0:3], dim=-1).nan_to_num(0.0))
        if video:
            snaps.append(env.snapshot(show))
    seated = env.plug_state()[3] & ~void
    ts = first[first >= 0].float() * env.step_dt
    if video:
        npz = os.path.splitext(video)[0] + ".npz"
        render.record(snaps, npz, {"step_dt": env.step_dt, "title": title})
        render.render(npz, video, worlds=worlds)
    p = seated.float().mean().item()
    return {"seated_end": round(p, 4), "se": round((p * (1 - p) / n) ** 0.5, 4),
            "ever_seated": round(((first >= 0) & ~void).float().mean().item(), 4),
            "voided": round(void.float().mean().item(), 4),
            "t_seat_median_s": round(ts.median().item(), 2) if len(ts) else None,
            "peak_force_p95_N": round(peak.quantile(0.95).item(), 1),
            "end_force_mean_N": round(torch.linalg.vector_norm(env._wrench()[:, 0:3], dim=-1)[~void].mean().item(), 2),
            "slip_mm_mean": round(1e3 * env.slip_total[~void].mean().item(), 2) if env.cfg.slip else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.25])
    p.add_argument("--video", action="store_true", help="a video of the first worlds at full uncertainty")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    cfg_env = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "config.json")))["env"]
    # The run's env as trained (scalar settings; the error/domain ranges are this code's defaults).
    same = {k: v for k, v in cfg_env.items() if k in {f.name for f in dataclasses.fields(EnvCfg)}
            and not isinstance(v, (dict, list)) and k not in ("num_envs", "curriculum", "noise_scale")}
    full = ErrorCfg(**cfg_env.get("errors", {}))  # the run's full range (scale 1)
    domain = DomainCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_env.get("domain", {}).items()
                          if k in {f.name for f in dataclasses.fields(DomainCfg)}})
    env = UsbNewtonEnv(EnvCfg(num_envs=a.num_envs, curriculum="off", errors=full, domain=domain, **same))
    policy = load(env, a.checkpoint)
    stem = os.path.splitext(os.path.abspath(a.checkpoint))[0]
    conditions = [("none", ZERO, 1.0), ("in_hand", only(full, *IN_HAND), 1.0), ("target", only(full, *TARGET), 1.0)]
    conditions += [(f"both x{s:g}", full, s) for s in a.scales]
    results, t0 = {}, time.time()
    for name, errors, scale in conditions:
        vid = f"{stem}_eval_full.mp4" if (a.video and name == "both x1") else None
        results[name] = episode(env, policy, errors, scale, video=vid,
                                title=f"{os.path.basename(a.checkpoint)}: full uncertainty, deterministic")
        print(f"{name:12s} {json.dumps(results[name])}", flush=True)
    out = f"{stem}_eval.json"
    with open(out, "w") as f:
        json.dump({"checkpoint": a.checkpoint, "num_envs": a.num_envs, "errors_full": dataclasses.asdict(full),
                   "results": results, "artifacts(wall,floor,kick)": env.artifact_counts.int().tolist()}, f, indent=1)
    print(f"wrote {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
