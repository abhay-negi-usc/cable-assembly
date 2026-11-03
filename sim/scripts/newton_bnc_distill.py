"""Distil a privileged teacher (newton_bnc_train.py --privileged: its actor sees the true errors and
poses) into a deployable student (a GRU on the policy observations only), DAgger-style: the student
acts, the teacher labels every state it visits, and the student regresses the teacher's actions with
backpropagation through time over --chunk steps. The output is an ordinary checkpoint (the student's
init checkpoint with its actor replaced) plus a config.json, so newton_bnc_eval.py and the trace tools
read it like a trained run.

    ./run.sh python scripts/newton_bnc_distill.py --teacher logs/rsl_rl/bnc_newton/<priv>/model_<it>.pt \\
        --student logs/rsl_rl/bnc_newton/<run>/model_<it>.pt --hours 3
"""

import argparse
import json
import os
import sys
import time

from cable_assembly_sim.newton_bnc.env import BncNewtonEnv, EnvCfg  # first: configures newton

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_bnc_train import train_cfg  # noqa: E402


def run_args(checkpoint):
    with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "config.json")) as f:
        return json.load(f)["args"]


def load_actor(env, checkpoint):
    from rsl_rl.runners import OnPolicyRunner

    ra = run_args(checkpoint)
    args = argparse.Namespace(seed=0, max_iters=1, save_interval=1, rnn=ra.get("rnn", ""),
                              privileged=ra.get("privileged", False))
    runner = OnPolicyRunner(env, train_cfg(args), log_dir=None, device=str(env.device))
    runner.load(checkpoint)
    return runner.alg.get_policy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True, help="checkpoint to start the student from (a GRU run)")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=64, help="steps per BPTT update")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta0", type=float, default=0.5, help="initial chance the teacher acts (decays to 0)")
    p.add_argument("--beta_iters", type=int, default=100)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--save_interval", type=int, default=25)
    a = p.parse_args()

    ta, sa = run_args(a.teacher), run_args(a.student)
    for k in ("episode_s", "base_yaw", "base_push", "max_torque"):
        assert ta.get(k) == sa.get(k), f"teacher and student differ in {k}: {ta.get(k)} vs {sa.get(k)}"
    env = BncNewtonEnv(EnvCfg(num_envs=a.num_envs, episode_s=sa["episode_s"], base_yaw=sa["base_yaw"],
                              base_push=sa.get("base_push", 0.0), max_torque=sa.get("max_torque", 1.0),
                              artifacts_terminal=True))
    teacher = load_actor(env, a.teacher).eval()
    student = load_actor(env, a.student)
    student.train()
    opt = torch.optim.Adam([q for n, q in student.named_parameters() if "std" not in n], lr=a.lr)

    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", "bnc_newton",
                           time.strftime("%Y-%m-%d_%H-%M-%S") + "_distill")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"args": {**sa, "distilled_from": a.teacher, "student_init": a.student, **vars(a)}}, f, indent=1)
    base = torch.load(a.student, map_location="cpu", weights_only=False)
    print(f"logging to {os.path.abspath(log_dir)}", flush=True)

    deadline = time.time() + a.hours * 3600
    it = 0
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    obs = env.get_observations()
    student.reset()
    while time.time() < deadline:
        beta = a.beta0 * max(0.0, 1.0 - it / a.beta_iters)
        loss = 0.0
        for _ in range(a.chunk):
            act_s = student(obs)
            with torch.no_grad():
                act_t = teacher(obs)
            loss = loss + ((act_s - act_t) ** 2).mean()
            use_t = (torch.rand(env.num_envs, 1, device=env.device) < beta).float()
            act = use_t * act_t + (1 - use_t) * act_s.detach()
            with torch.no_grad():
                obs, _, dones, _ = env.step(act)
            # Clear finished episodes' memory out of place (RSL-RL's reset zeroes in place, which breaks BPTT).
            h = student.rnn.hidden_state
            if h is not None:
                student.rnn.hidden_state = h * (~dones).float()[None, :, None]
        opt.zero_grad()
        (loss / a.chunk).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        student.detach_hidden_state()
        it += 1
        rec = torch.cat(env._outcomes) if env._outcomes else torch.zeros(1, 4)
        if it % 5 == 0:
            print(f"it {it} loss {loss.item() / a.chunk:.4f} beta {beta:.2f} seated_at_end {rec[:, 0].mean():.4f} "
                  f"ever {rec[:, 2].mean():.4f} artifacts {env.artifact_counts.int().tolist()}", flush=True)
        if it % a.save_interval == 0 or time.time() >= deadline:
            ck = dict(base)
            ck["actor_state_dict"] = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            ck["iter"] = it
            torch.save(ck, os.path.join(log_dir, f"model_{it}.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
