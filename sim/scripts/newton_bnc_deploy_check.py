"""The robot's code path in closed loop on the sim: every world driven by the app's own TaskSpace + numpy
Policy (cable_assembly/skills/rl_insertion.py) instead of the env's step -- the socket's turn set to the
plug's as the robot does (roll_matched), the lead clamped in the OBSERVED socket's axes, the wrench
measured with the env's noise and bias -- at full estimation error, randomised domain.

It answers what the robot needs and the env's own eval cannot:
  - does the deployed path seat as often as the eval says (98.1% for v16 model_1100)?
  - SEATED WITHOUT GROUND TRUTH: for the advance-after-first-touch rule (SeatDetector), at each
    threshold, how often it fires on a plug truly seated (and stays), how often on one that is not
    (a false seat: collar clocking on an unseated plug), and how often it never fires on a seated one;
  - how far the plug turns about its axis (the arm's A7 must have that room).

    ./run.sh python scripts/newton_bnc_deploy_check.py --policy ../configs/policies/bnc_insert_v16_1100.npz --num_envs 256
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))


def _T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default=None, help="the per-step traces (.npz)")
    a = p.parse_args()

    from cable_assembly.skills import rl_insertion as RL
    from cable_assembly_sim.newton_bnc.env import BncNewtonEnv, DomainCfg, EnvCfg

    torch.manual_seed(a.seed)
    spec = RL.Policy(a.policy).spec
    e = spec["env"]
    dom = DomainCfg(**{k: tuple(v) for k, v in e["domain"].items()})
    env = BncNewtonEnv(EnvCfg(num_envs=a.num_envs, episode_s=e["episode_s"], base_yaw=e["base_yaw"],
                              base_push=e["base_push"], max_torque=e["max_torque"], max_force=e["max_force"],
                              noise_scale=a.noise_scale, domain=dom, wrench_noise=tuple(e["wrench_noise"]),
                              wrench_bias=tuple(e["wrench_bias"]), void_artifacts=False))
    n, dev = env.num_envs, env.device
    ids = torch.arange(n, device=dev)
    env._reset(ids)
    env._read()
    pols = [RL.Policy(a.policy) for _ in range(n)]
    tss = [RL.TaskSpace(spec) for _ in range(n)]
    Pb = [_T(env.R_bel[i].cpu().numpy(), env.p_bel[i].cpu().numpy()) for i in range(n)]
    S = [RL.roll_matched(_T(env.R_obs[i].cpu().numpy(), env.p_obs[i].cpu().numpy()), Pb[i]) for i in range(n)]
    for i in range(n):
        tss[i].reset(S[i], Pb[i])
    wn = np.array([e["wrench_noise"][0]] * 3 + [e["wrench_noise"][1]] * 3)
    T_steps = env.max_episode_length - 1
    h_bel, push, h_true, seated, yaw = (np.zeros((T_steps, n)) for _ in range(5))
    rng = np.random.default_rng(a.seed)
    for k in range(T_steps):
        w_all = env._wrench().cpu().numpy()
        R_bel, p_bel = env.R_bel.cpu().numpy(), env.p_bel.cpu().numpy()
        bias, lvl = env.wrench_bias.cpu().numpy(), env.noise_lvl.cpu().numpy()
        Rt, pt = np.zeros((n, 3, 3)), np.zeros((n, 3))
        for i in range(n):
            P = _T(R_bel[i], p_bel[i])
            Ro = S[i][:3, :3].T
            w = np.concatenate([Ro @ w_all[i, :3], Ro @ w_all[i, 3:]]) + rng.standard_normal(6) * wn * lvl[i] + bias[i]
            o = tss[i].observe(P, w)
            T = tss[i].apply(pols[i](o))
            Rt[i], pt[i] = T[:3, :3], T[:3, 3]
            h_bel[k, i], push[k, i] = tss[i].p_rel[2], tss[i].meas_push
            x = Ro @ P[:3, 0]
            yaw[k, i] = np.arctan2(x[1], x[0])
        env.R_tgt = torch.tensor(Rt, dtype=torch.float32, device=dev)
        env.p_tgt = torch.tensor(pt, dtype=torch.float32, device=dev)
        env._drive()
        env._simulate()
        env._guard()
        env._read()
        h, _, _, st, _ = env.plug_state()
        h_true[k], seated[k] = h.cpu().numpy(), st.cpu().numpy()
        if k % 40 == 0:
            print(f"step {k}/{T_steps}: seated now {seated[k].mean():.3f}", flush=True)

    dt = 1.0 / e["policy_hz"]
    ever = seated.any(0)
    end = seated[-1].astype(bool)
    turn = np.degrees(np.abs(np.unwrap(yaw, axis=0) - yaw[0]).max(0))
    print(f"\nDEPLOYED PATH, {n} worlds, noise x{a.noise_scale}: seated at the end {end.mean():.3f}, ever {ever.mean():.3f}")
    print(f"turn about the axis: p50 {np.percentile(turn, 50):.0f}, p95 {np.percentile(turn, 95):.0f}, max {turn.max():.0f} deg")
    res = {"seated_end": float(end.mean()), "seated_ever": float(ever.mean()), "turn_p95": float(np.percentile(turn, 95)),
           "turn_max": float(turn.max()), "rules": []}
    for touch_n in (5.0,):
        for adv_mm in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
            for hold_s in (0.25, 0.5):
                fire = np.full(n, -1)
                for i in range(n):
                    d = RL.SeatDetector(touch_n, adv_mm / 1000.0, hold_s)
                    for k in range(T_steps):
                        if d(h_bel[k, i], push[k, i], k * dt):
                            fire[i] = k
                            break
                fired = fire >= 0
                ok = np.array([fired[i] and seated[fire[i], i] > 0 for i in range(n)])
                false = fired & ~ok
                missed = ~fired & end
                t = fire[ok] * dt
                r = {"touch_n": touch_n, "advance_mm": adv_mm, "hold_s": hold_s, "correct": float(ok.mean()),
                     "false_seat": float(false.mean()), "missed": float(missed.mean()),
                     "t_p50": float(np.median(t)) if len(t) else None}
                res["rules"].append(r)
                print(f"advance >= {adv_mm:.1f} mm for {hold_s:.2f} s: seated when declared {ok.mean():.3f}, "
                      f"FALSE seat {false.mean():.3f}, seated but never declared {missed.mean():.3f}, "
                      f"declared at p50 {r['t_p50']} s")
    if a.out:
        np.savez(a.out, h_bel=h_bel, push=push, h_true=h_true, seated=seated, yaw=yaw)
        json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
