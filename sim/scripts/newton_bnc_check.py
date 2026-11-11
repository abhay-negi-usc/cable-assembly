"""Scripted baseline for the Newton BNC env (src/cable_assembly_sim/newton_bnc): the bench's
connector clocking from the policy's own observations -- push down until the force says contact, then
rock about the plug axis while pushing -- at a given error scale. Checks the env end to end and gives
the number a trained policy has to beat.

    ./run.sh python scripts/newton_bnc_check.py --num_envs 64 --noise_scale 0 1
    ./run.sh python scripts/newton_bnc_check.py --turn_band 0 230     # within the bench wrist's budget
"""

import argparse
import math
import time

from cable_assembly_sim.newton_bnc.env import BncNewtonEnv, EnvCfg  # first: configures newton

import torch


def scripted(env, obs, t):
    """Down at full speed until the push exceeds 5 N, then push at 0.6 and rock +/-45 deg at 0.5 Hz."""
    cfg = env.cfg
    h = obs[:, (cfg.history - 1) * 15: cfg.history * 15]
    fz = h[:, 11] * 10.0  # the hand's push on the plug, observed socket z (negative = down)
    a = torch.zeros(env.num_envs, 6, device=env.device)
    contact = fz < -5.0
    a[:, 2] = torch.where(contact, -0.6, -1.0)
    rock = math.sin(2 * math.pi * 0.5 * t * env.step_dt)
    a[:, 5] = torch.where(contact, torch.full_like(fz, rock), torch.zeros_like(fz))
    return a


def run(num_envs, noise_scale, randomize, seed=0, turn_band=None):
    torch.manual_seed(seed)
    env = BncNewtonEnv(EnvCfg(num_envs=num_envs, noise_scale=noise_scale, randomize_domain=randomize,
                              turn_band_deg=tuple(turn_band) if turn_band else None))
    obs = env.get_observations()["policy"]
    seated_any = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    t0 = time.time()
    for t in range(env.max_episode_length - 1):
        obs = env.step(scripted(env, obs, t))[0]["policy"]
        seated_any |= env.plug_state()[3]
    h, r, tilt, seated, engaged = env.plug_state()
    dt = time.time() - t0
    print(f"noise {noise_scale:.2f} domain {'random' if randomize else 'nominal'}: seated at end "
          f"{seated.float().mean():.2f}, ever {seated_any.float().mean():.2f}, engaged {engaged.float().mean():.2f}; "
          f"height mm median {h.median() * 1000:.2f}, finite {torch.isfinite(h).all().item()}; "
          f"{num_envs * (env.max_episode_length - 1) / dt:.0f} env-steps/s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--noise_scale", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--nominal", action="store_true", help="no domain randomisation")
    p.add_argument("--turn_band", type=float, nargs=2, default=None,
                   help="the wrist's turn budget [deg] (env.py turn_band_deg): the rock is clamped inside it")
    a = p.parse_args()
    for s in a.noise_scale:
        run(a.num_envs, s, not a.nominal, turn_band=a.turn_band)
