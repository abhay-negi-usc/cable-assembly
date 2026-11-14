"""Validate the Newton USB-A insertion env (src/cable_assembly_sim/newton_usb) before training on it.

1. ZERO UNCERTAINTY: the plug exactly where the robot believes it, the socket exactly where it is seen.
   The base controller alone (the env's constant push, zero policy action) must seat every plug, with
   the domain nominal and randomised. Its video is written (--video).
2. THE DETENT holds a plug pushed with less than its resistance, and lets one pushed with more through.
3. A SCRIPTED SEARCH (push + a spiral in the observed socket's plane, from the policy's own observations)
   at a few uncertainty scales: the number a trained policy has to beat.

    ./run_usb.sh python scripts/newton_usb_check.py --video logs/usb_check/zero_uncertainty.mp4
"""

import argparse
import math
import os
import time

from cable_assembly_sim.newton_usb.env import DomainCfg, EnvCfg, UsbNewtonEnv  # first: configures newton

import torch

from cable_assembly_sim.newton_usb import render


def spiral(env, obs, t):
    """Push down; once the measured push says the tip is on the face, spiral outward in the observed socket's
    plane (0.25 mm pitch) until the plug drops in, then keep pushing."""
    cfg = env.cfg
    h = obs[:, (cfg.history - 1) * 15: cfg.history * 15]
    z = h[:, 2] * 0.01  # believed plug height over the observed socket [m]
    push = env.meas_push
    a = torch.zeros(env.num_envs, 6, device=env.device)
    on_face = (push > 4.0) & (z > 0.0085)  # pressed, but not in (in: the tip below the face)
    env._spiral_t = torch.where(on_face, env._spiral_t + 1, env._spiral_t) if hasattr(env, "_spiral_t") else torch.zeros(env.num_envs, device=env.device)
    k = env._spiral_t
    ang = 0.6 * k
    rad = 0.00025 * ang / (2 * math.pi)
    tgt = torch.stack((rad * torch.cos(ang), rad * torch.sin(ang)), -1)
    cur = h[:, 0:2] * 0.01
    a[:, 0:2] = torch.where(on_face[:, None], ((tgt - cur) / cfg.pos_scale).clamp(-1, 1), torch.zeros_like(cur))
    a[:, 2] = torch.where(on_face, 0.3, 0.0)  # ease off while searching (the base push stays)
    return a


@torch.inference_mode()
def run(env, act, video=None, worlds=2, label=""):
    n = env.num_envs
    ids = torch.arange(n, device=env.device)
    env._reset(ids)
    env._read()
    env._obs = env._observe(reset_ids=ids)
    if hasattr(env, "_spiral_t"):
        del env._spiral_t
    obs = env.get_observations()["policy"]
    ever = torch.zeros(n, dtype=torch.bool, device=env.device)
    first = torch.full((n,), -1, device=env.device)
    void = torch.zeros(n, dtype=torch.bool, device=env.device)
    snaps = []
    show = torch.arange(min(worlds, n), device=env.device)
    if video:
        snaps.append(env.snapshot(show))
    t0 = time.time()
    T = env.max_episode_length - 1
    for t in range(T):
        obs = env.step(act(env, obs, t))[0]["policy"]
        void |= env.voided
        seated = env.plug_state()[3] & ~void
        first[(first < 0) & seated] = t + 1
        ever |= seated
        if video:
            snaps.append(env.snapshot(show))
    h, r, tilt, seated, engaged = env.plug_state()
    seated = seated & ~void
    ts = first[first >= 0].float() * env.step_dt
    print(f"{label:38s} seated at end {seated.float().mean():.3f}  ever {ever.float().mean():.3f}  voided {void.float().mean():.3f}  "
          f"t_seat median {ts.median().item() if len(ts) else float('nan'):.2f} s  "
          f"({n * T / (time.time() - t0):.0f} env-steps/s)", flush=True)
    if video:
        npz = os.path.splitext(video)[0] + ".npz"
        render.record(snaps, npz, {"step_dt": env.step_dt, "title": label})
        render.render(npz, video, worlds=worlds)
        print(f"  video: {os.path.abspath(video)}", flush=True)
    return seated.float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--video", default="logs/usb_check/zero_uncertainty.mp4")
    p.add_argument("--video_worlds", type=int, default=2)
    p.add_argument("--scales", type=float, nargs="*", default=[0.25, 0.5, 1.0])
    a = p.parse_args()
    torch.manual_seed(0)
    zero = lambda env, obs, t: torch.zeros(env.num_envs, 6, device=env.device)  # noqa: E731

    # 1. Zero uncertainty: the base push alone.
    env = UsbNewtonEnv(EnvCfg(num_envs=a.num_envs, noise_scale=0.0, randomize_domain=False))
    ok = run(env, zero, video=a.video, worlds=a.video_worlds, label="zero uncertainty, nominal domain, push")
    env.cfg.randomize_domain = True
    ok_dr = run(env, zero, label="zero uncertainty, randomised domain, push")

    # 2. The detent: 12 N of resistance, the arm's push capped below and above it.
    for cap, expect in ((8.0, False), (30.0, True)):
        env.cfg.randomize_domain, env.cfg.domain, env.cfg.max_force = False, DomainCfg(detent=(12.0, 12.0)), cap
        s = run(env, zero, label=f"detent 12 N, push capped at {cap:.0f} N")
        h = env.plug_state()[0]
        print(f"  plug height median {h.median() * 1e3:.2f} mm (seated < 0.8; springs reached at 8.2)", flush=True)
        assert (s > 0.95) == expect, f"the detent {'stopped' if expect else 'let through'} a plug pushed with {cap} N"
    env.cfg.max_force, env.cfg.domain, env.cfg.randomize_domain = EnvCfg.max_force, DomainCfg(), True

    # 3. Under uncertainty: the push alone and a scripted spiral search.
    for s in a.scales:
        env.cfg.noise_scale = s
        run(env, zero, label=f"uncertainty x{s:g}, push")
        run(env, spiral, label=f"uncertainty x{s:g}, push + spiral search")
    assert ok > 0.99 and ok_dr > 0.97, "the env does not insert at zero uncertainty"
    print("zero-uncertainty validation passed", flush=True)


if __name__ == "__main__":
    main()
