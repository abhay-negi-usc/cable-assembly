"""Validate the cable-held USB-A insertion env (src/cable_assembly_sim/newton_usb_cable) before training on it.

ZERO UNCERTAINTY: the socket where it is seen and the grasp where the robot believes (the grasp's control
error is still drawn every episode, and known). The base controller alone (the env's constant push, zero
policy action) must seat the plug through the bending cable: with the domain nominal (and a video), and with
it randomised (the cable's EI 3e-4 to 3e-3 N m^2, friction, arm stiffness, detent, mass, latency). Then the
push alone under uncertainty, for reference.

    ./run_usb.sh python scripts/newton_usb_cable_check.py --video logs/usb_cable_check/zero_uncertainty.mp4
"""

import argparse
import os
import time

from cable_assembly_sim.newton_usb_cable.env import CableEnvCfg, UsbCableEnv  # first: configures newton

import torch

from cable_assembly_sim.newton_usb_cable import render


@torch.inference_mode()
def run(env, video=None, worlds=2, label=""):
    n = env.num_envs
    ids = torch.arange(n, device=env.device)
    env._reset(ids)
    env._read()
    env._obs = env._observe(reset_ids=ids)
    zero = torch.zeros(n, 6, device=env.device)
    first = torch.full((n,), -1, device=env.device)
    void = torch.zeros(n, dtype=torch.bool, device=env.device)
    peak_bend = torch.zeros(n, device=env.device)
    peak_f = torch.zeros(n, device=env.device)
    show = torch.arange(min(worlds, n), device=env.device)
    snaps = [env.snapshot(show)] if video else []
    t0 = time.time()
    T = env.max_episode_length - 1
    for t in range(T):
        env.step(zero)
        void |= env.voided
        seated = env.plug_state()[3] & ~void
        first[(first < 0) & seated] = t + 1
        peak_bend = torch.maximum(peak_bend, env.cable_bend().nan_to_num(0.0))
        peak_f = torch.maximum(peak_f, torch.linalg.vector_norm(env._wrench()[:, 0:3], dim=-1).nan_to_num(0.0))
        if video:
            snaps.append(env.snapshot(show))
    seated = env.plug_state()[3] & ~void
    ts = first[first >= 0].float() * env.step_dt
    print(f"{label:44s} seated {seated.float().mean():.3f}  voided {void.float().mean():.3f}  "
          f"t_seat median {ts.median().item() if len(ts) else float('nan'):.2f} s  peak force {peak_f.mean():.1f} N  "
          f"peak cable bend {torch.rad2deg(peak_bend).mean():.1f} deg (max {torch.rad2deg(peak_bend).max():.1f})  "
          f"nonfinite {env.nonfinite_count}  ({n * T / (time.time() - t0):.0f} env-steps/s)", flush=True)
    if video:
        os.makedirs(os.path.dirname(os.path.abspath(video)), exist_ok=True)
        npz = os.path.splitext(video)[0] + ".npz"
        render.record(snaps, npz, {"step_dt": env.step_dt, "title": label})
        render.render(npz, video, worlds=worlds)
        print(f"  video: {os.path.abspath(video)}", flush=True)
    return seated.float().mean().item(), seated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--episode_s", type=float, default=10.0)
    p.add_argument("--video", default="logs/usb_cable_check/zero_uncertainty.mp4")
    p.add_argument("--video_worlds", type=int, default=3)
    p.add_argument("--scales", type=float, nargs="*", default=[0.25, 0.5])
    a = p.parse_args()
    torch.manual_seed(0)
    env = UsbCableEnv(CableEnvCfg(num_envs=a.num_envs, episode_s=a.episode_s, noise_scale=0.0, randomize_domain=False))
    ok, _ = run(env, video=a.video, worlds=a.video_worlds,
                label="zero uncertainty, nominal domain, random grasp, push")
    env.cfg.randomize_domain = True
    ok_dr, seated = run(env, label="zero uncertainty, randomised domain, push")
    EI = env.EI
    for lo, hi in ((3e-4, 6e-4), (6e-4, 1.2e-3), (1.2e-3, 3.1e-3)):
        m = (EI >= lo) & (EI < hi)
        print(f"  EI {lo:.1e}-{hi:.1e}: seated {seated[m].float().mean():.3f} ({int(m.sum())} worlds)", flush=True)
    for s in a.scales:
        env.cfg.noise_scale = s
        run(env, label=f"uncertainty x{s:g}, randomised domain, push")
    print(f"zero uncertainty: nominal {ok:.3f}, randomised {ok_dr:.3f}", flush=True)


if __name__ == "__main__":
    main()
