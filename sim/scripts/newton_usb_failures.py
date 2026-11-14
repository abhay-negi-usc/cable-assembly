"""Where and why a USB-A insertion policy fails: one deterministic episode per world at each scale, recording
where the plug's tip first touched (in the true socket frame) and where the episode ended, then binning the
outcome by the landing offset across the plug's thin direction (socket y) and by the end state:

    seated      in at the end
    no_touch    never touched anything
    face        ended on the housing face / plate (tip above the shell's top), not in the opening
    panel       ended off the plate, on the panel below it
    in_opening  ended with the tip inside the opening but not seated (wedged, jammed, or slow)

    ./run_usb.sh python scripts/newton_usb_failures.py --checkpoint <run>/model_<it>.pt --scales 0.75 1.0
"""

import argparse
import dataclasses
import json
import os
import sys

from cable_assembly_sim.newton_usb.env import DomainCfg, EnvCfg, ErrorCfg, UsbNewtonEnv  # first: configures newton
from cable_assembly_sim.newton_usb.model import FACE_Z, PANEL_TOP_Z, SHELL_TOP_Z, TIP_DEPTH

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_usb_eval import load  # noqa: E402

TOUCH_N = 1.0
EDGES_MM = [0.0, 1.0, 2.0, 3.0, 4.0, 4.8, 6.0, 8.0, 99.0]


@torch.inference_mode()
def run(env, policy, scale):
    n, dev = env.num_envs, env.device
    env.cfg.noise_scale = scale
    ids = torch.arange(n, device=dev)
    env._reset(ids)
    env._read()
    env._obs = env._observe(reset_ids=ids)
    policy.reset()
    obs = env.get_observations()
    err = env.err.clone()
    land = torch.full((n, 3), float("nan"), device=dev)
    void = torch.zeros(n, dtype=torch.bool, device=dev)
    peak = torch.zeros(n, device=dev)
    contact_steps = torch.zeros(n, device=dev)
    capped_steps = torch.zeros(n, device=dev)
    for _ in range(env.max_episode_length - 1):
        obs = env.step(policy(obs))[0]
        void |= env.voided
        f = torch.linalg.vector_norm(env.true_wrench[:, 0:3], dim=-1).nan_to_num(0.0)
        tip = env.p_true - env.R_true[:, :, 2] * TIP_DEPTH
        new = torch.isnan(land[:, 0]) & (f > TOUCH_N)
        land[new] = tip[new]
        peak = torch.maximum(peak, f)
        contact_steps += (f > TOUCH_N).float()
        capped_steps += (f > 0.9 * env.cfg.max_force).float()
    h, r, tilt, seated, engaged = env.plug_state()
    seated &= ~void
    tip = env.p_true - env.R_true[:, :, 2] * TIP_DEPTH
    tip_r = torch.linalg.vector_norm(tip[:, 0:2], dim=-1)
    state = torch.full((n,), 4, device=dev)  # in_opening
    state[tip[:, 2] > SHELL_TOP_Z] = 2  # face
    state[(tip[:, 2] < FACE_Z - 0.0012) & (tip_r > 0.0105)] = 3  # panel
    state[torch.isnan(land[:, 0])] = 1  # no_touch
    state[seated] = 0
    state[void] = 5
    dom = torch.stack((env.mu, env.k_lin, env.k_rot, env.detent_f), -1)
    return dict(err=err, dom=dom, land=land, state=state, peak=peak, contact_s=contact_steps * env.step_dt,
                capped_s=capped_steps * env.step_dt, end_tilt_deg=torch.rad2deg(tilt), end_tip=tip)


STATES = ["seated", "no_touch", "face", "panel", "in_opening", "voided"]


def summarise(d, scale):
    n = len(d["state"])
    st = d["state"].cpu()
    ly = (d["land"][:, 1].abs() * 1e3).cpu()
    lx = (d["land"][:, 0].abs() * 1e3).cpu()
    out = {"scale": scale, "n": n, "end_state": {s: round((st == i).float().mean().item(), 3) for i, s in enumerate(STATES)}}
    rows = []
    for lo, hi in zip(EDGES_MM[:-1], EDGES_MM[1:]):
        m = (ly >= lo) & (ly < hi)
        if m.sum() == 0:
            continue
        rows.append({"landing_|y|_mm": f"{lo:g}-{hi:g}", "share": round(m.float().mean().item(), 3),
                     "success": round((st[m] == 0).float().mean().item(), 3),
                     **{s: round((st[m] == i).float().mean().item(), 3) for i, s in enumerate(STATES) if i}})
    out["by_landing_y"] = rows
    rows = []
    for lo, hi in zip([0, 2, 4, 6, 8], [2, 4, 6, 8, 99]):
        m = (lx >= lo) & (lx < hi)
        if m.sum():
            rows.append({"landing_|x|_mm": f"{lo:g}-{hi:g}", "share": round(m.float().mean().item(), 3),
                         "success": round((st[m] == 0).float().mean().item(), 3)})
    out["by_landing_x"] = rows
    # Success against each estimation error (normalised by its full range), split in thirds of |error|.
    names = ["hand_y", "hand_z", "hand_rot", "sock_x", "sock_y", "sock_z", "sock_roll", "sock_pitch", "sock_yaw"]
    e = d["err"].abs().cpu() / scale
    out["by_error_thirds"] = {nm: [round((st[(e[:, i] >= a) & (e[:, i] < b)] == 0).float().mean().item(), 3)
                                   for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))] for i, nm in enumerate(names)}
    fail = st != 0
    jam = st == 4
    for i, nm in enumerate(["mu", "k_lin", "k_rot", "detent_N"]):
        v = d["dom"][:, i].cpu()
        qs = torch.quantile(v, torch.tensor([1 / 3, 2 / 3]))
        bins = [(v < qs[0]), (v >= qs[0]) & (v < qs[1]), (v >= qs[1])]
        out.setdefault("by_domain_thirds(success|in_opening)", {})[nm] = [
            (round((st[b] == 0).float().mean().item(), 3), round(jam[b].float().mean().item(), 3)) for b in bins]
    for k in ("peak", "contact_s", "capped_s", "end_tilt_deg"):
        v = d[k].cpu()
        out[f"{k}_mean_success/fail"] = [round(v[~fail].mean().item(), 2), round(v[fail].mean().item(), 2)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0])
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    cfg_env = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "config.json")))["env"]
    same = {k: v for k, v in cfg_env.items() if k in {f.name for f in dataclasses.fields(EnvCfg)}
            and not isinstance(v, (dict, list)) and k not in ("num_envs", "curriculum", "noise_scale")}
    full = ErrorCfg(**cfg_env.get("errors", {}))
    domain = DomainCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_env.get("domain", {}).items()
                          if k in {f.name for f in dataclasses.fields(DomainCfg)}})
    env = UsbNewtonEnv(EnvCfg(num_envs=a.num_envs, curriculum="off", errors=full, domain=domain, **same))
    policy = load(env, a.checkpoint)
    res = []
    for s in a.scales:
        d = run(env, policy, s)
        res.append(summarise(d, s))
        print(json.dumps(res[-1], indent=1), flush=True)
        torch.save({k: v.cpu() for k, v in d.items()}, os.path.splitext(a.checkpoint)[0] + f"_failures_x{s:g}.pt")
    with open(os.path.splitext(a.checkpoint)[0] + "_failures.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
