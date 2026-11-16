"""Where and why a cable-held USB-A insertion policy fails (newton_usb_cable). One deterministic episode per world
at each pose-error scale; records where the plug's tip first touched (true socket frame), the end state, the
grasp's peak load, and bins success by the landing offset, by each estimation error and by the domain.

End states (the first of these to happen, frozen there):
    seated      in at the end
    slipped     the grasp's load reached its slip margin (episode ended as a failure)
    voided      a simulation artifact or a non-finite state
    no_touch    never touched anything
    face        ended on the housing face / plate, not in the opening
    panel       ended off the plate, on the panel
    in_opening  ended with the tip inside the opening but not seated (wedged, tilted, slow)

    ./run_usb.sh python scripts/newton_usb_cable_failures.py --checkpoint <run>/model_<it>.pt --scales 0.6 1.0
"""

import argparse
import dataclasses
import json
import os
import sys

from cable_assembly_sim.newton_usb_cable.env import CableEnvCfg, UsbCableEnv  # first: configures newton
from cable_assembly_sim.newton_usb.model import FACE_Z, SHELL_TOP_Z, TIP_DEPTH

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_usb_eval import load  # noqa: E402

TOUCH_N = 0.5
STATES = ["seated", "slipped", "voided", "no_touch", "face", "panel", "in_opening"]
ERR_NAMES = ["grasp_y", "grasp_z", "grasp_rot", "grasp_roll", "sock_x", "sock_y", "sock_z", "sock_roll", "sock_pitch",
             "sock_yaw"]
ERR_FULL = [1, 1, 1, 1, 1, 1, 10 / 7, 0.122 / 0.12, 0.122 / 0.12, 0.122 / 0.12]  # |err| at full scale, per entry


def build_cfg(cfg_env, n):
    """A CableEnvCfg from a run's config.json (nested dataclasses rebuilt), the curriculum off."""
    def make(cls, d):
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            sub = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
            if isinstance(v, dict) and dataclasses.is_dataclass(sub):
                kw[f.name] = make(type(sub), v)
            elif isinstance(v, list):
                kw[f.name] = tuple(v)
            else:
                kw[f.name] = v
        return cls(**kw)

    cfg = make(CableEnvCfg, cfg_env)
    return dataclasses.replace(cfg, num_envs=n, curriculum="off", curriculum_mix=0.0, artifacts_terminal=False)


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
    err, EI, mu, grip, det = env.err.clone(), env.EI.clone(), env.pad_mu.clone(), env.grip.clone(), env.detent_f.clone()
    land = torch.full((n, 3), float("nan"), device=dev)
    ended = torch.full((n,), -1, dtype=torch.long, device=dev)  # 1 slipped, 2 voided: frozen at the first
    peak_f = torch.zeros(n, device=dev)
    peak_load = torch.zeros(n, device=dev)
    capped = torch.zeros(n, device=dev)
    end_tip = torch.zeros(n, 3, device=dev)
    end_h = torch.zeros(n, device=dev)
    end_tilt = torch.zeros(n, device=dev)
    end_seated = torch.zeros(n, dtype=torch.bool, device=dev)
    for t in range(env.max_episode_length - 1):
        live = ended < 0
        # Read the state before stepping (a world that ends this step is reset inside step()).
        f = torch.linalg.vector_norm(env._wrench()[:, 0:3], dim=-1).nan_to_num(0.0)
        tip = env.p_true - env.R_true[:, :, 2] * TIP_DEPTH
        h, r, tilt, seated, engaged = env.plug_state()
        new = live & torch.isnan(land[:, 0]) & (f > TOUCH_N)
        land[new] = tip[new]
        upd = live
        end_tip[upd], end_h[upd], end_tilt[upd], end_seated[upd] = tip[upd], h[upd], tilt[upd], seated[upd]
        obs = env.step(policy(obs))[0]
        load = getattr(env, "grasp_util", torch.zeros(n, device=dev))
        slipped = live & (load >= 1.0) & ~env.voided
        void = live & env.voided
        ended[slipped] = 1
        ended[void & ~slipped] = 2
        live2 = live & ~slipped & ~void
        f2 = torch.linalg.vector_norm(env._wrench()[:, 0:3], dim=-1).nan_to_num(0.0)
        peak_f = torch.where(live2, torch.maximum(peak_f, f2), peak_f)
        peak_load = torch.where(live, torch.maximum(peak_load, load), peak_load)
        capped += (live2 & (f2 > 0.9 * env.cfg.max_force)).float()
    # Worlds still running: their final state.
    live = ended < 0
    tip = env.p_true - env.R_true[:, :, 2] * TIP_DEPTH
    h, r, tilt, seated, engaged = env.plug_state()
    end_tip[live], end_tilt[live], end_seated[live] = tip[live], tilt[live], seated[live]
    tip_r = torch.linalg.vector_norm(end_tip[:, 0:2], dim=-1)
    state = torch.full((n,), 6, device=dev)  # in_opening
    state[end_tip[:, 2] > SHELL_TOP_Z] = 4  # face
    state[(end_tip[:, 2] < FACE_Z - 0.0012) & (tip_r > 0.0105)] = 5  # panel
    state[torch.isnan(land[:, 0])] = 3  # no_touch
    state[end_seated & live] = 0
    state[ended == 1] = 1
    state[ended == 2] = 2
    return dict(err=err, EI=EI, mu=mu, grip=grip, detent=det, land=land, state=state, peak_f=peak_f,
                peak_load=peak_load, capped_s=capped * env.step_dt, end_tilt_deg=torch.rad2deg(end_tilt), end_tip=end_tip)


def summarise(d, scale):
    st = d["state"].cpu()
    out = {"scale": scale, "n": len(st),
           "end_state": {s: round((st == i).float().mean().item(), 3) for i, s in enumerate(STATES)}}
    for axis, k in (("x", 0), ("y", 1)):
        v = (d["land"][:, k].abs() * 1e3).cpu()
        rows = []
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 6), (6, 8), (8, 99)):
            m = (v >= lo) & (v < hi)
            if m.sum():
                rows.append({f"landing_|{axis}|_mm": f"{lo:g}-{hi:g}", "share": round(m.float().mean().item(), 3),
                             "success": round((st[m] == 0).float().mean().item(), 3),
                             "face": round((st[m] == 4).float().mean().item(), 3),
                             "in_opening": round((st[m] == 6).float().mean().item(), 3)})
        out[f"by_landing_{axis}"] = rows
    e = d["err"].abs().cpu() / max(scale, 1e-6)
    out["by_error_thirds(success)"] = {}
    for i, nm in enumerate(ERR_NAMES):
        full = ERR_FULL[i]
        out["by_error_thirds(success)"][nm] = [
            round((st[(e[:, i] >= a * full) & (e[:, i] < b * full)] == 0).float().mean().item(), 3)
            for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))]
    out["by_domain_thirds(success)"] = {}
    for nm in ("EI", "mu", "grip", "detent"):
        v = d[nm].cpu()
        qs = torch.quantile(v.float(), torch.tensor([1 / 3, 2 / 3]))
        bins = [(v < qs[0]), (v >= qs[0]) & (v < qs[1]), (v >= qs[1])]
        out["by_domain_thirds(success)"][nm] = [round((st[b] == 0).float().mean().item(), 3) for b in bins]
    fail = st != 0
    for k in ("peak_f", "peak_load", "capped_s", "end_tilt_deg"):
        v = d[k].cpu()
        out[f"{k}_mean_success/fail"] = [round(v[~fail].mean().item(), 2), round(v[fail].mean().item(), 2)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--scales", type=float, nargs="+", default=[0.6, 1.0])
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    cfg_env = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "config.json")))["env"]
    env = UsbCableEnv(build_cfg(cfg_env, a.num_envs))
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
