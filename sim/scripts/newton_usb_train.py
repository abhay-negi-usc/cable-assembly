"""Train the USB-A insertion policy in Newton (src/cable_assembly_sim/newton_usb) with RSL-RL PPO.

The actor observes proprioception and the measured wrench only (a history of frames, and a GRU); the critic
also sees the truth (asymmetric actor-critic). Both estimation errors -- the plug in the hand and the socket's
pose -- are drawn every episode, scaled by a CURRICULUM that starts easy (--curriculum_start) and steps the
scale up (--curriculum_step) whenever episodes at the current scale seat at least --promote_at of the time;
friction, stiffness, the detent, the hand's mass and a step of latency are randomised.

Every --monitor_every iterations: a deterministic evaluation on its own small env (half its worlds at the
curriculum's scale, half at full uncertainty), appended to <run>/eval.csv; and in the background, the
metric plots (<run>/plots/metrics.png) and a low-res video of that evaluation (<run>/videos/iter_<it>.mp4).

    ./run_usb.sh python scripts/newton_usb_train.py --hours 8
    tensorboard --logdir logs/rsl_rl/usb_newton
"""

import argparse
import copy
import csv
import dataclasses
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

from cable_assembly_sim.newton_usb.env import DomainCfg, EnvCfg, ErrorCfg, UsbNewtonEnv  # first: configures newton

import torch
from rsl_rl.runners import OnPolicyRunner

from cable_assembly_sim.newton_usb import render

HERE = os.path.dirname(os.path.abspath(__file__))


def train_cfg(args):
    mlp = dict(class_name="MLPModel", hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    rnn = getattr(args, "rnn", "gru")
    actor = (dict(class_name="RNNModel", rnn_type=rnn, rnn_hidden_dim=256, rnn_num_layers=1, hidden_dims=[256, 128],
                  activation="elu", obs_normalization=True) if rnn else mlp)
    return {
        "seed": args.seed, "device": "cuda:0", "num_steps_per_env": getattr(args, "steps_per_env", 32),
        "max_iterations": args.max_iters, "empirical_normalization": {},
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "clip_actions": None, "check_for_nan": True, "save_interval": args.save_interval,
        "experiment_name": "usb_newton", "run_name": "", "logger": logger_cfg(args), "class_name": "OnPolicyRunner",
        "actor": {**actor, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": getattr(args, "init_std", None) or 0.5,
                                              "std_type": "scalar", "std_range": [getattr(args, "min_std", 1e-3), 10.0]}},
        "critic": {**mlp, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3.0e-4,
                      "schedule": "adaptive", "gamma": getattr(args, "gamma", 0.99), "lam": 0.95, "entropy_coef": getattr(args, "entropy", 1.0e-3),
                      "desired_kl": 0.01, "max_grad_norm": 1.0, "optimizer": "adam", "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True, "clip_param": 0.2, "normalize_advantage_per_mini_batch": False,
                      "share_cnn_encoders": False, "use_mixed_precision": False, "rnd_cfg": None, "symmetry_cfg": None},
        "torch_compile_mode": None, "policy": {},
    }


# New runs (checkpoints, videos, TensorBoard, W&B's local files) go to the external drive, found by its NTFS
# volume UUID because its mount point varies (/media/rp/Elements1, ...). Override with --log_root or $USB_LOG_ROOT.
EXTERNAL_DRIVE_UUID = os.environ.get("USB_LOG_DRIVE_UUID", "D214F87114F859C5")
EXTERNAL_SUBDIR = os.path.join("abhay_ws", "cable-assembly", "logs", "rsl_rl", "usb_newton")
INTERNAL_ROOT = os.path.join(HERE, "..", "logs", "rsl_rl", "usb_newton")


def log_root(arg):
    """Where new run directories go: --log_root ('internal' for sim/logs/...), else $USB_LOG_ROOT, else the external
    drive. Fails rather than falling back to the (nearly full) internal disk when the drive is not mounted."""
    root = arg or os.environ.get("USB_LOG_ROOT")
    if root == "internal":
        return INTERNAL_ROOT
    if root:
        return root
    try:
        mnt = subprocess.run(["findmnt", "-n", "-o", "TARGET", "--source", f"UUID={EXTERNAL_DRIVE_UUID}"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        raise SystemExit(f"the external drive (UUID {EXTERNAL_DRIVE_UUID}) is not mounted: mount it, or pass "
                         f"--log_root <dir> (--log_root internal for sim/logs/rsl_rl/usb_newton)")
    return os.path.join(mnt, EXTERNAL_SUBDIR)


def logger_cfg(args):
    """TensorBoard, and with --wandb_project also Weights & Biases (scalars, evaluation videos, some checkpoints)."""
    project = getattr(args, "wandb_project", "")
    if not project:
        return "tensorboard"
    return {"class_name": "cable_assembly_sim.newton_usb.wandb_writer:UsbWandbWriter", "project_name": project,
            "run_name": getattr(args, "run_name", ""), "group": "usb_newton", "run_id": getattr(args, "wandb_id", None),
            "tags": ["slip"] if getattr(args, "slip", 0) else ["fixed-grasp"],
            "config": {"args": {k: v for k, v in vars(args).items()}}}


@torch.inference_mode()
def evaluate(env: UsbNewtonEnv, actor, scales, video_worlds=2):
    """One deterministic episode per world: the first half at scales[0], the rest at scales[1]. Returns the
    outcomes per half and the snapshots of the first `video_worlds` worlds of the second half."""
    n, dev = env.num_envs, env.device
    half = n // 2
    ids = torch.arange(n, device=dev)
    # Each half drawn at its own scale (the env scales errors at reset by cfg.noise_scale).
    for sl, s in ((slice(0, half), scales[0]), (slice(half, n), scales[1])):
        env.cfg.noise_scale = s
        env._reset(ids[sl])
    env._read()
    env._obs = env._observe(reset_ids=ids)
    actor.reset()
    obs = env.get_observations()
    show = torch.arange(half, half + min(video_worlds, n - half), device=dev)
    snaps = [env.snapshot(show)]
    first = torch.full((n,), -1, device=dev)
    void = torch.zeros(n, dtype=torch.bool, device=dev)
    peak_f = torch.zeros(n, device=dev)
    peak_v = torch.zeros(n, device=dev)
    for t in range(env.max_episode_length - 1):
        obs = env.step(actor(obs))[0]
        void |= env.voided
        seated = env.plug_state()[3] & ~void
        first[(first < 0) & seated] = t + 1
        f = torch.linalg.vector_norm(env._wrench()[:, 0:3], dim=-1).nan_to_num(0.0)
        peak_f = torch.maximum(peak_f, f)
        peak_v = torch.maximum(peak_v, torch.where(f > env.cfg.contact_force, env.speed, 0.0))
        snaps.append(env.snapshot(show))
    seated = env.plug_state()[3] & ~void
    end_f = torch.linalg.vector_norm(env._wrench()[:, 0:3], dim=-1).nan_to_num(0.0)
    out = {}
    for name, sl in (("cur", slice(0, half)), ("full", slice(half, n))):
        f = first[sl]
        ts = f[f >= 0].float() * env.step_dt
        out[f"seated_{name}"] = seated[sl].float().mean().item()
        out[f"t_seat_{name}"] = ts.median().item() if len(ts) else float("nan")
        out[f"voided_{name}"] = void[sl].float().mean().item()
        ok = ~void[sl]
        out[f"peak_force_{name}"] = peak_f[sl][ok].mean().item() if ok.any() else float("nan")
        out[f"end_force_{name}"] = end_f[sl][ok].mean().item() if ok.any() else float("nan")
        out[f"contact_speed_{name}"] = 1e3 * peak_v[sl][ok].mean().item() if ok.any() else float("nan")
    return out, snaps


class EvalPolicy(torch.nn.Module):
    """A deterministic copy of the actor for batched evaluation, with its own recurrent state (built as rsl_rl's
    JIT export is -- from the observation normaliser, the RNN, the MLP head -- but over a batch of worlds; the
    training actor itself holds autograd state and cannot be deep-copied)."""

    def __init__(self, model):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn) if hasattr(model, "rnn") else None
        self.mlp = copy.deepcopy(model.mlp)
        self.out = model.distribution.as_deterministic_output_module() if model.distribution is not None else torch.nn.Identity()
        self.groups = list(model.obs_groups)
        self.h = None
        self.eval()

    def reset(self):
        self.h = None

    def forward(self, obs):
        x = self.obs_normalizer(torch.cat([obs[g] for g in self.groups], -1))
        if self.rnn is not None:
            y, self.h = self.rnn(x.unsqueeze(0), self.h)
            x = y.squeeze(0)
        return self.out(self.mlp(x))


def init_weights(runner, path, init_std=None):
    """Start from another run's weights: the actor exactly (same observations), the critic copied into this
    env's critic, whose privileged input may be wider (EnvCfg.slip adds the grasp's state): the old inputs keep
    their weights and normaliser statistics, new ones start at zero weight and unit variance."""
    ck = torch.load(path, weights_only=False, map_location=runner.device)
    runner.alg.load({"actor_state_dict": ck["actor_state_dict"]}, {"actor": True}, strict=True)
    critic = runner.alg._raw_critic
    new = critic.state_dict()
    for k, v in ck["critic_state_dict"].items():
        t = new[k]
        if t.shape == v.shape:
            new[k] = v
        elif t.dim() == v.dim() and t.shape[:-1] == v.shape[:-1] and t.shape[-1] > v.shape[-1]:
            w = v.shape[-1]
            t = t.clone()
            if "_mean" in k:
                t.zero_()
            elif "_var" in k or "_std" in k:
                t.fill_(1.0)
            else:
                t.zero_()
            t[..., :w] = v
            new[k] = t
        else:
            raise ValueError(f"critic {k}: {tuple(v.shape)} -> {tuple(t.shape)}")
    critic.load_state_dict(new)
    if init_std is not None:  # restart exploration: the source's std may have collapsed
        for name, prm in runner.alg.get_policy().named_parameters():
            if name.endswith("distribution.std_param"):
                with torch.no_grad():
                    prm.fill_(init_std)
                print(f"action std reset to {init_std}", flush=True)
    print(f"initialised from {path} (critic inputs {ck['critic_state_dict']['mlp.0.weight'].shape[1]} -> "
          f"{new['mlp.0.weight'].shape[1]})", flush=True)


class Runner(OnPolicyRunner):
    """Stops on the clock (after saving), and evaluates / plots / films at every monitor point."""

    deadline = float("inf")
    monitor_every = 100
    eval_env: UsbNewtonEnv | None = None
    train_env: UsbNewtonEnv | None = None
    _monitor_proc = None
    _finished = False

    def save(self, path, infos=None):
        super().save(path, infos)
        it = self.current_learning_iteration
        if self.eval_env is not None and it % self.monitor_every == 0:
            self.monitor(it)
        if time.time() > self.deadline:
            print(f"time budget reached at iteration {it}: saved {path}", flush=True)
            if self.eval_env is not None:
                self.monitor(it, final=True)
            self.finish(it, path)
            raise SystemExit(0)

    def finish(self, it, path=None):
        """Upload the last checkpoint and any videos rendered since the last iteration's upload, and close the
        writer (W&B marks the run finished)."""
        w = self.logger.writer
        if w is None or self._finished:
            return
        self._finished = True
        if hasattr(w, "save_video"):
            if self._monitor_proc is not None:
                self._monitor_proc.wait()
            for video in sorted(pathlib.Path(self.logger.log_dir).rglob("*.mp4")):
                w.save_video(video, it)
            if path is not None and os.path.exists(path):
                w.save_file(path)
        self.logger.stop_logging_writer()

    def monitor(self, it, final=False):
        run = self.logger.log_dir
        t0 = time.time()
        actor = EvalPolicy(self.alg.get_policy())  # its own hidden state: training's is untouched
        cur = self.train_env.cfg.noise_scale
        res, snaps = evaluate(self.eval_env, actor, (cur, 1.0))
        row = {"iteration": it, "scale": cur, **{k: f"{v:.4f}" for k, v in res.items()}}
        path = os.path.join(run, "eval.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        if self.logger.writer is not None:
            for k, v in res.items():
                self.logger.writer.add_scalar(f"Eval/{k}", v, it)
            self.logger.writer.flush()
        os.makedirs(os.path.join(run, "videos"), exist_ok=True)
        npz = os.path.join(run, "videos", f"iter_{it:05d}.npz")
        render.record(snaps, npz, {"step_dt": self.eval_env.step_dt,
                                   "title": f"iteration {it}: full uncertainty (curriculum at {cur:.2f})"})
        print(f"[monitor] it {it}: seated {res['seated_cur']:.3f} at scale {cur:.2f}, {res['seated_full']:.3f} at full "
              f"({time.time() - t0:.0f} s)", flush=True)
        # Plots and video in the background (matplotlib on the CPU; training goes on).
        if self._monitor_proc is not None and self._monitor_proc.poll() is None and not final:
            return  # the last one is still rendering: skip this one's video (its npz stays)
        cmd = [sys.executable, os.path.join(HERE, "newton_usb_monitor.py"), run, "--video", npz]
        log = open(os.path.join(run, "monitor.log"), "a")
        self._monitor_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        if final:
            self._monitor_proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--eval_envs", type=int, default=64)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--monitor_every", type=int, default=100, help="iterations between evaluations / plots / videos")
    p.add_argument("--hours", type=float, default=0.0, help="stop at the first checkpoint after this long")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", default=None, help="a model_*.pt to start from")
    p.add_argument("--rnn", default="gru", choices=["", "gru", "lstm"], help="recurrent actor ('' = MLP)")
    p.add_argument("--episode_s", type=float, default=EnvCfg.episode_s)
    p.add_argument("--base_push", type=float, default=EnvCfg.base_push)
    p.add_argument("--init_std", type=float, default=None, help="initial action std (default 0.5); with --init_from, resets it")
    p.add_argument("--entropy", type=float, default=1.0e-3,
                   help="3e-3 (the first run) let the action std climb from 0.24 to 0.43 once success had saturated")
    p.add_argument("--steps_per_env", type=int, default=32, help="rollout length = the GRU's truncated-BPTT window")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--min_std", type=float, default=1e-3, help="floor on the action std (stage 1d collapsed to 0.03)")
    p.add_argument("--w_align", type=float, default=EnvCfg.w_align)
    p.add_argument("--max_force", type=float, default=EnvCfg.max_force)
    p.add_argument("--curriculum_mix", type=float, default=EnvCfg.curriculum_mix)
    p.add_argument("--promote_stall", type=int, default=EnvCfg.promote_stall)
    p.add_argument("--promote_floor", type=float, default=EnvCfg.promote_floor)
    p.add_argument("--history", type=int, default=EnvCfg.history)
    p.add_argument("--curriculum", default="adaptive", choices=["adaptive", "linear", "off"])
    p.add_argument("--curriculum_start", type=float, default=EnvCfg.curriculum_start)
    p.add_argument("--curriculum_step", type=float, default=EnvCfg.curriculum_step)
    p.add_argument("--curriculum_max", type=float, default=EnvCfg.curriculum_max)
    p.add_argument("--promote_at", type=float, default=EnvCfg.promote_at)
    p.add_argument("--promote_window", type=int, default=EnvCfg.promote_window)
    p.add_argument("--noise_scale", type=float, default=1.0, help="with --curriculum off: the fixed scale")
    p.add_argument("--artifacts_terminal", type=int, default=1, help="simulator-artifact episodes end as failures (1) or neutral time-outs (0)")
    p.add_argument("--slip", type=int, default=0, help="the plug slips in the pads (pad friction and grip randomised)")
    p.add_argument("--wandb_project", default="usb_insertion", help="Weights & Biases project ('': TensorBoard only)")
    p.add_argument("--log_root", default=None,
                   help="parent of new run directories (default: the external drive; 'internal': sim/logs/rsl_rl/usb_newton)")
    p.add_argument("--log_dir", default=None,
                   help="write into this (existing) run directory, e.g. with --resume to continue a run in place")
    p.add_argument("--wandb_id", default=None, help="the W&B run id to continue (with --resume and --log_dir)")
    p.add_argument("--run_name", default="", help="appended to the run directory's name in W&B (e.g. stage1d)")
    p.add_argument("--init_from", default=None,
                   help="a model_*.pt to start the weights from (a new run, iteration 0): the actor as it is, the critic "
                        "widened to this env's privileged inputs (new inputs start at zero weight)")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    cfg = EnvCfg(num_envs=a.num_envs, episode_s=a.episode_s, base_push=a.base_push, history=a.history,
                 curriculum=a.curriculum, curriculum_start=a.curriculum_start, curriculum_step=a.curriculum_step,
                 curriculum_max=a.curriculum_max, promote_at=a.promote_at, promote_window=a.promote_window,
                 curriculum_mix=a.curriculum_mix, promote_stall=a.promote_stall, promote_floor=a.promote_floor,
                 w_align=a.w_align, max_force=a.max_force,
                 curriculum_steps=4000 * a.steps_per_env if a.curriculum == "linear" else 0, noise_scale=a.noise_scale,
                 artifacts_terminal=bool(a.artifacts_terminal), slip=bool(a.slip))
    env = UsbNewtonEnv(cfg)
    eval_cfg = dataclasses.replace(cfg, num_envs=a.eval_envs, curriculum="off", errors=ErrorCfg(), domain=DomainCfg(),
                                   artifacts_terminal=False)
    eval_env = UsbNewtonEnv(eval_cfg) if a.monitor_every > 0 else None

    log_dir = a.log_dir or os.path.join(log_root(a.log_root), time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    tc = train_cfg(a)
    cfg_name = "config.json"
    if os.path.exists(os.path.join(log_dir, cfg_name)):  # continuing a run in place: keep its original config
        cfg_name = f"config_resumed_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(os.path.join(log_dir, cfg_name), "w") as f:
        json.dump({"env": dataclasses.asdict(cfg), "train": tc, "args": vars(a)}, f, indent=1)
    runner = Runner(env, tc, log_dir=log_dir, device="cuda:0")
    runner.train_env, runner.eval_env = env, eval_env
    runner.monitor_every = max(a.monitor_every, 1)
    if a.monitor_every % a.save_interval:
        raise SystemExit("--monitor_every must be a multiple of --save_interval (monitoring runs at checkpoints)")
    if a.resume:
        runner.load(a.resume)
    elif a.init_from:
        init_weights(runner, a.init_from, a.init_std)
    if a.hours > 0:
        runner.deadline = time.time() + a.hours * 3600
    print(f"logging to {os.path.abspath(log_dir)}", flush=True)
    try:
        runner.learn(num_learning_iterations=a.max_iters, init_at_random_ep_len=True)
    except KeyboardInterrupt:  # stopped (e.g. by newton_usb_chain.py): close the W&B run cleanly
        it = runner.current_learning_iteration
        print(f"interrupted at iteration {it}", flush=True)
        ckpts = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")), key=os.path.getmtime)
        runner.finish(it, ckpts[-1] if ckpts else None)


if __name__ == "__main__":
    main()
