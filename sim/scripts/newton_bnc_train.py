"""Train the BNC insertion policy in Newton (src/cable_assembly_sim/newton_bnc) with RSL-RL PPO.

Robust to the in-hand SE(2) error and the 6-DOF target error (a curriculum ramps both to full range
over the first --curriculum_iters), with friction (plug on socket, collar on its seat), arm/grasp
stiffness and washer stiffness randomised every episode. Checkpoints every --save_interval
iterations, so a run can be stopped at any time; --hours stops it on the clock.

    ./run.sh python scripts/newton_bnc_train.py --hours 4
    tensorboard --logdir logs/rsl_rl/bnc_newton
"""

import argparse
import dataclasses
import json
import os
import time

from cable_assembly_sim.newton_bnc.env import WIDE_DOMAIN, BncNewtonEnv, DomainCfg, EnvCfg  # first: configures newton

import torch
from rsl_rl.runners import OnPolicyRunner

from cable_assembly_sim.newton_bnc.model import ModelCfg as ModelCfgDefault


def _actor_group(args):
    """The actor's observation group: policy (deployable), or a privileged one for a teacher."""
    priv = getattr(args, "privileged", "")
    return "critic" if priv is True else (priv or getattr(args, "actor_obs", "policy") or "policy")  # True: runs before the teacher group


def _critic_group(args):
    return "critic_pw" if getattr(args, "actor_obs", "") == "pose_wrench" else "critic"


def train_cfg(args):
    mlp = dict(class_name="MLPModel", hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    # The actor may be recurrent (--rnn gru|lstm): the errors are hidden and only contact over time
    # reveals them. The critic sees them (privileged), so it stays an MLP.
    rnn = getattr(args, "rnn", "")
    actor = (dict(class_name="RNNModel", rnn_type=rnn, rnn_hidden_dim=256, rnn_num_layers=1,
                  hidden_dims=[256, 128], activation="elu", obs_normalization=True) if rnn else mlp)
    return {
        "seed": args.seed, "device": "cuda:0", "num_steps_per_env": getattr(args, "steps_per_env", 32), "max_iterations": args.max_iters,
        "empirical_normalization": {}, "obs_groups": {"actor": [_actor_group(args)], "critic": [_critic_group(args)]},
        "clip_actions": None, "check_for_nan": True, "save_interval": args.save_interval,
        "experiment_name": "bnc_newton", "run_name": "", "logger": "tensorboard",
        "class_name": "OnPolicyRunner",
        "actor": {**actor, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": getattr(args, "init_std", 0.6),
                                              "std_type": "scalar", "std_range": [1e-3, 10.0]}},
        "critic": {**mlp, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3.0e-4,
                      "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "entropy_coef": getattr(args, "entropy", 3.0e-3), "desired_kl": 0.01,
                      "max_grad_norm": 1.0, "optimizer": "adam", "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True, "clip_param": 0.2, "normalize_advantage_per_mini_batch": False,
                      "share_cnn_encoders": False, "use_mixed_precision": False, "rnd_cfg": None, "symmetry_cfg": None},
        "torch_compile_mode": None, "policy": {},
    }


def load_padded(runner, path):
    """Load a checkpoint whose actor / critic took fewer observations (new ones appended at the end): the
    first layers get zero columns for them and the normalisers mean 0 / std 1, so the loaded policy acts
    exactly as before until training uses them. The optimizer state is then dropped."""
    d = torch.load(path, weights_only=False, map_location="cpu")
    padded = False
    for key, model in (("actor_state_dict", runner.alg._raw_actor), ("critic_state_dict", runner.alg._raw_critic)):
        cur = model.state_dict()
        for k, t in d[key].items():
            c = cur.get(k)
            if c is None or c.shape == t.shape:
                continue
            assert c.dim() == t.dim() and c.shape[:-1] == t.shape[:-1] and c.shape[-1] > t.shape[-1], (k, c.shape, t.shape)
            fill = 1.0 if k.endswith(("_var", "_std")) else 0.0
            new = torch.full(c.shape, fill, dtype=t.dtype)
            new[..., : t.shape[-1]] = t
            d[key][k] = new
            padded = True
    runner.alg.load(d, {"actor": True, "critic": True, "optimizer": not padded, "iteration": True}, strict=True)
    runner.current_learning_iteration = d["iter"]
    print(f"loaded {path}" + (" (padded for new observations; fresh optimizer)" if padded else ""), flush=True)


class TimedRunner(OnPolicyRunner):
    """Stops (after saving) once the wall clock passes the deadline."""

    deadline = float("inf")

    def save(self, path, infos=None):
        super().save(path, infos)
        if time.time() > self.deadline:
            print(f"time budget reached at iteration {self.current_learning_iteration}: saved {path}", flush=True)
            if self.logger.writer is not None:
                self.logger.stop_logging_writer()
            raise SystemExit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--curriculum_iters", type=int, default=300)
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--hours", type=float, default=0.0, help="stop at the first checkpoint after this long")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", default=None, help="a model_*.pt to start from")
    p.add_argument("--rnn", default="", choices=["", "gru", "lstm"], help="recurrent actor")
    p.add_argument("--episode_s", type=float, default=EnvCfg.episode_s)
    p.add_argument("--max_force", type=float, default=EnvCfg.max_force)
    p.add_argument("--max_torque", type=float, default=EnvCfg.max_torque)
    p.add_argument("--base_push", type=float, default=EnvCfg.base_push, help="base push (env.py)")
    p.add_argument("--init_std", type=float, default=0.6)
    p.add_argument("--steps_per_env", type=int, default=32, help="rollout length per PPO update")
    p.add_argument("--privileged", default="", choices=["", "critic", "teacher", "teacher_gated"],
                   help="a privileged actor (not deployable): critic = everything; teacher = without the collar angle")
    p.add_argument("--contact_ke", type=float, default=ModelCfgDefault.contact_ke, help="contact stiffness; damping scales with its sqrt")
    p.add_argument("--physics_hz", type=float, default=EnvCfg.physics_hz)
    p.add_argument("--artifacts_terminal", action="store_true", help="artifact episodes end as failures")
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="train errors at this multiple of the spec ranges (after any curriculum); >1 hardens the edges")
    p.add_argument("--jam_shaping", type=float, default=0.0, help="potential-based reward for lifting out of a jam (env.py)")
    p.add_argument("--entropy", type=float, default=3.0e-3)
    p.add_argument("--base_yaw", type=float, default=EnvCfg.base_yaw, help="base twist while pushing (env.py)")
    p.add_argument("--actor_obs", default="policy", choices=["policy", "pose_wrench"],
                   help="pose_wrench: the actor knows only the noisy connector pose and the measured wrench (history)")
    p.add_argument("--pose_noise", type=float, nargs=2, default=list(EnvCfg.pose_noise), help="per-step std: m, deg")
    p.add_argument("--noise_level", type=float, nargs=2, default=list(EnvCfg.noise_level), help="per-episode sensor-noise multiplier range")
    p.add_argument("--wide_dr", action="store_true", help="env.WIDE_DOMAIN: wider friction/stiffness, mass, action delay")
    p.add_argument("--turn_band", type=float, nargs=2, default=None, help="the wrist's turn budget [deg] (env.py turn_band_deg)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    from cable_assembly_sim.newton_bnc.model import ModelCfg

    model = ModelCfg(contact_ke=args.contact_ke, contact_kd=ModelCfg.contact_kd * (args.contact_ke / ModelCfg.contact_ke) ** 0.5)
    cfg = EnvCfg(num_envs=args.num_envs, model=model, physics_hz=args.physics_hz, curriculum_steps=args.curriculum_iters * 32, episode_s=args.episode_s,
                 max_force=args.max_force, max_torque=args.max_torque, base_yaw=args.base_yaw, base_push=args.base_push,
                 artifacts_terminal=args.artifacts_terminal, jam_shaping=args.jam_shaping,
                 noise_scale=args.noise_scale, pose_noise=tuple(args.pose_noise), noise_level=tuple(args.noise_level),
                 domain=WIDE_DOMAIN if args.wide_dr else DomainCfg(),
                 turn_band_deg=tuple(args.turn_band) if args.turn_band else None)
    env = BncNewtonEnv(cfg)
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", "bnc_newton", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    tc = train_cfg(args)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"env": dataclasses.asdict(cfg), "train": tc, "args": vars(args)}, f, indent=1)
    runner = TimedRunner(env, tc, log_dir=log_dir, device="cuda:0")
    if args.resume:
        load_padded(runner, args.resume)
    if args.hours > 0:
        runner.deadline = time.time() + args.hours * 3600
    print(f"logging to {os.path.abspath(log_dir)}", flush=True)
    runner.learn(num_learning_iterations=args.max_iters, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
