"""The robot's copy of the env (cable_assembly/skills/rl_insertion.py: TaskSpace, Policy) against the env
itself: the same observation and the same clamped target, step for step, while the exported policy
drives the plug in. With no estimation error and no sensor noise the two must agree exactly -- the
robot runs what the policy was trained on. GPU, ~1.5 min:

    ./run.sh python -m pytest tests/test_rl_deploy_parity.py
    PARITY_POLICY=/abs/path/policy.npz ./run.sh python -m pytest tests/test_rl_deploy_parity.py   # another export
"""

import dataclasses
import json
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
envmod = pytest.importorskip("cable_assembly_sim.newton_bnc.env")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
RL = pytest.importorskip("cable_assembly.skills.rl_insertion")

POLICY = os.environ.get("PARITY_POLICY") or os.path.join(os.path.dirname(__file__), "..", "..", "configs", "policies",
                                                         "bnc_insert_v16_1100.npz")
if not os.path.isfile(POLICY):
    pytest.skip("no exported policy (scripts/newton_bnc_export.py)", allow_module_level=True)
ZERO = envmod.ErrorCfg(**{f.name: 0.0 for f in dataclasses.fields(envmod.ErrorCfg)})


def _T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def test_the_robots_observation_and_target_are_the_envs():
    spec = RL.Policy(POLICY).spec
    e = spec["env"]
    n = 4
    cfg = envmod.EnvCfg(num_envs=n, errors=ZERO, randomize_domain=False, wrench_noise=(0.0, 0.0), wrench_bias=(0.0, 0.0),
                        episode_s=e["episode_s"], base_yaw=e["base_yaw"], base_push=e["base_push"],
                        max_torque=e["max_torque"], max_force=e["max_force"],
                        turn_band_deg=tuple(e["turn_band_deg"]) if e.get("turn_band_deg") else None,
                        turn_margin_deg=e.get("turn_margin_deg", 20.0))
    env = envmod.BncNewtonEnv(cfg)
    # As the eval starts an episode: the env's very first observation (at construction) has no history
    # and a velocity from an identity pose -- only ever seen once in training.
    ids = torch.arange(n, device=env.device)
    env._reset(ids)
    env._read()
    env._obs = env._observe(reset_ids=ids)
    pols = [RL.Policy(POLICY) for _ in range(n)]
    tss = [RL.TaskSpace(spec) for _ in range(n)]
    S = [_T(env.R_obs[i].cpu().numpy(), env.p_obs[i].cpu().numpy()) for i in range(n)]

    def state(i):
        P = _T(env.R_bel[i].cpu().numpy(), env.p_bel[i].cpu().numpy())
        w = env._wrench()[i].cpu().numpy()
        Ro = S[i][:3, :3].T
        return P, np.concatenate([Ro @ w[:3], Ro @ w[3:]])

    for i in range(n):
        tss[i].reset(S[i], state(i)[0])
    worst_obs = worst_p = worst_r = 0.0
    obs_env = env.get_observations()["policy"].cpu().numpy()
    seated = 0
    for k in range(200):
        acts = []
        for i in range(n):
            P, w = state(i)
            o = tss[i].observe(P, w)
            d = np.abs(o - obs_env[i])
            if d.max() > 1e-3 and os.environ.get('PARITY_DEBUG'):
                j = int(np.argmax(d))
                print('step', k, 'world', i, 'index', j, 'app', o[j], 'env', obs_env[i][j], 'bad idx', np.nonzero(d > 1e-3)[0].tolist())
            worst_obs = max(worst_obs, float(d.max()))
            acts.append(pols[i](o))
        targets = [tss[i].apply(acts[i]) for i in range(n)]
        obs, _, dones, _ = env.step(torch.tensor(np.array(acts), dtype=torch.float32, device=env.device))
        assert not dones.any()
        # the env's target after its own clamping (set in step, before the simulation)
        for i in range(n):
            worst_p = max(worst_p, float(np.max(np.abs(env.p_tgt[i].cpu().numpy() - targets[i][:3, 3]))))
            worst_r = max(worst_r, float(np.linalg.norm(RL.rotvec(env.R_tgt[i].cpu().numpy() @ targets[i][:3, :3].T))))
        obs_env = obs["policy"].cpu().numpy()
    seated = int((env.first_seated >= 0).sum())
    assert worst_obs < 1e-3, worst_obs        # float32 env vs float64 here
    assert worst_p < 1e-6 and worst_r < 1e-5, (worst_p, worst_r)
    assert seated == n                        # no error, nominal domain: every plug seats within 10 s
