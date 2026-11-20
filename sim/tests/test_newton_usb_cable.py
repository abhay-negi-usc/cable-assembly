"""The cable-held USB-A env (src/cable_assembly_sim/newton_usb_cable): grasp geometry, the cable-to-plug weld,
the belief at zero uncertainty, and a short push. GPU; needs the USDs from scripts/build_usb.py (McMaster's CAD,
gitignored) -- skipped without them.

    ./run_usb.sh python -m pytest tests/test_newton_usb_cable.py
"""

import pytest
import torch

from cable_assembly_sim.newton_usb.model import PLUG_DY, PLUG_USD, SOCKET_USD

pytestmark = pytest.mark.skipif(not (PLUG_USD.exists() and SOCKET_USD.exists()),
                                reason="run scripts/build_usb.py first (McMaster CAD)")


@pytest.fixture(scope="module")
def env():
    from cable_assembly_sim.newton_usb_cable.env import CableEnvCfg, UsbCableEnv

    return UsbCableEnv(CableEnvCfg(num_envs=32, episode_s=4.0, noise_scale=1.0))


def _weld_gap(env):
    from cable_assembly_sim.newton_usb.env import quat_to_rot
    from cable_assembly_sim.newton_usb_cable.model import OVERMOLD_BACK_Z

    um, L = env.um, env.cfg.model.link_len
    bq = env._bq()
    R, p = quat_to_rot(bq[:, um.link_bodies[-1], 3:7]), bq[:, um.link_bodies[-1], 0:3]
    Rp, pp = quat_to_rot(bq[:, um.plug_body, 3:7]), bq[:, um.plug_body, 0:3]
    a = p + R @ torch.tensor([0.0, 0.0, -L], device=p.device)
    b = pp + Rp @ torch.tensor([0.0, PLUG_DY, OVERMOLD_BACK_Z], device=p.device)
    return (a - b).norm(dim=-1)


@torch.inference_mode()
def test_reset_is_consistent_with_the_weld(env):
    """Every world's grasp (its own anchor for the cable) puts the cable's end on the overmold at reset."""
    env._reset(torch.arange(env.num_envs, device=env.device))
    env._read()
    assert _weld_gap(env).max() < 1e-6
    env.step(torch.zeros(env.num_envs, 6, device=env.device))
    assert _weld_gap(env).max() < 1e-4  # m: the weld holds within 0.1 mm


@torch.inference_mode()
def test_zero_uncertainty_belief_is_exact_but_grasp_varies(env):
    env.cfg.noise_scale = 0.0
    env._reset(torch.arange(env.num_envs, device=env.device))
    env._read()
    assert (env.p_true - env.p_bel).norm(dim=-1).max() < 1e-5
    assert env.err_ctrl.abs().amax(0).min() > 0.3  # every control-error DOF is drawn
    assert env.err[:, 0:4].abs().max() == 0.0  # no knowledge error
    env.cfg.noise_scale = 1.0


@torch.inference_mode()
def test_push_seats_at_zero_uncertainty(env):
    env.cfg.noise_scale, env.cfg.randomize_domain = 0.0, False
    env._reset(torch.arange(env.num_envs, device=env.device))
    env._read()
    zero = torch.zeros(env.num_envs, 6, device=env.device)
    for _ in range(env.max_episode_length - 2):
        env.step(zero)
    seated = env.plug_state()[3] & ~env.voided
    assert seated.float().mean() > 0.95
    assert env.nonfinite_count == 0
    env.cfg.noise_scale, env.cfg.randomize_domain = 1.0, True
