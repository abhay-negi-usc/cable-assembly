"""The Newton USB-A env (src/cable_assembly_sim/newton_usb): geometry and physics checks. GPU; needs the USDs from
scripts/build_usb.py (McMaster's CAD, gitignored) -- skipped without them.

    ./run_usb.sh python -m pytest tests/test_newton_usb.py
"""

import math

import numpy as np
import pytest

from cable_assembly_sim.newton_usb.model import PLUG_USD, SOCKET_USD, pad_clearance

pytestmark = pytest.mark.skipif(not (PLUG_USD.exists() and SOCKET_USD.exists()),
                                reason="run scripts/build_usb.py first (McMaster CAD)")


def test_seated_plug_does_not_interpenetrate():
    """Authored seated, the solid plug sits in the solid receptacle within meshing tolerance."""
    import trimesh
    from cable_assembly_sim.newton_usb.model import _read_meshes

    plug = _read_meshes(PLUG_USD, ("plug",))["plug"]
    rec = _read_meshes(SOCKET_USD, ("receptacle",))["receptacle"]
    P = trimesh.Trimesh(plug[0], plug[1].reshape(-1, 3), process=False)
    R = trimesh.Trimesh(rec[0], rec[1].reshape(-1, 3), process=False)
    assert P.is_watertight and R.is_watertight
    v = P.vertices[np.random.default_rng(0).choice(len(P.vertices), 4000, replace=False)]
    assert trimesh.proximity.signed_distance(R, v).max() < 5e-5  # m: 0.05 mm


def test_pads_clear_the_adapter_when_seated():
    assert pad_clearance() > 0.003
    assert pad_clearance(tilt_deg=20.0, grasp_z=0.020) < 0.0  # the BNC's grip point would hit it


@pytest.fixture(scope="module")
def env():
    from cable_assembly_sim.newton_usb.env import EnvCfg, UsbNewtonEnv

    return UsbNewtonEnv(EnvCfg(num_envs=32, noise_scale=0.0, randomize_domain=False))


def _push(env, steps):
    import torch

    for _ in range(steps):
        env.step(torch.zeros(env.num_envs, 6, device=env.device))


def _restart(env):
    import torch

    ids = torch.arange(env.num_envs, device=env.device)
    env._reset(ids)
    env._read()
    env._obs = env._observe(reset_ids=ids)


def test_zero_uncertainty_push_seats(env):
    """The plug where the robot believes it and the socket where it is seen: the base push alone seats it,
    centred, untilted, not through anything."""
    _restart(env)
    _push(env, 100)
    h, r, tilt, seated, _ = env.plug_state()
    assert seated.float().mean() == 1.0
    assert h.abs().max() < 2e-4 and r.max() < 4e-4 and tilt.max() < math.radians(1.0)
    assert env.artifact_counts.sum() == 0 and env.nonfinite_count == 0 and env.overflow_steps == 0


def test_detent_holds_a_light_push(env):
    """12 N of detent: a push capped at 8 N stalls in the springs; at 30 N it seats."""
    from cable_assembly_sim.newton_usb.env import DomainCfg, EnvCfg

    env.cfg.domain = DomainCfg(detent=(12.0, 12.0))
    try:
        for cap, seats in ((8.0, False), (30.0, True)):
            env.cfg.max_force = cap
            _restart(env)
            _push(env, 120)
            seated = env.plug_state()[3]
            assert bool(seated.all()) == seats and bool(seated.any()) == seats
    finally:
        env.cfg.domain, env.cfg.max_force = DomainCfg(), EnvCfg.max_force


def test_grasp_slip_is_stable_and_bounded():
    """The plug slipping in a 20 N grip (EnvCfg.slip), 256 worlds with friction, stiffness, detent and grip
    randomised, pushed with the 20 N cap: plugs seat, the in-hand rotation stays a turn about the closing axis,
    and nothing flies off. (Two earlier versions blew up -- composing the in-hand rotation by matrix products,
    and writing the targets mid-physics-step -- and only with many worlds slipping at once.)"""
    import torch
    from cable_assembly_sim.newton_usb.env import EnvCfg, UsbNewtonEnv, rot_x

    torch.manual_seed(0)
    env = UsbNewtonEnv(EnvCfg(num_envs=256, noise_scale=0.0, slip=True, max_force=20.0))
    _restart(env)
    void = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(80):
        env.step(torch.zeros(env.num_envs, 6, device=env.device))
        void |= env.voided
    seated = env.plug_state()[3] & ~void
    assert void.float().mean() < 0.02, f"{void.float().mean():.3f} of worlds voided"
    assert seated.float().mean() > 0.95
    assert env.slip_total.max() > 5e-4, "the 20 N push never slipped a weak grip: is slip off?"
    assert torch.allclose(env.R_E, rot_x(env.e_rot), atol=1e-5)
    assert env.p_E[:, 0].abs().max() == 0.0
