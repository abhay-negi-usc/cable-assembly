"""Regression tests for the Newton BNC env's contacts (src/cable_assembly_sim/newton_bnc): the parts must
not pass through each other under the forces the policy can apply. Speed tuning broke this twice
(collision every 2nd substep; a contact buffer too small for seated plugs) without any error. GPU only,
~1-2 min:

    ./run.sh python -m pytest tests/test_newton_bnc_contacts.py
"""

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
envmod = pytest.importorskip("cable_assembly_sim.newton_bnc.env")
newton = pytest.importorskip("newton")
from cable_assembly_sim.newton_bnc.model import ModelCfg  # noqa: E402

ZERO = envmod.ErrorCfg(**{f.name: 0.0 for f in dataclasses.fields(envmod.ErrorCfg)})


def make_env(n, collar_deg, start_height, max_force, model=None, **kw):
    """n worlds, no estimation error, nominal friction, the collar at collar_deg off its slot entries.
    Artifacts are not voided: the tests look at them."""
    cfg = envmod.EnvCfg(num_envs=n, errors=ZERO, randomize_domain=False, void_artifacts=False,
                        max_force=max_force, start_height=(start_height, start_height), **kw)
    if model is not None:
        cfg.model = model
    env = envmod.BncNewtonEnv(cfg)
    q = env._q()
    yaw = torch.atan2(env.R_true[:, 1, 0], env.R_true[:, 0, 0])
    q[:, env.bm.turn_q] = envmod.wrap(math.radians(collar_deg) - yaw)
    newton.eval_fk(env.model, env.s0.joint_q, env.s0.joint_qd, env.s0)
    env._read()
    return env


def run(env, action, steps, check_finite=True):
    """Hold one action; return the worst artifact flags and the plug's min height, max off-axis."""
    a = torch.tensor(action, device=env.device).expand(env.num_envs, 6)
    art = torch.zeros(env.num_envs, 3, dtype=torch.bool, device=env.device)
    h_min = torch.full((env.num_envs,), math.inf, device=env.device)
    r_max = torch.zeros(env.num_envs, device=env.device)
    for _ in range(steps):
        env.step(a)
        assert not (check_finite and env.voided.any()), "non-finite physics"
        h, r, *_ = env.plug_state()
        art |= env._artifacts()
        h_min, r_max = torch.minimum(h_min, h), torch.maximum(r_max, r)
    return art, h_min, r_max


@torch.inference_mode()
def test_pins_hold_a_misclocked_collar():
    """Collar 45 deg off its slot, pushed straight down at 50 N (over the env's cap): it must stay on
    the pins. Colliding every 2nd substep let 38-55% of worlds through at 25 N (52 of 64 here). About
    1% still pass with the fixed settings (contact softness; voided in training), so up to 2 may."""
    env = make_env(64, 45.0, 0.010, 50.0, start_xy=0.0, start_yaw_deg=0.0)
    art, h_min, _ = run(env, [0, 0, -1, 0, 0, 0], 60)
    assert int(art[:, 0].sum()) <= 2, f"pins through the collar in {int(art[:, 0].sum())} of 64 worlds"
    assert int((h_min < 0.0050).sum()) <= 2, f"{int((h_min < 0.0050).sum())} plugs below 5 mm on a misclocked collar"


@torch.inference_mode()
def test_seated_plugs_stay_on_axis_and_the_buffer_holds():
    """Plugs at the slot floor, pushed down and sideways at the env's force cap: the sleeve stays on the
    barrel, the body behind the collar's stop, and the shared contact buffer does not overflow (256
    contacts a world overflowed with seated plugs, dropping contacts in random worlds)."""
    env = make_env(256, 0.0, 0.0008, envmod.EnvCfg.max_force, start_xy=0.0, start_yaw_deg=0.0)
    art, h_min, r_max = run(env, [0.6, 0.3, -1, 0, 0, 0], 60)
    assert int(env.overflow_steps) == 0, "contact buffer overflowed"
    assert not art.any(), f"artifacts (pass-through, off-axis, overrun): {art.sum(0).tolist()}"
    assert r_max.max() < 0.0012, f"seated plug {r_max.max() * 1e3:.2f} mm off-axis"


@torch.inference_mode()
def test_overflow_is_counted():
    """The overflow counter itself: a buffer far too small must register."""
    env = make_env(64, 0.0, 0.0008, envmod.EnvCfg.max_force, model=ModelCfg(contacts_per_world=16),
                   start_xy=0.0, start_yaw_deg=0.0)
    run(env, [0, 0, -1, 0, 0, 0], 5, check_finite=False)  # dropped contacts may blow up: not the point here
    assert int(env.overflow_steps) > 0


@torch.inference_mode()
def test_history_advances_when_other_worlds_reset():
    """A reset in one world must not stall the others' observation history. It did until 2026-10-30:
    with staggered episodes (a reset nearly every step) every history stayed frozen near its reset
    frame in training, while evaluation (all worlds reset together) saw it move."""
    env = make_env(8, 0.0, 0.015, envmod.EnvCfg.max_force)
    H = env.cfg.history
    a = torch.zeros(8, 6, device=env.device)
    a[:, 2] = -0.5
    prev = env.get_observations()["policy"][:, : H * 15].view(8, H, 15).clone()
    for _ in range(4):
        env.episode_length_buf[0] = env.max_episode_length  # world 0 times out on this step
        obs = env.step(a)[0]["policy"][:, : H * 15].view(8, H, 15)
        assert torch.equal(obs[1:, :-1], prev[1:, 1:]), "history did not advance in the worlds that did not reset"
        assert not torch.equal(obs[1:, -1], prev[1:, -1]), "no new frame"
        assert torch.equal(obs[0], obs[0, :1].expand(H, -1)), "the reset world's history is not its first frame"
        prev = obs.clone()


@torch.inference_mode()
def test_no_mm_scale_interference_against_the_cad():
    """Plugs driven onto and beside the barrel (6 mm target offsets, 12 deg tilt, 5 deg in-hand turn) by
    the push-and-twist base controller: no CAD part of the plug -- colliders or not -- may sit more than
    0.3 mm inside the socket. The speed-tuned model reached 0.98 mm (sphere stand-ins for the fingers and
    insulator, coarse contact reduction, soft contacts at 500 Hz); the accurate one ~0.13 mm."""
    o3d = pytest.importorskip("open3d")
    import numpy as np
    from pxr import Usd, UsdGeom

    from cable_assembly_sim.newton_bnc.model import PLUG_USD, SOCKET_USD

    err = dataclasses.replace(ZERO, target_xy=0.006, target_tilt_deg=12.0, in_hand_rot_deg=5.0)
    torch.manual_seed(3)
    env = envmod.BncNewtonEnv(envmod.EnvCfg(num_envs=32, episode_s=3.0, base_yaw=0.8, base_push=0.6,
                                            void_artifacts=False, errors=err))
    a = torch.zeros(32, 6, device=env.device)
    for _ in range(env.max_episode_length - 1):
        env.step(a)
    bq = env.s0.body_q.numpy().reshape(32, env.bm.bodies_per_world, 7)

    def read_all(path):
        st = Usd.Stage.Open(str(path))
        out = {}
        for prim in st.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(prim)
                X = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                out[prim.GetName()] = (np.array(m.GetPointsAttr().Get()) @ X[:3, :3].T + X[:3, 3],
                                       np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3))
        return out

    def mesh(p, f):
        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p.astype(np.float64)), o3d.utility.Vector3iVector(f))

    plug, sock = read_all(PLUG_USD), read_all(SOCKET_USD)
    S = np.concatenate([np.asarray(mesh(*sock[k]).sample_points_uniformly(60000).points)
                        for k in ("socket_0", "socket_1")]).astype(np.float32)
    worst = 0.0
    for w in range(32):
        for part in ("fingers", "insulator", "disc", "collar"):
            b = bq[w, env.bm.collar_body if part == "collar" else env.bm.plug_body]
            R = o3d.geometry.get_rotation_matrix_from_quaternion([b[6], b[3], b[4], b[5]])
            p, f = plug[part]
            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh(p @ R.T + b[:3], f)))
            sd = sc.compute_signed_distance(o3d.core.Tensor(S)).numpy()
            worst = max(worst, -float(np.sort(sd)[3]))  # the 4th deepest sample: not single-point mesh noise
    assert worst < 0.0003, f"plug parts {worst * 1e3:.2f} mm inside the socket"
