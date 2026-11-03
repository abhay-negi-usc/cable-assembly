"""The Newton BNC env's rotation helpers (src/cable_assembly_sim/newton_bnc/env.py), on the CPU."""

import math

import pytest

torch = pytest.importorskip("torch")
env = pytest.importorskip("cable_assembly_sim.newton_bnc.env")


def test_euler_round_trip():
    e = (torch.rand(256, 3) * 2 - 1) * torch.tensor([1.2, 1.2, math.pi])
    R = env.euler_xyz_to_rot(e)
    assert torch.allclose(env.rot_to_euler_xyz(R), e, atol=1e-5)
    assert torch.allclose(R @ R.mT, torch.eye(3).expand(256, 3, 3), atol=1e-5)


def test_euler_is_intrinsic_xyz():
    """Newton's D6 (angular axes x, y, z) composes Rx(a) Ry(b) Rz(c); checked against eval_fk."""
    from scipy.spatial.transform import Rotation

    e = torch.tensor([[0.2, -0.3, 0.5]])
    ref = Rotation.from_euler("XYZ", e[0].numpy()).as_matrix()
    assert torch.allclose(env.euler_xyz_to_rot(e)[0], torch.tensor(ref, dtype=torch.float32), atol=1e-6)


def test_rotvec_round_trip():
    v = (torch.rand(256, 3) * 2 - 1) * 1.5
    assert torch.allclose(env.rot_to_rotvec(env.rotvec_to_rot(v)), v, atol=1e-4)
    small = torch.tensor([[1e-6, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert torch.allclose(env.rot_to_rotvec(env.rotvec_to_rot(small)), small, atol=1e-6)
