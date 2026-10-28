"""UR10e numpy kinematics: the model the simulated controller and the dry run stand on.

fk_links itself is held against Universal Robots' URDF in test_behaviors.py; these tests hold the
IK and the Jacobian against fk_links, so the whole chain is anchored to UR's published geometry.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from cable_assembly.robot.kinematics import ur10e as k


def _same_angles(a, b, atol=1e-7):
    return np.allclose([k._wrap(v) for v in np.asarray(a) - np.asarray(b)], 0.0, atol=atol)


def test_every_analytic_branch_reaches_the_pose_and_one_of_them_is_the_original():
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = k.fk(q)
        sols = k.ik_all(T)
        assert 1 <= len(sols) <= 8
        assert any(_same_angles(s, q) for s in sols), q
        for s in sols:
            assert np.abs(k.fk(s) - T).max() < 1e-9


def test_ik_returns_the_seed_itself_for_joints_anywhere_in_their_two_turn_range():
    """A wrist_3 prewound past 180 deg must come back where it is, not a full turn away -- the
    analytic angles are wrapped to (-pi, pi] and have to be shifted toward qnear."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.uniform(-2.0 * np.pi + 0.1, 2.0 * np.pi - 0.1, 6)
        q[2] = rng.uniform(-np.pi + 0.1, np.pi - 0.1)          # the elbow cannot exceed a half turn
        assert np.allclose(k.ik(k.fk(q), q), q, atol=1e-7), q


def test_ik_picks_the_branch_nearest_the_seed_not_the_first_one_found():
    rng = np.random.default_rng(2)
    q = rng.uniform(-np.pi, np.pi, 6)
    T = k.fk(q)
    for s in k.ik_all(T):
        near = s + rng.normal(0.0, 0.02, 6)                    # seed just beside another branch
        assert _same_angles(k.ik(T, near), s)


def test_out_of_reach_has_no_solution():
    T = np.eye(4)
    T[:3, 3] = [2.5, 0.0, 0.5]                                 # past the ~1.54 m true reach
    assert k.ik_all(T) == [] and k.ik(T, np.zeros(6)) is None


def test_wrist_singularity_keeps_the_seeds_wrist_3():
    """q5 = 0 aligns axes 4 and 6: only q4 + q6 is determined, so q6 stays at the seed's value."""
    q = np.array([0.3, -1.2, 1.4, -1.8, 0.0, 0.7])
    sol = k.ik(k.fk(q), q)
    assert sol is not None and np.abs(k.fk(sol) - k.fk(q)).max() < 1e-9
    assert abs(sol[5] - q[5]) < 1e-9


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    eps = 1e-7
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 6)
        J = k.jacobian(q)
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            Tp, Tm = k.fk(q + dq), k.fk(q - dq)
            assert np.allclose(J[:3, i], (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps), atol=1e-6)
            w = Rotation.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * eps)
            assert np.allclose(J[3:, i], w, atol=1e-6)


def test_collision_model_uses_this_chain_not_a_copy():
    from cable_assembly.robot import collision
    assert collision.fk_links is k.fk_links and collision.UR10E_D is k.UR10E_D
