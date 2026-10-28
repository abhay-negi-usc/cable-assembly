"""KUKA iiwa 14 numpy kinematics. On this arm the model IS the IK the robot moves by (FRI takes
joint commands), so it is held to the upstream URDF, and the IK to the FK, closely.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from cable_assembly.robot.kinematics import iiwa14 as k

URDF = os.path.join(os.path.dirname(k.__file__), '..', 'description', 'iiwa14', 'iiwa14.urdf')
LIM = k.JOINT_LIMITS


def _urdf_chain():
    """[(origin 4x4, axis)] for A1..A7 plus the fixed tool0 joint, read from the vendored URDF."""
    joints = {j.get('name'): j for j in ET.parse(URDF).getroot().findall('joint')}
    chain = []
    for name in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'link_7-tool0']:
        j = joints[name]
        o = j.find('origin')
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in o.get('rpy').split()]).as_matrix()
        T[:3, 3] = [float(v) for v in o.get('xyz').split()]
        ax = j.find('axis')
        chain.append((T, None if ax is None else np.array([float(v) for v in ax.get('xyz').split()])))
    return chain


def _urdf_fk(chain, q):
    T, out = np.eye(4), []
    for i, (origin, axis) in enumerate(chain):
        T = T @ origin
        if axis is not None:
            R = np.eye(4)
            R[:3, :3] = Rotation.from_rotvec(axis * q[i]).as_matrix()
            T = T @ R
        out.append(T.copy())
    return out


def _random_q(rng, shrink=0.95):
    return rng.uniform(LIM[:, 0] * shrink, LIM[:, 1] * shrink)


def test_fk_is_the_upstream_urdf_lateral_offsets_and_all():
    """The model drops the URDF's +/-10, 10.5 and 70.7 mm lateral offsets on the claim that each
    pair cancels along the joint axis between them. This is the check of that claim."""
    chain = _urdf_chain()
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = _random_q(rng, 1.0)
        theirs = _urdf_fk(chain, q)
        mine = k.fk_links(q)
        for i in range(7):                      # each link_i lies on the same joint axis line
            assert np.allclose(mine[i + 1][:3, :3], theirs[i][:3, :3], atol=1e-12)
        assert np.abs(mine[7] - theirs[6]).max() < 1e-12, 'link_7 itself coincides exactly'
        assert np.abs(mine[8] - theirs[7]).max() < 1e-12, 'and so does tool0'


def test_limits_match_the_urdf():
    joints = {j.get('name'): j for j in ET.parse(URDF).getroot().findall('joint')}
    for i, name in enumerate(k.JOINT_NAMES):
        lim = joints[name].find('limit')
        assert np.isclose(float(lim.get('lower')), LIM[i, 0])
        assert np.isclose(float(lim.get('upper')), LIM[i, 1])
        assert np.isclose(float(lim.get('velocity')), k.VELOCITY_LIMITS[i])


def test_geometry_at_known_poses():
    T = k.fk(np.zeros(7))
    assert np.allclose(T[:3, 3], [0.0, 0.0, 1.306]) and np.allclose(T[:3, :3], np.eye(3))
    T = k.fk(k.HOME)
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0]), 'HOME points the tool straight down'
    assert np.allclose(T[:3, 3], [0.556, 0.0, 0.398], atol=1e-3)
    assert k.within(k.HOME)


def test_every_branch_reaches_the_pose_and_one_is_the_original():
    rng = np.random.default_rng(1)
    for _ in range(500):
        q = _random_q(rng)
        T = k.fk(q)
        sols = k.ik_all(T, k.arm_angle(q), q)
        assert len(sols) == 8, 'shoulder x elbow x wrist, away from singularities'
        assert any(np.allclose(s, q, atol=1e-7) for s in sols), q
        for s in sols:
            assert np.abs(k.fk(s) - T).max() < 1e-9
            assert abs(k.arm_angle(s) - k.arm_angle(q)) < 1e-7, 'every branch keeps the arm angle'


def test_ik_returns_the_seed_itself():
    rng = np.random.default_rng(2)
    for _ in range(500):
        q = _random_q(rng)
        assert np.allclose(k.ik(k.fk(q), q), q, atol=1e-7), q


def test_ik_keeps_the_arm_angle_along_a_path():
    """A small Cartesian step must not swing the elbow: the solution keeps qnear's arm angle."""
    q = k.HOME.copy()
    psi0 = k.arm_angle(q)
    for _ in range(50):
        T = k.fk(q)
        T[:3, 3] += [0.002, 0.001, -0.001]
        q = k.ik(T, q)
        assert q is not None and abs(k.arm_angle(q) - psi0) < 1e-9


def test_ik_searches_the_arm_angle_when_the_seeds_breaks_a_limit():
    """At qnear's own arm angle every branch may need a joint past its limit; the solver then
    takes the nearest arm angle that fits rather than giving up on a reachable pose."""
    q = np.radians([0.0, 60.0, 0.0, -60.0, 0.0, 60.0, 0.0])
    T = k.fk(q)
    tight = LIM.copy()
    tight[2] = np.radians([20.0, 60.0])          # forbid q3 near 0 -> qnear's arm angle won't do
    sol = k.ik(T, q, tight)
    assert sol is not None and k.within(sol, tight)
    assert np.abs(k.fk(sol) - T).max() < 1e-9
    assert abs(k.arm_angle(sol) - k.arm_angle(q)) > 1e-3, 'it had to move the elbow'


def test_singularities_keep_the_seeds_free_joint():
    seed = np.radians([10.0, 40.0, 25.0, -70.0, 15.0, 0.0, 33.0])   # wrist straight: q6 = 0
    sol = k.ik(k.fk(seed), seed)
    assert sol is not None and np.abs(k.fk(sol) - k.fk(seed)).max() < 1e-9
    assert abs(sol[6] - seed[6]) < 1e-9, 'q6 = 0: A7 stays where it is, A5 takes the rest'

    seed = np.radians([10.0, 40.0, 25.0, 0.0, 15.0, 50.0, 33.0])    # elbow straight: q4 = 0
    sols = k.ik_all(k.fk(seed), 0.0, seed)
    assert sols and all(np.abs(k.fk(s) - k.fk(seed)).max() < 1e-9 for s in sols)
    assert k.arm_angle(seed) is None, 'a straight arm has no arm angle'
    sol = k.ik(k.fk(seed), seed)
    assert sol is not None and np.abs(k.fk(sol) - k.fk(seed)).max() < 1e-9


def test_out_of_reach_has_no_solution():
    T = np.eye(4)
    T[:3, 3] = [1.4, 0.0, 0.36]
    assert k.ik_all(T, 0.0) == [] and k.ik(T, k.HOME) is None


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    eps = 1e-7
    for _ in range(20):
        q = _random_q(rng)
        J = k.jacobian(q)
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            Tp, Tm = k.fk(q + dq), k.fk(q - dq)
            assert np.allclose(J[:3, i], (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps), atol=1e-6)
            w = Rotation.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * eps)
            assert np.allclose(J[3:, i], w, atol=1e-6)
