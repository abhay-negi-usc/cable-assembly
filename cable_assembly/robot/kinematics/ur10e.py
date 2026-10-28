"""UR10e kinematics in ROS `base_link`: FK, geometric Jacobian, and the closed-form 8-branch IK.

THE DH TABLE LIVES HERE AND NOWHERE ELSE. collision.py builds its capsules from fk_links() below,
and the tests hold fk_links() against Universal Robots' URDF to nanometres -- two descriptions
from different upstream files, so a typo in either shows up as a disagreement, not a silent offset.

THE IK IS THE SAME KIND OF SOLVER THE CONTROLLER RUNS: analytic, all eight branches (shoulder
left/right x wrist up/down x elbow up/down), then the one nearest `qnear`. Every joint of a UR10e
turns +/-360 deg, so each analytic angle (wrapped to (-pi, pi]) is also shifted by whole turns
toward `qnear` -- without that, a wrist_3 prewound past 180 deg would come back a full turn away
from where it is, which is exactly the move the collar-clocking prewind exists to avoid. On the
real arm `URArm.ik` still asks the controller; this model is what the simulated controller and the
dry run use, so "nearest" is Euclidean in joint space -- close to, not guaranteed identical to,
the controller's choice between two near-equidistant branches.

Frames: the DH chain produces the UR `base` frame; BASE_LINK_FROM_UR_BASE (Rz(pi)) carries every
result into ROS `base_link`, the frame every config in this repo is measured in.
"""

import numpy as np

from ...transforms import BASE_LINK_FROM_UR_BASE, UR_JOINTS, inverse

JOINT_NAMES = tuple(UR_JOINTS)
# The joint that rolls the tool about its own axis -- what a connector or collar twist spends.
TOOL_ROLL_JOINT = 5
# Every UR10e joint turns +/-360 deg. The INSTALLATION safety configuration can narrow any of
# them, which only the controller knows -- see URArm.joints_ok.
JOINT_LIMITS = np.array([[-2.0 * np.pi, 2.0 * np.pi]] * 6)

# ---- The published Denavit-Hartenberg parameters (metres, radians) ---------------------------
# Standard DH: T_i = Rz(theta_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i). UR's own UR10e figures; the
# frame they produce is the UR `base`, NOT ROS `base_link`.
UR10E_A = (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0)
UR10E_D = (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655)
UR10E_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)

_UR_BASE_FROM_BASE_LINK = BASE_LINK_FROM_UR_BASE      # Rz(pi) is its own inverse


def dh_matrix(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_links(q):
    """Every joint frame plus tool0, in ROS `base_link`. Returns a (7, 4, 4) array.

    Index i is the frame at the OUTBOARD end of link i; index 6 is tool0. Index i is also the
    frame whose z axis joint i turns about (index 0 = the base), which is what jacobian() uses."""
    T = np.eye(4)
    out = []
    for i in range(6):
        T = T @ dh_matrix(float(q[i]), UR10E_D[i], UR10E_A[i], UR10E_ALPHA[i])
        out.append(T.copy())
    origins = [np.eye(4)] + out
    return np.array([BASE_LINK_FROM_UR_BASE @ T_i for T_i in origins])


def fk(q):
    """Joints -> tool0 in base_link (4x4)."""
    return fk_links(q)[6]


def jacobian(q):
    """Geometric Jacobian of tool0 in base_link, (6, 6): rows [v; w], v at tool0's origin."""
    frames = fk_links(q)
    p_e = frames[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z, p = frames[i][:3, 2], frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def _wrap(a):
    """Angle -> (-pi, pi]."""
    return float(np.pi - (np.pi - a) % (2.0 * np.pi))


def ik_all(T_base_tool0, wrist_singular_q6=0.0):
    """Every analytic solution for a tool0 pose in base_link: a list of up to eight (6,) arrays,
    each joint in (-pi, pi]. Empty when the pose is out of reach.

    At the wrist singularity (q5 ~ 0, axes 4 and 6 aligned) only q4 + q6 is determined; q6 is then
    taken as `wrist_singular_q6` (pass the current wrist_3 to keep it still) and q4 absorbs the
    rest. Each candidate is checked by FK before it is returned, so a numerically marginal branch
    is dropped rather than returned wrong."""
    d1, d4, d5, d6 = UR10E_D[0], UR10E_D[3], UR10E_D[4], UR10E_D[5]
    a2, a3 = UR10E_A[1], UR10E_A[2]
    T06 = _UR_BASE_FROM_BASE_LINK @ np.asarray(T_base_tool0, dtype=float)

    p05 = T06 @ np.array([0.0, 0.0, -d6, 1.0])
    r05 = np.hypot(p05[0], p05[1])
    if r05 < abs(d4):
        return []                                    # inside the shoulder offset cylinder
    phi = np.arctan2(p05[1], p05[0])
    psi = np.arccos(np.clip(d4 / r05, -1.0, 1.0))

    T60 = inverse(T06)
    sols = []
    for q1 in (phi + psi + np.pi / 2.0, phi - psi + np.pi / 2.0):
        s1, c1 = np.sin(q1), np.cos(q1)
        c5 = (T06[0, 3] * s1 - T06[1, 3] * c1 - d4) / d6
        if abs(c5) > 1.0 + 1e-9:
            continue
        for q5 in (np.arccos(np.clip(c5, -1.0, 1.0)), -np.arccos(np.clip(c5, -1.0, 1.0))):
            s5 = np.sin(q5)
            if abs(s5) < 1e-9:
                q6 = float(wrist_singular_q6)
            else:
                q6 = np.arctan2((-T60[1, 0] * s1 + T60[1, 1] * c1) / s5,
                                (T60[0, 0] * s1 - T60[0, 1] * c1) / s5)
            T01 = dh_matrix(q1, d1, 0.0, UR10E_ALPHA[0])
            T45 = dh_matrix(q5, d5, 0.0, UR10E_ALPHA[4])
            T56 = dh_matrix(q6, d6, 0.0, UR10E_ALPHA[5])
            T14 = inverse(T01) @ T06 @ inverse(T45 @ T56)
            # Joints 2-4 are three parallel axes: a planar 2-link problem for the wrist-1 origin.
            p13 = T14 @ np.array([0.0, -d4, 0.0, 1.0])
            r2 = p13[0] ** 2 + p13[1] ** 2
            c3 = (r2 - a2 ** 2 - a3 ** 2) / (2.0 * a2 * a3)
            if abs(c3) > 1.0 + 1e-9:
                continue
            for q3 in (np.arccos(np.clip(c3, -1.0, 1.0)), -np.arccos(np.clip(c3, -1.0, 1.0))):
                q2 = (np.arctan2(p13[1], p13[0])
                      - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3)))
                T13 = (dh_matrix(q2, 0.0, a2, UR10E_ALPHA[1])
                       @ dh_matrix(q3, 0.0, a3, UR10E_ALPHA[2]))
                T34 = inverse(T13) @ T14
                q4 = np.arctan2(T34[1, 0], T34[0, 0])
                q = np.array([_wrap(v) for v in (q1, q2, q3, q4, q5, q6)])
                T = fk(q)
                if (np.linalg.norm(T[:3, 3] - np.asarray(T_base_tool0)[:3, 3]) < 1e-6
                        and np.linalg.norm(T[:3, :3] - np.asarray(T_base_tool0)[:3, :3]) < 1e-6):
                    sols.append(q)
    return sols


def nearest_turn(q, qnear, limits=JOINT_LIMITS):
    """Shift each joint of `q` by whole turns to land closest to `qnear` inside `limits`."""
    out = np.array(q, dtype=float)
    for i in range(len(out)):
        lo, hi = limits[i]
        k = np.round((qnear[i] - out[i]) / (2.0 * np.pi))
        best = None
        for kk in (k - 1, k, k + 1):
            v = out[i] + 2.0 * np.pi * kk
            if lo - 1e-9 <= v <= hi + 1e-9 and (best is None or abs(v - qnear[i]) < abs(best - qnear[i])):
                best = v
        if best is None:
            return None
        out[i] = best
    return out


def ik(T_base_tool0, qnear, limits=JOINT_LIMITS):
    """The analytic solution nearest `qnear` (Euclidean in joint space, after shifting each joint
    by whole turns toward it), as a (6,) array -- or None if the pose is out of reach or no branch
    fits `limits`."""
    qnear = np.asarray(qnear, dtype=float)
    best, best_d = None, np.inf
    for q in ik_all(T_base_tool0, wrist_singular_q6=qnear[TOOL_ROLL_JOINT]):
        q = nearest_turn(q, qnear, limits)
        if q is None:
            continue
        d = float(np.linalg.norm(q - qnear))
        if d < best_d:
            best, best_d = q, d
    return best
