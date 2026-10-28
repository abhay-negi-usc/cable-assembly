"""KUKA LBR iiwa 14 R820 kinematics: FK, geometric Jacobian, and the closed-form IK parameterised
by the ARM ANGLE.

THE CHAIN is read off lbr_description's iiwa14.urdf (lbr-stack, Apache-2.0; the kinematic part is
vendored as description/iiwa14/iiwa14.urdf and the tests hold this module against it). Every link
frame is aligned with the base at q = 0 and the joint axes alternate z / y. The URDF's lateral
offsets (A2 -/+10 mm, A4 +/-10.5 mm, A6 -/+70.7 mm) each lie ALONG the y axis of the joint they
straddle, so they cancel and the arm is exactly

    a SPHERICAL SHOULDER   A1-A3 axes meet 0.360 m above the base,
    an ELBOW               A4, 0.420 m further along the upper arm,
    a SPHERICAL WRIST      A5-A7 axes meet 0.400 m beyond the elbow,

    R_0_7 = Rz(q1) Ry(q2) Rz(q3) Ry(-q4) Rz(q5) Ry(q6) Rz(q7)        (A4 turns about -y)

THE FLANGE (tool0 here) is the mounting face of the media flange: link_7 sits 0.091 m beyond the
wrist point and the media flange adds 0.035 m, so tool0 is 0.126 m out -- the d7 of KUKA's own
iiwa 14 R820 table. Bring-up must confirm this against the flange pose Sunrise reports: every
tool frame (hand-eye, fingertip) is measured FROM tool0, so a wrong d7 is a rigid shift of all
of them that a hand-eye calibration would silently absorb on one arm angle and not another.

REDUNDANCY. Seven joints for a six-DOF pose leave one free parameter, the arm angle psi: the
rotation of the elbow about the shoulder-wrist line. At a fixed psi a pose has up to eight
solutions -- shoulder (sign of q2) x elbow (sign of q4) x wrist (sign of q6), the same eight-branch
structure as the UR10e. psi here is GEOMETRIC (measured from the vertical plane through the
shoulder-wrist line), not KUKA's q3 = 0 reference plane; it is only ever compared with itself.

ik() keeps qnear's arm angle and picks the branch nearest qnear, so successive solutions along a
path stay on one configuration and the elbow does not swing. Only when that solution breaks a
joint limit does it search psi outward from qnear's -- the nearest arm angle that fits.

Joint limits are the mechanical ones (+/-170, 120, 170, 120, 170, 120, 175 deg). Unlike the UR,
no joint turns past a half turn, so there is no whole-turn ambiguity to resolve -- and the tool
roll (A7) has 350 deg of travel, not the UR wrist_3's 720.
"""

import numpy as np

NAME = 'iiwa14'
JOINT_NAMES = ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')
DOF = 7
# The joint that rolls the tool about its own axis, and the one that tips the wrist across it.
TOOL_ROLL_JOINT = 6
JOINT_ROLES = {'tool_roll': 6, 'wrist_nod': 5}
JOINT_LIMITS = np.radians([[-170.0, 170.0], [-120.0, 120.0], [-170.0, 170.0], [-120.0, 120.0],
                           [-170.0, 170.0], [-120.0, 120.0], [-175.0, 175.0]])
# Rated joint speeds (URDF / KUKA data sheet), rad/s.
VELOCITY_LIMITS = np.radians([85.0, 85.0, 100.0, 75.0, 130.0, 135.0, 135.0])
# A working pose: tool pointing straight down, 0.56 m in front of the base and 0.40 m above it.
# q2 - q4 + q6 = 180 deg is what points the tool down when q1 = q3 = q5 = 0.
HOME = np.radians([0.0, 30.0, 0.0, -90.0, 0.0, 60.0, 0.0])

D_BS = 0.360          # base -> shoulder (A2 axis)
D_SE = 0.420          # shoulder -> elbow (A4 axis)
D_EW = 0.400          # elbow -> wrist (A6 axis)
D_WF = 0.126          # wrist -> tool0 (media-flange face)

# Per joint: the offset from the previous frame to this joint (along the previous frame's z, the
# URDF's lateral offsets dropped -- see above) and the axis it turns about, in its own frame.
# The last entry is the fixed tool0 frame.
_Z_OFFSETS = (0.1475, 0.2125, 0.2280, 0.1920, 0.2075, 0.1925, 0.0910, 0.0350)
_AXES = ('z', 'y', 'z', '-y', 'z', 'y', 'z')


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


_STRAIGHT = 1e-7      # |q4| below this is a straight arm (see ik_all)

_ROT = {'z': _rz, 'y': _ry, '-y': lambda a: _ry(-a)}
_AXIS_VEC = {'z': np.array([0.0, 0.0, 1.0]), 'y': np.array([0.0, 1.0, 0.0]),
             '-y': np.array([0.0, -1.0, 0.0])}


def fk_links(q):
    """Every link frame plus tool0, in the robot base frame. Returns a (9, 4, 4) array.

    Index 0 is the base, index i (1..7) is link_i -- the frame joint Ai turns -- and index 8 is
    tool0. Origins match the URDF's link frames up to the cancelling lateral offsets, i.e. they
    lie on the same joint axes."""
    out = [np.eye(4)]
    T = np.eye(4)
    for i in range(7):
        step = np.eye(4)
        step[2, 3] = _Z_OFFSETS[i]
        step[:3, :3] = _ROT[_AXES[i]](float(q[i]))
        T = T @ step
        out.append(T.copy())
    tool = np.eye(4)
    tool[2, 3] = _Z_OFFSETS[7]
    out.append(T @ tool)
    return np.array(out)


def fk(q):
    """Joints -> tool0 in the base frame (4x4)."""
    return fk_links(q)[8]


def jacobian(q):
    """Geometric Jacobian of tool0 in the base frame, (6, 7): rows [v; w], v at tool0's origin."""
    frames = fk_links(q)
    p_e = frames[8][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        R, p = frames[i + 1][:3, :3], frames[i + 1][:3, 3]
        z = R @ _AXIS_VEC[_AXES[i]]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def _wrap(a):
    """Angle -> (-pi, pi]."""
    return float(np.pi - (np.pi - a) % (2.0 * np.pi))


def _reference_axes(u):
    """Two unit vectors spanning the plane normal to the shoulder-wrist direction `u`: v0 points
    'up' (the base z axis projected into the plane) and w0 = u x v0. psi = 0 puts the elbow on v0.
    When the wrist is straight above or below the shoulder, base x stands in for z."""
    for ref in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
        v = ref - np.dot(ref, u) * u
        n = np.linalg.norm(v)
        if n > 1e-6:
            v = v / n
            return v, np.cross(u, v)
    raise ValueError('degenerate shoulder-wrist direction')


def _shoulder_wrist(T_base_tool0):
    T = np.asarray(T_base_tool0, dtype=float)
    S = np.array([0.0, 0.0, D_BS])
    W = T[:3, 3] - D_WF * T[:3, 2]
    return S, W


def arm_angle(q):
    """The arm angle psi of a configuration (rad), or None when the arm is straight (q4 ~ 0) and
    the elbow sits on the shoulder-wrist line, where psi is undefined."""
    frames = fk_links(q)
    S, E, W = frames[2][:3, 3], frames[4][:3, 3], frames[6][:3, 3]
    sw = W - S
    u = sw / np.linalg.norm(sw)
    e = (E - S) - np.dot(E - S, u) * u
    if np.linalg.norm(e) < 1e-6:
        return None
    v0, w0 = _reference_axes(u)
    return float(np.arctan2(np.dot(e, w0), np.dot(e, v0)))


def _zyz(R, c_singular):
    """(a, b, c) with Rz(a) Ry(b) Rz(c) = R: two solutions (b > 0 and b < 0), or one at b = 0,
    where only a + c is determined and c is taken as `c_singular`."""
    sb = np.hypot(R[0, 2], R[1, 2])
    if sb < 1e-9:
        if R[2, 2] > 0.0:
            c = float(c_singular)
            return [(np.arctan2(R[1, 0], R[0, 0]) - c, 0.0, c)]
        c = float(c_singular)                          # b = pi: outside every iiwa limit anyway
        return [(np.arctan2(-R[0, 1], R[1, 1]) + c, np.pi, c)]
    b = np.arctan2(sb, R[2, 2])
    a = np.arctan2(R[1, 2], R[0, 2])
    c = np.arctan2(R[2, 1], -R[2, 0])
    return [(a, b, c), (a + np.pi, -b, c + np.pi)]


def ik_all(T_base_tool0, psi, seed=None):
    """Every analytic solution for a tool0 pose at arm angle `psi`: a list of up to eight (7,)
    arrays, each joint in (-pi, pi], NOT filtered by joint limits. Empty when out of reach.

    At the three singularities only a sum of two joint angles is determined, and the one that
    would otherwise be arbitrary is taken from `seed` so it stays still: q1 at the shoulder (q2 ~ 0),
    q3 at the elbow (q4 ~ 0, arm straight), q7 at the wrist (q6 ~ 0). Each candidate is checked by
    FK before it is returned."""
    seed = np.zeros(7) if seed is None else np.asarray(seed, dtype=float)
    T = np.asarray(T_base_tool0, dtype=float)
    R07 = T[:3, :3]
    S, W = _shoulder_wrist(T)
    sw = W - S
    L = float(np.linalg.norm(sw))
    if L < 1e-9:
        return []
    c4 = (L * L - D_SE ** 2 - D_EW ** 2) / (2.0 * D_SE * D_EW)
    if abs(c4) > 1.0 + 1e-9:
        return []
    q4_abs = float(np.arccos(np.clip(c4, -1.0, 1.0)))
    u = sw / L
    # A STRAIGHT ARM is snapped, not solved: arccos near 1 turns rounding in L into a spurious
    # q4 of ~1e-8 rad, and q3 would then be read off a vector that is all noise. Below
    # _STRAIGHT the elbow is put exactly on the shoulder-wrist line and q3 kept at the seed's.
    straight = q4_abs < _STRAIGHT
    if straight:
        q4_abs, E = 0.0, S + D_SE * u
    else:
        v0, w0 = _reference_axes(u)
        ca = (D_SE ** 2 + L * L - D_EW ** 2) / (2.0 * D_SE * L)
        sa = np.sqrt(max(0.0, 1.0 - ca * ca))
        E = S + D_SE * (ca * u + sa * (np.cos(psi) * v0 + np.sin(psi) * w0))
    e1 = (E - S) / D_SE

    rxy = np.hypot(e1[0], e1[1])
    if rxy < 1e-9:
        shoulders = [(seed[0], 0.0 if e1[2] > 0.0 else np.pi)]
    else:
        q2 = np.arctan2(rxy, e1[2])
        q1 = np.arctan2(e1[1], e1[0])
        shoulders = [(q1, q2), (q1 + np.pi, -q2)]

    elbows = [0.0] if straight else [q4_abs, -q4_abs]
    sols = []
    for q1, q2 in shoulders:
        R02 = _rz(q1) @ _ry(q2)
        ew = R02.T @ (W - E)              # = Rz(q3) Ry(-q4) [0, 0, D_EW]
        for q4 in elbows:
            s4 = np.sin(q4)
            q3 = seed[2] if straight else np.arctan2(-ew[1] / s4, -ew[0] / s4)
            R04 = R02 @ _rz(q3) @ _ry(-q4)
            for q5, q6, q7 in _zyz(R04.T @ R07, seed[6]):
                q = np.array([_wrap(v) for v in (q1, q2, q3, q4, q5, q6, q7)])
                Tq = fk(q)
                if (np.linalg.norm(Tq[:3, 3] - T[:3, 3]) < 1e-6
                        and np.linalg.norm(Tq[:3, :3] - R07) < 1e-6):
                    sols.append(q)
    return sols


def within(q, limits=JOINT_LIMITS, tol=1e-9):
    q = np.asarray(q, dtype=float)
    return bool(np.all(q >= limits[:, 0] - tol) and np.all(q <= limits[:, 1] + tol))


def _nearest(sols, qnear, limits):
    best, best_d = None, np.inf
    for q in sols:
        if not within(q, limits):
            continue
        d = float(np.linalg.norm(q - qnear))
        if d < best_d:
            best, best_d = q, d
    return best, best_d


def ik(T_base_tool0, qnear, limits=JOINT_LIMITS, psi=None, psi_step=np.radians(2.0)):
    """The solution nearest `qnear` inside `limits`, as a (7,) array -- or None.

    The arm angle is held at `psi` (default: qnear's own) and the nearest of its branches taken.
    If none of them fits the limits, psi is searched outward from there in `psi_step` steps and
    the first arm angle with a fitting branch wins -- the smallest elbow swing that works. When
    qnear's arm is straight (psi undefined) every arm angle is tried and the nearest solution in
    joint space wins."""
    qnear = np.asarray(qnear, dtype=float)
    psi0 = arm_angle(qnear) if psi is None else float(psi)
    n = int(np.ceil(np.pi / psi_step))
    if psi0 is None:
        best, best_d = None, np.inf
        for k in range(-n, n):
            q, d = _nearest(ik_all(T_base_tool0, k * psi_step, qnear), qnear, limits)
            if d < best_d:
                best, best_d = q, d
        return best
    for k in range(n + 1):
        for p in ((psi0,) if k == 0 else (psi0 + k * psi_step, psi0 - k * psi_step)):
            q, _d = _nearest(ik_all(T_base_tool0, p, qnear), qnear, limits)
            if q is not None:
                return q
    return None
