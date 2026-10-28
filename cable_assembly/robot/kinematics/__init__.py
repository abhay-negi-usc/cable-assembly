"""Pure-numpy arm kinematics: forward kinematics, Jacobian, and closed-form inverse kinematics.

No robot connection, no controller. On hardware the UR10e's own controller still solves IK
(backends/ur_rtde.py); for the iiwa these models ARE the IK, since FRI takes joint commands only.
They are also what make the simulator and the tests geometrically real -- before them a dry run's
fk() returned one fixed pose whatever was commanded.

EVERY MODEL MODULE HAS THE SAME SHAPE, so the arm backends and the task code never branch on the
robot:

    NAME, JOINT_NAMES, DOF          identity
    TOOL_ROLL_JOINT, JOINT_ROLES    which joint does what ('tool_roll', 'wrist_nod')
    JOINT_LIMITS, VELOCITY_LIMITS   (DOF, 2) rad and (DOF,) rad/s, the mechanical ratings
    HOME                            a safe start pose for the simulator
    fk(q), fk_links(q)              tool0 / every frame, in the robot's base frame
    jacobian(q)                     (6, DOF) geometric, [v; w] at tool0
    ik(T, qnear, limits)            the solution nearest qnear, or None
    within(q, limits)               joint-limit check
"""

from . import iiwa14, ur10e

MODELS = {m.NAME: m for m in (ur10e, iiwa14)}


def get(name):
    """The kinematics module for a robot type (`robot.type` in the config)."""
    try:
        return MODELS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f'unknown robot type {name!r}; known: {sorted(MODELS)}') from None
