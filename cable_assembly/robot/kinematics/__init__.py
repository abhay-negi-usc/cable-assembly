"""Pure-numpy arm kinematics: forward kinematics, Jacobian, and closed-form inverse kinematics.

No robot connection, no controller. On hardware the UR10e's own controller still solves IK
(arm.py), so these models are what make a dry run, a simulated controller, and the tests
GEOMETRICALLY real -- before them a dry run's fk() returned one fixed pose whatever was commanded.
"""
