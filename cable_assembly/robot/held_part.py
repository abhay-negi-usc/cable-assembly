"""THE PART IN THE HAND -- what the gripper holds, modelled on the tool from its CAD (the user,
2026-10-31: the connector, after the pickup, at the pose the pick expects it at, its CAD in the
gripper).

    part = HeldPart('plug', mesh, 0.001, T_tool0_obj, grip_half_m, extra_spheres=cable)
    robot.hold(part)        # the plain-move check carries it, and the live view draws it
    robot.release()         # ... until released -- or the gripper opens, which lets go of it

WHERE IT IS is the caller's: its CAD frame in tool0. bnc_assembly gives the EXPECTED pose, where the
commanded grasp puts it -- no estimate: in the hand the plug is too near the camera for its depth,
and the estimate takes too long.

ITS COLLISION MODEL is tool_geometry.part_spheres on the CAD's vertices (in the part's own frame, so
the grid follows its axes, then carried to tool0), plus whatever the caller adds -- the cable leaving
it, which no CAD has; the fingers closed `grip_half_m` from the middle on it.
"""

import numpy as np

from . import tool_geometry


class HeldPart:
    def __init__(self, name, mesh_path, scale, T_tool0_obj, grip_half_m, extra_spheres=()):
        self.name, self.mesh_path, self.scale = str(name), str(mesh_path), float(scale)
        self.T_tool0_obj = np.asarray(T_tool0_obj, dtype=float)
        self.grip_half_m = float(grip_half_m)
        points, _ = tool_geometry.read_ply(self.mesh_path)
        R, t = self.T_tool0_obj[:3, :3], self.T_tool0_obj[:3, 3]
        self.spheres = ([(R @ c + t, r) for c, r in tool_geometry.part_spheres(points * self.scale)]
                        + [(np.asarray(c, dtype=float), float(r)) for c, r in extra_spheres])

    def live(self):
        """For the live view's datagram: the mesh to draw (its path, the scale to m) and its pose in
        tool0 -- the viewer reads the mesh itself (a datagram cannot carry it)."""
        return {'name': self.name, 'mesh': self.mesh_path, 'scale': self.scale,
                'T_tool0': self.T_tool0_obj.tolist()}
