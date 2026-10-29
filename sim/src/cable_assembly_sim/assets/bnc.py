"""The BNC plug and socket: Isaac Lab's NIST Task Board connectors (Factory/NIST on the Isaac Sim 6.2
asset server). SDF colliders, 50 g each, both authored in the SOCKET frame with the plug seated, so
a plug pose equal to the socket pose is the mated, locked connector.

The plug is ONE rigid body. A real BNC plug's bayonet collar turns on its body; here the J-slots are
cut into the whole plug, so locking it turns the plug itself -1/4 turn about the socket axis.

Geometry, measured off the collision meshes (socket frame, z up the socket axis, plug seated):

    socket   flange r 11.7 mm at z 2-6; barrel r 4.08-4.16 inside, 4.70-4.76 outside, up to its
             rim at z 21.17; two bayonet pins, r 5.5 mm, at +/-90 deg about z, centred at z 16.5
    plug     sleeve bore r 4.90-4.98 mm (~0.2 mm on the barrel); mouth at z 10.67, rear at z 32.0;
             inside it, from z 11.9, the inner contact r 4.00-4.08, which enters the barrel bore
             with under 0.1 mm (spring fingers on a real BNC, rigid here: where plugs jam);
             outside: a lip r 6.96 at z 10.7-13 and the knurled collar r 7.0-7.1 at z 19-24.5
    slots    straight in at plug angle 0 / 180 deg up to plug z ~14.5, then a helix to +/-90 deg
             at z ~16.5 -- so the plug enters turned +90 deg and seats at 0 (or -90 -> 180)

The plug's height above its seat therefore reads the whole insertion: 10.5 mm at the barrel rim,
9.3 mm at the inner fit, ~6.8 mm where a mis-clocked plug lands on the pins, 2.5 mm where the
twist begins, 0 when locked.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

NIST_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory/NIST"

PLUG_MOUTH_Z = 0.01067
"""The plug's lowest point, in its own frame [m]."""
SOCKET_RIM_Z = 0.02117
"""The top of the socket barrel, in the socket frame [m]."""
PLUG_DIAMETER = 0.0142
"""Across the knurled collar, where the 2F-85 holds it [m]."""
TWIST_START_HEIGHT = 0.0025
"""Plug height above its seat at which the pins leave the straight slots for the helix [m]."""
ENTRY_YAW_DEG = 90.0
"""Plug yaw about the socket axis at which the pins line up with the straight slots."""

# Contact settings of Isaac Lab's NIST Factory assets (isaaclab_tasks/contrib/nist).
_COLLISION = PhysxCollisionCfg(contact_offset=0.0025, rest_offset=0.0)

BNC_SOCKET_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BncSocket",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{NIST_DIR}/bnc_socket.usd",
        # Bolted to the fixture: moved only by resets.
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            PhysxRigidBodyCfg(solver_position_iteration_count=192, solver_velocity_iteration_count=1),
        ],
        collision_props=_COLLISION,
    ),
)

BNC_PLUG_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BncPlug",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{NIST_DIR}/bnc_plug.usd",
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            max_depenetration_velocity=5.0, solver_position_iteration_count=192, solver_velocity_iteration_count=1
        ),
        collision_props=_COLLISION,
    ),
)
