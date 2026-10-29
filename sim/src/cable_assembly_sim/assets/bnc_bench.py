"""The iiwa bench's own BNC parts: McMaster-Carr 6641T31 (the cable's plug) and 4441N11 (the panel
adapter it plugs into), built from McMaster's STEP files by scripts/build_bnc.py.

Unlike Isaac Lab's NIST plug (assets/bnc.py), this plug is an ARTICULATION, as the real one is a
body with a bayonet collar on it: the collar turns freely on the body and slides 1.2 mm along it
against its wave washer (the `collar_slide` drive). Both parts are authored in the SOCKET frame --
the adapter's own CAD frame: z up its axis, its two bayonet pins on +/-y -- with the plug SEATED
(its insulator on the socket's dielectric), so a plug pose equal to the socket pose is seated, and
collar joints at 0 put the collar's slot entries on the pins.

Heights in the socket frame [m], measured off the CAD (build_bnc.py prints the parts' extents):

    socket  barrel r 4.83 mm up to SOCKET_RIM_Z; pins r 5.51 mm, z 6.91-8.89 mm; the hex ring around
            the barrel up to 8.93 mm, the flange r 15.5 mm at 3.8-5.3 mm
    plug    collar front 2.71 mm above the plug origin, the knurled band r 7.22 mm at 11.6-16.2 mm;
            the strain relief from 17.8 mm (r 7.97 mm at its widest, 21-25 mm) to 46.5 mm

So the plug's HEIGHT above its seat reads the bayonet: 9.4 mm when the collar meets the barrel,
6.18 mm resting on the pins (collar off its slots), 1.08 mm at the bottom of the straight slots
(the 5 mm advance the bench's connector clocking calls seated), 0 home -- reached only by turning
the collar into its helix, which a push alone does not do (it self-locks at friction 0.3).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

from . import CABLE_ASSEMBLY_SIM_ASSETS_DIR

BNC_DIR = CABLE_ASSEMBLY_SIM_ASSETS_DIR / "bnc"

SOCKET_RIM_Z = 0.01214
PIN_TOP_Z = 0.00889
COLLAR_ON_PINS_HEIGHT = 0.00618
"""Plug height when the collar's front rests on the pins."""
SLOT_FLOOR_HEIGHT = 0.00108
"""Plug height when the pins reach the bottom of the collar's straight slots."""
STRAIN_RELIEF_Z = (0.0178, 0.0465)
"""The strain relief (the moulded boot) along the plug axis, from its front to the cable."""
COLLAR_LOCK_DEG = -96.0
"""collar_turn at the locking pocket (measured: the collar settles there when let go)."""

COLLAR_SLIDE_STIFFNESS = 5000.0
"""N/m: the wave washer, as scripts/build_bnc.py authors it."""
COLLAR_SLIDE_PRELOAD = 0.001
"""collar_slide target [m], behind its rear stop: ~5 N of washer preload at rest."""

# Contact settings of Isaac Lab's NIST Factory connectors (isaaclab_tasks/contrib/nist).
_COLLISION = PhysxCollisionCfg(contact_offset=0.0025, rest_offset=0.0)

BENCH_BNC_SOCKET_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BncSocket",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(BNC_DIR / "bnc_socket.usd"),
        # Bolted to the fixture: moved only by resets.
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            PhysxRigidBodyCfg(solver_position_iteration_count=192, solver_velocity_iteration_count=1),
        ],
        collision_props=_COLLISION,
    ),
)

BENCH_BNC_PLUG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/BncPlug",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(BNC_DIR / "bnc_plug.usd"),
        rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=5.0),
        # Self-collision ON: the collar must bear on its seat on the body (scripts/build_bnc.py).
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=True, solver_position_iteration_count=192, solver_velocity_iteration_count=1
        ),
        collision_props=_COLLISION,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"collar_slide": 0.0, "collar_turn": 0.0}),
    actuators={
        "collar_slide": ImplicitActuatorCfg(
            joint_names_expr=["collar_slide"], stiffness=COLLAR_SLIDE_STIFFNESS, damping=10.0
        ),
        # Free: what turns the collar with the body is the friction of its seat, under whatever axial
        # load presses it there (the bench: the collar follows the strain relief to find its slots,
        # but not far enough to lock).
        "collar_turn": ImplicitActuatorCfg(joint_names_expr=["collar_turn"], stiffness=0.0, damping=0.0),
    },
)
"""The plug: body (the gripper holds its strain relief) -> collar_slide -> collar_turn -> collar."""
