"""KUKA LBR iiwa 14 R820 with a Robotiq 2F-85 on the media flange -- the iiwa bench's tool
(configs/cells/iiwa14_bench.yaml). The USD is built by scripts/build_iiwa14.py from the same
lbr-stack description the real arm's kinematics are tested against.

Joint names are the real arm's (A1..A7), and tool0 is the media-flange face, 0.035 m beyond link_7,
as in cable_assembly/robot/kinematics/iiwa14.py. So a joint vector or a tool0 pose means the same
thing in the simulator and on the bench.
"""

import math

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import CABLE_ASSEMBLY_SIM_ASSETS_DIR

ARM_JOINTS = [f"A{i}" for i in range(1, 8)]
TOOL0 = "tool0"
GRIPPER_DRIVE = "finger_joint"
"""The 2F-85's opening, as one angle: 0 is open, ~45 deg closed (limit 47 deg)."""
GRIPPER_JOINT_SIGNS = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}
"""Each 2F-85 joint's target per radian of finger_joint, from Isaac Lab's UR10e 2F-85 gear-assembly
task (set_finger_joint_pos_robotiq_2f85). The linkage is commanded as a whole: a target on
finger_joint alone leaves the finger joints' PD holding them open, and it closes only half way."""

# kinematics/iiwa14.py HOME: tool pointing straight down, A7 at mid-range.
HOME_DEG = [0.0, 30.0, 0.0, -90.0, 0.0, 60.0, 0.0]

# KUKA's rated maximum torques for the iiwa 14 R820 [Nm]. lbr-stack's URDF caps every joint at 200,
# which is too much for A5..A7 and too little for A1/A2.
EFFORT_LIMIT_NM = {"A1": 320.0, "A2": 320.0, "A3": 176.0, "A4": 176.0, "A5": 110.0, "A6": 40.0, "A7": 40.0}
# Rated speeds [deg/s], the same as kinematics.iiwa14.VELOCITY_LIMITS and the URDF.
VELOCITY_LIMIT_DEG_S = {"A1": 85.0, "A2": 85.0, "A3": 100.0, "A4": 75.0, "A5": 130.0, "A6": 135.0, "A7": 135.0}

IIWA14_2F85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(CABLE_ASSEMBLY_SIM_ASSETS_DIR / "iiwa14_2f85.usda"),
        activate_contact_sensors=True,
        # The real arm runs KUKA's joint position control, which compensates gravity. With gravity
        # off, the PD below only has to set the tracking response, not hold the arm up.
        rigid_props=PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=[
            # 192 position iterations, as Isaac Lab's Factory tasks use. At 16 the solver cannot hold the
            # 2F-85's 40 g mimic linkage against a stiffly held 30 kg arm: the fingers stalled
            # asymmetrically at ~24 deg of 45 and pushed A5/A7 6 deg off target.
            PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=192, solver_velocity_iteration_count=1
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={**{j: math.radians(q) for j, q in zip(ARM_JOINTS, HOME_DEG)}, ".*finger.*|.*knuckle.*": 0.0},
    ),
    actuators={
        # UNTUNED: critically damped at A1 with the arm stretched out (~2.5 kg m^2). To be matched to
        # the real arm's step response through the FRI driver at bring-up.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            joint_effort_limit=EFFORT_LIMIT_NM,
            actuator_velocity_limit={j: math.radians(v) for j, v in VELOCITY_LIMIT_DEG_S.items()},
            stiffness=1000.0,
            damping=100.0,
        ),
        # The 2F-85 groups below are Isaac Lab's UR10e_ROBOTIQ_2F_85_CFG, for the same Robotiq USD.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_DRIVE],
            joint_effort_limit=10.0,
            joint_velocity_limit=1.0,
            stiffness=11.25,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            joint_effort_limit=1.0,
            joint_velocity_limit=1.0,
            stiffness=0.2,
            damping=0.001,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
            joint_effort_limit=1.0,
            joint_velocity_limit=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""iiwa 14 + Robotiq 2F-85, fixed base, joint position targets on A1..A7."""
