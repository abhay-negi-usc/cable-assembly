"""BNC insertion on the iiwa 14 + 2F-85: the plug starts in the gripper above the socket, clocked to
the bayonet entry, and the episode succeeds when it is pushed home and turned to lock.

Laid out after Isaac Lab's Factory peg-insert (isaaclab_tasks/contrib/factory): a direct env with
the held part placed in the closing gripper at reset, keypoint rewards, and an asymmetric critic.
What differs is the arm interface: Factory drives the Franka with an operational-space torque
controller; here the policy's pose deltas go through differential IK to A1..A7 POSITION targets,
which is what the FRI backend streams to the real arm. And the reset only writes state, so
environments reset one at a time (bnc_insert_env.BncInsertEnv._reset_idx).
"""

import math

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass, replace
from isaaclab_tasks.utils import PresetCfg

from cable_assembly_sim.assets import IIWA14_2F85_CFG
from cable_assembly_sim.assets.bnc import BNC_PLUG_CFG, BNC_SOCKET_CFG


@configclass
class BncInsertPhysicsCfg(PresetCfg):
    """PhysX only: under Newton the robot model goes non-finite (sim/README.md)."""

    # Factory's PhysX settings, but 64 position iterations (a cap: the assets ask for 192). Measured
    # with scripts/check_bnc_insert.py, 32 envs, 2026-10-29: 64 locked 88% at 35 ms a step for 512
    # envs; 192 locked 69% (the rigid inner fit jams more) at 2.6x the cost; 32 locked 28%.
    isaacsim_physx = PhysxCfg(
        solver_type=1,
        max_position_iteration_count=64,
        max_velocity_iteration_count=1,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.00625,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
        gpu_collision_stack_size=2**28,
        gpu_max_num_partitions=1,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = isaacsim_physx


@configclass
class BncInsertSceneCfg(InteractiveSceneCfg):
    # The bench top, level with the robot's base (ground_plane z 0 in configs/cells/iiwa14_bench.yaml).
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    # The 2F-85 drive of Isaac Lab's UR10e gear-assembly task (contrib/deploy), which holds a part
    # against contact: at the asset's softer drive the plug slid 3 mm up the pads when pushed home.
    robot = replace(
        IIWA14_2F85_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            **IIWA14_2F85_CFG.actuators,
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                joint_effort_limit=10.0,
                joint_velocity_limit=1.0,
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
                armature=0.0,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                joint_effort_limit=10.0,
                joint_velocity_limit=10.0,
                stiffness=10.0,
                damping=0.05,
                friction=0.0,
                armature=0.0,
            ),
        },
    )
    socket = BNC_SOCKET_CFG
    plug = BNC_PLUG_CFG


@configclass
class BncInsertEnvCfg(DirectRLEnvCfg):
    # 120 Hz physics, the policy at 15 Hz (Factory's rates).
    decimation = 8
    episode_length_s = 10.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=8,
        physics=BncInsertPhysicsCfg(),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    # A physics step costs nearly the same for 16 envs as for 512 (19 ms at 128, 27 ms at 512), so
    # train wide. 512 fits the RTX 3080's 10 GB.
    scene: BncInsertSceneCfg = BncInsertSceneCfg(num_envs=512, env_spacing=2.0, clone_in_fabric=True)

    # (dx, dy, dz, rx, ry, rz) of the held plug, in the socket frame, each in [-1, 1].
    action_space = 6
    # Policy: plug position (3) and yaw (sin, cos) relative to the OBSERVED socket, its linear and
    # angular velocity (6), the previous action (6).
    observation_space = 17
    # Critic: the policy's terms without observation noise (17), the TRUE plug pose relative to the
    # socket (7), the plug in the gripper (7), joint positions and velocities (14).
    state_space = 45

    # ---- The bench (PLACEHOLDERS: the iiwa bench's socket fixture is not measured yet) -----------
    socket_pos = (0.60, 0.0, 0.0)
    """Socket origin in the robot base frame [m]: on the bench, 0.6 m in front of the base."""
    socket_yaw_deg = 135.0
    """Socket clocking about its axis. 135 puts the -90 deg locking turn symmetric about A7 = 0
    (A7 = -45 at entry, +45 locked) at the IK seed below, so A7 has room either way."""
    socket_pos_noise = (0.02, 0.02)
    """Uniform +/- per episode [m]: where the fixture sits."""
    socket_yaw_noise_deg = 30.0

    # ---- What the policy is told about the socket (the vision estimate) -------------------------
    socket_obs_pos_noise = 0.001
    """Gaussian std of the observed socket position, fixed per episode [m] (Factory's)."""
    socket_obs_yaw_noise_deg = 1.0

    friction = 0.75
    """Static and dynamic friction of the plug, the socket and the robot (Factory's 0.75)."""

    # ---- The grasp ----------------------------------------------------------------------------
    grasp_depth = 0.1731
    """Plug origin along tool0 z [m] with the coupler at 0: the pad tips (162.1 mm past tool0 when
    closed on the plug) at 11 mm up the plug, holding the front lip and the knurled collar, so the
    fingertips stay 5 mm above the socket flange with the plug locked. The plug points back up the
    tool axis (its z is tool0's -z), its x along tool0 x; the jaws close along tool0 y."""
    gripper_grasp_deg = 39.5
    """finger_joint written at reset: a 14.8 mm gap, 0.3 mm either side of the plug (the pads touch
    it at ~39.8; 2 mm of gap per degree here)."""
    gripper_close_deg = 45.0
    """finger_joint target throughout: past contact, so the drive squeezes the plug."""
    in_hand_noise = (0.001, 0.0, 0.0015)
    """Uniform +/- of the plug in the gripper, in the plug frame [m]: x along the pads, y across
    them (the jaws centre it), z along its axis."""

    # ---- The start ----------------------------------------------------------------------------
    start_height = (0.013, 0.023)
    """Plug height above its seat [m]: its mouth 2.5-12.5 mm above the socket rim."""
    start_xy_noise = 0.004
    start_yaw_noise_deg = 10.0
    """About the bayonet entry (+90 deg). The straight slots leave the pins ~+/-4.5 deg."""
    ik_seed_deg = (0.0, 52.6, 0.0, -85.7, 0.0, 41.8, -45.0)
    """A1..A7 holding the plug 18 mm above the nominal socket at its entry clocking (closed-form IK,
    cable_assembly/robot/kinematics/iiwa14.py). The reset's DLS IK starts here."""
    ik_iterations = 30

    # ---- The action ---------------------------------------------------------------------------
    pos_action_scale = 0.002
    """Target displacement per policy step at |a| = 1 [m] (30 mm/s)."""
    rot_action_scale = math.radians(2.0)
    """Target rotation per policy step at |a| = 1 [rad] (30 deg/s)."""
    max_target_lead = (0.002, math.radians(4.0))
    """How far the target may run ahead of the measured held pose, per axis [m] and in angle [rad].
    In contact this, times the arm's stiffness, caps the force and torque."""
    pos_action_bounds = ((-0.02, -0.02, -0.01), (0.02, 0.02, 0.04))
    """(low, high) of the target in the observed socket frame [m], as Factory bounds its workspace.
    z reaches below the seat so a plug held high in the gripper can still be pushed home."""
    lock_tilt = True
    """Keep the plug axis vertical: only the yaw of a rotation action is used (Factory's choice)."""
    ema_factor = 0.2
    ik_damping = 0.05

    # ---- Rewards and success -------------------------------------------------------------------
    # Factory's keypoint terms: four points along the plug axis against the same points at the seat.
    # They do not see yaw, so it is the depth -- reachable only through the bayonet -- that pays.
    num_keypoints = 4
    keypoint_scale = 0.15
    keypoint_coef_baseline = (5.0, 4.0)
    keypoint_coef_coarse = (50.0, 2.0)
    keypoint_coef_fine = (100.0, 0.0)
    center_tol = 0.0025
    """Plug axis within this of the socket axis [m] to count as engaged or locked."""
    engage_height = 0.005
    """Below this the pins are in the straight slots: lower than a mis-clocked plug can go."""
    success_height = 0.0005
    success_yaw_deg = 15.0
    """Within this of a locked clocking (0 or 180 deg)."""

    def play_mode(self):
        """Watch env 0's connector up close: the camera follows its socket, which moves +/-2 cm."""
        from isaaclab_visualizers.kit import KitVisualizerCfg

        super().play_mode()
        # The plug is 14 mm across, and Kit ignores focal_length, so the camera sits 11 cm off the
        # socket, on the far side from the robot, level with the barrel rim. It looks between the pads
        # at the entry clocking (the jaws close along the plug's y); a pad hides the plug once locked.
        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(0.08, 0.08, 0.04), lookat=(0.0, 0.0, 0.025), origin_type="asset", origin_track_path="socket"
        )
