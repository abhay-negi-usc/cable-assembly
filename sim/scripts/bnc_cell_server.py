"""Serve the iiwa 14 BNC cell in Isaac Sim to cable_assembly's `isaac_sim` arm backend, so that the
bench's own BNC assembly (cable_assembly/apps/bnc_assembly.py) runs against the simulated cell.

LOCKSTEP. The simulation advances only when the client asks it to, by exactly the time it asks
for: a servo command of dt steps dt, a settle of 0.5 s steps 0.5 s. The bench's control loops --
the 125 Hz admittance over servo_l, the force guards' persistence, the engage timeouts -- then
run in SIMULATED time however slowly the physics runs (tens of times slower than real time with
the 0.1 mm SDF fits), which a free-running simulation behind a real-time client could not give.

THE CELL, one of it:
  - the iiwa 14 + Robotiq 2F-85 (assets/iiwa14.py), its joint drives stiff like KUKA's position
    control (the compliance is the app's admittance, not the arm's);
  - McMaster's BNC socket (assets/bnc_bench.py) at the cell's target -- read from the SAME frames
    file the app plans with (configs/cells/iiwa14_sim.frames.yaml), so the two cannot disagree;
  - McMaster's BNC plug, held at its strain relief.

THE WRENCH is what a 6-axis F/T sensor between the flange and the gripper would read (Isaac Lab's
joint-wrench sensor on the gripper mount), as the external force ON the tool in base_link axes,
moment about the flange -- the convention every cable_assembly backend reports in. The real iiwa
estimates it from joint torques instead; this is that estimate without its model error.

PROTOCOL: multiprocessing.connection on localhost. Each request is (command, kwargs); each reply a
dict of plain lists and floats (the two environments' numpy versions differ):

    info()                              static facts: dt, joints, the socket pose, the grasp
    reset(q, collar_deg, grasp_depth_m) arm at q (rad), the plug in hand, gripper closed on it
    step(q=None, seconds=None)          track arm target q (rad) for `seconds` (default one dt)
    gripper(counts, speed, force)       Robotiq-style target (0 open .. 255 closed)
    state()                             the measurement: t, q, wrench, gripper, and the plug's truth
    note(text)                          a label for the video's caption (the client's phase)
    close()

    cd sim
    ./run.sh python scripts/bnc_cell_server.py                  # headless; --viz kit to watch
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--port", type=int, default=6007)
parser.add_argument("--authkey", default="cable-assembly-sim")
parser.add_argument("--frames", default=os.path.join(REPO, "configs", "cells", "iiwa14_sim.frames.yaml"))
parser.add_argument("--target", default="bnc_connector_finger_holder", help="the socket target in --frames")
parser.add_argument("--dt", type=float, default=1 / 250, help="physics step [s]")
parser.add_argument("--position_iterations", type=int, default=64)
parser.add_argument("--friction", type=float, default=0.3, help="static and dynamic: any surface without its own (the bench)")
parser.add_argument("--grip_friction", type=float, default=1.0, help="the robot's shapes: rubber pads on the rubber strain relief")
parser.add_argument("--grip_contact_deg", type=float, default=38.9, help="finger_joint where the pads meet the strain relief")
parser.add_argument("--collar_drag_nm", type=float, default=None, help="collar_turn joint friction")
parser.add_argument("--physics", default="isaacsim_physx")
parser.add_argument("--video", default=None, help="record a close-up of the connector to this .mp4 (one per backend connection)")
parser.add_argument("--video_fps", type=float, default=10.0, help="frames per SIMULATED second (played back in real time)")
add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True  # the launcher only sees the physics config, not the scene's camera

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from multiprocessing.connection import Listener  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, JointWrenchSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402

from cable_assembly_sim.assets import IIWA14_2F85_CFG  # noqa: E402
from cable_assembly_sim.assets.bnc_bench import (  # noqa: E402
    BENCH_BNC_PLUG_CFG,
    BENCH_BNC_SOCKET_CFG,
    COLLAR_SLIDE_PRELOAD,
)
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_DRIVE, GRIPPER_JOINT_SIGNS, TOOL0  # noqa: E402

# The connector datum of the app's frames (+X the insertion axis, into the socket) in the socket
# frame (z up its axis): X = -z, Y = y, Z = x, i.e. a quarter turn about y.
R_SOCKET_CONN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

# The Robotiq's encoder against the simulated finger_joint: the 2F-85's fingertip angle is linear
# in counts (cable_assembly/robot/gripper_kinematics.py: 0.2186 deg a count, pads meeting at ~216),
# and the simulated linkage's finger_joint spans the same ~45 deg.
DEG_PER_COUNT = 0.2186
FINGER_MAX_DEG = 45.0
# finger_joint speed at the Robotiq's speed 0 .. 255 [rad/s]: the real stroke takes ~0.5-2 s.
FINGER_SPEED = (0.25, 1.5)


def socket_pose_from_frames(path, name):
    """T_base_socket from the target `name` in a cable_assembly frames file (xyz_mm, rpy_deg,
    extrinsic XYZ, parent base_link)."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    t = (doc.get("targets") or {}).get(name)
    if t is None:
        raise SystemExit(f"{path} has no targets.{name}: the cell needs the socket target")
    T_base_conn = np.eye(4)
    T_base_conn[:3, :3] = Rotation.from_euler("xyz", np.radians(t.get("rpy_deg", [0, 0, 0]))).as_matrix()
    T_base_conn[:3, 3] = np.asarray(t.get("xyz_mm", [0, 0, 0]), dtype=float) / 1000.0
    T_socket_conn = np.eye(4)
    T_socket_conn[:3, :3] = R_SOCKET_CONN
    return T_base_conn @ np.linalg.inv(T_socket_conn)


def pose7(T):
    """4x4 -> (x, y, z, qx, qy, qz, qw), Isaac Lab's root-pose layout."""
    return [*T[:3, 3].tolist(), *Rotation.from_matrix(T[:3, :3]).as_quat().tolist()]


def matrix(pose):
    """(x, y, z, qx, qy, qz, qw) -> 4x4."""
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def set_friction(asset, value):
    """Static and dynamic friction of every shape of `asset` (Factory's factory_utils.set_friction)."""
    import warp as wp

    materials = wp.to_torch(asset.root_view.get_material_properties())
    materials[..., 0] = value
    materials[..., 1] = value
    env_ids = torch.arange(1, device="cpu", dtype=torch.int32)
    asset.root_view.set_material_properties(wp.from_torch(materials.contiguous()), wp.from_torch(env_ids))


@configclass
class CellSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = replace(
        IIWA14_2F85_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            **IIWA14_2F85_CFG.actuators,
            # KUKA's joint position control is stiff: ~0.4 mm at the tool for 10 N at the bench poses.
            # Every compliance in the BNC methodology is the app's admittance on top of it.
            "arm": replace(IIWA14_2F85_CFG.actuators["arm"], stiffness=1.0e4, damping=100.0),
            # The 2F-85 at full force (the bench runs it at force 255, ~235 N at the pads): ~11 N m on
            # finger_joint squeezing the strain relief (stalled ~5 deg short of its closed target),
            # through the ~55 mm fingertip circle (robot/gripper_kinematics.py) ~200 N a pad. Isaac
            # Lab's gear-assembly drive (40) gave ~60 N, and the plug slid out of the hand under the
            # engage's wiggle.
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_DRIVE], joint_effort_limit=15.0, joint_velocity_limit=2.0,
                stiffness=120.0, damping=3.0, friction=0.0, armature=0.0,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"], joint_effort_limit=10.0, joint_velocity_limit=10.0,
                stiffness=10.0, damping=0.05, friction=0.0, armature=0.0,
            ),
        },
    )
    socket = BENCH_BNC_SOCKET_CFG
    plug = BENCH_BNC_PLUG_CFG
    # The wrist F/T: the incoming joint wrench of every robot link; the gripper mount's is read.
    wrist = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


VIDEO_SIZE = (1024, 576)


@configclass
class CellSceneWithVideoCfg(CellSceneCfg):
    video_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoCam",
        update_period=0.0,
        width=VIDEO_SIZE[0],
        height=VIDEO_SIZE[1],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
    )


class Cell:
    def __init__(self, sim, scene, T_base_socket):
        self.sim, self.scene, self.dt = sim, scene, sim.get_physics_dt()
        self.robot, self.plug, self.socket = scene["robot"], scene["plug"], scene["socket"]
        self.wrist = scene["wrist"]
        dev = sim.device
        self.dev = dev
        self.origin = scene.env_origins[0].cpu().numpy()
        self.arm_ids, names = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
        assert names == ARM_JOINTS, names
        self.grip_ids, grip_names = self.robot.find_joints(list(GRIPPER_JOINT_SIGNS), preserve_order=True)
        self.grip_signs = torch.tensor([GRIPPER_JOINT_SIGNS[n] for n in grip_names], device=dev)
        self.finger_id = self.grip_ids[grip_names.index(GRIPPER_DRIVE)]
        self.tool0_id = self.robot.find_bodies(TOOL0)[0][0]
        mount = self.wrist.find_bodies("base_link")[0]
        if len(mount) != 1:
            raise SystemExit(f"expected one gripper base_link among {self.wrist.body_names}")
        self.mount_id = mount[0]
        self.body_id = self.plug.find_bodies("body")[0][0]
        self.slide_id, self.turn_id = self.plug.find_joints(["collar_slide", "collar_turn"], preserve_order=True)[0]
        self.T_base_socket = T_base_socket
        T_world_socket = T_base_socket.copy()
        T_world_socket[:3, 3] += self.origin
        self.socket.write_root_pose_to_sim_index(
            root_pose=torch.tensor([pose7(T_world_socket)], device=dev), env_ids=torch.tensor([0], device=dev)
        )
        set_friction(self.robot, args_cli.grip_friction)
        self.cam = scene["video_cam"] if args_cli.video else None
        if self.cam is not None:
            # A close-up from across the socket, the arm coming in from behind it.
            target = T_world_socket[:3, 3] + np.array([0.0, 0.0, 0.03])
            eye = T_world_socket[:3, 3] + np.array([0.16, 0.13, 0.08])
            self.cam.set_world_poses_from_view(torch.tensor([eye], device=dev, dtype=torch.float32),
                                               torch.tensor([target], device=dev, dtype=torch.float32))
        self.writer = None
        self.next_frame = 0.0
        self.label = ""
        self.t = 0.0
        self.steps = 0
        self.render_every = max(1, round(1 / 30 / self.dt))
        self.q_target = self.robot.data.joint_pos.torch[0, self.arm_ids].clone()
        self.finger_target = 0.0  # rad, what the drive tracks
        self.finger_goal = 0.0  # rad, where the Robotiq was told to go
        self.finger_speed = FINGER_SPEED[1]
        self.finger_still_s = 0.0
        self.prev_finger = 0.0
        self.grasp_depth = None

    # ---------------------------------------------------------------- stepping
    def _write_targets(self):
        r = self.robot
        r.set_joint_position_target_index(target=self.q_target.unsqueeze(0), joint_ids=self.arm_ids)
        r.set_joint_position_target_index(target=(self.grip_signs * self.finger_target).unsqueeze(0), joint_ids=self.grip_ids)
        tgt = torch.zeros((1, 2), device=self.dev)
        tgt[0, 0] = COLLAR_SLIDE_PRELOAD
        self.plug.set_joint_position_target_index(target=tgt, joint_ids=[self.slide_id, self.turn_id])

    def advance(self, seconds):
        n = max(1, round(seconds / self.dt))
        gui = getattr(self.sim, "has_gui", False)
        gui = gui() if callable(gui) else bool(gui)
        for _ in range(n):
            self.steps += 1
            # A viewport (--viz kit) is redrawn at ~30 frames per simulated second.
            render = gui and self.steps % self.render_every == 0
            # The Robotiq moves its fingers at the commanded speed; the drive tracks that ramp.
            step = self.finger_speed * self.dt
            self.finger_target += float(np.clip(self.finger_goal - self.finger_target, -step, step))
            self._write_targets()
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(self.dt)
            self.t += self.dt
            # Stalled = the finger angle stopped changing. Not the joint velocity: squeezed against a
            # compliant contact, PhysX keeps reporting the drive's push as a velocity it never moves.
            finger = float(self.robot.data.joint_pos.torch[0, self.finger_id])
            v = abs(finger - self.prev_finger) / self.dt
            self.prev_finger = finger
            self.finger_still_s = self.finger_still_s + self.dt if v < 0.02 else 0.0
            if self.writer is not None and self.t >= self.next_frame:
                self.capture()
                self.next_frame += 1.0 / args_cli.video_fps

    # ---------------------------------------------------------------- video
    def start_video(self, path):
        import imageio

        # Fragmented, so the file plays while the run is still being recorded.
        self.writer = imageio.get_writer(path, fps=args_cli.video_fps, codec="libx264", quality=8,
                                         ffmpeg_params=["-movflags", "frag_keyframe+empty_moov"])
        self.video_path = path
        self.next_frame = self.t
        print(f"[cell] recording {path}", flush=True)

    def stop_video(self):
        if self.writer is not None:
            self.writer.close()
            print(f"[cell] wrote {self.video_path}", flush=True)
        self.writer = None

    def capture(self):
        from PIL import Image, ImageDraw

        self.sim.render()
        self.cam.update(self.dt, force_recompute=True)
        rgb = self.cam.data.output["rgb"]
        rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[0, ..., :3].cpu().numpy()
        img = Image.fromarray(rgb.astype(np.uint8))
        tr = self.truth()
        lines = [
            f"t {self.t:6.2f} s (simulated)   {self.label}",
            f"plug {tr['height_mm']:6.2f} mm above home, {tr['lateral_mm']:4.2f} mm off axis",
            f"collar {tr['collar_abs_deg']:+6.1f} deg in the socket (locks at -96), spring {tr['slide_mm']:5.2f} mm",
        ]
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 22 * len(lines) + 8), fill=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((10, 6 + 22 * i), line, fill=(255, 255, 255))
        self.writer.append_data(np.asarray(img))

    def note(self, text):
        self.label = str(text)
        return {}

    # ---------------------------------------------------------------- commands
    def info(self):
        return {
            "dt": self.dt,
            "arm_joints": ARM_JOINTS,
            "T_base_socket": self.T_base_socket.tolist(),
            "R_socket_conn": R_SOCKET_CONN.tolist(),
            "deg_per_count": DEG_PER_COUNT,
        }

    def reset(self, q, collar_deg=0.0, grasp_depth_m=0.1785, settle_s=0.3):
        """The arm at q, the plug between the open pads at its strain relief, then the fingers
        closed on it and left to squeeze for 0.3 s."""
        r, dev = self.robot, self.dev
        joint_pos = r.data.default_joint_pos.torch.clone()
        joint_pos[0, self.arm_ids] = torch.tensor(q, device=dev, dtype=torch.float32)
        # The pads written a hair clear of the strain relief (15.9 mm wide: they meet it at ~38.9 deg,
        # ~2 mm of gap a degree) and driven straight at closed, so they squeeze it within a few
        # milliseconds -- ramped like a Robotiq command, the plug fell out before they closed.
        open_deg = args_cli.grip_contact_deg - 0.1
        joint_pos[0, self.grip_ids] = self.grip_signs * math.radians(open_deg)
        r.write_joint_position_to_sim_index(position=joint_pos)
        r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
        self.q_target = joint_pos[0, self.arm_ids].clone()
        self.finger_target = self.finger_goal = math.radians(FINGER_MAX_DEG)
        self._write_targets()
        # The plug, axis down the tool's z, at grasp_depth_m from tool0 (its frame's z points back
        # up the tool; its x along the tool's x).
        T_world_tool0 = matrix(r.data.body_link_pose_w.torch[0, self.tool0_id].cpu().numpy())
        T_tool0_plug = np.eye(4)
        T_tool0_plug[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T_tool0_plug[2, 3] = grasp_depth_m
        T_world_plug = T_world_tool0 @ T_tool0_plug
        self.plug.write_root_pose_to_sim_index(root_pose=torch.tensor([pose7(T_world_plug)], device=dev))
        self.plug.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
        pq = torch.zeros((1, 2), device=dev)
        pq[0, 1] = math.radians(collar_deg)
        self.plug.write_joint_position_to_sim_index(position=pq, joint_ids=[self.slide_id, self.turn_id])
        self.plug.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pq), joint_ids=[self.slide_id, self.turn_id])
        self.grasp_depth = grasp_depth_m
        self.finger_still_s = 0.0
        if settle_s > 0.0:
            self.advance(settle_s)
        return self.state()

    def step(self, q=None, seconds=None):
        if q is not None:
            self.q_target = torch.tensor(q, device=self.dev, dtype=torch.float32)
        self.advance(self.dt if seconds is None else float(seconds))
        return self.state()

    def gripper(self, counts, speed=255, force=128):
        self.finger_goal = math.radians(min(FINGER_MAX_DEG, max(0.0, float(counts)) * DEG_PER_COUNT))
        self.finger_speed = FINGER_SPEED[0] + (FINGER_SPEED[1] - FINGER_SPEED[0]) * float(speed) / 255.0
        self.finger_still_s = 0.0
        return self.state()

    def state(self):
        r = self.robot
        q = r.data.joint_pos.torch[0, self.arm_ids].cpu().numpy()
        # The wrist wrench: the mount joint's reaction on the gripper (child frame = tool0's axes,
        # moment about tool0), negated to the force the world exerts ON the tool, then into base_link.
        f = self.wrist.data.force.torch[0, self.mount_id].cpu().numpy()
        tau = self.wrist.data.torque.torch[0, self.mount_id].cpu().numpy()
        T_world_tool0 = matrix(r.data.body_link_pose_w.torch[0, self.tool0_id].cpu().numpy())
        R = T_world_tool0[:3, :3]
        wrench = np.concatenate([R @ -f, R @ -tau])
        finger = float(r.data.joint_pos.torch[0, self.finger_id])
        reached = abs(finger - self.finger_goal) < math.radians(1.0)
        moving = self.finger_still_s < 0.05 and not reached
        obj = 0 if moving else (3 if reached else (2 if self.finger_goal > finger else 1))
        return {
            "t": self.t,
            "q": q.tolist(),
            "wrench": wrench.tolist(),
            "gripper": {
                "counts": max(0.0, math.degrees(finger) / DEG_PER_COUNT),
                "goal_counts": math.degrees(self.finger_goal) / DEG_PER_COUNT,
                "obj": obj,
                "finger_deg": math.degrees(finger),
                "finger_speed_deg_s": math.degrees(float(r.data.joint_vel.torch[0, self.finger_id])),
            },
            "truth": self.truth(),
        }

    def truth(self):
        """The plug in the socket frame: what the app can only infer."""
        T_world_body = matrix(self.plug.data.body_link_pose_w.torch[0, self.body_id].cpu().numpy())
        T_world_socket = self.T_base_socket.copy()
        T_world_socket[:3, 3] += self.origin
        T = np.linalg.inv(T_world_socket) @ T_world_body
        jp = self.plug.data.joint_pos.torch[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, T[2, 2]))))
        # The collar's clocking IN THE SOCKET: the body's yaw plus the collar's turn on it, folded
        # into (-180, 180] -- and its two slots make it two-fold, so -96 and +84 are both locked.
        body_yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        collar_abs = (body_yaw + math.degrees(float(jp[self.turn_id])) + 180.0) % 360.0 - 180.0
        # ... and in the hand: where the grasp put it (tool0 frame), against where it was placed.
        T_world_tool0 = matrix(self.robot.data.body_link_pose_w.torch[0, self.tool0_id].cpu().numpy())
        T_tool = np.linalg.inv(T_world_tool0) @ T_world_body
        return {
            "plug_in_socket": T.tolist(),
            "height_mm": T[2, 3] * 1000.0,
            "lateral_mm": float(np.hypot(T[0, 3], T[1, 3]) * 1000.0),
            "tilt_deg": tilt,
            "collar_deg": math.degrees(float(jp[self.turn_id])),
            "body_yaw_deg": body_yaw,
            "collar_abs_deg": collar_abs,
            "slide_mm": float(jp[self.slide_id]) * 1000.0,
            "plug_in_tool_mm": (T_tool[:3, 3] * 1000.0).tolist(),
            "plug_axis_off_tool_deg": math.degrees(math.acos(max(-1.0, min(1.0, -T_tool[2, 2])))),
        }


def main():
    T_base_socket = socket_pose_from_frames(args_cli.frames, args_cli.target)
    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        if hasattr(physics_cfg, "max_position_iteration_count"):
            physics_cfg.max_position_iteration_count = args_cli.position_iterations
        material = sim_utils.RigidBodyMaterialCfg(static_friction=args_cli.friction, dynamic_friction=args_cli.friction)
        sim = SimulationContext(
            sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device, physics=physics_cfg, physics_material=material)
        )
        sim.set_camera_view(
            (T_base_socket[0, 3] + 0.25, T_base_socket[1, 3] - 0.25, T_base_socket[2, 3] + 0.2), tuple(T_base_socket[:3, 3])
        )
        scene_cfg = (CellSceneWithVideoCfg if args_cli.video else CellSceneCfg)(num_envs=1, env_spacing=3.0)
        if args_cli.collar_drag_nm is not None:
            scene_cfg.plug.actuators["collar_turn"].friction = args_cli.collar_drag_nm
        scene = instantiate(scene_cfg)
        sim.reset()
        cell = Cell(sim, scene, T_base_socket)
        print(f"[cell] socket at {np.round(T_base_socket[:3, 3] * 1000, 1).tolist()} mm, dt {cell.dt * 1000:.1f} ms; "
              f"waiting for the cable_assembly backend on localhost:{args_cli.port}", flush=True)
        runs = 0
        with Listener(("localhost", args_cli.port), authkey=args_cli.authkey.encode()) as listener:
            while True:
                with listener.accept() as conn:
                    print("[cell] backend connected", flush=True)
                    runs += 1
                    if args_cli.video:
                        root, ext = os.path.splitext(args_cli.video)
                        cell.start_video(args_cli.video if runs == 1 else f"{root}_{runs}{ext}")
                    while True:
                        try:
                            cmd, kwargs = conn.recv()
                        except EOFError:
                            print("[cell] backend disconnected", flush=True)
                            cell.stop_video()
                            break
                        if cmd == "close":
                            cell.stop_video()
                            conn.send({"ok": True})
                            return
                        try:
                            conn.send({"ok": True, **getattr(cell, cmd)(**kwargs)})
                        except Exception as exc:  # noqa: BLE001 -- report it to the client, keep serving
                            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
