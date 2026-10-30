"""LIVE VIEW: the real iiwa 14 mirrored in Isaac Sim while a cable_assembly task runs on it.

    cd sim
    ./run.sh python scripts/live_view.py                 # opens the viewport, listens on UDP 6008
    # in the `cable` env, any order, any time:
    python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench

The task streams its measured joints, the Robotiq's position and its phase ~60 times a second
(cable_assembly/robot/live_view.py; the iiwa14_bench cell has `live_view.enabled`). This sets them
on the iiwa 14 + 2F-85 model (assets/iiwa14.py) every frame. It is a picture, not a simulation:
nothing is stepped, no physics acts on the arm, and the model shows exactly the joints the arm
reported. Either side can be started, stopped and restarted without the other noticing. When the
samples stop, the arm stays where it was last seen and the panel says how long ago that was.

THE SCENE: the robot at the origin of its base frame and a floor at the bench height the hand-eye
calibration measured (the cell's ground_plane.z_mm). The socket fixture, the cable and the D415 are
not modelled.

KEEPING OUT OF THE ARM'S WAY. The FRI loop runs in the task's own driver process at normal
priority (this user has no real-time limit), on the same PC. So this viewer lowers its own priority
(--nice 10), caps its frame rate (--fps 30) and runs PhysX on the CPU (--device cpu, nothing to
simulate anyway). SAM3 shares the GPU, and there the viewer could crowd it out: Kit takes 2 GiB
at start, SAM3 4.4 GiB, and the card has 10 GB. So before Kit starts, this checks that
the GPU can hold both. If not, it does not start, and it lists what is using the GPU (another Isaac
Sim, as a rule). --task_gpu_gib 0 skips the check. To keep the viewer off the robot PC, run it on
another machine with --bind 0.0.0.0 and point the cell's live_view.host at it.

    ./run.sh python scripts/live_view.py --fps 60                 # smoother, more GPU
    ./run.sh python scripts/live_view.py --bind 0.0.0.0           # accept a task on another PC
    HEADLESS=1 ./run.sh python scripts/live_view.py --duration_s 20 --screenshot /tmp/live.png
"""

import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--cell", default="iiwa14_bench", help="configs/cells/<cell>.yaml: the bench height and live_view.port")
parser.add_argument("--port", type=int, default=None, help="UDP port to listen on (default: the cell's live_view.port)")
parser.add_argument("--bind", default="127.0.0.1", help="address to listen on; 0.0.0.0 for a task on another PC")
parser.add_argument("--fps", type=float, default=30.0, help="frame-rate cap")
parser.add_argument("--stale_s", type=float, default=0.5, help="no sample for this long = the stream is lost")
parser.add_argument("--nice", type=int, default=10, help="lower this process's CPU priority by this much (0 = no)")
parser.add_argument("--eye", type=float, nargs=3, default=(1.45, -1.1, 1.05), help="viewport camera position, m")
parser.add_argument("--lookat", type=float, nargs=3, default=(0.35, 0.05, 0.35), help="viewport camera target, m")
parser.add_argument("--duration_s", type=float, default=None, help="exit after this long (a smoke test)")
parser.add_argument("--screenshot", default=None, help="on exit, save what the viewport's eye sees to this .png (a camera: works headless too)")
parser.add_argument("--task_gpu_gib", type=float, default=None, help="GPU memory to leave for the task's SAM3, GiB (default 5.0; 0 when --bind is not loopback: the task is on another PC; 0 = don't check)")
parser.add_argument("--rgbd", action="append", default=[], metavar="NPZ", help="a captured RGB-D frame (frame.npz / view_NN.npz), a run directory (all its views), or `latest`, to start the camera layer with; repeatable")
parser.add_argument("--rgbd_stride", type=int, default=2, help="use every n-th pixel of a --rgbd capture")
parser.add_argument("--voxel_mm", type=float, default=2.0, help="the camera layer keeps one Gaussian per cube this size")
parser.add_argument("--max_points", type=int, default=300_000, help="past this many, the camera layer drops its oldest frames' (each rewrite of 300k stalls the viewer ~0.1 s)")
parser.add_argument("--splat_hz", type=float, default=1.0, help="rewrite the camera layer's splat at most this often")
parser.add_argument("--frames_port", type=int, default=None, help="TCP port for the task's camera frames (default: the cell's live_view.frames_port)")
parser.add_argument("--no_frames", action="store_true", help="do not listen for the task's camera frames")
parser.add_argument("--physics", default="isaacsim_physx")
add_launcher_args(parser)
parser.set_defaults(device="cpu", visualizer="kit")
args_cli = parser.parse_args()
if args_cli.screenshot:
    args_cli.enable_cameras = True  # the launcher only sees the physics config, not the scene's camera
if args_cli.nice > 0:
    os.nice(args_cli.nice)  # before Kit starts its threads, which inherit it

import time  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402

from cable_assembly_sim.assets import IIWA14_2F85_CFG  # noqa: E402
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_JOINT_SIGNS  # noqa: E402
from cable_assembly_sim import rgbd  # noqa: E402
from cable_assembly_sim.live import FRAMES_PORT, FrameReceiver, Overlays, Stream, check_gpu, finger_rad, load_cell  # noqa: E402

# This viewer's GPU memory, GiB, measured: Kit's RTX renderer, 1.99 GiB with the GUI and about the
# same headless with --screenshot's camera. Kit takes it at start and keeps it.
VIEWER_GPU_GIB = 2.0

# The panel's status colour by Stream.describe()'s state. omni.ui colours are 0xAABBGGRR.
COLORS = {"live": 0xFF66DD66, "dry_run": 0xFF33BBFF, "stale": 0xFF5566FF, "waiting": 0xFFAAAAAA}


@configclass
class LiveSceneCfg(InteractiveSceneCfg):
    floor = AssetBaseCfg(prim_path="/World/floor", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = replace(IIWA14_2F85_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class LiveSceneWithCameraCfg(LiveSceneCfg):
    """... and a camera at the viewport's eye, for --screenshot (a headless Kit has no viewport)."""

    shot_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ShotCam",
        update_period=0.0,
        width=1280,
        height=720,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.01, 20.0)),  # the viewport's lens
    )


class Panel:
    """A small Kit window: is the stream live, the task's phase, the joints, the gripper. Docked
    over the Property panel (nothing to edit here), so it never covers the arm."""

    def __init__(self, title):
        from omni import ui

        self.window = ui.Window(title, width=560, height=260)
        self.window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        with self.window.frame, ui.VStack(spacing=4, height=0):
            self.status = ui.Label("", style={"font_size": 20})
            self.phase = ui.Label("", style={"font_size": 20})
            self.joints = ui.Label("", style={"font_size": 16})
            self.gripper = ui.Label("", style={"font_size": 16})
            self.camera = ui.Label("", style={"font_size": 16})
        self._last = None

    def show(self, status, color, phase, joints, gripper, camera=""):
        content = (status, color, phase, joints, gripper, camera)
        if content == self._last:
            return
        self._last = content
        self.status.text, self.status.style = status, {"font_size": 20, "color": color}
        self.phase.text, self.joints.text, self.gripper.text, self.camera.text = phase, joints, gripper, camera


def screenshot(sim, cam, path):
    """What the viewport's eye sees, to a .png."""
    from PIL import Image

    sim.render()
    cam.update(sim.get_physics_dt(), force_recompute=True)
    rgb = cam.data.output["rgb"]
    rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[0, ..., :3].cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(rgb.astype("uint8")).save(path)
    print(f"[live] wrote {path}", flush=True)


def main():
    check_gpu(args_cli.task_gpu_gib, args_cli.bind, VIEWER_GPU_GIB)

    cell, cell_path = load_cell(args_cli.cell)
    lv = cell.get("live_view") or {}
    port = args_cli.port or int(lv.get("port", 6008))
    bench_z = float((cell.get("ground_plane") or {}).get("z_mm", 0.0)) / 1000.0
    stream = Stream(args_cli.bind, port, ARM_JOINTS)
    camera = rgbd.CameraLayer(voxel_m=args_cli.voxel_mm / 1000.0, max_points=args_cli.max_points)
    capture_q = camera.seed(rgbd.expand(args_cli.rgbd), bench_z, stride=args_cli.rgbd_stride)
    overlays = Overlays()  # the task's findings: their text in the panel (drawn only in live_view_gl.py)
    frames = None
    if not args_cli.no_frames:
        frames_port = args_cli.frames_port or int(lv.get("frames_port", FRAMES_PORT))
        try:
            frames = FrameReceiver(args_cli.bind, frames_port, camera.add, overlays.put)
        except OSError as exc:
            raise SystemExit(f"[live] cannot listen for camera frames on tcp://{args_cli.bind}:{frames_port} ({exc}): "
                             "is another viewer running? (--no_frames, or --frames_port)") from None

    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / args_cli.fps, device=args_cli.device, physics=physics_cfg))
        scene_cfg = (LiveSceneWithCameraCfg if args_cli.screenshot else LiveSceneCfg)(num_envs=1, env_spacing=2.0)
        scene_cfg.floor.init_state.pos = (0.0, 0.0, bench_z)
        scene = instantiate(scene_cfg)
        sim.reset()
        origin = scene.env_origins[0].cpu().numpy()
        eye, lookat = origin + args_cli.eye, origin + args_cli.lookat
        sim.set_camera_view(tuple(eye), tuple(lookat))
        cam = scene["shot_cam"] if args_cli.screenshot else None
        if cam is not None:
            cam.set_world_poses_from_view(torch.tensor([eye], device=sim.device, dtype=torch.float32),
                                          torch.tensor([lookat], device=sim.device, dtype=torch.float32))

        robot = scene["robot"]
        arm_ids, names = robot.find_joints(ARM_JOINTS, preserve_order=True)
        assert names == ARM_JOINTS, names
        grip_ids, grip_names = robot.find_joints(list(GRIPPER_JOINT_SIGNS), preserve_order=True)
        grip_signs = torch.tensor([GRIPPER_JOINT_SIGNS[n] for n in grip_names], device=sim.device)
        joint_pos = robot.data.default_joint_pos.torch.clone()
        if capture_q is not None:  # until the first sample: where the last --rgbd frame was taken
            joint_pos[0, arm_ids] = torch.tensor(capture_q, device=sim.device, dtype=joint_pos.dtype)
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.set_joint_position_target_index(target=joint_pos)
            scene.write_data_to_sim()
            sim.forward()
        zeros = torch.zeros_like(joint_pos)
        finger = 0.0
        panel = Panel(f"Live: iiwa 14 ({args_cli.cell})") if sim.has_gui else None
        # The camera layer as a Gaussian splat. splat imports pxr: only now that Kit is up.
        import omni.usd

        from cable_assembly_sim import splat

        layer_prim = splat.define(omni.usd.get_context().get_stage(), "/World/CameraLayer")
        drawn, t_drawn = -1, 0.0
        print(f"[live] {cell_path}: floor at {bench_z * 1000:.1f} mm. Listening on udp://{args_cli.bind}:{port}; "
              f"start the task (live_view.enabled) in any order.", flush=True)

        t_start = time.monotonic()
        while sim.is_running():
            t0 = time.monotonic()
            if args_cli.duration_s is not None and t0 - t_start > args_cli.duration_s:
                break
            stream.poll()
            msg = stream.latest
            if msg is not None:
                joint_pos[0, arm_ids] = torch.tensor(msg["q"], device=sim.device, dtype=joint_pos.dtype)
                if msg.get("gripper") is not None:
                    finger = finger_rad(msg["gripper"])
                joint_pos[0, grip_ids] = grip_signs * finger
                # Set, not driven: the state written, and the same targets, so nothing moves it.
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=zeros)
                robot.set_joint_position_target_index(target=joint_pos)
                scene.write_data_to_sim()
                sim.forward()

            stream.check_live(args_cli.stale_s)
            if camera.map.version != drawn and t0 - t_drawn >= 1.0 / args_cli.splat_hz:
                arrays, drawn = camera.splat_arrays()
                splat.write(layer_prim, **arrays)
                t_drawn = t0
            if panel is not None:
                state, *lines = stream.describe(finger)
                panel.show(lines[0], COLORS[state], *lines[1:], "\n".join([camera.describe(), *overlays.text()]))
            sim.render()
            dt = t0 + 1.0 / args_cli.fps - time.monotonic()
            if dt > 0.0:
                time.sleep(dt)

        if cam is not None:
            screenshot(sim, cam, args_cli.screenshot)
        stream.close()
        if frames is not None:
            frames.close()


if __name__ == "__main__":
    main()
