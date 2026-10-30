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
import math
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
parser.add_argument("--physics", default="isaacsim_physx")
add_launcher_args(parser)
parser.set_defaults(device="cpu", visualizer="kit")
args_cli = parser.parse_args()
if args_cli.screenshot:
    args_cli.enable_cameras = True  # the launcher only sees the physics config, not the scene's camera
if args_cli.nice > 0:
    os.nice(args_cli.nice)  # before Kit starts its threads, which inherit it

import json  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402

from cable_assembly_sim.assets import IIWA14_2F85_CFG  # noqa: E402
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_JOINT_SIGNS  # noqa: E402

# Robotiq counts -> the model's finger_joint, as scripts/bnc_cell_server.py maps them: the 2F-85's
# fingertip angle is linear in counts (cable_assembly/robot/gripper_kinematics.py, 0.2186 deg a
# count), and the model's linkage closes at 45 deg.
DEG_PER_COUNT = 0.2186
FINGER_MAX_DEG = 45.0

# GPU memory, GiB. The viewer's is measured: Kit's RTX renderer, 1.99 GiB with the GUI and about
# the same headless with --screenshot's camera; Kit takes it at start and keeps it. SAM3's is
# measured too: 4.40 GiB from its first frame on (3.33 of it weights), on the D415 test frame with
# the viewer running. The extra 0.6 is a margin.
VIEWER_GPU_GIB = 2.0
TASK_GPU_GIB = 5.0
TASK_CMD = re.compile(r"cable_assembly[./]apps")  # python -m cable_assembly.apps.bnc_pickup ...

# omni.ui colours are 0xAABBGGRR.
GREEN, AMBER, RED, GREY = 0xFF66DD66, 0xFF33BBFF, 0xFF5566FF, 0xFFAAAAAA


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


def load_cell(name):
    path = os.path.join(REPO, "configs", "cells", f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}, path


class Stream:
    """The task's datagrams (cable_assembly/robot/live_view.py), read without ever blocking."""

    def __init__(self, bind, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port))
        self.sock.setblocking(False)
        self.latest = None  # the newest valid sample
        self.sender = None
        self.t_rx = None  # time.monotonic() it arrived
        self._arrivals = []  # for the rate shown
        self._rejected = set()

    def poll(self):
        """Drain the socket, keep the newest sample this model can show. True if one arrived."""
        got = False
        while True:
            try:
                data, addr = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                return got
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            why = self._unusable(msg)
            if why:
                if why not in self._rejected:  # once each, not 60 times a second
                    self._rejected.add(why)
                    print(f"[live] ignoring the stream from {addr[0]}:{addr[1]}: {why}", flush=True)
                continue
            now = time.monotonic()
            if self.sender != addr:
                print(f"[live] receiving {msg['robot']} from {addr[0]}:{addr[1]}"
                      f"{' (a DRY RUN)' if msg.get('dry_run') else ''}", flush=True)
            self.latest, self.sender, self.t_rx = msg, addr, now
            self._arrivals.append(now)
            got = True

    @staticmethod
    def _unusable(msg):
        if not isinstance(msg, dict) or msg.get("v") != 1:
            return f"not a version-1 live-view sample ({str(msg)[:60]})"
        if msg.get("robot") != "iiwa14" or msg.get("joints") != ARM_JOINTS:
            return f"this model is the iiwa 14 ({ARM_JOINTS}); the task drives a {msg.get('robot')} ({msg.get('joints')})"
        q = msg.get("q")
        if not isinstance(q, list) or len(q) != 7 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in q):
            return f"bad joint vector {q}"
        return None

    def rate_hz(self):
        now = time.monotonic()
        self._arrivals = [t for t in self._arrivals if now - t < 1.0]
        return len(self._arrivals)

    def close(self):
        self.sock.close()


class Panel:
    """A small Kit window: is the stream live, the task's phase, the joints, the gripper. Docked
    over the Property panel (nothing to edit here), so it never covers the arm."""

    def __init__(self, title):
        from omni import ui

        self.window = ui.Window(title, width=560, height=170)
        self.window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        with self.window.frame, ui.VStack(spacing=4, height=0):
            self.status = ui.Label("", style={"font_size": 20})
            self.phase = ui.Label("", style={"font_size": 20})
            self.joints = ui.Label("", style={"font_size": 16})
            self.gripper = ui.Label("", style={"font_size": 16})
        self._last = None

    def show(self, status, color, phase, joints, gripper):
        content = (status, color, phase, joints, gripper)
        if content == self._last:
            return
        self._last = content
        self.status.text, self.status.style = status, {"font_size": 20, "color": color}
        self.phase.text, self.joints.text, self.gripper.text = phase, joints, gripper


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


def gpu_users():
    """(free GiB, [(GiB, pid, command)]) on GPU 0 -- where Kit and SAM3 both run -- or None
    without nvidia-smi."""

    def smi(query):
        out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return [[f.strip() for f in line.split(",")] for line in out.splitlines() if line.strip()]

    try:
        uuid, free = smi("--query-gpu=uuid,memory.free")[0]
        users = []
        for gpu, pid, mib in smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
            if gpu != uuid:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = f.read().decode(errors="replace").strip("\0").split("\0")
                cmd = " ".join([os.path.basename(argv[0]), *argv[1:]])  # python, not its venv path
            except OSError:
                cmd = "?"
            users.append((float(mib) / 1024.0, int(pid), cmd))
        return float(free) / 1024.0, users
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def gpu_shortfall(free_gib, users, task_gib):
    """Why the viewer must not start, or None. Kit takes VIEWER_GPU_GIB at start and keeps it. The
    task loads SAM3 as it starts and needs up to `task_gib` by its first frame, so a viewer started
    into a GPU that cannot hold both leaves the task to die of torch.OutOfMemoryError. Started
    after the task, the viewer only has to leave what the task has not taken yet."""
    task_has = sum(gib for gib, _, cmd in users if TASK_CMD.search(cmd))
    if free_gib >= VIEWER_GPU_GIB + max(task_gib - task_has, 0.0):
        return None
    lines = [(f"not starting. The GPU has {free_gib:.1f} GiB free; this viewer takes {VIEWER_GPU_GIB:.1f}, "
              f"and the task's SAM3 needs up to {task_gib:.1f} (it holds {task_has:.1f} now). The task "
              "would run out of GPU memory. Using the GPU:")]
    lines += [f"    {gib:4.1f} GiB  pid {pid:<8d} {cmd[:140]}" for gib, pid, cmd in sorted(users, reverse=True)]
    lines.append("Stop one of them, run the viewer on another PC (sim/README.md), or pass --task_gpu_gib 0 to start anyway.")
    return "\n".join(lines)


def main():
    task_gib = args_cli.task_gpu_gib
    if task_gib is None:  # a task on another PC does not share this GPU
        task_gib = TASK_GPU_GIB if args_cli.bind in ("127.0.0.1", "localhost") else 0.0
    if task_gib > 0.0:
        gpu = gpu_users()
        if gpu is None:
            print("[live] nvidia-smi failed: GPU memory not checked.", flush=True)
        elif why := gpu_shortfall(*gpu, task_gib):
            raise SystemExit(f"[live] {why}")

    cell, cell_path = load_cell(args_cli.cell)
    lv = cell.get("live_view") or {}
    port = args_cli.port or int(lv.get("port", 6008))
    bench_z = float((cell.get("ground_plane") or {}).get("z_mm", 0.0)) / 1000.0
    stream = Stream(args_cli.bind, port)

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
        zeros = torch.zeros_like(joint_pos)
        finger = 0.0
        panel = Panel(f"Live: iiwa 14 ({args_cli.cell})") if sim.has_gui else None
        print(f"[live] {cell_path}: floor at {bench_z * 1000:.1f} mm. Listening on udp://{args_cli.bind}:{port}; "
              f"start the task (live_view.enabled) in any order.", flush=True)

        t_start = time.monotonic()
        live = ever_live = False
        while sim.is_running():
            t0 = time.monotonic()
            if args_cli.duration_s is not None and t0 - t_start > args_cli.duration_s:
                break
            stream.poll()
            msg = stream.latest
            age = None if stream.t_rx is None else t0 - stream.t_rx
            if msg is not None:
                joint_pos[0, arm_ids] = torch.tensor(msg["q"], device=sim.device, dtype=joint_pos.dtype)
                if msg.get("gripper") is not None:
                    finger = math.radians(min(FINGER_MAX_DEG, max(0.0, float(msg["gripper"])) * DEG_PER_COUNT))
                joint_pos[0, grip_ids] = grip_signs * finger
                # Set, not driven: the state written, and the same targets, so nothing moves it.
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=zeros)
                robot.set_joint_position_target_index(target=joint_pos)
                scene.write_data_to_sim()
                sim.forward()

            now_live = age is not None and age < args_cli.stale_s
            if now_live != live and msg is not None and (ever_live or not now_live):
                print(f"[live] {'stream back' if now_live else f'stream lost: no sample for {args_cli.stale_s:.1f} s'}", flush=True)
            live = now_live
            ever_live |= live
            if panel is not None:
                if msg is None:
                    status, color = f"waiting for the task on udp://{args_cli.bind}:{port}", GREY
                elif live:
                    status, color = f"LIVE  {stream.rate_hz()} Hz from {stream.sender[0]}", GREEN
                    if msg.get("dry_run"):
                        status, color = status + "  (DRY RUN)", AMBER
                else:
                    status, color = f"NO DATA for {age:.1f} s -- showing the last pose", RED
                q_deg = " ".join(f"{math.degrees(x):+6.1f}" for x in msg["q"]) if msg else "--"
                g = msg.get("gripper") if msg else None
                panel.show(status, color,
                           f"phase: {msg.get('phase') or '--'}" if msg else "phase: --",
                           f"A1..A7 [deg]  {q_deg}",
                           f"gripper: {g} counts ({math.degrees(finger):.0f} deg)" if g is not None else "gripper: --")
            sim.render()
            dt = t0 + 1.0 / args_cli.fps - time.monotonic()
            if dt > 0.0:
                time.sleep(dt)

        if cam is not None:
            screenshot(sim, cam, args_cli.screenshot)
        stream.close()


if __name__ == "__main__":
    main()
