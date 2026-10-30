"""LIVE VIEW, LIGHT: the real iiwa 14 mirrored in Newton's OpenGL viewer while a task runs on it.

    cd sim
    ./run.sh python scripts/live_view_gl.py               # opens a window, listens on UDP 6008
    # in the `cable` env, any order, any time:
    python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench

The same stream and the same iiwa 14 + 2F-85 model as scripts/live_view.py, drawn by Newton's
ViewerGL instead of Isaac Sim: no Kit and no RTX renderer, so it starts in seconds and leaves the
GPU to SAM3. The model is set from the reported joints by forward kinematics on the CPU (Warp);
nothing is simulated. Newton's FK puts tool0 where cable_assembly's own iiwa14 FK does.

What it gives up against the Isaac Sim viewer: plain OpenGL shading, and no Gaussian splats (Newton
draws a Gaussian as a sphere). For those, run scripts/live_view.py.

The panel at the top says LIVE (with the rate), or NO DATA and for how long, with the task's phase,
the joints and the gripper. Mouse: left-drag orbits, right-drag pans, wheel zooms; WASD moves.

    ./run.sh python scripts/live_view_gl.py --fps 30                 # half the CPU (0.33 core, not 0.66)
    ./run.sh python scripts/live_view_gl.py --bind 0.0.0.0           # accept a task on another PC
    ./run.sh python scripts/live_view_gl.py --headless --duration_s 10 --screenshot /tmp/live_gl.png
    ./run.sh python scripts/live_view_gl.py --rgbd latest            # the newest splat scan / capture
    ./run.sh python scripts/live_view_gl.py --rgbd ../data/experiments/splat_scan_<stamp>   # a run of choice

THE CAMERA LAYER: what the wrist camera has seen, as coloured spheres where the depth puts its
pixels in the base frame (src/cable_assembly_sim/rgbd.py), with the newest view's cone in yellow.
Frames come from the task as it captures them (live_view.frames in the cell, over TCP 6009) and
from --rgbd captures (cable_assembly.apps.capture_rgbd) at start. A new frame replaces what it
sees: where the cable lay and is gone, the bench takes its place; what the camera cannot see now
-- outside its view, behind the fingers -- stays as last seen. The gripper is always left out.
Until a task streams, the arm stands at the joints the last --rgbd frame was taken at; the viewer
prints the height and tilt of the surface those points lie on (the bench's, on a good hand-eye).
"""

import argparse
import math
import os
import time

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--cell", default="iiwa14_bench", help="configs/cells/<cell>.yaml: the bench height and live_view.port")
parser.add_argument("--port", type=int, default=None, help="UDP port to listen on (default: the cell's live_view.port)")
parser.add_argument("--bind", default="127.0.0.1", help="address to listen on; 0.0.0.0 for a task on another PC")
parser.add_argument("--fps", type=float, default=60.0, help="frame-rate cap")
parser.add_argument("--stale_s", type=float, default=0.5, help="no sample for this long = the stream is lost")
parser.add_argument("--nice", type=int, default=10, help="lower this process's CPU priority by this much (0 = no)")
parser.add_argument("--eye", type=float, nargs=3, default=(1.45, -1.1, 1.05), help="camera position, m")
parser.add_argument("--lookat", type=float, nargs=3, default=(0.35, 0.05, 0.35), help="camera target, m")
parser.add_argument("--size", type=int, nargs=2, default=(1600, 900), help="window size, px")
parser.add_argument("--headless", action="store_true", default=os.environ.get("HEADLESS") == "1", help="no window (or HEADLESS=1)")
parser.add_argument("--duration_s", type=float, default=None, help="exit after this long (a smoke test)")
parser.add_argument("--screenshot", default=None, help="on exit, save the view to this .png")
parser.add_argument("--rgbd", action="append", default=[], metavar="NPZ", help="a captured RGB-D frame (frame.npz / view_NN.npz), a run directory (all its views), or `latest`, to start the camera layer with; repeatable")
parser.add_argument("--rgbd_stride", type=int, default=2, help="use every n-th pixel of a --rgbd capture (the task's frames come at half resolution already)")
parser.add_argument("--voxel_mm", type=float, default=2.0, help="the camera layer keeps one point per cube this size")
parser.add_argument("--max_points", type=int, default=400_000, help="past this many, the camera layer drops its oldest frames' points")
parser.add_argument("--frames_port", type=int, default=None, help="TCP port for the task's camera frames (default: the cell's live_view.frames_port)")
parser.add_argument("--no_frames", action="store_true", help="do not listen for the task's camera frames")
parser.add_argument("--cuda", action="store_true", help="let Warp use the GPU's CUDA (default: hidden, no CUDA context)")
parser.add_argument("--task_gpu_gib", type=float, default=None, help="GPU memory to leave for the task's SAM3, GiB (default 5.0; 0 when --bind is not loopback: the task is on another PC; 0 = don't check)")
args = parser.parse_args()
if not args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before Warp loads: the FK runs on the CPU anyway
if args.nice > 0:
    os.nice(args.nice)

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402
from newton.viewer import ViewerGL  # noqa: E402

from cable_assembly_sim.assets import CABLE_ASSEMBLY_SIM_ASSETS_DIR  # noqa: E402
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_JOINT_SIGNS, HOME_DEG  # noqa: E402
from cable_assembly_sim import rgbd  # noqa: E402
from cable_assembly_sim.live import FRAMES_PORT, FrameReceiver, Overlays, Stream, check_gpu, finger_rad, load_cell  # noqa: E402

# This viewer's GPU memory, GiB, measured: 386 MiB for its OpenGL window and buffers; no CUDA context.
VIEWER_GPU_GIB = 0.4
# Vertical field of view, deg: the Isaac Sim viewport's 18.15 mm lens on its 16:9 aperture.
FOV_DEG = 36.0
# The panel's status colour by Stream.describe()'s state, RGBA.
COLORS = {"live": (0.4, 0.87, 0.4, 1.0), "dry_run": (1.0, 0.73, 0.2, 1.0),
          "stale": (1.0, 0.4, 0.33, 1.0), "waiting": (0.67, 0.67, 0.67, 1.0)}


def build_model(bench_z):
    """The iiwa 14 + 2F-85 (the USD Isaac Sim uses) on a floor at the bench height, on the CPU."""
    builder = newton.ModelBuilder()
    builder.add_usd(str(CABLE_ASSEMBLY_SIM_ASSETS_DIR / "iiwa14_2f85.usda"), collapse_fixed_joints=False,
                    enable_self_collisions=False, hide_collision_shapes=True)
    builder.add_ground_plane(height=bench_z)
    return builder.finalize(device="cpu")


def joint_slots(model, names):
    """Each named joint's index in model.joint_q (labels are USD paths: /iiwa14/Physics/A1, ...)."""
    starts = model.joint_q_start.numpy()
    by_name = {label.rsplit("/", 1)[-1]: int(starts[j]) for j, label in enumerate(model.joint_label)}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"[live] the model has no joints {missing}: {sorted(by_name)}")
    return np.array([by_name[n] for n in names])


def aim(viewer, eye, lookat):
    """Point Newton's Z-up camera from `eye` at `lookat` (it takes a position, pitch and yaw)."""
    d = np.asarray(lookat, dtype=float) - np.asarray(eye, dtype=float)
    d /= np.linalg.norm(d)
    viewer.set_camera(wp.vec3(*map(float, eye)), math.degrees(math.asin(d[2])), math.degrees(math.atan2(d[1], d[0])))
    viewer.camera.fov = FOV_DEG


class Panel:
    """The status lines in their own small window, right of Newton's settings panel (whose
    sections would bury them). H hides all the UI, this window included."""

    def __init__(self, title):
        self.title, self.content = title, None

    def draw(self, imgui):
        if self.content is None:
            return
        state, status, *lines = self.content
        imgui.set_next_window_pos(imgui.ImVec2(330, 12), imgui.Cond_.first_use_ever)
        imgui.begin(self.title, None, imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_collapse)
        imgui.text_colored(imgui.ImVec4(*COLORS[state]), status)
        for line in lines:
            imgui.text(line)
        imgui.end()


class CameraLayer(rgbd.CameraLayer):
    """The camera layer (rgbd.CameraLayer), drawn as one sphere per point with the newest frame's
    view cone."""

    def __init__(self):
        super().__init__(voxel_m=args.voxel_mm / 1000.0, max_points=args.max_points)
        self.drawn = -1

    def draw(self, viewer):
        """Hand the layer to the viewer when it has changed; the viewer keeps drawing it."""
        p, c, sizes, _, version, last = self.map.snapshot()
        if version == self.drawn:
            return
        self.drawn = version
        if len(p):
            viewer.log_points("/camera", wp.array(p, dtype=wp.vec3, device="cpu"),
                              wp.array(0.5 * sizes, dtype=wp.float32, device="cpu"),
                              wp.array(c, dtype=wp.vec3, device="cpu"))
        else:
            viewer.log_points("/camera", None)
        if last is not None:
            starts, ends = rgbd.frustum(last)
            viewer.log_lines("/camera/view", wp.array(starts, dtype=wp.vec3, device="cpu"),
                             wp.array(ends, dtype=wp.vec3, device="cpu"), (1.0, 0.85, 0.2))


class OverlayDrawer:
    """The task's overlays (live.Overlays): points as spheres, polylines and pose axes as lines,
    each item under its overlay's name, redrawn when they change."""

    AXES = ((1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.3, 0.5, 1.0))

    def __init__(self, overlays):
        self.overlays, self.drawn, self.names = overlays, -1, set()

    def draw(self, viewer):
        by_name, version = self.overlays.snapshot()
        if version == self.drawn:
            return
        self.drawn = version
        names = set()
        for name, overlay in by_name.items():
            for i, item in enumerate(overlay.get("items", [])):
                key = f"/overlay/{name}/{i}"
                if self._draw(viewer, key, item):
                    names.add(key)
        for key in self.names - names:                     # gone from a newer overlay: hide
            viewer.log_points(key, None)
            viewer.log_lines(key, None, None, None)
        self.names = names

    def _draw(self, viewer, key, item):
        kind = item.get("type")
        if kind == "points" and item.get("points"):
            p = np.asarray(item["points"], dtype=np.float32).reshape(-1, 3)
            # log_points takes colours only as one per point (log_lines also takes one for all).
            colors = wp.full(len(p), wp.vec3(*item.get("color", (1.0, 1.0, 1.0))), dtype=wp.vec3, device="cpu")
            viewer.log_points(key, wp.array(p, dtype=wp.vec3, device="cpu"), float(item.get("radius", 0.003)), colors)
            return True
        if kind == "line" and len(item.get("points", [])) >= 2:
            p = np.asarray(item["points"], dtype=np.float32).reshape(-1, 3)
            viewer.log_lines(key, wp.array(p[:-1], dtype=wp.vec3, device="cpu"), wp.array(p[1:], dtype=wp.vec3, device="cpu"),
                             tuple(item.get("color", (1.0, 1.0, 1.0))))
            return True
        if kind == "pose":
            T = np.asarray(item["T"], dtype=np.float32)
            n = float(item.get("length", 0.03))
            starts = np.repeat(T[None, :3, 3], 3, axis=0)
            ends = starts + n * T[:3, :3].T
            colors = np.asarray(self.AXES, dtype=np.float32)
            viewer.log_lines(key, wp.array(starts, dtype=wp.vec3, device="cpu"), wp.array(ends, dtype=wp.vec3, device="cpu"),
                             wp.array(colors, dtype=wp.vec3, device="cpu"))
            return True
        return False


def main():
    check_gpu(args.task_gpu_gib, args.bind, VIEWER_GPU_GIB)
    cell, cell_path = load_cell(args.cell)
    port = args.port or int((cell.get("live_view") or {}).get("port", 6008))
    bench_z = float((cell.get("ground_plane") or {}).get("z_mm", 0.0)) / 1000.0
    camera = CameraLayer()
    overlays = Overlays()
    drawer = OverlayDrawer(overlays)
    capture_q = camera.seed(rgbd.expand(args.rgbd), bench_z, stride=args.rgbd_stride)
    stream = Stream(args.bind, port, ARM_JOINTS)
    frames = None
    if not args.no_frames:
        frames_port = args.frames_port or int((cell.get("live_view") or {}).get("frames_port", FRAMES_PORT))
        try:
            frames = FrameReceiver(args.bind, frames_port, camera.add, overlays.put)
        except OSError as exc:
            raise SystemExit(f"[live] cannot listen for camera frames on tcp://{args.bind}:{frames_port} ({exc}): "
                             "is another viewer running? (--no_frames, or --frames_port)") from None

    model = build_model(bench_z)
    state = model.state()
    arm = joint_slots(model, ARM_JOINTS)
    grip = joint_slots(model, list(GRIPPER_JOINT_SIGNS))
    grip_signs = np.array(list(GRIPPER_JOINT_SIGNS.values()))
    q = model.joint_q.numpy().copy()
    # Until the first sample: where the (last) captured frame was taken, so the arm, the camera's
    # view cone and the scene it saw can be compared; else HOME, as the Isaac Sim viewer starts.
    q[arm] = capture_q if capture_q is not None else np.radians(HOME_DEG)
    joint_q = wp.array(q, dtype=float, device="cpu")
    newton.eval_fk(model, joint_q, model.joint_qd, state)
    finger = 0.0

    viewer = ViewerGL(width=args.size[0], height=args.size[1], headless=args.headless,
                      enable_cuda_interop=ViewerGL.CudaInterop.NONE)
    viewer.set_model(model)
    aim(viewer, args.eye, args.lookat)
    panel = Panel(f"Live: iiwa 14 ({args.cell})")
    viewer.register_ui_callback(panel.draw, position="free")
    print(f"[live] {cell_path}: floor at {bench_z * 1000:.1f} mm. Listening on udp://{args.bind}:{port}"
          f"{'' if frames is None else ' and, for camera frames, ' + frames.address}; "
          f"start the task (live_view.enabled) in any order.", flush=True)

    t_start = time.monotonic()
    while viewer.is_running():
        t0 = time.monotonic()
        if args.duration_s is not None and t0 - t_start > args.duration_s:
            break
        if stream.poll():
            msg = stream.latest
            q[arm] = msg["q"]
            if msg.get("gripper") is not None:
                finger = finger_rad(msg["gripper"])
            q[grip] = grip_signs * finger
            joint_q.assign(q)
            newton.eval_fk(model, joint_q, model.joint_qd, state)
        stream.check_live(args.stale_s)
        panel.content = (*stream.describe(finger), camera.describe(), *overlays.text())
        viewer.begin_frame(t0 - t_start)
        camera.draw(viewer)
        drawer.draw(viewer)
        viewer.log_state(state)
        viewer.end_frame()
        dt = t0 + 1.0 / args.fps - time.monotonic()
        if dt > 0.0:
            time.sleep(dt)

    if args.screenshot:
        from PIL import Image

        os.makedirs(os.path.dirname(os.path.abspath(args.screenshot)), exist_ok=True)
        Image.fromarray(viewer.get_frame(render_ui=not args.headless).numpy()).save(args.screenshot)
        print(f"[live] wrote {args.screenshot}", flush=True)
    viewer.close()
    stream.close()
    if frames is not None:
        frames.close()


if __name__ == "__main__":
    main()
