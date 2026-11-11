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
calibration measured (the cell's ground_plane.z_mm). The socket fixture and the cable are not
modelled; the connector in the hand is, once the task holds it (its `held`: the CAD at the pose the
grasp expects, drawn under tool0 -- HeldPrim).

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
parser.add_argument("--splat", choices=("2dgs", "3dgs"), default="2dgs", help="the camera layer's Gaussians: 2dgs surfels lying on the surface (normals from depth), or 3dgs discs facing the camera (src/cable_assembly_sim/rgbd.py)")
parser.add_argument("--splat_hz", type=float, default=1.0, help="rewrite the camera layer's splat at most this often")
parser.add_argument("--frames_port", type=int, default=None, help="TCP port for the task's camera frames (default: the cell's live_view.frames_port)")
parser.add_argument("--no_frames", action="store_true", help="do not listen for the task's camera frames")
parser.add_argument("--no_camera_window", action="store_true", help="no window with the camera's newest frame")
parser.add_argument("--camera_window_px", type=int, default=480, help="the camera window's image width, px")
parser.add_argument("--physics", default="isaacsim_physx")
add_launcher_args(parser)
parser.set_defaults(device="cpu", visualizer="kit")
args_cli = parser.parse_args()
if args_cli.screenshot:
    args_cli.enable_cameras = True  # the launcher only sees the physics config, not the scene's camera
if args_cli.nice > 0:
    os.nice(args_cli.nice)  # before Kit starts its threads, which inherit it

import time  # noqa: E402

import numpy as np  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402

from cable_assembly_sim.assets.iiwa14 import bench_cfg  # noqa: E402
from cable_assembly_sim.assets.iiwa14 import ARM_JOINTS, GRIPPER_JOINT_SIGNS  # noqa: E402
from cable_assembly_sim import rgbd  # noqa: E402
from cable_assembly_sim.live import FRAMES_PORT, FrameReceiver, LatestImage, NamedImages, Overlays, Stream, check_gpu, finger_rad, load_cell  # noqa: E402

# This viewer's GPU memory, GiB, measured: Kit's RTX renderer, 1.99 GiB with the GUI and about the
# same headless with --screenshot's camera. Kit takes it at start and keeps it.
VIEWER_GPU_GIB = 2.0

# The panel's status colour by Stream.describe()'s state. omni.ui colours are 0xAABBGGRR.
COLORS = {"live": 0xFF66DD66, "dry_run": 0xFF33BBFF, "stale": 0xFF5566FF, "waiting": 0xFFAAAAAA}


@configclass
class LiveSceneCfg(InteractiveSceneCfg):
    floor = AssetBaseCfg(prim_path="/World/floor", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = replace(bench_cfg(), prim_path="{ENV_REGEX_NS}/Robot")  # the bench's V3 gripper, as it is


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
    over the Property panel (nothing to edit here), so it never covers the arm.

    ITS BUTTONS: Reset puts the viewer back as it started (main's reset: the camera layer, the overlays,
    the held part, the images and the last sample forgotten; the arm at its start pose; the viewport
    camera back) -- a live task's next sample and its next frames draw it again. Reset view moves only
    the viewport camera back. A click sets a request the render loop takes (take())."""

    def __init__(self, title):
        from omni import ui

        self._asked = set()
        self.window = ui.Window(title, width=560, height=300)
        self.window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        with self.window.frame, ui.VStack(spacing=4, height=0):
            self.status = ui.Label("", style={"font_size": 20})
            # the buttons first: the lines below grow with the overlays' text
            with ui.HStack(spacing=8, height=28):
                ui.Button("Reset", tooltip="Clear the camera layer, overlays, held part and images; the arm to its "
                          "start pose; the view back", clicked_fn=lambda: self._asked.add("reset"))
                ui.Button("Reset view", tooltip="The viewport camera back to --eye / --lookat",
                          clicked_fn=lambda: self._asked.add("view"))
            self.phase = ui.Label("", style={"font_size": 20})
            self.joints = ui.Label("", style={"font_size": 16})
            self.gripper = ui.Label("", style={"font_size": 16})
            self.camera = ui.Label("", style={"font_size": 16})
        self._last = None

    def take(self, what):
        """Whether `what` ('reset' | 'view') was clicked since the last take."""
        if what in self._asked:
            self._asked.discard(what)
            return True
        return False

    def show(self, status, color, phase, joints, gripper, camera=""):
        content = (status, color, phase, joints, gripper, camera)
        if content == self._last:
            return
        self._last = content
        self.status.text, self.status.style = status, {"font_size": 20, "color": color}
        self.phase.text, self.joints.text, self.gripper.text, self.camera.text = phase, joints, gripper, camera


class CameraWindow:
    """The wrist camera's newest frame (live.LatestImage) in a Kit window of its own, its phase and age
    under it; the image re-sent only when a newer frame has come. A failure turns it off, said once."""

    def __init__(self, image, width=480):
        from omni import ui

        self.image, self.shown, self.ok = image, -1, True
        self.window = ui.Window("Camera", width=width + 20, height=int(width * 0.6) + 60)
        self.provider = ui.ByteImageProvider()
        with self.window.frame, ui.VStack(spacing=4):
            ui.ImageWithProvider(self.provider, width=width, height=int(width * 9 / 16))
            self.label = ui.Label("no frame yet: the task sends one each time it captures")

    def reset(self):
        """Blank (the viewer's reset), until the next frame."""
        if not self.ok:
            return
        try:
            blank = np.zeros((9, 16, 4), np.uint8)
            blank[..., 3] = 255
            if hasattr(self.provider, "set_data_array"):
                self.provider.set_data_array(blank, [16, 9])
            else:
                self.provider.set_bytes_data(blank.ravel().tolist(), [16, 9])
            self.label.text = "no frame yet: the task sends one each time it captures"
        except Exception as exc:    # noqa: BLE001 -- the window only
            print(f"[live] the camera window is off: {exc}", flush=True)
            self.ok = False

    def update(self):
        if not self.ok:
            return
        rgb, phase, age, version = self.image.get()
        if rgb is None:
            return
        try:
            if version != self.shown:
                h, w = rgb.shape[:2]
                rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
                if hasattr(self.provider, "set_data_array"):
                    self.provider.set_data_array(rgba, [w, h])
                else:
                    self.provider.set_bytes_data(rgba.ravel().tolist(), [w, h])
                self.shown = version
            self.label.text = f"{phase or '--'}, {age:.1f} s ago"
        except Exception as exc:    # noqa: BLE001 -- the window only
            print(f"[live] the camera window is off: {exc}", flush=True)
            self.ok = False


class NamedImageWindows:
    """The task's named images (live.NamedImages: the behavior tree with the mode and autonomy state, the
    in-hand CAD fit while it searches), each in a Kit window of its own name, made when the first comes."""

    def __init__(self, images, width=560):
        self.images, self.width, self.windows, self.shown, self.ok = images, int(width), {}, {}, True

    def reset(self):
        """Hide every window (the viewer's reset); each shows again when its next image comes."""
        for win, _prov, _label in self.windows.values():
            win.visible = False
        self.shown.clear()

    def update(self):
        if not self.ok:
            return
        try:
            from omni import ui

            for name, rgb, age, version in self.images.items():
                if rgb is None:
                    continue
                h, w = rgb.shape[:2]
                if name not in self.windows:
                    win = ui.Window(name, width=self.width + 20, height=int(self.width * h / w) + 50)
                    prov = ui.ByteImageProvider()
                    with win.frame, ui.CollapsableFrame(name, collapsed=False), ui.VStack(spacing=4):
                        ui.ImageWithProvider(prov, width=self.width, height=int(self.width * h / w))
                        label = ui.Label("")
                    self.windows[name] = (win, prov, label)
                win, prov, label = self.windows[name]
                if self.shown.get(name) != version:
                    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
                    if hasattr(prov, "set_data_array"):
                        prov.set_data_array(rgba, [w, h])
                    else:
                        prov.set_bytes_data(rgba.ravel().tolist(), [w, h])
                    self.shown[name] = version
                    win.visible = True
                label.text = f"{age:.1f} s ago"
        except Exception as exc:    # noqa: BLE001 -- the windows only
            print(f"[live] the named-image windows are off: {exc}", flush=True)
            self.ok = False


class HeldPrim:
    """The part in the hand -- the task's `held` (cable_assembly Robot.hold: the connector after the
    pick, at the pose the grasp expects it at) -- its CAD as a mesh prim under the robot's tool0,
    at T_tool0, so it moves with the gripper drawn (as the camera mount does). Rebuilt only when
    the part changes; hidden when the task lets go. The mesh is read from its path (trimesh); a
    path this PC does not have (a task on another PC) is said once and skipped."""

    COLOR = (0.78, 0.78, 0.82)          # nickel-plated, as live_view_gl.py's

    def __init__(self, stage):
        from pxr import UsdGeom

        tool0 = next((q for q in stage.Traverse() if q.GetName() == "tool0" and "/Robot" in str(q.GetPath())), None)
        if tool0 is None:
            print("[live] no tool0 prim under the robot: the held part will not be drawn.", flush=True)
        self.stage, self.path = stage, None if tool0 is None else tool0.GetPath().AppendChild("held_part")
        self.shown, self.failed = None, set()
        self._UsdGeom = UsdGeom

    def update(self, held):
        if self.path is None:
            return
        key = None if not held else (held["mesh"], float(held.get("scale", 1.0)),
                                     tuple(np.asarray(held["T_tool0"], dtype=float).ravel()))
        if key == self.shown:
            return
        UsdGeom = self._UsdGeom
        prim = self.stage.GetPrimAtPath(self.path)
        if key is None or key[0] in self.failed:
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
            self.shown = key
            return
        from pxr import Gf, Vt

        if self.shown is None or self.shown[:2] != key[:2] or not prim.IsValid():
            try:
                import trimesh

                m = trimesh.load(key[0], process=False, force="mesh")
            except Exception as exc:    # noqa: BLE001 -- only this part goes undrawn
                print(f"[live] the held part's mesh {key[0]}: {exc} -- not drawn.", flush=True)
                self.failed.add(key[0])
                return
            v = np.asarray(m.vertices, dtype=np.float32) * key[1]
            f = np.asarray(m.faces, dtype=np.int32)
            mesh = UsdGeom.Mesh.Define(self.stage, self.path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(f.ravel())))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*self.COLOR)])
            mesh.CreateExtentAttr([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))])
            print(f"[live] holding {held.get('name', '?')}: {len(f)} triangles from {key[0]}", flush=True)
            prim = mesh.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        # Gf matrices act on row vectors: the transpose of T_tool0.
        xf.AddTransformOp().Set(Gf.Matrix4d(np.asarray(held["T_tool0"], dtype=float).T.tolist()))
        UsdGeom.Imageable(prim).MakeVisible()
        self.shown = key


class OverlayPrims:
    """The task's overlays (live.Overlays) as prims under /World/Overlays/<name>, rebuilt when they
    change: meshes (a CAD file at a pose, e.g. the socket where the 6D estimate puts it) as meshes,
    poses as RGB axes, polylines as curves, points as points. As live_view_gl.py's OverlayDrawer."""

    AXES = ((1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.3, 0.5, 1.0))
    ROOT = "/World/Overlays"

    def __init__(self, stage, overlays):
        self.stage, self.overlays, self.drawn = stage, overlays, -1
        self.meshes = {}            # (path, scale) -> (points (N, 3) m, faces (M, 3)), or None

    def _mesh(self, path, scale):
        key = (path, float(scale))
        if key not in self.meshes:
            try:
                import trimesh

                m = trimesh.load(path, process=False, force="mesh")
                self.meshes[key] = (np.asarray(m.vertices, dtype=np.float32) * float(scale), np.asarray(m.faces, dtype=np.int32))
            except Exception as exc:    # noqa: BLE001 -- only this part goes undrawn
                print(f"[live] the mesh {path}: {exc} -- not drawn.", flush=True)
                self.meshes[key] = None
        return self.meshes[key]

    def update(self):
        by_name, version = self.overlays.snapshot()
        if version == self.drawn:
            return
        self.drawn = version
        from pxr import Gf, Sdf, UsdGeom, Vt

        self.stage.RemovePrim(self.ROOT)
        UsdGeom.Scope.Define(self.stage, self.ROOT)
        for name, overlay in by_name.items():
            base = Sdf.Path(self.ROOT).AppendChild(Sdf.Path.MakeValidIdentifier(name) if hasattr(Sdf.Path, "MakeValidIdentifier")
                                                   else "".join(c if c.isalnum() else "_" for c in name))
            UsdGeom.Xform.Define(self.stage, base)
            for i, item in enumerate(overlay.get("items", [])):
                path, kind = base.AppendChild(f"item_{i}"), item.get("type")
                color = Gf.Vec3f(*item.get("color", (0.8, 0.8, 0.8)))
                if kind == "mesh" and item.get("mesh"):
                    got = self._mesh(item["mesh"], item.get("scale", 1.0))
                    if got is None:
                        continue
                    v, f = got
                    mesh = UsdGeom.Mesh.Define(self.stage, path)
                    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v)))
                    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
                    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(f.ravel())))
                    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                    mesh.CreateDisplayColorAttr([color])
                    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))])
                    mesh.AddTransformOp().Set(Gf.Matrix4d(np.asarray(item["T"], dtype=float).T.tolist()))
                elif kind == "pose":
                    # RGB axes as thin solid rods (RTX draws them from any side, at any zoom).
                    T = np.asarray(item["T"], dtype=float)
                    n = float(item.get("length", 0.03))
                    frame = UsdGeom.Xform.Define(self.stage, path)
                    frame.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))
                    for axis, (name_, rgb) in enumerate(zip("XYZ", self.AXES)):
                        rod = UsdGeom.Cylinder.Define(self.stage, path.AppendChild(f"axis_{name_.lower()}"))
                        rod.CreateAxisAttr(name_)
                        rod.CreateHeightAttr(n)
                        rod.CreateRadiusAttr(0.0012)
                        rod.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
                        offset = [0.0, 0.0, 0.0]
                        offset[axis] = n / 2.0
                        rod.AddTranslateOp().Set(Gf.Vec3d(*offset))
                elif kind == "line":
                    pts = [np.asarray(p, dtype=float) for p in item.get("points", [])]
                    if len(pts) < 2:
                        continue
                    curves = UsdGeom.BasisCurves.Define(self.stage, path)
                    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
                    curves.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
                    curves.CreateCurveVertexCountsAttr([len(pts)])
                    curves.CreateWidthsAttr([0.002])
                    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                    curves.CreateDisplayColorAttr([color])
                elif kind == "points" and item.get("points"):
                    pts = np.asarray(item["points"], dtype=np.float32).reshape(-1, 3)
                    points = UsdGeom.Points.Define(self.stage, path)
                    points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts)))
                    points.CreateWidthsAttr([2.0 * float(item.get("radius", 0.003))])
                    points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                    points.CreateDisplayColorAttr([color])


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
    camera = rgbd.CameraLayer(voxel_m=args_cli.voxel_mm / 1000.0, max_points=args_cli.max_points,
                              splat=args_cli.splat)
    capture_q = camera.seed(rgbd.expand(args_cli.rgbd), bench_z, stride=args_cli.rgbd_stride)
    overlays = Overlays()  # the task's findings: drawn (OverlayPrims) and their text in the panel
    latest = LatestImage()  # the camera window's frame
    named = NamedImages()  # the task's named images: the behavior tree, the in-hand fit
    frames = None
    if not args_cli.no_frames:
        frames_port = args_cli.frames_port or int(lv.get("frames_port", FRAMES_PORT))
        try:
            frames = FrameReceiver(args_cli.bind, frames_port, lambda f: (camera.add(f), latest.put(f)), overlays.put,
                                   on_preview=latest.put, on_image=named.put)
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
        q_start = joint_pos.clone()  # what Reset puts the arm back to
        zeros = torch.zeros_like(joint_pos)
        finger = 0.0
        panel = Panel(f"Live: iiwa 14 ({args_cli.cell})") if sim.has_gui else None
        camera_window = CameraWindow(latest, args_cli.camera_window_px) if sim.has_gui and not args_cli.no_camera_window else None
        named_windows = NamedImageWindows(named) if sim.has_gui and not args_cli.no_camera_window else None
        # The camera layer as a Gaussian splat. splat imports pxr: only now that Kit is up.
        import omni.usd

        from cable_assembly_sim import splat

        layer_prim = splat.define(omni.usd.get_context().get_stage(), "/World/CameraLayer")
        held = HeldPrim(omni.usd.get_context().get_stage())
        overlay_prims = OverlayPrims(omni.usd.get_context().get_stage(), overlays)  # the socket, the plug, ...
        drawn, t_drawn = -1, 0.0

        def set_view():
            sim.set_camera_view(tuple(eye), tuple(lookat))

        def reset():
            """Back as the viewer started: everything heard from the task forgotten -- the camera layer
            (re-seeded from --rgbd, if given), the overlays, the held part, the images, the last sample --
            the arm at its start pose, the fingers open, the viewport camera back. A task still running
            draws it all again with its next sample, overlay and frame."""
            nonlocal finger, t_drawn
            stream.forget()
            camera.clear()
            if args_cli.rgbd:
                camera.seed(rgbd.expand(args_cli.rgbd), bench_z, stride=args_cli.rgbd_stride)
            overlays.clear()
            latest.clear()
            named.clear()
            if camera_window is not None:
                camera_window.reset()
            if named_windows is not None:
                named_windows.reset()
            finger = 0.0
            joint_pos[:] = q_start
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=zeros)
            robot.set_joint_position_target_index(target=joint_pos)
            scene.write_data_to_sim()
            sim.forward()
            t_drawn = 0.0  # the emptied layer drawn at once, not after --splat_hz
            set_view()
            print("[live] RESET: the camera layer, overlays, held part, images and last sample cleared; "
                  "the arm at its start pose.", flush=True)

        print(f"[live] {cell_path}: floor at {bench_z * 1000:.1f} mm. Listening on udp://{args_cli.bind}:{port}; "
              f"start the task (live_view.enabled) in any order.", flush=True)

        t_start = time.monotonic()
        while sim.is_running():
            t0 = time.monotonic()
            if args_cli.duration_s is not None and t0 - t_start > args_cli.duration_s:
                break
            if panel is not None:
                if panel.take("reset"):
                    reset()
                elif panel.take("view"):
                    set_view()
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

            held.update((msg or {}).get("held"))
            if camera_window is not None:
                camera_window.update()
            if named_windows is not None:
                named_windows.update()
            overlay_prims.update()
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
