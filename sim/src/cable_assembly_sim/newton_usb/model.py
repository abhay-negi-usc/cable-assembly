"""The USB-A plug (McMaster 4974T135) and its panel-mount receptacle (4574N11) as a Newton model: the plug
driven by a Cartesian spring-damper at the fingertips, over the receptacle, replicated into many worlds.

THE ARM, as in newton_bnc/model.py: not simulated link by link. The iiwa runs in position control (FRI
streaming joint targets), so the plug feels the arm's Cartesian stiffness in series with the grasp, and
the arm's reflected mass. Both are one D6 joint from the world to the plug, its six DOFs driven to a
target, at the FINGERTIP point on the plug (the pads' middle, on the overmold). Gravity is off (the arm
compensates it; the plug is 7 g).

THE GRASP is the bench's tilted one (bnc_assembly.yaml, pickup.fingertip_in_connector rpy [0, -70, 0]):
the jaws close ACROSS the plug (its 16 mm width, plug x), and the fingers come in at GRASP_TILT_DEG to
the plug axis, over toward its free end. So the 2F-85's pads -- 37.8 x 22.4 x 11.6 mm rubber blocks --
stand at that tilt either side of the overmold, their lower corners reaching down toward the adapter.
They are colliders (`pad_0`, `pad_1`, against the adapter only): with the in-hand error the plug can
sit far enough up the pads that they meet the adapter's face before the plug seats. They move with the
hand, not with the plug, so the env sets their pose in the plug body per world (set_pad_pose).

THE PARTS come from scripts/build_usb.py's USDs (McMaster's geometry, gitignored), read with pxr, as SDF
mesh colliders: `plug` (the shell + insert as one solid, the tongue cavity cut in), `receptacle` (the
pocket, the tongue, the lead-in, the face, the rim and the flange, made solid). Socket frame: z up the receptacle axis, the plug authored seated (a plug pose equal to
the socket's is seated). The overmold and the 5 cm cable stub are drawn only: rigid with the plug.

THE DETENT. A USB-A receptacle's shell has spring tabs that press on the plug's shell once it is ~2 mm
in: they are what resists insertion (10-35 N by the USB 2.0 spec). Rigid, they would only jam, so
build_usb.py cuts them out, and the env puts their resistance on the D6's axial DOF as Coulomb friction
(MuJoCo frictionloss, env._set_detent): implicit, so it holds a plug still and lets it slide, and it adds
no side force (the tabs squeeze from all four sides). A one-sided push onto the pocket wall, tried first,
tilted the plug 4 deg about the fingertips through the grasp's compliance and jammed it on the tongue.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from cable_assembly_sim.newton_bnc.model import _coarser_contact_reduction, _tube_inertia  # noqa: F401  (configures newton)

import newton  # noqa: E402

USB_DIR = Path(__file__).resolve().parents[1] / "assets" / "data" / "usb"
PLUG_USD = USB_DIR / "usb_plug.usd"
SOCKET_USD = USB_DIR / "usb_socket.usd"
SDF_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cable_assembly_sim" / "sdf"

# Heights in the socket frame [m] (scripts/build_usb.py, off the CAD). The plug frame's origin is where
# its tip's centre is when seated, raised by TIP_DEPTH: plug z = 0 is seated.
TIP_DEPTH = 0.004837
"""The plug's tip sits this far below the plug frame's origin (the pocket floor is at z -4.837 mm)."""
FACE_Z = 0.0062
"""The housing's face round the opening (its lead-in from 6.2 down to 5.54 mm)."""
SHELL_TOP_Z = 0.00554
POCKET_TOP_Z = 0.00504
TONGUE_TOP_Z = 0.00531
DETENT_TOP_Z = 0.0034
"""The tops of the receptacle shell's spring tabs (they reach the plug's shell below this)."""
PLUG_LEN = 0.012
"""Exposed plug shell: seated, its tip is 12 mm below the overmold's front face."""
PLUG_HALF = (0.0060, 0.00225)
PLUG_DY = 0.000456
"""The plug's centreline in the socket's y (the shells' centres split with the tongue's)."""

GRASP_Z = 0.032
"""The fingertip point (the pads' middle) up the plug axis [m]: on the overmold (z 7.2-43.2 mm), 11 mm
from its cable end. The BNC's 23 mm from the junction would put the pads' lower corners 1.4 mm into the
adapter's face at a seated plug; here they clear its rim by 3.8 mm (GRASP_TILT_DEG, PAD_*)."""
GRASP_TILT_DEG = 20.0
"""Between the fingers (tool z) and the plug axis: the bench's pick pitch, -70 deg (90 - 70)."""
PANEL_TOP_Z = 0.00490
"""The panel the adapter mounts in: its front face at the housing plate's underside (build_usb.PLATE_UNDERSIDE_Z),
the plate (r 11.7 mm, top 6.65 mm) standing proud of it."""
PANEL_R = (0.0116, 0.045)
"""The panel's hole (just inside the plate's rim: no gap to drop a corner into) and its outer edge [m]."""
PANEL_THICK = 0.003

OVERMOLD_HALF_W = 0.008
"""Half the overmold's width: where the pads' faces sit, either side along plug x."""
PAD_HALF = (0.0112, 0.0058, 0.0189)
"""The bench pads' half sizes (scripts/build_iiwa14.py): along the finger's width (F x), across
(F y, the closing direction) and along the finger (F z)."""


def grasp_rotation(tilt_deg: float = GRASP_TILT_DEG) -> np.ndarray:
    """R_PF: the fingertip frame F in the plug frame. F y = plug x (the jaws close across the plug's width);
    F z, the fingers' direction, runs down the plug toward its tip, tilted by tilt_deg toward -y (the
    wrist stands above the plug's +y side); F x = F y x F z."""
    t = math.radians(tilt_deg)
    fz = np.array([0.0, -math.sin(t), -math.cos(t)])
    fy = np.array([1.0, 0.0, 0.0])
    fx = np.cross(fy, fz)
    return np.stack((fx, fy, fz), -1)


def pad_offsets(tilt_deg: float = GRASP_TILT_DEG) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each pad's (rotation, centre) relative to the plug-aligned frame at the fingertip point (no error)."""
    R = grasp_rotation(tilt_deg)
    across = OVERMOLD_HALF_W + PAD_HALF[1]
    return [(R, R @ np.array([0.0, s * across, 0.0])) for s in (-1.0, 1.0)]


def pad_clearance(tilt_deg: float = GRASP_TILT_DEG, grasp_z: float = GRASP_Z) -> float:
    """Height of the pads' lowest corner over the adapter's top (6.65 mm), seated, no error [m]."""
    R = grasp_rotation(tilt_deg)
    hx, hy, hz = PAD_HALF
    lowest = min((R @ np.array([sx * hx, sy * hy, sz * hz]))[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
    return grasp_z + lowest - 0.00665


def panel_mesh(segments: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """The panel: an annular plate round the adapter, in the socket frame (points [m], triangle indices).
    Off the CAD (the adapter alone has no surround): with socket estimates off by up to 7 mm the plug lands
    on it beside the 11.7 mm housing, which an adapter floating in space would let it fall past."""
    th = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = lambda r, z: np.stack((r * np.cos(th), r * np.sin(th), np.full_like(th, z)), -1)  # noqa: E731
    z1, z0 = PANEL_TOP_Z, PANEL_TOP_Z - PANEL_THICK
    pts = np.concatenate((ring(PANEL_R[0], z1), ring(PANEL_R[1], z1), ring(PANEL_R[0], z0), ring(PANEL_R[1], z0)))
    i = np.arange(segments)
    j = (i + 1) % segments
    IT, OT, IB, OB = 0, segments, 2 * segments, 3 * segments
    tris = [np.stack(t, -1) for t in (
        (IT + i, OT + i, OT + j), (IT + i, OT + j, IT + j),      # top, +z
        (IB + i, OB + j, OB + i), (IB + i, IB + j, OB + j),      # bottom, -z
        (OB + i, OB + j, OT + j), (OB + i, OT + j, OT + i),      # outer wall, outward
        (IB + i, IT + j, IB + j), (IB + i, IT + i, IT + j))]     # hole's wall, toward the axis
    return pts.astype(np.float32), np.concatenate(tris).astype(np.int32).flatten()


@dataclass
class ModelCfg:
    num_worlds: int = 256
    hand_mass: float = 0.25
    """Reflected mass of arm + gripper at the fingertips [kg] (given to the plug body)."""
    contact_ke: float = 1.0e6
    contact_kd: float = 158.0
    """As newton_bnc (CAD-checked there): ~0.04 mm interference at 1 kHz."""
    gap: float = 0.0003
    sdf_resolution: int = 256
    socket_sdf_resolution: int = 512
    """The adapter's front is 33 mm across: 512 gives 0.065 mm cells."""
    contacts_per_world: int = 768
    rows_per_world: int = 3600
    triangle_pairs: int = 4_000_000
    lin_ke: float = 1.0e4
    lin_kd: float = 150.0
    rot_ke: float = 10.0
    rot_kd: float = 0.15
    pads: bool = True
    """The gripper's pads as colliders against the adapter."""
    panel: bool = True
    """The panel round the adapter (panel_mesh)."""
    panel_sdf_resolution: int = 256


def _read_meshes(path: Path, names) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from pxr import Usd, UsdGeom

    if not path.exists():
        raise FileNotFoundError(f"{path}: run `python scripts/build_usb.py` first")
    stage = Usd.Stage.Open(str(path))
    out = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() not in names:
            continue
        mesh = UsdGeom.Mesh(prim)
        X = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64) @ X[:3, :3].T + X[:3, 3]
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        out[prim.GetName()] = (pts.astype(np.float32), idx)
    missing = set(names) - set(out)
    if missing:
        raise ValueError(f"{path}: no mesh named {sorted(missing)}: rebuild with scripts/build_usb.py")
    return out


def read_drawn() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every mesh in both USDs (socket frame, plug seated), for drawing."""
    return {**_read_meshes(PLUG_USD, ("plug", "overmold", "cable", "contacts")),
            **_read_meshes(SOCKET_USD, ("adapter",))}


def _sdf_mesh(points, indices, resolution, gap) -> newton.Mesh:
    mesh = newton.Mesh(points, indices)
    SDF_CACHE.mkdir(parents=True, exist_ok=True)
    mesh.build_sdf(max_resolution=resolution, narrow_band_range=(-0.002, 0.002), margin=gap, cache_dir=SDF_CACHE)
    return mesh


@dataclass
class UsbModel:
    model: newton.Model
    plug_body: int
    bodies_per_world: int
    coords_per_world: int
    dofs_per_world: int
    shapes_per_world: int
    plug_shapes: list[int]
    socket_shapes: list[int]  # the receptacle and the panel (carry mu_parts with the plug)
    pad_shapes: list[int]


def build(cfg: ModelCfg, device=None) -> UsbModel:
    plug = _read_meshes(PLUG_USD, ("plug",))
    socket = _read_meshes(SOCKET_USD, ("receptacle",))
    meshes = {
        "plug": _sdf_mesh(*plug["plug"], cfg.sdf_resolution, cfg.gap),
        "receptacle": _sdf_mesh(*socket["receptacle"], cfg.socket_sdf_resolution, cfg.gap),
    }
    if cfg.panel:
        meshes["panel"] = _sdf_mesh(*panel_mesh(), cfg.panel_sdf_resolution, cfg.gap)
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.3, ke=cfg.contact_ke, kd=cfg.contact_kd, gap=cfg.gap, density=0.0, mu_torsional=0.0, mu_rolling=0.0,
    )
    world = newton.ModelBuilder(gravity=0.0)
    world.default_shape_cfg = shape_cfg
    shape_names = []

    def add_mesh(body, name):
        s = world.add_shape_mesh(body, mesh=meshes[name], cfg=shape_cfg, label=name)
        shape_names.append((s, name))

    add_mesh(-1, "receptacle")
    if cfg.panel:
        add_mesh(-1, "panel")
    m = cfg.hand_mass
    body = world.add_link(
        xform=wp.transform_identity(), com=wp.vec3(0.0, 0.0, GRASP_Z), mass=m,
        inertia=_tube_inertia(m, 0.0, 0.03, 0.06), lock_inertia=True, label="plug_body",
    )
    add_mesh(body, "plug")
    if cfg.pads:
        pad_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5, ke=cfg.contact_ke, kd=cfg.contact_kd, gap=cfg.gap, density=0.0, mu_torsional=0.0, mu_rolling=0.0)
        for k, (R, c) in enumerate(pad_offsets()):
            q = wp.quat_from_matrix(wp.mat33(*R.flatten().tolist()))
            pos = wp.vec3(float(c[0]), float(c[1]), float(c[2] + GRASP_Z))
            s = world.add_shape_box(body, xform=wp.transform(pos, q), hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2],
                                    cfg=pad_cfg, label=f"pad_{k}")
            shape_names.append((s, f"pad_{k}"))

    Dof = newton.ModelBuilder.JointDofConfig
    # The linear z DOF (world z = the socket axis) carries the detent as Coulomb friction (env._set_detent);
    # it is built non-zero so MuJoCo-Warp keeps its friction row.
    lin = [Dof(axis=a, target_ke=cfg.lin_ke, target_kd=cfg.lin_kd, friction=1.0e-4 if a[2] else 0.0)
           for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    ang = [Dof(axis=a, target_ke=cfg.rot_ke, target_kd=cfg.rot_kd) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    j_d6 = world.add_joint_d6(
        -1, body, linear_axes=lin, angular_axes=ang, parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, GRASP_Z), wp.quat_identity()), label="arm",
    )
    world.add_articulation([j_d6])

    # Only the pairs that can touch: the plug and the pads on the adapter and the panel.
    keep = {(a, b) for a in ("plug", "pad_0", "pad_1") for b in ("receptacle", "panel")}
    for i, (sa, na) in enumerate(shape_names):
        for sb, nb in shape_names[i + 1:]:
            if (na, nb) not in keep and (nb, na) not in keep:
                world.add_shape_collision_filter_pair(sa, sb)

    builder = newton.ModelBuilder(gravity=0.0)
    builder.replicate(world, world_count=cfg.num_worlds, spacing=(0.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    model.rigid_contact_max = cfg.contacts_per_world * cfg.num_worlds
    names = dict(shape_names)
    return UsbModel(
        model=model, plug_body=body, bodies_per_world=world.body_count, coords_per_world=world.joint_coord_count,
        dofs_per_world=world.joint_dof_count, shapes_per_world=world.shape_count,
        plug_shapes=[s for s, n in names.items() if n == "plug"],
        socket_shapes=[s for s, n in names.items() if n in ("receptacle", "panel")],
        pad_shapes=sorted(s for s, n in names.items() if n.startswith("pad_")),
    )


def make_solver(um: UsbModel, cfg: ModelCfg):
    pipeline = newton.CollisionPipeline(
        um.model, reduce_contacts=True, rigid_contact_max=um.model.rigid_contact_max, broad_phase="explicit",
        max_triangle_pairs=cfg.triangle_pairs,
    )
    solver = newton.solvers.SolverMuJoCo(
        um.model, use_mujoco_contacts=False, solver="newton", integrator="implicitfast", cone="elliptic",
        njmax=cfg.rows_per_world, nconmax=cfg.contacts_per_world, iterations=int(os.environ.get("USB_ITERS", 20)),
        ls_iterations=int(os.environ.get("USB_LS", 50)), impratio=1.0,
    )
    return pipeline, solver
