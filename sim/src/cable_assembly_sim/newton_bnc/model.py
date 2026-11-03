"""The bench BNC pair as a Newton model: the McMaster plug (body -> collar_slide -> collar_turn ->
collar) driven by a Cartesian spring-damper, over the 4441N11 socket, replicated into many worlds.

THE ARM. Not simulated link by link: the iiwa runs in position control (FRI streaming joint targets
from IK), so what the plug feels is the arm's Cartesian stiffness in series with the grasp, and the
arm's reflected mass. Both are lumped into one D6 joint from the world to the plug, its six DOFs
driven to a target, at the FINGERTIP point on the plug (the middle of the pads, on the strain
relief): that is where the grasp's rotational compliance is centred. The plug body's mass is that
reflected mass. Gravity is off, as if the arm compensated it (the washer's 5 N preload dominates the
collar's 0.06 N weight anyway).

THE PARTS come from scripts/build_bnc.py's USDs (McMaster's geometry, gitignored), read here with
pxr and turned into SDF mesh colliders. Socket frame: z up the socket axis, pins on +/-y; the plug
authored seated, so a plug pose equal to the socket's is seated and collar joints at 0 put the
collar's slot entries on the pins.

FRICTION. MuJoCo uses the larger of the two shapes' friction. So the collar's own is ~0, the socket
and the body's parts carry `mu_parts`, the seat ring carries `mu_seat`: collar on socket (pins,
barrel) is mu_parts, collar on its seat is mu_seat -- each set per world (set_friction).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


def _coarser_contact_reduction(polyhedron="octahedron", directions=3, voxel_slots=24):
    """Newton keeps up to 240 contacts per mesh pair (20 normal bins x 7 + 100 voxel slots), set by
    module constants read at import. A BNC pair needs far fewer, and every one is solver rows, so this
    loads that module with smaller constants BEFORE newton imports it (it then finds it loaded)."""
    import importlib.util
    import sys

    name = "newton._src.geometry.contact_reduction"
    if name in sys.modules:
        if "newton" in sys.modules and sys.modules[name].NUM_VOXEL_DEPTH_SLOTS != voxel_slots:
            raise RuntimeError("newton was imported before cable_assembly_sim.newton_bnc.model")
        return
    root = Path(importlib.util.find_spec("newton").origin).parent
    src = (root / "_src" / "geometry" / "contact_reduction.py").read_text()
    for old, new in (('NORMAL_BINNING_POLYHEDRON = "icosahedron"', f'NORMAL_BINNING_POLYHEDRON = "{polyhedron}"'),
                     ("NUM_SPATIAL_DIRECTIONS = 6", f"NUM_SPATIAL_DIRECTIONS = {directions}"),
                     ("NUM_VOXEL_DEPTH_SLOTS = 100", f"NUM_VOXEL_DEPTH_SLOTS = {voxel_slots}")):
        if old not in src:
            raise RuntimeError(f"newton's contact_reduction.py changed: no {old!r}")
        src = src.replace(old, new)
    import types

    module = types.ModuleType(name)
    module.__file__ = str(root / "_src" / "geometry" / "contact_reduction.py")
    module.__package__ = "newton._src.geometry"
    sys.modules[name] = module
    exec(compile(src, module.__file__, "exec"), module.__dict__)


_coarser_contact_reduction(*[c(v) for c, v in zip((str, int, int), os.environ.get("BNC_CONTACT_REDUCTION", "icosahedron,6,100").split(","))])

import newton  # noqa: E402

BNC_DIR = Path(__file__).resolve().parents[1] / "assets" / "data" / "bnc"
PLUG_USD = BNC_DIR / "bnc_plug.usd"
SOCKET_USD = BNC_DIR / "bnc_socket.usd"
SDF_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cable_assembly_sim" / "sdf"

# Heights in the socket frame [m] (assets/bnc_bench.py, measured off the CAD).
SOCKET_RIM_Z = 0.01214
COLLAR_ON_PINS_HEIGHT = 0.00618
SLOT_FLOOR_HEIGHT = 0.00108
STRAIN_RELIEF_Z = (0.0178, 0.0465)

GRASP_Z = 0.030
"""The fingertip point (the pads' middle) along the plug axis [m]: on the strain relief, as the
bench holds it (bnc_assembly.yaml). The D6 drive acts here."""

COLLAR_SLIDE_RANGE = (-0.0012, 0.00005)
COLLAR_SLIDE_STIFFNESS = 5000.0
COLLAR_SLIDE_PRELOAD = 0.001

BODY_PARTS = ("seat",)
"""The body's MESH colliders: none. What of it touches anything is stood in for by spheres (below):
the boot never reaches the socket; the seat ring, the contact fingers and the insulator face act as a
load-bearing ring, a lateral guide and a stop, which a few spheres reproduce at one contact each
(their meshes cost ~230 contacts a world)."""

SPHERE_R = 0.0005
SEAT_SPHERES = (6, 0.0059, 0.01708 + SPHERE_R)
"""(count, ring radius, centre z): the collar's seat, bottoms at the seat ring's face (17.08 mm),
r 4.95-6.80 mm -- where the collar's rear flange bears when pushed back."""
FINGER_SPHERES = ((8, 0.00397 - SPHERE_R, 0.00404 + 0.0008), (8, 0.00397 - SPHERE_R, 0.0080))
"""The contact fingers (r 3.97 mm, from z 4.04 mm): their tips and part way up, in the barrel bore."""
INSULATOR_SPHERES = (4, 0.0025, 0.003726 + SPHERE_R)
"""The insulator face (z 3.726 mm), which rests on the socket's dielectric when locked."""
COLLIDER_PARTS = BODY_PARTS + ("collar", "socket_0", "socket_1")
INNER_SOLID = (0.00397, 0.00373, 0.01623)
"""(radius, z from, z to) of the plug's finger sleeve + insulator as one solid collider (plug frame)."""
BARREL_CORE = (0.004826, 0.0020, 0.0119)
"""(radius, z from, z to) of a solid collision core inside the socket's barrel, at its outer radius (4.826
mm), from above the flange to just under the rim; collar only. The barrel wall is 0.7 mm: once a collar
edge sank past its middle, the wall's SDF flipped and pulled the collar through (tilted landings on the
rim, ~1-3% of full-error episodes). The core's SDF points out for 2 mm."""


@dataclass
class ModelCfg:
    num_worlds: int = 256
    hand_mass: float = 0.25
    """Reflected mass of arm + gripper at the fingertips [kg] (given to the plug body)."""
    contact_ke: float = 1.0e6
    contact_kd: float = 158.0
    """Contact stiffness and damping (damping ~ sqrt(ke)). With Newton's full contact reduction, the plug's
    inner parts as meshes and 1 kHz physics: worst interference 0.13 mm, median 0.04 mm in provoked
    straddling landings (was 0.98 mm with 1e5, 24 reduction slots, sphere stand-ins, 500 Hz)."""
    gap: float = 0.0003
    sdf_resolution: int = 256
    contacts_per_world: int = 768
    """The contact buffer is this times the worlds, shared: 1024 plugs seated and pushed make ~290 a
    world, and past the buffer contacts are dropped in random worlds (parts pass through each other)."""
    rows_per_world: int = 3600
    triangle_pairs: int = 4_000_000
    """Newton's mesh-mesh candidate buffer (default 1M), shared by all worlds: a plug pushed deep
    into the socket touches thousands of triangles, and on overflow contacts are silently dropped."""
    """MuJoCo constraint rows (njmax): ~3 per elliptic contact, plus limits."""
    lin_ke: float = 1.0e4
    lin_kd: float = 150.0
    rot_ke: float = 10.0
    rot_kd: float = 0.15
    collar_armature: float = 2.0e-6
    slide_limit_ke: float = 1.0e6
    slide_limit_kd: float = 200.0
    """The collar slide's stops (the washer fully compressed at the rear: a metal shoulder)."""
    inner_colliders: str = "solid"
    """The plug's contact-finger sleeve and insulator, against the socket:
    - "solid": one solid cylinder, r 3.97 mm, z 3.73-16.23 mm (plug frame) -- what they are to the socket,
      whose bore holds nothing above z 3.73 (the dielectric, flush with the centre contact). Default.
    - "mesh": the CAD. The finger sleeve is a 0.16 mm shell: its SDF gives contacts from both faces with
      flipped normals near the bore's 0.7 mm wall, which launched seated plugs upward at ~50 N.
    - "spheres": two rings of 8 + 4 (fast; the barrel wall fits through their ~1.7 mm gaps)."""
    barrel_core: bool = False
    """Add the BARREL_CORE collider (see there)."""
    decimate: tuple[tuple[str, float], ...] = ()
    """(part, fraction of triangles kept) for the collision meshes. Collision is ~70% of the step and
    scales with the triangles near contact."""


def _read_meshes(path: Path, names=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from pxr import Usd, UsdGeom

    if not path.exists():
        raise FileNotFoundError(f"{path}: run `./run.sh --extra cad python scripts/build_bnc.py` first")
    stage = Usd.Stage.Open(str(path))
    out = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() not in (names or COLLIDER_PARTS):
            continue
        mesh = UsdGeom.Mesh(prim)
        X = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64) @ X[:3, :3].T + X[:3, 3]
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f"{prim.GetPath()}: not triangles")
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        out[prim.GetName()] = (pts.astype(np.float32), idx)
    return out


def _sdf_mesh(points: np.ndarray, indices: np.ndarray, cfg: ModelCfg) -> newton.Mesh:
    mesh = newton.Mesh(points, indices)
    SDF_CACHE.mkdir(parents=True, exist_ok=True)
    mesh.build_sdf(
        max_resolution=cfg.sdf_resolution,
        narrow_band_range=(-0.002, 0.002),
        margin=cfg.gap,
        cache_dir=SDF_CACHE,
    )
    return mesh


def _cylinder_mesh(radius, z0, z1, segments=128):
    """A closed cylinder about z, outward-facing triangles."""
    a = 2 * np.pi * np.arange(segments) / segments
    ring = np.stack((radius * np.cos(a), radius * np.sin(a)), -1)
    pts = np.concatenate((np.c_[ring, np.full(segments, z0)], np.c_[ring, np.full(segments, z1)],
                          [[0.0, 0.0, z0], [0.0, 0.0, z1]])).astype(np.float32)
    k, k1 = np.arange(segments), (np.arange(segments) + 1) % segments
    lo, hi, c0, c1 = k, k + segments, 2 * segments, 2 * segments + 1
    side = np.concatenate((np.stack((lo, k1, hi), -1), np.stack((k1, k1 + segments, hi), -1)))
    caps = np.concatenate((np.stack((np.full(segments, c0), k1, lo), -1),
                           np.stack((np.full(segments, c1), hi, k1 + segments), -1)))
    return pts, np.concatenate((side, caps)).astype(np.int32).flatten()


def _tube_inertia(mass, r_in, r_out, length):
    ixx = mass * (3 * (r_in**2 + r_out**2) + length**2) / 12.0
    izz = mass * (r_in**2 + r_out**2) / 2.0
    return wp.mat33(ixx, 0.0, 0.0, 0.0, ixx, 0.0, 0.0, 0.0, izz)


@dataclass
class BncModel:
    model: newton.Model
    plug_body: int  # per-world body indices are offset by bodies_per_world
    collar_body: int
    bodies_per_world: int
    coords_per_world: int
    dofs_per_world: int
    d6_q: slice  # coordinate slice of the D6 joint within a world
    slide_q: int
    turn_q: int
    shapes_per_world: int
    mu_parts_shapes: list[int]  # within-world shape indices that carry mu_parts
    mu_seat_shapes: list[int]
    collar_shapes: list[int]


def build(cfg: ModelCfg, device=None) -> BncModel:
    inner_mesh = cfg.inner_colliders == "mesh"
    inner_solid = cfg.inner_colliders == "solid"
    plug = _read_meshes(PLUG_USD, COLLIDER_PARTS + (("fingers", "insulator") if inner_mesh else ()))
    socket = _read_meshes(SOCKET_USD)
    parts = {**plug, **socket}
    for name, keep in cfg.decimate:
        import fast_simplification

        pts, idx = parts[name]
        p2, f2 = fast_simplification.simplify(pts, idx.reshape(-1, 3), target_reduction=1.0 - keep)
        parts[name] = (p2.astype(np.float32), f2.astype(np.int32).flatten())
    if cfg.barrel_core:
        parts["barrel_core"] = _cylinder_mesh(*BARREL_CORE)
    if inner_solid:
        parts["fingers"] = _cylinder_mesh(*INNER_SOLID)
    meshes = {name: _sdf_mesh(*pi, cfg) for name, pi in parts.items()}

    shape_cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.3, ke=cfg.contact_ke, kd=cfg.contact_kd, gap=cfg.gap, density=0.0,
        mu_torsional=0.0, mu_rolling=0.0,
    )
    world = newton.ModelBuilder(gravity=0.0)
    world.default_shape_cfg = shape_cfg

    shape_names = []

    def add_shape(body, name):
        s = world.add_shape_mesh(body, mesh=meshes[name], cfg=shape_cfg, label=name)
        shape_names.append((s, name))

    # The socket, bolted down: static shapes on the world body.
    for name in ("socket_0", "socket_1") + (("barrel_core",) if cfg.barrel_core else ()):
        add_shape(-1, name)

    # The plug body, carrying the arm's reflected mass at the fingertip point.
    m = cfg.hand_mass
    body = world.add_link(
        xform=wp.transform_identity(), com=wp.vec3(0.0, 0.0, GRASP_Z), mass=m,
        inertia=_tube_inertia(m, 0.0, 0.03, 0.06), lock_inertia=True, label="plug_body",
    )
    for name in BODY_PARTS:
        add_shape(body, name)

    def add_ring(body, name, count, radius, z):
        for k in range(count):
            a = 2 * math.pi * (k + 0.5) / count
            pos = wp.vec3(radius * math.cos(a), radius * math.sin(a), z)
            sh = world.add_shape_sphere(body, xform=wp.transform(pos, wp.quat_identity()), radius=SPHERE_R,
                                        cfg=shape_cfg, label=f"{name}_{k}")
            shape_names.append((sh, name))

    if inner_mesh:
        add_shape(body, "fingers")
        add_shape(body, "insulator")
    elif inner_solid:
        add_shape(body, "fingers")
    else:
        for ring in FINGER_SPHERES:
            add_ring(body, "fingers", *ring)
        add_ring(body, "insulator", *INSULATOR_SPHERES)
    slider = world.add_link(
        xform=wp.transform_identity(), mass=0.001, inertia=_tube_inertia(0.001, 0.0045, 0.0055, 0.001),
        lock_inertia=True, label="slider",
    )
    collar = world.add_link(
        xform=wp.transform_identity(), com=wp.vec3(0.0, 0.0, 0.010), mass=0.0062,
        inertia=_tube_inertia(0.0062, 0.00484, 0.0071, 0.0144), lock_inertia=True, label="collar",
    )
    add_shape(collar, "collar")

    Dof = newton.ModelBuilder.JointDofConfig
    lin = [Dof(axis=a, target_ke=cfg.lin_ke, target_kd=cfg.lin_kd) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    ang = [Dof(axis=a, target_ke=cfg.rot_ke, target_kd=cfg.rot_kd) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    j_d6 = world.add_joint_d6(
        -1, body, linear_axes=lin, angular_axes=ang,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, GRASP_Z), wp.quat_identity()),
        label="arm",
    )
    j_slide = world.add_joint_prismatic(
        body, slider, axis=(0.0, 0.0, 1.0),
        limit_lower=COLLAR_SLIDE_RANGE[0], limit_upper=COLLAR_SLIDE_RANGE[1],
        limit_ke=cfg.slide_limit_ke, limit_kd=cfg.slide_limit_kd,
        target_pos=COLLAR_SLIDE_PRELOAD, target_ke=COLLAR_SLIDE_STIFFNESS, target_kd=2.0,
        armature=cfg.collar_armature * 1000, label="collar_slide",
    )
    j_turn = world.add_joint_revolute(
        slider, collar, axis=(0.0, 0.0, 1.0), armature=cfg.collar_armature, damping=1.0e-5,
        label="collar_turn",
    )
    world.add_articulation([j_d6, j_slide, j_turn])

    # Only the pairs that can touch: the collar on the socket (pins, barrel, the well) and on its seat,
    # the contact fingers and the insulator in the barrel. Every other pair is contacts to pay for.
    keep = {("collar", "socket_0"), ("collar", "socket_1"), ("collar", "seat"),
            ("fingers", "socket_1"), ("insulator", "socket_1"), ("collar", "barrel_core")}
    for i, (sa, na) in enumerate(shape_names):
        for sb, nb in shape_names[i + 1:]:
            if (na, nb) not in keep and (nb, na) not in keep:
                world.add_shape_collision_filter_pair(sa, sb)

    builder = newton.ModelBuilder(gravity=0.0)
    builder.replicate(world, world_count=cfg.num_worlds, spacing=(0.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    model.rigid_contact_max = cfg.contacts_per_world * cfg.num_worlds

    names = dict(shape_names)
    mu_parts = [s for s, n in names.items() if n in ("socket_0", "socket_1", "insulator", "fingers", "barrel_core")]
    mu_seat = [s for s, n in names.items() if n == "seat"]
    collar_s = [s for s, n in names.items() if n == "collar"]
    return BncModel(
        model=model, plug_body=body, collar_body=collar,
        bodies_per_world=world.body_count, coords_per_world=world.joint_coord_count,
        dofs_per_world=world.joint_dof_count, d6_q=slice(0, 6), slide_q=6, turn_q=7,
        shapes_per_world=world.shape_count, mu_parts_shapes=mu_parts, mu_seat_shapes=mu_seat,
        collar_shapes=collar_s,
    )


def make_solver(bm: BncModel, cfg: ModelCfg):
    pipeline = newton.CollisionPipeline(
        bm.model, reduce_contacts=True, rigid_contact_max=bm.model.rigid_contact_max, broad_phase="explicit",
        max_triangle_pairs=cfg.triangle_pairs,
    )
    solver = newton.solvers.SolverMuJoCo(
        bm.model, use_mujoco_contacts=False, solver="newton", integrator="implicitfast", cone="elliptic",
        njmax=cfg.rows_per_world, nconmax=cfg.contacts_per_world, iterations=int(os.environ.get("BNC_ITERS", 20)), ls_iterations=int(os.environ.get("BNC_LS", 50)),
        impratio=1.0,
    )
    return pipeline, solver


def euler_xyz(qx, qy, qz, qw):
    """Angles (a, b, c) with R = Rx(a) Ry(b) Rz(c) (numpy, for checks)."""
    import scipy.spatial.transform as T

    return T.Rotation.from_quat([qx, qy, qz, qw]).as_euler("XYZ")


def deg(x):
    return math.radians(x)
