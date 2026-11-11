"""Build the KUKA iiwa 14 + Robotiq 2F-85 USD that the Isaac Lab tasks spawn.

WHY A BUILD SCRIPT RATHER THAN A CHECKED-IN USD. Neither NVIDIA nor KUKA publishes an iiwa 14 USD:
Isaac Sim's `Robots/Kuka/LBR_iiwa` and Isaac Lab's KukaAllegro are both the iiwa 7 R800 (0.34 m
base -> A2, Drake's iiwa 7 masses). So the arm is converted here from the description the real arm
already runs on -- lbr-stack's, the same source as cable_assembly/robot/description/iiwa14/ -- and
the result is reproducible from a pinned commit instead of being a binary nobody can diff.

Three stages, each re-runnable on its own:

    urdf     Fetch lbr-stack's iiwa 14 description at PIN and expand it into a plain URDF. Upstream
             ships xacro only, but the macro body is literal links and joints plus joint_limits.yaml,
             so it is expanded here without ROS. Stdlib + PyYAML only. The URDF is committed;
             tests/test_sim_iiwa14.py holds it to the real arm's kinematics.
    usd      Convert that URDF with Isaac Lab's URDF converter (the standalone importer when it is
             installed, else Isaac Sim's), exactly as Isaac Lab's scripts/tools/convert_urdf.py does.
    gripper  Fetch Isaac Sim's Robotiq 2F-85 and mount it on tool0 with a fixed joint -- the same
             construction Isaac Sim's own ur10e.usd uses for its "Robotiq_2f_85" gripper variant --
             and put the lab's camera mount and the D405 (cable_assembly/robot/description/
             tool/*.stl) on tool0, as seen
             (visual only), where the CAD puts them.

    cd sim
    ./run.sh python scripts/build_iiwa14.py                        # all three
    python scripts/build_iiwa14.py --stage urdf                    # any Python with PyYAML

Only the URDF and its LICENSE are tracked here (the tool's STLs are in the robot description).
The meshes, the converted USD,
the Robotiq files and the tool's USD are fetched or generated into the same directory and ignored
by git.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "src", "cable_assembly_sim", "assets", "data")
IIWA_DIR = os.path.join(DATA, "iiwa14")
URDF = os.path.join(IIWA_DIR, "iiwa14.urdf")
USD_DIR = os.path.join(IIWA_DIR, "usd")
ROBOTIQ_DIR = os.path.join(DATA, "robotiq_2f85")
ROBOT_USD = os.path.join(DATA, "iiwa14_2f85.usda")

# ---- upstream ---------------------------------------------------------------------------------
# The commit the lab's checkout (~/jeon_team_ws/lbr-stack) is on, and that the vendored kinematic
# URDF was taken from. That checkout's iiwa14.urdf is a LOCAL xacro expansion with one local edit
# (link_ee moved for an earlier tool); building from upstream here keeps that edit out.
PIN = "4a5599ce053044e5daf4df0030c01ec0629ae7c5"
RAW = f"https://raw.githubusercontent.com/lbr-stack/lbr_fri_ros2_stack/{PIN}/"
XACRO = "lbr_description/urdf/iiwa14/iiwa14_description.xacro"
LIMITS = "lbr_description/urdf/iiwa14/joint_limits.yaml"
MESHES = "lbr_description/meshes/iiwa14/"
LINKS = [f"link_{i}" for i in range(8)]
XACRO_NS = "{http://www.ros.org/wiki/xacro}"

# tool0 is upstream's own link_ee, renamed to what cable_assembly calls it: the media-flange face,
# 0.035 m beyond link_7. It gets a token inertial because the converter makes every link a rigid
# body; on a fixed joint inside a reduced-coordinate articulation the token mass changes nothing.
TOOL0_OFFSET_M = 0.035
TOOL0_INERTIAL = {"mass": "0.001", "ixx": "1e-07", "iyy": "1e-07", "izz": "1e-07"}

# ---- the gripper mount ------------------------------------------------------------------------
# PLACEHOLDER, like every bench measurement in configs/cells/iiwa14_bench.yaml: the coupler
# between the iiwa's media flange and the 2F-85, and the gripper's clocking about tool0's Z.
# Re-run the gripper stage with the measured values (--coupler_mm, --clocking_deg). The 2F-85's
# own frame has the fingers along +Z and opening along +/-Y.
COUPLER_MM = 0.0
CLOCKING_DEG = 0.0
ROBOTIQ_SUBDIR = "Robots/Robotiq/2F-85/"
ROBOTIQ_ROOT_FILE = "Robotiq_2F_85_edit.usd"
# ---- the camera mount -------------------------------------------------------------------------
# The lab's flange mount and the camera on it (metres) are in the CAD assembly frame the camera
# pose was given in (2026-10-29: the D415 at [0, -68.79, 12.65] mm, rpy [-15, 0, 0] from the
# flange face). Its z is tool0's; the flange face is at z = 72.400 mm (the floor of the mount's
# recess, which the flange's rim sits in; the plate is 7.94 mm from there to the coupling's seat at
# 80.343); and it is turned -90 deg about z from tool0's axes (the camera on tool -X). Checked on
# the meshes: the D415's front face normal was the CAD optical axis to 0.01, and the mount's round
# edges are centred on z. Since 2026-11-10 the camera is a D405 by the same 1/4-20 screw: Intel's
# mesh placed on the bracket by analysis/d405_tool_mesh.py (cable_assembly/robot/tool_geometry.py).
# Visual only -- 115k triangles of collider would slow every physics step.
# The STLs live with the robot description (cable_assembly's motion planner reads them too); the
# USD made from them is generated here.
TOOL_DIR = os.path.join(HERE, "..", "..", "cable_assembly", "robot", "description", "tool")
TOOL_USD = os.path.join(DATA, "tool", "tool_meshes.usdc")
TOOL_PARTS = {  # prim name: (STL, display colour)
    "flange_mount": ("realsense_flangemount.stl", (0.22, 0.22, 0.24)),
    "d405": ("d405.stl", (0.08, 0.08, 0.09)),
}
CAD_FLANGE_FACE_Z_M = 0.0724
CAD_TO_TOOL0_YAW_DEG = -90.0
# ---- the bench's gripper, as it is (--stage bench -> iiwa14_2f85_bench.usda) ------------------------
# Isaac Sim's 2F-85 is Robotiq's V4; the bench's is a V3 (the user, 2026-11-02), with its own pads.
# Same linkage (the outer knuckle pivots +/-30.6 mm off the axis and turns 47 deg in both), so the
# bench USD keeps Isaac's articulation -- joints, mimics, colliders, physics -- and swaps what is SEEN:
# each link's V4 visuals switched off, Robotiq's V3 CAD (their 2019-02-07 model set, downloaded by hand,
# gitignored) in their place, plus the coupling and the bench's pads. iiwa14_2f85.usda is left as it
# is: the sim cell and the bnc_insert task are built on its 0 mm coupler and V4 pads.
V3_DIR = os.path.join(DATA, "robotiq_2f85_v3")
V3_STEP = "2-Finger_85_Adaptive_Gripper-opened.STEP"
V3_COUPLING_STEP = "C-062-COUPLING_C-50-4M6-1D6.STEP"
V3_USD = os.path.join(V3_DIR, "v3_visuals.usdc")
BENCH_USD = os.path.join(DATA, "iiwa14_2f85_bench.usda")
# The V3 STEP's frame (mm): x across the opening, y along the fingers, z = x cross y. Into the gripper's
# base frame (x = STEP z, y = STEP x, z = STEP y + this): the outer knuckle's pivot at STEP y -35.10
# lands on Isaac's, base z 54.66 mm -- at joint 0 (open) every link frame IS the base frame.
V3_Z_OFF_MM = 89.76
# THE COUPLER, from the CAD stack: tool0 (the flange face) -> the camera mount's 7.94 mm plate -> the
# C-062 coupling (11.0 mm from its bottom rim to the seat its 2.9 mm rim wraps the base's skirt in) ->
# the V3 base's seating face, 0.14 mm below its frame. The wrist images' best fit agrees (16-20 mm).
BENCH_COUPLER_MM = 7.94 + 11.0 + 0.14
V3_COUPLING_Z_MM = -4.14          # the coupling's STEP z -> base z (its seat, STEP z 4.0, on the base face)
# The bench's pads (the user, 2026-11-02): rectangular blocks, 37.8 mm along the finger (away from the
# flange), 22.4 mm wide, 11.6 mm thick in the closing direction, their backs FLUSH WITH THE INNER FACE
# OF THE FINGERTIP'S MOUNT (48.85 mm off the axis, open: 74.5 mm between the pads' gripping faces).
# Centred along the finger where Robotiq's own pad (C-203) sits, which the wrist images bear out.
PAD_MM = (37.8, 22.4, 11.6)
PAD_CENTRE_ALONG_MM = 40.65       # STEP y
PAD_MOUNT_FACE_MM = 48.85         # |STEP x|, open: the fingertip's mount face, the pad's back on it
V3_COLORS = {"metal": (0.05, 0.05, 0.055), "pins": (0.6, 0.6, 0.62), "coupling": (0.2, 0.2, 0.22),
             "pad": (0.85, 0.5, 0.58)}
# Fallback when isaaclab is not importable (e.g. building the gripper stage from a bare usd-core).
ISAAC_NUCLEUS_FALLBACK = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


# ================================================================================================
# urdf
# ================================================================================================

_LIMIT_EXPR = re.compile(r"^joint_limits\['(A[1-7])'\]\['(lower|upper|velocity|effort)'\]( \* PI / 180)?$")


def _evaluate(expr: str, props: dict[str, str], limits: dict) -> str:
    """One ${...} from the iiwa 14 xacro. Only the two forms upstream uses are accepted, so an
    upstream change to the macro fails here loudly instead of producing a silently wrong URDF."""
    if expr in props:
        return props[expr]
    m = _LIMIT_EXPR.match(expr)
    if m is None:
        raise ValueError(f"unexpected xacro expression ${{{expr}}} -- upstream changed; update _evaluate")
    value = float(limits[m.group(1)][m.group(2)])
    if m.group(3):
        value = value * float(props["PI"]) / 180.0
    return repr(value)


def expand_xacro(xacro_text: str, limits: dict) -> ET.Element:
    """lbr-stack's iiwa14 macro -> a plain <robot>: its links and joints with every ${...} resolved,
    package:// meshes made relative, and link_ee renamed tool0. The gazebo and ros2_control macro
    calls are dropped -- they configure Gazebo and the real controller, not the model."""
    macro = next(e for e in ET.fromstring(xacro_text) if e.tag == XACRO_NS + "macro" and e.get("name") == "iiwa14")
    props = {e.get("name"): e.get("value") for e in macro if e.tag == XACRO_NS + "property"}
    robot = ET.Element("robot", name="iiwa14")
    for e in macro:
        if e.tag not in ("link", "joint"):
            continue
        e = copy.deepcopy(e)
        for node in e.iter():
            for key, value in node.attrib.items():
                if "${" in value:
                    node.set(key, re.sub(r"\$\{([^}]*)\}", lambda m: _evaluate(m.group(1), props, limits), value))
                if key == "filename" and value.startswith("package://" + MESHES):
                    node.set(key, "meshes/" + value[len("package://" + MESHES):])
        robot.append(e)

    ee_joint = robot.find("joint[@name='joint_ee']")
    ee_link = robot.find("link[@name='link_ee']")
    xyz = [float(v) for v in ee_joint.find("origin").get("xyz").split()]
    if xyz != [0.0, 0.0, TOOL0_OFFSET_M]:
        raise ValueError(f"upstream link_ee moved to {xyz}; tool0 is the media flange at {TOOL0_OFFSET_M} m")
    ee_joint.set("name", "link_7-tool0")
    ee_joint.find("child").set("link", "tool0")
    ee_link.set("name", "tool0")
    inertial = ET.SubElement(ee_link, "inertial")
    ET.SubElement(inertial, "origin", rpy="0 0 0", xyz="0 0 0")
    ET.SubElement(inertial, "mass", value=TOOL0_INERTIAL["mass"])
    ET.SubElement(inertial, "inertia", ixx=TOOL0_INERTIAL["ixx"], ixy="0", ixz="0",
                  iyy=TOOL0_INERTIAL["iyy"], iyz="0", izz=TOOL0_INERTIAL["izz"])
    return robot


def build_urdf() -> None:
    limits = yaml.safe_load(_fetch(RAW + LIMITS))
    robot = expand_xacro(_fetch(RAW + XACRO).decode(), limits)
    ET.indent(robot, space="  ")
    header = (
        "<?xml version=\"1.0\"?>\n"
        # No "--" in here: XML forbids it inside a comment, and ElementTree enforces that.
        "<!-- GENERATED by sim/scripts/build_iiwa14.py; do not hand-edit.\n"
        f"     KUKA LBR iiwa 14 R820, expanded from lbr-stack/lbr_fri_ros2_stack @ {PIN}\n"
        f"     {XACRO} + joint_limits.yaml (Apache-2.0, see LICENSE beside this file).\n"
        "     Every number is upstream's. Ours: no ROS (gazebo / ros2_control dropped), relative mesh\n"
        f"     paths, and upstream's link_ee renamed tool0 (the media-flange face, {TOOL0_OFFSET_M} m beyond\n"
        "     link_7) with a token inertial so the converter can make it a rigid body. -->\n"
    )
    _write(URDF, (header + ET.tostring(robot, encoding="unicode") + "\n").encode())
    _write(os.path.join(IIWA_DIR, "LICENSE"), _fetch(RAW + "LICENSE"))
    for kind, ext in (("visual", "dae"), ("collision", "stl")):
        for link in LINKS:
            rel = f"meshes/{kind}/{link}.{ext}"
            _write(os.path.join(IIWA_DIR, rel), _fetch(RAW + MESHES + rel.split("/", 1)[1]))
    print("wrote", os.path.relpath(URDF), "and its meshes")


# ================================================================================================
# usd
# ================================================================================================


def build_usd(args, then=None) -> None:
    """URDF -> USD with Isaac Lab's converter. Mirrors scripts/tools/convert_urdf.py of Isaac Lab
    3.0: prefer the kit-less standalone importer, and run the converter inside launch_simulation.

    `then` runs inside the simulation context too. When the importer needs Kit (it does once Isaac
    Sim is installed), closing that context ends the process -- nothing after it runs, and unflushed
    output is lost."""
    from isaaclab.app import launch_simulation
    from isaaclab.utils.version import standalone_importers_available

    args.require_kit = not standalone_importers_available()
    args.physics = "isaacsim_physx" if args.require_kit else "newton_mjwarp"

    from isaaclab.physics import PhysicsCfg
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    # The importer writes <usd_dir>/<robot name>/ and numbers a fresh folder instead of
    # overwriting, so start from an empty directory to keep the output path fixed.
    shutil.rmtree(USD_DIR, ignore_errors=True)
    cfg = UrdfConverterCfg(
        asset_path=os.path.abspath(URDF),
        usd_dir=os.path.abspath(USD_DIR),
        fix_base=True,
        # Keep tool0 as its own body: it is the frame every tool offset is measured from, and the
        # gripper's fixed joint attaches to it.
        merge_fixed_joints=False,
        force_usd_conversion=True,
        # One hull per link: the arm should never touch anything during an insertion, and the
        # contact that matters (plug in socket) is SDF-based on the connector assets.
        self_collision=False,
        collision_type="Convex Hull",
        # Gains are set per actuator in assets/iiwa14.py at spawn; the USD keeps the importer's.
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type="force", target_type="position"),
        robot_type="Manipulator",
    )
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args):
        usd_path = UrdfConverter(cfg).usd_path
        print("wrote", os.path.relpath(usd_path), flush=True)
        if then is not None:
            then()


def converted_usd() -> str:
    """The converter's root layer from a previous usd stage."""
    for dirpath, _dirs, files in os.walk(USD_DIR):
        for f in files:
            if f == "iiwa14.usda" or f == "iiwa14.usd":
                return os.path.join(dirpath, f)
    raise FileNotFoundError(f"no converted iiwa14 USD under {USD_DIR}; run --stage usd first")


# ================================================================================================
# gripper
# ================================================================================================


def _nucleus_dir() -> str:
    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    except ImportError:
        return ISAAC_NUCLEUS_FALLBACK
    return ISAAC_NUCLEUS_DIR


def fetch_robotiq() -> None:
    """Every file of Isaac Sim's 2F-85 folder. The root layer pulls in payloads, parts and
    materials by relative path, so the folder is mirrored whole; the asset server is a public S3
    bucket, which lists by prefix."""
    base = _nucleus_dir().rstrip("/") + "/"
    split = urllib.parse.urlsplit(base)
    prefix = split.path.lstrip("/") + ROBOTIQ_SUBDIR
    host = f"{split.scheme}://{split.netloc}/"
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        listing = _fetch(host + "?" + urllib.parse.urlencode(query)).decode()
        for key in re.findall(r"<Key>([^<]+)</Key>", listing):
            rel = key[len(prefix):]
            if rel and not rel.endswith("/") and not rel.endswith(".bzEmpty"):
                _write(os.path.join(ROBOTIQ_DIR, rel), _fetch(host + urllib.parse.quote(key)))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", listing)
        if m is None:
            break
        token = m.group(1)
    if not os.path.isfile(os.path.join(ROBOTIQ_DIR, ROBOTIQ_ROOT_FILE)):
        raise FileNotFoundError(f"{ROBOTIQ_ROOT_FILE} not found under {base + ROBOTIQ_SUBDIR}")
    print("fetched the Robotiq 2F-85 into", os.path.relpath(ROBOTIQ_DIR), flush=True)


def read_stl(path: str):
    """A binary STL's triangles as (points (V, 3), triangle vertex indices (T, 3)), duplicate
    vertices merged (to 1 um)."""
    import numpy as np

    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), np.uint32)[0])
        tri = np.frombuffer(f.read(50 * n), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"]
    flat = tri.reshape(-1, 3).astype(np.float64)
    points, index = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    return points, index.reshape(-1, 3)


def build_tool_meshes() -> None:
    """data/tool/tool_meshes.usdc: the camera mount and the D405 as meshes, in the CAD frame."""
    from pxr import Gf, Usd, UsdGeom, Vt

    os.makedirs(os.path.dirname(TOOL_USD), exist_ok=True)
    if os.path.exists(TOOL_USD):
        os.remove(TOOL_USD)
    stage = Usd.Stage.CreateNew(TOOL_USD)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Tool")
    stage.SetDefaultPrim(root.GetPrim())
    for name, (stl, color) in TOOL_PARTS.items():
        points, faces = read_stl(os.path.join(TOOL_DIR, stl))
        mesh = UsdGeom.Mesh.Define(stage, f"/Tool/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype("float32")))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype("int32")))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
        print(f"  {name}: {len(faces)} triangles, {len(points)} vertices from {stl}", flush=True)
    stage.GetRootLayer().customLayerData = {"cable_assembly": "generated by scripts/build_iiwa14.py from data/tool/*.stl"}
    stage.GetRootLayer().Save()
    print(f"wrote {os.path.relpath(TOOL_USD)}", flush=True)


def _step_triangles(path: str, size_mm: float = 1.5):
    """A STEP file's solids as {name: triangles (T, 3, 3), mm, in the file's frame} (gmsh/OpenCASCADE)."""
    import gmsh
    import numpy as np

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.zeros((int(tags.max()) + 1, 3))
        xyz[tags.astype(int)] = coords.reshape(-1, 3)
        out = {}
        for dim, tag in gmsh.model.getEntities(3):
            tris = []
            for _, s in gmsh.model.getBoundary([(dim, tag)], oriented=False, recursive=False):
                types, _, nodes = gmsh.model.mesh.getElements(2, abs(s))
                tris += [xyz[np.asarray(n, int).reshape(-1, 3)] for t, n in zip(types, nodes) if t == 2]
            out[f"{tag:02d}|{gmsh.model.getEntityName(dim, tag)}"] = np.concatenate(tris)
        return out
    finally:
        gmsh.finalize()


def _box_triangles(lo, hi):
    import numpy as np

    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v[np.array(f)]


def build_v3_visuals() -> None:
    """robotiq_2f85_v3/v3_visuals.usdc: the V3 2F-85's parts as meshes in the gripper's base frame (m),
    one Xform per Isaac link (/V3/<link>), open -- the coupling and the bench's pads included."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, Vt

    step = os.path.join(V3_DIR, V3_STEP)
    if not os.path.isfile(step):
        raise FileNotFoundError(f"{step}: Robotiq's 2F-85 V3 CAD (their 2019-02-07 model set) goes here, by hand")

    def to_base(T):                                   # STEP mm -> base m (see V3_Z_OFF_MM)
        return np.stack([T[..., 2], T[..., 0], T[..., 1] + V3_Z_OFF_MM], axis=-1) / 1000.0

    links = {}                                        # link -> [(triangles in base m, colour key)]
    for name, T in _step_triangles(step).items():
        group = name.split("Group")[1][0] if "Group" in name else None
        side = "left" if T[..., 0].mean() < 0.0 else "right"   # left: base y < 0 (Isaac's finger_joint at -30.6)
        link, color = {"1": ("base_link", "metal"), "6": ("base_link", "metal"), "7": ("base_link", "pins"),
                       "2": (f"{side}_inner_knuckle", "metal"), "3": (f"{side}_inner_finger", "metal"),
                       "4": (f"{side}_outer_finger", "metal"), "5": (f"{side}_outer_knuckle", "metal")}.get(group, (None, None))
        if link is None:
            raise ValueError(f"{V3_STEP}: part {name!r} is not one of the 2F-85's seven")
        links.setdefault(link, []).append((to_base(T), color))
    coupling = os.path.join(V3_DIR, V3_COUPLING_STEP)
    if os.path.isfile(coupling):
        for T in _step_triangles(coupling, 1.0).values():
            links["base_link"].append(((T + [0.0, 0.0, V3_COUPLING_Z_MM]) / 1000.0, "coupling"))
    L, W, H = PAD_MM
    for side, s in (("left", -1.0), ("right", 1.0)):
        x0, x1 = sorted((s * PAD_MOUNT_FACE_MM, s * (PAD_MOUNT_FACE_MM - H)))
        pad = _box_triangles((x0, PAD_CENTRE_ALONG_MM - L / 2, -W / 2), (x1, PAD_CENTRE_ALONG_MM + L / 2, W / 2))
        links[f"{side}_inner_finger"].append((to_base(pad), "pad"))

    if os.path.exists(V3_USD):
        os.remove(V3_USD)
    stage = Usd.Stage.CreateNew(V3_USD)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/V3")
    stage.SetDefaultPrim(root.GetPrim())
    for link, items in links.items():
        UsdGeom.Xform.Define(stage, f"/V3/{link}")
        for color in sorted({c for _, c in items}):
            tri = np.concatenate([T for T, c in items if c == color]).reshape(-1, 3)
            points, index = np.unique(np.round(tri, 7), axis=0, return_inverse=True)
            mesh = UsdGeom.Mesh.Define(stage, f"/V3/{link}/{color}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype("float32")))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(index) // 3)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(index.astype("int32").ravel()))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*V3_COLORS[color])])
            mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
        print(f"  {link}: {sum(len(T) for T, _ in items)} triangles", flush=True)
    stage.GetRootLayer().customLayerData = {"cable_assembly": "generated by scripts/build_iiwa14.py --stage bench "
                                                              f"from {V3_STEP} (+ coupling, pads {PAD_MM} mm)"}
    stage.GetRootLayer().Save()
    print(f"wrote {os.path.relpath(V3_USD)}", flush=True)


def build_gripper(coupler_mm: float, clocking_deg: float, out: str = ROBOT_USD, v3: bool = False) -> None:
    """Author iiwa14_2f85.usda: the converted arm, plus the 2F-85 as a payload under the robot root
    joined to tool0 by a fixed joint. As in Isaac Sim's ur10e.usd, the gripper's own articulation
    root is deleted so that it becomes part of the arm's articulation.

    `v3`: the bench's gripper as it is seen (see V3_DIR) -- each link's V4 visuals deactivated and the V3
    meshes (build_v3_visuals) put on the same link; the articulation is untouched. Written to `out`."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    arm_path = converted_usd()
    arm = Usd.Stage.Open(arm_path)
    root_name = arm.GetDefaultPrim().GetName()
    tool0 = next(p for p in arm.Traverse() if p.GetName() == "tool0")
    now = Usd.TimeCode.Default()
    T_root_tool0 = UsdGeom.Xformable(tool0).ComputeLocalToWorldTransform(now) * UsdGeom.Xformable(
        arm.GetDefaultPrim()
    ).ComputeLocalToWorldTransform(now).GetInverse()

    # Mount: tool0 -> 2F-85 base_link. Gf matrices are row-vector, so a chain reads left to right.
    T_tool0_grip = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), clocking_deg)) * Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(0, 0, coupler_mm / 1000.0)
    )
    T_root_grip = T_tool0_grip * T_root_tool0

    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = stage.DefinePrim(f"/{root_name}", "Xform")
    robot.GetReferences().AddReference(os.path.relpath(arm_path, DATA))
    stage.SetDefaultPrim(robot)

    gripper = UsdGeom.Xform.Define(stage, f"/{root_name}/gripper")
    gripper.GetPrim().GetPayloads().AddPayload(os.path.relpath(os.path.join(ROBOTIQ_DIR, ROBOTIQ_ROOT_FILE), DATA))
    gripper.AddTransformOp().Set(T_root_grip)
    inner = stage.OverridePrim(gripper.GetPath().AppendChild("Robotiq_2F_85"))
    schemas = Sdf.TokenListOp()
    schemas.deletedItems = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    inner.SetMetadata("apiSchemas", schemas)

    # The converter roots the articulation AT link_0, beside the world root_joint. PhysX then builds a
    # floating base held by a maximal-coordinate joint (6 base DOFs, offsetting every Jacobian
    # column). Rooted at the robot prim instead, root_joint makes the base fixed -- the same move
    # Isaac Lab's own fix_root_link makes (isaaclab/sim/schemas/schemas.py). Newton's root schema
    # goes too: under Kit it carries PhysicsArticulationRootAPI with it, a second root.
    root_link = next(p for p in arm.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    if root_link != arm.GetDefaultPrim():
        over = stage.OverridePrim(root_link.GetPath().ReplacePrefix(arm.GetDefaultPrim().GetPath(), robot.GetPath()))
        root_schemas = Sdf.TokenListOp()
        root_schemas.deletedItems = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI", "NewtonArticulationRootAPI"]
        over.SetMetadata("apiSchemas", root_schemas)
        UsdPhysics.ArticulationRootAPI.Apply(robot)

    # The camera mount and the D405, on tool0 where the CAD has them (see TOOL_PARTS): a child of
    # the tool0 body, so it moves with it; meshes only, no collision.
    build_tool_meshes()
    tool0_here = tool0.GetPath().ReplacePrefix(arm.GetDefaultPrim().GetPath(), robot.GetPath())
    stage.OverridePrim(tool0_here)
    mount = UsdGeom.Xform.Define(stage, tool0_here.AppendChild("camera_mount"))
    mount.GetPrim().GetReferences().AddReference(os.path.relpath(TOOL_USD, DATA))
    T_tool0_cad = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -CAD_FLANGE_FACE_Z_M)) * Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(0, 0, 1), CAD_TO_TOOL0_YAW_DEG)
    )
    mount.AddTransformOp().Set(T_tool0_cad)

    joint = UsdPhysics.FixedJoint.Define(stage, f"/{root_name}/gripper_joint")
    joint.CreateBody0Rel().SetTargets([tool0.GetPath().ReplacePrefix(arm.GetDefaultPrim().GetPath(), robot.GetPath())])
    joint.CreateBody1Rel().SetTargets([inner.GetPath().AppendChild("base_link")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(T_tool0_grip.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(T_tool0_grip.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    if v3:
        # Each link's V4 visuals (an instance root) DEACTIVATED -- invisible is not enough: Newton's USD
        # importer (live_view_gl.py) draws invisible meshes -- and the V3 meshes in a sibling on the same
        # body. The bench USD is for the live views, which simulate nothing: the links, joints and their
        # mass properties stay, and nothing collides there.
        build_v3_visuals()
        v3_stage = Usd.Stage.Open(V3_USD)
        for link_prim in v3_stage.GetPrimAtPath("/V3").GetChildren():
            link = inner.GetPath().AppendChild(link_prim.GetName())
            stage.OverridePrim(link.AppendChild("visuals")).SetActive(False)
            seen = UsdGeom.Xform.Define(stage, link.AppendChild("visuals_v3"))
            seen.GetPrim().GetReferences().AddReference(os.path.relpath(V3_USD, DATA), link_prim.GetPath())

    stage.GetRootLayer().customLayerData = {
        "cable_assembly": json.dumps({"coupler_mm": coupler_mm, "clocking_deg": clocking_deg, "pin": PIN,
                                      "gripper_visuals": "Robotiq 2F-85 V3 + bench pads" if v3 else "Isaac Sim 2F-85 (V4)"})
    }
    stage.GetRootLayer().Save()
    print(f"wrote {os.path.relpath(out)} (coupler {coupler_mm} mm, clocking {clocking_deg} deg"
          f"{', V3 visuals' if v3 else ''})", flush=True)


# ================================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=["all", "urdf", "usd", "gripper", "bench"], default="all",
                        help="bench: iiwa14_2f85_bench.usda, the bench's V3 gripper on its measured coupler")
    parser.add_argument("--coupler_mm", type=float, default=COUPLER_MM, help="tool0 -> 2F-85 base along Z")
    parser.add_argument("--clocking_deg", type=float, default=CLOCKING_DEG, help="2F-85 rotation about tool0 Z")
    try:
        from isaaclab.app import add_launcher_args
    except ImportError:
        add_launcher_args = None
    else:
        add_launcher_args(parser)
    args = parser.parse_args()
    if args.stage in ("all", "usd") and add_launcher_args is None:
        parser.error("the usd stage needs Isaac Lab: run it as `./run.sh python scripts/build_iiwa14.py` in sim/")

    def gripper() -> None:
        if not os.path.isfile(os.path.join(ROBOTIQ_DIR, ROBOTIQ_ROOT_FILE)):
            fetch_robotiq()
        build_gripper(args.coupler_mm, args.clocking_deg)

    def bench() -> None:
        if not os.path.isfile(os.path.join(ROBOTIQ_DIR, ROBOTIQ_ROOT_FILE)):
            fetch_robotiq()
        build_gripper(BENCH_COUPLER_MM, args.clocking_deg, out=BENCH_USD, v3=True)

    if args.stage in ("all", "urdf"):
        build_urdf()
    if args.stage in ("all", "usd"):
        build_usd(args, then=gripper if args.stage == "all" else None)
    elif args.stage == "gripper":
        gripper()
    elif args.stage == "bench":
        bench()


if __name__ == "__main__":
    main()
