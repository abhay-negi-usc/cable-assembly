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
             construction Isaac Sim's own ur10e.usd uses for its "Robotiq_2f_85" gripper variant.

    cd sim
    uv run --extra importers python scripts/build_iiwa14.py        # all three
    python scripts/build_iiwa14.py --stage urdf                    # any Python with PyYAML

Only the URDF and its LICENSE are tracked. The meshes, the converted USD and the Robotiq files
are fetched or generated into the same directory and ignored by git.
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


def build_usd(args) -> str:
    """URDF -> USD with Isaac Lab's converter. Mirrors scripts/tools/convert_urdf.py of Isaac Lab
    3.0: prefer the kit-less standalone importer, and run the converter inside launch_simulation."""
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
    print("wrote", os.path.relpath(usd_path))
    return usd_path


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
    print("fetched the Robotiq 2F-85 into", os.path.relpath(ROBOTIQ_DIR))


def build_gripper(coupler_mm: float, clocking_deg: float) -> None:
    """Author iiwa14_2f85.usda: the converted arm, plus the 2F-85 as a payload under the robot root
    joined to tool0 by a fixed joint. As in Isaac Sim's ur10e.usd, the gripper's own articulation
    root is deleted so that it becomes part of the arm's articulation."""
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

    stage = Usd.Stage.CreateNew(ROBOT_USD)
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

    joint = UsdPhysics.FixedJoint.Define(stage, f"/{root_name}/gripper_joint")
    joint.CreateBody0Rel().SetTargets([tool0.GetPath().ReplacePrefix(arm.GetDefaultPrim().GetPath(), robot.GetPath())])
    joint.CreateBody1Rel().SetTargets([inner.GetPath().AppendChild("base_link")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(T_tool0_grip.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(T_tool0_grip.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    stage.GetRootLayer().customLayerData = {
        "cable_assembly": json.dumps({"coupler_mm": coupler_mm, "clocking_deg": clocking_deg, "pin": PIN})
    }
    stage.GetRootLayer().Save()
    print(f"wrote {os.path.relpath(ROBOT_USD)} (coupler {coupler_mm} mm, clocking {clocking_deg} deg)")


# ================================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=["all", "urdf", "usd", "gripper"], default="all")
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
        parser.error("the usd stage needs Isaac Lab: run it as `uv run --extra importers python ...` in sim/")

    if args.stage in ("all", "urdf"):
        build_urdf()
    if args.stage in ("all", "usd"):
        build_usd(args)
    if args.stage in ("all", "gripper"):
        if not os.path.isfile(os.path.join(ROBOTIQ_DIR, ROBOTIQ_ROOT_FILE)):
            fetch_robotiq()
        build_gripper(args.coupler_mm, args.clocking_deg)


if __name__ == "__main__":
    main()
