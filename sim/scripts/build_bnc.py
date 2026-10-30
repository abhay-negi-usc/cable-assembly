"""Build the bench's BNC plug and socket USDs from McMaster-Carr's STEP files.

THE PARTS. The iiwa bench plugs McMaster 6641T31 (a 12 in coaxial cord with a BNC plug at one end)
into 4441N11 (a panel-mount coaxial adapter). Their STEP files are downloaded by hand into
assets/data/bnc/ -- McMaster's CAD is theirs to distribute, so that directory is ignored by git --
and this script writes the two USDs the BNC task spawns beside them:

    bnc_plug.usd    an ARTICULATION, because the plug's bayonet collar turns and slides on its body:
                        body (strain relief, contact fingers, insulator, pin, a stub of cable)
                          -> collar_slide   prismatic along the axis, sprung: the collar's wave washer
                          -> collar_turn    revolute about the axis, free
                          -> collar
    bnc_socket.usd  one rigid body: the adapter's BNC half, in front of its mounting boss

Both are authored in the SOCKET frame (the adapter's own CAD frame: z up its axis, the pins on +/-y,
z = 0 at the bottom of the boss) with the plug SEATED -- its insulator on the socket's dielectric --
as Isaac Lab's NIST connectors are: a plug pose equal to the socket pose is a seated plug, and the
collar joints at 0 put the collar's slot entries on the socket's pins.

WHAT IS CHANGED FROM MCMASTER'S CAD, and why:
  - The collar is cut free of the disc that closes the back of the plug; in the CAD they are one
    solid, and the collar has to turn on the body.
  - The collar gets its second J-slot. The CAD cuts one slot in full; the other is only its entry
    notch, which would stop the collar turning with both pins in. The collar kept is the CAD collar
    intersected with itself turned half a turn.
  - The plug keeps PLUG_END_Z_MM of the cord: the other connector and the rest of the cable go.
    The wave washer (a torus in the CAD) goes too; the slide's drive is the spring.
  - Only the socket's BNC half, z >= 0, is kept.

    cd sim
    ./run.sh --extra cad python scripts/build_bnc.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BNC_DIR = os.path.join(HERE, "..", "src", "cable_assembly_sim", "assets", "data", "bnc")
PLUG_STEP = os.path.join(BNC_DIR, "6641T31_coaxial_cord.step")
SOCKET_STEP = os.path.join(BNC_DIR, "4441N11_panel_mount_coaxial_adapter.step")
PLUG_USD = os.path.join(BNC_DIR, "bnc_plug.usd")
SOCKET_USD = os.path.join(BNC_DIR, "bnc_socket.usd")

# ---- geometry, in the STEP files' own frames [mm] ------------------------------------------------
# The cord runs along z with the BNC plug at +z, mating face at z 152.40. Its solids, told apart by
# their extents (radius, z range) rather than their STEP order:
PLUG_SOLIDS = {
    "collar": (7.24, 136.53, 152.40),  # with the disc that closes the plug's back fused in
    "boot": (7.97, 108.65, 137.29),  # the strain relief: where the gripper holds the plug
    "insulator": (3.81, 138.88, 151.38),
    "fingers": (3.97, 138.88, 151.19),  # the outer contact, slotted into four fingers
    "pin": (0.57, 145.03, 150.34),
    "washer": (5.33, 138.88, 139.83),  # the wave washer, modelled as a torus: dropped
    "cable": (2.54, -118.57, 111.82),
}
PLUG_END_Z_MM = 100.0
"""Keep the cord from here to the plug: the strain relief ends at 108.65, so 8.65 mm of cable."""
COLLAR_SPLIT_R_MM = 4.84
"""Radius that cuts the collar from the fused back disc: between the disc's rear stub (4.762) and
the collar's bore (4.921)."""
SOCKET_KEEP_Z_MM = 0.0
"""Keep the socket from here up: its BNC half, with the well the collar goes down into."""
SEAT_Z_MM = 151.38 + 3.73
"""Seated, the plug's insulator face (151.38 along the cord) rests on the socket's dielectric (3.73
up the socket), plug facing down: z_socket = SEAT_Z_MM - z_cord, y_socket = -y_cord."""

MESH_SIZE_MM = 0.35
"""Largest triangle edge. Its chord error on the 4.8 mm fits is under 0.01 mm, a tenth of their
clearance."""

# Densities [g/cm^3]: nickel-plated brass shells, PTFE insulator, a moulded PVC strain relief, and
# RG-58 (40 g/m) for the cable stub.
DENSITY = {"collar": 8.5, "disc": 8.5, "fingers": 8.5, "pin": 8.5, "insulator": 2.2, "boot": 1.3, "cable": 2.0}
SLIDER_MASS_KG = 0.001
"""The massless link between the slide and the turn, given the mass and inertia of a 1 g ring the
collar's size: a near-zero inertia in the chain blew the solver up when the collar hit the pins."""
SLIDER_RING_R_M = 0.005

# The collar's wave washer: the slide's drive, pulling the collar back against its rear stop.
SLIDE_TRAVEL_MM = 1.2
"""Forward (towards the socket) travel of the collar on the body. Locking needs ~0.7 mm: the pocket
holds the collar 0.31 mm forward with the plug seated, and the detent ridge 0.39 mm more."""
SLIDE_STIFFNESS_N_PER_M = 5000.0
SLIDE_PRELOAD_MM = 1.0
"""Drive target behind the rear stop: 5 N of preload at rest, 11 N fully compressed."""

COLOURS = {
    "collar": (0.72, 0.72, 0.74),
    "fingers": (0.72, 0.72, 0.74),
    "disc": (0.72, 0.72, 0.74),
    "pin": (0.85, 0.66, 0.25),
    "insulator": (0.92, 0.92, 0.90),
    "boot": (0.05, 0.05, 0.05),
    "cable": (0.05, 0.05, 0.05),
    "socket": (0.72, 0.72, 0.74),
    "seat": (0.30, 0.30, 0.32),
}
# Pieces that touch something: the socket, each other across the slide, or the gripper's pads.
COLLIDERS = {"collar", "boot", "insulator", "fingers", "socket", "seat"}

# THE COLLAR'S SEAT on the body: where its rear flange bears (through the wave washer) when the
# plug is pushed onto the pins. The CAD leaves this 0.75 mm between the flange and the strain
# relief empty; a ring on the body fills it. Its friction is what turns the collar WITH the body
# while the collar is pushed back onto it -- the bench's collar follows the strain relief to find
# its slots -- and what lets it go once the helix pulls the collar forward off the seat to lock,
# which turning the strain relief then cannot do. Friction that scales with the axial load, which
# no constant joint friction can give. Its friction (0.35, 0.325 against the collar's 0.3) is set
# between two bounds: over the pins' (0.3 at r 5.3 mm) so the collar follows when pushed onto
# them, and under the helix's self-locking (seat torque / pin r tan 20 deg < 1), which at 0.6 jammed
# the body against its collar at the end of the cam. Socket frame [mm], plug seated: (inner r,
# outer r), (z, z).
COLLAR_SEAT_R_MM = (4.95, 6.80)
COLLAR_SEAT_Z_MM = (17.08, 17.82)
# Their surfaces (static = dynamic friction): the moulded strain relief is rubber, where the
# gripper's rubber pads hold it; the rest is plated metal and PTFE. PhysX averages the two sides of
# a contact, so the pads (1.0, set on the robot) hold the boot at 1.0 and the collar rides the
# socket's pins at 0.3 -- the value the bayonet's self-locking was measured at.
FRICTION = {"boot": 1.0, "collar": 0.3, "insulator": 0.3, "fingers": 0.3, "socket": 0.3, "seat": 0.35}
# The strain relief is RUBBER, and rigid it is not the part the bench holds: flat pads squeezing a
# rigid bulge sit on its crest, and the first axial shift puts them on its taper, where the squeeze
# itself ejects the hand (the plug walked 29 mm out of a 200 N grip under the engage's loads). A
# compliant contact lets the pads bed ~1 mm into it at the 2F-85's ~200 N, as into the moulding --
# and, the bulge standing only 0.75 mm proud of the collar, reach the collar too when they grip
# both. (stiffness N/m, damping N s/m)
COMPLIANT = {"boot": (2.0e5, 50.0)}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _classify(gmsh, volumes: list[int]) -> dict[str, int]:
    """Name the cord's solids by their extents, and refuse a file that does not match."""
    named = {}
    for name, (r, z0, z1) in PLUG_SOLIDS.items():
        best, err = None, math.inf
        for tag in volumes:
            b = gmsh.model.getBoundingBox(3, tag)
            e = abs(max(abs(v) for v in (b[0], b[1], b[3], b[4])) - r) + abs(b[2] - z0) + abs(b[5] - z1)
            if e < err:
                best, err = tag, e
        if err > 0.5:
            raise SystemExit(f"{os.path.basename(PLUG_STEP)}: no solid looks like the {name} (off by {err:.2f} mm)")
        named[name] = best
    return named


def _meshes(gmsh, volumes: dict[str, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mesh every surface once, then gather each solid's closed shell: (points [mm], triangles)."""
    import trimesh

    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MM)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags, coords = tags[order], coords.reshape(-1, 3)[order]
    out = {}
    for name, vol in volumes.items():
        tris = []
        for _, surf in gmsh.model.getBoundary([(3, vol)], oriented=True):
            types, _, nodes = gmsh.model.mesh.getElements(2, abs(surf))
            for t, n in zip(types, nodes):
                if t == 2:  # 3-node triangles
                    tri = np.searchsorted(tags, np.asarray(n).reshape(-1, 3))
                    tris.append(tri if surf > 0 else tri[:, ::-1])
        mesh = trimesh.Trimesh(coords, np.vstack(tris), process=True)
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight:
            raise SystemExit(f"the {name} shell is not closed: an SDF collider needs a closed mesh")
        out[name] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return out


def plug_geometry(gmsh) -> tuple[dict, dict]:
    """The plug's pieces in the socket frame [m] and their masses [kg]."""
    gmsh.clear()
    volumes = [t for d, t in gmsh.model.occ.importShapes(PLUG_STEP) if d == 3]
    gmsh.model.occ.synchronize()
    named = _classify(gmsh, volumes)
    occ = gmsh.model.occ

    # The cord's far end and the washer go; the cable is cut back to a stub.
    keep = {named[n] for n in ("collar", "boot", "insulator", "fingers", "pin", "cable")}
    occ.remove([(3, t) for t in volumes if t not in keep], recursive=True)
    box = occ.addBox(-20, -20, PLUG_END_Z_MM, 40, 40, 100)
    (cable,), _ = occ.intersect([(3, named["cable"])], [(3, box)])

    # The collar, cut free of the back disc ...
    cyl = occ.addCylinder(0, 0, 130, 0, 0, 30, COLLAR_SPLIT_R_MM)
    ring, _ = occ.cut([(3, named["collar"])], [(3, cyl)], removeObject=False, removeTool=False)
    (disc,), _ = occ.intersect([(3, named["collar"])], [(3, cyl)])
    ring_volume = sum(occ.getMass(*d) for d in ring)
    # ... and given its second J-slot: material only where the collar turned half a turn has it too.
    turned = occ.copy(ring)
    occ.rotate(turned, 0, 0, 0, 0, 0, 1, math.pi)
    collar, _ = occ.intersect(ring, turned)
    if len(collar) != 1:
        raise SystemExit(f"the collar came out as {len(collar)} solids")
    occ.synchronize()
    print(f"collar: {ring_volume:.1f} mm^3 as cut free, {occ.getMass(*collar[0]):.1f} with its second slot")

    pieces = {
        "collar": collar[0][1],
        "disc": disc[1],
        "boot": named["boot"],
        "insulator": named["insulator"],
        "fingers": named["fingers"],
        "pin": named["pin"],
        "cable": cable[1],
    }
    masses = {n: occ.getMass(3, t) * DENSITY[n] * 1e-6 for n, t in pieces.items()}  # mm^3 * g/cm^3 -> kg
    meshes = _meshes(gmsh, pieces)
    # Into the socket frame, seated: half a turn about x, then up to the seat. The collar's slot
    # entries (+/-90 deg about the cord's z) land on the socket's pins (+/-90 deg about its z).
    for name, (p, f) in meshes.items():
        meshes[name] = (np.column_stack([p[:, 0], -p[:, 1], SEAT_Z_MM - p[:, 2]]) * 1e-3, f[:, ::-1])
    return meshes, masses


def socket_geometry(gmsh) -> tuple[dict, float]:
    gmsh.clear()
    volumes = [(d, t) for d, t in gmsh.model.occ.importShapes(SOCKET_STEP) if d == 3]
    if len(volumes) != 1:
        raise SystemExit(f"{os.path.basename(SOCKET_STEP)}: expected one solid, found {len(volumes)}")
    box = gmsh.model.occ.addBox(-40, -40, SOCKET_KEEP_Z_MM, 80, 80, 100)
    # Cut there, the barrel and the outer ring are separate solids (they join further back): one
    # rigid body all the same.
    pieces, _ = gmsh.model.occ.intersect(volumes, [(3, box)])
    gmsh.model.occ.synchronize()
    mass = sum(gmsh.model.occ.getMass(*p) for p in pieces) * 8.5e-6
    meshes = _meshes(gmsh, {f"socket_{i}": t for i, (_, t) in enumerate(pieces)})
    return {n: (p * 1e-3, f) for n, (p, f) in meshes.items()}, mass


# ---- USD ------------------------------------------------------------------------------------------


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    metal = rgb not in (COLOURS["boot"], COLOURS["insulator"])
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.9 if metal else 0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35 if metal else 0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, friction: float, compliant=None):
    from pxr import Sdf, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    if compliant is not None:
        # PhysX's compliant contact, by name: usd-core does not know PhysX's schemas.
        prim = mat.GetPrim()
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(compliant[0])
        prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(compliant[1])
    return mat


def _mesh(stage, path: str, points, faces, material, collide: bool, physics=None) -> None:
    from pxr import UsdGeom, UsdPhysics, UsdShade, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        # Signed-distance-field contact, as Isaac Lab's NIST connectors: the fits are 0.1 mm. PhysX
        # reads approximation "sdf" only with its SDF schema applied -- by name, as usd-core does not
        # know PhysX's schemas; without it a moving part falls back to its convex hull.
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("sdf")
        mesh.GetPrim().AddAppliedSchema("PhysxSDFMeshCollisionAPI")
        if physics is not None:
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
                physics, UsdShade.Tokens.weakerThanDescendants, "physics"
            )


def _stage(path: str, root: str, meta: dict):
    from pxr import Usd, UsdGeom

    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, f"/{root}").GetPrim()
    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().customLayerData = {"cable_assembly": json.dumps(meta)}
    return stage, prim


def _link(stage, path: str, mass: float, inertia=None):
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    if inertia is not None:
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    return prim


def seat_geometry():
    """The collar's seat (see COLLAR_SEAT_R_MM): a ring on the body, in the socket frame [m], and its mass."""
    import trimesh

    (r0, r1), (z0, z1) = COLLAR_SEAT_R_MM, COLLAR_SEAT_Z_MM
    ring = trimesh.creation.annulus(r_min=r0 * 1e-3, r_max=r1 * 1e-3, height=(z1 - z0) * 1e-3, sections=96)
    ring.apply_translation([0.0, 0.0, 0.5 * (z0 + z1) * 1e-3])
    return (np.asarray(ring.vertices), np.asarray(ring.faces)), float(ring.volume) * DENSITY["collar"] * 1e3


def write_plug(meshes: dict, masses: dict, meta: dict) -> None:
    from pxr import Gf, UsdPhysics

    meshes["seat"], masses["seat"] = seat_geometry()
    stage, root = _stage(PLUG_USD, "BncPlug", meta)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    looks = {n: _material(stage, f"/BncPlug/Looks/{n}", c) for n, c in COLOURS.items() if n != "socket"}
    body_pieces = ("boot", "insulator", "fingers", "pin", "disc", "cable", "seat")
    surfaces = {
        n: _physics_material(stage, f"/BncPlug/Surfaces/{n}", f, COMPLIANT.get(n))
        for n, f in FRICTION.items()
        if n != "socket"
    }
    body = _link(stage, "/BncPlug/body", sum(masses[n] for n in body_pieces))
    for n in body_pieces:
        _mesh(stage, f"{body.GetPath()}/{n}", *meshes[n], looks[n], n in COLLIDERS, surfaces.get(n))
    i = SLIDER_MASS_KG * SLIDER_RING_R_M**2
    slider = _link(stage, "/BncPlug/slider", SLIDER_MASS_KG, (i / 2, i / 2, i))
    collar = _link(stage, "/BncPlug/collar", masses["collar"])
    _mesh(stage, f"{collar.GetPath()}/collar", *meshes["collar"], looks["collar"], True, surfaces["collar"])

    def frames(joint):
        for k in (0, 1):
            getattr(joint, f"CreateLocalPos{k}Attr")().Set(Gf.Vec3f(0, 0, 0))
            getattr(joint, f"CreateLocalRot{k}Attr")().Set(Gf.Quatf(1, 0, 0, 0))

    slide = UsdPhysics.PrismaticJoint.Define(stage, "/BncPlug/joints/collar_slide")
    slide.CreateBody0Rel().SetTargets([body.GetPath()])
    slide.CreateBody1Rel().SetTargets([slider.GetPath()])
    slide.CreateAxisAttr("Z")
    # The plug points down -z: the collar compresses its washer moving -z on the body. The rear
    # limit stands 0.05 mm behind the seat (0.01 mm), so the seat's CONTACT -- and its friction --
    # carries the collar's load, not the limit.
    slide.CreateLowerLimitAttr(-SLIDE_TRAVEL_MM * 1e-3)
    slide.CreateUpperLimitAttr(0.05e-3)
    frames(slide)
    drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(SLIDE_STIFFNESS_N_PER_M)
    drive.CreateDampingAttr(2.0 * math.sqrt(SLIDE_STIFFNESS_N_PER_M * masses["collar"]))
    drive.CreateTargetPositionAttr(SLIDE_PRELOAD_MM * 1e-3)

    turn = UsdPhysics.RevoluteJoint.Define(stage, "/BncPlug/joints/collar_turn")
    turn.CreateBody0Rel().SetTargets([slider.GetPath()])
    turn.CreateBody1Rel().SetTargets([collar.GetPath()])
    turn.CreateAxisAttr("Z")
    frames(turn)
    # Free, but with a drive of zero gains: PhysX applies the gains and targets Isaac Lab writes only to
    # a joint that has one (a collar twist, a brake), and ignores them silently otherwise.
    drive = UsdPhysics.DriveAPI.Apply(turn.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    stage.GetRootLayer().Save()


def write_socket(meshes: dict, mass: float, meta: dict) -> None:
    stage, _ = _stage(SOCKET_USD, "BncSocket", meta)
    _link(stage, "/BncSocket/socket", mass)
    look = _material(stage, "/BncSocket/Looks/socket", COLOURS["socket"])
    surface = _physics_material(stage, "/BncSocket/Surfaces/socket", FRICTION["socket"])
    for name, (points, faces) in meshes.items():
        _mesh(stage, f"/BncSocket/socket/{name}", points, faces, look, True, surface)
    stage.GetRootLayer().Save()


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    for path in (PLUG_STEP, SOCKET_STEP):
        if not os.path.isfile(path):
            raise SystemExit(f"missing {os.path.relpath(path)}: download it from McMaster-Carr (3-D STEP)")
    try:
        import gmsh
    except ImportError:
        raise SystemExit("gmsh is missing: run `./run.sh --extra cad python scripts/build_bnc.py`") from None

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")  # the files are in inches
    try:
        plug, masses = plug_geometry(gmsh)
        socket, socket_mass = socket_geometry(gmsh)
    finally:
        gmsh.finalize()

    meta = {
        "plug_step": {"file": os.path.basename(PLUG_STEP), "sha256": _sha256(PLUG_STEP)},
        "socket_step": {"file": os.path.basename(SOCKET_STEP), "sha256": _sha256(SOCKET_STEP)},
        "seat_z_mm": SEAT_Z_MM,
        "slide_travel_mm": SLIDE_TRAVEL_MM,
        "slide_stiffness_n_per_m": SLIDE_STIFFNESS_N_PER_M,
        "slide_preload_mm": SLIDE_PRELOAD_MM,
        "friction": FRICTION,
        "collar_seat_r_mm": COLLAR_SEAT_R_MM,
        "collar_seat_z_mm": COLLAR_SEAT_Z_MM,
        "compliant": COMPLIANT,
    }
    write_plug(plug, masses, meta)
    write_socket(socket, socket_mass, meta)
    for name, (p, f) in {**plug, **socket}.items():
        z = p[:, 2] * 1e3
        grams = f"  {masses[name] * 1e3:6.2f} g" if name in masses else ""
        print(f"  {name:9s} {len(f):6d} triangles  z {z.min():7.2f}..{z.max():7.2f} mm{grams}")
    print(f"  plug {sum(masses.values()) * 1e3:.1f} g, socket {socket_mass * 1e3:.1f} g")
    print(f"wrote {os.path.relpath(PLUG_USD)} and {os.path.relpath(SOCKET_USD)}")


if __name__ == "__main__":
    main()
