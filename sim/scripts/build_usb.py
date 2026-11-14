"""Build the USB-A plug and receptacle USDs from McMaster-Carr's STEP files.

THE PARTS. 4974T135 is a USB extension cord: a USB-A PLUG at one end (+z in its STEP), a USB-A
receptacle at the other. 4574N11 is a panel-mount USB adapter (a D-series round body): a USB-A
RECEPTACLE on its front, above a square flange. Their STEP files are downloaded by hand into
assets/data/usb/ (McMaster's CAD is theirs to distribute: that directory is ignored by git), and this
script writes beside them:

    usb_plug.usd     one rigid body: the plug's shell + insert as ONE solid collider (`plug`), and the
                     overmold, contacts and a stub of cable for drawing
    usb_socket.usd   one rigid body: the adapter's front as a solid collider (`receptacle`: the pocket, the
                     tongue, the lead-in, the face, the rim and the flange), and the CAD for drawing

Both are authored in the SOCKET frame -- the adapter's own CAD frame: z up the receptacle axis, the
plug going in along -z, the flange's front face at z = 0 -- with the plug SEATED (its tip on the
pocket floor), as build_bnc.py does: a plug pose equal to the socket pose is a seated plug.

The cord's plug frame -> socket frame, seated: half a turn about y (which keeps the plug's tongue
cavity on the receptacle tongue's side, +y), the shells centred on each other, then down onto the
floor:  (x, y, z)_socket = (-x, y + PLUG_DY, SEAT_Z - z)_cord.

WHAT IS CHANGED FROM McMASTER'S CAD, and why (the BNC lesson: thin-shell SDFs give contacts from both
faces with flipped normals, and launched seated plugs):
  - The PLUG collider is the shell's outer envelope (12.0 x 4.5 mm, its rounded corners) minus the
    tongue cavity (11.35 x 1.95 mm, 12 mm deep): what the 0.33 mm sheet-metal shell and the insert are
    to the receptacle, as one solid. The four contacts and the shell's latch dimples go.
  - The RECEPTACLE collider is the adapter's CAD with its 0.3 mm sheet-metal shell made solid outward
    (filled out to the housing's opening) and its pocket cut clean at the shell's inner size (12.50 x
    5.12 mm, the USB-A spec) with a 0.5 mm flare at its mouth (the shell's lip). The cut also takes out
    the shell's six detent springs, which stand 0.3 mm into the plug's path: rigid, they would only
    jam. The env models them as the insertion resistance they are to the arm (env.py, `detent`).
    The plastic tongue (11.10 x 1.84 mm) is kept as the CAD has it, chamfer and all, but 0.15 mm thinner on
    each broad face (TONGUE_RELIEF): the give of the contacts on it and on the plug's cavity floor.
  - Only the adapter's front is kept (|x|, |y| < 16.5, z > -5.5), its hollow D-body filled under the
    housing's top plate.

    cd sim
    ./run.sh --extra cad python scripts/build_usb.py           # or any env with gmsh, trimesh, usd-core
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_bnc import _material, _mesh, _physics_material, _sha256, _stage  # noqa: E402  (shared USD helpers)

USB_DIR = os.path.join(HERE, "..", "src", "cable_assembly_sim", "assets", "data", "usb")
PLUG_STEP = os.path.join(USB_DIR, "4974T135_usb_cord.step")
SOCKET_STEP = os.path.join(USB_DIR, "4574N11_panel_mount_usb_adapter.step")
PLUG_USD = os.path.join(USB_DIR, "usb_plug.usd")
SOCKET_USD = os.path.join(USB_DIR, "usb_socket.usd")

# ---- the cord's plug end, in its STEP frame [mm] ---------------------------------------------------
# Solids told apart by their extents (half-width x, half-height y, z range), not their STEP order.
PLUG_SOLIDS = {
    "shell": (6.00, 2.25, 455.20, 469.20),  # 0.33 mm sheet metal
    "insert": (5.65, 1.93, 454.20, 469.20),  # the contacts' carrier: fills y < -0.025, and all of y below z 457.2
    "overmold": (8.66, 4.01, 421.20, 457.20),  # 16 x 8 mm (its bounding box is a little larger): what the gripper holds
}
CONTACT_Z = (463.45, 467.96)
"""The four contacts (z range); drawn only."""
SHELL_HALF = (6.00, 2.25)
SHELL_CORNER_R = 0.50
SHELL_Z = (457.20, 469.20)
"""The exposed shell: from the overmold's front face to the tip."""
TIP_ROUND_R = 0.25
"""The shell's rolled tip edge."""
CAVITY_HALF_X = 5.675
CAVITY_Y = (-0.025, 1.925)
"""The tongue cavity: between the insert's top face and the shell's inside, the shell's inside width."""
CABLE_KEEP_MM = 50.0
"""The cable kept past the overmold (z 421.2 in the cord's frame): 5 cm of the constant-diameter cord is
all the task needs. Part of the plug's rigid body (its mass, and drawn); not a collider -- it runs up the
plug axis between the pads, clear of the socket. Its bending stiffness and pull are not modelled."""
OVERMOLD_BACK_Z = 421.2

# ---- the adapter, in its STEP frame = the socket frame [mm] ----------------------------------------
POCKET_HALF_X = 6.25
POCKET_Y = (-2.089, 3.031)
"""The receptacle shell's inside: 12.50 x 5.12 mm, centred at y 0.471."""
POCKET_FLOOR_Z = -4.837
POCKET_TOP_Z = 5.04
POCKET_FLARE = 0.5
"""The shell's lip: the pocket widens by this much per side from POCKET_TOP_Z to SHELL_TOP_Z."""
SHELL_TOP_Z = 5.54
PLATE_UNDERSIDE_Z = 4.90
"""The underside of the housing's top plate (z 4.90-6.20 round the opening, its lead-in chamfered)."""
OPENING_HALF_X = 6.931
OPENING_Y = (-2.897, 3.839)
"""The housing's rectangular opening above the shell (z 5.54-6.2); the solid fill goes out to it."""
TONGUE_BOX = ((-5.60, 5.60), (0.40, 2.36), (-5.5, 5.40))
"""A box round the tongue (11.10 x 1.84, z to 5.31): inside it the CAD is kept as is, but for ..."""
TONGUE_Y = (0.471, 2.311)
TONGUE_RELIEF = 0.15
""" ... this much off each of its broad faces. The tongue carries the receptacle's four contacts and the plug's
cavity floor its four spring contacts: in the real fit they flex by a few tenths of a mm. Rigid, the CAD's
0.04 / 0.07 mm under / over the tongue wedged a plug turned 3 deg in the hand hard enough to drive the tongue's
edge past the middle of the plug's 0.33 mm wall, whose SDF then fired it out (3.5% of full-error episodes)."""
RECEPTACLE_CROP = ((-16.5, 16.5), (-16.5, 16.5), (-5.5, 6.70))
"""All of the adapter's front: a plug that slides off the face meets the rim and the flange, never a void
(at |x| < 11 a plug slid 10.8 mm sideways off the collider's edge, in 1 of 256 full-error episodes)."""
BODY_FILL_R = 11.70
"""The fill under the housing's top plate stays inside the D-body's outer wall (r 11.8): outside it the CAD
(the wall, the flange) is kept as it is."""
DRAW_CROP_Z = -24.0

PLUG_DY = 0.456
"""The plug's y in the socket, seated: the shells' centres (0.471) and the cavity on the tongue (0.441)
split -- 0.30-0.33 mm of shell clearance, 0.04 / 0.07 mm under / over the tongue."""
SEAT_Z = SHELL_Z[1] + POCKET_FLOOR_Z
"""Seated, the plug's tip is on the pocket floor: z_socket = SEAT_Z - z_cord."""

MESH_SIZE_MM = 0.25
COARSE_MM = 1.0
FINE_REGION = ((-8.5, 8.5), (-4.5, 5.5), (-5.0, 7.0))
"""The receptacle is meshed at MESH_SIZE_MM inside this box (the pocket, the tongue, the lead-in and the
face round it, where the plug can land), at COARSE_MM outside."""
DRAW_MESH_SIZE_MM = 0.6
DENSITY = {"plug": 6.0, "overmold": 1.3, "cable": 1.5}
"""g/cm^3: the shell + insert as steel and nylon together, a PVC overmold, a 5.5 mm cable."""

COLOURS = {
    "plug": (0.75, 0.75, 0.78),
    "overmold": (0.08, 0.08, 0.08),
    "cable": (0.08, 0.08, 0.08),
    "contacts": (0.85, 0.66, 0.25),
    "receptacle": (0.75, 0.75, 0.78),
    "housing": (0.10, 0.10, 0.11),
    "adapter": (0.10, 0.10, 0.11),
}
FRICTION = {"plug": 0.3, "receptacle": 0.3, "housing": 0.4}
"""Plated steel on plated steel and on the nylon tongue; the env randomises the plug-on-socket value."""


def _box(occ, lo, hi):
    return occ.addBox(lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])


def _crop(occ, vols, crop):
    (x0, x1), (y0, y1), (z0, z1) = crop
    out, _ = occ.intersect(vols, [(3, _box(occ, (x0, y0, z0), (x1, y1, z1)))])
    return out


def _vertical_edges(gmsh, vol, z0, z1):
    """Edges of `vol` that run the full height z0..z1 (a box's four upright edges)."""
    out = []
    for _, s in gmsh.model.getBoundary([(3, vol)], oriented=False):
        for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            b = gmsh.model.getBoundingBox(1, c)
            if abs(b[2] - z0) < 1e-3 and abs(b[5] - z1) < 1e-3 and abs(b[0] - b[3]) < 1e-6 and abs(b[1] - b[4]) < 1e-6:
                out.append(abs(c))
    return sorted(set(out))


def _rect_wire(occ, half_x, y0, y1, z):
    pts = [occ.addPoint(x, y, z) for x, y in ((-half_x, y0), (half_x, y0), (half_x, y1), (-half_x, y1))]
    lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return occ.addWire(lines)


def _mesh_volumes(gmsh, volumes: dict[str, list[int]], size: float, fine=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mesh, then gather each named group of solids as one closed shell: (points [mm], triangles)."""
    import trimesh

    gmsh.option.setNumber("Mesh.MeshSizeMax", size if fine is None else COARSE_MM)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.1)
    if fine is not None:
        # Fine where the parts meet, coarse elsewhere (each triangle near a contact is solver work).
        (x0, x1), (y0, y1), (z0, z1) = fine
        gmsh.model.mesh.setSizeCallback(
            lambda dim, tag, x, y, z, lc: min(lc, size if (x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1) else COARSE_MM))
    gmsh.model.mesh.generate(2)
    if fine is not None:
        gmsh.model.mesh.removeSizeCallback()
    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags, coords = tags[order], coords.reshape(-1, 3)[order]
    out = {}
    for name, vols in volumes.items():
        tris = []
        for vol in vols:
            for _, surf in gmsh.model.getBoundary([(3, vol)], oriented=True):
                types, _, nodes = gmsh.model.mesh.getElements(2, abs(surf))
                for t, n in zip(types, nodes):
                    if t == 2:
                        tri = np.searchsorted(tags, np.asarray(n).reshape(-1, 3))
                        tris.append(tri if surf > 0 else tri[:, ::-1])
        mesh = trimesh.Trimesh(coords, np.vstack(tris), process=True)
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight:
            raise SystemExit(f"the {name} shell is not closed: an SDF collider needs a closed mesh")
        out[name] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return out


def _classify(gmsh, volumes):
    named = {}
    for name, (hx, hy, z0, z1) in PLUG_SOLIDS.items():
        best, err = None, np.inf
        for tag in volumes:
            b = gmsh.model.getBoundingBox(3, tag)
            e = abs(max(-b[0], b[3]) - hx) + abs(max(-b[1], b[4]) - hy) + abs(b[2] - z0) + abs(b[5] - z1)
            if e < err:
                best, err = tag, e
        if err > 0.5:
            raise SystemExit(f"{os.path.basename(PLUG_STEP)}: no solid looks like the {name} (off by {err:.2f} mm)")
        named[name] = best
    return named


def _to_socket(points_mm: np.ndarray) -> np.ndarray:
    """Cord frame [mm] -> socket frame [m], seated (a proper rotation: no re-winding needed)."""
    p = points_mm
    return np.column_stack([-p[:, 0], p[:, 1] + PLUG_DY, SEAT_Z - p[:, 2]]) * 1e-3


def plug_geometry(gmsh):
    """(collider meshes, drawn meshes, masses) of the plug, in the socket frame [m] / [kg]."""
    gmsh.clear()
    occ = gmsh.model.occ
    volumes = [t for d, t in occ.importShapes(PLUG_STEP) if d == 3]
    occ.synchronize()
    named = _classify(gmsh, volumes)

    # The collider: the shell's envelope, its upright edges rounded as the CAD's, its tip rolled ...
    (hx, hy), (z0, z1) = SHELL_HALF, SHELL_Z
    env = _box(occ, (-hx, -hy, z0), (hx, hy, z1))
    occ.synchronize()
    env = occ.fillet([env], _vertical_edges(gmsh, env, z0, z1), [SHELL_CORNER_R])[0][1]
    occ.synchronize()
    tip_edges = [abs(c) for _, s in gmsh.model.getBoundary([(3, env)], oriented=False)
                 for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)
                 if abs(gmsh.model.getBoundingBox(1, abs(c))[2] - z1) < 1e-3 and abs(gmsh.model.getBoundingBox(1, abs(c))[5] - z1) < 1e-3]
    env = occ.fillet([env], sorted(set(tip_edges)), [TIP_ROUND_R])[0][1]
    # ... less the tongue cavity, open at the tip.
    cav = _box(occ, (-CAVITY_HALF_X, CAVITY_Y[0], z0), (CAVITY_HALF_X, CAVITY_Y[1], z1 + 1.0))
    (plug,), _ = occ.cut([(3, env)], [(3, cav)])
    occ.synchronize()
    plug_vol = occ.getMass(*plug)
    colliders = _mesh_volumes(gmsh, {"plug": [plug[1]]}, MESH_SIZE_MM)

    # Drawn: the overmold, the contacts, a stub of cable.
    gmsh.clear()
    volumes = [t for d, t in occ.importShapes(PLUG_STEP) if d == 3]
    occ.synchronize()
    named = _classify(gmsh, volumes)
    contacts = []
    for t in volumes:
        b = gmsh.model.getBoundingBox(3, t)
        if b[2] > CONTACT_Z[0] - 0.5 and b[5] < CONTACT_Z[1] + 0.5 and b[5] - b[2] < 6:
            contacts.append(t)
    cable = None
    for t in volumes:
        b = gmsh.model.getBoundingBox(3, t)
        if b[5] - b[2] > 500:  # the cord itself
            cable = t
    (cable_stub,), _ = occ.intersect([(3, cable)], [(3, _box(occ, (-5, -5, OVERMOLD_BACK_Z - CABLE_KEEP_MM), (5, 5, OVERMOLD_BACK_Z + 0.3)))])
    occ.remove([(3, t) for t in volumes if t not in contacts + [named["overmold"], cable]], recursive=True)
    occ.synchronize()
    masses = {"plug": plug_vol * DENSITY["plug"] * 1e-6,
              "overmold": occ.getMass(3, named["overmold"]) * DENSITY["overmold"] * 1e-6,
              "cable": occ.getMass(*cable_stub) * DENSITY["cable"] * 1e-6}
    drawn = _mesh_volumes(gmsh, {"overmold": [named["overmold"]], "contacts": contacts, "cable": [cable_stub[1]]},
                          DRAW_MESH_SIZE_MM)
    to = lambda d: {n: (_to_socket(p), f) for n, (p, f) in d.items()}  # noqa: E731
    return to(colliders), to(drawn), masses


def socket_geometry(gmsh):
    """(collider meshes, drawn meshes, mass) of the adapter, in the socket frame [m] / [kg]."""
    gmsh.clear()
    occ = gmsh.model.occ
    vols = [(d, t) for d, t in occ.importShapes(SOCKET_STEP) if d == 3]
    if len(vols) != 1:
        raise SystemExit(f"{os.path.basename(SOCKET_STEP)}: expected one solid, found {len(vols)}")
    occ.synchronize()
    mass = occ.getMass(*vols[0]) * 2.0e-6  # mostly glass-filled nylon, some steel

    # The pocket: the shell's inside, from the floor up, flaring at the lip, and open above.
    (y0, y1), hx = POCKET_Y, POCKET_HALF_X
    pocket = _box(occ, (-hx, y0, POCKET_FLOOR_Z), (hx, y1, POCKET_TOP_Z))
    f = POCKET_FLARE
    lip = occ.addThruSections([_rect_wire(occ, hx, y0, y1, POCKET_TOP_Z),
                               _rect_wire(occ, hx + f, y0 - f, y1 + f, SHELL_TOP_Z)], makeSolid=True)
    top = _box(occ, (-hx - f, y0 - f, SHELL_TOP_Z), (hx + f, y1 + f, RECEPTACLE_CROP[2][1] + 1.0))
    (pocket_all,), _ = occ.fuse([(3, pocket)], [lip[0], (3, top)])
    tongue = _box(occ, *zip(*TONGUE_BOX))
    cut_tool, _ = occ.cut([pocket_all], [(3, tongue)], removeObject=False, removeTool=True)
    # The solid fill: the whole crop up to the underside of the housing's top plate (the D-series body
    # is hollow behind the shell, which left 0.7 mm walls and a 0.66 mm floor), then the housing's
    # opening up to the shell's top -- all of it outside the pocket.
    (cx0, cx1), (cy0, cy1), (cz0, _) = RECEPTACLE_CROP
    (ox0, ox1), (oy0, oy1) = (-OPENING_HALF_X, OPENING_HALF_X), OPENING_Y
    fill = occ.addCylinder(0, 0, cz0, 0, 0, PLATE_UNDERSIDE_Z - cz0, BODY_FILL_R)
    fill2 = _box(occ, (ox0 - 0.6, oy0 - 0.6, PLATE_UNDERSIDE_Z - 0.1), (ox1 + 0.6, oy1 + 0.6, SHELL_TOP_Z))  # and the undercut
    fill, _ = occ.fuse([(3, fill)], [(3, fill2)])
    fill, _ = occ.cut(fill, [pocket_all], removeObject=True, removeTool=True)

    adapter_copy = occ.copy(vols)
    rec = _crop(occ, adapter_copy, RECEPTACLE_CROP)
    rec, _ = occ.fuse(rec, fill)
    rec, _ = occ.fuse(rec[:1], rec[1:]) if len(rec) > 1 else (rec, None)
    rec, _ = occ.cut(rec, cut_tool)
    (tx0, tx1), _, (tz0, tz1) = TONGUE_BOX
    relief = [(3, _box(occ, (tx0, TONGUE_Y[0] - 0.1, tz0), (tx1, TONGUE_Y[0] + TONGUE_RELIEF, tz1 + 0.5))),
              (3, _box(occ, (tx0, TONGUE_Y[1] - TONGUE_RELIEF, tz0), (tx1, TONGUE_Y[1] + 0.1, tz1 + 0.5)))]
    rec, _ = occ.cut(rec, relief)
    occ.synchronize()
    draw_copy = occ.copy(vols)
    drawn = _crop(occ, draw_copy, ((-40, 40), (-40, 40), (DRAW_CROP_Z, 10)))
    occ.remove(vols, recursive=True)
    occ.synchronize()
    colliders = _mesh_volumes(gmsh, {"receptacle": [t for _, t in rec]}, MESH_SIZE_MM, FINE_REGION)
    coarse = _mesh_volumes(gmsh, {"adapter": [t for _, t in drawn]}, DRAW_MESH_SIZE_MM)
    to = lambda d: {n: (p * 1e-3, f) for n, (p, f) in d.items()}  # noqa: E731
    return to(colliders), to(coarse), mass


def _inertia_box(mass, dims):
    a, b, c = dims
    return (mass * (b * b + c * c) / 12, mass * (a * a + c * c) / 12, mass * (a * a + b * b) / 12)


def write_plug(colliders, drawn, masses, meta) -> None:
    from pxr import Gf, UsdPhysics

    stage, _ = _stage(PLUG_USD, "UsbPlug", meta)
    looks = {n: _material(stage, f"/UsbPlug/Looks/{n}", COLOURS[n]) for n in ("plug", "overmold", "cable", "contacts")}
    surface = _physics_material(stage, "/UsbPlug/Surfaces/plug", FRICTION["plug"])
    body = stage.DefinePrim("/UsbPlug/body", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(body)
    m = sum(masses.values())
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_api.CreateMassAttr(float(m))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*_inertia_box(m, (0.016, 0.008, 0.090))))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, PLUG_DY * 1e-3, 0.040))
    _mesh(stage, "/UsbPlug/body/plug", *colliders["plug"], looks["plug"], True, surface)
    for n, (p, f) in drawn.items():
        _mesh(stage, f"/UsbPlug/body/{n}", p, f, looks[n], False)
    stage.GetRootLayer().Save()


def write_socket(colliders, coarse, mass, meta) -> None:
    from pxr import UsdPhysics

    stage, _ = _stage(SOCKET_USD, "UsbSocket", meta)
    looks = {n: _material(stage, f"/UsbSocket/Looks/{n}", COLOURS[n]) for n in ("receptacle", "adapter")}
    body = stage.DefinePrim("/UsbSocket/socket", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(float(mass))
    _mesh(stage, "/UsbSocket/socket/receptacle", *colliders["receptacle"], looks["receptacle"], True,
          _physics_material(stage, "/UsbSocket/Surfaces/receptacle", FRICTION["receptacle"]))
    _mesh(stage, "/UsbSocket/socket/adapter", *coarse["adapter"], looks["adapter"], False)
    # Drawn only: hide the colliders under the full adapter.
    stage.GetPrimAtPath("/UsbSocket/socket/receptacle").GetAttribute("visibility").Set("invisible")
    stage.GetRootLayer().Save()


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    for path in (PLUG_STEP, SOCKET_STEP):
        if not os.path.isfile(path):
            raise SystemExit(f"missing {os.path.relpath(path)}: download it from McMaster-Carr (3-D STEP)")
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")  # the files are in inches
    try:
        p_col, p_draw, p_mass = plug_geometry(gmsh)
        s_col, s_coarse, s_mass = socket_geometry(gmsh)
    finally:
        gmsh.finalize()
    meta = {
        "plug_step": {"file": os.path.basename(PLUG_STEP), "sha256": _sha256(PLUG_STEP)},
        "socket_step": {"file": os.path.basename(SOCKET_STEP), "sha256": _sha256(SOCKET_STEP)},
        "seat_z_mm": SEAT_Z, "plug_dy_mm": PLUG_DY, "pocket_floor_z_mm": POCKET_FLOOR_Z,
        "pocket_top_z_mm": POCKET_TOP_Z, "shell_top_z_mm": SHELL_TOP_Z, "friction": FRICTION,
        "tongue_relief_mm": TONGUE_RELIEF,
    }
    write_plug(p_col, p_draw, p_mass, meta)
    write_socket(s_col, s_coarse, s_mass, meta)
    for name, (p, f) in {**p_col, **p_draw, **s_col, **s_coarse}.items():
        lo, hi = p.min(0) * 1e3, p.max(0) * 1e3
        print(f"  {name:10s} {len(f):7d} triangles  x {lo[0]:7.2f}..{hi[0]:6.2f}  y {lo[1]:7.2f}..{hi[1]:6.2f}  "
              f"z {lo[2]:7.2f}..{hi[2]:7.2f} mm")
    print(f"  plug {sum(p_mass.values()) * 1e3:.1f} g ({', '.join(f'{k} {v * 1e3:.1f}' for k, v in p_mass.items())}), "
          f"adapter {s_mass * 1e3:.1f} g")
    print(f"wrote {os.path.relpath(PLUG_USD)} and {os.path.relpath(SOCKET_USD)}")


if __name__ == "__main__":
    main()
