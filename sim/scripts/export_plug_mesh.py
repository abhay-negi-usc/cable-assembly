"""Export the BNC plug's outside (scripts/build_bnc.py's bnc_plug.usd) as a mesh the task's cable
model fits to camera points: cable_assembly/robot/description/bnc/bnc_plug.npz.

    cd sim && ./run.sh python scripts/export_plug_mesh.py        # after scripts/build_bnc.py

THE PARTS a camera can see: the boot (the strain relief), the collar, the disc and the seat ring.
The insulator, contact fingers and pin are inside the shell, and the cable stub is the cable's (the
task models it as a deformable rod).

THE FRAME is the task's connector frame: the origin on the axis where the cable leaves the boot
(the junction SAM3 marks), x along the axis toward the plug's mating face, z as the plug USD's x.
The USD has the plug facing down its z (mating face low): x_conn = -z_usd, y_conn = y_usd,
z_conn = x_usd, origin at the boot's top.

Gitignored with the rest of McMaster-Carr's geometry: theirs to distribute, not ours.
"""

import os

import numpy as np
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
PLUG_USD = os.path.join(HERE, "..", "src", "cable_assembly_sim", "assets", "data", "bnc", "bnc_plug.usd")
OUT = os.path.join(HERE, "..", "..", "cable_assembly", "robot", "description", "bnc", "bnc_plug.npz")
OUTSIDE = ("boot", "collar", "disc", "seat")


def main():
    stage = Usd.Stage.Open(PLUG_USD)
    now = Usd.TimeCode.Default()
    points, faces, part, offset = [], [], [], 0
    boot_top = None
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() not in OUTSIDE:
            continue
        mesh = UsdGeom.Mesh(prim)
        X = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(now)).T
        p = np.array(mesh.GetPointsAttr().Get(), dtype=float) @ X[:3, :3].T + X[:3, 3]
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f"{prim.GetPath()}: not triangles")
        points.append(p)
        faces.append(idx.reshape(-1, 3) + offset)
        part += [prim.GetName()] * len(p)
        offset += len(p)
        if prim.GetName() == "boot":
            boot_top = float(p[:, 2].max())
    if boot_top is None:
        raise SystemExit(f"no boot in {PLUG_USD}: run scripts/build_bnc.py first")
    P = np.vstack(points)
    P = np.stack([-(P[:, 2] - boot_top), P[:, 1], P[:, 0]], axis=1)       # into the connector frame
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, points=P, faces=np.vstack(faces), part=np.array(part),
                        source=os.path.relpath(PLUG_USD, os.path.dirname(OUT)))
    r = np.hypot(P[:, 1], P[:, 2])
    print(f"wrote {os.path.relpath(OUT)}: {len(P)} points, {sum(len(f) for f in faces)} triangles; "
          f"x {P[:, 0].min() * 1000:.1f}..{P[:, 0].max() * 1000:.1f} mm, radius up to {r.max() * 1000:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
