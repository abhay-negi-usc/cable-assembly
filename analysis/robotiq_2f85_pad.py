"""Robotiq 2F-85 fingertip pad, isolated from the gripper's assembly STEP for 3D printing.

The assembly (sim/src/cable_assembly_sim/assets/data/robotiq_2f85_v3/2-Finger_85_Pad_203.STEP) holds the pad as
two mirrored solids (one per finger, solids 14 and 15). They are the same part: a 6 mm plate, 38.3 x 23.5 mm,
two distal corners rounded R5, with three holes through it on its centre line -- two 2.5 mm (M3 tap drill; the
inner finger's two counterbored M3 clearance holes line up with them) and one 3 mm (a dowel pin) between them.
Since the holes go through, they show up as recesses in the gripping face.

Printed holes come out undersized (an M3 bolt would not thread into the 2.5 mm ones as drawn), so every hole is
opened up by --oversize mm on its diameter (default 0.2: 2.5 -> 2.7 mm, 3.0 -> 3.2 mm) in both outputs.

Writes, in the PRINT frame (mm): the gripping face at z = 0 lying on the bed, the body in +z (back face at
z = 6), y along the pad's length toward its rounded (distal) end, origin at the centre of its outline:

  * <out>/2f85_pad.step / .stl        -- the pad as Robotiq drew it (through holes), holes oversized;
  * <out>/2f85_pad_flat.step / .stl   -- the holes stop --skin mm short of the gripping face (blind from the back),
                                         so the gripping face is one unbroken flat. The screws must not reach
                                         past 6 - skin mm into the pad.

Print two of either one (the left and right pads are identical). Gripping face down gives the best surface and
the blind holes need no supports.

  python -I analysis/robotiq_2f85_pad.py [--skin 1.5] [--oversize 0.2] [--out DIR]
(needs OCP, e.g. the rag6dpose env.)
"""
import argparse
import os

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STEP = os.path.join(REPO, 'sim/src/cable_assembly_sim/assets/data/robotiq_2f85_v3/2-Finger_85_Pad_203.STEP')
OUT = os.path.join(REPO, 'cable_assembly/robot/description/tool')
PAD_SOLID = 14          # the -x finger's pad; 15 is its mirror image
THICK = 6.0             # the pad's thickness (mm)


def solids(shape):
    e, out = TopExp_Explorer(shape, TopAbs_SOLID), []
    while e.More():
        out.append(TopoDS.Solid_s(e.Current()))
        e.Next()
    return out


def faces(shape):
    e = TopExp_Explorer(shape, TopAbs_FACE)
    while e.More():
        f = TopoDS.Face_s(e.Current())
        yield f, BRepAdaptor_Surface(f)
        e.Next()


def area(f):
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(f, g)
    return g.Mass()


def volume(s):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    return g.Mass()


def bbox(s):
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(s, b, False, False)
    return b.Get()


def print_frame(pad):
    """gp_Trsf taking the assembly frame to the print frame (see the module docstring)."""
    # The two big parallel planes are the gripping face and the back; the gripping face is the one nearer the
    # gripper's centre plane x = 0 (the pads face each other across it).
    big = sorted(((area(f), f, a) for f, a in faces(pad) if a.GetType() == GeomAbs_Plane), key=lambda t: -t[0])[:2]
    loc = [a.Plane().Location() for _, _, a in big]
    grip = min(range(2), key=lambda i: abs(loc[i].X()))
    n = big[grip][2].Plane().Axis().Direction()
    into = gp_Vec(loc[grip], loc[1 - grip])                  # gripping face -> back
    z = gp_Dir(n.X(), n.Y(), n.Z()) if gp_Vec(n).Dot(into) > 0 else gp_Dir(-n.X(), -n.Y(), -n.Z())
    # Length direction: the flat end face (normal ~ +-y), pointed toward the rounded end (larger y here).
    end = max(((area(f), a) for f, a in faces(pad) if a.GetType() == GeomAbs_Plane
               and abs(a.Plane().Axis().Direction().Y()) > 0.9), key=lambda t: t[0])[1]
    d = end.Plane().Axis().Direction()
    y = gp_Dir(d.X(), d.Y(), d.Z()) if d.Y() > 0 else gp_Dir(-d.X(), -d.Y(), -d.Z())
    # Origin: the gripping face's centre (its bbox centre, projected onto the face).
    b = bbox(big[grip][1])
    c = gp_Pnt(*[(b[i] + b[i + 3]) / 2 for i in range(3)])
    c = c.Translated(gp_Vec(z).Multiplied(-gp_Vec(loc[grip], c).Dot(gp_Vec(z))))
    t = gp_Trsf()
    t.SetTransformation(gp_Ax3(c, z, y.Crossed(z)))          # x = y cross z, right-handed
    return t


def holes(pad):
    """(x, y, r) of each hole in the print frame (axes along z), merged across the half-cylinder faces."""
    out = []
    for f, a in faces(pad):
        if a.GetType() != GeomAbs_Cylinder:
            continue
        cyl = a.Cylinder()
        p, r = cyl.Location(), cyl.Radius()
        if r > 2.5:                                           # the R5 corner rounds
            continue
        if any(abs(p.X() - h[0]) < 0.05 and abs(p.Y() - h[1]) < 0.05 for h in out):
            continue
        out.append((p.X(), p.Y(), r))
    return sorted(out, key=lambda h: h[1])


def enlarge(pad, hs, oversize):
    """The pad with each hole re-cut through at a diameter `oversize` mm larger; returns (pad, new holes)."""
    s, new = pad, [(x, y, r + oversize / 2) for x, y, r in hs]
    for x, y, r in new:
        c = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1)), r, THICK + 2.0).Shape()
        s = BRepAlgoAPI_Cut(s, c).Shape()
    u = ShapeUpgrade_UnifySameDomain(s, True, True, True)
    u.Build()
    return solids(u.Shape())[0], new


def plug(pad, hs, skin):
    """The pad with each hole filled from the gripping face (z = 0) to z = skin."""
    s = pad
    for x, y, r in hs:
        c = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, 0.0), gp_Dir(0, 0, 1)), r, skin).Shape()
        s = BRepAlgoAPI_Fuse(s, c).Shape()
    u = ShapeUpgrade_UnifySameDomain(s, True, True, True)
    u.Build()
    return solids(u.Shape())[0]


def write(shape, stem):
    assert BRepCheck_Analyzer(shape).IsValid(), stem
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    assert w.Write(stem + '.step') == IFSelect_RetDone
    BRepMesh_IncrementalMesh(shape, 0.01, False, 0.1, True)
    s = StlAPI_Writer()
    s.ASCIIMode = False
    assert s.Write(shape, stem + '.stl')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', default=STEP)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--skin', type=float, default=1.5, help='solid skin left over the holes on the flat pad (mm)')
    ap.add_argument('--oversize', type=float, default=0.2,
                    help='added to every hole diameter, in both outputs (mm); 0 keeps the holes as drawn')
    a = ap.parse_args()

    r = STEPControl_Reader()
    assert r.ReadFile(a.step) == IFSelect_RetDone
    r.TransferRoots()
    raw = solids(r.OneShape())[PAD_SOLID]
    pad = TopoDS.Solid_s(BRepBuilderAPI_Transform(raw, print_frame(raw), True).Shape())
    b = bbox(pad)                                             # centre it in x, y (the face was tilted)
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, 0.0))
    pad = TopoDS.Solid_s(BRepBuilderAPI_Transform(pad, t, True).Shape())

    b = bbox(pad)
    assert abs(b[2]) < 1e-3 and abs(b[5] - THICK) < 1e-3, b
    drawn = holes(pad)
    hs = drawn
    if a.oversize:
        pad, hs = enlarge(pad, drawn, a.oversize)
        assert [round(2 * h[2], 3) for h in holes(pad)] == [round(2 * h[2], 3) for h in hs], holes(pad)
    flat = plug(pad, hs, a.skin)

    os.makedirs(a.out, exist_ok=True)
    write(pad, os.path.join(a.out, '2f85_pad'))
    write(flat, os.path.join(a.out, '2f85_pad_flat'))
    print('pad bbox (mm): x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f' % (b[0], b[3], b[1], b[4], b[2], b[5]))
    for (x, y, rr), d in zip(hs, drawn):
        print('  hole d %.2f (drawn %.2f) at (x %.2f, y %.2f)' % (2 * rr, 2 * d[2], x, y))
    print('volume: through holes %.1f mm^3, flat %.1f mm^3 (skin %.1f mm, holes %.1f mm deep from the back)'
          % (volume(pad), volume(flat), a.skin, THICK - a.skin))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
