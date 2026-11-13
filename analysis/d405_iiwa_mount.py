"""The iiwa's D405 flange mount (cable_assembly/robot/description/tool/d405_iiwa_mount.step, the user's,
2026-11-12): its mesh for the planner and the sim, and the faces and holes the D405 is placed by.

THE MOUNT'S FRAME (its STEP's, millimetres) is the CAD frame tool_geometry.py uses: z along the flange's axis,
z = 0 the face that sits on the flange (a flat 75 mm disc, a 32 mm bore, eight 6.6 mm holes on the 50 mm
circle), the Robotiq C-062 coupling on the other side -- its 63 mm x 3.0 mm recess over the mount's 62.9 mm x
2.5 mm boss, its rim on the z = 10 shoulder. The bracket runs out along +x and carries the D405:

  * its BOTTOM (the 1/4-20 tripod thread) on the inner face of the upright tab at the end -- the 1/4-20's
    7 mm clearance hole (11 mm counterbore on the outside) through it, the screw CENTRED in that hole;
  * its BACK on the bracket's plate, 15 deg off the flange plane -- the two M3's 3.35 mm clearance holes
    through it, 20 mm apart, the camera's two back threads CENTRED on them.

The two faces are square to each other, so the camera's pose follows from them: its optical axis along the
plate's normal (out of the plate), its image's down along the tab's (into the tab), and its tripod screw where
the 1/4-20 hole's axis meets the tab's face. The back holes check it: they land 21.0 mm up the camera's 42 mm
(mid-height, symmetric about the screw) and the screw 8.36 mm in front of its back (Intel: 23 mm deep, the
screw 14.65 mm behind the front), within a few hundredths.

    cd sim && ./run.sh --extra cad python ../analysis/d405_iiwa_mount.py         (gmsh)

prints the numbers tool_geometry.py carries and writes d405_iiwa_mount.stl (metres, the CAD frame); then
python analysis/d405_tool_mesh.py <Intel's d405.stl> places the camera's mesh on it.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(HERE, '..', 'cable_assembly', 'robot', 'description', 'tool')
STEP = os.path.join(TOOL_DIR, 'd405_iiwa_mount.step')
STL = os.path.join(TOOL_DIR, 'd405_iiwa_mount.stl')


def write_stl(path, points, tris, title):
    v = np.asarray(points, dtype=np.float64)[np.asarray(tris)]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    rec = np.zeros(len(v), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = n, v
    with open(path, 'wb') as f:
        f.write(title.encode()[:80].ljust(80, b' '))
        f.write(np.uint32(len(v)).tobytes())
        f.write(rec.tobytes())


def faces(gmsh):
    """[(tag, kind, nodes (N, 3) mm, area mm2)] of every surface, meshed."""
    out = []
    for _, s in gmsh.model.getEntities(2):
        _, xyz, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=True)
        out.append((s, gmsh.model.getType(2, s), xyz.reshape(-1, 3), gmsh.model.occ.getMass(2, s)))
    return out


def plane(P):
    c = P.mean(axis=0)
    n = np.linalg.svd(P - c)[2][2]
    return n / np.linalg.norm(n), c


def cylinder(P, axis):
    """(a point on the axis, radius) of a cylinder's nodes whose axis is `axis`."""
    a = axis / np.linalg.norm(axis)
    e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    x, y = P @ e1, P @ e2
    cx, cy, k = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)], x * x + y * y, rcond=None)[0]
    centre = cx * e1 + cy * e2 + (P @ a).mean() * a
    return centre, float(np.sqrt(k + cx * cx + cy * cy))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--step', default=STEP)
    ap.add_argument('--out', default=STL)
    ap.add_argument('--size_mm', type=float, default=2.0, help='the mesh size (the planner uses 1 cm spheres)')
    a = ap.parse_args(argv)
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('Mesh.MeshSizeMax', a.size_mm)
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.2)
    gmsh.model.occ.importShapes(a.step)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    F = faces(gmsh)

    # ---- the mesh, metres
    nodes, xyz, _ = gmsh.model.mesh.getNodes()
    idx = {int(n): i for i, n in enumerate(nodes)}
    tris_by_face = []
    for _, s in gmsh.model.getEntities(2):
        for typ, conn in zip(*gmsh.model.mesh.getElements(2, s)[::2]):
            if typ == 2:
                tris_by_face.append(np.vectorize(idx.get)(np.asarray(conn, int)).reshape(-1, 3))
    tris = np.concatenate(tris_by_face)
    P = xyz.reshape(-1, 3)
    used = np.unique(tris)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, tris = P[used], remap[tris]
    # outward winding: every triangle turned to face away from the solid (a point just off it, outside)
    c = P[tris].mean(axis=1)
    n = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    # one winding throughout (gmsh orients each face on its own): flip every face whose triangles point into the
    # solid, judged by a ray from each face's middle triangle along its normal crossing the mesh an odd number
    # of times -- then the signed volume is the solid's
    ftag = np.concatenate([np.full(len(t), k) for k, t in enumerate(tris_by_face)])
    def crossings(o, d):
        v0, v1, v2 = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        e1, e2 = v1 - v0, v2 - v0
        h = np.cross(d, e2)
        det = (e1 * h).sum(1)
        ok = np.abs(det) > 1e-12
        f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = o - v0
        u = f * (sv * h).sum(1)
        qv = np.cross(sv, e1)
        v = f * (qv @ d) if qv.ndim == 1 else f * (qv * d).sum(1)
        t = f * (qv * e2).sum(1)
        return int(np.sum(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)))
    flipped = 0
    for k in range(len(tris_by_face)):
        sel = np.where(ftag == k)[0]
        j = sel[np.argmax(np.linalg.norm(np.cross(P[tris[sel, 1]] - P[tris[sel, 0]], P[tris[sel, 2]] - P[tris[sel, 0]]), axis=1))]
        o = c[j] + 1e-3 * n[j]
        d = n[j] + np.array([1e-4, 2e-4, 3e-4])
        if crossings(o, d / np.linalg.norm(d)) % 2 == 1:
            tris[sel] = tris[sel][:, ::-1]
            flipped += 1
    v0, v1, v2 = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    vol = float(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6.0)
    print(f'{flipped} of {len(tris_by_face)} faces turned outward; signed volume {vol:.0f} mm3 '
          f'(the solid: {gmsh.model.occ.getMass(3, 1):.0f})')
    gmsh.finalize()
    write_stl(a.out, P * 1e-3, tris, 'D405 iiwa flange mount (d405_iiwa_mount.step), CAD frame, m')
    print(f'{os.path.relpath(a.out)}: {len(tris)} triangles, bbox '
          f'{np.round(P.min(0), 2)} .. {np.round(P.max(0), 2)} mm')

    # ---- the faces the D405 sits on
    def biggest(kind, test):
        cand = [(f[3], f) for f in F if f[1] == kind and test(f)]
        return max(cand, key=lambda t: t[0])[1]

    # the tab's inner face (the camera's bottom): the large plane facing ~(0.966, 0, 0.259) with the camera on
    # its -x side; the plate's top (its back): the large plane ~(-0.259, 0, 0.966) along the bracket
    tilted = [f for f in F if f[1] == 'Plane' and f[3] > 100]
    tab = [f for f in tilted if abs(plane(f[2])[0] @ [0.966, 0.0, 0.259]) > 0.99]
    back = [f for f in tilted if abs(plane(f[2])[0] @ [-0.259, 0.0, 0.966]) > 0.99]
    n_t = np.array([np.cos(np.radians(15.0)), 0.0, np.sin(np.radians(15.0))])
    n_b = np.array([-n_t[2], 0.0, n_t[0]])
    tab_d = sorted({round(float(plane(f[2])[1] @ n_t), 3) for f in tab})
    back_d = sorted({round(float(plane(f[2])[1] @ n_b), 3) for f in back})
    print(f'tab faces along {np.round(n_t, 4)}: {tab_d} mm; plate faces along {np.round(n_b, 4)}: {back_d} mm')
    # the 1/4-20's clearance hole (7 mm) and the M3's (3.35 mm): cylinders along n_t and n_b
    holes = {}
    for s, kind, Pn, area in F:
        if kind != 'Cylinder':
            continue
        for name, ax, r_want in (('quarter20', n_t, 3.5), ('m3', n_b, 1.675)):
            ctr, r = cylinder(Pn, ax)
            res = np.std(np.linalg.norm((Pn - ctr) - np.outer((Pn - ctr) @ ax, ax), axis=1) - r)
            if abs(r - r_want) < 0.05 and res < 0.05:
                holes.setdefault(name, []).append(ctr)
    q = holes['quarter20'][0]
    # the faces either side of each hole: the seat is the tab's face on the camera's side of the 1/4-20 (its
    # counterbore on the other), the back the plate's face on the camera's side of the M3's
    d_seat = max(d for d in tab_d if d < q @ n_t)
    d_back = min(d for d in back_d if d > holes['m3'][0] @ n_b)
    screw = q - ((q @ n_t) - d_seat) * n_t
    m3 = sorted([h - ((h @ n_b) - d_back) * n_b for h in holes['m3']], key=lambda p: p[1])
    m3 = [p for i, p in enumerate(m3) if i == 0 or np.linalg.norm(p - m3[i - 1]) > 1.0]
    print(f'seat (tab inner face): n.p = {d_seat:.3f} mm; back (plate top): n.p = {d_back:.3f} mm '
          f'(square: {n_t @ n_b:.1e})')
    print(f'1/4-20 axis on the seat: {np.round(screw, 3)} mm, {screw @ n_b - d_back:.3f} mm in front of the back '
          f'(D405: 8.35)')
    for p in m3:
        print(f'M3 hole on the back: {np.round(p, 3)} mm, {d_seat - p @ n_t:.3f} mm up from the seat, '
              f'{p[1] - screw[1]:+.3f} mm across from the screw')
    return 0


if __name__ == '__main__':
    sys.exit(main())
