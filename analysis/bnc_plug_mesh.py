"""The BNC plug's mesh for perception/slot_roll.py, from a STEP of McMaster 6641T31 -- the user's two-J-slot version
(2026-11-06: "i made a step file with the j slot on both sides - this connector matches the real BNC"):
~/Downloads/6641T31_Coaxial Cord_two_J_slots.STEP.

The solids at z >= 105 mm (the cord's +z connector end), tessellated by RAG-6DPose's own prepare_models.build_mesh
-- exactly as it makes obj_000003 from McMaster's one-slot STEP -- so the mesh is in obj_000003's frame (mm, the
cord along z, the mating face at 152.4) and the plug belief, the camera geometry and P6.MATING_Z apply to it as
they are. McMaster's CAD is not ours to distribute: the output goes to data/ (git-ignored).

Needs OCP + trimesh -- the rag6dpose env (the cable env has neither):

    ~/miniconda3/envs/rag6dpose/bin/python analysis/bnc_plug_mesh.py \\
        ~/Downloads/"6641T31_Coaxial Cord_two_J_slots.STEP" data/cad/bnc_plug_two_j_slots.ply
"""

import argparse
import os
import sys

RAG = os.path.expanduser('~/abhay_ws/RAG-6DPose-code-abhay')
Z_MIN_MM = 105.0             # prepare_models' obj_000003 filter: (2, 105.0, inf) -- solids wholly above z 105 mm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('step')
    ap.add_argument('out', nargs='?', default='data/cad/bnc_plug_two_j_slots.ply')
    ap.add_argument('--rag', default=RAG, help='the RAG-6DPose checkout (its pose_pipeline.prepare_models)')
    a = ap.parse_args(argv)
    sys.path.insert(0, a.rag)
    from pose_pipeline.prepare_models import build_mesh   # noqa: E402 -- OCP + trimesh, the rag6dpose env
    mesh = build_mesh(a.step, (2, Z_MIN_MM, float('inf')))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mesh.export(a.out)
    lo, hi = mesh.bounds
    print(f'{a.out}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, z {lo[2]:.2f}..{hi[2]:.2f} mm, '
          f'r max {max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1])):.2f} mm, watertight {mesh.is_watertight}')


if __name__ == '__main__':
    main()
