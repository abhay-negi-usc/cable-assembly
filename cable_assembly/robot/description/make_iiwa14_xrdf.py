"""Generate iiwa14/iiwa14.xrdf, the cuMotion description of the KUKA LBR iiwa 14 R820.

cuMotion loads a robot as URDF + XRDF. The URDF is the vendored kinematic chain (iiwa14/iiwa14.urdf);
the XRDF adds what cuMotion needs on top of it: the c-space, acceleration and jerk ceilings, the
tool frame, and the COLLISION SPHERES each link is approximated by.

THE SPHERES ARE FITTED, NOT HAND-PLACED. Each link's upstream collision mesh (lbr_description, the
same pinned commit the URDF came from) is filled with spheres by cuMotion's own generator, then
moved from the mesh's frame into the link frame by the URDF's <collision><origin>. The meshes are
only read here: the XRDF carries the spheres, so nothing at run time needs them.

    python -m cable_assembly.robot.description.make_iiwa14_xrdf          # writes the XRDF
    python -m cable_assembly.robot.description.make_iiwa14_xrdf --check  # coverage report only

Meshes come from sim/'s copy (sim/scripts/build_iiwa14.py fetches them) when it exists, otherwise
straight from upstream at PIN. Re-run after changing SPHERES and diff the result.

ARM ONLY. The gripper, camera and spacer are cell facts (configs/cells/), not the robot's, so
they are not in this file; they get attached to tool0 where the cell is known.
"""
import argparse
import os
import struct
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'iiwa14', 'iiwa14.xrdf')
URDF = os.path.join(HERE, 'iiwa14', 'iiwa14.urdf')
SIM_MESHES = os.path.join(HERE, '..', '..', '..', 'sim', 'src', 'cable_assembly_sim', 'assets',
                          'data', 'iiwa14', 'meshes', 'collision')

PIN = '4a5599ce053044e5daf4df0030c01ec0629ae7c5'     # same commit as iiwa14.urdf (see README.md)
RAW = (f'https://raw.githubusercontent.com/lbr-stack/lbr_fri_ros2_stack/{PIN}/'
       'lbr_description/meshes/iiwa14/collision/')

JOINTS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

# Each mesh's <collision><origin> in upstream's iiwa14 description (all rpy 0): the meshes are
# modelled in the BASE frame at q = 0, so this is minus the link frame's height there.
MESH_ORIGIN = {
    'link_0': (0.0, 0.0, 0.0),
    'link_1': (0.0, 0.0, -0.1475),
    'link_2': (0.0, 0.01, -0.36),
    'link_3': (0.0, 0.0, -0.588),
    'link_4': (0.0, -0.0105, -0.78),
    'link_5': (0.0, 0.0, -0.9875),
    'link_6': (0.0, 0.0707, -1.18),
    'link_7': (0.0, 0.0, -1.271),
}

# Spheres per link. cuMotion's generator picks spheres INSIDE the mesh, tangent to it, so edges and
# corners stick out between them; each link's spheres are then grown by the one offset that brings
# 99% of its surface to within COVER_TOL_M of them. More spheres = a tighter fit for less growth
# (--check prints both), at a GPU cost that scales with the total. link_0 never moves, so it only
# matters against the wrist and gets fewer.
SPHERES = {'link_0': 16, 'link_1': 32, 'link_2': 32, 'link_3': 32, 'link_4': 32, 'link_5': 32,
           'link_6': 32, 'link_7': 16}
COVER_TOL_M = 0.002

# Ceilings, not run speeds. Velocity comes from the URDF (KUKA's rated limits); the cell's speed
# caps and phase scales are applied when a trajectory is executed, as for every other move. These
# match the FRI driver's defaults (backends/iiwa_fri.py: stop_accel_rad_s2 4.0 over a 0.05 s ramp).
ACCEL_LIMIT = 4.0      # rad/s^2
JERK_LIMIT = 80.0      # rad/s^3

# A neutral pose clear of every limit, elbow up; RMPflow uses it as its posture attractor.
DEFAULT_Q_DEG = [0.0, 30.0, 0.0, -60.0, 0.0, 90.0, 0.0]

# Pairs whose spheres overlap by construction: --check's random sweep finds exactly the neighbours
# across each joint touching at every configuration (fraction 1.000) and every other pair touching
# only in real self-collisions, so only the neighbours are ignored.
SELF_IGNORE = {f'link_{i}': [f'link_{i + 1}'] for i in range(7)}


def read_stl(path):
    """Vertices (n, 3) and triangles (m, 3) of a binary STL, shared corners merged."""
    with open(path, 'rb') as fh:
        data = fh.read()
    n = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f'{path}: not a binary STL')
    rec = np.frombuffer(data, dtype=np.dtype([('normal', '<f4', 3), ('v', '<f4', (3, 3)),
                                              ('attr', '<u2')]), count=n, offset=84)
    corners = rec['v'].reshape(-1, 3).astype(np.float64)
    verts, idx = np.unique(np.round(corners, 7), axis=0, return_inverse=True)
    return verts, idx.reshape(-1, 3).astype(np.int32)


def mesh_path(link, cache):
    local = os.path.join(SIM_MESHES, link + '.stl')
    if os.path.exists(local):
        return local
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, link + '.stl')
    if not os.path.exists(dest):
        with urllib.request.urlopen(RAW + link + '.stl') as r:
            data = r.read()
        with open(dest, 'wb') as fh:
            fh.write(data)
    return dest


def fit(cache, rng):
    """{link: [(center in link frame, radius)]}, each link's mesh in the link frame, and the radius
    offset each link needed."""
    import cumotion
    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    spheres, meshes, offsets = {}, {}, {}
    for link, num in SPHERES.items():
        verts, tris = read_stl(mesh_path(link, cache))
        verts = verts + np.array(MESH_ORIGIN[link])
        gen = cumotion.create_collision_sphere_generator(list(verts), list(tris))
        tangent = [(np.asarray(s.center).ravel(), float(s.radius))
                   for s in gen.generate_spheres(num, 0.0)]
        _worst, p99 = coverage({link: tangent}, {link: (verts, tris)}, rng)[link]
        offsets[link] = max(0.0, 1e-3 * p99 - COVER_TOL_M)
        spheres[link] = [(c, r + offsets[link]) for c, r in tangent]
        meshes[link] = (verts, tris)
    return spheres, meshes, offsets


def coverage(spheres, meshes, rng):
    """Per link: how far the mesh surface pokes OUT of its spheres (max, 99th percentile), in mm.

    Surface points are the vertices plus random points on every triangle, so a long flat face
    with few vertices is still sampled."""
    report = {}
    for link, (verts, tris) in meshes.items():
        a, b, c = (verts[tris[:, k]] for k in range(3))
        u = rng.random((len(tris), 4, 2))
        flip = u.sum(-1) > 1
        u[flip] = 1 - u[flip]
        pts = (a[:, None] + u[..., :1] * (b - a)[:, None] + u[..., 1:] * (c - a)[:, None])
        pts = np.vstack([verts, pts.reshape(-1, 3)])
        C = np.array([s[0] for s in spheres[link]])
        R = np.array([s[1] for s in spheres[link]])
        out = np.min(np.linalg.norm(pts[:, None] - C[None], axis=-1) - R[None], axis=1)
        report[link] = (1e3 * out.max(), 1e3 * np.percentile(out, 99))
    return report


def render(spheres):
    lines = [
        '# GENERATED by cable_assembly/robot/description/make_iiwa14_xrdf.py -- edit that, not this.',
        '# cuMotion description of the KUKA LBR iiwa 14 R820, for use with iiwa14.urdf next to it.',
        f'# Collision spheres fitted to lbr_description\'s collision meshes @ {PIN[:8]}.',
        '# ARM ONLY: the tool (gripper, camera, spacer) is attached to tool0 from the cell config.',
        '',
        'format: xrdf',
        'format_version: 2.0',
        '',
        'default_joint_positions:',
    ]
    lines += [f'  {j}: {np.radians(d):.6f}' for j, d in zip(JOINTS, DEFAULT_Q_DEG)]
    lines += [
        '',
        'cspace:',
        '  joint_names: [' + ', '.join(f'"{j}"' for j in JOINTS) + ']',
        f'  acceleration_limits: [{", ".join([str(ACCEL_LIMIT)] * 7)}]',
        f'  jerk_limits: [{", ".join([str(JERK_LIMIT)] * 7)}]',
        '',
        'tool_frames: ["tool0"]',
        '',
        'modifiers:',
        '  - set_base_frame: "link_0"',
        '',
        'world_collision:',
        '  geometry: "iiwa14_collision_spheres"',
        '',
        'self_collision:',
        '  geometry: "iiwa14_collision_spheres"',
        '  ignore:',
    ]
    lines += [f'    {a}: [{", ".join(f"{b!r}".replace(chr(39), chr(34)) for b in bs)}]'
              for a, bs in SELF_IGNORE.items()]
    lines += ['', 'geometry:', '  iiwa14_collision_spheres:', '    spheres:']
    for link, ss in spheres.items():
        lines.append(f'      {link}:')
        for c, r in ss:
            lines.append(f'        - center: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]')
            lines.append(f'          radius: {r:.4f}')
    return '\n'.join(lines) + '\n'


def self_contact_sweep(xrdf_text, n=2000, seed=0):
    """Fraction of random in-limit configurations at which each link pair's spheres touch.

    A pair that touches at nearly every configuration overlaps by construction and belongs in
    SELF_IGNORE; a pair that touches only sometimes is a real self-collision and must stay."""
    import cumotion
    robot = cumotion.load_robot_from_memory(xrdf_text, open(URDF).read())
    kin = robot.kinematics()
    lo = np.array([kin.cspace_coord_limits(i).lower for i in range(7)])
    hi = np.array([kin.cspace_coord_limits(i).upper for i in range(7)])
    spheres = {}
    import yaml
    for link, ss in yaml.safe_load(xrdf_text)['geometry']['iiwa14_collision_spheres'][
            'spheres'].items():
        spheres[link] = (np.array([s['center'] for s in ss]), np.array([s['radius'] for s in ss]))
    links = list(spheres)
    hits = {}
    rng = np.random.default_rng(seed)
    for q in rng.uniform(lo, hi, (n, 7)):
        world = {}
        for link, (C, R) in spheres.items():
            T = kin.pose(q, link).matrix()
            world[link] = (C @ T[:3, :3].T + T[:3, 3], R)
        for i, a in enumerate(links):
            for b in links[i + 1:]:
                (Ca, Ra), (Cb, Rb) = world[a], world[b]
                d = np.linalg.norm(Ca[:, None] - Cb[None], axis=-1) - Ra[:, None] - Rb[None]
                if d.min() < 0:
                    hits[(a, b)] = hits.get((a, b), 0) + 1
    return {k: v / n for k, v in sorted(hits.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='report only; do not write the XRDF')
    ap.add_argument('--cache', default=os.path.expanduser('~/.cache/cable_assembly/iiwa14_meshes'),
                    help='where upstream meshes are downloaded when sim/ has no copy')
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    spheres, meshes, offsets = fit(args.cache, rng)
    print('link     spheres  grown by (mm)  mesh still outside: max / p99 (mm)')
    for link, (worst, p99) in coverage(spheres, meshes, rng).items():
        print(f'{link:8s} {len(spheres[link]):7d}  {1e3 * offsets[link]:13.1f}  '
              f'{worst:6.1f} / {p99:4.1f}')
    text = render(spheres)
    ignored = {(a, b) for a, bs in SELF_IGNORE.items() for b in bs}
    print('\nself-contact over random configurations (pair: fraction; * = ignored):')
    for (a, b), f in self_contact_sweep(text).items():
        print(f'  {a}-{b}: {f:.3f}{" *" if (a, b) in ignored else ""}')
    if not args.check:
        with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(text)
        print('\nwrote', os.path.relpath(OUT))


if __name__ == '__main__':
    main()
