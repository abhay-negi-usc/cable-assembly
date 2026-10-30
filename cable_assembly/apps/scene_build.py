"""Scene build -- a collision world for cuMotion from RGB-D frames (perception/scene.py), checked.
No robot needed: it reads frames already captured.

    python -m cable_assembly.apps.scene_build --show        # the latest splat scan, drawn in the live view
    python -m cable_assembly.apps.scene_build --rgbd data/experiments/splat_scan_<stamp>   # a scan of choice
    python -m cable_assembly.apps.scene_build --rgbd .../view_03.npz .../frame.npz         # frames of choice
                                                    # --cell: the frames' own, unless given

With no --rgbd, the frames are the newest splat scan's views, from data/experiments under the repo
root or under sim/ (where the app was started decides which). --rgbd takes run directories as well
as files and globs. The frames (capture_rgbd's / splat_scan's frame.npz) go into an OccupancyScene
over the cell's `scene:` box, in the order given (a later frame replaces what it sees). Written to
data/experiments/scene_build_<stamp>/scene.npz (OccupancyScene.load reads it back).

THE CHECK, with cuMotion and the iiwa description (robot/description/iiwa14): the scene as an SDF
obstacle plus the bench, and the arm at each frame's own joints must be clear of it -- a camera
cannot see the arm it is on through the arm, so a collision there means the robot was not left out
of the frame, or the hand-eye is off. Reported: at each frame's pose, how far the arm's moving links
are from the scene and from the bench -- link_2 out, as splat_scan's clearance check counts. The
base and link_1 stand beside the bench and only turn in place (65 and ~190 mm from the bench box at
any pose: all a single minimum over every link ever said). The gripper is not in the description
yet, so it is not checked.

--show sends the solid voxels to the live view (live_view_gl.py) as the overlay 'scene'.
"""

import os
import sys

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..perception.scene import OccupancyScene
from ._common import experiment_dir, frame_files

log = urlog.get('scene')


def main():
    parser = urconfig.arg_parser('Build a collision scene for cuMotion from RGB-D frames', 'bnc_assembly')
    parser.add_argument('--rgbd', nargs='*', default=[], metavar='NPZ',
                        help='frame.npz files, globs or run directories, in order (default: the latest splat scan)')
    parser.add_argument('--voxel-mm', type=float, default=None, help="voxel size (default: the cell's scene.voxel_mm, 10)")
    parser.add_argument('--show', action='store_true', help='send the solid voxels to the live view')
    parser.add_argument('--no-check', action='store_true', help='skip the cuMotion check')
    args = parser.parse_args()
    paths, what = frame_files(urconfig.from_args(args), args.rgbd)
    if not paths:
        sys.exit(f'scene_build: {what}.')
    if args.cell is None:                    # the cell the frames were captured on
        with np.load(paths[0]) as z:
            args.cell = str(z['cell']) if 'cell' in z.files and str(z['cell']) else None
    cfg = urconfig.from_args(args)
    urlog.setup()
    log.info('Cell: %s. Frames: %s (%d).', cfg.get('cell'), what, len(paths))
    scene = OccupancyScene.for_cell(cfg, None if args.voxel_mm is None else args.voxel_mm / 1000.0)
    bench_z = float(cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0
    frames = []
    for path in paths:
        with np.load(path) as z:
            frame = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
        cleared, filled = scene.add(frame)
        frames.append((path, frame))
        log.info('%s: %d voxels occupied, %d cleared.', path, filled, cleared)
    solid = scene.solid()
    log.info('Scene: %d frames; %d voxels occupied, %d solid (with what lies below), of %s at %.0f mm, '
             'over (%.2f, %.2f, %.3f)-(%.2f, %.2f, %.3f) m.', scene.frames, int(scene.occ.sum()), int(solid.sum()),
             'x'.join(map(str, scene.shape)), 1000 * scene.voxel, *scene.lo, *scene.hi)
    out = experiment_dir(cfg, 'scene_build')
    scene.save(os.path.join(out, 'scene.npz'))
    log.info('Wrote %s', os.path.join(out, 'scene.npz'))
    ok = True
    if not args.no_check:
        ok = check(scene, frames, bench_z)
    if args.show:
        show(cfg, scene)
    sys.exit(0 if ok else 1)


def check(scene, frames, bench_z):
    """The arm at each frame's joints, against the scene, in cuMotion. False on a collision."""
    try:
        import cumotion
    except ImportError:
        log.warning('No cuMotion here (setup-env.sh step 5): the scene is not checked.')
        return True
    from ..robot.kinematics import iiwa14
    desc = os.path.join(os.path.dirname(iiwa14.__file__), '..', 'description', 'iiwa14')
    xrdf, urdf = os.path.join(desc, 'iiwa14.xrdf'), os.path.join(desc, 'iiwa14.urdf')
    if not os.path.isfile(xrdf):
        log.warning('No %s (robot/description/make_iiwa14_xrdf.py writes it): the scene is not checked.', xrdf)
        return True
    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    robot = cumotion.load_robot_from_file(xrdf, urdf)
    world, handles = scene.to_cumotion(bench_z=bench_z)
    insp = cumotion.create_robot_world_inspector(robot, world.add_world_view())
    moving = [i for i in range(insp.num_world_collision_spheres())
              if insp.world_collision_sphere_frame_name(i) not in ('link_0', 'link_1')]
    ok = True
    for path, frame in frames:
        if 'joints_rad' not in frame or str(frame.get('robot')) != 'iiwa14':
            continue
        q = np.asarray(frame['joints_rad'], dtype=float)
        d_scene = min(insp.distance_to_obstacle(handles['sdf'], i, q) for i in moving)
        d_bench = (min(insp.distance_to_obstacle(handles['bench'], i, q) for i in moving)
                   if handles['bench'] is not None else np.inf)
        hit = min(d_scene, d_bench) < 0.0
        ok &= not hit
        log.info('  the arm where %s was taken: %s (%.0f mm from the scene, %.0f mm from the bench)',
                 os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path),
                 'IN COLLISION' if hit else 'clear', 1000 * d_scene, 1000 * d_bench)
    return ok


def show(cfg, scene, max_points=60000):
    """The solid voxels to the live view, as one overlay of points."""
    from ..robot import live_view
    c = cfg.get_path('live_view', {}) or {}
    idx = np.argwhere(scene.solid())
    if len(idx) > max_points:
        idx = idx[np.linspace(0, len(idx) - 1, max_points).astype(int)]
    tx = live_view.FrameSender('iiwa14', np.eye(4), None, str(c.get('host', '127.0.0.1')),
                               int(c.get('frames_port', live_view.FRAMES_PORT)))
    tx.send_overlay('scene', [{'type': 'points', 'points': scene.centres(idx).round(4).tolist(),
                               'color': [1.0, 0.25, 0.25], 'radius': 0.4 * scene.voxel}],
                    [f'scene: {int(scene.solid().sum())} solid voxels at {1000 * scene.voxel:.0f} mm'])
    import time
    time.sleep(1.0)                                     # let the sender's thread deliver it
    tx.close()
    log.info('Sent %d voxels to the live view.', len(idx))


if __name__ == '__main__':
    main()
