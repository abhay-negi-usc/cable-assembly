"""Capture RGB-D -- one wrist-camera colour + depth frame and the pose it was taken from, for the
live view's Gaussian layer. IT NEVER MOVES THE ARM.

    python -m cable_assembly.apps.capture_rgbd --cell iiwa14_bench                 # the iiwa bench
    python -m cable_assembly.apps.capture_rgbd --cell iiwa14_bench --frames 20     # median of 20
    python -m cable_assembly.apps.capture_rgbd --cell iiwa14_bench --joints-deg -27 50 66 -96 120 -57 -123
                                                                   # no arm: label the frame with these

THE SETUP. Leave the arm where the camera sees the bench -- where a task ended, or jogged there on
the pendant -- with the FRI app running as for any task. This connects, reads the joints (the FRI
session holds the arm where it is; nothing here commands a move), takes the frames, checks the
joints again and disconnects. Talks to the ARM and the CAMERA only: no gripper, no Robot.

THE POSE is the camera's at capture: tool0 from the measured joints (forward kinematics) times the
cell's hand_eye (tool0 -> color optical frame), as robot.camera() gives it during a task. If the
joints change during the capture the frame is refused: it would mix two viewpoints.

THE DEPTH. camera.enable_depth is forced on, so the depth is aligned to the colour image and in
metres (perception/camera.py), from the D415's 848x480 depth mode: at 1280x720 it cannot see the
bench from the pick view (0.35 m; see camera.depth_width). Over --frames frames the arm is still, so they differ only by the
D415's noise: each pixel keeps the median of the frames that saw something there.

THE OUTPUT, data/experiments/capture_rgbd_<stamp>/:
    frame.npz   rgb (H,W,3 uint8), depth_m (H,W float32, 0 = no reading), K (3x3),
                T_base_cam, T_base_tool0 and T_tool0_fingertip (4x4, m), joints_rad, robot, cell, stamp
    color.png, depth.png (a false-colour preview, near = warm)
The sim side reads frame.npz without cable_assembly (sim/src/cable_assembly_sim/rgbd.py):
    cd sim && ./run.sh python scripts/live_view_gl.py --rgbd ../data/experiments/capture_rgbd_<stamp>/frame.npz
"""

import os
import sys
import time
import warnings

import cv2
import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..transforms import from_cfg
from ._common import experiment_dir

log = urlog.get('capture-rgbd')

# A joint that moved more than this between the first and the last frame spoils the capture.
_STILL_RAD = np.radians(0.05)


def main():
    parser = urconfig.arg_parser('Capture one RGB-D frame and the camera pose it was taken from',
                                 'bnc_assembly')
    parser.add_argument('--frames', type=int, default=10,
                        help='frames whose per-pixel median depth is kept (default 10)')
    parser.add_argument('--joints-deg', type=float, nargs='+', default=None, metavar='Q',
                        help='do not connect to the arm: label the frame with these joints (a test of '
                             'the camera side; the pose is only as right as the joints given)')
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    urlog.setup()
    if args.dry_run:
        log.error('A dry run has no camera: nothing to capture.')
        sys.exit(1)
    cfg.set_path('camera.enable_depth', True)

    from ..perception import RealSenseCamera
    from ..robot import kinematics, make_arm
    model = kinematics.get(cfg.get_path('robot.type'))
    T_tool0_cam = from_cfg(cfg.section('hand_eye'))
    # Where the fingertips are, so the viewer can leave the gripper out of the picture.
    T_tool0_tip = from_cfg(cfg.section('fingertip_grasp')) if cfg.get('fingertip_grasp') else None

    arm = None
    if args.joints_deg is None:
        arm = make_arm(cfg)
        read_q = lambda: np.asarray(arm.q(), dtype=float)       # noqa: E731
    else:
        if len(args.joints_deg) != model.DOF:
            parser.error(f'--joints-deg needs {model.DOF} values for the {model.NAME}')
        q_given = np.radians(args.joints_deg)
        read_q = lambda: q_given                                # noqa: E731
        log.warning('NOT CONNECTED TO THE ARM: the frame is labelled with the joints given.')

    camera = None
    try:
        camera = RealSenseCamera(cfg)
        ok = capture(cfg, camera, model, T_tool0_cam, T_tool0_tip, read_q, max(1, args.frames))
    except KeyboardInterrupt:
        log.warning('Interrupted.')
        ok = False
    finally:
        if camera is not None:
            camera.close()
        if arm is not None:
            arm.disconnect()
    sys.exit(0 if ok else 1)


def capture(cfg, camera, model, T_tool0_cam, T_tool0_tip, read_q, n_frames):
    """Take the frames, check the arm stayed still, write frame.npz and the previews."""
    got = still_frames(camera, read_q, n_frames)
    if got is None:
        log.error('The arm moved during the capture: the frames are from different viewpoints. '
                  'Nothing written.')
        return False
    q0, frames = got
    if frames[-1].depth is None:
        log.error('The camera returned no depth (is this a depth-capable RealSense?).')
        return False
    out = experiment_dir(cfg, 'capture_rgbd')
    path = os.path.join(out, 'frame.npz')
    depth, T_base_cam = save_frame(path, frames, q0, model, T_tool0_cam, T_tool0_tip, cfg)
    cv2.imwrite(os.path.join(out, 'color.png'), frames[-1].color)
    cv2.imwrite(os.path.join(out, 'depth.png'), depth_preview(depth))

    seen = depth > 0.0
    log.info('Wrote %s: %dx%d, depth on %.0f%% of the pixels (%.2f-%.2f m, median of %d frames). '
             'Camera at (%.3f, %.3f, %.3f) m in the base frame, looking %s.',
             path, depth.shape[1], depth.shape[0], 100.0 * seen.mean(),
             np.percentile(depth[seen], 2) if seen.any() else 0.0,
             np.percentile(depth[seen], 98) if seen.any() else 0.0, n_frames,
             *T_base_cam[:3, 3], 'down' if T_base_cam[2, 2] < -0.5 else 'sideways' if abs(T_base_cam[2, 2]) < 0.5 else 'UP')
    log.info('To see it: cd sim && ./run.sh python scripts/live_view_gl.py --rgbd %s', os.path.abspath(path))
    return True


def still_frames(camera, read_q, n_frames):
    """(joints, frames): `n_frames` captures with the joints read either side, or None if they
    moved between the two reads (the frames would mix viewpoints)."""
    q0 = np.asarray(read_q(), dtype=float)
    frames = [camera.capture() for _ in range(n_frames)]
    q1 = np.asarray(read_q(), dtype=float)
    return None if np.max(np.abs(q1 - q0)) > _STILL_RAD else (q0, frames)


def median_depth(frames):
    """Each pixel's median over the frames that read it; 0 where none did."""
    stack = np.stack([f.depth for f in frames]).astype(np.float32)
    stack[stack <= 0.0] = np.nan
    with warnings.catch_warnings():                          # pixels no frame saw: 0
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)


def save_frame(path, frames, q, model, T_tool0_cam, T_tool0_tip, cfg):
    """Frames of one still view -> one frame.npz (the format in the module docstring; the sim's
    rgbd.load reads it). Returns (depth, T_base_cam)."""
    depth = median_depth(frames)
    T_base_tool0 = model.fk(q)
    T_base_cam = T_base_tool0 @ T_tool0_cam
    extra = {} if T_tool0_tip is None else {'T_tool0_fingertip': T_tool0_tip}
    np.savez_compressed(
        path, rgb=np.ascontiguousarray(frames[-1].color[..., ::-1]), depth_m=depth,
        K=np.asarray(frames[-1].K, dtype=float), T_base_cam=T_base_cam, T_base_tool0=T_base_tool0,
        joints_rad=np.asarray(q, dtype=float), robot=model.NAME, cell=str(cfg.get('cell', '')),
        stamp=time.time(), **extra)
    return depth, T_base_cam


def depth_preview(depth):
    """A false-colour image of the depth: near is warm, no reading is black."""
    seen = depth > 0.0
    if not seen.any():
        return np.zeros(depth.shape + (3,), np.uint8)
    lo, hi = np.percentile(depth[seen], [2, 98])
    t = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap((255 * (1.0 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~seen] = 0
    return img


if __name__ == '__main__':
    main()
