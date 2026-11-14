"""LIVE 6D pose test: a static RealSense (default the D435i) -> the RGB-D pose6d estimator -> CAD overlay.

No robot. The camera frame IS the base frame (T_base_cam = I), so the pose printed is camera-relative.

    python scripts/live_pose6d.py                         # usb_plug, D435i 317422075514
    python scripts/live_pose6d.py --part usb_socket       # the adapter
    python scripts/live_pose6d.py --track                 # feed each estimate back as the next prior

Keys in the window: q quit, space pause, p drop the prior (re-detect from scratch),
n NEXT PLACEMENT (you moved the plug/camera: the frames after it are a new static station).

EVERY ESTIMATE IS SAVED to data/experiments/live_pose6d_<stamp>/ (or --save-dir): est_NNNN.npz with
color, depth_m, K, T_cam_obj, found, score, terms, dt_s, placement, tracked, label (--label, e.g. the tape
distance) -- for scripts/analyze_live_pose6d.py, which says at what camera range the estimate is
reliable. Hold each placement still for ~10+ estimates (repeatability is measured within a placement).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cable_assembly import config as urconfig          # noqa: E402
from cable_assembly import log as urlog                # noqa: E402
from cable_assembly.perception import RealSenseCamera  # noqa: E402
from cable_assembly.perception import pose6d           # noqa: E402


def draw(img, K, T, pts, ok):
    P = (T[:3, :3] @ pts.T).T + T[:3, 3]
    P = P[P[:, 2] > 1e-3]
    uv = (K @ (P / P[:, 2:3]).T).T[:, :2].astype(int)
    col = (0, 255, 0) if ok else (0, 165, 255)
    for u, v in uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, col, -1)
    o = T[:3, 3]
    L = 0.02
    for k, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        a = o + L * T[:3, k]
        p0 = (K @ (o / o[2]))[:2].astype(int)
        p1 = (K @ (a / a[2]))[:2].astype(int)
        cv2.line(img, tuple(p0), tuple(p1), c, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='usb_assembly')
    ap.add_argument('--part', default=pose6d.USB, help='usb_plug | usb_socket | ... (pose6d part names)')
    ap.add_argument('--serial', default='317422075514', help='RealSense serial (default: the D435i)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--track', action='store_true', help='use the last good estimate as the prior')
    ap.add_argument('--out', default='/tmp/live_pose6d.png', help='latest overlay also written here')
    ap.add_argument('--save-dir', default=None, help='default data/experiments/live_pose6d_<stamp>')
    ap.add_argument('--label', default='', help='free text saved with each estimate (e.g. dist=300mm)')
    ap.add_argument('--lost-after', type=int, default=2,
                    help='--track: drop the prior after this many rejected estimates in a row -> full re-detect '
                         '(the prior confines SAM2 to the OLD pose\'s box, so a moved plug is never found again)')
    ap.add_argument('--rgb-only', action='store_true',
                    help='PURE RGB + CAD: the depth is not used (the silhouette size sets the range); '
                         'depth is still recorded so the analysis can check the range against it')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--set', action='append', default=[], help='config override key=value')
    args = ap.parse_args()
    urlog.setup()

    cfg = urconfig.load(args.config, args.set)
    cfg.set_path('camera.serial_no', args.serial)
    cfg.set_path('camera.width', args.width)
    cfg.set_path('camera.height', args.height)
    cfg.set_path('camera.fps', args.fps)
    cfg.set_path('camera.enable_depth', True)
    cfg.set_path('camera.depth_width', 848)       # D435i: best depth mode, ~0.3 m min range
    cfg.set_path('camera.depth_height', 480)
    cfg.set_path('pose6d.gripper_mask', None)     # no gripper in a static view
    cfg.set_path('pose6d.preload', [args.part])
    cfg.set_path('dry_run', False)

    cam = RealSenseCamera(cfg)
    svc = pose6d.service(cfg)
    pts = svc.model_points(args.part, n=1500)
    prior, paused, last = None, False, None
    save_dir = None
    if not args.no_save:
        save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                                                 'experiments', time.strftime('live_pose6d_%Y%m%d_%H%M%S'))
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'model_points.npy'), pts)
        print('saving to', os.path.abspath(save_dir), flush=True)
    n_est, placement, misses = 0, 0, 0
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.canvas.manager.set_window_title('live pose6d')
    view, keys = None, []
    fig.canvas.mpl_connect('key_press_event', lambda e: keys.append(ord(e.key) if e.key and len(e.key) == 1 else 0))
    try:
        while True:
            f = cam.capture()
            f.T_base_cam = np.eye(4)
            img = cv2.cvtColor(np.asarray(f.color), cv2.COLOR_RGB2BGR) if f.color.ndim == 3 else f.color.copy()
            if not paused:
                t0 = time.time()
                try:
                    out = svc.estimate(args.part, [f], prior=prior if args.track else None, hide_gripper=False,
                                       use_depth=not args.rgb_only)
                    T, ok = out.get('T_base_obj'), bool(out.get('found'))
                    if T is None:      # rejected at the gate: keep the view's own candidate, for the analysis
                        v0 = next((v for v in out.get('views', []) if v.get('T_base_obj') is not None), None)
                        if v0 is not None:
                            T = np.asarray(v0['T_base_obj'], float)
                            out = dict(out, score=v0.get('score'), terms=v0.get('terms'))
                    dt = time.time() - t0
                    misses = 0 if ok else misses + 1
                    used_prior = prior is not None
                    if args.track and prior is not None and misses >= args.lost_after:
                        print(f'lost ({misses} rejected in a row): dropping the prior -> full re-detect', flush=True)
                        prior = None
                    if save_dir:
                        np.savez_compressed(
                            os.path.join(save_dir, f'est_{n_est:04d}.npz'),
                            color=np.asarray(f.color), depth_m=np.asarray(f.depth, np.float32),
                            K=np.asarray(f.K, float), T_cam_obj=np.full((4, 4), np.nan) if T is None else T,
                            found=ok, score=np.nan if out.get('score') is None else float(out['score']),
                            terms=json.dumps(out.get('terms') or {}), dt_s=dt, placement=placement,
                            tracked=bool(args.track and used_prior), label=args.label, mode='rgb' if args.rgb_only else 'rgbd',
                            part=args.part, stamp=time.time())
                        n_est += 1
                    if T is not None:
                        last = (T, ok)
                        if ok and args.track:
                            prior = T
                        rpy = np.degrees(cv2.Rodrigues(T[:3, :3])[0].ravel())
                        print(f'{dt:5.2f}s found={ok} score={out.get("score")} '
                              f'xyz_mm={np.round(T[:3, 3] * 1000, 1)} rotvec_deg={np.round(rpy, 1)}', flush=True)
                    else:
                        print(f'{dt:5.2f}s no candidate: {out.get("why")}', flush=True)
                        last = None
                except pose6d.Pose6DError as e:
                    print('estimator error:', e, flush=True)
                    last = None
            if last is not None:
                draw(img, f.K, last[0], pts, last[1])
            cv2.putText(img, f'{args.part}  place={placement} n={n_est}  {"PAUSED" if paused else ""} track={args.track} {"RGB" if args.rgb_only else "RGBD"}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(args.out, img)
            if view is None:
                view = ax.imshow(img[:, :, ::-1])
                ax.axis('off')
            else:
                view.set_data(img[:, :, ::-1])
            plt.pause(0.001)
            if not plt.fignum_exists(fig.number):
                break
            k = keys.pop() if keys else 0
            if k == ord('q'):
                break
            if k == ord(' '):
                paused = not paused
            if k == ord('p'):
                prior = None
            if k == ord('n'):
                placement += 1
                prior = None
                print(f'--- placement {placement} ---', flush=True)
    finally:
        cam.close()
        pose6d.shutdown()


if __name__ == '__main__':
    main()
