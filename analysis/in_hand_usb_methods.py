"""IN-HAND POSE OF THE USB PLUG: several vision methods compared, each with its uncertainty (2026-11-14).

THE DATA: the socket-refine views of the USB runs (data/pose6d/*_6d_pose_the_socket_refine/view_*.npz). The plug
is in the gripper in every one of them, and the camera is on the flange, so within a run the plug's TRUE pose in
the tool frame is one constant: every method's spread across the views of a run is its repeatability (background,
lighting, the arm's pose -- not the viewpoint, which does not change relative to the plug). On top of that each
prior-based method is restarted from perturbed priors (its basin: how much the answer depends on the start).
Run 122035 was pushed home in the socket, so there the connector's pose in the tool at the push is a REFERENCE
(the refined socket target, seen from tool0 at the push) -- an accuracy check, uncertain itself by the socket
estimate (~1-2 mm, ~2 deg) and any slip in the jaws between the views and the push.

THE METHODS
  server (pose6d_server.py, GroundingDINO + SAM2 + CAD render-and-compare), each with depth and without:
    detect    no prior: detection from scratch, coarse + refine (what the scan does)
    refine6   the prior's box -> SAM2 mask, a free 6-DOF refinement from the prior
    jaw       the prior's box -> mask, the 4-DOF jaw-constrained search (slide x/z, pivot, turn) from the prior
  local, no learned segmentation:
    icp       depth points near the prior-posed CAD, trimmed point-to-point ICP onto the CAD surface (6 DOF)
    pca       MODEL-FREE: depth points in a tube ahead of the pads, their principal axis and far end
              (the mating tip) -- 5 DOF (no turn about the axis), no CAD

    ~/miniconda3/envs/cable/bin/python analysis/in_hand_usb_methods.py [--perturb 3] [--out DIR]
Writes DIR/results.npz + summary.csv/.txt + per-run overlay sheets.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cable_assembly import config as C                                   # noqa: E402
from cable_assembly.perception import pose6d as P6                       # noqa: E402
from cable_assembly.perception.camera import Frame                       # noqa: E402
from cable_assembly.skills.pick import fingertip_in_connector            # noqa: E402
from cable_assembly.transforms import from_cfg, inverse, xyzrpy_to_matrix  # noqa: E402
from cable_assembly.robot.kinematics import iiwa14 as K                 # noqa: E402

PART = P6.USB
EXP = 'data/experiments/usb_assembly_20261113_'
RUNS = {   # run -> its refine views; how it was loaded
    '103025': ('20261113_103206', 'hand load, D405'),
    '103559': ('20261113_103747', 'hand load, D405'),
    '115816': ('20261113_120541', 'hand load, D405'),
    '122035': ('20261113_123057', 'hand load, D405, pushed home'),
    '180404': ('20261113_182710', 'picked by the cord, D435i'),
}
# 122035's reference: the refined socket target (log line 140) and the push-home instant (line 179-180)
REF = {'122035': dict(target_xyz_mm=[812.91, 224.65, 270.96], target_rpy_deg=[-0.09, -0.61, -2.37],
                      t_home='12:32:04.87')}


def _clean(d):
    """config_resolved.json carries each pose in both units (xyz_mm and the derived xyz): keep the SI."""
    if isinstance(d, dict):
        out = {k: _clean(v) for k, v in d.items()}
        for a, b in (('xyz', 'xyz_mm'), ('rpy', 'rpy_deg')):
            if a in out and b in out:
                del out[b]                      # from_cfg reads the SI keys
        return out
    if isinstance(d, list):
        return [_clean(v) for v in d]
    return d


def run_cfg(run):
    return C.Config(_clean(json.load(open(EXP + run + '/config_resolved.json'))))


def load_run(run):
    cfg = run_cfg(run)
    T_t0c = from_cfg(cfg.section('hand_eye'))
    T_t0f = from_cfg(cfg.section('fingertip_grasp'))
    grip = fingertip_in_connector(cfg)
    T_ftip_plug = inverse(grip) @ inverse(P6.T_cord_junction(PART))     # the nominal belief's plug (pose6d mode)
    ihc = EXP + run + '/trial_001/in_hand_close.json'
    if os.path.isfile(ihc):                      # picked: the rigidly held estimate, the bogus half turn taken out
        J = json.load(open(ihc))
        Tr = np.asarray(J['T_ftip_conn_rigid'])
        turn = xyzrpy_to_matrix([0, 0, 0], [np.pi, 0, 0])
        # T_ftip_conn = J_ftip @ T_jc; the plug = J_ftip @ inv(Tcj); J_ftip = rigid @ inv(T_jc), its x-turn undone
        T_jc = _junction_conn(cfg, grip)
        J_ftip = Tr @ inverse(T_jc) @ inverse(turn)
        T_ftip_plug = J_ftip @ inverse(P6.T_cord_junction(PART))
    views = []
    for p in sorted(glob.glob(f'data/pose6d/{RUNS[run][0]}_6d_pose_the_socket_refine/view_*.npz')):
        z = np.load(p)
        views.append(Frame(np.ascontiguousarray(z['rgb'][..., ::-1]), np.asarray(z['K'], float), None, 0.0,
                           np.asarray(z['T_base_cam'], float), np.asarray(z['depth_m'], np.float32)))
    return cfg, T_t0c, T_t0f, T_t0f @ T_ftip_plug, views, grip


def _junction_conn(cfg, grip):
    """T_junction_connector as bnc_assembly builds it for pose6d (its y, z zeroed)."""
    from cable_assembly import tool_frames
    init_name = cfg.get_path('estimation.initial_connector_frame')
    init = cfg.get_path('estimation.initial_connector_in_fingertip')
    T_ftip = from_cfg(cfg.section('fingertip_grasp'))
    if init_name:
        frames = tool_frames.load_frames(cfg)
        T_cat = inverse(T_ftip) @ frames[init_name]
    else:
        T_cat = from_cfg(init) if init else from_cfg(cfg.section('junction_in_fingertip'))
    T_jc = inverse(from_cfg(cfg.section('junction_in_fingertip'))) @ T_cat
    T_jc[1:3, 3] = 0.0
    return T_jc


def reference(run, cfg, grip):
    if run not in REF:
        return None
    r = REF[run]
    rows = list(csv.reader(open(EXP + run + '/trial_001/motion.csv')))
    h = rows[0]
    k = min(range(1, len(rows)), key=lambda i: abs(_sec(rows[i][0]) - _sec(r['t_home'])))
    q = np.radians([float(rows[k][h.index(f'q{i}_deg')]) for i in range(1, 8)])
    T_b_t0 = np.asarray(K.fk(q))
    T_b_conn = xyzrpy_to_matrix(np.array(r['target_xyz_mm']) / 1000.0, np.radians(r['target_rpy_deg']))
    T_jc = _junction_conn(cfg, grip)
    return inverse(T_b_t0) @ T_b_conn @ inverse(T_jc) @ inverse(P6.T_cord_junction(PART))


def _sec(s):
    hh, mm, ss = s.split(':')
    return 3600 * int(hh) + 60 * int(mm) + float(ss)


def perturb(T, rng, mm=4.0, deg=4.0):
    d = rng.uniform(-mm, mm, 3) / 1000.0
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R = cv2.Rodrigues(ax * np.radians(rng.uniform(-deg, deg)))[0]
    c = T[:3, :3] @ P6.CENTRE_M[PART] + T[:3, 3]           # about the part's middle
    out = T.copy()
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ (T[:3, 3] - c) + c + d
    return out


# ------------------------------------------------------------------------------------------ local methods
def cloud(frame, step=2):
    d = frame.depth[::step, ::step]
    v, u = np.nonzero(d > 0)
    z = d[v, u].astype(float)
    u, v = u * step, v * step
    K_ = frame.K
    return np.c_[(u - K_[0, 2]) * z / K_[0, 0], (v - K_[1, 2]) * z / K_[1, 1], z]


def icp(P_cam, M, T0, radii=(0.012, 0.008, 0.005, 0.004), iters=15):
    """Trimmed point-to-point ICP of the model points M (part frame) onto the scene P_cam, from T0 (cam_obj)."""
    tree_scene = cKDTree(P_cam)
    T = T0.copy()
    rms = np.nan
    for r in radii:
        for _ in range(iters):
            Mc = (T[:3, :3] @ M.T).T + T[:3, 3]
            d, i = tree_scene.query(Mc, distance_upper_bound=r)
            ok = np.isfinite(d)
            # visible side only: the model points whose match is near; need enough of them
            if ok.sum() < 30:
                return None, np.nan, int(ok.sum())
            A, B = Mc[ok], P_cam[i[ok]]
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            dT = np.eye(4)
            dT[:3, :3], dT[:3, 3] = R, cb - R @ ca
            T = dT @ T
            rms = float(np.sqrt(np.mean(d[ok] ** 2)))
            if np.linalg.norm(dT[:3, 3]) < 1e-5:
                break
    return T, rms, int(ok.sum())


def pca_tip(P_t0, T_t0_prior):
    """MODEL-FREE: points (tool0 frame) in a tube about the prior's axis, from just past the pads' middle to well
    beyond the tip; their principal axis, oriented away from the pads, and its far end (98th pct). No CAD used
    but the tube (which a rough grasp gives)."""
    a0 = T_t0_prior[:3, 2]
    j = T_t0_prior[:3, :3] @ [0, 0, P6.JUNCTION_Z[PART]] + T_t0_prior[:3, 3]
    s = (P_t0 - j) @ a0
    rad = np.linalg.norm((P_t0 - j) - np.outer(s, a0), axis=1)
    sel = (s > 0.004) & (s < 0.090) & (rad < 0.018)
    Q = P_t0[sel]
    if len(Q) < 40:
        return None, len(Q)
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)
    ax = Vt[0] * np.sign(Vt[0] @ a0)
    t = (Q - c) @ ax
    tip = c + ax * np.percentile(t, 98)
    return (ax, tip), len(Q)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perturb', type=int, default=3, help='perturbed restarts per view for prior-based methods')
    ap.add_argument('--out', default='data/in_hand_methods_usb')
    ap.add_argument('--runs', default=','.join(RUNS))
    ap.add_argument('--no-server', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []           # dict per estimate
    svc = None
    if not args.no_server:
        cfg0 = run_cfg('122035')
        cfg0.set_path('pose6d.preload', [PART])
        svc = P6.Pose6DService(cfg0)
        svc.depth_min_m = 0.0                       # depth exactly as each method asks
        M = svc.model_points(PART, n=3000)
    else:
        M = np.load(os.path.join(args.out, 'model_points.npy'))
    np.save(os.path.join(args.out, 'model_points.npy'), M)
    gate = dict(min_iou=0.0, min_edge=0.0, min_cover=0.0)           # always a pose back; the real gate applied after
    jaw_cfg = dict(slide_mm=25.0, pivot_deg=30.0, turn_deg=20.0, trim_mm=6.0)
    for run in args.runs.split(','):
        cfg, T_t0c, T_t0f, T_t0p_prior, views, grip = load_run(run)
        ref = reference(run, cfg, grip)
        print(f'== {run} ({RUNS[run][1]}): {len(views)} views', flush=True)
        for vi, f in enumerate(views):
            T_b_t0 = f.T_base_cam @ inverse(T_t0c)
            T_ct = inverse(T_t0c)                                     # tool in camera
            inits = [T_t0p_prior] + [perturb(T_t0p_prior, rng) for _ in range(args.perturb)]

            def add(method, depth, k, T_t0p, terms=None, dt=np.nan, extra=None):
                rows.append(dict(run=run, view=vi, method=method, depth=depth, init=k,
                                 T=None if T_t0p is None else np.asarray(T_t0p, float), terms=terms or {}, dt=dt,
                                 extra=extra or {}))
            if svc is not None:
                for depth in (True, False):
                    t = time.time()
                    try:
                        r = svc.estimate(PART, [f], prior=None, use_depth=depth, hide_gripper=False, **gate)
                        T = inverse(T_b_t0) @ r['T_base_obj'] if r.get('T_base_obj') is not None else None
                        add('detect', depth, 0, T, r.get('terms'), time.time() - t)
                    except P6.Pose6DError as e:
                        add('detect', depth, 0, None, {'error': str(e)})
                    for k, T0 in enumerate(inits):
                        prior = T_b_t0 @ T0
                        for meth, extra in (('refine6', {}),
                                            ('jaw', {'jaw': dict(jaw_cfg, T_cam_tool=T_ct.tolist())})):
                            t = time.time()
                            try:
                                r = svc.estimate(PART, [f], prior=prior, use_depth=depth, hide_gripper=False,
                                                 refine_only=True, joint_iters=0, quiet=True, **gate, **extra)
                                T = inverse(T_b_t0) @ r['T_base_obj'] if r.get('T_base_obj') is not None else None
                                add(meth, depth, k, T, r.get('terms'), time.time() - t)
                            except P6.Pose6DError as e:
                                add(meth, depth, k, None, {'error': str(e)})
            # local: ICP and PCA on depth
            P_cam = cloud(f)
            P_t0 = (T_t0c[:3, :3] @ P_cam.T).T + T_t0c[:3, 3]
            near = cKDTree((T_t0p_prior[:3, :3] @ M.T).T + T_t0p_prior[:3, 3])
            dmin, _ = near.query(P_t0, distance_upper_bound=0.03)
            P_crop = P_cam[np.isfinite(dmin)]
            for k, T0 in enumerate(inits):
                t = time.time()
                Tc, rms, n = icp(P_crop, M, T_ct @ T0) if len(P_crop) > 50 else (None, np.nan, len(P_crop))
                add('icp', True, k, None if Tc is None else T_t0c @ Tc, {'rms_mm': 1000 * rms, 'n': n},
                    time.time() - t)
            got, n = pca_tip(P_t0, T_t0p_prior)
            add('pca', True, 0, None, {'n': n}, extra=None if got is None else {'axis': got[0], 'tip': got[1]})
            print(f'  view {vi}: {sum(1 for r in rows if r["run"] == run and r["view"] == vi)} estimates', flush=True)
        np.save(os.path.join(args.out, f'prior_{run}.npy'), T_t0p_prior)
        if ref is not None:
            np.save(os.path.join(args.out, f'ref_{run}.npy'), ref)
    np.save(os.path.join(args.out, 'rows.npy'), np.array(rows, dtype=object), allow_pickle=True)
    if svc is not None:
        svc.close()
    print('saved', len(rows), 'estimates to', args.out)


if __name__ == '__main__':
    main()
