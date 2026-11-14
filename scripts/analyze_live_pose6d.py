"""How far should the RGB-D camera be from the connector? -- from scripts/live_pose6d.py's saved estimates.

    python scripts/analyze_live_pose6d.py data/experiments/live_pose6d_<stamp> [more dirs ...]

THE PROTOCOL (what makes the answer mean something): put the plug at one range, hold it STILL, let ~10+
estimates run, press `n`, move to the next range (e.g. 150, 200, 250 ... 600 mm), repeat. Use --track
as in the real pipeline (or not, but the same for every placement). No ground truth is needed: at each
placement the plug does not move, so the spread of the accepted estimates IS the estimator's error
(repeatability), and the rest is measured against the camera's own depth.

PER ESTIMATE:  range (|t|, mm), accepted, score + its terms (iou, edge_fwd, edge_cover, depth), runtime,
               the plug's size in the image (px, from the CAD points projected), and the DEPTH on the plug:
               the fraction of its pixels with a reading and the median |measured - model| depth (mm).
PER PLACEMENT: median range, acceptance rate, translation spread (RMS about the median, mm), rotation
               spread (median geodesic to the chordal mean, deg; flips > 90 deg counted apart -- a
               USB-A's wide faces look alike), depth coverage/residual, median runtime.
THE ANSWER: the range band whose placements meet ALL of --min-accept, --max-trans-mm, --max-rot-deg,
               --min-depth-valid. Written to <first dir>/analysis/ as per_estimate.csv, per_placement.csv,
               range_study.png and summary.txt.
"""
import argparse
import glob
import json
import os

import numpy as np


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def depth_stats(depth, K, T, pts):
    """(px size, valid fraction, median |depth - model| mm, median signed mm) over the plug's visible pixels."""
    H, W = depth.shape
    P = (T[:3, :3] @ pts.T).T + T[:3, 3]
    P = P[P[:, 2] > 1e-3]
    if len(P) == 0:
        return np.nan, np.nan, np.nan, np.nan
    uv = (K @ (P / P[:, 2:3]).T).T[:, :2]
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inside.sum() < 5:
        return np.nan, 0.0, np.nan, np.nan
    size_px = float(np.linalg.norm(uv[inside].max(0) - uv[inside].min(0)))
    u, v, z = u[inside], v[inside], P[inside, 2]
    # z-buffer: the nearest model point per pixel = the visible surface
    zb = {}
    for a, b, c in zip(u, v, z):
        if c < zb.get((a, b), np.inf):
            zb[(a, b)] = c
    keys = np.array(list(zb.keys()))
    zm = np.array(list(zb.values()))
    dm = depth[keys[:, 1], keys[:, 0]]
    ok = dm > 0
    if ok.sum() == 0:
        return size_px, 0.0, np.nan, np.nan
    r = (dm[ok] - zm[ok]) * 1000.0
    return size_px, float(ok.mean()), float(np.median(np.abs(r))), float(np.median(r))


def load(dirs):
    rows, pts = [], None
    for d in dirs:
        mp = os.path.join(d, 'model_points.npy')
        pts_d = np.load(mp) if os.path.exists(mp) else None
        for f in sorted(glob.glob(os.path.join(d, 'est_*.npz'))):
            z = np.load(f, allow_pickle=False)
            T = z['T_cam_obj']
            row = dict(file=os.path.relpath(f), run=os.path.basename(os.path.normpath(d)),
                       placement=int(z['placement']), label=str(z['label']), mode=str(z['mode']) if 'mode' in z else 'rgbd', found=bool(z['found']),
                       score=float(z['score']), dt_s=float(z['dt_s']), tracked=bool(z['tracked']), T=T)
            row.update({k: float(v) for k, v in json.loads(str(z['terms'])).items()})
            valid = np.isfinite(T).all()
            # THE PLUG'S CENTRE, not T's origin: the CAD origin is far off the part (usb_plug: ~0.4 m)
            c = pts_d.mean(0) if pts_d is not None else np.zeros(3)
            p = T[:3, :3] @ c + T[:3, 3] if valid else np.full(3, np.nan)
            row['p_cam'] = p
            row['range_mm'] = float(np.linalg.norm(p) * 1000)
            row['z_mm'] = float(p[2] * 1000)
            if valid and pts_d is not None:
                (row['size_px'], row['depth_valid'], row['depth_abs_mm'],
                 row['depth_bias_mm']) = depth_stats(z['depth_m'], z['K'], T, pts_d)
            rows.append(row)
        pts = pts if pts is not None else pts_d
    return rows


def per_placement(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['run'], r['mode'], r['placement']), []).append(r)
    out = []
    for (run, mode, pl), rs in sorted(groups.items()):
        acc = [r for r in rs if r['found'] and np.isfinite(r['T']).all()]
        g = dict(run=run, mode=mode, placement=pl, label=rs[0]['label'], n=len(rs), n_accepted=len(acc),
                 accept_rate=len(acc) / len(rs), dt_s=float(np.median([r['dt_s'] for r in rs])))
        src = acc if acc else [r for r in rs if np.isfinite(r['T']).all()]
        for k in ('range_mm', 'z_mm', 'size_px', 'depth_valid', 'depth_abs_mm', 'depth_bias_mm', 'score',
                  'iou', 'edge_fwd', 'edge_cover', 'depth'):
            vals = [r[k] for r in src if k in r and np.isfinite(r[k])]
            g[k] = float(np.median(vals)) if vals else np.nan
        if len(acc) >= 3:
            t = np.array([r['p_cam'] for r in acc]) * 1000
            g['trans_rms_mm'] = float(np.sqrt(np.mean(np.sum((t - np.median(t, 0)) ** 2, 1))))
            g['trans_axial_std_mm'] = float(np.std(t @ (np.median(t, 0) / np.linalg.norm(np.median(t, 0)))))
            Rs = np.array([r['T'][:3, :3] for r in acc])
            Rm = chordal_mean(Rs)
            ang = np.array([rot_angle_deg(Rm.T @ R) for R in Rs])
            g['flips'] = int((ang > 90).sum())
            g['rot_med_deg'] = float(np.median(ang[ang <= 90])) if (ang <= 90).any() else np.nan
            g['rot_p90_deg'] = float(np.percentile(ang[ang <= 90], 90)) if (ang <= 90).any() else np.nan
        else:
            g.update(trans_rms_mm=np.nan, trans_axial_std_mm=np.nan, flips=0, rot_med_deg=np.nan, rot_p90_deg=np.nan)
        out.append(g)
    return out


def write_csv(path, rows, keys):
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join('' if r.get(k) is None else (f'{r[k]:.4g}' if isinstance(r.get(k), float) else str(r.get(k)))
                             for k in keys) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--min-accept', type=float, default=0.9)
    ap.add_argument('--max-trans-mm', type=float, default=1.0, help='user spec +/-1 mm')
    ap.add_argument('--max-rot-deg', type=float, default=2.0, help='p90 rotation spread')
    ap.add_argument('--min-depth-valid', type=float, default=0.5)
    args = ap.parse_args()

    rows = load(args.dirs)
    if not rows:
        raise SystemExit('no est_*.npz found')
    pls = per_placement(rows)
    out = os.path.join(args.dirs[0], 'analysis')
    os.makedirs(out, exist_ok=True)
    ek = ['file', 'run', 'mode', 'placement', 'label', 'found', 'tracked', 'score', 'iou', 'edge_fwd', 'edge_cover', 'depth',
          'range_mm', 'z_mm', 'size_px', 'depth_valid', 'depth_abs_mm', 'depth_bias_mm', 'dt_s']
    write_csv(os.path.join(out, 'per_estimate.csv'), rows, ek)
    pk = ['run', 'mode', 'placement', 'label', 'n', 'n_accepted', 'accept_rate', 'range_mm', 'z_mm', 'size_px', 'trans_rms_mm',
          'trans_axial_std_mm', 'rot_med_deg', 'rot_p90_deg', 'flips', 'depth_valid', 'depth_abs_mm', 'depth_bias_mm',
          'score', 'iou', 'edge_fwd', 'edge_cover', 'depth', 'dt_s']
    pls.sort(key=lambda g: (np.nan_to_num(g['range_mm'], nan=1e9)))
    write_csv(os.path.join(out, 'per_placement.csv'), pls, pk)

    def good(g):
        return (g['accept_rate'] >= args.min_accept and g['trans_rms_mm'] <= args.max_trans_mm
                and g['rot_p90_deg'] <= args.max_rot_deg and g['flips'] == 0
                and g['depth_valid'] >= args.min_depth_valid)

    lines = [f'{len(rows)} estimates, {len(pls)} placements ({", ".join(args.dirs)})', '',
             f'{"mode":>4} {"pl":>3} {"label":>12} {"range":>7} {"n":>3} {"acc":>5} {"t_rms":>6} {"rot90":>6} {"flip":>4} '
             f'{"dvalid":>6} {"d_err":>6} {"px":>5} {"score":>6} {"dt":>5}  ok']
    for g in pls:
        lines.append(f'{g["mode"]:>4} {g["placement"]:>3} {g["label"][:12]:>12} {g["range_mm"]:7.0f} {g["n"]:3d} {g["accept_rate"]:5.2f} '
                     f'{g["trans_rms_mm"]:6.2f} {g["rot_p90_deg"]:6.2f} {g["flips"]:4d} {g["depth_valid"]:6.2f} '
                     f'{g["depth_abs_mm"]:6.2f} {g["size_px"]:5.0f} {g["score"]:6.3f} {g["dt_s"]:5.1f}  '
                     f'{"YES" if good(g) else "-"}')
        crit = (f'accept >= {args.min_accept}, trans RMS <= {args.max_trans_mm} mm, rot p90 <= {args.max_rot_deg} deg, '
            f'no flips, depth valid >= {args.min_depth_valid}')
    for mode in sorted({g['mode'] for g in pls}):
      P = [g for g in pls if g['mode'] == mode]
      ok = [g['range_mm'] for g in P if good(g)]
      lines.append('')
      lines.append(f'[{mode}]')
      if ok:
        best = min((g for g in P if good(g)), key=lambda g: g['trans_rms_mm'] + 0.5 * g['rot_p90_deg'])
        lines.append(f'MEETS ALL ({crit}): {min(ok):.0f} - {max(ok):.0f} mm camera-to-plug range '
                     f'({len(ok)}/{len(P)} placements); tightest at {best["range_mm"]:.0f} mm '
                     f'({best["trans_rms_mm"]:.2f} mm, {best["rot_p90_deg"]:.2f} deg).')
        bad_in = [g['range_mm'] for g in P if not good(g) and min(ok) < g['range_mm'] < max(ok)]
        if bad_in:
            lines.append(f'  but failing placements INSIDE that band at {", ".join(f"{r:.0f}" for r in bad_in)} mm '
                         '-- not a clean band, take more data there.')
      else:
        lines.append(f'NO placement meets all of: {crit}.')
      if mode == 'rgb':
        lines.append('  (rgb: depth_bias_mm = measured depth - the RGB-only model surface = its RANGE ERROR, '
                     'the one accuracy number here)')
    lines.append('Repeatability only (static plug): a bias common to all estimates at one range is NOT seen; '
                 'depth_bias_mm is the one hint of it.')
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(out, 'summary.txt'), 'w').write(txt + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = np.array([g['range_mm'] for g in pls])
    col = ['tab:green' if good(g) else ('tab:red' if g['mode'] == 'rgbd' else 'tab:orange') for g in pls]
    panels = [('accept_rate', 'acceptance rate', args.min_accept), ('trans_rms_mm', 'translation RMS (mm)', args.max_trans_mm),
              ('rot_p90_deg', 'rotation p90 (deg)', args.max_rot_deg), ('depth_valid', 'depth valid on plug', args.min_depth_valid),
              ('depth_bias_mm', 'depth - model, median (mm)', None), ('size_px', 'plug size in image (px)', None),
              ('score', 'score', None), ('dt_s', 'runtime (s)', None)]
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (k, title, thr) in zip(axs.ravel(), panels):
        for mk, m in (('o', 'rgbd'), ('^', 'rgb')):
            sel = [i for i, g in enumerate(pls) if g['mode'] == m]
            if sel:
                ax.scatter(r[sel], [pls[i][k] for i in sel], c=[col[i] for i in sel], marker=mk, s=40, zorder=3,
                           label=m)
        ax.legend(fontsize=7)
        if k in ('depth_bias_mm', 'size_px', 'score', 'dt_s'):
            rr = [x['range_mm'] for x in rows if k in x and np.isfinite(x.get(k, np.nan))]
            vv = [x[k] for x in rows if k in x and np.isfinite(x.get(k, np.nan))]
            ax.scatter(rr, vv, s=6, c='0.6', zorder=2)
        if thr is not None:
            ax.axhline(thr, ls='--', c='k', lw=0.8)
        ax.set_title(title)
        ax.set_xlabel('camera-to-plug range (mm)')
        ax.grid(alpha=0.3)
    fig.suptitle('pose6d vs camera range (green = meets all criteria; grey = single estimates)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'range_study.png'), dpi=110)
    print('wrote', os.path.abspath(out))


if __name__ == '__main__':
    main()
