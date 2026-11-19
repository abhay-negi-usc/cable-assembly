"""Report for analysis/in_hand_usb_methods.py: each method's uncertainty (2026-11-14).

    ~/miniconda3/envs/cable/bin/python analysis/in_hand_usb_methods_report.py [DIR]

Per method (and depth on/off), from the plug's pose in the TOOL frame:
  pass       the in-hand gate (iou >= 0.5, edge_fwd >= 0.4, edge_cover >= 0.5; pose6d.in_hand) -- server methods
  REPEAT     within a run, across its views (the plug is fixed in the camera): the RMS of the mating tip about the
             run's mean (mm), split along the plug's axis / across it; the axis's RMS tilt; the turn about it
  BASIN      per view, across the restarts from perturbed priors (+-4 mm, +-4 deg): RMS of the tip about the view's mean
  SIGMA      both pooled: every estimate of a run about the run's mean (what one estimate is worth)
  ACCURACY   run 122035 only (pushed home): the run-mean tip and axis against the reference
  CONSENSUS  each run: the method's mean tip against the median of all methods' means (agreement, not truth)
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cable_assembly.perception import pose6d as P6   # noqa: E402

PART = P6.USB
TIP = np.array([0.0, 0.0, P6.MATING_Z[PART]])
GATE = dict(iou=0.5, edge_fwd=0.4, edge_cover=0.5)


def tip(T):
    return T[:3, :3] @ TIP + T[:3, 3]


def passes(terms):
    return all(float(terms.get(k, -1)) >= v for k, v in GATE.items())


def mean_axis(A):
    m = np.mean(A, axis=0)
    return m / np.linalg.norm(m)


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))


def turn(Ts, axis):
    """Each pose's turn about `axis` (deg), from its x axis projected onto the plane normal to it, relative to the first."""
    xs = []
    for T in Ts:
        x = T[:3, 0] - (T[:3, 0] @ axis) * axis
        xs.append(x / np.linalg.norm(x))
    e1 = xs[0]
    e2 = np.cross(axis, e1)
    a = np.degrees([np.arctan2(x @ e2, x @ e1) for x in xs])
    a = (a - np.degrees(np.angle(np.mean(np.exp(1j * np.radians(a)))))+ 180) % 360 - 180
    return a


def split(d, axis):
    along = d @ axis
    lat = np.linalg.norm(d - np.outer(along, axis), axis=1)
    return along, lat


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'data/in_hand_methods_usb'
    rows = list(np.load(os.path.join(out, 'rows.npy'), allow_pickle=True))
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    if only:
        rows = [r for r in rows if r['run'] in only]
    runs = sorted({r['run'] for r in rows})
    refs = {run: np.load(os.path.join(out, f'ref_{run}.npy')) for run in runs
            if os.path.isfile(os.path.join(out, f'ref_{run}.npy'))}
    # each estimate -> (tip, axis, T or None)
    for r in rows:
        if r['method'] == 'pca':
            e = r['extra']
            r['tip'], r['axis'] = (None, None) if not e else (np.asarray(e['tip']), np.asarray(e['axis']))
        elif r['T'] is not None:
            r['tip'], r['axis'] = tip(r['T']), r['T'][:3, 2]
        else:
            r['tip'] = r['axis'] = None
        r['pass'] = passes(r['terms']) if r['method'] in ('detect', 'refine6', 'jaw') else (
            r['tip'] is not None and (r['method'] != 'icp' or r['terms'].get('rms_mm', 99) < 3.0))
    keys = []
    for r in rows:
        k = (r['method'], r['depth'])
        if k not in keys:
            keys.append(k)
    # consensus per run: the median of the methods' mean tips (init-0, passed estimates)
    means = {}
    for k in keys:
        for run in runs:
            E = [r for r in rows if (r['method'], r['depth']) == k and r['run'] == run and r['init'] == 0
                 and r['tip'] is not None]
            if E:
                means[(k, run)] = (np.mean([e['tip'] for e in E], 0), mean_axis([e['axis'] for e in E]))
    consensus = {run: np.median([v[0] for (k, rr), v in means.items() if rr == run], 0) for run in runs}

    lines, table = [], []
    for k in keys:
        name = f'{k[0]} ({"RGB-D" if k[1] else "RGB"})' if k[0] in ('detect', 'refine6', 'jaw') else (
            'icp (depth, CAD)' if k[0] == 'icp' else 'pca (depth, model-free)')
        R = [r for r in rows if (r['method'], r['depth']) == k]
        got = [r for r in R if r['tip'] is not None]
        rep_t, rep_al, rep_lat, rep_ax, rep_turn, basin, pooled, cons = [], [], [], [], [], [], [], []
        for run in runs:
            E0 = [r for r in got if r['run'] == run and r['init'] == 0]
            if len(E0) >= 2:
                P = np.array([e['tip'] for e in E0])
                ax = mean_axis([e['axis'] for e in E0])
                d = P - P.mean(0)
                al, la = split(d, ax)
                rep_t += list(np.linalg.norm(d, axis=1))
                rep_al += list(al)
                rep_lat += list(la)
                rep_ax += [ang(e['axis'], ax) for e in E0]
                if E0[0].get('T') is not None and k[0] != 'pca':
                    rep_turn += list(turn([e['T'] for e in E0], ax))
            Er = [r for r in got if r['run'] == run]
            if len(Er) >= 2:
                P = np.array([e['tip'] for e in Er])
                pooled += list(np.linalg.norm(P - P.mean(0), axis=1))
            for v in {r['view'] for r in Er}:
                Ev = [r for r in Er if r['view'] == v]
                if len(Ev) >= 2:
                    P = np.array([e['tip'] for e in Ev])
                    basin += list(np.linalg.norm(P - P.mean(0), axis=1))
            if (k, run) in means:
                cons.append(np.linalg.norm(means[(k, run)][0] - consensus[run]))
        rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float('nan')   # noqa: E731
        acc = {}
        for run, ref in refs.items():
            if (k, run) in means:
                m_tip, m_ax = means[(k, run)]
                d = m_tip - tip(ref)
                al, la = split(d[None], ref[:3, 2])
                acc = dict(acc_mm=float(np.linalg.norm(d)) * 1000, acc_along_mm=float(al[0]) * 1000,
                           acc_lat_mm=float(la[0]) * 1000, acc_axis_deg=ang(m_ax, ref[:3, 2]))
                Er = [r for r in got if r['run'] == run and r['init'] == 0]
                acc['acc_each_mm'] = float(np.median([np.linalg.norm(e['tip'] - tip(ref)) for e in Er])) * 1000
        row = dict(method=name, n=len(R), found=len(got), pass_rate=np.mean([r['pass'] for r in R]),
                   rep_tip_mm=rms(rep_t) * 1000, rep_along_mm=rms(rep_al) * 1000, rep_lat_mm=rms(rep_lat) * 1000,
                   rep_axis_deg=rms(rep_ax), rep_turn_deg=rms(rep_turn), basin_mm=rms(basin) * 1000,
                   sigma_mm=rms(pooled) * 1000, consensus_mm=float(np.median(cons)) * 1000 if cons else np.nan,
                   dt_s=float(np.nanmedian([r['dt'] for r in R])) if R else np.nan, **acc)
        table.append(row)
    cols = ['method', 'found', 'n', 'pass_rate', 'rep_tip_mm', 'rep_along_mm', 'rep_lat_mm', 'rep_axis_deg',
            'rep_turn_deg', 'basin_mm', 'sigma_mm', 'consensus_mm', 'acc_mm', 'acc_along_mm', 'acc_lat_mm',
            'acc_axis_deg', 'acc_each_mm', 'dt_s']
    tag = '' if not only else '_' + '_'.join(only)
    with open(os.path.join(out, f'summary{tag}.csv'), 'w') as fh:
        w = csv.DictWriter(fh, cols, extrasaction='ignore')
        w.writeheader()
        for r in table:
            w.writerow(r)
    hdr = f"{'method':26s} {'found':>7s} {'pass':>5s} | {'rep tip':>7s} {'along':>6s} {'lat':>5s} {'axis':>5s} {'turn':>5s} | {'basin':>6s} {'sigma':>6s} | {'cons':>5s} | {'acc':>6s} {'along':>6s} {'lat':>5s} {'axis':>5s} | {'s/est':>5s}"
    lines.append(hdr)
    lines.append('-' * len(hdr))
    f = lambda v, p=1: '   nan' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:6.{p}f}'   # noqa: E731
    for r in table:
        lines.append(f"{r['method']:26s} {r['found']:3d}/{r['n']:<3d} {r['pass_rate']:5.2f} | {f(r['rep_tip_mm'])} "
                     f"{f(r['rep_along_mm'])} {f(r['rep_lat_mm'])[1:]} {f(r['rep_axis_deg'])[1:]} {f(r['rep_turn_deg'])[1:]} | "
                     f"{f(r['basin_mm'])} {f(r['sigma_mm'])} | {f(r['consensus_mm'])[1:]} | {f(r.get('acc_mm'))} "
                     f"{f(r.get('acc_along_mm'))} {f(r.get('acc_lat_mm'))[1:]} {f(r.get('acc_axis_deg'))[1:]} | {f(r['dt_s'])[1:]}")
    lines.append('mm / deg. rep = across a run\'s views (init 0); basin = across restarts of one view; sigma = pooled; '
                 'cons = from the methods\' median; acc = run 122035 vs the pushed-home reference.')
    # per-run detail of tips (init 0 means) for the record
    lines.append('')
    lines.append('per run: method mean tip (tool0, mm) [n]')
    for run in runs:
        lines.append(f'  {run}: consensus {np.round(consensus[run] * 1000, 1).tolist()}'
                     + (f'  REFERENCE {np.round(tip(refs[run]) * 1000, 1).tolist()}' if run in refs else ''))
        for k in keys:
            if (k, run) in means:
                lines.append(f'      {k[0]:8s} {"D" if k[1] else "-"} {np.round(means[(k, run)][0] * 1000, 1).tolist()}')
    txt = '\n'.join(lines)
    open(os.path.join(out, f'summary{tag}.txt'), 'w').write(txt)
    print(txt)
    if not only:
        overlays(out, rows, runs, refs)


def overlays(out, rows, runs, refs):
    """View 0 of each run with every method's init-0 estimate's CAD drawn (and the reference where there is one)."""
    import glob
    import json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import in_hand_usb_methods as H
    M = np.load(os.path.join(out, 'model_points.npy'))[::3]
    cols = {'detect': (0, 0, 255), 'refine6': (0, 165, 255), 'jaw': (0, 255, 0), 'icp': (255, 0, 255),
            'pca': (255, 255, 0), 'ref': (255, 255, 255)}
    for run in runs:
        _cfg, T_t0c, _f, _p, views, _g = H.load_run(run)
        f = views[0]
        tiles = []
        for depth in (True, False):
            img = f.color.copy()
            ests = [r for r in rows if r['run'] == run and r['view'] == 0 and r['init'] == 0
                    and (r['depth'] == depth or r['method'] in ('icp', 'pca'))]
            items = [(r['method'], r) for r in ests] + ([('ref', {'T': refs[run]})] if run in refs else [])
            for m, r in items:
                if m == 'pca':
                    if r.get('tip') is None:
                        continue
                    T_ct = np.linalg.inv(T_t0c)
                    a = T_ct[:3, :3] @ r['tip'] + T_ct[:3, 3]
                    b = T_ct[:3, :3] @ (r['tip'] - 0.05 * r['axis']) + T_ct[:3, 3]
                    pa, pb = (f.K @ (a / a[2]))[:2], (f.K @ (b / b[2]))[:2]
                    cv2.line(img, tuple(map(int, pa)), tuple(map(int, pb)), cols[m], 2)
                    cv2.circle(img, tuple(map(int, pa)), 6, cols[m], 2)
                    continue
                T = r.get('T')
                if T is None:
                    continue
                Tc = np.linalg.inv(T_t0c) @ T
                P = (Tc[:3, :3] @ M.T).T + Tc[:3, 3]
                uv = (f.K @ (P / P[:, 2:3]).T).T[:, :2].astype(int)
                for u, v in uv:
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        img[v, u] = cols[m]
            # crop round the plug
            T = refs.get(run, _p)
            Tc = np.linalg.inv(T_t0c) @ (T if run in refs else _p)
            c = Tc[:3, :3] @ (TIP - [0, 0, 0.025]) + Tc[:3, 3]
            u, v = (f.K @ (c / c[2]))[:2].astype(int)
            s = int(0.09 * f.K[0, 0] / max(c[2], 0.05))
            crop = img[max(0, v - s):v + s, max(0, u - s):u + s]
            crop = cv2.resize(crop, (480, 480))
            cv2.putText(crop, f'{run} {"RGB-D" if depth else "RGB"}', (8, 24), 0, 0.7, (255, 255, 255), 2)
            tiles.append(crop)
        leg = np.zeros((480, 200, 3), np.uint8)
        for i, (m, c) in enumerate(cols.items()):
            cv2.putText(leg, m, (10, 40 + 30 * i), 0, 0.7, c, 2)
        cv2.imwrite(os.path.join(out, f'overlay_{run}.jpg'), np.hstack(tiles + [leg]))


if __name__ == '__main__':
    main()
