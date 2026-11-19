"""Pictures of the in-hand USB estimates: per view, the current estimator (server jaw fit, RGB) against the improved
RGB-only method (analysis/in_hand_rgb_usb.py), with the depth-PCA proxy truth where there is one (D405 runs).
Writes data/in_hand_rgb_usb/viz/{view}.jpg crops and errors.json.
    ~/miniconda3/envs/cable/bin/python analysis/in_hand_rgb_usb_viz.py
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import in_hand_rgb_usb as R  # noqa: E402

OUT = 'data/in_hand_rgb_usb/viz'
os.makedirs(OUT, exist_ok=True)
SAT = 120                                       # the unsaturated trim: drops the blue fixture (255 kept it)
LEN = 0.030                                     # the drawn axis: tip to 30 mm back along it
COL = dict(truth=(122, 175, 27), current=(52, 104, 235), improved=(214, 120, 42))   # BGR: aqua, orange, blue


def proj(X, k):
    T_ct, K = R.V[f'{k}__T_ct'], R.V[f'{k}__K']
    c = T_ct[:3, :3] @ X + T_ct[:3, 3]
    return (K @ (c / c[2]))[:2]


def draw(img, est, k, col, thick=3):
    if est is None:
        return
    a, b = proj(est['tip'], k), proj(est['tip'] - LEN * est['axis'], k)
    cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), col, thick, cv2.LINE_AA)
    cv2.circle(img, tuple(np.int32(a)), 9, col, thick, cv2.LINE_AA)


def main():
    base = R.jaw_baseline()
    d405 = [k for k in R.KEYS if not k.startswith('180404')]
    Wv = {}
    for k in d405:
        e = R.scale_estimate(k, 12.0, sat=SAT)
        gt = R.V[f'{k}__pca_tip']
        if e is None or not np.all(np.isfinite(gt)):
            continue
        Zgt = (R.V[f'{k}__T_ct'][:3, :3] @ gt + R.V[f'{k}__T_ct'][:3, 3])[2]
        Wv[k] = e['w'] * Zgt / R.V[f'{k}__K'][0, 0] * 1000
    Wall = float(np.median(list(Wv.values())))
    rows = []
    for k in R.KEYS:
        run = k.split('_')[0]
        W = Wall if run == '180404' else float(np.median([w for kk, w in Wv.items() if not kk.startswith(run)]))
        cur, new = base.get(('jaw', k)), R.rgb_inhand(k, W, sat=SAT)
        gt = None
        if np.all(np.isfinite(R.V[f'{k}__pca_tip'])) and run != '180404':
            gt = dict(tip=R.V[f'{k}__pca_tip'], axis=R.V[f'{k}__pca_axis'])
        img = cv2.cvtColor(R.V[f'{k}__rgb'], cv2.COLOR_RGB2BGR).copy()
        if new is not None:                       # the mask the improved method used
            cnt, _ = cv2.findContours(new['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnt, -1, (255, 255, 255), 1, cv2.LINE_AA)
        draw(img, gt, k, COL['truth'], 5)
        draw(img, cur, k, COL['current'])
        draw(img, new, k, COL['improved'])
        # crop around the fingertip / the drawn things
        pts = [R.V[f'{k}__ftip_uv']] + [proj(e['tip'], k) for e in (gt, cur, new) if e is not None]
        c = np.median(np.array(pts), 0)
        h, w = img.shape[:2]
        s = 300 if run == '180404' else 170
        x0 = int(np.clip(c[0] - s, 0, w - 2 * s)); y0 = int(np.clip(c[1] - s, 0, h - 2 * s))
        crop = cv2.resize(img[y0:y0 + 2 * s, x0:x0 + 2 * s], (480, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{OUT}/{k}.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        r = dict(view=k, run=run, cam='D435i' if run == '180404' else 'D405', kept=new is not None)
        for name, e in (('current', cur), ('improved', new)):
            s_ = R.score(e, k) if gt is not None else None
            r[name] = {kk: round(float(v), 2) for kk, v in s_.items()} if s_ else None
            r[name + '_tip'] = None if e is None else [round(float(v) * 1000, 1) for v in e['tip']]
        rows.append(r)
        print(k, r['current'], r['improved'])
    json.dump(dict(W_eff=Wall, rows=rows), open(f'{OUT}/errors.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
