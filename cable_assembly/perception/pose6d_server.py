"""The 6D pose service: the CAD-based estimator in ~/abhay_ws/RAG-6DPose-code-abhay (pose_pipeline:
Grounding DINO + SAM2 for the part's mask, render-and-compare against its mesh for the pose), kept
loaded in a process of its own and asked for poses from SEVERAL VIEWS at once.

    <rag6dpose env python> -u pose6d_server.py --repo ~/abhay_ws/RAG-6DPose-code-abhay [--preload adapter,...]

WHY A PROCESS OF ITS OWN. It runs in the `rag6dpose` env (Python 3.10, transformers 5, its own
torch), which the `cable` env cannot import, and loading its models takes ~10 s -- so it is started
once and kept (cable_assembly/perception/pose6d.py starts it and talks to it). --preload: these
objects' estimators (and their rivals') are loaded and every model run once before it is ready.

THE PROTOCOL: one JSON object per line on stdin, one reply per line on stdout; everything else
(progress) goes to stderr. First it prints {"ready": true, ...}. Requests:

    {"op": "estimate", "id": n, "object": "adapter",
     "views": [{"npz": "/dev/shm/...", "T_base_cam": 4x4 (m)}, ...],
     "prior": 4x4 T_base_obj (m) or null, "symmetry": "none" | "axis_z" | "z2" | "z4",
     "min_iou": .75, "min_edge": .5, "min_cover": .7, "max_candidates": 4, "refine_top": 2,
     "consistent_deg": 10, "consistent_mm": 10, "joint_iters": 40, "use_depth": true, "refine_only": false, "quiet": false}
    {"op": "model", "id": n, "object": "adapter", "n": 800}    -> {"points": (n, 3) m, the CAD frame}
    {"op": "quit"}

use_depth false: the views' depth is ignored (the silhouette's area sets the range). refine_only
true: a view where the prior's projection gives no mask (out of the image) has no pose -- no
search of the whole image (a tracker's request: ~40 s otherwise, and maybe another part). quiet true: no
progress lines for it (a tracker's, one after another, would bury the app's prompts).

Each view's npz holds color (H, W, 3 uint8 BGR), depth_mm (H, W float32, 0 = none), K (3, 3).

AN ESTIMATE FROM SEVERAL VIEWS:
  1. EACH VIEW on its own: with no prior, the pipeline's own search (candidates from the text
     prompts, coarse SO(3) search, refinement of the best, the acceptance gate, and the object's
     `rivals` in objects.yaml: a candidate a rival's model fits better is that part, not this one,
     and the next is tried); with a prior --
     the pose a first view found -- its box is the prior's projection (SAM2 on it, no detection)
     and the prior is the refinement's seed, which is a few seconds instead of ~10.
  2. THE CONSISTENT VIEWS: those whose pose agrees with the best-scoring view's within
     consistent_deg / consistent_mm, modulo the part's symmetry (axis_z: only the z axis is
     compared, the plug turns freely about it; z2 / z4: half / quarter turns about z -- the
     socket's 26 x 31 mm flange looks the same turned half round). The rest are left out.
  3. ONE POSE FOR ALL OF THEM: a joint refinement in the base frame -- the same shrinking random
     search as the pipeline's, but each candidate is scored in every consistent view at once
     (its pose carried into each camera) and the mean kept. Different viewpoints pin different
     axes: depth along one camera's ray is across another's.
  The reply carries every view's pose and terms, which views were used, and the joint pose with
  its per-view terms (the gate is applied to their mean).

SELF-CONTAINED: numpy, torch, OpenCV, PyYAML and the repo's pose_pipeline -- never cable_assembly.
It uses the estimator's internals (Observation, _score, _so3_exp), which the repo may change.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml


def say(msg):
    print(f'[pose6d] {msg}', file=sys.stderr, flush=True)


def reply(obj):
    sys.stdout.write(json.dumps(obj) + '\n')
    sys.stdout.flush()


def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _metres(T_mm):
    T = np.array(T_mm, dtype=float)
    T[:3, 3] *= 1e-3
    return T


def rotation_gap_deg(R1, R2, symmetry='none'):
    """The angle between two model orientations, modulo the part's symmetry."""
    if symmetry == 'axis_z':
        return float(np.degrees(np.arccos(np.clip(R1[:, 2] @ R2[:, 2], -1.0, 1.0))))
    turns = {'z4': [0.0, 90.0, 180.0, 270.0], 'z2': [0.0, 180.0]}.get(symmetry, [0.0])
    return min(float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2 @ _rz(k)) - 1.0) / 2.0, -1.0, 1.0))))
               for k in turns)


class Service:
    def __init__(self, repo, device='cuda'):
        self.repo = Path(repo).expanduser().resolve()
        sys.path.insert(0, str(self.repo))
        import torch
        from pose_pipeline import estimator as E
        from pose_pipeline.camera import Frame
        from pose_pipeline.segmentation import Segmenter
        self.torch, self.E, self.Frame = torch, E, Frame
        with open(self.repo / 'pose_pipeline' / 'objects.yaml') as f:
            cfg = yaml.safe_load(f)
        self.objects = cfg['objects']
        self.models_dir = self.repo / cfg['models_dir']
        with open(self.models_dir / 'models_info.json') as f:
            self.info = json.load(f)
        self.device = device
        t0 = time.time()
        self.seg = Segmenter(device)
        self.est = {}
        say(f'models loaded in {time.time() - t0:.1f} s; objects {sorted(self.objects)}')

    def estimator(self, name):
        if name not in self.est:
            obj = self.objects[name]
            self.est[name] = self.E.PoseEstimator(self.models_dir / f"obj_{obj['obj_id']:06d}.ply", device=self.device)
        return self.est[name]

    def preload(self, names):
        """Load `names`' estimators (and their rivals') now, and run the detector, SAM2 and each
        estimator once on a blank view -- so the run's first request pays for neither the loading
        nor the GPU's first-call warm-up."""
        t0 = time.time()
        names = [n for n in names if n in self.objects]
        for n in list(names):
            names += [r for r in self.objects[n].get('rivals') or [] if r not in names]
        for n in names:
            self.estimator(n)
        try:
            K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
            color = np.full((480, 640, 3), 127, np.uint8)
            frame = self.Frame(color, np.zeros((480, 640), np.float32), K)
            box = [280.0, 200.0, 360.0, 280.0]
            mask = np.zeros((480, 640), bool)
            mask[210:270, 290:350] = True
            if names:
                self.seg.detect(color, self.objects[names[0]]['prompts'][:1])
            self.seg.segment(color, [box])
            R0 = self.torch.eye(3, device=self.device)[None]
            t0_mm = self.torch.tensor([[0.0, 0.0, 400.0]], device=self.device)
            for n in names:
                self.est[n].refine(frame, mask, R0, t0_mm, iters=2)
        except Exception as exc:                                    # noqa: BLE001 -- a warm-up only
            say(f'warm-up skipped: {type(exc).__name__}: {exc}')
        if names:
            say(f'preloaded {", ".join(names)} in {time.time() - t0:.1f} s')

    # ------------------------------------------------------------------ one view
    def _frame(self, path, use_depth=True):
        z = np.load(path)
        depth = z['depth_mm'].astype(np.float32)
        if not use_depth:
            depth = np.zeros_like(depth)
        return self.Frame(np.ascontiguousarray(z['color']), depth, np.asarray(z['K'], float))

    def model(self, req):
        """Points on the part's CAD surface (m, its frame), for drawing it."""
        pts = self.estimator(req['object']).model.pts.cpu().numpy()
        n = max(1, int(req.get('n', 800)))
        pick = np.random.default_rng(0).choice(len(pts), size=min(n, len(pts)), replace=False)
        return {'points': (pts[pick] / 1000.0).round(5).tolist()}

    def _prior_mask(self, frame, est, T_cam_obj_mm, pad=0.15):
        """SAM2 on the box the prior's projection fills, or None when it falls outside the image."""
        p = est.model.pts[::40].cpu().numpy() @ T_cam_obj_mm[:3, :3].T + T_cam_obj_mm[:3, 3]
        p = p[p[:, 2] > 1.0]
        if not len(p):
            return None
        uv = (p[:, :2] / p[:, 2:3]) * [frame.K[0, 0], frame.K[1, 1]] + [frame.K[0, 2], frame.K[1, 2]]
        h, w = frame.depth.shape
        (x0, y0), (x1, y1) = uv.min(axis=0), uv.max(axis=0)
        dx, dy = pad * (x1 - x0), pad * (y1 - y0)
        box = [max(0.0, x0 - dx), max(0.0, y0 - dy), min(w - 1.0, x1 + dx), min(h - 1.0, y1 + dy)]
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return None
        masks, _ = self.seg.segment(frame.color, [box])
        return (masks[0], box) if len(masks) and masks[0].sum() >= 50 else None

    def _view(self, name, frame, T_cam_obj_prior_mm, req):
        """(PoseResult, mask, box) for one view, or (None, None, None)."""
        est = self.estimator(name)
        torch = self.torch
        if T_cam_obj_prior_mm is not None:
            got = self._prior_mask(frame, est, T_cam_obj_prior_mm)
            if got is not None:
                mask, box = got
                R0 = torch.tensor(T_cam_obj_prior_mm[None, :3, :3], dtype=torch.float32, device=self.device)
                t0 = torch.tensor(T_cam_obj_prior_mm[None, :3, 3], dtype=torch.float32, device=self.device)
                return est.refine(frame, mask, R0, t0), mask, box
            if req.get('refine_only'):                 # tracking: out of view is no pose, not a search
                return None, None, None
        obj = self.objects[name]
        cands = self.seg.candidates(frame, obj['prompts'], model_size_mm=self.info[str(obj['obj_id'])]['diameter'],
                                    size_tol=(0.3, 1.2))
        cands = sorted(cands, key=lambda c: -c.det_score)[:int(req.get('max_candidates', 4))]
        if not cands:
            return None, None, None
        coarse = [est.coarse(frame, c.mask) for c in cands]
        refined = []
        for i in np.argsort([-c[0] for c in coarse])[:int(req.get('refine_top', 2))]:
            r = est.refine(frame, cands[i].mask, *coarse[i][1:])
            refined.append(((self._accept(r.terms, req), r.score), r, cands[i].mask, cands[i].box.tolist()))
        refined.sort(key=lambda x: x[0], reverse=True)
        for key, r, mask, box in refined:
            if not key[0]:
                break                                  # none left that passes the gate: the best, rejected
            rival = self._rival(name, frame, mask, r.score)
            if rival is None:
                return r, mask, box
            say(f'{name}: the candidate at {[round(v) for v in box]} is a {rival[0]} (score {rival[1]:.3f} '
                f'against {r.score:.3f}) -- left out')
        if refined[0][0][0]:                           # every one that passed is a rival's
            return None, None, None
        return refined[0][1], refined[0][2], refined[0][3]

    def _rival(self, name, frame, mask, score):
        """(rival, its score) for the first of the object's `rivals` (objects.yaml: other parts it
        can be mistaken for -- an RJ45's for a BNC plug's, both ~40 mm and dark at 0.4 m) whose
        model explains `mask` better than its own did with `score`; None if none does."""
        for other in self.objects[name].get('rivals') or []:
            r = self.estimator(other).estimate(frame, mask)
            if r.score > score:
                return other, r.score
        return None

    @staticmethod
    def _accept(terms, req):
        return (terms['iou'] >= req.get('min_iou', 0.75) and terms['edge_fwd'] >= req.get('min_edge', 0.5)
                and terms['edge_cover'] >= req.get('min_cover', 0.7))

    # ------------------------------------------------------------------ all views together
    def _joint(self, est, used, T_base_obj_mm, iters=40, n_samples=64):
        """The base-frame pose that scores best on average over `used` [(obs, T_cam_base_mm)]."""
        torch, E = self.torch, self.E
        dev = self.device
        Rcb = torch.tensor(np.stack([T[:3, :3] for _, T in used]), dtype=torch.float32, device=dev)
        tcb = torch.tensor(np.stack([T[:3, 3] for _, T in used]), dtype=torch.float32, device=dev)

        def score(Rb, tb):
            total, terms = 0.0, []
            for k, (obs, _) in enumerate(used):
                s, tm = est._score(obs, Rcb[k] @ Rb, (Rcb[k] @ tb[..., None])[..., 0] + tcb[k], est.sub_fine)
                total = total + s
                terms.append(tm)
            return total / len(used), terms

        R = torch.tensor(T_base_obj_mm[None, :3, :3], dtype=torch.float32, device=dev)
        t = torch.tensor(T_base_obj_mm[None, :3, 3], dtype=torch.float32, device=dev)
        s_cur, _ = score(R, t)
        s_start = float(s_cur[0])
        rot_sigma, trans_sigma = np.radians(4.0), 0.04 * est.model.radius
        c = est.model.center
        for it in range(iters):
            dR = E._so3_exp(torch.randn(n_samples, 3, device=dev) * rot_sigma)
            Rn = dR @ R
            tn = t + torch.randn(n_samples, 3, device=dev) * trans_sigma
            tn = tn + (R @ c)[0] - (dR @ (R @ c)[0][:, None])[..., 0]          # turn about the part's centre
            s, _ = score(Rn, tn)
            b = int(torch.argmax(s))
            if s[b] > s_cur[0]:
                R, t, s_cur = Rn[b:b + 1], tn[b:b + 1], s[b:b + 1]
            if it % 5 == 4:
                rot_sigma *= 0.7
                trans_sigma *= 0.7
        _, terms = score(R, t)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R[0].cpu().numpy(), t[0].cpu().numpy()
        per_view = [{k: float(v[0]) for k, v in tm.items()} for tm in terms]
        return T, float(s_cur[0]), s_start, per_view

    def estimate(self, req):
        torch = self.torch
        name = req['object']
        est = self.estimator(name)
        sym = req.get('symmetry', 'none')
        talk = say if not req.get('quiet') else (lambda msg: None)
        prior = req.get('prior')
        prior_mm = None
        if prior is not None:
            prior_mm = np.asarray(prior, float).copy()
            prior_mm[:3, 3] *= 1000.0
        views, kept = [], []
        with torch.no_grad():
            for k, v in enumerate(req['views']):
                t0 = time.time()
                frame = self._frame(v['npz'], bool(req.get('use_depth', True)))
                T_base_cam = np.asarray(v['T_base_cam'], float).copy()
                T_base_cam[:3, 3] *= 1000.0
                T_cam_base = np.linalg.inv(T_base_cam)
                r, mask, box = self._view(name, frame, None if prior_mm is None else T_cam_base @ prior_mm, req)
                if r is None:
                    views.append({'found': False, 'why': 'no candidate'})
                    talk(f'{name} view {k}: no candidate ({time.time() - t0:.1f} s)')
                    continue
                T_base_obj = T_base_cam @ r.T
                ok = self._accept(r.terms, req)
                views.append({'found': ok, 'score': r.score, 'terms': r.terms, 'box': box,
                              'T_base_obj': _metres(T_base_obj).tolist()})
                talk(f'{name} view {k}: score {r.score:.3f} {"accepted" if ok else "REJECTED"} '
                    f'{ {a: round(b, 3) for a, b in r.terms.items()} } ({time.time() - t0:.1f} s)')
                if ok:
                    side = self.E._mask_side(mask)
                    obs = self.E.Observation(frame, mask, scale=min(1.0, 160 / (side * 2.2)), device=self.device)
                    kept.append((k, T_base_obj, r.score, obs, T_cam_base))
            if not kept:
                return {'found': False, 'views': views, 'why': 'no view passed the gate'}
            ref = max(kept, key=lambda x: x[2])
            c_model = est.model.center.cpu().numpy()
            used = []
            for k, T, s, obs, Tcb in kept:
                d_deg = rotation_gap_deg(ref[1][:3, :3], T[:3, :3], sym)
                d_mm = float(np.linalg.norm((T[:3, :3] @ c_model + T[:3, 3]) - (ref[1][:3, :3] @ c_model + ref[1][:3, 3])))
                agree = d_deg <= req.get('consistent_deg', 10.0) and d_mm <= req.get('consistent_mm', 10.0)
                views[k].update(consistent=agree, gap_deg=d_deg, gap_mm=d_mm)
                if agree:
                    used.append((obs, Tcb))
                else:
                    talk(f'{name} view {k}: {d_deg:.1f} deg / {d_mm:.1f} mm from the best view -- left out')
            t0 = time.time()
            T, s, s0, per_view = self._joint(est, used, ref[1], iters=int(req.get('joint_iters', 40)))
        mean = {key: float(np.mean([p[key] for p in per_view])) for key in per_view[0]}
        found = self._accept(mean, req)
        T_m = _metres(T)
        talk(f'{name}: {len(used)} views jointly, score {s0:.3f} -> {s:.3f} ({time.time() - t0:.1f} s); '
            f'mean {({a: round(b, 3) for a, b in mean.items()})}')
        return {'found': found, 'T_base_obj': T_m.tolist(), 'score': s, 'score_best_view': s0, 'terms': mean,
                'per_view_terms': per_view, 'views': views, 'n_used': len(used)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repo', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--preload', default='', help='objects to load before ready, comma-separated')
    args = ap.parse_args()
    try:
        svc = Service(args.repo, args.device)
        svc.preload([n for n in args.preload.split(',') if n])
    except Exception as exc:                                        # told to the client, then raised
        reply({'ready': False, 'error': f'{type(exc).__name__}: {exc}'})
        raise
    reply({'ready': True, 'objects': sorted(svc.objects)})
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get('op') == 'quit':
            break
        t0 = time.time()
        try:
            ops = {'estimate': svc.estimate, 'model': svc.model}
            op = ops.get(req.get('op'))
            out = op(req) if op is not None else {'error': f"unknown op {req.get('op')!r}"}
        except Exception as exc:                                    # noqa: BLE001 -- one bad request
            traceback.print_exc(file=sys.stderr)
            out = {'found': False, 'error': f'{type(exc).__name__}: {exc}'}
        out.update(id=req.get('id'), elapsed_s=time.time() - t0)
        reply(out)


if __name__ == '__main__':
    main()
