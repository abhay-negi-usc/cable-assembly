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
     "consistent_deg": 10, "consistent_mm": 10, "joint_iters": 40, "use_depth": true, "refine_only": false, "quiet": false,
     "jaw": null or {"T_cam_tool": 4x4 (m), "slide_mm": 25, "pivot_deg": 30, "turn_deg": 20, "trim_mm": 6}}
    {"op": "model", "id": n, "object": "adapter", "n": 800}    -> {"points": (n, 3) m, the CAD frame}
    {"op": "quit"}

use_depth false: the views' depth is ignored (the silhouette's area sets the range). refine_only
true: a view where the prior's projection gives no mask (out of the image) has no pose -- no
search of the whole image (a tracker's request: ~40 s otherwise, and maybe another part). quiet true: no
progress lines for it (a tracker's, one after another, would bury the app's prompts). jaw (with a prior): the part
is HELD IN A PARALLEL JAW -- its pose is the prior moved only as the jaws let it (see _jaw_fit), not a free refinement.

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

import cv2
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
        self.sessions = {}                       # op session_view / session_joint: views taken so far
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
        """(Frame, robot_mask or None): robot_mask, sent by the client, is where the robot's own gripper is
        in the image (HxW bool) -- see _view."""
        z = np.load(path)
        depth = z['depth_mm'].astype(np.float32)
        if not use_depth:
            depth = np.zeros_like(depth)
        robot = z['robot_mask'].astype(bool) if 'robot_mask' in z.files else None
        return self.Frame(np.ascontiguousarray(z['color']), depth, np.asarray(z['K'], float)), robot

    def _hidden(self, frame, robot):
        """`frame` with the robot's pixels painted the image's median colour: what the detector sees.
        2026-11-03: the wrist camera's view of the gripper's own knuckle was taken for the BNC plug (det
        0.21) and the real plug, against the finger pad, never proposed; hidden, the plug is (0.63)."""
        if robot is None or not robot.any():
            return frame
        color = frame.color.copy()
        color[robot] = np.median(frame.color[~robot], axis=0).astype(np.uint8)
        return self.Frame(color, frame.depth, frame.K)

    @staticmethod
    def _off_robot(mask, robot, keep=0.5):
        """`mask` less the robot's pixels -- or None when that leaves under `keep` of it (or 50 px):
        then the candidate IS the robot."""
        if robot is None:
            return mask
        m = mask & ~robot
        return m if m.sum() >= max(50, keep * mask.sum()) else None

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

    # ------------------------------------------------------------------ a part held in a parallel jaw
    def _trim(self, est, frame, mask, T_cam_obj_mm, reach_mm):
        """`mask` within the outline of the model's box at `T_cam_obj_mm`, grown by `reach_mm` -- what lies beyond
        the model (the cord it hangs on) cut off."""
        pts = est.model.pts[::20].cpu().numpy()
        lo, hi = pts.min(axis=0) - reach_mm, pts.max(axis=0) + reach_mm
        K = frame.K
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        corners = corners @ T_cam_obj_mm[:3, :3].T + T_cam_obj_mm[:3, 3]
        if (corners[:, 2] <= 1.0).any():
            return mask
        uv = (corners[:, :2] / corners[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        region = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(region, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
        out = mask & (region > 0)
        return out if out.sum() >= max(50, 0.2 * mask.sum()) else mask

    def _jaw_fit(self, est, frame, mask, T_cam_obj_mm, jaw, round_part=False):
        """(PoseResult, the mask it was scored on) for a part HELD IN A PARALLEL JAW, from the prior.

        THE JAWS FIX 2 DOF (cable-assembly's jaw_constrained; the user, 2026-11-13: "the gripper only constrains
        2DOF"): two flat pads closing along the tool's y hold the part's middle on their mid-plane and its axis
        parallel to their faces. What it can do is SLIDE along the pads (tool x, z), PIVOT about the closing axis
        and TURN about its own axis (the model's z) -- and those four are all that is searched, from the prior (the
        grasp as commanded, which has the rest), on the estimator's own score: a grid over slide and pivot, then
        the turn, refined `levels` times about the best. A round part's turn shows in no view and is left alone.
        A free 6-DOF refinement of one near view spends its freedom on what the jaws do not allow (2026-11-13, the
        USB plug hand-loaded: it sat 14 mm short along the plug for a higher IoU against the mask).

        THE MASK IS THE PRIOR'S BOX, and a cord's plug brings its strain relief and cord with it -- the same black,
        not in the model. After the first fit the mask is cut to the model's outline there (+ trim_mm) and the fit
        finished on that: the terms the gate reads are of the part alone.

        jaw: T_cam_tool (4x4, m: the tool frame in the camera's -- the camera is on the flange, so one for every
        view), slide_mm, pivot_deg and turn_deg (the search's half-ranges; turn_deg 0: no turn), trim_mm."""
        torch, E = self.torch, self.E
        T_ct = np.asarray(jaw['T_cam_tool'], float).copy()
        T_ct[:3, 3] *= 1000.0
        T_tool_obj = np.linalg.inv(T_ct) @ T_cam_obj_mm
        c_obj = est.model.center.cpu().numpy()
        c = T_tool_obj[:3, :3] @ c_obj + T_tool_obj[:3, 3]
        n, levels = int(jaw.get('steps', 9)), int(jaw.get('levels', 4))

        def pose(dx, dz, th, roll):                           # turned about its axis and pivoted, both about its
            D, A = np.eye(4), np.eye(4)                       # centre, then slid
            D[:3, :3] = cv2.Rodrigues(np.array([0.0, th, 0.0]))[0]
            D[:3, 3] = c - D[:3, :3] @ c + [dx, 0.0, dz]
            A[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, roll]))[0]
            A[:3, 3] = c_obj - A[:3, :3] @ c_obj
            return T_ct @ D @ T_tool_obj @ A

        def scores(obs, P):
            Ts = np.stack([pose(*q) for q in P])
            R = torch.tensor(Ts[:, :3, :3], dtype=torch.float32, device=self.device)
            t = torch.tensor(Ts[:, :3, 3], dtype=torch.float32, device=self.device)
            return torch.cat([est._score(obs, R[i:i + 256], t[i:i + 256], est.sub_fine)[0] for i in range(0, len(P), 256)])

        def search(obs, best, span, levels):
            best = np.array(best, dtype=float)
            for _ in range(levels):
                g = [np.linspace(best[i] - span[i], best[i] + span[i], n) for i in range(3)]
                P = np.array(np.meshgrid(*g, indexing='ij')).reshape(3, -1).T
                P = np.c_[P, np.full(len(P), best[3])]
                best = P[int(torch.argmax(scores(obs, P)))]
                if span[3] > 0:                               # the turn, with the rest where it is
                    P = np.tile(best, (2 * n + 1, 1))
                    P[:, 3] = np.linspace(best[3] - span[3], best[3] + span[3], 2 * n + 1)
                    best = P[int(torch.argmax(scores(obs, P)))]
                span = span * 2.0 / (n - 1)                   # one step either side of the best
            return best, span

        def observe(m):
            return E.Observation(frame, m, scale=min(1.0, 240.0 / (E._mask_side(m) * 2.2)), device=self.device)

        slide = float(jaw.get('slide_mm', 25.0))
        span = np.array([slide, slide, np.radians(float(jaw.get('pivot_deg', 30.0))),
                         0.0 if round_part else np.radians(float(jaw.get('turn_deg', 20.0)))])
        best, span = search(observe(mask), np.zeros(4), span, levels)
        trim = float(jaw.get('trim_mm', 6.0))
        if trim > 0:
            mask = self._trim(est, frame, mask, pose(*best), trim)
            floor = np.array([2.0, 2.0, np.radians(3.0), 0.0 if span[3] == 0 else np.radians(3.0)])
            best, span = search(observe(mask), best, np.maximum(span * (n - 1) / 2.0, floor), 2)
        T = pose(*best)
        s, terms = est._score(observe(mask), torch.tensor(T[None, :3, :3], dtype=torch.float32, device=self.device),
                              torch.tensor(T[None, :3, 3], dtype=torch.float32, device=self.device), est.sub_fine)
        return E.PoseResult(T, float(s[0]), {k: float(v[0]) for k, v in terms.items()}), mask

    def _view(self, name, frame, T_cam_obj_prior_mm, req, robot=None):
        """(PoseResult, mask, box) for one view, or (None, None, None). `robot`: the robot's own pixels
        (HxW bool, or None) -- hidden from the detector and cut out of every candidate's mask."""
        est = self.estimator(name)
        torch = self.torch
        if T_cam_obj_prior_mm is not None:
            got = self._prior_mask(self._hidden(frame, robot), est, T_cam_obj_prior_mm)
            if got is not None and self._off_robot(got[0], robot) is not None:
                mask, box = self._off_robot(got[0], robot), got[1]
                if req.get('jaw'):
                    r, mask = self._jaw_fit(est, frame, mask, T_cam_obj_prior_mm, req['jaw'],
                                            round_part=req.get('symmetry') == 'axis_z')
                    return r, mask, box
                R0 = torch.tensor(T_cam_obj_prior_mm[None, :3, :3], dtype=torch.float32, device=self.device)
                t0 = torch.tensor(T_cam_obj_prior_mm[None, :3, 3], dtype=torch.float32, device=self.device)
                return est.refine(frame, mask, R0, t0), mask, box
            if req.get('refine_only'):                 # tracking: out of view is no pose, not a search
                return None, None, None
        obj = self.objects[name]
        cands = self.seg.candidates(self._hidden(frame, robot), obj['prompts'],
                                    model_size_mm=self.info[str(obj['obj_id'])]['diameter'],
                                    size_tol=tuple(req.get('size_tol', (0.3, 1.5))))
        for c in cands:
            c.mask = self._off_robot(c.mask, robot)
        cands = [c for c in cands if c.mask is not None]
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

    # ---- one estimate = each view refined (or searched), then one pose refined against them all.
    # Split in two so a client can send each view AS IT IS TAKEN (op 'session_view') while the arm
    # drives to the next, and ask for the joint pose once the last is in (op 'session_joint') -- the
    # orbit's moves and the estimator's work overlap (the user, 2026-11-04). op 'estimate' is the two
    # in one request, the same code: its results do not change.
    def _start(self, req):
        prior = req.get('prior')
        prior_mm = None
        if prior is not None:
            prior_mm = np.asarray(prior, float).copy()
            prior_mm[:3, 3] *= 1000.0
        return {'req': dict(req), 'name': req['object'], 'prior_mm': prior_mm, 'views': [], 'kept': []}

    def _add_view(self, ses, v):
        """One view into a session: refined from the prior (or searched), gated, kept if it passes."""
        req, name = ses['req'], ses['name']
        talk = say if not req.get('quiet') else (lambda msg: None)
        k = len(ses['views'])
        with self.torch.no_grad():
            t0 = time.time()
            frame, robot = self._frame(v['npz'], bool(req.get('use_depth', True)))
            T_base_cam = np.asarray(v['T_base_cam'], float).copy()
            T_base_cam[:3, 3] *= 1000.0
            T_cam_base = np.linalg.inv(T_base_cam)
            prior_mm = ses['prior_mm']
            r, mask, box = self._view(name, frame, None if prior_mm is None else T_cam_base @ prior_mm, req, robot)
            if r is None:
                ses['views'].append({'found': False, 'why': 'no candidate'})
                talk(f'{name} view {k}: no candidate ({time.time() - t0:.1f} s)')
                return ses['views'][-1]
            T_base_obj = T_base_cam @ r.T
            ok = self._accept(r.terms, req)
            ses['views'].append({'found': ok, 'score': r.score, 'terms': r.terms, 'box': box,
                                 'T_base_obj': _metres(T_base_obj).tolist()})
            talk(f'{name} view {k}: score {r.score:.3f} {"accepted" if ok else "REJECTED"} '
                 f'{ {a: round(b, 3) for a, b in r.terms.items()} } ({time.time() - t0:.1f} s)')
            if ok:
                side = self.E._mask_side(mask)
                obs = self.E.Observation(frame, mask, scale=min(1.0, 160 / (side * 2.2)), device=self.device)
                ses['kept'].append((k, T_base_obj, r.score, obs, T_cam_base))
        return ses['views'][-1]

    def _finish(self, ses):
        """The joint pose of a session's kept views (those consistent with the best)."""
        req, name, views, kept = ses['req'], ses['name'], ses['views'], ses['kept']
        talk = say if not req.get('quiet') else (lambda msg: None)
        if not kept:
            return {'found': False, 'views': views, 'why': 'no view passed the gate'}
        est = self.estimator(name)
        sym = req.get('symmetry', 'none')
        with self.torch.no_grad():
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
            if len(used) == 1:
                # ONE VIEW: the joint pass would refine it against itself (0.9-1.7 s for ~0.01 of score on
                # 2026-11-04) -- its own refined pose is the answer (the user, 2026-11-04).
                k1 = [k for k, *_ in kept if views[k].get('consistent')][0]
                v1 = views[k1]
                talk(f'{name}: 1 view -- its own fit, no joint pass')
                return {'found': self._accept(v1['terms'], req), 'T_base_obj': v1['T_base_obj'], 'score': v1['score'],
                        'score_best_view': v1['score'], 'terms': dict(v1['terms']),
                        'per_view_terms': [dict(v1['terms'])], 'views': views, 'n_used': 1}
            t0 = time.time()
            T, s, s0, per_view = self._joint(est, used, ref[1], iters=int(req.get('joint_iters', 40)))
        mean = {key: float(np.mean([p[key] for p in per_view])) for key in per_view[0]}
        found = self._accept(mean, req)
        T_m = _metres(T)
        talk(f'{name}: {len(used)} views jointly, score {s0:.3f} -> {s:.3f} ({time.time() - t0:.1f} s); '
             f'mean {({a: round(b, 3) for a, b in mean.items()})}')
        return {'found': found, 'T_base_obj': T_m.tolist(), 'score': s, 'score_best_view': s0, 'terms': mean,
                'per_view_terms': per_view, 'views': views, 'n_used': len(used)}

    def estimate(self, req):
        self.estimator(req['object'])
        ses = self._start(req)
        for v in req['views']:
            self._add_view(ses, v)
        return self._finish(ses)

    def session_view(self, req):
        """op 'session_view': {session, view, and -- with the first view of a session -- what 'estimate'
        takes but views}. The view's own reply."""
        sid = str(req['session'])
        if sid not in self.sessions:
            self.estimator(req['object'])
            self.sessions[sid] = self._start(req)
        return {'view': self._add_view(self.sessions[sid], req['view'])}

    def session_joint(self, req):
        """op 'session_joint': {session}. The joint reply, as 'estimate' gives; the session is dropped."""
        ses = self.sessions.pop(str(req['session']), None)
        if ses is None:
            return {'error': f"no session {req['session']!r}"}
        return self._finish(ses)


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
            ops = {'estimate': svc.estimate, 'model': svc.model, 'session_view': svc.session_view,
                   'session_joint': svc.session_joint}
            op = ops.get(req.get('op'))
            out = op(req) if op is not None else {'error': f"unknown op {req.get('op')!r}"}
        except Exception as exc:                                    # noqa: BLE001 -- one bad request
            traceback.print_exc(file=sys.stderr)
            out = {'found': False, 'error': f'{type(exc).__name__}: {exc}'}
        out.update(id=req.get('id'), elapsed_s=time.time() - t0)
        reply(out)


if __name__ == '__main__':
    main()
