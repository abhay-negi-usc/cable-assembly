"""2D masks of the held USB plug (RGB only) for analysis/in_hand_rgb_usb.py -- run in the rag6dpose env:
    ~/miniconda3/envs/rag6dpose/bin/python analysis/in_hand_rgb_segment.py
Variants:  box   SAM2 on the prior's projected box, grown 60%
           dino  GroundingDINO ("usb plug", "usb connector") inside a crop round the prior, best box near it -> SAM2
           pts   SAM2 with the grown box + positive points along the prior's axis past the pads, negatives on the pads
"""
import sys
import numpy as np
import torch
sys.path.insert(0, '/home/lm-2023/abhay_ws/RAG-6DPose-code-abhay')
from pose_pipeline.segmentation import Segmenter
from PIL import Image

z = np.load('data/in_hand_rgb_usb/views.npz')
keys = sorted({k.split('__')[0] for k in z.files})
seg = Segmenter('cuda')
out = {}


@torch.no_grad()
def sam(rgb, box, pos=(), neg=()):
    img = Image.fromarray(rgb)
    kw = dict(images=img, input_boxes=[[list(map(float, box))]], return_tensors='pt')
    if len(pos) or len(neg):
        pts = [list(map(float, p)) for p in list(pos) + list(neg)]
        kw['input_points'] = [[pts]]
        kw['input_labels'] = [[[1] * len(pos) + [0] * len(neg)]]
    inp = seg.sam_proc(**kw).to('cuda')
    o = seg.sam(**inp, multimask_output=False)
    m = seg.sam_proc.post_process_masks(o.pred_masks.cpu(), inp['original_sizes'])[0]
    return m[0, 0].numpy().astype(bool), float(o.iou_scores.reshape(-1)[0])


for k in keys:
    rgb = z[f'{k}__rgb']
    b = z[f'{k}__prior_uv_box']
    c, h = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) / 2
    grown = np.r_[c - 1.6 * h, c + 1.6 * h]
    m, s = sam(rgb, grown)
    out[f'{k}__box'] = m
    # dino in a crop
    s_px = int(max(h) * 3)
    x0, y0 = max(0, int(c[0] - s_px)), max(0, int(c[1] - s_px))
    crop = np.ascontiguousarray(rgb[y0:int(c[1] + s_px), x0:int(c[0] + s_px)])
    boxes, scores = seg.detect(crop[:, :, ::-1], ['usb plug', 'usb connector'], threshold=0.15)
    if len(boxes):
        bc = (boxes[:, :2] + boxes[:, 2:]) / 2 + [x0, y0]
        i = int(np.argmin(np.linalg.norm(bc - c, axis=1) - 50 * scores))
        bb = boxes[i] + [x0, y0, x0, y0]
        out[f'{k}__dino'], _ = sam(rgb, bb)
        out[f'{k}__dino_box'] = bb
    # points along the prior axis: from the fingertip pixel toward the box centre and beyond
    f = z[f'{k}__ftip_uv']
    d = (c - f) / max(np.linalg.norm(c - f), 1)
    L = np.linalg.norm(c - f)
    pos = [f + d * L * t for t in (0.6, 1.0, 1.4)]
    nperp = np.array([-d[1], d[0]])
    neg = [f - d * L * 0.4 + nperp * L * 0.5, f - d * L * 0.4 - nperp * L * 0.5]
    out[f'{k}__pts'], _ = sam(rgb, grown, pos, neg)
    print(k, {v: int(out.get(f'{k}__{v}', np.zeros(1)).sum()) for v in ('box', 'dino', 'pts')}, flush=True)
np.savez_compressed('data/in_hand_rgb_usb/masks.npz', **out)
