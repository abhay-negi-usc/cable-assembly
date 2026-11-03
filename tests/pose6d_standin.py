"""A stand-in for cable_assembly/perception/pose6d_server.py in tests: the same protocol, no models.

It answers with the poses in the JSON file $POSE6D_TRUTH ({"cord_end_pos_z": 4x4, "adapter": 4x4},
base frame, metres), each view's a little off (0.5 mm, 0.3 deg, by view), the joint pose the truth
itself. $POSE6D_TRUTH_CAM instead: the poses in each view's CAMERA frame -- a part that moves with
the camera, as one in the gripper does (the joint pose is then the first view's). $POSE6D_MISS_FIRST=1:
no candidate in a request's first view when it has no prior; $POSE6D_MISS_SINGLE=1: none in any
one-view request, prior or not (only views together find it). Every view's npz is opened and
checked, as the real one reads it. "model": points on a cylinder where each part is. Its ready line
says what --preload named ("preloaded")."""

import json
import os
import sys

import numpy as np


def main():
    assert '--device' in sys.argv
    in_cam = 'POSE6D_TRUTH_CAM' in os.environ
    with open(os.environ['POSE6D_TRUTH_CAM' if in_cam else 'POSE6D_TRUTH']) as f:
        truth = json.load(f)
    miss_first = os.environ.get('POSE6D_MISS_FIRST') == '1'
    miss_single = os.environ.get('POSE6D_MISS_SINGLE') == '1'
    pre = sys.argv[sys.argv.index('--preload') + 1].split(',') if '--preload' in sys.argv else []
    print(json.dumps({'ready': True, 'objects': sorted(truth), 'preloaded': [n for n in pre if n]}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('op') == 'quit':
            break
        if req.get('op') == 'model':
            z0, z1 = (0.10865, 0.1524) if req['object'] == 'cord_end_pos_z' else (-0.02, 0.012)
            a = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
            pts = [[0.006 * np.cos(t), 0.006 * np.sin(t), z] for z in np.linspace(z0, z1, 10) for t in a]
            print(json.dumps({'id': req['id'], 'points': pts[:int(req.get('n', 800))]}), flush=True)
            continue
        T = np.asarray(truth[req['object']], float)
        joint = None
        views = []
        for k, v in enumerate(req['views']):
            z = np.load(v['npz'])
            assert z['color'].ndim == 3 and z['depth_mm'].shape == z['color'].shape[:2] and z['K'].shape == (3, 3)
            assert np.asarray(v['T_base_cam']).shape == (4, 4)
            if (miss_first and k == 0 and req.get('prior') is None) or (miss_single and len(req['views']) == 1):
                views.append({'found': False, 'why': 'no candidate'})
                continue
            Tv = np.asarray(v['T_base_cam'], float) @ T if in_cam else T.copy()
            joint = Tv.copy() if joint is None else joint
            Tv[:3, 3] += 0.0005 * np.array([np.cos(k), np.sin(k), 0.3])
            views.append({'found': True, 'score': 2.2, 'terms': {'iou': 0.85, 'edge_fwd': 0.65, 'edge_cover': 0.95},
                          'T_base_obj': Tv.tolist(), 'consistent': True})
        found = any(v['found'] for v in views)
        out = {'id': req['id'], 'found': found, 'views': views, 'n_used': sum(v['found'] for v in views),
               'terms': {'iou': 0.85, 'edge_fwd': 0.65, 'edge_cover': 0.95}, 'score': 2.25, 'score_best_view': 2.2}
        if found:
            out['T_base_obj'] = (joint if in_cam else T).tolist()
        else:
            out['why'] = 'no view passed the gate'
        print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
