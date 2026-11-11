"""Where the robot's own gripper is in the wrist camera's image -- learned from the saved views.

WHY. The camera is bolted to the gripper, so the open gripper covers the same pixels in every wrist
view. The 6D estimator's detector does not know that: on 2026-11-03 it took the 2F-85's left knuckle,
bottom-left of the connector view, for the BNC plug (det 0.21) and never proposed the real plug,
which lay against the right finger pad; with the gripper's pixels hidden it found the plug (0.63).
perception/pose6d.py sends this mask with every view of a part NOT in the hand; pose6d_server.py
hides it from the detector and cuts it out of every candidate.

LEARNED, NOT DRAWN: across hundreds of views of different scenes from different poses, the pixels
that never change are the gripper (the sphere envelope of robot/tool_geometry.py covers 37% of the
image, the space between the fingers and whatever lies there included; this, ~12%). Rebuild it after
the camera or the gripper moves on the mount:

    python -m cable_assembly.perception.gripper_mask configs/cells/iiwa14_bench.gripper_mask.png

It reads every data/pose6d/*/view_*.npz (the gripper OPEN in all of them: views are taken before a
pick) of the NEWEST CAMERA -- the newest view's image size and focal length: after a camera swap
(the iiwa's D415 for a D405, 2026-11-10) the older views are another camera's, its gripper elsewhere.
"""

import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _camera(z):
    """A saved view's camera, as far as the view can tell: its image size and focal length."""
    return z['rgb'].shape[:2], (round(float(z['K'][0, 0])) if 'K' in z.files else None)


def learn(paths, still=20.0, grow_px=4):
    """(H, W) bool from saved views (npz with 'rgb'), oldest first, those of the last one's camera
    (_camera): the pixels whose grey level varies less than `still` across them, in regions touching
    the image's bottom edge (where the gripper is), holes filled, grown by `grow_px`."""
    import cv2
    grey, shape = [], None
    cameras = {}
    for p in paths:
        with np.load(p) as z:
            cameras.setdefault(_camera(z), []).append(p)
    with np.load(paths[-1]) as z:
        newest = _camera(z)
    shape, paths = newest[0], cameras[newest]
    for p in paths:
        with np.load(p) as z:
            grey.append(cv2.cvtColor(z['rgb'], cv2.COLOR_RGB2GRAY).astype(np.float32))
    if len(grey) < 20:
        raise ValueError(f'{len(grey)} views of {shape} from the newest camera: too few to tell the '
                         'gripper from the scene')
    low = (np.stack(grey).std(0) < still).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _n, lab = cv2.connectedComponents(low)
    keep = np.isin(lab, [k for k in np.unique(lab[-1, :]) if k > 0])
    _n, out = cv2.connectedComponents((~keep).astype(np.uint8))
    border = set(np.unique(out[0, :])) | set(np.unique(out[:, 0])) | set(np.unique(out[:, -1]))
    mask = keep | (~np.isin(out, list(border)) & ~keep)
    k = 2 * int(grow_px) + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def load(path, shape):
    """The mask at `path` (a PNG, nonzero = gripper), resized to `shape` (H, W) -- or None."""
    if not path:
        return None
    import cv2
    path = path if os.path.isabs(path) else os.path.join(REPO, path)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != tuple(shape):
        m = cv2.resize(m, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
    return m > 0


def main(argv=None):
    import cv2
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    paths = sorted(glob.glob(os.path.join(REPO, 'data', 'pose6d', '*', 'view_*.npz')))
    mask = learn(paths)
    cv2.imwrite(argv[0], mask.astype(np.uint8) * 255)
    print(f'{argv[0]}: {mask.shape[1]}x{mask.shape[0]}, {100.0 * mask.mean():.1f}% gripper, from {len(paths)} views')
    return 0


if __name__ == '__main__':
    sys.exit(main())
