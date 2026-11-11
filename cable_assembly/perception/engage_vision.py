"""ENGAGEMENT, SEEN BY THE WRIST CAMERA (the user, 2026-11-06: "capture images during the contact and engage
phase and explore ideas for vision based verification of engagement (and potentially also aligning)").

RECORDING. EngageImageRecorder takes the D415's newest colour image `rate_hz` times a second from a thread of its
own (camera.preview(): no wait on the servo loop, nothing taken while a capture is under way) for as long as the
engage runs -- contact, engage, the seat confirmation, the wiggle fade-out, the settle -- and keeps each frame
with what the run knew at that instant:

    <trial>/engage_frames/NNNN_<stage>.jpg
    <trial>/engage_frames/frames.csv    k, wall, stage, depth_mm (connector vs the target, + = in), axial_n,
                                        radial_n (in the connector frame), the flange pose (mm, rotvec), the wrench
    <trial>/engage_frames/meta.json     K, tool0 -> camera, the target, tool0 -> plug (the belief), the 6D
                                        socket pose, the parts and their CAD -- all a replay needs

Colour only: at the socket the plug's nose is ~0.15-0.25 m from the D415, inside its ~0.3 m depth minimum.

REPLAY. `python -m cable_assembly.perception.engage_vision <trial folder>` draws on every recorded frame the
socket's CAD where the 6D estimate put it and the plug's where the belief holds it, z-buffered against each other
(render()), with the stage and the forces -- <trial>/engage_frames/overlay/ and a contact sheet. Where the drawn
plug and socket sit on the real ones is what a vision check would measure; where they are hidden (by the plug
itself, by the fingers) is what it cannot.
"""

import csv
import json
import os
import threading
import time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

from .. import log as urlog

log = urlog.get('engage-vision')

FRAMES_DIR = 'engage_frames'
COLS = ['k', 'wall', 'mono_s', 'stage', 'depth_mm', 'axial_n', 'radial_n', 'flange_x_mm', 'flange_y_mm',
        'flange_z_mm', 'flange_rx', 'flange_ry', 'flange_rz', 'fx_n', 'fy_n', 'fz_n', 'tx_nm', 'ty_nm', 'tz_nm',
        'file']


def _jsonable(v):
    return np.asarray(v, dtype=float).tolist() if isinstance(v, np.ndarray) else v


class EngageImageRecorder:
    """Frames through the engage, each with the run's state (see the module docstring). start() / stop();
    `state_fn()` -> {'stage', 'depth_mm', 'axial_n', 'radial_n'}; never raises into the run. Stops by itself after
    `max_s` (an engage that ended without stopping it)."""

    def __init__(self, camera, robot, out_dir, state_fn, rate_hz=5.0, quality=90, max_s=120.0, meta=None):
        self.camera, self.robot, self.state_fn = camera, robot, state_fn
        self.dir = os.path.join(out_dir, FRAMES_DIR)
        self.period = 1.0 / max(float(rate_hz), 0.2)
        self.quality, self.max_s = int(quality), float(max_s)
        self.meta = dict(meta or {})
        self.n = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        meta = {k: _jsonable(v) for k, v in self.meta.items()}
        meta['started'] = datetime.now().isoformat(timespec='milliseconds')
        meta['rate_hz'] = 1.0 / self.period
        with open(os.path.join(self.dir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=1, default=str)
        self._thread = threading.Thread(target=self._run, name='engage-images', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        log.info('ENGAGE IMAGES: %d frame%s in %s (replay: python -m cable_assembly.perception.engage_vision %s).',
                 self.n, '' if self.n == 1 else 's', self.dir, os.path.dirname(self.dir))

    def _row(self, k, fname):
        st = dict(self.state_fn() or {})
        T = st.pop('tool0', None)
        if T is None:
            T = self.robot.tool0()
        try:
            w = np.asarray(self.robot.arm.wrench(), dtype=float)
        except Exception:                              # noqa: BLE001
            w = np.full(6, np.nan)
        rv = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_rotvec()
        now = time.time()
        return [k, datetime.fromtimestamp(now).strftime('%H:%M:%S.%f')[:-3], f'{time.monotonic():.4f}',
                st.get('stage', ''), f"{st.get('depth_mm', float('nan')):.3f}", f"{st.get('axial_n', float('nan')):.3f}",
                f"{st.get('radial_n', float('nan')):.3f}"] + [f'{1000 * v:.2f}' for v in np.asarray(T)[:3, 3]] + \
            [f'{v:.5f}' for v in rv] + [f'{v:.3f}' for v in w] + [fname]

    def _run(self):
        import cv2
        t0 = time.monotonic()
        path = os.path.join(self.dir, 'frames.csv')
        with open(path, 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(COLS)
            while not self._stop.is_set() and time.monotonic() - t0 < self.max_s:
                t_next = time.monotonic() + self.period
                try:
                    img = self.camera.preview()
                    if img is not None:
                        stage = str((self.state_fn() or {}).get('stage', '')) or 'engage'
                        fname = f'{self.n:04d}_{stage}.jpg'
                        row = self._row(self.n, fname)
                        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                        if ok:
                            with open(os.path.join(self.dir, fname), 'wb') as f:
                                f.write(jpg.tobytes())
                            wr.writerow(row)
                            fh.flush()
                            self.n += 1
                except Exception as exc:               # noqa: BLE001 -- a record, never a reason to stop the run
                    log.debug('engage image skipped (%s)', exc)
                self._stop.wait(max(0.0, t_next - time.monotonic()))


# ---------------------------------------------------------------------------- the CAD drawn into a view
_MESHES = {}


def load_mesh(path, scale):
    """(vertices m, triangles) of a PLY, read once."""
    key = (path, float(scale))
    if key not in _MESHES:
        from ..robot import tool_geometry
        v, f = tool_geometry.read_ply(path)
        _MESHES[key] = (np.asarray(v, dtype=float) * float(scale), np.asarray(f, dtype=np.int64))
    return _MESHES[key]


def render(img, K, parts, alpha=0.45, bins=96, outline=True):
    """`parts` = [(vertices m, triangles, T_cam_obj, (b, g, r))] drawn into `img` (BGR, a copy returned) with a
    painter's z-order -- triangles depth-sorted into `bins` layers, far to near, shaded by their facing -- so each
    part hides what lies behind it, the other parts included. Also returns the visible-pixel mask of each part
    (uint8), for what a vision check could see of it."""
    import cv2
    K = np.asarray(K, dtype=float)
    h, w = img.shape[:2]
    ids = np.zeros((h, w), np.uint8)
    shade = np.zeros((h, w), np.float32)
    tris, depth, pid, lum = [], [], [], []
    for i, (V, F, T, _col) in enumerate(parts):
        P = V @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
        tri = P[F]                                         # (T, 3, 3) in the camera frame
        z = tri[:, :, 2]
        keep = np.all(z > 0.02, axis=1)
        tri, z = tri[keep], z[keep]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        c = tri.mean(axis=1)
        facing = -np.einsum('ij,ij->i', n, c) / (np.linalg.norm(n, axis=1) * np.linalg.norm(c, axis=1) + 1e-12)
        uv = np.stack([K[0, 0] * tri[:, :, 0] / z + K[0, 2], K[1, 1] * tri[:, :, 1] / z + K[1, 2]], axis=2)
        tris.append(uv)
        depth.append(z.mean(axis=1))
        pid.append(np.full(len(z), i + 1, np.uint8))
        lum.append(0.35 + 0.65 * np.abs(facing))
    if not tris:
        return img.copy(), [np.zeros((h, w), np.uint8) for _ in parts]
    uv, d, pi, lu = np.concatenate(tris), np.concatenate(depth), np.concatenate(pid), np.concatenate(lum)
    order = np.argsort(-d)                                 # far first
    uv, d, pi, lu = uv[order], d[order], pi[order], lu[order]
    edges = np.linspace(0, len(d), bins + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        for p in np.unique(pi[a:b]):
            for lv in range(6):
                sel = (pi[a:b] == p) & (np.minimum((lu[a:b] * 6).astype(int), 5) == lv)
                if not np.any(sel):
                    continue
                polys = [np.round(t * 16).astype(np.int32) for t in uv[a:b][sel]]
                cv2.fillPoly(ids, polys, int(p), lineType=cv2.LINE_8, shift=4)
                cv2.fillPoly(shade, polys, (lv + 0.5) / 6.0, lineType=cv2.LINE_8, shift=4)
    out = img.copy().astype(np.float32)
    masks = []
    for i, (_V, _F, _T, col) in enumerate(parts):
        m = ids == i + 1
        masks.append(m.astype(np.uint8))
        colr = np.asarray(col, np.float32)[None, :] * (0.4 + 0.6 * shade[m][:, None])
        out[m] = (1.0 - alpha) * out[m] + alpha * colr
    out = np.clip(out, 0, 255).astype(np.uint8)
    if outline:
        for m, (_V, _F, _T, col) in zip(masks, parts):
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, tuple(int(c) for c in col), 1, cv2.LINE_AA)
    return out, masks


# ---------------------------------------------------------------------------- replay of a recorded engage
def _mat(rv_xyz_mm):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rv_xyz_mm[3:]).as_matrix()
    T[:3, 3] = np.asarray(rv_xyz_mm[:3]) / 1000.0
    return T


def replay(trial_dir, every=1, sheet_cols=6):
    """Every recorded frame with the socket's and the plug's CAD drawn on it -> engage_frames/overlay/."""
    import cv2
    d = os.path.join(trial_dir, FRAMES_DIR) if os.path.basename(trial_dir.rstrip('/')) != FRAMES_DIR else trial_dir
    meta = json.load(open(os.path.join(d, 'meta.json')))
    rows = list(csv.DictReader(open(os.path.join(d, 'frames.csv'))))
    K = np.asarray(meta['K'], dtype=float)
    T_tool0_cam = np.asarray(meta['T_tool0_cam'], dtype=float)
    T_tool0_plug = np.asarray(meta['T_tool0_plug'], dtype=float)
    T_base_socket = np.asarray(meta['socket_pose'], dtype=float) if meta.get('socket_pose') else None
    plug = load_mesh(*meta['plug_cad'])
    sock = load_mesh(*meta['socket_cad']) if meta.get('socket_cad') else None
    out_dir = os.path.join(d, 'overlay')
    os.makedirs(out_dir, exist_ok=True)
    thumbs = []
    for r in rows[::max(1, int(every))]:
        img = cv2.imread(os.path.join(d, r['file']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        T_base_tool0 = _mat([float(r[k]) for k in ('flange_x_mm', 'flange_y_mm', 'flange_z_mm',
                                                     'flange_rx', 'flange_ry', 'flange_rz')])
        T_cam_base = np.linalg.inv(T_base_tool0 @ T_tool0_cam)
        parts = [(plug[0], plug[1], np.linalg.inv(T_tool0_cam) @ T_tool0_plug, (255, 0, 255))]
        if sock is not None and T_base_socket is not None:
            parts.append((sock[0], sock[1], T_cam_base @ T_base_socket, (255, 200, 0)))
        out, _ = render(img, K, parts)
        txt = (f"{r['stage']}  depth {float(r['depth_mm']):+.2f} mm  axial {float(r['axial_n']):.1f} N  "
               f"radial {float(r['radial_n']):.1f} N  {r['wall']}")
        cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, r['file']), out)
        thumbs.append(cv2.resize(out, (out.shape[1] // 4, out.shape[0] // 4), interpolation=cv2.INTER_AREA))
    if thumbs:
        n = len(thumbs)
        rows_n = (n + sheet_cols - 1) // sheet_cols
        th, tw = thumbs[0].shape[:2]
        sheet = np.zeros((rows_n * th, sheet_cols * tw, 3), np.uint8)
        for i, t in enumerate(thumbs):
            sheet[(i // sheet_cols) * th:(i // sheet_cols + 1) * th, (i % sheet_cols) * tw:(i % sheet_cols + 1) * tw] = t
        cv2.imwrite(os.path.join(d, 'overlay_sheet.jpg'), sheet)
    print(f'{len(thumbs)} frames drawn: {out_dir} (sheet: {os.path.join(d, "overlay_sheet.jpg")}); '
          'magenta: the plug where the belief holds it, cyan: the socket where the 6D estimate put it.')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Draw the socket and plug CAD on a recorded engage\'s frames.')
    ap.add_argument('trial', help='a trial folder (with engage_frames/) or the engage_frames folder')
    ap.add_argument('--every', type=int, default=1, help='every Nth frame only')
    a = ap.parse_args(argv)
    replay(a.trial, a.every)


if __name__ == '__main__':
    main()
