"""perception/engage_vision.py: the wrist camera's frames through the engage, and their replay with the CAD."""

import csv
import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np

from cable_assembly.perception import engage_vision as EV

K = np.array([[930.0, 0.0, 640.0], [0.0, 930.0, 360.0], [0.0, 0.0, 1.0]])


def cube_ply(path, half=0.01):
    """A 2 cm cube as the 6D models' PLY (binary little-endian, float xyz, uchar-count int faces)."""
    v = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)], '<f4')
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [(a, b, c) for q in quads for (a, b, c) in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    with open(path, 'wb') as fh:
        fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty float x\nproperty float y\n'
                 f'property float z\nelement face {len(tris)}\nproperty list uchar int vertex_indices\n'
                 'end_header\n'.encode())
        fh.write(v.tobytes())
        for t in tris:
            fh.write(np.array([3], 'u1').tobytes() + np.array(t, '<i4').tobytes())
    return path


def test_render_hides_what_is_behind():
    """The painter's order: a near cube hides the far one where they overlap; the masks say what is seen."""
    from cable_assembly.robot import tool_geometry
    import tempfile
    p = cube_ply(os.path.join(tempfile.mkdtemp(), 'cube.ply'))
    V, F = tool_geometry.read_ply(p)
    near, far = np.eye(4), np.eye(4)
    near[:3, 3], far[:3, 3] = [0.0, 0.0, 0.20], [0.02, 0.0, 0.30]    # far: half behind the near one
    img = np.full((720, 1280, 3), 200, np.uint8)
    out, (m_near, m_far) = EV.render(img, K, [(V, F, near, (255, 0, 255)), (V, F, far, (255, 200, 0))])
    assert m_near.sum() > 0 and m_far.sum() > 0
    assert not np.any(m_near & m_far), 'a pixel is one part or the other'
    _, (alone,) = EV.render(img, K, [(V, F, far, (255, 200, 0))])
    assert m_far.sum() < alone.sum(), 'the near cube hides part of the far one'
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_frames_are_recorded_with_the_state_and_replayed(tmp_path):
    p = cube_ply(str(tmp_path / 'cube.ply'))

    class Cam:
        n = 0

        def preview(self):
            Cam.n += 1
            return np.full((720, 1280, 3), 120 + Cam.n % 50, np.uint8)
    T_tool0 = np.eye(4)
    T_tool0[:3, 3] = [0.5, 0.1, 0.3]
    robot = SimpleNamespace(tool0=lambda: T_tool0, arm=SimpleNamespace(wrench=lambda: np.arange(6.0)))
    stage = ['contact']
    T_tool0_cam = np.eye(4)
    T_tool0_plug = np.eye(4)
    T_tool0_plug[:3, 3] = [0.0, 0.02, 0.2]
    T_base_socket = T_tool0 @ T_tool0_cam @ np.array([[1, 0, 0, 0.0], [0, 1, 0, -0.01], [0, 0, 1, 0.25], [0, 0, 0, 1]])
    meta = {'K': K, 'T_tool0_cam': T_tool0_cam, 'T_tool0_plug': T_tool0_plug, 'socket_pose': T_base_socket,
            'plug_cad': (p, 1.0), 'socket_cad': (p, 1.0)}
    rec = EV.EngageImageRecorder(Cam(), robot, str(tmp_path),
                                 lambda: {'stage': stage[0], 'depth_mm': -3.0, 'axial_n': 5.0, 'radial_n': 1.5},
                                 rate_hz=40.0, meta=meta).start()

    def advance():
        time.sleep(0.15)
        stage[0] = 'engage'
    th = threading.Thread(target=advance)
    th.start()
    time.sleep(0.4)
    rec.stop()
    th.join()
    d = tmp_path / EV.FRAMES_DIR
    rows = list(csv.DictReader(open(d / 'frames.csv')))
    assert len(rows) >= 6 and rec.n == len(rows)
    assert {r['stage'] for r in rows} == {'contact', 'engage'}
    assert float(rows[0]['flange_x_mm']) == 500.0 and float(rows[0]['axial_n']) == 5.0
    assert all((d / r['file']).exists() for r in rows)
    assert json.load(open(d / 'meta.json'))['plug_cad'][0] == p
    EV.replay(str(tmp_path))
    assert len(os.listdir(d / 'overlay')) == len(rows) and (d / 'overlay_sheet.jpg').exists()
