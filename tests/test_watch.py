"""perception/depth.py and skills/watch.py: what the wrist camera's depth says about a pick -- where
the connector is, the cables' shape in 3D, and whether the grasp took the cable off the bench.
Synthetic frames with known answers: a camera 0.35 m above the bench looking straight down, a cable
lying on it, the depth exact."""

import importlib.util
import os
import time

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.perception import depth as D
from cable_assembly.perception.camera import Frame
from cable_assembly.robot import live_view
from cable_assembly.skills.watch import PickWatcher

BENCH_Z = 0.0062
CAM_Z = 0.35
# Straight down at (0.6, 0.15): optical x along base x, y along -y, z along -z.
T = np.array([[1.0, 0, 0, 0.6], [0, -1.0, 0, 0.15], [0, 0, -1.0, CAM_Z], [0, 0, 0, 1.0]])
K = np.array([[232.0, 0, 160.0], [0, 232.0, 90.0], [0, 0, 1.0]])
W, H = 320, 180
CABLE_TOP = 0.004                    # the cable's top, above the bench
ROWS, COLS = slice(85, 95), slice(40, 280)


def _frame(cable=True, covered=None):
    depth = np.full((H, W), CAM_Z - BENCH_Z, np.float32)
    mask = np.zeros((H, W), bool)
    if cable:
        depth[ROWS, COLS] = CAM_Z - BENCH_Z - CABLE_TOP
        mask[ROWS, COLS] = True
    if covered is not None:
        depth[covered] = 0.15
    return Frame(np.zeros((H, W, 3), np.uint8), K, np.zeros(5), 0.0, T, depth), mask


@pytest.fixture(scope='module')
def cfg():
    return C.load('bnc_assembly', ['cell=iiwa14_bench'])


class Sent:
    def __init__(self):
        self.overlays = {}

    def send_overlay(self, name, items, text=()):
        self.overlays[name] = (items, list(text))


# ---------------------------------------------------------------------------- depth.py

def test_a_pixel_lifts_to_the_point_it_sees_and_projects_back():
    frame, mask = _frame()
    p = D.lift(frame, (100, 90), mask=mask)
    np.testing.assert_allclose(p, [0.6 + (100 - 160) / 232 * (CAM_Z - BENCH_Z - CABLE_TOP), 0.15, BENCH_Z + CABLE_TOP], atol=1e-6)
    uv, z = D.project(frame, p[None])
    np.testing.assert_allclose(uv[0], [100, 90], atol=1e-6)
    assert D.lift(frame, (100, 40), mask=mask) is None              # no cable pixels near it
    hole = _frame()[0]
    hole.depth[80:100, 90:110] = 0.0
    assert D.lift(hole, (100, 90)) is None                           # no reading near it


def test_a_cables_centreline_comes_out_on_its_top_at_its_length():
    frame, mask = _frame()
    line = D.centreline_3d(frame, mask, step_px=8.0)
    np.testing.assert_allclose(line[:, 2], BENCH_Z + CABLE_TOP, atol=1e-5)
    true_len = (COLS.stop - 1 - COLS.start) / 232 * (CAM_Z - BENCH_Z - CABLE_TOP)
    length = np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1))
    assert length == pytest.approx(true_len, rel=0.08)
    assert np.all(np.abs(line[:, 1] - 0.15 + (90 - 90) / 232) < 0.01)  # along the strip's row


def test_the_spot_check_says_what_is_there():
    before, mask = _frame()
    p = D.lift(before, (160, 90), mask=mask)                          # the cable's top
    assert D.spot_check(before, p, BENCH_Z)[0] == 'still there'
    assert D.spot_check(_frame(cable=False)[0], p, BENCH_Z)[0] == 'gone'
    cover = np.zeros((H, W), bool)
    cover[70:110, 140:180] = True
    assert D.spot_check(_frame(covered=cover)[0], p, BENCH_Z)[0] == 'covered'
    far = p + [1.0, 0, 0]
    assert D.spot_check(before, far, BENCH_Z)[0] == 'unseen'


# ---------------------------------------------------------------------------- the watcher

def test_the_scan_finds_the_junction_by_depth_against_the_plane_estimate(cfg):
    frame, mask = _frame()
    sent = Sent()
    w = PickWatcher(cfg, None, None, sent)
    uv = (220.0, 90.0)
    ray = D.rays(frame, [uv])[0]
    p_plane = T[:3, 3] + (BENCH_Z - T[2, 3]) / ray[2] * ray       # the pick's estimate: on the bench
    w.on_scan(frame, [{'junction': (*uv, 0.0), 'cable_mask': mask, 'ends': []}],
              [{'junction': p_plane, 'heading': np.array([1.0, 0, 0])}])
    o = w.last['cables'][0]
    assert o['top_mm'] == pytest.approx(1000 * CABLE_TOP, abs=0.01)
    # The plane estimate sits where the ray meets the bench, CABLE_TOP further along it: off by
    # CABLE_TOP * tan(the ray's angle) horizontally.
    assert o['gap_xy_mm'] == pytest.approx(1000 * CABLE_TOP * (220 - 160) / 232, abs=0.01)
    assert o['line'] is not None
    assert set(sent.overlays) == {'connector', 'cables'}
    assert 'depth vs plane' in sent.overlays['connector'][1][0]


class Cam:
    def __init__(self, frames):
        self.frames = list(frames)

    def capture(self):
        return self.frames.pop(0)


@pytest.mark.parametrize('after, verdict', [(lambda: _frame(cable=False)[0], 'gone'),
                                            (lambda: _frame()[0], 'still there')])
def test_after_the_lift_it_reports_what_is_left(cfg, after, verdict):
    frame, mask = _frame()
    sent = Sent()
    w = PickWatcher(cfg, Cam([after()]), None, sent)
    w.detect_after = False
    uv = (160.0, 90.0)
    ray = D.rays(frame, [uv])[0]
    p_plane = T[:3, 3] + (BENCH_Z - T[2, 3]) / ray[2] * ray
    w.on_scan(frame, [{'junction': (*uv, 0.0), 'cable_mask': mask, 'ends': []}],
              [{'junction': p_plane, 'heading': np.array([1.0, 0, 0])}])
    assert w.after_lift(p_plane) == verdict
    assert w.verdicts == [verdict] and 'grasp' in sent.overlays


def test_off_or_blind_it_does_nothing(cfg):
    frame, mask = _frame()
    off = PickWatcher(C.load('bnc_assembly', ['cell=iiwa14_bench', 'watch.enabled=false']), None, None, Sent())
    off.on_scan(frame, [{'junction': (160.0, 90.0, 0.0), 'cable_mask': mask, 'ends': []}], [None])
    assert off.last is None and off.after_lift(np.zeros(3)) is None
    blind = PickWatcher(cfg, None, None, Sent())
    no_depth = Frame(frame.color, K, np.zeros(5), 0.0, T, None)
    blind.on_scan(no_depth, [{'junction': (160.0, 90.0, 0.0), 'cable_mask': mask, 'ends': []}], [None])
    assert blind.last is None


def test_the_scanner_tells_its_listeners_and_survives_a_bad_one(cfg):
    from cable_assembly.skills.ground_pick import GroundPlaneScanner

    class Det:
        def detect_all(self, frame, n):
            return []

    scanner = GroundPlaneScanner(None, None, Det(), None, cfg)
    heard = []
    scanner.listeners += [lambda *a: 1 / 0, lambda f, c, p: heard.append((f, c, p))]
    frame, _ = _frame()
    out = scanner._project_all([{'junction': (160.0, 90.0, 0.0), 'ends': []}], frame)
    assert heard == [(frame, [{'junction': (160.0, 90.0, 0.0), 'ends': []}], out)]


# ---------------------------------------------------------------------------- overlays on the wire

def test_an_overlay_reaches_the_viewer_between_frames():
    spec = importlib.util.spec_from_file_location(
        'live', os.path.join(os.path.dirname(__file__), '..', 'sim', 'src', 'cable_assembly_sim', 'live.py'))
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    frames, overlays = [], live.Overlays()
    rx = live.FrameReceiver('127.0.0.1', 0, frames.append, overlays.put)
    tx = live_view.FrameSender('iiwa14', np.eye(4), None, '127.0.0.1', rx.port)
    try:
        frame, _ = _frame()
        tx(frame)
        tx.send_overlay('grasp', [{'type': 'points', 'points': [[0.6, 0.15, 0.01]], 'color': [0, 1, 0], 'radius': 0.005}],
                        ['grasp (camera): picked'])
        t0 = time.monotonic()
        while (not frames or not overlays.snapshot()[0]) and time.monotonic() - t0 < 3.0:
            time.sleep(0.01)
        by_name, version = overlays.snapshot()
        assert len(frames) == 1 and version == 1
        assert by_name['grasp']['items'][0]['points'] == [[0.6, 0.15, 0.01]]
        assert overlays.text() == ['grasp (camera): picked']
    finally:
        tx.close()
        rx.close()
