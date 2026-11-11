"""skills/ground_pick.py DepthPlaneScanner and the 6D scanner's fallback to it (the user, 2026-11-04: when
the 6D estimator fails, SAM3 + the centreline trace + the junction on the ground plane from the depth)."""

from types import SimpleNamespace

import numpy as np

from cable_assembly import config as C
from cable_assembly.transforms import xyzrpy_to_matrix

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _bench_frame(z_bench, tilt_deg=0.0):
    """A camera 0.4 m over a bench at height z_bench (tilted about x), looking straight down."""
    T = xyzrpy_to_matrix([0.6, 0.0, 0.4], [np.pi, 0.0, 0.0])          # optical z down
    n = np.array([0.0, -np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))])
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 600.0, (v - 240) / 600.0, np.ones_like(u, float)], -1) @ T[:3, :3].T
    p0 = np.array([0.6, 0.0, z_bench])
    t = ((p0 - T[:3, 3]) @ n) / (rays @ n)
    return SimpleNamespace(depth=(t * 1.0).astype(np.float32), K=K, T_base_cam=T)   # depth = optical z (ray z = 1)


def _scanner(tmp_path):
    from cable_assembly.skills.ground_pick import DepthPlaneScanner
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    return DepthPlaneScanner(None, None, SimpleNamespace(detect_all=lambda *a, **k: []), None, cfg,
                             data_root=str(tmp_path), plane_cfg=cfg.get_path('pose6d.fallback.plane'))


def test_the_bench_plane_is_fitted_from_the_depth(tmp_path):
    sc = _scanner(tmp_path)
    fr = _bench_frame(0.012, tilt_deg=2.0)                              # 6 mm off the calibrated, tilted
    n, d, k = sc.fit_plane(fr)
    assert k > 2000 and abs(np.degrees(np.arccos(n[2])) - 2.0) < 0.1
    assert abs(d / n[2] - 0.012) < 0.0005                                # its height under the base
    P = sc._project((320.0, 240.0), fr)                                  # straight down from the camera
    assert abs(P[2] - (0.012 - 0.0 * np.tan(np.radians(2.0)))) < 0.001 and abs(P[0] - 0.6) < 1e-6
    T = sc._pose({'junction': P, 'heading': np.array([1.0, 0.0, 0.0])})
    np.testing.assert_allclose(T[:3, 2], n, atol=1e-9)                   # z = the plane's normal
    np.testing.assert_allclose(T[:3, 3], P + sc.axis_up * n, atol=1e-9)  # raised to the axis


def test_no_depth_falls_back_to_the_calibrated_plane(tmp_path):
    sc = _scanner(tmp_path)
    fr = SimpleNamespace(depth=None, K=K, T_base_cam=xyzrpy_to_matrix([0.6, 0.0, 0.4], [np.pi, 0.0, 0.0]))
    P = sc._project((320.0, 240.0), fr)
    assert abs(P[2] - sc.plane_z) < 1e-9


def test_the_6d_scanner_falls_back_when_the_estimator_finds_nothing(monkeypatch, tmp_path):
    # (off on the bench since 2026-11-04 -- "only use RAG 6D pose estimation"; switched on here)
    from cable_assembly.skills import pose6d_pick as PP
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', f'data_dir={tmp_path}',
                                  'pose6d.fallback.enabled=true', 'pose6d.retry_views=[]'])
    robot = SimpleNamespace(arm=SimpleNamespace(dry_run=True, q=lambda: np.zeros(7), fk=lambda q: np.eye(4),
                                                move_l=lambda *a, **k: True))
    sc = PP.Pose6DScanner(robot, None, None, cfg, data_root=str(tmp_path))
    monkeypatch.setattr(PP, 'estimate_part', lambda *a, **k: (None, []))
    monkeypatch.setattr(PP, 'close_cfg', lambda cfg: {'enabled': False})
    T_fb = xyzrpy_to_matrix([0.59, 0.02, 0.013], [0.0, 0.0, 0.7])
    # GPU memory: the 6D server stopped before SAM3 loads, SAM3 dropped after the scan, the server back
    calls = []
    from cable_assembly.perception import pose6d as P6
    from cable_assembly.perception import sam3
    from cable_assembly.skills import ground_pick
    det = SimpleNamespace(core='model', detector='head')
    monkeypatch.setattr(P6, 'shutdown', lambda: calls.append('6d stopped'))
    monkeypatch.setattr(P6, 'preload', lambda cfg: calls.append('6d restarted'))
    monkeypatch.setattr(sam3, 'make_detector', lambda cfg: calls.append('sam3 loaded') or det)

    class FakeScanner:
        def __init__(self, *a, **k):
            pass

        def scan(self, confirm=None):
            calls.append('scanned')
            return T_fb
    monkeypatch.setattr(ground_pick, 'DepthPlaneScanner', FakeScanner)
    T = sc.scan()
    np.testing.assert_allclose(T, T_fb)
    assert sc.used_fallback and sc.last is None
    assert calls == ['6d stopped', 'sam3 loaded', 'scanned', '6d restarted']
    assert det.core is None and det.detector is None                      # SAM3 let go of

    # SAM3 does not load (out of memory): no pose, and the 6D server is still started again
    calls.clear()
    monkeypatch.setattr(sam3, 'make_detector', lambda cfg: (_ for _ in ()).throw(RuntimeError('CUDA out of memory')))
    assert sc.scan() is None and calls == ['6d stopped', '6d restarted']


def test_a_near_miss_is_looked_at_closer_before_any_fallback(monkeypatch, tmp_path):
    # 2026-11-04 13:53: the plug against the finger, its fit in the right place but IoU 0.69 < 0.75.
    # The close view centred on that candidate decides -- SAM3 is not loaded when it passes.
    from cable_assembly.skills import pose6d_pick as PP
    from cable_assembly.perception import sam3
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', f'data_dir={tmp_path}'])
    T_fit = xyzrpy_to_matrix([0.57, 0.07, 0.008], [0.0, 1.45, 1.2])
    T_close = T_fit @ xyzrpy_to_matrix([0.0, 0.0, 0.003], [0.0, 0.0, 0.0])
    seen = {}

    def fake_estimate_part(*a, hints=None, **k):
        hints.append((T_fit, {'score': 2.17, 'terms': {'iou': 0.69, 'edge_fwd': 0.70, 'edge_cover': 1.0}}))
        return None, []

    def fake_close(cfg_, robot, camera, part, prior, sc, label, max_shift=None):
        seen['prior'], seen['max_shift'] = prior, max_shift
        return {'found': True, 'T_base_obj': T_close, 'score': 2.4, 'terms': {'iou': 0.86}}, []
    monkeypatch.setattr(PP, 'estimate_part', fake_estimate_part)
    monkeypatch.setattr(PP, 'servo_close_view', fake_close)
    monkeypatch.setattr(sam3, 'make_detector', lambda cfg: (_ for _ in ()).throw(AssertionError('SAM3 loaded')))
    arm = SimpleNamespace(dry_run=True, q=lambda: np.zeros(7), fk=lambda q: np.eye(4), move_l=lambda *a, **k: True)
    sc = PP.Pose6DScanner(SimpleNamespace(arm=arm), None, None, cfg, data_root=str(tmp_path))
    monkeypatch.setattr(sc, '_draw', lambda *a, **k: None)
    T = sc.scan()
    np.testing.assert_allclose(seen['prior'], T_fit)
    assert T is not None and not sc.used_fallback
    from cable_assembly.perception import pose6d as P6
    np.testing.assert_allclose(T, P6.junction_from_plug(sc._rest_on_plane({'T_base_obj': T_close})['T_base_obj']),
                               atol=1e-9)


def test_not_found_the_camera_is_moved_in_x_and_y_until_it_is_and_comes_back(monkeypatch, tmp_path):
    # the user, 2026-11-04: "only use RAG 6D pose estimation, if it does not work then perturb the camera in
    # the x-y plane by +/- 5 cm". The first offset out of reach is skipped; found from the next; back home.
    from cable_assembly.skills import pose6d_pick as PP
    from cable_assembly.perception import sam3
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', f'data_dir={tmp_path}',
                                  'pose6d.close_view.enabled=false'])
    assert not cfg.get_path('pose6d.fallback.enabled')
    T_home = xyzrpy_to_matrix([0.47, 0.15, 0.39], [3.0, 0.4, 0.2])
    state = {'T': T_home.copy()}
    moves = []

    def move_l(T, label='', caps=None):
        moves.append(np.asarray(T).copy())
        if abs(T[0, 3] - T_home[0, 3] + 0.05) < 1e-9:          # -50 mm in x: out of reach
            return False
        state['T'] = np.asarray(T).copy()
        return True
    arm = SimpleNamespace(dry_run=True, q=lambda: state['T'], fk=lambda q: q, move_l=move_l)
    T_plug = xyzrpy_to_matrix([0.6, 0.0, 0.01], [0.0, 1.5, 0.3])
    tries = []

    def fake_estimate_part(cfg_, robot, camera, part, confirm=None, label=None, hints=None, **k):
        tries.append(tuple(np.round(state['T'][:2, 3] - T_home[:2, 3], 3)))
        if len(tries) < 3:                                      # the scan pose and +50 x: nothing
            return None, []
        return {'found': True, 'T_base_obj': T_plug, 'score': 2.4, 'terms': {'iou': 0.86}}, []
    monkeypatch.setattr(PP, 'estimate_part', fake_estimate_part)
    monkeypatch.setattr(sam3, 'make_detector', lambda cfg: (_ for _ in ()).throw(AssertionError('SAM3 loaded')))
    sc = PP.Pose6DScanner(SimpleNamespace(arm=arm), None, None, cfg, data_root=str(tmp_path))
    monkeypatch.setattr(sc, '_draw', lambda *a, **k: None)
    T = sc.scan()
    assert T is not None
    assert tries == [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05)]     # -50 x skipped (unreachable), found at +50 y
    np.testing.assert_allclose(state['T'], T_home)              # back at the scan pose for the pick

    # found nowhere: no pose, SAM3 never loaded, still back home
    tries.clear()
    state['T'] = T_home.copy()
    monkeypatch.setattr(PP, 'estimate_part', lambda *a, **k: (tries.append(1), (None, []))[1])
    assert sc.scan() is None and len(tries) == 4                # the scan pose + 3 reachable moved views
    np.testing.assert_allclose(state['T'], T_home)
