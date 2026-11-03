"""perception/cable_model.py and skills/cable_tracker.py: the connector fitted rigidly, the cable as
a Warp rod -- fitted to points of it, resting on whatever it lies on, carried with the connector
when the gripper holds it. Synthetic points with known answers."""

import os
import types

import numpy as np
import pytest

pytest.importorskip('warp')

from cable_assembly.perception.cable_model import (
    PLUG_NPZ,
    CableRod,
    ConnectorModel,
    HeightField,
)

BENCH = 0.0062
R = 0.0025


def _pose(origin, heading_deg, tilt_deg=0.0):
    a = np.radians(heading_deg)
    x = np.array([np.cos(np.radians(tilt_deg)) * np.cos(a), np.cos(np.radians(tilt_deg)) * np.sin(a),
                  np.sin(np.radians(tilt_deg))])
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, np.cross(x, y)], axis=1)
    T[:3, 3] = origin
    return T


def _seen_side(model, T, rng, noise=0.0004):
    """The model's surface points a camera above sees (the upper half), placed at T, with noise."""
    p = model.samples @ T[:3, :3].T + T[:3, 3]
    up = (model.samples[:, 1:] @ [0.0, 1.0]) > -0.002                       # the connector frame's z up
    p = p[up]
    return p + rng.normal(0, noise, p.shape)


@pytest.mark.parametrize('cad', [False, True])
def test_the_plug_is_fitted_from_a_nearby_pose(cad):
    if cad and not os.path.isfile(PLUG_NPZ):
        pytest.skip('no bnc_plug.npz (sim/scripts/export_plug_mesh.py makes it from McMaster CAD)')
    model = ConnectorModel(path=PLUG_NPZ if cad else '/nonexistent')
    assert model.from_cad == cad
    rng = np.random.default_rng(1)
    T_true = _pose([0.6, 0.1, BENCH + 0.0074], 80.0)
    pts = _seen_side(model, T_true, rng)
    T0 = T_true @ _pose([0.003, 0.0015, -0.001], 3.0, 2.0)                # 3 mm, ~3 deg off
    T, st = model.fit(pts, T0)
    a = T[:3, 0] @ T_true[:3, 0]
    off = T[:3, 3] - T_true[:3, 3]
    along = abs(off @ T_true[:3, 0])
    across = np.linalg.norm(off - (off @ T_true[:3, 0]) * T_true[:3, 0])
    assert np.degrees(np.arccos(min(a, 1.0))) < 1.0
    assert across < 0.0007
    assert along < (0.001 if cad else 0.003)                               # the CAD's steps pin it along
    assert st['rms_m'] < 0.001 and st['inliers'] > 0.8


def _bench_and_block():
    """A bench, and a 20 mm block on it under the middle of the cable."""
    xs, ys = np.meshgrid(np.arange(0.45, 0.95, 0.003), np.arange(-0.15, 0.35, 0.003))
    z = np.full(xs.shape, BENCH)
    block = (xs > 0.66) & (xs < 0.70) & (ys > -0.10) & (ys < 0.30)
    z[block] = BENCH + 0.020
    return np.stack([xs.ravel(), ys.ravel(), z.ravel()], axis=1)


def _cable_over_block():
    """The cable's axis: a straight run from the junction across the block, draped over it."""
    x = np.linspace(0.60, 0.80, 400)
    top = np.where((x > 0.66) & (x < 0.70), BENCH + 0.020 + R, BENCH + R)
    # a drape: between the bench and the block top, a 25 mm ramp each side
    for x0, x1 in ((0.635, 0.66), (0.70, 0.725)):
        m = (x >= x0) & (x <= x1)
        t = (x[m] - x0) / (x1 - x0)
        top[m] = BENCH + R + 0.020 * (t if x0 < 0.66 else 1 - t)
    return np.stack([x, np.full_like(x, 0.1), top], axis=1)


def test_the_rod_fits_a_cable_draped_over_a_block_and_rests_on_it():
    axis = _cable_over_block()
    rod = CableRod(axis[0], [1.0, 0.0, 0.0], length_m=0.20, segment_m=0.01)
    rod.set_environment(HeightField(_bench_and_block(), cell_m=0.004))
    rod.trace(axis)
    res = rod.fit(axis)
    x = rod.nodes()
    floor = HeightField(_bench_and_block(), cell_m=0.004).at(x[:, :2]) + R
    assert res < 0.0015
    assert np.all(x[:, 2] >= floor - 1e-4)                                 # nowhere through it
    assert np.max(x[:, 2]) > BENCH + 0.018                                 # over the block, no plane
    assert rod.length() == pytest.approx(0.20, rel=0.02)


def test_lifting_the_connector_lifts_the_cable_off_what_it_lay_on():
    axis = _cable_over_block()
    rod = CableRod(axis[0], [1.0, 0.0, 0.0], length_m=0.20, segment_m=0.01)
    field = HeightField(_bench_and_block(), cell_m=0.004)
    rod.set_environment(field)
    rod.trace(axis)
    rod.fit(axis)
    start = axis[0].copy()
    for k in range(150):
        a = min(k / 60.0, 1.0)
        rod.pin(start + [0.0, 0.0, 0.15 * a], (1 - a) * np.array([1.0, 0.0, 0.0]) + a * np.array([0.0, 0.0, -1.0]))
        rod.step(1.0 / 60.0, pull=False)
    x = rod.nodes()
    assert np.all(x[:, 2] >= field.at(x[:, :2]) + R - 1e-4)
    assert x[2, 2] > start[2] + 0.10                                       # hanging below the connector
    assert rod.length() == pytest.approx(0.20, rel=0.03)


class _Sent:
    def __init__(self):
        self.overlays = {}

    def send_overlay(self, name, items, text=()):
        self.overlays[name] = (items, list(text))


def test_the_tracker_carries_the_connector_while_the_gripper_holds_it():
    from cable_assembly import config as C
    from cable_assembly.skills.cable_tracker import CableTracker
    tip = np.eye(4)
    tip[2, 3] = 0.19
    robot = types.SimpleNamespace(T_tool0_fingertip=tip, arm=None, gripper=None)
    sent = _Sent()
    tr = CableTracker(C.load('bnc_assembly', ['cell=iiwa14_bench', 'cable_model.capture_hz=0.0001']), robot, None, sent)
    axis = _cable_over_block()
    T_conn = _pose(axis[0] + [0.0, 0.0, 0.005], 180.0)                    # the plug toward -x, cable +x
    con = _seen_side(ConnectorModel(path='/nonexistent'), T_conn, np.random.default_rng(2), 0.0)
    tr.fit(T_conn, con, axis + [0.0, 0.0, R], _bench_and_block())
    assert 'cable_model' in sent.overlays and tr.rod is not None
    fitted = tr.T_conn.copy()                                              # the plug model's fit
    tool = np.eye(4)
    tool[:3, :3] = np.diag([1.0, -1.0, -1.0])                              # pointing down
    tool[:3, 3] = T_conn[:3, 3] + [0.01, 0.0, 0.19]                       # the fingertip on the plug
    tr.update(now=1.0, tool=tool, counts=0)
    assert tr.T_tool0_conn is None                                         # open: not held
    tr.update(now=1.1, tool=tool, counts=220)
    assert tr.T_tool0_conn is not None                                     # closed on it: held
    lifted = tool.copy()
    lifted[2, 3] += 0.10
    for k in range(60):
        tr.update(now=1.2 + k / 30, tool=lifted, counts=220)
    np.testing.assert_allclose(tr.T_conn[:3, 3], fitted[:3, 3] + [0.0, 0.0, 0.10], atol=1e-9)
    assert tr.rod.nodes()[0, 2] == pytest.approx(fitted[2, 3] + 0.10)
    assert 'carried by the gripper' in sent.overlays['cable_model'][1][0]
    tr.update(now=4.0, tool=lifted, counts=0)
    assert tr.T_tool0_conn is None                                         # let go: left where it is
    tr.send()
    assert 'where it lies' in sent.overlays['cable_model'][1][0]
