"""bnc_assembly's axis probe (connector clocking, 2026-11-06): the socket's real axis found by turning with the
side load nulled -- the plug stays on the real axis while the wrist turns about the estimated one."""

import csv
import types

import numpy as np

from cable_assembly.apps.bnc_assembly import _ClockTrace, fit_axis_offset
from cable_assembly.transforms import inverse, translation_matrix, xyzrpy_to_matrix

B_TRUE = np.array([0.6, -3.5])          # the real axis off the estimated one (screw-axis y, z), mm: the 02:33 run's
D_TRUE = np.array([1.8, 1.6])           # the in-hand belief's error (connector y, z), mm


def Rx(th):
    return xyzrpy_to_matrix([0.0, 0.0, 0.0], [th, 0.0, 0.0])


def test_the_fit_finds_the_axis_and_the_hand_error_from_a_short_turn():
    rng = np.random.default_rng(0)
    th = np.radians(np.r_[np.linspace(0, 60, 80), np.linspace(60, 0, 80)])
    c, s = np.cos(th), np.sin(th)
    y = B_TRUE[0] - (c * D_TRUE[0] - s * D_TRUE[1]) + rng.normal(0, 0.15, th.size)
    z = B_TRUE[1] - (s * D_TRUE[0] + c * D_TRUE[1]) + rng.normal(0, 0.15, th.size)
    B, d, res, cond = fit_axis_offset(th, y, z)
    assert np.linalg.norm(B - B_TRUE) < 0.3 and np.linalg.norm(d - D_TRUE) < 0.3, (B, d)
    assert res.mean() < 0.3 and np.isfinite(cond)


def test_the_trace_of_a_plug_held_on_the_real_axis_gives_it_back():
    """The real geometry: a screw axis rolled -135 deg, the plug gripped at 70 deg -- the wrist turns about the
    real axis (the nulled plug stays on it); the trace samples where the BELIEF puts the connector."""
    T_axis0 = translation_matrix([0.807, 0.109, 0.268]) @ xyzrpy_to_matrix([0, 0, 0], np.radians([-135.0, -2.4, -4.4]))
    T_tc0 = xyzrpy_to_matrix([0.012, -0.004, 0.17], np.radians([0.0, -70.0, 0.0]))   # the belief, tool0 -> connector
    T_tc_true = T_tc0 @ translation_matrix([0.0, D_TRUE[0] / 1000, D_TRUE[1] / 1000])
    T_true = T_axis0 @ translation_matrix([0.0, B_TRUE[0] / 1000, B_TRUE[1] / 1000])
    pose = {'T': T_true @ inverse(T_tc_true)}
    robot = types.SimpleNamespace(tool0=lambda: pose['T'],
                                  arm=types.SimpleNamespace(wrench=lambda: np.zeros(6)))
    adm = types.SimpleNamespace(delta=np.zeros(6))
    import tempfile
    path = tempfile.mktemp(suffix='.csv')
    drawn = []
    trace = _ClockTrace(robot, adm, path, T_axis0, T_tc0, every=1, draw=lambda items, text: drawn.append(items))
    trace.phase = 'probe 1/2'
    for th in np.radians(np.r_[np.linspace(0, 60, 60), np.linspace(60, 0, 60)]):
        pose['T'] = T_true @ Rx(th) @ inverse(T_tc_true)          # the plug ON the real axis, turned by th
        trace.step()
    trace.close()
    P = np.array(trace.probe)
    assert np.isclose(np.degrees(np.ptp(P[:, 0])), 60.0, atol=0.1), 'theta: the roll about the screw axis'
    B, d, res, _ = fit_axis_offset(P[:, 0], P[:, 1], P[:, 2])
    assert np.allclose(B, B_TRUE, atol=1e-6) and np.allclose(d, D_TRUE, atol=1e-6) and res.max() < 1e-6
    # applied as connector_clocking applies it, the engaged pose IS the real one: about the real axis, belief fixed
    T_clk = T_axis0 @ translation_matrix([0.0, B[0] / 1000, B[1] / 1000])
    T_tc = T_tc0 @ translation_matrix([0.0, d[0] / 1000, d[1] / 1000])
    assert np.allclose(T_clk @ inverse(T_tc), T_true @ inverse(T_tc_true), atol=1e-9)
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 120 and rows[0]['phase'] == 'probe 1/2' and drawn, 'recorded and drawn'
    assert drawn[-1][0]['type'] == 'line' and drawn[-1][-1]['type'] == 'points'


def test_after_the_probe_the_search_is_checked_against_the_corrected_axis():
    T_axis0 = xyzrpy_to_matrix([0.8, 0.1, 0.27], np.radians([-135.0, 0, 0]))
    T_tc0 = xyzrpy_to_matrix([0.0, 0.0, 0.17], np.radians([0.0, -70.0, 0.0]))
    T_tc_true = T_tc0 @ translation_matrix([0.0, D_TRUE[0] / 1000, D_TRUE[1] / 1000])
    T_true = T_axis0 @ translation_matrix([0.0, B_TRUE[0] / 1000, B_TRUE[1] / 1000])
    pose = {'T': T_true @ inverse(T_tc_true)}
    robot = types.SimpleNamespace(tool0=lambda: pose['T'], arm=types.SimpleNamespace(wrench=lambda: np.zeros(6)))
    import tempfile
    trace = _ClockTrace(robot, types.SimpleNamespace(delta=np.zeros(6)), tempfile.mktemp(suffix='.csv'), T_axis0, T_tc0)
    trace.every = 1
    trace.check = (inverse(T_true), T_tc_true)            # corrected exactly
    trace.phase = 'leg 1/3'
    for th in np.radians(np.linspace(0, 240, 50)):
        pose['T'] = T_true @ Rx(th) @ inverse(T_tc_true)
        trace.step()
    trace.close()
    assert len(trace.off) == 50 and max(trace.off) < 1e-6, 'on the corrected axis all the way round'


def test_the_clocking_axis_is_levelled_onto_the_base_plane_keeping_its_point_and_heading():
    """connector_clocking.level_axis (2026-11-06): the 02:33 socket frame, pitched -2.4 deg by the 6D estimate --
    turned onto the base plane by the smallest rotation, about its own origin."""
    from cable_assembly.apps.bnc_assembly import level_axis
    T = xyzrpy_to_matrix([0.807, 0.109, 0.268], np.radians([-135.0, -2.4, -4.4]))
    L, tilt = level_axis(T)
    assert abs(tilt - 2.4) < 0.05
    assert abs(L[2, 0]) < 1e-12, 'its +X in the base plane'
    assert np.allclose(L[:3, 3], T[:3, 3]), 'the same point'
    heading = lambda M: np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    assert abs(heading(L) - heading(T)) < 1e-9, 'the same heading'
    turn = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ L[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    assert abs(turn - tilt) < 1e-6, 'the smallest rotation: no roll added about the axis'
    assert np.allclose(L[:3, :3] @ L[:3, :3].T, np.eye(3), atol=1e-12)
    # the screw about it starts where the arm stands: the belief reset re-derives the connector from the frame
    T_tool0 = T @ inverse(xyzrpy_to_matrix([0.0, 0.0, 0.17], np.radians([0.0, -70.0, 0.0])))
    T_tc = inverse(T_tool0) @ L
    assert np.allclose(L @ inverse(T_tc), T_tool0, atol=1e-12)


def test_an_axis_meant_to_be_off_the_plane_is_left_alone():
    from cable_assembly.apps.bnc_assembly import level_axis
    up = xyzrpy_to_matrix([0.6, 0.0, 0.1], np.radians([0.0, -80.0, 0.0]))     # a socket facing up
    L, tilt = level_axis(up, max_deg=15.0)
    assert L is up and abs(tilt - 80.0) < 1e-6
    flat = xyzrpy_to_matrix([0.8, 0.1, 0.27], np.radians([-135.0, 0.0, 3.0]))
    L, tilt = level_axis(flat)
    assert L is flat and tilt < 1e-6
