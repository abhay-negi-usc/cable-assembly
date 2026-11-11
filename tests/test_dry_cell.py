"""apps/_dry_cell.py: what a --dry-run stands in for (the 6D estimates, the part in the jaws, contact), so
ethernet_assembly runs end to end with no camera -- the 2026-11-06 11:26 dry run stopped at its first socket view
("a view without depth")."""

import csv
import glob
import os
import sys
import types

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.apps import _dry_cell as D
from cable_assembly.perception import pose6d as P6
from cable_assembly.transforms import translation_matrix, xyzrpy_to_matrix


def _cfg(*over):
    return C.load('ethernet_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', *over])


class _Arm:
    """tcp_pose() = the tool where we put it."""
    def __init__(self, T=None):
        self.T = np.eye(4) if T is None else T

    def tcp_pose(self):
        return self.T


class _Grip:
    def __init__(self, counts=205):
        self.counts = counts

    def position(self):
        return self.counts


def _jack(*over, counts=205):
    """A jack at the origin (x into it), the plug's datum on the flange itself."""
    j = D.VirtualJack(_cfg(*over), _Grip(counts))
    j.target(np.eye(4))
    j.hold(np.eye(4))
    return j


def _at(x_mm, y_mm=0.0, z_mm=0.0):
    return _Arm(translation_matrix([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]))


def test_enabled_only_in_a_dry_run_that_asks_for_it():
    assert D.enabled(_cfg())
    assert not D.enabled(C.load('ethernet_assembly', ['cell=iiwa14_bench']))            # the arm
    assert not D.enabled(C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true']))   # not asked


def test_the_jack_resists_inside_floors_at_the_seat_and_latches():
    j = _jack()
    assert np.allclose(j(_at(-15.0)), 0.0), 'outside: free'
    w = j(_at(-11.0))                                         # through the opening, square to it
    assert j.entered and w[0] == pytest.approx(-j.f_ins)      # pushes back out (-x) by the insertion force
    assert not j.latched
    assert j(_at(0.5))[0] == pytest.approx(-(j.k_floor * 0.0005)), 'on the floor: the floor, the latch caught'
    assert j.latched
    f_hold = j(_at(-1.5))[0]                                  # pulled back: the latch holds it in (+x)
    assert f_hold == pytest.approx(j.k_latch * 0.001)
    assert np.allclose(j(_at(-1.5, 0.8))[1], -j.k_side * 0.0005), 'the side walls past the clearance'


def test_the_face_blocks_a_plug_beside_the_opening_and_an_unlatched_one_comes_out():
    j = _jack()
    w = j(_at(-11.0, 2.0))                                    # 2 mm off: on the face
    assert not j.entered and w[0] == pytest.approx(-j.k_wall * 0.0011)
    j = _jack('dry_cell.latches=false')
    j(_at(-11.0))
    j(_at(0.5))
    assert not j.latched and j(_at(-5.0))[0] == pytest.approx(-j.f_ins), 'nothing holds it in'
    assert np.allclose(j(_at(-20.0)), 0.0) and not j.entered


def test_released_the_plug_pushes_on_nothing():
    j = _jack(counts=3)                                       # the gripper open
    assert np.allclose(j(_at(0.5)), 0.0)


def test_on_the_kukas_spring_the_arm_settles_where_spring_and_jack_balance():
    j = _jack()
    j.K = 3000.0
    j(_at(-11.0))                                             # in
    T = j.deflect(None, translation_matrix([0.006, 0.0, 0.0]))   # commanded 6 mm past the floor
    x = T[0, 3]
    assert 0.0 < x < 0.006
    assert j.K * (0.006 - x) == pytest.approx(j._axial(x), rel=1e-6), 'K (commanded - actual) = the jack at actual'
    T = j.deflect(None, translation_matrix([-0.030, 0.0, 0.0]))  # out of the jack: on the command
    assert np.allclose(T[:3, 3], [-0.030, 0.0, 0.0])


def test_the_socket_stands_in_from_its_own_measurement_else_the_recorded_mate(tmp_path):
    cfg = _cfg(f'data_dir={tmp_path}')
    T_jc = translation_matrix([0.03444, 0.0, 0.0])
    mate = xyzrpy_to_matrix([0.8, 0.1, 0.26], [0.0, 0.0, 0.0])
    T, src = D.socket_pose(cfg, P6.RJ45_SOCKET, {'bnc_holder': mate}, 'rj45_connector', T_jc)
    assert 'recorded mate' in src
    np.testing.assert_allclose(P6.target_from_socket(T, T_jc, socket=P6.RJ45_SOCKET), mate, atol=1e-9)
    os.makedirs(tmp_path / 'pose6d')
    import json
    S = xyzrpy_to_matrix([0.7, 0.0, 0.2], [0.0, 0.5, 0.0])
    with open(tmp_path / 'pose6d' / 'last_socket.json', 'w') as fh:
        json.dump({'part': P6.SOCKET, 'T_base_obj': S.tolist()}, fh)
    assert 'recorded mate' in D.socket_pose(cfg, P6.RJ45_SOCKET, {'m': mate}, 'x', T_jc)[1], "not the BNC's"
    with open(tmp_path / 'pose6d' / 'last_socket.json', 'w') as fh:
        json.dump({'part': P6.RJ45_SOCKET, 'T_base_obj': S.tolist()}, fh)
    T, src = D.socket_pose(cfg, P6.RJ45_SOCKET, {'m': mate}, 'x', T_jc)
    assert 'last measurement' in src and np.allclose(T, S)
    assert D.socket_pose(cfg, P6.RJ45_SOCKET, {}, 'x', T_jc)[0] is not None
    assert D.socket_pose(_cfg(f'data_dir={tmp_path / "none"}'), P6.RJ45_SOCKET, {}, 'x', T_jc)[0] is None


def test_the_dry_scanner_gives_what_the_6d_scanner_gives():
    T_jc = translation_matrix([0.03444, 0.0, 0.0])
    T_lying = xyzrpy_to_matrix([0.59, 0.07, 0.013], np.radians([0.0, 0.0, 75.0]))
    s = D.DryScanner(_cfg('dry_cell.plug.latch_deg=180'), T_jc, lambda: T_lying)
    T = s.scan()
    assert T[2, 2] > 0.999, 'the levelled junction: grasped from above'
    assert abs(abs(P6.turn_deg(s.turn)) - 180.0) < 1e-6
    np.testing.assert_allclose(T @ T_jc, T_lying, atol=1e-9)


def test_a_dry_run_close_stalls_on_the_part():
    from cable_assembly.robot.gripper import Robotiq2F85
    g = Robotiq2F85.__new__(Robotiq2F85)
    g.dry_run, g._sim_pos, g.last_counts = True, 3, 3
    g.go_to(228, 'grasp')
    assert g.position() == 228, 'by default: shut on nothing'
    g.dry_run_object_counts = 205
    g.go_to(228, 'grasp')
    assert g.position() == 205
    g.go_to(3, 'open')
    assert g.position() == 3


def _run(tmp_path, monkeypatch, app, *sets):
    monkeypatch.setattr(sys, 'argv', ['x', '--cell', 'iiwa14_bench', '--dry-run', '--no-prompts',
                                      '--set', f'data_dir={tmp_path}', '--set', 'live_view.enabled=false', *sets])
    monkeypatch.setattr(sys, 'stdin', types.SimpleNamespace(readline=lambda: '', read=lambda *a: ''))
    with pytest.raises(SystemExit) as ex:
        app.main()
    rows = list(csv.DictReader(open(glob.glob(str(tmp_path / 'experiments' / 'ethernet_assembly_*' / 'trials.csv'))[0])))
    return ex.value.code, rows


@pytest.mark.parametrize('impedance', [False, True])
def test_the_ethernet_dry_run_assembles_end_to_end(tmp_path, monkeypatch, impedance):
    pytest.importorskip('cumotion')
    from cable_assembly.apps import ethernet_assembly, ethernet_assembly_impedance
    code, rows = _run(tmp_path, monkeypatch, ethernet_assembly_impedance if impedance else ethernet_assembly,
                      '--set', 'dry_cell.plug.latch_deg=180')
    assert code == 0, rows
    assert rows and 'assembled, latch tug-verified' in str(rows[0]), rows
