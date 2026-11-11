"""ethernet_assembly: bnc_assembly's pipeline for an RJ45 (a latch task) -- its config (extends, over_cell),
the cut it enforces, the plug's latch turn in the grasp, and where its target is."""

import logging
import os
import textwrap

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.perception import pose6d as P6
from cable_assembly.transforms import from_cfg, inverse, xyzrpy_to_matrix


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(textwrap.dedent(text))
    return str(path)


# ------------------------------------------------------------------ the config loader
def test_extends_deep_merges_the_variant_over_its_base(tmp_path):
    _write(tmp_path / 'base.yaml', """
        a: {b: 1, c: [1, 2], d: {e: 5, f: 6}}
        keep: yes
        gone: 3
        """)
    child = _write(tmp_path / 'child.yaml', """
        extends: base.yaml
        a: {c: [9], d: {e: 50}}
        gone: null
        new: 7
        """)
    cfg = C.load(child)
    assert cfg['a'] == {'b': 1, 'c': [9], 'd': {'e': 50, 'f': 6}}, 'nested keys merge; a list replaces the whole'
    assert cfg['keep'] is True and cfg['gone'] is None and cfg['new'] == 7
    assert 'extends' not in cfg
    assert cfg['_config_path'] == os.path.abspath(child)
    assert cfg['_config_bases'] == [os.path.abspath(tmp_path / 'base.yaml')]


def test_extends_refuses_a_loop_and_a_missing_base(tmp_path):
    _write(tmp_path / 'x.yaml', 'extends: y\n')
    _write(tmp_path / 'y.yaml', 'extends: x.yaml\n')
    with pytest.raises(ValueError, match='loops'):
        C.load(str(tmp_path / 'x.yaml'))
    _write(tmp_path / 'z.yaml', 'extends: nowhere.yaml\n')
    with pytest.raises(FileNotFoundError):
        C.load(str(tmp_path / 'z.yaml'))


def test_over_cell_beats_the_cell_and_set_beats_both(tmp_path):
    _write(tmp_path / 'cells' / 'bench.yaml', 'k: {a: cell, b: cell}\nonly_cell: 1\n')
    task = _write(tmp_path / 'task.yaml', """
        cell: bench
        k: {a: task, b: task}
        over_cell:
          k: {a: insisted}
        """)
    cfg = C.load(task)
    assert cfg['k'] == {'a': 'insisted', 'b': 'cell'} and cfg['only_cell'] == 1
    assert 'over_cell' not in cfg
    assert C.load(task, ['k.a=set'])['k']['a'] == 'set'


# ------------------------------------------------------------------ the ethernet config on the bench
def test_the_ethernet_config_is_the_bnc_one_cut_for_an_rj45():
    cfg = C.load('ethernet_assembly', ['cell=iiwa14_bench'])
    g = cfg.get_path
    assert cfg['_config_bases'][-1].endswith('bnc_assembly.yaml')
    assert g('pose6d.plug') == P6.RJ45 and P6.socket_part(cfg) == P6.RJ45_SOCKET
    assert g('pose_method') == 'pose6d' and g('assembly.insertion_mode') == 'engage'
    # the bench cell's BNC tuning kept off it (over_cell): it switches these on, or sets them, for the BNC
    assert g('assembly.collar_clocking.enabled') is False and g('assembly.disassembly.enabled') is False
    assert g('assembly.connector_clocking.enabled') is False
    assert g('assembly.engage_clock_deg') == 0.0
    assert g('pickup.belief_offset.xyz_mm') == [0.0, 0.0, 0.0]
    # the bench's facts still the cell's
    assert g('robot.type') == 'iiwa14' and g('assembly.standoff.distance_mm') == 50.0
    assert g('campaign.reset') == 'manual'
    assert g('cable') == 'rj45' and g('grasp_check.connector_counts') == [198, 212]
    assert g('assembly.engage.compliance_frame') == 'connector'
    assert g('assembly.engage.push_home.enabled') is True and g('assembly.tug_verify.enabled') is True
    # ... and the BNC's own config untouched by any of it
    bnc = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    assert bnc.get_path('pose6d.plug') == P6.PLUG and bnc.get_path('assembly.collar_clocking.enabled') is True
    assert bnc.get_path('assembly.engage.compliance_frame') == 'tool0'
    assert not bnc.get_path('assembly.engage.push_home.enabled', False)


def test_the_connector_datum_is_the_plug_tip_and_the_target_its_place_on_the_jack_floor():
    """estimation.initial_connector_in_fingertip against cables.yaml rj45's junction_in_fingertip: the datum 34.44 mm
    up the axis from the junction -- the mating face; seated, that is the jack's floor (z 6.18), on its axis."""
    cfg = C.load('ethernet_assembly', ['cell=iiwa14_bench'])
    T_jc = inverse(from_cfg(cfg.section('junction_in_fingertip'))) @ from_cfg(
        cfg.get_path('estimation.initial_connector_in_fingertip'))
    np.testing.assert_allclose(T_jc[:3, 3], [P6.MATING_Z[P6.RJ45] - P6.JUNCTION_Z[P6.RJ45], 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(T_jc[:3, :3], np.eye(3), atol=1e-9)
    T_target = P6.target_from_socket(np.eye(4), T_jc, socket=P6.RJ45_SOCKET)
    np.testing.assert_allclose(T_target[:3, 3], [0.0, 0.0, 0.00618], atol=1e-6)
    np.testing.assert_allclose(T_target[:3, 0], [0.0, 0.0, -1.0], atol=1e-9)    # going in, down the jack's -z
    np.testing.assert_allclose(T_target[:3, 2], [0.0, -1.0, 0.0], atol=1e-9)    # latch (z) to the slot (-y)


# ------------------------------------------------------------------ the cut
def test_configure_enforces_the_cut_and_refuses_what_cannot_work(caplog):
    from cable_assembly.apps import ethernet_assembly as app
    over = ['cell=iiwa14_bench', 'pose6d.plug=plug', 'assembly.collar_clocking.enabled=true',
            'assembly.connector_clocking.enabled=true', 'assembly.disassembly.enabled=true',
            'assembly.engage_clock_deg=-135']
    cfg = C.load('ethernet_assembly', over)
    assert app.configure(cfg)
    assert cfg.get_path('pose6d.plug') == P6.RJ45
    for k in app._OFF:
        assert cfg.get_path(k) is False, k
    assert cfg.get_path('assembly.engage_clock_deg') == 0.0
    assert app.configure(cfg), 'idempotent'
    with caplog.at_level(logging.ERROR):
        assert not app.configure(C.load('ethernet_assembly', ['cell=iiwa14_bench', 'pose_method=sam3']))
        assert not app.configure(C.load('ethernet_assembly', ['cell=iiwa14_bench', 'assembly.insertion_mode=rl']))
        assert not app.configure(C.load('ethernet_assembly', ['cell=iiwa14_bench', 'campaign.trials=5',
                                                              'campaign.reset=auto']))
    assert app.configure(C.load('ethernet_assembly', ['cell=iiwa14_bench', 'campaign.trials=5']))
    # (a dry run: on a real arm the precheck starts the 6D models loading)
    assert app.precheck(C.load('ethernet_assembly', ['cell=iiwa14_bench', 'campaign.trials=5', 'robot.dry_run=true']))


def test_the_app_is_the_bnc_pipeline_as_a_latch_task():
    import inspect

    from cable_assembly.apps import bnc_assembly as bnc
    from cable_assembly.apps import ethernet_assembly as app
    assert app.ETHERNET.latch and not bnc.BNC.latch and app.ETHERNET.name == 'ethernet_assembly'
    src = inspect.getsource(bnc.build_and_run)
    for leaf in ("'Verify: latch (tug)'", "'Release and escape'", "'Verdict: assembled and verified'"):
        assert leaf in src, leaf
    latch = src[src.index('    def step_latch():'):src.index('    # ---------------------------------------------------------------- LOCK')]
    assert 'hold_after=True, keep_on_fail=True' in latch, 'the grip kept through the tug, passed or failed'
    assert "'misses'" in latch, 'an unlatched plug counts as a miss'


# ------------------------------------------------------------------ the latch's turn in the grasp
def _junction(heading_deg, latch_deg):
    """A junction frame lying on the bench, heading `heading_deg`, its z (an RJ45's latch) `latch_deg` from up."""
    return xyzrpy_to_matrix([0.6, 0.1, 0.011], np.radians([latch_deg, 0.0, heading_deg]))


@pytest.mark.parametrize('latch_deg, held_deg', [(0.0, 0.0), (180.0, 180.0), (-172.0, 180.0), (8.0, 0.0),
                                                 (80.0, 90.0), (-95.0, -90.0)])
def test_an_rj45_is_grasped_from_above_whichever_way_up_it_lies(latch_deg, held_deg):
    M = _junction(74.0, latch_deg)
    G, turn = P6.grasp_junction(M, P6.RJ45, square_deg=90.0)
    np.testing.assert_allclose(G[:3, 0], M[:3, 0], atol=1e-9)              # the same axis
    np.testing.assert_allclose(G[:3, 3], M[:3, 3], atol=1e-12)
    assert G[2, 2] > 0.999, 'the grasp frame z up: the pick comes from above'
    assert abs(((P6.turn_deg(turn) - held_deg) + 180.0) % 360.0 - 180.0) < 1e-6
    # unsquared: exactly the measured frame back
    G2, turn2 = P6.grasp_junction(M, P6.RJ45, square_deg=None)
    np.testing.assert_allclose(G2 @ turn2, M, atol=1e-9)


def test_a_round_plug_has_no_turn():
    M = _junction(30.0, 0.0)
    G, turn = P6.grasp_junction(M, P6.PLUG)
    np.testing.assert_allclose(turn, np.eye(4))
    np.testing.assert_allclose(G, M, atol=1e-9)


def test_the_6d_scanner_grasps_the_levelled_frame_and_keeps_the_turn():
    from cable_assembly.skills.pose6d_pick import Pose6DScanner
    cfg = C.load('ethernet_assembly', ['cell=iiwa14_bench'])
    sc = Pose6DScanner(None, None, None, cfg)
    np.testing.assert_allclose(sc.turn, np.eye(4))
    M = _junction(74.0, 178.0)                                             # lying latch down
    G = sc._grasp_frame(M, 'test')
    assert G[2, 2] > 0.999 and abs(abs(P6.turn_deg(sc.turn)) - 180.0) < 1e-6
    # the held junction the belief is built from: the grasp frame turned -- within the squaring of the measured
    np.testing.assert_allclose((G @ sc.turn)[:3, 0], M[:3, 0], atol=1e-9)
    assert float((G @ sc.turn)[:3, 2] @ M[:3, 2]) > np.cos(np.radians(2.5))


# ------------------------------------------------------------------ on the KUKA's own impedance
def test_the_impedance_version_sets_its_defaults_first_so_the_operators_set_wins():
    from cable_assembly.apps import ethernet_assembly as app
    from cable_assembly.apps.ethernet_assembly_impedance import DEFAULTS, with_defaults
    argv = with_defaults(['--cell', 'iiwa14_bench', '--set', 'assembly.tug_verify.pull_force_n=20.0'])
    assert argv[:2 * len(DEFAULTS)] == [x for kv in DEFAULTS for x in ('--set', kv)]
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == '--set']
    cfg = C.load('ethernet_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true'] + sets)
    assert cfg.get_path('robot.compliance_controller') == 'kuka_impedance'
    assert app.configure(cfg)
    assert cfg.get_path('assembly.tug_verify.pull_force_n') == 20.0, "the operator's own --set wins"
    # its own: the pull sized for whatever the pendant's stiffness is -- K x the threshold + tell_apart_n
    cfg = C.load('ethernet_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true'] + list(DEFAULTS))
    assert app.precheck(cfg)
    K = min(cfg.get_path('robot.fri.cartesian_impedance.stiffness')[:3])
    assert app._springs(cfg) == (K, K)
    thr = cfg.get_path('assembly.tug_verify.displacement_threshold_mm')
    pull = cfg.get_path('assembly.tug_verify.pull_force_n')
    assert pull == pytest.approx(K * thr / 1000.0 + 10.0, abs=0.05)
    assert pull - K * thr / 1000.0 == pytest.approx(10.0, abs=0.05), 'an unlatched plug told apart up to 10 N'
    for k in (2000.0, 3000.0):
        c = C.load('ethernet_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', *DEFAULTS,
                                         f'robot.fri.cartesian_impedance.stiffness=[{k},{k},{k},50,50,50]'])
        assert app.configure(c)
        assert c.get_path('assembly.tug_verify.pull_force_n') == pytest.approx(k * thr / 1000.0 + 10.0, abs=0.05)


def test_the_latch_precheck_refuses_a_tug_that_cannot_fail_and_a_push_that_cannot_arrive(caplog):
    from cable_assembly.apps import ethernet_assembly as app
    bench = ['cell=iiwa14_bench']
    adm = C.load('ethernet_assembly', bench)
    assert app._springs(adm) == (1500.0, 500.0), 'the engage in connector axes (x = 1500), the tug at 500'
    assert app.latch_precheck(adm)
    kuka = bench + ['robot.compliance_controller=kuka_impedance', 'robot.fri.cartesian_impedance.stiffness=[2000,2000,2000,200,200,200]']
    with caplog.at_level(logging.WARNING):
        # the admittance version's 8 N on 2000 N/m: a 4 mm offset, told apart only under 2 N -- warned, not refused
        caplog.clear()
        assert app.latch_precheck(C.load('ethernet_assembly', kuka))
        assert 'MARGINAL' in caplog.text
        # 5 N: a 2.5 mm offset, inside the 3 mm threshold -- the tug could never fail
        assert not app.latch_precheck(C.load('ethernet_assembly', kuka + ['assembly.tug_verify.pull_force_n=5.0']))
        # 12 N on 500 N/m needs 24 mm of lead: past a 20 mm push
        assert not app.latch_precheck(C.load('ethernet_assembly', bench + ['assembly.engage.stiffness=[500,500,500,10,10,10]']))
