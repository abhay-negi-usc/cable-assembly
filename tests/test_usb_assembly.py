"""usb_assembly: bnc_assembly's pipeline for a USB-A plug -- INSERTION ONLY (a push-fit task: no clocking, no tug;
the user, 2026-11-12). Its config (extends bnc_assembly, the bench's BNC tuning kept off it), the plug's and the
receptacle's frames, and the dry run end to end on both compliances, picked and hand-loaded."""

import csv
import glob
import sys
import types

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.apps import bnc_assembly, usb_assembly
from cable_assembly.perception import pose6d as P6


def test_the_usb_config_is_the_bnc_one_cut_to_the_insertion():
    cfg = C.load('usb_assembly', ['cell=iiwa14_bench'])
    g = cfg.get_path
    assert cfg['_config_bases'][-1].endswith('bnc_assembly.yaml')
    assert g('pose6d.plug') == P6.USB and P6.socket_part(cfg) == P6.USB_SOCKET
    assert g('pose_method') == 'pose6d' and g('assembly.insertion_mode') == 'engage'
    for off in ('connector_clocking', 'collar_clocking', 'disassembly', 'tug_verify'):
        assert g(f'assembly.{off}.enabled') is False, off
    assert g('assembly.engage_clock_deg') == 0.0 and g('pickup.belief_offset.xyz_mm') == [0.0, 0.0, 0.0]
    assert g('campaign.reset') == 'manual' and g('cable') == 'usb'
    # pushed home above the sliding force, and the advance not stopped under it
    ph, fl = g('assembly.engage.push_home'), g('assembly.engage.force_limit')
    assert ph['enabled'] is True and ph['force_n'] > fl['target_n'] > fl['full_rate_below_n'] > 6.0
    assert usb_assembly.configure(cfg) and usb_assembly.USB.latch and not usb_assembly.USB.verify
    # ... and the BNC's own config untouched by any of it
    bnc = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    assert bnc.get_path('pose6d.plug') == P6.PLUG and bnc.get_path('assembly.collar_clocking.enabled') is True
    assert bnc_assembly.BNC.verify and not bnc_assembly.BNC.latch


def test_the_cut_is_enforced_whatever_the_config_says():
    cfg = C.load('usb_assembly', ['cell=iiwa14_bench', 'assembly.tug_verify.enabled=true',
                                  'assembly.collar_clocking.enabled=true', 'pose6d.plug=plug'])
    assert usb_assembly.configure(cfg)
    assert cfg.get_path('pose6d.plug') == P6.USB
    assert cfg.get_path('assembly.tug_verify.enabled') is False and cfg.get_path('assembly.collar_clocking.enabled') is False
    assert not usb_assembly.configure(C.load('usb_assembly', ['cell=iiwa14_bench', 'assembly.insertion_mode=rl']))
    assert not usb_assembly.configure(C.load('usb_assembly', ['cell=iiwa14_bench', 'campaign.trials=3',
                                                              'campaign.reset=auto']))


def test_the_usb_plug_seats_cavity_to_tongue_at_its_depth():
    """The parts' frames (perception/pose6d.py): the plug's axis z, its shell's front 48 mm from the junction; home,
    that front is 3.7 mm below the adapter's origin, on its axis, and the plug's -y (its cavity) is over the
    receptacle's +y (its tongue)."""
    assert P6.SYMMETRY[P6.USB] == 'none' and P6.SYMMETRY[P6.USB_SOCKET] == 'none'
    assert P6.MATING_Z[P6.USB] - P6.JUNCTION_Z[P6.USB] == pytest.approx(0.048)
    T = P6.T_socket_cord_seated(P6.USB_SOCKET)
    front = T @ [0.0, 0.0, P6.MATING_Z[P6.USB], 1.0]
    np.testing.assert_allclose(front[:3], [0.0, 0.0, -0.0037], atol=1e-9)
    np.testing.assert_allclose(T[:3, :3] @ [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    # the junction frame: x along the insertion; the target's connector datum is the shell's front
    J = P6.seated_junction_in_socket(P6.USB_SOCKET)
    np.testing.assert_allclose(J[:3, 0], [0.0, 0.0, -1.0], atol=1e-12)          # into the receptacle (its -z)
    np.testing.assert_allclose((J @ [0.048, 0.0, 0.0, 1.0])[:3], [0.0, 0.0, -0.0037], atol=1e-9)


def _run(tmp_path, monkeypatch, app, *extra):
    monkeypatch.setattr(sys, 'argv', ['x', '--cell', 'iiwa14_bench', '--dry-run', '--no-prompts',
                                      '--set', f'data_dir={tmp_path}', '--set', 'live_view.enabled=false', *extra])
    monkeypatch.setattr(sys, 'stdin', types.SimpleNamespace(readline=lambda: '', read=lambda *a: ''))
    with pytest.raises(SystemExit) as ex:
        app.main()
    rows = list(csv.DictReader(open(glob.glob(str(tmp_path / 'experiments' / 'usb_assembly_*' / 'trials.csv'))[0])))
    return ex.value.code, rows


@pytest.mark.parametrize('impedance, hand_load', [(False, False), (True, False), (True, True)])
def test_the_usb_dry_run_inserts_and_stops_there(tmp_path, monkeypatch, impedance, hand_load):
    pytest.importorskip('cumotion')
    from cable_assembly.apps import usb_assembly_impedance
    code, rows = _run(tmp_path, monkeypatch, usb_assembly_impedance if impedance else usb_assembly,
                      *(['--hand-load'] if hand_load else []))
    assert code == 0, rows
    assert rows and 'inserted, pushed home' in str(rows[0]), rows
    assert 'tug' not in str(rows[0].get('outcome', '')).lower()


def test_the_usb_socket_is_the_one_recorded_and_the_right_way_up():
    """The bench has four look-alike panel sockets, each in the same box, and the estimator cannot tell the USB
    socket's half turn about its axis (2026-11-12: found at iou 0.91, the wrong way up). The frames file's
    usb_connector says which socket it is and which way up: its tongue (the adapter's +y) DOWN, facing the arm;
    the app takes the estimate's half turn from it, and the socket's model is the adapter AS MOUNTED."""
    from cable_assembly import tool_frames
    from cable_assembly.transforms import from_cfg, inverse
    cfg = C.load('usb_assembly', ['cell=iiwa14_bench'])
    assert usb_assembly.configure(cfg)
    assert P6.OBJECTS[P6.USB_SOCKET] == 'usb_adapter_mount' and P6.DRAWN[P6.USB_SOCKET] == 'usb_adapter'
    assert cfg.get_path('assembly.visual_target.half_turn_from_recorded') is True
    assert cfg.get_path('pose6d.min_edge') == pytest.approx(0.4)
    targets = tool_frames.load_targets(cfg)
    T_jc = inverse(from_cfg(cfg.section('junction_in_fingertip'))) @ from_cfg(
        cfg.get_path('estimation.initial_connector_in_fingertip'))
    S = targets['usb_connector'] @ inverse(T_jc) @ inverse(P6.seated_junction_in_socket(P6.USB_SOCKET))
    assert S[2, 1] < -0.98, 'the tongue (+y) down'
    assert S[0, 2] < -0.98, 'the opening (+z) toward the arm (base -x)'
    # the TOP-LEFT place on the bench's wall (the user, 2026-11-13: moved there from bottom-left, [808.6, 227.8,
    # 148.4]; top right is the BNC, bottom right the RJ45) -- its way up kept as recorded, not re-measured
    np.testing.assert_allclose(S[:3, 3], [0.8106, 0.2240, 0.2707], atol=2e-3)
    # the BNC's own run is untouched by any of it
    bnc = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    assert not bnc.get_path('assembly.visual_target.half_turn_from_recorded', False)
    assert bnc.get_path('pose6d.min_edge') == pytest.approx(0.5)


def test_the_held_usb_plug_is_fitted_as_the_jaws_let_it_move_and_its_grasp_band_takes_the_relief():
    """2026-11-13, hand-loaded: every in-hand estimate failed ("no estimate passed the gate"). The plug is fitted
    in the hand by the jaws' free DOF alone (pose6d.in_hand.jaw_fit, on for the USB plug, off for the BNC), on a
    gate that takes its ragged strain-relief mask; and the grasp check takes the plug held by its 10.5 mm strain
    relief (206 counts, measured) as well as by its 16 mm body -- the cord (217) still a miss."""
    # the OVERMOLD grasp's (the cable grasp, pickup.cable_grasp, fits the plug free:
    # test_held_by_its_cable_the_plug_is_tracked_free_and_from_further_back)
    usb = C.load('usb_assembly', ['cell=iiwa14_bench', 'pickup.cable_grasp.enabled=false'])
    bnc = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    ih = usb.get_path('pose6d.in_hand')
    assert ih['jaw_constraint'] is True and ih['jaw_fit']['enabled'] is True and ih['min_iou'] == 0.5
    assert ih['jaw_fit']['turn_deg'] > 0 and ih['jaw_fit']['slide_mm'] >= 20 and ih['jaw_fit']['pivot_deg'] >= 25
    assert bnc.get_path('pose6d.in_hand.jaw_fit.enabled') is False and bnc.get_path('pose6d.in_hand.min_iou') == 0.6
    lo, hi = usb.get_path('grasp_check.connector_counts')
    assert lo <= 191 and 206 <= hi < usb.get_path('grasp_check.cable_counts') == 217
    # the tracker reads it: the search's ranges, as the request carries them (the *_m / *_rad the loader adds left out)
    from types import SimpleNamespace
    from cable_assembly.skills import pose6d_pick as PP
    tr = PP.ConnectorTracker(usb, SimpleNamespace(), None)
    assert tr.jaw_fit == {'slide_mm': 25.0, 'pivot_deg': 30.0, 'turn_deg': 20.0, 'trim_mm': 6.0}
    assert PP.ConnectorTracker(bnc, SimpleNamespace(), None).jaw_fit is None


@pytest.mark.parametrize('case, extra, ok, verdict, says', [
    ('upside down', ['--set', 'dry_cell.plug.upside_down=true'], False, 'wrong_way_up', 'WRONG WAY UP'),
    ('on the face', ['--set', 'dry_cell.in_hand_error={xyz_mm: [0.0, 5.0, 0.0], rpy_deg: [0.0, 0.0, 0.0]}'], False,
     'free', 'ENGAGE FAILED'),
    ('home, nothing met on the way', ['--set', 'dry_cell.jack.insertion_n=0.0'], True, 'in', 'inserted, pushed home'),
])
def test_which_way_up_is_read_from_the_contact(tmp_path, monkeypatch, case, extra, ok, verdict, says):
    """assembly.engage.way_up (the user, 2026-11-13: "i want to use proprioception + wrench to determine that
    orientation"). The wrong way up a USB-A plug stops 1.3 mm inside the mouth: the engage stalls with no travel, as
    on the face beside the mouth. The plug is then probed sideways under a light press: CAPTURED and short of the
    seat -- the wrong way up (no re-aim, said to the operator); FREE -- a miss, as before; captured AT the seat -- it
    was home all along (it met nothing before the floor), and the run goes on to the push home."""
    pytest.importorskip('cumotion')
    import json
    from cable_assembly.apps import usb_assembly_impedance
    code, rows = _run(tmp_path, monkeypatch, usb_assembly_impedance, '--hand-load',
                      '--set', 'assembly.engage.timeout_s=10', *extra)
    assert (code == 0) is ok, rows
    way = json.loads(rows[0]['way_up'])
    assert way['verdict'] == verdict, way
    assert says in str(rows[0]), rows
    if verdict == 'wrong_way_up':
        assert way['depth_mm'] == pytest.approx(way['blocked_at_mm'], abs=1.0) and way['blocked_at_mm'] == -9.01
        assert max(way['went_mm'].values()) <= 0.5 * way['probe_mm'] and rows[0]['reaims'] in ('0', '', None)
    if verdict == 'free':
        assert max(way['went_mm'].values()) > 0.5 * way['probe_mm']


def test_the_way_up_check_is_the_usb_plugs_and_the_impedances():
    usb, bnc = C.load('usb_assembly', ['cell=iiwa14_bench']), C.load('bnc_assembly', ['cell=iiwa14_bench'])
    w = usb.get_path('assembly.engage.way_up')
    assert w['enabled'] is True and w['home_mm'] - w['blocked_mm'] == pytest.approx(9.01)
    assert bnc.get_path('assembly.engage.way_up.enabled') is False        # a BNC goes in any way round
    # the CAD the two depths are from: the receptacle's mouth at z 6.65, its tongue's front at 5.31, the seat -3.70
    assert w['home_mm'] == pytest.approx(6.65 + 3.70) and w['blocked_mm'] == pytest.approx(6.65 - 5.31)
    assert usb.get_path('dry_cell.jack.blocked_mm') == w['blocked_mm']


def test_the_cable_grasp_is_three_numbers_written_into_the_one_grasp_transform():
    """pickup.cable_grasp (the user, 2026-11-13: "USB pickup via cable ... the grasp angle (wrt cable axis) and
    distance from the junction, and z height to be tunable parameters in yaml"): the bite behind the junction on
    the cable, the gripper's axis that far from the cable's -- compiled into pickup.fingertip_in_connector at
    load, so the pick, the belief and the hand load's drawing follow; off, the overmold's grasp; the BNC's untouched."""
    from cable_assembly.skills.pick import fingertip_in_connector, held_belief
    from cable_assembly.transforms import from_cfg, inverse
    cfg = C.load('usb_assembly', ['cell=iiwa14_bench'])
    cg = cfg.get_path('pickup.cable_grasp')
    assert cg['enabled'] is True
    d, a, z = cg['distance_mm'], cg['angle_deg'], cg['z_mm']
    assert cfg.get_path('pickup.fingertip_in_connector.xyz_mm') == [-d, 0.0, z]
    assert cfg.get_path('pickup.fingertip_in_connector.rpy_deg') == [0.0, a - 90.0, 0.0]
    assert cfg.get_path('pickup.fingertip_in_connector_on_connector.xyz_mm') == [20.0, 0.0, 0.0]    # kept beside it
    G = fingertip_in_connector(cfg)                               # the fingertip in the plug's junction frame
    np.testing.assert_allclose(G[:3, 3], [-d / 1000.0, 0.0, z / 1000.0], atol=1e-12)       # behind it, on the cable
    tool_z = -G[:3, 2]                                            # the gripper's approach (the fingertip's z is back up it)
    assert np.degrees(np.arccos(tool_z @ [1.0, 0.0, 0.0])) == pytest.approx(a)             # from the cable's axis...
    assert tool_z[2] < 0.0 and tool_z[0] > 0.0 and abs(tool_z[1]) < 1e-12                  # ...down, toward the plug
    # the belief: the plug rigid on the cable, its datum (the shell's front) 48 mm + the distance ahead of the bite
    T_fc = held_belief(from_cfg(cfg.get_path('estimation.initial_connector_in_fingertip')),
                       from_cfg(cfg.section('junction_in_fingertip')), G)
    assert np.linalg.norm(T_fc[:3, 3]) == pytest.approx(np.hypot((48.0 + d) / 1000.0, z / 1000.0), abs=1e-9)
    # tunable: --set, and each number does what its name says
    c2 = C.load('usb_assembly', ['cell=iiwa14_bench', 'pickup.cable_grasp.angle_deg=90',
                                 'pickup.cable_grasp.distance_mm=30', 'pickup.cable_grasp.z_mm=1.5'])
    np.testing.assert_allclose(fingertip_in_connector(c2), [[1, 0, 0, -0.030], [0, 1, 0, 0], [0, 0, 1, 0.0015],
                                                            [0, 0, 0, 1]], atol=1e-12)     # straight down on it
    off = C.load('usb_assembly', ['cell=iiwa14_bench', 'pickup.cable_grasp.enabled=false'])
    assert off.get_path('pickup.fingertip_in_connector.xyz_mm') == [20.0, 0.0, 0.0]
    assert off.get_path('pickup.fingertip_in_connector.rpy_deg') == [0.0, -70.0, 0.0]
    bnc = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    assert bnc.get_path('pickup.cable_grasp') is None
    assert bnc.get_path('pickup.fingertip_in_connector.rpy_deg') == [0.0, -70.0, 0.0]
    for bad in ('pickup.cable_grasp.distance_mm=-5', 'pickup.cable_grasp.angle_deg=0', 'pickup.cable_grasp.angle_deg=120'):
        with pytest.raises(ValueError):
            C.load('usb_assembly', ['cell=iiwa14_bench', bad])


def test_held_by_its_cable_the_plug_is_tracked_free_and_from_further_back():
    """The in-hand tracker with pickup.cable_grasp: the fingers close `distance_mm` behind the plug and it still
    takes the plug as held; and the jaws, holding the cable, pin none of the plug's pose -- no jaw constraint, no
    jaw fit (2026-11-13: the jaw fit 7.5 mm from the depth's on the cable-held views, the free fit 2.2)."""
    from cable_assembly.skills.pose6d_pick import ConnectorTracker
    cfg = C.load('usb_assembly', ['cell=iiwa14_bench'])
    tr = ConnectorTracker(cfg, robot=None, camera=None)
    assert tr.attach_back_m == pytest.approx(cfg.get_path('pickup.cable_grasp.distance_mm') / 1000.0)
    assert tr.jaw_constraint is False and tr.jaw_fit is None
    off = ConnectorTracker(C.load('usb_assembly', ['cell=iiwa14_bench', 'pickup.cable_grasp.enabled=false']), None, None)
    assert off.attach_back_m == 0.0 and off.jaw_constraint is True and off.jaw_fit is not None


def test_repositioning_grabs_the_cable_too_and_no_count_is_checked():
    """The user, 2026-11-13: "also when repositioning, grab by cable rather than connector" and "turn off grasp
    checking for usb". The reorient recovery's square pick is on the cable (assembly.reorient_recovery.cable_grasp,
    straight down by default); and a close on the cable -- 223 counts, which the count classifier calls "grabbed the
    CABLE" and would reseat onto the plug -- is the grasp: no band, no reseat, no slip check. The BNC keeps them all."""
    from types import SimpleNamespace

    from cable_assembly.skills.pick import GraspCheck, GraspRecovery
    cfg = C.load('usb_assembly', ['cell=iiwa14_bench'])
    rr = cfg.get_path('assembly.reorient_recovery')
    cg = rr['cable_grasp']
    assert cg['enabled'] is True and cg['angle_deg'] == 90.0
    assert rr['fingertip_in_connector']['xyz_mm'] == [-cg['distance_mm'], 0.0, cg['z_mm']]
    assert rr['fingertip_in_connector']['rpy_deg'] == [0.0, 0.0, 0.0]
    assert rr['fingertip_in_connector_on_connector']['xyz_mm'] == [20.0, 0.0, -2.0]
    off = C.load('usb_assembly', ['cell=iiwa14_bench', 'assembly.reorient_recovery.cable_grasp.enabled=false'])
    assert off.get_path('assembly.reorient_recovery.fingertip_in_connector.xyz_mm') == [20.0, 0.0, -2.0]
    tilted = C.load('usb_assembly', ['cell=iiwa14_bench', 'assembly.reorient_recovery.cable_grasp.angle_deg=60'])
    assert tilted.get_path('assembly.reorient_recovery.fingertip_in_connector.rpy_deg') == [0.0, -30.0, 0.0]

    check, rec = GraspCheck(cfg), GraspRecovery(cfg)
    assert not check.enabled and not check.lift_check_enabled and not rec.enabled and not rec.check_counts

    class Gripper:
        closes = 0

        def close(self, label=''):
            Gripper.closes += 1
            return True

        def position(self):
            return 223                                    # on the cable

    robot = SimpleNamespace(gripper=Gripper())
    rec.settle_s = 0.0
    assert rec.grasp_with_recovery(robot, None, check) == 'ok' and Gripper.closes == 1
    bnc = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    assert GraspCheck(bnc).enabled and GraspCheck(bnc).lift_check_enabled and GraspRecovery(bnc).check_counts
    assert bnc.get_path('assembly.reorient_recovery.cable_grasp') is None
