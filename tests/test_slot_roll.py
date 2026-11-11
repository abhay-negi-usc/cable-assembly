"""perception/slot_roll.py and connector clocking's slot-aimed search (2026-11-06: "using the slot estimate the
connector roll orientation, but still do connector clocking to account for errors ... +15, -15, +30, -30")."""

import os
import tempfile
import types

import numpy as np
import pytest

from cable_assembly.perception import slot_roll as SR

CAD = '/home/lm-2023/abhay_ws/RAG-6DPose-code-abhay/data/bnc/models/obj_000003.ply'
needs_cad = pytest.mark.skipif(not os.path.exists(CAD), reason='the NIST BNC plug CAD is not on this machine')

# the bench's view of the held plug (02:31 run): the camera behind it, ~35 deg off its axis
T_CAM_PLUG = np.array([[0.0104, -0.9999, -0.0047, 0.0013],
                       [0.8265, 0.0112, -0.5628, 0.0979],
                       [0.5628, 0.0019, 0.8266, 0.0656],
                       [0.0, 0.0, 0.0, 1.0]])
K = np.array([[929.87, 0.0, 649.76], [0.0, 928.94, 355.75], [0.0, 0.0, 1.0]])


def _orthonormal(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T = T.copy()
    T[:3, :3] = U @ Vt
    return T


@needs_cad
def test_the_model_finds_the_slotted_neck_and_both_entries():
    m = SR.SlotModel.from_cad(CAD, 0.001, mating_z=0.1524)
    assert 144.5 < 1000 * m.neck[0] < 145.2 and 150.0 < 1000 * m.neck[1] < 150.5, m.neck
    assert abs(1000 * m.r_neck - 5.72) < 0.05
    assert abs(m.entry_deg - 90.0) < 3.0, 'the full J opens at 90 deg in the CAD'
    zs = np.full(8, 0.1495)
    on = m.window(zs, np.array([88.0, 92.0, 268.0, 272.0, 0.0, 180.0, 135.0, 315.0]), 0.0)
    assert on[:4].all() and not on[6:].any(), 'both entries -- the second copied 180 deg round'
    hel = m.window(np.array([0.1460, 0.1460]), np.array([20.0, 200.0]), 0.0)
    assert hel.all(), 'the helix on both sides (the CAD draws only one)'


@needs_cad
@pytest.mark.parametrize('roll', [30.0, 105.0, 150.0])
def test_a_rendered_plug_gives_its_roll_back(roll):
    """The plug drawn from the model at a known roll, the belief 5 px off -- the estimate undoes both."""
    m = SR.SlotModel.from_cad(CAD, 0.001, mating_z=0.1524)
    T = _orthonormal(T_CAM_PLUG)
    K_true = K.copy()
    K_true[0, 2] += 5.0
    K_true[1, 2] -= 4.0
    box = SR._box(m, K_true, T, (720, 1280), 0)
    zh, phh, rh = SR.march(m, K_true, T, box)
    img = np.full((720, 1280), 200.0)
    x0, y0, x1, y1 = box
    hit = np.isfinite(zh)
    seg = m.seg[m.iz(np.nan_to_num(zh, nan=m.z_min))]
    sub = img[y0:y1, x0:x1]
    sub[hit] = 90.0 + 50.0 * (seg[hit] % 2)
    neck = hit & (np.nan_to_num(zh) >= m.neck[0]) & (np.nan_to_num(zh) <= m.neck[1]) & \
        (np.abs(np.nan_to_num(rh) - m.r_neck) < 3e-4)
    sub[neck] = 170.0
    for off in (0.0, 180.0):                              # the entries: dark down the whole neck, as on the bench
        d = np.abs((np.nan_to_num(phh) - (m.entry_deg + roll + off) + 180.0) % 360.0 - 180.0)
        sub[neck & (d < 9.0)] = 30.0
    rng = np.random.default_rng(1)
    img = np.clip(img + rng.normal(0, 4.0, img.shape), 0, 255).astype(np.uint8)
    e = SR.estimate(np.repeat(img[:, :, None], 3, axis=2), K, T, m)
    err = abs(SR.wrap(e['roll_deg'] - roll))
    assert err <= 6.0, (e['roll_deg'], roll)
    assert abs(e['shift_px'][0] - 5) <= 2 and abs(e['shift_px'][1] + 4) <= 2, e['shift_px']
    assert e['margin'] > 0.3
    f = SR.fuse([e, e], m)
    assert f['roll_deg'] == e['roll_deg'] and f['n_frames'] == 2


def test_the_search_goes_nearest_first_both_ways():
    assert SR.plan_legs(20.0, 15.0, 45.0, -120.0, 120.0) == [20.0, 35.0, 5.0, 50.0, -10.0, 65.0, -25.0]
    legs = SR.plan_legs(80.0, 15.0, 90.0, -120.0, 120.0)
    assert legs[:5] == [80.0, 95.0, 65.0, 110.0, 50.0]
    assert legs[5] == -55.0, 'past the band: the other slot, 180 round'
    assert all(-120.0 <= v <= 120.0 for v in legs)
    covered = {round(SR.wrap(v)) for v in legs}
    assert all(any(abs(SR.wrap(a - c)) <= 15.0 for c in covered) for a in range(-90, 90, 5)), \
        'every alignment within a step of some leg'


def test_the_turn_is_the_nearest_inside_the_band():
    assert SR.nearest_turn(30.0, 100.0, -120.0, 120.0) == pytest.approx(70.0)
    assert SR.nearest_turn(30.0, 100.0 - 180.0, -120.0, 120.0) == pytest.approx(70.0), 'mod 180'
    assert SR.nearest_turn(-60.0, 80.0, -120.0, 120.0) == pytest.approx(-40.0)
    assert SR.nearest_turn(0.0, 100.0, 10.0, 60.0) is None


def test_the_angle_is_clockwise_looking_into_the_socket_and_turns_with_the_wrist():
    from cable_assembly.transforms import xyzrpy_to_matrix
    axis = np.array([1.0, 0.0, 0.0])                     # a wall socket: in along base +x
    assert SR.angle_about_deg([0, 0, 1], axis) == pytest.approx(0.0)
    assert SR.angle_about_deg([0, -1, 0], axis) == pytest.approx(90.0), 'right-handed about +x'
    T_plug = xyzrpy_to_matrix([0.8, 0.1, 0.27], np.radians([20.0, 90.0, 0.0]))   # plug z along base +x
    a0 = SR.entry_angle_deg(T_plug, 90.0, axis)
    for th in (15.0, -40.0, 75.0):
        R = np.eye(4)
        R[:3, :3] = xyzrpy_to_matrix([0, 0, 0], [np.radians(th), 0, 0])[:3, :3]
        assert SR.wrap(SR.entry_angle_deg(R @ T_plug, 90.0, axis) - a0, 360.0) == pytest.approx(th, abs=1e-6)


def test_the_pins_are_learned_per_socket_and_averaged_mod_180():
    path = os.path.join(tempfile.mkdtemp(), 'pins.json')
    st = SR.PinStore(path)
    assert st.get([0.8, 0.1, 0.27]) is None
    st.add(85.0, [0.8, 0.1, 0.27], trial=1)
    st.add(-85.0, [0.801, 0.1, 0.27], trial=2)            # 10 deg from the first, across the seam
    got = st.get([0.8, 0.1, 0.27])
    assert got['n'] == 2 and abs(SR.wrap(got['pin_deg'] - 90.0)) < 1e-6 and got['spread_deg'] < 6.0
    assert st.get([0.85, 0.1, 0.27]) is None, 'a socket 50 mm away is another fixture'


def test_the_trace_finds_the_drop_in_at_the_foot_of_the_rise():
    from cable_assembly.apps.bnc_assembly import _ClockTrace
    from cable_assembly.transforms import inverse, translation_matrix, xyzrpy_to_matrix
    T_axis = xyzrpy_to_matrix([0.8, 0.1, 0.27], np.radians([-135.0, 0, 0]))
    T_tc = xyzrpy_to_matrix([0.0, 0.0, 0.17], np.radians([0.0, -70.0, 0.0]))
    pose = {'T': T_axis @ inverse(T_tc)}
    robot = types.SimpleNamespace(tool0=lambda: pose['T'], arm=types.SimpleNamespace(wrench=lambda: np.zeros(6)))
    trace = _ClockTrace(robot, types.SimpleNamespace(delta=np.zeros(6)), tempfile.mktemp(suffix='.csv'), T_axis, T_tc,
                        every=1)
    trace.drop_mm, trace.drop_window_deg = 1.5, 30.0
    trace.phase = 'leg 1/7'
    for th in np.linspace(0.0, 60.0, 241):
        x = 0.0002 * th / 60.0 + (0.0028 * min(1.0, (th - 33.0) / 4.0) if th > 33.0 else 0.0)   # creep, then the drop
        pose['T'] = T_axis @ translation_matrix([x, 0, 0]) @ xyzrpy_to_matrix([0, 0, 0], [np.radians(th), 0, 0]) \
            @ inverse(T_tc)
        trace.step()
    trace.close()
    assert trace.drop is not None and abs(trace.drop['theta_deg'] - 33.0) < 1.0, trace.drop
    assert trace.drop['phase'] == 'leg 1/7'


def _trace_run(xs_mm, push_mm, ths_deg):
    from cable_assembly.apps.bnc_assembly import _ClockTrace
    from cable_assembly.transforms import inverse, translation_matrix, xyzrpy_to_matrix
    T_axis = xyzrpy_to_matrix([0.8, 0.1, 0.27], np.radians([-135.0, 0, 0]))
    T_tc = xyzrpy_to_matrix([0.0, 0.0, 0.17], np.radians([0.0, -70.0, 0.0]))
    pose = {'T': T_axis @ inverse(T_tc)}
    robot = types.SimpleNamespace(tool0=lambda: pose['T'], arm=types.SimpleNamespace(wrench=lambda: np.zeros(6)))
    trace = _ClockTrace(robot, types.SimpleNamespace(delta=np.zeros(6)), tempfile.mktemp(suffix='.csv'), T_axis, T_tc,
                        every=1)
    trace.drop_mm, trace.drop_window_deg = 1.5, 30.0
    trace.phase = 'leg 1/7'
    for x, p, th in zip(xs_mm, push_mm, ths_deg):
        trace.push_mm = p
        pose['T'] = T_axis @ translation_matrix([x / 1000.0, 0, 0]) @ \
            xyzrpy_to_matrix([0, 0, 0], [np.radians(th), 0, 0]) @ inverse(T_tc)
        trace.step()
    trace.close()
    return trace.drop


def test_a_plug_held_back_by_the_face_then_dropping_onto_the_pins_is_a_drop_in():
    th = np.linspace(0.0, 60.0, 241)
    push = np.minimum(3.0, th / 10.0)                     # the push built over the first 30 deg
    x = np.where(th < 40.0, 0.2, np.minimum(3.0, 0.2 + (th - 40.0) * 0.9))   # blocked, then it catches up
    drop = _trace_run(x, push, th)
    assert drop is not None and abs(drop['theta_deg'] - 40.0) < 1.0, drop


def test_an_arm_that_just_follows_the_push_is_not_a_drop_in():
    """A dry run (or a plug free of the face): the measured advance IS the push, ramped over a short leg."""
    th = np.linspace(0.0, 20.0, 81)
    push = np.minimum(3.0, th / 5.0)
    assert _trace_run(push, push, th) is None


TWO_SLOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cad',
                        'bnc_plug_two_j_slots.ply')


@needs_cad
@pytest.mark.skipif(not os.path.exists(TWO_SLOT), reason='analysis/bnc_plug_mesh.py has not made the two-slot mesh')
def test_the_two_slot_plug_is_the_one_slot_cad_with_its_j_copied_round():
    """The user's two-J-slot 6641T31 (2026-11-06), meshed into obj_000003's frame: the same solid, the same neck,
    and its slot map exactly the one-slot CAD's J copied 180 deg round -- what the model did before it existed."""
    one = SR.SlotModel(*_mesh(CAD), mating_z=0.1524, symmetric=False)
    two = SR.SlotModel(*_mesh(TWO_SLOT), mating_z=0.1524, symmetric=False)
    assert np.abs(one.R - two.R[:len(one.R)]).max() < 2e-5 and one.neck == two.neck
    assert not (two.slot ^ np.roll(two.slot, two.slot.shape[1] // 2, axis=1)).any(), '180 deg symmetric'
    assert not (two.slot ^ (one.slot | np.roll(one.slot, one.slot.shape[1] // 2, axis=1))).any()
    assert abs(SR.wrap(two.entry_deg - one.entry_deg)) < 1.0, 'the same entries, mod 180'


def _mesh(path):
    from cable_assembly.perception.engage_vision import load_mesh
    return load_mesh(path, 0.001)
