"""skills/rl_insertion.py: the trained insertion policy on the robot side.

Parity with the sim env itself is tested in sim/tests/test_rl_deploy_parity.py (GPU). Here: the frames
map the app's connector onto the sim's plug, the action keeps the training clamps, the seated rule, and
a dry run of the loop on the simulated iiwa."""

import os

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.perception import pose6d as P6
from cable_assembly.skills import rl_insertion as RL
from cable_assembly.transforms import inverse, translation_matrix, xyzrpy_to_matrix

POLICY = os.path.join(os.path.dirname(__file__), '..', 'configs', 'policies', 'bnc_insert_v16_1100.npz')
needs_policy = pytest.mark.skipif(not os.path.isfile(POLICY), reason='no exported policy')


def test_the_sim_plug_runs_up_the_cable_from_its_seat_point():
    # The sim's plug: z up the cable (away from the socket), origin SEAT_Z_M up the CAD's axis -- the
    # insulator face (CAD z 151.38 mm) is at the sim's +3.726 mm.
    T = RL.T_cad_simplug()
    face_cad = np.array([0.0, 0.0, 0.15138, 1.0])
    np.testing.assert_allclose((inverse(T) @ face_cad)[:3], [0.0, 0.0, 0.00373], atol=1e-5)
    # the connector's +X (toward the mating face) is the sim plug's -z
    T_plug = xyzrpy_to_matrix([0.5, 0.1, 0.01], [0.0, np.pi / 2, 0.3])
    J = P6.junction_from_plug(T_plug)
    T_jc = translation_matrix([0.03, 0.0, 0.0])
    T_cs = RL.T_conn_simplug(T_jc, P6.PLUG)
    conn = J @ T_jc
    sim = conn @ T_cs
    np.testing.assert_allclose(sim[:3, 2], -conn[:3, 0], atol=1e-9)
    np.testing.assert_allclose(sim, T_plug @ T, atol=1e-9)        # the same plug, whichever way round


@needs_policy
def test_the_action_keeps_the_training_clamps():
    spec = RL.Policy(POLICY).spec
    e = spec['env']
    ts = RL.TaskSpace(spec)
    S = xyzrpy_to_matrix([0.8, 0.1, 0.27], [0.3, -1.2, 0.5])
    P = S @ translation_matrix([0.0, 0.0, 0.0175])
    ts.reset(S, P)
    ts.observe(P, np.zeros(6))
    for _ in range(20):                                 # push everything hard one way
        T = ts.apply(np.ones(6))
    rel = inverse(S) @ T
    lead = inverse(P) @ T
    assert np.linalg.norm(lead[:3, 3]) <= np.sqrt(3) * e['max_lead'][0] + 1e-9
    assert np.all(rel[:3, 3] <= np.array(e['pos_bounds'][1]) + 1e-9)
    tilt = np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1)))
    assert tilt <= e['max_tilt_deg'] + 1e-6
    # the spring's pull at G stays under max_force at the sim's nominal stiffness
    G = np.r_[ts.G, 1.0]
    assert ts.k_lin * np.linalg.norm((T @ G - P @ G)[:3]) <= e['max_force'] + 1e-6
    # pushing (measured push > base_yaw_push) adds the base twist: the target turns +z of S
    ts2 = RL.TaskSpace(spec)
    ts2.reset(S, P)
    ts2.observe(P, np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0]))
    T2 = ts2.apply(np.zeros(6))
    assert (inverse(S)[:3, :3] @ RL.rotvec(T2[:3, :3] @ P[:3, :3].T))[2] > 0.0


@needs_policy
def test_a_turn_band_rocks_the_plug_inside_it():
    # A policy trained with the wrist's budget (env turn_band_deg): the target's turn about S's axis is
    # clamped inside the band, the base twist reverses near its edges, and the policy sees both. A
    # plug that follows its target exactly while pushing winds up and rocks back, never past the band.
    spec = RL.Policy(POLICY).spec
    spec = {**spec, 'env': {**spec['env'], 'turn_band_deg': [0.0, 230.0], 'turn_margin_deg': 20.0}}
    S = xyzrpy_to_matrix([0.8, 0.1, 0.27], [0.3, -1.2, 0.5])

    def run(yaw_action):
        ts = RL.TaskSpace(spec)
        P = S @ translation_matrix([0.0, 0.0, 0.0175])
        ts.reset(S, P)
        turns, dirs = [], []
        for _ in range(600):
            o = ts.observe(P, np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0]))
            P = ts.apply(np.r_[0.0, 0.0, 0.0, 0.0, 0.0, yaw_action])
            turns.append(np.degrees(ts.turn))
            dirs.append(ts.yaw_dir)
        return o, np.array(turns), np.array(dirs)

    o, turns, dirs = run(0.0)                                 # the base twist alone rocks
    assert o.shape[0] == RL.Policy(POLICY).obs_dim + 2
    assert o[-1] in (-1.0, 1.0) and -1.0 - 1e-9 <= o[-2] <= 1.0 + 1e-9
    assert turns.min() >= -1e-6 and turns.max() <= 230.0 + 1e-6
    assert turns.max() > 200.0 and np.sum(np.diff(dirs) != 0) >= 2
    _, turns, _ = run(1.0)                                    # a policy twisting on is held at the edge
    assert turns.max() == pytest.approx(230.0, abs=1e-6)


def test_seated_is_advance_after_the_first_touch_held():
    d = RL.SeatDetector(5.0, 0.0045, 0.5)
    t = 0.0
    assert not d(0.012, 2.0, t)                          # no touch yet: no datum
    assert not d(0.010, 6.0, 0.05)                       # touch at 10 mm
    assert not d(0.0062, 8.0, 0.1)                       # on the pins: 3.8 mm
    assert not d(0.0050, 8.0, 0.2)                       # 5 mm: starts the hold
    assert not d(0.0050, 8.0, 0.6)
    assert d(0.0049, 8.0, 0.71)                          # held 0.5 s


@needs_policy
def test_a_dry_run_of_the_policy_never_touches_and_times_out():
    # The dry-run arm measures no wrench: the base push drives the plug down to the bound and the
    # policy runs its whole episode without a first touch -- a miss, not a seat.
    from cable_assembly.robot import Robot
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false',
                                  'robot.world.enabled=false'])
    robot = Robot(cfg, with_gripper=False, needs=())
    pol = RL.Policy(POLICY)
    pol.spec['env']['episode_s'] = 1.0
    T0 = robot.arm.tcp_pose()
    T_tool0_conn = translation_matrix([0.0, 0.0, 0.2])
    T_cs = RL.T_conn_simplug(translation_matrix([0.03, 0.0, 0.0]), P6.PLUG)
    # the target: the believed connector 17.5 mm further along its +X
    T_target = T0 @ T_tool0_conn @ translation_matrix([0.0175, 0.0, 0.0])
    rc = {'admittance': {'stiffness': [1e4] * 3 + [10.0] * 3, 'mass': [10.0] * 3 + [0.01] * 3},
          'touch_n': 5.0, 'seated_advance_mm': 4.5, 'hold_s': 0.5, 'max_turn_deg': 230.0}
    status, info = RL.run(robot, pol, T_target, T_tool0_conn, T_cs, rc)
    assert status == 'timeout' and info['h_touch_mm'] is None and info['steps'] == 20
