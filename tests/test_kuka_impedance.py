"""robot/admittance.py KukaImpedanceController + apps/bnc_assembly_impedance.py: the KUKA's own Cartesian impedance
in place of the software admittance (the user, 2026-11-06)."""

import types

import numpy as np
import pytest

from cable_assembly.robot.admittance import (AdmittanceController, KukaImpedanceController, KUKA_DEFAULT_STIFFNESS,
                                             make_compliance)

BENCH = {'stiffness': [500.0, 500.0, 500.0, 5.0, 10.0, 10.0], 'mass': [50.0] * 3 + [5.0] * 3,
         'damping_ratio': [10.0] * 6, 'selected_axes': [1, 1, 1, 1, 0, 0], 'force_deadband_n': 3.0,
         'reference_rate_hz': 125.0}


class Arm:
    """servo_l recorded; a constant external wrench; tcp_pose a fixed offset off the command (the KUKA's give)."""

    def __init__(self, wrench=(0, 0, 0, 0, 0, 0), compliance='kuka_impedance', kuka=True, dry_run=False):
        self.cmds, self.w = [], np.asarray(wrench, dtype=float)
        self.compliance, self.dry_run = compliance, dry_run
        self.supports_kuka_impedance = kuka
        self.kuka_stiffness = [1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]
        self.sag = np.eye(4)

    def wrench(self):
        return self.w

    def servo_l(self, T, dt, lookahead=0.1, gain=300):
        self.cmds.append(np.array(T, dtype=float))

    def tcp_pose(self):
        return self.cmds[-1] @ self.sag

    def servo_stop(self):
        pass


def test_the_reference_is_streamed_as_it_is_whatever_the_wrench():
    arm = Arm(wrench=[10.0, -4.0, 6.0, 0.5, 0.3, -0.2])
    adm = make_compliance(arm, BENCH)
    assert isinstance(adm, KukaImpedanceController)
    T0, T1 = np.eye(4), np.eye(4)
    T1[:3, 3] = [0.0, 0.0, 0.01]
    assert adm.ramp(T0, T1, 1.0) == 'done'
    assert np.allclose(arm.cmds[-1], T1) and np.allclose(adm.delta, 0.0), 'nothing integrated here'


def test_its_stiffness_is_the_controllers_so_pulls_and_pushes_are_sized_for_it():
    adm = KukaImpedanceController(Arm(), BENCH)
    assert np.allclose(adm.S, [1000.0] * 3 + [100.0] * 3) and np.all(adm.selected == 1)
    arm = Arm()
    arm.kuka_stiffness = None
    assert np.allclose(KukaImpedanceController(arm, BENCH).S, KUKA_DEFAULT_STIFFNESS)


def test_nulled_axes_drift_with_the_force_and_stop_at_their_cap():
    arm = Arm(wrench=[0.0, 4.5, 0.0, 0.0, 0.0, 0.0])
    adm = KukaImpedanceController(arm, BENCH)
    adm.null_axes([1, 2], damping=2000.0, deadband=1.5, max_offset=0.005)
    for _ in range(125):                                   # 1 s
        adm._step(np.eye(4), 1.0 / 125.0)
    assert adm.delta[1] == pytest.approx((4.5 - 1.5) / 2000.0, rel=1e-6), 'at (F - deadband) / D'
    assert adm.delta[0] == 0.0 and adm.delta[2] == 0.0
    for _ in range(125 * 20):
        adm._step(np.eye(4), 1.0 / 125.0)
    assert adm.delta[1] == pytest.approx(0.005), 'held at max_offset'


def test_nulling_in_series_with_the_kuka_spring_takes_the_side_load_off():
    """A socket holding the connector 2 mm across the screw axis (5 kN/m), the KUKA's 2 kN/m spring between it and
    the command: the old fixed set-point held ~2.9 N across; the drift walks it under the deadband."""
    K_env, K_k, EPS = 5000.0, 2000.0, 0.002

    class SeriesArm(Arm):
        def tcp_pose(self):
            T = self.cmds[-1].copy()
            y_cmd = T[1, 3]
            T[1, 3] = (K_k * y_cmd + K_env * EPS) / (K_k + K_env)    # where the two springs balance
            return T

        def wrench(self):
            if not self.cmds:
                return np.zeros(6)
            return np.array([0.0, K_env * (EPS - self.tcp_pose()[1, 3]), 0.0, 0.0, 0.0, 0.0])

    arm = SeriesArm()
    adm = KukaImpedanceController(arm, BENCH)
    adm._step(np.eye(4), 1.0 / 125.0)
    held = arm.wrench()[1]
    adm.null_axes([1, 2], damping=2000.0, deadband=1.5, max_offset=0.005)
    for _ in range(125 * 30):
        adm._step(np.eye(4), 1.0 / 125.0)
    assert held > 2.5 and abs(arm.wrench()[1]) <= 1.5 + 0.05, (held, arm.wrench()[1])


def test_give_reads_the_kuka_springs_real_deflection():
    arm = Arm()
    arm.sag = np.eye(4)
    arm.sag[:3, 3] = [0.0, 0.0, -0.003]                    # 3 mm sag along tool z
    adm = KukaImpedanceController(arm, BENCH)
    adm._step(np.eye(4), 1.0 / 125.0)
    assert np.allclose(adm.give(), [0, 0, -0.003, 0, 0, 0], atol=1e-12)
    a = AdmittanceController(Arm(compliance='admittance'), BENCH)
    assert np.allclose(a.give(), a.delta), "the admittance's give is its delta"


def test_the_factory_picks_by_the_arm_and_refuses_what_has_no_kuka():
    assert type(make_compliance(Arm(compliance='admittance'), BENCH)) is AdmittanceController
    with pytest.raises(ValueError, match='needs a KUKA iiwa'):
        make_compliance(Arm(kuka=False), BENCH)
    assert isinstance(make_compliance(Arm(kuka=False, dry_run=True), BENCH), KukaImpedanceController)
    with pytest.raises(ValueError):
        make_compliance(Arm(compliance='forcemode'), BENCH)


@pytest.mark.parametrize('mode,name', [(0, 'POSITION'), (2, 'JOINT IMPEDANCE'), (3, 'NO CONTROL'), (-1, None)])
def test_the_impedance_version_is_refused_unless_the_controller_is_in_cartesian_impedance(mode, name):
    from cable_assembly.robot.arm import ArmError
    from cable_assembly.robot.backends.iiwa_fri import _check_kuka_impedance
    with pytest.raises(ArmError, match='CARTESIAN_IMPEDANCE_CONTROL'):
        _check_kuka_impedance(mode, name, KUKA_DEFAULT_STIFFNESS)
    _check_kuka_impedance(1, 'CARTESIAN IMPEDANCE', KUKA_DEFAULT_STIFFNESS)


def test_no_load_data_in_sunrise_is_refused():
    from cable_assembly.robot.arm import ArmError
    from cable_assembly.robot.backends.iiwa_fri import _check_load_data
    with pytest.raises(ArmError, match='LOAD DATA'):
        _check_load_data([0.3, -0.5, -11.8, 0.1, 0.2, 0.0], 1.31)    # 2026-10-29: ~12 N untared = no load data
    _check_load_data([0.3, -0.5, 1.1, 0.0, 0.0, 0.0], 1.31)


def test_the_entry_point_sets_its_defaults_first_so_the_operators_set_wins():
    from cable_assembly import config as C
    from cable_assembly.apps.bnc_assembly_impedance import DEFAULTS, size_clocking_push, with_defaults
    argv = with_defaults(['--cell', 'iiwa14_bench', '--set', 'assembly.connector_clocking.push_mm=3.0'])
    assert argv[:2 * len(DEFAULTS)] == [x for kv in DEFAULTS for x in ('--set', kv)]
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == '--set']
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench'] + sets)
    assert cfg.get_path('robot.compliance_controller') == 'kuka_impedance'
    assert size_clocking_push(cfg, sets) is None
    assert cfg.get_path('assembly.connector_clocking.push_mm') == 3.0, "the operator's own --set wins"


def test_the_clocking_push_is_sized_for_5_n_on_the_robots_own_spring():
    """This robot's LBRServer runs 3000 N/m / 50 Nm/rad (its project, 2026-11-06), not KUKA's 2000 / 200."""
    from cable_assembly import config as C
    from cable_assembly.apps.bnc_assembly import clocking_precheck
    from cable_assembly.apps.bnc_assembly_impedance import DEFAULTS, size_clocking_push
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench'] + list(DEFAULTS))
    K = cfg.get_path('robot.fri.cartesian_impedance.stiffness')
    assert K == [3000.0, 3000.0, 3000.0, 50.0, 50.0, 50.0]
    push = size_clocking_push(cfg, list(DEFAULTS))
    assert push == pytest.approx(1.67) and cfg.get_path('assembly.connector_clocking.push_mm') == push
    assert K[0] * push / 1000.0 == pytest.approx(5.0, abs=0.02), '5 N of press'
    assert clocking_precheck(cfg), 'under the early-out'


class _Passed(Exception):
    pass


def _path_start(compliance, q_meas_deg, q_cmd_deg, start_deg):
    """IiwaFRIArm.move_joint_path's start check alone: True accepted (it went on to time the path), False refused."""
    from cable_assembly.robot.backends.iiwa_fri import IiwaFRIArm
    arm = IiwaFRIArm.__new__(IiwaFRIArm)
    arm.compliance = compliance
    s = types.SimpleNamespace(q=np.radians(q_meas_deg), q_cmd=np.radians(q_cmd_deg))
    arm._commanding = lambda label: s
    arm.joints_ok = lambda q: True

    def stop(*_a, **_k):
        raise _Passed
    arm._limits = stop
    q = np.radians(np.vstack([start_deg, np.asarray(start_deg) + 1.0]))
    try:
        return arm.move_joint_path([0.0, 1.0], q, label='grasp-align')
    except _Passed:
        return True


def test_under_the_kukas_impedance_a_plan_starts_where_the_arm_stands_not_where_it_was_told():
    """2026-11-06 11:38: at rest at the scan pose, A5 -2.9 / A7 +2.9 deg off the command (the impedance holds the
    tool, the null space is free); the plan, from the measured joints, was refused as starting 2.9 deg away."""
    cmd = np.zeros(7)
    meas = np.array([1.29, 0.07, -1.05, -0.2, -2.88, 0.37, 2.92])
    assert _path_start('admittance', meas, cmd, meas) is False, 'position control: the command IS where it is'
    assert _path_start('admittance', cmd, cmd, cmd) is True
    assert _path_start('kuka_impedance', meas, cmd, meas) is True, 'impedance: it starts where the arm stands'
    assert _path_start('kuka_impedance', meas, cmd, cmd) is False, 'not where it was only told to be'
    far = meas * 3.0                                   # ~8.8 deg off: something holds it
    assert _path_start('kuka_impedance', far, cmd, far) is False
