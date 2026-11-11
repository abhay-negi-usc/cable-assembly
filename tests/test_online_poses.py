"""pose6d.online (skills/pose6d_pick.OnlinePoses + robot/admittance's reference_correction): the socket and the held
connector re-estimated through contact, and the insertion's references corrected to them -- gradually, within
limits, never along the insertion axis unless asked."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cable_assembly.config import Config
from cable_assembly.perception import pose6d as P6
from cable_assembly.robot.admittance import AdmittanceController
from cable_assembly.skills import pose6d_pick as PP
from cable_assembly.transforms import inverse


def _T(xyz_mm=(0, 0, 0), rotvec_deg=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    T[:3, 3] = np.asarray(xyz_mm, dtype=float) / 1000.0
    return T


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class _Sock:
    part = P6.SOCKET

    def __init__(self, T=None, n=3):
        self.T, self.n = T, n

    def measured(self, since=None):
        return None if self.T is None else (self.T, self.n, self.n)


class _Conn:
    part = P6.PLUG

    def __init__(self, T=None, n=3):
        self.T, self.n = T, n

    def in_hand(self, since=None):
        return None if self.T is None else (self.T, self.n, self.n)


class _Robot:
    T_tool0_fingertip = _T((0, 0, 182))


CFG = Config({'pose6d': {'online': {'enabled': True, 'min_estimates': 2, 'rate_mm_s': 1.0, 'rate_deg_s': 1.0,
                                    'max_socket_mm': 5.0, 'max_socket_deg': 5.0, 'max_hand_mm': 5.0,
                                    'max_hand_deg': 5.0, 'axial': False}}})
T_S0 = _T((800, 100, 270), (0, 90, 0))           # the socket, its axis (z) along base -x
X = _T((5, 0, 0), (0, 0, 30))                    # the target in the socket's frame (any fixed transform)
TARGET = lambda T: T @ X                         # noqa: E731
F0 = _T((0, 0, 20), (0, -70, 0))                 # the in-hand belief (fingertip -> connector)
T_JC = np.eye(4)


def _online(monkeypatch, sock=None, conn=None, cfg=CFG):
    clock = _Clock()
    monkeypatch.setattr(PP.time, 'monotonic', clock)
    oc = PP.OnlinePoses(cfg, _Robot(), sock, conn, T_S0, F0, T_JC, TARGET)
    return oc, clock


def _run(oc, clock, seconds, dt=0.01):
    for _ in range(int(round(seconds / dt))):
        clock.t += dt
        oc.correct(np.eye(4))


def test_with_nothing_measured_the_reference_is_untouched(monkeypatch):
    oc, clock = _online(monkeypatch, _Sock(), _Conn())
    ref = _T((700, 50, 300), (10, 20, 30))
    _run(oc, clock, 1.0)
    np.testing.assert_allclose(oc.correct(ref), ref, atol=1e-12)


def test_the_socket_correction_is_the_reference_rebuilt_on_the_new_socket(monkeypatch):
    D = _T((0, 2.0, 0))                          # 2 mm across the axis, in the socket's own frame
    oc, clock = _online(monkeypatch, _Sock(T_S0 @ D))
    _run(oc, clock, 1.0)                         # 1 mm/s: half way after 1 s
    assert oc.applied()[2][0] == pytest.approx(1.0, abs=0.02)
    _run(oc, clock, 2.0)
    row, C0 = _T((-20, 1, 0), (2, 0, 0)), _Robot.T_tool0_fingertip @ F0
    ref = TARGET(T_S0) @ row @ inverse(C0)
    np.testing.assert_allclose(oc.correct(ref), TARGET(T_S0 @ D) @ row @ inverse(C0), atol=1e-9)
    np.testing.assert_allclose(oc.uncorrect(oc.correct(ref)), ref, atol=1e-9)


def test_along_the_axis_and_the_turn_about_it_are_not_taken(monkeypatch):
    D = _T((0, 0, 3.0), (0, 0, 40))              # 3 mm along the socket's axis, 40 deg about it: none observable
    oc, clock = _online(monkeypatch, _Sock(T_S0 @ D))
    _run(oc, clock, 5.0)
    assert oc.applied()[2] == pytest.approx((0.0, 0.0), abs=1e-6)


def test_an_estimate_past_the_limit_is_not_followed(monkeypatch):
    oc, clock = _online(monkeypatch, _Sock(T_S0 @ _T((0, 8.0, 0))))
    _run(oc, clock, 10.0)
    assert oc.applied()[2][0] == pytest.approx(0.0, abs=1e-9)


def test_too_few_agreeing_estimates_are_not_followed(monkeypatch):
    oc, clock = _online(monkeypatch, _Sock(T_S0 @ _T((0, 2.0, 0)), n=1))
    _run(oc, clock, 5.0)
    assert oc.applied()[2][0] == pytest.approx(0.0, abs=1e-9)


def test_the_hand_correction_is_the_reference_rebuilt_on_the_measured_grasp(monkeypatch):
    C0 = _Robot.T_tool0_fingertip @ F0
    E = _T((0, 1.5, -1.0))                       # across the connector's axis (+x), in its own frame
    T_tool0_plug = _Robot.T_tool0_fingertip @ P6.plug_from_belief(inverse(_Robot.T_tool0_fingertip) @ C0 @ E,
                                                                  T_JC, P6.PLUG)
    oc, clock = _online(monkeypatch, None, _Conn(T_tool0_plug))
    _run(oc, clock, 4.0)
    _T_s, F, _s, (h_mm, _h_deg) = oc.applied()
    assert h_mm == pytest.approx(np.hypot(1.5, 1.0), abs=1e-3)
    row = _T((-10, 0, 0))
    ref = TARGET(T_S0) @ row @ inverse(C0)
    np.testing.assert_allclose(oc.correct(ref), TARGET(T_S0) @ row @ inverse(C0 @ E), atol=1e-6)
    np.testing.assert_allclose(_Robot.T_tool0_fingertip @ F, C0 @ E, atol=1e-6)


class _Arm:
    dry_run = False

    def __init__(self):
        self.sent = []
        self.reference_correction = None

    def wrench(self):
        return np.zeros(6)

    def servo_l(self, T, dt, lookahead, gain):
        self.sent.append(np.array(T))


class _Shift:
    """A fixed correction: 3 mm along base y."""
    L = _T((0, 3, 0))

    def correct(self, T):
        return self.L @ T

    def uncorrect(self, T):
        return inverse(self.L) @ T


def test_the_controller_commands_the_corrected_reference_and_rebases_without_it_twice():
    arm = _Arm()
    adm = AdmittanceController(arm, {})
    ref = _T((600, 0, 300))
    adm._step(ref, 0.008)
    np.testing.assert_allclose(arm.sent[-1], ref)          # no correction set
    arm.reference_correction = _Shift()
    adm._step(ref, 0.008)
    np.testing.assert_allclose(arm.sent[-1], _Shift.L @ ref, atol=1e-12)
    # rebase on where the arm is (the corrected pose): the reference handed back, corrected, is that pose again
    back = adm.rebase(_Shift.L @ ref)
    adm._step(back, 0.008)
    np.testing.assert_allclose(arm.sent[-1], _Shift.L @ ref, atol=1e-12)
    adm.warmup(back, seconds=0.02)
    np.testing.assert_allclose(arm.sent[-1], _Shift.L @ ref, atol=1e-12)
