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


def test_the_socket_is_not_estimated_from_a_view_the_tool_hides_it_in(monkeypatch):
    """pose6d.online.socket.min_visible (2026-11-12: at the contact the gripper and the plug hid half the socket,
    and its single-view estimates there led the insertion 8.5 mm off it): a frame with too little of the
    socket's outline clear of the tool's mask is not sent to the estimator, and the agreed pose stays."""
    import threading
    from types import SimpleNamespace
    K = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]])
    size = (1280, 720)
    disc = np.array([[0.03 * np.cos(a), 0.03 * np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 40, endpoint=False)])
    T_sock = _T((0, 0, 300))                                 # 0.3 m in front of the camera, on its axis
    clear = np.zeros((720 // 4 + 1, 1280 // 4 + 1), bool)
    top_half = clear.copy()
    top_half[:360 // 4] = True                               # the tool over the image's top half
    assert PP.visible_fraction(K, size, clear, np.eye(4), disc + T_sock[:3, 3]) == pytest.approx(1.0, abs=0.05)
    assert PP.visible_fraction(K, size, top_half, np.eye(4), disc + T_sock[:3, 3]) == pytest.approx(0.5, abs=0.08)
    assert PP.visible_fraction(K, size, ~clear, np.eye(4), disc + T_sock[:3, 3]) == 0.0
    assert PP.visible_fraction(K, size, clear, np.eye(4), disc - T_sock[:3, 3]) == 0.0      # behind the camera

    asked = []

    class _Service:
        def estimate(self, part, frames, **kw):
            asked.append(kw.get('prior'))
            return {'found': True, 'T_base_obj': T_sock @ _T((0, 1.0, 0)), 'score': 2.0}

    monkeypatch.setattr(P6, 'service', lambda cfg: _Service())
    mask = [top_half]
    monkeypatch.setattr(PP, 'tool_mask', lambda robot, K_, size_: mask[0])
    tr = object.__new__(PP.SocketTracker)
    tr.__dict__.update(cfg=None, robot=SimpleNamespace(held=None), camera=None, part=P6.SOCKET, _lock=threading.Lock(),
                       _epoch=0, T_base_plug=T_sock, min_visible=0.8, visible=None, _pts=disc, _mask=(None, None),
                       _hidden_said=None, stats={'estimates': 0, 'accepted': 0}, use_depth=True, depth_min_m=0.07,
                       request={}, history=7, agree_m=0.003, agree_rad=np.radians(5.0), base=[], last=None,
                       _version=0)
    frame = SimpleNamespace(color=np.zeros((720, 1280, 3), np.uint8), depth=np.ones((720, 1280), np.float32), K=K,
                            T_base_cam=np.eye(4))
    assert tr.step(frame) is True
    assert not asked and tr.measured() is None, 'half hidden: not estimated from'
    assert tr.stats.get('hidden') == 1 and tr.visible == pytest.approx(0.5, abs=0.08)
    assert np.allclose(tr.T_base_plug, T_sock), 'the pose stays the last agreed one'
    mask[0] = clear                                           # backed out: all of it in view
    tr._mask = (None, None)
    assert tr.step(frame) is True
    assert len(asked) == 1 and tr.measured()[2] == 1 and tr.stats['accepted'] == 1
    tr.min_visible = 0.0                                      # the gate off: every frame, as before
    mask[0] = top_half
    tr._mask = (None, None)
    assert tr.step(frame) is True and len(asked) == 2


def test_the_jaws_fix_two_dof_and_the_estimate_gives_the_other_four():
    """jaw_constrained (the user, 2026-11-13: "the gripper only constrains 2DOF"): flat pads closing along tool y
    hold the part on their mid-plane, its axis in their plane -- of a measured pose, the slide along the pads
    (x, z), the pivot about y and the turn about the part's own axis are taken; the shift along y and the axis's
    tilt out of the plane stay the commanded grasp's. A round part's turn about its axis is the grasp's too."""
    y = np.array([0.0, 1.0, 0.0])
    ref = _T((-15.0, 0.0, 223.0), (0.0, -70.0, 0.0))             # the plug's axis (its +X) in the x-z plane
    slide_pivot = _T((4.0, 0.0, -6.0)) @ ref @ _T((0, 0, 0))     # slid along the pads ...
    slide_pivot[:3, :3] = Rotation.from_rotvec(np.radians([0.0, 8.0, 0.0])).as_matrix() @ ref[:3, :3]   # ... and pivoted
    got, dy, tilt = PP.jaw_constrained(slide_pivot, ref, round_part=True)
    assert np.allclose(got, slide_pivot, atol=1e-9) and abs(dy) < 1e-12 and tilt < 1e-9     # all of it is taken
    # the same, read with an error the jaws do not allow: 3 mm along y, 6 deg out of the plane (about x)
    bad = slide_pivot.copy()
    bad[:3, :3] = Rotation.from_rotvec(np.radians([6.0, 0.0, 0.0])).as_matrix() @ slide_pivot[:3, :3]
    bad[1, 3] += 0.003
    got, dy, tilt = PP.jaw_constrained(bad, ref, round_part=True)
    assert dy == pytest.approx(0.003) and np.degrees(tilt) > 2.0
    assert got[1, 3] == pytest.approx(ref[1, 3]) and abs(got[:3, 0] @ y) < 1e-9                 # on the mid-plane, in it
    assert np.allclose(got[[0, 2], 3], slide_pivot[[0, 2], 3])                                  # the slide kept
    pivot = np.degrees(np.arctan2(np.cross(ref[:3, 0], got[:3, 0]) @ y, ref[:3, 0] @ got[:3, 0]))
    assert pivot == pytest.approx(8.0, abs=0.7)                                                 # the pivot kept
    # A PART WITH NO SYMMETRY ROLLS BETWEEN THE PADS, and the roll is seen: slid, pivoted and turned 12 deg about
    # its own axis, all of it is taken ...
    rolled = slide_pivot @ _T((0, 0, 0), (12.0, 0.0, 0.0))
    got, dy, tilt = PP.jaw_constrained(rolled, ref, round_part=False)
    assert np.allclose(got, rolled, atol=1e-9) and abs(dy) < 1e-12 and tilt < 1e-9
    # ... a round one's turn is the grasp's (no view tells it) ...
    got_round = PP.jaw_constrained(rolled, ref, round_part=True)[0]
    assert np.allclose(got_round, slide_pivot, atol=1e-9)
    # ... and with the error the jaws do not allow -- 3 mm along y, the axis tipped 6 deg out of the pads' plane
    # (about the in-plane direction square to it: no turn about the axis in that) -- it comes back, the turn kept
    bad = rolled.copy()
    tip = np.cross(y, rolled[:3, 0])
    bad[:3, :3] = Rotation.from_rotvec(np.radians(6.0) * tip / np.linalg.norm(tip)).as_matrix() @ rolled[:3, :3]
    bad[1, 3] += 0.003
    got, dy, tilt = PP.jaw_constrained(bad, ref, round_part=False)
    assert dy == pytest.approx(0.003) and np.degrees(tilt) == pytest.approx(6.0, abs=1e-6)
    assert got[1, 3] == pytest.approx(ref[1, 3]) and abs(got[:3, 0] @ y) < 1e-9
    assert np.allclose(got[:3, :3], rolled[:3, :3], atol=1e-9)


def test_a_held_plugs_estimate_is_asked_as_a_jaw_fit(monkeypatch):
    """pose6d.in_hand.jaw_fit: held, each request carries `jaw` -- the tool frame in the camera's and the search's
    ranges -- so the server searches the jaws' free DOF alone; lying on the bench, or with it off, it does not."""
    import threading
    from types import SimpleNamespace
    T_cam = _T((-62.0, -8.0, 30.0), (14.0, 0.0, 90.0))
    tool = _T((500.0, 100.0, 300.0), (180.0, 0.0, 20.0))
    asked = []

    class _Service:
        def estimate(self, part, frames, prior=None, **kw):
            asked.append(kw)
            return {'found': False}

    monkeypatch.setattr(P6, 'service', lambda cfg: _Service())
    tr = object.__new__(PP.ConnectorTracker)
    tr.__dict__.update(cfg=None, robot=SimpleNamespace(T_tool0_cam=T_cam), part=P6.USB, _lock=threading.Lock(),
                       use_depth=False, depth_min_m=0.07, request={'refine_only': True}, jaw_constraint=True,
                       jaw_fit={'slide_mm': 25.0, 'pivot_deg': 30.0}, T_tool0_plug=_T((0, 0, 200.0)), T_base_plug=None,
                       stats={'estimates': 0, 'accepted': 0}, last=None, _epoch=0, held=[], history=9)
    frame = SimpleNamespace(T_base_cam=tool @ T_cam, depth=np.zeros((4, 4)))
    assert tr.step(frame) is True
    jaw = asked[-1]['jaw']
    assert jaw['slide_mm'] == 25.0 and jaw['pivot_deg'] == 30.0
    assert np.allclose(np.asarray(jaw['T_cam_tool']), inverse(T_cam), atol=1e-9)     # the camera is on the flange
    tr.T_tool0_plug, tr.T_base_plug = None, tool @ _T((0, 0, 200.0))                 # lying still: a plain refine
    assert tr.step(frame) is True and 'jaw' not in asked[-1]
    tr.T_tool0_plug, tr.jaw_fit = _T((0, 0, 200.0)), None                            # off
    assert tr.step(frame) is True and 'jaw' not in asked[-1]


def test_the_in_hand_belief_is_updated_from_the_sockets_views(monkeypatch):
    """ConnectorTracker.from_views (the user, 2026-11-12: "when socket is viewed from multiple angles, also update
    the connector in-hand pose"): one estimate a view, the agreeing ones taken with the jaws' fixed DOF kept; too
    few agreeing, the belief stays."""
    import threading
    from types import SimpleNamespace
    T_tf = _T((0, 0, 184.1))
    F0 = inverse(T_tf) @ _T((-15.0, 0.0, 223.0), (0.0, -70.0, 0.0))        # the belief, connector in the fingertip frame
    T_jc = np.eye(4)
    T_cam = _T((-62.0, -8.0, 30.0))
    true_plug = _T((3.0, 2.0, -5.0)) @ T_tf @ P6.plug_from_belief(F0, T_jc, P6.PLUG)   # 2 mm of it along y: not real
    replies = []

    class _Service:
        def estimate(self, part, frames, prior=None, **kw):
            tool = frames[0].T_base_cam @ inverse(T_cam)
            ok = replies.pop(0)
            return {'found': ok, 'T_base_obj': tool @ true_plug, 'score': 2.0}

    monkeypatch.setattr(P6, 'service', lambda cfg: _Service())
    tr = object.__new__(PP.ConnectorTracker)
    tr.__dict__.update(cfg=None, robot=SimpleNamespace(T_tool0_cam=T_cam), part=P6.PLUG, _lock=threading.Lock(),
                       update_belief=True, use_depth=False, depth_min_m=0.07, request={}, agree_m=0.003,
                       agree_rad=np.radians(5.0), min_estimates=3, max_corr_m=0.015, max_corr_rad=np.radians(20.0),
                       jaw_constraint=True, jaw_ref=F0, last_measure={})
    views = [(None, SimpleNamespace(T_base_cam=_T((500 + 30 * k, 100, 300), (180.0, 0.0, 10.0 * k)) @ T_cam, depth=None))
             for k in range(5)]
    replies[:] = [True, True, False, True, True]
    F = tr.from_views(views, F0, T_tf, T_jc, label='test')
    moved = inverse(T_tf @ F0) @ (T_tf @ F)
    assert tr.last_measure['outcome'] == 'taken' and tr.last_measure['jaw_shift_mm'] == pytest.approx(2.0, abs=0.05)
    C0, C1 = T_tf @ F0, T_tf @ F
    assert C1[0, 3] - C0[0, 3] == pytest.approx(0.003, abs=1e-6) and C1[2, 3] - C0[2, 3] == pytest.approx(-0.005, abs=1e-6)
    assert C1[1, 3] == pytest.approx(C0[1, 3], abs=1e-9), 'the shift along the closing axis is not taken'
    assert moved is not None
    replies[:] = [True, False, False, True, False]                 # two views only: not enough
    assert np.allclose(tr.from_views(views, F0, T_tf, T_jc, label='test'), F0) and tr.last_measure['outcome'] == 'none'
