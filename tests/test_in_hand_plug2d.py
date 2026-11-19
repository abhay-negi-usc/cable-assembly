"""perception/in_hand_plug2d.py: the held plug's tip and axis from its 2D outline, and the tracker's use of it.

What must hold: a plug drawn (as a cylinder of the effective width, on a cord between the pads) at a known pose is
found there -- tip to ~1 mm, axis to ~1 deg -- from a belief several mm / deg off; its turn about its own axis is
the belief's; a view the gates refuse is no estimate, said why; and the ConnectorTracker takes it as a reply."""

import types

import numpy as np

from cable_assembly.perception import in_hand_plug2d as IP

# The D435i on the flange mount, as on run 20261113_180404 (its hand-eye and colour intrinsics, 1280x720), and the
# plug held by its cable as it hung there: its axis leaning back over the pads, the tip ~5 cm out.
K = np.array([[910.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]])
T_TOOL0_CAM = np.array([[0.011, 0.969, 0.246, -0.070], [-1.0, 0.011, 0.002, -0.002], [-0.001, -0.246, 0.969, 0.025],
                        [0.0, 0.0, 0.0, 1.0]])
AXIS = np.array([-0.684, 0.0, 0.729]) / np.linalg.norm([-0.684, 0.0, 0.729])
FTIP_Z = 0.182
MATING_Z = 0.4692
W_MM = 16.1


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _cam():
    T = T_TOOL0_CAM.copy()
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    return np.linalg.inv(T)                                   # T_cam_tool0


def _plug(tip, axis, roll=0.0):
    """T_tool0_plug: CAD +z along `axis`, its tip (MATING_Z) at `tip`, turned `roll` about the axis."""
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    R = np.c_[x, np.cross(z, x), z] @ _rpy(0.0, 0.0, roll)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tip, float) - R[:, 2] * MATING_Z
    return T


def _draw(T_cam_tool0, a, b, d_mm, img):
    """A cylinder from a to b (tool0, m), d_mm across, filled into `img` -- its silhouette, hull per slice."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = (b - a) / np.linalg.norm(b - a)
    u = np.cross(ax, [0.3, 1.0, 0.1])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    t = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    ring = (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)) * d_mm / 2000.0
    import cv2
    s = np.linspace(0.0, 1.0, 40)
    for s0, s1 in zip(s[:-1], s[1:]):
        P = np.r_[a + s0 * (b - a) + ring, a + s1 * (b - a) + ring]
        uv = np.array([IP.project(K, T_cam_tool0, p) for p in P])
        cv2.fillConvexPoly(img, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)


def _model_pts():
    """Points on a plug-sized box round the CAD axis's last 40 mm (its frame, m)."""
    g = np.linspace(-1.0, 1.0, 5)
    return np.array([[0.008 * a, 0.004 * b, z] for a in g for b in g for z in np.linspace(MATING_Z - 0.04, MATING_Z, 9)])


def _off(est_tip, tip):
    """(its error across the camera's line of sight to `tip`, m; its range error, a fraction): the bearing is the
    image's, exact; the range is the width's -- read behind the tip, where this plug is nearer the camera, which
    the effective width calibrated on the bench absorbs and a drawn band does not (~4% short here)."""
    o = np.linalg.inv(_cam())[:3, 3]
    r = (tip - o) / np.linalg.norm(tip - o)
    e = est_tip - tip
    return float(np.linalg.norm(e - (e @ r) * r)), float(np.linalg.norm(est_tip - o) / np.linalg.norm(tip - o) - 1.0)


def _scene(tip, axis):
    """(colour image BGR, the plug-and-cord mask) of a plug whose cord runs from the grip to 35 mm before `tip`."""
    Tc = _cam()
    grip = np.array([0.0, 0.0, FTIP_Z])
    tip = np.asarray(tip, float)
    mask = np.zeros((720, 1280), np.uint8)
    body = tip - 0.035 * np.asarray(axis, float) / np.linalg.norm(axis)
    _draw(Tc, grip, body, 5.5, mask)
    # the plug: a flat band square to the line of sight -- its outline's far end is the tip, its width as read
    # W_MM at the tip's range (the effective width the method is calibrated to; a drawn cylinder's end cap, seen
    # at 57 deg, would put the outline's end past the tip)
    import cv2
    ray = tip - np.linalg.inv(Tc)[:3, 3]
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    side = np.cross(ax, ray)
    side *= W_MM / 0.9 / 2000.0 / np.linalg.norm(side)     # widths() reads the 5th-95th percentile: 0.9 of it
    quad = np.array([IP.project(K, Tc, p) for p in (body + side, tip + side, tip - side, body - side)])
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
    color = np.full((720, 1280, 3), 235, np.uint8)
    color[mask > 0] = (30, 30, 30)
    return color, mask > 0


def test_hanging_out_of_the_pads_plane_tip_and_axis_from_the_grip():
    """Off the pads' mid-plane (as the D405 runs hung: the plug 20 deg out of it), the axis runs tip to grip."""
    axis = _rpy(np.radians(20.0), 0.0, 0.0) @ AXIS
    tip = np.array([0.0, 0.0, FTIP_Z]) + 0.055 * axis
    color, mask = _scene(tip, axis)
    off = _plug(tip + [0.006, 0.0, -0.005], _rpy(0, np.radians(8.0), 0) @ axis, roll=0.7)
    e = IP.estimate(color, mask, K, _cam(), FTIP_Z, off, MATING_Z, IP.settings({'width_mm': W_MM}))
    assert 'why' not in e, e.get('why')
    assert e['axis_from'] == 'grip'
    across, rng = _off(e['tip'], tip)
    assert across < 0.001 and abs(rng) < 0.06
    # the axis from the tip to the grip, 55 mm: the tip's range error (above) turns it a few degrees
    assert np.degrees(np.arccos(np.clip(e['axis'] @ axis, -1, 1))) < 8.0
    T = IP.pose(e['tip'], e['axis'], off, MATING_Z)
    assert np.allclose(T[:3, :3] @ [0, 0, MATING_Z] + T[:3, 3], e['tip'])
    assert np.allclose(T[:3, 2], e['axis'])
    # the turn about the axis is the belief's: the belief's x, brought onto the new axis by the least turn
    assert abs(T[:3, 0] @ off[:3, 0]) > 0.95


def test_in_the_pads_plane_seen_end_on_tip_measured_axis_from_the_belief():
    """In the pads' mid-plane, before the D435i (2.5 mm off it): no grip point -- the tip is still measured, the
    axis is the belief's in the plane of sight; with end_on 'refuse' there is no estimate."""
    tip = np.array([0.0, 0.0, FTIP_Z]) + 0.055 * AXIS
    color, mask = _scene(tip, AXIS)
    off = _plug(tip + [0.006, 0.0, -0.005], _rpy(0, np.radians(8.0), 0) @ AXIS)
    e = IP.estimate(color, mask, K, _cam(), FTIP_Z, off, MATING_Z, IP.settings({'width_mm': W_MM}))
    assert 'why' not in e, e.get('why')
    assert e['axis_from'] == 'belief'
    across, rng = _off(e['tip'], tip)
    assert across < 0.001 and abs(rng) < 0.06
    assert e['axis'] @ off[:3, 2] > np.cos(np.radians(1.0))
    assert 'end-on' in IP.estimate(color, mask, K, _cam(), FTIP_Z, off, MATING_Z,
                                   IP.settings({'width_mm': W_MM, 'end_on': 'refuse'}))['why']


def test_gates_refuse_and_say_why():
    axis = AXIS
    tip = np.array([0.0, 0.0, FTIP_Z]) + 0.055 * axis
    color, mask = _scene(tip, axis)
    s = IP.settings({'width_mm': W_MM})
    backwards = _plug(np.array([0.0, 0.0, FTIP_Z]) - 0.058 * axis, -axis)
    assert 'points' in IP.estimate(color, mask, K, _cam(), FTIP_Z, backwards, MATING_Z, s)['why']
    assert 'range' in IP.estimate(color, mask, K, _cam(), FTIP_Z, _plug(tip, axis), MATING_Z,
                                  dict(s, z_range_mm=(250.0, 300.0)))['why']
    assert 'mask' in IP.estimate(color, np.zeros_like(mask), K, _cam(), FTIP_Z, _plug(tip, axis), MATING_Z, s)['why']
    assert 'mask' in IP.estimate(np.full_like(color, (200, 40, 230)), mask, K, _cam(), FTIP_Z, _plug(tip, axis),
                                 MATING_Z, s)['why']                  # all of it saturated: the pads' pink


def test_prompts_box_round_the_belief():
    axis = AXIS
    tip = np.array([0.0, 0.0, FTIP_Z]) + 0.055 * axis
    T = _plug(tip, axis)
    box, pos, neg, f = IP.prompts(K, _cam(), T, _model_pts(), FTIP_Z, (720, 1280))
    uv = IP.project(K, _cam() @ T, [0.0, 0.0, MATING_Z])
    assert box[0] <= uv[0] <= box[2] and box[1] <= uv[1] <= box[3]
    assert len(pos) >= 2 and len(neg) == 2


def test_tracker_takes_it_as_a_reply(monkeypatch):
    from cable_assembly import config as C
    from cable_assembly.perception import pose6d as P6
    from cable_assembly.skills import pose6d_pick as PP
    cfg = C.load('usb_assembly', ['pose6d.in_hand.rgb2d.enabled=true',     # off in usb_assembly.yaml since 2026-11-18
                                  'pose6d.in_hand.depth_axis.enabled=false'])
    T_tool0_cam = np.linalg.inv(_cam())
    Tf = np.eye(4)
    Tf[2, 3] = FTIP_Z
    robot = types.SimpleNamespace(T_tool0_cam=T_tool0_cam, T_tool0_fingertip=Tf)
    tr = PP.ConnectorTracker(cfg, robot, None)
    assert tr.rgb2d is not None and tr.agree_m == 0.006
    axis = AXIS
    tip = np.array([0.0, 0.0, FTIP_Z]) + 0.055 * axis
    color, mask = _scene(tip, axis)
    monkeypatch.setitem(P6.MATING_Z, tr.part, MATING_Z)
    svc = types.SimpleNamespace(model_points=lambda part: _model_pts(),
                                segment=lambda frame, box, pos, neg: mask)
    tool = np.eye(4)
    tool[:3, 3] = [0.5, 0.1, 0.3]
    frame = types.SimpleNamespace(color=color, K=K)
    prior = tool @ _plug(tip + [0.004, 0.0, 0.003], axis)
    r = tr._rgb2d(svc, frame, tool, prior)
    assert r['found'] and r['method'] == 'rgb2d'
    got = np.linalg.inv(tool) @ r['T_base_obj']
    across, rng = _off(got[:3, :3] @ [0, 0, MATING_Z] + got[:3, 3], tip)
    assert across < 0.001 and abs(rng) < 0.06
    assert r['axis_from'] == 'belief'                 # in the pads' plane, before the D435i
    svc.segment = lambda frame, box, pos, neg: np.zeros_like(mask)
    r = tr._rgb2d(svc, frame, tool, prior)
    assert not r['found'] and tr.rgb2d_last['why']
