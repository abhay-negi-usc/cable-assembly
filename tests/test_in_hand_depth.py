"""perception/in_hand_depth.py: the held plug's axis and tip from depth alone, on a synthetic depth image of a box plug
whose metal shell reads no depth (as the D435i's does not, 2026-11-18)."""

import numpy as np

from cable_assembly.perception import in_hand_depth as ID

K = np.array([[1370.0, 0.0, 960.0], [0.0, 1370.0, 540.0], [0.0, 0.0, 1.0]])
SHAPE = (1080, 1920)
JUNCTION_Z, MATING_Z = 0.4212, 0.4692       # the USB plug's CAD: its rear and its mating face along z (m)
BODY = (0.436, 0.460, 0.016, 0.008)         # the overmold: z from, z to, width (x), thickness (y)
RELIEF = (0.4212, 0.436, 0.009, 0.008)      # the strain relief behind it


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def _plug(tilt_deg=45.0):
    """T_tool0_plug: the plug's axis (CAD z) tilted from tool z toward -x, its tip ~220 mm out (the hand load's), and
    turned about its own axis so its WIDE face (CAD y, the 8 mm thickness, along the line of sight) faces the camera,
    as hand-loaded on the bench (the shell's holes in view)."""
    T = np.eye(4)
    T[:3, :3] = _rot([0, 1, 0], np.radians(-tilt_deg))
    tip = np.array([-0.039, 0.0, 0.218])
    T[:3, 3] = tip - T[:3, :3] @ [0, 0, MATING_Z]
    cam = np.linalg.inv(_camera())[:3, 3]
    v = T[:3, :3].T @ (tip - cam)                       # the line of sight in the plug's frame
    T[:3, :3] = T[:3, :3] @ _rot([0, 0, 1], np.arctan2(-v[0], v[1]))
    T[:3, 3] = tip - T[:3, :3] @ [0, 0, MATING_Z]
    return T


def _camera():
    """T_cam_tool0: the camera 75 mm off the flange, looking along tool z (as the D435i on its mount)."""
    T_tool0_cam = np.eye(4)
    T_tool0_cam[:3, :3] = _rot([1, 0, 0], np.radians(20.0))      # tipped toward the plug
    T_tool0_cam[:3, 3] = [0.0, 0.075, 0.0]
    return np.linalg.inv(T_tool0_cam)


def _box_surface(z0, z1, w, t, n=60000, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    pts = []
    for ax, s in ((0, w / 2), (0, -w / 2), (1, t / 2), (1, -t / 2)):
        p = np.c_[rng.uniform(-w / 2, w / 2, n), rng.uniform(-t / 2, t / 2, n), rng.uniform(z0, z1, n)]
        p[:, ax] = s
        pts.append(p)
    return np.vstack(pts)


def _depth(T_tool0_plug, T_cam_tool0, background_m=0.45, noise_m=0.0005):
    """A z-buffered depth image of the plug's overmold and strain relief (the shell: none) over a far plane."""
    D = np.full(SHAPE, background_m)
    P = np.vstack([_box_surface(*BODY), _box_surface(*RELIEF)])
    T = T_cam_tool0 @ T_tool0_plug
    C = P @ T[:3, :3].T + T[:3, 3]
    C[:, 2] += np.random.default_rng(1).normal(0.0, noise_m, len(C))
    uv = (K @ (C / C[:, 2:3]).T).T[:, :2].round().astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < SHAPE[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < SHAPE[0])
    for (u, v), z in zip(uv[ok], C[ok, 2]):
        if z < D[v, u]:
            D[v, u] = z
    return D


def _true_tip(T):
    return T[:3, :3] @ [0, 0, MATING_Z] + T[:3, 3]


def test_the_axis_and_the_tip_are_found_from_a_belief_off_by_10_deg_and_8_mm():
    truth, cam = _plug(45.0), _camera()
    D = _depth(truth, cam)
    prior = truth.copy()                                # turned 10 deg about the tip (the CAD origin is 445 mm back)
    prior[:3, :3] = _rot([1, 0, 0], np.radians(10.0)) @ truth[:3, :3]
    prior[:3, 3] = _true_tip(truth) - prior[:3, :3] @ [0, 0, MATING_Z] + [0.005, -0.004, 0.005]
    s = ID.settings({'front_to_tip_mm': 9.2, 'surface_to_axis_mm': 4.0})   # the box's: no taper at its front
    e = ID.estimate(D, K, cam, prior, JUNCTION_Z, MATING_Z, s)
    assert 'why' not in e, e.get('why')
    assert np.degrees(np.arccos(e['axis'] @ truth[:3, 2])) < 2.0
    assert np.linalg.norm(e['tip'] - _true_tip(truth)) < 0.002
    assert e['n'] >= s['min_points'] and e['elong'] > s['min_elong']


def test_a_belief_far_off_finds_no_plug_and_says_why():
    truth, cam = _plug(45.0), _camera()
    D = _depth(truth, cam)
    prior = truth.copy()
    prior[:3, 3] += [0.06, 0.0, 0.0]                    # 60 mm to the side: nothing within radius_mm of its axis
    e = ID.estimate(D, K, cam, prior, JUNCTION_Z, MATING_Z, ID.settings({}))
    assert 'why' in e and 'depth points' in e['why']


def test_the_gates_refuse_a_tip_far_from_the_belief():
    truth, cam = _plug(45.0), _camera()
    D = _depth(truth, cam)
    s = ID.settings({'front_to_tip_mm': 9.2, 'max_tip_mm': 0.5})
    prior = truth.copy()
    prior[:3, 3] += prior[:3, :3] @ [0.0, 0.0, 0.004]   # 4 mm along its axis: inside the region, past max_tip_mm
    e = ID.estimate(D, K, cam, prior, JUNCTION_Z, MATING_Z, s)
    assert 'why' in e and 'tip' in e['why']


def test_settings_read_the_config_block_and_ignore_the_loaders_metric_copies():
    s = ID.settings({'radius_mm': 25, 'radius_m': 0.025, 'tip_from': 'END', 'depth_range_mm': [90, 500]})
    assert s['radius_mm'] == 25.0 and 'radius_m' not in s and s['tip_from'] == 'end'
    assert s['depth_range_mm'] == (90.0, 500.0)
