"""perception/in_hand_fit.py: the held plug's slide, shift and pivot fitted with the gripper's CAD.

Synthetic wrist images drawn from the bench's own geometry (the D415 on the flange, the pads at the
fingertip, the plug pitched 70 deg in the jaws): a known pose is put in, the fit must give it back;
and what the pads rule in and out."""

import types

import numpy as np

from cable_assembly.perception import in_hand_fit as F

# The bench (2026-11-05): tool0 -> the D415's colour frame, tool0 -> the closed pads' tips, the camera's K.
T_TOOL0_CAM = np.array([[0.0113, 0.9692, 0.2461, -0.0698], [-0.9999, 0.0112, 0.0019, 0.0],
                        [-0.0009, -0.2462, 0.9692, 0.0175], [0.0, 0.0, 0.0, 1.0]])
T_TOOL0_FTIP = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.182], [0.0, 0.0, 0.0, 1.0]])
K = np.array([[929.9, 0.0, 649.8], [0.0, 928.9, 355.7], [0.0, 0.0, 1.0]])
# The belief at the 70 deg pickup: the plug's CAD z (toward the mating face) 20 deg off tool z, in x-z.
_c, _s = np.cos(np.radians(20.0)), np.sin(np.radians(20.0))
T_PRIOR = np.array([[_c, 0.0, -_s, 0.0403], [0.0, 1.0, 0.0, 0.0], [_s, 0.0, _c, 0.0566], [0.0, 0.0, 0.0, 1.0]])
# A BNC-like plug, CAD z (m) -> radius: the boot from the junction, the collar, the nose.
PROFILE = [(0.1086, 0.130, 0.0055), (0.130, 0.142, 0.0073), (0.142, 0.1524, 0.0066)]
BOOT_D = 0.011


def plug_vertices():
    a = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    pts = []
    for z0, z1, r in PROFILE:
        for z in np.arange(z0, z1 + 1e-9, 0.0005):
            pts += [[r * np.cos(t), r * np.sin(t), z] for t in a]
    return np.array(pts)


def fitter(**kw):
    return F.InHandCadFit(plug_vertices(), T_TOOL0_CAM, T_TOOL0_FTIP, **kw)


def boot_counts(fit):
    """The Robotiq position at which the bench's pads are BOOT_D apart."""
    from cable_assembly.robot import gripper_kinematics
    return int(round(float(gripper_kinematics.counts_from_gap(BOOT_D - fit.counts_gap_offset))))


def wrist_image(fit, T_plug, gap_seen, seed=0, plug=True):
    """The bench seen by the wrist camera: a light, slightly noisy bench, the pink pads `gap_seen` apart,
    and the dark plug with its cable at T_plug."""
    rng = np.random.default_rng(seed)
    img = np.clip(205 + rng.normal(0, 2.0, (720, 1280, 3)), 0, 255).astype(np.uint8)
    T_cam_tool0 = np.linalg.inv(T_TOOL0_CAM)
    pads = fit.pads.mask(gap_seen, T_cam_tool0, K, (720, 1280))
    img[pads > 0] = (140, 110, 230)
    if plug:
        m = fit.sil.mask(T_cam_tool0 @ T_plug, K, (720, 1280))          # the plug and its cable stub
        img[m > 0] = (30, 30, 34)
    return types.SimpleNamespace(color=img, K=K, depth=None)


def test_the_pads_are_found_from_their_edges():
    fit = fitter(gap_offset_m=0.0)
    # the pads alone: their CAD meets their edges where it was drawn, at the gap it was drawn at
    reg = fit.register(wrist_image(fit, T_PRIOR, BOOT_D, plug=False).color, K)
    assert reg is not None and reg['residual_px'] <= 1.0 and max(abs(v) for v in reg['shift']) <= 1
    assert abs(reg['gap'] - BOOT_D) < 0.0005
    # with the plug between them it hides the far half of each pad's inner face: the image's gap reads
    # WIDE -- which is why the fit takes the gap from the gripper's counts
    reg = fit.register(wrist_image(fit, T_PRIOR, BOOT_D).color, K)
    assert reg['gap'] > BOOT_D + 0.001 and max(abs(v) for v in reg['shift']) <= 1


def test_a_known_pose_comes_back():
    fit = fitter()
    T_true = fit.moved(T_PRIOR, 0.003, 0.001, np.radians(2.0))
    r = fit.refine([wrist_image(fit, T_true, BOOT_D)], T_PRIOR, counts=boot_counts(fit))
    assert r['ok'], r['why']
    assert r['gap_from'] == 'counts' and abs(r['gap'] - BOOT_D) < 0.0003
    assert abs(r['slide'] - 0.003) <= 0.001 and abs(np.degrees(r['pivot']) - 2.0) <= 1.5
    assert abs(r['shift'] - 0.001) <= 0.0015
    # the nose -- what has to find the socket -- within a millimetre of where it is
    nose = np.array([0.0, 0.0, 0.1524, 1.0])
    assert np.linalg.norm((r['T_tool0_plug'] @ nose - T_true @ nose)[:3]) < 0.001
    # the roll about the plug's axis is the belief's, never fitted
    assert np.allclose(r['T_tool0_plug'][:3, 1], T_true[:3, 1], atol=1e-6)


def test_the_pads_rule_out_what_they_cannot_be_holding():
    fit = fitter()
    # THE BELIEF ITSELF is not a pinch: it puts the collar's first 3 mm between pads closed on the boot --
    # on the bench's grasps the fit moved it out by +1..+6 mm, which is what this rule asks
    assert not fit.held(T_PRIOR, BOOT_D)[0]
    on_boot = fit.moved(T_PRIOR, 0.004, 0.0, 0.0)
    assert fit.held(on_boot, BOOT_D)[0]
    # slid 10 mm back into the hand, the collar (14.6 mm) is between pads 11 mm apart: impossible
    ok, why = fit.held(fit.moved(on_boot, -0.010, 0.0, 0.0), BOOT_D)
    assert not ok and 'between pads' in why
    # ... but with the pads on the collar (197 counts: ~15 mm), that is just the grasp
    assert fit.held(fit.moved(on_boot, -0.010, 0.0, 0.0), 0.0146)[0]
    # and pads closed on the collar cannot be holding the thin boot only
    ok, why = fit.held(on_boot, 0.0146)
    assert not ok and 'touch too little' in why


def test_the_belief_is_put_between_the_pads_first():
    fit = fitter()
    # a belief turned 3 deg out of the jaws' plane and 2 mm off their midplane
    R = F._rot(np.array([1.0, 0.0, 0.0]), np.radians(3.0))
    T = T_PRIOR.copy()
    T[:3, :3] = R @ T[:3, :3]
    T[1, 3] += 0.002
    T0, d_m, d_deg = fit.in_plane(T)
    y = fit.pads.closing
    assert abs(T0[:3, 2] @ y) < 1e-9, 'its axis parallel to the pads'
    assert abs((T0[:3, 3] - fit.pads.centre) @ y) < 1e-9, 'on their midplane'
    assert 0.0015 < d_m < 0.0025 and 2.5 < d_deg < 3.5
    # the three moves keep it there
    T1 = fit.moved(T0, 0.004, -0.002, np.radians(5.0))
    assert abs(T1[:3, 2] @ y) < 1e-9 and abs((T1[:3, 3] - fit.pads.centre) @ y) < 1e-9


def test_no_pads_no_fit():
    fit = fitter()
    f = wrist_image(fit, T_PRIOR, BOOT_D)
    f.color[:] = 205                                   # nothing pink in view
    r = fit.refine([f], T_PRIOR)
    assert not r['ok'] and 'pads' in r['why']
    assert np.allclose(r['T_tool0_plug'], fit.in_plane(T_PRIOR)[0])


def test_a_saved_fit_replays(tmp_path):
    fit = fitter()
    f = wrist_image(fit, T_PRIOR, BOOT_D)
    p = str(tmp_path / 'case.npz')
    F.save_case(p, [f], T_PRIOR, T_TOOL0_CAM, T_TOOL0_FTIP, counts=204)
    frames, T_prior, T_cam, T_ftip, counts = F.load_case(p)
    assert counts == 204
    assert np.array_equal(frames[0].color, f.color) and frames[0].depth is None
    assert np.allclose(T_prior, T_PRIOR) and np.allclose(T_cam, T_TOOL0_CAM) and np.allclose(T_ftip, T_TOOL0_FTIP)


def test_the_matching_is_shown_as_it_happens():
    """refine(progress=): the pads, the segmentation, each better candidate, the verdict -- each drawable."""
    fit = fitter()
    T_true = fit.moved(T_PRIOR, 0.003, 0.001, np.radians(2.0))
    states = []
    r = fit.refine([wrist_image(fit, T_true, BOOT_D)], T_PRIOR, counts=boot_counts(fit), progress=states.append)
    stages = [s['stage'] for s in states]
    assert stages[0] == 'pads' and 'segment' in stages and 'coarse' in stages and stages[-1] == 'done'
    assert states[-1]['result'] is r and r['ok']
    assert len(states[0]['reg']['edges']['top']) > 10 and len(states[0]['reg']['edges']['inner']) > 10
    for s in (states[0], states[stages.index('coarse')], states[-1]):
        img = fit.live_image(s)
        assert img.ndim == 3 and img.shape[1] == 1100 and img.dtype == np.uint8
