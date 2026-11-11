"""perception/in_hand_rgb.py: the held plug's slide along its axis, from the wrist camera's colour
image alone.

What must hold: a plug drawn slid is found where it was drawn, to a millimetre; a pivot (not
fitted) is taken up as a slide that keeps the nose within a few millimetres; a plug further off than searched, a frame with no plug, and a belief whose axis
misses the pinch are refused rather than fitted; and the belief the app takes back describes the
plug the fit found (the inverse of pose6d.plug_from_belief)."""

import os

import numpy as np
import pytest

from cable_assembly.perception import in_hand_rgb as IH
from cable_assembly.perception.camera import Frame

# The iiwa bench's wrist camera on tool0 (hand_eye) and its colour intrinsics, and the fingertip.
T_TOOL0_CAM = np.array([[1.130e-02, 9.692e-01, 2.461e-01, -6.980e-02],
                        [-9.999e-01, 1.120e-02, 1.900e-03, 0.0],
                        [-9.000e-04, -2.462e-01, 9.692e-01, 1.750e-02],
                        [0.0, 0.0, 0.0, 1.0]])
K = np.array([[929.87, 0.0, 649.76], [0.0, 928.94, 355.75], [0.0, 0.0, 1.0]])
TIP = np.array([0.0, 0.0, 0.190])


def _plug_vertices():
    """A BNC-like round plug, z 0..44 mm: boot 10 mm across, collar 14.5, nose 13.8."""
    pts = []
    for z0, z1, d in ((0.0, 0.020, 0.010), (0.020, 0.038, 0.0145), (0.038, 0.044, 0.0138)):
        for z in np.linspace(z0, z1, 12):
            a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            pts.append(np.stack([d / 2 * np.cos(a), d / 2 * np.sin(a), np.full_like(a, z)], axis=1))
    return np.vstack(pts)


def _belief():
    """The plug pitched 20 deg in tool0's x-z plane with its axis through the fingertip, its nose
    ~25 mm past it (the iiwa bench's 70 deg pickup)."""
    a = np.radians(-20.0)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = TIP - R[:, 2] * 0.020
    return T


def _image(T_tool0_plug, verts, seed=0):
    """A wrist frame: a light, speckled bench, the plug dark where it is, the jaws (black) below
    their edge, blur and noise."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((720, 1280, 3), 205, np.uint8)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    for _ in range(40):                                        # specks and scratches on the bench
        c = tuple(int(v) for v in rng.integers([0, 0], [1280, 720]))
        cv2.circle(img, c, int(rng.integers(1, 4)), (90, 90, 90), -1)
    sil = IH.Silhouette(*IH.radius_profile(verts))
    m = sil.mask(np.linalg.inv(T_TOOL0_CAM) @ T_tool0_plug, K, img.shape[:2])
    img[m > 0] = (40, 40, 45)
    img[m > 0][::7] = (170, 170, 175)                          # a few highlights on the metal
    ref = IH.InHandRefiner(verts, T_TOOL0_CAM, TIP)
    img[ref.jaw_row(K) + 5:, 450:850] = (25, 25, 25)           # the fingers
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return Frame(img, K, None, 0.0)


@pytest.fixture(scope='module')
def verts():
    return _plug_vertices()


def _nose(T):
    return (T @ [0.0, 0.0, 0.044, 1.0])[:3]


@pytest.mark.parametrize('slide_mm', [0.0, -8.0, 6.0, -12.0])
def test_a_slid_plug_is_found_where_it_is(verts, slide_mm):
    T0 = _belief()
    truth = IH.moved(T0, slide_mm / 1000.0, 0.0, TIP)
    r = IH.InHandRefiner(verts, T_TOOL0_CAM, TIP).refine([_image(truth, verts, seed=k) for k in range(3)], T0)
    assert r['ok'], r['why']
    assert abs(1000 * r['slide'] - slide_mm) <= 1.0
    assert r['iou'] > 0.9
    assert np.linalg.norm(_nose(r['T_tool0_plug']) - _nose(truth)) < 0.001


@pytest.mark.parametrize('slide_mm,pivot_deg', [(0.0, 3.0), (0.0, -4.0), (5.0, 4.0)])
def test_a_pivot_is_taken_up_as_a_slide_that_keeps_the_nose_close(verts, slide_mm, pivot_deg):
    """Pivot is not fitted (one view cannot tell it from a slide: the module docstring): a few
    degrees of it come out as a slide, and the nose stays within 3 mm of where it is."""
    T0 = _belief()
    truth = IH.moved(T0, slide_mm / 1000.0, np.radians(pivot_deg), TIP)
    r = IH.InHandRefiner(verts, T_TOOL0_CAM, TIP).refine([_image(truth, verts, seed=k) for k in range(3)], T0)
    assert r['ok'], r['why']
    assert r['pivot'] == 0.0
    assert np.linalg.norm(_nose(r['T_tool0_plug']) - _nose(truth)) < 0.003


def test_further_off_than_searched_is_refused(verts):
    T0 = _belief()
    truth = IH.moved(T0, 0.022, 0.0, TIP)
    ref = IH.InHandRefiner(verts, T_TOOL0_CAM, TIP, slide_range_m=0.010)
    r = ref.refine([_image(truth, verts)], T0)
    assert not r['ok']
    assert r['T_tool0_plug'] is not None and np.allclose(r['T_tool0_plug'], T0)


def test_no_plug_in_view_is_refused(verts):
    import cv2
    T0 = _belief()
    frame = _image(T0, verts)
    blank = np.full_like(frame.color, 205)
    blank = cv2.GaussianBlur(blank, (3, 3), 0)
    r = IH.InHandRefiner(verts, T_TOOL0_CAM, TIP).refine([Frame(blank, K, None, 0.0)], T0)
    assert not r['ok']


def test_a_belief_off_the_pinch_is_refused_not_fitted(verts):
    """Off the pinch, a slide fits the image as well but describes another plug (the module
    docstring): refused, said why."""
    T0 = _belief()
    off = T0.copy()
    off[:3, 3] += np.cross([0, 1, 0], T0[:3, 2]) * 0.010       # 10 mm across, in the pitch plane
    r = IH.InHandRefiner(verts, T_TOOL0_CAM, TIP).refine([_image(T0, verts)], off)
    assert not r['ok'] and 'pinch' in r['why'] and not r['per_frame']


def test_the_belief_taken_back_describes_the_fitted_plug():
    """bnc_assembly turns the fitted plug pose back into a belief; pose6d.plug_from_belief of it
    must give the same plug."""
    from cable_assembly.perception import pose6d as P6
    from cable_assembly.transforms import inverse
    part = P6.PLUG
    T_tool0_tip = np.eye(4)
    T_tool0_tip[:3, 3] = TIP
    T_jc = np.eye(4)
    T_jc[0, 3] = 0.0457
    T_plug = IH.moved(_belief(), 0.004, np.radians(-2.0), TIP)
    T_ftip_conn = inverse(T_tool0_tip) @ T_plug @ P6.T_cord_junction(part) @ T_jc
    np.testing.assert_allclose(T_tool0_tip @ P6.plug_from_belief(T_ftip_conn, T_jc, part), T_plug, atol=1e-9)


GRASP_PNG = os.path.join(os.path.dirname(__file__), '..', 'sim', 'data', 'grasp_images', '20261102_193648',
                         'grasp_000_grasp_206counts_ok.png')


@pytest.mark.skipif(not os.path.isfile(GRASP_PNG), reason='the 2026-11-02 bench grasp image is not here')
def test_the_2026_11_02_bench_grasp():
    """The real plug after the 19:36 grasp, from the belief with belief_offset zero (its axis
    through the pinch): a sharp fit near the belief -- it was ~1 mm out along the plug."""
    import cv2
    from cable_assembly import config as C
    from cable_assembly.perception import pose6d as P6
    from cable_assembly.robot import tool_geometry
    from cable_assembly.skills.pick import fingertip_in_connector
    from cable_assembly.transforms import from_cfg, inverse
    # The grasp as it was made: the fingertip at 190 mm then (182 since that evening).
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pickup.belief_offset.xyz_mm=[0,0,0]',
                                  'fingertip_grasp.xyz_mm=[0,0,190]'])
    part = P6.plug_part(cfg)
    mesh = P6.cad_mesh(cfg, part)
    if mesh is None:
        pytest.skip('no RAG-6D CAD here')
    verts, _ = tool_geometry.read_ply(mesh[0])
    T_tool0_tip = from_cfg(cfg.section('fingertip_grasp'))
    T_jif = from_cfg(cfg.section('junction_in_fingertip'))
    grip = fingertip_in_connector(cfg)
    T_jc = inverse(T_jif) @ from_cfg(cfg.get_path('estimation.initial_connector_in_fingertip'))
    T_jc[1:3, 3] = 0.0
    T_prior = T_tool0_tip @ P6.plug_from_belief(inverse(grip) @ T_jc, T_jc, part)
    ref = IH.InHandRefiner(np.asarray(verts) * mesh[1], from_cfg(cfg.section('hand_eye')), T_tool0_tip[:3, 3])
    r = ref.refine([Frame(cv2.imread(GRASP_PNG), K, None, 0.0)], T_prior)
    assert r['ok'], r['why']
    assert r['iou'] > 0.9
    assert abs(1000 * r['slide']) < 4.0
