"""The socket re-measured by its landmarks (perception/socket_landmarks.py): its two cap screws, and the BNC
barrel's rim, found in a wrist image near where the 6D pose puts them, and the socket's translation solved from
them with the orientation held. Drawn scenes here; the real frames it was tuned on are a run's data."""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from cable_assembly.perception import socket_landmarks as SL  # noqa: E402

K = np.array([[641.7, 0.0, 631.0], [0.0, 641.3, 367.6], [0.0, 0.0, 1.0]])
SIZE = (720, 1280)


def socket_in_camera(shift_mm=(0.0, 0.0, 0.0)):
    """The socket as the wrist camera has it at the stand-off: ~166 mm away, its axis 54 deg off the line of sight,
    pointing up the image and toward the camera."""
    z = np.array([0.0, -0.809, -0.588])
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = np.array([0.012, 0.002, 0.166]) + np.asarray(shift_mm) / 1000.0
    return T


def _fill(img, T, centre_mm, radius_mm, value, ss=4):
    """A disc about the part's z, at `centre_mm`, drawn into the `ss`-times-enlarged image."""
    a = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
    pts = (np.asarray(centre_mm)[None, :] + radius_mm * np.stack([np.cos(a), np.sin(a), 0 * a], axis=1)) / 1000.0
    uv = SL._project(K, T, pts) * ss
    cv2.fillPoly(img, [np.round(uv).astype(np.int32)], float(value), lineType=cv2.LINE_AA)


def scene(T, part='socket', heads=(True, True), rim=True, ss=4):
    """The landmarks of `part` at pose T on a dark flange: grey head tops with dark hex sockets, a bright rim."""
    spec = SL.landmarks_for(part)
    big = np.full((SIZE[0] * ss, SIZE[1] * ss), 28.0, np.float32)
    if rim and spec.get('rim'):
        r = spec['rim']
        _fill(big, T, (0.0, 0.0, r['z']), r['r_out'], 150.0, ss)
        _fill(big, T, (0.0, 0.0, r['z']), r['r_in'], 45.0, ss)
    for top, on in zip(spec['screws'], heads):
        if on:
            _fill(big, T, top, SL.HEAD_D_MM / 2.0, 115.0, ss)
            _fill(big, T, top, SL.HEX_AF_MM / 2.0, 22.0, ss)
    img = cv2.resize(big, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_AREA)
    img += np.random.default_rng(0).normal(0.0, 1.5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_the_translation_is_recovered_with_the_orientation_held():
    truth = (-0.6, -3.2, 1.0)                                # the 2026-11-12 run's kind of error
    img = scene(socket_in_camera(truth))
    r = SL.fit(img, K, socket_in_camera())
    assert r['ok'], r['why']
    t = np.array(r['t_cam_mm'])
    assert np.linalg.norm(t[:2] - truth[:2]) < 0.3           # across the line of sight: a fraction of a pixel
    assert abs(t[2] - truth[2]) < 1.5                        # along it: from the screws' spacing, the weak one
    assert r['residual_px'] < 1.0
    assert r['rim'] is not None and r['rim']['used']
    np.testing.assert_allclose(r['T_cam_obj'][:3, :3], socket_in_camera()[:3, :3])     # never the orientation
    for s in r['screws']:
        assert s['contrast'] > 30


def test_no_correction_where_the_pose_was_right():
    r = SL.fit(scene(socket_in_camera()), K, socket_in_camera())
    assert r['ok'] and np.linalg.norm(r['t_cam_mm'][:2]) < 0.3 and abs(r['t_cam_mm'][2]) < 1.5


def test_screws_alone_are_enough():
    truth = (1.5, 2.0, -1.0)
    img = scene(socket_in_camera(truth), rim=False)
    r = SL.fit(img, K, socket_in_camera())
    assert r['ok'], r['why']
    assert not (r['rim'] or {}).get('used')
    assert np.linalg.norm(np.array(r['t_cam_mm'])[:2] - truth[:2]) < 0.3
    usb = SL.fit(scene(socket_in_camera(truth), part='usb_socket'), K, socket_in_camera(), part='usb_socket')
    assert usb['ok'], usb['why']
    assert np.linalg.norm(np.array(usb['t_cam_mm'])[:2] - truth[:2]) < 0.3


def test_no_answer_without_both_screws():
    T = socket_in_camera()
    one = SL.fit(scene(T, heads=(True, False)), K, T)
    assert not one['ok'] and 'screw head 1' in one['why'] and one['t_cam_mm'] is None
    none = SL.fit(np.full(SIZE, 30, np.uint8), K, T)
    assert not none['ok'] and 'not found' in none['why']
    assert not SL.fit(scene(T), K, T, part='plug')['ok']                               # a part with no landmarks


def test_a_screw_behind_the_held_plug_is_not_looked_for():
    T = socket_in_camera()
    img = scene(T)
    hidden = np.zeros(SIZE, bool)
    u, v = SL._project(K, T, np.array([SL.LANDMARKS['socket']['screws'][0]]) / 1000.0)[0]
    hidden[int(v) - 20:int(v) + 20, int(u) - 20:int(u) + 20] = True
    r = SL.fit(img, K, T, hidden=hidden)
    assert not r['ok'] and 'behind the held plug' in r['why']


def test_a_socket_further_off_than_the_search_is_left_alone():
    img = scene(socket_in_camera((0.0, -12.0, 0.0)))
    r = SL.fit(img, K, socket_in_camera(), search_mm=8.0)
    assert not r['ok']
    assert SL.fit(img, K, socket_in_camera(), search_mm=15.0)['ok']


def test_a_shadow_on_a_bright_face_is_not_a_screw():
    """What beat the real heads on contrast on the 2026-11-12 frames: a soft dark patch on something bright -- darker
    in the middle than round it, as a hex socket in a head is, but brighter still further out: no head top standing
    out of a dark flange."""
    T = socket_in_camera()
    img = np.full(SIZE, 170.0, np.float32)
    yy, xx = np.mgrid[:SIZE[0], :SIZE[1]]
    for top in SL.LANDMARKS['socket']['screws']:
        u, v = SL._project(K, T, np.array([top]) / 1000.0)[0]
        img -= 140.0 * np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * 6.0 ** 2))
    r = SL.fit(np.clip(img, 0, 255).astype(np.uint8), K, T)
    assert not r['ok'] and 'not found' in r['why']


def test_frames_are_fused_and_carried_to_the_base_frame():
    truth = np.array([-0.6, -3.2, 0.5])
    T_bc = np.eye(4)
    T_bc[:3, :3] = cv2.Rodrigues(np.array([0.3, -1.1, 0.4]))[0]
    T_bc[:3, 3] = [0.6, 0.1, 0.4]
    T_base_obj = T_bc @ socket_in_camera()
    img = scene(socket_in_camera(truth))
    m = SL.measure([img, None, img], K, [T_bc] * 3, T_base_obj, agree_mm=1.0)
    assert m['ok'] and m['n'] == 2 and m['of'] == 3
    d = np.linalg.inv(T_bc[:3, :3]) @ m['shift_m'] * 1000.0                             # back in the camera frame
    assert np.linalg.norm(d[:2] - truth[:2]) < 0.3
    np.testing.assert_allclose(m['T_base_obj'][:3, :3], T_base_obj[:3, :3])
    np.testing.assert_allclose(m['T_base_obj'][:3, 3] - T_base_obj[:3, 3], m['shift_m'])
    axis = T_base_obj[:3, 2]
    assert m['along_m'] == pytest.approx(float(m['shift_m'] @ axis))
    assert m['across_m'] == pytest.approx(float(np.linalg.norm(m['shift_m'] - m['along_m'] * axis)))
    assert not SL.measure([None, None, img], K, [T_bc] * 3, T_base_obj)['ok']            # one frame of three


def test_frames_that_disagree_give_no_answer():
    a = {'ok': True, 't_cam_mm': [0.0, -3.0, 0.0], 'residual_px': 0.5}
    b = {'ok': True, 't_cam_mm': [0.0, 0.0, 0.0], 'residual_px': 0.5}
    f = SL.fuse([a, b, dict(a, t_cam_mm=[0.0, -6.0, 0.0])], agree_mm=1.0)
    assert not f['ok'] and 'disagree' in f['why']
    g = SL.fuse([a, dict(a, t_cam_mm=[0.2, -3.1, 0.3]), dict(a, t_cam_mm=[-0.1, -2.9, -0.2])], agree_mm=1.0)
    assert g['ok'] and np.allclose(g['t_cam_mm'], [0.0, -3.0, 0.0], atol=0.2)


def test_the_plug_mask_covers_where_the_plug_is():
    V = np.array([[x, y, z] for x in (-0.007, 0.007) for y in (-0.007, 0.007) for z in (0.11, 0.15)])
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.05]
    m = SL.plug_mask(SIZE, K, T, V, grow_px=0)
    assert m[int(K[1, 2]), int(K[0, 2])] and not m[10, 10] and 0 < m.sum() < m.size / 20


def test_the_drawing_is_made_fitted_or_not():
    T = socket_in_camera()
    img = cv2.cvtColor(scene(socket_in_camera((0.0, -3.0, 0.0))), cv2.COLOR_GRAY2BGR)
    assert SL.draw(img, K, T, SL.fit(img, K, T)).shape[2] == 3
    assert SL.draw(img, K, T, {'ok': False, 'why': 'nothing', 'screws': []}) is not None
