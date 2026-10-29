"""The hand-eye solve, held to a KNOWN camera mount through the real chain: markers rendered from
the true mount, detected, PnP'd, solved -- and the app's sweep on the simulated arm.

What the tolerances are: with a 38.8 mm marker 25 cm away, 0.2 mm / 0.02 deg of arm-pose noise
and image noise, the mount comes back within ~1 mm and ~0.5 deg, and a marker's base pose --
what the camera's view of the bench actually depends on -- within ~0.5 mm.
"""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cable_assembly import config as C
from cable_assembly.perception import hand_eye as he
from cable_assembly.robot.kinematics import iiwa14
from cable_assembly.transforms import inverse, pose_error, translation_matrix, xyzrpy_to_matrix

K = np.array([[910.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]])
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
X_TRUE = xyzrpy_to_matrix([0.03, -0.075, 0.045], np.radians([1.5, -2.0, 91.0]))
SIZE = 0.0388


def _scene(axes, tilt_deg=15.0, seed=1):
    rng = np.random.default_rng(seed)
    T0 = iiwa14.fk(np.radians([0, 20, 0, -80, 0, 80, 0]))
    cam0 = T0 @ X_TRUE
    markers = {7: cam0 @ xyzrpy_to_matrix([0.07, 0.0, 0.25], np.radians([180, 0, 15])),
               3: cam0 @ xyzrpy_to_matrix([-0.08, 0.0, 0.26], np.radians([180, 0, -20]))}
    pivot = X_TRUE[:3, :3] @ [0.0, 0.0, 0.25] + X_TRUE[:3, 3]
    det = he.make_detector(DICT)
    views = []
    for ax, k in axes:
        R = np.eye(4)
        R[:3, :3] = Rotation.from_rotvec(np.radians(tilt_deg * k) * np.asarray(ax, float)
                                         / np.linalg.norm(ax)).as_matrix()
        T = T0 @ translation_matrix(pivot) @ R @ translation_matrix(-pivot)
        img = he.render_markers(K, 1280, 720, [(m, SIZE, inverse(T @ X_TRUE) @ Tm)
                                               for m, Tm in markers.items()], DICT)
        img = np.clip(img + rng.normal(0.0, 3.0, img.shape), 0, 255).astype(np.uint8)
        noise = xyzrpy_to_matrix(rng.normal(0.0, 2e-4, 3), rng.normal(0.0, np.radians(0.02), 3))
        views.append({'T_base_tool0': T @ noise, 'corners': he.detect_corners(det, img)})
    return views, markers


SWEEP = [((1, 0, 0), 1), ((1, 0, 0), -1), ((0, 1, 0), 1), ((0, 1, 0), -1), ((1, 1, 0), 1),
         ((1, -1, 0), -1), ((-1, 1, 0), 1), ((1, 1, 0), -1), ((0, 0, 1), 2), ((0, 0, 1), -2),
         ((1, 0, 0), 0)]


@pytest.mark.parametrize('seed', [1, 2])
def test_the_solve_recovers_a_known_mount(seed):
    views, markers = _scene(SWEEP, seed=seed)
    r = he.solve(views, K, np.zeros(5), {7: SIZE, 3: SIZE})
    lin, ang = pose_error(r['X'], X_TRUE)
    assert lin < 1.5e-3 and np.degrees(ang) < 0.6, (lin, np.degrees(ang))
    for m, Tm in markers.items():
        assert abs(r['markers'][m][2, 3] - Tm[2, 3]) < 1e-3, 'the bench height reading'
    assert r['rms_px'] < 1.0 and r['second_axis_deg'] > 3.0


def test_views_turning_about_one_axis_are_refused():
    views, _ = _scene([((1, 0, 0), k) for k in (-1, -0.5, 0, 0.5, 1)])
    with pytest.raises(ValueError, match='one axis'):
        he.solve(views, K, np.zeros(5), {7: SIZE, 3: SIZE})


def test_too_few_views_is_an_error():
    views, _ = _scene(SWEEP[:2])
    with pytest.raises(ValueError, match='>= 3'):
        he.solve(views, K, np.zeros(5), {7: SIZE, 3: SIZE})


def test_the_app_sweep_on_the_simulator(tmp_path):
    from cable_assembly.apps import hand_eye_calibration as app
    from cable_assembly.robot import make_arm

    cfg = C.load('hand_eye_calibration', ['robot.dry_run=true', f'data_dir={tmp_path}',
                                          'hand_eye_calibration.settle_s=0.0'])
    arm = make_arm(cfg)
    cam = app._camera(cfg, arm)
    try:
        r = app.calibrate(cfg, arm, cam, assume_yes=True)
    finally:
        cam.close()
        arm.disconnect()
    assert r, 'the sweep and solve ran'
    lin, ang = pose_error(r['X'], cam.X_true)
    assert lin < 1e-3 and np.degrees(ang) < 0.2, (lin, np.degrees(ang))
    out = next((tmp_path / 'experiments').iterdir())
    text = (out / 'result.yaml').read_text()
    assert 'hand_eye:' in text and 'ground_plane:' in text and 'camera:' in text
    assert len(list(out.glob('view_*.jpg'))) == 13


@pytest.mark.parametrize('error, axis_mm, step', [
    (None, 0.0, (100.0, 0.0)),                                   # the configured mount
    (([0.0, 70.0, 0.0], [0.0, 0.0, 0.0]), 70.0, (100.0, 0.0)),   # 70 mm off along tool y
    (([0.0, 0.0, 0.0], [0.0, 0.0, 90.0]), 0.0, (0.0, 100.0)),    # turned 90 deg about the tool axis
])
def test_marker_snapshot_reads_the_mount_check(tmp_path, caplog, error, axis_mm, step):
    """marker_snapshot's MOUNT CHECK with marker 44 under the tool axis and 13 a ruled 100 mm
    along base +x, rendered through a TRUE mount (the configured one moved by `error`, in tool0)
    and read through the configured one. The snapshot's detector has no subpixel refinement
    (~1.5% of the range at half a metre), so the tolerance is a few mm -- the errors it is there
    to catch are 70 mm and 90 deg."""
    import re

    from cable_assembly.apps import marker_snapshot
    from cable_assembly.perception.camera import Frame
    from cable_assembly.robot import Robot

    cfg = C.load('marker_calibration', ['cell=iiwa14_bench', 'robot.dry_run=true',
                                        'markers.13=40.25', 'markers.44=40.25',
                                        f'snapshot.out_dir={tmp_path}'])
    robot = Robot(cfg, with_gripper=False, needs=('hand_eye',))
    try:
        # The check assumes the tool straight down -- not the cell's pick pose, which may tilt it.
        assert robot.arm.move_j(np.radians([0, 20, 0, -80, 0, 80, 0]), label='straight down')
        T0 = robot.tool0()
        X_true = robot.T_tool0_cam if error is None else (
            xyzrpy_to_matrix(np.array(error[0]) / 1000.0, np.radians(error[1])) @ robot.T_tool0_cam)
        bench = {44: translation_matrix([T0[0, 3], T0[1, 3], 0.0]),
                 13: translation_matrix([T0[0, 3] + 0.1, T0[1, 3], 0.0])}

        class Cam:
            width, height = 1280, 720

            def capture(self):
                T_cam = T0 @ X_true
                img = he.render_markers(K, 1280, 720, [(m, 0.04025, inverse(T_cam) @ T)
                                                       for m, T in bench.items()], DICT)
                return Frame(img, K, np.zeros(5), 0.0, robot.camera())

        with caplog.at_level('INFO'):
            assert marker_snapshot.build_and_run(cfg, robot, Cam(), None)
        text = '\n'.join(r.getMessage() for r in caplog.records)
        read = {int(m.group(1)): (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                for m in re.finditer(r'id (\d+) +([\d.]+) mm from the tool axis \(base dx +'
                                     r'([-+\d.]+) dy +([-+\d.]+)\)', text)}
        assert set(read) == {13, 44}, text
        assert abs(read[44][0] - axis_mm) < 5.0, text
        step_read = np.subtract(read[13][1:], read[44][1:])
        assert np.allclose(step_read, step, atol=5.0), text
    finally:
        robot.close()
