"""robot/payload.py and apps/payload_calibration: the tool's mass and CoG from the external joint
torques its weight causes -- the fit, its sign agreement with the FRI backend's compensation, and
the app's sweep on the simulated iiwa."""
import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.apps import payload_calibration as app
from cable_assembly.robot import make_arm
from cable_assembly.robot import payload as pl
from cable_assembly.robot.backends.iiwa_fri import wrench_from_torques
from cable_assembly.robot.kinematics import iiwa14

PICK = np.radians([-27, 50, 66, -96, 120, -57, -123])     # the iiwa bench's pick pose
TRUE = (1.25, np.array([-0.018, 0.004, 0.072]))
WRIST = [0, 4, 5, 6]                                       # A1 and the wrist (0-based)


def _wrist_poses(q0, d6=(0, 30, -30), d7=(0, 90, 180), d5=(40, -40)):
    out = []
    for a in d6:
        for b in d7:
            q = np.array(q0)
            q[5] += np.radians(a)
            q[6] += np.radians(b)
            out.append(q)
    for a in d5:
        q = np.array(q0)
        q[4] += np.radians(a)
        out.append(q)
    return out


def _samples(qs, bias=np.zeros(7), noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return [(q, pl.gravity_torques(iiwa14, q, *TRUE) + bias + rng.normal(0.0, noise, 7))
            for q in qs]


def test_the_fit_recovers_the_payload_through_wrist_offsets_and_noise():
    bias = np.array([0.4, 0.0, 0.0, 0.0, -0.12, 0.08, 0.04])
    r = pl.fit(iiwa14, _samples(_wrist_poses(PICK), bias, noise=0.03), bias_joints=WRIST)
    assert abs(r['mass_kg'] - TRUE[0]) < 0.01
    assert np.allclose(r['cog_m'], TRUE[1], atol=0.002)
    assert np.allclose(r['bias_nm'], bias, atol=0.05)
    assert r['rms_nm'] < 0.05


def test_the_fitted_payload_is_what_the_backend_subtracts():
    """Same sign, same frame: the FRI backend's compensate_payload, given the fitted payload,
    reads ZERO wrench from torques that are the tool's weight alone, at every pose (undamped:
    the damping alone leaves ~3 mN)."""
    samples = _samples(_wrist_poses(PICK))
    r = pl.fit(iiwa14, samples)
    for q, tau in samples:
        assert np.allclose(wrench_from_torques(iiwa14, q, tau), 0.0, atol=1e-6) is False
        w = wrench_from_torques(iiwa14, q, tau, damping=0.0, payload=(r['mass_kg'], r['cog_m']))
        assert np.allclose(w, 0.0, atol=1e-3), w


def test_an_offset_on_the_joints_that_carry_the_mass_is_refused():
    """The wrist turns the tool about ONE point: with the elbow still, the mass shows only in the
    torque level on A2-A4, which offsets there would absorb."""
    with pytest.raises(ValueError, match='undetermined'):
        pl.fit(iiwa14, _samples(_wrist_poses(PICK)), bias_joints=range(7))


def test_given_the_mass_an_offset_on_every_joint_leaves_the_cog_exact():
    """Step 2: with the mass known, KUKA's model error on A2-A4 -- which step 1 cannot take --
    is an offset like any other, and the CoG comes from how the torques turn."""
    bias = np.array([0.4, 0.5, 0.6, 0.1, -0.45, 0.15, -0.12])     # the bench's, 2026-10-29
    r = pl.fit(iiwa14, _samples(_wrist_poses(PICK), bias), bias_joints=range(7), mass_kg=TRUE[0])
    assert np.allclose(r['cog_m'], TRUE[1], atol=1e-6)
    assert np.allclose(r['bias_nm'], bias, atol=1e-6)
    assert r['sigma_mass_kg'] == 0.0


def test_turning_about_one_axis_only_is_refused():
    down = np.radians([0, 20, 0, -80, 0, 80, 0])             # the tool straight down
    qs = [down + np.radians([0, 0, 0, 0, 0, 0, a]) for a in (-150, -90, -30, 30, 90, 150)]
    with pytest.raises(ValueError, match='gravity turned only'):
        pl.fit(iiwa14, _samples(qs))


def _cfg(tmp_path, *extra):
    return C.load('payload_calibration', ['robot.dry_run=true', f'data_dir={tmp_path}', *extra])


def test_the_app_sweeps_the_wrist_and_writes_the_cell_block(tmp_path):
    cfg = _cfg(tmp_path)
    arm = make_arm(cfg)
    try:
        r = app.calibrate(cfg, arm, app._torque_reader(arm), assume_yes=True)
        assert r, 'the dry run fits'
        # The dry run's model error includes A2-A4, which the fit assumes clean -- as on the arm.
        assert abs(r['mass_kg'] - app.DRY_RUN_PAYLOAD[0]) < 0.05
        assert np.allclose(r['cog_m'], app.DRY_RUN_PAYLOAD[1], atol=0.008)
        assert np.allclose(arm.q(), PICK, atol=1e-9), 'back at the start'
    finally:
        arm.disconnect()
    out = next((tmp_path / 'experiments').iterdir())
    text = (out / 'result.yaml').read_text()
    assert 'mass_kg: 1.2' in text and 'cog_mm:' in text and 'compensate_payload: true' in text
    assert len((out / 'samples.csv').read_text().splitlines()) > 20


def test_a_refit_from_the_saved_readings_needs_no_arm(tmp_path):
    cfg = _cfg(tmp_path)
    arm = make_arm(cfg)
    try:
        assert app.calibrate(cfg, arm, app._torque_reader(arm), assume_yes=True)
    finally:
        arm.disconnect()
    first = next((tmp_path / 'experiments').iterdir())
    cfg = _cfg(tmp_path, f'payload_calibration.mass_kg={app.DRY_RUN_PAYLOAD[0]}')
    r = app.refit(cfg, str(first / 'samples.csv'))
    assert r['mass_kg'] == app.DRY_RUN_PAYLOAD[0]
    assert np.allclose(r['cog_m'], app.DRY_RUN_PAYLOAD[1], atol=0.0015), 'the right mass: exact'


def test_a_start_too_near_the_bench_refuses_before_moving(tmp_path):
    cfg = _cfg(tmp_path, 'ground_plane.z_mm=250')            # the bench just under the tool
    arm = make_arm(cfg)
    try:
        moves = []
        go = arm.move_j
        arm.move_j = lambda q, *a, **k: moves.append(k.get('label')) or go(q, *a, **k)
        assert not app.calibrate(cfg, arm, app._torque_reader(arm), assume_yes=True)
        assert moves == ['start'], 'only the dry run\'s move to its start view'
    finally:
        arm.disconnect()
