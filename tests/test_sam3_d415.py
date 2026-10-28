"""SAM3 junction detection on a REAL D415 frame -- the one test that runs the actual model.

Every other perception test feeds synthetic masks to the geometry, so a change that only shows
up through SAM3 itself -- a sam3 or torch bump, a prompt or threshold retune in
bnc_assembly.yaml, an autocast change in sam3_backend -- would otherwise reach the robot untested.

Fixture: tests/fixtures/d415_bnc_junction_1920x1080.jpg, captured 2026-10-27 through
RealSenseCamera from the iiwa tool's D415 (serial 840412060409) at 1920x1080: a BNC cable on a
white surface, the Robotiq fingertips across the bottom of the frame. The expected junction --
where the cable meets the connector's strain-relief boot, heading toward the connector tip --
was checked by eye on the detector's overlay, then pinned from the detector's output.

Needs CUDA and the SAM3 weights in the HuggingFace cache (upstream SAM3 hard-codes CUDA), and is
skipped otherwise, so CPU-only machines still run the rest of the suite. `-m "not gpu"` skips it
on purpose: it adds the model load (~8 s once cached) to the run.
"""
import os
import time

import numpy as np
import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'd415_bnc_junction_1920x1080.jpg')
# The D415's factory color intrinsics at 1920x1080, as RealSenseCamera reported them at capture.
K_D415_1080P = np.array([[1394.8, 0.0, 974.6], [0.0, 1393.4, 533.6], [0.0, 0.0, 1.0]])
# Measured on this file: identical across repeated runs on the RTX 3080 (torch 2.11 cu128).
EXPECTED_UV = (1301.8, 575.1)
EXPECTED_YAW_DEG = -138.6           # image frame (y down): up-left, from junction to connector tip
# The connector is ~150 px long in this frame, so 10 px is a small fraction of it -- but enough
# slack for fp16-autocast differences between GPUs and torch builds.
TOL_PX = 10.0
TOL_DEG = 5.0


@pytest.fixture(scope='module')
def junction_detector():
    torch = pytest.importorskip('torch')
    pytest.importorskip('sam3')
    pytest.importorskip('cv2')
    if not torch.cuda.is_available():
        pytest.skip('SAM3 needs CUDA')
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache('facebook/sam3', 'sam3.pt'), str):
        pytest.skip('SAM3 weights not cached (gated: request access, then `hf auth login`)')
    from cable_assembly import config
    from cable_assembly.perception.sam3 import make_detector
    return make_detector(config.load('bnc_assembly'))


@pytest.mark.gpu
def test_sam3_finds_the_bnc_junction_on_a_real_d415_frame(junction_detector):
    import cv2
    from cable_assembly.perception.camera import Frame

    bgr = cv2.imread(FIXTURE)
    assert bgr is not None and bgr.shape == (1080, 1920, 3), FIXTURE
    junctions = junction_detector.detect(Frame(bgr, K_D415_1080P, np.zeros(5), time.monotonic()))

    assert len(junctions) == 1, junctions
    u, v, yaw = junctions[0]
    assert np.hypot(u - EXPECTED_UV[0], v - EXPECTED_UV[1]) < TOL_PX, (u, v)
    d_yaw = (np.degrees(yaw) - EXPECTED_YAW_DEG + 180.0) % 360.0 - 180.0
    assert abs(d_yaw) < TOL_DEG, np.degrees(yaw)
