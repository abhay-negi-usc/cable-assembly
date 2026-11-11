"""robot/trial_recorder.py: the console and the arm, recorded as the run goes."""

import csv
import logging
import os
import time

import numpy as np

from cable_assembly import config as C


def test_a_trial_records_the_console_and_the_arm(tmp_path):
    from cable_assembly.robot import Robot
    from cable_assembly import log as urlog
    from cable_assembly.robot.trial_recorder import TrialRecorder
    urlog.setup()
    logging.getLogger().setLevel(logging.INFO)           # as the app does (pytest owns the root handlers)
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false',
                                  'robot.world.enabled=false'])
    robot = Robot(cfg, with_gripper=False, needs=())
    rec = TrialRecorder(robot, str(tmp_path), rate_hz=200.0, bench_z_m=0.0062).start()
    logging.getLogger('arm').info('[grasp-align] a move')
    q = np.asarray(robot.arm.q()) + np.radians([5, 0, 0, 0, 0, 0, 0])
    robot.arm.move_j(q, label='test move')
    time.sleep(0.3)
    rec.stop()
    text = open(os.path.join(tmp_path, 'run.log')).read()
    assert 'test move' in text or 'a move' in text
    rows = list(csv.DictReader(open(os.path.join(tmp_path, 'motion.csv'))))
    assert len(rows) > 20
    last = rows[-1]
    np.testing.assert_allclose(float(last['q1_deg']), np.degrees(q[0]), atol=1e-2)
    assert float(last['fingertip_over_bench_mm']) > 0.0 and 'pad_tips_lowest_over_bench_mm' in last
    assert last['arm_last_line']
