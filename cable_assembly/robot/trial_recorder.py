"""TRIAL DATA for debugging a run after the fact (the user, 2026-11-04: "the gripper suddenly pitches down
very quickly and collides with the ground plane" during the pickup -- record the next run).

Two files in the run's experiment folder, written as the run goes (a crash keeps what came before it):

  run.log       the console, every line, with milliseconds -- which move, which check, which refusal
  motion.csv    the arm at `rate_hz` from a background thread: wall time (to line up with run.log),
                measured and COMMANDED joints (the FRI driver's reference) and commanded joint speed,
                the flange pose, the fingertip's height and the lowest point of the open gripper's
                pad tips over the bench, the tared wrench, KUKA's external joint torques, and the
                last line the arm logged (the move it is in)

    robot.record_trial: true      # configs (bnc_assembly.yaml); off: nothing is written

A RUN OF MANY TRIALS (bnc_assembly --trials N) keeps one recorder and SWITCHES the motion file at each trial's
start (switch(folder): trial_NNN/motion.csv), so no trial's file is gigabytes and none is missed in between.
"""

import csv
import logging
import os
import threading
import time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

from .. import log as urlog

log = urlog.get('trial')

# The open V3's pad tips (the user's touch-off, 2026-11-04: 165 mm on tool z, fingers open) and how far
# either side of the axis they are (CAD: the open pads' outer edges 74 mm, their gripping faces 37 mm).
PAD_TIP_Z_M = 0.165
PAD_TIP_Y_M = (0.037, 0.074)


class _LastLine(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.last = ''

    def emit(self, record):
        try:
            self.last = record.getMessage()[:160].replace('\n', ' ')
        except Exception:                                  # noqa: BLE001
            pass


class TrialRecorder:
    def __init__(self, robot, out_dir, rate_hz=100.0, bench_z_m=0.0, console=True, motion=True):
        self.robot, self.out_dir = robot, out_dir
        self.dt = 1.0 / float(rate_hz)
        self.bench_z = float(bench_z_m)
        self.console, self.motion = bool(console), bool(motion)
        self._stop = threading.Event()
        self._thread = None
        self._file_handler = None
        self._arm_line = _LastLine()
        self._motion_dir = out_dir           # where motion.csv goes; switch() moves it
        self._switch = None
        self.rows = 0

    def start(self):
        os.makedirs(self.out_dir, exist_ok=True)
        if self.console:
            fh = logging.FileHandler(os.path.join(self.out_dir, 'run.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(levelname)-7s %(name)s: %(message)s',
                                              datefmt='%H:%M:%S'))
            logging.getLogger().addHandler(fh)
            self._file_handler = fh
        if self.motion:
            logging.getLogger('arm').addHandler(self._arm_line)
            self._thread = threading.Thread(target=self._loop, name='trial-recorder', daemon=True)
            self._thread.start()
        log.info('TRIAL DATA: %s%s%s.', f"the console to {os.path.join(self.out_dir, 'run.log')}" if self.console else '',
                 ', ' if self.console and self.motion else '',
                 f"the arm at {1.0 / self.dt:.0f} Hz to {os.path.join(self.out_dir, 'motion.csv')}" if self.motion else '')
        return self

    def switch(self, out_dir):
        """From now on the motion goes to `out_dir`/motion.csv (a new trial's folder); taken up within a sample."""
        os.makedirs(out_dir, exist_ok=True)
        self._switch = out_dir

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logging.getLogger('arm').removeHandler(self._arm_line)
        if self._file_handler is not None:
            log.info('TRIAL DATA: %d motion samples kept.', self.rows)
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    # ---- the sampler ---------------------------------------------------------------------------------
    def _sample(self):
        arm = self.robot.arm
        st = arm._state() if hasattr(arm, '_state') else None
        q = np.asarray(st.q if st is not None else arm.q(), dtype=float)
        n = len(q)
        q_cmd = np.asarray(st.q_cmd, dtype=float) if st is not None else np.full(n, np.nan)
        qd_cmd = np.asarray(st.qd_cmd, dtype=float) if st is not None else np.full(n, np.nan)
        tau = np.asarray(st.tau_ext, dtype=float) if st is not None else np.full(n, np.nan)
        T = arm.fk(q)
        try:
            w = np.asarray(arm.wrench(), dtype=float)
        except Exception:                                  # noqa: BLE001
            w = np.full(6, np.nan)
        ftip = T @ self.robot.T_tool0_fingertip
        pads = [T @ np.array([x, s * y, PAD_TIP_Z_M, 1.0]) for x in (-0.011, 0.011)
                for y in PAD_TIP_Y_M for s in (-1.0, 1.0)]
        low = min(p[2] for p in pads) - self.bench_z
        rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
        now = time.time()
        return ([datetime.fromtimestamp(now).strftime('%H:%M:%S.%f')[:-3], f'{time.monotonic():.4f}']
                + [f'{np.degrees(v):.3f}' for v in q] + [f'{np.degrees(v):.3f}' for v in q_cmd]
                + [f'{np.degrees(v):.2f}' for v in qd_cmd]
                + [f'{1000 * v:.2f}' for v in T[:3, 3]] + [f'{v:.5f}' for v in rv]
                + [f'{1000 * (ftip[2, 3] - self.bench_z):.2f}', f'{1000 * low:.2f}']
                + [f'{v:.3f}' for v in w] + [f'{v:.3f}' for v in tau]
                + [str(int(st.session)) if st is not None else '', str(int(st.fault)) if st is not None else '']
                + [self._fmt(st, k) for k in self.DIAG]
                + [f'{np.degrees(v):.3f}' for v in self._vec(st, 'q_robot_cmd', n)]
                + [f'{np.degrees(v):.3f}' for v in self._vec(st, 'q_ipo', n)]
                + [self._arm_line.last])

    # FRI's own view (fri_driver.STATE): quality 0 POOR .. 3 EXCELLENT; control_mode 0 POSITION,
    # 1 CARTESIAN IMPEDANCE, 2 JOINT IMPEDANCE, 3 NO CONTROL; tracking 0..1; step_gap_s: this PC's
    # loop period (a stall shows as a long one); sample_time_s: the controller's.
    DIAG = ('quality', 'safety', 'drive', 'command_mode', 'control_mode', 'tracking', 'sample_time', 'step_gap',
            'robot_time')

    @staticmethod
    def _fmt(st, k):
        if st is None or k not in st:
            return ''
        return f'{float(st[k]):.6f}' if k in ('step_gap', 'sample_time', 'tracking', 'robot_time') else str(int(st[k]))

    @staticmethod
    def _vec(st, k, n):
        return np.asarray(st[k], dtype=float) if st is not None and k in st else np.full(n, np.nan)

    def _loop(self):
        arm = self.robot.arm
        n = len(arm.q())
        head = (['wall', 'mono_s'] + [f'q{i + 1}_deg' for i in range(n)] + [f'qcmd{i + 1}_deg' for i in range(n)]
                + [f'qdcmd{i + 1}_deg_s' for i in range(n)] + ['flange_x_mm', 'flange_y_mm', 'flange_z_mm',
                                                                'flange_rx', 'flange_ry', 'flange_rz']
                + ['fingertip_over_bench_mm', 'pad_tips_lowest_over_bench_mm']
                + ['fx_n', 'fy_n', 'fz_n', 'tx_nm', 'ty_nm', 'tz_nm'] + [f'tau{i + 1}_nm' for i in range(n)]
                + ['session', 'fault'] + list(self.DIAG)
                + [f'q_robot_cmd{i + 1}_deg' for i in range(n)] + [f'q_ipo{i + 1}_deg' for i in range(n)]
                + ['arm_last_line'])
        t_next = time.monotonic()
        while not self._stop.is_set():
            path = os.path.join(self._motion_dir, 'motion.csv')
            with open(path, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(head)
                t_flush = time.monotonic() + 1.0
                while not self._stop.is_set():
                    if self._switch is not None:           # a new trial: its own file from here
                        self._motion_dir, self._switch = self._switch, None
                        break
                    try:
                        w.writerow(self._sample())
                        self.rows += 1
                    except Exception as exc:               # noqa: BLE001 -- never take the run down
                        w.writerow([f'sample failed: {type(exc).__name__}: {exc}'])
                    now = time.monotonic()
                    if now >= t_flush:
                        fh.flush()
                        t_flush = now + 1.0
                    t_next += self.dt
                    time.sleep(max(0.0, t_next - time.monotonic()))
                    if time.monotonic() - t_next > 0.5:    # fell behind (a stall): resync, don't burst
                        t_next = time.monotonic()
