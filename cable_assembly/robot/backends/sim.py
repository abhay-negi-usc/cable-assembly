"""SimArm -- any arm model with no hardware: the dry run, and the robot under the tests.

It replaces URArm's old `dry_run` branches, whose fk() returned one fixed pose whatever was
commanded and whose ik() returned the joints the arm was already at -- so every geometric result
in a dry run was meaningless and an unreachable target "succeeded". Here the joints are state, FK
and IK are the model's (kinematics/ur10e.py, kinematics/iiwa14.py), a target outside the joint
limits is refused, move_l really travels a line, and a guarded move is stepped along its path so
a guard (fed by a contact model) can stop it part-way, as on the real arm.

NOT A DYNAMICS SIMULATOR. Moves are kinematic and take no time; the wrench is whatever the
`contact` callable says (zero by default). Enough to run a whole task through its logic,
reachability and joint limits on either robot -- not to predict cycle time or tune a force law.
"""

import numpy as np

from ... import log as urlog
from ...transforms import slerp_matrix
from ..arm import Arm

log = urlog.get('arm')

# Guarded moves are stepped at this resolution so a guard sees the contact build up.
_STEP_RAD = np.radians(0.5)
_STEP_M = 0.001


class SimArm(Arm):
    """Kinematic simulation of `model` (a kinematics module). `contact(arm) -> wrench` supplies
    the raw [f; tau] in base_link (moment about the flange), before taring; None = free space."""

    dry_run = True
    supports_force_mode = True          # accepted and logged; the simulated arm stays rigid

    def __init__(self, cfg, model, frames=None, contact=None):
        super().__init__(cfg, model)
        r = cfg.section('robot')
        q0 = r.get('dry_run_joints')
        self._q = np.asarray(q0 if q0 is not None else model.HOME, dtype=float)
        if self._q.shape != (self.dof,):
            raise ValueError(f'robot.dry_run_joints has {self._q.size} values; a {model.NAME} '
                             f'has {self.dof} joints')
        self.contact = contact
        self._tare = np.zeros(6)
        log.warning('DRY RUN: simulated %s, no robot connection. Moves are kinematic and '
                    'instant; the wrench is %s.', model.NAME,
                    'the contact model' if contact else 'zero (free space)')
        if frames is not None:
            self.publish_frames(frames, cfg)

    # ------------------------------------------------------------------ state
    def q(self):
        return list(self._q)

    def tcp_pose(self):
        return self.model.fk(self._q)

    def fk(self, q):
        return self.model.fk(np.asarray(q, dtype=float))

    def ik(self, T_base_tool0, qnear=None):
        seed = self._q if qnear is None else np.asarray(qnear, dtype=float)
        q = self.model.ik(T_base_tool0, seed, self.model.JOINT_LIMITS)
        if q is None:
            log.error('No IK solution (unreachable, or no branch inside the joint limits).')
            return None
        return list(q)

    def joints_ok(self, q):
        q = np.asarray(q, dtype=float)
        return q.shape == (self.dof,) and self.model.within(q, self.model.JOINT_LIMITS)

    # ------------------------------------------------------------------ motion
    def _go(self, path, label):
        """Step through `path` (joint vectors), polling the guards. False if one trips."""
        for q in path:
            self._q = np.asarray(q, dtype=float)
            if self._guards and self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                return False
        return True

    def move_j(self, q_target, speed=None, accel=None, label='move', caps=None):
        q_target = np.asarray(q_target, dtype=float)
        if not self.joints_ok(q_target):
            log.error('[%s] move rejected: target outside the %s joint limits (deg %s).', label,
                      self.model.NAME, np.round(np.degrees(q_target), 1).tolist())
            return False
        log.info('[dry-run] %s -> q=%s', label, np.round(q_target, 3))
        if not self._guards:
            self._q = q_target
            return True
        q0 = self._q.copy()
        n = max(1, int(np.ceil(np.max(np.abs(q_target - q0)) / _STEP_RAD)))
        return self._go((q0 + (q_target - q0) * (k / n) for k in range(1, n + 1)), label)

    def move_l(self, T_base_tool0, speed=None, accel=None, label='move', caps=None):
        T0 = self.tcp_pose()
        d = float(np.linalg.norm(np.asarray(T_base_tool0)[:3, 3] - T0[:3, 3]))
        n = max(1, int(np.ceil(d / _STEP_M)))
        path, q = [], self._q
        for k in range(1, n + 1):
            q = self.model.ik(slerp_matrix(T0, T_base_tool0, k / n), q, self.model.JOINT_LIMITS)
            if q is None:
                log.error('[%s] linear move rejected: the line leaves the reachable workspace '
                          'or the joint limits at %.0f%%.', label, 100.0 * k / n)
                return False
            path.append(q)
        log.info('[dry-run] %s -> linear', label)
        return self._go(path, label)

    def servo_j(self, q, dt, lookahead=0.1, gain=300):
        q = np.asarray(q, dtype=float)
        if not self.joints_ok(q):
            log.error('servo_j target outside the joint limits -- holding.')
            return
        self._q = q

    def servo_l(self, T_base_tool0, dt, lookahead=0.1, gain=300):
        q = self.model.ik(T_base_tool0, self._q, self.model.JOINT_LIMITS)
        if q is None:
            log.error('servo_l target unreachable -- holding.')
            return
        self._q = q

    def servo_stop(self):
        pass

    def stop(self):
        pass

    def _settle(self):
        pass

    # ------------------------------------------------------------------ force / torque
    def _raw_wrench(self):
        if self.contact is None:
            return np.zeros(6)
        return np.asarray(self.contact(self), dtype=float)

    def zero_ft(self, settle=True):
        self._tare = self._raw_wrench()
        return True

    def wrench(self):
        return self._raw_wrench() - self._tare

    def force_mode(self, T_base_task, selection, wrench, limits, mode=2, damping=0.005,
                   gain_scaling=0.8):
        self._force_mode = True
        return True

    # ------------------------------------------------------------------ lifecycle
    def disconnect(self):
        pass
