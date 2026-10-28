"""A fake UR controller behind the ur_rtde API, so the REAL URArm runs with no robot.

One object serves both RTDEControlInterface and RTDEReceiveInterface. It runs the numpy UR10e
model (cable_assembly/robot/kinematics/ur10e.py), speaks the controller's UR `base` frame (Rz(pi)
from base_link) exactly as the real one does, and records every call in `calls` -- the
controller-facing command stream, which is what a golden trace compares.

Async moves advance one step per getAsyncOperationProgress() poll, so URArm's guard loop sees a
contact build up the way it would on the arm; `contact(q) -> wrench` (base_link, moment about the
flange) supplies the external force. `wrist_3_limits` narrows wrist_3 the way an installation
safety configuration does, for exercising the bisected tool-roll window.

    ctl = FakeRTDE()
    install(monkeypatch, ctl)          # ur_rtde imports now return ctl
    arm = make_arm(cfg)                # a real URArm, talking to ctl
"""

import sys
import types

import numpy as np

from cable_assembly.robot.kinematics import ur10e
from cable_assembly.transforms import (BASE_LINK_FROM_UR_BASE, matrix_to_rtde, rtde_to_matrix,
                                       slerp_matrix)

_R_UR = BASE_LINK_FROM_UR_BASE[:3, :3]            # Rz(pi): its own inverse


class FakeRTDE:
    STEP_RAD = np.radians(1.0)
    STEP_M = 0.002

    def __init__(self, q0=None, contact=None, wrist_3_limits=None):
        self.q = np.asarray(ur10e.HOME if q0 is None else q0, dtype=float)
        self.contact = contact
        self.limits = ur10e.JOINT_LIMITS.copy()
        if wrist_3_limits is not None:
            self.limits[5] = wrist_3_limits
        self.calls = []
        self._path = []
        self._step = -1
        self._tare = np.zeros(6)

    def _log(self, name, *args):
        self.calls.append((name, args))

    # ---- receive -------------------------------------------------------------------------
    def getActualQ(self):
        return list(self.q)

    def getActualTCPPose(self):
        return list(matrix_to_rtde(ur10e.fk(self.q)))

    def _raw_wrench(self):
        if self.contact is None:
            return np.zeros(6)
        return np.asarray(self.contact(self.q), dtype=float)

    def getActualTCPForce(self):
        w = self._raw_wrench() - self._tare                 # base_link
        return list(np.concatenate([_R_UR @ w[:3], _R_UR @ w[3:]]))   # -> UR base

    # ---- kinematics + safety -------------------------------------------------------------
    def getForwardKinematics(self, q):
        self._log('getForwardKinematics', tuple(q))
        return list(matrix_to_rtde(ur10e.fk(q)))

    def getInverseKinematics(self, pose, qnear):
        self._log('getInverseKinematics', tuple(pose), tuple(qnear))
        q = ur10e.ik(rtde_to_matrix(pose), qnear, self.limits)
        return [] if q is None else list(q)

    def isPoseWithinSafetyLimits(self, pose):
        return bool(ur10e.ik_all(rtde_to_matrix(pose)))

    def isJointsWithinSafetyLimits(self, q):
        return ur10e.within(q, self.limits)

    def getTCPOffset(self):
        return [0.0] * 6

    def setPayload(self, mass, cog):
        self._log('setPayload', mass, tuple(cog))
        return True

    # ---- motion --------------------------------------------------------------------------
    def _run(self, path, asynchronous):
        if not asynchronous:
            if path:
                self.q = np.asarray(path[-1], dtype=float)
            return True
        self._path, self._step = list(path), 0
        return True

    def moveJ(self, q, speed, accel, asynchronous=False):
        self._log('moveJ', tuple(q), speed, accel, asynchronous)
        q = np.asarray(q, dtype=float)
        if not ur10e.within(q, self.limits):
            return False
        n = max(1, int(np.ceil(np.max(np.abs(q - self.q)) / self.STEP_RAD)))
        return self._run([self.q + (q - self.q) * (k / n) for k in range(1, n + 1)],
                         asynchronous)

    def moveL(self, pose, speed, accel, asynchronous=False):
        self._log('moveL', tuple(pose), speed, accel, asynchronous)
        T0, T1 = ur10e.fk(self.q), rtde_to_matrix(pose)
        n = max(1, int(np.ceil(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / self.STEP_M)))
        path, q = [], self.q
        for k in range(1, n + 1):
            q = ur10e.ik(slerp_matrix(T0, T1, k / n), q, self.limits)
            if q is None:
                return False
            path.append(q)
        return self._run(path, asynchronous)

    def getAsyncOperationProgress(self):
        """Advance the running async move one step; -1 once there is none."""
        if self._step < 0 or self._step >= len(self._path):
            self._path, self._step = [], -1
            return -1
        self.q = np.asarray(self._path[self._step], dtype=float)
        self._step += 1
        return self._step - 1

    def stopJ(self, decel=2.0):
        self._log('stopJ', decel)
        self._path, self._step = [], -1
        return True

    def initPeriod(self):
        return 0

    def waitPeriod(self, t0):
        pass

    def servoJ(self, q, speed, accel, dt, lookahead, gain):
        self._log('servoJ', tuple(q), dt, lookahead, gain)
        self.q = np.asarray(q, dtype=float)
        return True

    def servoL(self, pose, speed, accel, dt, lookahead, gain):
        self._log('servoL', tuple(pose), dt, lookahead, gain)
        q = ur10e.ik(rtde_to_matrix(pose), self.q, self.limits)
        if q is not None:
            self.q = q
        return True

    def servoStop(self, decel=10.0):
        self._log('servoStop')
        return True

    # ---- force ---------------------------------------------------------------------------
    def zeroFtSensor(self):
        self._log('zeroFtSensor')
        self._tare = self._raw_wrench()
        return True

    def forceModeSetDamping(self, d):
        self._log('forceModeSetDamping', d)

    def forceModeSetGainScaling(self, g):
        self._log('forceModeSetGainScaling', g)

    def forceMode(self, task, selection, wrench, mode, limits):
        self._log('forceMode', tuple(task), tuple(selection), tuple(wrench), mode, tuple(limits))
        return True

    def forceModeStop(self):
        self._log('forceModeStop')
        return True

    # ---- lifecycle -----------------------------------------------------------------------
    def stopScript(self):
        self._log('stopScript')

    def disconnect(self):
        self._log('disconnect')


def install(monkeypatch, controller):
    """Make `from rtde_control import RTDEControlInterface` (and rtde_receive) hand back
    `controller`, whatever IP is asked for."""
    monkeypatch.setitem(sys.modules, 'rtde_control',
                        types.SimpleNamespace(RTDEControlInterface=lambda ip: controller))
    monkeypatch.setitem(sys.modules, 'rtde_receive',
                        types.SimpleNamespace(RTDEReceiveInterface=lambda ip: controller))
