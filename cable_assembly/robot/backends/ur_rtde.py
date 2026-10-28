"""URArm -- the UR10e over RTDE.

Replaces, in one class, what took four ROS mechanisms:

    MoveIt /compute_ik                      -> getInverseKinematics()
    FollowJointTrajectory action + JTC      -> moveJ()
    force_torque_sensor_broadcaster/wrench  -> getActualTCPForce()
    admittance_controller + switch_controller + set_parameters
                                            -> forceMode() / endForceMode()

Three of those swaps delete a whole bug class rather than just re-homing it:

  * IK. MoveIt's KDL solver is a LOCAL search seeded from one joint state, so a perfectly
    reachable pose whose solution lives in another IK branch returned error -31 and the old code
    papered over it by retrying with up to 12 RANDOM seeds. The UR controller's own IK is
    ANALYTIC -- it enumerates all eight branches and returns the one nearest `qnear`. There is
    nothing to retry, and the answer is the closest solution rather than a lucky one.

  * Compliance. The ros2_control admittance controller had to be installed, loaded inactive,
    parameterised over a service, and ACTIVATED -- which deactivated the trajectory controller.
    Activating it also made the arm jump to whatever reference it had, which is what caused the
    joint-0 velocity fault (the fix was an elaborate dance pinning the reference to the arm's
    current pose on both sides of both switches). forceMode() has no such transition: it is a
    mode of the SAME controller, and it starts from where the arm is. The _hold_reference dance
    is not ported because there is nothing left for it to fix.

  * Speed caps. `_move_duration_for` computed a DURATION from a velocity cap and then trusted the
    controller to respect it, bounding only the AVERAGE velocity (peak ~1.5x). moveJ takes a
    joint-velocity limit directly, and the controller enforces it exactly.

Frames: RTDE speaks the UR `base` frame, this class speaks ROS `base_link` (the frame every
config in this repo is measured in). transforms.rtde_to_matrix / matrix_to_rtde are the only
place that conversion happens -- see the note there.

HARDWARE ONLY. A dry run is the simulator (backends/sim.py) running the same UR10e model; there
is no offline branch in here, so every line below talks to a controller.
"""

import time

import numpy as np

from ... import log as urlog
from ...transforms import BASE_LINK_FROM_UR_BASE, matrix_to_rtde, rtde_to_matrix
from ..arm import Arm, ArmError
from ..kinematics import ur10e

log = urlog.get('arm')


# ur_rtde VALIDATION BOUNDS. rtde_control checks every speed/acceleration argument against
# these and raises ValueError("The value is not within [0;3.14]") rather than clamping, which
# aborts the run wherever it happens -- mid-sweep, or worse, mid-insertion. The joint-velocity
# ceiling is the UR's own limit of pi rad/s; the others are ur_rtde's documented maxima.
#
# WHY THIS BITES HERE: speed.phase_scale multiplies the GLOBAL caps, so any scale above
# ~2.0x with the shipped 90 deg/s joint cap lands past 180 deg/s. The config comments say as
# much ("past ~2.0x the joint-velocity cap is at the UR's own ceiling") but nothing enforced
# it, so the ceiling was discovered as a traceback instead of a warning.
RTDE_MAX_JOINT_VELOCITY = 3.14        # rad/s
RTDE_MAX_JOINT_ACCEL = 40.0           # rad/s^2
RTDE_MAX_TOOL_VELOCITY = 3.0          # m/s
RTDE_MAX_TOOL_ACCEL = 150.0           # m/s^2


def clamp_rtde(value, ceiling, what, label=''):
    """Hold a commanded speed/accel inside ur_rtde's accepted range, warning ONCE per move if
    the configured value asked for more. Returning the ceiling is the honest thing: the arm
    could never have gone faster anyway -- the controller clamps internally -- so the only
    thing the exception bought was a dead run."""
    v = float(value)
    if v > ceiling:
        log.warning('[%s] commanded %s %.2f exceeds the %.2f the controller accepts -- '
                    'CLAMPED. Lower the speed.phase_scale for this phase (or the global cap); '
                    'the arm was never going to move faster than the ceiling.',
                    label or '?', what, v, ceiling)
        return ceiling
    return max(v, 0.0)


class URArm(Arm):
    """The UR10e over RTDE. Blocking; every move returns True/False."""

    supports_force_mode = True
    JOINT_VELOCITY_CEILING = RTDE_MAX_JOINT_VELOCITY

    def __init__(self, cfg, frames=None):
        super().__init__(cfg, ur10e)
        r = cfg.section('robot')
        self.ip = r.get('ip', '192.168.125.2')
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ArmError(
                'ur_rtde is not installed. `pip install ur_rtde` (Linux wheels exist for '
                'cp38-cp312). Use --dry-run to plan without a robot.') from exc

        log.info('Connecting to the UR at %s ...', self.ip)
        # RTDEControlInterface uploads its OWN control script to the robot. The External
        # Control URCap program must NOT be playing -- if it is, the two fight over the
        # controller and this constructor hangs or the arm refuses to move.
        self.rtde_c = RTDEControlInterface(self.ip)
        self.rtde_r = RTDEReceiveInterface(self.ip)
        log.info('Connected.')
        self._set_payload(r.get('payload', {}) or {})
        self._report_tcp_offset()

        if frames is not None:
            self.publish_frames(frames, cfg)

    # ------------------------------------------------------------------ state
    def q(self):
        """Current joint positions (rad), in UR_JOINTS order."""
        return list(self.rtde_r.getActualQ())

    def tcp_pose(self):
        """Current tool0 pose in base_link, as a 4x4."""
        return rtde_to_matrix(self.rtde_r.getActualTCPPose())

    def fk(self, q):
        """Forward kinematics: joints -> tool0 in base_link."""
        return rtde_to_matrix(self.rtde_c.getForwardKinematics(list(q)))

    def ik(self, T_base_tool0, qnear=None):
        """Inverse kinematics: tool0 pose in base_link -> joints, or None if unreachable.

        The controller's analytic solver returns the branch nearest `qnear` (default: where the
        arm is now), so successive calls along a path stay on one branch and the arm does not
        flip its elbow mid-trajectory."""
        pose = matrix_to_rtde(T_base_tool0)
        if not self.rtde_c.isPoseWithinSafetyLimits(pose):
            log.error('Pose is outside the robot safety limits: %s', np.round(pose, 4))
            return None
        qnear = list(qnear) if qnear is not None else self.q()
        q = self.rtde_c.getInverseKinematics(pose, qnear)
        if not q or not self.rtde_c.isJointsWithinSafetyLimits(list(q)):
            log.error('No IK solution (unreachable, in collision, or at a singularity).')
            return None
        return list(q)

    def joints_ok(self, q):
        """True if `q` sits inside the controller's JOINT safety limits.

        Distinct from `isPoseWithinSafetyLimits`, which ik() already calls: a pose can be legal
        while the joints that reach it are not, and wrist_3 in particular is usually limited by
        the INSTALLATION safety configuration rather than by the robot model. Asking the
        controller means a restricted wrist reports itself here instead of mid-motion."""
        return bool(self.rtde_c.isJointsWithinSafetyLimits(list(q)))

    # ------------------------------------------------------------------ motion
    def move_j(self, q_target, speed=None, accel=None, label='move', caps=None):
        """Joint-space move to `q_target`. Blocking; returns True on success.

        SYNCHRONOUS unless a guard is armed. A synchronous moveJ blocks until the arm has actually
        finished -- simplest, and immune to the async-progress race: right after an async moveJ,
        ur_rtde's getAsyncOperationProgress() can report 'no operation' for a tick before the move
        registers, and a wait that reads that as 'done' returns instantly, so the next command
        supersedes the move before it happens and the arm never visibly moves. Only a guarded move
        needs to run async, so the guard can be polled while the arm is in motion."""
        auto_speed, auto_accel, which = self._speeds(q_target, caps)
        if speed is None:
            speed = auto_speed
        else:
            which = 'explicit'
        accel = auto_accel if accel is None else accel
        speed = clamp_rtde(speed, RTDE_MAX_JOINT_VELOCITY, 'joint velocity (rad/s)', label)
        accel = clamp_rtde(accel, RTDE_MAX_JOINT_ACCEL, 'joint accel (rad/s^2)', label)
        # Log the COMMANDED speed and the cap that produced it: if the arm visibly moves slower
        # than this line says, the throttle is on the CONTROLLER side (pendant speed slider,
        # safety Reduced mode / restricted limits), not in this code or the config.
        log.info('[%s] moveJ %.2f rad/s (%.0f deg/s leading joint, %s cap), accel %.2f rad/s^2',
                 label, speed, np.degrees(speed), which, accel)

        if not self._guards:
            ok = self.rtde_c.moveJ(list(q_target), speed, accel, False)   # blocks until finished
            if not ok:
                log.error('[%s] move rejected/failed. On the pendant: Remote Control on, brakes '
                          'released (green "Normal"), speed slider up, and NOT in Simulation mode?',
                          label)
            return bool(ok)

        self.rtde_c.moveJ(list(q_target), speed, accel, True)             # async -> pollable guard
        return self._await_move(label)

    def move_l(self, T_base_tool0, speed=None, accel=None, label='move', caps=None):
        """Straight-line Cartesian move (the tool travels a line in space, not a joint arc).
        `caps` (a `speed:`-style mapping) overrides the global limits for this move; moveL's RTDE
        speed is a TCP linear speed, so the translation cap is the one applied.

        Synchronous unless a guard is armed, for the same reason as move_j."""
        pose = matrix_to_rtde(T_base_tool0)
        speed = self._limits(caps)[2] if speed is None else speed
        accel = self.cart_accel if accel is None else accel
        speed = clamp_rtde(speed, RTDE_MAX_TOOL_VELOCITY, 'tool velocity (m/s)', label)
        accel = clamp_rtde(accel, RTDE_MAX_TOOL_ACCEL, 'tool accel (m/s^2)', label)
        log.info('[%s] moveL %.3f m/s, accel %.2f m/s^2', label, speed, accel)
        if not self._guards:
            ok = self.rtde_c.moveL(pose, speed, accel, False)
            if not ok:
                log.error('[%s] linear move rejected/failed (unreachable, or the checks above).',
                          label)
            return bool(ok)
        self.rtde_c.moveL(pose, speed, accel, True)
        return self._await_move(label)

    def _await_move(self, label):
        """Poll an ASYNC move until it finishes, a guard trips, or it times out.

        Waits for the move to actually START before accepting 'no operation' as finished. Right
        after an async move, getAsyncOperationProgress() can read < 0 for a tick before the
        controller registers it; returning then would skip the move. So < 0 counts as 'done' only
        once we have seen it running -- or after a short grace, which covers a genuinely instant
        (near-zero) move that never registers progress at all."""
        deadline = time.monotonic() + self.move_timeout
        start_grace = time.monotonic() + 1.0
        last_log = 0.0
        started = False
        while True:
            prog = self.rtde_c.getAsyncOperationProgress()
            if prog >= 0:
                started = True
            elif started or time.monotonic() > start_grace:
                time.sleep(0.02)                        # let the final setpoint land
                return True

            if self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                self.rtde_c.stopJ(2.0)
                return False

            now = time.monotonic()
            if now > deadline:
                log.error(
                    '[%s] not finished after %.0fs. On the pendant: Remote Control on, no '
                    'protective/e-stop, speed slider up, NOT in Simulation mode? Stopping.',
                    label, self.move_timeout)
                self.rtde_c.stopJ(2.0)
                return False
            if now - last_log > 5.0:
                log.info('[%s] executing...', label)
                last_log = now
            time.sleep(0.01)

    def servo_j(self, q, dt, lookahead=0.1, gain=300):
        """One step of a streamed joint reference -- the compliant-insertion inner loop.

        Unlike moveJ this does NOT plan a profile; it drives toward `q` and returns immediately.
        Call it at a steady rate or the motion will be jerky."""
        t0 = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(list(q), 0.0, 0.0, dt, lookahead, gain)
        self.rtde_c.waitPeriod(t0)

    def servo_l(self, T_base_tool0, dt, lookahead=0.1, gain=300):
        """One step of a streamed CARTESIAN reference -- the software-admittance inner loop.

        Drives tool0 toward the pose (the controller's own IK handles it) and returns immediately.
        Like servo_j it plans no profile, so call it at a steady rate; the initPeriod/waitPeriod
        pair holds that rate. `dt` is the control period."""
        t0 = self.rtde_c.initPeriod()
        self.rtde_c.servoL(matrix_to_rtde(T_base_tool0), 0.0, 0.0, dt, lookahead, gain)
        self.rtde_c.waitPeriod(t0)

    def servo_stop(self):
        self.rtde_c.servoStop()

    def stop(self):
        self.rtde_c.stopJ(2.0)

    # ------------------------------------------------------------------ force / torque
    def tcp_offset(self):
        """The controller's configured flange -> TCP offset, [x, y, z, rx, ry, rz] (m, rotvec)."""
        return np.asarray(self.rtde_c.getTCPOffset(), dtype=float)

    def _report_tcp_offset(self):
        """Read and report the controller's configured TCP offset -- it decides where the logged
        MOMENT is referenced, and nothing else in this stack can tell.

        wrench() returns a force plus a moment about a point the controller chooses, and the two
        candidate readings of the documentation disagree about which point that is: the SW5.19
        URScript page for get_tcp_force() says the tool flange, while the RTDE field ur_rtde
        actually reads (actual_TCP_force) is described as being at the TCP. They are the same
        point only when this offset is zero.

        wrench_in() references the moment at tcp_pose(), i.e. the controller's TCP -- correct
        under the RTDE reading unconditionally, and under the URScript reading when the offset is
        zero. A NON-ZERO offset therefore means the two readings differ by exactly that vector,
        and it also means `tip_frame: tool0` ("the pendant all-zeros TCP") is no longer true, so
        every pose in this stack inherits the same shift. Worth a loud line either way."""
        try:
            off = self.tcp_offset()
        except Exception as exc:                       # noqa: BLE001 -- never fatal at connect
            log.warning('Could not read the controller TCP offset (%s). Poses assume it is zero '
                        '(tip_frame: tool0) and wrench_in references the moment at tcp_pose().',
                        exc)
            return
        d_mm = float(np.linalg.norm(off[:3])) * 1000.0
        if d_mm < 0.05:
            log.info('Controller TCP offset is zero -- tool flange == TCP == tool0, so the logged '
                     'moment is referenced there and every frame in this repo means what it says.')
        else:
            log.warning(
                'Controller TCP offset is NON-ZERO: %s mm. Two consequences, both silent '
                'otherwise. (1) tip_frame: tool0 is documented as "the pendant all-zeros TCP", '
                'which is no longer true -- every pose read from RTDE is the TCP, shifted by this '
                'from the flange. (2) The logged wrench MOMENT is referenced at whichever point '
                'the firmware uses, and the URScript and RTDE docs disagree by exactly this '
                'vector; wrench_in() references it at tcp_pose(). Zero the TCP on the pendant, or '
                'confirm which end the moment comes from before trusting the torque columns.',
                np.round(off[:3] * 1000.0, 2).tolist())

    def _set_payload(self, payload):
        """Tell the controller the tool's mass + CoG so getActualTCPForce() subtracts the tool's
        own weight and reports true EXTERNAL force. Without this, the wrench reads the tool weight
        (~tens of N) as contact -- and a tare only cancels it at the tare pose and only while idle,
        so a force guard trips spuriously the moment the arm is under active control. Configure
        robot.payload.mass_kg + cog_m for correct readings everywhere (guard, admittance, touch)."""
        mass = float(payload.get('mass_kg', 0.0))
        cog = [float(v) for v in payload.get('cog_m', [0.0, 0.0, 0.0])]
        if mass > 0.0:
            try:
                self.rtde_c.setPayload(mass, cog)
                log.info('Payload set: %.2f kg, CoG %s m.', mass, cog)
            except Exception as exc:                       # noqa: BLE001
                log.warning('setPayload failed: %s', exc)
        else:
            log.warning('No robot.payload configured -- getActualTCPForce() will read the tool '
                        "weight as external force, so the force guard/admittance can't be trusted. "
                        'Set robot.payload.mass_kg and cog_m (mass + centre of gravity of the '
                        'gripper + coupler + camera).')

    def zero_ft(self, settle=True):
        """Tare the wrist F/T. Everything after this is CONTACT force -- so do it with the tool
        hanging free, not in contact. `settle=False` skips the sample-wait + residual check, for
        taring INSIDE a servo loop (where blocking would drop servo mode)."""
        ok = self.rtde_c.zeroFtSensor()
        if not settle:
            return ok
        time.sleep(0.3)                     # let a fresh tared sample arrive before anyone reads
        residual = float(np.linalg.norm(self.wrench()[:3]))
        if residual > 5.0:
            log.warning(
                'Residual force %.1f N after taring. The tool weight is NOT being compensated '
                '(set robot.payload) -- force thresholds will be wrong and force mode will drift.',
                residual)
        return ok

    def wrench(self):
        """Contact wrench [fx, fy, fz, tx, ty, tz]: force and the moment ABOUT THE TOOL FLANGE,
        both expressed in ROS base_link axes, tared.

        THE REFERENCE POINT IS THE FLANGE, NOT THE BASE ORIGIN. The UR script manual for
        get_tcp_force says the components are "all measured at the tool flange with the
        orientation of the robot base coordinate system" -- so this is a moment about a point
        roughly a metre from the base origin, and anything that re-expresses it has to move that
        reference point before rotating. wrench_in() does; a bare rotation would not.

        getActualTCPForce() reports in the UR `base` frame, which differs from ROS `base_link` by
        Rz(pi) -- the SAME bridge every pose crosses via rtde_to_matrix. It must be applied here
        too, or the wrench is silently mixed with base_link poses and its x/y components come out
        NEGATED while z is fine. That asymmetry is invisible to anything reading a MAGNITUDE (the
        force guard, force(), torque()) but inverts the compliant axes of the admittance law and
        mislabels the logged wrench columns. Pure rotation about a shared origin -> no cross term.

        NOTE this is the BASE frame, whereas the ROS broadcaster published in tool0_controller.
        Force MAGNITUDE is frame-invariant so the force guards carry over unchanged; TORQUE
        magnitude is NOT (it depends on the reference origin), so a torque threshold tuned
        against the ROS stack needs re-checking. Use wrench_in() to get it in a tool frame."""
        w = np.asarray(self.rtde_r.getActualTCPForce(), dtype=float)
        R = BASE_LINK_FROM_UR_BASE[:3, :3]                   # UR base -> ROS base_link
        return np.concatenate([R @ w[:3], R @ w[3:]])

    # ------------------------------------------------------------------ compliance
    def force_mode(self, T_base_task, selection, wrench, limits, mode=2, damping=0.005,
                   gain_scaling=0.8):
        """Enter compliant (force) control.

        `selection` marks the COMPLIANT axes [x, y, z, rx, ry, rz] of the task frame: 1 = yield to
        contact and regulate toward the requested `wrench`, 0 = stay rigidly position-controlled.
        `limits` bounds the compliant axes' speed (m/s, rad/s) and the non-compliant ones'
        deviation (m, rad).

        This is a mode of the running controller, not a different controller: the arm keeps
        tracking whatever you servo to it, but yields on the selected axes. Nothing is switched
        out, so there is no activation jump -- the joint-0 velocity fault that the ros2_control
        admittance controller produced on every switch simply cannot happen here."""
        self.rtde_c.forceModeSetDamping(damping)
        self.rtde_c.forceModeSetGainScaling(gain_scaling)
        self.rtde_c.forceMode(matrix_to_rtde(T_base_task), list(selection),
                              list(wrench), int(mode), list(limits))
        self._force_mode = True
        return True

    def end_force_mode(self):
        """Leave force mode. Idempotent and safe to call from a finally: -- and it must be, or a
        crash mid-insertion leaves the arm compliant."""
        if self._force_mode:
            self.rtde_c.forceModeStop()
        self._force_mode = False

    # ------------------------------------------------------------------ lifecycle
    def disconnect(self):
        try:
            self.end_force_mode()
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
        except Exception as exc:                       # noqa: BLE001 -- best-effort teardown
            log.warning('Teardown: %s', exc)
        finally:
            try:
                self.rtde_c.disconnect()
                self.rtde_r.disconnect()
            except Exception:                          # noqa: BLE001
                pass
