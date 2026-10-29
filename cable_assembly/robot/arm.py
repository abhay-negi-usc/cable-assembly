"""Arm -- the one interface every robot backend implements, and the only one the task code sees.

THE LAYERS, top to bottom:

    apps/ (bnc_assembly, ...)       the task: what to do, in frames and forces, never joints by name
    skills/, behaviors/             scan, pick, insert, guarded and compliant motion
    Robot (robot.py)                arm + gripper + camera + the frame graph + motion primitives
    Arm (this module)               the robot-independent contract below
    backends/                       ur_rtde.URArm   UR10e over RTDE (the controller solves IK)
                                    iiwa_fri.IiwaFRIArm
                                                    KUKA iiwa 14 over FRI (IK, trajectories and
                                                    the wrench client-side; the FRI loop in its
                                                    own process, fri_driver.py)
                                    sim.SimArm      any model, no hardware -- the dry run
    kinematics/                     ur10e, iiwa14 -- pure numpy FK / Jacobian / IK

`robot.type` in the config picks the model (ur10e | iiwa14), and with it the backend
(backends.make_arm); `robot.dry_run` swaps in the simulator for either. Nothing above this module
may import a backend or name a joint: a robot difference the task needs is a property here
(`tool_roll_joint`, `tool_roll_window`, `joint_index('wrist_nod')`), answered by the backend.

UNITS AND FRAMES. Poses are 4x4 tool0-in-base matrices, base being ROS `base_link` (for the UR,
the backend bridges the controller's Rz(pi)-rotated `base`). Joints are rad, in the model's
JOINT_NAMES order. Wrenches are [fx, fy, fz, tx, ty, tz], N and Nm, tared, in base axes, with
the moment about the tool flange.

WHAT IS SHARED lives here, written once: the speed-cap arithmetic, the force guards polled
inside moves, IK + move composition, re-referencing a wrench to another frame, and the usable
tool-roll range. A backend supplies the primitives (state, IK, the moves, the servo step, the
wrench) and nothing else.
"""

import time

import numpy as np

from .. import log as urlog
from ..transforms import inverse, pose_error

log = urlog.get('arm')


class ArmError(RuntimeError):
    pass


_DEFAULT_LIMITS = (1.05, 1.2, 0.25, 0.0)     # rad/s, rad/s^2, m/s, rad/s (0 = unbounded)


def parse_limits(spd, base=_DEFAULT_LIMITS):
    """A `speed:`-style mapping -> (joint_vel, joint_accel, cart_vel, cart_rot) in SI units.

    The four canonical keys carry their unit in the name -- max_joint_velocity_deg_s,
    max_joint_acceleration_deg_s2, max_cartesian_translation_mm_s, max_cartesian_rotation_deg_s.
    Precedence preserves older configs' exact behaviour: the deg joint spellings win over the
    legacy rad ones (more explicit), but the legacy max_cartesian_velocity_m_s wins over the mm/s
    key (in older configs m/s governed moveL while mm/s only paced the compliant reference).
    Absent keys fall back to `base` -- pass the global limits there to make a partial override
    block (e.g. assembly.speed) inherit the rest."""
    spd = spd or {}
    jv = (np.radians(float(spd['max_joint_velocity_deg_s']))
          if spd.get('max_joint_velocity_deg_s') is not None
          else float(spd.get('max_joint_velocity_rad_s', 0.0)) or base[0])
    ja = (np.radians(float(spd['max_joint_acceleration_deg_s2']))
          if spd.get('max_joint_acceleration_deg_s2') is not None
          else float(spd.get('joint_acceleration_rad_s2', 0.0)) or base[1])
    mm = spd.get('max_cartesian_translation_mm_s')
    cv = (float(spd.get('max_cartesian_velocity_m_s', 0.0))
          or (float(mm) / 1000.0 if mm is not None else 0.0) or base[2])
    cr = (np.radians(float(spd['max_cartesian_rotation_deg_s']))
          if spd.get('max_cartesian_rotation_deg_s') is not None else base[3])
    return jv, ja, cv, cr


class Arm:
    """Blocking arm control. Every move returns True/False; nothing spins an executor.

    Subclasses implement the primitives marked "BACKEND" below and call Arm.__init__ first."""

    # True only for the simulator. Read by the callers that must not prompt an operator or wait
    # on hardware in a dry run -- never by motion code, which behaves the same either way.
    dry_run = False
    # forceMode-style controller compliance (the UR has it). Without it, compliance is the
    # software AdmittanceController over servo_l, which every backend supports.
    supports_force_mode = False
    # The controller's hard ceiling on a commanded joint speed, rad/s -- None if it clamps
    # silently or has none. set_speed_scale warns when a phase would land past it.
    JOINT_VELOCITY_CEILING = None

    def __init__(self, cfg, model):
        self.model = model
        self.joint_names = list(model.JOINT_NAMES)
        self.dof = int(model.DOF)
        self.tool_roll_joint = int(model.TOOL_ROLL_JOINT)
        # The (DOF, 2) limits, rad, this backend holds commands inside. The model's by default; a
        # backend may narrow them (the iiwa keeps a margin off the mechanical stops). Where the
        # CONTROLLER owns the real limits (the UR's safety configuration) joints_ok() asks it.
        self.joint_limits = np.array(model.JOINT_LIMITS, dtype=float)

        # Speed caps. The GLOBAL `speed:` block sets the four limits for every move; a caller may
        # pass a `caps=` mapping (same schema) to move_j / move_l to override them for one move
        # (e.g. assembly.speed).
        speed = cfg.section('speed')
        (self.max_joint_vel, self.joint_accel,
         self.max_cart_vel, self.max_cart_rot) = parse_limits(speed)
        self.cart_accel = float(speed.get('cartesian_acceleration_m_s2', 0.5))
        self.speed_scale = 1.0            # the CURRENT phase's scaling (set_speed_scale)

        self.move_timeout = float(cfg.get('move_timeout_s', 60.0))
        self.settle_s = float(cfg.get('settle_s', 0.5))
        self.debug = bool(cfg.get('debug', False))

        self._force_mode = False
        self._guards = []

    def joint_index(self, role):
        """The index of the joint playing `role` on this robot: 'tool_roll' (turns the tool about
        its own axis -- UR wrist_3, iiwa A7) or 'wrist_nod' (tips the wrist across it -- UR
        wrist_1, iiwa A6)."""
        try:
            return int(self.model.JOINT_ROLES[role])
        except KeyError:
            raise ValueError(f'{self.model.NAME} has no joint role {role!r}; known: '
                             f'{sorted(self.model.JOINT_ROLES)}') from None

    # ------------------------------------------------------------------ frames
    def publish_frames(self, frames, cfg):
        """Register base_link -> tool0 as a LIVE edge: read from the arm on every lookup, so it is
        fresh by construction and can never go stale the way a republished tf could."""
        self.base_frame = cfg.get('base_frame', 'base_link')
        self.tip_frame = cfg.get('tip_frame', 'tool0')
        frames.set_live(self.base_frame, self.tip_frame, self.tcp_pose)

    # ------------------------------------------------------------------ BACKEND: state
    def q(self):
        """Current joint positions (rad), in joint_names order."""
        raise NotImplementedError

    def tcp_pose(self):
        """Current tool0 pose in base_link, as a 4x4."""
        raise NotImplementedError

    def fk(self, q):
        """Forward kinematics: joints -> tool0 in base_link."""
        raise NotImplementedError

    def ik(self, T_base_tool0, qnear=None):
        """Inverse kinematics: tool0 pose in base_link -> joints (a list), or None if unreachable.
        Returns the solution nearest `qnear` (default: where the arm is now), so successive calls
        along a path stay on one branch and the arm does not flip its elbow mid-trajectory."""
        raise NotImplementedError

    def joints_ok(self, q):
        """True if `q` sits inside the joint limits the robot will actually enforce."""
        raise NotImplementedError

    # ------------------------------------------------------------------ BACKEND: motion
    def move_j(self, q_target, speed=None, accel=None, label='move', caps=None):
        """Joint-space move to `q_target`. Blocking; returns True on success. While a guard is
        armed (add_guard) it is polled during the motion, and a trip stops the arm and returns
        False."""
        raise NotImplementedError

    def move_l(self, T_base_tool0, speed=None, accel=None, label='move', caps=None):
        """Straight-line Cartesian move of tool0. Blocking, guarded like move_j."""
        raise NotImplementedError

    def servo_j(self, q, dt, lookahead=0.1, gain=300):
        """One step of a streamed joint reference. Returns once the period `dt` has elapsed, so a
        loop calling it runs at 1/dt."""
        raise NotImplementedError

    def servo_l(self, T_base_tool0, dt, lookahead=0.1, gain=300):
        """One step of a streamed CARTESIAN reference -- the software-admittance inner loop.
        Paced like servo_j. `lookahead` and `gain` are the UR servo's tuning; other backends may
        ignore them."""
        raise NotImplementedError

    def servo_stop(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ BACKEND: force / torque
    def zero_ft(self, settle=True):
        """Tare the wrench. Everything after this is CONTACT force -- so do it with the tool
        hanging free. `settle=False` skips any wait, for taring INSIDE a servo loop."""
        raise NotImplementedError

    def wrench(self):
        """Contact wrench [fx, fy, fz, tx, ty, tz], tared: force and the moment ABOUT THE TOOL
        FLANGE, both in base_link axes."""
        raise NotImplementedError

    def force_mode(self, T_base_task, selection, wrench, limits, mode=2, damping=0.005,
                   gain_scaling=0.8):
        """Controller compliance (see supports_force_mode)."""
        raise ArmError(f'{type(self).__name__} ({self.model.NAME}) has no controller force mode; '
                       'use the AdmittanceController, which runs on every backend.')

    def end_force_mode(self):
        """Leave force mode. Idempotent and safe to call from a finally:."""
        self._force_mode = False

    @property
    def in_force_mode(self):
        return self._force_mode

    # ------------------------------------------------------------------ BACKEND: lifecycle
    def disconnect(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disconnect()
        return False

    # ------------------------------------------------------------------ guards
    def add_guard(self, guard):
        """Arm a callable checked DURING every move; returning True cancels the motion.

        This is the force guard's hook. In the ROS stack the equivalent (_abort_move) could only
        be polled between executor spins; here the move loop owns the thread and polls at 100 Hz,
        so a collision stops the arm within ~10 ms of tripping instead of within a spin."""
        self._guards.append(guard)

    def clear_guards(self):
        self._guards = []

    def _tripped(self):
        for guard in self._guards:
            if guard():
                return True
        return False

    # ------------------------------------------------------------------ speed caps
    def set_speed_scale(self, scale, phase=''):
        """Set the CURRENT phase's scaling of the global speed limits (speed.phase_scale.<phase>).
        Applies to every following move until changed; a per-move `caps` still overrides it."""
        self.speed_scale = max(float(scale), 1e-3)
        log.info('Speed scale%s: %.2fx the global limits',
                 f' [{phase}]' if phase else '', self.speed_scale)
        # WARN WHEN THE PHASE IS ENTERED, not when a move inside it is rejected: the scale
        # multiplies the global caps, so a large one lands past the controller's own ceiling
        # and every move in the phase then runs clamped. Said here, it names the phase to fix.
        jv = self.max_joint_vel * self.speed_scale
        ceiling = self.JOINT_VELOCITY_CEILING
        if ceiling is not None and jv > ceiling:
            log.warning('speed.phase_scale%s = %.2f puts the joint cap at %.2f rad/s '
                        '(%.0f deg/s), past the %.2f rad/s the controller accepts -- every '
                        'move in this phase will be CLAMPED to the ceiling. Lower the scale, '
                        'or raise the CARTESIAN caps instead (they are what actually pace a '
                        'free-space move).',
                        f' [{phase}]' if phase else '', self.speed_scale, jv,
                        np.degrees(jv), ceiling)

    def _limits(self, caps):
        """The four limits for one move. None = the global limits x the CURRENT phase scale
        (set_speed_scale). Otherwise `caps` REPLACES the phase scale for this move: either a
        `speed:`-style mapping (absolute overrides; absent keys inherit the global values), or a
        bare NUMBER scaling all four global limits (e.g. 0.2 = a fifth of every limit)."""
        mine = (self.max_joint_vel, self.joint_accel, self.max_cart_vel, self.max_cart_rot)
        if caps is None:
            s = self.speed_scale
            return tuple(v * s for v in mine)
        if isinstance(caps, (int, float)):
            return tuple(v * float(caps) for v in mine)
        return parse_limits(caps, mine)

    def _speeds(self, q_target, caps=None):
        """(joint speed, accel, name-of-binding-cap) honouring EVERY cap -- whichever binds.

        The joint caps are direct. Each Cartesian cap is converted into an equivalent joint
        speed: the tool travels `d` (metres, or radians of tool rotation) while the largest joint
        travels `dj` radians, so holding the tool under the cap means holding that joint under
        `dj/d * cap`. The min over all of them respects every limit simultaneously."""
        jv, ja, cv, cr = self._limits(caps)
        q_now = np.asarray(self.q(), dtype=float)
        dj = float(np.max(np.abs(np.asarray(q_target, dtype=float) - q_now)))
        speed, which = jv, 'joint-velocity'
        if dj > 1e-6 and (cv > 0.0 or cr > 0.0):
            lin, ang = pose_error(self.tcp_pose(), self.fk(q_target))
            # SANITY-GATE the ratios: lin/dj is the move's effective lever arm, physically bounded
            # by the reach (~1.3 m; tool rotation similarly by the sum of joint rotations). A ratio
            # beyond that means the pose delta is NOT this move's motion but the constant few-mm
            # disagreement between the measured pose and the FK model -- which DOMINATES when the
            # arm is already AT the target (dj ~ 0) and would collapse the bound to the 1e-3 floor
            # ("moveJ 0.00 rad/s"). Sub-mm moves need no cartesian pacing either way.
            if cv > 0.0 and 1e-3 < lin <= 2.0 * dj and dj / lin * cv < speed:
                speed, which = dj / lin * cv, 'cartesian-translation'
            if cr > 0.0 and 1e-3 < ang <= 8.0 * dj and dj / ang * cr < speed:
                speed, which = dj / ang * cr, 'cartesian-rotation'
        return max(speed, 1e-3), ja, which

    # ------------------------------------------------------------------ composed motion
    def _settle(self):
        """Let the arm come to rest after a move before anyone measures. Real arms wait."""
        time.sleep(self.settle_s)

    def move_to(self, T_base_tool0, label='move', qnear=None):
        """IK + joint move to a tool0 pose. The workhorse: every app move goes through here."""
        if self.debug:
            from ..transforms import fmt_delta, fmt_pose
            log.info('[%s] target %s', label, fmt_pose(T_base_tool0))
            log.info('[%s] delta from current: %s', label,
                     fmt_delta(self.tcp_pose(), T_base_tool0))
        q = self.ik(T_base_tool0, qnear)
        if q is None:
            from ..transforms import fmt_delta, fmt_pose
            log.error('[%s] IK UNREACHABLE.', label)
            log.error('  current: %s', fmt_pose(self.tcp_pose()))
            log.error('  target:  %s', fmt_pose(T_base_tool0))
            log.error('  delta:   %s', fmt_delta(self.tcp_pose(), T_base_tool0))
            return False
        if not self.move_j(q, label=label):
            return False
        self._settle()
        return True

    def move_frame_to(self, T_base_target, T_tool0_frame, label='move', qnear=None):
        """Move so that the frame rigidly attached to tool0 by `T_tool0_frame` lands on
        `T_base_target`.

        This is how a fingertip, a grasp TCP, or a held object gets commanded: you say where the
        THING should go, and the tool0 pose that puts it there is inv()'d out. Every demo that
        controls something other than the flange goes through this one method."""
        return self.move_to(T_base_target @ inverse(T_tool0_frame), label, qnear)

    def tool_roll_window(self, q_ref, span=2.0 * np.pi, tol=np.radians(0.25)):
        """The contiguous interval of the TOOL-ROLL joint, in rad, that joints_ok() accepts
        around `q_ref[tool_roll_joint]`, with every other joint held at `q_ref`.

        A twist of the tool (connector or collar clocking) spends exactly this range, so it is
        asked rather than assumed: on the UR it is bisected against the CONTROLLER's joint-limit
        check, because the usable wrist_3 range is an installation safety setting that can be
        narrower than the model's +/-360; on the iiwa it is A7's +/-175. Capped at `span` each
        way."""
        j = self.tool_roll_joint
        q0 = float(q_ref[j])

        def ok(v):
            q = list(q_ref)
            q[j] = float(v)
            return self.joints_ok(q)

        def edge(direction):
            if not ok(q0 + direction * tol):
                return q0                                  # already at the limit
            if ok(q0 + direction * span):
                return q0 + direction * span               # nothing within `span`
            good, bad = tol, span                          # good is inside, bad is outside
            while bad - good > tol:
                mid = 0.5 * (good + bad)
                if ok(q0 + direction * mid):
                    good = mid
                else:
                    bad = mid
            return q0 + direction * good

        return edge(-1.0), edge(+1.0)

    # ------------------------------------------------------------------ wrench, re-referenced
    def wrench_in(self, T_base_frame, T_base_flange=None):
        """The contact wrench moved TO another frame's origin and expressed IN its axes.

        Two separate operations, and the first one is the easy one to skip:

          1. MOVE THE REFERENCE POINT. wrench() reports the moment about the TOOL FLANGE. A
             moment is only meaningful about a stated point, so re-referencing it to the target
             frame's origin is a physical change, not a rotation:
             tau_new = tau + (p_flange - p_target) x f.
          2. ROTATE into the target frame's axes: R^T applied to both halves.

        SKIPPING STEP 1 IS A SILENT, LARGE ERROR. Composing only the rotation implicitly claims
        the moment was about the BASE ORIGIN, which adds a phantom lever of |base -> flange| --
        about a metre on these arms. Measured on the 2026-08 BNC logs before this fix: the logged
        connector torque was 90% explained (R^2 0.90) by a single cross product p x f with
        |p| = 1024 mm, against a predicted |base -> flange| of 1050 mm, and tau came out
        perpendicular to f in 98.8% of rows -- the signature of a pure lever-arm artefact rather
        than a contact moment. A BNC contact moment should show a lever of a few millimetres.

        `T_base_flange` lets a caller that already read the arm pose pass it in: it saves a
        second round trip and, more importantly, guarantees the pose and the wrench come from the
        SAME sample rather than two reads a cycle apart.
        """
        w = self.wrench()
        f_b, tau_b = np.asarray(w[:3], dtype=float), np.asarray(w[3:], dtype=float)
        T = np.asarray(T_base_frame, dtype=float)
        p_flange = (np.asarray(T_base_flange, dtype=float)[:3, 3] if T_base_flange is not None
                    else self.tcp_pose()[:3, 3])
        tau_b = tau_b + np.cross(p_flange - T[:3, 3], f_b)     # (1) re-reference the moment
        R = T[:3, :3]
        return np.concatenate([R.T @ f_b, R.T @ tau_b])        # (2) rotate into the frame

    def force(self):
        return float(np.linalg.norm(self.wrench()[:3]))

    def torque(self):
        return float(np.linalg.norm(self.wrench()[3:]))
