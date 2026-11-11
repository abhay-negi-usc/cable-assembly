"""AdmittanceController -- the admittance control law, in software, over servoL.

Reimplements the ros2_control admittance_controller the dev branch used, so there is a real,
finite RESTORING STIFFNESS again (forceMode has none -- it is pure force control and yields
freely). No controller to install, load, switch, or parameterise: this is a plain servo loop.

THE LAW, per axis (the frame it RUNS in is tool0 -- see FRAME below):

    F_ext = M x'' + D x' + S (x - x_d)      =>     x'' = M^-1 ( F_ext - D x' - S (x - x_d) )

integrated each cycle (v += x'' dt ; x += v dt). We track the DISPLACEMENT delta = x - x_d from
the reference rather than x itself: contact force grows delta, the spring S pulls it back to zero,
so when contact eases the arm RETURNS to the reference (the ideal insertion trajectory). A static
force F holds a steady deflection delta = F / S -- with S = 2000 N/m, 20 N deflects 10 mm.

    D = damping_ratio * 2 sqrt(M S)         (damping_ratio = 1 -> critically damped)

SIGN + FRAME BRIDGE. F_ext is the EXTERNAL force ON the tool (what compliance YIELDS to): a push in
+x must grow delta in +x so the tool moves WITH the push. arm.wrench() already carries that sign --
but it must also be in ROS base_link, NOT the raw UR `base` that getActualTCPForce() reports (they
differ by Rz(pi), so x/y come out negated and z does not). arm.wrench() applies that bridge. If it
is ever dropped, the symptom is the giveaway: SOME axes comply and others push back. A global sign
error would invert ALL axes together; a per-axis split is always a FRAME error.

FRAME. The law runs in the TOOL0 frame, matching the dev controller. arm.wrench() reports the
wrench at the TCP in ROS base_link axes (never say just "base" here -- that ambiguity IS the bug
above), so each cycle it is rotated into tool0 (by R^T, R = the tool0 orientation in base_link -- a
pure rotation, no lever arm, since the wrench is already referenced to the TCP = tool0 origin). delta is then a TOOL0-frame displacement -- delta[:3] a translation and
delta[3:] a rotation vector about the TOOL axes -- and it is applied to the reference by
POST-multiplying: T_cmd = T_ref @ Delta. So "z compliant" means yielding along the tool's own z
wherever the tool points, which is what an insertion wants.

SAFETY. delta is clamped (max_delta_m / max_delta_rad): a bad tare or an F/T fault would otherwise
integrate into a runaway. The caller still arms a force guard on top for the hard limit.

FORCE NULLING (null_axes; the user, 2026-11-06, for connector clocking stuck on misalignment and friction): on
the axes named, NO SPRING -- S = 0 and a plain damper D, so a force there drives the pose at F/D until it is
gone, instead of being held as a preload S x offset. A deadband keeps it from chasing the wrench estimate's error
(what the null holds is the MEASURED force at zero, so the estimate's error becomes a real load); a per-axis
clamp bounds how far it may go. Usually with set_frame(): the law run at the part, in its own axes.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .. import log as urlog

log = urlog.get('admittance')


class AdmittanceController:
    """Software admittance over servoL. Holds the integrator state so a chunked insertion is one
    continuous, compliant motion."""

    def __init__(self, arm, cfg_section):
        c = cfg_section or {}
        self.arm = arm
        self.M = np.asarray(c.get('mass', [5.0, 5.0, 5.0, 0.5, 0.5, 0.5]), dtype=float)
        self.S = np.asarray(c.get('stiffness', [2000.0] * 3 + [15.0] * 3), dtype=float)
        zeta = np.asarray(c.get('damping_ratio', [1.0] * 6), dtype=float)
        self.D = zeta * 2.0 * np.sqrt(self.M * self.S)          # critically damped at zeta = 1
        for i, d in enumerate(c.get('damping') or []):           # explicit per axis (N s/m, N m s/rad)
            if d is not None:
                self.D[i] = float(d)
        self.selected = np.asarray([1.0 if v else 0.0
                                    for v in c.get('selected_axes', [1] * 6)], dtype=float)
        self.rate = float(c.get('reference_rate_hz', 125.0))
        self.lookahead = float(c.get('lookahead_time_s', 0.1))
        self.gain = float(c.get('servo_gain', 300.0))
        # Warm-up: hold the start pose this long before the guard is trusted. servoL ENGAGING
        # produces a joint-torque transient that getActualTCPForce reports as tens of N of phantom
        # force -- enough to trip a 30 N guard on the very first cycle, before anything is touched.
        # Holding still lets it decay first.
        self.warmup_s = float(c.get('warmup_s', 0.5))
        self.max_delta = float(c.get('max_delta_m', 0.05))      # runaway clamp (translation)
        self.max_delta_rot = float(c.get('max_delta_rad', 0.5))
        self.clamp = np.array([self.max_delta] * 3 + [self.max_delta_rot] * 3)   # per axis (null_axes)
        # DEADBANDS, per tool axis: a force under force_deadband_n (a moment under
        # torque_deadband_nm) is not yielded to, and over it only the excess is. For an arm whose
        # wrench is ESTIMATED from its joint torques (the iiwa), whose error while moving -- friction
        # that flips with each joint's direction -- would otherwise drive the spring, the motion flip
        # the friction, and the two cycle: +-3.5 mm at 0.8 Hz with nothing touched (2026-10-30). 0 = off.
        self.deadband = np.array([float(c.get('force_deadband_n', 0.0))] * 3
                                 + [float(c.get('torque_deadband_nm', 0.0))] * 3)
        # THE FRAME THE LAW RUNS IN, in tool0 (default: tool0 itself). The RL insertion's runs about
        # the point on the plug its policy was trained with the arm's spring centred on
        # (skills/rl_insertion.py): the wrench is referred to that point, in its axes, and Delta
        # displaces it there -- T_cmd = T_ref @ F @ Delta @ F^-1.
        self.frame = np.asarray(c.get('frame', np.eye(4)), dtype=float)
        self._frame_inv = np.linalg.inv(self.frame)
        self.nulled = []                                          # the axes null_axes() freed
        self._delta = np.zeros(6)
        self._vel = np.zeros(6)
        # LAST CYCLE, FOR DIAGNOSIS. The commanded pose is `ref @ Delta`, and when an
        # insertion misbehaves the first question is always WHICH OF THE TWO MOVED: the
        # reference (a trajectory/waypoint problem) or Delta (a controller/compliance one).
        # A log of the measured pose alone cannot tell those apart, so record both.
        self._last_ref = None
        self._last_cmd = None

    def set_frame(self, T_tool0_frame):
        """Run the law at `T_tool0_frame` (a frame in tool0: the held part's), in its axes -- see FRAME. Only
        between motions (it re-interprets delta): call before reset()."""
        self.frame = np.asarray(T_tool0_frame, dtype=float)
        self._frame_inv = np.linalg.inv(self.frame)

    def zero_axes(self, axes):
        """Zero delta and its rate on `axes` -- ONLY in the breath the caller moves the reference by what they held
        (connector clocking's axis probe: the lateral give it measured becomes the new screw axis). Otherwise the
        command steps by -delta there: see reset()."""
        for i in axes:
            self._delta[i] = 0.0
            self._vel[i] = 0.0

    def null_axes(self, axes, damping, deadband=None, max_offset=None):
        """FORCE NULLING on `axes` (indices into the law frame's [x, y, z, rx, ry, rz]): no spring, a damper
        `damping` (F = D v), the force under `deadband` not yielded to, at most `max_offset` (m / rad) of give.
        See FORCE NULLING above."""
        for i in axes:
            self.S[i] = 0.0
            self.D[i] = float(damping)
            self.selected[i] = 1.0
            if deadband is not None:
                self.deadband[i] = float(deadband)
            if max_offset is not None:
                self.clamp[i] = float(max_offset)
        self.nulled = sorted(set(self.nulled) | set(int(i) for i in axes))

    @property
    def delta(self):
        """The compliant offset in the tool0 frame, [m,m,m,rad,rad,rad]. Read-only copy."""
        return self._delta.copy()

    @property
    def last_ref(self):
        """The reference pose commanded on the most recent cycle (None before the first)."""
        return None if self._last_ref is None else self._last_ref.copy()

    @property
    def last_cmd(self):
        """The pose actually sent to servoL last cycle -- `last_ref @ Delta`."""
        return None if self._last_cmd is None else self._last_cmd.copy()

    def reset(self):
        """Zero the integrator -- call before an insertion so it starts on the reference.

        THIS IS AN ASSERTION, NOT A MOTION: "the tool is AT the reference, and unloaded". The
        commanded pose is `T_ref @ Delta`, so zeroing Delta moves the COMMAND by exactly -Delta.
        Call it while the spring is loaded and the arm steps by the whole accumulated deflection
        in one cycle -- into whatever it was pressing against.

        SO IT IS ONLY VALID WHEN ONE OF THESE HOLDS:

          1. Delta is already ~0 -- free space, nothing touching. The normal case: move_j to the
             start, then reset() + warmup(start).
          2. The caller re-references to where the tool ACTUALLY is in the same breath. That is
             what `rebase()` does, and it is the only safe way to reset under load.

        Neither is checked here, because the controller cannot know which pose the caller is about
        to command. It has been got wrong: resetting mid-contact while keeping a reference 30 mm
        deep drove the connector 30 mm further into the socket, then let it spring back out.
        """
        self._delta = np.zeros(6)
        self._vel = np.zeros(6)

    def rebase(self, T_measured):
        """Re-reference the spring onto where the tool ACTUALLY is, and return that pose.

        The safe way to clear the integrator UNDER LOAD: the deflection is discarded and the
        reference becomes the deflected pose, so the commanded pose does not move at all. Use it
        whenever a compliant motion is interrupted -- an operator prompt, a phase change -- and
        the next one has to start from reality rather than from a stale reference.

        The trade, stated plainly: this FORGETS the contact equilibrium, so the spring will begin
        yielding again from zero against whatever force is still present, and the tool will drift
        off the contact until that force decays. If the preload is meant to be HELD, keep the
        reference and the deflection instead of rebasing.
        """
        self.reset()
        return np.array(T_measured, dtype=float)

    def warmup(self, T_ref, seconds=None, tare_fn=None):
        """Hold T_ref via servoL for `seconds` (default warmup_s), engaging servo mode and letting
        the joint-torque transient in getActualTCPForce decay BEFORE the guard is armed.

        If `tare_fn` is given it is called MIDWAY -- once the servo is engaged and the arm is
        holding still -- so the F/T is zeroed against the SERVO-ACTIVE reading the guard will
        actually see. This matters when the payload is not configured: taring while idle leaves an
        offset (the tool weight) that only appears once the arm is under active control. servoL
        keeps streaming around the tare so servo mode is not dropped, so `tare_fn` must NOT block
        (use zero_ft(settle=False)). The ODE is not run here; the integrator is zeroed afterward."""
        s = self.warmup_s if seconds is None else seconds
        if self.arm.dry_run:
            return
        if s <= 0:
            if tare_fn is not None:
                tare_fn()
            return
        dt = 1.0 / self.rate
        steps = max(2, int(s * self.rate))
        for i in range(steps):
            self.arm.servo_l(T_ref, dt, self.lookahead, self.gain)
            if tare_fn is not None and i == steps // 2:    # tare while servo-active and static
                tare_fn()
        self.reset()

    def _command_pose(self, T_ref):
        """The reference tool0 pose displaced by the compliant delta IN THE TOOL0 FRAME
        (post-multiply): T_cmd = T_ref @ Delta."""
        Delta = np.eye(4)
        Delta[:3, :3] = Rotation.from_rotvec(self._delta[3:]).as_matrix()
        Delta[:3, 3] = self._delta[:3]
        return T_ref @ self.frame @ Delta @ self._frame_inv

    def _step(self, T_ref, dt, on_step=None):
        """Advance the ODE one cycle (in the tool0 frame) and command the compliant pose. `on_step`
        (if given) is called AFTER the servo command each cycle -- a hook for data logging."""
        R = T_ref[:3, :3] @ self.frame[:3, :3]              # the law's frame's orientation in base_link
        wb = np.asarray(self.arm.wrench(), dtype=float)     # external force ON the tool, base_link (tared)
        tb = wb[3:]
        if self.frame[:3, 3].any():                         # the moment about the frame's origin
            tb = tb + np.cross(-(T_ref[:3, :3] @ self.frame[:3, 3]), wb[:3])
        w = np.concatenate([R.T @ wb[:3], R.T @ tb]) * self.selected   # -> the frame's axes
        w = np.sign(w) * np.maximum(np.abs(w) - self.deadband, 0.0)
        accel = (w - self.D * self._vel - self.S * self._delta) / self.M
        self._vel += accel * dt
        self._delta += self._vel * dt
        at = np.abs(self._delta) >= self.clamp
        self._delta = np.clip(self._delta, -self.clamp, self.clamp)
        self._vel[at & (np.sign(self._vel) == np.sign(self._delta))] = 0.0   # no wind-up against the clamp
        T_cmd = self._command_pose(T_ref)
        self._last_ref, self._last_cmd = np.array(T_ref, dtype=float), T_cmd
        self.arm.servo_l(T_cmd, dt, self.lookahead, self.gain)
        if on_step is not None:
            on_step()

    def ramp(self, T_ref_start, T_ref_end, duration, guard=None, on_step=None):
        """Ramp the tool0 reference start -> end over `duration` s under admittance. Returns
        'seated' if the guard trips (contact), else 'done'. Integrator state persists across calls,
        so consecutive ramps form one continuous compliant motion. `on_step` (if given) is called
        once per servo cycle -- used by data-collection callers to log at the servo rate."""
        from ..transforms import slerp_matrix
        dt = 1.0 / self.rate
        steps = max(1, int(duration * self.rate))
        for k in range(1, steps + 1):
            self._step(slerp_matrix(T_ref_start, T_ref_end, k / steps), dt, on_step)
            if guard is not None and guard.check():
                return 'seated'
        return 'done'

    def ramp_joint_path(self, q_start, q_end, duration, guard=None, waypoints=60, tare_fn=None):
        """Admittance around a JOINT-interpolated reference path -- for a compliant move to a joint
        target (e.g. a home reset). The tool follows a predictable joint-space path (no
        Cartesian-slerp singularity surprises from an arbitrary start) while the spring yields to
        contact. FK is evaluated at `waypoints` coarse points and slerped between, so the servo
        loop stays real-time (no per-cycle FK). `tare_fn` (if given) tares mid-warmup, once the
        servo is engaged. Returns 'seated' if the guard trips, else 'done'."""
        from ..transforms import slerp_matrix
        dt = 1.0 / self.rate
        steps = max(1, int(duration * self.rate))
        q0, q1 = np.asarray(q_start, dtype=float), np.asarray(q_end, dtype=float)
        n = max(1, min(int(waypoints), steps))
        wp = [self.arm.fk(q0 + (q1 - q0) * (j / n)) for j in range(n + 1)]   # precompute FK
        self.warmup(wp[0], tare_fn=tare_fn)         # settle the servo (+ tare) before the guard
        if guard is not None:
            guard.reset()
        for k in range(1, steps + 1):
            u = (k / steps) * n
            j = min(int(u), n - 1)
            self._step(slerp_matrix(wp[j], wp[j + 1], u - j), dt)
            if guard is not None and guard.check():
                return 'seated'
        return 'done'

    def hold(self, T_ref, seconds, guard=None, on_step=None):
        """Hold the reference for `seconds` under admittance (let the mate settle / keep yielding)."""
        return self.ramp(T_ref, T_ref, seconds, guard, on_step)

    def give(self):
        """How far the tool has yielded from the command's reference, in the law frame's axes [m, m, m, rad, rad,
        rad]: this controller's delta (the pose follows its command rigidly)."""
        return self._delta.copy()

    def stop(self):
        self.arm.servo_stop()


# ---------------------------------------------------------------------------- the KUKA's own impedance
KUKA_DEFAULT_STIFFNESS = [2000.0, 2000.0, 2000.0, 200.0, 200.0, 200.0]   # CartesianImpedanceControlMode's own


class KukaImpedanceController(AdmittanceController):
    """THE KUKA'S CARTESIAN IMPEDANCE in place of the software admittance (the user, 2026-11-06: "write a new
    version of the script which uses the kuka's impedance control in place of admittance control").

    With the FRI app in CARTESIAN_IMPEDANCE_CONTROL (client command mode POSITION), the joint positions streamed
    are the set-point of the controller's own spring-damper, run at 1 kHz on its joint-torque sensors. So this
    streams the REFERENCE as it is -- no mass, no spring, no damper integrated here: a contact deflects the arm
    off the command by F / K_kuka, and the KUKA, not this loop, decides how. Same interface, so every maneuver
    runs unchanged (ramp, hold, warmup, guards, on_step).

    WHAT STAYS HERE:
      * FORCE NULLING (null_axes): on those axes the set-point DRIFTS with the measured force, at (|F| - deadband)
        / D, clamped -- the KUKA spring in series. The preload a misaligned part would hold is walked off.
      * S is the KUKA's stiffness (robot.fri.cartesian_impedance.stiffness, which must match the pendant's
        app): the maneuvers that size a pull or a push from force / stiffness (tug verify, the collar's seat
        push, the engage's unload) then size it for the spring that is really there.
    WHAT DOES NOT: the per-maneuver stiffness / mass / damping_ratio / force deadbands in the config -- the KUKA's
    impedance is set once for the whole FRI session. give() reads the real deflection (measured vs commanded)."""

    kuka = True

    def __init__(self, arm, cfg_section):
        super().__init__(arm, cfg_section)
        K = getattr(arm, 'kuka_stiffness', None)
        self.S = np.asarray(KUKA_DEFAULT_STIFFNESS if K is None else K, dtype=float).copy()
        self.selected = np.ones(6)                       # the KUKA's spring acts on every axis

    def _step(self, T_ref, dt, on_step=None):
        if self.nulled:
            R = T_ref[:3, :3] @ self.frame[:3, :3]
            wb = np.asarray(self.arm.wrench(), dtype=float)
            tb = wb[3:]
            if self.frame[:3, 3].any():
                tb = tb + np.cross(-(T_ref[:3, :3] @ self.frame[:3, 3]), wb[:3])
            w = np.concatenate([R.T @ wb[:3], R.T @ tb])
            w = np.sign(w) * np.maximum(np.abs(w) - self.deadband, 0.0)
            for i in self.nulled:
                self._delta[i] = float(np.clip(self._delta[i] + w[i] / max(self.D[i], 1e-9) * dt,
                                               -self.clamp[i], self.clamp[i]))
        T_cmd = self._command_pose(T_ref)
        self._last_ref, self._last_cmd = np.array(T_ref, dtype=float), T_cmd
        self.arm.servo_l(T_cmd, dt, self.lookahead, self.gain)
        if on_step is not None:
            on_step()

    def give(self):
        """The KUKA spring's deflection: where the tool IS against where it was last commanded, in the law frame's
        axes [m, m, m, rad, rad, rad] (zeros before the first command, or if the pose cannot be read)."""
        if self._last_cmd is None:
            return np.zeros(6)
        try:
            T_meas = np.asarray(self.arm.tcp_pose(), dtype=float)
        except Exception:                                # noqa: BLE001 -- a diagnostic
            return np.zeros(6)
        d = np.linalg.inv(self._last_cmd @ self.frame) @ (T_meas @ self.frame)
        return np.concatenate([d[:3, 3], Rotation.from_matrix(d[:3, :3]).as_rotvec()])


def make_compliance(arm, cfg_section):
    """The compliance a maneuver runs under: the software admittance (robot.compliance_controller admittance, the
    default) or the KUKA's own Cartesian impedance (kuka_impedance; an iiwa over FRI, or a dry run)."""
    mode = str(getattr(arm, 'compliance', 'admittance') or 'admittance').lower()
    if mode == 'admittance':
        return AdmittanceController(arm, cfg_section)
    if mode == 'kuka_impedance':
        if not getattr(arm, 'supports_kuka_impedance', False) and not getattr(arm, 'dry_run', False):
            raise ValueError(f'robot.compliance_controller kuka_impedance needs a KUKA iiwa over FRI, not '
                             f'{type(arm).__name__}')
        return KukaImpedanceController(arm, cfg_section)
    raise ValueError(f"robot.compliance_controller {mode!r}: 'admittance' or 'kuka_impedance'")
