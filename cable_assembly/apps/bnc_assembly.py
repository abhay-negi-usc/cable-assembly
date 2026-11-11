"""BNC ASSEMBLY -- cable_pick_estimate_assemble's pipeline with estimator_eval's estimator.

Same shape as cable_pick_estimate_assemble (scan, grasp, slip-checked lift, stand-off, then an
assemble/estimate loop that corrects the in-hand belief), but everything downstream of the grasp
is estimator_eval's:

  * ESTIMATOR: the same `estimation:` block, including `commit: argmin`. `_argmin_estimate` and
    `_landscape` are IMPORTED from estimator_eval rather than reimplemented, so the two apps
    cannot drift.
  * COLLECTION: `assembly.collection.mode` = attempts | offset_sweep | peck. offset_sweep
    commands one insertion per deliberate offset with the belief held FIXED across passes (so
    their evidence fuses exactly); peck keeps advancing past each force stop.
  * COMPLIANCE / GUARD: read from the TOP-LEVEL `compliance:` and `force_guard:` blocks, the same
    names estimator_eval uses, so tuned values copy across verbatim.
  * FINAL INSERTION: `assembly.final_insertion` -- stiffness, mass, damping, settle, dwell and the
    guard overrides in one place, each inheriting the shared value when null.

STATE VOCABULARY -- the four words this app reports progress in, in order:

    ENGAGED     the initial assembly mated the connector. Where the estimate/insert loop ends.
    SEATED      connector clocking succeeded: the bayonet cams pulled the connector home.
    LOCKED      collar clocking succeeded: the locking collar has been turned.
    ASSEMBLED   all of the above -- the connector/cable is done.

`CLOCK_STATES` is the progression and the code WALKS it rather than setting flags, so a state
cannot be reported without the step that earns it having succeeded.

CAREFUL -- 'seated' IS OVERLOADED. `AdmittanceController.ramp` returns 'seated' to mean "a guard
tripped and I stopped early"; that is the ROBOT layer's word and says nothing about the assembly
state. The clocking code reads that return into a local named `stopped`.

WHAT THIS APP ADDS BEYOND THE MATE -- two operations that run only once the connector is ENGAGED
and the operator has called the assembly successful, each with its own compliance, force guard and
speed scale (`assembly.connector_clocking`, `assembly.collar_clocking`; a failed screw and a finished
collar turn share one escape, `assembly.clocking_retract`):

  * CONNECTOR CLOCKING. An OSCILLATING screw about the connector's +X -- rock between the roll
    positions in `sweep_deg` while pushing along that same axis at a VIRTUAL target past where the
    connector can physically go, so compliance follows whatever path the bayonet cams allow. It
    rocks rather than turning further because a pin that missed its slot will not find it by
    turning harder -- it rides the rim and jams -- but it will find it by crossing back and forth
    under a steady axial load, which is held across every reversal. Progress is MEASURED (advance
    along +X) and ends the motion the moment it is reached; otherwise the legs run to `max_tries`.
    A leg stopped by the force guard is normal -- the next leg reverses from where it stopped. The
    gripper stays CLOSED throughout; there is no regrasp.
  * COLLAR CLOCKING. Only if the sweep succeeded. The gripper takes the ring AXIALLY -- fingers
    parallel to the cable, jaws closing across a diameter -- which puts tool0 ON the connector
    axis with its Z collinear, so turning the collar about that axis is a WRIST TWIST with the
    flange stationary. The socket is wall-mounted and that is what decides it: the obvious grasp
    (fingertip frame on the collar frame, approaching from the side) holds tool0 183 mm off the
    axis at +25 mm PAST the mating face and sweeps it through a 258 mm arc across the wall;
    axially it sits at -158 mm and travels 0 mm. Getting there is no longer an orbit, so the
    sequence seat-pushes where it stands, withdraws ALONG the cable, lifts the open fingers off,
    reorients in clear space, and advances back down the axis -- threading the cable through the
    open jaw. Every station is on the axis and gated against `wall_standoff_mm` before it moves.

TWO KINDS OF ANGLE, and keeping them apart is most of the arithmetic:

  * ABSOLUTE roll about the socket +X, wrt the TARGET frame -- `assembly.engage_clock_deg` and
    `connector_clocking.sweep_deg`. These are the angles obstacles and wrist limits live in.
  * ROTATIONS from the pose the connector was ENGAGED at -- what every stroke is built from, and
    what `collar_clocking.rotation_deg` / `prewind_deg` are measured in.

`engage_clock_deg` converts between them: it rolls the socket frame about its +X ONCE, and every
pose in the app is built from the rolled frame. As shipped, engage_clock_deg 0 with sweep_deg
[-75, +75] works the -75 .. +75 band: engage at 0, rock to -75, +75, -75, unwind to 0, turn the
collar to +90.

Two things are deliberately NOT rolled with it: `collar_clocking.axis_offset_mm`, a bench-measured
property of the FIXTURE that stays put while the plug turns (see `axis_offset_base`), and the
contact manifold, collected at one clock angle and describing different contact at another -- so
`insertion_mode: estimate` warns when the engage angle is non-zero.

THE BELIEF RESET at the start of connector clocking is the load-bearing idea. Once the mate is made the
connector's pose is known from a PHYSICAL CONSTRAINT -- it is at the target -- so that replaces the
estimate and the screw axis becomes the target's +X exactly instead of inheriting the accumulated
in-hand error.

WHAT IS NECESSARILY DIFFERENT from estimator_eval: the part is really picked and the true in-hand
pose is unknown, so there is no injected error, no err_before/after, and no truth to draw. Success
is the OPERATOR's call at the check (a dry run falls back to the kinematic tolerance).

OTHER PLUGS (Task): the pipeline up to the insertion is shared. apps/ethernet_assembly runs it for an
RJ45 (Task latch=True): no clocking -- the engage pushes it home (assembly.engage.push_home), the tug
checks its latch, a plug that comes back out is backed out and laid down like a missed engage. Its
latch side is measured by the 6D scan and carried in the belief (latch_turn, P6.grasp_junction), and
its target keeps the jack's turn (engage_clock_deg does not apply).

Units: robot poses are metres/radians; the manifold space is mm/deg. The conversion happens only at
the observation/correction boundary.
"""

import contextlib
import csv as _csv
import json
import logging
import os
import sys
import time
from datetime import datetime


import numpy as np

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..log import StepRunner
from ..perception import pose6d as P6
from ..robot import ForceGuard, make_compliance
from ..skills import manifold_debug, marker_localize as mloc, reset
from ..skills import trajectory as traj
from ..skills import wiggle as wigmod
from ..skills.manifold import mats_from_vec6, vec6_from_mats
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, belief_offset, connector_axis_height_m,
                           fingertip_in_connector, held_belief, offset_belief, retry_offset_x,
                           rigidly_held_junction, verify_cable_held)
from ..skills.pose6d_pick import OnlinePoses, in_hand_tracker, online_cfg, socket_tracker
from ..skills.solution_check import CheckedManifoldEstimator
from ..transforms import (from_cfg, inverse, matrix_to_xyzrpy, pose_error,
                          rotate_about_axis, slerp_matrix, translation_matrix, xyzrpy_to_matrix)
from ._cable import build_scanner, make_confirm
from ._common import prompts_off
from ._runner import run_app
from .cable_pick_assemble import _guarded as _guarded_move, _pick
from .cable_pick_estimate_assemble import (_corr_to_m, _observe, _plot_run, _save_observations)
from .estimator_eval import _argmin_estimate, _landscape
from .uncertain_sampling import _retract_ref

log = urlog.get('bnc-assembly')

# The assembly state progression, walked in order (see the module docstring).
CLOCK_STATES = ('engaged', 'seated', 'locked')

# DISASSEMBLY walks the same ladder DOWNWARD and then one rung below the bottom. 'removed' is
# not a clocking state -- it means the connector is out of the socket and in the fingers, which
# is the only state from which placing it down is meaningful.
UNCLOCK_STATES = ('locked', 'seated', 'engaged', 'removed')


class Task:
    """WHAT THE RUN ASSEMBLES. The pipeline up to the insertion is the same for every plug -- localize the
    socket, pick the cable, the connector in the hand, the stand-off, the insertion -- and so is the campaign
    round it; what comes after the insertion is the part's:

      the BNC (BNC, this app)   a bayonet: connector clocking seats it, collar clocking locks it, the tug
                                checks the lock, the disassembly undoes all three and puts the cable back.
      a LATCH (latch=True;      pushed home (assembly.engage.push_home), its latch clicks over by itself; the
      apps/ethernet_assembly)   tug, the grip kept, checks it caught -- a plug that comes back out is a miss,
                                backed out and laid down like one. No clocking, no disassembly (the latch is
                                not released by pulling): the cell is reset by hand between trials.

    `name` names the run's folders and records, `title` the tree."""

    def __init__(self, name, title, latch=False):
        self.name, self.title, self.latch = str(name), str(title), bool(latch)

    def __repr__(self):
        return f'Task({self.name!r}, latch={self.latch})'


BNC = Task('bnc_assembly', 'BNC assembly')


ENGAGE_TRACE_COLS =['t_s', 'stage', 'ref_x_mm', 'cmd_x_mm', 'meas_x_mm',
                     'delta_x_mm', 'delta_y_mm', 'delta_z_mm', 'delta_rot_deg',
                     'flange_x_mm', 'flange_y_mm', 'flange_z_mm', 'fx_n', 'fy_n', 'fz_n']


def _save_engage_trace(path, rows):
    """Per-cycle reference / delta / commanded / measured, so a rebound can be attributed."""
    import csv as _csv2
    with open(path, 'w', newline='') as fh:
        w = _csv2.writer(fh)
        w.writerow(ENGAGE_TRACE_COLS)
        w.writerows(rows)


def _report_engage_trace(trace):
    """Say WHAT MOVED, per stage, in one table.

    `T_cmd = ref @ Delta`, so an axial retreat is exactly one of three faults and they need
    different fixes. Printing the decomposition means a run answers that on the console
    instead of needing the CSV opened and cross-plotted:

      ref_x falls              -> the TRAJECTORY asked for it (waypoints, or a wiggle
                                  rotation levered into axial travel by a bad connector
                                  belief -- watch for it OSCILLATING, not just drifting)
      delta_x falls            -> the CONTROLLER yielded (compliance unloading by w/S)
      ref and cmd flat but
      meas/flange fall         -> the ARM did not track the command
      meas falls, flange does not -> the connector BELIEF moved, not the connector
    """
    if not trace:
        log.info('  (no engage trace collected)')
        return
    by_stage = {}
    for r in trace:
        by_stage.setdefault(r[1], []).append(r)
    log.info('  ENGAGE TRACE -- what moved, per stage (mm along the insertion axis):')
    log.info('    %-8s %7s  %8s %8s %8s %8s %9s', 'stage', 'cycles',
             'd_ref', 'd_delta', 'd_cmd', 'd_meas', 'd_flange')
    for name in ('contact', 'engage', 'confirm', 'settle'):
        rs = by_stage.get(name)
        if not rs:
            continue
        a, b = rs[0], rs[-1]

        def d(i, a=a, b=b):
            try:
                return float(b[i]) - float(a[i])
            except (TypeError, ValueError):
                return float('nan')
        flange = float(np.linalg.norm([d(9), d(10), d(11)]))
        log.info('    %-8s %7d  %+8.2f %+8.2f %+8.2f %+8.2f %9.2f',
                 name, len(rs), d(2), d(5), d(3), d(4), flange)
    xs = [float(r[4]) for r in trace if r[4] == r[4]]
    if xs:
        peak = max(xs)
        log.info('    deepest measured %.2f mm; ended %.2f mm -- BACKED OUT %.2f mm.',
                 peak, xs[-1], peak - xs[-1])


COMPLIANCE_KEYS = ('stiffness', 'mass', 'damping_ratio')


def _compliance_override(shared, block):
    """The shared `compliance:` with a maneuver's own physics laid over it.

    A maneuver sets `stiffness` (and optionally `mass`/`damping_ratio`) in its OWN block to
    run at a different compliance from everything else -- STATIC for that whole behaviour.
    `null`, or the key omitted, inherits; it is not the same as writing zeros.

    Returns a NEW dict: the shared section is read by every other maneuver, so overriding one
    must not quietly retune the rest."""
    comp = dict(shared or {})
    for k in COMPLIANCE_KEYS:
        if (block or {}).get(k) is not None:
            comp[k] = [float(v) for v in block[k]]
    # WHICH AXES YIELD, and the deadbands, may be a maneuver's own too (2026-11-06: connector clocking's roll
    # about the screw axis had a stiffness but was never compliant -- the shared selected_axes is translations only)
    if (block or {}).get('selected_axes') is not None:
        comp['selected_axes'] = [int(bool(v)) for v in block['selected_axes']]
    for k in ('force_deadband_n', 'torque_deadband_nm'):
        if (block or {}).get(k) is not None:
            comp[k] = float(block[k])
    return comp


def _retreat_state(state, expected):
    """The state BELOW `expected`, asserting that is where we actually are -- the mirror of
    _advance_state, so a disassembly step cannot claim a rung it never undid."""
    if state != expected:
        raise AssertionError(
            f'disassembly step expected the connector to be {expected!r}, but it is {state!r}')
    return UNCLOCK_STATES[UNCLOCK_STATES.index(expected) + 1]

# The six pose axes, in the order every 6-vector in this app uses (mm, mm, mm, deg, deg, deg).
DIM_KEYS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')


@contextlib.contextmanager
def _quiet_ik():
    """The arm's "No IK solution" ERROR silenced while a search probes candidates it expects to fail
    (the collar's roll search printed one per roll -- 54 lines in 17 s, 2026-11-05 23:47)."""
    lg = logging.getLogger('arm')
    old = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        lg.setLevel(old)


def _num(block, key, default):
    """A number from a config block, treating an EXPLICIT null the same as absent -- null
    inherits, by this config's convention."""
    v = (block or {}).get(key)
    return float(default) if v is None else float(v)


def _path_time(lin_mm, ang_deg, v_mm_s, w_deg_s, min_s):
    """Seconds for a segment of `lin_mm` translation and `ang_deg` rotation under both caps.

    THE CAPS MUST ALREADY BE RESOLVED. A None here means a caller skipped its own defaulting,
    which once raised a TypeError partway into a clocking stroke with the gripper closed on the
    part. Module level so the null path is testable without a robot."""
    if v_mm_s is None or w_deg_s is None:
        raise ValueError(
            f'_path_time needs resolved speed caps, got v={v_mm_s!r} w={w_deg_s!r}. A null '
            f'override in the config must be defaulted by the caller before it gets here.')
    return max((lin_mm / v_mm_s) if v_mm_s > 0 else 0.0,
               (ang_deg / w_deg_s) if w_deg_s > 0 else 0.0, min_s)


def _wrap_near(angle, centre):
    """`angle` (rad) shifted by whole turns so it lands within pi of `centre`.

    Every achieved clock angle here is recovered from a measured pose -- an Euler roll or
    `dot(rotvec, axis)` -- and both return a value in (-pi, pi]. Near a half turn that sign is a
    coin flip, and the sign IS the direction the unwind orbits. The achieved angle lies inside the
    swept band, so wrapping near the band centre picks the right branch."""
    return float(angle) + 2.0 * np.pi * np.round((float(centre) - float(angle)) / (2.0 * np.pi))


def _clocking_plan(connector, collar):
    """Which post-mate maneuvers to run, from their `enabled` flags in configs/bnc_assembly.yaml.

    Both are optional and independently switchable; an absent block or key means OFF. With
    connector clocking off a run ends ENGAGED, with collar clocking off it ends SEATED.

    ONE DEPENDENCY: collar clocking requires connector clocking, and the combination is REJECTED
    rather than silently reinterpreted -- without the belief reset it has no connector pose to
    place the collar against, and locking a connector still proud of the socket is worse than not
    locking it.

    Returns (connector_on, collar_on); raises ValueError carrying the operator-facing reason.
    Pure, so the rule is testable without a robot."""
    connector_on = bool((connector or {}).get('enabled', False))
    collar_on = bool((collar or {}).get('enabled', False))
    if collar_on and not connector_on:
        raise ValueError(
            'assembly.collar_clocking.enabled is true but assembly.connector_clocking.enabled is '
            'false. Collar clocking only runs on a SEATED connector and takes the connector pose '
            'it places the collar against from connector clocking. Enable connector clocking, or disable '
            'collar clocking.')
    return connector_on, collar_on


def _advance_state(state, expected):
    """The next state after `expected`, asserting that is where we actually are -- so an edit
    that advances twice, or skips a step, fails here instead of logging 'locked' for a connector
    that was never seated."""
    if state != expected:
        raise AssertionError(f'cannot advance from {state!r}: expected {expected!r}')
    return CLOCK_STATES[CLOCK_STATES.index(state) + 1]


def _fit_turn(theta, lo, hi):
    """`theta` shifted by whole turns into [lo, hi]. Returns (angle, clamped).

    A whole-turn shift is FREE: the collar is a body of revolution, so theta and theta +/- 360
    grip the same ring and differ only in which wrist_3 branch the turn then runs on. Clamping
    is the fallback and DOES move the grasp attitude, so it is reported separately.

    Pure, so the window arithmetic is testable without a robot."""
    theta = float(theta) + 2.0 * np.pi * np.round((0.5 * (lo + hi) - float(theta))
                                                  / (2.0 * np.pi))
    if lo <= theta <= hi:
        return float(theta), False
    # Clamped HALF A DEGREE INSIDE, not onto the edge: the post-advance check measures the same window, and a turn
    # planned to end exactly on it was refused by rounding (2026-11-06: -169.9 against a -169.9 limit, the unlock).
    eps = min(np.radians(0.5), 0.5 * max(hi - lo, 0.0))
    return float(np.clip(theta, lo + eps, hi - eps)), True


class _ScrewAdvance:
    """Progress detector for a clocking screw, shaped like ForceGuard so that
    AdmittanceController.ramp can terminate on it (ramp returns 'seated' the cycle check() first
    returns True).

    That early termination is the point: once the cams have pulled the connector far enough there
    is nothing to gain by finishing the rotation.

    Advance is measured from the MEASURED arm pose, never the commanded reference. Under
    admittance the two differ by the compliant deflection, and here that deflection IS the signal
    -- the reference is deliberately a virtual target the connector cannot reach."""

    def __init__(self, robot, T_tool0_conn, T_base_conn_engaged, threshold_m):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self._inv_engaged = inverse(np.asarray(T_base_conn_engaged, dtype=float))
        self.threshold_m = float(threshold_m)
        self.peak_m = 0.0                  # best advance seen across ALL tries (never reset)
        self.tripped_by = None

    def rebase(self, T_tool0_conn):
        """Adopt a new connector-in-gripper relationship, keeping the engaged reference frame.

        Needed after any REGRASP: the connector stays put in the socket while the gripper
        travels, so the two afterwards differ by exactly the progress made. Rebasing keeps advance
        measured from the ORIGINAL engaged pose instead of re-zeroing.

        NO CALLER IN THE CURRENT SWEEP -- the oscillating screw never lets go. Kept as the
        detector's contract for anything that does."""
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)

    def advance_m(self):
        """Connector translation along the ENGAGED connector frame's +X, in metres."""
        rel = self._inv_engaged @ (self.robot.tool0() @ self.T_tool0_conn)
        return float(rel[0, 3])

    def check(self):
        d = self.advance_m()
        self.peak_m = max(self.peak_m, d)
        if self.threshold_m > 0.0 and d >= self.threshold_m:
            self.tripped_by = (f'advance {d * 1000.0:.2f} mm >= '
                               f'{self.threshold_m * 1000.0:.2f} mm')
            return True
        return False

    def reset(self):
        self.tripped_by = None


class _AxialForce:
    """Trips on the contact force ALONG THE CONNECTOR'S OWN +X -- the insertion reaction.

    ForceGuard watches |f|, which a lateral graze raises without opposing the push at all -- so a
    magnitude limit tight enough to catch real resistance also stops on every glancing touch.
    Projecting onto the insertion axis separates the two, which lets the ENGAGE limit be set low
    without becoming a hair-trigger.

    Shaped like ForceGuard (check/reset/tripped_by) so _AnyGuard can OR it with the others.
    Persistence means the limit must hold CONTINUOUSLY for that long."""

    def __init__(self, robot, T_tool0_conn, max_force_n, persistence_s=0.0):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self.max_force_n = float(max_force_n)
        self.persistence_s = float(persistence_s)
        self.peak_n = 0.0                  # best axial force seen; never reset, for the log
        self.tripped_by = None
        self._over_since = None

    def axial_n(self):
        """|force along the connector +X|, in newtons. Magnitude, not signed -- an inverted sign
        convention would turn this guard off rather than make it noisy."""
        T_base_tool0 = self.robot.tool0()
        T_base_conn = T_base_tool0 @ self.T_tool0_conn
        w = self.robot.arm.wrench_in(T_base_conn, T_base_tool0)
        return abs(float(w[0]))

    def check(self):
        f = self.axial_n()
        self.peak_n = max(self.peak_n, f)
        if self.max_force_n <= 0.0 or f < self.max_force_n:
            self._over_since = None
            return False
        now = self.robot.arm.clock()
        if self.persistence_s > 0.0:
            if self._over_since is None:
                self._over_since = now
                return False
            if now - self._over_since < self.persistence_s:
                return False
        self.tripped_by = (f'axial force {f:.1f} N >= {self.max_force_n:.1f} N'
                           + (f' for {now - self._over_since:.2f} s'
                              if self.persistence_s > 0.0 and self._over_since else ''))
        return True

    def reset(self):
        self.tripped_by = None
        self._over_since = None


class _TravelReached:
    """Satisfied once the connector has ADVANCED far enough along its own +X since contact.

    THE POSITIVE SUCCESS SIGNAL. A force limit says something is resisting, which is also what a
    connector jammed on a rim does. Travel says the part actually WENT somewhere -- it is the one
    condition a jam cannot fake, so a mate that slides home under low force is recognised as the
    success it is instead of running until the force criterion eventually trips.

    Measured from the MEASURED arm pose, never the commanded reference: under admittance the two
    differ by the compliant deflection, and here it is the real motion that matters."""

    def __init__(self, robot, T_tool0_conn, T_base_conn_contact, threshold_mm):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self._inv_contact = inverse(np.asarray(T_base_conn_contact, dtype=float))
        self.threshold_mm = float(threshold_mm)
        self.peak_mm = 0.0
        self.tripped_by = None

    def travel_mm(self):
        """Advance along the connector's own +X since contact, in mm. Signed, then clipped at 0 --
        backing OFF is not progress and must never be counted as any."""
        T_now = self.robot.tool0() @ self.T_tool0_conn
        return max(0.0, float((self._inv_contact @ T_now)[0, 3]) * 1000.0)

    def check(self):
        d = self.travel_mm()
        self.peak_mm = max(self.peak_mm, d)
        if self.threshold_mm <= 0.0 or d < self.threshold_mm:
            return False
        self.tripped_by = f'travel {d:.2f} mm >= {self.threshold_mm:.2f} mm since contact'
        return True

    def reset(self):
        self.tripped_by = None


class _RadialConfirm:
    """Was there sustained force ACROSS the connector axis -- the signature of being IN a socket?

    THE AXIAL LIMIT ALONE CANNOT TELL A SEAT FROM A FACE. Pushing a connector flat against any
    surface -- the fixture body, the bench, the rim of the wrong hole -- develops axial reaction
    exactly like a real insertion does, and stops the engage in the same way. What a SOCKET adds is
    LATERAL constraint: once the barrel is inside, the walls resist sideways motion, so a connector
    being wiggled inside a socket pushes back across its own axis. A connector resting on a flat
    face does not; it just slides.

    So this measures |force| in the connector's OWN Y-Z plane -- the two axes perpendicular to the
    insertion direction -- and asks that it hold above a threshold CONTINUOUSLY. It is a
    satisfaction detector, not a canceller: nothing is stopped when it trips, it simply records
    that the condition was met."""

    def __init__(self, robot, T_tool0_conn, min_force_n, persistence_s=0.0):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self.min_force_n = float(min_force_n)
        self.persistence_s = float(persistence_s)
        self.peak_n = 0.0
        self.held_s = 0.0                  # longest continuous stretch above the threshold
        self.satisfied = False
        self._over_since = None

    def radial_n(self):
        """|force across the connector +X|, in newtons -- the Y-Z magnitude in the CONNECTOR
        frame, so it means the same thing whatever attitude the part is held at."""
        T_base_tool0 = self.robot.tool0()
        T_base_conn = T_base_tool0 @ self.T_tool0_conn
        w = self.robot.arm.wrench_in(T_base_conn, T_base_tool0)
        return float(np.linalg.norm(w[1:3]))

    def check(self):
        f = self.radial_n()
        self.peak_n = max(self.peak_n, f)
        if self.min_force_n <= 0.0:
            self.satisfied = True                      # no bar set -> nothing to fail
            return True
        now = self.robot.arm.clock()
        if f < self.min_force_n:
            self._over_since = None
            return False
        if self._over_since is None:
            self._over_since = now
        self.held_s = max(self.held_s, now - self._over_since)
        if self.held_s >= self.persistence_s:
            self.satisfied = True
        return self.satisfied

    def reset(self):
        self.satisfied = False
        self._over_since = None
        self.held_s = 0.0


class _AnyGuard:
    """ORs several ForceGuard-shaped watchdogs onto one ramp, remembering WHICH one tripped.

    The screw needs a success detector and a force limit watching the same motion, and the two
    mean opposite things -- satisfied vs jammed -- so which fired has to survive the call.
    `ramp` only reports 'seated'."""

    def __init__(self, *guards):
        self.guards = [g for g in guards if g is not None]
        self.tripped = None
        self.tripped_by = None

    def check(self):
        for g in self.guards:
            if g.check():
                self.tripped = g
                self.tripped_by = getattr(g, 'tripped_by', None)
                return True
        return False

    def reset(self):
        self.tripped = None
        self.tripped_by = None
        for g in self.guards:
            g.reset()



def level_axis(T, up=(0.0, 0.0, 1.0), max_deg=15.0):
    """THE CLOCKING AXIS PARALLEL TO THE BASE PLANE (the user, 2026-11-06; connector_clocking.level_axis): T with its
    +X turned onto the plane normal to `up` by the smallest rotation -- about the axis square to both, through T's
    own origin, so its position and its heading in that plane are kept -> (T_levelled, tilt_deg). Returned as given
    when the tilt is over `max_deg` (an axis meant to be off the plane -- a socket facing up -- is not levelled)."""
    T = np.asarray(T, dtype=float)
    n = np.asarray(up, dtype=float)
    n = n / np.linalg.norm(n)
    x = T[:3, 0] / np.linalg.norm(T[:3, 0])
    s = float(np.clip(x @ n, -1.0, 1.0))
    tilt = float(np.degrees(np.arcsin(abs(s))))
    if tilt > max_deg or tilt < 1e-9:
        return T, tilt
    xh = x - s * n
    xh /= np.linalg.norm(xh)
    k = np.cross(x, xh)
    k /= np.linalg.norm(k)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(k * np.arccos(float(np.clip(x @ xh, -1.0, 1.0)))).as_matrix()
    out = T.copy()
    out[:3, :3] = R @ T[:3, :3]
    return out, tilt


def fit_axis_offset(theta, y, z):
    """THE SOCKET'S TRUE AXIS FROM A TURN (connector clocking's axis probe; the user, 2026-11-06). With the side
    load nulled, the plug turns on the socket's REAL axis while the wrist turns about the one the estimate gave;
    the connector where the in-hand belief puts it (rigid in the grip) then traces

        p(theta) = B - R(theta) d          (the screw axis's y, z; theta the roll since the probe began)

    B the real axis's offset from the screw axis, d the in-hand belief's lateral error (in the connector's frame,
    which turns with theta). Linear in (B, d) once theta is measured -- better posed than fitting a circle to a
    short arc. 02:33 run: B (+0.6, -3.5) mm, |d| 2.4 mm, 0.29 mm residual. -> (B, d, residuals, condition) in the
    units given."""
    th = np.asarray(theta, dtype=float)
    c, s = np.cos(th), np.sin(th)
    n = len(th)
    A = np.zeros((2 * n, 4))
    b = np.empty(2 * n)
    A[0::2, 0], A[0::2, 2], A[0::2, 3], b[0::2] = 1.0, -c, s, np.asarray(y, dtype=float)
    A[1::2, 1], A[1::2, 2], A[1::2, 3], b[1::2] = 1.0, -s, -c, np.asarray(z, dtype=float)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = A @ sol
    res = np.hypot(b[0::2] - pred[0::2], b[1::2] - pred[1::2])
    return sol[:2], sol[2:], res, float(np.linalg.cond(A))


class _ClockTrace:
    """CONNECTOR CLOCKING, SAMPLED AND DRAWN (the user, 2026-11-06: "record data for post-analysis"; "add a live
    visualization of the axis in sim"). step() is screw_ramp's on_step: every `every`-th servo cycle the
    connector where the belief puts it, in the frame of the screw axis as the probe began (theta its roll,
    x the advance, y/z off the axis), the controller's give, the wrench and the flange -> <trial>/
    clocking_trace.csv; the probe's (theta, y, z) kept for fit_axis_offset. Drawn live in the viewer (overlay
    "clocking_axis"): the screw axis -- and, once probed, the socket's -- and the connector's trace."""

    COLS = ['t_s', 'wall', 'phase', 'theta_deg', 'x_mm', 'y_mm', 'z_mm', 'null_x_mm', 'null_y_mm', 'null_z_mm',
            'null_rx_deg', 'null_ry_deg', 'null_rz_deg', 'fx_n', 'fy_n', 'fz_n', 'tx_nm', 'ty_nm', 'tz_nm',
            'flange_x_mm', 'flange_y_mm', 'flange_z_mm', 'flange_roll_deg', 'flange_pitch_deg', 'flange_yaw_deg']

    def __init__(self, robot, adm, path, T_axis, T_tool0_conn, every=3, draw=None):
        self.robot, self.adm = robot, adm
        self.T_axis0, self.T_tc0 = np.array(T_axis, dtype=float), np.array(T_tool0_conn, dtype=float)
        self._inv0 = inverse(self.T_axis0)
        self.axes = [(self.T_axis0, [1.0, 0.55, 0.1])]        # (frame, colour): the screw axis
        self.every, self.k, self.phase, self.n = max(1, int(every)), 0, '', 0
        self.probe, self.pts, self.text = [], [], []
        self.check, self.off = None, []        # (inv(axis), tool0 -> connector): the corrected pair, checked live
        # THE DROP-IN (slot_roll): the first time the connector gains drop_mm on the commanded push within
        # drop_window_deg of turn -- the pins falling into the slot entries: held back by the socket face it lags the
        # push, dropped in it catches up (an arm that just follows the push, as in a dry run, never does). Kept: the
        # roll at the FOOT of that rise (where they met). Catches a drop once the push is over drop_mm.
        self.drop_mm, self.drop_window_deg, self.drop = None, 30.0, None
        self.push_mm = 0.0                     # the push the reference commands now (connector_clocking's ref_at)
        self._hist, self._turn, self._th_last = [], 0.0, None
        self.t0, self._drawn, self.draw_fn = time.monotonic(), 0.0, draw
        self.fh = open(path, 'w', newline='')
        self.w = _csv.writer(self.fh)
        self.w.writerow(self.COLS)

    def step(self):
        self.k += 1
        if self.k % self.every:
            return
        try:
            T = self.robot.tool0()
            conn = T @ self.T_tc0
            rel = self._inv0 @ conn
            th = float(np.arctan2(rel[2, 1], rel[1, 1]))
            x, y, z = (rel[:3, 3] * 1000.0).tolist()
            if self.phase.startswith('probe'):
                self.probe.append((th, y, z))
            elif self.check is not None:           # after the probe: how far off the CORRECTED axis it runs
                r2 = self.check[0] @ (T @ self.check[1])
                self.off.append(float(np.hypot(r2[1, 3], r2[2, 3])) * 1000.0)
            if self.drop_mm and self.drop is None and self.phase != 'settle':
                self._drop_step(th, x - self.push_mm)
            self.pts.append(conn[:3, 3].copy())
            d = self.adm.delta
            try:
                w = np.asarray(self.robot.arm.wrench(), dtype=float)
            except Exception:                          # noqa: BLE001
                w = np.full(6, np.nan)
            fx, fr = matrix_to_xyzrpy(T)
            self.w.writerow([f'{time.monotonic() - self.t0:.3f}', datetime.now().strftime('%H:%M:%S.%f')[:-3],
                             self.phase, f'{np.degrees(th):.3f}', f'{x:.3f}', f'{y:.3f}', f'{z:.3f}']
                            + [f'{1000 * v:.3f}' for v in d[:3]] + [f'{np.degrees(v):.3f}' for v in d[3:]]
                            + [f'{v:.3f}' for v in w] + [f'{1000 * v:.2f}' for v in fx]
                            + [f'{np.degrees(v):.3f}' for v in fr])
            self.n += 1
            if self.n % 100 == 0:
                self.fh.flush()
            if self.draw_fn is not None and time.monotonic() - self._drawn > 0.3:
                self.draw()
        except Exception as exc:                       # noqa: BLE001 -- a record, never a reason to stop
            log.debug('clocking trace: sample skipped (%s)', exc)

    def _drop_step(self, th, x_mm):
        if self._th_last is not None:
            self._turn += abs(float(np.arctan2(np.sin(th - self._th_last), np.cos(th - self._th_last))))
        self._th_last = th
        self._hist.append((self._turn, th, x_mm, self.phase))
        win = np.radians(self.drop_window_deg)
        while self._hist and self._turn - self._hist[0][0] > win:
            self._hist.pop(0)
        xs = np.array([h[2] for h in self._hist])
        if x_mm - xs.min() >= self.drop_mm:
            # the foot: the LAST sample still near the low -- where the rise began (a slow creep before it would
            # put the minimum itself at the window's start)
            i = int(np.nonzero(xs <= xs.min() + 0.25 * self.drop_mm)[0].max())
            self.drop = {'theta_deg': float(np.degrees(self._hist[i][1])), 'theta_at_trip_deg': float(np.degrees(th)),
                         'lag_foot_mm': float(self._hist[i][2]), 'lag_mm': float(x_mm), 'phase': self._hist[i][3],
                         't_s': round(time.monotonic() - self.t0, 3)}

    def draw(self):
        self._drawn = time.monotonic()
        items = []
        for Tx, col in self.axes:
            o, ax = Tx[:3, 3], Tx[:3, 0]
            items.append({'type': 'line', 'points': [(o - 0.06 * ax).tolist(), (o + 0.03 * ax).tolist()],
                          'color': col})
        if self.pts:
            items.append({'type': 'points', 'points': np.round(np.array(self.pts[-800:]), 5).tolist(),
                          'color': [0.2, 0.85, 1.0], 'radius': 0.0006})
        try:
            self.draw_fn(items, self.text)
        except Exception as exc:                       # noqa: BLE001 -- drawing only
            log.debug('clocking trace: not drawn (%s)', exc)

    def close(self):
        if self.draw_fn is not None:
            self.draw()
        try:
            self.fh.close()
        except OSError:
            pass


def _passed(phase, condition, nxt=None):
    """ONE line, the same shape everywhere, when a phase's success condition is met: WHICH
    condition, measured against WHICH limit, and where the run goes next -- so a log read after
    the fact says why each transition happened, not only that it did."""
    log.info('PASSED %s: %s%s', phase, condition, (' -> %s' % nxt) if nxt else '')


def _engage_report(status, s, depth_mm, det, guard, combo, confirm=None):
    """Print WHICH condition ended the engage, and where every OTHER one stood when it did.

    THREE THINGS CAN END THIS MOTION and they mean completely different things:

        PATH COMPLETE   the whole trajectory ran. Nothing resisted enough to stop it.
        AXIAL FORCE     the connector met the socket hard enough, along its own +X, for long
                        enough. A NORMAL end -- the bayonet screw drives the rest.
        FORCE GUARD     the general wrench limit, in any direction. A JAM.

    Reporting only the winner has cost real bench time: 'stopped on force' reads the same
    whether the axial limit was met at 2 mm or at 19.8 mm of a 20 mm path, and whether the
    general guard was idle or a hair under its own limit. So all three are printed every time,
    with the one that fired marked, and each shown against ITS OWN limit -- a bare number
    cannot be judged without the threshold it was tested against."""
    fired = {'complete': 'PATH COMPLETE', 'force': 'AXIAL FORCE LIMIT',
             'travel': 'TRAVEL REACHED', 'timeout': 'TIMED OUT',
             'guard': 'GENERAL FORCE GUARD',
             'unconfirmed': 'SUCCEEDED BUT SEAT NOT CONFIRMED'}.get(status, status.upper())
    meaning = {
        'complete': 'the full path ran without meeting the axial limit',
        'force': 'a NORMAL end -- the clocking screw drives the rest',
        'guard': 'a JAM: the general wrench limit, not the axial one',
        'travel': 'the part actually WENT somewhere -- the one thing a jam cannot fake',
        'timeout': 'neither travel nor force arrived in the time allowed',
        'unconfirmed': 'something resisted, but it does not behave like a SOCKET',
    }.get(status, '')
    say = log.warning if status in ('guard', 'unconfirmed', 'timeout') else log.info

    def mark(*names):
        """'>>' against the condition that ended it. `elapsed` covers TWO outcomes -- the path
        running out and the timeout -- because both mean the budget was spent without either
        success condition arriving, and leaving neither marked would show a report with no
        winner at all."""
        return '>>' if status in names else '  '

    pct = 100.0 * s['elapsed_s'] / max(s['duration_s'], 1e-9)
    say('  --- ENGAGE ENDED: %s --- %s', fired, meaning)
    say('   %s elapsed         %.2f of %.2f s (%.0f%%) -- drove %.1f of %.1f mm total',
        mark('timeout', 'complete'), s['elapsed_s'], s['duration_s'], pct, s['driven_mm'],
        s['total_mm'])
    if 'travel_mm' in s:
        say('   %s travel          %.2f mm of %.2f mm since contact (contact at %.2f mm)',
            mark('travel'), s['travel_mm'], s['travel_limit_mm'], s.get('contact_mm', 0.0))
    if s['axial_limit_n'] > 0:
        say('   %s axial force     %.1f N of %.1f N limit (peak %.1f N%s)',
            mark('force'), s['axial_n'], s['axial_limit_n'], s['axial_peak_n'],
            ', persistence %.2f s' % s['axial_persist_s'] if s['axial_persist_s'] > 0 else '')
    else:
        say('   %s axial force     NO LIMIT SET (peak %.1f N seen) -- this condition can never '
            'end the engage', mark('force'), s['axial_peak_n'])
    if guard is not None and getattr(guard, 'enabled', False):
        say('   %s force guard     |F| %.1f N of %.1f N (peak %.1f) | tau %.2f Nm of %.2f Nm '
            '(peak %.2f)', mark('guard'), s['force_n'], guard.max_force, guard.peak_force,
            s['torque_nm'], guard.max_torque, guard.peak_torque)
    else:
        say('   %s force guard     DISABLED -- nothing was watching for a jam', mark('guard'))
    say('      depth           %+.2f mm past the recorded mate', depth_mm)
    # THE CONFIRMATION, when it ran. Both numbers against their own limits, like everything else
    # here -- 'not confirmed' is unreadable without knowing WHICH of the two failed and by how far.
    if confirm:
        say('   %s radial force    %.1f N peak, held %.2f s (want >= %.1f N for %.2f s) -- %s',
            '>>' if not confirm['radial_ok'] else '  ', confirm['radial_peak_n'],
            confirm['radial_held_s'], confirm['radial_limit_n'],
            confirm['radial_persist_s'],
            'OK' if confirm['radial_ok'] else 'FAILED')
        say('      wiggle travel   %.2f mm over %.2f s -- REPORTED ONLY, not a criterion',
            confirm['travel_mm'], confirm['cycles_s'])
    if status != 'complete' and combo is not None and combo.tripped_by:
        say('      tripped by      %s', combo.tripped_by)


def engage_wiggle(cfg):
    """(the engage's Wiggle, where its numbers came from): configs/wiggle_sampling.yaml under any
    assembly.engage.wiggle override, validated against its caps. The Wiggle is None when every
    amplitude is 0. Raises wiggle.WiggleError."""
    en = cfg.section('assembly').get('engage', {}) or {}
    blk, src = wigmod.from_shared(cfg, en.get('wiggle'), en.get('wiggle_from', 'wiggle_sampling.yaml'), 'engage')
    w = wigmod.Wiggle.from_cfg(blk, 'engage')
    if w is not None:
        w.validate(rate_hz=float(en.get('sample_rate_hz', 25.0)),
                   cap_v=blk.get('max_speed_mm_s') or en.get('max_oscillation_speed_mm_s'),
                   cap_w=blk.get('max_rotation_deg_s') or en.get('max_oscillation_rotation_deg_s'))
    return w, src


def precheck(cfg):
    """The config checks that need no hardware, run before the camera opens and the arm connects:
    a refusal here ends the run without an FRI session (2026-11-05: each wiggle refusal, raised
    after the connect, faulted the pendant's app)."""
    try:
        engage_wiggle(cfg)
    except wigmod.WiggleError as exc:
        log.error('%s', exc)
        return False
    if not clocking_precheck(cfg):
        return False
    return campaign_precheck(cfg)


def precheck_and_preload(cfg):
    """precheck -- then, passed, the 6D estimator's models loading in the background BEFORE the arm connects, while
    the operator starts the FRI app on the pendant (2026-11-06 11:22: they loaded during the first move and the FRI
    link dropped, the PC stalled ~0.4 s; step_start also waits for them before the arm moves)."""
    if not precheck(cfg):
        return False
    if cfg.get_path('robot.dry_run', False):
        return True
    if str(cfg.get_path('robot.backend') or '') == 'iiwa_fri':
        from ..robot.backends.iiwa_fri import reserve_fri_cpu
        reserve_fri_cpu(cfg)                    # before the 6D server starts, so it runs off the driver's core
    if str(cfg.get('pose_method') or 'sam3').strip().lower() == 'pose6d':
        # TO THE END, BEFORE THE ARM CONNECTS (2026-11-06 11:30: started early, it was still loading when the
        # session began -- and the link dropped 0.3 s after the weights went to the GPU). The FRI app on the pendant
        # waits for this PC (LBRServer: 100 s), so it may be started before or after.
        t0 = time.monotonic()
        P6.preload(cfg)
        log.info('Loading the 6D pose models BEFORE connecting to the arm (~15 s) -- they stalled the FRI link when '
                 'loaded while it was commanding. Start the FRI app on the pendant now if it is not running.')
        if P6.wait_ready(120.0):
            log.info('6D pose models loaded in %.0f s -- connecting to the arm.', time.monotonic() - t0)
        else:
            log.warning('The 6D pose models are still loading after 120 s -- connecting to the arm anyway.')
    return True


def clocking_precheck(cfg):
    """Connector clocking's push must stay under its early-out: the early-out reads ADVANCE, and a plug nothing
    holds back (an engage that ended short) follows the push -- a push at or past it would be called seated with no
    cam ever turning it in (a dry run did, 2026-11-06, at push 9 mm)."""
    cc = cfg.get_path('assembly.connector_clocking') or {}
    if not cc.get('enabled', True):
        return True
    push, need = _num(cc, 'push_mm', 5.0), _num(cc, 'success_advance_mm', 5.0)
    if need > 0 and push >= need - 0.25:
        log.error('assembly.connector_clocking.push_mm %.1f must stay under success_advance_mm %.1f (by 0.25 mm): a '
                  'plug that is not held back would follow the push and be called SEATED. Lower the push (stiffen '
                  'the spring for more press).', push, need)
        return False
    return True


def campaign_precheck(cfg):
    """More than one trial needs something to put the cell back between them: disassembly (with its place,
    after connector clocking or the RL policy) -- or campaign.reset: manual, the operator. Refused here,
    before anything connects."""
    cp = dict(cfg.get('campaign') or {})
    a = cfg.section('assembly')
    dis = a.get('disassembly', {}) or {}
    n = int(cp['trials']) if cp.get('trials') else max(1, int(_num(dis, 'cycles', 1)))
    mode = str(cp.get('reset') or 'auto').strip().lower()
    if mode not in ('auto', 'manual'):
        log.error("campaign.reset %r must be 'auto' or 'manual'.", mode)
        return False
    if n <= 1 or mode == 'manual':
        return True
    cc = bool((a.get('connector_clocking') or {}).get('enabled', False))
    rl = str(a.get('insertion_mode') or 'estimate').strip().lower() == 'rl'
    missing = ('assembly.disassembly.enabled' if not dis.get('enabled', False) else
               'assembly.disassembly.place.enabled' if not (dis.get('place') or {}).get('enabled', True) else
               'assembly.connector_clocking.enabled (disassembly undoes it)' if not (cc or rl) else None)
    if missing:
        log.error('%d TRIALS asked for, but nothing puts the cell back between them: %s is off. Turn it on, or set '
                  'campaign.reset: manual to reset the cell by hand after each trial.', n, missing)
        return False
    return True


def build_and_run(cfg, robot, camera, args, task=None):
    """The run. `task` (Task; default BNC) is what is assembled -- see Task."""
    task = BNC if task is None else task
    a = cfg.section('assembly')
    camp = None                  # the run as a behavior tree (behaviors/campaign.py): made once its folder is

    def _guarded(robot_, guard, move_fn, label=None):
        """A free-space move with the force guard armed (apps/_common.guarded) -- and a TRIP TOLD TO THE
        CAMPAIGN: an unexpected contact makes the step's failure one for a person, not for a recovery."""
        ok = _guarded_move(robot_, guard, move_fn, label)
        if not ok and getattr(guard, 'tripped_by', None) and camp is not None:
            camp.collision(guard.tripped_by)
        return ok

    def ask(prompt):
        """input(), shown on the board as WAITING FOR OPERATOR. None on EOF: each caller keeps its own rule."""
        if camp is not None:
            return camp.ask(prompt)
        try:
            return input(prompt)
        except EOFError:
            return None

    # The ESTIMATOR first: a missing/stale manifold CSV or a bad metric key must fail before
    # the robot moves, not after a part is in the fingers. But only when this run will USE it:
    # insertion_mode engage reads no map, and requiring one anyway stopped the default mode from
    # starting wherever the (never-committed) manifold CSV was absent.
    estimator = None
    if str(a.get('insertion_mode') or 'estimate').strip().lower() == 'estimate':
        estimator = CheckedManifoldEstimator(cfg.section('estimation'))
    commit = str(cfg.get_path('estimation.commit', 'aggregator')).strip().lower()
    if commit not in ('aggregator', 'argmin'):
        log.error("estimation.commit %r must be 'aggregator' or 'argmin'.", commit)
        return False
    if estimator is not None:
        log.info('Estimator: commit %s, dims %s, manifold %s.', commit, estimator.estimate_dims,
                 cfg.get_path('estimation.manifold_csv'))
    else:
        log.info('Estimator: not built -- insertion_mode engage reads no map.')

    # A DRY RUN WITH dry_cell.enabled (apps/_dry_cell.py): no camera to scan with -- the dry cell's stand-in scanner,
    # made below once the connector datum is known; the 6D estimator is never started.
    from . import _dry_cell
    if _dry_cell.enabled(cfg):
        scanner, _detector, _est = None, None, None
    else:
        scanner, _detector, _est = build_scanner(cfg, robot, camera)
    # pose_method pose6d: the connector kept estimated after the scan, through the pick and in the
    # hand (pose6d.in_hand), drawn in the live view -- and read once, before the stand-off.
    conn_tracker = in_hand_tracker(cfg, robot, camera, scanner)
    # pose6d.online: the socket, and the connector in the hand, kept estimated through every contact phase
    # (the user, 2026-11-11) -- step_insert's online_begin below.
    # Started only on a socket the 6D estimator measured (socket_obj).
    sock_tracker = socket_tracker(cfg, robot, camera)
    geom, check = GraspGeometry(cfg), GraspCheck(cfg)
    recovery, grasp = GraspRecovery(cfg), GraspController(cfg)
    recorder = GraspImageRecorder(cfg)
    confirm = make_confirm(cfg)

    # COMPLIANCE + GUARD from the top-level blocks -- same names estimator_eval reads, so a
    # tuned pair of blocks can be copied between the two configs without translation.
    adm = make_compliance(robot.arm, cfg.section('compliance'))
    guard_shared = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_shared = float(cfg.get_path('compliance.settle_s', 0.5))
    hold_shared = float(cfg.get_path('compliance.hold_after_insertion_s', 0.0))

    # ---- FINAL INSERTION: one grouped block, each key inheriting the shared value when null.
    fi = a.get('final_insertion', {}) or {}
    fi_on = bool(fi.get('enabled', True))
    fi_settle = None if fi.get('settle_s') is None else float(fi['settle_s'])
    fi_hold = None if fi.get('hold_after_insertion_s') is None \
        else float(fi['hold_after_insertion_s'])
    # Pacing, dwell and noise for the COMMIT. Zero-noise by default: the jitter exists to gather
    # varied contact while PROBING and has no place in the attempt meant to seat.
    fi_v = None if fi.get('speed_translation_mm_s') is None \
        else float(fi['speed_translation_mm_s'])
    fi_wr = None if fi.get('speed_rotation_deg_s') is None \
        else float(fi['speed_rotation_deg_s'])
    fi_pause = float(fi.get('pause_s', 0.0) or 0.0)
    # PRELOAD -- the press, on the COMMIT only. Drives the commit's reference this far PAST the
    # mate along the connector's +X; the part stops at the mate, the reference keeps going, and
    # the spring turns the leftover travel into contact force. 0 disables.
    #
    # It must NOT live as a trajectory row: uncertain_sampling anchors that path, so the contact
    # map was collected with the press removed while this app drove it -- every observation 10 mm
    # deeper than anything the map held. Named here it survives anchoring and lands only on the
    # insertion meant to SEAT.
    #
    # It buys the break-in SPIKE, not a sustained press: it decays as the spring yields over
    # D/S ~ 6-9 s, and what it can HOLD is only stiffness x preload.
    fi_preload_mm = float(fi.get('preload_mm', 10.0) or 0.0)
    fin = fi.get('trajectory_noise', {}) or {}
    fi_noise_on = bool(fin.get('enabled', False))
    fi_noise_std = [float(v) for v in (fin.get('std') or [0.0] * 6)]
    fi_noise_w = max(1, int(fin.get('smooth_window', 10)))
    if fi_noise_on and len(fi_noise_std) != 6:
        log.error('final_insertion.trajectory_noise.std must have 6 entries.')
        return False
    for nm, v in (('speed_translation_mm_s', fi_v), ('speed_rotation_deg_s', fi_wr),
                  ('pause_s', fi_pause), ('preload_mm', fi_preload_mm)):
        if v is not None and v < 0:
            log.error('assembly.final_insertion.%s must be >= 0 (got %.2f).', nm, v)
            return False
    adm_final, guard_final = adm, None
    if fi_on:
        comp_final = dict(cfg.section('compliance'))
        for k in ('stiffness', 'mass', 'damping_ratio'):
            if fi.get(k) is not None:
                comp_final[k] = [float(v) for v in fi[k]]
        adm_final = make_compliance(robot.arm, comp_final)
        gsec = dict(cfg.section('force_guard'))
        over_g = {k: fi[k] for k in ('max_force_n', 'max_torque_nm', 'persistence_s')
                  if fi.get(k) is not None}
        guard_final = ForceGuard(robot.arm, {**gsec, **over_g}) if over_g else None
        log.info('Final insertion ON: stiffness %s, guard %.0f N.', comp_final.get('stiffness'),
                 float(over_g.get('max_force_n', gsec.get('max_force_n', 0.0))))
        if fi_preload_mm > 0:
            s_ins = max(float(v) for v in (comp_final.get('stiffness') or [0.0])[:3])
            log.info('   PRELOAD %.1f mm past the mate on the commit (probing passes do NOT '
                     'press). The spring holds %.1f N of it at %.0f N/m; the contact peak is a '
                     'transient several times larger.',
                     fi_preload_mm, s_ins * fi_preload_mm / 1000.0, s_ins)

    # ---- INSERTION MODE ----------------------------------------------------------------------
    # ESTIMATE  insert, observe, fit the contact manifold, correct the in-hand belief, repeat.
    #           Needs a map covering the contact states production actually visits.
    # ENGAGE    no estimation: drive the trajectory home with an optional oscillation superimposed,
    #           letting compliance find the mate. Needs no map, so it is immune to every failure
    #           mode of the estimator.
    # RL        the trained policy (skills/rl_insertion.py, assembly.rl_insertion): from above the socket
    #           to the plug SEATED -- engage and connector clocking both -- robust to the in-hand and
    #           socket errors it was trained on. Collar clocking then runs as after connector clocking.
    ins_mode = str(a.get('insertion_mode') or 'estimate').strip().lower()
    if ins_mode not in ('estimate', 'engage', 'rl'):
        log.error("assembly.insertion_mode %r must be 'estimate', 'engage' or 'rl'. The old\n"
                  "standalone 'wiggle' is retired: engage does the same oscillation, but\n"
                  "superimposed on the trajectory rather than replacing it -- set\n"
                  "assembly.engage.amplitude instead.", ins_mode)
        return False
    # ---- ENGAGE: the trajectory-following insertion ------------------------------------------
    en = a.get('engage', {}) or {}
    en_amp = [float((en.get('amplitude') or {}).get(d, 0.0)) for d in DIM_KEYS]
    en_frq = [float((en.get('frequency_hz') or {}).get(d, 0.0)) for d in DIM_KEYS]
    en_rate = float(en.get('sample_rate_hz', 25.0))
    # How many MISSED engagements (full path run, axial limit never reached) the run will recover
    # from by putting the cable down and re-picking. A budget, not a loop: a cell that keeps
    # missing has a wrong target pose or a wrong in-hand belief, and retrying wears the part.
    max_engage_misses = max(0, int(en.get('max_misses', 2)))
    # ---- CONTACT: a separate, gentle phase that FINDS the socket before engaging into it ------
    # Driving the whole insertion in one go means the wiggle and the full approach speed are
    # already running when first contact happens, so the first touch is the least controlled
    # moment of the phase. Splitting it puts a slow, un-oscillated approach in front: creep along
    # the trajectory until a light force says the parts are touching, and let the engagement start
    # from a KNOWN contact pose rather than from a guess about where the socket is.
    _ct = en.get('contact', {}) or {}
    ct_on = bool(_ct.get('enabled', True))
    ct_force_n = _num(_ct, 'force_n', 1.0)
    ct_pers_s = _num(_ct, 'persistence_s', 0.10)
    # ENGAGE TERMINATION -- travel is the positive signal, force the resistive one, timeout the
    # backstop. See engage_insertion.
    en_travel_mm = _num(en, 'travel_mm', 5.0)
    en_timeout_s = _num(en, 'timeout_s', 15.0)
    en_align_s = _num(en, 'align_s', 2.0)
    # FORCE-LIMITED ADVANCE (the user, 2026-11-06: "the linear motion of the engage phase is too fast leading to the
    # preload increasing too high before the wiggle amplitude can align the connector, so the connector and socket
    # get stuck misaligned at face on face contact mode due to friction"). The reference used to advance at the
    # engage rate whatever the part did: at a face the spring wound up -- 01:13, 3.8 mm of it, 20-28 N -- and
    # friction (~mu x that) outgrew the most the wiggle can push sideways (stiffness x amplitude: 500 N/m x 8 mm
    # = 4 N). Now the advance follows the AXIAL FORCE: full rate below half the target, slower to a stop at it, and
    # backing off above it -- the wiggle works under a light, steady load, and the moment the connector drops in the
    # force falls and the advance resumes.
    _fl = en.get('force_limit', {}) or {}
    fl_on = bool(_fl.get('enabled', True))
    fl_target = _num(_fl, 'target_n', 6.0)
    fl_soft = _num(_fl, 'full_rate_below_n', 0.5 * fl_target)
    fl_back = _num(_fl, 'back_off', 0.5)
    fl_max_back_mm = _num(_fl, 'max_back_mm', 2.0)
    fl_tau = _num(_fl, 'filter_s', 0.2)
    if fl_on and not 0.0 <= fl_soft < fl_target:
        log.error('assembly.engage.force_limit: full_rate_below_n (%.1f) must be under target_n (%.1f).',
                  fl_soft, fl_target)
        return False
    # THE FAILED-ENGAGE RETRACT. Much longer than the ordinary between-attempt back-off, because
    # the arm is about to TRAVEL to the place: anything still in the socket levers it off its
    # fixture. `min_mm` is the MEASURED clearance the recovery refuses to travel without.
    _fr = en.get('fail_retract', {}) or {}
    fail_retract_m = _num(_fr, 'distance_mm', 100.0) / 1000.0
    fail_retract_min_m = _num(_fr, 'min_mm', 20.0) / 1000.0
    # CONFIRMATION -- the axial stop says "something resisted", not "it is in the socket". See
    # _RadialConfirm and the confirm block in engage_insertion.
    _cf = en.get('confirm', {}) or {}
    cf_on = bool(_cf.get('enabled', True))
    cf_radial_n = _num(_cf, 'radial_force_n', 1.0)
    cf_radial_s = _num(_cf, 'radial_persistence_s', 0.10)
    cf_max_mm = _num(_cf, 'max_wiggle_mm', 1.0)
    cf_cycles = max(1, int(_cf.get('cycles', 1)))
    # AT MOST THIS LONG, whatever the wiggle's frequency (2026-11-06: at 0.1 Hz "1 cycle" was 10 s of an
    # 8 mm wiggle against a captured connector, which angled it between the fingers) -- and it ENDS the
    # moment the radial reaction is confirmed: that is the only criterion, nothing is learnt by going on.
    cf_max_s = _num(_cf, 'max_s', 3.0)
    cf_fallback_s = _num(_cf, 'fallback_s', 0.5)
    # PUSH HOME -- a LATCHING plug (an RJ45; Task.latch): the engage's success says it is IN the jack, not that it
    # is home, and the latch only clicks over at the bottom. So, the spring kept loaded, on along the connector +X
    # until the floor pushes back -- force_n axial held persistence_s -- then the press is taken off again (the
    # tug that follows tares there). See engage_insertion. Off: the BNC's engage, unchanged.
    _ph = en.get('push_home', {}) or {}
    ph_on = bool(_ph.get('enabled', False))
    ph_force = _num(_ph, 'force_n', 12.0)
    ph_pers = _num(_ph, 'persistence_s', 0.3)
    ph_max_mm = _num(_ph, 'max_mm', 20.0)
    ph_dwell_s = _num(_ph, 'dwell_s', 1.5)
    ph_timeout_s = _num(_ph, 'timeout_s', 20.0)
    ph_unload_n = _num(_ph, 'unload_to_n', 0.0)
    if ph_on and (ph_force <= 0.0 or ph_max_mm <= 0.0):
        log.error('assembly.engage.push_home: force_n and max_mm must be > 0 (got %.1f N, %.1f mm).',
                  ph_force, ph_max_mm)
        return False
    en_pre_mm = float(en.get('preload_mm', 0.0) or 0.0)
    # THE AXES THE ENGAGE'S SPRING RUNS IN (assembly.engage.compliance_frame): tool0's, as it always has (the
    # admittance's own default), or the held CONNECTOR's -- x the insertion axis -- where the stiffness is not the
    # same along every axis. Under a pitched grasp the two differ by ~70 deg: a stiffer x in tool0 is mostly
    # ACROSS the insertion.
    en_law_frame = str(en.get('compliance_frame') or 'tool0').strip().lower()
    if en_law_frame not in ('tool0', 'connector'):
        log.error("assembly.engage.compliance_frame %r must be 'tool0' or 'connector'.", en.get('compliance_frame'))
        return False
    # RETIRED: assembly.engage.speed_mm_s. The engage reference rate is now speed.phase_scale
    # .engage, so every phase's speed is set in one table instead of one phase carrying a private
    # mm/s that silently outranked its own scale. Fail loudly rather than ignore a stale key --
    # a run that quietly ran at a different speed than the file says is the failure to avoid.
    en_speed_retired = en.get('speed_mm_s')
    en_fmax = float(en.get('max_axial_force_n', 0.0) or 0.0)
    en_fpers = float(en.get('persistence_s', 0.0) or 0.0)
    # OSCILLATION SPEED CAP. phase_scale.engage paces the PATH; the oscillation adds its own velocity
    # (amplitude x 2*pi*f) on top. The cap dilates the oscillation CLOCK only, leaving amplitude
    # and the frequency ratio untouched, so it costs wall-clock and nothing else. null = uncapped.
    #
    # ONE WIGGLE IMPLEMENTATION (cable_assembly/skills/wiggle.py), with parameters from
    # configs/wiggle_sampling.yaml unless engage overrides them -- a second copy here is how this
    # app and the sampler would drive different excitations while both claiming to be "the wiggle",
    # and observations from one would stop being comparable with a map from the other.
    en_wig = None
    try:
        en_wig, _esrc = engage_wiggle(cfg)
        if en_wig is not None:
            en_amp = list(en_wig.amp)
            en_frq = list(en_wig.frq)
            log.info('ENGAGE wiggle: %s (parameters from %s).', en_wig.describe(), _esrc)
    except wigmod.WiggleError as exc:
        log.error('%s', exc)
        return False
    en_scale, en_pv, en_pw, en_orbit = traj.wiggle_time_scale(
        en_amp, en_frq, en.get('max_oscillation_speed_mm_s'),
        en.get('max_oscillation_rotation_deg_s'))
    if ins_mode == 'engage':
        if en_rate <= 0:
            log.error('assembly.engage.sample_rate_hz must be > 0.')
            return False
        if en_pre_mm < 0:
            log.error('assembly.engage.preload_mm must be >= 0.')
            return False
        if en_speed_retired is not None:
            # Read the cap here rather than using g_v: this validation runs long before the speed
            # block is unpacked, and the whole point is to fail BEFORE anything moves.
            _cap = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 3.5))
            log.error(
                'assembly.engage.speed_mm_s is RETIRED -- the engage reference rate now comes '
                'from speed.phase_scale.engage, like every other phase. Delete the key and set '
                'speed.phase_scale.engage: %.4g to keep the %.2f mm/s it was asking for '
                '(= %.2f / max_cartesian_translation_mm_s %.4g).',
                float(en_speed_retired) / max(_cap, 1e-9), float(en_speed_retired),
                float(en_speed_retired), _cap)
            return False
        for i, d in enumerate(DIM_KEYS):
            if abs(en_amp[i]) > 0.0 and en_frq[i] <= 0.0:
                log.error('assembly.engage.frequency_hz.%s must be > 0 when its amplitude is '
                          '%.3f (an amplitude with no frequency is a constant offset, which '
                          'belongs in the trajectory).', d, en_amp[i])
                return False
        # EFFECTIVE frequency: the speed cap dilates the oscillation clock, so the raw value
        # would reject a configuration that samples perfectly well.
        f_live = [en_frq[i] * en_scale for i in range(6) if abs(en_amp[i]) > 0.0]
        if f_live and en_rate < 4.0 * max(f_live):
            log.error('assembly.engage.sample_rate_hz %.1f Hz is too coarse for a %.2f Hz '
                      'component (need >= 4x) -- the sampled sine would alias.',
                      en_rate, max(f_live))
            return False
        if en_fmax <= 0:
            log.warning('assembly.engage.max_axial_force_n is 0 -- the engage will run the whole '
                        'trajectory no matter how hard it presses.')

    # ---- ENGAGE CLOCK ANGLE: the roll the connector is MATED at ------------------------------
    # Parsed here, ahead of the clocking blocks, because the connector sweep is stated in ABSOLUTE
    # roll positions about the socket +X and needs this to convert them into rotations from the
    # engaged pose. The frame it rolls (T_base_tconn) is built where the target is loaded.
    eng_clock = np.radians(_num(a, 'engage_clock_deg', 0.0))
    rl_cfg, rl_policy = dict(a.get('rl_insertion') or {}), None
    if ins_mode == 'rl':
        from ..skills import rl_insertion as RLI
        _pp = urconfig.resolve(cfg, os.path.expanduser(str(rl_cfg.get('policy', ''))))   # configs/-relative
        try:
            rl_policy = RLI.Policy(_pp)
        except (OSError, KeyError, ValueError) as exc:
            log.error('assembly.rl_insertion.policy %r: %s (export one with sim/scripts/newton_bnc_export.py).',
                      rl_cfg.get('policy'), exc)
            return False
        # THE PLUG TURNS ONE WAY: the policy's base twist is +z of its socket, -X of the connector, so
        # the run starts at the TOP of the band the wrist can roll through and turns down it -- the
        # stand-off, the start and everything after are at this clock angle (it replaces
        # engage_clock_deg).
        eng_clock = np.radians(float(rl_cfg.get('start_clock_deg', 120.0)))
        _e = rl_policy.spec['env']
        log.info('RL INSERTION: policy %s (iteration %s, trained %s); starts %.1f mm above the socket at a '
                 '%+.0f deg clock, up to %.0f s, turning at most %.0f deg; seated = %.1f mm of advance after '
                 'the first %.0f N push, held %.2f s.', _pp, rl_policy.spec.get('iteration'),
                 os.path.basename(os.path.dirname(rl_policy.spec.get('checkpoint', ''))),
                 1000.0 * float(np.mean(_e['start_height'])), np.degrees(eng_clock), float(_e['episode_s']),
                 float(rl_cfg.get('max_turn_deg', 250.0)), float(rl_cfg.get('seated_advance_mm', 6.0)),
                 float(rl_cfg.get('touch_n', 5.0)), float(rl_cfg.get('hold_s', 0.5)))
    R_clock = xyzrpy_to_matrix([0.0, 0.0, 0.0], [eng_clock, 0.0, 0.0])

    # ---- CLOCKING (post-mate): CABLE clocking, then COLLAR clocking --------------------------
    # Both run only after a mate the operator called successful, each with its OWN compliance and
    # force guard. The guard override is not optional in practice: force_guard: is tuned for a
    # light probing insertion (5 N) and a press-and-twist exceeds that on the first cycle.
    cc = a.get('connector_clocking', {}) or {}
    cl = a.get('collar_clocking', {}) or {}
    if a.get('cable_clocking') is not None:
        # NOT a warning: `enabled` defaults to False, so a config still on the old name would read
        # as "the maneuver is off" and the run would end ENGAGED with nothing said about why.
        log.error('assembly.cable_clocking has been renamed to assembly.connector_clocking. '
                  'Rename the block -- leaving it under the old name would silently disable the '
                  'maneuver and end the run ENGAGED.')
        return False

    try:
        cc_on, cl_on = _clocking_plan(cc, cl)
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if task.latch:
        # A LATCHING PLUG HAS NO BAYONET: nothing to clock, and no disassembly (which undoes the clocking). The
        # app that runs this task turns them off; here only refused if they come back on.
        _dis_on = bool((a.get('disassembly') or {}).get('enabled', False))
        if cc_on or cl_on or _dis_on:
            log.error('%s: a latching plug has no connector clocking, collar clocking or disassembly -- %s on.',
                      task.name, ', '.join(n for n, v in (('connector_clocking', cc_on), ('collar_clocking', cl_on),
                                                          ('disassembly', _dis_on)) if v))
            return False
        if ins_mode != 'engage':
            log.error("%s: assembly.insertion_mode must be 'engage' (got %r) -- 'rl' is the BNC's policy and "
                      "'estimate' needs a contact map of this plug.", task.name, ins_mode)
            return False
        log.info('%s: no clocking -- engaged, pushed home (assembly.engage.push_home %s), the latch tugged '
                 '(assembly.tug_verify %s).', task.title.upper(),
                 'ON' if bool((en.get('push_home') or {}).get('enabled', False)) else 'off',
                 'ON' if bool((a.get('tug_verify') or {}).get('enabled', True)) else 'off')
    else:
        log.info('Post-mate clocking: connector clocking %s, collar clocking %s (a run therefore ends %s '
                 'at best).', 'ON' if cc_on else 'off', 'ON' if cl_on else 'off',
                 'LOCKED/ASSEMBLED' if cl_on else ('SEATED' if cc_on else 'ENGAGED'))
    # ---- THE CONNECTOR SWEEP: absolute roll POSITIONS, visited in turn, one per try ---------------
    # `sweep_deg` is roll angles about the socket +X stated WRT THE TARGET FRAME, not wrt where
    # the connector was engaged -- the reachable band is a property of the FIXTURE, so stating it
    # absolutely means it does not have to be re-derived when engage_clock_deg moves.
    #
    # `rotation_deg` is the legacy single relative stroke and still works: it becomes a one-entry
    # sweep at engage_clock_deg + rotation_deg, so the two are one code path rather than two.
    _sw = cc.get('sweep_deg')
    if _sw is None:
        cc_sweep = [eng_clock + np.radians(_num(cc, 'rotation_deg', 90.0))]
    else:
        try:
            cc_sweep = [np.radians(float(v)) for v in _sw]
        except (TypeError, ValueError):
            log.error('assembly.connector_clocking.sweep_deg must be a list of numbers (roll angles '
                      'in deg wrt the target frame), got %r.', _sw)
            return False
        if not cc_sweep:
            log.error('assembly.connector_clocking.sweep_deg is empty -- give it at least one roll '
                      'position, or delete it to use the legacy rotation_deg stroke.')
            return False
    # ROTATIONS FROM THE ENGAGED POSE -- what the strokes are built from. 0.0 joins the span
    # because the arm starts there.
    cc_legs = [th - eng_clock for th in cc_sweep]
    cc_lo, cc_hi = min(cc_legs + [0.0]), max(cc_legs + [0.0])
    # Band centre, the branch every measured clock angle is wrapped onto (_wrap_near). A band
    # wider than a full turn has no unambiguous branch, so it is refused rather than read wrong.
    cc_mid = 0.5 * (cc_lo + cc_hi)
    if cc_hi - cc_lo > 2.0 * np.pi:
        log.error('assembly.connector_clocking.sweep_deg spans %.0f deg from the engaged roll '
                  '(%+.1f deg) -- more than one turn, so a measured clock angle cannot be told '
                  'from itself plus 360. Narrow the sweep or move assembly.engage_clock_deg.',
                  np.degrees(cc_hi - cc_lo), np.degrees(eng_clock))
        return False
    cc_push_m = _num(cc, 'push_mm', 5.0) / 1000.0
    cc_need_m = _num(cc, 'success_advance_mm', 5.0) / 1000.0
    cc_level = bool(cc.get('level_axis', True))
    cc_level_max = _num(cc, 'level_max_deg', 15.0)
    cc_tries = max(1, int(_num(cc, 'max_tries', 3)))
    cc_settle = _num(cc, 'settle_s', settle_shared)
    cc_hold = _num(cc, 'hold_after_s', 0.0)
    cc_open_after = bool(cc.get('open_gripper_after', True))
    cl_off_m = _num(cl, 'collar_offset_mm', 25.0) / 1000.0
    cl_rot = np.radians(_num(cl, 'rotation_deg', 90.0))
    cl_push_m = _num(cl, 'push_mm', 0.0) / 1000.0
    # GRASP CLOCK ANGLE: the ABSOLUTE roll about the socket +X to take the collar at, wrt the
    # TARGET frame -- the same convention connector_clocking.sweep_deg uses. null = the engaged roll.
    # The turn then runs from here to here + rotation_deg. (Replaces prewind_deg, which existed
    # only because the old radial approach reached the ring by orbiting from wherever the sweep
    # left the arm; an axial approach is placed in free space, so the angle is simply stated.)
    _gc = cl.get('grasp_clock_deg')
    cl_grasp_clock = None if _gc is None else np.radians(float(_gc))
    # RETRACT: how far to back straight off along the TARGET connector -X, from wherever the
    # seat push left the arm, before the gripper is pitched onto the axis. Replaces
    # retreat_mm, which measured the reorient station BACKWARD FROM THE COLLAR and so moved
    # with collar_offset_mm; this is a plain relative back-off from the pose we are in.
    cl_retract_m = _num(cl, 'retract_mm', 300.0) / 1000.0
    # RETREAT: how far BEHIND the approach station, along the connector -X, to reorient. Away
    # from the wall, and the furthest-from-it station in the whole maneuver.
    cl_retreat_m = _num(cl, 'retreat_mm', 100.0) / 1000.0
    # WALL STANDOFF: the largest tool0 station along the connector +X any planned pose may take,
    # mm in the TARGET frame (+X points INTO the wall the socket is mounted on, 0 = the mating
    # face). Checked pre-motion against every station. null = no check.
    cl_wall_mm = cl.get('wall_standoff_mm')
    cl_wall_mm = None if cl_wall_mm is None else float(cl_wall_mm)
    cl_settle = _num(cl, 'settle_s', settle_shared)
    cc_v = None if cc.get('speed_translation_mm_s') is None \
        else float(cc['speed_translation_mm_s'])
    cc_w = None if cc.get('speed_rotation_deg_s') is None else float(cc['speed_rotation_deg_s'])
    cl_v = None if cl.get('speed_translation_mm_s') is None \
        else float(cl['speed_translation_mm_s'])
    cl_w = None if cl.get('speed_rotation_deg_s') is None else float(cl['speed_rotation_deg_s'])
    # TARE: default OFF, unlike the insertion. The arm stands in a MATED, loaded pose when
    # clocking starts, so re-zeroing the F/T there would define the mate load as zero and the
    # guard would only see force ADDED by the screw. Off, it sees absolute force, at the cost of
    # the spring yielding to the standing load during warm-up.
    cc_tare = tare if bool(cc.get('tare_before', False)) else None
    cl_tare = tare if bool(cl.get('tare_before', False)) else None
    cl_tilt_deg = _num(cl, 'max_offaxis_tilt_deg', 5.0)
    # WRIST-3 MARGIN: how far inside each end of the usable wrist_3 range the collar
    # turn must stay. The turn is servoed under admittance, so the arm does not track
    # the reference exactly and a start placed hard against the limit can still walk
    # into it.
    cl_w3_margin = np.radians(_num(cl, 'wrist3_margin_deg', 5.0))
    cl_axis_off = [float(v) for v in (cl.get('axis_offset_mm') or [0.0, 0.0, 0.0])]
    if len(cl_axis_off) != 3:
        log.error('assembly.collar_clocking.axis_offset_mm must have 3 entries (connector-frame '
                  'xyz, mm), got %r.', cl.get('axis_offset_mm'))
        return False
    tv = a.get('tug_verify', {}) or {}
    tv_on = bool(tv.get('enabled', True))
    tv_force = _num(tv, 'pull_force_n', 3.0)
    tv_time = _num(tv, 'pull_time_s', 3.0)
    tv_thresh_m = _num(tv, 'displacement_threshold_mm', 3.0) / 1000.0
    tv_extract_m = _num(tv, 'extraction_distance_mm', 50.0) / 1000.0
    # The tug gets its OWN spring: the pull only works if the spring offset
    # (pull_force_n / stiffness) EXCEEDS displacement_threshold_mm. Under that, an unlocked
    # connector could never move far enough to show and the check goes vacuous. Null inherits.
    _tv_phys = dict(cfg.section('compliance'))
    for _k in ('stiffness', 'mass', 'damping_ratio'):
        if tv.get(_k):
            _tv_phys[_k] = tv[_k]
    adm_tug = make_compliance(robot.arm, _tv_phys) if tv_on else None
    ground_z = float(cfg.get_path('ground_plane.z_m', -0.76))
    # ---- DISASSEMBLY (optional) -----------------------------------------------------------
    dis = a.get('disassembly', {}) or {}
    dis_on = bool(dis.get('enabled', False))
    dis_unlock = bool(dis.get('unlock_collar', True))
    dis_extract_m = _num(dis, 'extract_mm', 60.0) / 1000.0
    dis_place = dis.get('place', {}) or {}
    dis_place_on = bool(dis_place.get('enabled', True))
    dis_clear_m = _num(dis_place, 'clearance_mm', 25.0) / 1000.0
    dis_rise_m = _num(dis_place, 'retreat_mm', 100.0) / 1000.0
    # CYCLES: repeat the whole localize -> pick -> assemble -> disassemble pass. Only
    # meaningful with disassembly ON, because only disassembly puts the cell back into a
    # state a second pass can start from -- so it is CLAMPED to 1 rather than silently
    # looping a run that would try to insert an already-mated connector.
    n_cycles = max(1, int(_num(dis, 'cycles', 1)))
    if n_cycles > 1 and not (dis_on and dis_place_on):
        log.warning('assembly.disassembly.cycles = %d but %s -- a second pass has nothing to '
                    'pick and nowhere to start from. Running ONE cycle.', n_cycles,
                    'disassembly is off' if not dis_on else 'disassembly.place is off')
        n_cycles = 1
    if dis_on and not cc_on:
        # Nothing to unwind and, more to the point, no ENGAGED pose to unwind FROM: the
        # clocking maneuvers are what establish the frames disassembly reverses.
        log.error('assembly.disassembly.enabled needs connector_clocking enabled -- the '
                  'disassembly reverses the clocking, and without it there is nothing to '
                  'reverse and no engaged pose to reverse from.')
        return False

    sp = cl.get('seat_push', {}) or {}
    sp_on = bool(sp.get('enabled', True))
    sp_force = _num(sp, 'force_n', 5.0)
    sp_persist = _num(sp, 'persistence_s', 1.0)
    sp_travel_m = _num(sp, 'max_travel_mm', 15.0) / 1000.0
    # A dedicated guard: the push SUCCEEDS by tripping it (force held for the persistence), so
    # its limits are the push spec, not the global safety limits -- those stay on guard_shared.
    guard_push = ForceGuard(robot.arm, dict(cfg.section('force_guard'),
                                            max_force_n=sp_force,
                                            persistence_s=sp_persist)) if sp_on else None
    if cc_on and cl_on and cc_open_after:
        log.error('assembly.collar_clocking needs connector_clocking.open_gripper_after FALSE: '
                  'the seat push presses with the pads still closed on the connector, so the '
                  'sweep must hand it over held. The release happens after the push, before the '
                  'retreat.')
        return False
    if cc_on and cc_tries > 1 and len(cc_sweep) < 2:
        # Not fatal (one position is a legal legacy sweep) but the extra tries become
        # zero-rotation no-ops.
        log.warning('assembly.connector_clocking: max_tries is %d but the sweep has only ONE roll '
                    'position (%+.1f deg). Legs 2..%d have nothing to turn. Give sweep_deg a '
                    'second position to rock across the slot.',
                    cc_tries, np.degrees(cc_sweep[0]), cc_tries)
    if cc_on and cc_need_m <= 0.0:
        log.error('assembly.connector_clocking.success_advance_mm must be > 0 (got %.2f) -- a zero '
                  'early-out threshold trips on the first servo cycle, ending the stroke before '
                  'it has turned anything.', cc_need_m * 1000.0)
        return False

    # THE KUKA'S IMPEDANCE in place of the software admittance (apps/bnc_assembly_impedance.py;
    # robot.compliance_controller kuka_impedance): every maneuver's controller streams its reference as it is and
    # the controller's spring yields -- robot/admittance.py KukaImpedanceController.
    kuka_imp = getattr(robot.arm, 'compliance', 'admittance') == 'kuka_impedance'
    if kuka_imp:
        from ..robot.admittance import KUKA_DEFAULT_STIFFNESS
        kuka_K = [float(v) for v in (getattr(robot.arm, 'kuka_stiffness', None) or KUKA_DEFAULT_STIFFNESS)]

    def _clock_physics(block, name):
        """(AdmittanceController, ForceGuard) for one clocking maneuver -- every compliance key
        inheriting compliance: and every guard key force_guard: when absent or null, the same
        inheritance rule final_insertion uses."""
        comp = _compliance_override(cfg.section('compliance'), block)
        gsec = dict(cfg.section('force_guard'))
        over = {k: block[k] for k in ('max_force_n', 'max_torque_nm', 'persistence_s')
                if block.get(k) is not None}
        # NOT `enabled` -- that is the MANEUVER's own switch in this block. Reusing it would
        # make "run the screw with no guard" inexpressible.
        if block.get('force_guard_enabled') is not None:
            over['enabled'] = bool(block['force_guard_enabled'])
        g = ForceGuard(robot.arm, {**gsec, **over})
        # SAY WHAT THE STIFFNESS MEANS, not just what it is. This is an ADMITTANCE loop: to hold
        # any force it must deflect by w/S, so the commanded pose necessarily backs off the
        # contact by exactly that much. Printing the give next to the limit that provokes it
        # makes the trade visible at the moment it is chosen -- a soft axial stiffness against a
        # modest force limit can give more than the whole insertion is trying to travel, which
        # reads on the bench as "it pushed in, then came back out".
        stiff = [float(v) for v in (comp.get('stiffness') or [2000.0] * 3 + [15.0] * 3)]
        if kuka_imp:                                      # the spring that is really there: the controller's
            stiff = list(kuka_K)
            comp = dict(comp, selected_axes=[1] * 6)
        lim = block.get('max_axial_force_n')
        lim = float(lim) if lim is not None else float(g.max_force)
        give_mm = (lim / stiff[0] * 1000.0) if stiff[0] > 0 else float('inf')
        log.info('%s ON (%s): stiffness %s, compliant axes %s, guard %.1f N / %.1f Nm.', name,
                 "the KUKA's Cartesian impedance" if kuka_imp else 'software admittance',
                 stiff if kuka_imp else comp.get('stiffness'),
                 ''.join(a for a, on in zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), comp.get('selected_axes', [1] * 6))
                         if on) or 'none', g.max_force, g.max_torque)
        log.info('    axial give %.1f mm at %.1f N (= force / stiffness %.0f N/m) -- the '
                 'commanded pose backs off by this much to hold that force.',
                 give_mm, lim, stiff[0])
        return make_compliance(robot.arm, comp), g

    adm_cc = guard_cc = adm_cl = guard_cl = None
    if cc_on:
        adm_cc, guard_cc = _clock_physics(cc, 'Connector clocking')
    # FORCE NULLING ACROSS THE CONNECTOR (the user, 2026-11-06: "connector clocking is getting stuck due to
    # misalignment and friction - can we implement force control/nulling here to eliminate preloads? perhaps in
    # the connector y and z directions"). The screw turns about the MEASURED socket axis; the connector's own is
    # off it by whatever the socket estimate and the grasp got wrong, and a 1500 N/m spring held that offset as a
    # side load (1.5 N a millimetre) the turn then drags the plug against. Across the connector -- its own y and
    # z, the law run at the connector -- there is now no spring: a side load over the deadband moves the pose
    # until it is gone (robot/admittance.py null_axes). The push along x and the stiff rotations stay as they were.
    _cn = (cc.get('force_null') or {}) if cc_on else {}
    cc_null_on = bool(cc_on and _cn.get('enabled', True))
    # THE AXIS PROBE (the user, 2026-11-06: "implement the force based method"): see connector_clocking.
    _ap = (cc.get('axis_probe') or {}) if cc_on else {}
    ap_on = bool(cc_on and _ap.get('enabled', True))
    ap_turns = [float(v) for v in (_ap.get('turn_deg') or [60.0, 0.0])]
    ap_apply = bool(_ap.get('apply', True))
    ap_min_pts = int(_ap.get('min_points', 40))
    ap_min_span = _num(_ap, 'min_span_deg', 30.0)
    ap_max_res = _num(_ap, 'max_residual_mm', 0.5)
    ap_max_shift = _num(_ap, 'max_shift_mm', 6.0)
    ap_max_hand = _num(_ap, 'max_hand_mm', 6.0)
    ap_every = int(_ap.get('sample_every', 3))
    # THE SLOT ESTIMATE (the user, 2026-11-06: "using the slot estimate the connector roll orientation, but still do
    # connector clocking to account for errors ... try both angles in increasing increments"): see connector_clocking
    # and perception/slot_roll.py.
    _sr = (cc.get('slot_roll') or {}) if cc_on else {}
    sr_on = bool(cc_on and _sr.get('enabled', True))
    sr_capture = [str(v) for v in (_sr.get('capture') or ['standoff', 'engaged'])]
    sr_use = [str(v) for v in (_sr.get('use') or ['standoff'])]
    sr_frames = max(1, int(_sr.get('frames', 3)))
    sr_symmetric = bool(_sr.get('symmetric', True))
    sr_cad = _sr.get('cad')
    if sr_cad:
        sr_cad = os.path.expanduser(str(sr_cad))
        sr_cad = sr_cad if os.path.isabs(sr_cad) else os.path.join(cfg.get('data_dir', 'data'), sr_cad)
    sr_cad_scale = _num(_sr, 'cad_scale', 0.001)
    sr_min_margin = _num(_sr, 'min_margin', 0.1)
    sr_pin = _sr.get('pin_deg')
    sr_pin = None if sr_pin is None else float(sr_pin)
    sr_learn = bool(_sr.get('learn', True))
    sr_learn_move = _num(_sr, 'learn_max_move_mm', 10.0)
    sr_drop_mm = _num(_sr, 'drop_mm', 1.5)
    sr_drop_win = _num(_sr, 'drop_window_deg', 30.0)
    sr_step = _num(_sr, 'step_deg', 15.0)
    sr_max = _num(_sr, 'max_deg', 90.0)
    sr_orient_probe = bool(_sr.get('orient_probe', True))
    # AIM THE LEGS BY IT, or only measure it (the user, 2026-11-06: back to the plain rocking between the sweep's
    # negative and positive ends): false = the plan is still worked out and recorded -- with the aim error a drop-in
    # shows -- but the legs are sweep_deg's and the probe turns its own way.
    sr_aim = bool(_sr.get('aim', True))
    sr_store_path = os.path.join(cfg.get('data_dir', 'data'), str(_sr.get('store', 'bnc_pin_angle.json')))
    if sr_on and sr_step <= 0:
        log.error('assembly.connector_clocking.slot_roll.step_deg must be > 0 (got %s).', sr_step)
        return False
    if cc_null_on:
        # y, z: across the axis. ry, rz: TILT about the connector's y and z through its origin -- the mating face --
        # so the plug can turn coaxial with the socket where it meets it, like a remote-centre compliance (the
        # user, 2026-11-06: "connector clocking appears to be at an angle which leads to a runaway effect of
        # misalignment and slip. perhaps force control can help keep it aligned with the socket"). x (the push)
        # and rx (the screw itself) never.
        _ax = {'y': 1, 'z': 2, 'ry': 4, 'rz': 5}
        _axes = [str(v).lower() for v in (_cn.get('axes') or ['y', 'z'])]
        if not _axes or any(v not in _ax for v in _axes):
            log.error("assembly.connector_clocking.force_null.axes %r: of 'y', 'z', 'ry', 'rz' only -- across the "
                      "connector and tilts about it (x is the push, rx the screw).", _cn.get('axes'))
            return False
        _dead = _cn.get('deadband_n')
        _dead = float(cfg.get_path('compliance.force_deadband_n', 0.0) or 0.0) if _dead is None else float(_dead)
        _damp = _num(_cn, 'damping_n_s_per_m', _num(_cn, 'damping_n_s_m', 2000.0))
        _cap = _num(_cn, 'max_offset_mm', 5.0)
        _lin = [_ax[v] for v in _axes if v in ('y', 'z')]
        _rot = [_ax[v] for v in _axes if v in ('ry', 'rz')]
        if _lin:
            adm_cc.null_axes(_lin, _damp, _dead, _cap / 1000.0)
            log.info('    FORCE NULLING across the connector (%s): no spring there -- a side load over %.1f N moves '
                     'the pose at F / %.0f N s/m until it is gone (1 N over: %.1f mm/s), at most %.1f mm.',
                     ', '.join(v for v in _axes if v in ('y', 'z')), _dead, _damp, 1000.0 / _damp, _cap)
        if _rot:
            _tdead = _num(_cn, 'torque_deadband_nm', 0.5)
            _tdamp = _num(_cn, 'rot_damping_nm_s_per_rad', _num(_cn, 'rot_damping_nm_s_rad', 100.0))
            _tcap = _num(_cn, 'max_tilt_deg', 3.0)
            adm_cc.null_axes(_rot, _tdamp, _tdead, np.radians(_tcap))
            log.info('    MOMENT NULLING about the connector (%s), at its mating face: no spring there -- a moment over '
                     '%.2f Nm tilts the gripper at M / %.0f Nm s/rad (1 Nm over: %.2f deg/s) until it is gone, at most '
                     '%.1f deg -- the plug kept coaxial with the socket it is in.',
                     ', '.join(v for v in _axes if v in ('ry', 'rz')), _tdead, _tdamp, np.degrees(1.0 / _tdamp), _tcap)
    if cl_on:
        adm_cl, guard_cl = _clock_physics(cl, 'Collar clocking')
    adm_en = guard_en = None
    if ins_mode == 'engage':
        # Its OWN compliance and guard, same override schema as the clocking blocks: the engage
        # wants neither the probing block's stop-on-contact nor a clocking block's jam limit.
        adm_en, guard_en = _clock_physics(en, 'Engage insertion')
    if kuka_imp:
        # WHAT EVERY VIRTUAL OFFSET MEANS ON THE CONTROLLER'S SPRING -- the config's stiffness, mass, damping and
        # force deadbands are not what yields any more, so each push / wiggle is stated as the force it makes.
        _K = list(kuka_K)
        log.warning('COMPLIANCE: THE KUKA\'S CARTESIAN IMPEDANCE, %s N/m, Nm/rad (robot.fri.cartesian_impedance -- it '
                    'must match the FRI app on the pendant), in place of the software admittance: the per-maneuver '
                    'stiffness / mass / damping / deadbands in the config are NOT used; force nulling still runs, as a '
                    'set-point drift. The tool\'s load data must be set in Sunrise.', [round(v) for v in _K])
        if cc_on:
            log.info('  connector clocking: push %.1f mm -> %.1f N of press (early-out at %.1f mm of advance).',
                     cc_push_m * 1000.0, _K[0] * cc_push_m, cc_need_m * 1000.0)
        if cl_on:
            log.info('  collar clocking: the seat push sized for %.1f N on this spring; the turn\'s push %.1f mm -> '
                     '%.1f N.', sp_force, cl_push_m * 1000.0, _K[0] * cl_push_m)
        if ins_mode == 'engage' and en_wig is not None:
            _lat = max(abs(v) for v in en_wig.amp[:3]) if en_wig.amp else 0.0
            _rot = max(abs(v) for v in en_wig.amp[3:]) if len(en_wig.amp) > 3 else 0.0
            log.info('  engage wiggle: up to %.1f mm / %.1f deg -> up to %.1f N / %.2f Nm against a socket that holds '
                     'still (the admittance gave way at its own stiffness instead).', _lat, _rot, _K[1] * _lat / 1000.0,
                     _K[4] * np.radians(_rot))

    # ---- Target from the SHARED catalogue (the same record estimator_eval assembles to) ----
    tname = a.get('target_frame')
    targets = tool_frames.load_targets(cfg)
    # ONE PERCEPTION METHOD (pose_method): 'pose6d' finds the connector AND the socket with the
    # 6D pose estimator (perception/pose6d.py); 'sam3' is the SAM3 scans and the targets/markers.
    pose_method = str(cfg.get('pose_method') or 'sam3').strip().lower()
    if pose_method not in ('sam3', 'pose6d'):
        log.error("pose_method must be 'sam3' or 'pose6d', got %r.", cfg.get('pose_method'))
        return False
    # The socket from the 6D estimator: target_source pose6d, or pose_method pose6d -- unless the
    # target is asked for as `kinematic`, the recorded mate (a fixture whose socket the estimator
    # cannot see well, e.g. inside a housing: the connector from 6D, the socket from the record).
    _src = str(a.get('target_source') or '').strip().lower()
    pose6d_target = (pose_method == 'pose6d' and _src != 'kinematic') or _src == 'pose6d'
    # The socket's pose: measured by the 6D estimator before every pick in pose6d mode. A recorded
    # targets: entry, if there is one, is then only what the measurement is checked against; with
    # none, a placeholder stands until the first measurement replaces it, and nothing is planned
    # from it before (pose6d_measured).
    pose6d_measured = [False]
    socket_obj = [None]                 # the socket's last 6D pose (T_base_obj): the refine's prior
    # THE MOVE TO THE STAND-OFF, PLANNED (assembly.standoff.plan): cuMotion round a collision scene
    # built from depth views of the socket and what is round it (taken before the pick -- the 6D
    # socket views, or look_at_socket's), the camera mount, the D405, the gripper AND the held
    # connector with the cable leaving it on the tool. Off: the plain joint move to the IK solution.
    _st = a.get('standoff', {}) or {}
    plan_standoff = bool(_st.get('plan', False))
    socket_views = [[]]                      # [(joints, Frame)] of the socket, this cycle's
    if not tname or tname not in targets:
        if not (pose6d_target and tname):
            log.error('assembly.target_frame %r needs a targets: entry in %s.',
                      tname, tool_frames.frames_path(cfg))
            return False
        log.info('No targets: entry for %r -- the 6D estimator measures the socket before each pick.', tname)
    if tname in targets:
        T_base_socket = targets[tname]
    else:
        T_base_socket = np.eye(4)              # the placeholder the first 6D measurement replaces

    # ---- WHERE THE TARGET COMES FROM: assembly.target_source -----------------------------------
    # kinematic  the recorded targets: pose above. It is a reading taken by hand-guiding to a good
    #            mate, and it is only true while nothing moves -- the fixture, the robot base, the
    #            calibration. The 2026-08 hose campaign lost a session to exactly that drift.
    # visual     fiducials bolted around the socket, calibrated once by apps/marker_calibration.
    #            Each marker carries the target in ITS OWN frame, so the rig travels WITH the
    #            fixture: unbolt it, move it, and the run still finds it.
    #
    # THE VISUAL ANSWER IS NOT MORE ACCURATE THAN THE RECORD IT WAS CALIBRATED FROM -- the rig was
    # solved against this same targets: entry, so it inherits every error that reading carried.
    # What it adds is INVARIANCE, and a live check: markers that disagree with each other, or a
    # fixture that has moved further than max_shift_mm, are things the kinematic path cannot
    # notice at all.
    # pose6d   the 6D pose estimator, from a view of the socket and an orbit about it, converted
    #          to the target the plug's connector frame reaches when seated (locate_target_pose6d).
    tgt_source = str(a.get('target_source') or 'kinematic').strip().lower()
    if pose_method == 'pose6d' and tgt_source == 'kinematic':
        log.info('pose_method pose6d, assembly.target_source kinematic: the connector from the 6D estimator, '
                 'the socket from the recorded targets: entry %r.', tname)
    elif pose_method == 'pose6d':
        if tgt_source != 'pose6d':
            log.info('pose_method pose6d: the socket comes from the 6D estimator too (assembly.target_source '
                     '%r set aside).', tgt_source)
        tgt_source = 'pose6d'
    if tgt_source not in ('kinematic', 'visual', 'pose6d'):
        log.error("assembly.target_source must be 'kinematic', 'visual' or 'pose6d', got %r.",
                  a.get('target_source'))
        return False
    vt = a.get('visual_target', {}) or {}
    vt_rig = None
    if tgt_source == 'visual':
        # VALIDATED HERE, at parse time, not after the arm has already homed and driven to a view
        # pose: a missing rig is a config typo, and finding out about it three moves in wastes a
        # run and leaves the operator guessing which name was wrong.
        try:
            rigs = tool_frames.load_marker_rigs(cfg)
        except ValueError as exc:
            log.error('assembly.target_source is visual but %s', exc)
            return False
        vt_rig = rigs.get(tname)
        if vt_rig is None:
            log.error('assembly.target_source is visual but %s has no marker_rigs: entry for %r '
                      '(rigs present: %s). Calibrate one with cable_assembly.apps.marker_calibration.',
                      tool_frames.frames_path(cfg), tname,
                      ', '.join(sorted(rigs)) or 'none')
            return False
        log.info('Target source: VISUAL -- rig %r, markers %s (%s mm).', tname,
                 ', '.join(str(m) for m in sorted(vt_rig['markers'])),
                 ', '.join('%.1f' % (m['size_m'] * 1000.0)
                           for _i, m in sorted(vt_rig['markers'].items())))
    elif tgt_source == 'pose6d':
        log.info('Target source: 6D POSE -- the socket measured from several views before each pick%s.',
                 ', checked against the recorded targets: entry' if tname in targets else '')
    else:
        log.info('Target source: KINEMATIC -- the recorded targets: pose of %r.', tname)

    # ---- ENGAGE CLOCK ANGLE: the roll the connector is MATED at ------------------------------
    # A BNC is free about its own axis until the bayonet pins pick up, so the clock angle it is
    # ENGAGED at is a free parameter. Applied ONCE, here, as a roll of the FRAME: every pose in
    # the app is built from T_base_tconn, so rolling it rolls the trajectory, stand-off,
    # observations, belief reset, screw axis and collar frame rigidly together. Rolling any one of
    # them instead would put the part at one clock angle and its reference at another.
    #
    # It does not place the sweep -- sweep_deg states its positions against the target frame -- so
    # this only decides where the connector starts, and which way the first leg turns. +X is
    # unchanged by a roll about +X, so the insertion axis, push, retract legs and every depth
    # reading are identical at any value.
    #
    # NOT FOR A PLUG THAT GOES IN ONE WAY ROUND (no symmetry: the RJ45). Its 6D target keeps the jack's measured
    # turn -- latch to slot (P6.target_from_socket) -- and any roll of it would put the latch off the slot. So 0,
    # whatever the config (the bench cell's -135 is a reach choice for the free-turning BNC).
    if pose_method == 'pose6d' and P6.SYMMETRY[P6.plug_part(cfg)] == 'none' and abs(eng_clock) > 1e-9:
        log.warning('assembly.engage_clock_deg %+.1f SET ASIDE: the %s goes in one way round (its latch to the '
                    "jack's slot, the turn the 6D target keeps) -- engaged at 0.", np.degrees(eng_clock),
                    P6.plug_part(cfg))
        eng_clock = 0.0
        R_clock = np.eye(4)
    T_base_tconn = T_base_socket @ R_clock
    log.info('Clock angles about the socket +X: engage at %+.1f deg, connector sweep visits %s deg '
             'in turn (up to %d leg%s) -- the run works the %+.1f .. %+.1f deg band about the '
             'declared roll of %r.',
             np.degrees(eng_clock), [round(float(np.degrees(t)), 1) for t in cc_sweep],
             cc_tries, '' if cc_tries == 1 else 's',
             np.degrees(eng_clock + cc_lo), np.degrees(eng_clock + cc_hi), tname)
    if abs(eng_clock) > 1e-9:
        if ins_mode == 'estimate':
            # Rolling the frame keeps observations near identity, so they land in the map's
            # coordinates -- but the bayonet slots and keyway are NOT bodies of revolution, so the
            # contact they describe at another clock angle is different contact.
            log.warning('  insertion_mode is ESTIMATE and the clock angle is not 0. The manifold '
                        'was collected at ONE clock angle; the observations will be expressed in '
                        'the rolled frame but the CONTACT they describe is a different part of '
                        'the socket. Re-collect the map at this clock angle, or engage with '
                        'insertion_mode: engage (which matches nothing and is unaffected).')
    # FRAME FOR THE POST-ENGAGEMENT MANEUVERS -- connector clocking, collar clocking, the escape's
    # target-frame leg and the tug all build their axes and stations from T_clk. 'target' binds it
    # to the recorded socket pose; 'believed' rebinds it, when clocking starts, to the
    # estimator-corrected in-hand belief. The ENGAGEMENT itself always uses the target frame.
    pe_frame = str(a.get('post_engage_frame') or 'target').lower()
    if pe_frame not in ('target', 'believed'):
        log.error("assembly.post_engage_frame must be 'target' or 'believed', got %r.",
                  a.get('post_engage_frame'))
        return False
    T_clk = T_base_tconn
    csv_in = urconfig.resolve(cfg, a.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(a.get('trajectory_angles_deg', False)))
    dense = traj.resample(mats, float(a.get('translational_resolution_m', 0.001)),
                          float(a.get('rotational_resolution_deg', 1.0)))

    # ---- ANCHORING: the path ENDS on the recorded mate ---------------------------------------
    # Normalises the trajectory so its last row lands exactly on the target, whatever that row
    # says -- the same thing uncertain_sampling does when building the contact map. A no-op for a
    # conforming CSV, and what keeps a non-conforming one (the shipped CSV once carried a +10 mm
    # press in its last row) from silently putting every observation deeper than the map goes.
    # A press belongs in final_insertion.preload_mm, which applies to the COMMIT alone.
    T_base_targetobj = T_base_tconn @ inverse(mats[-1])
    T_base_commit = T_base_targetobj @ translation_matrix([fi_preload_mm / 1000.0, 0.0, 0.0])

    def _anchor_target(T_socket):
        """Rebuild EVERY frame the run plans from, off a (re)measured socket pose.

        These four are the only parse-time products of the target, and each nested maneuver reads
        them through the closure at call time -- so rebinding them here is enough to move the whole
        run onto a new socket pose. Rebinding a subset instead would put the trajectory at one
        place and the clocking axes at another, which is the failure mode this exists to prevent."""
        nonlocal T_base_socket, T_base_tconn, T_clk, T_base_targetobj, T_base_commit
        T_base_socket = T_socket
        T_base_tconn = T_base_socket @ R_clock
        T_clk = T_base_tconn
        T_base_targetobj = T_base_tconn @ inverse(mats[-1])
        T_base_commit = T_base_targetobj @ translation_matrix([fi_preload_mm / 1000.0, 0.0, 0.0])

    _sh_m, _sh_r = pose_error(T_base_targetobj, T_base_tconn)
    if _sh_m * 1000.0 > 1e-6 or np.degrees(_sh_r) > 1e-6:
        log.warning('trajectory_csv\'s last row is NOT identity (off by %.2f mm / %.2f deg) -- '
                    'ANCHORED onto the recorded mate, like uncertain_sampling. A press belongs '
                    'in assembly.final_insertion.preload_mm, not in a trajectory row.',
                    _sh_m * 1000.0, np.degrees(_sh_r))

    # ---- COLLECTION MODE (estimator_eval.eval.collection semantics) ----
    col = a.get('collection', {}) or {}
    col_mode = str(col.get('mode', 'attempts')).strip().lower()
    if col_mode not in ('attempts', 'offset_sweep', 'peck'):
        log.error("assembly.collection.mode %r must be 'attempts', 'offset_sweep' or 'peck'.",
                  col_mode)
        return False
    sweep_offsets = col.get('sweep_offsets')
    if sweep_offsets is None:
        sweep_offsets = [[0.0, 0.0, 0.0, 0.0, float(p), 0.0]
                         for p in np.arange(-4.0, 4.01, 2.0)]
    sweep_offsets = [[float(v) for v in o] for o in sweep_offsets]
    if col_mode == 'offset_sweep' and (not sweep_offsets
                                       or any(len(o) != 6 for o in sweep_offsets)):
        log.error('assembly.collection.sweep_offsets must be a non-empty list of 6-vectors '
                  '[x, y, z (m), roll, pitch, yaw (deg)].')
        return False
    peck_back_m = float(col.get('peck_retract_mm', 5.0)) / 1000.0
    peck_timeout = float(col.get('peck_timeout_s', 30.0))
    if col_mode == 'offset_sweep':
        log.info('Collection: offset_sweep, %d passes per attempt (pitch %s).',
                 len(sweep_offsets), [round(o[4], 1) for o in sweep_offsets])
    elif col_mode == 'peck':
        log.info('Collection: peck, %.1f mm back-off per stop, %.0f s budget.',
                 peck_back_m * 1000.0, peck_timeout)

    # ---- Trajectory noise (same semantics as estimator_eval.eval.trajectory_noise) ----
    tn = a.get('trajectory_noise', {}) or {}
    tn_on = bool(tn.get('enabled', False))
    tn_std = tn.get('std') or [0.0] * 6
    tn_std = [float(v) for v in tn_std]
    if len(tn_std) != 6:
        log.error('assembly.trajectory_noise.std must have 6 entries.')
        return False
    tn_w = max(1, int(tn.get('smooth_window', 10)))
    tn_da, tn_dt = float(tn.get('noise_decay_attempt', 0.0)), \
        float(tn.get('noise_decay_traj', 0.0))
    noise_rng = np.random.default_rng()
    # The place-scatter RNG, in a list so the closures share ONE stream. Seeded when the config
    # asks, so a scattered multi-cycle run can be re-created exactly when cycle 34 goes wrong.
    _ps_seed = ((cfg.section('assembly').get('disassembly', {}) or {})
                .get('place_scatter', {}) or {}).get('seed')
    _place_rng = [np.random.default_rng(None if _ps_seed is None else int(_ps_seed))]

    spd = cfg.section('speed')
    scales = spd.get('phase_scale', {}) or {}

    def phase(name):
        robot.arm.set_speed_scale(float(scales.get(name, 1.0)), name)

    g_v = float(spd.get('max_cartesian_translation_mm_s', 3.5))
    g_w = float(spd.get('max_cartesian_rotation_deg_s', 5.0))
    s_asm = float(scales.get('assemble', 1.0))
    # ENGAGE HAS ITS OWN SCALE, falling back to `assemble`. The insertion is the slowest thing
    # the arm does -- contact force scales hard with approach speed, and whole oscillation cycles
    # have to fit inside the insertion -- so it needs to be settable without also slowing every
    # other move that runs under the assemble phase. Absent -> assemble, so a config that never
    # split them behaves exactly as before.
    s_eng = float(scales.get('engage', scales.get('assemble', 1.0)))
    s_ret = float(scales.get('retract', 1.0))
    s_std = float(scales.get('standoff', 1.0))
    min_seg_s = 1.0 / adm.rate

    def caps(v=None, w=None):
        """Resolve a maneuver's speed overrides. A null means "no override" = the GLOBAL cap
        times the assemble phase scale. Every duration goes through here, so a null can never
        reach the arithmetic unresolved."""
        return (g_v * s_asm if v is None else v), (g_w * s_asm if w is None else w)

    def seg_time(A, B, v=None, w=None):
        lin_m, ang_rad = pose_error(A, B)
        cv, cw = caps(v, w)
        return _path_time(lin_m * 1000.0, np.degrees(ang_rad), cv, cw, min_seg_s)

    def screw_ramp(adm, ref_at, guard, v, w, ang_deg, label='', on_step=None):
        """Ramp along the EXACT screw path instead of the straight chord between its endpoints.

        admittance.ramp SLERPs the rotation but LERPs the translation, so one ramp draws a
        STRAIGHT LINE between the two tool0 positions. Harmless at servo-rate steps, badly wrong
        for a whole clocking stroke: a `theta` turn about an axis `r` from tool0 bows into an arc
        and the chord cuts inside it by r(1 - cos(theta/2)) -- 52 mm at 90 deg with r = 178 mm,
        dragging whatever sits on the axis through that excursion. Both ENDPOINTS stay exactly
        right, which is why it hides in every logged pose. Subdividing on the true screw keeps
        each lerp to a fraction of a degree, where the chord error is microns.

        The speed cap is applied to the ARC, which seg_time cannot do -- it measures the chord and
        under-counts a 90 deg turn's real path by ~11%.

        Returns ('done' | 'seated', f), f being the FRACTION of the stroke the reference reached.
        A caller that continues the motion needs it: restarting from the nominal endpoint would
        jump across the arc the guard just refused, and restarting from the MEASURED pose would
        throw away the deflection holding the part loaded."""
        SAMPLES = 64
        coarse = [ref_at(k / SAMPLES) for k in range(SAMPLES + 1)]
        pts = [T[:3, 3] for T in coarse]
        arc_mm = sum(float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(SAMPLES)) * 1000.0
        chord = pts[-1] - pts[0]
        chord_mm = float(np.linalg.norm(chord)) * 1000.0
        sag_mm = 0.0
        if chord_mm > 1e-6:
            u = chord / np.linalg.norm(chord)
            sag_mm = max(float(np.linalg.norm((p - pts[0]) - np.dot(p - pts[0], u) * u))
                         for p in pts) * 1000.0
        cv, cw = caps(v, w)          # a null override must be resolved BEFORE the arithmetic
        dur = _path_time(arc_mm, ang_deg, cv, cw, min_seg_s)
        n = max(1, int(np.ceil(dur * adm.rate)))
        log.info('  %s%.1f deg about a fixed axis: arc %.1f mm in %d steps over %.2f s '
                 '(a single ramp would cut the chord and pull the axis %.1f mm off true)',
                 label, ang_deg, arc_mm, n, dur, sag_mm)
        dt, prev = dur / n, coarse[0]
        for k in range(1, n + 1):
            cur = ref_at(k / n)
            out = adm.ramp(prev, cur, dt, guard, on_step=on_step)
            prev = cur
            if out == 'seated':
                return 'seated', k / n
        return 'done', 1.0

    retract_m = float(a.get('retract_distance_m', 0.05))
    decim = max(1, int(a.get('log_decimation', 5)))
    tol = a.get('success_tolerance', {}) or {}
    tol_pos_m, tol_rot_rad = float(tol.get('pos_mm', 2.0)) / 1000.0, \
        np.radians(float(tol.get('rot_deg', 3.0)))
    max_attempts = int(a.get('max_attempts', 5))
    accumulate = bool(a.get('accumulate_observations', True))
    dbg = a.get('debug_match', {}) or {}
    dbg_on = bool(dbg.get('enabled', False))
    dbg_live = dbg.get('live', True)
    dbg_live = (os.path.join(cfg.get('data_dir', 'data'), 'experiments')
                if dbg_live is True else (dbg_live or None))

    frames = tool_frames.load_frames(cfg)
    tool_frames.check_drift(frames, cfg)      # warns if frames.yaml and the config sections drift

    # THE BELIEVED IN-HAND POSE, fingertip-relative (the estimator updates it, so it must be the
    # local pose, not the flattened tool0 chain). Preference order:
    #   1. estimation.initial_connector_frame -- a name in the shared frames catalogue, parented
    #      to `fingertip` there. Exists so the value lives in ONE place.
    #   2. estimation.initial_connector_in_fingertip -- the raw pose, kept as an override.
    #   3. cables.yaml junction_in_fingertip -- "wherever the pick put it".
    init = cfg.get_path('estimation.initial_connector_in_fingertip')
    init_name = cfg.get_path('estimation.initial_connector_frame')
    if init_name:
        if init_name not in frames:
            log.error('estimation.initial_connector_frame %r is not in %s.', init_name,
                      tool_frames.frames_path(cfg))
            return False
        T_ftip_conn = inverse(robot.T_tool0_fingertip) @ frames[init_name]
        log.info('Believed in-hand pose from the shared catalogue: %r.', init_name)
        if init:
            # Both given: the catalogue wins, but a silent disagreement is how the two drift.
            d_lin, d_ang = pose_error(from_cfg(init), T_ftip_conn)
            if d_lin * 1000.0 > 0.5 or np.degrees(d_ang) > 0.2:
                log.warning('estimation.initial_connector_in_fingertip disagrees with frame %r by '
                            '%.2f mm / %.2f deg. The FRAME is being used; delete the inline pose '
                            'or align it.', init_name, d_lin * 1000.0, np.degrees(d_ang))
    else:
        T_ftip_conn = from_cfg(init) if init else from_cfg(cfg.section('junction_in_fingertip'))
    # PICKUP PITCH: the frames catalogue declares the SQUARE grip, so a pitched pickup rotates
    # the part in the hand by the same angle. Applied here rather than in frames.yaml, so the
    # declared frame stays the physical truth and the pitch stays a run-time choice.
    _grip_off = fingertip_in_connector(cfg)
    _oxyz, _orpy = matrix_to_xyzrpy(_grip_off)
    # THE RAW NOMINAL, kept before the grasp is folded in. held_belief maps "the frames
    # catalogue's SQUARE grip" to "the grip this run actually takes", so it must always be
    # applied to the catalogue value -- never to a belief that already carries a grasp. Feeding
    # it its own output composes the two grasps: the reorient recovery did exactly that and put
    # the connector down 75 deg off horizontal, nearly axis-vertical.
    T_ftip_conn_catalogue = np.array(T_ftip_conn, dtype=float)
    T_ftip_conn = held_belief(T_ftip_conn, from_cfg(cfg.section('junction_in_fingertip')),
                              _grip_off)
    # THE CONNECTOR DATUM ON THE PART (pose_method pose6d): junction -> connector, ONE transform for
    # the socket target, the in-hand prior, the measured belief and the pick's belief. The
    # catalogue's (the initial connector pose vs junction_in_fingertip: for the BNC 45.7 mm along
    # the axis) -- but for a ROUND plug, ON ITS AXIS: the 6D junction is on the axis, where the
    # catalogue's was where the ground-plane scan saw it, on the bench one radius (7.5 mm) below.
    # Carried over, that offset would put the frame the clocking turns about off the plug's axis.
    # ANY 6D PLUG, round or not: an RJ45's junction is on its cord's axis as well (P6.T_cord_junction).
    T_junction_conn = inverse(from_cfg(cfg.section('junction_in_fingertip'))) @ T_ftip_conn_catalogue
    if pose_method == 'pose6d':
        T_junction_conn[1:3, 3] = 0.0
        T_ftip_conn = inverse(_grip_off) @ T_junction_conn      # = held_belief, the datum on the axis
    # THE DRY CELL (a dry run, dry_cell.enabled): its scanner, its virtual jack on the arm's contact hook, its
    # gripper stall -- see apps/_dry_cell.py. Its plug lies where the reset pose puts a cable (set below).
    dry = _dry_cell.DryCell.setup(cfg, robot, T_junction_conn)
    if dry is not None:
        scanner = dry.scanner
    log.info('fingertip_in_connector xyz %s mm rpy %s deg (target fingertip wrt the detected '
             'connector: +x along the connector, +y across the cable, +z up) -> in-hand '
             'belief derived to match. The pick and the belief come from this ONE transform, '
             'so they cannot drift apart.',
             np.round(_oxyz * 1000.0, 2).tolist(), np.round(np.degrees(_orpy), 2).tolist())
    # THE MEASURED RESIDUAL, applied last and to the BELIEF ONLY: how the part actually seats
    # in the jaws at this approach angle, which no frame can predict. Moves nothing at pickup.
    _bel_off = belief_offset(cfg)
    _bo_xyz, _bo_rpy = matrix_to_xyzrpy(_bel_off)
    if float(np.linalg.norm(_bo_xyz)) > 0.0 or float(np.linalg.norm(_bo_rpy)) > 0.0:
        T_ftip_conn = offset_belief(T_ftip_conn, _bel_off)
        log.info('Belief offset %s mm / %s deg (CONNECTOR frame: +x along the connector axis) '
                 'applied to the in-hand pose ONLY -- the grasp command is unchanged.',
                 np.round(_bo_xyz * 1000.0, 2).tolist(),
                 np.round(np.degrees(_bo_rpy), 2).tolist())

    # START HELD: the connector is ALREADY in the hand at a KNOWN grasp -- a simulated cell
    # (robot.backend isaac_sim puts it there), or a bench where it was placed by hand -- so there
    # is no pick to derive the belief from. It is the held frame itself (frames: <target_frame>,
    # tool0 -> connector), and the reset, the gripper warm-up and the pick that would open the
    # hand are skipped (a missed engage still falls back to a full pick).
    start_held = bool(a.get('start_held', False))
    if start_held:
        if tname not in frames:
            log.error('assembly.start_held needs the held frame %r in %s.', tname,
                      tool_frames.frames_path(cfg))
            return False
        T_ftip_conn = inverse(robot.T_tool0_fingertip) @ frames[tname]
        _hx, _hr = matrix_to_xyzrpy(T_ftip_conn)
        log.info('START HELD: the connector starts in the hand; belief = frame %r: %s mm / %s deg '
                 'in the fingertip frame. No pick.', tname, np.round(_hx * 1000.0, 2).tolist(),
                 np.round(np.degrees(_hr), 2).tolist())

    live = a.get('live_plot', True)
    live_path = None
    if live:
        live_path = live if isinstance(live, str) else os.path.join(
            cfg.get('data_dir', 'data'), 'experiments', f'{task.name}_live.png')
        os.makedirs(os.path.dirname(live_path) or '.', exist_ok=True)
    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'{task.name}_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Experiment folder: %s', out_dir)
    # TRIAL DATA (robot.record_trial): the console (run.log) and the arm at 100 Hz into this folder -- the motion
    # until the first trial starts, then into each trial's own folder (TrialRecorder.switch).
    _trial = None
    if cfg.get_path('robot.record_trial', False):
        import atexit
        from ..robot.trial_recorder import TrialRecorder
        _trial = TrialRecorder(robot, out_dir, float(cfg.get_path('robot.record_trial_hz', 100.0) or 100.0),
                               float(cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0).start()
        atexit.register(_trial.stop)

    def end_reset_with_snapshot(label='end reset'):
        """The end-of-run bookend: home (the marker VIEW pose) + one image from that view,
        saved to the experiment folder. The capture is best-effort -- it never fails a run
        that already finished."""
        ok = reset.reset_robot(robot, cfg, label)
        if ok and camera is not None and not robot.arm.dry_run                 and bool(cfg.get('end_view_image', True)):
            try:
                import cv2
                frame = camera.capture()
                path = os.path.join(out_dir, 'end_view.jpg')
                cv2.imwrite(path, frame.color)
                log.info('End-of-run image from the home view: %s', path)
            except Exception as exc:           # noqa: BLE001 -- never fail a finished run
                log.warning('end-of-run image skipped (%s)', exc)
        return ok

    no_prompts = prompts_off(cfg)
    if no_prompts:
        log.warning('skip_prompts: running with NO confirmations -- the reset, the pre-contact '
                    'stand-off and the success calls are all skipped (success falls back to '
                    'the tolerance check). Only the cable labelling still asks.')
    # BEHAVIOUR GATES ARE UNCONDITIONAL, and deliberately do NOT read confirm_each_step / --yes.
    # --yes means "stop asking me to confirm each little step"; it does not mean "drive the part
    # into the socket without telling me". Every phase_gate here sits at a BEHAVIOUR boundary --
    # contact, engage, clocking, the collar, the tug, the escape -- and each is a point where an
    # operator wants to look at what just happened before the next thing starts. Silencing those
    # with a convenience flag is how a prompt that was asked for never appears (which happened:
    # the failed-engage prompt was invisible under --yes).
    #
    # The escapes are `--no-prompts` for genuinely unattended runs, and
    # assembly.gate_between_behaviors: false to turn them off in config. Dry runs never prompt.
    gates_on = bool(a.get('gate_between_behaviors', True)) and not no_prompts

    def phase_gate(name, ahead):
        """Continue/abort between two BEHAVIOURS. True = go on.

        Aborting LEAVES THE ARM WHERE IT IS rather than escaping automatically: the right recovery
        depends on how far the mate went and whether it will release, which this cannot judge. EOF
        continues, so a headless run never hangs; dry runs skip the prompt."""
        if robot.arm.dry_run or not gates_on:
            return True
        ans = ask(f'\n[{name}] {ahead}\n    Enter to continue (q to abort): ')
        if ans is None:
            return True
        if ans.strip().lower() in ('q', 'quit', 'n', 'no'):
            log.warning('ABORTED before %s by the user. The arm is LEFT WHERE IT IS -- the '
                        'connector may still be engaged and held. Free it by hand (release the '
                        'gripper / press the unlock button) before commanding any motion.', name)
            return False
        _passed('GATE', 'the operator pressed Enter', name)
        return True

    def retract_from(last_ref, T_tool0_conn, adm_ctl=None):
        """The compliant UN-GUARDED escape along the believed connector's own -X.

        Un-guarded on purpose: a seated or jammed connector is already over the force limit, so a
        guarded retract would block the very motion that frees it (see ForceGuard.disable)."""
        ctl = adm_ctl if adm_ctl is not None else adm
        T_out = _retract_ref(last_ref, T_tool0_conn, retract_m)
        ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret, g_w * s_ret), guard=None)
        robot.arm.servo_stop()
        return T_out

    def run_insertion(adm_ctl, refs, T_tool0_conn, peck=False, guard_ctl=None, settle=None,
                      hold=None, speed=None, pause=None, retract=True):
        """One admittance insertion along refs, collecting observations. Mirrors estimator_eval's
        run_insertion, including peck (a force stop backs off and advances again rather than
        ending the pass) and the un-guarded post-insertion dwell.

        retract=False LEAVES THE ARM AT THE STOP. The insertion meant to SEAT must stay put: a
        retract here pulls the connector back out with it (the gripper holds the cable), and
        connector clocking would then read the retracted pose as its "engaged pose" and anchor the
        whole clocking sequence there. Only an attempt about to be RETRIED should back off, and
        that is the caller's decision (see retract_from).

        Returns (obs, lin, ang, stops, last_ref); `last_ref` is what retract_from needs later."""
        guard = guard_ctl if guard_ctl is not None else guard_shared
        settle_s = settle if settle is not None else settle_shared
        hold_s = hold if hold is not None else hold_shared
        sv, sw = speed if speed is not None else (None, None)
        obs, cnt = [], [0]

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim == 0:
                obs.append(_observe(robot, T_tool0_conn, T_base_tconn))

        adm_ctl.reset()
        adm_ctl.warmup(refs[0], tare_fn=tare)
        guard.reset()
        last_ref, i, t0, stops = refs[0], 1, robot.arm.clock(), []
        prev = refs[0]
        while i < len(refs):
            res = adm_ctl.ramp(prev, refs[i], seg_time(prev, refs[i], sv, sw), guard,
                               on_step=log_cb)
            last_ref = refs[i]
            if res == 'seated':
                stops.append(float(matrix_to_xyzrpy(
                    inverse(T_base_tconn) @ robot.tool0() @ T_tool0_conn)[0][0] * 1000.0))
                if not peck or robot.arm.clock() - t0 > peck_timeout:
                    break
                # back off along the BELIEVED connector's -X, un-guarded (we are at the limit
                # by construction), then keep advancing from where we are.
                T_out = _retract_ref(last_ref, T_tool0_conn, peck_back_m)
                adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret,
                                                       g_w * s_ret), guard=None,
                             on_step=log_cb)
                guard.reset()
                prev = T_out
                i += 1
                continue
            prev = refs[i]
            i += 1
        adm_ctl.hold(last_ref, settle_s, guard, on_step=log_cb)
        if hold_s > 0:                                 # dwell: un-guarded, unlogged
            adm_ctl.hold(last_ref, hold_s, guard=None)
        if pause:
            # Servo RELEASED, so the arm applies no force -- inspection time, not press time.
            adm_ctl.stop()
            log.info('   pausing %.1f s at the seat (servo stopped).', float(pause))
            robot.arm.sleep(float(pause))
        lin, ang = pose_error(robot.tool0() @ T_tool0_conn, T_base_tconn)
        if retract:
            retract_from(last_ref, T_tool0_conn, adm_ctl)
        else:
            # servo_stop only ends the servoL stream; the controller holds the last commanded
            # pose, so the connector keeps its seat.
            robot.arm.servo_stop()
            log.info('   holding the seat (no retract) -- the pose is the engaged pose.')
        return obs, lin, ang, stops, last_ref

    # ====================================================================================
    # POST-MATE CLOCKING. Both maneuvers run only after a successful mate and share one
    # escape. Diagnostics land in clocking.csv rather than estimates.csv, which is already
    # closed by the time these run.
    # ====================================================================================
    clock_rows = []

    def axis_offset_base():
        """collar_clocking.axis_offset_mm as a BASE-frame vector.

        THE OFFSET BELONGS TO THE FIXTURE, NOT THE PLUG'S ROLL -- it is the bench-measured shift
        from the DECLARED socket axis to the barrel centreline the collar turns about, so it is a
        fixed line in space however far round the plug is mated.

        `engage_clock_deg` rolls T_clk about its own +X, so T_clk's Y and Z are no longer the axes
        the number was measured in. Resolving it there would swing a measured -Z correction round
        to -Y at a 90 deg clock angle. Hence the roll-FREE basis. With engage_clock_deg 0 this is
        exactly T_clk's own basis."""
        return (T_clk[:3, :3] @ R_clock[:3, :3].T) @ (np.asarray(cl_axis_off, dtype=float)
                                                      / 1000.0)

    def clocking_retract(label='clocking retract', gripper_leg=True):
        """The post-clocking escape, in two legs.

        First along the GRIPPER's own axis, lifting the open fingers off the connector; then
        along the TARGET frame's axis, backing the arm away from the socket. Guarded straight
        lines, not the compliant `_retract_ref` used between attempts: the part is released by
        now, so there is no held connector to thread back out along its own axis.

        `gripper_leg=False` drops the first leg and backs straight out along the target -X.
        That is the right shape after the COLLAR turn, because there tool0 sits ON the
        connector axis with its Z collinear -- so the gripper -Z leg points essentially where
        the target -X leg already goes, and running both just adds a second, differently
        parametrised move for the same escape."""
        r = a.get('clocking_retract', {}) or {}

        def leg(vec, dist, in_target):
            v = np.asarray(vec, dtype=float)
            n = float(np.linalg.norm(v))
            if n < 1e-9 or abs(float(dist)) < 1e-9:
                return True
            step = v / n * abs(float(dist))
            if in_target:
                # a direction in the TARGET frame -> rotate it into base and left-multiply
                T = translation_matrix(T_clk[:3, :3] @ step) @ robot.tool0()
                what = f'target {np.round(v / n, 3).tolist()}'
            else:
                # a direction in the GRIPPER's own (tool0) frame -> right-multiply
                T = robot.tool0() @ translation_matrix(step)
                what = f'gripper {np.round(v / n, 3).tolist()}'
            return _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                T, label=f'{label} ({what}, {abs(float(dist)) * 1000.0:.0f} mm)'))

        phase('clock_retract')
        if not gripper_leg:
            log.info('  escape: target -X only (tool0 is already on the connector axis, so the '
                     'gripper -Z leg would repeat it).')
        return ((gripper_leg is False
                 or leg(r.get('gripper_axis', [0.0, 0.0, -1.0]),
                        r.get('gripper_distance_m', 0.100), False))
                and leg(r.get('target_axis', [-1.0, 0.0, 0.0]),
                        r.get('target_distance_m', 0.100), True))

    def tug_verify_in_place(hold_after=False, keep_on_fail=False):
        """TUG VERIFICATION, from wherever the collar turn ended, WITHOUT letting go.

        collar_clocking returns with the fingers still closed on the locked collar. That is
        already a grip on the assembly, and already on the connector axis -- so the pull happens
        here rather than after a release, a retract, a drive back to the historical engaged pose
        and a blind re-grasp. Each of those was a chance to disturb the very thing being measured,
        and the re-grasp could miss the connector outright.

        A LOCKED bayonet holds the pull; an unlocked one backs out and the displacement says so.
        The pull is a SPRING pull, not a position ramp: the reference is offset along the
        connector -X by pull_force / stiffness, so the force is applied at zero displacement and
        DROPS as the connector comes out -- it can never exceed pull_force_n on one that holds.

        Returns 'verified' (held; released and retracted), 'failed' (backed out; the cable has
        been EXTRACTED, carried home and released), 'terminated' (the GLOBAL guard tripped
        mid-pull -- stop where we are), or 'error'."""
        axn_t = T_clk[:3, 0] / float(np.linalg.norm(T_clk[:3, 0]))
        # NO re-approach and NO re-grip: pull from the pose the turn left, on the collar it is
        # already holding. The axis is still the socket's, so the pull direction is unchanged.
        T_grasp = robot.tool0()
        phase('assemble')
        # THE PULL. Effective axial stiffness of the (diagonal, tool0-frame) spring along the
        # base-frame pull direction: compliances add, 1/S_eff = sum(u_i^2 / S_i).
        u = T_grasp[:3, :3].T @ axn_t
        S_eff = 1.0 / float(np.sum((u ** 2) / adm_tug.S[:3]))
        if tv_force / S_eff <= tv_thresh_m:
            log.warning('TUG VERIFY: the spring offset (%.1f mm at %.0f N/m) does not exceed '
                        'the %.1f mm threshold -- an unlocked connector cannot move past it, so '
                        'this verification cannot fail. Soften tug_verify.stiffness or raise '
                        'pull_force_n.', tv_force / S_eff * 1000.0, S_eff, tv_thresh_m * 1000.0)
        T_pull = translation_matrix(-(tv_force / S_eff) * axn_t) @ T_grasp
        log.info('TUG VERIFY: pulling %.1f N along the connector -X for %.1f s '
                 '(spring %.0f N/m -> %.1f mm reference offset); verified if the connector '
                 'moves <= %.1f mm.', tv_force, tv_time, S_eff, tv_force / S_eff * 1000.0,
                 tv_thresh_m * 1000.0)
        adm_tug.reset()
        adm_tug.warmup(T_grasp, tare_fn=tare)       # tare while gripping and static
        guard_shared.reset()
        phase('assemble')
        res = adm_tug.ramp(T_grasp, T_pull, seg_time(T_grasp, T_pull), guard_shared)
        if res != 'seated':
            res = adm_tug.hold(T_pull, tv_time, guard_shared)
        disp = float(np.dot(T_grasp[:3, 3] - robot.tool0()[:3, 3], axn_t))
        adm_tug.reset()
        adm_tug.stop()
        robot.arm.servo_stop()
        if res == 'seated':
            log.error('TUG VERIFY: the GLOBAL force guard tripped during the pull (%s) -- '
                      'terminating the script where it stands.', guard_shared.tripped_by)
            return 'terminated'
        held = disp <= tv_thresh_m
        clock_rows.append({'maneuver': 'tug_verify', 'try': 1, 'ramp_result': res,
                           'advance_mm': round(-disp * 1000.0, 3),
                           'need_mm': round(tv_thresh_m * 1000.0, 3),
                           'success': bool(held), 'force_stop': False,
                           'state_after': 'locked' if held else 'extracted',
                           'stopped_by': ''})
        if held:
            log.info('TUG VERIFIED -- %.1f N for %.1f s moved the connector %.2f mm '
                     '(<= %.1f mm): the %s holds.', tv_force, tv_time, disp * 1000.0,
                     tv_thresh_m * 1000.0, 'latch' if task.latch else 'lock')
            _passed('TUG VERIFY', 'a %.1f N pull for %.1f s moved the connector %.2f mm <= '
                    '%.1f mm' % (tv_force, tv_time, disp * 1000.0, tv_thresh_m * 1000.0),
                    'release, then the escape' if task.latch else
                    'DISASSEMBLY (grip held)' if hold_after else 'release and retract')
            if hold_after:
                # DISASSEMBLY IS NEXT and it starts from exactly this state: fingers closed on
                # the collar, tool0 on the connector axis. Releasing and retracting here only
                # to re-approach and re-grip would add the three free-space moves and the blind
                # re-grasp that doing the tug in place exists to avoid.
                log.info('  holding the grip -- disassembly starts from here.')
                return 'verified_held'
            if not robot.gripper.open('release (tug verified)'):
                log.error('TUG VERIFY: gripper did not release after the tug.')
                return 'error'
            # Straight out along the target -X when disassembly follows: it re-approaches
            # the collar on the axis anyway, so lifting off the gripper -Z first only adds a
            # move in the direction the next leg already travels.
            return ('verified' if clocking_retract('tug retract', gripper_leg=not dis_on)
                    else 'error')
        # FAILED: never locked, and already part-way out in the fingers.
        if keep_on_fail:
            # THE TREE TAKES IT FROM HERE (the user, 2026-11-06: "in order to [repeat] the robot must disassemble and
            # place after assembly"): the grip kept, the arm where the pull left it -- the escape, then the
            # disassembly (unlock, extract, PLACE where the next pick finds it) follow, as after a good assembly.
            # The old way below dropped it: reset_robot opens the gripper before it moves, so "carried home" was
            # let go of at the socket (03:08).
            log.error('TUG VERIFY FAILED -- the connector backed out %.2f mm (> %.1f mm) under a %.1f N pull: NOT '
                      '%s. Grip kept: %s.', disp * 1000.0, tv_thresh_m * 1000.0, tv_force,
                      'latched' if task.latch else 'locked',
                      'a miss -- backed out and laid down by the recovery' if task.latch else
                      'the escape and the disassembly put the cable back')
            return 'failed_held'
        # The old standalone handling: EXTRACT it fully along the connector -X, carry it home and release. The
        # global guard stays armed, so an extraction that snags terminates rather than tearing at the fixture.
        log.error('TUG VERIFY FAILED -- the connector backed out %.2f mm (> %.1f mm) under a '
                  '%.1f N pull: NOT locked. Extracting the cable.',
                  disp * 1000.0, tv_thresh_m * 1000.0, tv_force)
        phase('retract')
        T_now = robot.tool0()
        T_out = translation_matrix(-tv_extract_m * axn_t) @ T_now
        adm_tug.reset()
        adm_tug.warmup(T_now)
        guard_shared.reset()
        res2 = adm_tug.ramp(T_now, T_out, seg_time(T_now, T_out), guard_shared)
        adm_tug.reset()
        adm_tug.stop()
        robot.arm.servo_stop()
        if res2 == 'seated':
            log.error('TUG: the GLOBAL force guard tripped during the extraction (%s) -- '
                      'terminating the script with the cable still held.',
                      guard_shared.tripped_by)
            return 'terminated'
        phase('reset')
        if not reset.reset_robot(robot, cfg, 'tug failure (carry the cable home)'):
            log.error('TUG: could not go home with the extracted cable -- terminating.')
            return 'terminated'
        robot.gripper.open('release (failed cable, at home)')
        return 'failed'


    def retract_along_target(last_ref, distance_m):
        """Pull straight out along the TARGET CONNECTOR's -X, compliantly and un-guarded.

        NOT the believed connector's -X, which is what retract_from uses. The difference matters
        precisely here: a missed engagement is EVIDENCE THAT THE BELIEF IS WRONG, and retracting
        along a wrong axis pulls at an angle -- which is how a recovery levers the socket off its
        fixture instead of freeing the part. The socket's own axis is the one direction that is
        known good, because it is what the trajectory was recorded against.

        A pure BASE-frame translation: the attitude is left exactly as it is, so nothing rotates
        while the part is still inside anything."""
        axis = np.asarray(T_base_tconn[:3, 0], dtype=float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        T_out = translation_matrix(-float(distance_m) * axis) @ np.asarray(last_ref, dtype=float)
        adm.ramp(last_ref, T_out,
                 seg_time(last_ref, T_out, g_v * s_ret, g_w * s_ret), guard=None)
        robot.arm.servo_stop()
        return T_out

    def place_after_failed_engage(miss_ref, T_tool0_conn_held, why=''):
        """After an insertion that did not mate (or a connector clocking that did not seat, re-gripped): back
        out, lay the cable down, and leave the arm at the pick pose so the trial can start again from the view.

        WHY A NO-CONTACT ENGAGE IS A FAILURE. The engage drives the recorded path and stops on the
        axial force limit -- meeting the socket IS the success condition. Running the entire path,
        preload included, without ever reaching that limit means the connector never touched
        anything: it went past the socket, or beside it. Continuing to the bayonet search from
        there would clock a connector that is not in a socket.

        AND WHY THE RECOVERY IS A FULL RE-PICK, NOT A RE-TRY FROM HERE. The most likely causes are
        a wrong target pose or a wrong in-hand belief, and neither is improved by pushing again
        from the same place with the same numbers. Putting the cable down and starting over
        re-localizes the target, resets the belief to nominal, re-detects the cable and re-grasps
        it -- so every input to the engagement is measured again.

        THE QUESTION BEFORE IT MOVES is the campaign's now (behaviors/campaign.py: every recovery is asked
        about first unless --no-prompts, the same terms this asked on), with the cell state in it."""
        if why:
            log.warning('BACK OUT AND PLACE, after: %s', why)

        # ---- RETRACT FIRST, AND PROVE IT ----------------------------------------------------
        # THE ARM IS ABOUT TO TRAVEL SIDEWAYS. If the connector is still in or against the socket
        # when that happens it levers the socket off its fixture -- the recovery breaks the very
        # thing the run exists to mate into. So the retraction is MANDATORY here, not best-effort:
        # a failure to back out ABORTS the recovery and leaves the arm where it is, because
        # stopping with the part held is recoverable by hand and a snapped socket is not.
        #
        # And it is VERIFIED, not assumed. retract_from drives a compliant REFERENCE; a connector
        # that is jammed can leave the reference free while the part has not moved at all. The
        # measured depth before and after is the only thing that says it actually came out.
        def conn_depth_mm():
            return float((inverse(T_base_tconn)
                          @ (robot.tool0() @ T_tool0_conn_held))[0, 3]) * 1000.0

        depth_before = conn_depth_mm()
        phase('retract')
        log.info('FAILED ENGAGE: backing out %.0f mm along the TARGET connector -X BEFORE '
                 'travelling (depth now %+.2f mm).', fail_retract_m * 1000.0, depth_before)
        try:
            retract_along_target(miss_ref, fail_retract_m)
        except Exception as exc:                       # noqa: BLE001
            log.error('RETRACT FAILED after the missed engage (%s: %s). REFUSING to travel to the '
                      'place with the connector still in the socket -- that would break it. The '
                      'arm is LEFT WHERE IT IS with the cable held; free it by hand.',
                      type(exc).__name__, exc)
            return False

        backed_out_mm = depth_before - conn_depth_mm()
        if backed_out_mm < fail_retract_min_m * 1000.0:
            log.error('RETRACT DID NOT CLEAR: the connector came out only %.2f mm of the %.0f mm '
                      'commanded (needed at least %.0f mm) -- it is most likely still engaged or '
                      'jammed. REFUSING to travel to the place, which would lever the socket off '
                      'its fixture. The arm is LEFT WHERE IT IS with the cable held.',
                      backed_out_mm, fail_retract_m * 1000.0, fail_retract_min_m * 1000.0)
            return False
        ok = True
        log.info('  retracted %.2f mm -- clear of the socket, travelling to the place.',
                 backed_out_mm)

        T_place = aligned_place_pose(dis_clear_m)
        T_ftip_place = T_place @ inverse(_upright(T_ftip_conn))
        up = translation_matrix([0.0, 0.0, float(dis_rise_m)])
        log.info('FAILED ENGAGE: laying the cable down at xyz %s mm, then back to the pick pose.',
                 np.round(T_place[:3, 3] * 1000.0, 0).tolist())
        phase('reconfigure')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(up @ T_ftip_place, 'failed-engage place '
                                                                       '(above)')):
            log.error('Could not reach the place stand-off after a failed engage.')
            return False
        phase('lift')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(T_ftip_place, 'failed-engage place (down)')):
            log.error('Could not lower to the place pose after a failed engage.')
            return False
        robot.gripper.open('release (failed engage)')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(up @ T_ftip_place, 'failed-engage place '
                                                                       '(rise)')):
            return False
        phase('reconfigure')
        if not robot.arm.move_j(q_pick, label='pick pose (after failed engage)'):
            log.error('Could not return to the pick pose after a failed engage.')
            return False
        return ok

    def celebrate(state, cycle, tug_res, ret_ok):
        """CELEBRATE -- a short flourish after a VERIFIED assembly. Cosmetic, and built so it
        cannot undo the work it is celebrating.

        EVERY GATE LIVES HERE, not at the call site, so there is one place that decides whether a
        run earned a celebration and one place to read when it did not happen.

        WHY IT IS WRIST-ONLY. It runs with the gripper open, near a fixture the arm has just spent
        a minute carefully not hitting, so the design constraint is that it must be incapable of
        translating the tool into anything. wrist_3 rotates about tool0's OWN axis, so it moves the
        tool0 ORIGIN by exactly zero -- that is the same fact the collar prewind rests on. wrist_1
        does move the tool, so it is small, taken after a rise into open air, and COLLISION-CHECKED
        like any other joint move rather than assumed safe (a moveJ interpolates in joint space and
        can swing the tool through the bench between two clear endpoints).

        IT ONLY RUNS ON SUCCESS. Celebrating a failed mate is worse than not celebrating: it wastes
        the diagnostic value of where the arm stopped, and it moves the evidence. The expected
        terminal state depends on what was enabled -- 'locked' with collar clocking on, 'seated'
        with only connector clocking, 'engaged' with neither -- so the bar is "the run reached the
        end it was configured to reach", not a hardcoded state.

        Off by default. Nobody wants this firing 50 times during a data-collection run."""
        cb = a.get('celebrate', {}) or {}
        if not bool(cb.get('enabled', False)):
            return True

        # ---- 1. DID THIS CYCLE ACTUALLY SUCCEED? ------------------------------------------
        # The same bar the run itself uses for ok_cycle. Note 'terminated' is NOT success: it
        # means the tug verification aborted part-way, so nothing was verified -- an earlier
        # version of this gate tested `!= 'failed'` and let that through.
        if not ret_ok:
            log.info('CELEBRATE skipped: the escape did not complete, so the arm is not in a '
                     'known clear state.')
            return True
        if tug_res not in (None, 'skipped', 'verified'):
            log.info('CELEBRATE skipped: tug verification returned %r -- the assembly is NOT '
                     'verified, so there is nothing to celebrate.', tug_res)
            return True
        want = 'locked' if cl_on else ('seated' if cc_on else 'engaged')
        if state != want:
            log.info('CELEBRATE skipped: the run ended %r, not the configured %r.', state, want)
            return True

        # ---- 2. IS THIS ONE OF THE CYCLES TO CELEBRATE? -----------------------------------
        # `last` is the useful one for a long run: a 50-cycle collection should not stop to dance
        # 50 times, but finishing the set is worth marking. It is safe against a mid-run failure
        # for free -- a failed cycle stops the loop, so the final cycle is never reached.
        when = str(cb.get('when', 'last')).strip().lower()
        every_n = max(1, int(cb.get('every_n', 1)))
        if when in ('last', 'end'):
            due = (cycle >= n_cycles)
        elif when in ('every', 'each', 'cycle'):
            due = True
        elif when in ('every_n', 'nth'):
            due = (cycle % every_n == 0)
        else:
            log.warning("CELEBRATE: assembly.celebrate.when=%r is not one of "
                        "last | every | every_n -- treating it as 'last'.", when)
            due = (cycle >= n_cycles)
        if not due:
            log.info('CELEBRATE skipped: cycle %d/%d, when=%r%s.', cycle, n_cycles, when,
                     f' (every {every_n})' if when in ('every_n', 'nth') else '')
            return True

        rise_m = _num(cb, 'rise_mm', 100.0) / 1000.0
        nod_deg = _num(cb, 'nod_deg', 15.0)
        spin_deg = _num(cb, 'spin_deg', 180.0)
        reps = max(1, int(cb.get('repeats', 2)))
        flourish = bool(cb.get('gripper_flourish', True))

        log.info('--- CELEBRATE --- %s: rise %.0f mm, nod %+.0f deg (wrist_1), spin %+.0f deg '
                 '(wrist_3), x%d.', 'assembly verified', rise_m * 1000, nod_deg, spin_deg, reps)
        phase('retract')

        if rise_m > 0.0:
            T_up = translation_matrix([0.0, 0.0, rise_m]) @ robot.tool0()
            if not _guarded(robot, guard_shared,
                            lambda: robot.arm.move_l(T_up, label='celebrate (rise)')):
                log.warning('CELEBRATE: could not rise -- skipping the rest. The assembly is '
                            'unaffected.')
                return True

        model = grasp.collision_model()
        q0 = list(robot.arm.q())

        def wiggle_joint(index, amplitude_deg, label):
            """Rock one joint +amp, -amp, back to centre. Refuses on a collision or a joint limit
            rather than clipping -- a flourish is never worth a forced move."""
            for target in (+amplitude_deg, -amplitude_deg, 0.0):
                q = list(q0)
                q[index] = q0[index] + np.radians(target)
                if not robot.arm.joints_ok(q):
                    log.warning('CELEBRATE: %s %+.0f deg is outside the joint limits -- stopping '
                                'the flourish here.', label, target)
                    return False
                if model is not None:
                    ok, body, _over, _f = model.check_path(list(robot.arm.q()), q)
                    if not ok:
                        log.warning('CELEBRATE: %s %+.0f deg would put %s through the ground -- '
                                    'stopping the flourish here.', label, target, body)
                        return False
                if not _guarded(robot, guard_shared,
                                lambda _q=q: robot.arm.move_j(_q, label=f'celebrate ({label})')):
                    return False
            return True

        for _ in range(reps):
            if not wiggle_joint(robot.arm.joint_index('wrist_nod'), nod_deg, 'nod'):
                break
            if not wiggle_joint(robot.arm.joint_index('tool_roll'), spin_deg, 'spin'):
                break
            if flourish:
                robot.gripper.close('celebrate (clap)')
                robot.gripper.open('celebrate (clap)')
        log.info('--- CELEBRATE ENDED --- the arm is clear and the assembly is untouched.')
        return True

    def disassembly(state, screw_deg, T_base_conn_d):
        """DISASSEMBLY -- unwind the clocking, pull the connector out, put the cable down.

        THE STATE LADDER RUN BACKWARDS. Assembly walks engaged -> seated -> locked; this walks
        locked -> seated -> engaged -> removed, each rung claimed only by the step that undoes
        it (_retreat_state asserts the rung we are actually on).

        ONE MOTION DROPS TWO RUNGS, and that is a fact about the grasp rather than a shortcut:
        the axial grip closes on the collar AND the connector body together, so the reverse
        turn backs the collar off its lock and carries the bayonet round to its slots in the
        same stroke. Assembly needs two maneuvers there because it grips differently for each
        (the bayonet at the junction, the collar on the ring); disassembly does not.

        WHERE IT STARTS, and why that matters. It runs straight after the tug, which leaves the
        fingers CLOSED on the collar with tool0 ON the connector axis -- the same reason the tug
        itself happens in place. Every rotation below is therefore a wrist twist about the axis
        the arm is already on: no re-approach, no re-grip, no chance to lose the part between
        steps.

        THE GEOMETRY. Both rotations are about the SAME LINE the assembly turned about: the
        socket frame's own +X through its origin (T_clk), which is what connector_clocking
        screwed about and -- with collar_clocking.axis_offset_mm at zero -- what the collar
        turned about too. They are run through screw_ramp for the same reason the assembly is:
        a rotation about a line 178 mm from tool0 bows into an arc that a straight ramp would
        cut across.

        THE DIRECTIONS, and this is the part that is easy to get backwards:
          * the collar was locked by turning +rotation_deg, so the reverse turn is
            -rotation_deg -- or until the torque limit, since the ring runs to a stop at both
            ends of its travel and torque building is the intended end in either direction;
          * that same turn carries the bayonet round with it, because the jaws hold the
            connector body and the collar together -- so there is no separate angle to undo
            and nothing to aim at the mated roll;
          * the pull is along the socket -X, the exact reverse of the insertion axis.

        Returns (ok, state)."""
        axn_d = T_clk[:3, 0] / float(np.linalg.norm(T_clk[:3, 0]))
        if float(np.linalg.norm(np.asarray(cl_axis_off, dtype=float))) > 0.0:
            log.warning('DISASSEMBLY: collar_clocking.axis_offset_mm is non-zero, but the '
                        'unwind turns about the SOCKET axis. Re-check the collar stays on the '
                        'ring through the unlock.')

        # ---- 1. LOCKED/SEATED -> ENGAGED: ONE reverse turn undoes both --------------------
        # THE AXIAL GRASP HOLDS BOTH. Closing on the ring closes on the connector body with
        # it, so the single reverse turn backs the collar off its lock AND carries the bayonet
        # round to where its pins line up with the slots -- there is no second, separate
        # connector rotation to make, and making one would turn a part that is already free.
        # That is why this drops two rungs of the ladder in one motion.
        #
        # The assembly RELEASED and retracted before this, so there is no grip to inherit: the
        # turn drives the SAME approach the lock did (retract along the cable, reorient onto
        # the axis, advance down it, close on the ring) with the rotation negated and the seat
        # push off. Sharing that geometry is what keeps the unlock landing on the ring the lock
        # turned, instead of on a station computed a second, divergent way.
        if dis_unlock and state in ('locked', 'seated'):
            if not phase_gate('UNLOCK',
                              'Re-approach on the axis, close on the collar (which grips the '
                              'connector body with it), and turn %.0f deg BACK -- or until the '
                              'torque limit. That releases the lock and the bayonet together.'
                              % np.degrees(cl_rot)):
                return False, state
            if not collar_clocking(T_base_conn_d, screw_deg, cl_rot=-cl_rot, sp_on=False,
                                   unlocking=True):
                log.error('DISASSEMBLY: the reverse turn did not complete -- stopping with the '
                          'connector still in the socket.')
                return False, state
            if state == 'locked':
                state = _retreat_state(state, 'locked')      # -> seated (the lock is off)
            state = _retreat_state(state, 'seated')          # -> engaged (the bayonet is free)
            log.info('  the reverse turn released the collar AND the bayonet -- the connector '
                     'is ENGAGED only, and free to pull.')
        elif state in ('locked', 'seated'):
            log.warning('DISASSEMBLY: unlock_collar is off, so nothing has released the '
                        'bayonet -- the pull below will be against a %s connector.', state)

        # THE CONNECTOR ON THE TOOL for the extract and the place (the live view draws it, the move
        # check carries it): still mated here, so it is AT T_base_conn_d -- exact, not a belief.
        if hasattr(robot, 'hold') and getattr(robot, 'held', None) is None:
            held_d = held_connector(inverse(robot.tool0() @ robot.T_tool0_fingertip) @ T_base_conn_d)
            if held_d is not None:
                robot.hold(held_d)

        # ---- 2. ENGAGED -> REMOVED: pull it straight out ----------------------------------
        if not phase_gate('EXTRACT',
                          'Pull the connector %.0f mm straight out along the socket -X. The '
                          'bayonet should be free; the global guard stops a pull that snags.'
                          % (dis_extract_m * 1000.0)):
            return False, state
        phase('retract')
        T_now = robot.tool0()
        T_out = translation_matrix(-dis_extract_m * axn_d) @ T_now
        adm_cl.reset()
        adm_cl.warmup(T_now)
        guard_shared.reset()
        res = adm_cl.ramp(T_now, T_out, seg_time(T_now, T_out), guard_shared)
        pulled = float(np.dot(T_now[:3, 3] - robot.tool0()[:3, 3], axn_d))
        adm_cl.reset()
        adm_cl.stop()
        robot.arm.servo_stop()
        clock_rows.append({'maneuver': 'extract', 'try': 1, 'ramp_result': res,
                           'advance_mm': round(-pulled * 1000.0, 3),
                           'need_mm': round(dis_extract_m * 1000.0, 3),
                           'success': res != 'seated', 'force_stop': res == 'seated',
                           'state_after': 'removed' if res != 'seated' else 'engaged',
                           'stopped_by': guard_shared.tripped_by or ''})
        if res == 'seated':
            log.error('EXTRACT: the global guard tripped after %.1f mm (%s) -- the connector '
                      'is still in the socket and still held. Free it by hand.',
                      pulled * 1000.0, guard_shared.tripped_by)
            return False, state
        log.info('EXTRACTED %.1f mm along the socket -X -- the connector is out and in the '
                 'fingers.', pulled * 1000.0)
        _passed('EXTRACT', 'pulled %.1f mm of the %.1f mm along the socket -X without the global '
                'guard tripping' % (pulled * 1000.0, dis_extract_m * 1000.0),
                'PLACE THE CABLE' if dis_place_on else 'done (cable held)')
        state = _retreat_state(state, 'engaged')

        # ---- 3. put the cable down --------------------------------------------------------
        if not dis_place_on:
            log.info('DISASSEMBLY: place is off -- the cable stays in the fingers.')
            return True, state
        # SCATTER THE PLACE, so each cycle re-picks from a genuinely different pose. Drawn with a
        # REACHABILITY RETRY: an unlucky draw that lands outside the workspace would otherwise kill
        # an unattended 50-cycle run, and falling back to the centre is always safe because that is
        # the pose the run would have used anyway.
        T_place = aligned_place_pose(dis_clear_m)
        for _try in range(8):
            s = place_scatter()
            if s == (0.0, 0.0, 0.0):
                break                                   # scatter off: the centre IS the pose
            T_try = aligned_place_pose(dis_clear_m, scatter=s)
            if robot.arm.dry_run or robot.arm.ik(T_try @ inverse(T_ftip_conn)) is not None:
                T_place = T_try
                log.info('PLACE SCATTER: %+.1f, %+.1f mm and %+.1f deg about the configured place '
                         '(draw %d/8).', s[0] * 1000, s[1] * 1000, np.degrees(s[2]), _try + 1)
                break
            log.info('  place scatter draw %d is unreachable -- redrawing.', _try + 1)
        else:
            log.warning('PLACE SCATTER: no reachable draw in 8 tries -- placing at the '
                        'un-scattered centre instead. Reduce assembly.disassembly.place_scatter.')
        if not phase_gate('PLACE THE CABLE',
                          'Lay the cable down ALONG THE SOCKET AXIS at xyz %s mm, %.0f mm '
                          'clear of the ground, release and rise.'
                          % (np.round(T_place[:3, 3] * 1000.0, 0).tolist(),
                             dis_clear_m * 1000.0)):
            return False, state
        # AIMED, NOT INHERITED. The old version drove to the pick pose and descended, which
        # laid the part down in whatever attitude the grip happened to have -- and with the
        # COAXIAL grip the connector hangs axis-down, so it was set on its end. The pose is
        # built from the socket heading instead, and the arm is commanded to put the CONNECTOR
        # there via the belief it is currently holding it with.
        T_ftip_place = T_place @ inverse(T_ftip_conn)
        up = translation_matrix([0.0, 0.0, float(dis_rise_m)])
        log.info('PLACE: laying the connector along the socket axis -- heading %+.1f deg '
                 '(socket %+.1f deg), %.0f mm above the ground plane.',
                 np.degrees(np.arctan2(T_place[1, 0], T_place[0, 0])),
                 np.degrees(np.arctan2(T_base_tconn[1, 0], T_base_tconn[0, 0])),
                 (T_place[2, 3] - ground_z) * 1000.0)
        phase('reconfigure')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(up @ T_ftip_place, 'place (above)')):
            log.error('DISASSEMBLY: could not reach the place stand-off.')
            return False, state
        phase('lift')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(T_ftip_place, 'place (down)')):
            log.error('DISASSEMBLY: the descent hit something before the release height.')
            return False, state
        if not robot.gripper.open('release (cable placed)'):
            log.error('DISASSEMBLY: the gripper did not open to release the cable.')
            return False, state
        T_up = translation_matrix([0.0, 0.0, dis_rise_m]) @ robot.tool0()
        if not _guarded(robot, guard_shared,
                        lambda: robot.arm.move_l(T_up, label='rise clear of the cable')):
            log.error('DISASSEMBLY: could not rise after releasing.')
            return False, state
        log.info('CABLE PLACED %.0f mm above the ground plane and released.',
                 dis_clear_m * 1000.0)
        _passed('PLACE', 'reached the place %.0f mm above the ground with no guard trip, gripper '
                'opened, rose %.0f mm' % (dis_clear_m * 1000.0, dis_rise_m * 1000.0), 'end reset')
        return True, state

    def rl_insertion():
        """THE TRAINED POLICY, from above the socket to the plug seated (skills/rl_insertion.py): the
        arm goes from the stand-off straight down the target axis to where the policy starts -- the
        believed plug start_height above the socket, its axis on the socket's -- and the policy runs.
        Returns (status, info): 'seated' is the success, anything else a miss."""
        from ..skills import rl_insertion as RLI
        T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
        T_conn_sp = RLI.T_conn_simplug(T_junction_conn, P6.plug_part(cfg))
        e = rl_policy.spec['env']
        h = float(np.mean(e['start_height']))
        T_start = T_base_tconn @ translation_matrix([-h, 0.0, 0.0]) @ inverse(T_tool0_conn)
        phase('assemble')
        if not robot.arm.move_l(T_start, label='RL start (%.1f mm above the socket)' % (1000.0 * h)):
            log.error('RL INSERTION: could not reach the start above the socket.')
            return 'start', {}
        _q = np.asarray(robot.arm.q(), dtype=float)
        _sm = getattr(robot.arm.model, 'singularity_margin', None)
        log.info('RL INSERTION: at the start, joints %s deg%s.', np.round(np.degrees(_q), 1).tolist(),
                 '' if _sm is None else ' (nearest singularity: A%d at %.0f deg)' % (_sm(_q)[1] + 1, np.degrees(_sm(_q)[0])))
        rows = []

        def on_step(k, obs, act, P_bel, P_tgt, w):
            _x = (inverse(T_base_tconn @ T_conn_sp) @ P_bel)[:3, 3]
            rows.append({'step': k, 'x_mm': 1000 * _x[0], 'y_mm': 1000 * _x[1], 'z_mm': 1000 * _x[2],
                         **{f'a{i}': float(v) for i, v in enumerate(act)},
                         **{f'w{i}': float(v) for i, v in enumerate(w)}})
        try:
            status, info = RLI.run(robot, rl_policy, T_base_tconn, T_tool0_conn, T_conn_sp,
                                   {'admittance': dict(rl_cfg.get('admittance') or {}),
                                    'touch_n': float(rl_cfg.get('touch_n', 5.0)),
                                    'seated_advance_mm': float(rl_cfg.get('seated_advance_mm', 6.0)),
                                    'hold_s': float(rl_cfg.get('hold_s', 0.5)),
                                    'max_turn_deg': float(rl_cfg.get('max_turn_deg', 250.0))},
                                   guard_n=float(rl_cfg.get('guard_force_n', 40.0)), on_step=on_step)
        finally:
            if rows and out_dir:
                with open(os.path.join(out_dir, 'rl_trace.csv'), 'w', newline='') as fh:
                    w_ = _csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w_.writeheader()
                    w_.writerows(rows)
        (log.info if status == 'seated' else log.error)(
            'RL INSERTION: %s after %d steps -- advance %.1f mm since the first touch (%s), turned %+.0f deg '
            'about the axis, peak %.1f N.', status.upper(), info.get('steps', 0), info.get('advance_mm', 0.0),
            'at %.1f mm' % info['h_touch_mm'] if info.get('h_touch_mm') is not None else 'never touched',
            info.get('turn_deg', 0.0), info.get('peak_n', 0.0))
        if status == 'seated':
            _passed('RL INSERTION', '%.1f mm of advance after the first %.0f N push, held %.2f s' % (
                info['advance_mm'], float(rl_cfg.get('touch_n', 5.0)), float(rl_cfg.get('hold_s', 0.5))), 'SEATED')
        return status, info

    def rl_seated():
        """What connector_clocking hands collar clocking, for a plug the policy SEATED: the belief reset
        onto the seat -- the connector AT the target, at the roll it reached -- and that roll."""
        T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
        T_now = robot.tool0()
        got = matrix_to_xyzrpy(inverse(T_clk) @ (T_now @ T_tool0_conn))
        th = _wrap_near(float(got[1][0]), cc_mid)
        T_seat = T_clk @ xyzrpy_to_matrix([0.0, 0.0, 0.0], [th, 0.0, 0.0])
        log.info('RL INSERTION: the belief reset onto the seat -- it read %+.2f mm along +X of the target and '
                 '%+.1f deg about it (%+.1f wrt the target frame).', got[0][0] * 1000.0, np.degrees(th),
                 np.degrees(eng_clock + th))
        T_tool0_conn_now = inverse(T_now) @ T_seat
        if cc_open_after:
            phase('retract')
            if not robot.gripper.open('release (after the RL insertion)'):
                log.error('Gripper did not open after the RL insertion.')
                return False, T_tool0_conn_now, T_seat, float(np.degrees(th))
        return True, T_tool0_conn_now, T_seat, float(np.degrees(th))

    # ---- THE SLOT ROLL, SEEN (perception/slot_roll.py; the user, 2026-11-06) -----------------------------------------
    # The camera rides with the plug, so wherever the arm stands still with it held, the wrist image shows the same
    # slots: captured at the stand-off and again engaged (slot_roll.capture), several frames each, fused, and used by
    # connector clocking to aim its search. Per grasp: cleared in step_in_hand.
    slot_obs = {}

    def _slot_model():
        """slot_roll.cad (the two-J-slot plug, in the 6D plug CAD's frame), else the 6D plug CAD -- whose second J
        the model then copies 180 round."""
        from ..perception import slot_roll as SR
        part = P6.plug_part(cfg)
        path, scale = P6.cad_mesh(cfg, part)
        if sr_cad and os.path.exists(sr_cad):
            path, scale = sr_cad, sr_cad_scale
        elif sr_cad and not slot_obs.get('_cad_warned'):
            slot_obs['_cad_warned'] = True
            log.warning('slot_roll.cad %s not found -- the 6D plug CAD instead, its one J copied 180 deg round '
                        '(make it: analysis/bnc_plug_mesh.py).', sr_cad)
        return SR.SlotModel.from_cad(path, scale, mating_z=P6.MATING_Z.get(part), symmetric=sr_symmetric)

    def slot_capture(name):
        """`sr_frames` wrist images of the held plug, the arm still, each estimated against the belief it was held
        at -> slot_obs[name]; the images and drawings to <trial>/slot_roll/. Never a reason to fail."""
        if (not sr_on or name not in sr_capture or camera is None
                or not hasattr(camera, 'preview') or getattr(camera, 'K', None) is None):
            return None
        import cv2
        from ..perception import slot_roll as SR
        try:
            model = _slot_model()
            T_tool0_plug = robot.T_tool0_fingertip @ P6.plug_from_belief(T_ftip_conn, T_junction_conn,
                                                                          P6.plug_part(cfg))
            T_cam_plug = inverse(robot.T_tool0_cam) @ T_tool0_plug
            d = os.path.join(out_dir, 'slot_roll')
            os.makedirs(d, exist_ok=True)
            res, t0 = [], time.monotonic()
            time.sleep(0.2)                                  # a frame from after the arm stopped
            for k in range(sr_frames):
                if k:
                    time.sleep(0.15)
                img = camera.preview()
                if img is None:
                    continue
                cv2.imwrite(os.path.join(d, f'{name}_{k}.jpg'), img)
                e = SR.estimate(img, camera.K, T_cam_plug, model)
                if e is None:
                    continue
                vis = SR.draw(img, e, model)
                if vis is not None:
                    cv2.imwrite(os.path.join(d, f'{name}_{k}_est.png'), vis)
                res.append(e)
            f = SR.fuse(res, model)
            slot_obs[name] = {'_raw': res, 'T_tool0_plug': T_tool0_plug, 'fused': SR.public(f),
                              'frames': [SR.public(e) for e in res]}
            if f is None:
                log.warning('SLOT ROLL (%s): the plug\'s slotted band was not found in %d frame%s.', name, sr_frames,
                            '' if sr_frames == 1 else 's')
            else:
                log.info('SLOT ROLL (%s): %.0f deg (frames %s), margin %.2f, %.1f s -- %s/%s_*_est.png.', name,
                         f['roll_deg'], [round(e['roll_deg']) for e in res], f['margin'], time.monotonic() - t0, d,
                         name)
            _write_slot_json()
            return f
        except Exception as exc:                         # noqa: BLE001 -- a measurement, never a reason to stop
            log.warning('SLOT ROLL (%s): not estimated (%s).', name, exc)
            return None

    def slot_fused():
        """This grasp's captures named in slot_roll.use (else any it has) measured against the SAME belief (the
        latest's), fused -> (estimate, tool0 -> plug it is relative to), or (None, None)."""
        from ..perception import slot_roll as SR
        caps = [slot_obs[n] for n in sr_use if n in slot_obs and slot_obs[n].get('_raw')]
        if not caps:
            caps = [slot_obs[n] for n in sr_capture if n in slot_obs and slot_obs[n].get('_raw')]
        if not caps:
            return None, None
        T = caps[-1]['T_tool0_plug']
        raw = [e for c in caps if np.allclose(c['T_tool0_plug'], T, atol=1e-6) for e in c['_raw']]
        return SR.fuse(raw, _slot_model()), T

    def _write_slot_json(extra=None):
        """<trial>/slot_roll.json: every capture (frames and fusion), and -- once clocking ran -- the plan and the
        drop-in."""
        try:
            rep = {'model': _slot_model().summary(),
                   'captures': {n: {'fused': c['fused'], 'frames': c['frames'],
                                    'T_tool0_plug': np.asarray(c['T_tool0_plug']).tolist()}
                                for n, c in slot_obs.items() if not n.startswith('_')}}
            rep.update(slot_obs.get('_clocking') or {})
            rep.update(extra or {})
            with open(os.path.join(out_dir, 'slot_roll.json'), 'w') as fh:
                json.dump(rep, fh, indent=1, default=lambda o: np.asarray(o).tolist())
        except (OSError, TypeError, ValueError) as exc:
            log.warning('slot_roll.json not written (%s)', exc)

    def _engage_images(stage, T_tool0_conn_e, conn_x_mm):
        """An EngageImageRecorder for this engage (assembly.engage.record_images), started -- or None (off, no
        camera, a dry run). Its meta is everything a replay needs to draw the plug and the socket."""
        ri = dict(en.get('record_images') or {})
        if not ri.get('enabled', True) or camera is None or robot.arm.dry_run or not hasattr(camera, 'preview'):
            return None
        from ..perception.engage_vision import EngageImageRecorder

        def state():
            T = robot.tool0()
            Tc = T @ T_tool0_conn_e
            try:
                w = robot.arm.wrench_in(Tc, T)
            except Exception:                      # noqa: BLE001
                w = np.full(6, np.nan)
            return {'stage': stage[0], 'tool0': T, 'depth_mm': conn_x_mm(T), 'axial_n': float(w[0]),
                    'radial_n': float(np.linalg.norm(w[1:3]))}
        try:
            part, spart = P6.plug_part(cfg), P6.socket_part(cfg)
            T_sock = None
            try:
                with open(os.path.join(cfg.get('data_dir', 'data'), 'pose6d', 'last_socket.json')) as fh:
                    T_sock = json.load(fh).get('T_base_obj')
            except (OSError, ValueError):
                pass
            meta = {'K': getattr(camera, 'K', None), 'T_tool0_cam': robot.T_tool0_cam,
                    'T_tool0_fingertip': robot.T_tool0_fingertip, 'T_base_target': T_base_tconn,
                    'T_tool0_conn': T_tool0_conn_e,
                    'T_tool0_plug': robot.T_tool0_fingertip @ P6.plug_from_belief(T_ftip_conn, T_junction_conn, part),
                    'socket_pose': T_sock if tgt_source == 'pose6d' else None,     # the 6D estimate's CAD pose
                    'plug': part, 'socket': spart, 'plug_cad': P6.cad_mesh(cfg, part),
                    'socket_cad': P6.cad_mesh(cfg, spart), 'insertion_mode': ins_mode}
            return EngageImageRecorder(camera, robot, out_dir, state, float(ri.get('rate_hz', 5.0)),
                                       int(ri.get('jpeg_quality', 90)), float(ri.get('max_s', 120.0)), meta).start()
        except Exception as exc:                   # noqa: BLE001 -- a record, never a reason to stop the run
            log.warning('engage images not recorded (%s)', exc)
            return None

    _engage_extra = {}            # what the last engage adds to its (status, ref, depth): the push home's result

    def engage_insertion():
        """ENGAGE -- drive the assembly trajectory home, optionally rocking, stop on axial force.

        Follows configs/assembly_trajectory.csv, extends it by `preload_mm` past the mate, and
        superimposes an oscillation on the way. With every amplitude at 0 it is a plain direct
        insertion, so direct and wiggle are one code path at two settings.

        TWO CLOCKS, deliberately. The PATH advances by DISTANCE (speed.phase_scale.engage x the
        global translation cap); the OSCILLATION
        advances by TIME, so its frequency is the frequency configured regardless of how fast the
        path is driven. Pacing both by distance would make the oscillation frequency a function of
        the insert speed. MIND THE DURATION: the insertion lasts path_length / that rate, so a
        frequency chosen without reference to that can complete less than one cycle and act as a
        constant offset -- the start-up line reports cycles-per-insertion for that reason.

        TERMINATION is the point of the phase: reaching the end of the path is SUCCESS, and
        hitting the axial force limit is a NORMAL end, not a failure -- meeting resistance partway
        is exactly what the bayonet screw is for. That is why the limit is AXIAL (see _AxialForce)
        and can be set low.

        Returns (status, last_ref, depth_mm) with status 'complete' or 'force'."""
        _engage_extra.clear()
        T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn

        # ---- the path: trajectory rows, then the preload, all as connector-wrt-target 6-vecs ----
        rows6 = [np.asarray(vec6_from_mats(r), dtype=float) for r in dense]
        rows6 = [np.concatenate([v[:3] * 1000.0, v[3:]]) for v in rows6]   # -> mm / deg
        if en_pre_mm > 0.0:
            step = float(a.get('translational_resolution_m', 0.001)) * 1000.0
            n_pre = max(1, int(round(en_pre_mm / max(step, 1e-6))))
            base = rows6[-1].copy()
            for k in range(1, n_pre + 1):
                v = base.copy()
                v[0] += en_pre_mm * k / n_pre        # extend along the connector's own +X
                rows6.append(v)
        seg = [float(np.linalg.norm(rows6[i + 1][:3] - rows6[i][:3]))
               for i in range(len(rows6) - 1)]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total_mm = float(cum[-1])

        v_mm_s = g_v * s_eng
        dt = 1.0 / en_rate

        def path_at(d_mm):
            """The trajectory 6-vec `d_mm` along the path, linearly between rows."""
            d = float(np.clip(d_mm, 0.0, total_mm))
            j = int(np.searchsorted(cum, d, side='right') - 1)
            j = max(0, min(j, len(rows6) - 2))
            span = cum[j + 1] - cum[j]
            f = 0.0 if span <= 1e-12 else (d - cum[j]) / span
            return rows6[j] + f * (rows6[j + 1] - rows6[j])

        def ref_at(t, dur=None):
            """The path point at time t, with the wiggle RIGHT-MULTIPLIED onto it -- so the
            offset acts in the connector's OWN frame and a misaligned part rocks about its own
            axes, which is what the part physically does and what wiggle_sampling collected."""
            T = _corr_to_m(mats_from_vec6(path_at(v_mm_s * t)))
            if en_wig is not None:
                T = T @ en_wig.delta(t, dur)
            return traj_ref(T, T_tool0_conn)

        live = [f'{d} {en_amp[i]:+.2f}@{en_frq[i]:.2f}Hz'
                for i, d in enumerate(DIM_KEYS) if abs(en_amp[i]) > 0.0]
        log.info('  engage reference rate %.2f mm/s = %.4g x %.4g mm/s '
                 '(speed.phase_scale.engage x speed.max_cartesian_translation_mm_s). The arm '
                 'speed scale does NOT apply: this phase is servo-driven.',
                 v_mm_s, s_eng, g_v)
        log.info('--- ENGAGE --- %s over %.1f mm (%.1f mm trajectory + %.1f mm preload) at '
                 '%.2f mm/s -> %.1f s; %s.',
                 'oscillating ' + ', '.join(live) if live else 'DIRECT (no oscillation)',
                 total_mm, total_mm - en_pre_mm, en_pre_mm, v_mm_s, total_mm / max(v_mm_s, 1e-9),
                 f'axial force limit {en_fmax:.1f} N' if en_fmax > 0 else 'NO force limit')
        if live:
            dur = total_mm / max(v_mm_s, 1e-9)
            log.info('  oscillation peak %.2f mm/s / %.2f deg/s%s; the PATH runs at %.2f mm/s, so '
                     'the two add.', en_pv * en_scale, en_pw * en_scale,
                     f' (capped, time scale {en_scale:.3f})' if en_scale < 1.0 else ' (uncapped)',
                     v_mm_s)
            cyc = {d: en_frq[i] * en_scale * dur
                   for i, d in enumerate(DIM_KEYS) if abs(en_amp[i]) > 0.0}
            worst = min(cyc.values()) if cyc else 0.0
            (log.info if worst >= 1.0 else log.warning)(
                '  cycles completed during the %.1f s insertion: %s%s', dur,
                {k: round(v, 2) for k, v in cyc.items()},
                '.' if worst >= 1.0 else ' -- under one full cycle acts as a constant OFFSET, not '
                'a wiggle. Raise the frequency or lower speed.phase_scale.engage.')

        first = ref_at(0.0)
        phase('standoff')
        q = robot.arm.ik(first, seed_q)
        if q is None:
            log.error('Could not reach the engage start pose: it does not IK-solve.')
            return 'unreachable', first, float('nan')
        if not _guarded(robot, guard_shared, lambda: robot.arm.move_j(q, label='engage start')):
            # NOT an IK failure (2026-11-06 12:51: the FRI link had dropped, and this was reported as "the engage
            # path does not IK-solve"): the move itself did not run.
            log.error('Could not reach the engage start pose: the move did not complete%s.',
                      (' -- ' + robot.arm.link_down()) if robot.arm.link_down() else '')
            return 'move_failed', first, float('nan')

        det = _AxialForce(robot, T_tool0_conn, en_fmax, en_fpers)
        combo = _AnyGuard(det, guard_en)
        obs, cnt = [], [0]

        # ---- ENGAGE TRACE ------------------------------------------------------------
        # `obs` records only where the connector was BELIEVED to be, plus the wrench. That
        # cannot answer the question a rebound actually poses -- WHAT MOVED? The commanded
        # pose is `ref @ Delta`, so a retreat is one of three different faults:
        #
        #   ref_x falls          -> the TRAJECTORY commanded it (waypoints//wiggle lever arm)
        #   delta_x falls        -> the CONTROLLER yielded (compliance unloading, w/S)
        #   both flat, meas falls-> the ARM did not track, or the belief frame is moving
        #
        # flange_* is read straight off the arm and does NOT depend on T_tool0_conn, so it
        # stays honest even when the connector-wrt-fingertip belief is well off -- which is
        # exactly the case where meas_x and the truth part company.
        stage = ['contact']
        trace = []
        t_trace0 = [robot.arm.clock()]

        def _conn_x_mm(T_base_tool0):
            """Insertion depth of the connector wrt the TARGET frame, for any tool0 pose."""
            if T_base_tool0 is None:
                return float('nan')
            rel = inverse(T_base_tconn) @ (T_base_tool0 @ T_tool0_conn)
            return float(matrix_to_xyzrpy(rel)[0][0] * 1000.0)

        # THE WRIST CAMERA THROUGH CONTACT AND ENGAGE (the user, 2026-11-06): its newest frame `rate_hz` times a
        # second, from a thread of its own, each with the stage, the depth and the forces at that instant --
        # <trial>/engage_frames/ (perception/engage_vision.py; replay with the CAD drawn on: python -m
        # cable_assembly.perception.engage_vision <trial>). Stopped on every way out of the engage.
        rec_img = _engage_images(stage, T_tool0_conn, _conn_x_mm)

        def _done(status_, ref_, depth_):
            if rec_img is not None:
                rec_img.stop()
            return status_, ref_, depth_

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim != 0:
                return
            row = _observe(robot, T_tool0_conn, T_base_tconn)
            obs.append(row)
            d = adm_en.give()
            flange = robot.tool0()
            trace.append([
                round(robot.arm.clock() - t_trace0[0], 4), stage[0],
                round(_conn_x_mm(adm_en.last_ref), 4),
                round(_conn_x_mm(adm_en.last_cmd), 4),
                round(float(row[0]), 4),
                round(float(d[0]) * 1000.0, 4), round(float(d[1]) * 1000.0, 4),
                round(float(d[2]) * 1000.0, 4),
                round(float(np.degrees(np.linalg.norm(d[3:]))), 4),
                round(float(flange[0, 3]) * 1000.0, 4),
                round(float(flange[1, 3]) * 1000.0, 4),
                round(float(flange[2, 3]) * 1000.0, 4),
                round(float(row[6]), 3), round(float(row[7]), 3), round(float(row[8]), 3),
            ])

        # NO phase() HERE, deliberately. The engage is driven entirely by servo_l, and servo_l
        # calls servoL directly -- it never reads arm.speed_scale. So a phase scale cannot pace
        # this motion; the ONLY thing that does is v_mm_s (the reference rate, above). Setting the
        # phase here would be worse than useless: it would leave the arm at engage's very low
        # scale for every ordinary move that follows, until the next phase() call.
        adm_en.set_frame(T_tool0_conn if en_law_frame == 'connector' else np.eye(4))   # before the reset
        adm_en.reset()
        adm_en.warmup(first, tare_fn=tare)
        prev = last_ref = first
        # REAL ELAPSED TIME everywhere below, not i*dt. The servo loop does not necessarily cycle
        # at reference_rate_hz -- on 19 Aug it ran at ~332 Hz against a configured 125, which
        # delivered every frequency 2.65x high. Reading the clock makes the delivered frequency
        # the configured one whatever the loop does.
        dur_s = total_mm / max(v_mm_s, 1e-9)

        # ================= STAGE 1: CONTACT -- find the socket, gently =====================
        # Slow and UN-OSCILLATED. Driving the whole insertion in one go means the wiggle and the
        # full approach speed are already running when the parts first touch, which makes first
        # contact the least controlled moment of the phase. Creeping in first means the engagement
        # starts from a MEASURED contact pose instead of a guess about where the socket is.
        contact_d = 0.0
        if ct_on:
            ct_v = g_v * float(scales.get('contact', s_eng))
            ct_dur = total_mm / max(ct_v, 1e-9)
            ct_det = _AxialForce(robot, T_tool0_conn, ct_force_n, ct_pers_s)
            ct_combo = _AnyGuard(ct_det, guard_en)
            ct_combo.reset()
            log.info('--- CONTACT --- creeping in at %.2f mm/s (no oscillation) until %.1f N '
                     'holds for %.2f s, or %.1f mm of path runs out.',
                     ct_v, ct_force_n, ct_pers_s, total_mm)
            ct_status = 'no_contact'
            t_c0 = robot.arm.clock()
            while True:
                tc = robot.arm.clock() - t_c0
                if tc >= ct_dur:
                    break
                d = ct_v * tc
                cur = traj_ref(_corr_to_m(mats_from_vec6(path_at(d))), T_tool0_conn)
                res = adm_en.ramp(prev, cur, dt, ct_combo, on_step=log_cb)
                prev = last_ref = cur
                if res == 'seated':
                    ct_status = 'contact' if ct_combo.tripped is ct_det else 'guard'
                    contact_d = d
                    break
            if ct_status != 'contact':
                robot.arm.servo_stop()
                why = ('the general force guard tripped (%s)' % ct_combo.tripped_by
                       if ct_status == 'guard' else
                       'the whole %.1f mm path ran without %.1f N of axial reaction'
                       % (total_mm, ct_force_n))
                log.error('--- CONTACT FAILED --- %s. The connector never met the socket, so '
                          'there is nothing to engage into.', why)
                depth0 = float(matrix_to_xyzrpy(
                    inverse(T_base_tconn) @ (robot.tool0() @ T_tool0_conn))[0][0] * 1000.0)
                return _done(('guard' if ct_status == 'guard' else 'complete'), last_ref, depth0)
            log.info('--- CONTACT MADE --- %.2f mm along the path, %.1f N axial (peak %.1f N). '
                     'Engaging from here.', contact_d, ct_det.axial_n(), ct_det.peak_n)
            _passed('CONTACT', ct_det.tripped_by, 'ENGAGE')
            robot.arm.servo_stop()          # settle before asking; do not hold a servo open on a
            if not phase_gate(              # prompt that may sit for minutes
                    'ENGAGE (from contact)',
                    'The connector is TOUCHING the socket %.2f mm along the path (%.1f N axial). '
                    'Look at where it actually met before the alignment and the wiggle start.'
                    % (contact_d, ct_det.axial_n())):
                return _done('aborted', last_ref, float(matrix_to_xyzrpy(
                    inverse(T_base_tconn) @ (robot.tool0() @ T_tool0_conn))[0][0] * 1000.0))
            # REBASE, not reset: the arm is already touching (1 N -> ~2 mm of yield), so
            # re-engaging on `last_ref` with a zeroed Delta would push in by that much. rebase()
            # re-references onto the MEASURED pose, so the command does not move at all.
            last_ref = prev = adm_en.rebase(robot.tool0())
            adm_en.warmup(last_ref)         # re-engage the servo where the arm actually is

        # ================= STAGE 2: ENGAGE -- align laterally AND insert ===================
        # ALIGNMENT AND INSERTION RUN TOGETHER, not one then the other. Contact is made wherever
        # the estimates said the socket was, which is not exactly on the trajectory; the reference
        # therefore starts at the pose the arm is ACTUALLY at and fades onto the trajectory over
        # `align_s` while it advances. Starting instead at the trajectory would be a step the
        # admittance spring would have to absorb at the worst possible moment -- in contact.
        #
        # THE INSERTION AXIS IS EXCLUDED from the correction. x is DEPTH, and depth is what the
        # advance is for; folding the depth error into the alignment would either shove the part
        # in or pull it back out the instant contact was made.
        # As the phase's own references describe it: the online correction (pose6d.online) is applied to
        # eng_ref's below by the controller, so the measured pose must not carry it into err6 as well.
        _oc = getattr(robot.arm, 'reference_correction', None)
        _tool_now = robot.tool0() if _oc is None else _oc.uncorrect(robot.tool0())
        T_meas = inverse(T_base_targetobj) @ (_tool_now @ T_tool0_conn)
        cur6 = np.asarray(vec6_from_mats(T_meas), dtype=float)
        cur6 = np.concatenate([cur6[:3] * 1000.0, cur6[3:]])
        err6 = cur6 - path_at(contact_d)
        err6[0] = 0.0                       # ignore the insertion axis
        log.info('  aligning to the trajectory over %.1f s: lateral %+.2f, %+.2f mm, '
                 'rot %+.2f, %+.2f, %+.2f deg (x excluded -- that is the advance).',
                 en_align_s, err6[1], err6[2], err6[3], err6[4], err6[5])

        # THE WIGGLE'S WINDOW: force-limited, the advance may wait at a face for the wiggle to find the way in, so
        # the wiggle keeps going until the timeout -- not only for the path's own time (dur_s), past which it tapers
        # to nothing.
        wig_dur = en_timeout_s if fl_on else dur_s

        def eng_ref(t, d_mm):
            """Trajectory at depth `d_mm` along the path, plus the decaying lateral error, plus wiggle."""
            fade = max(0.0, 1.0 - t / max(en_align_s, 1e-9))
            v6 = path_at(d_mm) + fade * err6
            T = _corr_to_m(mats_from_vec6(v6))
            if en_wig is not None:
                T = T @ en_wig.delta(t, wig_dur)
            return traj_ref(T, T_tool0_conn)

        T_base_conn_contact = robot.tool0() @ T_tool0_conn
        stage[0] = 'engage'
        trav = _TravelReached(robot, T_tool0_conn, T_base_conn_contact, en_travel_mm)
        combo = _AnyGuard(trav, det, guard_en)
        combo.reset()
        log.info('--- ENGAGE --- from contact, advancing to the %.1f mm path end at %.2f mm/s '
                 'with the wiggle running. SUCCESS on %.1f mm of travel OR %.1f N held %.2f s; '
                 'TIMEOUT at %.1f s.',
                 total_mm, v_mm_s, en_travel_mm, en_fmax, en_fpers, en_timeout_s)
        if fl_on:
            log.info('  FORCE-LIMITED ADVANCE: full rate below %.1f N axial, slowing to a stop at %.1f N, backing off '
                     'above it (at most %.0f%% of the rate, never more than %.1f mm behind the contact) -- the wiggle '
                     'aligns under that light load. A connector that never drops in ends on the TIMEOUT (%.0f N held '
                     'is out of reach while the force is limited).', fl_soft, fl_target, 100.0 * fl_back,
                     fl_max_back_mm, en_fmax)
        status = 'timeout'
        t_wig0 = robot.arm.clock()
        d_ref, t_prev = contact_d, 0.0               # the reference's depth along the path: it can wait now
        f_axial = det.axial_n()
        f_peak_fl, t_held = 0.0, 0.0
        while True:
            t = robot.arm.clock() - t_wig0
            if t >= en_timeout_s:
                break
            step_s, t_prev = max(0.0, t - t_prev), t
            rate = v_mm_s
            if fl_on:
                f_axial += min(1.0, step_s / max(fl_tau, 1e-3)) * (det.axial_n() - f_axial)
                f_peak_fl = max(f_peak_fl, f_axial)
                rate = v_mm_s * float(np.clip((fl_target - f_axial) / (fl_target - fl_soft), -fl_back, 1.0))
                if rate < 0.25 * v_mm_s:
                    t_held += step_s
            d_ref = max(d_ref + rate * step_s, contact_d - fl_max_back_mm)
            if d_ref >= total_mm:
                status = 'complete'
                break
            cur = eng_ref(t, d_ref)
            res = adm_en.ramp(prev, cur, dt, combo, on_step=log_cb)
            prev = last_ref = cur
            if res != 'seated':
                continue
            status = ('travel' if combo.tripped is trav
                      else 'force' if combo.tripped is det else 'guard')
            break

        if status in ('force', 'travel'):
            _passed('ENGAGE', combo.tripped_by,
                    'SEAT CONFIRM' if cf_on else 'PUSH HOME' if ph_on else 'settle, then CONNECTOR CLOCKING')

        # The reference the loop actually ended on. The settle below holds THIS (see there for
        # why it must not re-reference to the measured pose); the confirmation stage, if it runs,
        # advances it further and updates this.
        ref_live = last_ref
        d_end = float(np.clip(d_ref, 0.0, total_mm))      # how far along the path the reference got
        if fl_on:
            log.info('  force-limited advance: the reference went %.2f mm from the contact in %.2f s (%.2f mm/s on '
                     'average, %.2f the rate); held back %.2f s by the force (filtered peak %.1f N).',
                     d_end - contact_d, t, (d_end - contact_d) / max(t, 1e-9), v_mm_s, t_held, f_peak_fl)

        # SNAPSHOT EVERY CONDITION AT THE MOMENT IT STOPPED, before the settle hold moves
        # anything. Read once, here, rather than per servo cycle.
        t_end = min(t, en_timeout_s)
        w_end = robot.arm.wrench()
        end_state = {
            # The ENGAGE stage is budgeted by TIME now (timeout), not by path length, so it is
            # reported against that -- with the contact stage's advance folded into driven_mm so
            # the depth figure still means "how far along the path we got".
            'elapsed_s': t_end, 'duration_s': en_timeout_s,
            'driven_mm': d_end, 'total_mm': total_mm,
            'contact_mm': contact_d, 'travel_mm': trav.peak_mm, 'travel_limit_mm': en_travel_mm,
            'axial_n': det.axial_n(), 'axial_peak_n': det.peak_n,
            'axial_limit_n': en_fmax, 'axial_persist_s': en_fpers,
            'force_n': float(np.linalg.norm(w_end[:3])),
            'torque_nm': float(np.linalg.norm(w_end[3:])),
        }

        # ---- CONFIRM THE SEAT ---------------------------------------------------------------
        # THE AXIAL STOP ONLY SAYS "SOMETHING RESISTED". Pushing a connector flat against the
        # fixture body, the bench, or the rim of the wrong hole develops axial reaction exactly
        # like a real insertion and stops the engage identically. Two things separate a socket
        # from a face, and both are measured by WIGGLING IN PLACE at the stopped depth:
        #
        #   RADIAL FORCE -- a barrel inside a socket is laterally constrained, so rocking it
        #       pushes back ACROSS its own axis. A connector resting on a flat face just slides.
        #   CONSTRAINED MOTION -- the wiggle reference keeps commanding the same amplitude, but a
        #       captured connector cannot follow it. Measured travel far below the commanded
        #       amplitude IS the capture; travel that tracks the command means it is still free.
        #
        # Wiggling IN PLACE, not advancing: the depth is already decided, and driving further
        # while testing would confound the two.
        confirm = None
        if status in ('force', 'travel') and cf_on:
            rad = _RadialConfirm(robot, T_tool0_conn, cf_radial_n, cf_radial_s)
            live_f = [en_frq[i] * en_scale for i in range(6)
                      if abs(en_amp[i]) > 0.0 and en_frq[i] > 0.0]
            cyc_s = min((cf_cycles / min(live_f)) if live_f else float(cf_fallback_s), cf_max_s)
            # WHERE THE ENGAGE STOPPED: its reference was path_at(contact_d + v t) plus the fading
            # lateral correction, so the wiggle continues from exactly that. Measured from the path
            # START (v t_end alone, as this was) the reference stepped back by the contact depth
            # in one cycle -- ~14 mm, a yank on a connector the bayonet had already caught; found
            # running this on the simulated cell, where it pulled the plug out of the grip.
            d_stop = d_end
            err_stop = max(0.0, 1.0 - t_end / max(en_align_s, 1e-9)) * err6
            pts = []

            def conf_step():
                rad.check()
                pts.append((robot.tool0() @ T_tool0_conn)[:3, 3])

            log.info('  confirming the seat: wiggling in place for up to %.2f s -- until radial >= %.1f N '
                     'holds %.2f s (the confirmation; it stops there). Travel (< %.2f mm) reported only.',
                     cyc_s, cf_radial_n, cf_radial_s, cf_max_mm)
            # NO reset() HERE. The spring is LOADED -- it is holding the contact equilibrium,
            # and Delta is the difference between the deep reference and where the part actually
            # is (w/S: at 15 N against 500 N/m that is 30 mm). Zeroing it while keeping
            # `last_ref` would step the command by the whole 30 mm straight into the socket, then
            # let it spring back out. Continue from the reference the engage ended on, with the
            # deflection intact, so the wiggle rides on the equilibrium instead of through it.
            stage[0] = 'confirm'
            prev_c = last_ref
            t_c0 = robot.arm.clock()
            while True:
                tc = robot.arm.clock() - t_c0
                if tc >= cyc_s or rad.satisfied:          # confirmed: stop wiggling a captured connector
                    break
                Tc = _corr_to_m(mats_from_vec6(path_at(d_stop) + err_stop))
                if en_wig is not None:
                    Tc = Tc @ en_wig.delta(t_end + tc, wig_dur)   # keep the wiggle phase continuous
                cur_c = traj_ref(Tc, T_tool0_conn)
                adm_en.ramp(prev_c, cur_c, dt, guard=None, on_step=conf_step)
                prev_c = cur_c
            robot.arm.servo_stop()
            ref_live = prev_c            # the wiggle moved it; the settle holds where we ARE
            if rad.satisfied:
                log.info('  radial reaction confirmed after %.2f s of the confirmation -- wiggle stopped.', tc)

            travel_mm = 0.0
            if len(pts) >= 2:
                P = np.asarray(pts, dtype=float)
                travel_mm = float(np.max(P.max(axis=0) - P.min(axis=0))) * 1000.0
            wiggled = bool(live_f)
            confirm = {
                'radial_ok': bool(rad.satisfied), 'radial_peak_n': rad.peak_n,
                'radial_held_s': rad.held_s, 'radial_limit_n': cf_radial_n,
                'radial_persist_s': cf_radial_s,
                'travel_mm': travel_mm, 'travel_limit_mm': cf_max_mm,
                'travel_ok': bool(travel_mm < cf_max_mm) if wiggled else None,
                'cycles_s': cyc_s, 'wiggled': wiggled,
            }
            # TRAVEL IS MEASURED AND REPORTED, BUT NO LONGER DECIDES. The commanded amplitude
            # is only an upper bound on what a FREE connector would do, and the admittance spring
            # already suppresses most of it -- so a small measured travel does not by itself mean
            # captured, and the test rejected good seats. Radial force is the discriminating
            # signal; the travel number stays in the log because it is useful to read.
            if confirm['radial_ok']:
                _passed('SEAT CONFIRM', 'radial force %.1f N peak, held %.2f s >= %.1f N for '
                        '%.2f s' % (rad.peak_n, rad.held_s, cf_radial_n, cf_radial_s),
                        'PUSH HOME' if ph_on else 'settle, then CONNECTOR CLOCKING')
                log.info('  SEAT CONFIRMED: radial %.1f N held %.2f s (>= %.1f N / %.2f s). '
                         '(travel %.2f mm -- reported only, not a criterion.)',
                         rad.peak_n, rad.held_s, cf_radial_n, cf_radial_s, travel_mm)
            else:
                status = 'unconfirmed'
                log.error('  SEAT NOT CONFIRMED -- the axial limit was met but there was no '
                          'sustained RADIAL reaction, so the connector is most likely against a '
                          'face rather than in a socket. radial peak %.1f N, longest hold %.2f s '
                          '(want >= %.1f N for %.2f s). (travel %.2f mm -- reported only.)',
                          rad.peak_n, rad.held_s, cf_radial_n, cf_radial_s, travel_mm)

        # THE WIGGLE BROUGHT BACK TO THE AXIS BEFORE THE SETTLE (the user, 2026-11-06: "the wiggle continued too
        # long and ended up angling the connector between the fingers"). Wherever the engage or the confirmation
        # stopped, the reference still carries the wiggle's offset at that moment -- up to its full amplitude
        # (6.4 mm at 3.5 s of the 8 mm / 0.1 Hz one) -- and the settle, then connector clocking, would hold the
        # captured connector pushed sideways by it. Ramped out (no step) at the engage's own pace, on a seat only:
        # a miss is backed out along the socket axis by the recovery anyway.
        if en_wig is not None and status in ('force', 'travel'):
            # the confirmation's own reference without the wiggle: where the engage stopped, its lateral
            # correction as it was then
            d_fade = d_end
            T_axis = traj_ref(_corr_to_m(mats_from_vec6(
                path_at(d_fade) + max(0.0, 1.0 - t_end / max(en_align_s, 1e-9)) * err6)), T_tool0_conn)
            off_mm, off_rad = pose_error(ref_live, T_axis)
            if off_mm > 1e-4 or off_rad > np.radians(0.05):
                fade_s = max(1.0, 1000.0 * off_mm / max(v_mm_s, 1e-9), np.degrees(off_rad) / 5.0)
                log.info('  wiggle out: the reference back onto the axis, %.2f mm / %.2f deg over %.1f s, before '
                         'the settle.', 1000.0 * off_mm, np.degrees(off_rad), fade_s)
                stage[0] = 'unwiggle'
                adm_en.ramp(ref_live, T_axis, fade_s, guard=None, on_step=log_cb)
                ref_live = last_ref = T_axis

        # ---- PUSH HOME (assembly.engage.push_home: a LATCHING plug, the RJ45) ----------------------------------
        # The engage's success says the plug is IN the jack -- some millimetres past where it touched -- not that
        # it is HOME, and its latch clicks over only at the bottom. So on along the connector's +X, from the
        # reference the engage ended on (the wiggle faded out, the spring still loaded: no reset, as for the
        # confirmation), until the floor pushes back: force_n axial held persistence_s. Then the press is taken
        # off -- the reference backed off by force / stiffness -- so the tug that follows tares on a plug at rest,
        # not on the press (a tare under it would read the latch's hold as a pull it is not).
        push = None
        if ph_on and status in ('force', 'travel'):
            stage[0] = 'push_home'
            err_end = max(0.0, 1.0 - t_end / max(en_align_s, 1e-9)) * err6

            def push_ref(d_mm):
                """The engage's axis reference `d_mm` along the path -- and past its end, on along the +X."""
                v6 = np.array(path_at(min(d_mm, total_mm)) + err_end, dtype=float)
                v6[0] += max(0.0, d_mm - total_mm)
                return traj_ref(_corr_to_m(mats_from_vec6(v6)), T_tool0_conn)

            ph_v = g_v * float(scales.get('push_home', s_eng))
            ph_det = _AxialForce(robot, T_tool0_conn, ph_force, ph_pers)
            ph_combo = _AnyGuard(ph_det, guard_en)
            ph_combo.reset()
            depth_ph0 = _conn_x_mm(robot.tool0())
            log.info('--- PUSH HOME --- on along the connector +X at %.2f mm/s until %.1f N axial holds %.2f s (the '
                     'plug on the floor, the latch over), the reference at most %.1f mm past where the engage '
                     'stopped (depth now %+.2f mm).', ph_v, ph_force, ph_pers, ph_max_mm, depth_ph0)
            ph_status, t_cap, d_p, prev_p = 'max', None, d_end, ref_live
            t_p0 = robot.arm.clock()
            while True:
                tp = robot.arm.clock() - t_p0
                if tp >= ph_timeout_s:
                    ph_status = 'timeout'
                    break
                d_p = min(d_end + ph_v * tp, d_end + ph_max_mm)
                if d_p >= d_end + ph_max_mm:               # the reference at its limit: let the spring catch up
                    t_cap = tp if t_cap is None else t_cap
                    if tp - t_cap >= ph_dwell_s:
                        break
                cur = push_ref(d_p)
                res = adm_en.ramp(prev_p, cur, dt, ph_combo, on_step=log_cb)
                prev_p = cur
                if res == 'seated':
                    ph_status = 'home' if ph_combo.tripped is ph_det else 'guard'
                    break
            ref_live = last_ref = prev_p
            depth_ph = _conn_x_mm(robot.tool0())
            f_ph = ph_det.axial_n()
            push = {'status': ph_status, 'axial_n': round(float(f_ph), 2), 'peak_n': round(float(ph_det.peak_n), 2),
                    'depth_before_mm': round(depth_ph0, 2), 'depth_mm': round(depth_ph, 2),
                    'reference_mm': round(float(d_p - d_end), 2), 's': round(float(tp), 2)}
            if ph_status == 'home':
                _passed('PUSH HOME', ph_det.tripped_by, 'unload, then the LATCH TUG')
                log.info('  HOME: %.1f N axial after the plug moved %+.2f mm (depth %+.2f mm, by the belief).',
                         f_ph, depth_ph - depth_ph0, depth_ph)
            elif ph_status == 'guard':
                log.error('PUSH HOME: the general force guard tripped (%s) -- a JAM, not the floor. Treated as a failed '
                          'engage.', ph_combo.tripped_by)
                status = 'guard'
            else:
                log.warning('PUSH HOME: %s without %.1f N held (peak %.1f N); the plug moved %+.2f mm (depth %+.2f mm). '
                            'Not known to be home -- the tug will say whether it latched.',
                            'the reference ran its %.0f mm' % ph_max_mm if ph_status == 'max' else
                            'timed out after %.0f s' % ph_timeout_s, ph_force, ph_det.peak_n,
                            depth_ph - depth_ph0, depth_ph)
            if ph_status != 'guard':
                # UNLOAD: the reference back onto where the plug IS -- the spring's lead along the axis, which is
                # what holds the press -- less what keeps unload_to_n (at the spring's stiffness along the axis, in
                # the axes its law runs in).
                ax_b = np.asarray(T_base_tconn[:3, 0], dtype=float)
                ax_b = ax_b / max(float(np.linalg.norm(ax_b)), 1e-12)
                u = (robot.tool0()[:3, :3] @ adm_en.frame[:3, :3]).T @ ax_b
                s_ax = 1.0 / float(np.sum((u ** 2) / np.maximum(adm_en.S[:3], 1e-9)))
                lead_mm = _conn_x_mm(prev_p) - _conn_x_mm(robot.tool0())
                back_mm = max(0.0, lead_mm - 1000.0 * ph_unload_n / s_ax)
                push['lead_mm'] = round(float(lead_mm), 2)
                if back_mm > 0.05:
                    T_back = push_ref(d_p - back_mm)
                    log.info('  unloading: the reference back %.2f mm onto the plug (its lead %.2f mm; %.0f N/m along '
                             'the axis) before the tug.', back_mm, lead_mm, s_ax)
                    stage[0] = 'unload'
                    adm_en.ramp(prev_p, T_back, max(0.5, back_mm / max(ph_v, 1e-9)), guard=None, on_step=log_cb)
                    ref_live = last_ref = T_back
        _engage_extra['push_home'] = push

        # SETTLE ON THE LOADED EQUILIBRIUM, NOT ON A FRESH ZERO.
        #
        # `T_cmd = ref @ Delta`, and at the moment engage stops Delta IS the contact deflection --
        # w/S, which at the 5 N exit force against 500 N/m is 10 mm of back-off. The old
        # `stay = tool0(); reset(); hold(stay)` reads as safe, and it does not JUMP: re-referencing
        # to the measured pose while zeroing Delta starts the hold exactly where the arm already
        # is. But it has thrown the equilibrium away, so the integrator spends the whole settle
        # RE-DERIVING it from the new zero -- driving Delta back to w/S and walking the connector
        # that far out of the socket, a fraction of a mm per cycle. THAT is the rebound: not a
        # step at the exit, but a 5 s unloading RAMP that begins the instant the exit fires.
        #
        # Holding the reference engage ended on, with Delta intact, commands the pose the arm is
        # already at and keeps the preload. The loop still yields if the mate relaxes -- which is
        # what a settle is for -- it just no longer double-counts the compliance.
        stage[0] = 'settle'
        if settle_shared > 0:
            adm_en.hold(ref_live, settle_shared, guard=None, on_step=log_cb)
        adm_en.stop()
        robot.arm.servo_stop()

        depth = float(matrix_to_xyzrpy(
            inverse(T_base_tconn) @ (robot.tool0() @ T_tool0_conn))[0][0] * 1000.0)
        _engage_report(status, end_state, depth, det, guard_en, combo, confirm)
        _report_engage_trace(trace)
        if trace:
            _save_engage_trace(os.path.join(out_dir, 'engage_trace.csv'), trace)
        if obs:
            _save_observations(os.path.join(out_dir, 'engage_observations.csv'), obs)
        return _done(status, last_ref, depth)

    def _slot_outcome(rep, drop, ok, verdict):
        """After the search: where the plug dropped onto the pins (the trace's drop-in), what that says the pins'
        angle is, how far the aim was off -- logged, kept in slot_roll.json and the trial's data, and (learn, a
        trusted estimate) added to the pin store."""
        from ..perception import slot_roll as SR
        period = 180.0 if sr_symmetric else 360.0
        rep.update({'drop_in': drop, 'sweep_ok': bool(ok), 'verdict': verdict})
        t = tr()
        f = rep.get('fused') or {}
        t.data.update(slot_roll_deg=f.get('roll_deg'), slot_margin=f.get('margin'),
                      slot_plan_deg=(rep.get('plan') or {}).get('theta_star_deg'))
        if drop is None:
            log.info('  slot roll: no drop-in seen in the trace (the plug never gained %.1f mm on the push within '
                     '%.0f deg of turn)%s.',
                     sr_drop_mm, sr_drop_win, '' if ok else ' -- and the search did not seat it')
        elif 'alpha0_deg' in rep:
            implied = SR.wrap(rep['alpha0_deg'] + drop['theta_deg'], period)
            rep['pin_implied_deg'] = implied
            t.data.update(slot_drop_deg=round(drop['theta_deg'], 1), slot_pin_implied_deg=round(implied, 1))
            plan = rep.get('plan')
            if plan:
                miss = SR.wrap(drop['theta_deg'] - plan['theta_star_deg'], period)
                rep['aim_error_deg'] = miss
                t.data['slot_aim_error_deg'] = round(miss, 1)
            log.info('  slot roll: DROP-IN at %+.1f deg from the engaged roll (%s; the plug rose %.1f mm) -> the '
                     'pins at %+.0f deg about the socket axis%s.', drop['theta_deg'], drop['phase'],
                     drop['lag_mm'] - drop['lag_foot_mm'], implied,
                     (('; the aim was %+.0f deg off' if plan.get('applied', True) else
                       '; the recorded (unused) aim would have been %+.0f deg off') % rep['aim_error_deg'])
                     if plan else '')
            if sr_learn and f.get('margin', -1.0) >= sr_min_margin:
                try:
                    SR.PinStore(sr_store_path).add(implied, rep['socket_xyz_m'], run=os.path.basename(run_dir),
                                                   trial=int(cycle), roll_deg=f.get('roll_deg'),
                                                   margin=f.get('margin'), drop_theta_deg=drop['theta_deg'],
                                                   alpha0_deg=rep['alpha0_deg'])
                    log.info('  slot roll: the pins\' angle %+.0f deg learned -> %s.', implied, sr_store_path)
                except OSError as exc:
                    log.warning('  slot roll: the pin angle not stored (%s).', exc)
        slot_obs['_clocking'] = {'clocking': rep}
        _write_slot_json()

    def connector_clocking():
        """CONNECTOR CLOCKING -- the bayonet search, and the belief reset that makes it well posed.

        THE BELIEF RESET: the mate is made, so the connector's pose is known from a PHYSICAL
        CONSTRAINT rather than estimated -- it is AT the target. That replaces the estimated
        belief, which is what makes the screw axis the target's +X exactly instead of inheriting
        the accumulated in-hand error. The pose the arm stands at becomes the engaged pose and
        every measurement below is relative to it.

        THE MOTION IS AN OSCILLATING SCREW. `sweep_deg` lists roll positions about the connector's
        +X -- absolute, wrt the TARGET frame -- visited IN TURN, one per try, while pushing
        `push_mm` along that same axis. Translation ALONG the rotation axis commutes with it, so
        each leg is a true helix. The push target is VIRTUAL: it aims past where the connector can
        go and lets compliance follow whatever path the cams allow.

        WHY IT ROCKS INSTEAD OF TURNING FURTHER. A bayonet pin that missed its slot will not find
        it by turning harder -- it rides the rim and jams. Rocking across the slot under a steady
        axial load is what finds it. Hence the press being established on the first leg and HELD
        across every reversal, and the search ending on measured x-advance rather than any
        commanded angle.

        A LEG THAT JAMS IS NORMAL: the guard stopping one part-way is how the far edge of the slot
        is found. The next leg reverses from where the reference actually stopped (screw_ramp's
        returned fraction), not from the endpoint it never reached.

        SUCCESS IS ASSUMED when the sweep runs. `success_advance_mm` is the EARLY-OUT and
        telemetry, never a post-hoc verdict -- cross-checking against measured advance turned
        soft-spring deflection into false failures. The only not-seated outcome is a sweep in
        which EVERY leg was guard-stopped: nothing swung freely, so the connector is wedged rather
        than searching.

        NO REGRASP -- the retry IS the reversal and the gripper stays closed throughout. (The old
        ratchet existed because a one-way stroke could only be retried by giving back the wrist
        range it had spent; an oscillation never spends range it does not immediately return.)

        Returns (ok, T_tool0_conn, T_base_conn, achieved_deg) -- where the connector ended up,
        read from the MEASURED arm pose, so it is the ACHIEVED roll and not the commanded one."""
        nonlocal T_clk                                  # the axis probe may move it onto the socket's real one
        T_tool0_engaged = robot.tool0()
        T_tool0_conn = inverse(T_tool0_engaged) @ T_clk
        moved = matrix_to_xyzrpy(inverse(robot.T_tool0_fingertip @ T_ftip_conn) @ T_tool0_conn)
        log.info('--- CONNECTOR CLOCKING --- belief reset: connector assumed AT the %s frame '
                 '(shifts the in-hand belief by %s mm, %s deg)', pe_frame,
                 np.round(moved[0] * 1000.0, 2).tolist(),
                 np.round(np.degrees(moved[1]), 2).tolist())
        log.info('  oscillating screw: roll to %s deg (wrt the target frame) in turn, up to %d '
                 'leg%s, while pushing %+.1f mm along the connector +X and HOLDING it; early-out '
                 'at %.1f mm of CUMULATIVE advance (a completed sweep = seated).',
                 [round(float(np.degrees(t)), 1) for t in cc_sweep], cc_tries,
                 '' if cc_tries == 1 else 's', cc_push_m * 1000.0, cc_need_m * 1000.0)
        log.info('  starting at the engaged roll of %+.1f deg, so leg 1 turns %+.1f deg (%s) and '
                 'the band worked is %+.1f .. %+.1f deg about the socket +X.',
                 np.degrees(eng_clock), np.degrees(cc_legs[0]),
                 'NEGATIVE' if cc_legs[0] < 0 else 'POSITIVE',
                 np.degrees(eng_clock + cc_lo), np.degrees(eng_clock + cc_hi))

        # Each leg is a BASE-frame motion about the fixed axis line L (engaged connector origin,
        # along its +X) -- independent of the connector-in-gripper belief, and the SAME line every
        # leg, since rotation about +X and push along +X both leave L invariant.
        ref_start = T_tool0_engaged

        def arm_at(th_rel, push):
            """The arm pose at roll `th_rel` (rad from the ENGAGED roll) and axial offset `push`."""
            return (T_clk
                    @ xyzrpy_to_matrix([push, 0.0, 0.0], [th_rel, 0.0, 0.0])
                    @ inverse(T_clk)) @ ref_start

        # ---- REACHABILITY OF THE SWEPT BAND, before anything turns ----------------------------
        # The legs are servoed, not planned, so without this the arm finds out a pose is
        # unreachable by failing at it mid-turn, gripped and cammed part-way home. The band is the
        # union of every leg, so one scan covers all of them however the tries fall.
        #
        # ADVISORY, NOT A GATE: a marginal pose can solve here and not on the arm, so refusing
        # would abort runs that would have completed.
        #
        # THE SEED IS CHAINED from the previous sample, because the solver returns the branch
        # nearest its seed and the servo walks each leg in fractions of a degree. Seeding every
        # sample from the current joints would let the far end resolve onto an unreachable branch
        # and report a range problem that is really a seeding artefact.
        #
        # arm.ik logs 'No IK solution' at ERROR per unsolved pose, so a marginal band prints a
        # burst of them here. They belong to this scan; nothing has moved.
        bad, seed_scan = [], robot.arm.q()
        for _k in range(25):
            _th = cc_lo + (cc_hi - cc_lo) * (_k / 24.0)
            _q = robot.arm.ik(arm_at(_th, cc_push_m), seed_scan)
            if _q is None:
                bad.append(_th)
            else:
                seed_scan = _q
        if bad:
            log.warning('  CONNECTOR CLOCKING: %d of 25 poses sampled across the %.0f deg band have '
                        'no IK solution (first at %+.0f deg wrt the target frame). The sweep will '
                        'be attempted anyway, but it is likely to run out of wrist range there -- '
                        'move assembly.connector_clocking.sweep_deg, or the engage angle, rather than '
                        'letting a leg discover it mid-turn.',
                        len(bad), np.degrees(cc_hi - cc_lo), np.degrees(eng_clock + bad[0]))

        # ---- THE SEARCH AIMED BY THE SLOTS (the user, 2026-11-06: "using the slot estimate the connector roll
        # orientation, but still do connector clocking to account for errors (with the idea that it requires less
        # clocking) - perhaps try both angles in increasing increments so for example, +15, -15, +30, -30, etc") ----
        # The wrist images gave where the slot entries point on the tool; the pins' angle about the socket axis is
        # configured (slot_roll.pin_deg) or learned from earlier drop-ins. The turn that brings one onto the other
        # is theta*; the legs visit theta*, theta* +- step, +- 2 step, ... -- nearest first, so an error in either
        # costs legs, not the search. Without both: the plain sweep, and the slots still measured and the drop-in
        # still recorded (that is how the pins are learned).
        legs_run, tries_run, slot_rep = cc_legs, cc_tries, None
        if sr_on:
            try:
                from ..perception import slot_roll as SR
                period = 180.0 if sr_symmetric else 360.0
                f, T_tp = slot_fused()
                slot_rep = {'fused': SR.public(f), 'plan': None, 'why_no_plan': ''}
                if f is not None:
                    e_tool0 = SR.entry_dir(T_tp, f['entry_deg'])
                    axis = T_clk[:3, 0]
                    alpha0 = SR.wrap(SR.angle_about_deg(T_tool0_engaged[:3, :3] @ e_tool0, axis), period)
                    slot_rep.update({'alpha0_deg': alpha0, 'entry_tool0': e_tool0.tolist(), 'axis_base': axis.tolist(),
                                     'socket_xyz_m': T_clk[:3, 3].tolist()})
                    pin, pin_src = sr_pin, 'slot_roll.pin_deg'
                    if pin is None:
                        got = SR.PinStore(sr_store_path).get(T_clk[:3, 3], sr_learn_move, period=period)
                        if got is not None:
                            pin, pin_src = got['pin_deg'], 'learned from %d drop-in%s (spread %.0f deg)' % (
                                got['n'], '' if got['n'] == 1 else 's', got['spread_deg'])
                    slot_rep.update({'pin_deg': pin, 'pin_source': pin_src if pin is not None else None})
                    if f['margin'] < sr_min_margin:
                        slot_rep['why_no_plan'] = 'the slot estimate is flat (margin %.2f < min_margin %.2f)' % (
                            f['margin'], sr_min_margin)
                    elif pin is None:
                        slot_rep['why_no_plan'] = ('the pins\' angle is unknown -- set slot_roll.pin_deg, or let a '
                                                   'drop-in teach it (slot_roll.learn)')
                    else:
                        th_star = SR.nearest_turn(alpha0, pin, np.degrees(cc_lo), np.degrees(cc_hi), period)
                        if th_star is None:
                            slot_rep['why_no_plan'] = 'no turn inside the band brings the entries onto the pins'
                        else:
                            legs_deg = SR.plan_legs(th_star, sr_step, sr_max, np.degrees(cc_lo), np.degrees(cc_hi),
                                                    period)
                            slot_rep['plan'] = {'theta_star_deg': th_star, 'legs_deg': legs_deg, 'step_deg': sr_step,
                                                'max_deg': sr_max, 'applied': sr_aim}
                            if sr_aim:
                                legs_run, tries_run = [np.radians(v) for v in legs_deg], len(legs_deg)
                            log.info('  SLOT-AIMED SEARCH%s: the entries at %+.0f deg about the socket axis (slot '
                                     'roll %.0f deg, margin %.2f, %d frame%s), the pins at %+.0f (%s) -> turn %+.0f '
                                     'deg; legs %s (from the engaged roll)%s.', '' if sr_aim else ' (RECORDED ONLY)',
                                     alpha0, f['roll_deg'], f['margin'], f['n_frames'],
                                     '' if f['n_frames'] == 1 else 's', pin, pin_src, th_star,
                                     [round(v) for v in legs_deg],
                                     '' if sr_aim else ' -- slot_roll.aim is false: the plain sweep %s runs' % (
                                         [round(float(np.degrees(eng_clock + t))) for t in cc_legs],))
                else:
                    slot_rep['why_no_plan'] = 'no slot estimate (not captured, or the slotted band not found)'
                if slot_rep['plan'] is None:
                    log.info('  slot-aimed search OFF for this trial: %s -- the plain sweep %s.',
                             slot_rep['why_no_plan'], [round(float(np.degrees(eng_clock + t))) for t in cc_legs])
                slot_obs['_clocking'] = {'clocking': slot_rep}
                _write_slot_json()
            except Exception as exc:                 # noqa: BLE001 -- the aim is a measurement, never a reason to stop
                log.warning('  slot-aimed search OFF: the plan failed (%s) -- the plain sweep.', exc)
                legs_run, tries_run, slot_rep = cc_legs, cc_tries, None

        det = _ScrewAdvance(robot, T_tool0_conn, T_clk, cc_need_m)
        combo = _AnyGuard(det, guard_cc)
        # ONE WARM-UP FOR THE WHOLE SWEEP, not one per leg: adm.ramp keeps its integrator across
        # calls, so resetting between legs would dump the axial deflection holding the connector
        # loaded and the press would rebuild on every reversal. Taring is once, here, for the same
        # reason.
        phase('connector_clock')
        if cc_null_on:
            adm_cc.set_frame(T_tool0_conn)          # the law at the connector, in its axes: y and z across it
        adm_cc.reset()
        adm_cc.warmup(ref_start, tare_fn=cc_tare)

        # THE WHOLE MANEUVER SAMPLED (clocking_trace.csv) AND DRAWN in the live viewer: the screw axis, the
        # socket's once probed, the connector's trace.
        def _draw_axis(items, text):
            from ..skills.pose6d_pick import send_overlay
            send_overlay(robot, 'clocking_axis', items, text)
        trace = _ClockTrace(robot, adm_cc, os.path.join(out_dir, 'clocking_trace.csv'), T_clk, T_tool0_conn,
                            every=ap_every, draw=_draw_axis)
        if sr_on:
            trace.drop_mm, trace.drop_window_deg = sr_drop_mm, sr_drop_win

        def ref_at(th_rel, push):
            """arm_at, the push it commands told to the trace (a drop-in is the plug catching up with it)."""
            trace.push_mm = push * 1000.0
            return arm_at(th_rel, push)
        th_at, push_at = 0.0, 0.0        # where the REFERENCE stands: roll from engaged, and push
        seated_in_probe = False

        # ---- THE AXIS PROBE: FIND THE SOCKET'S REAL AXIS BY TURNING (the user, 2026-11-06) -------------------
        # The screw turns about the axis the socket estimate gave -- 3.6 mm off the real one on 02:33, the plug
        # then dragged round it. With the side load nulled the plug stays on the REAL axis while the wrist turns,
        # so where the belief puts it traces p = B - R(theta) d (fit_axis_offset): B the real axis, d the in-hand
        # error. Turned there and back (turn_deg, from the engaged roll) under the clocking push, fitted, and --
        # trusted -- the screw axis moved onto B and the belief by d before the bayonet search. Recorded either way.
        if ap_on and not cc_null_on:
            log.warning('  AXIS PROBE skipped: it needs the side load nulled (connector_clocking.force_null with y '
                        'and z) -- without it the plug cannot follow the real axis, and there is nothing to fit.')
        elif ap_on:
            band_lo, band_hi = min(cc_lo, 0.0), max(cc_hi, 0.0)
            targets = []
            # TOWARD THE SLOTS' TURN (slot_roll.orient_probe): the probe's arc then searches too
            _flip = bool(slot_rep and slot_rep.get('plan') and slot_rep['plan'].get('applied') and sr_orient_probe
                         and slot_rep['plan']['theta_star_deg'] < 0)
            for v in ([-x for x in ap_turns] if _flip else ap_turns):
                t = np.radians(v)
                if not band_lo - 1e-9 <= t <= band_hi + 1e-9:
                    t = -t if band_lo - 1e-9 <= -t <= band_hi + 1e-9 else float(np.clip(t, band_lo, band_hi))
                targets.append(t)
            T_axis0, T_tc0 = T_clk.copy(), T_tool0_conn.copy()
            log.info('  --- AXIS PROBE --- turning %s deg from the engaged roll under the %.1f mm push, the side load '
                     'nulled: where the belief puts the connector traces the socket\'s REAL axis (and the in-hand '
                     'error); %s.', [round(float(np.degrees(t)), 1) for t in targets], cc_push_m * 1000.0,
                     'the screw axis moved onto it when trusted' if ap_apply else 'measured and recorded only')
            for j, th_to in enumerate(targets, 1):
                turn = th_to - th_at
                if abs(turn) < 1e-9:
                    continue
                trace.phase = f'probe {j}/{len(targets)}'
                combo.reset()
                _a, _pa = th_at, push_at
                res, f_done = screw_ramp(
                    adm_cc, lambda f, _s=_a, _e=th_to, _p=_pa: ref_at(_s + (_e - _s) * f, _p + (cc_push_m - _p) * f),
                    combo, cc_v, cc_w, abs(np.degrees(turn)),
                    label='axis probe %d/%d (%+.0f -> %+.0f deg) ' % (j, len(targets), np.degrees(eng_clock + _a),
                                                                       np.degrees(eng_clock + th_to)),
                    on_step=trace.step)
                th_at = _a + turn * f_done
                push_at = _pa + (cc_push_m - _pa) * f_done
                if res == 'seated':
                    if combo.tripped is det:
                        seated_in_probe = True
                        log.info('  axis probe: SEATED while probing (%s) -- no search needed.', det.tripped_by)
                    else:
                        log.warning('  axis probe: the force guard stopped it at %+.1f deg (%s) -- fitting what it '
                                    'saw.', np.degrees(eng_clock + th_at), combo.tripped_by)
                    break
            P = np.array(trace.probe, dtype=float).reshape(-1, 3)
            rep_ = {'targets_deg': [float(np.degrees(t)) for t in targets], 'n': int(len(P)),
                    'seated_in_probe': seated_in_probe, 'applied': False, 'why': '',
                    'T_axis_before': T_axis0.tolist(), 'T_tool0_conn_before': T_tc0.tolist()}
            why = None
            if len(P) < ap_min_pts:
                why = f'{len(P)} samples, under min_points {ap_min_pts}'
            else:
                B, dh, res_mm, cond = fit_axis_offset(P[:, 0], P[:, 1], P[:, 2])
                span = float(np.degrees(np.ptp(P[:, 0])))
                B_base = (T_axis0[:3, :3] @ np.array([0.0, B[0], B[1]])).tolist()
                rep_.update({'B_mm': B.tolist(), 'd_mm': dh.tolist(), 'B_base_mm': B_base, 'span_deg': span,
                             'residual_mean_mm': float(res_mm.mean()), 'residual_max_mm': float(res_mm.max()),
                             'cond': cond})
                if span < ap_min_span:
                    why = f'the turn spanned {span:.0f} deg, under min_span_deg {ap_min_span:.0f}'
                elif res_mm.mean() > ap_max_res:
                    why = f'the trace fits poorly ({res_mm.mean():.2f} mm mean, over max_residual_mm {ap_max_res:.2f})'
                elif np.linalg.norm(B) > ap_max_shift:
                    why = f'the axis would move {np.linalg.norm(B):.1f} mm, over max_shift_mm {ap_max_shift:.1f}'
                elif np.linalg.norm(dh) > ap_max_hand:
                    why = f'the in-hand error {np.linalg.norm(dh):.1f} mm is over max_hand_mm {ap_max_hand:.1f}'
                log.info('  axis probe: %d samples over %.0f deg -> the socket\'s axis %.2f mm from the screw axis '
                         '(y %+.2f, z %+.2f; base %+.1f, %+.1f, %+.1f mm), the in-hand error %.2f mm (y %+.2f, '
                         'z %+.2f); fit residual %.2f mm mean / %.2f max.', len(P), span, np.linalg.norm(B), B[0],
                         B[1], *B_base, np.linalg.norm(dh), dh[0], dh[1], res_mm.mean(), res_mm.max())
            if why is None and ap_apply and not seated_in_probe:
                # MOVED ONTO THE SOCKET'S AXIS: the screw axis through B, the belief by d -- and the lateral give
                # the null held handed to the reference in the same breath (zero_axes), so the command stays put.
                T_clk = T_axis0 @ translation_matrix([0.0, B[0] / 1000.0, B[1] / 1000.0])
                T_tool0_conn = T_tc0 @ translation_matrix([0.0, dh[0] / 1000.0, dh[1] / 1000.0])
                ref_start = T_clk @ inverse(T_tool0_conn)
                adm_cc.zero_axes([1, 2])
                peak = det.peak_m
                det = _ScrewAdvance(robot, T_tool0_conn, T_clk, cc_need_m)
                det.peak_m = peak
                combo = _AnyGuard(det, guard_cc)
                trace.axes = [(T_axis0, [0.55, 0.55, 0.55]), (T_clk, [0.2, 1.0, 0.3])]
                trace.check = (inverse(T_clk), T_tool0_conn)
                rep_.update({'applied': True, 'T_axis_after': T_clk.tolist(), 'T_tool0_conn_after': T_tool0_conn.tolist()})
                log.info('  AXIS PROBE APPLIED: the screw axis moved %.2f mm onto the socket\'s, the belief by %.2f mm '
                         '-- the bayonet search turns about it (and collar clocking after).', np.linalg.norm(B),
                         np.linalg.norm(dh))
            elif why is not None or not ap_apply:
                rep_['why'] = why or 'axis_probe.apply is false'
                (log.warning if why else log.info)('  axis probe NOT applied: %s -- the screw stays on the estimated '
                                                   'axis.', rep_['why'])
            trace.text = ['axis probe: ' + ('socket axis %.1f mm off, in-hand %.1f mm, residual %.2f mm: %s' % (
                np.linalg.norm(rep_['B_mm']), np.linalg.norm(rep_['d_mm']), rep_['residual_mean_mm'],
                'APPLIED' if rep_['applied'] else 'not applied (%s)' % rep_['why']) if 'B_mm' in rep_
                else 'not fitted (%s)' % rep_['why'])]
            trace.draw()
            try:
                with open(os.path.join(out_dir, 'axis_probe.json'), 'w') as fh:
                    json.dump(rep_, fh, indent=1, default=str)
            except OSError as exc:
                log.warning('axis_probe.json not written (%s)', exc)
            clock_rows.append({'maneuver': 'axis_probe', 'try': 1, 'ramp_result': 'seated' if seated_in_probe else '',
                               'success': bool(rep_['applied']), 'stopped_by': rep_['why'],
                               'null_y_mm': round(rep_.get('B_mm', [np.nan, np.nan])[0], 3),
                               'null_z_mm': round(rep_.get('B_mm', [np.nan, np.nan])[1], 3)})

        ok = swung = False
        ran = 0                          # legs that actually turned something
        if seated_in_probe:
            ok = True
        for k in (range(1, tries_run + 1) if not seated_in_probe else ()):
            th_to = legs_run[(k - 1) % len(legs_run)]
            turn = th_to - th_at
            if abs(turn) < 1e-9:
                log.info('  leg %d/%d: the reference is already at %+.1f deg -- nothing to turn.',
                         k, tries_run, np.degrees(eng_clock + th_to))
                continue
            ran += 1
            combo.reset()
            _a, _pa = th_at, push_at
            # SUBDIVIDED ON THE TRUE SCREW -- see screw_ramp. The push ramps to its full value
            # on the FIRST leg and stays there (_pa == cc_push_m for every leg after), so the
            # axial load is established once and carried through the reversals.
            trace.phase = f'leg {k}/{tries_run}'
            res, f_done = screw_ramp(
                adm_cc,
                lambda f, _s=_a, _e=th_to, _p=_pa: ref_at(_s + (_e - _s) * f,
                                                          _p + (cc_push_m - _p) * f),
                combo, cc_v, cc_w, abs(np.degrees(turn)),
                label='leg %d/%d (%+.0f -> %+.0f deg) ' % (
                    k, tries_run, np.degrees(eng_clock + _a), np.degrees(eng_clock + th_to)),
                on_step=trace.step)
            adv = det.advance_m()
            # `stopped`, not `seated`: ramp's 'seated' means "a guard tripped" and says nothing
            # about the assembly state. WHICH guard fired decides between the two outcomes.
            stopped = res == 'seated'
            early_seat = stopped and combo.tripped is det
            jammed = stopped and not early_seat
            swung = swung or res == 'done'          # at least one leg swung its whole arc
            # CONTINUE FROM WHERE THE REFERENCE ACTUALLY GOT TO -- a jammed leg stopped short,
            # and the next leg reverses from there, not from the endpoint it never reached.
            th_at = _a + turn * f_done
            push_at = _pa + (cc_push_m - _pa) * f_done
            null_mm = adm_cc.delta[1:3] * 1000.0
            null_deg = np.degrees(adm_cc.delta[4:6])
            give_rx = float(np.degrees(adm_cc.give()[3]))         # the screw's own roll compliance
            if adm_cc.selected[3]:
                log.info('  leg %d/%d: the roll about the screw axis gave %+.2f deg (rx compliance).', k, tries_run,
                         give_rx)
            if cc_null_on:
                _capped = bool(np.any(np.abs(adm_cc.delta[adm_cc.nulled])
                                      >= adm_cc.clamp[adm_cc.nulled] - 1e-6))
                (log.warning if _capped else log.info)(
                    '  leg %d/%d: nulling moved the connector y %+.2f, z %+.2f mm off the screw axis and tilted it '
                    'ry %+.2f, rz %+.2f deg%s.', k, tries_run, null_mm[0], null_mm[1], null_deg[0], null_deg[1],
                    ' -- AT ITS LIMIT (force_null.max_offset_mm / max_tilt_deg): more misalignment than it may take up'
                    if _capped else '')
            clock_rows.append({'maneuver': 'connector_clocking', 'try': k, 'ramp_result': res,
                               'null_y_mm': round(float(null_mm[0]), 3) if cc_null_on else '',
                               'null_z_mm': round(float(null_mm[1]), 3) if cc_null_on else '',
                               'null_ry_deg': round(float(null_deg[0]), 3) if cc_null_on else '',
                               'null_rz_deg': round(float(null_deg[1]), 3) if cc_null_on else '',
                               'give_rx_deg': round(give_rx, 3) if adm_cc.selected[3] else '',
                               'target_deg': round(float(np.degrees(eng_clock + th_to)), 3),
                               'reached_deg': round(float(np.degrees(eng_clock + th_at)), 3),
                               'advance_mm': round(adv * 1000.0, 3),
                               'peak_advance_mm': round(det.peak_m * 1000.0, 3),
                               'need_mm': round(cc_need_m * 1000.0, 3),
                               'success': bool(early_seat), 'force_stop': bool(jammed),
                               'state_after': 'seated' if early_seat else 'engaged',
                               'stopped_by': combo.tripped_by or ''})
            if early_seat:
                ok = True
                log.info('  leg %d/%d: SEATED at %+.1f deg -- early out: %s (advance %.2f mm)',
                         k, tries_run, np.degrees(eng_clock + th_at), det.tripped_by,
                         adv * 1000.0)
                break
            if jammed:
                log.warning('  leg %d/%d: the force guard stopped the turn at %+.1f deg of the '
                            '%+.1f deg aimed for (%s); advance %.2f mm -- the next leg reverses '
                            'from there.', k, tries_run, np.degrees(eng_clock + th_at),
                            np.degrees(eng_clock + th_to), combo.tripped_by, adv * 1000.0)
            else:
                log.info('  leg %d/%d: turned to %+.1f deg without the cams picking up '
                         '(advance %.2f mm of the %.2f mm needed).', k, tries_run,
                         np.degrees(eng_clock + th_at), adv * 1000.0, cc_need_m * 1000.0)
        nxt_cc = 'COLLAR CLOCKING' if cl_on else 'TUG / ESCAPE'
        if ok:
            verdict = 'early_out'                  # the advance threshold fired mid-leg
            _passed('CONNECTOR CLOCKING', '%s on leg %d/%d at %+.1f deg' % (
                det.tripped_by, ran, tries_run, np.degrees(eng_clock + th_at)), nxt_cc)
        else:
            # SUCCESS IS ASSUMED when the sweep runs -- see the docstring. Not-seated means EVERY
            # leg was cut short by the guard, i.e. a wedge rather than a search.
            ok = swung
            verdict = 'completed' if ok else 'all_legs_guard_stopped'
            if ok:
                _passed('CONNECTOR CLOCKING', 'at least one of %d leg%s swung its whole arc '
                        'without the force guard stopping it (success assumed; peak advance '
                        '%.2f mm of the %.2f mm early-out)' % (ran, '' if ran == 1 else 's',
                        det.peak_m * 1000.0, cc_need_m * 1000.0), nxt_cc)
            log.info('  sweep finished after %d leg%s: %s (peak advance %.2f mm, needed %.2f).',
                     ran, '' if ran == 1 else 's',
                     'SEATED -- at least one leg swung its whole arc, success assumed' if ok
                     else 'NOT SEATED -- every leg was stopped by the force guard',
                     det.peak_m * 1000.0, cc_need_m * 1000.0)
        # ONE SUMMARY ROW on top of the per-leg rows: when the sweep seats by COMPLETION rather
        # than by the early-out, every per-leg row reads success=False and nothing else in the CSV
        # carries the verdict.
        clock_rows.append({'maneuver': 'connector_sweep', 'try': ran, 'ramp_result': verdict,
                           'reached_deg': round(float(np.degrees(eng_clock + th_at)), 3),
                           'advance_mm': round(det.advance_m() * 1000.0, 3),
                           'peak_advance_mm': round(det.peak_m * 1000.0, 3),
                           'need_mm': round(cc_need_m * 1000.0, 3),
                           'success': bool(ok), 'force_stop': bool(not swung),
                           'state_after': 'seated' if ok else 'engaged',
                           'stopped_by': ''})

        # Settle at wherever the sweep actually ended -- ON THE LIVE REFERENCE, Delta intact.
        # `th_at`/`push_at` track where the REFERENCE stands, so arm_at() rebuilds it exactly.
        #
        # This block used to read `last = tool0(); reset(); hold(last)`, on the reasoning that
        # zeroing made the hold command "the pose the arm is AT rather than that pose plus the
        # deflection". It does -- for one cycle. Then the integrator re-derives the deflection
        # from the new zero and unloads it, which is the same rebound the engage settle had. Here
        # it is worse than cosmetic: this maneuver exists to PUSH +X and HOLD it (see the log
        # above), and unloading is precisely what it must not do.
        if trace.off:
            # THE PROBE CHECKED BY WHAT FOLLOWED: right, the plug turns ON the corrected axis -- where the belief
            # puts it stays near it through the search. Off by much more than the fit's residual: not right.
            _om, _ox = float(np.mean(trace.off)), float(np.max(trace.off))
            log.info('  axis probe, checked by the search: the connector ran %.2f mm (mean) / %.2f mm (max) off the '
                     'corrected axis over %d samples (the trace about the old one: up to %.1f mm).', _om, _ox,
                     len(trace.off), np.max(np.hypot(*np.array(trace.probe)[:, 1:].T)) if trace.probe else float('nan'))
            try:
                with open(os.path.join(out_dir, 'axis_probe.json')) as fh:
                    _r = json.load(fh)
                _r.update({'search_off_axis_mean_mm': _om, 'search_off_axis_max_mm': _ox, 'search_n': len(trace.off)})
                with open(os.path.join(out_dir, 'axis_probe.json'), 'w') as fh:
                    json.dump(_r, fh, indent=1, default=str)
            except (OSError, ValueError):
                pass
        ref_live_cc = arm_at(th_at, push_at)
        trace.phase = 'settle'
        if cc_settle > 0:
            adm_cc.hold(ref_live_cc, cc_settle, guard=None, on_step=trace.step)
        if cc_hold > 0:
            adm_cc.hold(ref_live_cc, cc_hold, guard=None, on_step=trace.step)
        adm_cc.stop()
        robot.arm.servo_stop()
        trace.close()
        if slot_rep is not None:
            try:
                _slot_outcome(slot_rep, trace.drop, ok, verdict)
            except Exception as exc:                     # noqa: BLE001 -- a record, never a reason to stop
                log.warning('  slot roll: the outcome not recorded (%s).', exc)

        # det.T_tool0_conn, NOT the local from the belief reset: the detector owns the
        # connector-in-gripper relation, and reading a stale copy here would hand collar clocking
        # a connector pose wrong by exactly whatever the detector has since accounted for.
        T_tool0_conn_now = det.T_tool0_conn
        T_base_conn = robot.tool0() @ T_tool0_conn_now
        got = matrix_to_xyzrpy(inverse(T_clk) @ T_base_conn)
        # WRAPPED ONTO THE SWEEP'S OWN BRANCH. The Euler roll caps at +/-180, and this number is
        # handed to collar clocking as the angle to UNWIND -- so a sign that flipped at the band
        # edge would orbit the open gripper the long way round the part. cc_mid is the centre of
        # the band the roll must lie in, which is the branch to read it on.
        screw_rad = _wrap_near(float(got[1][0]), cc_mid)
        log.info('  achieved: %+.2f mm along +X, %+.2f deg about it from the engaged roll -- so '
                 '%+.1f deg wrt the target frame (aimed for %s). Peak advance %.2f mm.',
                 got[0][0] * 1000.0, np.degrees(screw_rad),
                 np.degrees(eng_clock + screw_rad),
                 [round(float(np.degrees(t)), 1) for t in cc_sweep], det.peak_m * 1000.0)
        # RE-ANCHOR ON A SEAT THE SWEEP MADE. The belief reset at the top put the connector AT
        # the target as the sweep began -- true only if the engage had reached the mate. Started
        # on the pins (the collar off its slots: the case this maneuver exists for), it was short
        # by the slot depth, and the advance that then seated it read as the connector going that
        # far PAST the mate. Collar clocking, planned from that, reached for the ring 5 mm too
        # deep and closed the pads on the socket (found running on the simulated cell). A seat is
        # the one moment the connector's depth is known again: put it back at the target.
        if ok and abs(got[0][0]) > 0.5e-3:
            log.info('  re-anchoring the in-hand belief on the seat: the connector read %+.2f mm '
                     'along +X of the target, and seated it is AT the target.', got[0][0] * 1000.0)
            T_tool0_conn_now = T_tool0_conn_now @ translation_matrix([-got[0][0], 0.0, 0.0])
            T_base_conn = robot.tool0() @ T_tool0_conn_now
        if robot.arm.dry_run and not ok:
            # arm.fk is a fixed stand-in offline, so tcp_pose never moves and advance is
            # unmeasurable by construction -- pass it so the rest of the sequence is exercised.
            log.info('  dry run: advance is unmeasurable (tcp_pose is a fixed stand-in); '
                     'treating the screw as successful to exercise the rest of the sequence.')
            ok = True
        if cc_open_after:
            phase('retract')
            if not robot.gripper.open('release (post connector clocking)'):
                log.error('Gripper did not open after connector clocking.')
                return False, T_tool0_conn_now, T_base_conn
        return ok, T_tool0_conn_now, T_base_conn, float(np.degrees(screw_rad))

    def collar_clocking(T_base_conn, screw_deg=None, cl_rot=cl_rot, sp_on=sp_on,
                        unlocking=False):
        """COLLAR CLOCKING -- grasp the locking collar AXIALLY and twist the wrist.

        RUNS IN BOTH DIRECTIONS. With `unlocking` the identical approach is driven with the
        turn NEGATED and the seat push off, which is what disassembly uses: the station, the
        wall check, the wrist-3 budget and the IK scan are all the same geometry, so sharing
        them is what keeps the unlock landing on the ring the lock turned. The wrist-margin
        arithmetic below already reads the SIGN of the rotation (sorted() and max(0, -rot)),
        so a negative turn is budgeted correctly rather than accidentally.

        The gripper is brought onto the connector axis POINTING ALONG IT, fingers parallel to the
        cable, so tool0 sits on the axis 183 mm back and its Z is collinear with it. Turning the
        collar about that axis is then a rotation about tool0's own Z -- a WRIST TWIST, with the
        flange stationary.

        WHY, and the number that decides it: the socket is wall-mounted. The old radial grasp
        (fingertip frame ON the collar frame, jaws closing across the ring from the side) put
        tool0 183 mm OFF the axis at +25 mm past the mating face, and a 90 deg turn swept it
        through a 258 mm arc at that depth -- straight across the wall. Axially tool0 sits at
        -158 mm and travels 0 mm for the same turn. Same grasp on the ring (the jaws still close
        across a diameter), 183 mm more wall clearance, and no lateral sweep at all.

        WHAT IT COSTS: the approach is no longer a pure orbit from where the connector sweep left the
        arm. Radial and axial differ by a 90 deg pitch about the jaw-closing axis, which cannot be
        done with the open fingers still around the cable -- so the sequence retreats along the
        connector -X first, reorients in clear space, and only then advances back down the axis.
        Every station is measured along the axis and gated against `wall_standoff_mm`.

        THE CABLE THREADS THE JAW. Advancing along +X with the gripper on the axis runs the cable
        down between the open fingers. That is the intended motion, and it is the one part of this
        that geometry cannot verify: the advance is guarded so a snag stops it, but a cable stiff
        enough to hold itself off the axis will be pushed rather than threaded. Jog it once with
        the phase gate before trusting it.

        A force-guard stop during the TURN is NOT failure. A collar that has reached its lock
        stops turning, which is the intended end state and is indistinguishable here from jamming;
        the achieved angle is logged for the operator to judge.

        TODO: grasp verification and failure recovery on the close -- a missed collar currently
        turns an empty gripper."""
        # THE AXIS IS THE SOCKET'S, NOT THE ARM'S. T_base_conn is rebuilt at RUNTIME from
        # robot.tool0(), so it carries every deviation the sweep accumulated -- spring yield, an
        # advance that stopped short. Turning about it means turning about where the ARM ended up
        # rather than about the part.
        #
        # What the measurement DOES legitimately know is how far the bayonet cammed the connector
        # in along its own axis. Keep that (project the measured origin onto the true axis) and
        # discard the lateral and angular drift, which the socket forbids.
        axis = T_clk[:3, 0]
        axn = axis / float(np.linalg.norm(axis))
        cammed = float(np.dot(T_base_conn[:3, 3] - T_clk[:3, 3], axn))
        point = T_clk[:3, 3] + cammed * axn
        # COLLAR AXIS OFFSET (collar_clocking.axis_offset_mm, SOCKET-frame xyz). The declared
        # frame origin is the mating-face reference, not necessarily on the barrel centreline the
        # collar turns about. Shifts the LINE the whole maneuver works about; Y/Z move the line, X
        # only slides the reference point along it. Scoped to collar clocking -- the connector sweep
        # still turns about the unoffset axis. Resolved in the ROLL-FREE basis, see
        # axis_offset_base.
        off_conn = np.asarray(cl_axis_off, dtype=float) / 1000.0
        if float(np.linalg.norm(off_conn)) > 0.0:
            point = point + axis_offset_base()
            log.info('  collar axis OFFSET by %s mm (socket frame, NOT rolled by the %+.1f deg '
                     'engage clock angle) -> the line moves %.2f mm laterally.',
                     np.round(cl_axis_off, 2).tolist(), np.degrees(eng_clock),
                     float(np.linalg.norm(off_conn - np.dot(off_conn, [1.0, 0.0, 0.0])
                                          * np.array([1.0, 0.0, 0.0]))) * 1000.0)
        T_base_axis = T_clk.copy()
        T_base_axis[:3, 3] = point
        here = robot.tool0()

        # THE CONNECTOR MUST BE STRAIGHT IN THE SOCKET before anything is threaded down its axis.
        # This is the gate max_offaxis_tilt_deg now guards: the measured connector frame's tilt
        # from the socket axis. A cocked connector means the cable does not lie on the line the
        # gripper is about to advance along, so the jaw would meet it side-on instead of
        # swallowing it. (It used to gate the ARM's orientation, which only meant anything while
        # the approach was an orbit from wherever the sweep ended.)
        _d = T_base_conn[:3, 3] - point
        _lat = float(np.linalg.norm(_d - np.dot(_d, axn) * axn))
        _tilt = float(np.degrees(np.arccos(np.clip(
            abs(float(np.dot(T_base_conn[:3, 0] / np.linalg.norm(T_base_conn[:3, 0]), axn))),
            0.0, 1.0))))
        log.info('  axis: %s, advanced %+.2f mm by the sweep. The measured connector frame sits '
                 '%.2f mm lateral / %.2f deg tilted from it (gate %.1f deg).',
                 'SOCKET (target frame)' if pe_frame == 'target' else 'BELIEVED connector',
                 cammed * 1000.0, _lat * 1000.0, _tilt, cl_tilt_deg)
        if _tilt > cl_tilt_deg:
            log.error('COLLAR CLOCKING: the connector is %.1f deg off the socket axis, over the '
                      '%.1f deg gate (collar_clocking.max_offaxis_tilt_deg). The cable will not '
                      'lie on the line the gripper advances down -- refusing to thread it.',
                      _tilt, cl_tilt_deg)
            return False

        # THE COLLAR SITS ON THE CONNECTOR AXIS, collar_offset_mm along it from the connector
        # frame ORIGIN. Nothing here is back-calculated from how the part sits in the fingers: the
        # belief reset put the connector AT T_clk, so the ring's station is a property of the
        # CONNECTOR, and T_base_axis already carries that axis and the cammed origin.
        #
        # Deriving it through the FINGERTIP (collar = the live pad station + offset) is what put
        # the target 45.7 mm behind where it belongs: that 45.7 mm is where the PADS sit relative
        # to the mating face -- a fact about the grasp, not about the collar -- and it dragged the
        # ring's station with it.
        collar_x = cl_off_m
        # WHERE TO GRASP, as an ABSOLUTE roll about the socket +X (wrt the target frame), the same
        # convention connector_clocking.sweep_deg uses. The collar is a body of revolution, so any
        # angle grips the same ring -- what this decides is where the TURN starts and ends, and
        # therefore how much wrist range it needs. null = the engaged roll.
        #
        # This replaces prewind_deg. A pre-wind existed because the radial approach could only
        # reach the ring by orbiting from wherever the sweep left the arm, so the start angle was
        # whatever that orbit could afford. The axial approach is placed in free space, so the
        # start angle is simply stated.
        # ---- THE APPROACH, IN FOUR LEGS -------------------------------------------------------
        # 1. REALIGN with the ENGAGEMENT POSE -- the flange attitude the connector actually went
        #    in at, not the one the oscillating sweep happened to stop on.
        # 2. RETRACT `retract_mm` along the TARGET connector -X from there, straight away from the
        #    wall, the open fingers sliding ALONG the cable rather than across it.
        # 3. REORIENT onto the axis: tool0 +Z along the connector +X (so the fingertip axis IS the
        #    connector axis) and tool0 -Y on the connector -Z (the roll that leaves the wrist).
        # 4. ADVANCE back down the axis onto the collar, threading the cable into the open jaw.
        #
        # WHY LEG 1 EXISTS. Every station below is measured from where the arm stood when the
        # connector MATED. `robot.tool0()` at this point is wherever the last sweep leg stopped:
        # rolled by up to sweep_deg, and by a compliant stroke that yields to contact. Retracting
        # from THAT carries the sweep's attitude into the reorientation and makes the retract
        # distance mean something slightly different on every run. The engagement pose is a clean,
        # known attitude on the axis that the operator has already watched go in.
        #
        # IT IS RECONSTRUCTED, NOT STORED. connector_clocking captured the in-hand belief at
        # engagement (`_T_tool0_conn` = the connector expressed in tool0 THERE), and T_clk is that
        # same connector in base -- so the flange pose that produced it is T_clk @ inverse(belief).
        T_engaged = T_clk @ inverse(_T_tool0_conn)
        T_withdraw = translation_matrix(-cl_retract_m * axn) @ T_engaged
        ftip_p = (T_withdraw @ robot.T_tool0_fingertip)[:3, 3]

        _dp = ftip_p - point
        ftip_lat = float(np.linalg.norm(_dp - np.dot(_dp, axn) * axn))
        (log.info if ftip_lat < 1e-3 else log.warning)(
            '  fingertip sits %.2f mm off the connector axis while gripping. The axial '
            'pose puts it back ON the axis, so this is corrected rather than carried%s.',
            ftip_lat * 1000.0,
            '' if ftip_lat < 1e-3 else ' -- the pads close on a cable coaxial with the connector, '
            'so a non-zero value here is a FRAME error (compare frames.yaml fingertip against '
            'the connector frame), not a property of the part')

        # THE AXIAL POSE, BUILT DIRECTLY -- an END STATE, not a path.
        #
        # What this maneuver needs is: tool0 +Z along the connector +X, the fingertip ON the axis
        # at the collar station, so the jaws close across the ring and the turn becomes a wrist
        # twist. Constraining HOW the arm gets there -- one pitch about the connector -Y, pivoting
        # on the fingertip -- adds a requirement the frames need not satisfy. A rotation about a
        # single fixed axis carries z_now onto the connector axis ONLY when the two make the same
        # angle with that axis, and the attitude the sweep leaves has no reason to. A large
        # residual there is not a frame error; it is the path being over-specified.
        #
        # So state the target and let IK find the joints. The reorient is a move_j through free
        # space with the fingers open and the arm a retract's length back, so it is free to take
        # whatever combined rotation it likes -- nothing downstream depends on the route, only on
        # the pose it ends at, which is checked below.
        G_axial = (xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, -np.pi / 2.0, 0.0])
                   @ inverse(robot.T_tool0_fingertip))

        def at(station_m):
            """Fingertip at `station_m` along the connector axis, ON it, tool0 +Z along it."""
            return T_base_axis @ translation_matrix([station_m, 0.0, 0.0]) @ G_axial

        # Reorient at the station the withdraw reached, so the threading leg is exactly the
        # retract's length and no more.
        ftip_x = float((inverse(T_base_axis) @ T_withdraw @ robot.T_tool0_fingertip)[0, 3])
        T_reorient = at(ftip_x)
        adv_m = collar_x - ftip_x             # the guarded axial leg that threads the cable
        T_grip0 = at(collar_x)                # the grasp at roll 0

        # THE END STATE IS VERIFIED, since the path no longer proves it by construction. These are
        # exact by build, so a non-zero reading means T_base_axis or the fingertip frame is wrong.
        _rel = inverse(T_base_axis) @ T_grip0
        _zdot = float(np.dot(_rel[:3, 2], [1.0, 0.0, 0.0]))
        _ftip_off = float(np.linalg.norm(
            (inverse(T_base_axis) @ T_grip0 @ robot.T_tool0_fingertip)[1:3, 3])) * 1000.0
        log.info('  axial pose: tool0 +Z . connector +X = %.6f, tool0 %.4f mm off the axis, '
                 'fingertip %.4f mm off it, %.1f mm to thread.',
                 _zdot, float(np.linalg.norm(_rel[1:3, 3])) * 1000.0, _ftip_off, adv_m * 1000.0)
        if abs(_zdot - 1.0) > 1e-6 or _ftip_off > 1e-3:
            log.error('COLLAR CLOCKING: the axial grasp pose does not come out axial -- tool0 +Z '
                      'is %.3f deg off the connector +X and the fingertip %.3f mm off the axis. '
                      'That is a FRAME problem (frames.yaml fingertip, or the connector frame '
                      'rpy), not a reachability one.',
                      np.degrees(np.arccos(np.clip(_zdot, -1.0, 1.0))), _ftip_off)
            return False

        # ---- THE GRASP CLOCK ANGLE, and why it is worth solving for ---------------------------
        # The reorientation from the lift-off pose (radial, where the sweep left the wrist) to the
        # axial retreat pose is the largest single rotation in the app, and its size depends
        # ENTIRELY on this angle: measured on the shipped fixture it runs 180 deg at 0 and 90 deg
        # at 180. A 180 deg tool reorientation is where the analytic IK's nearest-branch answer
        # stops being reachable from the seed -- which is what "the axial retreat station is
        # unreachable" looks like.
        #
        # The collar is a body of revolution, so this angle is FREE: any of them grips the same
        # ring. So spend it. null = pick the cheapest angle that actually solves; a number pins it
        # (absolute, wrt the target frame, like connector_clocking.sweep_deg).
        #
        # The tool0 station along the axis does NOT depend on the angle -- every candidate is a
        # rotation ABOUT the axis -- so this cannot trade wall clearance for reach.
        def roll(T, th):
            """`T` rolled `th` about the connector axis. The collar is a body of revolution, so
            this is the one freedom the pitch leaves and the only thing the prewind spends."""
            return rotate_about_axis(T, axis, point, th)

        def _reachable(th):
            """True when the reorient, grasp and turn-end all IK-solve at roll `th`, ON ONE BRANCH:
            the grasp from the reorient's solution, the turn-end from the grasp's, joints A1..A6
            (all but the tool roll) within 10 deg of each other between grasp and turn-end, and the
            tool roll turning by the turn itself -- not a wrist flip, not the long way round."""
            _grip = roll(T_grip0, th)
            q_r = robot.arm.ik(roll(T_reorient, th), seed_c)
            q_g = None if q_r is None else robot.arm.ik(_grip, q_r)
            q_e = None if q_g is None else robot.arm.ik(
                translation_matrix(cl_push_m * axn) @ rotate_about_axis(_grip, axis, point, cl_rot), q_g)
            if q_e is None:
                return False
            q_g, q_e = np.asarray(q_g, dtype=float), np.asarray(q_e, dtype=float)
            j = robot.arm.tool_roll_joint
            others = [i for i in range(len(q_g)) if i != j]
            return (float(np.max(np.abs(q_e[others] - q_g[others]))) < np.radians(10.0)
                    and abs(abs(q_e[j] - q_g[j]) - abs(cl_rot)) < np.radians(5.0))

        def _nearest_reachable(th0):
            """The roll nearest `th0` (5 deg steps, a turn either way) at which _reachable holds, or
            None. The window above is A7's at the GRASP station; the reorient station, 150 mm back,
            has other A1..A6 and so another A7 for the same roll -- on 2026-11-04 the window's edge
            (a pinned +45 deg clamped to -246.8) put the reorient past A7's limit: "no IK"."""
            for k in range(0, 73):
                for t in ((th0,) if k == 0 else (th0 + np.radians(5.0 * k), th0 - np.radians(5.0 * k))):
                    if _reachable(t):
                        return float(t)
            return None

        seed_c = robot.arm.ik(T_withdraw, robot.arm.q()) or robot.arm.q()

        # ---- THE CLOCK ANGLE IS A WRIST-3 OFFSET, EXACTLY -------------------------------------
        # After the pitch, tool0 stands ON the collar axis with its Z collinear (the fingertip
        # frame is a pure 183 mm translation along tool0 Z, so nothing puts the flange off the
        # line). Rolling the grasp about that axis therefore leaves the FLANGE ORIGIN fixed and
        # turns it about its own Z, which is the joint-6 axis. Checked numerically across the
        # band: the tool0 origin stays 0.0000 mm off the line and tool0_Z . axis = 1.000000.
        #
        # Therefore JOINTS 1..5 ARE IDENTICAL AT EVERY CLOCK ANGLE, and q6(th) = q6(0) + th. The
        # station picks the first five joints; the clock angle picks the sixth and nothing else.
        #
        # That turns the angle from something to search for into something to CALCULATE -- and
        # into a PREWIND, because what the turn actually needs is wrist_3 range: rotation_deg of
        # it, from wherever the grasp starts. Spending that range is the whole job of this angle,
        # and it has to be spent BEFORE the arm threads itself down the axis. Discovering it
        # afterwards means finding out with the collar clamped in the fingers.
        #
        # (The old code scanned 24 candidate angles at 3 IK calls each -- 72 calls to explore a
        # family that is one joint. Two calls answer it, and a 15 deg grid could miss a feasible
        # window narrower than its own step.)
        q_grip0 = robot.arm.ik(T_grip0, seed_c)
        if q_grip0 is None:
            log.error('COLLAR CLOCKING: the collar grasp station does not IK-solve at all. That '
                      'is joints 1-5, i.e. REACH -- no clock angle can help, because the angle '
                      'only moves wrist_3. Reduce collar_clocking.retreat_mm (now %.0f mm) or '
                      'move the fixture.', cl_retreat_m * 1000.0)
            return False

        # THE REORIENT STATION'S REACH, before any roll is searched (2026-11-05 23:47: every roll failed
        # there, and the search below took 17 s and 54 "No IK" lines to find that out). Its joints
        # A1..A6 do not depend on the roll either -- the roll turns only the tool-roll joint -- so four
        # rolls a quarter-turn apart answer it. None solving is REACH (on the iiwa: the wrist centre
        # nearer the shoulder than the elbow folds) and the knob is how far back it is: retract_mm.
        def _solves_some_roll(T):
            with _quiet_ik():
                return any(robot.arm.ik(roll(T, np.radians(d)), seed_c) is not None for d in (0, 90, 180, 270))

        if not _solves_some_roll(T_reorient):
            fits = next((b for b in np.arange(adv_m - 0.010, 0.0199, -0.010)
                         if _solves_some_roll(at(collar_x - b))), None)
            log.error('COLLAR CLOCKING: the reorient station -- the gripper pitched onto the axis %.0f mm back '
                      'from the collar -- does not IK-solve at ANY roll about the axis. That is REACH, not the '
                      'grasp angle (the roll turns only %s): %s', adv_m * 1000.0,
                      robot.arm.joint_names[robot.arm.tool_roll_joint],
                      ('it reaches from %.0f mm back, so set assembly.collar_clocking.retract_mm to %.0f or less '
                       '(now %.0f).' % (fits * 1000.0, (cl_retract_m - (adv_m - fits)) * 1000.0,
                                        cl_retract_m * 1000.0)) if fits is not None else
                      'no shorter threading leg reaches either -- move the fixture.')
            return False

        # THE USABLE WRIST-3 RANGE, asked of the ARM at that exact configuration rather than
        # assumed (Arm.tool_roll_window: the controller's limit on the UR, A7's on the iiwa).
        j_roll = robot.arm.tool_roll_joint
        w_lo, w_hi = robot.arm.tool_roll_window(q_grip0)
        need = abs(cl_rot) + 2.0 * cl_w3_margin
        if (w_hi - w_lo) < need:
            log.error('COLLAR CLOCKING: wrist_3 has %.0f deg of range in this configuration '
                      '(%+.0f .. %+.0f deg) but the turn needs %.0f (rotation_deg %.0f plus '
                      '2 x %.1f deg of margin). NO grasp angle can fit it -- lower '
                      'collar_clocking.rotation_deg.',
                      np.degrees(w_hi - w_lo), np.degrees(w_lo), np.degrees(w_hi),
                      np.degrees(need), np.degrees(cl_rot), np.degrees(cl_w3_margin))
            return False

        # THE FEASIBLE CLOCK-ANGLE WINDOW. The turn runs q6 -> q6 + rotation_deg, so the START
        # must sit a full rotation inside whichever end it travels toward -- which is what makes
        # this a prewind rather than a reachability check.
        th_lo = (w_lo + cl_w3_margin) - q_grip0[j_roll] + max(0.0, -cl_rot)
        th_hi = (w_hi - cl_w3_margin) - q_grip0[j_roll] - max(0.0, cl_rot)

        # ---- THE ROLL: tool0 -Y ONTO THE CONNECTOR -Z -----------------------------------------
        # G_axial fixes two of the three rotational freedoms -- tool0 +Z along the connector +X --
        # and the roll ABOUT that axis is the third. The collar is a body of revolution, so every
        # roll grips the same ring; what the angle decides is the attitude the gripper arrives in
        # and how much wrist_3 the turn has left.
        #
        # Spend it on tool0 -Y || connector -Z. That is the same rule the reorientation before the
        # sweep uses, so the wrist arrives the way up the operator has already seen, and on the
        # shipped fixture it is also the CHEAPER reorientation (about 90 deg of tool swing against
        # 180 at zero roll -- and a 180 deg tool reorientation is where the analytic IK's
        # nearest-branch answer stops being reachable from the seed).
        #
        # CLOSED FORM, not a search. tool0 +Y at zero roll is perpendicular to the connector axis
        # by construction (it is a column of a rotation whose +Z IS the axis), so a single atan2
        # gives the angle about +X that carries it onto the connector +Z exactly. On the shipped
        # frames the build puts tool0 -Y on the connector +Z -- the wrong way up -- and this comes
        # out at 180 deg. _fit_turn below still shifts it by whole turns onto a wrist_3 branch
        # with room for the turn.
        #
        # A pinned collar_clocking.grasp_clock_deg overrides, stated as an ABSOLUTE roll about the
        # socket +X wrt the target frame (the convention connector_clocking.sweep_deg uses), for
        # when the attitude must be fixed rather than derived.
        v0 = G_axial[:3, 1]                        # tool0 +Y at zero roll, in connector coords
        want = np.array([0.0, 0.0, 1.0])           # connector +Z -- tool0 -Y then lands on the -Z
        th_want = float(np.arctan2(float(np.dot([1.0, 0.0, 0.0], np.cross(v0, want))),
                                   float(np.dot(v0, want))))
        pinned = cl_grasp_clock is not None
        if pinned:
            # _fit_turn already shifts by whole turns, so no branch-wrapping is needed here.
            th_now = float(matrix_to_xyzrpy(inverse(T_clk) @ T_reorient)[1][0])
            th_want = cl_grasp_clock - eng_clock - th_now
        else:
            log.info('  roll %+.1f deg about the connector axis lays tool0 -Y on the connector -Z '
                     '(collar_clocking.grasp_clock_deg is null, so it is derived).',
                     np.degrees(th_want))
        th_grasp, clamped = _fit_turn(th_want, th_lo, th_hi)

        q6_start = q_grip0[j_roll] + th_grasp
        log.info('--- WRIST-3 PREWIND --- grasp at %+.1f deg wrt the target frame puts wrist_3 at '
                 '%+.1f deg, ending at %+.1f after the %+.1f deg turn. The controller allows '
                 '%+.1f .. %+.1f deg; margin %.1f deg at each end.',
                 np.degrees(eng_clock + th_grasp), np.degrees(q6_start),
                 np.degrees(q6_start + cl_rot), np.degrees(cl_rot),
                 np.degrees(w_lo), np.degrees(w_hi), np.degrees(cl_w3_margin))
        if clamped:
            log.warning('  the %s roll (%+.1f deg) leaves no room for the turn on ANY whole-turn '
                        'branch, so it was CLAMPED to %+.1f deg. The ring gripped is the same; '
                        'the approach attitude is not -- check the reorientation looks sane.',
                        'pinned collar_clocking.grasp_clock_deg' if pinned
                        else 'derived (tool0 -Y on the connector -Z)',
                        np.degrees(th_want), np.degrees(th_grasp))
        elif abs(th_grasp - th_want) > np.radians(0.5):
            log.info('  (rolled %+.0f deg off the -Y/-Z attitude onto a wrist_3 branch with '
                     'room -- the collar is a body of revolution, so the grasp on the ring is '
                     'identical, but the gripper arrives turned by that much.)',
                     np.degrees(th_grasp - th_want))

        # WRIST_3 now has room. Whether the STATIONS solve in joints 1..5 is a different question,
        # and the only one left -- so a failure here names the knob that actually moves it.
        with _quiet_ik():
            _ok_grasp = _reachable(th_grasp)
        if not _ok_grasp:
            log.info('  the %+.1f deg roll does not solve at every station: searching the rolls either side '
                     '(5 deg steps).', np.degrees(eng_clock + th_grasp))
            with _quiet_ik():
                th_found = _nearest_reachable(th_grasp)
            if th_found is None:
                log.error('COLLAR CLOCKING: no roll about the axis has the reorient, grasp and turn-end '
                          'IK-solving on one branch with the %+.0f deg turn inside A7. That is REACH -- '
                          'the knob is collar_clocking.retract_mm (now %.0f mm), rotation_deg or the '
                          'fixture position.', np.degrees(cl_rot), cl_retract_m * 1000.0)
                return False
            log.warning('  the %+.1f deg roll does not solve at every station (the reorient, %.0f mm back, '
                        'has its own A7); the nearest that does is %+.1f deg wrt the target frame (%+.1f '
                        'from the wanted attitude). The ring gripped is the same.',
                        np.degrees(eng_clock + th_grasp), cl_retract_m * 1000.0,
                        np.degrees(eng_clock + th_found), np.degrees(th_found - th_want))
            th_grasp = th_found
            _qg = robot.arm.ik(roll(T_grip0, th_grasp), robot.arm.ik(roll(T_reorient, th_grasp), seed_c))
            q6_start = float(_qg[j_roll]) if _qg is not None else q6_start
        T_retreat = roll(T_reorient, th_grasp)      # backed off, already pitched onto the axis
        T_grip = roll(T_grip0, th_grasp)            # fingertip ON the collar
        T_end = translation_matrix(cl_push_m * axn) @ rotate_about_axis(
            T_grip, axis, point, cl_rot)
        log.info('--- COLLAR CLOCKING (axial) --- collar %.1f mm along the connector +X from the '
                 'ORIGIN, on the axis. Realign with the engagement pose, retract %.0f mm along '
                 'the connector -X, reorient onto the axis (tool0 +Z on the connector +X, tool0 '
                 '-Y on the -Z), advance %+.1f mm down the axis onto the ring (the fingertip '
                 'lands at the %+.1f mm station), grasp, then TWIST the wrist %+.1f deg while '
                 'pushing %+.1f mm.',
                 cl_off_m * 1000.0, cl_retract_m * 1000.0, adv_m * 1000.0, ftip_x * 1000.0,
                 np.degrees(cl_rot), cl_push_m * 1000.0)

        # ---- WALL CLEARANCE, checked before anything moves ------------------------------------
        # The socket is wall-mounted, so the number that matters is how far along the connector +X
        # the FLANGE gets: +X points into the wall, and tool0 is the bulkiest thing on the arm.
        # Every pose below is a planned station on the axis, so the whole maneuver's approach to
        # the wall is known up front rather than discovered by driving into it.
        stations = (('withdraw', T_withdraw), ('reorient', T_retreat),
                    ('collar grasp', T_grip), ('turn end', T_end))
        x_tool = {lab: float((inverse(T_clk) @ T)[0, 3]) * 1000.0 for lab, T in stations}
        worst_lab = max(x_tool, key=x_tool.get)
        log.info('  tool0 station along the connector +X (larger = closer to the wall): %s. '
                 'Closest: %s at %+.1f mm.',
                 ', '.join('%s %+.1f' % (k, v) for k, v in x_tool.items()),
                 worst_lab, x_tool[worst_lab])
        log.info('  (the sweep left the arm at %+.1f mm and the engagement pose is at %+.1f mm '
                 '-- both are INHERITED, poses the arm has already occupied rather than stations '
                 'chosen here, so they are reported and not gated.)',
                 float((inverse(T_clk) @ here)[0, 3]) * 1000.0,
                 float((inverse(T_clk) @ T_engaged)[0, 3]) * 1000.0)
        if cl_wall_mm is not None and x_tool[worst_lab] > cl_wall_mm:
            log.error('COLLAR CLOCKING: the %s pose puts tool0 at %+.1f mm along the connector '
                      '+X, past the %+.1f mm wall standoff (collar_clocking.wall_standoff_mm). '
                      'Refusing to move toward the wall.',
                      worst_lab, x_tool[worst_lab], cl_wall_mm)
            return False
        # REACHABILITY of every station, before anything grips: discovering mid-turn that the far
        # end is unreachable leaves the collar clamped in a stalled gripper.
        for lab, T_chk in stations[2:]:
            if robot.arm.ik(T_chk, robot.arm.q()) is None:
                log.error('COLLAR CLOCKING: the %s pose is unreachable. The twist needs %.0f deg '
                          'of wrist range from a grasp at %+.1f deg; adjust '
                          'collar_clocking.rotation_deg or grasp_clock_deg.', lab,
                          np.degrees(cl_rot), np.degrees(eng_clock + th_grasp))
                return False

        # ---- SEAT PUSH, done WHERE THE ARM ALREADY IS -----------------------------------------
        # Re-grip the junction and PRESS the connector deeper along +X until force_n is SUSTAINED
        # for persistence_s; the guard TRIP is the SUCCESS. Completing the travel without ever
        # building the force means it slid in freely, which also ends deeper.
        #
        # It runs FIRST, before the retreat, because the pads are already around the cable at the
        # junction -- exactly where the push wants them. (The radial sequence had to centre on the
        # axis first for the collar's sake and pressed from there, a few mm off where the grasp
        # actually held.) The press moves the CONNECTOR, so the collar stations are shifted by the
        # measured travel afterwards: the ring rides the connector.
        d_push = 0.0
        if sp_on:
            # NO RE-GRIP. The sweep left the pads closed on the connector at the junction --
            # exactly where the push wants them -- so opening and re-closing here would only
            # release a part that is already held, risk a worse bite, and cost the grasp check
            # a chance to abort a run that is going fine. connector_clocking.open_gripper_after
            # must therefore be FALSE; the release happens after the push instead, because the
            # retreat and the advance are the legs that genuinely need open fingers.
            if True:
                T_a = robot.tool0()
                # The spring must STRETCH force/S to apply force_n, so the reference travel has
                # to cover that stretch on top of any real seating motion.
                u_p = T_a[:3, :3].T @ axn
                S_p = 1.0 / float(np.sum((u_p ** 2) / adm_cl.S[:3]))
                if sp_travel_m < sp_force / S_p:
                    log.warning('SEAT PUSH: max_travel_mm (%.1f) cannot stretch the %.0f N/m '
                                'spring to %.1f N (needs %.1f mm) -- the push cannot reach its '
                                'force.', sp_travel_m * 1000.0, S_p, sp_force,
                                sp_force / S_p * 1000.0)
                T_b = translation_matrix(sp_travel_m * axn) @ T_a
                adm_cl.reset()
                adm_cl.warmup(T_a, tare_fn=tare)      # tare while gripping and static
                guard_push.reset()
                res_p = adm_cl.ramp(T_a, T_b, seg_time(T_a, T_b), guard_push)
                if res_p != 'seated':
                    # the reference has stopped; give the guard its full persistence window
                    res_p = adm_cl.hold(T_b, sp_persist + 0.5, guard_push)
                d_push = float(np.dot(robot.tool0()[:3, 3] - T_a[:3, 3], axn))
                adm_cl.reset()
                adm_cl.stop()
                robot.arm.servo_stop()
                pressed = res_p == 'seated'
                clock_rows.append({'maneuver': 'seat_push', 'try': 1, 'ramp_result': res_p,
                                   'advance_mm': round(d_push * 1000.0, 3),
                                   'success': True, 'force_stop': bool(pressed),
                                   'state_after': 'seated',
                                   'stopped_by': guard_push.tripped_by or ''})
                (log.info if pressed else log.warning)(
                    'SEAT PUSH: %s -- connector moved %+.2f mm.',
                    'held %.1f N for %.1f s (pressed home)' % (sp_force, sp_persist) if pressed
                    else 'never built %.1f N over %.1f mm of reference travel'
                         % (sp_force, sp_travel_m * 1000.0),
                    d_push * 1000.0)
                _passed('SEAT PUSH', (guard_push.tripped_by or 'force %.1f N held %.2f s'
                                      % (sp_force, sp_persist)) if pressed
                        else 'the %.1f mm reference travel ran out without %.1f N (the push '
                             'does not block the run)' % (sp_travel_m * 1000.0, sp_force),
                        'COLLAR %s' % ('UNLOCK' if unlocking else 'GRASP AND TURN'))
                if not robot.gripper.open('release (seat push)'):
                    log.error('SEAT PUSH: gripper did not release -- cannot retreat with the '
                              'junction clamped.')
                    return False
                if abs(d_push) > 1e-6:
                    # the connector (and its collar) moved deeper -- keep the ring in the sights.
                    # T_engaged and T_withdraw ride along too: the realign is what the retract is
                    # measured from, so it has to track the PART rather than the history.
                    T_engaged = translation_matrix(d_push * axn) @ T_engaged
                    T_withdraw = translation_matrix(d_push * axn) @ T_withdraw
                    T_retreat = translation_matrix(d_push * axn) @ T_retreat
                    T_grip = translation_matrix(d_push * axn) @ T_grip
                    T_end = translation_matrix(d_push * axn) @ T_end

        # PACED BY ITS OWN PHASE SCALE. The two legs below are FREE SPACE -- the fingers are
        # open and clear, nothing is being inserted -- but they used to run at the `standoff`
        # scale, which is contact-approach pacing (the global 25 mm/s / 30 deg/s at 1.0x). The
        # reorient is the longest single move in the app: a ~90 deg swing plus a few hundred mm of
        # travel, and at 25 mm/s the translation cap alone puts it near arm.move_timeout_s.
        # `collar_approach` exists so it can be paced like the escape it is. The axial ADVANCE
        # below deliberately does NOT use it -- that one threads the cable and stays slow.
        phase('collar_approach')
        # ---- 1. REALIGN WITH THE ENGAGEMENT POSE ----------------------------------------------
        # Undo the sweep. The difference between where the last leg stopped and the engagement
        # pose is essentially a roll about the connector axis, and a straight move_l would cut its
        # chord -- at 60 deg with tool0 183 mm off the axis the fingertip dips about 25 mm off
        # true, which with OPEN fingers around the cable is a swipe rather than a slide.
        #
        # So the path is parametrised BY THE FINGERTIP, not by the flange: the fingertip walks the
        # straight line between its two stations (both ON the axis, differing only by whatever the
        # seat push drove in) while the attitude slerps, and the gripper body swings around the
        # cable instead of through it. Compliant and guarded, like every other leg with something
        # in front of it.
        _now = robot.tool0()
        _ft_off = robot.T_tool0_fingertip[:3, 3]
        _p0 = (_now @ robot.T_tool0_fingertip)[:3, 3]
        _p1 = (T_engaged @ robot.T_tool0_fingertip)[:3, 3]

        def _to_engaged(f):
            """tool0 at fraction `f` of the realign, with the FINGERTIP on the straight line
            between its start and end stations."""
            T = slerp_matrix(_now, T_engaged, f)
            T[:3, 3] = (_p0 + f * (_p1 - _p0)) - T[:3, :3] @ _ft_off
            return T

        _re_lin, _re_ang = pose_error(_now, T_engaged)
        if unlocking:
            # UNLOCKING DOES NOT REALIGN. The realign exists to undo the bayonet sweep so the
            # collar stations are measured from the mated pose -- but on the way OUT the
            # connector is already mated and the arm is already clear of it (the escape ran
            # before this), so driving the OPEN gripper back onto the connector would be a
            # pointless approach to a pose we only want to leave again. The collar is turned
            # from wherever it sits: the station below is absolute (built from the axis
            # frame), so the approach still lands on the ring without it.
            log.info('  UNLOCKING -- skipping the realign with the engagement pose (%.1f deg / '
                     '%.1f mm away); the collar station is absolute, so the approach does not '
                     'need it.', np.degrees(_re_ang), _re_lin * 1000.0)
        elif _re_ang > np.radians(0.5) or _re_lin > 1e-4:
            adm_cl.reset()
            adm_cl.warmup(_now)
            guard_shared.reset()
            res_re, f_re = screw_ramp(adm_cl, _to_engaged, guard_shared, cl_v, cl_w,
                                      float(np.degrees(_re_ang)),
                                      label='realign with the engagement pose ')
            adm_cl.reset()
            adm_cl.stop()
            robot.arm.servo_stop()
            if res_re == 'seated':
                log.error('COLLAR CLOCKING: the force guard tripped (%s) %.0f%% of the way through '
                          'the %.1f deg realign with the engagement pose. The fingers are OPEN and '
                          'still around the cable, so nothing is clamped -- something is fouling '
                          'the swing.', guard_shared.tripped_by or 'unknown', f_re * 100.0,
                          np.degrees(_re_ang))
                return False
            log.info('  realigned with the engagement pose (%.1f deg, %.1f mm of flange travel).',
                     np.degrees(_re_ang), _re_lin * 1000.0)
        else:
            log.info('  already at the engagement pose (%.2f deg, %.2f mm) -- nothing to realign.',
                     np.degrees(_re_ang), _re_lin * 1000.0)

        # ---- 2. RETRACT ALONG THE CABLE, straight away from the wall --------------------------
        # A pure translation along the connector -X with the open fingers still around the cable:
        # they slide ALONG it rather than across it, so nothing is swept, and every millimetre is
        # away from the wall. It also puts the reorient that follows as far from the wall as the
        # maneuver ever gets.
        if cl_retract_m > 1e-6:
            if not _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                    T_withdraw, label='collar retract (connector -X)')):
                log.error('Could not retract %.0f mm along the connector -X from the engagement '
                          'pose (collar_clocking.retract_mm).', cl_retract_m * 1000.0)
                return False

        # ---- 3. REORIENT ONTO THE AXIS ---------------------------------------------------------
        # One move_j to the stated end pose. A joint move rather than a straight line because it
        # is the only large reorientation in the maneuver, and it happens a retract's length back
        # with the fingers clear, so the route it takes does not matter -- only the pose it
        # reaches, which was verified above.
        #
        # TWO FAILURES, REPORTED APART. "Could not reach" used to cover both an unreachable pose
        # and a move that started and did not finish, which are opposite problems: one is fixed by
        # moving the station, the other by pacing or the pendant.
        q_ret = robot.arm.ik(T_retreat, robot.arm.q())
        if q_ret is None:
            # UNREACHABLE. Say so, and say which retract distances DO solve -- that turns a dead
            # end into a number to put in the config. (IK is seeded from the current joints, so
            # this also catches a branch the arm cannot get to from where it stands.)
            ok_mm = [d for d in (100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0)
                     if robot.arm.ik(
                         translation_matrix((cl_retract_m - d / 1000.0) * axn) @ T_retreat,
                         robot.arm.q()) is not None]
            log.error('COLLAR CLOCKING: the pitched (axial) pose is UNREACHABLE -- no IK solution '
                      'with the fingertip at %+.1f mm along the connector +X (tool0 %+.1f mm, on '
                      'the axis, pointing at the socket). This is REACH or a joint limit, not '
                      'speed: collar_clocking.retract_mm (%.0f) is what puts the flange that far '
                      'back. %s',
                      ftip_x * 1000.0, x_tool['reorient'], cl_retract_m * 1000.0,
                      ('Retract distances that DO solve from here: %s mm.'
                       % ', '.join('%.0f' % d for d in ok_mm)) if ok_mm else
                      'NO retract distance from 100 to 500 mm solves -- the axial orientation '
                      'itself is out of reach at this fixture pose, not just the distance.')
            return False
        if not _guarded(robot, guard_shared, lambda: robot.arm.move_j(
                q_ret, label='collar pitch onto the axis')):
            log.error('COLLAR CLOCKING: the pitch onto the axis did not finish. The pose IS '
                      'reachable (IK solved), so this is the force guard tripping on the way, the '
                      'controller rejecting the move, or arm.move_timeout_s (%.0f s) running out '
                      '-- raise speed.phase_scale.collar_approach (now %.2fx) or move_timeout_s.',
                      robot.arm.move_timeout, float(scales.get('collar_approach', 1.0)))
            return False

        # ---- 4. ADVANCE DOWN THE AXIS, threading the cable into the open jaw ------------------
        # A PURE TRANSLATION along the connector +X: no rotation, so a straight move is exactly
        # the right path and there is no chord to cut. Guarded, because this is the leg that runs
        # the cable between the open fingers -- a snag must stop it rather than push through.
        # Compliant, for the same reason, and back on the SLOW standoff pacing: this is the one
        # leg of the approach with something in front of it.
        phase('standoff')
        adm_cl.reset()
        adm_cl.warmup(T_retreat)
        guard_shared.reset()
        res_adv = adm_cl.ramp(T_retreat, T_grip,
                              seg_time(T_retreat, T_grip, g_v * s_std, g_w * s_std),
                              guard_shared)
        adm_cl.stop()
        robot.arm.servo_stop()
        if res_adv == 'seated':
            log.error('COLLAR CLOCKING: the force guard tripped during the axial advance (%s) '
                      'after %.1f of %.1f mm. The cable most likely did not thread into the open '
                      'jaw -- the fingers are still clear of the ring, so nothing is clamped.',
                      guard_shared.tripped_by or 'unknown',
                      float(np.dot(robot.tool0()[:3, 3] - T_retreat[:3, 3], axn)) * 1000.0,
                      adv_m * 1000.0)
            return False

        # ---- DID THE PREWIND SURVIVE? --------------------------------------------------
        # The prewind was computed at the PLANNED grip pose; the arm reached the real one through
        # a compliant advance that yields to contact, so wrist_3 is where the servo left it, not
        # necessarily where the plan put it. Checked HERE, with the fingers still open and the
        # collar not yet clamped, because this is the last moment a refusal is free.
        q6_now = float(robot.arm.q()[j_roll])
        turn_lo, turn_hi = sorted((q6_now, q6_now + cl_rot))
        _tol = np.radians(0.25)                     # the servo's tracking, not a wrist that walked off the plan
        if turn_lo < w_lo + cl_w3_margin - _tol or turn_hi > w_hi - cl_w3_margin + _tol:
            log.error('COLLAR CLOCKING: wrist_3 is at %+.1f deg after the advance, so the %+.1f '
                      'deg turn would run %+.1f .. %+.1f and leave the usable range '
                      '%+.1f .. %+.1f (margin %.1f). The prewind aimed for %+.1f. Refusing to '
                      'turn -- the fingers are still OPEN, so nothing is clamped.',
                      np.degrees(q6_now), np.degrees(cl_rot), np.degrees(turn_lo),
                      np.degrees(turn_hi), np.degrees(w_lo), np.degrees(w_hi),
                      np.degrees(cl_w3_margin), np.degrees(q6_start))
            return False
        log.info('  wrist_3 at %+.1f deg after the advance (prewound to %+.1f) -- the %+.1f deg '
                 'turn fits with %.1f deg to spare.', np.degrees(q6_now), np.degrees(q6_start),
                 np.degrees(cl_rot),
                 min(turn_lo - w_lo, w_hi - turn_hi) * 180.0 / 3.141592653589793)

        # ---- TARE BEFORE THE GRASP -------------------------------------------------------
        # The arm is standing at the collar station with the fingers still OPEN and clear of the
        # ring: the only genuinely unloaded moment in the maneuver. The tare used to happen
        # inside the warmup BELOW, i.e. after the close, which folds in whatever the jaws
        # preload against the captive ring -- jaw-on-jaw force cancels at the wrist, but an
        # off-centre close reacts through the connector into the socket and does not. At
        # max_torque_nm 1.0 that offset is a large fraction of the whole limit, so the turn could
        # trip on the grasp rather than on the lock.
        #
        # An IDLE tare (settle=True), unlike the servo-active one warmup performs: the gripper
        # close blocks for ~1 s, and streaming servoL around a blocking call is exactly what
        # warmup's mid-hold tare exists to avoid. What that trades away is the idle-vs-servo
        # offset, which is the uncompensated tool weight -- and robot.payload IS configured here,
        # so it is already subtracted. settle=True also re-checks the residual and warns if it
        # is not.
        if cl_tare is not None:
            robot.arm.zero_ft(settle=True)
        if not robot.gripper.close('grasp collar'):
            log.error('Gripper did not close on the collar.')
            return False
        phase('collar_clock')
        start = robot.tool0()
        adm_cl.reset()
        adm_cl.warmup(start)          # NOT tare_fn=cl_tare -- zeroed above, with open fingers
        guard_cl.reset()
        # THE TWIST. Still written as a rotation about the connector axis LINE, unchanged from the
        # radial version -- but tool0 now sits ON that line, so it resolves to a rotation about
        # tool0's own Z and the flange stays put. screw_ramp still subdivides it: with push_mm set
        # the turn is a SCREW (the axial press keeps the collar's lugs on their ramps), and a
        # translation along the rotation axis commutes with it, so the composed path is exact.
        res, _f_turn = screw_ramp(
            adm_cl,
            lambda f: (translation_matrix(cl_push_m * f * axn)
                       @ rotate_about_axis(start, axis, point, cl_rot * f)),
            guard_cl, cl_v, cl_w, abs(np.degrees(cl_rot)), label='twist ')
        _lin, turned = pose_error(start, robot.tool0())
        adm_cl.reset()
        if cl_settle > 0:
            adm_cl.hold(robot.tool0(), cl_settle, guard=None)
        adm_cl.stop()
        robot.arm.servo_stop()
        stopped = res == 'seated'            # ramp's word for a guard trip -- not the state
        clock_rows.append({'maneuver': 'collar_clocking', 'try': 1, 'ramp_result': res,
                           'grasp_clock_deg': round(float(np.degrees(eng_clock + th_grasp)), 3),
                           'turned_deg': round(float(np.degrees(turned)), 3),
                           'commanded_deg': round(float(np.degrees(cl_rot)), 3),
                           'tool0_x_mm': round(x_tool['collar grasp'], 3),
                           'success': True, 'force_stop': bool(stopped),
                           'state_after': 'seated' if unlocking else 'locked',
                           'stopped_by': guard_cl.tripped_by or ''})
        _word = 'UNLOCKED' if unlocking else 'LOCKED'
        if stopped:
            # SYMMETRIC WITH THE LOCK: the collar runs to a stop at BOTH ends of its travel, so
            # the torque building is the intended termination in either direction -- turn the
            # full rotation_deg or until the ring stops turning, whichever comes first.
            log.info('  %s -- the collar stopped on the force guard (%s) after %.1f of %.1f '
                     'deg, which is what reaching the end of its travel looks like; check it.',
                     _word, guard_cl.tripped_by, np.degrees(turned),
                     abs(np.degrees(cl_rot)))
        else:
            log.info('  %s -- collar turned %.1f deg (commanded %.1f). The flange moved '
                     '%.1f mm: a wrist twist, not an arm swing.', _word, np.degrees(turned),
                     np.degrees(cl_rot), _lin * 1000.0)
        _passed('COLLAR %s' % ('UNLOCK' if unlocking else 'LOCK'),
                ('the force guard tripped -- the ring at the end of its travel (%s) after %.1f of '
                 '%.1f deg' % (guard_cl.tripped_by, np.degrees(turned), abs(np.degrees(cl_rot))))
                if stopped else
                ('the full %.1f deg turn ran (turned %.1f deg) without the torque limit'
                 % (abs(np.degrees(cl_rot)), np.degrees(turned))),
                'EXTRACT (collar and bayonet released together)' if unlocking
                else ('TUG VERIFY' if tv_on else 'ESCAPE'))
        return True


    def dry_locate_socket(q_return, socket, refine):
        """locate_target_pose6d in a DRY CELL (apps/_dry_cell.py): the same moves to the socket view and back, the
        socket at a known pose instead of the views' estimate, the run anchored on it, the virtual jack put there."""
        phase('visual_localize')
        q_view = vt.get('view_joints_deg')
        if q_view is not None and not robot.arm.move_j(list(np.radians(np.asarray(q_view, dtype=float))),
                                                       label='socket view pose'):
            log.error('6D POSE (dry cell): could not reach assembly.visual_target.view_joints_deg.')
            return False
        if q_return is not None and vt.get('return_home_after', True):
            phase('reset')
            if not robot.arm.move_j(q_return, label='home after the socket views'):
                return False
        if refine and pose6d_measured[0]:
            return True                          # the same pose: nothing to refine in a dry cell
        T_socket, src = _dry_cell.socket_pose(cfg, socket, targets, tname, T_junction_conn)
        if T_socket is None:
            log.error('6D POSE (dry cell): no socket pose to stand in -- %s. Set dry_cell.socket.', src)
            return False
        T_target = P6.target_from_socket(T_socket, T_junction_conn, socket=socket)
        log.info('6D POSE (dry cell): the %s stands in at %s mm -- %s; target at %s mm.', socket,
                 np.round(T_socket[:3, 3] * 1000.0, 1).tolist(), src, np.round(T_target[:3, 3] * 1000.0, 1).tolist())
        _anchor_target(T_target)
        socket_obj[0] = np.asarray(T_socket, dtype=float)
        pose6d_measured[0] = True
        dry.jack.target(T_base_tconn)
        return True

    def locate_target_pose6d(q_return, refine=False):
        """The socket from the 6D pose estimator, from several views, and the run re-anchored on it.

        IN TWO STAGES when assembly.visual_target.refine_before_standoff is on (the user, 2026-11-03):
        before the pick ONE view, a coarse pose -- enough to know where to look; then, with the
        connector in the hand and just before the stand-off, `refine=True`: the view pose again and
        the most spread of the orbit's reachable views (refine_before_standoff.orbit, 4 views in all
        on the bench), refined from the coarse pose, and the run re-anchored on that.

        RUNS BEFORE THE PICK, the gripper empty (or, starting held, with the connector in it): the
        camera goes to visual_target.view_joints_deg (home if unset), the socket is estimated from
        there and from an orbit about it (skills/pose6d_pick.py), and the arm comes back.

        THE TARGET is where the plug's CONNECTOR frame -- the one the in-hand belief is in -- sits
        when the plug is seated: the socket's pose, times the seated plug's junction frame in it
        (perception/pose6d.py, from the CAD), times the part's junction -> connector transform,
        the very one held_belief builds the belief with. So the target and the belief describe
        the same datum, whatever the cell's connector frame is."""
        from ..skills.pose6d_pick import aim_cfg, draw_socket, estimate_part, view_past_held
        socket = P6.socket_part(cfg)            # the one pose6d.plug goes into
        if dry is not None:
            return dry_locate_socket(q_return, socket, refine)
        if camera is None:
            log.error('assembly.target_source is pose6d but the app has no camera.')
            return False
        if conn_tracker is not None:
            conn_tracker.stop()                  # the estimator is the socket's now
        rb = vt.get('refine_before_standoff') or {}
        two_stage = bool(rb.get('enabled', False))
        if refine:
            orbit = dict(vt.get('orbit') or {}, **dict(rb.get('orbit') or {}))
            what = 'REFINED from %s views before the stand-off' % (1 + int(orbit.get('max_views') or 0))
        elif two_stage:
            orbit, what = {'enabled': False}, 'COARSE, one view (refined before the stand-off)'
        else:
            orbit, what = vt.get('orbit'), 'from the view pose and its orbit'
        if not phase_gate('6D POSE: THE SOCKET (%s)' % ('refine' if refine else 'coarse' if two_stage else 'scan'),
                          'The camera will look at the socket -- %s; the run is then anchored on the pose '
                          'the views agree on.' % what):
            return False
        phase('visual_localize')
        q_view = vt.get('view_joints_deg')
        if q_view is not None and not robot.arm.move_j(list(np.radians(np.asarray(q_view, dtype=float))),
                                                       label='socket view pose'):
            log.error('6D POSE: could not reach assembly.visual_target.view_joints_deg.')
            return False
        # THE RECORDED MATE, IF ANY, AS WHERE THE SOCKET SHOULD BE (the kinematic calibration with the
        # finger holder): the socket it implies seeds the first view -- refined from there rather than
        # searched for -- and the orbit, should the first view not find it.
        prior = None
        if tname in targets:
            prior = targets[tname] @ inverse(T_junction_conn) @ inverse(P6.seated_junction_in_socket(socket))
        # THE LAST MEASURED SOCKET, FROM AN EARLIER RUN (the user, 2026-11-04): the fixture does not move,
        # and the recorded target sits ~10 mm / 3-6 deg from where the 6D estimator puts the socket --
        # outside a refine's reach, so every coarse view failed its refine (1.5 s) and searched the
        # whole image (~14 s). The last accepted socket pose (data/pose6d/last_socket.json) is near
        # enough to refine from; not found from it, the image is searched as before.
        last_path = os.path.join(cfg.get('data_dir', 'data'), 'pose6d', 'last_socket.json')
        if not refine and vt.get('seed_from_last', True) and os.path.isfile(last_path):
            try:
                with open(last_path) as fh:
                    _last = json.load(fh)
                if _last.get('part') == socket:
                    prior = np.asarray(_last['T_base_obj'], dtype=float)
                    log.info('6D POSE: the coarse view refines from the socket measured at %s (%s).',
                             _last.get('time', '?'), last_path)
            except (OSError, ValueError, KeyError) as exc:
                log.warning('6D POSE: %s unreadable (%s) -- the recorded target as the prior.', last_path, exc)
        if refine and socket_obj[0] is not None:
            prior = socket_obj[0]                # the coarse estimate: where to look now
        # THE CONNECTOR IN THE HAND HIDES THE SOCKET from the view pose (the user, 2026-11-04): pitch
        # the camera until the socket sits above it in the image, servoing on the socket's estimate.
        ac = aim_cfg(cfg, vt.get('aim_past_held'))
        if ac['enabled'] and robot.held is not None and prior is not None:
            try:
                prior = view_past_held(cfg, robot, camera, socket, prior, ac, '6D pose: the socket')
            except P6.Pose6DError as exc:
                log.warning('6D POSE: aiming past the held connector failed (%s) -- viewing from here.', exc)
        try:
            res, _views = estimate_part(cfg, robot, camera, socket, confirm,
                                        '6D pose: the socket (%s)' % ('refine' if refine else 'coarse' if two_stage
                                                                       else 'scan'), prior=prior, orbit=orbit)
            socket_views[0] = list(_views)
            if refine:                     # what was SEEN, not what was planned (02:32: "4 views", 1 used)
                what = 'REFINED from %d view%s before the stand-off' % (len(_views), '' if len(_views) == 1 else 's')
        except P6.Pose6DError as exc:
            log.error('6D POSE: the estimator failed: %s', exc)
            res = None
        if q_return is not None and vt.get('return_home_after', True):
            phase('reset')
            if not robot.arm.move_j(q_return, label='home after the socket views'):
                log.error('6D POSE: could not return home after the socket views.')
                return False
        if res is None:
            log.error('6D POSE: no socket pose -- nothing is planned from a guess. Check the view pose '
                      'frames the socket, or run with pose_method: sam3 and a recorded target.')
            return False
        if refine and socket_obj[0] is not None:
            d_lin_c, d_ang_c = pose_error(socket_obj[0], res['T_base_obj'])
            log.info('6D POSE: the refined socket is %.2f mm / %.2f deg from the coarse one.',
                     1000.0 * d_lin_c, np.degrees(d_ang_c))
        T_target = P6.target_from_socket(res['T_base_obj'], T_junction_conn, socket=socket)
        draw_socket(cfg, robot, res, T_target, socket)
        _xyz, _rpy = matrix_to_xyzrpy(T_target)
        log.info('6D POSE: the socket at %s mm in base_link -> target %r at xyz %s mm, rpy %s deg.',
                 (res['T_base_obj'][:3, 3] * 1000).round(1).tolist(), tname,
                 [round(v * 1000.0, 2) for v in _xyz], [round(float(np.degrees(v)), 2) for v in _rpy])
        if tname in targets:
            d_lin, d_ang = pose_error(targets[tname], T_target)
            if P6.SYMMETRY[P6.plug_part(cfg)] == 'axis_z':
                # Round: the 6D target's turn about the insertion axis is a convention (z up), the
                # recorded one is however the holder held it -- compare the axis, not the turn.
                d_ang = float(np.arccos(np.clip(targets[tname][:3, 0] @ T_target[:3, 0], -1.0, 1.0)))
            lim_mm, lim_deg = vt.get('max_shift_mm', 25.0), vt.get('max_shift_deg', 10.0)
            over = ((lim_mm is not None and d_lin * 1000.0 > float(lim_mm))
                    or (lim_deg is not None and np.degrees(d_ang) > float(lim_deg)))
            (log.error if over else log.info)(
                '6D POSE: %.2f mm / %.2f deg from the recorded targets: entry%s.', d_lin * 1000.0, np.degrees(d_ang),
                ' -- OVER assembly.visual_target.max_shift_mm/_deg' if over else '')
            if over:
                return False
        _anchor_target(T_target)
        socket_obj[0] = np.asarray(res['T_base_obj'], dtype=float)
        pose6d_measured[0] = True
        try:                                     # the next run's coarse view starts from it (above)
            os.makedirs(os.path.dirname(last_path), exist_ok=True)
            with open(last_path, 'w') as fh:
                json.dump({'part': socket, 'T_base_obj': socket_obj[0].tolist(), 'what': what, 'n_used': res.get('n_used'),
                           'score': res.get('score'), 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, fh, indent=1)
        except OSError as exc:
            log.warning('6D POSE: could not keep the socket pose in %s (%s).', last_path, exc)
        log.info('6D POSE: the run is now anchored on the MEASURED socket pose (%s).', what)
        return True

    def look_at_socket(q_return):
        """Depth views of the socket and what is round it, for the stand-off plan, when nothing else
        looks there (target_source kinematic or visual): from visual_target.view_joints_deg and an
        orbit about the target (pose6d.orbit, unless standoff.scene is `view`), then back."""
        from ..skills.pose6d_pick import _orbit_cfg, orbit_views, save_views, still_view
        if camera is None:
            log.error('assembly.standoff.plan needs the camera (its scene is depth views of the socket).')
            return False
        q_view = vt.get('view_joints_deg')
        if q_view is None:
            log.error('assembly.standoff.plan needs assembly.visual_target.view_joints_deg -- where the socket '
                      'is seen from.')
            return False
        if not phase_gate('SCENE: THE SOCKET',
                          'The camera will look at the socket and round it, for the collision scene the move '
                          'to the stand-off is planned in.'):
            return False
        phase('visual_localize')
        if not robot.arm.move_j(list(np.radians(np.asarray(q_view, dtype=float))), label='socket view pose'):
            log.error('SCENE: could not reach assembly.visual_target.view_joints_deg.')
            return False
        oc = _orbit_cfg(cfg)
        first = still_view(camera, robot.arm, oc['frames'], oc['settle_s'])
        views = [] if first is None else [first]
        if str(_st.get('scene', 'orbit')).lower() == 'orbit' and oc['enabled']:
            views += orbit_views(robot, camera, T_base_tconn[:3, 3], oc, confirm, 'scene: the socket')
        if q_return is not None and vt.get('return_home_after', True):
            phase('reset')
            if not robot.arm.move_j(q_return, label='home after the socket views'):
                log.error('SCENE: could not return home after the socket views.')
                return False
        if not views:
            log.error('SCENE: no usable view of the socket.')
            return False
        socket_views[0] = views
        save_views(cfg, 'scene: the socket', views, [])        # to replay a plan offline
        log.info('SCENE: %d view%s of the socket for the stand-off plan.', len(views), '' if len(views) == 1 else 's')
        return True

    def socket_scene(q_return):
        """The socket's views for the stand-off plan (assembly.standoff.socket_views):
          scan    look at it now (look_at_socket)
          saved   the newest views a run kept (<data_dir>/pose6d/*_scene_the_socket) -- no look, no
                  move: right while the fixture has not moved, which a KINEMATIC target assumes anyway.
                  None kept yet: one scan, said why (it is kept for the next run)
          none    no views: the plan goes round the base plane alone -- the fixture is NOT avoided
          auto    (default) a scan for a pose6d target; for a kinematic one no look, and the plan goes
                  round what this run's camera has seen (robot.world) -- ONLY ONLINE OBSERVATIONS (the
                  user, 2026-11-03: "besides robot and ground plane all collision objects should be from
                  online RGBD observations"); `saved` brings an earlier run's views in on purpose
        A pose6d target's own views serve, so this is not called for one."""
        mode = str(_st.get('socket_views', 'auto')).strip().lower()
        if mode == 'auto':
            if tgt_source == 'kinematic':
                log.info('SCENE: the target is the kinematic calibration, so the socket is not looked at: the '
                         'stand-off plan goes round what this run\'s camera has seen and the base plane.')
                return True
            mode = 'scan'
        if mode == 'none':
            log.warning('SCENE: assembly.standoff.socket_views none -- the stand-off plan avoids the base plane '
                        'only, NOT the socket fixture.')
            return True
        if mode == 'saved':
            from ..skills.pose6d_pick import load_saved_views
            path, views = load_saved_views(cfg, 'scene: the socket')
            if views:
                socket_views[0] = views
                if getattr(robot, 'world', None) is not None:      # moves are checked against them too
                    robot.world.add_views(views, 'saved socket views')
                log.info('SCENE: the target is the kinematic calibration, so the socket is not looked at: the '
                         'stand-off plan uses the %d saved view%s in %s (socket_views scan looks again).',
                         len(views), '' if len(views) == 1 else 's', path)
                return True
            log.warning('SCENE: no saved views of the socket under %s yet -- looking once; later runs reuse them.',
                        os.path.join(cfg.get('data_dir', 'data'), 'pose6d'))
        elif mode != 'scan':
            log.error('assembly.standoff.socket_views %r: one of auto, scan, saved, none.', mode)
            return False
        return look_at_socket(q_return)

    def held_connector(T_ftip_conn_now):
        """The connector as a part in the hand (robot/held_part.py): its CAD -- the 6D estimator's
        model -- at the in-hand belief `T_ftip_conn_now`, and the cable leaving it (only its first
        `held_cable_mm`: further on it hangs where no model says). None, said why, with no CAD."""
        from ..robot import tool_geometry
        from ..robot.held_part import HeldPart
        part = P6.plug_part(cfg)
        mesh = P6.cad_mesh(cfg, part)
        if mesh is None:
            return None
        T_tool0_plug = robot.T_tool0_fingertip @ P6.plug_from_belief(T_ftip_conn_now, T_junction_conn, part)
        z_j = P6.JUNCTION_Z[part]
        cable = tool_geometry.rod_spheres(T_tool0_plug, z_j - float(_st.get('held_cable_mm', 60.0)) / 1000.0, z_j,
                                          float(_st.get('cable_radius_mm', 5.0)) / 1000.0)
        return HeldPart(part, mesh[0], mesh[1], T_tool0_plug, float(_st.get('held_radius_mm', 9.0)) / 1000.0, cable)

    def close_report(T_closed, T_ftip_conn_now):
        """Where the grasp CLOSED against where it was commanded, and what that does to the in-hand
        pose -- logged and kept (in_hand_close.json in the run's directory) every run, whichever
        belief is used. Three numbers, in the DETECTED CONNECTOR's frame (+x along it, +z up):
          reseats   the grasp target against detection @ fingertip_in_connector: the pick's reseat
                    nudges, which accumulate into the target
          reached   the fingertip at the close against that target: a compliant descent stopping
                    short on the bench, a guard
          in hand   the pre-pickup estimate carried rigidly from the close (pose6d.rigid_grasp's
                    belief) against the belief in use: what the expected-pose belief is missing,
                    if the estimate is right and nothing moved as the jaws closed."""
        T_det = getattr(geom, 'T_base_detection', None)
        if T_closed is None or T_det is None:
            return
        try:
            T_ftip_closed = T_closed @ robot.T_tool0_fingertip
            T_cmd = T_det @ _grip_off
            R = T_det[:3, :3].T

            def in_conn(a, b):
                return R @ (b[:3, 3] - a[:3, 3]), np.degrees(pose_error(a, b)[1])
            res_xyz, res_ang = in_conn(T_cmd, geom.T_base_grasp)
            rch_xyz, rch_ang = in_conn(geom.T_base_grasp, T_ftip_closed)
            T_rigid = rigidly_held_junction(robot.T_tool0_fingertip, T_closed, T_det @ latch_turn) @ T_junction_conn
            d = inverse(T_ftip_conn_now) @ T_rigid
            hand_xyz, hand_rpy = matrix_to_xyzrpy(d)
            log.info('AT THE CLOSE (detected connector frame, mm): reseats moved the target %s (%.1f deg); the '
                     'fingertip closed %s (%.1f deg) from the target; so the rigidly held estimate is %s mm / '
                     '%s deg (connector frame) from the belief in use.',
                     np.round(1000 * res_xyz, 1).tolist(), res_ang, np.round(1000 * rch_xyz, 1).tolist(), rch_ang,
                     np.round(1000 * hand_xyz, 1).tolist(), np.round(np.degrees(hand_rpy), 1).tolist())
            import json
            with open(os.path.join(out_dir, 'in_hand_close.json'), 'w') as fh:
                json.dump({'T_base_detection': T_det.tolist(), 'T_conn_ftip': _grip_off.tolist(),
                           'T_base_grasp_target': geom.T_base_grasp.tolist(), 'T_base_tool0_closed': T_closed.tolist(),
                           'T_tool0_fingertip': robot.T_tool0_fingertip.tolist(),
                           'T_ftip_conn_belief': T_ftip_conn_now.tolist(), 'T_ftip_conn_rigid': T_rigid.tolist(),
                           'reseat_mm': (1000 * res_xyz).tolist(), 'reached_mm': (1000 * rch_xyz).tolist(),
                           'rigid_vs_belief_mm': (1000 * hand_xyz).tolist(),
                           'rigid_vs_belief_deg': np.degrees(hand_rpy).tolist()}, fh, indent=1)
        except Exception as exc:            # noqa: BLE001 -- a report, never a reason to stop
            log.warning('AT THE CLOSE: could not report (%s).', exc)

    def plan_to_standoff(T_goal, T_ftip_conn_now):
        """The move from the lift to the stand-off, PLANNED round the socket views' scene with the held
        connector on the tool; the joints it ends at, or None with the arm left where it is."""
        from ..robot import tool_geometry
        from ..robot.planner import Planner
        from ..skills.pose6d_pick import scene_from_views
        # THE SCENE: what this run's camera has seen (robot.world: every still RGBD frame of the run, the
        # socket views among them; with robot.world.prior false nothing older), else the socket views.
        world = getattr(robot, 'world', None)
        scene = world.snapshot()[0] if world is not None else None
        if scene is not None and not scene.occ.any():
            scene = None
        if scene is None and socket_views[0]:
            scene = scene_from_views(cfg, robot, socket_views[0])
        if scene is None:
            log.warning('STAND-OFF: nothing seen yet -- planning round the base plane alone.')
        # THE HELD PART on the tool: the connector from the belief, and the cable leaving it (only its
        # first `held_cable_mm`: further on it hangs where no model says).
        # Its CAD (held_connector) where there is one, else a rod to the mating face.
        part = P6.plug_part(cfg)
        T_tool0_plug = robot.T_tool0_fingertip @ P6.plug_from_belief(T_ftip_conn_now, T_junction_conn, part)
        z_j = P6.JUNCTION_Z[part]
        r_held = float(_st.get('held_radius_mm', 9.0)) / 1000.0
        cad = held_connector(T_ftip_conn_now)
        held = cad.spheres if cad is not None else (
            tool_geometry.rod_spheres(T_tool0_plug, z_j, P6.MATING_Z[part], r_held)
            + tool_geometry.rod_spheres(T_tool0_plug, z_j - float(_st.get('held_cable_mm', 60.0)) / 1000.0, z_j,
                                        float(_st.get('cable_radius_mm', 5.0)) / 1000.0))
        planner = Planner.for_cell(cfg, held=held, grip_half_m=r_held)    # the fingers closed on it
        if planner is None:
            log.error('STAND-OFF: no cuMotion planner here -- refusing, the arm where it is. '
                      'assembly.standoff.plan: false moves without one.')
            return None
        # What the robot fills where the move starts is not an obstacle: the views were taken before
        # the pick, and may have caught the cable lying where the arm now stands.
        q_now = np.asarray(robot.arm.q(), dtype=float)
        freed = planner.free_start(scene, q_now)
        path, q_goal = planner.plan_to_pose(q_now, T_goal, scene, limits=getattr(robot.arm, 'joint_limits', None))
        if path is None:
            log.error('STAND-OFF: cuMotion: %s -- refusing, the arm where it is.', planner.why)
            return None
        dq = np.degrees(np.abs(q_goal - q_now))
        log.info('STAND-OFF: planned round the base plane and %d occupied voxels seen this run (%d freed where it '
                 'starts), the held connector on the tool: %.1f s, %.0f mm clearance at the closest; joints change up '
                 'to %.0f deg (A%d).', 0 if scene is None else int(scene.occ.sum()), freed,
                 path.duration, 1000 * path.clearance, dq.max(), int(np.argmax(dq)) + 1)
        if not phase_gate('MOVE TO THE STAND-OFF',
                          f'The arm moves along the planned path ({1000 * path.clearance:.0f} mm clearance at the '
                          f'closest), joints changing up to {dq.max():.0f} deg (A{int(np.argmax(dq)) + 1}).'):
            return None
        if not _guarded(robot, guard_shared, lambda: planner.execute(robot.arm, path, 'stand-off (planned)')):
            log.error('STAND-OFF: the planned move stopped short.')
            return None
        return list(q_goal)

    def locate_target_visually(q_return):
        """Drive to the view pose, sweep the markers, and re-anchor the run on what they say.

        RUNS BEFORE THE PICK, with the gripper empty and the arm at home: the camera has a clear
        view of the fixture, and nothing is being carried that a view sweep could disturb. After
        the mate the socket is behind a connector and a gripper, so this is the only moment the
        markers are worth looking at."""
        from ..perception import ArucoDetector

        if camera is None:
            log.error('assembly.target_source is visual but the app has no camera.')
            return False
        try:
            plan = mloc.ViewPlan(cfg.section('marker_views'))
        except ValueError as exc:
            log.error('marker_views: %s', exc)
            return False
        # The per-marker servo refinement is OPTIONAL at run time: visual_target.servo_refine
        # switches it off WITHOUT touching the shared marker_views.servo parameters, which
        # must stay identical to marker_calibration's for the runs where it is on.
        if not bool(vt.get('servo_refine', True)):
            plan.servo.enabled = False
            log.info('VISUAL TARGET: servo refinement OFF (visual_target.servo_refine) -- '
                     'sweep views only.')
        detector = ArucoDetector(cfg, sizes_m=tool_frames.marker_sizes(vt_rig))

        # Gate BEFORE the first sweep move: the camera is about to drive a ring of views a few
        # hundred mm off the fixture, and the whole run is anchored on what it measures.
        if not phase_gate('VISUAL LOCALIZATION',
                          'The camera will sweep the markers from the home view%s. The run is '
                          'then anchored on the pose they measure.'
                          % (' and servo to each one' if plan.servo.enabled else '')):
            return False

        phase('visual_localize')
        q_view = vt.get('view_joints_deg')
        if q_view is not None:
            log.info('VISUAL TARGET: driving to the view pose %s deg.',
                     list(np.round(np.asarray(q_view, dtype=float), 1)))
            if not robot.arm.move_j(list(np.radians(np.asarray(q_view, dtype=float))),
                                    label='marker view pose'):
                log.error('VISUAL TARGET: could not reach '
                          'assembly.visual_target.view_joints_deg.')
                return False
        else:
            log.info('VISUAL TARGET: sweeping from the HOME pose (reset.home_joints_deg IS the '
                     'marker view pose; set visual_target.view_joints_deg to sweep from '
                     'somewhere else).')

        # Every image the localization estimated from, annotated + indexed under
        # <experiment>/marker_images/ -- the only record of what the run actually saw.
        images = mloc.MarkerImageWriter(out_dir, detector,
                                        enabled=getattr(plan, 'save_images', True))
        T_vis = mloc.locate(robot, camera, detector, vt_rig, plan,
                            on_view=images.sweep_view, on_servo_view=images.servo_view)
        _vis_xyz, _vis_rpy = ((None, None) if T_vis is None else matrix_to_xyzrpy(T_vis))
        images.finish(
            ['visual target localization -- rig %r' % tname, '']
            + (['NO TARGET POSE -- the markers did not produce a fused answer.']
               if T_vis is None else
               ['located target in base_link:',
                '  xyz %+8.2f %+8.2f %+8.2f mm' % tuple(v * 1000.0 for v in _vis_xyz),
                '  rpy %+7.2f %+7.2f %+7.2f deg'
                % tuple(float(np.degrees(v)) for v in _vis_rpy)]))
        if q_return is not None and vt.get('return_home_after', True):
            # BACK TO HOME BEFORE THE PICK, whatever the localisation decided: the scan, the grasp
            # geometry and every retry offset are written from the home pose, and leaving the arm
            # parked at a view pose would silently change where the pick starts.
            phase('reset')
            if not robot.arm.move_j(q_return, label='home after the marker sweep'):
                log.error('VISUAL TARGET: could not return home after the sweep.')
                return False
        if T_vis is None:
            log.error('VISUAL TARGET: the markers did not produce a target pose. Nothing has '
                      'moved and the recorded mate is untouched -- re-run with '
                      'assembly.target_source: kinematic to use it, or fix the rig.')
            return False

        # HOW FAR THE FIXTURE HAS APPARENTLY MOVED. Not an error by itself -- the whole point of
        # the rig is that the fixture MAY move -- but a shift of the wrong ORDER (a metre, a
        # quarter turn) is a stale calibration or a marker on the wrong fixture, and driving an
        # insertion trajectory at it would be the expensive way to find out.
        d_lin, d_ang = pose_error(targets[tname], T_vis)
        lim_mm = vt.get('max_shift_mm', 25.0)
        lim_deg = vt.get('max_shift_deg', 10.0)
        over = ((lim_mm is not None and d_lin * 1000.0 > float(lim_mm))
                or (lim_deg is not None and np.degrees(d_ang) > float(lim_deg)))
        (log.error if over else log.info)(
            'VISUAL TARGET: the markers put %r %.2f mm / %.2f deg from the recorded mate%s.',
            tname, d_lin * 1000.0, np.degrees(d_ang),
            ' -- OVER assembly.visual_target.max_shift_mm/_deg (%s mm / %s deg)'
            % (lim_mm, lim_deg) if over else '')
        if over:
            log.error('  Refusing to plan an insertion at it. If the fixture really did move that '
                      'far, raise the limit or re-record the targets: entry; otherwise the rig is '
                      'stale or a marker is on the wrong fixture.')
            return False
        _anchor_target(T_vis)
        log.info('VISUAL TARGET: the run is now anchored on the MEASURED socket pose.')
        return True

    def place_scatter():
        """One uniform (dx_m, dy_m, dyaw_rad) draw for the disassembly place, or zeros.

        WHY SCATTER AT ALL. Without it a multi-cycle run re-picks the cable from the SAME spot
        every time, so the pick is only ever tested against one pose and the estimator only ever
        sees one presentation. Scattering the place makes each cycle a genuinely fresh detection
        and grasp, which is the whole value of running the loop repeatedly.

        CENTRED ON THE CONFIGURED PLACE, not on a fresh guess: the draw is ADDED to
        reorient_recovery.place_offsets, so turning the scatter off leaves the old behaviour
        exactly. x/y are in the TARGET CONNECTOR's frame, like the offsets they perturb, so the
        spread means the same thing whichever way the socket faces.

        Off by default, and `seed` makes a scattered run reproducible -- an unseeded random place
        is a run you cannot re-create when something goes wrong on cycle 34."""
        sc = (a.get('disassembly', {}) or {}).get('place_scatter', {}) or {}
        if not bool(sc.get('enabled', False)):
            return 0.0, 0.0, 0.0
        rx = abs(_num(sc, 'x_mm', 50.0)) / 1000.0
        ry = abs(_num(sc, 'y_mm', 50.0)) / 1000.0
        rw = np.radians(abs(_num(sc, 'yaw_deg', 30.0)))
        rng = _place_rng[0]
        return (float(rng.uniform(-rx, rx)), float(rng.uniform(-ry, ry)),
                float(rng.uniform(-rw, rw)))

    def aligned_place_pose(extra_clearance_m=0.0, scatter=(0.0, 0.0, 0.0)):
        """Where the cable is set down, in base_link -- LYING ALONG THE SOCKET AXIS.

        `scatter` is a (dx_m, dy_m, dyaw_rad) perturbation ADDED to the configured offsets -- see
        place_scatter(). The default of zeros is the deterministic pose.

        THE CONNECTOR +X COMES OUT PARALLEL TO THE TARGET CONNECTOR +X. That is the whole
        requirement and it is easy to lose: a place that simply descends from the pick pose
        inherits whatever attitude the grip has, and with a COAXIAL grip the connector hangs
        with its axis along the tool -- i.e. pointing at the floor. The part then lands on its
        end instead of on its side.

        So the orientation here is BUILT, not inherited: heading from the socket, roll and
        pitch zero. `extra_clearance_m` lifts the release point above the resting height.

        DEFINED RELATIVE TO THE TARGET CONNECTOR, because the whole point is to leave the cable
        ALIGNED WITH THE SOCKET: yaw 0 wrt the target puts the connector axis on the same
        compass heading the socket has, which is the one heading a coaxial approach is known to
        reach. Setting it down on an arbitrary heading would fail the same way again.

        ROLL AND PITCH ARE NOT FREE -- the part is going onto a flat bench, so its axis ends up
        horizontal and its z up, whatever attitude the socket itself has. Only the HEADING
        carries over. (The recorded socket is level to within half a degree, so here the two
        readings agree to well under a millimetre; the code does not lean on that.)"""
        r = a.get('reorient_recovery', {}) or {}
        off = r.get('place_offsets', {}) or {}
        z_mm = float(off.get('z_mm', -400.0))
        s_x, s_y, s_yaw = (float(v) for v in scatter)
        d = np.array([float(off.get('x_mm', 0.0)) / 1000.0 + s_x,
                      float(off.get('y_mm', 0.0)) / 1000.0 + s_y,
                      z_mm / 1000.0], dtype=float)
        p = T_base_tconn[:3, 3] + T_base_tconn[:3, :3] @ d      # offsets in the TARGET frame
        yaw = (float(np.arctan2(T_base_tconn[1, 0], T_base_tconn[0, 0]))
               + float(np.radians(float(off.get('yaw_deg', 0.0)))) + s_yaw)
        T = xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, 0.0, yaw])   # flat: roll = pitch = 0
        # THE GROUND DECIDES z, by default. A resting connector's AXIS sits one barrel-radius
        # up -- a property of the part, not a number to type twice -- and "place it on the
        # ground plane" is the actual requirement. The configured z_mm is still reported,
        # because a large gap between the two means one of the two is wrong.
        axis_z = ground_z + connector_axis_height_m(cfg) + float(extra_clearance_m)
        if bool(r.get('snap_to_ground', True)):
            log.info('REORIENT PLACE: z_mm %+.0f puts the connector axis %+.0f mm above the '
                     'bench; snapping to the ground plane at %+.0f mm instead (one barrel '
                     'radius, %.1f mm, up).', z_mm, (p[2] - ground_z) * 1000.0,
                     (axis_z - ground_z) * 1000.0, connector_axis_height_m(cfg) * 1000.0)
            p[2] = axis_z
        T[:3, 3] = p
        return T

    if dry is not None:
        # The dry cell's plug lies where the reset pose puts a cable -- the one place the coaxial grasp is known to reach.
        dry.scanner.plug_pose_fn = lambda: aligned_place_pose(0.0)

    def reorient_recovery():
        """PICK IT SQUARE, SET IT DOWN ALIGNED, TRY AGAIN.

        THE FAILURE THIS ANSWERS. The coaxial grasp reaches the connector from along its own
        axis, so whether the arm can get there at all depends on which way the cable happens to
        be lying. A cable pointing the wrong way has NO collision-free path to that grasp, and
        no number of retries produces one -- the geometry refused, not the attempt.

        A SQUARE grasp (pitch 0, straight down) has no such dependence: the approach is
        vertical whatever the heading. That is exactly why it is the fallback. So this picks
        the cable the easy way, sets it down pointing where the socket points, and hands back
        to the scan -- which now sees a cable the coaxial grasp CAN reach.

        ONCE PER PICK. If the coaxial grasp is still unreachable after the cable has been
        squared up, its heading was not the problem and repeating this would only shuffle the
        part around the bench."""
        r = a.get('reorient_recovery', {}) or {}
        if not bool(r.get('enabled', True)):
            log.error('The coaxial grasp is unreachable and reorient_recovery is off.')
            return False
        if not phase_gate('REORIENT THE CABLE',
                          'The coaxial grasp has no collision-free path to this cable. Next: '
                          'pick it SQUARE (pitch 0), set it down aligned with the socket, and '
                          're-scan.'):
            return False

        # THE SQUARE GRASP, WHOLE POSE, for this manoeuvre only -- restored in the finally,
        # so a recovery that fails half way cannot leave the run picking with the wrong
        # geometry. It overrides xyz as well as rpy: the bite point that suits a coaxial
        # approach is not the one that suits a vertical one, and the same override is what the
        # in-hand belief for the PLACE is derived from below, so the two cannot disagree.
        keep = cfg.get_path('pickup.fingertip_in_connector')
        square = dict(r.get('fingertip_in_connector')
                      or {'xyz_mm': [5.0, 0.0, 0.0], 'rpy_deg': [0.0, 0.0, 0.0]})
        try:
            cfg.set_path('pickup.fingertip_in_connector', square)
            log.info('REORIENT: picking SQUARE (xyz %s mm, rpy %s deg) instead of the coaxial '
                     '(xyz %s, rpy %s).', square.get('xyz_mm'), square.get('rpy_deg'),
                     (keep or {}).get('xyz_mm'), (keep or {}).get('rpy_deg'))
            # NO NEW SCAN AND NO SECOND PROMPT. The cable has not moved since the coaxial
            # attempt was refused -- only the way the arm means to approach it has changed --
            # so the detection and the cable the operator already identified both still hold.
            phase('scan')
            if _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                     T_conn=getattr(geom, 'T_base_detection', None)) != 'ok':
                log.error('REORIENT: the square pick failed too -- the cable heading is not '
                          'what is wrong here. Stopping.')
                return False
            # THE CONNECTOR ON THE TOOL while it is carried and set down (the user, 2026-11-03: the
            # live view lost it here) -- the scan's estimate held rigidly from the close where there
            # is one, else where the square grasp puts it. The release takes it off again.
            if hasattr(robot, 'hold'):
                T_closed_sq = getattr(geom, 'T_base_tool0_closed', None)
                T_det_sq = getattr(geom, 'T_base_detection', None)
                if T_closed_sq is not None and T_det_sq is not None:
                    T_ftip_sq_held = rigidly_held_junction(robot.T_tool0_fingertip, T_closed_sq,
                                                           T_det_sq @ _scan_turn()) @ T_junction_conn
                else:
                    T_ftip_sq_held = held_belief(T_ftip_conn_catalogue,
                                                 from_cfg(cfg.section('junction_in_fingertip')),
                                                 fingertip_in_connector(cfg)) @ _scan_turn()
                held_sq = held_connector(T_ftip_sq_held)
                if held_sq is not None:
                    robot.hold(held_sq)
            if not grasp.lift(robot, geom, 'reorient lift',
                              position_guard=lambda mv: _guarded(robot, guard_shared, mv)):
                return False

            # WHERE IT GOES. The belief for the SQUARE grasp (cfg is still overridden here)
            # turns "put the CONNECTOR there" into a fingertip pose.
            # RELEASED A LITTLE ABOVE THE RESTING HEIGHT, so the pads are not pressing the
            # cable into the bench when they open; the part drops the last few millimetres.
            T_place = aligned_place_pose(
                float(r.get('release_clearance_mm', 10.0)) / 1000.0)
            # THE BELIEF FOR THE SQUARE GRASP -- cfg is still overridden here, so this is
            # derived from the very transform the pick was commanded with. That is the whole
            # reason the override wraps the place as well as the pick: a place computed from
            # the COAXIAL belief would set the part down rotated by the difference.
            T_ftip_sq = held_belief(T_ftip_conn_catalogue,
                                    from_cfg(cfg.section('junction_in_fingertip')),
                                    fingertip_in_connector(cfg))
            T_ftip_target = T_place @ inverse(T_ftip_sq)
            log.info('REORIENT PLACE: connector to xyz %s mm on heading %+.1f deg (the socket '
                     'heading %+.1f deg + yaw offset).',
                     np.round(T_place[:3, 3] * 1000.0, 1).tolist(),
                     np.degrees(np.arctan2(T_place[1, 0], T_place[0, 0])),
                     np.degrees(np.arctan2(T_base_tconn[1, 0], T_base_tconn[0, 0])))

            above = translation_matrix(
                [0.0, 0.0, float(r.get('approach_mm', 120.0)) / 1000.0])
            for tag, ph, T in (('above', 'reconfigure', above @ T_ftip_target),
                               ('down', 'lift', T_ftip_target)):
                phase(ph)
                if not _guarded(robot, guard_shared,
                                lambda _T=T, _t=tag: robot.move_fingertip(
                                    _T, 'reorient place (%s)' % _t)):
                    return False
            if not robot.gripper.open('release (cable reoriented)'):
                return False
            phase('retract')
            if not _guarded(robot, guard_shared,
                            lambda: robot.move_fingertip(above @ T_ftip_target,
                                                         'reorient place (clear)')):
                return False
            if hasattr(scanner, 'reselect'):
                scanner.reselect()      # it is not where it was scanned from any more
            # LET IT SETTLE BEFORE LOOKING AT IT. A cable that has just been carried and set
            # down is still under whatever tension the move put into it, and it keeps moving
            # for a second or two after the fingers open -- the free length pulls the connector
            # round as it relaxes. Scanning into that gives a junction pose the part has
            # already left, and the coaxial grasp is the one approach that cannot absorb a
            # heading error. So wait, THEN take the picture.
            settle = float(r.get('settle_s', 5.0))
            if settle > 0.0 and not robot.arm.dry_run:
                log.info('REORIENT: waiting %.1f s for the cable to settle before re-scanning '
                         '-- it is still relaxing after being set down.', settle)
                robot.arm.sleep(settle)
            log.info('REORIENT COMPLETE -- the cable now points where the socket does. '
                     'Re-scanning and retrying the coaxial grasp.')
            return True
        finally:
            cfg.set_path('pickup.fingertip_in_connector', keep)

    # ==================================================================================
    # THE TASK AS A BEHAVIOR TREE, RUN N TIMES (the user, 2026-11-05: "add a visualization of a
    # behavior tree and an option for running it N times (default 100). record all data. add basic
    # failure recovery ... be conservative and stop operations and ask for intervention when
    # necessary. indicate/visualize the mode and autonomy state").
    #
    # One TRIAL is what one CYCLE was: localize -> pick -> in-hand -> stand-off -> insert -> seat ->
    # lock -> verify -> escape -> disassemble (which puts the cable back) -> home. Each step below is
    # the code the cycle loop ran, now a leaf of the tree (behaviors/campaign.py) -- and each failure
    # now says WHY and WHAT THE CELL IS (where the connector is, how far it is in, whether the fingers
    # hold it, where the arm is). The tree's Recover branch picks the recovery from that state, or asks
    # a person; the board (behaviors/board.py) draws the tree with the mode and the autonomy state.
    #
    # EVERYTHING INSIDE A TRIAL REPEATS, deliberately: the socket is re-measured (the fixture may move
    # between trials) and the cable re-found (after a place it is NOT where it was picked from).
    # ==================================================================================
    from ..behaviors import campaign as bt
    from ..behaviors.board import Board

    cp = dict(cfg.get('campaign') or {})
    n_trials = int(cp['trials']) if cp.get('trials') else n_cycles
    reset_mode = str(cp.get('reset') or 'auto').strip().lower()
    if reset_mode not in ('auto', 'manual'):
        log.error("campaign.reset %r must be 'auto' (disassembly puts the cable back) or 'manual' (you do, "
                  "after each trial).", reset_mode)
        return False
    auto_reset = dis_on and dis_place_on and (cc_on or ins_mode == 'rl')
    if n_trials > 1 and reset_mode == 'auto' and not auto_reset:
        log.error('%d TRIALS asked for, but nothing puts the cell back between them: %s. Turn on '
                  'assembly.disassembly (and its place), or set campaign.reset: manual to reset the cell by hand '
                  'after each trial. Nothing has moved.', n_trials,
                  'disassembly is off' if not dis_on else 'disassembly.place is off' if not dis_place_on
                  else 'connector clocking is off, so there is nothing for disassembly to undo')
        return False
    n_cycles = n_trials                  # celebrate() reads it
    run_dir = out_dir
    mode = 'autonomous' if no_prompts else 'supervised'
    _backend = str(cfg.get_path('robot.backend') or '')
    tags = (['DRY RUN'] if robot.arm.dry_run else []) + (
        [_backend.upper().replace('_', ' ')] if _backend in ('sim', 'isaac_sim') else []) + [ins_mode.upper()]
    _bcfg = dict(cp.get('board') or {})
    board = Board(run_dir, mode, tags, sender=getattr(robot, 'live_frames', None),
                  rate_hz=float(_bcfg.get('rate_hz', 4.0)))

    def _safe_stop():
        """Brake whatever was streaming; the arm then holds where it is."""
        try:
            robot.arm.servo_stop()
        except Exception as exc:                       # noqa: BLE001
            log.error('servo_stop failed (%s)', exc)

    def _op_open():
        ok = robot.gripper.open('operator: open')
        if camp.trial is not None and ok:
            camp.trial.set_cell(gripped=False)
        return ok

    def _op_home():
        if not robot.gripper.open('operator: open before going home'):
            return False
        phase('reset')
        ok = reset.reset_robot(robot, cfg, 'operator: home')
        if camp.trial is not None:
            camp.trial.set_cell(gripped=False, arm='home' if ok else 'unknown')
        return ok

    cycle = 0

    def _trial_start(t):
        """Each trial its own folder: the app's CSVs, the end view, the in-hand fits and -- robot.record_trial --
        the motion (the run's recorder switches its file; the console stays one run.log)."""
        nonlocal out_dir, cycle
        out_dir, cycle = t.dir, t.k
        clock_rows.clear()
        slot_obs.clear()
        if _trial is not None and _trial.motion:
            _trial.switch(t.dir)

    camp = bt.Campaign(
        run_dir, n_trials, board, max_attempts=int(cp.get('max_attempts', 3)),
        max_consecutive_failures=int(cp.get('max_consecutive_failures', 3)),
        recover=bool(cp.get('recover', True)),
        confirm_recoveries=bool(cp.get('confirm_recoveries', True)) and not no_prompts and not robot.arm.dry_run,
        safe_stop=_safe_stop,
        actions={'o': ('open the gripper', _op_open),
                 'h': ('open the gripper and send the arm home (a guarded position move: check its path is clear)',
                       _op_home)},
        on_trial_start=_trial_start, link_check=getattr(robot.arm, 'link_down', None),
        link_hint='restart the FRI app on the smartPAD (FRI control mode %s, client command mode POSITION)' % (
            'CARTESIAN_IMPEDANCE_CONTROL' if kuka_imp else 'POSITION_CONTROL'),
        meta={'app': task.name, 'config': cfg.get('_config_path'), 'cell': cfg.get('cell'),
              'argv': sys.argv[1:], 'mode': mode, 'tags': tags, 'insertion_mode': ins_mode,
              'reset': reset_mode, 'git': bt.git_state(os.path.dirname(os.path.dirname(os.path.dirname(
                  os.path.abspath(__file__)))))})
    try:
        with open(os.path.join(run_dir, 'config_resolved.json'), 'w') as fh:
            json.dump(dict(cfg), fh, indent=1, default=str)
    except (OSError, TypeError) as exc:
        log.warning('config not recorded (%s)', exc)

    def tr():
        return camp.trial

    q_home = robot.arm.q()
    q_pick = q_home
    seed_q = q_home
    _T_tool0_conn = None          # the connector in tool0 as connector clocking left it (collar clocking reads it)
    T_ftip_conn_nominal = np.array(T_ftip_conn, dtype=float)
    # THE PLUG'S TURN ABOUT ITS OWN AXIS IN THE HAND, from the grasp frame (P6.grasp_junction): the identity for a
    # round plug; an RJ45's latch side, measured by the 6D scan -- the pick grasps the levelled frame, so a plug
    # lying latch DOWN is taken from above like any other, and the belief is T_ftip_conn_nominal @ this. Set by
    # each pick (step_pick), cleared with the belief (step_localize).
    latch_turn = np.eye(4)

    def _scan_turn():
        """The turn the scanner measured for its last pick (Pose6DScanner.turn); the identity for any other."""
        T = getattr(scanner, 'turn', None)
        return np.eye(4) if T is None else np.asarray(T, dtype=float)

    def _upright(T_ftip_conn_now):
        """The belief with the plug's turn taken out: where the connector frame would be had it lain z up -- what a
        put-down is written in, so the hand comes down upright and the plug lands as it was picked (an RJ45
        latch down stays latch down)."""
        return np.asarray(T_ftip_conn_now, dtype=float) @ inverse(latch_turn)

    held_start_used = False
    T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
    st = a.get('standoff', {}) or {}
    T_standoff_row = translation_matrix(
        np.asarray(st.get('axis', [-1, 0, 0]), dtype=float)
        * float(st.get('distance_m', 0.01))) @ mats[0]

    def traj_ref(row, T_tool0_conn, commit=False):
        """A TRAJECTORY row -> a tool0 reference, anchored so the path ends on the mate.

        ANCHORED (T_base_targetobj = the mate with the CSV's last row normalised away), which is
        why a preload must never be written as a trajectory row: it would be normalised out here
        and driven everywhere else.

        `commit=True` adds the final insertion's preload, so the press applies to the attempt
        meant to SEAT and to nothing that collects observations."""
        return (T_base_commit if commit else T_base_targetobj) @ row @ inverse(T_tool0_conn)

    # ---------------------------------------------------------------- START
    def step_start():
        """The run's start: home and the gripper warm-up (both skipped holding a part: they open the hand)."""
        nonlocal q_home
        # NO FIRST MOVE WHILE THE 6D MODELS LOAD (2026-11-06 11:22: the load stalled the PC ~0.4 s and the FRI link
        # dropped before the start reset moved) -- usually already done, started before the arm connected.
        if pose_method == 'pose6d' and not robot.arm.dry_run and not P6.wait_ready(0.0):
            log.info('START: waiting for the 6D estimator to finish loading before the arm moves.')
            if not P6.wait_ready(90.0):
                log.warning('START: the 6D estimator is still loading after 90 s -- going on.')
        if start_held:
            log.info('START HELD: no start reset and no gripper warm-up -- both open the hand.')
        else:
            phase('reset')
            if not reset.reset_robot(robot, cfg, 'start reset'):
                return camp.fail('the start reset did not reach home', 'intervene', arm='unknown')
        # THE GROUND-COLLISION MODEL IS CHECKED ONCE, HERE, against the controller's own forward
        # kinematics -- it is built from a DH chain written in this repo, and a typo there would
        # make every clearance it reports confidently wrong. On a dry run it reports UNVERIFIED
        # rather than pretending to have checked. Never fatal: an unchecked path is worse than a
        # guarded one, but better than refusing to run at all.
        grasp.verify_collision_model(robot)
        # GRIPPER WARM-UP instead of a plain open: full stroke, two partial cycles, end open.
        if not start_held and not robot.gripper.warmup():
            return camp.fail('the gripper warm-up failed', 'intervene')
        q_home = robot.arm.q()
        return True

    # ---------------------------------------------------------------- PREPARE
    def step_prepare():
        """The trial starts from an EMPTY HAND AT HOME. A first trial holding a part (start_held) is the one
        exception. Anything else in the fingers is a person's to take out."""
        t = tr()
        if start_held and not held_start_used:
            t.set_cell(connector='hand', gripped=True, arm='unknown')
            return True
        if robot.arm.dry_run:
            t.set_cell(connector='bench', assembly=None, gripped=False, arm='home')
            return True
        lim = int(cp.get('empty_max_counts', 30))
        while True:
            try:
                counts = int(robot.gripper.position())
                why = None if counts <= lim else (
                    f'the gripper is at {counts} counts, not open (campaign.empty_max_counts {lim}) -- something may '
                    'be in it: o opens it; take it out, then r')
            except Exception as exc:                   # noqa: BLE001
                why = f'the gripper cannot be read ({exc})'
            if why is None:
                break
            t.set_cell(gripped=None)
            choice = camp.intervene(why, 'Prepare: empty hand, home')
            if choice == 'quit':
                return False
            if choice == 'next':
                t.end_trial = True
                return camp.fail('the cell was not ready: ' + why, 'intervene')
        far = float(np.degrees(np.max(np.abs(np.asarray(robot.arm.q()) - np.asarray(q_home)))))
        if far > 2.0:
            log.info('PREPARE: the arm is %.1f deg from home -- going home first.', far)
            phase('reset')
            if not reset.reset_robot(robot, cfg, 'trial start: home'):
                return camp.fail('could not go home before the trial', 'intervene', arm='unknown')
        t.set_cell(connector='bench', assembly=None, gripped=False, arm='home')
        return True

    # ---------------------------------------------------------------- LOCALIZE
    def step_localize():
        """The socket measured (marker sweep / 6D) and seen for the stand-off plan -- every trial afresh."""
        online_stop()                        # no more contact: pose6d.online off
        nonlocal T_ftip_conn, latch_turn
        t = tr()
        _post.clear()
        _miss.clear()
        for k in ('seated', 'locked', 'tug', 'disassembled', 'insert_status', 'insert_depth_mm', 'screw_deg',
                  'belief_vs_target_mm', 'belief_vs_target_deg', 'push_home', 'latched', 'latch_turn_deg'):
            t.data.pop(k, None)                        # this attempt's, not the last one's
        if t.k > 1 or t.attempt > 1:
            # A FRESH PICK HAS A FRESH IN-HAND ERROR. The estimator spent the last attempt correcting
            # the belief for the PREVIOUS grasp; carrying that correction into a new grasp would start
            # the next insertion from a confidently wrong pose. And the cable was PLACED, so it is not
            # where it was picked from: the cached junction selection is stale.
            T_ftip_conn = np.array(T_ftip_conn_nominal, dtype=float)
            latch_turn = np.eye(4)
            if hasattr(scanner, 'reselect'):
                scanner.reselect()
        held = start_held and not held_start_used
        cell = dict(connector='hand', gripped=True) if held else dict(connector='bench', gripped=False)
        if held:
            # START HELD: the connector is in the hand, so this pass has no localization and no pick.
            # In pose6d mode the socket has no pose until it is measured: that one look happens anyway.
            if tgt_source == 'pose6d' and not pose6d_measured[0] and not locate_target_pose6d(q_home):
                return camp.fail('the 6D estimator did not find the socket', arm='unknown', **cell)
        else:
            if tgt_source == 'visual' and not locate_target_visually(q_home):
                return camp.fail('the marker sweep did not localize the socket', arm='unknown', **cell)
            # ONCE A RUN, WITH THE SCENE (the user, 2026-11-11: "only localize socket before picking cable once
            # (should be together with general scene scan)"): the fixture does not move, so the socket's look before
            # the pick -- the view that is also the stand-off plan's scene -- is made at the first attempt and kept
            # for every attempt and trial after it. The refine before the stand-off (refine_before_standoff) still
            # measures it again each attempt, the connector in the hand; a look that failed is tried again.
            # assembly.visual_target.localize_once false: looked at before every pick, as before.
            if tgt_source == 'pose6d':
                if bool(vt.get('localize_once', True)) and pose6d_measured[0]:
                    log.info('LOCALIZE: the socket as the run measured it (%s mm) -- not looked at again before this '
                             'pick (assembly.visual_target.localize_once).',
                             np.round(np.asarray(socket_obj[0])[:3, 3] * 1000.0, 1).tolist()
                             if socket_obj[0] is not None else '?')
                elif not locate_target_pose6d(q_home):
                    return camp.fail('the 6D estimator did not find the socket', arm='unknown', **cell)
        if plan_standoff and tgt_source != 'pose6d' and not socket_scene(q_home):
            return camp.fail('the socket views for the stand-off plan failed', arm='unknown', **cell)
        t.set_cell(arm='unknown', **cell)
        return True

    # ---------------------------------------------------------------- PICK
    def _hand():
        """The cell from what the GRIPPER says, after a failure in the pick: open, or closed on nothing -> the hand
        is empty and the cable is on the bench, wherever it lies (the next scan finds it -- the user, 2026-11-06:
        "the arm has no way of knowing the cable state before and it doesn't need to yet"); closed on something
        -> the cable is in it. Unreadable -> unknown."""
        if robot.arm.dry_run:
            return dict(connector='bench', gripped=False)
        try:
            counts = int(robot.gripper.position())
        except Exception:                              # noqa: BLE001
            return dict(connector='unknown', gripped=None)
        empty = int(cfg.get_path('grasp_check.empty_counts', 228) or 228)
        if counts <= int(cp.get('empty_max_counts', 30)) or counts >= empty - 2:
            return dict(connector='bench', gripped=False)
        return dict(connector='hand', gripped=True)

    def step_pick():
        """Scan, grasp, slip-checked lift -- with the grasp's own retries and the reorient recovery inside."""
        nonlocal q_pick, held_start_used, T_ftip_conn, latch_turn
        t = tr()
        if start_held and not held_start_used:
            if conn_tracker is not None:         # tracked in the hand from the belief, measured below
                conn_tracker.start(T_tool0_plug=robot.T_tool0_fingertip
                                   @ P6.plug_from_belief(T_ftip_conn, T_junction_conn, conn_tracker.part))
            held_start_used = True
            q_pick = q_home
            if cfg.get('pick_joints_deg') is not None:
                q_pick = list(np.radians(np.asarray(cfg.get('pick_joints_deg'), dtype=float)))
            t.data['picked'] = 'start_held'
            return True
        # THE PICK POSE. Home is the marker VIEW pose (the sweep above, and the end-of-run
        # image); the scan, the grasp geometry and every retry offset are written from HERE.
        q_pick = q_home
        _pick_deg = cfg.get('pick_joints_deg')
        if _pick_deg is not None:
            q_pick = list(np.radians(np.asarray(_pick_deg, dtype=float)))
            # Gate BEFORE reconfiguring: this is a large joint move away from the marker view and
            # into the pick pose, and everything downstream (scan, grasp, retry offsets) is
            # written from where it lands.
            if not phase_gate('RECONFIGURE FOR PICKUP',
                              'The arm will leave the marker view and move to the pick pose %s deg.'
                              % list(np.round(np.asarray(_pick_deg, dtype=float), 1))):
                return camp.fail('the operator stopped before the pick', 'abort')
            # Its OWN phase, not 'reset': this is a large free-space traverse with an EMPTY
            # gripper and nothing near the workpiece, so it has no reason to be paced like a
            # move that ends in contact.
            phase('reconfigure')
            if not robot.arm.move_j(q_pick, label='pick pose'):
                return camp.fail('could not reach pick_joints_deg', arm='unknown')
        t.set_cell(arm='pick')

        attempt = 0
        reoriented = False
        runner = StepRunner(log, confirm=confirm is not None)
        if not phase_gate('PICK THE CABLE',
                          'Next the arm localizes the cable and grasps it.'):
            return camp.fail('the operator stopped before the pick', 'abort')
        while True:
            phase('scan')
            result = _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                           offset_x_m=retry_offset_x(attempt, check.retry_perturb_x_m))
            t.data['grasps'] = attempt + 1
            if result == 'ok':
                t.set_cell(connector='hand', gripped=True, arm='unknown')
                # THE PLUG'S TURN IN THE HAND (an RJ45's latch side; the identity for the BNC): the grasp was
                # written in the levelled frame, so the belief takes the turn the scan measured.
                latch_turn = _scan_turn()
                T_ftip_conn = np.asarray(T_ftip_conn_nominal, dtype=float) @ latch_turn
                if dry is not None:                  # the dry cell's plug, held where the belief says
                    dry.jack.hold(robot.T_tool0_fingertip @ T_ftip_conn)
                if not np.allclose(latch_turn, np.eye(4)):
                    t.data['latch_turn_deg'] = round(P6.turn_deg(latch_turn), 1)
                    log.info('THE CONNECTOR IN THE HAND, TURNED %+.0f deg about its own axis from the grasp frame -- '
                             'its latch side as the 6D scan saw it; the belief carries it.', P6.turn_deg(latch_turn))
                # THE CONNECTOR ON THE TOOL from the close: its CAD at the expected pose (the
                # in-hand belief) -- the lift's move check carries it, the live view draws it.
                held = held_connector(T_ftip_conn) if hasattr(robot, 'hold') else None
                if held is not None:
                    robot.hold(held)
                status = {}

                def do_lift(_s=status):
                    _s['r'] = grasp.lift_verified(
                        robot, geom, check, 'lift',
                        position_guard=lambda mv: _guarded(robot, guard_shared, mv))
                    return _s['r'] == 'ok'

                if runner.run([('lift (slip-checked)', do_lift)]):
                    break
                result = status.get('r')
                if result != 'slipped':
                    return camp.fail(f'the lift did not complete ({result})', arm='unknown', **_hand())
            if result == 'abort':
                # Nothing was touched: look again (the next attempt). Twice the same -> a person (campaign).
                return camp.fail(getattr(geom, 'pick_abort_reason', None)
                                 or 'no cable to pick: the scan found none (or its labelling was declined)', **_hand())
            # NO COLLISION-FREE PATH TO THE COAXIAL GRASP. Retrying the identical approach
            # cannot help -- the geometry, not the attempt, is what refused. Square the cable
            # up once and let the scan try again on a heading that works.
            if result == 'unreachable':
                if reoriented:
                    return camp.fail('the coaxial grasp is still unreachable after the cable was squared up -- '
                                     'its heading was not the problem', **_hand())
                reoriented = True
                t.data['reoriented'] = True
                if not reorient_recovery():
                    # Refused before it touched anything, or it let go of the cable: either way the hand says where
                    # the cable is -- empty: on the bench, and the next attempt looks again.
                    return camp.fail('the reorient recovery (square pick, aligned place) failed', arm='unknown',
                                     **_hand())
                continue
            if attempt >= check.max_retries:
                t.data['reset_cable_wanted'] = True
                return camp.fail(f'the grasp failed on all {check.max_retries + 1} attempts ({result})',
                                 connector='bench', gripped=False, arm='pick')
            attempt += 1
            log.warning('Grasp %s -- recovering (attempt %d/%d).', result, attempt + 1,
                        check.max_retries + 1)
            if result == 'slipped' and hasattr(scanner, 'reselect'):
                phase('scan')
                T_up = translation_matrix([0.0, 0.0, check.slip_raise_m]) @ robot.tool0()
                if not (robot.gripper.open('drop')
                        and _guarded(robot, guard_shared,
                                     lambda: robot.arm.move_l(T_up, label='slip recovery (up)'))):
                    return camp.fail('the slip recovery move failed', arm='unknown', **_hand())
                t.set_cell(connector='bench', gripped=False)
                scanner.reselect()
            else:
                phase('reset')
                if not (robot.gripper.open('drop')
                        and robot.arm.move_j(q_pick, label='pick pose')):
                    return camp.fail('could not return to the pick pose after a failed grasp', arm='unknown',
                                     **_hand())
                t.set_cell(connector='bench', gripped=False, arm='pick')
        t.set_cell(connector='hand', gripped=True, arm='unknown')
        return True

    # ---------------------------------------------------------------- IN HAND
    def step_in_hand():
        """The connector's pose in the hand: where the commanded grasp puts it (+ belief_offset), or the scan's
        estimate held rigidly / the 6D tracker's (pose6d.rigid_grasp / pose6d.in_hand). Never a reason to fail."""
        nonlocal T_ftip_conn
        slot_obs.clear()                         # a new grasp: its slots are where this one put them
        # ---- THE CONNECTOR IN THE HAND, HELD RIGIDLY (pose_method pose6d, pose6d.rigid_grasp) --
        # The scan's estimate -- made before the grasp, from views the camera reads well -- carried
        # with the flange from the moment the fingers closed on it (the user, 2026-10-30: in the
        # hand the plug is too near the camera for a good estimate). Where the flange WAS at the
        # close, not where the grasp was commanded: a reseat's or a retry's offset is in it. No
        # belief_offset: that is the part moving as the jaws close, which this assumes it does not.
        T_closed = getattr(geom, 'T_base_tool0_closed', None)
        close_report(T_closed, T_ftip_conn)
        if pose_method == 'pose6d' and bool(cfg.get_path('pose6d.rigid_grasp', True)) and T_closed is not None:
            T_rigid = rigidly_held_junction(robot.T_tool0_fingertip, T_closed,
                                            geom.T_base_detection @ latch_turn) @ T_junction_conn
            d_lin, d_ang = pose_error(inverse(_grip_off) @ T_junction_conn, T_rigid)
            log.info('THE CONNECTOR IN THE HAND: the scan\'s estimate, held rigidly from the close -- %.1f mm / '
                     '%.1f deg from where the commanded grasp put it%s.', 1000.0 * d_lin, np.degrees(d_ang),
                     '; belief_offset not applied' if np.any(_bo_xyz) or np.any(_bo_rpy) else '')
            T_ftip_conn = T_rigid
            geom.T_base_tool0_closed = None      # this grasp's: a later pass picks again

        # ---- THE CONNECTOR IN THE HAND, MEASURED (pose_method pose6d, pose6d.in_hand) ----------
        # The camera sees the held connector: once enough in-hand estimates agree, the belief takes
        # their axis and position (its turn about the axis is the pick's -- a view cannot tell it).
        # Replaces the whole belief, belief_offset included: the measurement sees how it seated.
        if conn_tracker is not None:
            T_ftip_conn = conn_tracker.measured_belief(T_ftip_conn, robot.T_tool0_fingertip, T_junction_conn)
        if not (pose_method == 'pose6d' and bool(cfg.get_path('pose6d.rigid_grasp', True))) and conn_tracker is None:
            log.info('THE CONNECTOR IN THE HAND: at the EXPECTED pose, where the commanded grasp puts it -- '
                     'no 6D estimate (pose6d.rigid_grasp and pose6d.in_hand off).')

        # NO FIT TO THE WRIST IMAGE (the user, 2026-11-06: "remove vision based in-hand pose estimation of
        # connector -- the estimator doesn't fit the outline very well"): the in_hand_rgb silhouette fit and
        # the in_hand_fit CAD fit are no longer run; the belief is the grasp's (+ belief_offset) or the 6D's above.
        if getattr(robot, 'held', None) is not None:      # the CAD on the tool follows the belief
            held = held_connector(T_ftip_conn)
            if held is not None and not np.allclose(held.T_tool0_obj, robot.held.T_tool0_obj, atol=1e-6):
                robot.hold(held)
        _x, _r = matrix_to_xyzrpy(T_ftip_conn)
        tr().data['belief_xyz_mm'] = np.round(_x * 1000.0, 2).tolist()
        tr().data['belief_rpy_deg'] = np.round(np.degrees(_r), 2).tolist()
        return True

    # ---------------------------------------------------------------- APPROACH
    def step_approach():
        """The socket looked at again, the stand-off reached (planned round what the camera saw), the cable
        checked still held, and -- SUPERVISED -- the unconditional gate before contact."""
        nonlocal seed_q, T_tool0_conn
        held = dict(connector='hand', gripped=True)
        # THE SOCKET LOOKED AT AGAIN, FROM SEVERAL VIEWS, just before the stand-off (the user,
        # 2026-11-03): the pick's single view was coarse; this is the pose the contact is made at.
        if (tgt_source == 'pose6d' and (vt.get('refine_before_standoff') or {}).get('enabled', False)
                and not locate_target_pose6d(None, refine=True)):
            return camp.fail('the socket refinement before the stand-off failed', arm='unknown', **held)
        phase('standoff')
        seed_q = robot.arm.q()
        T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
        if plan_standoff:
            q = plan_to_standoff(traj_ref(T_standoff_row, T_tool0_conn), T_ftip_conn)
            if q is None:
                return camp.fail('no path to the stand-off', arm='unknown', **held)
        else:
            q = robot.arm.ik(traj_ref(T_standoff_row, T_tool0_conn), seed_q)
            if q is None or not _guarded(robot, guard_shared,
                                         lambda: robot.arm.move_j(q, label='stand-off')):
                return camp.fail('could not reach the stand-off', arm='unknown', **held)
        seed_q = q
        if not verify_cable_held(robot, check, 'stand-off'):
            return camp.fail('the cable is no longer in the hand at the stand-off', connector='unknown',
                             gripped=False, arm='clear')
        tr().set_cell(arm='clear', **held)
        slot_capture('standoff')               # the slots seen, the arm still (connector clocking aims by them)
        # Asked even with --yes -- this is the boundary between free space and contact. Only
        # skip_prompts (--no-prompts) silences it.
        if not robot.arm.dry_run and not no_prompts:
            ans = ask('\n[stand-off] Ready to ENGAGE (contact ahead). Enter to continue (q to abort): ')
            if ans is not None and ans.strip().lower() in ('q', 'quit', 'n', 'no'):
                return camp.fail('stopped at the stand-off by the operator', 'abort')
        return True

    # ---------------------------------------------------------------- ONLINE 6D THROUGH CONTACT
    # pose6d.online (the user, 2026-11-11: "update both the in-hand connector pose and the socket pose via the
    # RGBD data online during all contact phases"). From the insertion on, the socket (SocketTracker) and the
    # connector in the hand (the in-hand ConnectorTracker, restarted from the belief: the socket refine had
    # stopped it) are estimated from the wrist camera's RGB-D frames, one refine after another, and drawn
    # live -- through the insertion, connector clocking, the collar lock and the tug, until the escape lets go.
    # THE INSERTION FOLLOWS THEM (steer, engage mode): every reference of the creep to contact and of the
    # engage is corrected to the agreed estimates in the compliance controller (OnlinePoses, the arm's
    # reference_correction), gradually and within limits; when it ends the corrected socket pose and in-hand
    # belief become the run's. AFTER IT, ESTIMATED, NOT FOLLOWED: once engaged the socket holds the plug and
    # the clocking frames are built from the engaged pose -- a reference moved by vision would push a seated
    # plug sideways against the socket that constrains it.
    _onl = online_cfg(cfg)
    online_on = bool(_onl.get('enabled', False)) and (sock_tracker is not None or conn_tracker is not None)
    online = [None]                      # the OnlinePoses steering the insertion, while it does

    def online_begin(steer):
        """Track the socket and the held connector from the run's beliefs; with `steer`, correct the contact
        references to them (the arm's reference_correction)."""
        if not online_on:
            return
        if conn_tracker is not None and not conn_tracker.running():
            conn_tracker.start(T_tool0_plug=robot.T_tool0_fingertip
                               @ P6.plug_from_belief(T_ftip_conn, T_junction_conn, conn_tracker.part))
        if sock_tracker is not None and socket_obj[0] is not None and not sock_tracker.running():
            sock_tracker.start(T_base_plug=socket_obj[0])
        if steer and bool(_onl.get('steer', True)) and socket_obj[0] is not None:
            _sock = P6.socket_part(cfg)
            online[0] = OnlinePoses(cfg, robot, sock_tracker, conn_tracker, socket_obj[0], T_ftip_conn,
                                    T_junction_conn,
                                    lambda T: P6.target_from_socket(T, T_junction_conn, socket=_sock))
            robot.arm.reference_correction = online[0]
            log.info('6D ONLINE: the socket and the connector in the hand are estimated through contact; the '
                     'insertion follows them (at most %s mm / %s deg for the socket, %s mm / %s deg in the hand, '
                     '%s mm/s, %s deg/s%s).', _onl.get('max_socket_mm', 5.0), _onl.get('max_socket_deg', 5.0),
                     _onl.get('max_hand_mm', 5.0), _onl.get('max_hand_deg', 5.0), _onl.get('rate_mm_s', 1.0),
                     _onl.get('rate_deg_s', 1.0), '' if _onl.get('axial', False) else '; not along the axis')
        else:
            log.info('6D ONLINE: the socket and the connector in the hand are estimated through contact (not '
                     'followed%s).', '' if steer else ' in this insertion mode')

    def online_end(T_last_ref=None):
        """Stop steering; the corrected socket pose and in-hand belief become the run's (the frames re-anchored on
        them). Returns `T_last_ref` as the arm was last commanded (corrected), for what backs out from it."""
        nonlocal T_ftip_conn, T_tool0_conn
        oc, online[0] = online[0], None
        robot.arm.reference_correction = None
        if oc is None:
            return T_last_ref
        out = None if T_last_ref is None else oc._L @ np.asarray(T_last_ref, dtype=float) @ oc._R
        T_s, F, (s_mm, s_deg), (h_mm, h_deg) = oc.applied()
        t = tr()
        t.data.update(online_socket_mm=round(s_mm, 2), online_socket_deg=round(s_deg, 2),
                      online_hand_mm=round(h_mm, 2), online_hand_deg=round(h_deg, 2),
                      online_estimates=dict(oc.used))
        log.info('6D ONLINE: the insertion followed the socket %.2f mm / %.2f deg and the connector in the hand '
                 '%.2f mm / %.2f deg (%d socket, %d in-hand estimates); the run is re-anchored on them.',
                 s_mm, s_deg, h_mm, h_deg, oc.used['socket'], oc.used['hand'])
        if s_mm > 1e-6 or s_deg > 1e-6:
            socket_obj[0] = np.asarray(T_s, dtype=float)
            _anchor_target(P6.target_from_socket(T_s, T_junction_conn, socket=P6.socket_part(cfg)))
        if h_mm > 1e-6 or h_deg > 1e-6:
            T_ftip_conn = np.asarray(F, dtype=float)
            T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
            if getattr(robot, 'held', None) is not None:      # the CAD on the tool follows the belief
                held = held_connector(T_ftip_conn)
                if held is not None:
                    robot.hold(held)
        return out

    def online_stop():
        """No more contact (the escape lets go, a recovery, a new attempt, the trial's end): steering off, both
        trackers stopped -- the GPU is the next look's."""
        if online[0] is not None:
            online_end()
        robot.arm.reference_correction = None
        if online_on:
            if sock_tracker is not None:
                sock_tracker.stop()
            if conn_tracker is not None and conn_tracker.running():
                conn_tracker.stop()

    # ---------------------------------------------------------------- INSERT
    def step_insert():
        """Into the socket: ENGAGE (the trajectory home with the wiggle), RL (the policy, through the seat)
        or ESTIMATE (insert, observe, correct the belief, repeat; then the commit)."""
        nonlocal T_ftip_conn, seed_q, T_tool0_conn
        t = tr()
        est_rows, success, engage_missed, why = [], False, False, ''
        last_ref = robot.tool0()
        acc = np.zeros((0, 12))
        T_cum = np.eye(4)
        trackc = [np.zeros(len(estimator.estimate_dims))] if estimator is not None else []
        trackr, trackg = [], []
        at_socket = dict(connector='socket', gripped=True, arm='socket')
        online_begin(steer=ins_mode == 'engage')
        try:
            if ins_mode == 'engage':
                # ENGAGE replaces the estimate loop and the commit.
                #
                # A FORCE STOP IS THE SUCCESS. Meeting the socket is what the phase is for, so
                # 'force' continues to the bayonet search. 'complete' means the whole path ran --
                # trajectory AND preload -- without ever reaching the axial limit, i.e. the
                # connector never touched anything. That is a MISS, not a completion, and clocking
                # from there would turn a connector that is not in a socket.
                en_status, last_ref, en_depth = engage_insertion()
                t.data.update(insert_status=en_status, insert_depth_mm=en_depth)
                if _engage_extra.get('push_home') is not None:
                    t.data['push_home'] = _engage_extra['push_home']
                # SUCCESS IS POSITIVE EVIDENCE ONLY: the part travelled, or it pushed back hard
                # enough for long enough. Everything else -- the path running out, the timeout,
                # a jam, a failed seat confirmation -- is a miss, and they all take the same
                # recovery: back out (checked), put the cable down and measure everything again.
                if en_status == 'aborted':
                    log.warning('ENGAGE aborted by the operator at the contact gate -- the arm is '
                                'LEFT WHERE IT IS with the cable held and touching the socket. '
                                'No recovery motion is attempted; that is what abort means.')
                    raise bt.StepFailed('the operator aborted at the contact gate', 'abort', **at_socket)
                elif en_status == 'unreachable':
                    # The path's start did not solve: the arm never left the stand-off, so the cable is in the
                    # hand, clear of the socket -- laid down at the reset pose, not backed out of anything.
                    raise bt.StepFailed('the engage path does not IK-solve from the stand-off', connector='hand',
                                        gripped=True, arm='clear')
                elif en_status == 'move_failed':
                    raise bt.StepFailed('the move to the engage start did not complete', connector='hand',
                                        gripped=True, arm='unknown')
                elif en_status in ('complete', 'timeout', 'guard'):
                    why = ('ENGAGE FAILED (%s): neither %.1f mm of travel nor %.1f N held %.2f s was reached'
                           % ({'complete': 'the path ran out',
                               'timeout': 'timed out after %.0f s' % en_timeout_s,
                               'guard': 'the general force guard tripped -- a JAM'}[en_status],
                              en_travel_mm, en_fmax, en_fpers))
                    log.error('%s, so the connector is not seated. Treating this as a FAILED engagement.', why)
                    engage_missed = True
                elif en_status == 'unconfirmed':
                    # The axial limit WAS met, so something resisted -- but the confirmation says
                    # it does not behave like a socket. Same recovery as a clean miss.
                    why = 'ENGAGE UNCONFIRMED: the connector met resistance but failed the seat confirmation'
                    log.error('%s, so it is most likely against a face rather than in the socket. Treating this '
                              'as a FAILED engagement.', why)
                    engage_missed = True
                else:
                    success = en_status in ('force', 'travel')
                est_rows.append({'attempt': 'engage', 'status': en_status,
                                 'depth_mm': en_depth, 'success': bool(success)})
            if ins_mode == 'rl':
                # THE POLICY replaces the engage, its confirmation and connector clocking: it ends SEATED
                # or it missed -- and a miss takes the engage's recovery (back out, put down, re-pick).
                rl_status, rl_info = rl_insertion()
                last_ref = robot.tool0()
                success = rl_status == 'seated'
                engage_missed = rl_status in ('timeout', 'turn')
                why = f'the RL insertion ended {rl_status}'
                t.data.update(insert_status=rl_status, insert_depth_mm=rl_info.get('advance_mm', 0.0))
                est_rows.append({'attempt': 'rl', 'status': rl_status, 'depth_mm': rl_info.get('advance_mm', 0.0),
                                 'success': bool(success)})
            for it in (range(1, max_attempts + 1) if ins_mode == 'estimate' else ()):
                T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
                e_xyz, e_rpy = matrix_to_xyzrpy(T_ftip_conn)
                log.info('--- attempt %d/%d --- in-hand estimate xyz=%s mm rpy=%s deg', it,
                         max_attempts, np.round(e_xyz * 1000, 2).tolist(),
                         np.round(np.degrees(e_rpy), 2).tolist())

                # COLLECTION PASSES: the sweep commands one insertion per deliberate offset with
                # the belief held FIXED across passes, so their evidence fuses exactly.
                passes = ([list(o) for o in sweep_offsets] if col_mode == 'offset_sweep'
                          else [None])
                obs, lin, ang = [], 0.0, 0.0
                for pi, poff in enumerate(passes):
                    if tn_on or poff is not None:
                        rows_t = traj.noised(dense, noise_rng, tn_std if tn_on else [0.0] * 6,
                                             tn_w, tn_dt, (1.0 - tn_da) ** (it - 1), poff)
                    else:
                        rows_t = dense
                    refs = [traj_ref(row, T_tool0_conn) for row in rows_t]
                    phase('standoff')
                    label = (f'attempt {it}'
                             + (f' sweep {pi + 1}/{len(passes)}' if poff is not None else '')
                             + ' start')
                    q = robot.arm.ik(refs[0], seed_q)
                    if q is None or not _guarded(robot, guard_shared,
                                                 lambda: robot.arm.move_j(q, label=label)):
                        raise bt.StepFailed('could not reach the insertion pass start', **at_socket)
                    seed_q = q
                    phase('assemble')
                    # Intermediate passes MUST back off -- the next realigns to a different offset's
                    # start. The LAST pass stays at its stop, so a mate the operator calls successful
                    # leaves the arm AT the seat for clocking to anchor on.
                    obs_i, lin, ang, stops, last_ref = run_insertion(
                        adm, refs, T_tool0_conn, peck=(col_mode == 'peck'),
                        retract=(pi < len(passes) - 1))
                    obs.extend(obs_i)
                    if poff is not None:
                        log.info('  sweep %d/%d (pitch %+.1f deg, z %+.1f mm): %d obs, stop %s mm.',
                                 pi + 1, len(passes), poff[4], poff[2] * 1000.0, len(obs_i),
                                 [round(s, 1) for s in stops])
                    if not verify_cable_held(robot, check, f'attempt {it} pass {pi + 1}'):
                        raise bt.StepFailed('the cable left the hand during the insertion', 'intervene',
                                            connector='unknown', gripped=False, arm='socket')

                log.info('check: believed connector vs target: %.2f mm, %.2f deg (tol %.2f mm, '
                         '%.2f deg)', lin * 1000, np.degrees(ang), tol_pos_m * 1000,
                         np.degrees(tol_rot_rad))
                _save_observations(os.path.join(out_dir, f'attempt_{it:02d}_observations.csv'), obs)
                row = {'attempt': it, 'n_observations': len(obs), 'n_passes': len(passes),
                       'check_pos_mm': lin * 1000.0, 'check_rot_deg': float(np.degrees(ang))}

                # SUCCESS is the operator's call -- they can see the physical mate; the kinematic
                # numbers only see the belief. A dry run has no operator.
                if robot.arm.dry_run or no_prompts:
                    row['success'] = bool(lin <= tol_pos_m and ang <= tol_rot_rad)
                else:
                    ans = ask(f'[check attempt {it}] Was the assembly SUCCESSFUL? '
                              '(y = done / Enter = retry / q = abort): ')
                    ans = (ans or '').strip().lower()
                    if ans in ('q', 'quit'):
                        est_rows.append(row)
                        raise bt.StepFailed('the operator stopped at the success check', 'abort', **at_socket)
                    row['success'] = ans in ('y', 'yes')
                if row['success']:
                    est_rows.append(row)
                    # NO retract: the connector is mated and the arm stays on it, which is what makes
                    # this pose the ENGAGED pose the clocking maneuvers anchor to.
                    log.info('ASSEMBLY COMPLETE on attempt %d -- holding the seat (connector ENGAGED).',
                             it)
                    success = True
                    break
                # Not successful: NOW back off, because the next thing (another attempt, or the final
                # insertion) approaches its own start under position control and needs the clearance.
                phase('retract')
                last_ref = retract_from(last_ref, T_tool0_conn)
                if it == max_attempts:
                    est_rows.append(row)
                    log.error('Attempt limit reached (%d) without a successful mate.', max_attempts)
                    break

                # ---- ESTIMATE. Accumulated rows are re-projected into the belief after every
                # update, exactly as in estimator_eval, so old evidence stays valid.
                obs_arr = np.asarray(obs, dtype=float).reshape(-1, 12)
                full = np.vstack([acc, obs_arr]) if accumulate and len(acc) else obs_arr
                vec6, w6 = estimator.prepare_observations(full[:, :6], full[:, 6:9], full[:, 9:12])
                if commit == 'argmin':
                    T_corr_mm, info, land_pack = _argmin_estimate(estimator, vec6, w6)
                else:
                    T_corr_mm, info = estimator.estimate(vec6, w6)
                    land_pack = None
                if T_corr_mm is None:
                    log.warning('Estimation skipped (%s) -- retrying with the UNCHANGED belief.',
                                info)
                    row['estimate'] = f'skipped: {info}'
                    est_rows.append(row)
                    trackc.append(trackc[-1])
                    trackr.append(float('nan'))
                    trackg.append(np.zeros(0))
                    if accumulate:
                        acc = full
                    continue
                log.info('belief correction: %s  (residual %.3f, %d obs, %s)',
                         {k: round(v, 3) for k, v in info['theta_corr'].items()},
                         info['final_residual'], info['n_observations'], commit)
                if dbg_on:
                    try:
                        manifold_debug.figures(
                            estimator, vec6, w6, dict(info['theta_corr']), None,
                            os.path.join(out_dir, f'attempt_{it:02d}_match.png'),
                            title=f'attempt {it} (no ground truth)',
                            max_rows=int(dbg.get('max_rows', 250)),
                            grid_n=int(dbg.get('grid_points', 41)), live_dir=dbg_live)
                    except Exception as exc:               # noqa: BLE001 -- never fatal
                        log.warning('match diagnostics skipped (%s)', exc)

                T_ftip_conn = T_ftip_conn @ _corr_to_m(T_corr_mm)      # believed @ corr ~= true
                T_cum = T_cum @ np.asarray(T_corr_mm, dtype=float)
                if accumulate:
                    from .estimator_eval import _rebase_rows
                    acc = _rebase_rows(full, T_corr_mm) if len(full) else full
                trackc.append(vec6_from_mats(T_cum)[estimator.idx])
                trackr.append(float(info['final_residual']))
                trackg.append(np.asarray(info['res_hist'], dtype=float)[:, -1]
                              if info.get('res_hist') is not None else np.zeros(0))
                _plot_run(os.path.join(out_dir, 'run_corrections.png'), estimator.estimate_dims,
                          trackc, trackr, trackg,
                          f'attempt {it}/{max_attempts} | check {lin * 1000:.1f} mm', live_path)
                row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
                row.update({'icp_residual': info['final_residual'], 'commit': commit})
                est_rows.append(row)

            # ---- FINAL INSERTION -- the COMMIT, from the final corrected belief, zero noise ----
            if fi_on and not success and ins_mode == 'estimate':
                log.info('FINAL INSERTION from the corrected belief (zero noise).')
                T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
                rows_f = (traj.noised(dense, noise_rng, fi_noise_std, fi_noise_w, 0.0, 1.0)
                          if fi_noise_on else dense)
                refs = [traj_ref(row_, T_tool0_conn, commit=True) for row_ in rows_f]
                phase('standoff')
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not _guarded(robot, guard_shared,
                                             lambda: robot.arm.move_j(q, label='final start')):
                    log.warning('IK/approach failed for the final insertion.')
                else:
                    seed_q = q
                    phase('assemble')
                    # retract=False: this is the attempt meant to SEAT. Backing out of it would undo
                    # the mate before the operator can judge it and would leave the clocking maneuvers
                    # anchored on a retracted pose.
                    obs_f, lin, ang, _stops, last_ref_f = run_insertion(
                        adm_final, refs, T_tool0_conn, guard_ctl=guard_final, settle=fi_settle,
                        hold=fi_hold, speed=(fi_v, fi_wr) if (fi_v or fi_wr) else None,
                        pause=fi_pause, retract=False)
                    last_ref = last_ref_f
                    _save_observations(os.path.join(out_dir, 'final_insertion_observations.csv'),
                                       obs_f)
                    frow = {'attempt': 'final_insertion', 'n_observations': len(obs_f),
                            'check_pos_mm': lin * 1000.0,
                            'check_rot_deg': float(np.degrees(ang))}
                    if robot.arm.dry_run or no_prompts:
                        frow['success'] = bool(lin <= tol_pos_m and ang <= tol_rot_rad)
                    else:
                        ansf = (ask('[final insertion] SUCCESSFUL? (y/n): ') or '').strip().lower()
                        frow['success'] = ansf in ('y', 'yes')
                    success = success or frow['success']
                    est_rows.append(frow)
                    if not success:
                        # Only NOW back off -- a failed commit has nothing to hold on to, and the
                        # recovery expects clearance.
                        phase('retract')
                        last_ref = retract_from(last_ref_f, T_tool0_conn, adm_final)
            if ins_mode == 'estimate' and not success:
                why = 'the estimate/insert loop and the final insertion did not mate it'
        finally:
            robot.arm.servo_stop()
            last_ref = online_end(last_ref)
            if est_rows:
                keys = sorted({k for r in est_rows for k in r}, key=str)
                with open(os.path.join(out_dir, 'estimates.csv'), 'w', newline='') as fh:
                    w = _csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(est_rows)
                log.info('Per-attempt log: %s', os.path.join(out_dir, 'estimates.csv'))
        t.data['engage_missed'] = bool(engage_missed)
        if engage_missed:
            t.data['misses'] = t.data.get('misses', 0) + 1
        if success:
            t.set_cell(assembly='seated' if ins_mode == 'rl' else 'engaged', **at_socket)
            return True
        # A MISS IS RECOVERABLE: the connector is in the hand at (or against, or past) the socket. The
        # recovery backs it out along the socket axis, CHECKS that it came out, lays it down, and the next
        # attempt measures everything again (place_after_failed_engage).
        _miss['ref'] = last_ref
        _miss['T_tool0_conn'] = robot.T_tool0_fingertip @ T_ftip_conn
        raise bt.StepFailed(why or 'the insertion did not mate the connector', assembly=None, **at_socket)

    _miss = {}                    # where a failed insertion stopped: the extract recovery backs out from there

    # ---------------------------------------------------------------- SEAT (connector clocking)
    def _at_the_mate():
        """The belief against the target, reported, and T_clk -- the frame the post-mate maneuvers (and the tug)
        build their axes from -- bound (assembly.post_engage_frame). Run once the plug is in the socket."""
        nonlocal T_clk
        t = tr()
        # ALWAYS REPORTED, whichever frame the maneuvers then use. At the mate the part is
        # physically IN the socket, so the believed connector and the target frame describe
        # the same thing -- and any gap between them is in-hand belief error, measured
        # exactly where it matters. Decomposed in the CONNECTOR's own axes because that is
        # where the fix lives: along-axis is insertion depth or belief_offset x, lateral and
        # vertical are the grasp, and roll is the clock angle.
        _believed = robot.tool0() @ T_tool0_conn
        _d = inverse(T_base_tconn) @ _believed
        _dx, _dr = matrix_to_xyzrpy(_d)
        log.info('BELIEF vs TARGET at the mate (connector frame): along-axis %+.2f mm, '
                 'lateral %+.2f mm, vertical %+.2f mm | roll %+.2f, pitch %+.2f, yaw %+.2f '
                 'deg. Zero means the belief agreed with the socket; anything else is the '
                 'in-hand error the insertion had to absorb.',
                 _dx[0] * 1000.0, _dx[1] * 1000.0, _dx[2] * 1000.0,
                 *np.degrees(_dr))
        t.data['belief_vs_target_mm'] = np.round(_dx * 1000.0, 2).tolist()
        t.data['belief_vs_target_deg'] = np.round(np.degrees(_dr), 2).tolist()
        if pe_frame == 'believed':
            # The believed connector frozen in base coordinates NOW, while the arm still grips
            # it -- the maneuvers need one consistent frame, not one re-derived per use.
            T_clk = _believed
            log.info('Post-engage frame: BELIEVED connector.')
        else:
            T_clk = T_base_tconn
            log.info('Post-engage frame: TARGET connector (recorded socket pose).')
        if cc_level:
            # LEVELLED: a wall socket's axis lies in the base plane; the 6D estimate pitches it a few degrees (it read
            # pitched down before the low view). The screw turns about the level axis through the same point, on the
            # same heading -- the belief reset that opens connector clocking re-derives the connector from it, so the
            # arm does not move here; it is the turn that stops precessing round the true axis.
            T_lev, _tilt = level_axis(T_clk, max_deg=cc_level_max)
            t.data['clock_axis_tilt_deg'] = round(_tilt, 2)
            if T_lev is not T_clk:
                log.info('CLOCKING AXIS LEVELLED: the screw axis stood %.2f deg off the base plane -- turned onto it '
                         'about its own origin, its heading kept (connector_clocking.level_axis).', _tilt)
                T_clk = T_lev
            elif _tilt > cc_level_max:
                log.warning('CLOCKING AXIS NOT LEVELLED: it stands %.1f deg off the base plane, over '
                            'connector_clocking.level_max_deg %.0f -- taken as meant (not a wall socket).', _tilt,
                            cc_level_max)

    def step_seat():
        """The bayonet: connector clocking rocks the plug under an axial load until the cams pull it home
        (or the RL policy already did)."""
        nonlocal _T_tool0_conn
        t = tr()
        state = 'engaged'                    # the initial assembly mated it; that is where we are
        _at_the_mate()
        if ins_mode != 'rl':
            slot_capture('engaged')            # the slots as the search will find them (the arm still, mated)
        if ins_mode != 'rl' and not phase_gate('CONNECTOR CLOCKING (insert)',
                                               'The connector is ENGAGED. Next is the bayonet screw, which '
                                               'cams it HOME -- check the engagement looks right first.'):
            robot.arm.servo_stop()
            return camp.fail('the operator stopped before connector clocking', 'abort')
        try:
            res = rl_seated() if ins_mode == 'rl' else connector_clocking()
        finally:
            robot.arm.servo_stop()
            _save_clocking()
        # (connector_clocking answers three values when the gripper would not open after it)
        cc_ok, _T_tool0_conn, T_base_conn = res[0], res[1], res[2]
        cc_screw_deg = res[3] if len(res) > 3 else None
        t.data.update(seated=bool(cc_ok), screw_deg=cc_screw_deg)
        _post.update(T_base_conn=T_base_conn, cc_screw_deg=cc_screw_deg, state=state)
        opened = cc_open_after and len(res) > 3
        if cc_ok:
            _post['state'] = _advance_state(state, 'engaged')          # -> seated
            t.set_cell(connector='socket', assembly='seated', gripped=not opened, arm='socket')
            return True
        log.error('CONNECTOR CLOCKING FAILED after %d tr%s (every stroke was stopped by the '
                  'force guard) -- the connector is ENGAGED but NOT SEATED.',
                  cc_tries, 'y' if cc_tries == 1 else 'ies')
        return camp.fail('connector clocking did not seat it', connector='socket', assembly='engaged',
                         gripped=not opened, arm='socket')

    _post = {}                    # what the post-mate steps hand on: T_base_conn, cc_screw_deg, state, tug_res

    def _save_clocking():
        """clock_rows -> <trial>/clocking.csv (every leg, the extract, the tug). Called on the way out of each
        post-mate step, so a failure or a Ctrl+C mid-leg still leaves it (the cycle loop's `finally` did that
        before the tree; restored 2026-11-06)."""
        if not clock_rows:
            return
        try:
            keys = list(dict.fromkeys(k for r in clock_rows for k in r))
            with open(os.path.join(out_dir, 'clocking.csv'), 'w', newline='') as fh:
                w = _csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(clock_rows)
        except OSError as exc:
            log.warning('clocking.csv not written (%s)', exc)

    # ---------------------------------------------------------------- LATCH (Task.latch: the RJ45)
    def step_latch():
        """A LATCHING PLUG, pushed home by the insertion (assembly.engage.push_home): its latch clicks over by
        itself. The tug -- IN PLACE, the grip kept, along the connector -X (tug_verify_in_place) -- says whether it
        caught: a latched plug holds the pull and is released and backed off; one that comes back out with it was
        not latched, and is a MISS -- still in the hand, partly out of the jack, backed out and laid down by the
        recovery like any other (and counted against assembly.engage.max_misses)."""
        t = tr()
        _at_the_mate()
        _post['state'] = 'engaged'               # the only rung of the ladder a latching plug has
        if not tv_on:
            _post['tug_res'] = None
            log.warning('LATCH NOT CHECKED: assembly.tug_verify is off -- the plug is pushed home, but whether its '
                        'latch caught is not known.')
            return True
        if not phase_gate('LATCH TUG',
                          'The plug is pushed home and still HELD. Next: pull %.1f N along the connector -X for '
                          '%.1f s without letting go -- a latched plug holds, an unlatched one comes back out.'
                          % (tv_force, tv_time)):
            _post['tug_res'] = 'skipped'
            log.warning('Latch tug skipped by the operator -- the assembly is UNVERIFIED.')
            return True
        try:
            # THE GRIP KEPT EITHER WAY: passed, the escape releases and backs straight off (the BNC's two-leg tug
            # retract, gripper -Z first, leaves the iiwa's reach from a pitched grasp); failed, the recovery backs
            # it out held.
            res = tug_verify_in_place(hold_after=True, keep_on_fail=True)
        finally:
            robot.arm.servo_stop()
            _save_clocking()
        t.data['tug'] = {'verified_held': 'verified', 'failed_held': 'failed'}.get(res, res)
        if res != 'verified_held' and res != 'failed_held':
            return camp.fail('the latch tug did not complete (%s)' % res, 'intervene', connector='socket',
                             gripped=None, arm='socket')
        if res == 'failed_held':
            t.data['latched'] = False
            t.data['misses'] = t.data.get('misses', 0) + 1
            _post['tug_res'] = 'failed'
            _miss['ref'] = robot.tool0()
            _miss['T_tool0_conn'] = robot.T_tool0_fingertip @ T_ftip_conn
            return camp.fail('the tug pulled the plug back out: its latch had NOT caught', connector='socket',
                             assembly=None, gripped=True, arm='socket')
        t.data['latched'] = True
        _post.update(tug_res='verified', grip_kept=True)
        t.set_cell(connector='socket', assembly='latched', gripped=True, arm='socket')
        return True

    # ---------------------------------------------------------------- LOCK (collar clocking)
    def step_lock():
        """The collar: taken axially and turned -- a wrist twist -- until it locks."""
        t = tr()
        if not phase_gate('COLLAR CLOCKING (lock)',
                          'The connector is SEATED. Next is the collar turn, which LOCKS it -- the '
                          'gripper will re-grasp the collar and rotate. Check the seat first.'):
            robot.arm.servo_stop()
            return camp.fail('the operator stopped before collar clocking', 'abort')
        try:
            ok = collar_clocking(_post['T_base_conn'], _post['cc_screw_deg'])
        finally:
            robot.arm.servo_stop()
            _save_clocking()
        t.data['locked'] = bool(ok)
        if ok:
            _post['state'] = _advance_state(_post['state'], 'seated')       # -> locked
            t.set_cell(assembly='locked', gripped=True, arm='socket')
            return True
        return camp.fail('collar clocking did not lock it', 'intervene', connector='socket', assembly='seated',
                         gripped=None, arm='socket')

    # ---------------------------------------------------------------- VERIFY (tug)
    def step_verify():
        """The tug, IN PLACE and BEFORE the escape: pull along the connector -X with the collar still held --
        a locked bayonet holds, an unlocked one backs out."""
        t = tr()
        # collar_clocking returns with the fingers still CLOSED on the locked collar, which is
        # already a grip on the assembly and already on the connector axis -- so the pull can
        # happen right here (the old order released, retracted, drove back and re-gripped).
        if _post.get('state') != 'locked':
            _post['tug_res'] = None
            return True
        if not phase_gate('TUG VERIFY',
                          'The collar is LOCKED and still HELD. Next: pull %.1f N along the '
                          'connector -X for %.1f s without letting go -- a locked bayonet '
                          'holds, an unlocked one backs out.' % (tv_force, tv_time)):
            _post['tug_res'] = 'skipped'
            log.warning('Tug verification skipped by the user -- the assembly is UNVERIFIED.')
            return True
        try:
            res = tug_verify_in_place(keep_on_fail=True)
        finally:
            robot.arm.servo_stop()
            _save_clocking()
        t.data['tug'] = 'failed' if res == 'failed_held' else res
        if res == 'terminated':
            return camp.fail('the tug verification was terminated', 'intervene', connector='socket', gripped=None,
                             arm='socket')
        if res == 'error':
            return camp.fail('the gripper did not let go after the tug', 'intervene', connector='socket', gripped=None,
                             arm='socket')
        if res == 'failed_held':
            # NOT LOCKED -- a verdict, not a stop: the escape and the disassembly still run, so the cable ends where
            # the next attempt picks it; the Verdict step then fails the attempt.
            _post['tug_res'] = 'failed'
            _post['not_assembled'] = 'the tug pulled the connector back out: it was NOT locked'
            t.set_cell(connector='socket', assembly='seated', gripped=True, arm='socket')
            return True
        _post['tug_res'] = res
        if res == 'verified':
            # tug_verify_in_place 'verified' releases and runs clocking_retract itself.
            t.set_cell(gripped=False, arm='clear')
        return True

    # ---------------------------------------------------------------- ESCAPE
    def _celebrate(ret_ok):
        """CELEBRATE, between a verified assembly and taking it apart, the arm clear: cosmetic, never fatal
        (every condition -- success, end state, how often -- is decided inside celebrate())."""
        try:
            celebrate(_post.get('state'), cycle, _post.get('tug_res'), ret_ok)
        except Exception as exc:                       # noqa: BLE001 -- cosmetic only
            log.warning('CELEBRATE raised (%s) -- ignored; the assembly stands.', exc)

    def step_escape():
        """Release and back off -- unless the tug already left the arm clear."""
        online_stop()                        # no more contact: pose6d.online off
        t = tr()
        if _post.get('tug_res') == 'verified' and not _post.get('grip_kept'):
            log.info('Escape not needed -- the tug verification already left the arm clear (released and '
                     'retracted).')
            _celebrate(True)
            return True
        if not (cc_on or ins_mode == 'rl'):
            # NO CLOCKING: release the mated connector and back straight off along it.
            d_out = float(a.get('release_retract_distance_m', 0.08))
            back = -(robot.tool0() @ (robot.T_tool0_fingertip @ T_ftip_conn))[:3, 0] * d_out
            phase('retract')
            if not robot.gripper.open('open gripper (release)'):
                return camp.fail('the gripper did not open to release the mated connector', 'intervene',
                                 gripped=None)
            ok = _guarded(robot, guard_shared,
                          lambda: robot.arm.move_l(translation_matrix(back) @ robot.tool0(),
                                                   label='retract (connector -X)'))
            t.set_cell(gripped=False, arm='clear' if ok else 'unknown')
            if ok and _post.get('tug_res') == 'verified':
                _celebrate(True)
            return ok or camp.fail('the retract after the release did not complete', 'intervene')
        if not phase_gate('ESCAPE',
                          'Clocking done. Next: RELEASE the gripper, then the two-leg retract '
                          '(the gripper backs off its own -Z, then away along the target -X).'):
            return camp.fail('the operator stopped before the escape', 'abort')
        # RELEASE FIRST: the fingers may still be CLOSED on the collar, and the retract's first leg is
        # written for OPEN fingers. Idempotent where it is already open.
        if not robot.gripper.open('release before escape'):
            return camp.fail('the gripper did not open before the escape -- not dragging the assembly with '
                             'clamped fingers', 'intervene', gripped=None)
        t.set_cell(gripped=False)
        try:
            # With the disassembly next, straight out along the socket -X only -- as the tug's own retract does: from
            # the axial collar grasp the gripper -Z leg is the SAME line, and both made 200 mm along the axis, past the
            # elbow's fold (2026-11-06 dry run: refused at 61% of the second leg).
            ok = clocking_retract(gripper_leg=not dis_on)
        finally:
            robot.arm.servo_stop()
        t.set_cell(arm='clear' if ok else 'unknown')
        if not ok:
            return camp.fail('the escape retract did not complete', 'intervene')
        if not _post.get('not_assembled'):
            _celebrate(True)
        return True

    # ---------------------------------------------------------------- DISASSEMBLE
    def step_disassemble():
        """Take it apart and put the cable back where the next trial picks it up: re-approach, unlock (one
        reverse turn frees the collar and the bayonet), pull it out, lay it down along the socket axis."""
        t = tr()
        state = _post.get('state')
        if state not in ('locked', 'seated'):
            return True                                # nothing in the socket to take apart (a failed tug)
        if not phase_gate('DISASSEMBLE',
                          'The assembly is complete and the arm is clear. Next it '
                          're-approaches, re-grips and takes it apart.'):
            return camp.fail('the operator stopped before the disassembly', 'abort')
        try:
            dis_ok, state = disassembly(state, _post['cc_screw_deg'], _post['T_base_conn'])
        finally:
            robot.arm.servo_stop()
            _save_clocking()
        _post['state'] = state
        t.data['disassembled'] = bool(dis_ok)
        if not dis_ok:
            return camp.fail('the disassembly did not finish (state %s): the part may still be held or still in '
                             'the socket' % state, 'intervene',
                             connector='socket' if state in ('locked', 'seated', 'engaged') else 'unknown',
                             assembly=state if state in ('locked', 'seated', 'engaged') else None,
                             gripped=None, arm='unknown')
        log.info('DISASSEMBLED -- the connector is %s.',
                 'out and placed on the ground' if dis_place_on else 'out and still in the fingers')
        t.set_cell(connector='bench' if dis_place_on else 'hand', assembly=None, gripped=not dis_place_on,
                   arm='clear')
        return True

    # ---------------------------------------------------------------- VERDICT
    def step_verdict():
        """Assembled, or not -- decided AFTER the disassembly has put the cable back, so a failed assembly (the tug
        pulling it out) still leaves the cell ready: the attempt fails here and the next one picks the placed cable."""
        why = _post.get('not_assembled')
        if not why:
            return True
        placed = tr().cell.get('connector') == 'bench'
        return camp.fail(why + (' -- disassembled and placed, ready to pick again' if placed else ''),
                         connector='bench' if placed else tr().cell.get('connector'), gripped=False if placed else None)

    # ---------------------------------------------------------------- END OF THE TRIAL
    def step_end():
        """Home + the end-of-trial image; the outcome named; with campaign.reset manual, the operator resets."""
        online_stop()                        # no more contact: pose6d.online off
        t = tr()
        state = _post.get('state')
        tug_res = _post.get('tug_res')
        if task.latch:
            if tug_res == 'verified':
                t.outcome = 'assembled, latch tug-verified'
            elif tug_res == 'failed':
                t.outcome = 'not latched (tug)'
            elif t.data.get('insert_status') in ('force', 'travel'):
                t.outcome = 'inserted, latch NOT checked'
        elif tug_res == 'failed':
            t.outcome = 'not locked (tug)'
        elif state in ('locked', 'removed') and t.data.get('locked'):
            t.outcome = 'assembled' + (', tug-verified' if tug_res == 'verified' else '')
        elif t.data.get('seated'):
            t.outcome = 'seated'
        elif t.cell.get('assembly') == 'engaged' or t.data.get('insert_status') in ('force', 'travel'):
            t.outcome = 'engaged'
        if attempts.status.name != 'SUCCESS':
            last = (t.failures[-1] if t.failures else {}).get('step', '?')
            t.outcome = f'failed at "{last}"' + (f' (had reached {t.outcome})' if t.outcome else '')
        if t.outcome:
            log.info('TRIAL %d: %s.', t.k, t.outcome.upper())
        phase('reset')
        ok = end_reset_with_snapshot()                       # always, even after a failed screw
        t.set_cell(arm='home' if ok else 'unknown', gripped=False if ok else t.cell['gripped'])
        if not ok:
            return camp.fail('the end-of-trial reset did not reach home', 'intervene')
        if reset_mode == 'manual' and t.k < n_trials:
            ans = ask(f'\n[RESET THE CELL] Trial {t.k}/{n_trials} done. Take the connector out of the socket'
                      + (' (press its latch)' if task.latch else '') + ' and '
                      'lay the cable in the pick area. Enter when the cell is ready (q to stop the run): ')
            if ans is None or ans.strip().lower() in ('q', 'quit'):
                camp.stop_run('stopped by the operator at the manual reset' if ans is not None
                              else 'no terminal to confirm the manual reset')
            t.set_cell(connector='bench', assembly=None, gripped=False)
        return True

    # ---------------------------------------------------------------- RECOVERIES
    def rec_place_held():
        """The cable in the hand, out of the socket: lay it down at the reset pose (where the coaxial grasp
        reaches it), release, rise, back to the pick pose."""
        online_stop()                        # no more contact: pose6d.online off
        t = tr()
        T_place = aligned_place_pose(dis_clear_m)
        T_ftip_place = T_place @ inverse(_upright(T_ftip_conn))
        up = translation_matrix([0.0, 0.0, float(dis_rise_m)])
        log.info('RECOVERY: laying the cable down at xyz %s mm (the reset pose), then back to the pick pose.',
                 np.round(T_place[:3, 3] * 1000.0, 0).tolist())
        phase('reconfigure')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(up @ T_ftip_place, 'recovery place (above)')):
            return camp.fail('could not reach above the reset pose', arm='unknown')
        phase('lift')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(T_ftip_place, 'recovery place (down)')):
            return camp.fail('could not lower to the reset pose', arm='unknown')
        if not robot.gripper.open('release (recovery)'):
            return camp.fail('the gripper did not open to release the cable', gripped=None)
        if hasattr(robot, 'release'):
            robot.release()
        t.set_cell(connector='bench', assembly=None, gripped=False)
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(up @ T_ftip_place, 'recovery place (rise)')):
            return camp.fail('could not rise after the release', arm='unknown')
        phase('reconfigure')
        if not robot.arm.move_j(q_pick, label='pick pose (after the recovery)'):
            return camp.fail('could not return to the pick pose', arm='unknown')
        t.set_cell(arm='pick')
        return True

    def rec_extract_and_place():
        """Gripped in (or against) the socket, not seated: back out along the SOCKET axis, CHECK it came
        out, then lay it down (place_after_failed_engage)."""
        online_stop()                        # no more contact: pose6d.online off
        t = tr()
        ref = _miss.get('ref')
        if ref is None:
            ref = robot.tool0()
        held_conn = _miss.get('T_tool0_conn')
        if held_conn is None:
            held_conn = robot.T_tool0_fingertip @ T_ftip_conn
        if not place_after_failed_engage(ref, held_conn, why=(t.failure or {}).get('why', '')):
            return camp.fail('the extract-and-place did not complete', connector='unknown', gripped=None,
                             arm='unknown')
        if hasattr(robot, 'release'):
            robot.release()
        t.set_cell(connector='bench', assembly=None, gripped=False, arm='pick')
        return True

    def rec_regrip_extract_and_place():
        """The fingers OPEN round an engaged connector (connector clocking opens them, even when it failed):
        close them on it again where they are, CHECK they hold it, then back out and lay it down."""
        online_stop()                        # no more contact: pose6d.online off
        t = tr()
        if not robot.gripper.close('re-grip (recovery)'):
            return camp.fail('the gripper did not close on the connector', gripped=None)
        if not verify_cable_held(robot, check, 're-grip (recovery)'):
            robot.gripper.open('release (re-grip missed)')
            return camp.fail('the re-grip did not take hold of the connector', gripped=False)
        t.set_cell(gripped=True)
        _miss['ref'] = robot.tool0()
        _miss['T_tool0_conn'] = _T_tool0_conn if _T_tool0_conn is not None else \
            robot.T_tool0_fingertip @ T_ftip_conn
        return rec_extract_and_place()

    def rec_reset_cable():
        """The grasp failed on every try with the cable on the bench: pick it SQUARE (straight down -- no
        dependence on its heading) and set it down aligned at the reset pose, where the coaxial grasp reaches
        it (reorient_recovery). Once per trial."""
        online_stop()                        # no more contact: pose6d.online off
        t = tr()
        t.data['cable_reset'] = True
        if not reorient_recovery():
            hand = _hand()
            if hand['gripped'] is False:
                # It did not get the cable (or let it go): the cell is as it was -- the next attempt looks again.
                log.warning('RECOVERY: the cable reset did not complete, but the hand is empty -- the next attempt '
                            'scans for the cable afresh.')
                t.set_cell(arm='unknown', **hand)
                return rec_home()
            return camp.fail('the square pick and aligned place did not complete', arm='unknown', **hand)
        t.set_cell(connector='bench', gripped=False, arm='unknown')
        return True

    def rec_home():
        """Nothing in the hand: open, home; the next attempt measures everything again."""
        online_stop()                        # no more contact: pose6d.online off
        t = tr()
        phase('reset')
        if not reset.reset_robot(robot, cfg, 'recovery: home'):
            return camp.fail('could not go home', arm='unknown')
        t.set_cell(arm='home', gripped=False)
        return True

    def _is(t, **want):
        return all(t.cell.get(k) == v for k, v in want.items())

    # ---------------------------------------------------------------- THE TREE
    S = lambda name, fn, **kw: bt.Step(name, fn, camp, **kw)        # noqa: E731
    head = (S('Localize the socket', step_localize, note=tgt_source),
            S('Pick the cable', step_pick),
            S('Connector pose in the hand', step_in_hand, note='from the grasp'),
            S('Approach: stand-off', step_approach),
            S('Insert', step_insert, note=ins_mode + (', pushed home' if ph_on else '')))
    if task.latch:
        # A LATCHING PLUG (the RJ45): no bayonet to clock, nothing to take apart -- the tug checks the latch.
        tail = (S('Verify: latch (tug)', step_latch, enabled=tv_on),
                S('Release and escape', step_escape))
    else:
        tail = (S('Seat: connector clocking', step_seat, enabled=bool(cc_on or ins_mode == 'rl'),
                  note='by the policy' if ins_mode == 'rl' else ''),
                S('Lock: collar clocking', step_lock, enabled=bool((cc_on or ins_mode == 'rl') and cl_on)),
                S('Verify: tug', step_verify, enabled=bool((cc_on or ins_mode == 'rl') and cl_on and tv_on)),
                S('Release and escape', step_escape),
                S('Disassemble: put the cable back', step_disassemble,
                  enabled=bool(dis_on and (cc_on or ins_mode == 'rl'))))
    assemble = bt.sequence('Assemble', *head, *tail, S('Verdict: assembled and verified', step_verdict))
    recover = bt.selector(
        'Recover',
        camp.operator_branch(),
        camp.recovery('cable in hand, clear of the socket?',
                      lambda t: _is(t, connector='hand', gripped=True) and t.cell['arm'] != 'socket',
                      'Place it at the reset pose', rec_place_held),
        camp.recovery('gripped in the socket, not seated?',
                      lambda t: _is(t, connector='socket', gripped=True) and t.cell['assembly'] in (None, 'engaged')
                      and t.data.get('misses', 0) <= max_engage_misses,
                      'Back out (checked) and place', rec_extract_and_place,
                      note=f'≤{max_engage_misses} misses a trial'),
        camp.recovery('fingers open round an engaged connector?',
                      lambda t: _is(t, connector='socket', assembly='engaged', gripped=False, arm='socket'),
                      'Re-grip, back out and place', rec_regrip_extract_and_place),
        camp.recovery('grasp failed, cable on the bench?',
                      lambda t: _is(t, connector='bench', gripped=False) and t.data.get('reset_cable_wanted')
                      and not t.data.get('cable_reset') and bool((a.get('reorient_recovery') or {}).get(
                          'enabled', True)),
                      'Reset the cable: square pick, aligned place', rec_reset_cable),
        camp.recovery('nothing in the hand?',
                      lambda t: t.cell['gripped'] is False and t.cell['connector'] in ('bench', 'unknown'),
                      'Go home', rec_home),
        camp.operator_branch(fallback=True))
    attempts = bt.AttemptLoop(f'Attempts (≤{camp.max_attempts})', assemble, recover, camp, camp.max_attempts)
    trial_node = bt.TrialSequence('Trial', S('Prepare: empty hand, home', step_prepare), attempts,
                                  S('End: home + snapshot', step_end), camp)
    root = bt.sequence(task.title,
                       S('Start: home + gripper warm-up', step_start),
                       bt.Trials(f'Trials ×{n_trials}', trial_node, camp))
    log.info('MODE %s%s -- %d trial%s, up to %d attempts each; recovery %s; %s reset between trials.',
             mode.upper(), (' (' + ', '.join(tags) + ')') if tags else '', n_trials, '' if n_trials == 1 else 's',
             camp.max_attempts, 'ON' if camp.recover_on else 'OFF (every failure asks)',
             'automatic (disassembly)' if reset_mode == 'auto' else 'MANUAL')
    try:
        ok = camp.run(root)
    finally:
        board.close()
    return ok and not camp.abort and all(t.success for t in camp.trials) and len(camp.trials) == n_trials


def add_args(parser):
    """--trials [N]: run the task N times (bare: 100) -- campaign.trials."""
    parser.add_argument('--trials', nargs='?', type=int, const=100, default=None, metavar='N',
                        help='run the task as a campaign of N trials (bare --trials: 100), recovering what it '
                             'safely can and asking for the rest; see campaign: in configs/bnc_assembly.yaml')

    def apply(args, cfg):
        if args.trials is not None:
            if args.trials < 1:
                parser.error('--trials must be at least 1')
            cfg.set_path('campaign.trials', int(args.trials))
    return apply


def main():
    run_app('BNC assembly: pick + estimator_eval-tuned manifold estimation',
            'bnc_assembly', build_and_run, needs_camera=True, precheck=precheck_and_preload, add_args=add_args)


if __name__ == '__main__':
    main()
