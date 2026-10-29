"""Arm bring-up -- the first session with an arm, before any calibration exists.

Written for the iiwa 14's FRI link, and works on any cell's arm. It talks to the ARM only (no
camera, no gripper, no Robot), so `calibrated: false` does not stop it.

    python -m cable_assembly.apps.arm_bringup                       # the iiwa bench (default cell)
    python -m cable_assembly.apps.arm_bringup --jog                 # + two small test moves
    python -m cable_assembly.apps.arm_bringup --cell ur10e_bench    # any other bench
    python -m cable_assembly.apps.arm_bringup --dry-run --jog       # rehearse on the simulator

WHAT EACH STEP CHECKS:

  1. CONNECT. On the iiwa: the FRI app on the smartPAD is running in POSITION client command
     mode, sends to this PC, and the session reaches COMMANDING_ACTIVE. The log says what it is
     still waiting for.
  2. THE MODEL. The joints, and tool0 as this repo's kinematics computes it from them. Compare
     it with the FLANGE pose on the smartPAD (base frame): a mismatch along the tool axis means
     the flange length in kinematics/iiwa14.py (D_WF, media flange assumed) is wrong.
  3. THE WRENCH SIGN. After a tare, push the tool by hand: a push DOWN must read fz < 0 and a push
     toward +x fx > 0 -- the force ON the tool, base axes, as every backend reports it. If not,
     stop: the force guards and the admittance would act backwards.
  4. --jog: the tool-roll joint turns `jog_deg` and back, then tool0 lifts `jog_mm` straight up
     and back, at `speed_scale` of the caps, under a force guard. Watch that the arm does what the
     log says before trusting anything faster.
"""

import sys
import time

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..transforms import fmt_pose, pose_error, translation_matrix

log = urlog.get('bringup')


def main():
    parser = urconfig.arg_parser('Arm bring-up: connect, check the model and the wrench sign',
                                 'arm_bringup')
    parser.add_argument('--jog', action='store_true',
                        help='also make two small, slow test moves (asks first)')
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    urlog.setup()
    b = cfg.section('bringup')

    from ..robot import make_arm
    arm = make_arm(cfg)
    ok = False
    try:
        report(arm)
        live_wrench(arm, float(b.get('wrench_seconds', 20.0)))
        ok = jog(arm, b, args.yes) if args.jog else True
    except KeyboardInterrupt:
        log.warning('Interrupted.')
    finally:
        arm.disconnect()
    sys.exit(0 if ok else 1)


def report(arm):
    q = np.degrees(arm.q())
    log.info('Arm: %s via %s.', arm.model.NAME, type(arm).__name__)
    for name, v, (lo, hi) in zip(arm.joint_names, q, np.degrees(arm.joint_limits)):
        log.info('  %-14s %8.2f deg   (limits %7.1f .. %6.1f)', name, v, lo, hi)
    log.info('tool0 from the model: %s', fmt_pose(arm.tcp_pose()))
    log.info('Compare with the FLANGE pose on the pendant. A difference along the tool axis is '
             'the flange length in the model; anything else is a joint-zero or a frame problem.')


def live_wrench(arm, seconds):
    if seconds <= 0:
        return
    # Before the tare, hands off: on the iiwa this is ~12 N (the Robotiq + D415) when the Sunrise
    # app has no load data for the tool, ~0 when it does. After the tare it reads ~0 either way.
    time.sleep(0.5)
    w = arm.wrench()
    log.info('Untared, hands off: |f| = %.1f N  f = [%.1f %.1f %.1f] N', np.linalg.norm(w[:3]),
             *w[:3])
    arm.zero_ft()
    log.info('Tared. For %.0f s: push the tool DOWN (fz must go NEGATIVE) and toward +x (fx '
             'POSITIVE). Ctrl-C to skip.', seconds)
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            w = arm.wrench()
            log.info('  f = [%6.1f %6.1f %6.1f] N   tau = [%5.2f %5.2f %5.2f] Nm  (base, about '
                     'the flange)', *w)
            time.sleep(0.2)
    except KeyboardInterrupt:
        log.info('Wrench check skipped.')


def jog(arm, b, assume_yes):
    deg, mm = float(b.get('jog_deg', 5.0)), float(b.get('jog_mm', 20.0))
    scale, f_max = float(b.get('speed_scale', 0.1)), float(b.get('guard_force_n', 15.0))
    if not assume_yes:
        answer = input(f'Turn {arm.joint_names[arm.tool_roll_joint]} by {deg:.0f} deg and back, '
                       f'then lift tool0 {mm:.0f} mm and back, at {scale:.0%} speed. '
                       'Hand on the e-stop? [yes/N] ')
        if answer.strip().lower() != 'yes':
            log.info('No moves made.')
            return True

    arm.zero_ft()
    arm.add_guard(lambda: arm.force() > f_max)
    try:
        q0 = np.asarray(arm.q())
        q1 = q0.copy()
        q1[arm.tool_roll_joint] += np.radians(deg)
        for q, label in ((q1, f'roll +{deg:.0f} deg'), (q0, 'roll back')):
            if not arm.move_j(q, label=label, caps=scale):
                return False
            log.info('  [%s] joint error %.3f deg', label,
                     np.degrees(np.max(np.abs(np.asarray(arm.q()) - q))))
        T0 = arm.tcp_pose()
        for T, label in ((translation_matrix([0.0, 0.0, mm / 1000.0]) @ T0, f'lift {mm:.0f} mm'),
                         (T0, 'lower back')):
            if not arm.move_l(T, label=label, caps=scale):
                return False
            lin, ang = pose_error(arm.tcp_pose(), T)
            log.info('  [%s] tool0 error %.2f mm, %.3f deg', label, 1000.0 * lin, np.degrees(ang))
    finally:
        arm.clear_guards()
    log.info('Test moves done.')
    return True


if __name__ == '__main__':
    main()
