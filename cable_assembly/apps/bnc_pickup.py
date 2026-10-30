"""BNC PICKUP -- the BNC demo's pick on its own: scan, grasp, slip-checked lift, then set the
cable back down where it came from, `pickup_test.cycles` times.

    python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench
    python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench --set pickup_test.cycles=5
    python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench --dry-run

THE SAME PICK, BY CONSTRUCTION. It reads configs/bnc_assembly.yaml and calls what bnc_assembly
calls -- build_scanner, cable_pick_assemble._pick, GraspController.lift_verified, the same
GraspCheck / GraspRecovery and retry offsets -- so a pick that works here works there, and a
setting tuned here is tuned there. Only what happens AFTER the lift differs.

IT NEEDS NO SOCKET. The pick never looks at the target, so this runs on a cell whose
`calibrated:` lists hand_eye, fingertip, ground_plane and poses -- a bench whose socket has not
been recorded yet. Two things bnc_assembly does are therefore absent: the marker sweep that
finds the socket, and the reorient recovery for an unreachable coaxial grasp (it sets the cable
down along the SOCKET's heading). An unreachable grasp ends the run with a message instead.

PUT BACK. After the lift the fingers go back down to the height they gripped at plus
`release_clearance_mm`, open, rise straight up by `approach_distance_mm`, and the arm returns
to the pick pose. The cable lands close to where it lay, so the next cycle re-detects the same
cable without asking which one again.
"""

import time

import numpy as np

from .. import log as urlog
from ..log import StepRunner
from ..robot import ForceGuard
from ..skills import reset
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, retry_offset_x)
from ..transforms import translation_matrix
from ._cable import build_scanner, make_confirm
from ._common import prompts_off
from ._runner import run_app
from .cable_pick_assemble import _guarded, _pick

log = urlog.get('bnc-pickup')

# What the pick relies on (robot.CALIBRATION_ITEMS): no socket target.
NEEDS = ('hand_eye', 'fingertip', 'ground_plane', 'poses')


def pick_and_lift(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder, guard,
                  phase, q_pick, watcher=None):
    """bnc_assembly's attempt loop, less the reorient. 'ok' | 'unreachable' | the last failure
    ('missed', 'empty', 'slipped') once the retries are spent | 'abort'."""
    attempt = 0
    runner = StepRunner(log, confirm=confirm is not None)
    while True:
        phase('scan')
        result = _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                       offset_x_m=retry_offset_x(attempt, check.retry_perturb_x_m))
        if result == 'ok':
            status = {}

            def do_lift(_s=status):
                _s['r'] = grasp.lift_verified(
                    robot, geom, check, 'lift',
                    position_guard=lambda mv: _guarded(robot, guard, mv))
                return _s['r'] == 'ok'

            if runner.run([('lift (slip-checked)', do_lift)]):
                if watcher is not None:                 # reports only; the fingers decided
                    watcher.after_lift(getattr(scanner, '_selection', None))
                return 'ok'
            result = status.get('r')
            if result != 'slipped':
                return 'abort'
        if result == 'abort':
            return 'abort'
        if result == 'unreachable':
            log.error('The grasp has no path to this cable as it lies. bnc_assembly would pick it '
                      'square and set it down along the socket heading; with no socket here, '
                      'turn the cable by hand and run again.')
            return 'unreachable'
        if attempt >= check.max_retries:
            log.error('Grasp failed on all %d attempts (%s).', check.max_retries + 1, result)
            return result
        attempt += 1
        log.warning('Grasp %s -- recovering (attempt %d/%d).', result, attempt + 1,
                    check.max_retries + 1)
        if result == 'slipped' and hasattr(scanner, 'reselect'):
            phase('scan')
            T_up = translation_matrix([0.0, 0.0, check.slip_raise_m]) @ robot.tool0()
            if not (robot.gripper.open('drop')
                    and _guarded(robot, guard,
                                 lambda: robot.arm.move_l(T_up, label='slip recovery (up)'))):
                return 'abort'
            scanner.reselect()
        else:
            phase('reset')
            if not (robot.gripper.open('drop')
                    and robot.arm.move_j(q_pick, label='pick pose')):
                return 'abort'


def put_back(robot, geom, guard, clearance_m, phase):
    """Lower the held cable to where it was gripped (+ `clearance_m`), open, rise straight up."""
    T_release = translation_matrix([0.0, 0.0, clearance_m]) @ geom.T_base_grasp
    T_clear = translation_matrix([0.0, 0.0, geom.approach_distance]) @ T_release
    phase('lift')
    if not _guarded(robot, guard, lambda: robot.move_fingertip(T_release, 'put back (down)')):
        log.error('The put-back descent stopped -- the cable is STILL HELD. Free it by hand.')
        return False
    if not robot.gripper.open('release (put back)'):
        return False
    phase('retract')
    return _guarded(robot, guard, lambda: robot.move_fingertip(T_clear, 'put back (clear)'))


def build_and_run(cfg, robot, camera, args):
    t = cfg.section('pickup_test')
    cycles = max(1, int(t.get('cycles', 1)))
    clearance = float(t.get('release_clearance_mm', 5.0)) / 1000.0
    settle = float(t.get('settle_s', 3.0))
    if cfg.get('pick_joints_deg') is None:
        log.error('pick_joints_deg is not set in the cell: the pose the scan looks from.')
        return False
    q_pick = list(np.radians(np.asarray(cfg.get('pick_joints_deg'), dtype=float)))

    scanner, _detector, _estimator = build_scanner(cfg, robot, camera)
    # What the depth says about each pick (watch: in the config): reports, decides nothing.
    watcher = None
    if (cfg.get('watch') or {}).get('enabled') and not robot.arm.dry_run and hasattr(scanner, 'listeners'):
        from ..skills.watch import PickWatcher
        watcher = PickWatcher(cfg, camera, scanner.detector, getattr(robot, 'live_frames', None))
        scanner.listeners.append(watcher.on_scan)
    geom, check = GraspGeometry(cfg), GraspCheck(cfg)
    recovery, grasp = GraspRecovery(cfg), GraspController(cfg)
    recorder = GraspImageRecorder(cfg)
    confirm = make_confirm(cfg)
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    scales = cfg.get_path('speed.phase_scale', {}) or {}
    gates_on = (bool(cfg.get_path('assembly.gate_between_behaviors', True))
                and not prompts_off(cfg))

    def phase(name):
        robot.arm.set_speed_scale(float(scales.get(name, 1.0)), name)

    def gate(name, ahead):
        if robot.arm.dry_run or not gates_on:
            return True
        try:
            ans = input(f'\n[{name}] {ahead}\n    Enter to continue (q to abort): ')
        except EOFError:
            return True
        return ans.strip().lower() not in ('q', 'quit', 'n', 'no')

    phase('reset')
    if not reset.reset_robot(robot, cfg, 'start reset'):
        return False
    grasp.verify_collision_model(robot)
    if not robot.gripper.warmup():
        return False

    results = []
    for cycle in range(1, cycles + 1):
        log.info('=' * 78)
        log.info('PICKUP %d/%d', cycle, cycles)
        log.info('=' * 78)
        if not gate('PICK THE CABLE', 'The arm moves to the pick pose %s deg, localizes the '
                    'cable and grasps it.' % list(np.round(np.degrees(q_pick), 1))):
            break
        phase('reconfigure')
        if not robot.arm.move_j(q_pick, label='pick pose'):
            log.error('Could not reach pick_joints_deg.')
            return False
        t0 = time.monotonic()
        r = pick_and_lift(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                          guard, phase, q_pick, watcher)
        results.append((r, time.monotonic() - t0))
        if r != 'ok':
            break
        log.info('PICKED and lifted in %.1f s. Putting it back.', results[-1][1])
        if not (gate('PUT IT BACK', 'The cable goes back down where it was and is released.')
                and put_back(robot, geom, guard, clearance, phase)):
            return False
        phase('reset')
        if not robot.arm.move_j(q_pick, label='pick pose'):
            return False
        if cycle < cycles and settle > 0.0 and not robot.arm.dry_run:
            log.info('Waiting %.1f s for the cable to settle before the next scan.', settle)
            time.sleep(settle)

    n_ok = sum(1 for r, _ in results if r == 'ok')
    log.info('=' * 78)
    log.info('PICKUP TEST: %d/%d picked (%s).', n_ok, cycles,
             ', '.join(f'{r} {s:.0f} s' for r, s in results) or 'none attempted')
    phase('reset')
    reset.reset_robot(robot, cfg, 'end reset')
    return n_ok == cycles


def main():
    run_app('BNC pickup test: the BNC pick, then the cable put back', 'bnc_assembly',
            build_and_run, needs_camera=True, needs=NEEDS)


if __name__ == '__main__':
    main()
