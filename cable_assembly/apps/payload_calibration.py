"""Payload calibration -- the tool's mass and centre of mass (robot.payload), measured by turning
the wrist through poses and reading the joint torques the tool's weight causes.

    python -m cable_assembly.apps.payload_calibration                  # the iiwa bench
    python -m cable_assembly.apps.payload_calibration --dry-run        # rehearse
    python -m cable_assembly.apps.payload_calibration --set payload_calibration.refit_from=<run>/samples.csv
                                                                       # solve again, arm untouched

WHY. When the Sunrise app carries no load data for the tool, KUKA's external torques carry its
weight: the wrench reads ~12 N with nothing touching it. A base-frame tare cancels that FORCE at
every orientation, but not the weight's MOMENT about the flange, which turns with the tool. With
the payload known, robot.fri.compensate_payload subtracts both, at every pose.

THE MOTION. Only the wrist moves: A5, A6 and A7 step through offsets about the start pose (where
the arm is, or `start_joints_deg`), each move slow and force-guarded, so the tool tilts several
ways while the shoulder and elbow stay put. Every pose AND every path between them is checked
before anything moves: inside the joint limits, and the tool -- flange, gripper, fingertips,
camera -- at least `clearance_mm` above the bench. The poses are visited twice, the second time
in reverse, so each is read arriving from both sides. Then the arm returns to the start.

WATCH the gripper and camera cables (A7 turns up to 180 deg), and the tool near the forearm at the
larger A6 bends; the guard stops a move on contact.

THE SOLVE is two steps (robot/payload.py says why): the MASS from the torque level on A2-A4 --
or `mass_kg`, weighed or summed from the parts -- then the CoG with that mass given and a constant
offset on every joint for KUKA's own model error. The torques fix m (wrist offset + CoG along the
tool axis), so that component moves ~1.3 mm per 10 g of mass error; the others hardly at all.

THE RESULT goes to data/experiments/payload_calibration_<stamp>/: result.yaml (the block for the
cell) and samples.csv. EITHER paste it into the cell with compensate_payload: true, OR enter the
same mass and CoG as the tool's load data in the Sunrise app -- NOT both, or the weight is
subtracted twice.
"""

import csv
import os
import sys
import time

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..robot import payload as pl
from ..transforms import from_cfg
from ._common import experiment_dir

log = urlog.get('payload')


def main():
    parser = urconfig.arg_parser("Payload calibration: the tool's mass and CoG from joint torques",
                                 'payload_calibration')
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    urlog.setup()

    refit_from = (cfg.section('payload_calibration') or {}).get('refit_from')
    if refit_from:
        sys.exit(0 if refit(cfg, refit_from) else 1)

    from ..robot import make_arm
    arm = make_arm(cfg)
    ok = False
    try:
        ok = bool(calibrate(cfg, arm, _torque_reader(arm), args.yes))
    except KeyboardInterrupt:
        log.warning('Interrupted.')
    finally:
        arm.disconnect()
    sys.exit(0 if ok else 1)


# A dry run's tool: NOT the configured payload, so the rehearsal shows the fit finding it. Its
# model error is on EVERY joint, A2-A4 included, where the fit assumes none -- as on the arm.
DRY_RUN_PAYLOAD = (1.25, np.array([-0.018, 0.004, 0.072]))
DRY_RUN_BIAS_NM = np.array([0.4, 0.15, -0.1, 0.15, -0.12, 0.08, 0.04])


def _torque_reader(arm):
    """() -> (q, tau_ext). The iiwa's are KUKA's; a dry run's are synthesised."""
    if not arm.dry_run:
        if not hasattr(arm, 'external_torques'):
            raise SystemExit(f'{type(arm).__name__} reports no joint torques to fit a payload to '
                             '(a UR measures its own: PolyScope\'s payload estimation).')
        return arm.external_torques
    rng = np.random.default_rng(0)
    bias = DRY_RUN_BIAS_NM[:arm.dof]
    log.warning('DRY RUN: torques synthesised for a %.2f kg tool with its CoG at %s mm, plus a '
                'per-joint bias and 0.03 Nm of noise.', DRY_RUN_PAYLOAD[0],
                np.round(DRY_RUN_PAYLOAD[1] * 1000.0, 1).tolist())

    def read():
        q = np.asarray(arm.q(), dtype=float)
        return q, (pl.gravity_torques(arm.model, q, *DRY_RUN_PAYLOAD) + bias
                   + rng.normal(0.0, 0.03, arm.dof))
    return read


def tool_points(cfg):
    """(N, 3) points on the tool in tool0 (m) the clearance check keeps above the bench: the
    flange rim, a box around the gripper and its open fingers, one around the camera."""
    tip = from_cfg(cfg.section('fingertip_grasp'))[:3, 3] if cfg.section('fingertip_grasp') \
        else np.array([0.0, 0.0, 0.2])
    pts = [[x, y, 0.0] for x in (-0.045, 0.045) for y in (-0.045, 0.045)]
    for z in (0.5 * tip[2], tip[2] + 0.01):
        pts += [[x, y, z] for x in (-0.04, 0.04) for y in (-0.075, 0.075)]
    if cfg.section('hand_eye'):
        cam = from_cfg(cfg.section('hand_eye'))[:3, 3]
        pts += [cam + [dx, dy, dz] for dx in (-0.05, 0.05) for dy in (-0.05, 0.05)
                for dz in (-0.05, 0.05)]
    return np.array(pts, dtype=float)


def plan(arm, q0, h):
    """[(label, q)] about `q0`: A6 x A7 offsets with A5 at its start, then A5 both ways at the
    first two A7s. A7 takes the first `a7_count` offsets that stay inside the limits."""
    w5, w6, w7 = arm.dof - 3, arm.dof - 2, arm.dof - 1        # the wrist: the last three joints

    def at(d5, d6, d7):
        q = np.array(q0, dtype=float)
        q[[w5, w6, w7]] += np.radians([d5, d6, d7])
        return q

    a7 = [float(d) for d in h.get('a7_offsets_deg', [0, 90, -90, 180, -180])
          if arm.joints_ok(at(0, 0, float(d)))][:int(h.get('a7_count', 3))]
    a6 = [float(d) for d in h.get('a6_offsets_deg', [0, 30, -30])]
    out = []
    for k, d7 in enumerate(a7):
        for d6 in (a6 if k % 2 == 0 else a6[::-1]):          # serpentine: short wrist moves
            out.append((f'A6 {d6:+.0f}, A7 {d7:+.0f}', at(0, d6, d7)))
    for d5 in h.get('a5_offsets_deg', [40, -40]):
        for d7 in a7[:2]:
            out.append((f'A5 {float(d5):+.0f}, A7 {d7:+.0f}', at(float(d5), 0, d7)))
    return out


def clearance(arm, pts, z_bench, qa, qb, steps=25):
    """The lowest tool point's height above the bench along the joint-space line qa -> qb (m)."""
    low = np.inf
    for s in np.linspace(0.0, 1.0, steps):
        T = arm.fk(qa + s * (qb - qa))
        low = min(low, float(np.min((T[:3, :3] @ pts.T)[2] + T[2, 3])) - z_bench)
    return low


def calibrate(cfg, arm, read, assume_yes=False):
    """The poses, the readings and the fit. Returns payload.fit's dict, or False."""
    h = cfg.section('payload_calibration') or {}
    scale = float(h.get('speed_scale', 0.3))
    f_max = float(h.get('guard_force_n', 20.0))
    min_clear = float(h.get('clearance_mm', 60.0)) / 1000.0
    z_bench = float(cfg.get_path('ground_plane.z_m', 0.0) or 0.0)
    pts = tool_points(cfg)

    q_start = h.get('start_joints_deg')
    if q_start is None and arm.dry_run and cfg.get('pick_joints_deg') is not None:
        q_start = cfg.get('pick_joints_deg')                  # a dry run starts at the cell's view
    if q_start is not None:
        if not arm.dry_run and not _ask(assume_yes, f'Move to start_joints_deg {list(q_start)}?'):
            return False
        if not arm.move_j(np.radians(np.asarray(q_start, dtype=float)), label='start',
                          caps=scale):
            return False
    q0 = np.asarray(arm.q(), dtype=float)
    if not arm.joints_ok(q0):
        log.error('The arm starts outside its joint limits (deg %s). Jog it back inside on the '
                  'pendant (the wrist un-flipped).', np.round(np.degrees(q0), 1).tolist())
        return False

    poses = []
    for label, q in plan(arm, q0, h):
        if not arm.joints_ok(q):
            log.info('  skipped %-20s outside the joint limits', label)
            continue
        c = clearance(arm, pts, z_bench, q, q, steps=1)
        if c < min_clear:
            log.info('  skipped %-20s the tool would be %.0f mm above the bench', label, c * 1000)
            continue
        poses.append((label, q))
    passes = int(h.get('passes', 2))
    seq = poses + (poses[-2::-1] if passes >= 2 else [])
    path = [q0] + [q for _, q in seq] + [q0]
    lows = [clearance(arm, pts, z_bench, a, b) for a, b in zip(path[:-1], path[1:])]
    if len(poses) < 6:
        log.error('Only %d usable poses (want >= 6). Start higher above the bench, or with the '
                  'wrist nearer the middle of its ranges.', len(poses))
        return False
    if min(lows) < min_clear:
        k = int(np.argmin(lows))
        log.error('The path into move %d dips to %.0f mm above the bench (want >= %.0f). Start '
                  'higher, or lower a6_offsets_deg / a5_offsets_deg.', k + 1, min(lows) * 1000,
                  min_clear * 1000)
        return False
    swing = np.degrees(np.ptp([q for _, q in poses], axis=0))
    if not _ask(assume_yes, f'The wrist will visit {len(poses)} poses{" twice" if passes >= 2 else ""}'
                f' ({len(seq)} moves; A5/A6/A7 span {swing[-3]:.0f}/{swing[-2]:.0f}/'
                f'{swing[-1]:.0f} deg, the tool >= {min(lows) * 1000:.0f} mm above the bench) at '
                f'{scale:.0%} speed, guarded at {f_max:.0f} N, then return here. Cables slack, '
                'hand on the e-stop?'):
        return False

    out = experiment_dir(cfg, 'payload_calibration')
    settle, average = float(h.get('settle_s', 2.0)), float(h.get('average_s', 1.0))
    samples, rows = [], []
    arm.zero_ft()
    arm.add_guard(lambda: arm.force() > f_max)
    try:
        for k, (label, q) in enumerate(seq, start=1):
            if not arm.move_j(q, label=label, caps=scale):
                return False
            q_m, tau, sd, moved = _measure(arm, read, settle, average)
            if moved > np.radians(0.05):
                log.error('The arm moved %.2f deg while reading -- not still; stopping.',
                          np.degrees(moved))
                return False
            samples.append((q_m, tau))
            rows.append((k, label, q_m, tau, sd))
            log.info('  %2d/%d %-20s tau_ext %s Nm', k, len(seq), label,
                     ' '.join(f'{t:+6.2f}' for t in tau))
        if not arm.move_j(q0, label='back to start', caps=scale):      # the path checked above
            return False
    finally:
        arm.clear_guards()

    return _solve(cfg, out, arm.model, rows)


def refit(cfg, path):
    """Solve again from an earlier run's samples.csv -- other settings, the arm untouched."""
    from ..robot import kinematics
    model = kinematics.get(cfg.get_path('robot.type', 'iiwa14'))
    rows = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            n = sum(1 for k in row if k.startswith('tau') and k.endswith('_nm'))
            rows.append((int(row['reading']), row['label'],
                         np.radians([float(row[f'q{i}_deg']) for i in range(1, n + 1)]),
                         np.array([float(row[f'tau{i}_nm']) for i in range(1, n + 1)]),
                         np.array([float(row[f'tau{i}_sd']) for i in range(1, n + 1)])))
    log.info('Refit of %d readings from %s.', len(rows), path)
    return _solve(cfg, experiment_dir(cfg, 'payload_calibration'), model, rows)


def _solve(cfg, out, model, rows):
    """The two steps, the report and the files. Returns step 2's fit dict, or False."""
    h = cfg.section('payload_calibration') or {}
    samples = [(q, tau) for _, _, q, tau, _ in rows]
    dof = len(samples[0][1])
    level = [int(j) - 1 for j in h.get('level_joints', [2, 3, 4])]
    try:
        mass = h.get('mass_kg')
        step1 = None
        if mass is None:
            step1 = pl.fit(model, samples, bias_joints=[j for j in range(dof) if j not in level])
            mass = step1['mass_kg']
        r = pl.fit(model, samples, bias_joints=range(dof), mass_kg=float(mass))
        dz = (pl.fit(model, samples, bias_joints=range(dof), mass_kg=float(mass) + 0.01)['cog_m']
              - r['cog_m'])
    except ValueError as exc:
        log.error('Cannot fit: %s', exc)
        return False
    _report(cfg, out, model, rows, step1, r, dz)
    return r


def _measure(arm, read, settle_s, average_s):
    """(mean q, mean tau_ext, tau std, joint travel while reading) with the arm still."""
    if arm.dry_run:
        settle_s = average_s = 0.0
    time.sleep(settle_s)
    qs, ts = [], []
    t_end = time.monotonic() + average_s
    while time.monotonic() < t_end or len(qs) < 5:
        q, t = read()
        qs.append(q)
        ts.append(t)
        if not arm.dry_run:
            time.sleep(0.01)
    qs, ts = np.array(qs), np.array(ts)
    return qs.mean(axis=0), ts.mean(axis=0), ts.std(axis=0), float(np.max(np.ptp(qs, axis=0)))


def _ask(assume_yes, question):
    if assume_yes:
        return True
    try:
        answer = input(question + ' [yes/N] ')
    except EOFError:
        answer = ''
    if answer.strip().lower() != 'yes':
        log.info('Stopped; nothing moved.')
        return False
    return True


def _spread(model, samples, payload=None, offsets=None):
    """The wrench's spread over the poses, about its mean -- what a tare cannot remove: (force N,
    moment Nm), each the largest per-axis standard deviation."""
    from ..robot.backends.iiwa_fri import wrench_from_torques
    off = 0.0 if offsets is None else offsets
    W = np.array([wrench_from_torques(model, q, t - off, payload=payload) for q, t in samples])
    return float(np.max(np.std(W[:, :3], axis=0))), float(np.max(np.std(W[:, 3:], axis=0)))


def _report(cfg, out, model, rows, step1, r, dz):
    samples = [(q, tau) for _, _, q, tau, _ in rows]
    m, c = r['mass_kg'], r['cog_m'] * 1000.0
    sc = r['sigma_cog_m'] * 1000.0
    log.info('=' * 78)
    log.info('PAYLOAD (tool0 = the flange frame), %d readings:', len(samples))
    if step1 is None:
        log.info('  1. mass %.3f kg -- GIVEN (mass_kg)', m)
    else:
        log.info('  1. mass %.3f kg (+-%.3f) from the torque level on A2-A4 -- the level is only '
                 'as right as KUKA\'s model there; a weighed mass_kg beats it', m,
                 step1['sigma_mass_kg'])
    log.info('  2. CoG [%.1f, %.1f, %.1f] mm (+-%.1f, %.1f, %.1f), the mass given and an offset '
             'on every joint', *c, *sc)
    log.info('     per 10 g more mass the CoG moves [%+.1f, %+.1f, %+.1f] mm', *(dz * 1000.0))
    log.info('     joint offsets (KUKA model error) %s Nm',
             ' '.join(f'{b:+.2f}' for b in r['bias_nm']))
    log.info('     residual %.3f Nm RMS (per reading: %s); gravity turned %.0f deg in the tool '
             'frame', r['rms_nm'], ' '.join(f'{v:.2f}' for v in r['residual_nm']),
             r['tilt_span_deg'])
    rows_ = [('tared only', _spread(model, samples)),
             ('+ this payload', _spread(model, samples, (m, r['cog_m']))),
             ('+ payload + offsets', _spread(model, samples, (m, r['cog_m']), r['bias_nm']))]
    log.info('  the wrench across these poses after a tare (force N / moment Nm): %s',
             ';  '.join(f'{n} {f:.2f} / {t:.2f}' for n, (f, t) in rows_))
    log.info('     what is left with both is the torque sensors\' hysteresis -- the readings '
             'differ by that much arriving from either side.')
    good = float(np.max(sc)) < 5.0 and r['rms_nm'] < 0.3
    (log.info if good else log.warning)(
        'Quality: %s.', 'GOOD' if good else 'CHECK -- want the CoG within 5 mm and a residual '
        '< 0.3 Nm. Look at the per-reading residuals for a pose that was touched or still moving.')

    cell = cfg.get('_cell_path', 'the cell file')
    with open(os.path.join(out, 'result.yaml'), 'w') as fh:
        fh.write(f'# payload_calibration, {len(samples)} readings, {r["rms_nm"]:.3f} Nm RMS, '
                 f'CoG +-{np.max(sc):.1f} mm; mass '
                 + ('given' if step1 is None else f'+-{step1["sigma_mass_kg"]:.3f} kg') + '\n'
                 f'# -> {cell}, its robot: section. Merge -- do not replace the fri: block.\n'
                 f'robot:\n  payload:\n    mass_kg: {m:.3f}\n'
                 f'    cog_mm: [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}]\n'
                 f'  fri:\n    compensate_payload: true\n'
                 f'# OR, instead of the above (never both): the tool\'s load data in the Sunrise '
                 f'app -- mass {m:.3f} kg, centre of mass X/Y/Z {c[0]:.1f}/{c[1]:.1f}/{c[2]:.1f} '
                 f'mm in the flange frame.\n')
    with open(os.path.join(out, 'samples.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        dof = len(samples[0][0])
        wr.writerow(['reading', 'label'] + [f'q{i + 1}_deg' for i in range(dof)]
                    + [f'tau{i + 1}_nm' for i in range(dof)] + [f'tau{i + 1}_sd' for i in range(dof)]
                    + ['residual_nm'])
        for (k, label, q, tau, sd), res in zip(rows, r['residual_nm']):
            wr.writerow([k, label, *np.round(np.degrees(q), 3), *np.round(tau, 4),
                         *np.round(sd, 4), round(float(res), 4)])
    log.info('Written: %s', os.path.join(out, 'result.yaml'))


if __name__ == '__main__':
    main()
