"""ETHERNET ASSEMBLY -- an RJ45 plug picked off the bench and plugged into its jack.

bnc_assembly's pipeline, run as a LATCH task (bnc_assembly.Task): the same localize -> pick -> plug in
the hand -> stand-off -> engage, the same behavior tree, recoveries, campaign (--trials) and records --
but no bayonet, so no connector clocking and no collar clocking. In their place:

  * PUSH HOME. The engage's success says the plug is IN the jack; the latch clicks over only at the
    bottom. So, the spring still loaded, on along the connector +X until the jack's floor pushes back
    (assembly.engage.push_home), then the press taken off.
  * THE LATCH TUG. In place, the grip kept, along the connector -X (assembly.tug_verify): a latched plug
    holds and is released and backed off -- ASSEMBLED. One that comes back out was not latched: a MISS,
    backed out (checked), laid down and picked again, like a missed engage.

WHAT IS DIFFERENT ABOUT THE PLUG, and where it is handled:
  * It goes in ONE way round. The 6D estimator measures both turns -- the plug's latch side before the
    pick, the jack's latch slot -- and the target keeps the jack's (pose6d.target_from_socket):
    assembly.engage_clock_deg does not apply.
  * It may lie latch DOWN. The pick grasps the levelled frame (from above, whichever way up it lies) and
    the in-hand belief carries the latch's turn (pose6d.grasp_junction, bnc_assembly latch_turn); a
    put-down keeps the hand upright, so the plug lands as it was picked.
  * No disassembly: pulling does not release a latch. More than one trial needs campaign.reset: manual
    -- the operator takes the plug out (pressing its latch) between trials.

THE CONFIG is configs/ethernet_assembly.yaml, which `extends: bnc_assembly.yaml` and names only what
differs; its `over_cell:` keeps the bench cell's BNC tuning (belief_offset, collar, disassembly, engage
roll) off this plug. `configure` below enforces the cut whatever the config says, and refuses what
cannot work (another plug, SAM3 -- it cannot see the latch -- another insertion mode).

    python -m cable_assembly.apps.ethernet_assembly --cell iiwa14_bench [--trials N] [--dry-run]

ON THE KUKA'S OWN IMPEDANCE instead of the software admittance: apps/ethernet_assembly_impedance (as
apps/bnc_assembly_impedance is to bnc_assembly) -- the same app with robot.compliance_controller
kuka_impedance and the tug resized for that spring; latch_precheck below checks either.
"""

from .. import log as urlog
from ..perception import pose6d as P6
from . import bnc_assembly as bnc
from ._runner import run_app

log = urlog.get('ethernet-assembly')

ETHERNET = bnc.Task('ethernet_assembly', 'Ethernet assembly', latch=True)

# What this task never does, whatever the config (or the bench cell) says.
_OFF = ('assembly.connector_clocking.enabled', 'assembly.collar_clocking.enabled', 'assembly.disassembly.enabled')


def configure(cfg):
    """The cut, enforced on `cfg` in place: the RJ45 plug, no clocking, no disassembly, the engage roll 0.
    False, said why, where the config asks for what cannot work. Idempotent."""
    plug = cfg.get_path('pose6d.plug')
    if plug != P6.RJ45:
        log.info('pose6d.plug %r -> %r: the plug this app assembles.', plug, P6.RJ45)
        cfg.set_path('pose6d.plug', P6.RJ45)
    method = str(cfg.get('pose_method') or 'sam3').strip().lower()
    if method != 'pose6d':
        log.error("pose_method %r: this app needs 'pose6d' -- only the 6D estimator measures which way round the "
                  'plug lies and the jack faces (its latch).', cfg.get('pose_method'))
        return False
    for path in _OFF:
        if cfg.get_path(path, False):
            log.info('%s -> false: an RJ45 has no bayonet to clock, and pulling does not unlatch it.', path)
            cfg.set_path(path, False)
    if float(cfg.get_path('assembly.engage_clock_deg', 0.0) or 0.0) != 0.0:
        log.info('assembly.engage_clock_deg %s -> 0: the plug goes in one way round, latch to slot.',
                 cfg.get_path('assembly.engage_clock_deg'))
        cfg.set_path('assembly.engage_clock_deg', 0.0)
        cfg.set_path('assembly.engage_clock_rad', 0.0)
    mode = str(cfg.get_path('assembly.insertion_mode') or 'engage').strip().lower()
    if mode != 'engage':
        log.error("assembly.insertion_mode %r: this app runs 'engage' only ('rl' is the BNC's policy, 'estimate' "
                  'needs a contact map of this plug).', mode)
        return False
    # THE TUG SIZED FOR THE SPRING IT RUNS ON (assembly.tug_verify.tell_apart_n, with pull_force_n null): the pull
    # that still moves an unlatched plug held by that much friction past the threshold -- K x threshold + it.
    tv = cfg.get_path('assembly.tug_verify') or {}
    if tv.get('pull_force_n') is None and tv.get('tell_apart_n') is not None:
        k_tug = _springs(cfg)[1]
        thr = float(tv.get('displacement_threshold_mm', 3.0))
        pull = round(k_tug * thr / 1000.0 + float(tv['tell_apart_n']), 1)
        log.info('assembly.tug_verify.pull_force_n -> %.1f N: %.0f N/m x the %.1f mm threshold + %.1f N '
                 '(tell_apart_n) -- an unlatched plug held by up to that much friction still comes out past it.',
                 pull, k_tug, thr, float(tv['tell_apart_n']))
        cfg.set_path('assembly.tug_verify.pull_force_n', pull)
    cp = dict(cfg.get('campaign') or {})
    n = int(cp.get('trials') or 1)
    reset = str(cp.get('reset') or 'auto').strip().lower()
    if n > 1 and reset != 'manual':
        log.error('%d TRIALS with campaign.reset %r: nothing puts the cable back between them here (no '
                  'disassembly: pulling does not release a latch). Set campaign.reset: manual -- you take the '
                  'plug out between trials.', n, reset)
        return False
    return True


def _springs(cfg):
    """(along the insertion for the push home, along the pull for the tug), N/m: the spring each really runs on -- the
    KUKA's Cartesian impedance (robot.compliance_controller kuka_impedance: robot.fri.cartesian_impedance.stiffness,
    one for the whole session) or the software admittance's (the engage's stiffness -- x the insertion axis in the
    connector frame, else the softest translation -- and the tug's)."""
    from ..robot.admittance import KUKA_DEFAULT_STIFFNESS

    def soft(K):
        return min(float(v) for v in list(K)[:3])
    if str(cfg.get_path('robot.compliance_controller') or 'admittance').strip().lower() == 'kuka_impedance':
        k = soft(cfg.get_path('robot.fri.cartesian_impedance.stiffness') or KUKA_DEFAULT_STIFFNESS)
        return k, k
    shared = cfg.get_path('compliance.stiffness') or [2000.0] * 3
    en = cfg.get_path('assembly.engage') or {}
    K_en = en.get('stiffness') or shared
    k_push = (float(K_en[0]) if str(en.get('compliance_frame') or 'tool0').lower() == 'connector' else soft(K_en))
    return k_push, soft(cfg.get_path('assembly.tug_verify.stiffness') or shared)


def latch_precheck(cfg):
    """The push home must be able to reach its force, and the tug must be able to FAIL: both are spring offsets.
    The tug pulls the reference pull / K behind the plug; an unlatched one, held by friction F, follows it until
    K x (offset - moved) = F -- so it is told apart only while F < pull - K x threshold. Refused at 0 or less (the
    tug could never fail), warned under 5 N. Stated, either way."""
    k_push, k_tug = _springs(cfg)
    ph = cfg.get_path('assembly.engage.push_home') or {}
    if ph.get('enabled', False):
        lead = 1000.0 * float(ph.get('force_n', 12.0)) / k_push
        if lead >= float(ph.get('max_mm', 20.0)):
            log.error('assembly.engage.push_home: %.1f N needs the reference %.1f mm ahead of the plug on %.0f N/m -- '
                      'at or past its max_mm %.1f: it could never be reached. Raise max_mm or lower force_n.',
                      float(ph.get('force_n', 12.0)), lead, k_push, float(ph.get('max_mm', 20.0)))
            return False
        log.info('PUSH HOME: %.1f N = the reference %.1f mm ahead of the plug (%.0f N/m), within max_mm %.1f.',
                 float(ph.get('force_n', 12.0)), lead, k_push, float(ph.get('max_mm', 20.0)))
    tv = cfg.get_path('assembly.tug_verify') or {}
    if tv.get('enabled', True):
        pull, thr = float(tv.get('pull_force_n', 3.0)), float(tv.get('displacement_threshold_mm', 3.0))
        tol = pull - k_tug * thr / 1000.0
        if tol <= 0.0:
            log.error('assembly.tug_verify: %.1f N on %.0f N/m offsets the reference only %.1f mm, not past the %.1f mm '
                      'threshold -- an unlatched plug could never move far enough to show: the tug could not fail. '
                      'Raise pull_force_n (or lower the threshold).', pull, k_tug, 1000.0 * pull / k_tug, thr)
            return False
        (log.warning if tol < 5.0 else log.info)(
            'LATCH TUG: %.1f N on %.0f N/m = a %.1f mm offset against the %.1f mm threshold: an unlatched plug is told '
            'apart while its friction in the jack is under %.1f N%s.', pull, k_tug, 1000.0 * pull / k_tug, thr, tol,
            ' -- MARGINAL, raise pull_force_n' if tol < 5.0 else '')
    return True


def precheck(cfg):
    """Before anything connects: the cut (configure), the push and the tug (latch_precheck), then bnc_assembly's own
    -- which also starts the 6D models loading on a real arm (precheck_and_preload)."""
    return configure(cfg) and latch_precheck(cfg) and bnc.precheck_and_preload(cfg)


def build_and_run(cfg, robot, camera, args):
    if not configure(cfg):
        return False
    return bnc.build_and_run(cfg, robot, camera, args, task=ETHERNET)


def main():
    run_app('Ethernet assembly: pick an RJ45 cable, plug it into its jack, tug its latch',
            'ethernet_assembly', build_and_run, needs_camera=True, precheck=precheck, add_args=bnc.add_args)


if __name__ == '__main__':
    main()
