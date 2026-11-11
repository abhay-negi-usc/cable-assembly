"""bnc_assembly on the KUKA's own Cartesian impedance, in place of the software admittance (the user, 2026-11-06:
"write a new version of the script which uses the kuka's impedance control in place of admittance control").

    python -m cable_assembly.apps.bnc_assembly_impedance --cell iiwa14_bench --no-prompts --trials

THE SAME TASK, TREE, RECOVERY AND RECORDS as bnc_assembly -- run with robot.compliance_controller kuka_impedance.
Every compliant maneuver (the grasp descent, the engage, connector and collar clocking, the tug) then streams its
reference as it is and the CONTROLLER's spring yields to contact, at 1 kHz on its joint-torque sensors, instead of
this PC integrating an admittance at 125 Hz on the wrench estimated from them
(robot/admittance.py KukaImpedanceController). Force nulling (connector clocking's y/z/ry/rz) stays, as a set-point
drift with the KUKA's spring in series. Free-space moves are the same joint streams as before -- under impedance.

ON THE PENDANT, BEFORE STARTING:
  * the FRI app (LBRServer) with FRI control mode CARTESIAN_IMPEDANCE_CONTROL and client command mode POSITION.
    The arm is REFUSED at connect in any other mode: in POSITION nothing here would yield to anything.
  * the tool's load data in Sunrise (the Robotiq + camera + coupler: robot.payload in the cell). Without it the
    impedance spring carries the tool's weight as a deflection and the arm sags off every checked path -- REFUSED
    at connect when the force at rest reads about the tool's weight. With it, the PC no longer subtracts the
    weight itself (robot.fri.compensate_payload is ignored in this mode).
  * its Cartesian stiffness, written to robot.fri.cartesian_impedance.stiffness (N/m x3, Nm/rad x3) -- it cannot
    be read over FRI. The app sizes its pushes and pulls by it; KUKA's default is 2000 N/m / 200 Nm/rad.

WHAT DIFFERS FROM bnc_assembly (a `--set` placed FIRST, so a --set of your own after it wins):
  robot.compliance_controller=kuka_impedance
  and connector clocking's push sized for CLOCKING_PRESS_N (5 N) on the configured spring, unless you --set
  assembly.connector_clocking.push_mm yourself: 5 N / 3000 N/m = 1.7 mm on this bench (the admittance version:
  4.5 mm at 500 N/m over its 3 N deadband, ~5.3 N); always under the 5 mm early-out
WHAT IT CANNOT DO: soften or stiffen per maneuver -- the FRI session runs ONE impedance throughout, so the
per-maneuver stiffness / mass / damping / deadbands in the config do not apply. The start-up log states what each
virtual offset (pushes, the engage wiggle) means as a force on that spring; a wiggle of several mm against a socket
that holds still is ~2000 N/m x that.
"""

import sys

from . import bnc_assembly
from ._runner import run_app

DEFAULTS = (
    'robot.compliance_controller=kuka_impedance',
)
CLOCKING_PRESS_N = 5.0          # connector clocking's press on the controller's spring (with no --set push_mm)
PUSH_KEY = 'assembly.connector_clocking.push_mm'


def size_clocking_push(cfg, overrides):
    """Connector clocking's push for CLOCKING_PRESS_N on the KUKA's spring along its push (x), unless the operator
    set one. -> the push in mm, or None when it was theirs."""
    if any(str(o).startswith(PUSH_KEY + '=') for o in overrides):
        return None
    K = cfg.get_path('robot.fri.cartesian_impedance.stiffness')
    if not K:
        from ..robot.admittance import KUKA_DEFAULT_STIFFNESS
        K = KUKA_DEFAULT_STIFFNESS
    push = round(1000.0 * CLOCKING_PRESS_N / float(K[0]), 2)
    cfg.set_path(PUSH_KEY, push)
    return push


def add_args(parser):
    apply0 = bnc_assembly.add_args(parser)

    def apply(args, cfg):
        if apply0 is not None:
            apply0(args, cfg)
        size_clocking_push(cfg, getattr(args, 'overrides', []))
    return apply


def with_defaults(argv):
    """The command line with DEFAULTS as `--set`s in FRONT -- the loader applies them in order, so the operator's own
    --set of the same key, after, wins."""
    return [x for kv in DEFAULTS for x in ('--set', kv)] + list(argv)


def main(argv=None):
    sys.argv = [sys.argv[0]] + with_defaults(sys.argv[1:] if argv is None else argv)
    run_app("BNC assembly on the KUKA's Cartesian impedance (no software admittance)", 'bnc_assembly',
            bnc_assembly.build_and_run, needs_camera=True, precheck=bnc_assembly.precheck_and_preload,
            add_args=add_args)


if __name__ == '__main__':
    main()
