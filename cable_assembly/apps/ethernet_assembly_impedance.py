"""ethernet_assembly on the KUKA's own Cartesian impedance, in place of the software admittance -- the RJ45's
counterpart of apps/bnc_assembly_impedance.py.

    python -m cable_assembly.apps.ethernet_assembly_impedance --cell iiwa14_bench --no-prompts --trials 10

THE SAME TASK, TREE, RECOVERY AND RECORDS as ethernet_assembly (configs/ethernet_assembly.yaml, its cut enforced by
ethernet_assembly.configure) -- run with robot.compliance_controller kuka_impedance. Every compliant maneuver (the
grasp descent, the contact creep and the engage, the push home, the latch tug, the backing out of a miss) then
streams its reference as it is and the CONTROLLER's spring yields to contact, at 1 kHz on its joint-torque sensors,
instead of this PC integrating an admittance at 125 Hz on the wrench estimated from them
(robot/admittance.py KukaImpedanceController). Free-space moves are the same joint streams -- under impedance.

ON THE PENDANT, BEFORE STARTING (as for bnc_assembly_impedance):
  * the FRI app (LBRServer) in FRI control mode CARTESIAN_IMPEDANCE_CONTROL, client command mode POSITION -- the arm
    is REFUSED at connect in any other mode;
  * the tool's load data in Sunrise (robot.payload in the cell) -- REFUSED at connect without it;
  * its Cartesian stiffness written to robot.fri.cartesian_impedance.stiffness -- FRI cannot read it, and the pushes
    and pulls below are sized by it (KUKA's default: 2000 N/m, 200 Nm/rad).

WHAT DIFFERS FROM ethernet_assembly (each a `--set` placed FIRST, so a --set of your own after it wins):
  robot.compliance_controller=kuka_impedance
  assembly.tug_verify.pull_force_n=null   THE LATCH TUG SIZED FOR THE KUKA'S SPRING. It is a spring pull: the
  assembly.tug_verify.tell_apart_n=10.0   reference is offset pull / K and the verdict is how far the plug follows
                                          it (> 3 mm: unlatched). The admittance version's 8 N is a 4 mm offset on
                                          2000 N/m (2.7 on 3000) -- an unlatched plug held by more than 2 N of its
                                          contacts' friction would stop inside the 3 mm and read LATCHED. So the
                                          pull is K x 3 mm + 10 N (ethernet_assembly.configure): an unlatched plug
                                          is told apart up to 10 N whatever the pendant's stiffness -- 16 N at KUKA's
                                          2000 N/m, 19 N at 3000 (the admittance version: 6.5 N). The precheck states
                                          it, and refuses a tug that cannot fail.
WHAT STAYS AS IT IS, on this spring:
  * the push home (assembly.engage.push_home): 12 N needs the reference 12 / K ahead of the plug (6 mm at 2000 N/m;
    the admittance's (12 - 3 N deadband) / 1500), inside its 20 mm; the unload backs off by the lead as measured.
  * the engage: the force-limited advance and every detector read the force, not a stiffness. Its wiggle (up to 3
    mm, wiggle_sampling.yaml) is up to K x 3 mm on this spring against a jack that holds still (6 N at 2000 N/m, 9 N
    at 3000) -- logged at start-up -- where the admittance's 500 N/m gave way at 1.5 N; it now does move the plug
    across the jack's face.
WHAT IT CANNOT DO: soften or stiffen per maneuver -- the FRI session runs ONE impedance throughout, so the engage's
stiffness and compliance_frame, the tug's stiffness and the deadbands in the config do not apply.
"""

import sys

from . import bnc_assembly, ethernet_assembly
from ._runner import run_app

DEFAULTS = (
    'robot.compliance_controller=kuka_impedance',
    'assembly.tug_verify.pull_force_n=null',
    'assembly.tug_verify.tell_apart_n=10.0',
)


def with_defaults(argv):
    """The command line with DEFAULTS as `--set`s in FRONT -- the loader applies them in order, so the operator's own
    --set of the same key, after, wins."""
    return [x for kv in DEFAULTS for x in ('--set', kv)] + list(argv)


def main(argv=None):
    sys.argv = [sys.argv[0]] + with_defaults(sys.argv[1:] if argv is None else argv)
    run_app("Ethernet assembly on the KUKA's Cartesian impedance (no software admittance)", 'ethernet_assembly',
            ethernet_assembly.build_and_run, needs_camera=True, precheck=ethernet_assembly.precheck,
            add_args=bnc_assembly.add_args)


if __name__ == '__main__':
    main()
