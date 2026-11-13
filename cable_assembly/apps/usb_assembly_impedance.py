"""usb_assembly on the KUKA's own Cartesian impedance, in place of the software admittance -- as
bnc_assembly_impedance is to bnc_assembly (the user, 2026-11-12: "write usb_assemble_impedance -- it only needs
insertion and does not need a clocking or verification phase").

    python -m cable_assembly.apps.usb_assembly_impedance --cell iiwa14_bench --hand-load --reaim
    python -m cable_assembly.apps.usb_assembly_impedance --cell iiwa14_bench [--trials N] [--dry-run]

THE SAME TASK, TREE, RECOVERY AND RECORDS as usb_assembly (see there: localize the receptacle, the plug in the hand,
the stand-off, the engage, the push home, release -- INSERTED; no clocking, no tug) -- run with
robot.compliance_controller kuka_impedance: the engage and the push home stream their references and the
CONTROLLER's spring yields to contact. A contact, and the seat's sideways hold, count only with the arm held back
on that spring (assembly.engage.held_back).

ON THE PENDANT, BEFORE STARTING (as for bnc_assembly_impedance): the FRI app in CARTESIAN_IMPEDANCE_CONTROL (client
command mode POSITION), the tool's load data in Sunrise, and its Cartesian stiffness written to
robot.fri.cartesian_impedance.stiffness -- the push home is sized by it (20 N on 3000 N/m: the reference 6.7 mm
ahead of the plug).

--hand-load: no cable scan and no pick -- the fingers open, you put the plug between the pads (as its CAD on the
tool shows it in the live view: a USB-A plug goes in one way up), Enter closes them; the in-hand RGB-D estimate
then measures how it sits. --reaim [N]: after a missed engage, back out, re-aim on the 6D estimates, engage again
with the spiral probe.
"""

import sys

from . import bnc_assembly, usb_assembly
from ._runner import run_app

DEFAULTS = (
    'robot.compliance_controller=kuka_impedance',
)


def with_defaults(argv):
    """The command line with DEFAULTS as `--set`s in FRONT -- the loader applies them in order, so the operator's own
    --set of the same key, after, wins."""
    return [x for kv in DEFAULTS for x in ('--set', kv)] + list(argv)


def main(argv=None):
    sys.argv = [sys.argv[0]] + with_defaults(sys.argv[1:] if argv is None else argv)
    run_app("USB assembly on the KUKA's Cartesian impedance (insertion only; no software admittance)", 'usb_assembly',
            usb_assembly.build_and_run, needs_camera=True, precheck=usb_assembly.precheck,
            add_args=bnc_assembly.add_args)


if __name__ == '__main__':
    main()
