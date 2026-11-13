"""bnc_assembly on the KUKA's impedance, the cable PUT IN THE GRIPPER BY HAND (the user, 2026-11-12: "running the
bnc_assembly_impedance script from the beginning takes too long -- write a new version in which i can put the
connector within the gripper ... use RGBD in hand pose estimation").

    python -m cable_assembly.apps.bnc_assembly_impedance_handload --cell iiwa14_bench [--reaim] [--trials N]

bnc_assembly_impedance (its defaults and checks, see there) with assembly.hand_load on: no cable scan and no pick.
Home, the socket found by the 6D estimator; then the fingers open a little wider than the plug, and

    [HAND LOAD] Put the BNC plug between the pads ...   Enter -> the gripper closes, the grasp check reads it
    [HAND LOAD] Gripped. Hands clear -- Enter            -> the in-hand RGB-D estimate, then the approach

and on as bnc_assembly: the socket refined, the stand-off, the engage (--reaim: re-aimed after a miss), the seat,
the lock, the tug. The plug goes in as the pick takes it -- its CAD is drawn on the tool in the live view
(sim/run.sh python scripts/live_view_gl.py); the in-hand 6D corrects how it was placed (within
pose6d.in_hand.max_correction_mm / _deg). assembly.hand_load (configs/bnc_assembly.yaml): the opening, the pose.
"""

import sys

from . import bnc_assembly, bnc_assembly_impedance
from ._runner import run_app

DEFAULTS = bnc_assembly_impedance.DEFAULTS + (
    'assembly.hand_load.enabled=true',
)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    sys.argv = [sys.argv[0]] + [x for kv in DEFAULTS for x in ('--set', kv)] + args
    run_app('BNC assembly on the KUKA impedance, the cable loaded into the gripper by hand', 'bnc_assembly',
            bnc_assembly.build_and_run, needs_camera=True, precheck=bnc_assembly.precheck_and_preload,
            add_args=bnc_assembly_impedance.add_args)


if __name__ == '__main__':
    main()
