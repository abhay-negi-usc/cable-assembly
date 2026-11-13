"""USB ASSEMBLY -- a USB-A plug into its receptacle: INSERTION ONLY.

bnc_assembly's pipeline run as a push-fit task (bnc_assembly.Task, latch=True, verify=False; the user, 2026-11-12:
"it only needs insertion and does not need a clocking or verification phase"): the same localize -> plug in the
hand -> stand-off -> engage, the same behavior tree, recoveries, campaign (--trials) and records -- and after the
engage only:

  * PUSH HOME. The engage's success says the plug is IN the receptacle; on along the connector +X until the
    receptacle's stop pushes back (assembly.engage.push_home), then the press taken off.
  * RELEASE AND BACK OFF. The trial ends INSERTED. No connector clocking, no collar clocking, no tug, no
    disassembly (a person pulls the plug out between trials: campaign.reset manual).

WHAT IT KEEPS FROM THE BNC'S: the 6D estimates (the receptacle from several views, the plug in the hand by RGB-D
with the jaws' 3 fixed DOF kept), the online estimates through the approach, the contact check on the KUKA's
spring, and -- with --reaim -- the re-aim after a missed engage. With --hand-load there is no cable scan and no
pick: the operator puts the plug between the pads.

WHAT IS DIFFERENT ABOUT THE PLUG: it goes in ONE way up (the receptacle's tongue into its cavity), and its two wide
faces look much alike -- the 6D estimator measures the plug's and the receptacle's turn, but the plug's can be half
a turn out (configs/usb_assembly.yaml pose6d.grasp_square_deg). Hand-loaded, put it in as the CAD on the tool
shows it. Its models (usb_plug, usb_adapter) are from McMaster's CAD (4974T135, 4574N11) and were not yet seen on
the bench when this was written.

THE CONFIG is configs/usb_assembly.yaml, which `extends: bnc_assembly.yaml` and names only what differs.
`configure` below enforces the cut whatever the config says.

    python -m cable_assembly.apps.usb_assembly --cell iiwa14_bench [--hand-load] [--reaim] [--trials N] [--dry-run]

ON THE KUKA'S OWN IMPEDANCE: apps/usb_assembly_impedance.
"""

from .. import log as urlog
from ..perception import pose6d as P6
from . import bnc_assembly as bnc
from . import ethernet_assembly as eth
from ._runner import run_app

log = urlog.get('usb-assembly')

USB = bnc.Task('usb_assembly', 'USB assembly', latch=True, verify=False)

# What this task never does, whatever the config (or the bench cell) says.
_OFF = ('assembly.connector_clocking.enabled', 'assembly.collar_clocking.enabled', 'assembly.disassembly.enabled',
        'assembly.tug_verify.enabled')


def configure(cfg):
    """The cut, enforced on `cfg` in place: the USB plug, no clocking, no tug, no disassembly, the engage roll 0.
    False, said why, where the config asks for what cannot work. Idempotent."""
    plug = cfg.get_path('pose6d.plug')
    if plug != P6.USB:
        log.info('pose6d.plug %r -> %r: the plug this app assembles.', plug, P6.USB)
        cfg.set_path('pose6d.plug', P6.USB)
    method = str(cfg.get('pose_method') or 'sam3').strip().lower()
    if method != 'pose6d':
        log.error("pose_method %r: this app needs 'pose6d' -- only the 6D estimator measures which way up the plug "
                  'and the receptacle are.', cfg.get('pose_method'))
        return False
    for path in _OFF:
        if cfg.get_path(path, False):
            log.info('%s -> false: a USB plug is pushed in and that is all -- no bayonet, no latch to tug.', path)
            cfg.set_path(path, False)
    if float(cfg.get_path('assembly.engage_clock_deg', 0.0) or 0.0) != 0.0:
        log.info('assembly.engage_clock_deg %s -> 0: the plug goes in one way up.',
                 cfg.get_path('assembly.engage_clock_deg'))
        cfg.set_path('assembly.engage_clock_deg', 0.0)
        cfg.set_path('assembly.engage_clock_rad', 0.0)
    mode = str(cfg.get_path('assembly.insertion_mode') or 'engage').strip().lower()
    if mode != 'engage':
        log.error("assembly.insertion_mode %r: this app runs 'engage' only ('rl' is the BNC's policy, 'estimate' "
                  'needs a contact map of this plug).', mode)
        return False
    cp = dict(cfg.get('campaign') or {})
    n = int(cp.get('trials') or 1)
    reset = str(cp.get('reset') or 'auto').strip().lower()
    if n > 1 and reset != 'manual':
        log.error('%d TRIALS with campaign.reset %r: nothing takes the plug back out between them here. Set '
                  'campaign.reset: manual -- you pull it out between trials.', n, reset)
        return False
    return True


def precheck(cfg):
    """Before anything connects: the cut (configure), the push home on the spring it runs on (ethernet_assembly's
    latch_precheck: the tug is off, so only the push is checked), then bnc_assembly's own -- which also starts the
    6D models loading on a real arm."""
    return configure(cfg) and eth.latch_precheck(cfg) and bnc.precheck_and_preload(cfg)


def build_and_run(cfg, robot, camera, args):
    if not configure(cfg):
        return False
    return bnc.build_and_run(cfg, robot, camera, args, task=USB)


def main():
    run_app('USB assembly: a USB-A plug into its receptacle (insertion only)',
            'usb_assembly', build_and_run, needs_camera=True, precheck=precheck, add_args=bnc.add_args)


if __name__ == '__main__':
    main()
