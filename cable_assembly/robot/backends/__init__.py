"""Arm backends, and the one place that picks between them.

    robot.type      ur10e | iiwa14                   the arm model (kinematics/), default ur10e
    robot.backend   ur_rtde | iiwa_fri | isaac_sim   the interface, default the hardware's by type
    robot.dry_run   true                             the kinematic simulator, either type (--dry-run)

isaac_sim is the iiwa 14 BNC cell simulated in Isaac Sim (sim/scripts/bnc_cell_server.py): the
physics and the contacts, with no camera -- see backends/isaac_sim.py.
"""

from ..arm import ArmError
from .. import kinematics
from .sim import SimArm

DEFAULT_BACKEND = {'ur10e': 'ur_rtde', 'iiwa14': 'iiwa_fri'}
# Which models each hardware backend can drive.
_BACKEND_MODELS = {'ur_rtde': ('ur10e',), 'iiwa_fri': ('iiwa14',), 'isaac_sim': ('iiwa14',)}
# Backends with no real sensors behind them: the camera and SAM3 behave as in a dry run.
_SIMULATED = ('isaac_sim',)


def simulated_perception(cfg):
    """True when there is no real camera to read: a dry run, or a simulated cell (which has the
    physics but, so far, no rendered camera). The camera returns black frames and SAM3 is not
    loaded -- a run on such a cell must not depend on perception (e.g. it starts held)."""
    r = cfg.section('robot')
    return bool(r.get('dry_run', False)) or str(r.get('backend') or '').strip().lower() in _SIMULATED


def make_arm(cfg, frames=None):
    """The Arm for `cfg`'s robot section: the simulator in a dry run, else the hardware backend."""
    r = cfg.section('robot')
    model = kinematics.get(r.get('type', 'ur10e'))
    if r.get('dry_run', False):
        return SimArm(cfg, model, frames=frames)
    backend = str(r.get('backend') or DEFAULT_BACKEND[model.NAME]).strip().lower()
    if backend not in _BACKEND_MODELS:
        raise ArmError(f'unknown robot.backend {backend!r}; known: {sorted(_BACKEND_MODELS)}')
    if model.NAME not in _BACKEND_MODELS[backend]:
        raise ArmError(f'robot.backend {backend!r} cannot drive a {model.NAME} '
                       f'(it drives {", ".join(_BACKEND_MODELS[backend])})')
    if backend == 'ur_rtde':
        from .ur_rtde import URArm
        return URArm(cfg, frames=frames)
    if backend == 'isaac_sim':
        from .isaac_sim import IsaacSimArm
        return IsaacSimArm(cfg, model, frames=frames)
    from .iiwa_fri import IiwaFRIArm
    return IiwaFRIArm(cfg, frames=frames)


__all__ = ['make_arm', 'simulated_perception', 'SimArm', 'DEFAULT_BACKEND']
