"""Arm backends, and the one place that picks between them.

    robot.type      ur10e | iiwa14         the arm model (kinematics/), default ur10e
    robot.backend   ur_rtde | iiwa_fri     the hardware interface, default by type
    robot.dry_run   true                   the simulator, for either type (--dry-run)
"""

from ..arm import ArmError
from .. import kinematics
from .sim import SimArm

DEFAULT_BACKEND = {'ur10e': 'ur_rtde', 'iiwa14': 'iiwa_fri'}
# Which models each hardware backend can drive.
_BACKEND_MODELS = {'ur_rtde': ('ur10e',), 'iiwa_fri': ('iiwa14',)}


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
    from .iiwa_fri import IiwaFRIArm
    return IiwaFRIArm(cfg, frames=frames)


__all__ = ['make_arm', 'SimArm', 'DEFAULT_BACKEND']
