"""bnc_pickup: bnc_assembly's pick, then the cable put back -- on the simulated iiwa, with the scan
stubbed to a cable lying on the bench under the pick pose (a dry run loads no SAM3)."""
import numpy as np

from cable_assembly import config as C
from cable_assembly.apps import bnc_pickup
from cable_assembly.robot import Robot
from cable_assembly.skills.pick import GraspCheck
from cable_assembly.transforms import frame_from_axis


class _Scanner:
    camera = None

    def __init__(self, T):
        self.T, self.scans = T, 0
        self.estimator = type('Estimator', (), {'reset': lambda self: None})()

    def scan(self, confirm=None):
        self.scans += 1
        return self.T.copy()

    def reselect(self):
        pass


def _setup(tmp_path, monkeypatch, heading_deg, *extra):
    """The iiwa cell, simulated, with a cable on the bench under the pick pose whose connector
    points `heading_deg` from base +x. The pitched BNC grasp reaches a connector pointing
    ACROSS the robot (60-120 or 240-300 deg) and not one pointing away from or toward it."""
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'skip_prompts=true',
                                  f'data_dir={tmp_path}', *extra])
    robot = Robot(cfg, needs=bnc_pickup.NEEDS)
    T_view = robot.arm.fk(np.radians(cfg.get('pick_joints_deg')))
    a = np.radians(heading_deg)
    T_cable = np.eye(4)
    T_cable[:3, :3] = frame_from_axis([np.cos(a), np.sin(a), 0.0], [0.0, 0.0, 1.0])
    T_cable[:3, 3] = [T_view[0, 3], T_view[1, 3], cfg.get_path('ground_plane.z_m')]
    scanner = _Scanner(T_cable)
    monkeypatch.setattr(bnc_pickup, 'build_scanner', lambda c, r, cam: (scanner, None, None))
    return cfg, robot, scanner


def test_an_unreachable_heading_stops_and_goes_home(tmp_path, monkeypatch):
    cfg, robot, scanner = _setup(tmp_path, monkeypatch, 0.0)
    try:
        assert not bnc_pickup.build_and_run(cfg, robot, None, None)
        assert scanner.scans == 1, 'the geometry refused; retrying the same approach cannot help'
        home = np.radians(cfg.get_path('reset.home_joints_deg'))
        assert np.allclose(robot.arm.q(), home, atol=1e-6)
    finally:
        robot.close()


def test_pick_lift_and_put_back(tmp_path, monkeypatch):
    cfg, robot, scanner = _setup(tmp_path, monkeypatch, 90.0, 'pickup_test.cycles=2')
    try:

        events = []
        close, release, move = robot.gripper.close, robot.gripper.open, robot.move_fingertip
        groove = GraspCheck(cfg).groove_counts

        def spy_close(*a, **k):
            # The simulated gripper closes to EMPTY by construction; a connector in the jaws
            # stalls the fingers in the groove band instead -- which is what the check reads.
            events.append(('close', robot.fingertip()))
            ok = close(*a, **k)
            robot.gripper._sim_pos = groove
            return ok

        def spy_open(label='open', *a, **k):
            events.append((label, robot.fingertip()))
            return release(label, *a, **k)

        def spy_move(T, label='', *a, **k):
            events.append((label, np.array(T)))
            return move(T, label, *a, **k)

        monkeypatch.setattr(robot.gripper, 'close', spy_close)
        monkeypatch.setattr(robot.gripper, 'open', spy_open)
        monkeypatch.setattr(robot, 'move_fingertip', spy_move)

        assert bnc_pickup.build_and_run(cfg, robot, None, None)
        assert scanner.scans == 2, 'one scan per cycle, no retries'

        grips = [T for e, T in events if e == 'close']
        downs = [T for e, T in events if e == 'put back (down)']
        assert len(downs) == 2
        for T_down in downs:
            # Released straight above where it was gripped, 5 mm up.
            T_grip = min(grips, key=lambda G: np.linalg.norm(G[:2, 3] - T_down[:2, 3]))
            assert np.allclose(T_down[:3, 3] - T_grip[:3, 3], [0.0, 0.0, 0.005], atol=1e-4)
            assert np.allclose(T_down[:3, :3], T_grip[:3, :3], atol=1e-4)
        labels = [e for e, _ in events]
        k = labels.index('release (put back)')
        assert labels[k - 1] == 'put back (down)' and labels[k + 1] == 'put back (clear)'
        home = np.radians(cfg.get_path('reset.home_joints_deg'))
        assert np.allclose(robot.arm.q(), home, atol=1e-6), 'ends at home'
    finally:
        robot.close()
