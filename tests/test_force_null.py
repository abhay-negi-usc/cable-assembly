"""robot/admittance.py null_axes: force nulling across a held part (connector clocking, 2026-11-06).

A connector held off-axis in a socket: the socket holds it laterally at its own axis, `eps` from the screw
axis the reference turns about, through a contact stiffness. The old controller's spring held that offset as
a side load; with the connector's y/z nulled, the pose gives until the side load is under the deadband."""

import numpy as np

from cable_assembly.robot.admittance import AdmittanceController

K_ENV = 5000.0                      # N/m: the socket + the soft pads + the arm, laterally
EPS = 0.002                         # the connector's axis 2 mm from the screw axis, along the connector's +y


def T_conn_in_tool0():
    """The bench's angled grasp: the connector pitched 70 deg about tool0 y, 180 mm out along tool z."""
    a = np.radians(-70.0)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
    T[:3, 3] = [0.0, 0.0, 0.18]
    return T


class SocketArm:
    """Tool0 commanded by servo_l; the socket pushes the connector back toward its axis, laterally only."""

    def __init__(self, T_tool0_conn):
        self.T_tc = T_tool0_conn
        self.T_cmd = np.eye(4)

    def servo_l(self, T, dt, lookahead=0.1, gain=300):
        self.T_cmd = np.array(T, dtype=float)

    def lateral_n(self):
        """The socket's push on the connector, in the connector's axes (x free)."""
        conn = self.T_cmd @ self.T_tc
        R = self.T_tc[:3, :3]                         # the reference is tool0 = identity
        d = R.T @ (conn[:3, 3] - self.T_tc[:3, 3])    # the connector's displacement, its axes
        return np.array([0.0, K_ENV * (EPS - d[1]), K_ENV * (0.0 - d[2])])

    def wrench(self):
        conn = self.T_cmd @ self.T_tc
        f = conn[:3, :3] @ self.lateral_n()           # base axes
        r = conn[:3, 3] - self.T_cmd[:3, 3]           # applied at the connector: its moment about tool0
        return np.concatenate([f, np.cross(r, f)])

    def servo_stop(self):
        pass


def settle(adm, arm, seconds=8.0):
    adm.reset()
    for _ in range(int(seconds * adm.rate)):
        adm._step(np.eye(4), 1.0 / adm.rate)
    return arm.lateral_n()


BENCH = {'stiffness': [1500.0, 1500.0, 1500.0, 10.0, 10.0, 10.0], 'mass': [50.0] * 3 + [5.0] * 3,
         'damping_ratio': [10.0] * 6, 'selected_axes': [1, 1, 1, 0, 0, 0], 'force_deadband_n': 3.0,
         'reference_rate_hz': 125.0}


def test_nulling_takes_the_side_load_off():
    T_tc = T_conn_in_tool0()
    arm = SocketArm(T_tc)
    spring = settle(AdmittanceController(arm, BENCH), arm)
    adm = AdmittanceController(arm, BENCH)
    adm.set_frame(T_tc)
    adm.null_axes([1, 2], damping=2000.0, deadband=1.5, max_offset=0.005)
    nulled = settle(adm, arm)
    assert abs(spring[1]) > 4.0, f'the old spring kept {spring[1]:.1f} N across'      # ~5.7 N
    assert abs(nulled[1]) <= 1.5 + 0.1, f'nulled: {nulled[1]:.2f} N left across'
    assert abs(adm.delta[1] - (EPS - 1.5 / K_ENV)) < 2e-4, 'it moved the connector toward the socket axis'
    assert abs(adm.delta[0]) < 1e-9 and abs(nulled[2]) < 1e-6, 'nothing along x or z to give to'


def test_the_null_gives_no_further_than_its_cap():
    T_tc = T_conn_in_tool0()
    arm = SocketArm(T_tc)
    adm = AdmittanceController(arm, BENCH)
    adm.set_frame(T_tc)
    adm.null_axes([1, 2], damping=2000.0, deadband=1.5, max_offset=0.001)
    left = settle(adm, arm)
    assert abs(adm.delta[1] - 0.001) < 1e-9, 'held at the cap'
    assert abs(left[1] - K_ENV * (EPS - 0.001)) < 0.1, 'and the rest of the offset is a load again'


def test_a_low_view_looks_up_at_the_socket_from_the_same_distance():
    """skills/pose6d_pick.py low_views (2026-11-06: the socket estimate pitched down, every view from above): the
    camera swung about the socket until it looks UP at it, the socket where it was in the image."""
    from scipy.spatial.transform import Rotation

    from cable_assembly.skills.pose6d_pick import low_views
    T_tool0_cam = np.eye(4)
    T_tool0_cam[:3, 3] = [-0.07, 0.0, 0.0175]
    pivot = np.array([0.80, 0.11, 0.27])
    # the camera 320 mm back, looking 40 deg down at it (camera z along the line of sight)
    a = np.array([np.cos(np.radians(40.0)), 0.0, -np.sin(np.radians(40.0))])
    R = Rotation.align_vectors([a, [0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])[0].as_matrix()
    cam = np.eye(4)
    cam[:3, :3], cam[:3, 3] = R, pivot - 0.32 * a
    T0 = cam @ np.linalg.inv(T_tool0_cam)
    in_cam0 = np.linalg.inv(cam) @ np.append(pivot, 1.0)
    for el, az in ((10.0, 0.0), (10.0, 20.0)):
        (name, T), = low_views(T0, T_tool0_cam, pivot, [{'elevation_deg': el, 'azimuth_deg': az}])
        c = T @ T_tool0_cam
        los = pivot - c[:3, 3]
        assert abs(np.linalg.norm(los) - 0.32) < 1e-9, 'the same distance'
        assert abs(np.degrees(np.arcsin(los[2] / np.linalg.norm(los))) - el) < 1e-6, 'looking up by el'
        assert np.allclose(np.linalg.inv(c) @ np.append(pivot, 1.0), in_cam0, atol=1e-9), 'the same image point'
        assert 'looking up +10' in name


def test_tilt_nulling_turns_the_plug_coaxial_with_the_socket():
    """ry/rz nulled about the connector's origin (the mating face): a socket whose axis is 2 deg off the screw
    axis (about the connector's y) is followed -- the moment the old stiff rotations held is taken off."""
    from scipy.spatial.transform import Rotation
    K_ROT, TILT = 200.0, np.radians(2.0)              # Nm/rad of the plug in the socket; its axis 2 deg off
    T_tc = T_conn_in_tool0()

    class TiltSocketArm(SocketArm):
        def wrench(self):
            conn = self.T_cmd @ self.T_tc
            R_true = self.T_tc[:3, :3] @ Rotation.from_rotvec([0.0, TILT, 0.0]).as_matrix()
            err = Rotation.from_matrix(conn[:3, :3].T @ R_true).as_rotvec()     # connector axes
            err[0] = 0.0                                                           # the screw turns freely
            return np.concatenate([np.zeros(3), conn[:3, :3] @ (K_ROT * err)])

    arm = TiltSocketArm(T_tc)
    stiff = AdmittanceController(arm, BENCH)
    stiff.set_frame(T_tc)
    settle(stiff, arm)
    assert abs(stiff.delta[4]) < 1e-9, 'the bench rotations are stiff: the 2 deg is held as a moment'
    adm = AdmittanceController(arm, BENCH)
    adm.set_frame(T_tc)
    adm.null_axes([4, 5], damping=100.0, deadband=0.5, max_offset=np.radians(3.0))
    settle(adm, arm)
    left = K_ROT * (TILT - adm.delta[4])
    assert abs(left) <= 0.5 + 0.05, f'{left:.2f} Nm left'
    assert abs(np.degrees(adm.delta[4]) - (2.0 - np.degrees(0.5 / K_ROT))) < 0.05, 'tilted to the socket'
    assert np.allclose(adm.delta[:3], 0.0, atol=1e-9), 'about the mating face: no slide'
