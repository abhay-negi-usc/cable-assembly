"""perception/pose6d.py and skills/pose6d_pick.py: the 6D pose estimator for the BNC connector and
socket, from several views. The frames are checked against the CAD's numbers (the plug seated in
the socket); the client, the scanner and the pick's wiring run against tests/pose6d_standin.py,
which speaks the service's protocol with known answers."""

import builtins
import json
import os
import sys
import time

import numpy as np
import pytest

from cable_assembly import config as C
from cable_assembly.perception import pose6d as P6
from cable_assembly.perception.camera import Frame

HERE = os.path.dirname(os.path.abspath(__file__))
STANDIN = os.path.join(HERE, 'pose6d_standin.py')


def _T(R=None, t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.eye(3) if R is None else R, t
    return T


# ---------------------------------------------------------------------------- the frames

def test_the_junction_frame_is_where_the_cable_leaves_the_strain_relief():
    # The cord lying along base +x, 10 mm up.
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])        # cord z -> base x
    T_plug = _T(R, [0.4, 0.1, 0.010])
    J = P6.junction_from_plug(T_plug)
    np.testing.assert_allclose(J[:3, 3], [0.4 + 0.10865, 0.1, 0.010], atol=1e-12)
    np.testing.assert_allclose(J[:3, 0], [1.0, 0.0, 0.0], atol=1e-12)          # toward the mating face
    np.testing.assert_allclose(J[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)          # z up, as every scanner
    # Tilted 30 deg nose-up: x follows the axis, z stays as near up as it can.
    Rt = np.array([[np.cos(np.radians(30)), 0, -np.sin(np.radians(30))], [0, 1, 0],
                   [np.sin(np.radians(30)), 0, np.cos(np.radians(30))]]) @ R
    J = P6.junction_from_plug(_T(Rt))
    assert np.degrees(np.arcsin(J[2, 0])) == pytest.approx(30.0)
    assert J[2, 2] > 0.8


def test_seated_the_plug_rests_on_the_socket_as_the_cad_says():
    """In the socket's frame: the plug on its axis, facing down, the insulator face (151.38 along
    the cord) on the dielectric (3.73 up the socket), the junction 46.46 mm up."""
    S = P6.T_socket_cord_seated()
    def to_socket(z_cord):
        return (S @ [0.0, 0.0, z_cord, 1.0])[:3]

    np.testing.assert_allclose(to_socket(0.15138), [0.0, 0.0, 0.00373], atol=1e-12)
    np.testing.assert_allclose(to_socket(0.15240), [0.0, 0.0, 0.00271], atol=1e-12)   # the mating face
    J = P6.seated_junction_in_socket()
    np.testing.assert_allclose(J[:3, 3], [0.0, 0.0, 0.04646], atol=1e-12)
    np.testing.assert_allclose(J[:3, 0], [0.0, 0.0, -1.0], atol=1e-12)          # x INTO the socket
    np.testing.assert_allclose(J[:3, :3] @ J[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(J[:3, :3]) == pytest.approx(1.0)


def test_the_target_is_the_seated_connector_frame_in_base():
    """A socket on a wall facing base -x: the target's x points into the wall (+x), at the seated
    junction; a junction -> connector offset carries straight through, as the belief's does."""
    R_s = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])       # socket z -> base -x
    T_socket = _T(R_s, [0.8, 0.0, 0.3])
    T = P6.target_from_socket(T_socket, np.eye(4))
    np.testing.assert_allclose(T[:3, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)          # its turn: z up, by convention
    np.testing.assert_allclose(T[:3, 3], [0.8 - 0.04646, 0.0, 0.3], atol=1e-12)
    # The socket's own turn about its axis (which a view cannot tell) changes nothing.
    turned = T_socket @ _T(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(P6.target_from_socket(turned, np.eye(4)), T, atol=1e-12)
    off = _T(t=[0.02, 0.0, 0.0])                                               # a datum 20 mm along x
    np.testing.assert_allclose(P6.target_from_socket(T_socket, off)[:3, 3], [0.8 - 0.02646, 0.0, 0.3], atol=1e-12)


def test_the_cad_meshes_have_the_frames_the_conversions_assume():
    info = os.path.expanduser('~/abhay_ws/RAG-6DPose-code-abhay/data/bnc/models/models_info.json')
    if not os.path.isfile(info):
        pytest.skip('no RAG-6DPose models (pose_pipeline.prepare_models)')
    with open(info) as f:
        m = json.load(f)
    assert m['3']['min_z'] / 1000.0 == pytest.approx(P6.JUNCTION_Z_M, abs=2e-5)      # the plug starts there
    assert (m['3']['min_z'] + m['3']['size_z'] / 2) / 1000.0 == pytest.approx(P6.CENTRE_M[P6.PLUG][2], abs=5e-4)
    assert (m['1']['min_z'] + m['1']['size_z']) == pytest.approx(12.14, abs=0.01)     # the barrel's top
    if '5' in m:                                                                     # the RJ45, turned
        assert m['5']['min_z'] / 1000.0 == pytest.approx(P6.JUNCTION_Z[P6.RJ45], abs=2e-5)
        assert (m['5']['min_z'] + m['5']['size_z']) / 1000.0 == pytest.approx(P6.MATING_Z[P6.RJ45], abs=2e-5)
        assert m['5']['min_x'] + m['5']['size_x'] / 2 == pytest.approx(0.0, abs=0.01)  # on the axis, across x
        assert (m['5']['min_y'] + m['5']['size_y'] / 2) / 1000.0 == pytest.approx(P6.CENTRE_M[P6.RJ45][1], abs=5e-4)
    if '6' in m:                                                                     # the RJ45 coupler
        assert m['6']['min_z'] + m['6']['size_z'] == pytest.approx(18.30, abs=0.01)  # the front jack's face
        assert (m['6']['min_z'] + m['6']['size_z'] / 2) / 1000.0 == pytest.approx(P6.CENTRE_M[P6.RJ45_SOCKET][2], abs=5e-4)


def test_the_rj45_junction_frame_keeps_the_measured_turn():
    """Lying on its side (latch toward base +y), cord along base +x: x toward the mating face, z
    along the latch side -- not turned up, as a round plug's would be."""
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])        # cord z -> base x, y -> y
    J = P6.junction_from_plug(_T(R, [0.4, 0.1, 0.006]), part=P6.RJ45)
    np.testing.assert_allclose(J[:3, 3], [0.4 + 0.11930, 0.1, 0.006], atol=1e-12)
    np.testing.assert_allclose(J[:3, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(J[:3, 2], [0.0, 1.0, 0.0], atol=1e-12)          # the latch side
    assert np.linalg.det(J[:3, :3]) == pytest.approx(1.0)
    # Its turn about the axis is in the belief too: the measurement carries it.
    from cable_assembly.transforms import inverse
    T_jc = _T(t=[0.015, 0.0, 0.0])
    belief = _T(_Rz(10), [0.01, 0.002, 0.03])
    plug = P6.plug_from_belief(belief, T_jc, P6.RJ45)
    np.testing.assert_allclose(P6.belief_from_plug(plug, T_jc, belief, P6.RJ45), belief, atol=1e-12)
    turned = P6.belief_from_plug(plug @ _T(_Rz(70)), T_jc, belief, P6.RJ45)
    np.testing.assert_allclose(turned, plug @ _T(_Rz(70)) @ P6.T_cord_junction(P6.RJ45) @ T_jc, atol=1e-12)
    assert np.linalg.norm(turned[:3, :3] - belief[:3, :3]) > 0.5
    np.testing.assert_allclose((turned @ inverse(T_jc))[:3, 0], (belief @ inverse(T_jc))[:3, 0], atol=1e-12)


def test_seated_the_rj45_plug_rests_on_the_jack_floor_latch_to_slot():
    """In the coupler's frame: the plug down its -z, its tip (153.74 along the cord) on the front
    jack's floor (6.18 up), its latch (+y) toward the latch slot (-y); the junction 34.44 mm
    further out, x INTO the jack, z (the latch side) toward -y."""
    S = P6.T_socket_cord_seated(P6.RJ45_SOCKET)
    np.testing.assert_allclose((S @ [0.0, 0.0, 0.15374, 1.0])[:3], [0.0, 0.0, 0.00618], atol=1e-12)
    np.testing.assert_allclose(S[:3, :3] @ [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], atol=1e-12)
    J = P6.seated_junction_in_socket(P6.RJ45_SOCKET)
    np.testing.assert_allclose(J[:3, 3], [0.0, 0.0, 0.00618 + 0.15374 - 0.11930], atol=1e-12)
    np.testing.assert_allclose(J[:3, 0], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(J[:3, 2], [0.0, -1.0, 0.0], atol=1e-12)
    assert np.linalg.det(J[:3, :3]) == pytest.approx(1.0)
    assert P6.SOCKET_FOR[P6.RJ45] == P6.RJ45_SOCKET and P6.PLUG_FOR[P6.SOCKET] == P6.PLUG


def test_the_rj45_target_keeps_the_jack_s_measured_turn():
    """A coupler on a wall facing base -x, its latch slot down: the target's x into the wall, its z
    (the plug's latch side) down with the slot -- and turned with the jack, not reset to up."""
    R_s = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])  # x -> +y, y -> up, z -> -x
    T_socket = _T(R_s, [0.8, 0.0, 0.3])
    T = P6.target_from_socket(T_socket, np.eye(4), socket=P6.RJ45_SOCKET)
    np.testing.assert_allclose(T[:3, 0], [1.0, 0.0, 0.0], atol=1e-12)          # into the wall
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)         # the latch, with the slot
    np.testing.assert_allclose(T[:3, 3], [0.8 - (0.00618 + 0.15374 - 0.11930), 0.0, 0.3], atol=1e-12)
    # Turned half round about its axis, the target turns too (the BNC's would not).
    turned = T_socket @ _T(_Rz(180))
    np.testing.assert_allclose(P6.target_from_socket(turned, np.eye(4), socket=P6.RJ45_SOCKET)[:3, 2],
                               [0.0, 0.0, 1.0], atol=1e-12)


def test_the_plug_is_chosen_in_the_config():
    assert P6.plug_part(C.load('bnc_assembly')) == P6.PLUG
    assert P6.plug_part(C.load('bnc_assembly', ['pose6d.plug=rj45_plug'])) == P6.RJ45
    assert P6.socket_part(C.load('bnc_assembly', ['pose6d.plug=rj45_plug'])) == P6.RJ45_SOCKET
    assert P6.socket_part(C.load('bnc_assembly')) == P6.SOCKET
    with pytest.raises(P6.Pose6DError):
        P6.plug_part(C.load('bnc_assembly', ['pose6d.plug=usb']))


# ---------------------------------------------------------------------------- the service

TRUTH_PLUG = _T(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]), [0.50, 0.10, 0.0136 - 0.0])
TRUTH_SOCKET = _T(np.eye(3), [0.7, -0.1, 0.05])


@pytest.fixture
def standin(tmp_path, monkeypatch):
    truth = tmp_path / 'truth.json'
    truth.write_text(json.dumps({'cord_end_pos_z': TRUTH_PLUG.tolist(), 'adapter_mount': TRUTH_SOCKET.tolist()}))
    monkeypatch.setenv('POSE6D_TRUTH', str(truth))
    repo = tmp_path / 'repo'
    (repo / 'pose_pipeline').mkdir(parents=True)
    over = [f'pose6d.python={sys.executable}', f'pose6d.server={STANDIN}', f'pose6d.repo={repo}']
    yield over
    P6.shutdown()


def _frame(T_base_cam=None):
    T_base_cam = np.eye(4) if T_base_cam is None else T_base_cam
    return Frame(np.zeros((90, 160, 3), np.uint8), np.array([[116.0, 0, 80], [0, 116.0, 45], [0, 0, 1]]),
                 np.zeros(5), time.monotonic(), T_base_cam, np.full((90, 160), 0.35, np.float32))


def test_the_client_runs_the_service_and_reads_its_poses(standin):
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d', *standin])
    svc = P6.service(cfg)
    assert P6.service(cfg) is svc                                             # one per run
    r = svc.estimate(P6.SOCKET, [_frame(), _frame()])
    assert r['found'] and r['n_used'] == 2
    np.testing.assert_allclose(r['T_base_obj'], TRUTH_SOCKET)
    assert isinstance(r['views'][1]['T_base_obj'], np.ndarray)
    assert not os.listdir(svc._dir)                                           # the frames cleaned up
    with pytest.raises(P6.Pose6DError):
        svc.estimate(P6.PLUG, [Frame(np.zeros((9, 9, 3), np.uint8), np.eye(3), None, 0.0, None, None)])
    P6.shutdown()
    assert P6._SERVICE is None


def test_the_estimator_is_preloaded_in_the_background_with_the_plug_and_socket(standin):
    over = ['cell=iiwa14_bench', 'pose_method=pose6d', 'assembly.target_source=pose6d']
    cfg = C.load('bnc_assembly', [*over, *standin])
    P6.preload(cfg).join(20.0)
    svc = P6._SERVICE
    assert svc is not None and P6.service(cfg) is svc                         # the run's one, already up
    assert svc.preload == [P6.PLUG, P6.SOCKET]
    assert svc.ready['preloaded'] == ['cord_end_pos_z', 'adapter_mount']             # loaded before it said ready
    assert set(svc._models) == {P6.PLUG, P6.SOCKET}                           # their points fetched too
    P6.shutdown()
    cfg = C.load('bnc_assembly', [*over, 'pose6d.plug=rj45_plug', *standin])
    assert P6.service(cfg).ready['preloaded'] == ['rj45_plug', 'rj45_adapter']
    P6.shutdown()
    # The iiwa cell measures the socket by default, so its model is loaded with the plug's.
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d', *standin])
    assert cfg.get_path('assembly.target_source') == 'pose6d'
    assert P6.service(cfg).ready['preloaded'] == ['cord_end_pos_z', 'adapter_mount']
    P6.shutdown()
    # The socket from the recorded mate (target_source kinematic): its model is not loaded.
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d',
                                  'assembly.target_source=kinematic', *standin])
    assert P6.service(cfg).ready['preloaded'] == ['cord_end_pos_z']
    P6.shutdown()
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d', 'pose6d.preload=[]', *standin])
    assert P6.service(cfg).ready['preloaded'] == []


def test_held_rigidly_the_scan_s_estimate_moves_with_the_flange_from_the_close():
    from cable_assembly.skills.pick import (
        grasp_pose,
        held_junction_in_fingertip,
        rigidly_held_junction,
    )
    from cable_assembly.transforms import inverse
    T_tool0_tip = _T(t=[0.0, 0.0, 0.19])
    T_junction = _T(_Rz(30), [0.5, 0.1, 0.012])                               # the scan's estimate
    G = _T(_Rz(180) @ _Rx(-60), [0.02, 0.0, 0.01])                            # the grasp: fingertip in it
    T_tool0_closed = grasp_pose(T_junction, G) @ inverse(T_tool0_tip)         # landed as commanded
    np.testing.assert_allclose(rigidly_held_junction(T_tool0_tip, T_tool0_closed, T_junction),
                               held_junction_in_fingertip(G), atol=1e-12)
    # Reseated 4 mm along its own approach before the close that held: the belief has it.
    T_reseated = T_tool0_closed @ T_tool0_tip @ _T(t=[0.0, 0.0, -0.004]) @ inverse(T_tool0_tip)
    J = rigidly_held_junction(T_tool0_tip, T_reseated, T_junction)
    np.testing.assert_allclose(T_reseated @ T_tool0_tip @ J, T_junction, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(J[:3, 3] - held_junction_in_fingertip(G)[:3, 3]), 0.004, atol=1e-9)


def test_the_cad_points_are_asked_for_once(standin):
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d', *standin])
    svc = P6.service(cfg)
    p = svc.model_points(P6.PLUG)
    n = svc._n
    assert p.ndim == 2 and p.shape[1] == 3 and len(p) > 50
    assert svc.model_points(P6.PLUG) is p and svc._n == n


# ---------------------------------------------------------------------------- in the hand

def _Rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Rx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_the_in_hand_belief_takes_the_measured_axis_and_keeps_its_turn():
    from cable_assembly.transforms import inverse
    T_jc = _T(t=[0.021, 0.0, -0.003])                                          # a junction -> connector datum
    belief = _T(_Rz(10) @ _Rx(25), [0.01, 0.002, 0.03])
    plug = P6.plug_from_belief(belief, T_jc)
    np.testing.assert_allclose(P6.belief_from_plug(plug, T_jc, belief), belief, atol=1e-12)
    # Turned about its own axis -- which no view can tell -- it is the same belief.
    np.testing.assert_allclose(P6.belief_from_plug(plug @ _T(_Rz(70)), T_jc, belief), belief, atol=1e-12)
    # Measured 2 mm off across its axis: the junction moves with it.
    moved = P6.belief_from_plug(plug @ _T(t=[0.0, 0.002, 0.0]), T_jc, belief)
    np.testing.assert_allclose((moved @ inverse(T_jc))[:3, 3] - (belief @ inverse(T_jc))[:3, 3],
                               plug[:3, :3] @ [0.0, 0.002, 0.0], atol=1e-12)


def test_the_in_hand_pose_is_the_medoid_and_an_outlier_does_not_agree():
    Ts = [TRUTH_PLUG @ _T(t=[0.0003 * k, 0.0, 0.0]) for k in range(4)] + [TRUTH_PLUG @ _T(t=[0.0, 0.02, 0.0])]
    T, n = P6.robust_pose(Ts, P6.PLUG, 0.003, np.radians(5.0))
    assert n == 4 and np.linalg.norm(T[:3, 3] - TRUTH_PLUG[:3, 3]) < 0.001
    # Turned 90 deg about its axis: the same round plug, but not the same RJ45.
    Ts = [TRUTH_PLUG @ _T(_Rz(90 * (k == 3))) for k in range(4)]
    assert P6.robust_pose(Ts, P6.PLUG, 0.003, np.radians(5.0))[1] == 4
    assert P6.robust_pose(Ts, P6.RJ45, 0.003, np.radians(5.0))[1] == 3


class _Sender:
    def __init__(self):
        self.overlays = []

    def send_overlay(self, name, items, text=()):
        self.overlays.append((name, items, list(text)))

    def close(self):
        pass


def test_the_tracker_follows_the_connector_into_the_hand_and_measures_it(standin, tmp_path, monkeypatch):
    """Lying: estimated in base. Gripped: carried in tool0, the estimates from the camera's own
    frame wherever the arm goes. Before the stand-off: the belief from the ones that agree, within
    the gates. Let go: over."""
    from types import SimpleNamespace

    from cable_assembly.skills.pose6d_pick import OVERLAY_CONNECTOR, ConnectorTracker
    from cable_assembly.transforms import inverse
    T_tool0_cam, T_tool0_tip = _T(_Rz(90), [0.0, -0.08, 0.05]), _T(t=[0.0, 0.0, 0.25])
    middle = TRUTH_PLUG[:3, :3] @ P6.CENTRE_M[P6.PLUG] + TRUTH_PLUG[:3, 3]
    A = _T(np.diag([1.0, -1.0, -1.0]), middle) @ inverse(T_tool0_tip)          # the fingertip on it, pointing down
    T_cam_plug = inverse(A @ T_tool0_cam) @ TRUTH_PLUG                         # in the camera: moves with it once held
    truth = tmp_path / 'truth_cam.json'
    truth.write_text(json.dumps({'cord_end_pos_z': T_cam_plug.tolist(), 'adapter_mount': np.eye(4).tolist()}))
    monkeypatch.setenv('POSE6D_TRUTH_CAM', str(truth))
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d', 'pose6d.in_hand.enabled=true', *standin])
    sender = _Sender()
    robot = SimpleNamespace(T_tool0_cam=T_tool0_cam, T_tool0_fingertip=T_tool0_tip, live_frames=sender, arm=None)
    tr = ConnectorTracker(cfg, robot, None)

    def frame_at(tool):
        return _frame(tool @ T_tool0_cam)

    tr.begin(T_base_plug=TRUTH_PLUG @ _T(t=[0.0, 0.002, 0.0]))                # the scan's, 2 mm off
    assert tr.step(frame_at(A))
    np.testing.assert_allclose(tr.T_base_plug[:3, 3], TRUTH_PLUG[:3, 3], atol=1e-3)
    assert not tr.hold(A, 120) and tr.T_tool0_plug is None                    # not closed yet
    assert tr.hold(A, 255) and tr.T_tool0_plug is not None                    # closed on it: held
    B = _T(_Rx(30) @ _Rz(-40), [0.45, -0.1, 0.30])                            # lifted and turned
    for _ in range(3):
        assert tr.step(frame_at(B))
    T, n_agree, n = tr.in_hand()
    T_tool0_plug_true = inverse(A) @ TRUTH_PLUG
    assert n_agree == n == 3
    np.testing.assert_allclose(T[:3, 3], T_tool0_plug_true[:3, 3], atol=1e-3)
    tr.draw(B)                                                                 # drawn where the gripper is
    name, items, text = sender.overlays[-1]
    assert name == OVERLAY_CONNECTOR and 'in the gripper' in text[0] and '3 of 3' in text[0]
    np.testing.assert_allclose(np.mean(items[0]['points'], axis=0),
                               (B @ T_tool0_plug_true)[:3, :3] @ P6.CENTRE_M[P6.PLUG] + (B @ T_tool0_plug_true)[:3, 3],
                               atol=1e-3)
    # The belief: the pick's was 5 mm and 3 deg off; the measured one is the truth's axis and place.
    T_jc = _T(t=[0.021, 0.0, 0.0])
    true_belief = P6.belief_from_plug(inverse(T_tool0_tip) @ T_tool0_plug_true, T_jc,
                                      inverse(T_tool0_tip) @ T_tool0_plug_true @ P6.T_cord_junction() @ T_jc)
    picked = true_belief @ _T(_Rx(3) @ _Rz(3), [0.0, 0.005, 0.0])
    tr.jaw_constraint = False            # the WHOLE measured pose here (5 mm of this error is along the closing axis,
    got = tr.measured_belief(picked, T_tool0_tip, T_jc, since=0.0)     # which the jaws' constraint keeps: test_online_poses)
    np.testing.assert_allclose(got[:3, 3], true_belief[:3, 3], atol=1e-3)
    assert np.degrees(np.arccos(np.clip(got[:3, 0] @ true_belief[:3, 0], -1, 1))) < 0.5
    tr.max_corr_m = 0.001                                                      # over the gate: the pick's stays
    assert tr.measured_belief(picked, T_tool0_tip, T_jc, since=0.0) is picked
    tr.max_corr_m, tr.min_estimates, tr.wait_s = 0.015, 5, 0.0                 # too few agree: the pick's stays
    assert tr.measured_belief(picked, T_tool0_tip, T_jc, since=0.0) is picked
    assert not tr.hold(B, 0) and tr.T_tool0_plug is None                       # let go: lying where it was
    np.testing.assert_allclose(tr.T_base_plug, B @ T, atol=1e-12)
    assert not tr.hold(_T(t=[0.1, 0.0, 0.0]) @ B, 255) and tr.T_tool0_plug is None   # closed away from it: not held
    assert not tr.hold(B, 0) and tr.hold(B, 255)                                       # closed on it again: held


def test_the_tracker_runs_on_its_own_and_follows_the_gripper(standin, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cable_assembly.skills.pose6d_pick import ConnectorTracker
    from cable_assembly.transforms import inverse
    T_tool0_cam, T_tool0_tip = _T(t=[0.0, -0.08, 0.05]), _T(t=[0.0, 0.0, 0.25])
    middle = TRUTH_PLUG[:3, :3] @ P6.CENTRE_M[P6.PLUG] + TRUTH_PLUG[:3, 3]
    A = _T(np.diag([1.0, -1.0, -1.0]), middle) @ inverse(T_tool0_tip)
    truth = tmp_path / 'truth_cam.json'
    truth.write_text(json.dumps({'cord_end_pos_z': (inverse(A @ T_tool0_cam) @ TRUTH_PLUG).tolist(),
                                 'adapter_mount': np.eye(4).tolist()}))
    monkeypatch.setenv('POSE6D_TRUTH_CAM', str(truth))
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d', 'pose6d.in_hand.enabled=true',
                                  'pose6d.in_hand.overlay_hz=50', 'pose6d.in_hand.gap_s=0', *standin])
    tool = {'T': A}
    arm = SimpleNamespace(q=lambda: np.zeros(7), fk=lambda q: tool['T'])
    gripper = SimpleNamespace(last_counts=0)
    camera = SimpleNamespace(capture=lambda **_: _frame(tool['T'] @ T_tool0_cam))
    sender = _Sender()
    robot = SimpleNamespace(T_tool0_cam=T_tool0_cam, T_tool0_fingertip=T_tool0_tip, live_frames=sender,
                            arm=arm, gripper=gripper)
    tr = ConnectorTracker(cfg, robot, camera)
    P6.service(cfg)                                                            # up before the clock starts

    def wait_for(cond, s=20.0):
        t_end = time.monotonic() + s
        while not cond() and time.monotonic() < t_end:
            time.sleep(0.01)
        return cond()

    tr.start(T_base_plug=TRUTH_PLUG)
    try:
        assert wait_for(lambda: tr.stats['accepted'] >= 2)                     # lying: estimated, drawn
        assert any('lying' in t[0] for _, _, t in sender.overlays)
        gripper.last_counts = 255                                              # closed on it
        assert wait_for(lambda: tr.T_tool0_plug is not None)
        tool['T'] = _T(_Rx(20), [0.4, 0.0, 0.3]) @ inverse(T_tool0_tip)        # lifted
        assert wait_for(lambda: (tr.in_hand() or (0, 0, 0))[2] >= 2)
        np.testing.assert_allclose(tr.in_hand()[0][:3, 3], (inverse(A) @ TRUTH_PLUG)[:3, 3], atol=1e-3)
        T_tool0_plug = tr.T_tool0_plug
        gripper.last_counts = 0                                                # let go: lying there, still tracked
        assert wait_for(lambda: tr.T_tool0_plug is None)
        np.testing.assert_allclose(tr.T_base_plug, tool['T'] @ T_tool0_plug, atol=1e-12)
        assert tr.running()
    finally:
        tr.stop()
    assert not tr.running()


def test_the_socket_its_mount_and_its_frame_are_drawn(standin):
    from types import SimpleNamespace

    from cable_assembly.skills.pose6d_pick import OVERLAY_SOCKET, draw_socket
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'pose_method=pose6d', *standin])
    sender = _Sender()
    reply = {'T_base_obj': TRUTH_SOCKET, 'n_used': 5, 'score': 2.3, 'terms': {'iou': 0.8},
             'views': [{'found': True, 'T_base_obj': TRUTH_SOCKET, 'consistent': True}, {'found': False}]}
    draw_socket(cfg, SimpleNamespace(live_frames=sender), reply, np.eye(4))
    name, items, text = sender.overlays[-1]
    # only the socket (its mesh, or its points without the CAD file), its mount where there is a model
    # of it, and its frame at the estimate -- no plug seated in it
    kinds = [i['type'] for i in items]
    assert name == OVERLAY_SOCKET and kinds[0] in ('mesh', 'points') and kinds[-1] == 'pose'
    assert kinds[1:-1] in ([], ['mesh'])
    np.testing.assert_allclose(items[-1]['T'], TRUTH_SOCKET)
    for i in items[:-1]:
        if i['type'] == 'mesh':
            np.testing.assert_allclose(i['T'], TRUTH_SOCKET)        # the mount at the socket's pose
    assert '5 views' in text[0]


# ---------------------------------------------------------------------------- the pick

class _BenchCamera:
    def __init__(self, robot):
        self.robot, self.listeners = robot, []

    def capture(self, **_):
        return _frame(self.robot.camera())


@pytest.fixture
def dry_robot(tmp_path, monkeypatch):
    from cable_assembly.robot import Robot
    monkeypatch.setattr(builtins, 'input', lambda prompt='': '')
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'live_view.enabled=false',
                                  'pick_joints_deg=[-27, 50, 66, -96, 120, -57, -123]',   # an orbit reaches 5 of 6
                                  f'data_dir={tmp_path}'])
    robot = Robot(cfg, with_gripper=False, needs=())
    assert robot.arm.move_j(np.radians(np.asarray(cfg.get('pick_joints_deg'), dtype=float)), label='pick pose')
    yield robot
    robot.close()


def test_the_pose6d_scanner_finds_the_connector_from_an_orbit_and_never_loads_sam3(standin, dry_robot, tmp_path):
    from cable_assembly.apps._cable import build_scanner
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'pose_method=pose6d',
                                  'pose6d.orbit.settle_s=0', 'pose6d.close_view.enabled=false',
                                  f'data_dir={tmp_path}', *standin])
    from cable_assembly.perception import sam3
    called = []
    original = sam3.make_detector
    sam3.make_detector = lambda c: called.append(c)                            # would load SAM3
    try:
        scanner, detector, _ = build_scanner(cfg, dry_robot, _BenchCamera(dry_robot))
    finally:
        sam3.make_detector = original
    assert detector is None and not called and scanner.detector is None
    q0 = np.asarray(dry_robot.arm.q())
    dry_robot.live_frames = sender = _Sender()
    T = scanner.scan()
    np.testing.assert_allclose(T, P6.junction_from_plug(TRUTH_PLUG), atol=1e-12)
    name, items, _ = sender.overlays[-1]                                       # drawn in the live view
    assert name == 'pose6d_connector' and items[0]['type'] == 'points' and items[-1]['type'] == 'pose'
    np.testing.assert_allclose(np.mean(items[0]['points'], axis=0),
                               TRUTH_PLUG[:3, :3] @ P6.CENTRE_M[P6.PLUG] + TRUTH_PLUG[:3, 3], atol=1e-3)
    assert scanner.pose_on_axis and scanner.last['n_used'] > 3                # the first view + the orbit's
    assert scanner.scene is not None
    np.testing.assert_allclose(dry_robot.arm.q(), q0, atol=1e-6)              # back where it looked from


def test_not_found_in_the_first_view_is_no_pose(standin, dry_robot, tmp_path, monkeypatch):
    from cable_assembly.skills.pose6d_pick import Pose6DScanner
    monkeypatch.setenv('POSE6D_MISS_FIRST', '1')
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'pose_method=pose6d',
                                  'pose6d.fallback.enabled=false', f'data_dir={tmp_path}', *standin])
    q0 = np.asarray(dry_robot.arm.q())
    assert Pose6DScanner(dry_robot, _BenchCamera(dry_robot), None, cfg).scan() is None
    np.testing.assert_allclose(dry_robot.arm.q(), q0, atol=1e-12)             # it did not orbit a guess


def test_a_recorded_pose_seeds_the_first_view_and_the_views_are_kept(standin, dry_robot, tmp_path, monkeypatch):
    """With a prior the first view refines from it -- found where a search of the image is not --
    and the views and replies land under <data_dir>/pose6d for a look without the robot."""
    import glob

    from cable_assembly.skills.pose6d_pick import estimate_part
    monkeypatch.setenv('POSE6D_MISS_FIRST', '1')                                # no prior: nothing found
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'pose6d.orbit.settle_s=0',
                                  f'data_dir={tmp_path}', *standin])
    prior = TRUTH_SOCKET @ _T(t=[0.003, -0.002, 0.0])
    r, views = estimate_part(cfg, dry_robot, _BenchCamera(dry_robot), P6.SOCKET, label='6D pose: the socket',
                             prior=prior)
    assert r is not None and r['found'] and len(views) > 3
    np.testing.assert_allclose(r['T_base_obj'], TRUTH_SOCKET)
    kept = glob.glob(str(tmp_path / 'pose6d' / '*_6d_pose_the_socket'))
    assert len(kept) == 1
    with open(os.path.join(kept[0], 'replies.json')) as f:
        rec = json.load(f)
    assert [x['what'] for x in rec['replies']] == ['first view, from the prior', 'all views']
    np.testing.assert_allclose(rec['prior'], prior)
    z = np.load(os.path.join(kept[0], 'view_00.npz'))
    assert z['rgb'].shape == (90, 160, 3) and z['depth_m'].shape == (90, 160) and z['T_base_cam'].shape == (4, 4)


def test_not_found_in_the_first_view_the_orbit_still_runs_about_the_recorded_pose(standin, dry_robot, tmp_path,
                                                                                   monkeypatch):
    from cable_assembly.skills.pose6d_pick import estimate_part
    monkeypatch.setenv('POSE6D_MISS_SINGLE', '1')                               # one view never finds it
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'pose6d.orbit.settle_s=0',
                                  f'data_dir={tmp_path}', *standin])
    q0 = np.asarray(dry_robot.arm.q())
    r, views = estimate_part(cfg, dry_robot, _BenchCamera(dry_robot), P6.SOCKET, prior=TRUTH_SOCKET)
    assert r is not None and r['found'] and r['n_used'] > 1
    np.testing.assert_allclose(dry_robot.arm.q(), q0, atol=1e-6)               # back where it looked from
    # Without a prior, the same miss is no pose -- and no orbit about a guess.
    r, views = estimate_part(cfg, dry_robot, _BenchCamera(dry_robot), P6.SOCKET)
    assert r is None and len(views) == 1



def test_saved_socket_views_come_back_as_they_were_kept(tmp_path):
    """assembly.standoff.socket_views saved: the newest views save_views kept are read back as the
    (joints, Frame) the scene is built from -- colour, depth, K, camera pose and joints unchanged --
    and nothing kept reads as nothing."""
    from cable_assembly.config import Config
    from cable_assembly.perception.camera import Frame
    from cable_assembly.skills.pose6d_pick import load_saved_views, save_views
    cfg = Config({'data_dir': str(tmp_path)})
    assert load_saved_views(cfg, 'scene: the socket') == (None, [])
    rng = np.random.default_rng(0)
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.1, 0.4]
    views = [(rng.normal(size=7), Frame(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), np.eye(3), None, 0.0, T,
                                        rng.random((6, 8)).astype(np.float32))) for _ in range(2)]
    save_views(cfg, 'scene: the socket', views, [])
    path, got = load_saved_views(cfg, 'scene: the socket')
    assert path is not None and len(got) == 2
    for (q, f), (q2, f2) in zip(views, got):
        np.testing.assert_allclose(q2, q)
        np.testing.assert_array_equal(f2.color, f.color)
        np.testing.assert_allclose(f2.depth, f.depth)
        np.testing.assert_allclose(f2.T_base_cam, T)


def test_a_view_waits_only_until_the_arm_is_still_and_is_retaken_if_the_image_moved():
    """No fixed settle (the user, 2026-11-03): a view is taken once the joints have stopped (not
    before, not the old 0.5 s every time), and one whose image moved across its frames -- the
    camera still ringing, i.e. blur -- is taken again."""
    import time as _t
    from cable_assembly.perception.camera import Frame
    from cable_assembly.skills import pose6d_pick as PP

    class Arm:
        def __init__(self, stop_s):
            self.t0, self.stop = _t.monotonic(), stop_s

        def q(self):
            return np.full(7, 0.01 * min(_t.monotonic() - self.t0, self.stop))

    t = PP.wait_still(Arm(0.15), 0.5)
    assert 0.15 <= t < 0.3
    assert PP.wait_still(Arm(0.0), 0.5) < 0.12

    rng = np.random.default_rng(0)
    img = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)

    class Camera:                            # the first view's frames drift 3 px; the retake's do not
        def __init__(self):
            self.n = 0

        def capture(self, **_):
            self.n += 1
            shift = 3 * (self.n - 1) if self.n <= 3 else 0
            return Frame(np.roll(img, shift, axis=1), np.eye(3), None, 0.0, np.eye(4), np.full((720, 1280), 0.5, np.float32))

    cam = Camera()
    got = PP.still_view(cam, Arm(0.0), 3, settle_s=0.2)
    assert got is not None and cam.n == 6, 'the drifting view was taken again'


def test_the_refine_orbit_takes_the_most_spread_of_the_reachable_views():
    from cable_assembly.skills.pose6d_pick import _spread
    from cable_assembly.transforms import translation_matrix
    ring = [(f'pan {k}', translation_matrix([0.1 * np.cos(a), 0.1 * np.sin(a), 0.3]))
            for k, a in enumerate(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))]
    got = [int(n.split()[1]) for n, _ in _spread(ring, np.eye(4), [0.0, 0.0, 0.3], 3)]
    gaps = sorted((b - a) % 8 for a, b in zip(got, got[1:] + got[:1]))
    assert gaps == [2, 3, 3]                                  # as near a third of the ring apart as 8 allow
    assert len(_spread(ring[:2], np.eye(4), [0.0, 0.0, 0.3], 3)) == 2   # fewer reachable: all of them


@pytest.mark.parametrize('side', [-1.0, 1.0])
def test_the_socket_view_turns_until_the_tool_no_longer_hides_the_socket(monkeypatch, side):
    # The refine view in miniature: the camera at tool0, looking along +z; the held plug a column 0.2 m out from
    # one edge of the image (side -1: the TOP, as on the D405's own mount since 2026-11-12; +1: the bottom, as on
    # the D415's) to its middle; the socket a disc 0.32 m out just past the middle -- its near edge behind the
    # plug's tip. The camera turns AWAY from the plug, whichever edge it comes from.
    from types import SimpleNamespace
    from cable_assembly.apps import splat_scan
    from cable_assembly.robot import tool_geometry as G
    from cable_assembly.skills import pose6d_pick as PP
    K = np.array([[930.0, 0.0, 640.0], [0.0, 930.0, 360.0], [0.0, 0.0, 1.0]])
    size = (1280, 720)
    plug = np.array([[x, side * y, 0.2] for x in (-0.006, 0.006) for y in np.linspace(0.0, 0.08, 9)])
    th = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    socket = np.column_stack([0.02 * np.cos(th), 0.02 * np.sin(th) + side * 0.01, np.zeros_like(th)])
    T_socket = np.eye(4)
    T_socket[2, 3] = 0.32
    arm = SimpleNamespace(q=lambda: np.zeros(7), fk=lambda q: np.eye(4))
    robot = SimpleNamespace(arm=arm, T_tool0_cam=np.eye(4), T_tool0_fingertip=np.eye(4), held=object())
    monkeypatch.setattr(PP, 'held_points', lambda held: plug)
    monkeypatch.setattr(G, 'gripper_spheres', lambda *a, **k: [])          # the plug alone hides it here
    monkeypatch.setattr(G, 'gripper_view_spheres', lambda *a, **k: [])
    monkeypatch.setattr(splat_scan, 'check_plan', lambda *a, **k: [('aim', np.eye(4))])
    ac = {'margin_px': 15.0, 'edge_px': 30.0, 'max_deg': 25.0, 'min_visible': 0.95, 'bench_z': 0.0,
          'clearance_m': 0.06}
    S = socket + T_socket[:3, 3]
    mask = PP.tool_mask(robot, K, size)
    vis, inside = PP.view_of(K, size, mask, np.eye(4), S, 15.0, 30.0)
    assert vis < 0.95 and inside                                              # hidden from here
    a, T = PP.aim_past_held(robot, K, T_socket, socket, ac, size=size)
    assert a is not None and 1.0 < np.degrees(abs(a)) < 12.0                  # a few degrees do it
    vis, inside = PP.view_of(K, size, mask, T, S, 15.0, 30.0)
    assert vis >= 0.95 and inside
    tip = PP._px(K, np.eye(4), plug[np.abs(plug[:, 1]) < 1e-9]).mean(axis=0)   # the plug's tip, fixed in the image
    d0 = np.linalg.norm(PP._px(K, np.eye(4), S).mean(axis=0) - tip)
    d1 = np.linalg.norm(PP._px(K, np.linalg.inv(T), S).mean(axis=0) - tip)
    assert d1 > d0 + 20.0                                                     # moved away from the plug
    # Already clear: no move. Nothing reachable: none.
    T_far = T_socket.copy()
    T_far[1, 3] = -side * 0.08
    assert PP.aim_past_held(robot, K, T_far, socket, ac, size=size) == (0.0, None)
    monkeypatch.setattr(splat_scan, 'check_plan', lambda *a, **k: None)
    assert PP.aim_past_held(robot, K, T_socket, socket, ac, size=size) == (None, None)


def test_the_close_view_servos_on_its_estimate_and_keeps_the_scan_when_it_strays(monkeypatch):
    # The camera is aimed at the scan's estimate, looks, is re-aimed at what it saw, and stops once the
    # aim no longer moves; a view that puts the plug far from the scan is not trusted.
    from types import SimpleNamespace
    from cable_assembly.apps import splat_scan
    from cable_assembly.skills import pose6d_pick as PP
    from cable_assembly.transforms import translation_matrix
    state = {'T': translation_matrix([0.0, 0.0, 0.5])}
    arm = SimpleNamespace(q=lambda: np.zeros(7), fk=lambda q: state['T'],
                          move_l=lambda T, **k: state.update(T=np.asarray(T)) or True)
    robot = SimpleNamespace(arm=arm, T_tool0_cam=np.eye(4), T_tool0_fingertip=np.eye(4))
    monkeypatch.setattr(splat_scan, 'check_plan', lambda *a, **k: [('close view', np.eye(4))])
    monkeypatch.setattr(PP, 'still_view', lambda *a, **k: (np.zeros(7), object()))
    monkeypatch.setattr(PP, 'save_views', lambda *a, **k: None)
    seen = [translation_matrix([0.004, 0.0, 0.0])] * 5
    looks = []

    def estimate(part, frames, prior=None):
        looks.append(prior)
        return {'found': True, 'T_base_obj': seen[len(looks) - 1], 'score': 2.5,
                'terms': {'iou': 0.9, 'edge_fwd': 0.7, 'edge_cover': 1.0}}
    monkeypatch.setattr(P6, 'service', lambda cfg: SimpleNamespace(estimate=estimate, request={}))
    sc = {'distance_m': 0.30, 'max_distance_m': 0.30, 'servo_iters': 3, 'tol_m': 0.003, 'tol_rad': np.radians(3.0),
          'max_shift_m': 0.015, 'max_shift_rad': np.radians(20.0), 'caps': None, 'frames': 1, 'settle_s': 0.0,
          'bench_z': 0.0, 'clearance_m': 0.06}
    r, views = PP.servo_close_view(None, robot, None, P6.PLUG, np.eye(4), sc, 'test')
    assert r is not None and np.allclose(r['T_base_obj'], seen[0])
    assert len(looks) == 2                                   # re-aimed once, then the aim stood still
    cam = state['T'][:3, 3]
    centre = seen[0][:3, :3] @ P6.CENTRE_M[P6.PLUG] + seen[0][:3, 3]
    assert abs(np.linalg.norm(cam - centre) - 0.30) < 1e-6  # 0.30 m from the plug's middle, aimed at it
    assert np.allclose(state['T'][:3, 2], (centre - cam) / 0.30)
    # Strayed: 40 mm from the scan -- not trusted, the caller keeps the scan's.
    looks.clear()
    seen[:] = [translation_matrix([0.04, 0.0, 0.0])] * 5
    r, _ = PP.servo_close_view(None, robot, None, P6.PLUG, np.eye(4), sc, 'test')
    assert r is None


def test_the_scan_ends_with_a_close_look_at_the_connector(standin, dry_robot, tmp_path):
    # pose6d.close_view on (the bench's): after the scan the camera stands 0.30 m from the plug's
    # middle, aimed at it, and the pose given the pick is the close view's.
    from cable_assembly.apps._cable import build_scanner
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'pose_method=pose6d',
                                  'pose6d.orbit.settle_s=0', f'data_dir={tmp_path}', *standin])
    scanner, _, _ = build_scanner(cfg, dry_robot, _BenchCamera(dry_robot))
    q_scan = np.asarray(dry_robot.arm.q())
    seen = []
    move_l = dry_robot.arm.move_l
    dry_robot.arm.move_l = lambda T, **k: (seen.append(np.asarray(T)), move_l(T, **k))[1]
    T = scanner.scan()
    np.testing.assert_allclose(T, P6.junction_from_plug(TRUTH_PLUG), atol=1e-6)
    centre = TRUTH_PLUG[:3, :3] @ P6.CENTRE_M[P6.PLUG] + TRUTH_PLUG[:3, 3]
    closest = min(seen, key=lambda T_: abs(np.linalg.norm((T_ @ dry_robot.T_tool0_cam)[:3, 3] - centre) - 0.30))
    cam = closest @ dry_robot.T_tool0_cam
    assert abs(np.linalg.norm(cam[:3, 3] - centre) - 0.30) < 0.002          # it looked from 0.30 m...
    assert np.degrees(np.arccos(cam[:3, 2] @ (centre - cam[:3, 3]) / np.linalg.norm(centre - cam[:3, 3]))) < 0.5
    assert any('connector_close' in d for d in os.listdir(os.path.join(tmp_path, 'pose6d')))
    # ...and came back to the scan pose: the pick's move to the pre-grasp starts there (from the close
    # view it dived the fingertip to 30 mm over the bench, 2026-11-04)
    np.testing.assert_allclose(dry_robot.arm.fk(dry_robot.arm.q()), dry_robot.arm.fk(q_scan), atol=1e-6)


def test_the_scene_scan_looks_round_the_socket_view_and_comes_back(dry_robot, tmp_path):
    """assembly.visual_target.scene_scan (the user, 2026-11-13: "a small scan at initial socket view to increase the
    range and accuracy of the scene"): from the cell's socket view, a ring about the socket and looks about the
    camera's own centre -- each a still view, the arm back where it looked from; off unless the cell says so."""
    from cable_assembly.skills import pose6d_pick as PP
    from cable_assembly.transforms import inverse
    cfg = C.load('usb_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true', 'pose6d.orbit.settle_s=0',
                                  f'data_dir={tmp_path}'])
    vt = cfg.get_path('assembly.visual_target')
    sc = PP.scan_cfg(cfg, vt['scene_scan'])
    assert sc['enabled'] and sc['looks'] == 4 and np.degrees(sc['look']) == pytest.approx(20.0)
    assert not PP.scan_cfg(cfg, None)['enabled'] and not PP.scan_cfg(C.load('bnc_assembly', []), None)['enabled']
    # a look turns the camera about its own centre: it stays where it was, its axis look_deg off
    Tc = dry_robot.T_tool0_cam
    arm = dry_robot.arm
    assert arm.move_j(np.radians(np.asarray(vt['view_joints_deg'], dtype=float)), label='socket view')
    q0 = np.asarray(arm.q())
    T0 = arm.fk(q0)
    looks = PP.look_views(T0, Tc, sc['look'], 4)
    for _name, T in looks:
        cam = T @ Tc
        np.testing.assert_allclose(cam[:3, 3], (T0 @ Tc)[:3, 3], atol=1e-9)
        assert np.degrees(np.arccos(cam[:3, 2] @ (T0 @ Tc)[:3, 2])) == pytest.approx(20.0)
    # the scan from the view pose: most of its 8 views reachable, taken, and the arm back
    seen = []
    move_l = arm.move_l
    arm.move_l = lambda T, **k: (seen.append(np.asarray(T)), move_l(T, **k))[1]
    pivot = np.array([0.804, 0.224, 0.150])                    # the USB socket on the bench's wall
    views = PP.scene_scan(dry_robot, _BenchCamera(dry_robot), pivot, sc, label='scene scan')
    assert 6 <= len(views) <= 8 and len(seen) == len(views) + 1
    np.testing.assert_allclose(arm.fk(arm.q()), T0, atol=1e-6)
    cams = np.array([np.asarray(f.T_base_cam)[:3, 3] for _q, f in views])
    moved = np.linalg.norm(cams - (T0 @ Tc)[:3, 3], axis=1)
    assert (moved > 0.05).sum() >= 2 and (moved < 1e-6).sum() >= 3      # ring views a baseline apart; looks in place
    for T in seen[:-1]:                                                  # a ring view still looks at the socket
        cam = T @ Tc
        if np.linalg.norm(cam[:3, 3] - (T0 @ Tc)[:3, 3]) > 0.05:
            to = inverse(cam) @ np.append(pivot, 1.0)
            to0 = inverse(T0 @ Tc) @ np.append(pivot, 1.0)
            np.testing.assert_allclose(to[:3], to0[:3], atol=1e-6)
    # off: nothing moves
    seen.clear()
    assert PP.scene_scan(dry_robot, _BenchCamera(dry_robot), pivot, PP.scan_cfg(cfg, {'enabled': False})) == []
    assert not seen
