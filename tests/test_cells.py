"""Cells: the bench a task runs on (configs/cells/, overlaid by cable_assembly/config.py).

The point of the layer is that the BNC task names no robot. These tests hold the task file to
that, hold every cell to the model it names, and pin the overlay rules: per key, cell wins,
every override logged, an uncalibrated cell never moves hardware.
"""
import glob
import os

import numpy as np
import pytest
import yaml

from cable_assembly import config as C
from cable_assembly.robot import kinematics

CELLS = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(C.CONFIG_DIR, 'cells', '*.yaml'))
               if not p.endswith('.frames.yaml'))
# Keys that are facts of a BENCH -- the robot, the tool on it, where things are -- never of a task.
BENCH_FACTS = ('robot', 'hand_eye', 'grasp_tcp_offset', 'fingertip_grasp', 'frames_file',
               'calibrated', 'camera.serial_no', 'camera.width', 'camera.height', 'camera.fps',
               'ground_plane.z_mm')


def _raw(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _leaves(d, prefix=''):
    for k, v in d.items():
        p = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict) and v:
            yield from _leaves(v, p)
        else:
            yield p, v


def test_both_benches_exist():
    assert {'ur10e_bench', 'iiwa14_bench'} <= set(CELLS)


def test_the_bnc_task_names_no_robot():
    task = _raw(os.path.join(C.CONFIG_DIR, 'bnc_assembly.yaml'))
    leaves = dict(_leaves(task))
    for key in BENCH_FACTS:
        assert key not in leaves and not any(k.startswith(key + '.') for k in leaves), \
            f'{key} is a bench fact -- it belongs in configs/cells/, not the task'
    joint_space = [k for k in leaves if k.endswith('joints_deg') or k.endswith('joint_limits_deg')]
    assert not joint_space, f'joint-space values are per robot, so per cell: {joint_space}'
    assert task.get('cell') == 'ur10e_bench', 'the task still runs on the UR by default'


@pytest.mark.parametrize('cell', CELLS)
def test_every_cell_is_complete_and_matches_its_robot(cell):
    from cable_assembly.robot.robot import CALIBRATION_ITEMS

    raw = _raw(C.cell_path({'_config_dir': C.CONFIG_DIR}, cell))
    c = raw.get('calibrated')
    assert isinstance(c, bool) or (isinstance(c, list) and set(c) <= set(CALIBRATION_ITEMS)), \
        'a cell states what is calibrated: true, false, or a list of CALIBRATION_ITEMS'
    model = kinematics.get(raw['robot']['type'])
    for key in ('hand_eye', 'fingertip_grasp', 'camera', 'ground_plane', 'frames_file'):
        assert key in raw, f'{cell} must define {key}'
    assert os.path.isfile(os.path.join(C.CONFIG_DIR, raw['frames_file']))
    for path, value in _leaves(raw):
        if path.endswith('joints_deg') and value is not None:
            assert len(value) == model.DOF, f'{cell} {path}: {model.NAME} has {model.DOF} joints'
            assert model.within(np.radians(value)), f'{cell} {path} is outside the joint limits'


@pytest.mark.parametrize('cell', CELLS)
def test_the_bnc_config_resolves_onto_each_cell(cell):
    cfg = C.load('bnc_assembly', [f'cell={cell}'])
    raw = _raw(C.cell_path(cfg, cell))
    assert cfg['cell'] == cell and cfg.get_path('robot.type') == raw['robot']['type']
    assert cfg.get_path('reset.home_joints_deg') == raw['reset']['home_joints_deg']
    # The overlay is PER KEY: the cell set reset.home_joints_deg, the task's reset tuning survives.
    assert cfg.get_path('reset.max_force_n') is not None
    assert cfg.get_path('ground_plane.max_cables') is not None
    assert cfg.get_path('ground_plane.z_m') is not None, 'units are normalised after the overlay'


def test_the_ur_cell_is_a_pure_split():
    """The UR bench's facts were MOVED out of the task, so its cell re-tunes nothing."""
    assert C.load('bnc_assembly')['_cell_overrides'] == []


def test_a_cell_value_wins_and_says_so(tmp_path, caplog):
    (tmp_path / 'cells').mkdir()
    (tmp_path / 'task.yaml').write_text(
        'cell: bench\nscan:\n  standoff_mm: 100.0\n  dwell_s: 2.0\n')
    (tmp_path / 'cells' / 'bench.yaml').write_text(
        'robot: {type: iiwa14}\nscan:\n  standoff_mm: 150.0\n')
    with caplog.at_level('INFO'):
        cfg = C.load(str(tmp_path / 'task.yaml'))
    assert cfg.get_path('scan.standoff_m') == pytest.approx(0.150)
    assert cfg.get_path('scan.dwell_s') == 2.0
    assert cfg['_cell_overrides'] == [('scan.standoff_mm', 100.0, 150.0)]
    assert any('overrides scan.standoff_mm' in r.getMessage() for r in caplog.records)


def test_the_cell_can_be_chosen_on_the_command_line():
    args = C.arg_parser('t', 'bnc_assembly').parse_args(['--cell', 'iiwa14_bench'])
    assert C.from_args(args).get_path('robot.type') == 'iiwa14'
    with pytest.raises(FileNotFoundError, match='iiwa14_bench'):
        C.load('bnc_assembly', ['cell=no_such_bench'])     # the error lists what exists


def test_an_uncalibrated_cell_never_moves_hardware():
    from cable_assembly.robot import Robot, SimArm

    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench'])
    assert cfg.get('calibrated') is not True, 'the iiwa bench is still being measured'
    with pytest.raises(RuntimeError, match='not calibrated'):
        Robot(cfg, with_gripper=False)

    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'robot.dry_run=true'])
    robot = Robot(cfg, with_gripper=False)
    try:
        assert isinstance(robot.arm, SimArm) and robot.arm.model.NAME == 'iiwa14'
        assert len(robot.arm.q()) == 7 and robot.arm.joint_names[6] == 'A7'
        assert robot.tool0()[2, 2] < -0.99, 'the iiwa starts with the tool pointing down'
        assert robot.arm.move_j(np.radians(cfg.get_path('reset.home_joints_deg')), label='home')
    finally:
        robot.close()


def test_a_cell_is_calibrated_in_stages():
    """A list under calibrated: lets the apps that need only those items run -- the pickup test
    before the socket exists -- and still stops the ones that need the rest."""
    from cable_assembly.apps.bnc_pickup import NEEDS
    from cable_assembly.robot import Robot
    from cable_assembly.robot.robot import missing_calibration

    staged = 'calibrated=[hand_eye, fingertip, ground_plane, poses]'
    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', staged])
    assert missing_calibration(cfg, NEEDS) == []
    assert missing_calibration(cfg) == ['target']
    with pytest.raises(RuntimeError, match='target'):
        Robot(cfg, with_gripper=False)              # the full BNC run: refused before connecting

    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'calibrated=[hand_eye, fingertip]'])
    assert missing_calibration(cfg, NEEDS) == ['ground_plane', 'poses']

    cfg = C.load('bnc_assembly', ['cell=iiwa14_bench', 'calibrated=[handeye]'])
    with pytest.raises(ValueError, match='handeye'):
        missing_calibration(cfg)                    # a typo is not silently "not calibrated"

    assert missing_calibration(C.load('bnc_assembly')) == [], 'the UR bench: calibrated: true'
