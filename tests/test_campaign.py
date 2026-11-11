"""behaviors/campaign.py + behaviors/board.py: the task as a behavior tree, run N times, recovering what it
safely can and asking a person for the rest -- with fake steps, no robot."""

import builtins
import json
import os
import threading
import time

import numpy as np
import pytest

from cable_assembly.behaviors import campaign as C
from cable_assembly.behaviors.board import REQUEST_PAUSE, REQUEST_STOP, Board


def answers(monkeypatch, *replies):
    """The terminal: each input() the next reply; EOF after the last."""
    it = iter(replies)

    def fake(*_a, **_k):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None
    monkeypatch.setattr(builtins, 'input', fake)


class Cell:
    """A cell of fake steps: `script[(trial, attempt, step)]` makes that step fail (a StepFailed's args)."""

    def __init__(self, tmp_path, n=3, script=None, **kw):
        self.board = Board(str(tmp_path), mode='autonomous', tags=['DRY RUN'])
        self.stops = []
        self.camp = C.Campaign(str(tmp_path), n, self.board, safe_stop=lambda: self.stops.append(1), **kw)
        self.script = dict(script or {})
        self.ran = []
        c = self.camp

        def step(name, cell_after):
            def fn():
                t = c.trial
                self.ran.append((t.k, t.attempt, name))
                f = self.script.get((t.k, t.attempt, name))
                if f is not None:
                    if isinstance(f, Exception):
                        raise f
                    raise C.StepFailed(*f[0], **f[1])
                t.set_cell(**cell_after)
                return True
            return C.Step(name, fn, c)

        def rec(name, cell_after, ok=True):
            def fn():
                self.ran.append((c.trial.k, c.trial.attempt, name))
                c.trial.set_cell(**cell_after)
                return ok if ok else c.fail(f'{name} failed')
            return fn

        self.place_ok = True
        assemble = C.sequence('Assemble',
                              step('pick', dict(connector='hand', gripped=True, arm='unknown')),
                              step('insert', dict(connector='socket', assembly='engaged', arm='socket')),
                              step('disassemble', dict(connector='bench', assembly=None, gripped=False, arm='clear')))
        recover = C.selector('Recover', c.operator_branch(),
                             c.recovery('gripped at the socket?',
                                        lambda t: t.cell['connector'] == 'socket' and t.cell['gripped'],
                                        'back out and place', lambda: rec('back out and place', dict(
                                            connector='bench', assembly=None, gripped=False, arm='pick'),
                                            self.place_ok)()),
                             c.recovery('nothing in the hand?', lambda t: t.cell['gripped'] is False,
                                        'go home', rec('go home', dict(arm='home'))),
                             c.operator_branch(fallback=True))
        trial = C.TrialSequence('Trial', C.Step('prepare', lambda: True, c),
                                C.AttemptLoop('Attempts', assemble, recover, c, c.max_attempts),
                                C.Step('end', rec('end', dict(arm='home')), c), c)
        self.root = C.sequence('root', C.Step('start', lambda: True, c), C.Trials('Trials', trial, c))

    def run(self):
        try:
            return self.camp.run(self.root)
        finally:
            self.board.close()

    def rows(self, tmp_path):
        with open(os.path.join(str(tmp_path), 'trials.jsonl')) as fh:
            return [json.loads(line) for line in fh]


MISS = (('the engage missed',), dict(connector='socket', assembly=None, gripped=True, arm='socket'))


def test_every_trial_runs_and_is_recorded(tmp_path, monkeypatch):
    answers(monkeypatch)
    cell = Cell(tmp_path, n=3)
    assert cell.run()
    rows = cell.rows(tmp_path)
    assert [r['trial'] for r in rows] == [1, 2, 3] and all(r['success'] for r in rows)
    assert [r for r in cell.ran if r[2] == 'end'] == [(1, 1, 'end'), (2, 1, 'end'), (3, 1, 'end')]
    for f in ('trials.csv', 'events.jsonl', 'campaign.json', 'bt.png', 'bt_status.json', 'trial_003'):
        assert os.path.exists(os.path.join(str(tmp_path), f)), f
    meta = json.load(open(os.path.join(str(tmp_path), 'campaign.json')))
    assert meta['state'] == 'done' and meta['trials_succeeded'] == 3
    st = json.load(open(os.path.join(str(tmp_path), 'bt_status.json')))
    assert st['autonomy'] == 'done' and st['mode'] == 'autonomous' and st['counts']['assembled'] == 3


def test_a_miss_is_backed_out_and_tried_again(tmp_path, monkeypatch):
    answers(monkeypatch)
    cell = Cell(tmp_path, n=1, script={(1, 1, 'insert'): MISS})
    assert cell.run()
    r = cell.rows(tmp_path)[0]
    assert r['success'] and r['attempts'] == 2
    assert r['recoveries'] == '1:back out and place:ok' and r['n_interventions'] == 0
    assert (1, 1, 'back out and place') in cell.ran and (1, 2, 'insert') in cell.ran


def test_the_same_failure_twice_asks_a_person(tmp_path, monkeypatch):
    answers(monkeypatch, 'n')                          # the operator: count it failed, next trial
    cell = Cell(tmp_path, n=2, script={(1, 1, 'insert'): MISS, (1, 2, 'insert'): MISS})
    assert cell.run()
    rows = cell.rows(tmp_path)
    assert not rows[0]['success'] and rows[0]['interventions'] == 'insert:next'
    assert 'the same failure as attempt 1' in rows[0]['last_failure']
    assert rows[1]['success'], 'the next trial ran'
    assert cell.stops, 'an intervention brakes the arm first'


def test_a_failed_recovery_is_never_followed_by_another(tmp_path, monkeypatch):
    answers(monkeypatch, 'q')
    cell = Cell(tmp_path, n=3, script={(1, 1, 'insert'): MISS})
    cell.place_ok = False
    assert not cell.run()
    assert (1, 1, 'go home') not in cell.ran, 'no second automatic recovery after one failed'
    assert cell.camp.abort and len(cell.camp.trials) == 1
    assert (1, 1, 'end') not in cell.ran, 'stopped by the operator: the arm is left where it is'
    meta = json.load(open(os.path.join(str(tmp_path), 'campaign.json')))
    assert meta['state'] == 'stopped'


def test_an_error_or_a_collision_goes_to_a_person(tmp_path, monkeypatch):
    answers(monkeypatch, 'n', 'n')
    cell = Cell(tmp_path, n=3, script={(1, 1, 'pick'): RuntimeError('boom')})
    orig = cell.camp.fail

    def fail(why, kind='recover', **c):              # trial 2's miss came with a free-space guard trip
        if cell.camp.trial.k == 2 and why == 'the engage missed':
            cell.camp.collision('|F| 31 N > 30 N')
        return orig(why, kind, **c)
    cell.camp.fail = fail
    cell.script[(2, 1, 'insert')] = MISS
    assert cell.run()
    rows = cell.rows(tmp_path)
    assert rows[0]['last_failure_kind'] == 'intervene' and 'RuntimeError: boom' in rows[0]['last_failure']
    assert rows[1]['last_failure_kind'] == 'intervene' and 'force guard tripped' in rows[1]['last_failure']
    assert rows[2]['success']


def test_no_terminal_means_stop(tmp_path, monkeypatch):
    answers(monkeypatch)                               # stdin closed
    cell = Cell(tmp_path, n=5, script={(1, 1, 'pick'): RuntimeError('boom')})
    assert not cell.run()
    assert cell.camp.abort and len(cell.camp.trials) == 1


def test_failed_trials_in_a_row_ask_before_the_next(tmp_path, monkeypatch):
    answers(monkeypatch, 'q')
    script = {(k, a, 'insert'): (((f'miss {k}.{a}',), MISS[1])) for k in (1, 2) for a in (1, 2, 3)}
    cell = Cell(tmp_path, n=5, script=script, max_attempts=3, max_consecutive_failures=2)
    assert not cell.run()
    assert len(cell.camp.trials) == 2, 'asked after the 2nd failed trial, and stopped there'
    assert [r['attempts'] for r in cell.rows(tmp_path)] == [3, 3]


def test_a_stop_request_ends_the_run_after_the_trial(tmp_path, monkeypatch):
    answers(monkeypatch)
    cell = Cell(tmp_path, n=5)
    open(os.path.join(str(tmp_path), REQUEST_STOP), 'w').close()
    assert cell.run()
    assert len(cell.camp.trials) == 1 and cell.camp.trials[0].success
    assert not os.path.exists(os.path.join(str(tmp_path), REQUEST_STOP)), 'taken (withdrawn) once seen'


def test_a_pause_holds_until_withdrawn(tmp_path, monkeypatch):
    answers(monkeypatch)
    cell = Cell(tmp_path, n=1)
    p = os.path.join(str(tmp_path), REQUEST_PAUSE)
    open(p, 'w').close()
    seen = {}

    def withdraw():
        time.sleep(0.6)
        seen['state'] = cell.board.autonomy
        os.remove(p)
    th = threading.Thread(target=withdraw)
    th.start()
    t0 = time.time()
    assert cell.run()
    th.join()
    assert time.time() - t0 >= 0.5 and seen['state'] == 'paused'


def test_the_board_draws_the_tree_with_the_mode_and_state(tmp_path):
    b = Board(str(tmp_path), mode='supervised', tags=['DRY RUN'])
    camp = C.Campaign(str(tmp_path), 2, b)
    guard = camp.recovery('cable in hand?', lambda t: True, 'place it', lambda: True)
    root = C.sequence('root', C.Step('a step', lambda: True, camp), C.Step('off step', lambda: True, camp, enabled=False),
                      C.selector('Recover', guard))
    b.set_tree(root)
    b.set_autonomy('intervention', 'trial 1: the engage missed')
    snap = b.snapshot()
    names = [n['name'] for n in snap['nodes']]
    assert 'cable in hand?  →  place it' in names, 'a guard is one row'
    assert [n['status'] for n in snap['nodes'] if n['name'] == 'off step'] == ['off']
    img = b.render(snap, blink=True)
    assert img.ndim == 3 and img.shape[1] == 1120 and img.shape[0] > 200
    assert b.panel_lines(snap)[0].startswith('SUPERVISED / DRY RUN: INTERVENTION NEEDED')
    b.close()
    assert os.path.exists(os.path.join(str(tmp_path), 'bt.png'))


def test_the_motion_record_switches_to_each_trial(tmp_path):
    from types import SimpleNamespace

    from cable_assembly.robot.trial_recorder import TrialRecorder

    class Arm:
        def q(self):
            return np.zeros(7)

        def fk(self, q):
            return np.eye(4)

        def wrench(self):
            return np.zeros(6)
    robot = SimpleNamespace(arm=Arm(), T_tool0_fingertip=np.eye(4))
    rec = TrialRecorder(robot, str(tmp_path), rate_hz=200.0, console=False).start()
    time.sleep(0.15)
    rec.switch(str(tmp_path / 'trial_001'))
    time.sleep(0.15)
    rec.stop()
    for f in (tmp_path / 'motion.csv', tmp_path / 'trial_001' / 'motion.csv'):
        lines = open(f).read().splitlines()
        assert lines[0].startswith('wall,') and len(lines) > 5, f
    assert not (tmp_path / 'run.log').exists(), 'console off: no run.log'


def urconfig_load_bench():
    from cable_assembly import config as urconfig
    return urconfig.load('bnc_assembly', ['cell=iiwa14_bench'])


def test_the_app_runs_as_the_tree(monkeypatch):
    """bnc_assembly: the task's steps are the tree's leaves, the recoveries its Recover branch, --trials 100."""
    import argparse
    import inspect

    from cable_assembly.apps import bnc_assembly as app
    src = inspect.getsource(app.build_and_run)
    for leaf in ('Localize the socket', 'Pick the cable', 'Connector pose in the hand', 'Approach: stand-off',
                 'Insert', 'Seat: connector clocking', 'Lock: collar clocking', 'Verify: tug', 'Release and escape',
                 'Disassemble: put the cable back', 'Place it at the reset pose', 'Back out (checked) and place',
                 'Re-grip, back out and place', 'Reset the cable: square pick, aligned place', 'Go home'):
        assert f"'{leaf}'" in src, leaf
    assert 'while cycle < n_cycles' not in src, 'the old loop is gone'
    # NO VISION FIT OF THE HELD PLUG (the user, 2026-11-06: "the estimator doesn't fit the outline very well")
    assert 'in_hand_from_rgb' not in src and 'in_hand_from_cad' not in src
    cfg0 = urconfig_load_bench()
    assert cfg0.get('in_hand_rgb') is None and cfg0.get('in_hand_fit') is None
    p = argparse.ArgumentParser()
    apply = app.add_args(p)
    from cable_assembly import config as urconfig
    for argv, want in (([], None), (['--trials'], 100), (['--trials', '7'], 7)):
        cfg = urconfig.load('bnc_assembly', ['cell=iiwa14_bench'])
        apply(p.parse_args(argv), cfg)
        assert cfg.get_path('campaign.trials') == want, argv
    cfg = urconfig.load('bnc_assembly', ['cell=iiwa14_bench', 'campaign.trials=100'])
    assert app.campaign_precheck(cfg)
    cfg = urconfig.load('bnc_assembly', ['cell=iiwa14_bench', 'campaign.trials=100', 'assembly.disassembly.enabled=false'])
    assert not app.campaign_precheck(cfg), 'nothing would put the cell back between trials'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))


def test_a_pick_failure_before_the_cable_is_held_is_not_a_persons():
    """The user, 2026-11-06: an intervention for a failed reorient, with the hand empty and nothing touched --
    "the arm has no way of knowing the cable state before and it doesn't need to yet". The pick's failures take
    the cell from the gripper (empty: the cable is on the bench, the next scan finds it) and are recoverable;
    only the same failure twice asks a person."""
    import inspect

    from cable_assembly.apps import bnc_assembly as app
    src = inspect.getsource(app.build_and_run)
    pick = src[src.index('    def step_pick():'):src.index('    # ---------------------------------------------------------------- IN HAND')]
    assert "'intervene'" not in pick, 'no pick failure goes straight to a person'
    assert pick.count('**_hand()') >= 5


def test_the_6d_scan_rejects_a_plug_off_the_bench():
    """2026-11-06 01:02: an estimate 457 mm UNDER the bench passed the gate; the pick planned against it."""
    from types import SimpleNamespace

    from cable_assembly import config as urconfig
    from cable_assembly.perception import pose6d as P6
    from cable_assembly.skills.pose6d_pick import Pose6DScanner
    from cable_assembly.transforms import xyzrpy_to_matrix
    cfg = urconfig.load('bnc_assembly', ['cell=iiwa14_bench'])
    me = SimpleNamespace(cfg=cfg, part=P6.plug_part(cfg))
    bench = float(cfg.get_path('ground_plane.z_mm')) / 1000.0

    def plug_with_junction_at(z):
        T = xyzrpy_to_matrix([0.57, 0.105, 0.0], np.radians([0.0, -11.2, 83.6]))
        J = P6.junction_from_plug(T, part=me.part)
        T[2, 3] += z - J[2, 3]
        return {'T_base_obj': T}
    assert Pose6DScanner._plausible(me, plug_with_junction_at(bench + 0.008), 'test') is not None
    assert Pose6DScanner._plausible(me, plug_with_junction_at(-0.4514), 'test') is None, 'the 01:02 estimate'
    assert Pose6DScanner._plausible(me, None, 'test') is None


def test_the_viewer_window_reads_the_boards_snapshot(tmp_path):
    """sim/src/cable_assembly_sim/bt_window.py (the live viewer's collapsible window): its title says the mode
    and state with the window folded, its text is ASCII (imgui's default font), and the snapshot's depth-first
    rows nest into the tree the board drew."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), '..', 'sim', 'src', 'cable_assembly_sim', 'bt_window.py')
    spec = importlib.util.spec_from_file_location('bt_window', path)
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)                         # imgui-free: only draw() needs it
    b = Board(str(tmp_path), mode='autonomous', tags=['ENGAGE'])
    camp = C.Campaign(str(tmp_path), 100, b)
    leaf = C.Step('Insert', lambda: True, camp)
    guard = camp.recovery('cable in hand?', lambda t: True, 'place it', lambda: True)
    root = C.sequence('BNC assembly', C.sequence('Assemble', leaf), C.selector('Recover', guard))
    b.set_tree(root)
    b.set_trial(3, 100)
    b.set_autonomy('recovering', 'backing out → the reset pose')
    snap = json.loads(json.dumps(b.snapshot(), default=str))      # as the wire carries it
    b.close()
    assert W.title(snap) == 'Behavior tree: AUTONOMOUS | RECOVERING | trial 3/100###behavior_tree'
    assert W.ascii_text('cable in hand?  →  place it ≤3 …') == 'cable in hand?  ->  place it <=3 ...'
    nodes = snap['nodes']
    assert [n['depth'] for n in nodes] == [0, 1, 2, 1, 2]
    assert W.subtree_end(nodes, 0) == 5 and W.subtree_end(nodes, 1) == 3 and W.subtree_end(nodes, 2) == 3
    assert W.title(None).startswith('Behavior tree: no task yet')


IDLE = 'the iiwa is not commanding (FRI session IDLE)'


def test_with_the_arm_unable_to_move_only_quit_gets_out(tmp_path, monkeypatch, capsys):
    """2026-11-06: the FRI session went IDLE mid-run; every choice re-ran moves that failed at once and the prompt
    came straight back. Now r / n are refused while the link is down, and q ends the run."""
    answers(monkeypatch, 'n', 'r', '', 'q')
    cell = Cell(tmp_path, n=5, script={(1, 1, 'pick'): RuntimeError('boom')}, link_check=lambda: IDLE)
    assert not cell.run()
    assert cell.camp.abort and len(cell.camp.trials) == 1
    out = capsys.readouterr().out
    assert out.count('refused: ' + IDLE) == 2 and 'THE ARM CANNOT MOVE' in out
    assert [r['k'] if 'k' in r else r['trial'] for r in cell.rows(tmp_path)] == [1]


def test_once_the_link_is_back_the_choice_is_taken(tmp_path, monkeypatch, capsys):
    state = {'down': True}
    replies = iter(['n', 'n'])

    def fake(*_a, **_k):
        try:
            r = next(replies)
        except StopIteration:
            raise EOFError from None
        if state.pop('restarted', None) is None and r == 'n':
            state['restarted'] = True             # the first 'n' refused; then the FRI app is restarted
        else:
            state['down'] = False
        return r
    import builtins
    monkeypatch.setattr(builtins, 'input', fake)
    cell = Cell(tmp_path, n=2, script={(1, 1, 'pick'): RuntimeError('boom')},
                link_check=lambda: IDLE if state['down'] else None)
    cell.run()
    assert len(cell.camp.trials) == 2, 'the second n, with the link back, went on to the next trial'
    assert 'refused' in capsys.readouterr().out


def test_no_trial_is_started_into_a_dead_link(tmp_path, monkeypatch):
    answers(monkeypatch, 'q')
    state = {'down': False}
    cell = Cell(tmp_path, n=5)
    cell.camp._link_check = lambda: IDLE if state['down'] else None
    orig = cell.camp.end_trial

    def end_trial(success):
        orig(success)
        state['down'] = True                      # the link drops after trial 1
    cell.camp.end_trial = end_trial
    assert not cell.run()
    assert len(cell.camp.trials) == 1, 'asked before trial 2 instead of failing it in a second'


def test_a_failure_with_the_link_down_is_the_links_and_no_recovery_is_tried(tmp_path, monkeypatch, capsys):
    """2026-11-06 12:51: the FRI session dropped at the stand-off; the engage start's refused move was reported as
    an IK failure and a recovery move that could not run was tried. Now: the link named, a person's, no recovery."""
    answers(monkeypatch, 'q')
    state = {'down': False}
    script = {(1, 1, 'insert'): (('the move to the engage start did not complete',),
                                 dict(connector='hand', gripped=True, arm='unknown'))}
    cell = Cell(tmp_path, n=3, script=script, link_check=lambda: IDLE if state['down'] else None,
                link_hint='restart the FRI app (CARTESIAN_IMPEDANCE_CONTROL)')
    orig = cell.camp.fail

    def fail(why, kind='recover', **c):
        state['down'] = True                      # the link went at the step that failed
        return orig(why, kind, **c)
    cell.camp.fail = fail
    assert not cell.run()
    f = cell.camp.trials[0].failures[0]
    assert f['kind'] == 'intervene' and f['why'].startswith('the arm lost its link (' + IDLE + ')'), f
    assert not [r for r in cell.ran if r[2] in ('back out and place', 'go home')], 'no recovery moved anything'
    assert 'CARTESIAN_IMPEDANCE_CONTROL' in capsys.readouterr().out, 'the hint for this run'
