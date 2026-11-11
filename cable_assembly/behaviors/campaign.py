"""THE TASK AS A BEHAVIOR TREE, RUN N TIMES, RECOVERING WHAT IT SAFELY CAN AND ASKING A PERSON FOR THE REST
(the user, 2026-11-05: "an option for running it N times (default 100). record all data. add basic failure
recovery ... including reseting ... be conservative and stop operations and ask for intervention when
necessary. indicate/visualize the mode and autonomy state").

THE TREE an app builds from these nodes (apps/bnc_assembly.py):

    BNC assembly                                    sequence
      Start: home + gripper warm-up                 step
      Trials ×N                                     Trials -- runs its child N times, whatever each gives
        Trial                                       TrialSequence -- prepare, attempts, and the end ALWAYS
          Prepare the trial                         step: the hand checked empty (else a person), the arm home
          Attempts (≤A)                             AttemptLoop -- Assemble; if it fails, Recover, try again
            Assemble                                sequence of the task's steps
            Recover                                 selector: the first recovery whose condition holds
              ◇ needs a person?        -> Ask the operator
              ◇ <state the app knows how to undo>  -> <that recovery>
              ...
              (fallback)               -> Ask the operator
          End the trial: home + snapshot            step (skipped only when the operator stopped the run)

WHAT DECIDES A RECOVERY is the CELL STATE the steps keep up to date (Trial.cell: where the connector is,
how far it is assembled, whether the fingers hold it, where the arm is) -- not the name of the step that
failed. A step that fails says why and, through `kind`, how sure it is:

    recover     the cell is in a state the recoveries describe: let them try
    intervene   only a person can judge it (an unexpected force in free space, an unknown state, an
                exception, a recovery that itself failed): stop and ask
    abort       the operator said stop (q at a gate): end the run, the arm left where it is

CONSERVATIVE BY CONSTRUCTION:
  * a recovery runs only when its condition matches the recorded state exactly; nothing matches -> a person
  * a recovery that fails is never followed by another automatic one -> a person
  * a free-space force-guard trip during a step makes its failure `intervene`, whatever the step said
  * an exception in a step brakes the arm (safe_stop) and asks a person
  * the same step failing the same way twice in a trial -> a person (it is the cell, not luck)
  * a trial gets at most `max_attempts`; `max_consecutive_failures` failed trials in a row -> a person
  * an intervention NEVER times out or continues by itself: it needs a deliberate choice at the terminal
    (--no-prompts silences the routine gates, not this). No terminal (stdin closed) -> the run stops.
  * SUPERVISED mode asks before every recovery as well

Pause and stop come as files in the run folder (request_pause, request_stop: the live viewer's Behavior tree
window writes them, or touch them): a pause is taken at the next step boundary and lasts until withdrawn
(Resume in the viewer, or Enter here); a stop ends
the run once the current trial has finished and reset.

THE RECORD, in the run folder: trials.csv / trials.jsonl (one row per trial: outcome, attempts, where and
why each attempt failed, recoveries, interventions, and what the steps measured), events.jsonl (every step
entered and left, every state change, every question and its answer), campaign.json (the run: config,
mode, git state; the summary at the end), and trial_NNN/ folders the app fills (its CSVs, motion, images).
"""

import csv
import json
import os
import select
import subprocess
import sys
import time

import py_trees
from py_trees.common import Status

from .. import log as urlog
from .board import REQUEST_PAUSE, REQUEST_STOP

log = urlog.get('campaign')

CELL_WORDS = {
    'connector': {'bench': 'the cable is on the bench', 'hand': 'the cable is in the hand',
                  'socket': 'the connector is in the socket', 'unknown': "the cable's whereabouts are unknown"},
    'arm': {'home': 'the arm is home', 'pick': 'the arm is at the pick pose', 'socket': 'the arm is at the socket',
            'clear': 'the arm is clear of the socket', 'unknown': 'the arm is somewhere in the workspace'},
}


class StepFailed(Exception):
    """Raised anywhere inside a step to fail it: `why`, `kind` (recover | intervene | abort) and what the
    cell is now (connector=, assembly=, gripped=, arm=). A `finally` on the way out still runs."""

    def __init__(self, why, kind='recover', **cell):
        super().__init__(why)
        self.why, self.kind, self.cell = why, kind, cell


class Trial:
    """One trial: its record (a row of trials.csv) and the cell state its steps keep."""

    def __init__(self, k, folder):
        self.k, self.dir = int(k), folder
        self.t_start, self.t_end = time.time(), None
        self.attempt = 1
        self.cell = {'connector': 'bench', 'assembly': None, 'gripped': False, 'arm': 'home'}
        self.failure = None                # this attempt's first failure: {step, why, kind, attempt}
        self.failures = []                 # every one
        self.recoveries = []               # {attempt, step, ok}
        self.recovery_failures = []
        self.interventions = []            # {step, why, choice, waited_s}
        self.collision = None              # a free-space force-guard trip during the current step
        self.recovery_failed = False
        self.end_trial = False             # the operator: count it failed, start the next
        self.retry_now = False             # the operator: try again (not counted against the budget)
        self.outcome = None                # the app's word for how far it got: assembled | seated | ...
        self.success = None
        self.data = {}                     # what the steps measured

    def set_cell(self, **kw):
        for k, v in kw.items():
            if k not in self.cell:
                raise KeyError(f'unknown cell key {k!r} (one of {sorted(self.cell)})')
            self.cell[k] = v

    def describe(self):
        c = self.cell
        words = [CELL_WORDS['connector'].get(c['connector'], str(c['connector']))
                 + (f" ({c['assembly']})" if c['assembly'] and c['connector'] == 'socket' else ''),
                 {True: 'the fingers are closed on it', False: 'the fingers are open',
                  None: 'whether the fingers hold it is unknown'}[c['gripped']],
                 CELL_WORDS['arm'].get(c['arm'], str(c['arm']))]
        return '; '.join(words)

    def row(self):
        last = self.failures[-1] if self.failures else {}
        r = {'trial': self.k, 'success': self.success, 'outcome': self.outcome, 'attempts': self.attempt,
             'seconds': round((self.t_end or time.time()) - self.t_start, 1),
             'started': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.t_start)),
             'failed_steps': ' | '.join(f"{f['attempt']}:{f['step']}" for f in self.failures),
             'last_failure': last.get('why', ''), 'last_failure_kind': last.get('kind', ''),
             'recoveries': ' | '.join(f"{x['attempt']}:{x['step']}:{'ok' if x['ok'] else 'FAILED'}"
                                      for x in self.recoveries),
             'interventions': ' | '.join(f"{x['step']}:{x['choice']}" for x in self.interventions),
             'n_recoveries': len(self.recoveries), 'n_interventions': len(self.interventions),
             'cell_at_end': json.dumps(self.cell)}
        for k, v in self.data.items():
            r[k] = v if isinstance(v, (int, float, str, bool)) or v is None else json.dumps(v, default=str)
        return r


# ---------------------------------------------------------------------------- nodes
class Step(py_trees.behaviour.Behaviour):
    """A blocking step: runs `fn` whole inside one tick. False, a StepFailed or an exception fail it.
    `enabled=False` draws it as off and passes it. `recovery=True` marks a recovery action."""

    def __init__(self, name, fn, camp, enabled=True, note='', recovery=False, operator=False):
        super().__init__(name=name)
        self.fn, self.camp = fn, camp
        self.bt_enabled, self.bt_note, self.bt_kind = bool(enabled), note, 'action'
        self.recovery, self.operator = bool(recovery), bool(operator)

    def update(self):
        c = self.camp
        self.bt_skipped = False
        if c.abort:
            self.bt_skipped = True                     # not run: the run was stopped (drawn idle, not failed)
            return Status.FAILURE
        if not self.bt_enabled:
            return Status.SUCCESS
        c.check_requests()
        if c.abort:
            self.bt_skipped = True
            return Status.FAILURE
        t = c.trial
        if self.recovery and not self.operator and c.confirm_recoveries:
            ans = c.ask(f'\n[RECOVERY] {self.name}. Cell: {t.describe() if t else "?"}\n'
                        '    Enter to run it (q to stop and intervene instead): ')
            if ans is not None and ans.strip().lower() in ('q', 'quit', 'n', 'no'):
                if t is not None:
                    t.recovery_failed = True
                return Status.FAILURE
        if t is not None:
            t.collision = None
        c.current_step = self
        c.board.enter(self)
        ok = False
        try:
            ok = self.fn() is not False
        except StepFailed as e:
            c.fail(e.why, e.kind, **e.cell)
        except Exception as exc:               # noqa: BLE001 -- an error is a reason to stop, not to crash
            log.exception('Step "%s" raised:', self.name)
            c.safe_stop()
            c.fail(f'{type(exc).__name__}: {exc}', 'intervene', arm='unknown')
        finally:
            c.current_step = None
        if not ok and t is not None and not self.recovery and t.failure is None:
            c.current_step = self
            c.fail(f'"{self.name}" did not succeed (no reason given)')
            c.current_step = None
        if self.recovery and not self.operator and t is not None:
            t.recoveries.append({'attempt': t.attempt, 'step': self.name, 'ok': bool(ok)})
            c.board.count('recoveries')
            if not ok:
                t.recovery_failed = True
        c.board.leave(self, ok)
        return Status.SUCCESS if ok else Status.FAILURE


class Condition(py_trees.behaviour.Behaviour):
    """A question about the cell: SUCCESS iff `fn()`."""

    def __init__(self, name, fn):
        super().__init__(name=name)
        self.fn, self.bt_kind = fn, 'condition'

    def update(self):
        try:
            return Status.SUCCESS if self.fn() else Status.FAILURE
        except Exception as exc:               # noqa: BLE001
            log.warning('condition "%s" raised (%s) -- taken as false.', self.name, exc)
            return Status.FAILURE


class AttemptLoop(py_trees.composites.Composite):
    """children = [assemble, recover]. Tick `assemble`; if it fails, tick `recover` in the same tick; if
    that succeeds and the trial has attempts left (or the operator asked to retry), start again from
    `assemble`. An `abort` failure skips recovery and ends the run."""

    def __init__(self, name, assemble, recover, camp, max_attempts):
        super().__init__(name=name, children=[assemble, recover])
        self.camp, self.max_attempts = camp, max(1, int(max_attempts))
        self.bt_kind = 'attempts'
        self._recovering = False

    def tick(self):
        c = self.camp
        if self.status != Status.RUNNING:
            for ch in self.children:
                if ch.status != Status.INVALID:
                    ch.stop(Status.INVALID)
            self._recovering = False
            self.initialise()
        if not self._recovering:
            self.current_child = self.children[0]
            for node in self.children[0].tick():
                yield node
            st = self.children[0].status
            if st == Status.RUNNING:
                self.status = Status.RUNNING
                yield self
                return
            if st == Status.SUCCESS:
                self.stop(Status.SUCCESS)
                yield self
                return
            f = c.trial.failure if c.trial else None
            if c.abort or (f and f['kind'] == 'abort'):
                c.stop_run(f"stopped by the operator at \"{f['step']}\"" if f else 'stopped')
                self.stop(Status.FAILURE)
                yield self
                return
            self._recovering = True
            c.begin_recovery()
        self.current_child = self.children[1]
        for node in self.children[1].tick():
            yield node
        st = self.children[1].status
        if st == Status.RUNNING:
            self.status = Status.RUNNING
            yield self
            return
        c.end_recovery(st == Status.SUCCESS)
        t = c.trial
        again = (st == Status.SUCCESS and not c.abort and not t.end_trial
                 and (t.retry_now or t.attempt < self.max_attempts))
        if again:
            if not t.retry_now:
                log.warning('Trial %d: attempt %d of %d failed and was recovered -- trying again.', t.k, t.attempt,
                            self.max_attempts)
            t.retry_now = False
            t.attempt += 1
            t.failure, t.recovery_failed, t.collision = None, False, None
            c.board.set_trial(t.k, c.n_trials, t.attempt)
            for ch in self.children:
                ch.stop(Status.INVALID)
            self._recovering = False
            self.status = Status.RUNNING
            yield self
            return
        if st == Status.SUCCESS and not t.end_trial and not c.abort:
            log.error('Trial %d: %d attempts used -- counted as FAILED (the cell was put back).', t.k, t.attempt)
        self.stop(Status.FAILURE)
        yield self


class TrialSequence(py_trees.composites.Composite):
    """children = [prepare, attempts, end]: prepare; the attempts if it succeeded; then the END ALWAYS -- the
    arm home and the trial's last picture after a failed trial as after a good one -- unless the operator
    stopped the run, which leaves the arm where it is. SUCCESS only when all three succeeded."""

    def __init__(self, name, prepare, attempts, end, camp):
        super().__init__(name=name, children=[prepare, attempts, end])
        self.camp, self.bt_kind, self._i = camp, 'sequence', 0

    def tick(self):
        if self.status != Status.RUNNING:
            for ch in self.children:
                if ch.status != Status.INVALID:
                    ch.stop(Status.INVALID)
            self._i = 0
            self.initialise()
        while self._i < 3:
            child = self.children[self._i]
            if self._i == 2 and self.camp.abort:
                break                                  # stopped: the arm stays where it is
            self.current_child = child
            for node in child.tick():
                yield node
            if child.status == Status.RUNNING:
                self.status = Status.RUNNING
                yield self
                return
            self._i += 1 if not (self._i == 0 and child.status != Status.SUCCESS) else 2
        ok = all(ch.status == Status.SUCCESS for ch in self.children)
        self.stop(Status.SUCCESS if ok else Status.FAILURE)
        yield self


class Trials(py_trees.decorators.Decorator):
    """Run the child (one trial) `camp.n_trials` times, whatever each gives, until the operator stops it."""

    def __init__(self, name, child, camp):
        super().__init__(name=name, child=child)
        self.camp, self.bt_kind = camp, 'trials'

    def initialise(self):
        self.camp.begin_trial(1)

    def update(self):
        st = self.decorated.status
        if st == Status.RUNNING:
            return Status.RUNNING
        c = self.camp
        c.end_trial(st == Status.SUCCESS)
        if c.more_trials():
            c.begin_trial(c.trial.k + 1)
            self.decorated.stop(Status.INVALID)
            return Status.RUNNING
        return Status.FAILURE if c.abort else Status.SUCCESS


def sequence(name, *children, note=''):
    node = py_trees.composites.Sequence(name=name, memory=True)
    node.add_children(list(children))
    node.bt_note = note
    return node


def selector(name, *children, note=''):
    node = py_trees.composites.Selector(name=name, memory=True)
    node.add_children(list(children))
    node.bt_note = note
    return node


# ---------------------------------------------------------------------------- the campaign
class Campaign:
    """The run: its trials, its record, its person in the loop. See the module docstring."""

    def __init__(self, out_dir, n_trials, board, max_attempts=3, max_consecutive_failures=3, recover=True,
                 confirm_recoveries=False, safe_stop=None, actions=None, on_trial_start=None, on_trial_end=None,
                 meta=None, link_check=None, link_hint='restart the FRI app on the smartPAD'):
        self.out_dir, self.n_trials, self.board = out_dir, max(1, int(n_trials)), board
        self.max_attempts = max(1, int(max_attempts))
        self.max_consecutive_failures = max(0, int(max_consecutive_failures))
        self.recover_on, self.confirm_recoveries = bool(recover), bool(confirm_recoveries)
        self._safe_stop = safe_stop
        self.actions = dict(actions or {})       # key -> (label, fn): what an intervention can do
        self.on_trial_start, self.on_trial_end = on_trial_start, on_trial_end
        self._link_check = link_check            # () -> None, or why the arm cannot be commanded now
        self.link_hint = str(link_hint)          # what brings it back, in words
        self.trial, self.trials = None, []
        self.current_step = None
        self.abort, self.stop_after_trial = False, False
        self.stop_reason = ''
        self.in_recovery = False
        self.consecutive_failures = 0
        self._stdin_ok = True
        self.meta = dict(meta or {})
        self.t_start = time.time()
        self.root = None
        self._write_meta()

    # ---------------------------------------------------------------- the tree's recover branch
    def recoverable(self):
        t = self.trial
        return (self.recover_on and t is not None and t.failure is not None and t.failure['kind'] == 'recover'
                and not t.recovery_failed)

    def recovery(self, question, pred, action, fn, note=''):
        """◇ question -> action: runs only when automatic recovery is allowed and `pred(trial)` holds."""
        node = sequence(question, Condition(question, lambda: self.recoverable() and bool(pred(self.trial))),
                        Step(action, fn, self, recovery=True), note=note)
        node.bt_guard = True
        return node

    def operator_branch(self, fallback=False):
        """Ask the operator: as the first branch (when a person is needed) or as the fallback."""
        step = Step('Ask the operator', self._operator_for_trial, self, recovery=True, operator=True)
        if fallback:
            return step
        q = 'needs a person?'
        node = sequence(q, Condition(q, lambda: not self.recoverable()), step)
        node.bt_guard = True
        return node

    # ---------------------------------------------------------------- the steps' interface
    def fail(self, why, kind='recover', **cell):
        """Fail the current step: why, how sure (kind), and what the cell is now. Returns False."""
        t = self.trial
        step = self.current_step.name if self.current_step is not None else '?'
        if t is None:
            log.error('FAILED at "%s": %s', step, why)
            return False
        if cell:
            t.set_cell(**cell)
        if self.in_recovery:
            t.recovery_failed = True
            t.recovery_failures.append({'attempt': t.attempt, 'step': step, 'why': why})
            log.error('RECOVERY FAILED at "%s": %s -- %s.', step, why, t.describe())
            self.board.event('recovery_failed', step=step, why=why, cell=dict(t.cell))
            return False
        if t.collision and kind == 'recover':
            kind = 'intervene'
            why = f'{why} -- and the force guard tripped in free space ({t.collision})'
        down = self.link_down()
        if down and kind != 'abort':
            # THE ARM'S LINK IS DOWN: whatever the step reports, no recovery can move it (2026-11-06 12:51: a dropped
            # FRI session read as "does not IK-solve", then a recovery move that could not run) -- a person's.
            kind = 'intervene'
            why = f'the arm lost its link ({down}) -- {why}'
        same = [f for f in t.failures if f['step'] == step and f['why'] == why and f['attempt'] < t.attempt]
        if same and kind == 'recover':
            # THE SAME FAILURE AGAIN: not bad luck but something about the cell -- a person, not another lap.
            kind = 'intervene'
            why = f"{why} (the same failure as attempt {same[-1]['attempt']})"
        rec = {'step': step, 'why': why, 'kind': kind, 'attempt': t.attempt}
        t.failures.append(rec)
        if t.failure is None:
            t.failure = rec
        log.error('TRIAL %d, attempt %d FAILED at "%s": %s [%s] -- %s.', t.k, t.attempt, step, why,
                  {'recover': 'recoverable', 'intervene': 'needs a person', 'abort': 'operator stop'}.get(kind, kind),
                  t.describe())
        self.board.failure(f'trial {t.k}.{t.attempt} "{step}": {why}')
        self.board.event('failure', step=step, why=why, failure_kind=kind, cell=dict(t.cell))
        return False

    def collision(self, tripped_by):
        """A free-space force-guard trip during the current step (an unexpected contact)."""
        if self.trial is not None:
            self.trial.collision = str(tripped_by)
            self.board.event('collision', step=getattr(self.current_step, 'name', '?'), tripped_by=str(tripped_by))

    def safe_stop(self):
        if self._safe_stop is not None:
            try:
                self._safe_stop()
            except Exception as exc:           # noqa: BLE001
                log.error('Braking after the error failed too (%s).', exc)

    def say(self, message):
        self.board.say(message)

    def link_down(self):
        """None while the arm can be commanded, else why not (the robot's link_check)."""
        if self._link_check is None:
            return None
        try:
            return self._link_check()
        except Exception as exc:                       # noqa: BLE001
            return f'the link check failed ({exc})'

    # ---------------------------------------------------------------- talking to the person
    def _readline(self, prompt, until=None):
        """A line from the terminal, or None when there is none (EOF). With `until`, also returns '' as
        soon as until() holds (a request withdrawn: the viewer's Resume)."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        if until is None or not self._stdin_ok:
            if not self._stdin_ok and until is not None:
                while not until():
                    time.sleep(0.25)
                return ''
            try:
                return input()
            except EOFError:
                self._stdin_ok = False
                return None
        while True:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.25)
            except (OSError, ValueError):
                r = []
            if r:
                line = sys.stdin.readline()
                if line == '':
                    self._stdin_ok = False
                    return self._readline('', until)
                return line.rstrip('\n')
            if until():
                print('', flush=True)
                return ''

    def ask(self, prompt):
        """input() that the board shows as WAITING FOR OPERATOR. None on EOF -- the caller decides."""
        prev = (self.board.autonomy, self.board.message)
        first = next((ln.strip() for ln in prompt.strip().splitlines() if ln.strip()), prompt.strip())
        self.board.set_autonomy('waiting', first)
        t0 = time.time()
        try:
            ans = self._readline(prompt)
        finally:
            self.board.set_autonomy(*prev)
        self.board.event('ask', prompt=first, answer=ans, waited_s=round(time.time() - t0, 1))
        return ans

    def intervene(self, why, step='?', allow=('r', 'n', 'q')):
        """STOP AND ASK. The arm holds still until the operator chooses; returns 'retry', 'next' or 'quit'.
        Never times out; no terminal -> 'quit'."""
        t = self.trial
        self.safe_stop()
        prev = (self.board.autonomy, self.board.message)
        self.board.set_autonomy('intervention', f'{step}: {why}')
        self.board.count('interventions')
        self.board.event('intervention', step=step, why=why, cell=dict(t.cell) if t else None)
        t0 = time.time()
        labels = {'r': 'retry this trial from the top (put the cell back first: cable on the bench, socket empty)',
                  'n': 'count this trial as FAILED and start the next (the cell has been put back)',
                  'q': 'quit the run'}
        opts = [k for k in allow if k in labels] + [k for k in self.actions if k not in labels]
        choice = None
        while choice is None:
            bar = '=' * 92
            lines = ['', bar, f'  INTERVENTION NEEDED -- trial {t.k if t else "-"}/{self.n_trials}'
                     + (f', attempt {t.attempt}' if t else '') + f', at "{step}"', f'  {why}']
            if t is not None:
                lines.append(f'  The cell now: {t.describe()}.')
            down = self.link_down()
            if down:
                lines += [f'  THE ARM CANNOT MOVE: {down}.',
                          f'  Every move would fail at once -- {self.link_hint}. r / n are refused until the',
                          '  session is commanding again. q quits the run (Ctrl+C also stops safely).']
            else:
                lines += ['  The arm is HOLDING STILL. Nothing moves until you choose.']
            lines += ['-' * 92]
            for k in opts:
                lines.append(f'  [{k}] {labels[k] if k in labels else self.actions[k][0]}')
            lines.append(bar)
            sys.stdout.write('\a')
            ans = self._readline('\n'.join(lines) + '\nchoice (a letter, then Enter): ')
            if ans is None:
                log.error('INTERVENTION with no terminal to answer it (stdin closed) -- stopping the run.')
                choice = 'quit'
                break
            k = ans.strip().lower()[:1]
            if k in ('r', 'n') and k in opts and self.link_down():
                print(f'  refused: {self.link_down()} -- {self.link_hint} first, or q to quit.', flush=True)
            elif k in ('r', 'n', 'q') and k in opts:
                choice = {'r': 'retry', 'n': 'next', 'q': 'quit'}[k]
            elif k in self.actions:
                label, fn = self.actions[k]
                self.board.event('intervention_action', step=step, action=label)
                try:
                    ok = fn()
                    print(f'  {label}: {"done" if ok is not False else "FAILED"}', flush=True)
                except Exception as exc:       # noqa: BLE001
                    print(f'  {label}: FAILED ({exc})', flush=True)
            else:
                print('  (type one of ' + ', '.join(opts) + ' and press Enter)', flush=True)
        waited = round(time.time() - t0, 1)
        if t is not None:
            t.interventions.append({'step': step, 'why': why, 'choice': choice, 'waited_s': waited})
        self.board.event('intervention_answer', step=step, choice=choice, waited_s=waited)
        log.warning('INTERVENTION at "%s": the operator chose %s after %.0f s.', step, choice.upper(), waited)
        self.board.set_autonomy(*prev)
        if choice == 'quit':
            self.stop_run(f'stopped by the operator at an intervention ("{step}")')
        return choice

    def _operator_for_trial(self):
        t = self.trial
        f = t.failure or {'step': '?', 'why': 'the attempt failed', 'kind': 'intervene'}
        why = f['why']
        if t.recovery_failed and t.recovery_failures:
            why = f"{why}; then the recovery failed: {t.recovery_failures[-1]['why']}"
        elif f['kind'] == 'recover' and not self.recover_on:
            why = f'{why} (automatic recovery is off: campaign.recover)'
        elif f['kind'] == 'recover':
            why = f'{why} (no automatic recovery matches this state)'
        choice = self.intervene(why, f['step'])
        if choice == 'retry':
            t.retry_now = True
            t.set_cell(connector='bench', assembly=None, gripped=False, arm='unknown')
            return True
        if choice == 'next':
            t.end_trial = True
            t.set_cell(connector='bench', assembly=None, gripped=False, arm='unknown')
            return True
        return False

    def check_requests(self):
        """A stop or pause asked for (the request files: the viewer's buttons), taken here, between steps."""
        if self.board.requested(REQUEST_STOP) and not self.stop_after_trial:
            self.board.withdraw(REQUEST_STOP)
            self.stop_after_trial = True
            self.stop_reason = 'stop requested'
            log.warning('STOP REQUESTED: the run ends once trial %s has finished and reset.',
                        self.trial.k if self.trial else '-')
            self.board.set_autonomy('stopping', 'stop requested: finishing this trial, then stopping')
        if self.board.requested(REQUEST_PAUSE):
            self.pause()

    def pause(self):
        prev = (self.board.autonomy, self.board.message)
        self.safe_stop()
        self.board.set_autonomy('paused', 'paused at a step boundary: Resume in the viewer, or Enter at the terminal')
        log.warning('PAUSED (requested). The arm holds still.')
        ans = self._readline('\n[PAUSED] Enter (or Resume in the viewer) to resume, q + Enter to stop the run: ',
                             until=lambda: not self.board.requested(REQUEST_PAUSE))
        self.board.withdraw(REQUEST_PAUSE)
        self.board.event('pause_end', answer=ans)
        self.board.set_autonomy(*prev)
        if ans is not None and ans.strip().lower() in ('q', 'quit'):
            self.stop_run('stopped by the operator while paused')

    def stop_run(self, reason):
        self.abort = True
        self.stop_reason = self.stop_reason or reason
        self.board.event('stop', reason=reason)

    # ---------------------------------------------------------------- trials
    def begin_trial(self, k):
        folder = os.path.join(self.out_dir, f'trial_{k:03d}')
        os.makedirs(folder, exist_ok=True)
        self.trial = Trial(k, folder)
        self.trials.append(self.trial)
        self.board.set_trial(k, self.n_trials, 1)
        if self.board.autonomy not in ('stopping',):
            self.board.set_autonomy('running', f'trial {k}/{self.n_trials}')
        self.board.event('trial_start', folder=folder)
        log.info('=' * 78)
        log.info('TRIAL %d/%d  (data: %s)', k, self.n_trials, folder)
        log.info('=' * 78)
        if self.on_trial_start is not None:
            self.on_trial_start(self.trial)

    def end_trial(self, success):
        t = self.trial
        t.t_end = time.time()
        t.success = bool(success)
        if t.outcome is None:
            last = (t.failures[-1] if t.failures else {}).get('step')
            t.outcome = 'assembled' if success else (
                (f'stopped at "{last}"' if self.abort else f'failed at "{last}"') if last else
                ('stopped' if self.abort else 'failed'))
        self.board.count('assembled' if success else 'failed')
        self.board.trial_done(t.t_end - t.t_start)
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1
        if self.on_trial_end is not None:
            try:
                self.on_trial_end(t)
            except Exception as exc:           # noqa: BLE001 -- the record is written regardless
                log.warning('trial end hook failed (%s)', exc)
        self._write_trial(t)
        self.board.event('trial_end', success=t.success, outcome=t.outcome, attempts=t.attempt,
                         seconds=round(t.t_end - t.t_start, 1))
        n_ok = sum(1 for x in self.trials if x.success)
        log.info('TRIAL %d/%d %s (%s) in %.0f s, %d attempt%s -- %d/%d so far.', t.k, self.n_trials,
                 'SUCCEEDED' if success else 'FAILED', t.outcome, t.t_end - t.t_start, t.attempt,
                 '' if t.attempt == 1 else 's', n_ok, len(self.trials))

    def more_trials(self):
        t = self.trial
        if self.abort:
            return False
        if self.stop_after_trial:
            return False
        if t.k >= self.n_trials:
            return False
        down = self.link_down()
        if down:
            # NO TRIAL STARTED THAT CANNOT MOVE (2026-11-06: with the FRI session IDLE, trials failed a second apiece)
            choice = self.intervene(f'{down} -- no trial can run', 'between trials', allow=('n', 'q'))
            if choice == 'quit':
                return False
        if 0 < self.max_consecutive_failures <= self.consecutive_failures:
            choice = self.intervene(f'{self.consecutive_failures} trials in a row have failed '
                                    f'(campaign.max_consecutive_failures) -- look at the cell before the next one',
                                    'between trials', allow=('n', 'q'))
            if choice == 'quit':
                return False
            self.consecutive_failures = 0
        return True

    def begin_recovery(self):
        t = self.trial
        self.in_recovery = True
        f = t.failure or {'step': '?', 'why': '?'}
        self.board.set_autonomy('recovering', f"attempt {t.attempt} failed at \"{f['step']}\": {f['why']}")

    def end_recovery(self, ok):
        self.in_recovery = False
        if self.board.autonomy == 'recovering':
            self.board.set_autonomy('stopping' if self.stop_after_trial else 'running',
                                    'recovered' if ok else 'recovery did not complete')

    # ---------------------------------------------------------------- running
    def run(self, root):
        """Tick the tree to the end. True when every trial ran (whatever their outcomes) and nobody stopped it."""
        self.root = root
        self.board.set_tree(root)
        tree = py_trees.trees.BehaviourTree(root)
        tree.setup()
        log.info('THE BEHAVIOR TREE:\n%s', py_trees.display.unicode_tree(root))
        try:
            while True:
                tree.tick()
                if root.status != Status.RUNNING:
                    break
        except BaseException:
            self.board.set_autonomy('fault', 'interrupted' if sys.exc_info()[0] is KeyboardInterrupt
                                    else f'error: {sys.exc_info()[1]}')
            self.finish(fault=True)
            raise
        self.finish()
        return root.status == Status.SUCCESS and not self.abort

    def finish(self, fault=False):
        n = len(self.trials)
        ok = sum(1 for t in self.trials if t.success)
        if self.trial is not None and self.trial.t_end is None and not fault:
            pass
        if not fault and not self.trials and not self.abort:
            self.abort, self.stop_reason = True, self.stop_reason or 'the start failed: no trial was run'
        state = 'fault' if fault else ('stopped' if self.abort else 'done')
        msg = (f'{ok}/{n} trials succeeded' + (f' of {self.n_trials} asked' if n < self.n_trials else '')
               + (f' -- {self.stop_reason}' if self.stop_reason else ''))
        if not fault:
            self.board.set_autonomy(state, msg)
        self.meta.update({'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'state': state, 'trials_run': n,
                          'trials_succeeded': ok, 'stop_reason': self.stop_reason,
                          'outcomes': {o: sum(1 for t in self.trials if t.outcome == o)
                                       for o in sorted({str(t.outcome) for t in self.trials})},
                          'recoveries': sum(len(t.recoveries) for t in self.trials),
                          'interventions': sum(len(t.interventions) for t in self.trials),
                          'seconds': round(time.time() - self.t_start, 1)})
        self._write_meta()
        log.info('RUN %s: %s. Records: %s (trials.csv, events.jsonl, campaign.json, trial_NNN/).',
                 state.upper(), msg, self.out_dir)

    # ---------------------------------------------------------------- the record
    def _write_meta(self):
        m = dict(self.meta)
        m.setdefault('started', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.t_start)))
        m.update({'n_trials': self.n_trials, 'max_attempts': self.max_attempts,
                  'max_consecutive_failures': self.max_consecutive_failures, 'recover': self.recover_on,
                  'confirm_recoveries': self.confirm_recoveries})
        try:
            with open(os.path.join(self.out_dir, 'campaign.json'), 'w') as fh:
                json.dump(m, fh, indent=1, default=str)
        except OSError as exc:
            log.warning('campaign.json not written (%s)', exc)

    def _write_trial(self, t):
        row = t.row()
        try:
            with open(os.path.join(self.out_dir, 'trials.jsonl'), 'a') as fh:
                fh.write(json.dumps({**row, 'failures': t.failures, 'recovery_failures': t.recovery_failures,
                                     'interventions_detail': t.interventions, 'data': t.data}, default=str) + '\n')
            rows = [x.row() for x in self.trials if x.t_end is not None]
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(self.out_dir, 'trials.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        except OSError as exc:
            log.warning('trial record not written (%s)', exc)


def git_state(repo):
    """{'commit', 'branch', 'dirty_files'} of the code that ran, or {} without git."""
    try:
        run = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, timeout=5).stdout
        return {'commit': run('rev-parse', 'HEAD').strip(), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD').strip(),
                'dirty_files': [ln[3:] for ln in run('status', '--porcelain').splitlines()]}
    except Exception:                          # noqa: BLE001
        return {}
