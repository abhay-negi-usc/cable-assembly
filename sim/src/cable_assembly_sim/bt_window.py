"""THE TASK'S BEHAVIOR TREE IN THE VIEWER -- a collapsible imgui window (the user, 2026-11-06: "integrate the
behavior tree visual into the simulation viewer as a collapsible window").

The task's board (cable_assembly/behaviors/board.py) sends its status as an overlay named "behavior_tree"
carrying a `bt` field: the mode, the autonomy state, the trial count, and every node of the tree in
drawing order with its depth, status, time and success/failure counts. Drawn here natively -- crisp at
any size, unlike the board's picture -- as:

  the TITLE BAR      "Behavior tree: AUTONOMOUS | RUNNING | trial 3/100", coloured by the autonomy state,
                     so it still says what the run is doing when the window is collapsed (its arrow, or a
                     double click on the bar). An INTERVENTION opens it again by itself.
  the HEADER         mode and state, what is happening, the trial and its counts, the last failure
  the TREE           a table: each composite a collapsible node (open at first), each step a leaf
                     coloured by its status, with its time and how often it has succeeded / failed;
                     "follow" keeps the path to the running step open
  PAUSE / STOP       buttons: pause at the next step boundary (then Resume), stop once this trial has finished
                     and reset. They write the run's request files (request_pause, request_stop in its folder,
                     which the task takes between steps) -- so only with the task on THIS PC; from another,
                     touch them there.

Nothing else talks to the task: the rest reads what the frame receiver stored (live.Overlays).
"""

REQUEST_PAUSE, REQUEST_STOP = 'request_pause', 'request_stop'     # cable_assembly/behaviors/board.py

import os
import time

# status -> RGBA, and the autonomy states' colours (as the board draws them)
STATUS = {
    'running': (0.36, 0.67, 1.00, 1.0), 'success': (0.42, 0.80, 0.52, 1.0), 'failure': (0.95, 0.36, 0.32, 1.0),
    'idle': (0.62, 0.64, 0.68, 1.0), 'off': (0.38, 0.40, 0.43, 1.0),
}
AUTONOMY = {
    'starting': ('STARTING', (0.67, 0.67, 0.71)), 'running': ('RUNNING', (0.28, 0.75, 0.43)),
    'recovering': ('RECOVERING', (0.96, 0.65, 0.14)), 'waiting': ('WAITING FOR OPERATOR', (0.94, 0.82, 0.24)),
    'intervention': ('INTERVENTION NEEDED', (0.94, 0.27, 0.24)), 'paused': ('PAUSED', (0.35, 0.59, 0.98)),
    'stopping': ('STOPPING AFTER THIS TRIAL', (0.96, 0.55, 0.20)), 'done': ('DONE', (0.59, 0.61, 0.64)),
    'stopped': ('STOPPED', (0.80, 0.37, 0.33)), 'fault': ('FAULT', (0.94, 0.27, 0.24)),
}
MODES = {'autonomous': ('AUTONOMOUS', (0.12, 0.49, 0.35)), 'supervised': ('SUPERVISED', (0.22, 0.39, 0.69))}
GLYPH = {'sequence': '->', 'selector': '?', 'attempts': '@', 'trials': 'x', 'guard': 'if', 'condition': '<>'}
# The default imgui font is ASCII + Latin-1: the board's arrows and signs, spelled out.
ASCII = str.maketrans({'→': '->', '≤': '<=', '≥': '>=', '…': '...', '✓': 'ok', '✗': 'x', '·': '|', '×': 'x'})


def ascii_text(s):
    return str(s).translate(ASCII).encode('ascii', 'replace').decode('ascii')


def fmt_s(s):
    if s is None:
        return ''
    s = float(s)
    if s < 60:
        return f'{s:.1f} s'
    if s < 3600:
        return f'{int(s // 60)}:{int(s % 60):02d}'
    return f'{int(s // 3600)} h {int(s % 3600 // 60):02d}'


def subtree_end(nodes, i):
    """The index after node i's subtree (its descendants follow it, deeper)."""
    d, j = nodes[i]['depth'], i + 1
    while j < len(nodes) and nodes[j]['depth'] > d:
        j += 1
    return j


def title(bt):
    """The window's title: readable with the window collapsed. ### keeps imgui's id fixed as it changes."""
    if not bt:
        return 'Behavior tree: no task yet###behavior_tree'
    mode = MODES.get(bt.get('mode'), (str(bt.get('mode', '')).upper(), None))[0]
    state = AUTONOMY.get(bt.get('autonomy'), (str(bt.get('autonomy', '')).upper(), None))[0]
    trial = f" | trial {bt.get('trial', 0)}/{bt.get('n_trials', 0)}" if bt.get('n_trials') else ''
    return ascii_text(f'Behavior tree: {mode} | {state}{trial}') + '###behavior_tree'


class BehaviorTreeWindow:
    """Draws the latest "behavior_tree" overlay from `overlays` (live.Overlays) as a collapsible window."""

    NAME = 'behavior_tree'

    def __init__(self, overlays, pos=(880, 12), size=(600, 700)):
        self.overlays, self.pos, self.size = overlays, pos, size
        self.follow = True
        self._last, self._t_new, self._state = None, None, None

    def latest(self):
        by_name, _ = self.overlays.snapshot()
        ov = by_name.get(self.NAME)
        bt = ov.get('bt') if ov else None
        if ov is not self._last:
            self._last, self._t_new = ov, time.monotonic()
        return bt

    def draw(self, imgui):
        bt = self.latest()
        state = (bt or {}).get('autonomy')
        col = AUTONOMY.get(state, (None, (0.30, 0.32, 0.36)))[1]
        imgui.set_next_window_pos(imgui.ImVec2(*self.pos), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(*self.size), imgui.Cond_.first_use_ever)
        if state == 'intervention' and self._state != 'intervention':
            imgui.set_next_window_collapsed(False, imgui.Cond_.always)    # a person is needed: show it
        self._state = state
        dark = imgui.ImVec4(col[0] * 0.55, col[1] * 0.55, col[2] * 0.55, 1.0)
        bright = imgui.ImVec4(col[0] * 0.8, col[1] * 0.8, col[2] * 0.8, 1.0)
        imgui.push_style_color(imgui.Col_.title_bg, dark)
        imgui.push_style_color(imgui.Col_.title_bg_collapsed, dark)
        imgui.push_style_color(imgui.Col_.title_bg_active, bright)
        expanded, _ = imgui.begin(title(bt), None, 0)
        imgui.pop_style_color(3)
        try:
            if expanded:
                self._body(imgui, bt)
        finally:
            imgui.end()

    # ------------------------------------------------------------------ the window's content
    def _body(self, imgui, bt):
        if not bt:
            imgui.text_wrapped('No behavior tree yet: it comes from a task that runs one '
                               '(python -m cable_assembly.apps.bnc_assembly ...) with live_view.frames on.')
            return
        c4 = lambda c: imgui.ImVec4(*c, 1.0) if len(c) == 3 else imgui.ImVec4(*c)     # noqa: E731
        mode, mcol = MODES.get(bt.get('mode'), (str(bt.get('mode', '')).upper(), (0.5, 0.5, 0.5)))
        state, scol = AUTONOMY.get(bt.get('autonomy'), (str(bt.get('autonomy', '')).upper(), (0.8, 0.8, 0.8)))
        blink = bt.get('autonomy') in ('intervention', 'waiting') and int(time.time() * 2) % 2 == 0
        imgui.text_colored(c4(tuple(min(1.0, v * 1.6) for v in mcol)), mode)
        for tag in bt.get('tags') or []:
            imgui.same_line()
            imgui.text_colored(c4((0.70, 0.62, 0.92)), ascii_text(f'[{tag}]'))
        imgui.same_line(spacing=14)
        imgui.text_colored(c4((1.0, 1.0, 1.0) if blink else scol), state)
        age = None if self._t_new is None else time.monotonic() - self._t_new
        if age is not None and age > 5.0 and bt.get('autonomy') not in ('done', 'stopped', 'fault'):
            imgui.same_line()
            imgui.text_colored(c4((0.62, 0.64, 0.68)), f'(no word for {age:.0f} s)')
        if bt.get('message'):
            imgui.text_wrapped(ascii_text(bt['message']))
        n = bt.get('counts') or {}
        done = int(bt.get('trials_done') or 0)
        rate = f"{100.0 * n.get('assembled', 0) / done:.0f}% assembled" if done else 'no trial finished yet'
        per = f" | {fmt_s(bt['seconds_per_trial'])}/trial" if bt.get('seconds_per_trial') else ''
        eta = f" | ETA {fmt_s(bt['eta_s'])}" if bt.get('eta_s') else ''
        imgui.text(f"trial {bt.get('trial', 0)}/{bt.get('n_trials', 0)}"
                   + (f"  attempt {bt['attempt']}" if (bt.get('attempt') or 1) > 1 else '')
                   + f"   assembled {n.get('assembled', 0)}  failed {n.get('failed', 0)}  "
                     f"recoveries {n.get('recoveries', 0)}  interventions {n.get('interventions', 0)}")
        imgui.text_colored(c4((0.62, 0.64, 0.68)), f"{rate}{per}{eta} | elapsed {fmt_s(bt.get('elapsed_s'))}")
        if bt.get('last_failure'):
            imgui.push_style_color(imgui.Col_.text, c4(STATUS['failure']))
            imgui.text_wrapped('last failure: ' + ascii_text(bt['last_failure']))
            imgui.pop_style_color()
        self._controls(imgui, bt, c4)
        _, self.follow = imgui.checkbox('follow the running step', self.follow)
        imgui.separator()
        self._tree(imgui, bt.get('nodes') or [], c4)

    def _controls(self, imgui, bt, c4):
        """Pause / Resume and Stop after this trial -- the run's request files, when its folder is on this PC."""
        if bt.get('autonomy') in ('done', 'stopped', 'fault'):
            return
        run = bt.get('run_dir_abs')
        if not run or not os.path.isdir(run):
            imgui.text_colored(c4((0.62, 0.64, 0.68)), ascii_text(
                f"pause / stop: on the task's PC, touch {bt.get('run_dir', '<run>')}/request_pause or request_stop"))
            return
        pause_f, stop_f = os.path.join(run, REQUEST_PAUSE), os.path.join(run, REQUEST_STOP)
        pausing = os.path.exists(pause_f)
        if imgui.button('Resume##bt' if pausing else 'Pause at the next step##bt'):
            self._request(pause_f, not pausing)
        imgui.same_line()
        stopping = bt.get('autonomy') == 'stopping' or os.path.exists(stop_f)
        imgui.begin_disabled(stopping)
        if imgui.button('Stopping after this trial##bt' if stopping else 'Stop after this trial##bt'):
            self._request(stop_f, True)
        imgui.end_disabled()

    @staticmethod
    def _request(path, on):
        try:
            if on:
                open(path, 'w').close()
            elif os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(f'[live] behavior tree: could not {"write" if on else "remove"} {path} ({exc})', flush=True)

    def _tree(self, imgui, nodes, c4):
        flags = (imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.resizable
                 | imgui.TableFlags_.scroll_y | imgui.TableFlags_.no_borders_in_body_until_resize)
        if not imgui.begin_table('bt_nodes', 3, flags):
            return
        try:
            imgui.table_setup_scroll_freeze(0, 1)
            imgui.table_setup_column('step', imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column('time', imgui.TableColumnFlags_.width_fixed, 64.0)
            imgui.table_setup_column('ok / failed', imgui.TableColumnFlags_.width_fixed, 84.0)
            imgui.table_headers_row()
            i = 0
            while i < len(nodes):
                i = self._row(imgui, nodes, i, c4)
        finally:
            imgui.end_table()

    def _row(self, imgui, nodes, i, c4):
        n = nodes[i]
        end = subtree_end(nodes, i)
        kind, status = n.get('kind', 'action'), n.get('status', 'idle')
        glyph = GLYPH.get(kind, '')
        label = ascii_text((f'{glyph} ' if glyph else '') + n.get('name', '?') + ('  (off)' if status == 'off' else ''))
        imgui.table_next_row()
        if status == 'running':
            imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg1,
                                     imgui.get_color_u32(imgui.ImVec4(0.16, 0.27, 0.42, 1.0)))
        imgui.table_next_column()
        imgui.push_style_color(imgui.Col_.text, c4(STATUS.get(status, STATUS['idle'])))
        base = imgui.TreeNodeFlags_.span_all_columns | imgui.TreeNodeFlags_.draw_lines_to_nodes
        if end > i + 1:                                # a branch: collapsible
            if self.follow and status == 'running':
                imgui.set_next_item_open(True, imgui.Cond_.always)
            else:
                imgui.set_next_item_open(True, imgui.Cond_.once)
            opened = imgui.tree_node_ex(f'{label}##bt{i}', base)
        else:
            imgui.tree_node_ex(f'{label}##bt{i}', base | imgui.TreeNodeFlags_.leaf
                               | imgui.TreeNodeFlags_.no_tree_push_on_open)
            opened = False
        imgui.pop_style_color()
        if n.get('note') and status != 'off':
            imgui.same_line()
            imgui.text_colored(c4((0.55, 0.57, 0.61)), ascii_text(n['note']))
        imgui.table_next_column()
        if n.get('seconds') is not None:
            imgui.text_colored(c4((0.62, 0.64, 0.68)), fmt_s(n['seconds']) + (' ...' if n.get('live') else ''))
        imgui.table_next_column()
        if n.get('ok') or n.get('fail'):
            imgui.text_colored(c4(STATUS['success']), f"{n.get('ok', 0)}")
            imgui.same_line()
            imgui.text_colored(c4(STATUS['failure'] if n.get('fail') else STATUS['idle']), f"/ {n.get('fail', 0)}")
        if end > i + 1:
            if opened:
                j = i + 1
                while j < end:
                    j = self._row(imgui, nodes, j, c4)
                imgui.tree_pop()
        return end
