"""THE BEHAVIOR TREE ON SCREEN -- the task's tree, its MODE and its AUTONOMY STATE, live (the user,
2026-11-05: "add a visualization of a behavior tree ... indicate/visualize the mode and autonomy state").

  MODE       how much a person is asked: AUTONOMOUS (--no-prompts: nothing but interventions) or
             SUPERVISED (the behaviour gates ask before each behaviour); tags say DRY RUN / the backend
  AUTONOMY   what the run is doing about it NOW:
               RUNNING        the tree is executing on its own
               RECOVERING     an attempt failed; a recovery is putting the cell back
               WAITING        a gate or prompt is waiting for Enter (SUPERVISED)
               INTERVENTION   stopped, arm holding: a person must act before anything moves again
               PAUSED         paused at a step boundary on request
               STOPPING       will stop once this trial has finished and reset
               DONE / STOPPED / FAULT   the run is over: all trials ran / it was stopped / an error escaped
  THE TREE   every node with its status (running, succeeded, failed, idle, off), its time, and how often it
             has succeeded and failed over the whole run -- where a 100-trial campaign's failures gather

Drawn in a thread of its own (on a change only, at most `rate_hz`) and published, none of which the task
waits on:
  the live viewer                  the tree and state as the overlay "behavior_tree" (every change, and every
                                   2 s): the collapsible "Behavior tree" window in sim/scripts/live_view_gl.py,
                                   drawn natively (sim/src/cable_assembly_sim/bt_window.py), with Pause / Stop
                                   buttons; the mode/state lines in its panel; the picture, for the Isaac Sim viewer
  <run>/bt.png + bt_status.json    the tree as a picture and as data -- the run's last state stays on disk
  <run>/events.jsonl               every step entered and left, every state change -- the RECORD

Other live images (the in-hand CAD fit) take the same road: board.image(name, bgr) -> <run>/<name>_live.jpg
and an image window of that name in the live viewer.

PAUSE AND STOP are files in the run folder, request_pause and request_stop, which the run takes between steps:
the viewer's buttons write them (same PC), or by hand -- touch <run>/request_stop. (The board's own window,
a separate process, is gone: the user, 2026-11-06.)
"""

import json
import os
import threading
import time
from datetime import datetime

import numpy as np

from .. import log as urlog

log = urlog.get('board')

REQUEST_PAUSE = 'request_pause'
REQUEST_STOP = 'request_stop'

# autonomy state -> (label, colour RGB)
AUTONOMY = {
    'starting': ('STARTING', (170, 170, 180)),
    'running': ('RUNNING', (72, 190, 110)),
    'recovering': ('RECOVERING', (245, 166, 35)),
    'waiting': ('WAITING FOR OPERATOR', (240, 210, 60)),
    'intervention': ('INTERVENTION NEEDED', (240, 70, 60)),
    'paused': ('PAUSED', (90, 150, 250)),
    'stopping': ('STOPPING AFTER THIS TRIAL', (245, 140, 50)),
    'done': ('DONE', (150, 155, 163)),
    'stopped': ('STOPPED', (205, 95, 85)),
    'fault': ('FAULT', (240, 70, 60)),
}
MODES = {
    'autonomous': ('AUTONOMOUS', (30, 125, 90)),
    'supervised': ('SUPERVISED', (55, 100, 175)),
}
STATUS_COL = {
    'running': (90, 170, 255), 'success': (90, 200, 120), 'failure': (240, 85, 75),
    'idle': (150, 156, 165), 'off': (88, 92, 100),
}
GLYPH = {'sequence': '→', 'selector': '?', 'attempts': '↻', 'trials': '⟳', 'condition': '◇', 'action': '▸',
         'guard': '◇'}
BG, PANEL, TEXT, DIM, LINE = (22, 24, 28), (33, 36, 42), (228, 231, 235), (125, 131, 140), (64, 69, 78)


def kind_of(node):
    """A node's kind for the board: its `bt_kind` if it says, else from its py_trees class."""
    k = getattr(node, 'bt_kind', None)
    if k:
        return k
    name = type(node).__name__.lower()
    for k in ('sequence', 'selector', 'parallel'):
        if k in name:
            return k
    return 'action'


def walk(node, depth=0, parent=None):
    """(node, depth, parent) depth-first, in drawing order. A GUARD (a sequence marked `bt_guard`: a condition
    then the step it allows) is one row, "◇ question -> step": its two children are not walked."""
    yield node, depth, parent
    if getattr(node, 'bt_guard', False):
        return
    for c in getattr(node, 'children', None) or []:
        yield from walk(c, depth + 1, node)


def _fmt_s(s):
    if s is None:
        return ''
    s = float(s)
    if s < 60:
        return f'{s:.1f} s'
    if s < 3600:
        return f'{int(s // 60)}:{int(s % 60):02d}'
    return f'{int(s // 3600)} h {int(s % 3600 // 60):02d}'


def _atomic_write(path, data, mode='wb'):
    tmp = path + '.tmp'
    with open(tmp, mode) as fh:
        fh.write(data)
    os.replace(tmp, path)


class Board:
    """The run's live status (see the module docstring). Thread-safe; every method returns at once."""

    def __init__(self, out_dir, mode='supervised', tags=(), sender=None, rate_hz=4.0,
                 title='BNC assembly'):
        self.out_dir, self.title = out_dir, title
        self.mode, self.tags = mode, [str(t) for t in tags]
        self.sender = sender
        self.period = 1.0 / max(float(rate_hz), 0.5)
        self.autonomy, self.message = 'starting', ''
        self.trial, self.n_trials, self.attempt = 0, 0, 0
        self.counts = {'assembled': 0, 'failed': 0, 'recoveries': 0, 'interventions': 0}
        self.trial_seconds = []
        self.last_failure = ''
        self.t0 = time.time()
        self.root = None
        self.order = []                    # [(node, depth, parent)]
        self.parent = {}                   # id -> parent id
        self.stats = {}                    # id -> {'ok', 'fail', 'last_s', 'since', 'note'}
        self.active = []                   # ids of the steps executing now (a stack)
        self._lock = threading.RLock()
        self._version, self._drawn, self._last_draw, self._last_sent = 0, -1, 0.0, 0.0
        self._images = {}
        os.makedirs(out_dir, exist_ok=True)
        self._events = open(os.path.join(out_dir, 'events.jsonl'), 'a', buffering=1)
        self._fonts = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='bt-board', daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ the model
    def set_tree(self, root):
        with self._lock:
            self.root = root
            self.order = list(walk(root))
            self.parent = {n.id: (p.id if p is not None else None) for n, _, p in self.order}
            for n, _, _ in self.order:
                if getattr(n, 'bt_guard', False):          # its step's activity shows on the guard's row
                    for c in n.children:
                        self.parent[c.id] = n.id
                        self.stats.setdefault(c.id, {'ok': 0, 'fail': 0, 'last_s': None, 'since': None, 'note': ''})
                self.stats.setdefault(n.id, {'ok': 0, 'fail': 0, 'last_s': None, 'since': None, 'note': ''})
            self._touch()

    def set_autonomy(self, state, message=None):
        with self._lock:
            if state != self.autonomy:
                self.event('autonomy', state=state, was=self.autonomy, message=message or '')
            self.autonomy = state
            if message is not None:
                self.message = message
            self._touch()

    def say(self, message):
        """The activity line under the state."""
        with self._lock:
            self.message = message
            self._touch()

    def set_trial(self, k, n, attempt=1):
        with self._lock:
            self.trial, self.n_trials, self.attempt = int(k), int(n), int(attempt)
            self._touch()

    def count(self, key, by=1):
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + by
            self._touch()

    def trial_done(self, seconds):
        with self._lock:
            self.trial_seconds.append(float(seconds))
            self._touch()

    def failure(self, text):
        with self._lock:
            self.last_failure = text
            self._touch()

    def enter(self, node):
        with self._lock:
            self.active.append(node.id)
            st = self.stats.setdefault(node.id, {'ok': 0, 'fail': 0, 'last_s': None, 'since': None, 'note': ''})
            st['since'] = time.time()
            st['note'] = ''
            self.event('enter', step=node.name)
            self._touch()

    def leave(self, node, ok, note=''):
        with self._lock:
            if node.id in self.active:
                self.active.remove(node.id)
            st = self.stats[node.id]
            dt = time.time() - st['since'] if st['since'] else None
            st['last_s'], st['since'], st['note'] = dt, None, note
            st['ok' if ok else 'fail'] += 1
            self.event('leave', step=node.name, ok=bool(ok), seconds=None if dt is None else round(dt, 3), note=note)
            self._touch()

    def note(self, node, text):
        with self._lock:
            if node is not None and node.id in self.stats:
                self.stats[node.id]['note'] = text
                self._touch()

    def image(self, name, bgr):
        """A live image beside the tree (the in-hand CAD fit): saved and sent from the board's thread,
        latest-only per name."""
        if bgr is None:
            return
        with self._lock:
            self._images[str(name)] = np.ascontiguousarray(bgr)
        self._wake()

    def event(self, kind, **fields):
        """One line of events.jsonl: the run's timeline."""
        rec = {'t': datetime.now().isoformat(timespec='milliseconds'), 'trial': self.trial,
               'attempt': self.attempt, 'kind': kind}
        rec.update(fields)
        try:
            self._events.write(json.dumps(rec, default=str) + '\n')
        except (OSError, ValueError):
            pass

    def _touch(self):
        self._version += 1

    def _wake(self):
        self._version += 1

    # ------------------------------------------------------------------ the derived view
    def _running_ids(self):
        ids = set()
        for i in self.active:
            while i is not None and i not in ids:
                ids.add(i)
                i = self.parent.get(i)
        return ids

    def status_of(self, node, running=None):
        if getattr(node, 'bt_enabled', True) is False:
            return 'off'
        if node.id in (running if running is not None else self._running_ids()):
            return 'running'
        if getattr(node, 'bt_guard', False):       # question false: not this branch -- not a failure
            cond, step = node.children
            if getattr(cond.status, 'name', '') == 'FAILURE':
                return 'idle'
            node = step
        if getattr(node, 'bt_skipped', False):     # not run: the run had been stopped
            return 'idle'
        s = getattr(node.status, 'name', str(node.status)).upper()
        return {'SUCCESS': 'success', 'FAILURE': 'failure', 'RUNNING': 'running'}.get(s, 'idle')

    def snapshot(self):
        with self._lock:
            running = self._running_ids()
            now = time.time()
            nodes = []
            for n, d, _ in self.order:
                guard = getattr(n, 'bt_guard', False)
                st = self.stats.get(n.children[1].id if guard else n.id, {})
                live = now - st['since'] if st.get('since') else None
                name = f'{n.children[0].name}  →  {n.children[1].name}' if guard else n.name
                nodes.append({'name': name, 'kind': 'guard' if guard else kind_of(n), 'depth': d,
                              'status': self.status_of(n, running),
                              'seconds': live if live is not None else st.get('last_s'), 'live': live is not None,
                              'ok': st.get('ok', 0), 'fail': st.get('fail', 0),
                              'note': st.get('note') or getattr(n, 'bt_note', '') or ''})
            done = len(self.trial_seconds)
            per = float(np.mean(self.trial_seconds)) if self.trial_seconds else None
            left = max(self.n_trials - done, 0)
            return {'t': datetime.now().isoformat(timespec='seconds'), 'title': self.title, 'mode': self.mode,
                    'tags': list(self.tags), 'autonomy': self.autonomy, 'message': self.message,
                    'trial': self.trial, 'n_trials': self.n_trials, 'attempt': self.attempt,
                    'counts': dict(self.counts), 'trials_done': done, 'seconds_per_trial': per,
                    'eta_s': per * left if per is not None and left else None, 'elapsed_s': now - self.t0,
                    'last_failure': self.last_failure, 'run_dir': self.out_dir,
                    'run_dir_abs': os.path.abspath(self.out_dir), 'nodes': nodes}

    def panel_lines(self, snap):
        """The live viewer's panel: the mode and state, the trial, what is happening."""
        mode = MODES.get(snap['mode'], (snap['mode'].upper(), None))[0]
        state = AUTONOMY.get(snap['autonomy'], (snap['autonomy'].upper(), None))[0]
        c = snap['counts']
        lines = [f"{mode}{' / ' + ' / '.join(snap['tags']) if snap['tags'] else ''}: {state}",
                 f"trial {snap['trial']}/{snap['n_trials']} (attempt {snap['attempt']}): {c['assembled']} assembled, "
                 f"{c['failed']} failed, {c['recoveries']} recoveries, {c['interventions']} interventions"]
        run = [n['name'] for n in snap['nodes'] if n['status'] == 'running' and n['kind'] in ('action', 'condition')]
        if run:
            lines.append('now: ' + ' > '.join(run))
        if snap['message']:
            lines.append(snap['message'][:110])
        return lines

    # ------------------------------------------------------------------ drawing
    def _font(self, size, bold=False):
        from PIL import ImageFont
        if self._fonts is None:
            self._fonts = {}
            try:
                import matplotlib.font_manager as fm
                self._paths = (fm.findfont('DejaVu Sans'), fm.findfont('DejaVu Sans:bold'))
            except Exception:                      # noqa: BLE001
                self._paths = (None, None)
        key = (size, bold)
        if key not in self._fonts:
            path = self._paths[1 if bold else 0]
            try:
                self._fonts[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            except OSError:
                self._fonts[key] = ImageFont.load_default()
        return self._fonts[key]

    def render(self, snap=None, blink=False):
        """The board as an RGB image (numpy, H x W x 3)."""
        from PIL import Image, ImageDraw
        snap = snap or self.snapshot()
        W, ROW, TOP = 1120, 27, 176
        H = TOP + ROW * len(snap['nodes']) + 52
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        f_title, f_big, f_body, f_small = self._font(19), self._font(30, True), self._font(17), self._font(14)
        f_bold = self._font(17, True)

        # ---- the header: mode, state, what is happening, the trial count
        d.rectangle([0, 0, W, TOP - 14], fill=PANEL)
        d.text((20, 12), f"{snap['title']}  ·  behavior tree", fill=DIM, font=f_title)
        x = 20
        mlabel, mcol = MODES.get(snap['mode'], (snap['mode'].upper(), (90, 90, 90)))
        for label, col in [(mlabel, mcol)] + [(t, (95, 80, 140)) for t in snap['tags']]:
            w = d.textlength(label, font=f_bold) + 22
            d.rounded_rectangle([x, 44, x + w, 74], radius=8, fill=col)
            d.text((x + 11, 49), label, fill=(245, 245, 245), font=f_bold)
            x += w + 10
        slabel, scol = AUTONOMY.get(snap['autonomy'], (snap['autonomy'].upper(), (200, 200, 200)))
        if snap['autonomy'] in ('intervention', 'waiting') and blink:
            w = d.textlength(slabel, font=f_big) + 24
            d.rounded_rectangle([x + 6, 38, x + 6 + w, 80], radius=8, fill=scol)
            d.text((x + 18, 41), slabel, fill=BG, font=f_big)
        else:
            d.text((x + 18, 41), slabel, fill=scol, font=f_big)
        msg = snap['message'] or ''
        for i, line in enumerate(self._wrap(d, msg, f_body, W - 440)[:2]):
            d.text((20, 92 + 24 * i), line, fill=TEXT, font=f_body)
        if snap['last_failure']:
            d.text((20, 140), ('last failure: ' + snap['last_failure'])[:120], fill=STATUS_COL['failure'], font=f_small)
        c = snap['counts']
        rx = W - 400
        d.text((rx, 12), f"trial {snap['trial']} / {snap['n_trials']}" +
               (f"   attempt {snap['attempt']}" if snap['attempt'] > 1 else ''), fill=TEXT, font=self._font(24, True))
        d.text((rx, 48), f"✓ {c['assembled']} assembled    ✗ {c['failed']} failed", fill=TEXT, font=f_body)
        d.text((rx, 72), f"↻ {c['recoveries']} recoveries    ⚑ {c['interventions']} interventions", fill=TEXT, font=f_body)
        done = snap['trials_done']
        rate = f"{100.0 * c['assembled'] / done:.0f}% assembled" if done else 'no trial finished yet'
        per = f" · {_fmt_s(snap['seconds_per_trial'])}/trial" if snap['seconds_per_trial'] else ''
        eta = f" · ETA {_fmt_s(snap['eta_s'])}" if snap['eta_s'] else ''
        d.text((rx, 96), rate + per + eta, fill=DIM, font=f_small)
        d.text((rx, 116), f"elapsed {_fmt_s(snap['elapsed_s'])}", fill=DIM, font=f_small)

        # ---- the tree
        nodes = snap['nodes']
        y0 = TOP
        stack = []                                   # (depth, row) of the open ancestors
        for i, n in enumerate(nodes):
            y = y0 + ROW * i
            depth = n['depth']
            while stack and stack[-1][0] >= depth:
                stack.pop()
            xg = 22 + 26 * depth
            col = STATUS_COL[n['status']]
            if n['status'] == 'running':
                d.rectangle([0, y - 2, W, y + ROW - 4], fill=(30, 48, 72))
            if stack:                                # the connector from the parent
                px = 22 + 26 * stack[-1][0] + 8
                d.line([px, y0 + ROW * stack[-1][1] + 20, px, y + 11], fill=LINE, width=1)
                d.line([px, y + 11, xg - 3, y + 11], fill=LINE, width=1)
            glyph = GLYPH.get(n['kind'], '▸')
            d.ellipse([xg - 2, y + 1, xg + 18, y + 21], outline=col, width=2,
                      fill=col if n['status'] in ('running', 'failure') else None)
            d.text((xg + 3 if glyph != '?' else xg + 4, y + 1), glyph,
                   fill=BG if n['status'] in ('running', 'failure') else col, font=f_small)
            name = n['name'] + ('  (off)' if n['status'] == 'off' else '')
            fnt = f_bold if n['kind'] in ('sequence', 'selector', 'attempts', 'trials') or n['status'] == 'running' else f_body
            tcol = TEXT if n['status'] == 'running' else col if n['status'] in ('failure', 'off') else \
                (200, 225, 205) if n['status'] == 'success' else (170, 175, 183)
            d.text((xg + 26, y + 1), name, fill=tcol, font=fnt)
            if n['note']:
                nx = xg + 34 + d.textlength(name, font=fnt)
                note = n['note']
                while note and d.textlength(note, font=f_small) > W - 290 - nx:
                    note = note[:-2] + '…' if len(note) > 2 else ''
                d.text((nx, y + 3), note, fill=DIM, font=f_small)
            if n['seconds'] is not None and n['kind'] in ('action', 'guard', 'attempts', 'trials', 'sequence', 'selector'):
                d.text((W - 270, y + 3), _fmt_s(n['seconds']) + (' …' if n['live'] else ''), fill=DIM, font=f_small)
            if n['ok'] or n['fail']:
                d.text((W - 170, y + 3), f"✓{n['ok']}", fill=STATUS_COL['success'], font=f_small)
                d.text((W - 110, y + 3), f"✗{n['fail']}", fill=STATUS_COL['failure'] if n['fail'] else DIM, font=f_small)
            stack.append((depth, i))
        # ---- the footer
        fy = H - 40
        d.line([0, fy - 6, W, fy - 6], fill=LINE)
        keys = 'pause / stop: the live viewer\'s Behavior tree window, or touch request_pause / request_stop here'
        kx = W - 20 - d.textlength(keys, font=f_small)
        run = f"run: {snap['run_dir']}"
        while run and d.textlength(run, font=f_small) > kx - 40:
            run = 'run: …' + run[7:]
        d.text((20, fy), run, fill=DIM, font=f_small)
        d.text((kx, fy), keys, fill=DIM, font=f_small)
        return np.asarray(img)

    @staticmethod
    def _wrap(d, text, font, width):
        words, lines, cur = text.split(), [], ''
        for w in words:
            t = (cur + ' ' + w).strip()
            if d.textlength(t, font=font) > width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        return lines

    # ------------------------------------------------------------------ publishing
    def _publish(self, blink=False):
        import cv2
        snap = self.snapshot()
        bgr = np.ascontiguousarray(self.render(snap, blink)[:, :, ::-1])
        ok, png = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            _atomic_write(os.path.join(self.out_dir, 'bt.png'), png.tobytes())
        _atomic_write(os.path.join(self.out_dir, 'bt_status.json'), json.dumps(snap, default=str, indent=1), 'w')
        if self.sender is not None:
            try:
                self.sender.send_image('Behavior tree', bgr)        # the Isaac Sim viewer shows the picture
            except Exception as exc:                # noqa: BLE001 -- a view only
                log.debug('board: not sent to the live viewer (%s)', exc)
        self._send_state(snap)

    def _send_state(self, snap=None):
        """The tree and the state to the live viewer's "Behavior tree" window (drawn there natively) and its
        panel's lines."""
        if self.sender is None:
            return
        snap = snap or self.snapshot()
        try:
            self.sender.send_overlay('behavior_tree', [], text=[], bt=snap)
            self.sender.send_overlay('autonomy', [], text=self.panel_lines(snap))
            self._last_sent = time.monotonic()
        except Exception as exc:                    # noqa: BLE001 -- a view only
            log.debug('board: not sent to the live viewer (%s)', exc)

    def _publish_images(self):
        import cv2
        with self._lock:
            images, self._images = self._images, {}
        for name, bgr in images.items():
            ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                _atomic_write(os.path.join(self.out_dir, f'{name}_live.jpg'), jpg.tobytes())
            if self.sender is not None:
                try:
                    self.sender.send_image(name.replace('_', ' '), bgr)
                except Exception as exc:            # noqa: BLE001
                    log.debug('board: image %s not sent (%s)', name, exc)

    def _run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self._images:
                    self._publish_images()
                # Redrawn on a change; else only to blink a state that wants a person (twice a second) or to
                # move a running step's clock on (once a second). It runs in the task's process, beside the
                # servo loops: kept to the rate the eye needs.
                blinking = self.autonomy in ('intervention', 'waiting')
                due = (blinking and t0 - self._last_draw > 0.5) or (self.active and t0 - self._last_draw > 1.0)
                if self._version != self._drawn or due:
                    self._drawn = self._version
                    self._last_draw = t0
                    self._publish(blink=blinking and int(time.time() * 2) % 2 == 0)
                elif t0 - self._last_sent > 2.0:
                    self._send_state()             # a viewer started (or restarted) since: it catches up
            except Exception as exc:                # noqa: BLE001 -- never take the run down
                log.debug('board: not drawn (%s)', exc)
            self._stop.wait(max(0.05, self.period - (time.monotonic() - t0)))

    def flush(self):
        """Draw now (the final state, before the run ends)."""
        try:
            if self._images:
                self._publish_images()
            self._publish()
        except Exception as exc:                    # noqa: BLE001
            log.debug('board: final draw failed (%s)', exc)

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.flush()
        try:
            self._events.close()
        except OSError:
            pass

    # ------------------------------------------------------------------ pause / stop requests (files)
    def requested(self, which):
        return os.path.exists(os.path.join(self.out_dir, which))

    def withdraw(self, which):
        try:
            os.remove(os.path.join(self.out_dir, which))
        except OSError:
            pass
