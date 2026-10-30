"""PickWatcher -- what the wrist camera's depth says about a pick, alongside it. OBSERVES ONLY: it
logs, and draws in the live view; nothing the pickup decides depends on it.

    WHERE THE CONNECTOR IS. At every scan, each detected cable's junction is lifted to 3D by the
        depth of the cable's own pixels next to it, and set against the estimate the pick uses:
        the junction pixel's ray met with the bench plane (ground_pick._project). Reported: the
        horizontal gap between the two, and how high the cable's top stands above the bench.
    THE CABLES' SHAPE. Each cable mask's centreline in 3D (perception.depth.centreline_3d), at
        every scan and once more after the lift.
    THE GRASP. After the lift, one still frame, and what it sees where the picked cable's junction
        was (depth.spot_check): the bench there means the cable went with the gripper; the cable's
        own height means it stayed. Beside GraspRecovery's finger-position verdict, which decides;
        this one only reports.

Configured by `watch:` (configs/bnc_assembly.yaml); needs camera.enable_depth -- without depth it
says so once and does nothing. The live view draws its results from FrameSender overlays named
'connector', 'cables' and 'grasp', with lines for the viewer's panel.
"""

import numpy as np

from .. import log as urlog
from ..perception import depth

log = urlog.get('watch')

# Overlay colours, RGB 0..1.
_DEPTH, _PLANE, _HEADING, _CABLE = (1.0, 0.55, 0.1), (0.95, 0.95, 0.95), (1.0, 0.85, 0.2), (0.2, 0.85, 1.0)
_VERDICT = {'gone': (0.3, 0.9, 0.3), 'still there': (1.0, 0.3, 0.3), 'covered': (0.6, 0.6, 1.0),
            'unseen': (0.6, 0.6, 0.6), 'unclear': (1.0, 0.8, 0.2)}
_SAYS = {'gone': 'picked -- the bench is back where the junction was',
         'still there': 'MISSED -- the cable is still where it was',
         'covered': 'something in front of the spot (the fingers, or what they hold)',
         'unseen': 'the spot is out of view, or has no depth',
         'unclear': 'a surface there that is neither the bench nor the cable'}


class PickWatcher:
    def __init__(self, cfg, camera, detector=None, sender=None):
        w = cfg.section('watch') if cfg.get('watch') else {}
        self.enabled = bool(w.get('enabled', False))
        self.after_lift_on = bool(w.get('after_lift', True))
        self.detect_after = bool(w.get('detect_after_lift', True))
        self.radius = int(w.get('junction_radius_px', 6))
        self.step = float(w.get('centreline_step_px', 8.0))
        self.bench_z = float(cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0
        self.camera, self.detector, self.sender = camera, detector, sender
        self.last = None            # the latest scan: {'cables': [observation, ...]}
        self.verdicts = []          # each after_lift's verdict, in order
        self._warned = False

    # ------------------------------------------------------------------ the scan
    def on_scan(self, frame, cables, projected):
        """A GroundPlaneScanner listener: every detection, with the ground-plane projections."""
        if not self.enabled or not self._has_depth(frame):
            return
        obs = [self.observe(frame, cab, proj) for cab, proj in zip(cables, projected)]
        self.last = {'cables': obs}
        for i, o in enumerate(obs, start=1):
            if 'gap_xy_mm' in o:
                log.info('  cable #%d: junction by depth %.1f mm from the ground-plane estimate, its top '
                         '%.1f mm above the bench; centreline %s', i, o['gap_xy_mm'], o['top_mm'],
                         'none' if o['line'] is None else f"{len(o['line'])} pts, {1000 * _length(o['line']):.0f} mm")
            else:
                log.info('  cable #%d: no depth at the junction', i)
        self._show_scan(obs)

    def observe(self, frame, cab, proj):
        """One cable: its junction lifted by depth, the plane estimate, the centreline in 3D."""
        u, v = cab['junction'][:2]
        mask = cab.get('cable_mask')
        p_depth = depth.lift(frame, (u, v), self.radius, mask) if mask is not None else None
        if p_depth is None:                  # the mask's edge missed the window: any pixel near it
            p_depth = depth.lift(frame, (u, v), self.radius)
        p_plane = None if proj is None else np.asarray(proj['junction'], dtype=float)
        line = depth.centreline_3d(frame, mask, self.step) if mask is not None else None
        o = {'uv': (float(u), float(v)), 'p_depth': p_depth, 'p_plane': p_plane, 'line': line,
             'heading': None if proj is None else np.asarray(proj['heading'], dtype=float)}
        if p_depth is not None and p_plane is not None:
            o['gap_xy_mm'] = 1000.0 * float(np.hypot(*(p_depth - p_plane)[:2]))
            o['top_mm'] = 1000.0 * float(p_depth[2] - self.bench_z)
        return o

    # ------------------------------------------------------------------ after the lift
    def after_lift(self, selection):
        """One still frame from the lift pose: what it sees where the picked cable's junction was.
        `selection` is the scanner's chosen junction (ground-plane, base frame). The verdict, or
        None when there is nothing to check."""
        if not (self.enabled and self.after_lift_on) or self.last is None or selection is None:
            return None
        picked = _nearest(self.last['cables'], np.asarray(selection, dtype=float))
        if picked is None or picked['p_depth'] is None:
            log.info('Grasp (camera): no depth junction for the picked cable; nothing to check.')
            return None
        frame = self.camera.capture()
        if not self._has_depth(frame):
            return None
        verdict, info = depth.spot_check(frame, picked['p_depth'], self.bench_z, self.radius)
        self.verdicts.append(verdict)
        seen = '--' if info['z_seen'] is None else f"{1000 * (info['z_seen'] - self.bench_z):.1f} mm"
        log.info('Grasp (camera): %s (surface seen %s above the bench where the junction stood %.1f mm '
                 'above it).', _SAYS[verdict], seen, 1000 * (info['expected_z'] - self.bench_z))
        self._send('grasp', [{'type': 'points', 'points': [picked['p_depth'].tolist()],
                              'color': list(_VERDICT[verdict]), 'radius': 0.006}],
                   [f'grasp (camera): {_SAYS[verdict]}'])
        if self.detect_after and self.detector is not None:
            cables = self.detector.detect_all(frame, 8)
            lines = [depth.centreline_3d(frame, c['cable_mask'], self.step) for c in cables
                     if c.get('cable_mask') is not None]
            self._show_lines([ln for ln in lines if ln is not None], 'after the lift')
        return verdict

    # ------------------------------------------------------------------ the live view
    def _show_scan(self, obs):
        items, text = [], []
        for i, o in enumerate(obs, start=1):
            if o['p_depth'] is not None:
                items.append({'type': 'points', 'points': [o['p_depth'].tolist()], 'color': list(_DEPTH), 'radius': 0.004})
            if o['p_plane'] is not None:
                items.append({'type': 'points', 'points': [o['p_plane'].tolist()], 'color': list(_PLANE), 'radius': 0.003})
                if o['heading'] is not None and np.linalg.norm(o['heading']) > 0:
                    tip = o['p_plane'] + 0.03 * o['heading'] / np.linalg.norm(o['heading'])
                    items.append({'type': 'line', 'points': [o['p_plane'].tolist(), tip.tolist()], 'color': list(_HEADING)})
            if 'gap_xy_mm' in o:
                text.append(f"connector #{i}: depth vs plane {o['gap_xy_mm']:.1f} mm, top {o['top_mm']:.1f} mm up")
        self._send('connector', items, text or ['connector: no depth at the junctions'])
        self._show_lines([o['line'] for o in obs if o['line'] is not None], 'at the scan')

    def _show_lines(self, lines, when):
        items = [{'type': 'line', 'points': ln.tolist(), 'color': list(_CABLE)} for ln in lines]
        lengths = ', '.join(f'{1000 * _length(ln):.0f}' for ln in lines)
        self._send('cables', items, [f'cables {when}: {len(lines)} traced in 3D' + (f' ({lengths} mm)' if lines else '')])

    def _send(self, name, items, text):
        if self.sender is not None:
            self.sender.send_overlay(name, items, text)

    def _has_depth(self, frame):
        if frame.depth is not None and frame.T_base_cam is not None:
            return True
        if not self._warned:
            self._warned = True
            log.warning('watch: the camera frames carry no %s; nothing to watch (camera.enable_depth).',
                        'depth' if frame.depth is None else 'pose')
        return False


def _length(line):
    return float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))


def _nearest(observations, point, within_m=0.05):
    """The observation whose plane junction is nearest `point`, if within `within_m`."""
    best, best_d = None, within_m
    for o in observations:
        if o['p_plane'] is None:
            continue
        d = float(np.linalg.norm(o['p_plane'][:2] - point[:2]))
        if d < best_d:
            best, best_d = o, d
    return best
