"""WorldModel -- what the wrist camera has seen, as the collision world EVERY move is checked against.

    world = WorldModel.for_robot(cfg, robot)      # None: switched off (robot.world.enabled false)
    camera.listeners.append(world)                # every frame the task captures goes in (Robot does this)
    world.add_views(views)                        # [(joints, Frame)] kept from an earlier run
    scene, version = world.snapshot()             # for the planner (PathCheck reads it before each move)

WHY (the user, 2026-11-02): reconfiguring to the pick pose, the arm swept into the structure the
sockets are mounted on -- a plain move_j, checked against the base plane only. The camera had just
looked at that structure from a dozen sides (the socket's 6D views); the live view drew it; the
planner knew nothing of it. Now the frames that build the live view's camera layer build this too.

THE SCENE is perception/scene.py's OccupancyScene: voxels in the cell's box, each new frame replacing
what it sees (a cable picked up leaves no ghost), solid below what is seen, the robot and the gripper
left out of every frame. A frame goes in only if the arm was STILL at its capture -- the camera's pose
now within `still_mm` / `still_deg` of the pose it was stamped with -- since one taken on the move is
placed by a pose read after it, and would put the scene where it is not.

IT REMEMBERS, AS THE LIVE VIEW'S CAMERA LAYER DOES (the user, 2026-11-02: "the scene had already been
scanned ... and was showing in the gaussian splat, yet the robot still planned through it"). The
viewer keeps every frame it is sent, across runs; a world built only from the run's own views did
not -- the collided run's 12 socket close-ups covered ~23 x 18 cm of the structure the splat showed
whole. So the world is SAVED (data/world/world_scene.npz in the repo, `save_s` after a change and at
exit) and every run starts from it; and the first time, and for whatever was saved since, it takes in
every view set kept on disk (robot.world.seed: globs under the repo's data/ and sim/data/ -- the 6D
views, splat scans, captures), oldest first. No survey is needed: what the camera has seen, in any
run, is avoided. What has since moved away stays until a frame sees through where it was -- or the
file is deleted (robot.world.reset: true starts empty).

THE LIVE VIEW can draw the occupied voxels (robot.world.overlay: true; OFF by default, the user's
choice 2026-11-02) as an overlay ('collision_world', red points): what the planner avoids, over the
camera layer it came from. At most `overlay_points` of them, at most once every `overlay_s`. Off, an
empty one is sent at start, clearing what an earlier run drew in a viewer still open.
"""

import atexit
import glob
import os
import threading
import time

import numpy as np

from .. import log as urlog

log = urlog.get('world')

OVERLAY = 'collision_world'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORLD_FILE = os.path.join(REPO, 'data', 'world', 'world_scene.npz')
# What the saved world's readings were filtered with. A world saved under another stamp is REBUILT from
# the view sets: what it holds cannot be re-filtered. 'v3-gripper': scene.py drops readings through the
# bench 2F-85 V3's outer finger links (tool_geometry.gripper_view_spheres) -- before it, the right outer
# finger, read 0.34-0.5 m away by the D415, stood in the world where the stand-off starts (2026-11-03).
FILTER = 'v3-gripper+free+on+held+slab'  # +free: seen-through voxels are never filled below (perception/scene.py)
# +slab (2026-11-12): the open gripper's whole slab and the pixels round it (scene.py GRIPPER_SLAB_*) -- from its
# own mount the D405 reads the outer finger links' far ends, which stood in the world where the gripper had been.
# +on (2026-11-11): readings ON the gripper's spheres dropped too -- the D405 sees its own outer fingers.
# +held (2026-11-11): frames taken while a part is held (and saved views of HELD_VIEWS) clear, never occupy --
# the cable hanging from the held plug stood in the world where the stand-off puts the gripper.
# Saved view sets taken with a part in the hand (their directory names): read into the world carve-only.
HELD_VIEWS = ('_refine', 'in_hand', 'held')
SEED = ['pose6d/*', 'experiments/splat_scan_*', 'experiments/fixture_survey_*', 'experiments/capture_rgbd_*']


class WorldModel:
    def __init__(self, cfg, robot, still_mm=3.0, still_deg=1.0, stride=2, overlay_points=15000, overlay_s=1.0,
                 path=WORLD_FILE, save_s=5.0, overlay=False):
        from ..perception.scene import OccupancyScene
        self.cfg, self.robot = cfg, robot
        self.scene = OccupancyScene.for_cell(cfg)
        self.path, self.save_s, self.overlay = path, float(save_s), bool(overlay)
        self._saved, self._t_saved = 0, 0.0
        self.still_m, self.still_rad = float(still_mm) / 1000.0, np.radians(float(still_deg))
        self.stride, self.overlay_points, self.overlay_s = int(stride), int(overlay_points), float(overlay_s)
        self.version = 0
        self.skipped_moving = 0
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._sent, self._t_sent = -1, 0.0

    @classmethod
    def for_robot(cls, cfg, robot):
        """The cell's world (robot.world: {enabled, still_mm, still_deg, stride, overlay_points}),
        or None: switched off, or a robot with no plain-move check to give it to."""
        c = dict(cfg.get_path('robot.world') or {})
        if not c.get('enabled', True) or getattr(robot.arm, 'path_check', None) is None:
            return None
        path = c.get('file') or WORLD_FILE
        if not os.path.isabs(path):
            path = os.path.join(REPO, path)
        if os.environ.get('CABLE_ASSEMBLY_WORLD_DIR') and not c.get('file'):     # the tests: a world of their own
            path = os.path.join(os.environ['CABLE_ASSEMBLY_WORLD_DIR'], 'world_scene.npz')
        return cls(cfg, robot, c.get('still_mm', 3.0), c.get('still_deg', 1.0), c.get('stride', 2),
                   c.get('overlay_points', 15000), c.get('overlay_s', 1.0), path, c.get('save_s', 5.0),
                   c.get('overlay', False))

    # ------------------------------------------------------------------ frames in
    def _frame(self, frame, q):
        """A camera Frame as the scene's frame dict (the robot left out by its joints and tool0)."""
        robot = self.robot
        T_base_cam = np.asarray(frame.T_base_cam, dtype=float)
        T_base_tool0 = T_base_cam @ np.linalg.inv(robot.T_tool0_cam)
        return {'depth_m': np.asarray(frame.depth, dtype=np.float32), 'K': np.asarray(frame.K, dtype=float),
                'T_base_cam': T_base_cam, 'T_base_tool0': T_base_tool0,
                'T_tool0_fingertip': robot.T_tool0_fingertip, 'joints_rad': np.asarray(q, dtype=float),
                'robot': robot.arm.model.NAME,
                # HOLDING A PART: the frame clears what it sees through and occupies nothing (FILTER +held)
                'carve_only': getattr(robot, 'held', None) is not None}

    def __call__(self, frame):
        """A camera listener: every frame captured, in -- if it has depth and the arm was still."""
        if getattr(frame, 'depth', None) is None or getattr(frame, 'T_base_cam', None) is None:
            return
        arm = self.robot.arm
        q = np.asarray(arm.q(), dtype=float)
        T_now = arm.fk(q) @ self.robot.T_tool0_cam
        T_then = np.asarray(frame.T_base_cam, dtype=float)
        moved = np.linalg.norm(T_now[:3, 3] - T_then[:3, 3])
        turned = np.arccos(np.clip((np.trace(T_now[:3, :3].T @ T_then[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        if moved > self.still_m or turned > self.still_rad:
            self.skipped_moving += 1
            return
        self._add(self._frame(frame, q))

    def seed(self):
        """The world as last saved, plus every view set kept on disk since (all of them, the first
        time), oldest first -- see the module docstring. The frames taken in."""
        from ..perception.camera import Frame
        from ..perception.scene import OccupancyScene
        c = dict(self.cfg.get_path('robot.world') or {})
        since = 0.0
        if not c.get('prior', True):
            # NO PRIOR (the user, 2026-11-03: "no prior collision other than the robot and its ground
            # plane"): nothing saved is loaded and no saved view set read; the world holds only what
            # this run's camera sees.
            log.info('WORLD: robot.world.prior false -- starting empty (the robot and the base plane only); '
                     'this run\'s frames are added as it sees them.')
            atexit.register(self.save, True)
            return 0
        if c.get('reset'):
            log.warning('WORLD: robot.world.reset -- starting empty (%s is overwritten).', self.path)
        elif os.path.isfile(self.path):
            try:
                saved = OccupancyScene.load(self.path)
                with np.load(self.path) as z:
                    stamp = str(z['filter']) if 'filter' in z.files else None
                if stamp != FILTER:
                    log.warning('WORLD: %s was filtered as %r, not %r -- rebuilt from the view sets.',
                                self.path, stamp, FILTER)
                elif saved.shape == self.scene.shape and np.allclose(saved.lo, self.scene.lo) and saved.voxel == self.scene.voxel:
                    self.scene, since = saved, os.path.getmtime(self.path)
                    log.info('WORLD: %d voxels from %s (%d frames, saved %s).', int(saved.occ.sum()), self.path,
                             saved.frames, time.strftime('%Y-%m-%d %H:%M', time.localtime(since)))
                else:
                    log.warning('WORLD: %s is for another box (the cell\'s scene: changed) -- rebuilt.', self.path)
            except Exception as exc:                  # noqa: BLE001 -- rebuilt from the views
                log.warning('WORLD: %s unreadable (%s) -- rebuilt.', self.path, exc)
        roots = []
        default = ((os.environ['CABLE_ASSEMBLY_WORLD_DIR'],) if os.environ.get('CABLE_ASSEMBLY_WORLD_DIR') else
                   (str(self.cfg.get('data_dir', 'data')), os.path.join(REPO, 'data'), os.path.join(REPO, 'sim', 'data')))
        for r in c.get('roots') or default:
            if os.path.isdir(r) and os.path.realpath(r) not in [os.path.realpath(x) for x in roots]:
                roots.append(r)
        dirs = {os.path.realpath(d) for root in roots for pattern in c.get('seed', SEED)
                for d in glob.glob(os.path.join(root, pattern)) if os.path.isdir(d) and os.path.getmtime(d) > since}
        total = 0
        if dirs:
            log.info('WORLD: reading %d saved view set%s into the world (once; robot.world.prior: false skips it).',
                     len(dirs), '' if len(dirs) == 1 else 's')
        for d in sorted(dirs, key=os.path.getmtime):
            views = []
            for path in sorted(glob.glob(os.path.join(d, '*.npz'))):
                try:
                    with np.load(path) as z:
                        if not {'joints_rad', 'depth_m', 'K', 'T_base_cam', 'rgb'} <= set(z.files):
                            continue
                        views.append((np.asarray(z['joints_rad'], dtype=float),
                                      Frame(np.ascontiguousarray(z['rgb'][..., ::-1]), np.asarray(z['K'], dtype=float), None,
                                            0.0, np.asarray(z['T_base_cam'], dtype=float),
                                            np.asarray(z['depth_m'], dtype=np.float32))))
                except Exception as exc:              # noqa: BLE001 -- one bad file is not the world
                    log.debug('WORLD: %s skipped: %s', path, exc)
            if views:
                self.add_views(views, f'views from {d}', quiet=True,
                               carve_only=any(k in os.path.basename(d) for k in HELD_VIEWS))
                total += len(views)
        log.info('WORLD: %d voxels occupied (%d frames taken in from %d view sets saved since).',
                 int(self.scene.occ.sum()), total, len(dirs))
        self.save(force=True)
        atexit.register(self.save, True)
        self.send_overlay(force=True)
        return total

    def add_views(self, views, label='views', quiet=False, carve_only=False):
        """[(joints, Frame)] -- views kept from before (a saved socket scene) -- in, in order. `carve_only`: views
        taken with a part in the hand (HELD_VIEWS) -- they clear, and occupy nothing."""
        n = 0
        for q, frame in views:
            if getattr(frame, 'depth', None) is not None and getattr(frame, 'T_base_cam', None) is not None:
                f = self._frame(frame, q)
                f['carve_only'] = bool(carve_only)
                self._add(f, send=False)
                n += 1
        (log.debug if quiet else log.info)('WORLD: %d %s in -- %d voxels occupied.', n, label, int(self.scene.occ.sum()))
        if not quiet:
            self.save()
            self.send_overlay(force=True)

    def _add(self, f, send=True):
        with self._lock:
            self.scene.add(f, stride=self.stride)
            self.version += 1
        if send:
            self.save()
            self.send_overlay()

    def save(self, force=False):
        """The world to self.path (atomically), at most every save_s unless `force`."""
        now = time.monotonic()
        with self._lock:
            if self.version == self._saved and not force:
                return
            if not force and now - self._t_saved < self.save_s:
                return
            occ, version = self.scene.occ.copy(), self.version
            tmp = self.scene.__class__(self.scene.lo, self.scene.hi, self.scene.voxel, self.scene.margin,
                                       fill_below=self.scene.fill_below)
            tmp.occ, tmp.frames = occ, self.scene.frames
            tmp.shape, tmp.hi, tmp.free = self.scene.shape, self.scene.hi, self.scene.free.copy()
        # ONE WRITER AT A TIME, each with its own part file (2026-11-11: the camera's listener and the run saved
        # at once into the same `.part.npz`; the second rename found it gone -- "No such file or directory").
        with self._save_lock:
            if version <= self._saved and not force:
                return                                # a newer one was written meanwhile
            part = '%s.%d.%d.part.npz' % (self.path, os.getpid(), threading.get_ident())
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                tmp.save(part, filter=np.array(FILTER))
                os.replace(part, self.path)
                self._saved, self._t_saved = version, now
            except Exception as exc:                  # noqa: BLE001 -- remembering is best-effort
                log.warning('WORLD: not saved to %s: %s', self.path, exc)
                try:
                    os.unlink(part)
                except OSError:
                    pass

    def snapshot(self):
        """(a copy of the scene, its version) -- consistent, for a planner in another thread."""
        import copy
        with self._lock:
            return copy.deepcopy(self.scene), self.version

    # ------------------------------------------------------------------ the live view
    def send_overlay(self, force=False):
        """The occupied voxels to the live view, at most every overlay_s (or now, `force`)."""
        tx = getattr(self.robot, 'live_frames', None)
        if tx is None or not hasattr(tx, 'send_overlay'):
            return
        if not self.overlay:
            if force and self._sent != 'cleared':          # once: clear an earlier run's from the viewer
                try:
                    tx.send_overlay(OVERLAY, [], [])
                    self._sent = 'cleared'
                except Exception as exc:              # noqa: BLE001 -- drawing only
                    log.debug('WORLD: overlay not cleared: %s', exc)
            return
        now = time.monotonic()
        with self._lock:
            if self.version == self._sent or (not force and now - self._t_sent < self.overlay_s):
                return
            idx = np.argwhere(self.scene.occ)
            pts = self.scene.centres(idx)
            version, voxel = self.version, self.scene.voxel
        if len(pts) > self.overlay_points:
            pts = pts[np.random.default_rng(0).choice(len(pts), self.overlay_points, replace=False)]
        try:
            tx.send_overlay(OVERLAY, [{'type': 'points', 'points': np.round(pts, 4).tolist(), 'color': [1.0, 0.25, 0.2],
                                       'radius': 0.35 * voxel}],
                            [f'collision world: {len(idx)} voxels from {self.scene.frames} frames '
                             f'({self.skipped_moving} taken on the move left out)'])
            self._sent, self._t_sent = version, now
        except Exception as exc:          # noqa: BLE001 -- drawing only
            log.debug('WORLD: overlay not sent: %s', exc)
