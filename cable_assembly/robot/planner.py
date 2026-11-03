"""Planner -- collision-free joint moves with cuMotion, round what the wrist camera has seen.

    planner = Planner.for_cell(cfg)                 # None without cuMotion or the iiwa description
    path = planner.plan(q_from, q_to, scene)        # a JointPath, or None (planner.why says why)
    path, q = planner.plan_to_pose(q_from, T, scene, limits)   # to a tool0 pose, over the arm angle
    planner.execute(arm, path, label)               # streamed where the arm can, else waypoints
    arm.path_check = PathCheck.for_cell(cfg)        # EVERY plain move_j / move_l, checked before it moves

THE ROBOT is the iiwa 14's cuMotion description (robot/description/iiwa14: the arm's collision
spheres, written by another tool) with the TOOL added on tool0 here, in memory: the camera mount
and the D415 from their STLs, and the 2F-85 as an envelope of spheres (robot/tool_geometry.py). So
the plan keeps the gripper and the camera off the scene, not just the arm. What the gripper HOLDS
can be added the same way (`held`: spheres in tool0, e.g. tool_geometry.rod_spheres for a connector
and the cable leaving it).

THE WORLD is, ALWAYS, the plane the robot's base stands on (z = robot.base_plane_z_mm in base_link,
0 by default: the base's underside) -- with or without a scene. Then, when there is one, an
OccupancyScene (perception/scene.py) -- the pick's orbit views, the socket's, any RGB-D frames -- as
cuMotion's signed-distance obstacle, plus the bench as a box under it. What no frame saw counts as
free: plan round what has been looked at. The base (link_0) is left out of the world check: it is
bolted to that plane (its spheres reach 31 mm below it) and never moves, so it can meet nothing.

THE PLAN is cuMotion's trajectory optimizer to a joint-space target, then checked again, sample by
sample, against the same world (min clearance reported); a plan that fails the check is refused.
Its timing is cuMotion's (the XRDF's limits), stretched until no joint passes the move's caps.

THE PLAIN MOVES (PathCheck): every move_j's straight joint line and every move_l's line is checked
against the same base plane before the arm moves; a blocked move_j goes round (a plan, as above),
a blocked move_l is refused.
"""

import os

import numpy as np
import yaml

from .. import log as urlog
from . import tool_geometry

log = urlog.get('planner')

DESC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'description', 'iiwa14')
# Links bolted to the base plane: out of the WORLD collision check (kept for self-collision).
BOLTED = ('link_0',)
# The base plane as cuMotion sees it: a slab this big (x, y, thickness), its top on the plane.
BASE_PLANE_SIZE_M = (4.0, 4.0, 0.2)


class JointPath:
    """A planned move: times (N,) from 0, joint positions (N, dof), and its clearance (m)."""

    def __init__(self, times, q, clearance):
        self.times, self.q, self.clearance = np.asarray(times, float), np.asarray(q, float), float(clearance)

    @property
    def duration(self):
        return float(self.times[-1])


def arm_limits(cfg):
    """(joint acceleration rad/s^2, jerk rad/s^3) the arm moves under: the `speed:` block's
    acceleration, reached over robot.fri.accel_ramp_s -- what a move_j is held to -- or None."""
    if cfg is None or not cfg.get('speed'):
        return None
    from .arm import parse_limits
    accel = float(parse_limits(cfg.section('speed'))[1])
    return accel, accel / float(cfg.get_path('robot.fri.accel_ramp_s', 0.1) or 0.1)


def robot_with_tool(cfg=None, xrdf_path=None, urdf_path=None, held=(), grip_half_m=None):
    """cuMotion's description of the iiwa 14 with the tool's spheres on tool0 -- and `held`'s,
    [(centre in tool0, radius)], what the gripper holds, its fingers closed `grip_half_m` from the
    middle (tool_geometry.tool_spheres) -- planning to the arm's own acceleration and jerk
    (arm_limits), not the description's 4 rad/s^2 and 80 rad/s^3: a plan to those the driver can
    only follow by catching up, flat out (the jerk)."""
    import cumotion
    xrdf_path = xrdf_path or os.path.join(DESC, 'iiwa14.xrdf')
    urdf_path = urdf_path or os.path.join(DESC, 'iiwa14.urdf')
    with open(xrdf_path) as f:
        doc = yaml.safe_load(f)
    lim = arm_limits(cfg)
    if lim is not None:
        doc['cspace']['acceleration_limits'] = [round(min(lim[0], a), 4) for a in doc['cspace']['acceleration_limits']]
        doc['cspace']['jerk_limits'] = [round(min(lim[1], j), 4) for j in doc['cspace']['jerk_limits']]
    spheres = tool_geometry.tool_spheres(cfg, grip_half_m) + list(held)
    geom = doc['world_collision']['geometry']
    doc['geometry'][geom]['spheres']['tool0'] = [
        {'center': [round(float(v), 5) for v in c], 'radius': round(float(r), 5)} for c, r in spheres]
    # The tool is bolted to link_7 and sits against it: not a self-collision.
    ignore = doc.setdefault('self_collision', {}).setdefault('ignore', {})
    ignore['tool0'] = ['link_7', 'link_6', 'link_5']
    # The WORLD check without the base (BOLTED): the same spheres, as their own group.
    world = dict(doc['geometry'][geom])
    world['spheres'] = {k: v for k, v in world['spheres'].items() if k not in BOLTED}
    doc['geometry'][geom + '_world'] = world
    doc['world_collision']['geometry'] = geom + '_world'
    with open(urdf_path) as f:
        urdf = f.read()
    return cumotion.load_robot_from_memory(yaml.safe_dump(doc, sort_keys=False), urdf), len(spheres)


class Planner:
    def __init__(self, cfg=None, dt=0.004, check_dt=0.01, held=(), grip_half_m=None):
        import cumotion
        self._cm = cumotion
        cumotion.set_log_level(cumotion.LogLevel.ERROR)
        self.held = [(np.asarray(c, dtype=float), float(r)) for c, r in held]
        self.tool = tool_geometry.tool_spheres(cfg, grip_half_m) + self.held
        self.robot, self.n_tool_spheres = robot_with_tool(cfg, held=self.held, grip_half_m=grip_half_m)
        self.bench_z = None if cfg is None else float(cfg.get_path('ground_plane.z_mm', 0.0) or 0.0) / 1000.0
        self.base_plane_z = 0.0 if cfg is None else float(cfg.get_path('robot.base_plane_z_mm', 0.0) or 0.0) / 1000.0
        self.dt, self.check_dt = float(dt), float(check_dt)
        self.why = None

    @classmethod
    def for_cell(cls, cfg, held=(), grip_half_m=None):
        """A Planner for the cell's robot (with `held` on the tool, the fingers closed `grip_half_m`
        from the middle on it), or None (said why) where there is none to be had."""
        if str(cfg.get_path('robot.type', '')) != 'iiwa14':
            log.warning('cuMotion planning: only the iiwa 14 has a description here (robot.type %s).',
                        cfg.get_path('robot.type'))
            return None
        try:
            return cls(cfg, held=held, grip_half_m=grip_half_m)
        except ImportError:
            log.warning('cuMotion planning: no cuMotion in this environment (setup-env.sh step 5).')
        except FileNotFoundError as exc:
            log.warning('cuMotion planning: %s (robot/description/make_iiwa14_xrdf.py writes the XRDF).', exc)
        return None

    def _world(self, scene=None):
        """cuMotion's world: the base plane, always; and `scene` (with the bench under it) if given."""
        cm = self._cm
        world = cm.create_world()
        if scene is not None:
            scene.to_cumotion(bench_z=self.bench_z, world=world)
        plane = cm.create_obstacle(cm.Obstacle.Type.CUBOID)
        plane.set_attribute(cm.Obstacle.Attribute.SIDE_LENGTHS, np.array(BASE_PLANE_SIZE_M, dtype=float))
        world.add_obstacle(plane, cm.Pose3.from_translation(
            np.array([0.0, 0.0, self.base_plane_z - 0.5 * BASE_PLANE_SIZE_M[2]])))
        return world.add_world_view()

    def clearance(self, q, scene=None):
        """The robot's (tool included) signed clearance from the scene at q, m."""
        insp = self._cm.create_robot_world_inspector(self.robot, self._world(scene))
        return float(insp.min_distance_to_obstacle(np.asarray(q, dtype=float)))

    def plan(self, q_from, q_to, scene=None):
        """A collision-free JointPath from q_from to q_to round `scene` (None: the base plane alone),
        or None (self.why)."""
        cm = self._cm
        q_from, q_to = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
        view = self._world(scene)
        insp = cm.create_robot_world_inspector(self.robot, view)
        for name, q in (('start', q_from), ('goal', q_to)):
            if insp.in_collision_with_obstacle(q):
                frame, d, what = self.closest(q, scene)
                self.why = (f'the {name} is in collision ({1000 * insp.min_distance_to_obstacle(q):.0f} mm): '
                            f'{frame} against {what} ({1000 * d:.0f} mm)')
                return None
        opt = cm.create_trajectory_optimizer(cm.create_default_trajectory_optimizer_config(
            robot_description=self.robot, tool_frame_name='tool0', world_view=view))
        res = opt.plan_to_cspace_target(q_from, cm.TrajectoryOptimizer.CSpaceTarget(q_to))
        if res.status() != cm.TrajectoryOptimizer.Results.Status.SUCCESS:
            self.why = f'cuMotion found no path ({res.status()})'
            return None
        traj = res.trajectory()
        dom = traj.domain()
        n = max(2, int(np.ceil((dom.upper - dom.lower) / self.dt)) + 1)
        t = np.linspace(dom.lower, dom.upper, n)
        # THE POSITIONS FROM THE VELOCITY: cuMotion's sampled positions carry numerical wiggle
        # (differentiated, 10-40 rad/s^2 where its plan keeps 1.4) that a position-controlled arm
        # follows as jerk; its analytic velocity is smooth. Integrated from the start, the
        # 0.03-deg drift spread along it, the path lands on the goal exactly.
        from scipy.integrate import cumulative_trapezoid
        v = np.array([np.asarray(traj.eval_all(x)[1], dtype=float).ravel() for x in t])
        q = q_from + cumulative_trapezoid(v, t, axis=0, initial=0.0)
        q -= np.outer((t - t[0]) / max(t[-1] - t[0], 1e-9), q[-1] - q_to)
        q[-1] = q_to
        clear = min(insp.min_distance_to_obstacle(x) for x in q)
        if clear < 0.0:
            self.why = f'the plan passes {-1000 * clear:.0f} mm into an obstacle'
            return None
        self.why = None
        return JointPath(t - t[0], q, clear)

    def _spheres(self, insp, q):
        """cuMotion's own world-collision spheres at q: (centres (N, 3), radii (N,), frame names)."""
        q = np.asarray(q, dtype=float)
        c = np.array([np.asarray(x, dtype=float).ravel() for x in insp.world_collision_sphere_positions(q)])
        r = np.asarray(insp.world_collision_sphere_radii(), dtype=float)
        return c, r, [insp.world_collision_sphere_frame_name(i) for i in range(len(r))]

    def closest(self, q, scene=None):
        """(frame, clearance m, 'the scene' | 'the bench' | 'the base plane') of the sphere nearest an
        obstacle at q."""
        insp = self._cm.create_robot_world_inspector(self.robot, self._world(scene))
        c, r, names = self._spheres(insp, q)
        d = {'the base plane': c[:, 2] - self.base_plane_z - r}
        if scene is not None:
            sdf = scene.sdf()
            ijk = np.floor((c - scene.lo) / scene.voxel).astype(int)
            inside = np.all((ijk >= 0) & (ijk < np.asarray(scene.shape)), axis=1)
            d['the scene'] = np.full(len(c), np.inf)
            d['the scene'][inside] = sdf[tuple(ijk[inside].T)] - r[inside]
            if self.bench_z is not None:
                d['the bench'] = c[:, 2] - self.bench_z - r
        what = min(d, key=lambda w: float(np.min(d[w])))
        k = int(np.argmin(d[what]))
        return names[k], float(d[what][k]), what

    def free_start(self, scene, q, margin_m=0.01, max_margin_m=0.03):
        """Free what the robot fills at `q` -- cuMotion's own collision spheres (links, tool, what it
        holds) -- from `scene`: where the arm stands is not an obstacle, whatever older views said
        (the cable it has since picked up, the arm's own image). The margin grows a centimetre at a
        time, to `max_margin_m`, while the start still touches what is left. Voxels freed."""
        if scene is None:
            return 0
        q = np.asarray(q, dtype=float)
        insp = self._cm.create_robot_world_inspector(self.robot, self._world(scene))
        c, r, _ = self._spheres(insp, q)
        freed, m = 0, float(margin_m)
        while True:
            freed += scene.clear_spheres(c, r + m)
            if m >= max_margin_m or not self._cm.create_robot_world_inspector(
                    self.robot, self._world(scene)).in_collision_with_obstacle(q):
                return freed
            m += 0.01

    def plan_to_pose(self, q_from, T_base_tool0, scene=None, limits=None, max_goals=8, psi_step_deg=10.0):
        """(JointPath, q_goal) to a tool0 pose round `scene` (None: the base plane alone), or (None,
        None) with self.why.

        The goal is chosen, not given: every IK solution over the arm angle (and its branches)
        inside `limits` (the arm's own, margin included), nearest to q_from first; the first that
        is clear of the scene AND that cuMotion reaches wins. So the arm reconfigures -- a
        different elbow, a different wrist -- when the nearest solution is blocked or unreachable."""
        from .kinematics import iiwa14 as K
        lim = K.JOINT_LIMITS if limits is None else np.asarray(limits, dtype=float)
        q_from = np.asarray(q_from, dtype=float)
        cands = []
        for psi in np.radians(np.arange(-180.0, 180.0, psi_step_deg)):
            for q in K.ik_all(T_base_tool0, psi, q_from):
                q = np.asarray(q, dtype=float)
                if np.all((q >= lim[:, 0]) & (q <= lim[:, 1])):
                    cands.append(q)
        if not cands:
            self.why = 'no IK solution inside the joint limits at any arm angle'
            return None, None
        # distinct goals, nearest (largest joint change smallest) first
        cands.sort(key=lambda q: float(np.max(np.abs(q - q_from))))
        goals = []
        for q in cands:
            if all(np.max(np.abs(q - g)) > np.radians(5.0) for g in goals):
                goals.append(q)
        insp = self._cm.create_robot_world_inspector(self.robot, self._world(scene))
        clear = [q for q in goals if not insp.in_collision_with_obstacle(q)]
        if not clear:
            best = max(goals, key=insp.min_distance_to_obstacle)
            frame, _d, what = self.closest(best, scene)
            self.why = (f'all {len(goals)} IK solutions at the goal are in collision (best '
                        f'{1000 * insp.min_distance_to_obstacle(best):.0f} mm: {frame} against {what})')
            return None, None
        whys = []
        for q in clear[:max_goals]:
            path = self.plan(q_from, q, scene)
            if path is not None:
                return path, q
            whys.append(self.why)
        self.why = f'no path to any of {min(len(clear), max_goals)} clear goals ({whys[0]})'
        return None, None

    def execute(self, arm, path, label, caps=None):
        """Drive `path`: the arm's own streamed move_joint_path where it has one (the iiwa), else
        joint moves through it (waypoints no more than 5 deg apart) -- a dry run, the UR."""
        if hasattr(arm, 'move_joint_path'):
            return arm.move_joint_path(path.times, path.q, label=label, caps=caps)
        keep, last = [], path.q[0]
        for q in path.q[1:]:
            if np.max(np.abs(q - last)) >= np.radians(5.0):
                keep.append(q)
                last = q
        keep.append(path.q[-1])
        for k, q in enumerate(keep, start=1):
            if not arm.move_j(q, label=f'{label} ({k}/{len(keep)})', caps=caps):
                return False
        return True


class PathCheck:
    """EVERY PLAIN MOVE against the base plane, before the arm moves: the straight joint line a
    move_j drives and the line a move_l drives (Arm.path_check; the Robot sets one for a cell with a
    cuMotion description). The planned moves above were always checked; the plain ones were not --
    and on 2026-10-31 the reset's move home, a straight joint line from the engage pose (A2 102 deg),
    put the gripper 200 mm into the bench: it hit at ~300 N before the force guard stopped it.

    A MOVE MAY GO NO NEARER THE PLANE THAN ITS ENDS DO. A pick puts the fingertips on the bench --
    cuMotion's spheres 10-15 mm INTO the plane there, on every real pick so far -- so a fixed
    clearance would refuse the grasp. Refused is a path that dips more than `tol_mm` below the
    nearer of its two ends (or of `margin_mm`, for ends that are both well clear) -- its END counted
    no deeper than `max_into_mm` into the plane (the real picks: 16 mm at most), so a move cannot
    make a goal through the bench acceptable by ending there. Its START always counts: where the arm
    already is, it must be able to leave.

    A blocked move_j goes ROUND, planned by cuMotion against the same plane (detour); a blocked
    move_l is refused -- a straight line is what it was asked for."""

    def __init__(self, cfg, margin_mm=10.0, tol_mm=5.0, max_into_mm=25.0):
        self.cfg = cfg
        self.margin, self.tol = float(margin_mm) / 1000.0, float(tol_mm) / 1000.0
        self.max_into = float(max_into_mm) / 1000.0
        self.why = None
        self._held = None                    # (spheres in tool0, grip_half_m): the part in the hand
        self._planner, self._insp, self._unavailable = None, None, False

    def hold(self, spheres, grip_half_m):
        """From now the tool carries `spheres` (a held part, robot/held_part.py), the fingers closed
        `grip_half_m` from the middle on it -- cuMotion's robot is rebuilt with them at the next move."""
        self._held = ([(np.asarray(c, dtype=float), float(r)) for c, r in spheres], float(grip_half_m))
        self._planner, self._insp, self._unavailable = None, None, False

    def release(self):
        """Nothing in the hand: the tool alone, fingers open."""
        if self._held is not None:
            self._held = None
            self._planner, self._insp, self._unavailable = None, None, False

    @classmethod
    def for_cell(cls, cfg):
        """The cell's check (robot.move_check: {enabled, margin_mm, tol_mm, max_into_mm}), or None:
        the UR (no description here), or switched off. cuMotion loads on the first move."""
        c = cfg.get_path('robot.move_check') or {}
        if str(cfg.get_path('robot.type', '')) != 'iiwa14' or not c.get('enabled', True):
            return None
        return cls(cfg, c.get('margin_mm', 10.0), c.get('tol_mm', 5.0), c.get('max_into_mm', 25.0))

    @property
    def planner(self):
        if self._planner is None and not self._unavailable:
            held, half = self._held if self._held is not None else ((), None)
            self._planner = Planner.for_cell(self.cfg, held=held, grip_half_m=half)
            if self._planner is None:
                self._unavailable = True
                log.error('PLAIN MOVES ARE NOT CHECKED against the base plane: no cuMotion planner '
                          '(above says why).')
        return self._planner

    def refuse(self, path):
        """Why the arm may not drive `path` (joint vectors, start to end), or None."""
        pl = self.planner
        if pl is None or len(path) == 0:
            return None
        if self._insp is None:
            self._insp = pl._cm.create_robot_world_inspector(pl.robot, pl._world())
        d = np.array([self._insp.min_distance_to_obstacle(np.asarray(q, dtype=float)) for q in path])
        k = int(np.argmin(d))
        if d[k] >= min(d[0], max(d[-1], -self.max_into), self.margin) - self.tol:
            return None
        frame, dist, what = pl.closest(path[k])
        return (f'{frame} would pass {abs(1000 * dist):.0f} mm {"into" if dist < 0 else "from"} '
                f'{what}, {100 * k / max(len(d) - 1, 1):.0f}% of the way (its ends are '
                f'{1000 * d[0]:.0f} and {1000 * d[-1]:.0f} mm clear)')

    def detour(self, q_from, q_to):
        """A cuMotion JointPath from q_from to q_to round the base plane, or None (self.why)."""
        pl = self.planner
        path = None if pl is None else pl.plan(q_from, q_to)
        self.why = None if path is not None else (pl.why if pl is not None else 'no cuMotion planner')
        return path
