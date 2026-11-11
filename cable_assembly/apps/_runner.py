"""Shared app entry point -- parse args, build the config, set up logging, run, tear down cleanly.

Replaces the identical `main()` + `rclpy.init/spin/shutdown` boilerplate that every ROS node
carried. An app is `run(cfg, robot, args)`; this wraps it so Ctrl-C, force-mode teardown and the
non-zero exit code are handled once.
"""

import logging
import os
import sys

from .. import config as urconfig
from .. import log as urlog

log = urlog.get('app')


def _gripper_port_ok(cfg):
    """False (logged) when the gripper's serial port is a device path that is not there -- the adapter unplugged.
    Anything else (a dry run, no port, a non-/dev port) is the gripper's own business."""
    if cfg.get_path('robot.dry_run', False) or str(cfg.get_path('robot.backend') or '') in ('sim', 'isaac_sim'):
        return True                                  # a simulated gripper: no port to look for
    port = str((cfg.section('gripper') or {}).get('port') or '')
    if not port.startswith('/dev/') or os.path.exists(port):
        return True
    log.error('The gripper port %s is not there: its USB-serial adapter is unplugged (or on a hub without power). '
              'Plug it in and check `ls /dev/serial/by-id/` lists it. Nothing has connected to the arm.', port)
    return False


def run_app(description, default_config, build_and_run, with_gripper=True, needs_camera=False,
            needs=None, precheck=None, add_args=None):
    """Standardised CLI + lifecycle. `build_and_run(cfg, robot, camera, args)` returns truthy on
    success. `robot` and `camera` are constructed here so teardown is guaranteed. `needs` is the
    calibration the app relies on: None (the default) is all of robot.CALIBRATION_ITEMS, () is
    none -- for an app that never moves the arm. `precheck(cfg)`, if given, runs before anything
    connects; falsy ends the run there. `add_args(parser)`, if given, adds the app's own options and returns
    `apply(args, cfg)`, which folds them into the config."""
    parser = urconfig.arg_parser(description, default_config)
    apply = add_args(parser) if add_args is not None else None
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    if apply is not None:
        apply(args, cfg)
    urlog.setup(logging.DEBUG if cfg.get('debug') else logging.INFO)
    log.info('Config: %s', cfg.get('_config_path'))
    if precheck is not None and not precheck(cfg):
        log.error('Stopped before connecting to anything: fix the setting above and run again.')
        sys.exit(1)

    # Imported here, not at module load, so `--help` and dry runs work without ur_rtde /
    # pyrealsense2 installed.
    from ..robot import Robot
    from ..robot.robot import CALIBRATION_ITEMS

    ok = False
    robot = camera = None
    try:
        # THE CAMERA BEFORE THE ARM: a camera that will not stream (2026-11-05, run after run) then
        # ends the run before the FRI session starts, instead of faulting the pendant's app.
        if needs_camera:
            from ..perception import RealSenseCamera
            camera = RealSenseCamera(cfg)
        # THE GRIPPER'S PORT BEFORE THE ARM TOO (2026-11-10: after a replug its USB-serial adapter was missing, and
        # the run found out only after the FRI session was up -- then closed it).
        if with_gripper and not _gripper_port_ok(cfg):
            sys.exit(1)
        robot = Robot(cfg, with_gripper=with_gripper,
                      needs=CALIBRATION_ITEMS if needs is None else needs)
        if camera is not None:
            camera.pose_fn = robot.camera
            robot.register_camera(camera)
        ok = bool(build_and_run(cfg, robot, camera, args))
    except KeyboardInterrupt:
        # BRAKE, THEN LET GO: the arm is braked to rest under the driver's stop limits and the session
        # ended (robot.close below). A second Ctrl+C would cut that short, so it is ignored until done.
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        log.warning('Interrupted (Ctrl+C) -- braking the arm to rest, then closing. Ctrl+C is ignored until '
                    'the arm has stopped.')
        if robot is not None:
            try:
                robot.arm.stop()
            except Exception as exc:                   # noqa: BLE001 -- closing still runs
                log.error('Braking on the interrupt failed (%s) -- closing anyway.', exc)
    except Exception:                          # noqa: BLE001 -- log the traceback, still tear down
        log.exception('Unhandled error:')
    finally:
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.close()
    sys.exit(0 if ok else 1)
