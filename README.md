# cable_assembly — ROS-free BNC cable assembly on a UR10e or a KUKA iiwa 14

Plain Python, no ROS. The UR10e is driven over **RTDE** (`ur_rtde`), the KUKA iiwa 14 over **FRI**
(`pyfri`, in progress), the Robotiq 2F-85 over **Modbus** (`pymodbus`), and the wrist RealSense
over **pyrealsense2** (D405 on the UR, D415 on the iiwa). One conda env, one
`python -m cable_assembly.apps.<demo>`.

This repo was seeded from `ur-assembly`'s `python-dev-refactor` branch (history included), which
is the `dev` ROS2 workspace refactored into plain Python; the ROS packages still exist, unchanged,
on that repo's `dev` and `main`. `docs/refactor_plan.md` is the refactor this repo continues.

## Why it is smaller

The ROS version was ~6,700 lines across 14 packages, much of it plumbing: cv_bridge conversions,
PoseArray assembly, tf2 broadcasters/listeners, controller-switch choreography, `rclpy` spin
loops, per-package launch files and `package.xml`/`setup.py`. Removing ROS removed all of it. What
remained — the actual control laws, the perception math, the assembly geometry — was already pure
`numpy`/`opencv`, and is now in one importable library.

Three changes did more than relocate code; they deleted whole failure classes:

| Concern | ROS | Here | What went away |
|---|---|---|---|
| **IK** | MoveIt KDL, a local solver → random-seed retries on error -31 | `getInverseKinematics(pose, qnear)` — the controller's **analytic** solver | The retry loop; IK either has a solution or it doesn't, and you get the branch nearest where you are. |
| **Compliance** | `admittance_controller` loaded inactive, parameterised over a service, **activated** (which deactivated the JTC and caused a joint-0 velocity fault, patched with a reference-holding dance) | **software admittance** over `servoL` (`robot/admittance.py`) — the same spring-mass-damper law, as a plain servo loop. `forceMode()` is used only where free-floating *is* the goal (`admittance_hold`): it has **no restoring stiffness**, so it cannot track a trajectory. | The controller install, the switch, the reference streaming, and `_hold_reference` entirely. |
| **Staleness** | tf2's 10 s cache expired slowly-republished transforms; "fixed" three times by widening `tf_cache_s` | `FrameGraph` keeps the latest transform forever; the **caller** states a `max_age` | The class of bug where a good estimate fails to look up because a buffer dropped it. |

The connector fusion also became simpler: SAM3 runs **in-process**, so a detection is captured,
run, and ingested synchronously — the elaborate image-stamp attribution the ROS scan needed (to
avoid crediting a lagged detection to the wrong viewpoint) is gone.

## Install

One script builds the whole environment (Linux, conda, Python **3.12** — `ur_rtde` and
`pyrealsense2` ship wheels only through cp312):

```bash
./setup-env.sh                  # creates the conda env "cable" (or: ./setup-env.sh <name>)
conda activate cable
hf auth login                   # once: the SAM3 weights are gated -- request access first at
                                #   https://huggingface.co/facebook/sam3
python -m pytest                # compare the FAILED set with tests/known_failures.txt
```

It does four things, and only the first lives in `environment.yml`:

1. **`conda env create -f environment.yml`** — Python 3.12, `numpy<2`, pybullet and `cmake<4` from
   conda-forge.
2. **`pip install` every `requirements/` layer**, including torch (pinned to the cu128 build) and
   SAM3. This is not a `pip:` section in `environment.yml` because conda runs that with `-U`,
   which replaces the conda-forge pybullet with a from-source build and lifts torch to a CUDA 13
   build that driver 570 cannot run.
3. **Activation hooks that park `PYTHONPATH`.** A sourced ROS workspace puts python3.10
   site-packages and ROS pytest plugins on it; they shadow the env's packages and break test
   collection. The hook restores it on `conda deactivate`.
4. **pyFRI, compiled against the controller's FRI version** (`FRI_CLIENT_VERSION`, default
   `1.15`) — with `--no-deps`, because its dependencies serve its example scripts and upgrade
   numpy to 2, which SAM3 cannot run on.

The layers in `requirements/` can also be installed on their own: `core` (math, fusion,
collision, `--dry-run`), `ur`, `iiwa`, `perception`, `sam3`, `dev`. Install the package itself
**editable** (`pip install -e .`, which the script does) — `configs/` lives beside it.

### SAM3

The detector code — the junction/neck/tip geometry on top of SAM3's masks — is **vendored** in
`cable_assembly/perception/`. Only the upstream `sam3` package and its weights are needed; there is
no `sam3-abhay` checkout and no `sam3.repo_path`. `requirements/sam3.txt` pins upstream
`facebookresearch/sam3` to a commit; the `sam3-abhay` fork's two package patches matter only on a
CPU-only machine or with a `.half()` model, neither of which applies here.

Notes:

- **GPU.** The cu128 torch wheels need NVIDIA driver ≥ 570 and do not carry Pascal (GTX 10xx)
  kernels; on such a card use `torch==2.4.1` from the cu121 index. `sam3_backend` keeps fp32
  weights under fp16 autocast with memory-efficient attention, which is what fit the old 6 GB
  card; a 10 GB RTX 3080 has headroom.
- **`sam3.mode` picks the detector** (`neck` | `junction` | `tip`); `junction` is the default in
  the cable configs.
- **sam3 under-declares its runtime deps** — `model_builder` imports the whole model zoo at load, so
  `einops`, `pycocotools`, and `psutil` are all required even for image inference (`triton`
  is pulled in by the torch wheel). If a later run raises `ModuleNotFoundError` for another package,
  install just that one — but do **not** install the guarded/lazy/eval-only ones its code also
  mentions: `xformers` (guarded, falls back to torch SDPA), `decord`/`torchcodec` (video only),
  `hydra`/`omegaconf` (training + the lazy multiplex builder), `detectron2` (agent/eval — an install
  nightmare), or `open_clip` (a docstring reference, not a real import). None are reached by
  `build_sam3_image_model`.
- **First run** downloads the checkpoint (a few GB) and takes ~30 s to load — the log says
  "Loading SAM3…".
- **Docker:** CUDA needs the container started with `--gpus all` (nvidia-container-toolkit),
  *separate* from the camera's USB passthrough (`-v /dev/bus/usb:/dev/bus/usb`, not `--device
  /dev/video*` — the pip `pyrealsense2` wheel uses the libusb backend). If the check above prints
  `cuda False`, add `--gpus all` or install the CPU torch wheel.

### Robot prerequisites

- **`ur_rtde` uploads its own control script**, so the External Control URCap must **not** be
  playing and the pendant must be in **Remote Control**. (This is the opposite of the ROS driver,
  which needed External Control running.)
- The gripper is on a serial port (`/dev/ttyUSB0`); prefer a stable `/dev/serial/by-id/...` path.
- Set `robot.ip` and `camera.serial_no` in the config (or `--set`).

## Run

Every demo: `python -m cable_assembly.apps.<name> [--config <name>] [--cell <bench>] [--set k=v] [--dry-run] [--yes] [--debug]`.

```bash
python -m cable_assembly.apps.cartesian --dry-run              # plan every move, no robot
python -m cable_assembly.apps.cable_pick_place                 # real run
python -m cable_assembly.apps.cable_pick_place --set scan.min_good_views=6 --set robot.ip=192.168.1.10
```

`--set key=value` overrides **any** config key, nested with dots (values parse as YAML). This
replaces the ROS `-p name:=value`, which only reached a handful of hand-declared parameters.

## Robots and cells

The task code never names a robot. A **cell** (`configs/cells/<name>.yaml`) is one bench: the arm
(`robot.type`), the tool on it (camera, hand-eye, fingertip), and the fixtures around it (ground
height, the frames file with the socket target and marker rig, every pose written in joints). A
task config picks its cell in one line, and the cell is overlaid on it key by key. The cell wins
any key both set, and the load logs each such override.

```bash
python -m cable_assembly.apps.bnc_assembly                                   # cell: ur10e_bench
python -m cable_assembly.apps.bnc_assembly --cell iiwa14_bench --dry-run     # the iiwa, simulated
```

| Cell | Arm | Tool | Status |
|---|---|---|---|
| `ur10e_bench` | UR10e over RTDE | Robotiq 2F-85 + D405 | calibrated |
| `iiwa14_bench` | iiwa 14 over FRI | Robotiq 2F-85 + D415 | **`calibrated: false`**: placeholders; `arm_bringup` and `--jog` passed on the arm (2026-10-28); the task runs as a dry run only |

The layers, top to bottom (see `cable_assembly/robot/arm.py`):

```
apps/            the task -- frames, forces, joint ROLES ('tool_roll'), never joint names
skills/, behaviors/
Robot            arm + gripper + camera + frame graph + motion primitives
Arm              the contract every backend meets: state, IK, moves, servo, wrench, guards
backends/        ur_rtde.URArm (UR10e) | iiwa_fri.IiwaFRIArm (iiwa 14) | sim.SimArm (any; --dry-run)
kinematics/      ur10e, iiwa14 -- numpy FK / Jacobian / closed-form IK, held to upstream URDFs
```

`--dry-run` runs the same task on `SimArm` with the selected robot's real kinematics: joint
limits, reachability and IK branches are checked as they would be on the arm. A cell's
`calibrated:` says what has been measured on its bench -- `true`, `false`, or a list of
`hand_eye`, `fingertip`, `ground_plane`, `poses`, `target` -- and an app refuses to drive
hardware until the items it relies on are listed.

### The iiwa over FRI

FRI takes one joint position per controller cycle and nothing else, so `IiwaFRIArm` does
client-side what the UR controller does for `URArm`: IK (`kinematics/iiwa14.py`), jerk-limited
joint moves and straight lines, streamed servoing for the admittance controller, and the wrench
as `pinv(Jᵀ)·τ_ext` from KUKA's external joint torques. The FRI loop itself runs in its own
process (`backends/fri_driver.py`, sharing a small file in `/dev/shm`), because pyFRI's
`step()` holds the GIL while it waits on the socket. That process keeps every command to the
robot continuous and under the joint caps whatever the task sends. It holds the arm still when
a session comes back after an e-stop, and brakes the arm and ends the session if the task
process dies.

First contact with the arm, before any calibration (no camera or gripper involved):

```bash
python -m cable_assembly.apps.arm_bringup          # connect, compare tool0 with the pendant's flange, check the wrench sign
python -m cable_assembly.apps.arm_bringup --jog    # + a 5° tool roll and a 20 mm lift at 10% speed, asked first
```

Start the FRI app on the smartPAD in POSITION client command mode, sending to this PC on UDP
port 30200; the app waits up to `robot.fri.connect_timeout_s` for the session. Every FRI
setting is explained in `configs/cells/iiwa14_bench.yaml`. `tests/fake_pyfri.py` stands in for
pyFRI with a simulated controller in real time, so the whole backend is tested without the arm.

### The BNC pickup on the iiwa

`apps/bnc_pickup` is `bnc_assembly`'s pick on its own -- same config, same functions -- followed
by putting the cable back where it was taken from. It needs no socket, so it runs once the cell
lists `hand_eye`, `fingertip`, `ground_plane` and `poses` under `calibrated:`. In order:

```bash
python -m cable_assembly.apps.gripper_control --port /dev/ttyUSB0   # the Robotiq answers (else ttyUSB1)
python -m cable_assembly.apps.hand_eye_calibration    # markers flat on the bench; prints hand_eye + the bench height
python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench                     # then --set pickup_test.cycles=5
```

The tool's weight: unless the Sunrise app carries its load data, KUKA's torques read it as a
~12 N contact. A tare cancels the force, not its moment about the flange. `apps/payload_calibration`
turns the wrist through ~13 poses about where the arm stands and fits the mass and CoG to the joint
torques; paste its `robot:` block into the cell (`compensate_payload: true`) -- or enter the same
numbers as load data in Sunrise, never both.

**Watching it live.** The iiwa cell streams the measured joints, the gripper and the current phase
over UDP (`live_view:` in the cell, `robot/live_view.py`). A model of the arm shows them as they
happen. Start it in another terminal, before or during the run:

```bash
cd sim && ./run.sh python scripts/live_view_gl.py     # Newton's OpenGL viewer: 0.4 GiB, SAM3 unslowed
cd sim && ./run.sh python scripts/live_view.py        # Isaac Sim: 2 GiB, SAM3 ~2x slower per frame
```

It only reads state the task already holds, and with no viewer the packets are dropped. The
OpenGL viewer also draws what the wrist camera has seen: every frame the task captures
(`live_view.frames`) merged into one layer of points in the base frame; see sim/README.md.

**The splat pick** (`--set scan.mode=splat`, `skills/splat_pick.py`, the `splat_pick:` block). The
cable is chosen from one image as before; then the camera orbits the chosen junction (one ring of
6 views, 20 deg off, at 50 mm/s, each line checked first and force-guarded), SAM3 finds it again
in every view, each view's depth places the junction on the connector's top, and the views are
fused: the median x-y, the mean heading. The one-image estimate assumes the junction lies on the
bench and is off along its ray by the connector's height times the ray's slope; the orbit's is
not (on a synthetic 12 mm connector 60 mm off the image centre: 0.1 mm against 1.6 mm). The pose
handed on is the same kind, so the grasp is unchanged. Fewer than 3 views finding the cable, views
more than 6 mm apart, or no depth: the one-image estimate, said so. The orbit's frames build the
live view's camera layer, and its views are kept (`splat_view_NN.npz`, for `scene_build --rgbd`).

```bash
cd sim && ./run.sh python scripts/live_view_gl.py                                   # optional: watch it
python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench --set scan.mode=splat  # repo root
```

**With no plane at all** (`--set scan.mode=splat_3d`). The same orbit, but its views' pixels round
the junction are fused into one 2 mm map, each cube labelled cable or connector by most of the
views' SAM3 masks, and the connector is fitted as a cylinder in it (robustly: an end face or a step
is outvoted): its axis, tilted however it lies, and where the views' junctions fall along it. The
pose handed on is ON that axis, so the pick's resting-on-the-bench correction is skipped. On a
synthetic connector raised 40 mm and tilted 15 deg: 0.7-1.3 mm and under 0.6 deg with 0.7-1.5 mm of
depth noise, where the plane modes put it 40 mm low and level. Too few connector cubes, a fitted
radius far from the connector's, or an axis steeper than 45 deg: the `splat` estimate, said so. The
labelled map and the axis are drawn in the live view (overlay `splat_pick`).

**What the depth says about a pick** (`watch:` in bnc_assembly.yaml, `skills/watch.py`). With
the camera's depth on, the pickup also reports -- and only reports; the fingers still decide:
where each cable's junction is by depth, against the ground-plane estimate it grasps at (on a real
D415 frame: 3.0 mm apart, the junction's top 8.6 mm up, which on a ray ~20 deg off vertical is
the gap the plane estimate makes); each cable's centreline in 3D; and after the lift, what the
camera sees where the junction was (`gone` -- the bench -- means the cable went with the gripper).
In the log and drawn in the live view.

**A collision scene for cuMotion** (`perception/scene.py`): RGB-D frames -> occupancy over a box in
front of the arm (`scene:` in the cell, 10 mm voxels; a later frame replaces what it sees; below
a seen surface counts as solid) -> a signed-distance grid, which cuMotion takes as an SDF obstacle,
plus the bench as a box. From frames already captured (a splat scan's views, say):

```bash
python -m cable_assembly.apps.scene_build --show                  # the latest splat scan's views
python -m cable_assembly.apps.scene_build --rgbd <run dir or frame.npz files> [--show]
```

The latest scan is looked for under both `data/experiments` and `sim/data/experiments` (an app
writes under the directory it was started in). Depth nearer than the D415's 0.3 m minimum is left
out: it reads the fingers as ~0.27 m, which stood a wall of obstacle under the gripper. It checks,
with cuMotion and `robot/description/iiwa14`, that the arm at each frame's own joints is clear of
what that frame saw, and reports how far its moving links (link_2 out; the gripper is not in the
description yet) are from the scene and from the bench. `tests/test_scene.py` has cuMotion plan round a block that a depth
camera saw from above, which a straight joint move would hit.

The pick pose (`pick_joints_deg`) and home are taught by jogging on the pendant (T1, with the
FRI app stopped) and copying A1-A7. The pitched BNC grasp reaches a connector pointing ACROSS
the robot, not one pointing straight away from or toward it; `bnc_assembly` recovers from the
latter by re-laying the cable along the socket, which the pickup test (no socket) cannot.

## Demos

Each demo has its own page in [`docs/`](docs/) — purpose, sequence, key config knobs, run command,
and caveats.

| ROS package | Replacement | Docs | Notes |
|---|---|---|---|
| `ur_cartesian_demo` | `cable_assembly.apps.cartesian` | [cartesian](docs/cartesian.md) | |
| `ur_gripper_demo` | `cable_assembly.apps.gripper` | [gripper](docs/gripper.md) | counts, not radians |
| `ur_pick_place_demo` | `cable_assembly.apps.pick_place` | [pick_place](docs/pick_place.md) | ArUco fiducial pick-place |
| `ur_visual_servo_demo` | `cable_assembly.apps.visual_servo` | [visual_servo](docs/visual_servo.md) | |
| `ur_admittance_demo` | `cable_assembly.apps.admittance_hold` | [admittance_hold](docs/admittance_hold.md) | forceMode, no controller |
| `ur_kinematic_assembly_demo` | `cable_assembly.apps.kinematic_assembly` | [kinematic_assembly](docs/kinematic_assembly.md) | |
| `ur_uncertain_assembly_sampling` | `cable_assembly.apps.uncertain_sampling` | [uncertain_sampling](docs/uncertain_sampling.md) | data-collection CSV |
| `ur_cable_pick_place_demo` | `cable_assembly.apps.cable_pick_place` | [cable_pick_place](docs/cable_pick_place.md) | |
| `ur_cable_touch_pick_place_demo` | `cable_assembly.apps.cable_touch_pick_place` | [cable_touch_pick_place](docs/cable_touch_pick_place.md) | |
| `ur_cable_pick_assemble_demo` | `cable_assembly.apps.cable_pick_assemble` | [cable_pick_assemble](docs/cable_pick_assemble.md) | |
| `ur_assembly_demo` | `pick_place` + the `insert` skill | — | fiducial pick + kinematic mate |
| `ur_vision_demo` | `cable_assembly.perception.aruco` | — | a library, not a node |
| `ur_tf_demo` | `Robot`'s `hand_eye` edge | — | one static transform, not a package |
| `ur_gripper_bringup` | **not needed** | — | no URDF / controller_manager without ROS |
| — (new) | `cable_assembly.apps.monitor` | [monitor](docs/monitor.md) | **live frame monitor** — read-only, works with pendant freedrive |
| — (new) | `cable_assembly.apps.gripper_control` | — | **interactive gripper jog** — type a count, moves there, prints the measured count |
| — (new) | `cable_assembly.apps.marker_calibration` | — | **fiducial rig calibration** — measures the target's pose in each ArUco marker's frame; paste the printed block into `configs/frames.yaml` `marker_rigs:` and set `assembly.target_source: visual` to let `bnc_assembly` find the socket by looking at it |

## Layout

```
cable_assembly/
  transforms.py      pose math — the ONLY place rpy/quat/rotvec conventions and the
                     base_link<->UR-base bridge live
  frames.py          FrameGraph — the tf2 replacement, staleness is a caller concern
  config.py, log.py  config loading + logging/step-runner
  robot/             arm.py (the Arm contract), backends/ (URArm over RTDE, IiwaFRIArm over
                     FRI + its driver process, SimArm), kinematics/ (UR10e, iiwa 14),
                     Robotiq2F85 (Modbus), ForceGuard, and the
                     Robot class: registers the gripper/camera, holds the FRAME REGISTRY
                     (register_frame with parent chains, resolved through live FK), and the
                     MOTION PRIMITIVES (move_joints / move_cartesian / move_relative) every
                     behavior builds on
  perception/        RealSenseCamera, ArUco, SAM3 adapter, ConnectorEstimator (multi-view fusion)
  skills/            reusable behaviours: servo, scan, pick, touch, insert, trajectory
  behaviors/         py_trees behavior-tree layer: a library of common robot behaviors
                     (guarded moves, admittance ramps, gripper, operator gates) the apps
                     compose their run trees from
  apps/              one thin script per demo
configs/             one yaml per demo (+ assembly_trajectory.csv); cells/ = one per bench
tests/               offline math/geometry tests (no robot/camera/torch)
sim/                 Isaac Lab 3.0 model of the iiwa 14 cell: RL training, and the live view of the real arm -- its own uv
                     environment (Python 3.12, CUDA 13 torch); see sim/README.md
```

A demo is a short script that **composes skills** — they hold no references to each other, so any
two combine (scan + touch + insert) without one being a base class of the other. The refactored
demos (calibration_check, wiggle_sampling, marker_calibration, cable_pick_estimate_assemble)
express their run sequence as a **py_trees behavior tree** built from `cable_assembly/behaviors` — the
tree is logged at `--debug` before the arm moves, so the whole procedure is visible up front.

The layering is: **Robot motion primitives → behaviors → scripts**. A new script is a Robot,
some registered frames, and a chain of behaviors — see `cable_assembly/apps/behavior_demo.py` for the
complete worked example (safe to run: `python -m cable_assembly.apps.behavior_demo --dry-run`):

```python
def build(cfg, robot, camera, args):
    robot.load_frame_catalogue()                       # everything in configs/frames.yaml
    robot.register_frame('probe', T, parent='fingertip')
    return bt.chain('demo',
        ('reset', robot, cfg),
        ('open_gripper', robot),
        ('move_relative', robot, delta, 'advance', {'expressed_in': 'probe'}),
        ('close_gripper', robot))

main = bt.app('Demo', 'my_config', build)
``` This is the
"modules and skills abstracted into common functions" the refactor was for; the ROS version
expressed the same sharing through a four-deep inheritance chain
(`PickPlace → CablePickPlace → CableTouchPickPlace / CablePickAssemble`).

## Conventions (get these wrong and everything is silently wrong)

- **rpy is EXTRINSIC XYZ** (`R = Rz(yaw) Ry(pitch) Rx(roll)`), matching the ROS stack — so every
  angle in the configs carried over untouched. CAD tools usually report intrinsic XYZ.
- **quaternions are `[x, y, z, w]`**.
- **Poses are `T_a_b`** ("b in a"); compose left-to-right: `T_a_c = T_a_b @ T_b_c`.
- **`tool0`** is the pendant all-zeros TCP — **not** `flange` (which is ~90° twisted).
- RTDE speaks the **UR `base`** frame (a 180° Z turn from ROS `base_link`) and **axis-angle**
  rotation vectors; both conversions are confined to `transforms.rtde_to_matrix` / `matrix_to_rtde`.
- **The `Rz(π)` base bridge applies to WRENCHES too, not just poses** — `arm.wrench()` applies it, so
  everything downstream is `base_link`. Miss it on a new RTDE value and `x`/`y` come out **negated**
  while `z` looks fine. Anything reading a *magnitude* (the force guard, `force()`, `torque()`) is
  blind to it, so it stays hidden until something reads *components* — which is exactly how it
  reached hardware once already. **Symptom key: some axes wrong ⇒ FRAME error; all axes wrong ⇒ SIGN
  error.** A global sign flip can never fix a per-axis split.

## Tests

```bash
python -m pytest                 # or: python tests/test_smoke.py
```

The suite is not all green: `tests/known_failures.txt` lists the expected failures and why.
A change is clean when the FAILED *set* matches that file (the file has the one-line diff).

Covers the pure-computation layers offline: the transform conventions, `FrameGraph` staleness, the
config loader, and the multi-view connector fusion against a synthetic ground truth (recovers the
origin to <5 mm and the axis to <5° on clean synthetic views).

## Hardware checklist still open

These are physical-calibration items the code cannot settle for you:

- `grasp_check.groove_counts` / `empty_counts` / `faces_max_counts` (225 / 228 / 223) — the three
  fingertip closure levels. Confirm each against what the gripper reports: a **seated** cable stops
  at the groove count (225 — SUCCESS, the middle band, *not* full closure), an **empty** close goes
  to 228, a cable on the **flat faces** stalls at ≤ 223 (a miss). The rad→counts fudge
  (`full_close_rad`, permanently "TODO: verify me" in the ROS configs) is **gone**: these compare to
  numbers the hardware reports directly.
- `grasp_check.detect_empty` — now **POSITION**-based (228 vs 225 are distinct), so it defaults
  **on** and is reliable; the old UNVERIFIED gOBJ path is gone.
- `pickup.mode` — `position` (stiff) by default. Set `compliance` for a compliant grasp descent
  (software admittance, same law as the assembly insert); validate the params on hardware.
- `touch.contact_z_offset_mm` — tune on the real connector.
- `assembly.target` — measure by jogging to a good mate and reading the chosen frame off the robot.
- **hand-eye `rpy`** — the configs carry `[0, 0, 0]` (identity), matching the `dev` branch exactly.
  The `dev` plan noted identity puts a detected marker ~0.6 m too low and proposed
  `[-1.5708, 0, -1.5708]` (an optical-frame rotation), but that change was never committed and is
  unverified. Left at identity here for faithful parity; resolve it as a hand-eye calibration on
  hardware and set `hand_eye.rpy` in the configs (or `--set hand_eye.rpy='[-1.5708,0,-1.5708]'`).
