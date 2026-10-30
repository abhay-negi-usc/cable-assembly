# cable_assembly_sim — the iiwa 14 BNC cell in Isaac Lab

An Isaac Lab 3.0 project for training the BNC insertion on the KUKA iiwa 14 + Robotiq 2F-85.
It is its own uv environment (Python 3.12, CUDA 13 torch), separate from the `cable` conda env,
and nothing in it imports `cable_assembly`.

**Status (2026-10-29):** the robot model runs under Isaac Sim's PhysX. `check_iiwa14.py` passes:
the simulated tool0 is within 0.001 mm / 0.021 deg of `kinematics/iiwa14.py` over HOME and five random
poses, and the gripper closes and opens on command. Under Newton the model goes non-finite on the
first move (checked before the fixed-base fix below; not re-run since).

The first task, BNC insertion with the plug already in the gripper, is built: the policy outputs
small pose deltas that IK turns into joint targets, the interface the FRI backend streams to the
real arm. A scripted controller locks the connector through that interface in 88% of 32 tries
(`check_bnc_insert.py`), and RSL-RL trains on it. No policy has been trained to the end yet.

## Install

Pinned in `pyproject.toml` / `uv.lock`: Isaac Lab 3.0.0, Isaac Sim 6.2.0, torch 2.11.0+cu130,
Newton 1.6.1, RSL-RL 5.5.1. No Isaac Lab checkout is needed; it comes from PyPI.

1. **NVIDIA driver ≥ 580.65** (CUDA 13). Isaac Sim 6.2 was tested on 595.58:
   `sudo apt install nvidia-driver-595 && sudo reboot`. The `cable` env's cu128 torch keeps working.
2. **Disk.** The Isaac Sim extra is tens of GB; Isaac Sim asks for 50 GB free.
3. **uv:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. In this directory, with the ROS paths out of the way (see `run.sh`):

   ```bash
   env -u PYTHONPATH -u LD_LIBRARY_PATH uv sync --extra importers --extra isaacsim
   ```

   About 12 GB to download (7.4 GB of it Isaac Sim's extension cache), 25 GB on disk.

Run everything through `./run.sh`: it is `uv run` without the ROS Humble / MoveIt / Webots
`PYTHONPATH` and `LD_LIBRARY_PATH` that `~/.bashrc` sets, which a Python 3.12 / CUDA 13 process
must not load. The first Isaac Sim launch asks you to accept NVIDIA's Omniverse EULA (or set
`OMNI_KIT_ACCEPT_EULA=YES`) and takes 10+ minutes to warm its caches.

## The robot

`scripts/build_iiwa14.py` builds `src/cable_assembly_sim/assets/data/iiwa14_2f85.usda`:

```bash
./run.sh python scripts/build_iiwa14.py      # urdf, usd and gripper stages
```

- **urdf** — lbr-stack's iiwa 14 xacro at the pinned commit, expanded without ROS. Upstream's
  link_ee becomes `tool0` (media-flange face, 35 mm past link_7), as in `kinematics/iiwa14.py`.
  The URDF is committed; `tests/test_sim_iiwa14.py` (main repo) holds its chain to the real arm's.
- **usd** — Isaac Lab's URDF converter, fixed base, unmerged (tool0 stays a body).
- **gripper** — Isaac Sim's Robotiq 2F-85, mounted on tool0 with a fixed joint, built the way
  Isaac Sim's `ur10e.usd` mounts it.

Check it in the simulator — it compares the simulated tool0 with `kinematics/iiwa14.py`:

```bash
./run.sh python scripts/check_iiwa14.py --physics isaacsim_physx              # headless
./run.sh python scripts/check_iiwa14.py --physics isaacsim_physx --viz kit    # with the viewport
```

`IIWA14_2F85_CFG` (`assets/iiwa14.py`) is the articulation config: joints `A1`..`A7`, body `tool0`,
KUKA's rated torques and speeds, HOME as in `kinematics/iiwa14.py`. Two things learned the hard way:

- **Command the 2F-85 as a whole.** One angle (`finger_joint`, 0 open, 45 deg closed) maps to all six
  linkage joints through `GRIPPER_JOINT_SIGNS`. A target on `finger_joint` alone stops half way.
- **192 PhysX position iterations** (Factory's count). At 16 the solver cannot hold the 40 g mimic
  linkage against the stiffly held arm: the fingers stalled at ~24 deg and shoved the wrist 6 deg.

Pass the backend with `--physics`: scripts build their `SimulationCfg` from the physics config
`launch_simulation` returns. Building it beforehand and passing it in silently runs PhysX.

The gripper stage also moves the articulation root from `link_0` to the robot prim. Where the
converter puts it, beside the world `root_joint`, PhysX builds a FLOATING base pinned by a
constraint: poses come out right, but every Jacobian gains six base columns in front of A1..A7.
`check_iiwa14.py` prints `fixed base: True base DOFs: 0` when the build is right.

## Live view of the real arm

`scripts/live_view.py` shows the bench iiwa in Isaac Sim while a task runs on it, e.g. the BNC
pickup:

```bash
./run.sh python scripts/live_view.py                                  # here: opens the viewport
python -m cable_assembly.apps.bnc_pickup --cell iiwa14_bench          # `cable` env, any order
```

The task sends its measured joints, the Robotiq's last-read position and its phase ~60 times a
second as UDP datagrams (`cable_assembly/robot/live_view.py`, on in the iiwa14_bench cell's
`live_view:`). The viewer sets them on the model every frame; nothing is simulated. A panel docked at
the right says LIVE (with the rate), or NO DATA and for how long. Either side can be started,
stopped or restarted at any time. Nothing here imports `cable_assembly`; the two share only the
datagram format and the cell yaml (the bench height, the port).

On the robot PC it shares the CPU with the FRI driver process and the GPU with SAM3. Measured on 2026-10-29 against
the fake KUKA controller (`tests/fake_pyfri.py`, 5 ms cycle):

- The viewer used ~0.6 of a core at nice 10, 30 fps, physics on the CPU (the defaults) and ~2 GB of VRAM.
- The FRI cycle period: median 5.00 ms, p99.9 5.29 ms and worst 6.45 ms with the viewer, against
  5.17 / 5.77 ms without it.

**GPU memory is the limit.** The RTX 3080 has 10 GB. The desktop takes ~0.7 GB and the viewer
2.0 GiB. SAM3 takes 4.4 GiB from its first frame on (3.3 GiB of it weights). Together they leave
2.6 GiB free, but not enough for another Isaac Sim. On 2026-10-29, with `bnc_cell_server.py`
holding 4.3 GiB, the pickup died loading SAM3 (torch.OutOfMemoryError, before the arm moved). So
the viewer checks first. If the free memory cannot take both the viewer and what the task's SAM3
has yet to take (4.4 GiB plus a 0.6 margin), it does not start and lists what is using the GPU.
`--task_gpu_gib 0` skips the check.

The viewer and SAM3 also share the GPU's time. SAM3 took 0.35 s a frame on the D415 test frame
alone, 0.65 s with the viewer rendering at 30 fps, and 0.44-0.53 s at `--fps 10`. The result was
the same each time.

To keep it off the robot PC entirely, run it elsewhere with `--bind 0.0.0.0` and set the cell's
`live_view.host` to that machine (ufw must let UDP 6008 in there). The GPU check is off then: the
task's SAM3 is on the other PC.

`--fps`, `--eye`/`--lookat` and `--port` adjust it. `HEADLESS=1 ... --duration_s 20 --screenshot x.png`
is the smoke test: it renders the model from the default view to a png.

## The task: BNC insertion

`CableAssembly-BNC-Insert-iiwa14-v0` (`src/cable_assembly_sim/tasks/bnc_insert/`), a direct env laid
out after Isaac Lab's Factory peg-insert. The plug starts in the 2F-85 above the socket, clocked to
the bayonet entry, and succeeds when it is pushed home and turned -90 deg to lock.

- **Parts.** Isaac Lab's NIST BNC plug and socket (`assets/bnc.py` has the measured geometry). The
  plug is one rigid body, so the J-slots turn the whole plug, not a collar.
- **Start.** Socket at (0.60, 0, 0) in the base frame +/-2 cm, clocked 135 +/- 30 deg (a PLACEHOLDER
  bench). Plug 13-23 mm above its seat, +/-4 mm, +90 +/- 10 deg (the slot entry), +/-1.5 mm in the
  gripper.
- **Action**, 15 Hz: a 6-D delta of the held-plug target in the observed socket frame, tilt locked
  (Factory's choice), turned into A1..A7 position targets by Isaac Lab's DLS differential IK. The
  target accumulates the deltas and may lead the measured pose by at most 2 mm / 4 deg.
- **Observations.** Policy (17): the held frame against the socket as SEEN (1 mm, 1 deg noise),
  velocities, last action. Critic (45): the same without noise, the true plug pose, the plug in the
  gripper, joint state.
- **Reward.** Factory's three keypoint terms (they ignore yaw: depth past the pins can only be had by
  turning), +1 engaged (in the straight slots), up to +1 for the turn once engaged, +1 locked.

```bash
./run.sh python scripts/check_bnc_insert.py            # a scripted insertion, 16 envs: can it lock?
./run.sh isaaclab train --rl_library rsl_rl --task CableAssembly-BNC-Insert-iiwa14-v0 physics=isaacsim_physx
./run.sh isaaclab play --rl_library rsl_rl --task CableAssembly-BNC-Insert-iiwa14-v0 --num_envs 4 --viz kit
HEADLESS=1 ./run.sh isaaclab play --rl_library rsl_rl --task CableAssembly-BNC-Insert-iiwa14-v0 --num_envs 1 \
    --video --video_length 450 --viz kit                # 30 s mp4 in the run's videos/play/
```

Play loads the newest checkpoint (`--checkpoint` picks another). Its camera follows env 0's socket from
11 cm away (`BncInsertEnvCfg.play_mode`). Recording needs `--viz kit`: `--video` on its own records
nothing when the task names a camera. `HEADLESS=1` keeps the window closed. It records 15 frames a
second, one per policy step.

Training runs 512 envs by default, ~16 s an iteration on the RTX 3080 (logs in `logs/rsl_rl/`). The
physics step costs nearly the same at 16 envs as at 512 (the solver's iterations, not the
contacts), so fewer envs buys little time.

`check_bnc_insert.py` drives the task with a controller that reads the TRUE plug pose: it answers
whether the physics, the grasp and the action chain let the connector lock, not whether a policy can
learn it. Learned the hard way:

- **The target needs memory.** Factory re-targets from the measured pose each step. With the
  plug's own 0.5 N weight on a compliant arm, that sagged 0.25 mm a step, without end.
- **The 2F-85 needs the gear-assembly task's drive** (stiffness 40, inner fingers 10). At the asset's
  softer drive the plug slid 3 mm up the pads when pushed.
- **The fit is tight.** The sleeve clears the barrel by ~0.2 mm, but the plug's inner contact (spring
  fingers on a real BNC, rigid here) enters the barrel bore with under 0.1 mm. Every scripted failure
  jams there, 9.3 mm up. Friction 0.5 instead of 0.75 changed nothing; a softer arm made it worse
  (the plug's weight sags it more, and it centres slower).
- **64 PhysX position iterations, not Factory's 192**: 88% locked against 69%, at 2.6x the speed.
  32 drops to 28%. The robot check keeps the asset's 192.
- **Resets write state; they do not step the physics.** RSL-RL starts environments at random points
  of their episodes, so resets come a few at a time, and stepping the shared simulation for them
  (Factory's way) would move every other environment too. The IK runs on joint writes, and the
  fingers are written 0.3 mm clear of the plug and squeeze it from the first step, as Isaac Lab's
  UR10e gear-assembly reset does.

### Open before training

- **Gripper mount — PLACEHOLDER.** Coupler length and clocking default to 0. Rebuild with the bench
  values: `./run.sh python scripts/build_iiwa14.py --stage gripper --coupler_mm .. --clocking_deg ..`.
  The open fingertips then sit 148 mm + coupler past tool0; `fingertip_grasp` in
  `configs/cells/iiwa14_bench.yaml` must agree, and so must the task's `grasp_depth` (173.1 mm at a
  0 mm coupler).
- **Arm gains — untuned.** Gravity is off (KUKA's position control compensates it) and the joint PD
  is a first guess. Match it to the real arm's step response through the FRI driver. In the BNC task
  it is also the contact compliance: the force at the plug is about this stiffness times the target
  lead, so the real arm must be run the same way (joint impedance in the FRI app, or the admittance
  layer) for a policy to transfer.
- **Bench layout — PLACEHOLDER.** The socket at 0.6 m in front of the base on a bench level with it,
  clocked 135 deg, is a guess; the iiwa bench's fixture is not measured yet.
- **Newton — not working.** `check_iiwa14.py --physics newton_mjwarp` goes NaN on the first move.
  Suspects: the 2F-85's `PhysxMimicJointAPI` linkage and the 1 g `tool0` body. Use `isaacsim_physx`
  (verified) or try `ovphysx` (standalone PhysX, untested).
- **No D415** on the model yet; the first task is state-based.
- **GPU memory.** 10 GB is under Isaac Sim's 16 GB guideline: train headless, size `--num_envs` to fit.

## Assets

- iiwa 14: lbr-stack (Apache-2.0, `data/iiwa14/LICENSE`).
- Robotiq 2F-85: Isaac Sim 6.2 asset server (CC-BY-4.0, `data/robotiq_2f85/LICENSE` once fetched).
- BNC plug / socket: Isaac Lab's `Factory/NIST/bnc_plug.usd` and `bnc_socket.usd` (Isaac Sim 6.1+),
  the NIST Task Board #1 connectors, read from the asset server at spawn and cached under `/tmp`.
  SDF colliders, 50 g each. The plug is one rigid body with the J-slots cut into it, so locking turns
  the whole plug -90 deg rather than a collar. Check their dimensions against the bench's parts.
