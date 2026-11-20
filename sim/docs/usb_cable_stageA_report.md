# Cable-held USB-A insertion — stage A (cable-stiffness curriculum) report

Date: 2026-11-14. Env: `src/cable_assembly_sim/newton_usb_cable` (Newton 1.6.1, MuJoCo-Warp). Trainer: RSL-RL 5.5.1 PPO.

## Summary

The gripper holds the **cable** 25 mm above the plug's overmold, and the plug has to be inserted through the
bending cable. Stage A trains with **zero pose uncertainty**: the in-hand plug pose and the socket pose are
observed exactly. The policy is **not told the cable's stiffness**, which is widened by a curriculum toward
softer, harder cables.

The final policy (`model_700`) seats **100%** of 512 deterministic evaluation episodes over the full stiffness
range, EI 3e-4 to 3e-3 N·m² (10×). This holds at each of iterations 500, 600 and 700. Peak force is about 8 N
and force at the end of the episode about 0.12 N, under a 15 N cap. Training took 5 h 08 min (700 iterations
× 1024 environments) on one GPU.

## Setup

| | |
|---|---|
| Observation (actor, 72) | 4-step history of: in-hand plug pose and observed socket pose, both in the hand frame (position / 0.1 m, rotation's first two columns) |
| Action (6) | delta pose of the commanded plug target in the observed socket frame: 1 mm and 1.5° per unit per step, at 20 Hz; a constant 0.4 push down the socket axis is added |
| Policy | GRU 256 → MLP 256-128 (ELU) → 6, with an observation normaliser; deterministic (mean) at deploy |
| Critic (privileged) | adds the grasp control error, EI, GJ ratio and cable bend |
| Cable | 5 rigid links on angular D6 springs (k = EI/l, GJ/l), welded to the overmold; capsule collision |
| Randomised every episode | EI (log-uniform, curriculum), GJ/EI 0.5–1, grasp control error (±1.5 / ±3 mm, ±3° / ±5° roll, known), friction 0.15–0.45, arm stiffness 5–20 kN/m and 5–20 N·m/rad, detent 3–12 N, hand mass, 0–1 step action delay, wrench noise |
| Curriculum | EI_lo = 3e-3 · 0.1^level; promote at 80% success; 30% of episodes at easier levels |
| Done when | 3 evaluations at level 1.0 average ≥ 95% (the watcher then starts stage B) |

## Results (deterministic evaluation, 512 episodes)

| Iteration | Level | Seated, current range | Seated, full range (softest) | Peak / end force [N] |
|---|---|---|---|---|
| 0 | 0.00 | 66.0% | 40.2% | 15.0 / 13.9 |
| 100 | 0.15 | 100% | 94.5% | 12.5 / 3.4 |
| 200 | 0.40 | 100% | 99.2% | 11.1 / 2.5 |
| 300 | 0.65 | 99.6% | 99.6% | 9.4 / 0.7 |
| 400 | 0.90 | 98.8% | 99.2% | 8.0 / 0.5 |
| 500 | 1.00 | 100% | 100% | 8.1 / 0.14 |
| 600 | 1.00 | 100% | 100% | 7.7 / 0.15 |
| **700** | **1.00** | **100%** | **100%** | **8.2 / 0.12** |

The fix for soft cables was **easing off after seating**. Before training, the failures on soft cables were
fully inserted plugs held at more than 3° tilt by the bowed cable under the push. End force fell from about
14 N to 0.12 N.

## Problems found and fixed

1. **Reward exploit: swinging the plug** (run A1). The policy bent the cable to swing the plug about 34° and
   lower the keypoint heights. Fix: `kp_upright` penalises the lean of the upper keypoints off the tip's vertical.
2. **Reward exploit: walking off the plate** (run A2). The policy walked the plug about 15 mm sideways onto the
   panel and stayed contact-free (force ~0.1 N, success 0%, reward rising). Fix: `kp_free_radius` = 10 mm
   penalises the tip's off-axis distance beyond 10 mm. The search region stays free of this penalty.
3. **NaN target that never cleared** (run A3, from iteration ~423). The commanded target carries over from step
   to step and was never reset between episodes. After it once went NaN, the lead clamps passed the NaN
   through, and one world blew up on every step through every reset. Training success read about 66% from
   then on; the separate evaluation environment was unaffected (100%). Fix: the target is reset to the
   believed pose at each reset (`UsbCableEnv._reset`). The fix is in stage B; A3 ran with the bug.

## Checkpoint for the robot

Run directory: `/media/rp/Elements1/abhay_ws/cable-assembly/logs/rsl_rl/usb_cable_newton/2026-11-14_12-14-55/`

- `model_700.pt` — the rsl_rl checkpoint (actor, critic, normalisers).
- `model_700_actor.ts.pt` — **the one to deploy**: a standalone TorchScript actor (CPU, needs only PyTorch).
  It was written by `scripts/newton_usb_export.py` and checked to reproduce the network exactly.
  Its interface is `action, h = actor(obs[1, 72], h)`, with `h = actor.initial_state(1)` at the start of each
  insertion.
- `config.json` (every env and training parameter), `eval.csv`, and `videos/iter_00700.mp4`.

**What the robot side must reproduce** (see `UsbCableEnv._observe_poses` and `UsbNewtonEnv._apply_action`):

- **Observation frame (18 values):**
  - `p_plug_in_hand / 0.1`, then `R_plug_in_hand[:, 0]`, then `R_plug_in_hand[:, 1]`;
  - `p_socket_in_hand / 0.1`, then `R_socket_in_hand[:, 0]`, then `R_socket_in_hand[:, 1]`.
  - The hand frame is the fingertip point on the cable axis (y along the jaw closing axis, z along the fingers;
    `newton_usb_cable/model.py`).
  - The 4 most recent frames are concatenated oldest first. At the start, repeat the first frame 4 times.
- **Action:**
  - Clip all components to [-1, 1], then subtract the 0.4 push from `a[2]` and clip it to [-1, 1] again.
  - Integrate the delta pose onto the commanded target, in the observed socket frame (z out of the socket).
  - Clamp the target's lead over the current pose to 3 mm laterally, 7 mm axially and 4°, and its tilt to 14°
    from the socket axis.
  - Send it to a Cartesian impedance controller of about 10 kN/m and 10 N·m/rad. Limit the spring force to
    15 N and the torque to 1 N·m.
- **Rate:** 20 Hz.

## Caveats before trusting it on hardware

- **No pose uncertainty in training.** The policy has never seen pose-estimate error. Real in-hand and socket
  estimates will be off by millimetres and degrees, and stage B (now running) is the stage that trains for
  that. Use this checkpoint for interface and integration testing, at low speed with a force limit, and expect
  the stage-B policy to be the one to evaluate for success.
- **Simulated cable.** The cable is a 5-link spring model, and its stiffness range comes from estimates, not
  measurements of your cable. It was not validated against the real cable.
- **Controller match.** The policy assumes a compliant Cartesian spring at the hand. A stiff position
  controller on the real arm would change the contact dynamics.

## Next

Stage B started at 17:23 from `model_700`: the pose-uncertainty curriculum (4-DOF grasp-knowledge error, 6-DOF
socket error, from 5% of full range), with the stiffness range kept at full. Log: `logs/rsl_rl/usb_newton/train_cableB.log`.
W&B: https://forge.coreweave.com/wandb/abhay-negi-usc-university-of-southern-california/usb_insertion/runs/8fo6vm5l

---

## Update (2026-11-15): low-force env, stage A4 and stage B3

### Why the env changed

An evaluation video showed the cable snapping from one bend direction to the other under load: about 900°/s,
with a 34–45° kink at the joint next to the gripper. Meanwhile the policy held 12.5 N against the face. A
20 N grip on a 5.5 mm cable cannot hold that load; the plug would slip in the gripper. The cable model could
not slip, so the policy was free to use loads a real grasp would never survive. Changes:

| | Before | Now |
|---|---|---|
| Force limit | 15 N | **5 N** |
| Latch (detent) force | 3–12 N | **1–3 N** (USB-A mates under 3 N) |
| Grasp slip | not modelled | **slip limit, as a failure** (below) |
| Cable | 5 links, damping ratio 0.3 | **10 links, damping ratio 1.0** |
| Actor observation | 4 frames × (in-hand plug pose, socket pose) = 72 | **8 frames × (poses + last action) = 192** |

**Grasp slip limit** (`CableEnvCfg.grasp_slip`, `UsbCableEnv.grasp_load`). The load on the grasp is taken from
the hand's spring wrench, in the grasp frame. Each per-world capacity below uses that episode's grip force N
(18–22 N) and pad friction μ (0.3–0.8). A Robotiq 2F-85 pad touches about 22 mm of cable, and the cable radius
is 2.75 mm.

| Load | Capacity | At μ = 0.55, N = 20 N |
|---|---|---|
| In-plane force (along and across the cable) | 2μN | 22 N |
| Twist about the cable | 2μN·r | 0.06 N·m |
| Moment turning the cable in the pads' plane | μN·l/2 | 0.12 N·m |
| Force pushing the cable out between the jaws | N | 20 N |
| Moment tipping the cable into the pads | N·l/3 | 0.15 N·m |

- **Combining the loads:** the friction terms combine elliptically, and the worst of the rest is taken.
- **Failure:** at 70% of capacity the episode ends as a failure, with a −20 penalty.
- **Penalty:** a quadratic penalty starts at half of that.
- **Uncertainty:** the moment capacities are the least certain numbers.

**History:** each frame now also carries the last action, so the policy can compare what it commanded with
what moved: contact, the cable's give, a stuck plug. `newton_usb_train.py --init_from` maps a narrower actor
into the wider one, with the new inputs at zero weight, so a 72-input checkpoint starts unchanged.

**Validation, push only (no policy), 256 worlds:**
- **Zero uncertainty:** **100%** seated over the whole EI range. The softest cables used to seat 68–89%;
  peak force is 5 N and peak bend under 1°.
- **Slip:** no slip at any uncertainty level.
- **Snap-through:** the fastest bend change in real dynamics is about 100°/s.

### Stage A4 (stiffness curriculum, zero pose error)

- **Setup:** started from `model_700`, at stiffness level 0.5, with 2048 environments (about 1.2× the
  samples per second of 1024).
- **Adaptation:** in the new env, `model_700` seated 61%. It had learned habits for a 15 N limit.
- **Retraining:** within 100 iterations it was back to 100%.

| Iteration | Level | Seated, softest cables | Peak / end force [N] |
|---|---|---|---|
| 0 | 0.50 | 61% | 4.5 / 2.0 |
| 100 | 0.75 | 100% | 2.2 / 0.24 |
| 200 | 1.00 | 100% | 2.9 / 0.24 |
| 300 | 1.00 | 100% | 3.2 / 0.41 |
| **400** | **1.00** | **100%** | **2.4 / 0.38** |

The policy seats with about 2–3 N peak and lets go to about 0.4 N. Slip terminations are negligible.

**Checkpoint (supersedes `model_700`):**
`/media/rp/Elements1/abhay_ws/cable-assembly/logs/rsl_rl/usb_cable_newton/2026-11-14_19-59-34/model_400.pt`.
The observation is now 192 inputs: 8 frames of (18 pose values + the last 6-dimensional action), oldest first.
Export it with `scripts/newton_usb_export.py`.

### Stage B3 (pose-uncertainty curriculum, from A4)

| Iteration | Level | Seated, current level | Seated, full pose error |
|---|---|---|---|
| 0 | 0.05 | 99.6% | 3.5% |
| 100 | 0.30 | 61% | 3.9% |
| 300 | 0.45 | 70% | 7.8% |
| 500 | 0.60 | 48% | 13.7% |
| **700** | 0.60 | 54% | **19.1%** (best) |
| 1100 | 0.65 | 54% | 14.8% |

The curriculum reached 0.60 in about 5 hours, then stayed at 0.60–0.65 for about 6 hours. Signs of the plateau:

- force pinned at the 5 N limit;
- action noise decayed to its 0.08 minimum;
- training success near the stall rule's 60% floor.

This is the same pattern as the plug-held env's plateau at level 0.75. Grasp slip stayed rare: about 160
terminations in the whole run, with the 99th-percentile load at 0.38 of the limit.

### Other fixes

- **A world stuck in NaN.** After a blow-up, MuJoCo-Warp kept the world's solver data, and its derived
  per-world constants (`body/dof_invweight0`, `actuator_acc0`) went NaN. Resetting the state did not clear
  them, so the world blew up again on every step, which also disturbed the batch. It happened in A3 (from
  iteration ~423) and in B (from ~170). Fix (`UsbCableEnv._heal`): restore the constants from a snapshot
  taken at the first step, copy the solver data rows from a healthy world, then reset. The commanded target is
  also reset at every episode.
- **The training script's force limit.** `--max_force` defaulted to the plug env's 15 N and overrode the cable
  env's limit. It is now passed only when given.

---

## Update (2026-11-16): peck search, stage B4

### Why B3 stalled

The cable failure breakdown (`scripts/newton_usb_cable_failures.py`) on B3's `model_700` found that
**75% of episodes at full pose error ended with the plug on the housing face**, never finding the opening.
Only 8% jammed inside it, and grasp slip was about zero. Success fell steeply with the landing offset and with
the socket x/y error. The plug is held through 25 mm of cable, which is too compliant sideways to drag the plug
across the face to search: a soft cable is only about 60 N/m sideways (3EI/L³).

### Peck search (`CableEnvCfg.search`)

A search runs in the controller; the policy's actions are added on top. It uses only signals the robot has:
the hand's spring force along the estimated socket axis, the measured plug pose and the estimated socket pose.

- **Touch:** a push over 1 N records the probe's height.
- **In the opening:** if the plug sinks 1 mm below that height, the search ends.
- **On the face:** if the push is held above 4 N without sinking, the controller lifts 1 mm, moves to the next
  point of a Vogel (sunflower) spiral about the estimated socket (1.5 mm spacing, centre first), and probes
  again. The 4 N threshold is above the strongest latch (3 N) and below the 5 N limit.

Seated with no policy, 512 worlds, 25 s episodes:

| Pose error | Push only | Push + peck search |
|---|---|---|
| ×0 | 100% | 100% |
| ×0.6 | 12.5% | 61% |
| ×1.0 | 7% | 22% |

Two faster settings were worse: 2–2.5 mm spacing missed the opening, and a shorter dwell gave up too early.
At full error the search is limited by time, at about 30 probes per episode.

### Stage B4 (pose curriculum, from A4's `model_400`, with the search)

Setup: 2048 environments, minimum action noise 0.1, 14 hours.

| Iteration | Level | Seated, current level | Seated, full pose error |
|---|---|---|---|
| 0 | 0.05 | 99.6% | 6.3% |
| 200 | 0.45 | 79% | 14% |
| 500 | 0.70 | 61% | 27% |
| 800 | 0.75 | 61% | 37% |
| 1300 | 0.85 | 55% | 40.6% |
| **1500** | **0.85** | **54%** | **45.3%** (best) |
| 1550 | 0.85 | 54% | 40.2% |

- **Comparison:** B4 passed the 0.60 level where B3 stalled. It reached 0.85 and more than doubled full-error
  success (45% against B3's 19%).
- **Grasp slip:** about 0.2% of episodes; the 99th-percentile load was about 0.4 of the slip margin.
- **Problems:** no non-finite (NaN) states and no errors.
- **Best stage-B checkpoint:**
  `/media/rp/Elements1/abhay_ws/cable-assembly/logs/rsl_rl/usb_cable_newton/2026-11-15_13-15-29/model_1500.pt`

### Next

In-gripper slip: the cable can slide and turn in the pads, with pad friction and the slip model randomised.
Training starts from `model_1500`.
