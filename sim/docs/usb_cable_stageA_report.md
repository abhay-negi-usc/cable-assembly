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
