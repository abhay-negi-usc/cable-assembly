# Handoff: cable-held USB-A insertion RL (Newton sim)

Date: 2026-11-18. Branch: `RL-sim-testing`. Read this first, then `docs/usb_cable_stageA_report.md` (detailed history)
and `docs/usb_insertion_rl_report.md` (the earlier plug-held env).

## The task

A KUKA iiwa with a Robotiq 2F-85 grips the **cable** of a USB-A plug about 25 mm above the overmold (a hard
constraint: the gripper must hold the cable) and inserts the plug into a panel-mount receptacle. The policy uses
delta-pose actions at 20 Hz on a compliant Cartesian controller. Pose estimation error is realistic and does not
improve during insertion (no re-estimation by vision). At full scale it is:

- grasp knowledge error: ±3 mm y and z, ±3° tilt, ±3° roll;
- socket error: ±7 mm x and y, ±10 mm z, ±7° on each rotation axis.

The real hardware's limits: the receptacle mates under 3 N, and the grip is 20 N on a 5.5 mm cable.

## Code map (`sim/`)

| File | What it is |
|---|---|
| `src/cable_assembly_sim/newton_usb_cable/model.py` | Newton model: hand (6-DOF spring), a 10-link cable (angular springs EI/l, GJ/l, damping ratio 1.0), the plug welded to the cable end, socket/panel SDF contact. MuJoCo-Warp solver. |
| `src/cable_assembly_sim/newton_usb_cable/env.py` | `UsbCableEnv(UsbNewtonEnv)` and `CableEnvCfg`: rewards, curriculum, observations, peck search, slip model, tap localisation (off), NaN healing. |
| `src/cable_assembly_sim/newton_usb/env.py` | Base env: reward, action application, latch, `_failure()` hook. |
| `scripts/newton_usb_train.py` | Training: RSL-RL PPO, GRU actor, privileged critic. `--init_from` maps narrower actors into wider observations (new inputs at zero weight). |
| `scripts/newton_usb_chain.py` | Watcher: starts the next stage when a run saturates. |
| `scripts/newton_usb_cable_failures.py` | Failure breakdown: end states, landing offset, error and domain thirds. Options `--episode_s`, `--tag`. |
| `scripts/newton_usb_cable_check.py` | Zero-uncertainty validation and video. |
| `scripts/newton_usb_export.py` | Writes a TorchScript actor for the robot. |
| `tests/test_newton_usb_cable.py` | 3 tests; all pass. |

**Run:** `./run_usb.sh python ...` (`.venv-usb`).

**Logs and checkpoints:** `/media/rp/Elements1/abhay_ws/cable-assembly/logs/rsl_rl/usb_cable_newton/<date>/`. These
go on the external drive.

**W&B:** only with the user's key: `WANDB_API_KEY=$(cat ~/.config/wandb/api_key_abhay)`, project
`usb_insertion`. Never use the sambanda-usc account.

**Conventions:** commit only when the user asks.

### Current `CableEnvCfg` defaults that matter

| Setting | Value |
|---|---|
| Force cap / latch force / rotation torque cap | 5 N / 1–3 N / **0.03 N·m** (the pads hold only ~0.015–0.09 N·m of twist) |
| Observation (`obs_mode="poses"`) | 8 frames × 30 values = 240: in-hand plug pose, socket estimate in the hand frame, last action, measured wrench |
| Wrench sensor model | noise 0.3 N / 0.005 N·m; per-episode bias of the same size |
| Peck search (`search=True`) | touch at 1 N; "stuck" at 4 N with no 1 mm sink; lift 1 mm; Vogel spiral at 1.5 mm spacing |
| Slip (`--slip 1`) | pad friction 0.15–0.8; grip 18–22 N; kinetic ratio 0.6–1; slip speed limits; creep in half the episodes |
| Slip failure | 6 mm slid or 15° turned |
| Slip penalty | 1 a mm |
| `grasp_slip` | a soft penalty on the grasp load (when `slip` is on, it is not terminal) |
| `pos_bounds` | **±30 mm** xy. Widened for tapping; D1/D2 trained at ±16 mm, so pass the old bounds to reproduce them exactly. |
| `tap_localize` | **False** (see open issues) |

### Launch D2's setup

```
F="--env usb_cable --obs poses --num_envs 2048 --eval_envs 512 --min_std 0.1 --entropy 3e-3 --gamma 0.995 \
   --steps_per_env 64 --w_align 0.3 --curriculum_mix 0.3 --promote_stall 10 --slip 1 --episode_s 40"
WANDB_API_KEY=$(cat ~/.config/wandb/api_key_abhay) nohup .venv-usb/bin/python scripts/newton_usb_train.py \
  --init_from <ckpt> --curriculum_on pose --curriculum_start 0.8 --run_name <name> --hours 14 $F > <log> 2>&1 &
```

Notes:

- About 41 s an iteration at 2048 environments, 25 s episodes, on an RTX 4070 Ti (7.3 of 12 GB).
- Evaluations every 100 iterations go to `eval.csv`; `seated_full` is success at full pose error.
- With 512 evaluation episodes the numbers swing ±6 points; use the 2048-episode failure script to compare
  checkpoints.

## History and results (success = seated at full pose error unless noted)

| Run | What changed | Result | Checkpoint dir |
|---|---|---|---|
| A3 | stiffness curriculum, zero pose error, 15 N cap, 5-link cable | 100% (zero error) | `2026-11-14_12-14-55/model_700` |
| A4 | 5 N cap, 1–3 N latch, slip limit, 10 links, action history | 100% (zero error), 2–3 N | `2026-11-14_19-59-34/model_400` |
| B3 | pose curriculum | stalled at level 0.6–0.65; **19%** | `2026-11-15_00-43-07/model_700` |
| B4 | + peck search | level 0.85; **45%** | `2026-11-15_13-15-29/model_1500` |
| C2 | + real in-gripper slip, 0.03 N·m torque cap | ~31% | `2026-11-16_06-49-26` |
| D1 | + measured wrench in the observation | 43–48% (25 s episodes) | `2026-11-16_11-43-42/model_1500` |
| **D2** | + 40 s episodes | level 1.0; **57%** (2048-episode breakdown), 52–61% in evaluations | **`2026-11-17_15-03-14/model_1400`** |

**Validation without a policy, 25 s episodes:** the peck search alone seats 100% / 61% / 22% at pose error
×0 / ×0.6 / ×1.0; the push alone seats 100% / 12.5% / 7%.

**D2 `model_1400`, failure breakdown** (2048 episodes, full error, 40 s):

| End state | Share |
|---|---|
| Seated | 56.7% |
| On the housing face | 27.7% |
| In the opening, not seated | 9.1% |
| Grasp lost to slip | 3.8% |
| Invalid simulation / off the plate | 2.8% |

| Landing \|y\| | Success |
|---|---|
| 0–1 mm | 77% |
| 3–4.5 mm | 58% |
| 6–8 mm | 47% |
| over 8 mm | 26% |

Failed episodes end tilted 7.8°, against 1.2° for successes.

## Problems found and fixed (so they are not re-introduced)

- **Reward exploits:** swinging the plug on the cable (fixed by `kp_upright`), and walking off the plate (fixed by
  `kp_free_radius`).
- **NaN worlds stuck forever:** MuJoCo-Warp keeps per-world solver data, and its derived constants go NaN.
  `_heal` restores the constants and copies data rows from a healthy world. The commanded target is reset every
  episode.
- **Cable snap-through and unrealistic forces:** fixed by the 5 N cap, 10 links, and damping ratio 1.
- **Slip model:** the sign was wrong (it increased the load), and the 1 N·m rotation cap caused a twist runaway
  through the pads, so the cap is now 0.03 N·m.
- **Training script:** `--max_force` and `--history` used to override the env defaults; now they apply only
  when given.

## Open issue: tap localisation (`tap_localize`, implemented, off)

The idea is to find the socket by touch before the peck search:

1. tap at the estimate;
2. bisect along 3 rays for the housing plate's edge (plate radius 11.6 mm, the panel 1.3 mm lower);
3. fit the circle and a plane;
4. correct `R_obs`/`p_obs` and pre-align the target.

The mechanics work: 99% of worlds complete it, and 94% seat at zero error. But the estimate does **not**
improve (×1.0: 5.5 → 6.5 mm). Cause: the socket tilt (±7°) changes the plate height by up to ±3.4 mm across
the taps, which swamps the 1.3 mm edge step (on-plate taps are classified against the highest tap minus 1 mm).

Lessons from debugging:

- Descend at 0.4 mm a step or slower; faster, the free-air spring force exceeds the 1 N touch threshold.
- Do not require "stopped" for a touch: a plug sliding into the opening keeps moving at about 2 N.
- Dwell up to 6 steps at -0.5 to pass the latch.

## Proposed next steps (in order)

1. **Make tapping tilt-robust** (most promising; it targets the far landings and the tilted catches):
   - estimate the plate plane first, from about 3 taps;
   - classify on-plate or off by the residual to that plane, or by the drop between neighbouring taps along a ray;
   - use a robust fit;
   - keep the taps under about 12 s.

   Validate without a policy: in `/tmp`, the scratch `tapcheck.py` pattern measured the estimate error before and
   after. Then retrain from D2 `model_1400` with `tap_localize=True`.
2. **Train D2 longer** at full error (curriculum start 1.0), with 2–3 seeds; pick checkpoints by the
   2048-episode failure script. This can run while step 1 is built.
3. **Let the policy steer the peck search:** add the search state (probe index, offset) to the observation, and let
   the policy bias the probe direction from the force cues.
4. **Tilt pre-alignment, or a slightly higher rotation cap** (0.04–0.06 N·m), tested by evaluation only first,
   for the tilted catches (watch the slip rate).

Constraint, from the user: the gripper must grip the cable; a shorter free length is not an option. Tapping is
acceptable.
