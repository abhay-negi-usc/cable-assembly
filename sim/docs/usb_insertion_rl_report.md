# USB-A insertion by RL in Newton: report (2026-11-11 to 2026-11-13)

Simulation-only results for a KUKA iiwa + Robotiq 2F-85 inserting a USB-A plug into a panel-mount receptacle. The
policy uses proprioception and the measured wrench only. It is trained under hidden socket-pose and in-hand-pose
errors, with low contact force. Code: `sim/src/cable_assembly_sim/newton_usb/`, `sim/scripts/newton_usb_*.py`;
how-to and run details: `sim/README.md`, section "The task in Newton: USB-A insertion".

## 1. Summary

- **Stage 1 (fixed grasp) is solved in simulation.** Checkpoint `model_2500` of run `2026-11-12_23-44-45`
  (stage 1e) seats 96.2% ± 0.6% of 1,024 episodes at the full uncertainty range: socket ±7 mm, ±10 mm height,
  ±7° tilt and yaw; plug ±3 mm / ±3° in the hand. It seats 99.8% with no estimation error. Force is capped at
  15 N, and the end force averages about 2.5 N.
- **Getting there took six training runs.** Two of them stalled at the same curriculum level (0.75) for
  different reasons, and that was found only by sorting failures by end state and by hidden variable (section 5).
- **Stage 2 (plug slip in a 20 N grip) is in progress**, started automatically from `model_2500`. After about
  630 iterations it seats about 73% at full range. More than half of the episodes slip more than 1 mm in the
  fingers, because the policy still pushes at the 15 N cap.

## 2. Task and setup

| Item | Value |
|---|---|
| Robot | iiwa (simulated as a Cartesian spring, stiffness 5–20 kN/m and 5–20 N·m/rad) + 2F-85, plug held by the overmold with the fingers at 20° to the plug axis |
| Parts | McMaster 4974T135 cord (USB-A plug) and 4574N11 panel-mount adapter |
| Clearances | shell 0.3 mm a side, tongue 0.04–0.07 mm, lead-in 0.7–0.8 mm |
| Simulator | Newton 1.6.1 (MuJoCo-Warp, SDF mesh contact), 1,024 parallel worlds on one GPU |
| Hidden errors (level 1) | socket ±7 mm x/y, ±10 mm z, ±7° roll/pitch/yaw; in-hand ±3 mm (two axes), ±3° |
| Domain randomisation | plug–socket friction 0.15–0.45, arm stiffness, latch (detent) resistance 3–12 N, hand mass, 0–1 step action delay, wrench noise and bias |
| Success | the true plug seated: tip within 0.8 mm of the floor, centred, tilt < 3° |

## 3. Method

- **Observation (actor, 78 numbers):**
  - the last 4 frames of the *believed* plug pose (forward kinematics through the nominal grasp), in the frame
    of the socket estimate;
  - the measured wrench (noisy, biased);
  - velocity, the last action, and the lead of the target over the arm.
  - No vision, and no true pose.
- **Action:** a 6-D increment of a target pose (≤ 1 mm, ≤ 1.5° a step at 20 Hz), plus a constant 0.4 mm
  downward push. Safety layers: lead limits, a 15 N force cap (isotropic), a torque cap, a 14° tilt bound and
  workspace bounds.
- **Networks:** a GRU actor (256 units, then 256 and 128) and an MLP critic (512, 256, 128). The critic also
  sees privileged inputs: the true pose, the errors, the domain parameters and the true wrench.
- **Algorithm:** PPO (RSL-RL 5.5.1).
  - Data: 1,024 environments × 64 steps per iteration; 5 epochs × 4 mini-batches.
  - Adaptive learning rate (KL 0.01), γ 0.995, λ 0.95.
  - Exploration: entropy 3e-3 and an action-std floor of 0.08.
- **Reward:**

  | Term | Value |
  |---|---|
  | Seated | +3 |
  | Engaged (tongue in its cavity) | +1 |
  | Depth shaping | three keypoints along the plug axis, **height only** |
  | Force | −((F − 6)/6)² |
  | Hold (once seated) | −0.9 (F − 2)/10 |
  | Axial contact speed | −0.5 ((v − 5 mm/s)/10 mm/s)² |
  | Tilt, only once the tip is in the opening | −0.3 · tilt / 3° |
  | Action size | −0.01 \|a\|² |
  | Action change | −0.1 \|Δa\|² |

- **Curriculum:** both errors scale with a level from 0 to 1.
  - The level rises by 0.05 when 80% of episodes at the current level succeed (windows of 1,024 episodes).
  - After 10 windows without promotion it rises anyway, if success is at least 60%.
  - 30% of episodes draw a random level below the current one.
- **Stages:**
  1. Fixed grasp, until 3 consecutive evaluations at full range average ≥ 95%.
  2. The plug can slip in the pads (grip and pad friction randomised), starting from stage 1's weights.
     `newton_usb_chain.py` makes the hand-off.
- **Evaluation:** deterministic episodes (the action mean).
  - In-training evaluations every 100 iterations: 32 episodes per level, later 256.
  - Final evaluations: 1,024 episodes.
  - `newton_usb_failures.py` classifies each failure by its end state and bins success by landing offset,
    each estimation error and each domain parameter.

## 4. Run history

| Run | Change | Outcome |
|---|---|---|
| `2026-11-11_21-59-38` (first) | Small error ranges (now level ~0.2–0.3), 35 N cap | 89% at its full range (512 eps), but it pushed at the 35 N cap throughout |
| `2026-11-12_01-36-15` (stage 1) | Target ranges, 20 N cap, force and speed penalties, 15 s episodes | Stalled at level 0.75 from about iteration 550; 35% at full range |
| `2026-11-12_10-31-34` (1b) | 3× force weights, axial-only speed penalty, 25 s episodes | Peak force still about 20 N, jittery; stopped |
| `2026-11-12_11-15-34` (1c) | Force penalty from 5 N, jitter penalty | Jitter halved, but success fell to 35% at 0.7; peak 18 N |
| `2026-11-12_18-11-10` (1d) | From scratch: 12 N cap, height-only shaping, exploration reset | Stalled at 0.75 again (iterations ~830–1650); ~42% at full range |
| `2026-11-12_23-44-45` (1e) | From 1d: 15 N cap, tilt term, γ 0.995, 64-step BPTT, std floor, mixed curriculum, stall rule | Full range at iteration ~300; 91.6% at 1850; **96% at 2500** → stage 2 |
| `2026-11-13_13-57-02` (stage 2) | From 1e `model_2500`, slip on | In progress: ~73% at full range at iteration ~630 |

Run directories up to 1d are in `sim/logs/rsl_rl/usb_newton/`; 1e and stage 2 are on the external drive under
`abhay_ws/cable-assembly/logs/rsl_rl/usb_newton/`. All runs are in the W&B project `usb_insertion`.

## 5. Analysis

### 5.1 Why two runs stalled at level 0.75

Both stage 1 (20 N cap) and stage 1d (12 N cap) stopped at level 0.75, which first looked like one cause.
Sorting failures by end state showed two causes:

| Episode end (level 0.75) | Stage 1 (20 N, 15 s) | Stage 1d (12 N, 25 s) |
|---|---|---|
| Seated | 65% | 72% |
| On the housing face (the search failed) | 19% | 9% |
| In the opening, not seated | 16% | 18% |

- **Stage 1 (20 N cap):** limited by search time (15 s episodes) and by jams in roughly equal measure. Latch
  strength barely mattered.
- **Stage 1d (12 N cap):** the stuck-in-the-opening failures were the strong latches. 11–12 N latches seated
  56% of the time *with no estimation error*, and 22% at level 0.75 (3–8 N latches: about 86%). A 12 N force cap
  leaves no margin over a 12 N latch plus wall friction under a tilt. This was a feasibility limit, not a learning
  problem, so no reward change could have fixed it. The stuck plugs sat at about 3° of tilt, against 1.3° for
  seated ones.
- **Ruled out:** arm rotational stiffness had no effect on success or jams. The hypothesis of "blind landings"
  (the tip landing with no overlap with the opening), computed from the error model as 17% of episodes at 0.75,
  explained only part of the loss.

### 5.2 What fixed it (stage 1e)

| Measured cause | Change | Effect |
|---|---|---|
| No force margin for strong latches | 15 N cap; latch range kept at 3–12 N | Success no longer depends on the latch (90–94% across its thirds at full range, model 1850) |
| Tilted jams | Tilt penalty once the tip is in the opening | Stuck-in-opening at full range: 28–31% → 1.5% (model 2500) |
| Search failures, short credit horizon | γ 0.995, 64-step BPTT window | On-face failures at full range: 27% → 2.3% (model 2500) |
| Exploration collapse (std 0.03) | Std reset to 0.2 with a floor of 0.08 | The policy kept improving for 2,500 iterations at full range |
| A curriculum that cannot pass a hard level | Mixed levels, promotion after a stall | Not needed in 1e (every promotion was at ≥ 80%); used once in stage 2 |

### 5.3 Final stage 1 policy (`model_2500`, 1,024 episodes per row)

| Errors | Seated | On face | In opening, not seated | Time at the force cap (success / fail) | End tilt (success / fail) |
|---|---|---|---|---|---|
| None | **99.8%** | 0% | 0.2% | 0.2 s / 0 s | 0.05° / 12.5° |
| Level 0.5 | **99.4%** | 0.1% | 0.5% | 0.4 s / 0.9 s | 0.1° / 1.8° |
| Full range (level 1) | **96.2% ± 0.6** | 2.3% | 1.5% | 0.9 s / 3.6 s | 0.3° / 7.7° |

At full range:

- **No error dominates any more.** Success is 93–99% in every third of every error axis. The largest effect is
  the largest third of socket x (92.8%) and y (94.2%) error.
- **By landing offset:** success is 99.4% for landings within 1 mm of the opening's centre across the plug and
  91.5% beyond 8 mm.
- **By latch strength:** 95.6–97.4% across its thirds.
- **Force:** peak about 15.7 N (the cap), held for under 1 s in successful episodes.
- **No simulator artifacts.**

## 6. Observations

- **The policy uses all the force it's allowed.**
  - Peak force sits at the cap in every run: 35 N, 20 N, 12 N and 15 N.
  - In 1e successes spend only about 1 s at the cap, mostly pushing through the latch.
  - End force after seating fell from about 4.3 N to 2.5 N over training.
  - The force penalty shapes *how long* the policy pushes hard, not *how hard*.
- **Peak axial contact speed stays near 27 mm/s**, against the 5 mm/s threshold of the speed penalty. Those are
  short transients at first touch: the policy moves at up to 20 mm/s before contact.
- **Small evaluations mislead.** In-training evaluations of 32 episodes swung between 59% and 100% for one
  checkpoint. Decisions (stopping a run, the stage-2 trigger) need 256 to 1,024 episodes.
- **The search is learned, not scripted.** The policy descends gently, slides on the face using the torque
  pattern, drops into the lead-in, corrects tilt, pushes through the latch, then relaxes.
- **In stage 2, slip is the dominant effect.** With pad friction 0.3–0.8 on a 20 N grip, the plug slips at
  12–32 N axial. The stage-1 policy pushes to 15 N, so more than half of the episodes slip more than 1 mm early
  in stage 2.
- **Simulator artifacts are rare** (a plug through a thin wall or kicked out: ≤ 0.1% of episodes) and are voided
  as failures.

## 7. Lessons learned

1. **Diagnose before tuning.** Classify failures by end state and bin success by every hidden variable before
   changing rewards. Stage 1b–1d spent about 15 h on reward and force-cap changes that targeted symptoms. A
   15-minute breakdown found the cause.
2. **Test feasibility at zero uncertainty.** If a domain setting fails with no estimation error, the cause is
   physics or constraints, not learning.
3. **Constraints and randomisation ranges must leave a margin.** A force cap equal to the maximum latch
   resistance makes part of the distribution unsolvable. The curriculum then stalls with no error message.
4. **Don't reward what the actor cannot observe in a way it can exploit.** Lateral shaping toward the true socket
   made a blind policy hover at the estimate, its expected position. Height-only shaping, plus a tilt term paid
   only once the tip is inside the opening, avoided this.
5. **Partial observability needs memory and horizon.** The search pays off seconds later. γ 0.99 with a 32-step
   (1.6 s) BPTT window was too short; γ 0.995 with 64 steps worked.
6. **Guard exploration.** The action std collapsed to 0.02–0.03 in every long run. A floor plus a reset on
   fine-tuning kept learning going.
7. **A success-threshold curriculum needs an escape.** Promotion at 80% can block forever if part of the
   distribution is infeasible. Mixed levels plus a stall rule keep it moving and keep easier cases in the batch.
8. **Fine-tune; don't restart.** Stage 1e reused 1d's weights and reached full range in about 300 iterations.
9. **Infrastructure matters on a shared, full disk.** Logs and checkpoints now go to the external drive by volume
   UUID, W&B runs go to the right account, and the watcher stops one stage and starts the next. Kill processes by
   PID (a `pkill -f` pattern once matched the shell that ran it).

## 8. Suggestions

1. **Measure on the bench.** These settings decided the outcome and are still assumptions:
   - the receptacle's real insertion (latch) force;
   - the actual socket-pose error of the D405 estimate;
   - the axial force at which the plug slips in the 20 N grip.

   If real socket error is ±3–4 mm, level 0.75 already covers it.
2. **Stage 2: keep force under the slip limit.** If slip does not fall within a few hours:
   - start the force penalty lower (e.g. 4 N) or make it steeper;
   - or give the critic the slip margin (it already sees grip and pad friction) and reward low slip directly.
3. **Remaining stage-1 failures (3.8%).** These are landings more than 6 mm off (2.3% on the face) and tilted
   jams (1.5%). Options:
   - a contact-gated search shaping term in potential-based form;
   - a stronger tilt term (0.3 → 1.0).
4. **Before hardware:**
   - zero-uncertainty video checks of the final stage-1 and stage-2 policies;
   - an evaluation with wrench noise and bias, and latency, beyond the training range;
   - distillation or export of the GRU actor to the robot's 20 Hz loop, which needs the same observation frames
     (believed pose in the socket-estimate frame) computed from the real iiwa's kinematics and its external
     wrench estimate.
5. **Make the scripts robust:**
   - checkpoint the curriculum level (it is passed by hand on resume today);
   - add the failure breakdown to the periodic monitor and W&B;
   - add `pgrep` patterns that match the chain's absolute-path launches.

## 9. Reproduce

```bash
cd sim
./run_usb.sh python scripts/newton_usb_train.py --run_name stage1e --hours 12 \
  --init_from <stage-1d run>/model_1650.pt --init_std 0.2 --min_std 0.08 --entropy 3e-3 --gamma 0.995 \
  --steps_per_env 64 --w_align 0.3 --curriculum_mix 0.3 --promote_stall 10 --curriculum_start 0.6
./run_usb.sh python scripts/newton_usb_failures.py --checkpoint <run>/model_<it>.pt --scales 0.0 0.75 1.0
./run_usb.sh python scripts/newton_usb_eval.py --checkpoint <run>/model_<it>.pt --video
```
