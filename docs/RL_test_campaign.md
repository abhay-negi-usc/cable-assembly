# RL test campaign: USB-A insertion policies on the iiwa 14

Branch `RL-hardware-testing` (from `phase-0` at 4bb54e1). It brings the sim-trained policies of
`RL-sim-testing` to the bench and tests each variant there, against its own simulation numbers.

**Decided (2026-11-19):** the bench grips at the **sim's grasp, 20° to the cable axis**. The policies are tested
as trained; there is no grasp fine-tune.

**What exists and what doesn't.** Commands marked **[exists]** run today. Commands marked **[to build]** are the
planned interface of the deployment code in [section 4](#4-deployment-code-to-build). They will not run until that
code lands.

Background reading on `RL-sim-testing`:

- `sim/docs/usb_cable_handoff.md`: the cable-held env, its runs and its open issues.
- `sim/docs/usb_cable_stageA_report.md`: the stage A–D history.
- `sim/docs/usb_insertion_rl_report.md`: the plug-held env.

---

## 0. Set up the branch and the sim env

### 0.1 Merge the sim work [exists]

```bash
cd ~/abhay_ws/cable-assembly
git switch RL-hardware-testing
git fetch origin RL-sim-testing
git merge origin/RL-sim-testing
```

The merge has **one conflict, in `sim/.gitignore`**: a comment line about the USB CAD. Keep both sides, i.e. keep
their line `# ... and of the USB parts (4974T135 cord, 4574N11 panel-mount adapter).`. Then:

```bash
git add sim/.gitignore && git commit
```

`sim/README.md` merges cleanly.

### 0.2 The USB sim env on this PC [exists]

`run_usb.sh` expects `sim/.venv-usb`, a CUDA 12 env made for machines whose driver predates CUDA 13. This PC's
driver (595) runs CUDA 13, and `sim/.venv` already has the exact versions the USB env pins: newton 1.6.1,
warp 1.17.0, mujoco-warp 3.12.0, rsl-rl-lib 5.5.1. So point `.venv-usb` at it (gitignored after the merge):

```bash
cd ~/abhay_ws/cable-assembly/sim
ln -s .venv .venv-usb
./run_usb.sh python -m pytest tests/test_newton_usb.py tests/test_newton_usb_cable.py -q
```

The USB STEP files are already in `src/cable_assembly_sim/assets/data/usb/`. If the tests say the USDs are
missing, build them once (about 1.5 min):

```bash
./run_usb.sh python scripts/build_usb.py
```

**GPU.** The RTX 3080 has 10 GB. Run sim evaluations with no Isaac Sim, live view or SAM3 server up, and never
on the same day as arm trials while the estimator server holds the GPU.

### 0.3 Checkpoints

Copy them from the training machine into `sim/logs/rsl_rl/` (gitignored). Copy the checkpoint, `config.json`
and `eval.csv` of each run. The evaluation and export scripts rebuild the env from `config.json`, so it must
sit beside the checkpoint. See [section 1](#1-the-policy-variants) for the list.

```bash
SRC=<train-host>:/media/rp/Elements1/abhay_ws/cable-assembly/logs/rsl_rl
DST=~/abhay_ws/cable-assembly/sim/logs/rsl_rl
get() {  # get <env dir> <run> <checkpoint>
  mkdir -p "$DST/$1/$2"
  rsync -av --include="$3" --include=config.json --include=eval.csv --exclude='*' "$SRC/$1/$2/" "$DST/$1/$2/"
}
get usb_cable_newton 2026-11-14_19-59-34 model_400.pt    # A4
get usb_cable_newton 2026-11-15_13-15-29 model_1500.pt   # B4
get usb_cable_newton 2026-11-16_11-43-42 model_1500.pt   # D1
get usb_cable_newton 2026-11-17_15-03-14 model_1400.pt   # D2
get usb_newton       2026-11-12_23-44-45 model_2500.pt   # P1
# optional:
get usb_cable_newton 2026-11-16_06-49-26 'model_*.pt'    # C2 (pick its best from eval.csv, then delete the rest)
get usb_newton       2026-11-13_13-57-02 'model_*.pt'    # P2 (the latest)
```

Set `V` once for the commands below:

```bash
export V=~/abhay_ws/cable-assembly/sim/logs/rsl_rl
```

---

## 1. The policy variants

Every cable-held variant was trained with the fingers **20° to the plug axis**. All variants act at 20 Hz in
delta-pose steps (≤ 1 mm, ≤ 1.5° a step) in the frame of the socket estimate.

| ID | Run (checkpoint) | Grasp | Actor input | Peck search | Force cap | Seated in sim, full pose error |
|---|---|---|---|---|---|---|
| **V0** | none (the controller with zero action) | cable | — | on | 5 N | 22% (25 s); 100% at zero error |
| **A4** | `usb_cable_newton/2026-11-14_19-59-34` (`model_400`) | cable | 192: 8 × (in-hand plug, socket in hand, last action) | off | 5 N | trained at **zero error** only (100% there) |
| **B4** | `usb_cable_newton/2026-11-15_13-15-29` (`model_1500`) | cable | 192 | on | 5 N | 45% |
| C2 *(opt.)* | `usb_cable_newton/2026-11-16_06-49-26` (best in `eval.csv`) | cable + slip | 192 | on | 5 N | ~31% |
| **D1** | `usb_cable_newton/2026-11-16_11-43-42` (`model_1500`) | cable + slip | 240: + measured wrench | on | 5 N | 43–48% (25 s) |
| **D2** | `usb_cable_newton/2026-11-17_15-03-14` (`model_1400`) | cable + slip | 240 | on | 5 N | **57%** (40 s) |
| **P1** | `usb_newton/2026-11-12_23-44-45` (`model_2500`) | **overmold** | 78: believed plug (FK + nominal grasp), wrench, velocity, last action | learned | 15 N | **96%** |
| P2 *(opt.)* | `usb_newton/2026-11-13_13-57-02` (latest) | overmold + slip | 78 | learned | 15 N | ~73% (in progress) |

What the variants need from the bench:

- **The cable variants (A4–D2)** observe the plug's pose **in the hand, live, every step**. In sim that is the
  true pose plus a fixed per-episode bias, with no frame noise and no latency. The bench has no 20 Hz in-hand
  tracker, so each one runs in one of two modes (`in_hand` in [section 4](#4-deployment-code-to-build)):
  - `frozen`: the in-hand estimate from before the insertion, held constant;
  - `tracked`: the online in-hand 6D, as fast as it runs, latest value held.

  The sim check in [section 3](#3-sim-checks-before-the-arm) says which mode each variant can survive.
- **P1/P2** need no in-hand vision during the insertion: their belief is forward kinematics through the nominal
  grasp. They do need the **overmold grasp** (`pickup.cable_grasp.enabled: false`), which is also 20° (pick
  pitch −70°). Hand-load them: the overmold grasp's in-hand estimate was off by 29 mm.
- **D1/D2** read the wrench **scaled by 5 N and 0.03 N·m**, with sim noise of 0.3 N / 0.005 N·m. The iiwa's
  joint-torque estimate is much noisier (measure it in [section 2](#2-bench-measurements-no-policy)). If the
  sim check says it breaks them, run them with the torque inputs zeroed as a separate row (`wrench_torque: zero`).

---

## 2. Bench measurements (no policy)

Do these before the sim checks: section 3 replays these numbers.

### 2.1 Set the bench grasp to the sim's

The sim's hand frame F is the fingertip point **on the cable axis, 45 mm behind the junction** (25 mm of free
cable plus 20 mm of the pads' hold, past the CAD's overmold back), with the fingers at 20°. In
`configs/usb_assembly.yaml`:

```yaml
pickup:
  cable_grasp:
    enabled: true
    distance_mm: 45.0
    angle_deg: 20.0
    z_mm: 0.0
```

Two checks:

- **Joint margins.** The IK table in that file gives 16° at the mate, 7° at the stand-off and **none** at the
  re-aim's back-out for 20°. So **never pass `--reaim`** in this campaign.
- **The cable is not the CAD's.** The lab cord's ribbed strain relief runs about 21 mm past the CAD's, so of the
  sim's 25 mm of free cable only about 4 mm bends on the bench. The plug will hang stiffer than any sim cable.

Hand-load once and look at the live view: the plug drawn on the tool must sit where the real one hangs.

```bash
python -m cable_assembly.apps.usb_assembly --cell iiwa14_bench --hand-load --dry-run   # [exists]
```

### 2.2 Measurements

Record every number in the [results sheet](#6-results-sheet).

| # | Measure | How | Sim value it replaces |
|---|---|---|---|
| M1 | **Wrench noise and bias at the hand** | Plug held by the cable, arm still for 30 s, then moving slowly (5 mm/s) in free air for 30 s. Log FRI's external wrench at 20 Hz, referred to F. Report the standard deviation and the drift of the mean per axis. **[to build]** `tools/log_wrench.py` | 0.3 N / 0.005 N·m noise, same bias |
| M2 | **Socket-estimate error** | Hand-guide the plug home in the socket and record that pose as the true socket (the way `usb_connector` was recorded). Then run the 6D socket estimate 10× from the stand-off and take the error to the true pose per axis. | ±7 mm xy, ±10 mm z, ±7° |
| M3 | **In-hand estimate error and repeatability** | Pick (or hand-load) 10×. Compare the in-hand estimate with a reference view, e.g. the plug's tip touched on a known point. | ±3 mm, ±3° |
| M4 | **Latch force** | Plug at the opening, a slow admittance push (1 mm/s); the peak axial force before it gives. 5 trials. | 1–3 N (cable); 3–12 N (P1) |
| M5 | **Slip force in the grip** | Plug held by the cable, pull or push its tip axially against a fixed stop until the cable slides in the pads. | 12–32 N at the 20 N grip |
| M6 | **The plug's lateral stiffness on its cable** | Push the tip sideways with the arm still: deflection against force. | about 60 N/m (soft) and up, from EI 3e-4 to 3e-3 |
| M7 | **In-hand tracker rate and latency** | Run the online in-hand 6D through a slow move; rate, latency, frame-to-frame jitter. | sim: every step, no latency, no noise |

`scale = measured error / sim full range` (from M2 and M3) is the **pose-error scale** the sim checks are read at.

---

## 3. Sim checks before the arm

All [exists] unless marked. Run from `~/abhay_ws/cable-assembly/sim`.

### 3.1 Reproduce each variant's own numbers [exists]

```bash
# The cable variants: 2048 episodes, end-state breakdown, at the scales that matter.
for R in usb_cable_newton/2026-11-14_19-59-34/model_400 usb_cable_newton/2026-11-15_13-15-29/model_1500 \
         usb_cable_newton/2026-11-16_11-43-42/model_1500 usb_cable_newton/2026-11-17_15-03-14/model_1400; do
  ./run_usb.sh python scripts/newton_usb_cable_failures.py --checkpoint $V/$R.pt --num_envs 2048 \
      --scales 0.0 0.3 0.6 1.0 --episode_s 40
done
# P1 (plug-held)
./run_usb.sh python scripts/newton_usb_failures.py --checkpoint $V/usb_newton/2026-11-12_23-44-45/model_2500.pt \
    --scales 0.0 0.3 0.6 1.0
# The cable env's own zero-uncertainty check and video (no policy); P1's evaluation with a video:
./run_usb.sh python scripts/newton_usb_cable_check.py
./run_usb.sh python scripts/newton_usb_eval.py --checkpoint $V/usb_newton/2026-11-12_23-44-45/model_2500.pt --video
```

**Older runs: a known trap.** The failure script rebuilds the env from `config.json`, and **any key that a run's
`config.json` lacks takes today's default.** Features added after a run default to ON:

| Key | Today's default | Runs that lack it |
|---|---|---|
| `search` | `True` | A4 |
| `obs_wrench` | `True` | A4, B4, C2 |
| `pos_bounds` | ±30 mm | runs older than D1/D2's ±16 mm |

So A4 would be evaluated with a peck search it never had, and A4/B4/C2 would get a 240-wide observation they
cannot load. Before trusting an older run's number, check its `config.json` for each key:

```bash
python - <<'EOF'
import json, os, glob
for f in sorted(glob.glob(os.path.expandvars('$V/usb_cable_newton/*/config.json'))):
    e = json.load(open(f))['env']
    print(f.split('/')[-2], {k: e.get(k, 'MISSING') for k in ('search', 'obs_wrench', 'obs_action', 'history', 'pos_bounds', 'slip', 'max_force')})
EOF
```

If the `'env'` key name is wrong, look at the top level of one `config.json` first. The fix **[to build]**: a
`build_cfg` that fills missing keys with "off" (`search=False`, `obs_wrench=False`, the run's own bounds), shared
by the failure script and the npz export.

Pass: within the handoff's numbers (D2 ≈ 57% at 1.0, 40 s), allowing ±3 points at 2048 episodes. If the
results swing, use `--num_envs 1024` and repeat with another `--seed`. Add the measured scale from
[2.2](#22-measurements) to `--scales`.

### 3.2 The bench's conditions, one at a time [to build]

These need options in the evaluation scripts (`--in_hand`, `--wrench_noise`, `--delay`), part of
[section 4](#4-deployment-code-to-build):

```bash
./run_usb.sh python scripts/newton_usb_cable_failures.py --checkpoint $V/<run>/model_<it>.pt --num_envs 2048 \
    --scales <measured> --episode_s 40 --tag <cond> \
    [--in_hand frozen|tracked:<hz>:<latency_ms>:<jitter_mm>] [--wrench_noise <N> <Nm> --wrench_bias <N> <Nm>] \
    [--wrench_torque zero] [--delay <steps>]
```

| Condition | Flags |
|---|---|
| nominal | — |
| frozen in-hand | `--in_hand frozen` |
| tracked in-hand (M7's numbers) | `--in_hand tracked:...` |
| bench wrench (M1) | `--wrench_noise ... --wrench_bias ...` |
| bench wrench, torque zeroed | `+ --wrench_torque zero` |
| latency | `--delay 2` |
| all of the bench's at once | everything above that applies |

**Go/no-go per variant:** a variant goes to the arm in the mode (frozen or tracked, torque kept or zeroed) where
"all of the bench's" stays within **15 points** of its nominal number at the measured scale. Otherwise it waits
for a fine-tune.

### 3.3 Export and parity [to build]

```bash
# The robot's format: numpy .npz with the weights, the normaliser and every env constant from config.json.
./run_usb.sh python scripts/newton_usb_export_npz.py $V/usb_cable_newton/2026-11-17_15-03-14/model_1400.pt \
    ../configs/policies/usb_cable_d2_1400.npz
# (same for each variant: usb_cable_a4_400, usb_cable_b4_1500, usb_cable_d1_1500, usb_plug_p1_2500)

# Parity: sim episodes replayed through the ROBOT's code (numpy actor + controller + peck search) step by step:
# the same observation -> the same action and target (max error < 1e-5). It also calibrates the seated detector.
./run_usb.sh python scripts/newton_usb_deploy_check.py --policy ../configs/policies/usb_cable_d2_1400.npz \
    --worlds 512 --scales 0.0 1.0
```

Pass: parity error < 1e-5. The seated detector, on 512 worlds, must have ≤ 1% false "seated" and ≤ 2% missed.

---

## 4. Deployment code to build

| File | What |
|---|---|
| `sim/scripts/newton_usb_export_npz.py` | Exports the checkpoint and `config.json` to `.npz`, modelled on `newton_bnc_export.py`. The spec carries the run's **own** constants, e.g. D1/D2's ±16 mm `pos_bounds`, not the current ±30 mm default. |
| `sim/scripts/newton_usb_deploy_check.py` | Parity, the seated-detector calibration, and the bench-condition evaluation hooks of 3.2. |
| `cable_assembly/skills/rl_usb_insertion.py` | Reuses `Policy` (numpy GRU) from `skills/rl_insertion.py`. Adds: <ul><li>the `poses` observation (cable) and the `wrench` observation (plug-held);</li><li>the peck-search state machine: touch 1 N, stuck 4 N with no 1 mm sink, 1 mm lift, Vogel spiral at 1.5 mm, 0.4 mm/step descent;</li><li>the base push;</li><li>lead, tilt (14°) and position bounds, force and torque caps on the spring's pull;</li><li>admittance about F at 1e4 N/m and 10 N·m/rad (the sim's spring);</li><li>`in_hand` frozen or tracked; `wrench_torque` kept or zeroed;</li><li>the seated detector;</li><li>guards: measured force over `guard_force_n`, time over `max_s`, F leaving the workspace box, in-hand slip over 6 mm or 15°.</li></ul> |
| `cable_assembly/apps/usb_assembly.py` | `insertion_mode: rl`, which today is refused at line 79. It runs `rl_usb_insertion` from the stand-off and then does the push-home or release as usual. |
| `configs/usb_assembly.yaml` | An `assembly.rl_usb` block (below). |
| `tools/log_wrench.py` | M1. |
| `tests/test_rl_usb_insertion.py` | Frame mapping (sim plug and hand frames ↔ CAD datum and tool0, including **which side of the plug the wrist stands on** at 20°), numpy actor vs TorchScript, peck-search unit cases. |

The planned config block:

```yaml
assembly:
  rl_usb:
    policy: policies/usb_cable_d2_1400.npz   # or 'none' = V0, the controller with zero action
    in_hand: frozen                          # frozen | tracked
    wrench_torque: keep                      # keep | zero
    socket_offset_mm: [0.0, 0.0, 0.0]        # added to the socket the policy believes (socket frame)
    socket_offset_deg: [0.0, 0.0, 0.0]
    guard_force_n: 10.0                      # cable variants; 25.0 for P1/P2 (15 N cap)
    max_s: 40.0
    log_every_step: true                     # obs, action, target, wrench, joints -> the trial's folder
```

---

## 5. Arm trials

Hand on the E-stop throughout. The live view is on. Never pass `--reaim`.

Every trial logs its steps to the trial folder under `data/experiments/` (`log_every_step`) so a failure can be
replayed in sim. A person pulls the plug out between trials (`campaign.reset: manual`).

Base command (cable variants: picked by the cable, or `--hand-load`):

```bash
python -m cable_assembly.apps.usb_assembly --cell iiwa14_bench --trials N \
    --set assembly.insertion_mode=rl --set assembly.rl_usb.policy=policies/<variant>.npz \
    --set assembly.rl_usb.in_hand=<frozen|tracked> [--hand-load]                      # [to build]
```

For P1/P2 add:

```bash
--hand-load --set pickup.cable_grasp.enabled=false --set assembly.rl_usb.guard_force_n=25
```

**Always do one `--dry-run` of the exact command first** (no motion; the live view shows the stand-off and the
target).

### The ladder

A variant climbs only after passing the rung below it. Its stop rules apply at every rung.

| Rung | Socket the policy believes | Trials | Pass |
|---|---|---|---|
| **R0: controller only** (V0, once for all) | the **taught** true pose (`--set assembly.target_source=kinematic` with the M2 record) | 5 | ≥ 4/5 seated. Push never over 6 N. The peck search behaves as in sim (touch, lift, spiral). |
| **R1: zero error** | taught | 5 | ≥ 4/5 seated |
| **R2: the real estimate** | 6D (`target_source: pose6d`, the default) | 20 | report the rate against sim at the measured scale |
| **R3: injected error** | taught + offsets, e.g. `--set assembly.rl_usb.socket_offset_mm=[4,0,0]` | 5 per offset | the curve below |
| **R4: baseline** | the current `engage` pipeline (`insertion_mode=engage`), the same conditions as R2 | 20 | the comparison |

**R3 offsets:** lateral 2, 4 and 7 mm (along x, then y, then diagonal), height ±5 mm, tilt 3° and 7°. Compare
each with the sim's success for the same fixed offset; that curve is the main result.

**Order of variants:**

1. **V0**
2. **P1** (the simplest: no live in-hand vision)
3. **A4**, R0–R1 only: zero error is all it was trained for
4. **D2**
5. **D1**
6. **B4**
7. C2 and P2 if time allows

**Stop the variant** and replay its log in sim before going on if any of these happen:

- a guard trips twice;
- the plug slips in the pads (`slipped`);
- the measured force passes the guard;
- the target leaves the workspace box;
- the policy oscillates, e.g. 2 Hz or faster, visible in the logged actions.

---

## 6. Results sheet

Bench measurements:

| # | Value | Date |
|---|---|---|
| M1 wrench noise / bias (N, N·m) | | |
| M2 socket error (mm, °) | | |
| M3 in-hand error (mm, °) | | |
| M4 latch force (N) | | |
| M5 slip force (N) | | |
| M6 lateral stiffness (N/m) | | |
| M7 tracker rate / latency / jitter | | |
| → pose-error scale | | |

Per variant:

| Variant | Mode | Sim nominal @ scale | Sim bench conditions @ scale | R0 | R1 | R2 | R3 (2/4/7 mm, 3/7°) | R4 baseline | Peak force | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | — | | | | — | — | — | | | |
| P1 | — | | | | | | | | | |
| A4 | frozen | | | | | — | — | | | |
| D2 | | | | | | | | | | |
| D1 | | | | | | | | | | |
| B4 | | | | | | | | | | |
