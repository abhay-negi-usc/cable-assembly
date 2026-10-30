# Robot description

Upstream robot descriptions, kept as the independent reference our own numpy kinematics
(`cable_assembly/robot/kinematics/`) are tested against.

- `ur10e/` also feeds the ground-collision check in `cable_assembly/robot/collision.py`. On the
  UR the controller owns the real kinematics, so a discrepancy there costs a wrong *clearance*,
  not a wrong move.
- `iiwa14/` is the kinematic chain only. On the iiwa, `kinematics/iiwa14.py` **is** the IK the arm
  moves by (FRI takes joint commands), so here a discrepancy would be a wrong move. That is why
  the test holds it to this file.

## `ur10e/`

| | |
|---|---|
| Source | [UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) @ `rolling` |
| License | BSD-3-Clause — see `ur10e/LICENSE`, which must stay with the meshes |
| Contents | `ur10e.urdf` (generated), `meshes/collision/*.stl` (upstream, 7 files, ~430 KB) |

**The URDF is generated, not copied.** Upstream ships the description as *xacro*, which needs a
ROS toolchain to expand; there is no ROS in this workspace and no plain `.urdf` in that repo. So
`fetch_ur10e.py` assembles one from Universal Robots' own published parameter files:

- `config/ur10e/default_kinematics.yaml` — the joint origins (this *is* the calibrated chain)
- `config/ur10e/visual_parameters.yaml` — where each collision mesh sits within its link
- `urdf/ur_macro.xacro` — the link/joint topology and the `flange` / `tool0` frames

Every **number** is upstream's; only the assembly is ours.

```
python -m cable_assembly.robot.description.fetch_ur10e
```

Re-run to refresh, then diff — a silent upstream change to the kinematics shows up there.

### Two independent cross-checks

The reason to trust it is that two descriptions written from *different* upstream files agree:

1. **Offline.** `ur10e.urdf` is built from the published **joint origins**; `collision.fk_links`
   is built from the published **DH table**. `tests/test_behaviors.py` asserts their `tool0`
   agrees over random configurations — it comes out at ~6e-8 m, which is pybullet's float32
   precision, not a modelling difference.
2. **On hardware.** `GroundCollisionModel.verify_against_controller()` compares both against
   `getForwardKinematics` at start-up and logs an error past 1 mm. A dry run reports
   **UNVERIFIED** rather than pretending to have checked.

### What is *not* from upstream

The **80 mm spacer** and the **Robotiq 2F-85** are not in this description. They are modelled as
parametric capsules in `collision.ToolModel`, sized from `pickup.collision.tool` in the app
config and anchored on the repo's own calibrated `fingertip_grasp` (183 mm along tool0). Swapping
in a real 2F-85 mesh would be a strict improvement; the fingertip **intersection allowance**
(see `collision.py`) would still have to be applied to the finger bodies alone.

### Nominal, not per-robot

These are UR's nominal figures (`hash: calib_5119701370761913513`). A specific arm's factory
calibration differs by a fraction of a millimetre — far inside the margins this model works to,
and `verify_against_controller` measures the difference on the real machine anyway.

## `iiwa14/`

| | |
|---|---|
| Source | [lbr-stack/lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack) @ `4a5599ce` — `lbr_description/urdf/iiwa14/iiwa14.urdf` |
| License | Apache-2.0 — see `iiwa14/LICENSE`, which must stay with the file |
| Contents | `iiwa14.urdf`: links `link_0`..`link_7`, joints `A1`..`A7` copied verbatim (origins, axes, limits), plus our `tool0` |

**Kinematics only.** No meshes or inertials, and upstream's `link_ee` is left out: in the lab
checkout it sits 222.8 mm beyond `link_7`, which is some earlier tool, not ours. `tool0` here is
the media-flange face, 35 mm beyond `link_7` (the commented-out value upstream).

**The cross-check.** `tests/test_kinematics_iiwa14.py` evaluates this URDF's chain, lateral offsets
included, against `kinematics.iiwa14.fk_links`. The model drops the offsets because they cancel in
pairs along the joint axes they straddle. The test is what proves that, rather than assuming it.

### `iiwa14.xrdf` — the cuMotion description

cuMotion (optional; `setup-env.sh` step 5) loads the arm as `iiwa14.urdf` + `iiwa14.xrdf`:

```python
robot = cumotion.load_robot_from_file('iiwa14/iiwa14.xrdf', 'iiwa14/iiwa14.urdf')
```

The XRDF is **generated** by `make_iiwa14_xrdf.py`. Edit the script and re-run it (`--check` only
reports):

```
python -m cable_assembly.robot.description.make_iiwa14_xrdf
```

- **Collision spheres** are fitted by cuMotion's generator to upstream's collision meshes, from
  the same pinned commit as the URDF. The meshes come from `sim/`'s copy when it exists, or are
  downloaded otherwise. Only the spheres are stored.
  - There are 224 spheres: 32 per moving link, 16 for `link_0` and 16 for `link_7`.
  - The generator places spheres inside the mesh, so each link's spheres are grown just enough
    that 99% of its surface lies within 2 mm of them. On the moving links that is 8.5–17.6 mm, so
    the model is conservative by about a centimetre.
- **Self-collision ignores only neighbouring links.** A random sweep shows only those pairs touch
  at every configuration.
- **Limits.** Position and velocity limits come from the URDF. Acceleration (4 rad/s²) and jerk
  (80 rad/s³) are the FRI driver's defaults. These are ceilings: the cell's speed caps still
  apply when a plan is executed.
- **Arm only.** The tool (gripper, camera, spacer) is a cell fact. It gets attached to `tool0`
  where the cell is known.
- `tests/test_cumotion_iiwa14.py` checks:
  - the joints and limits;
  - that cuMotion's FK equals `kinematics.iiwa14.fk`;
  - that `HOME` and the cell's pick pose are free of self-collision, while a folded arm is not;
  - a plan around an obstacle that the straight joint move hits.

**To verify at bring-up:** the 35 mm media flange. Compare `fk(q)` against the flange pose Sunrise
reports at a few configurations. A wrong `d7` is a rigid shift of every tool frame, and a hand-eye
calibration would absorb it at one arm angle but not at another.
