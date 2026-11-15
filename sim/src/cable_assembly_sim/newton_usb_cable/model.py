"""USB-A insertion with the gripper holding the CABLE, not the plug: the hand, a deformable cable, the plug,
over the panel-mount receptacle, as a Newton model replicated into many worlds (newton_usb's parts and
contact; MuJoCo-Warp, which runs on NVIDIA Warp).

THE CABLE (its constant-diameter part, 5.5 mm; capsule colliders against the adapter and the panel) is a discrete elastic rod: `cable_segments` rigid links of
`cable_free_len / cable_segments`, joined by 3-DOF angular joints (D6, angular axes only) whose drive springs
are its bending (about the link's x and y) and twisting (about z) stiffness, k = EI / l and GJ / l per joint,
set per world (the env randomises EI and GJ). Newton's own rod (`add_rod`) runs on VBD / XPBD only; the
MuJoCo solver keeps this task's SDF contact, the arm's spring and the detent. Joint armature keeps the light
links' explicit springs stable at 1 kHz.

TOPOLOGY. Articulation A: world -D6 (the arm: Cartesian spring, as newton_usb)-> hand -> link 1 -> ... ->
link N. Articulation B: world -D6 (6 free DOFs; the detent's Coulomb friction on its world-z DOF)-> plug.
A fixed loop joint from link N's end to the overmold's back (MuJoCo turns it into a weld) closes the cable
onto the plug. MuJoCo compiles the weld's relative pose from the bodies at zero joint angles, so the model
is built in its nominal pose: the plug seated (identity), the cable straight up its axis, the hand at the
nominal grasp. Hence the arm D6's anchor: its zero is the nominal hand pose (env converts).

THE GRASP. The hand frame F (its body's frame) is the fingertip point, on the cable's axis, `grasp_z` up
the plug frame's z, turned like newton_usb's grasp: F y the jaws' closing axis (= plug x), F z the fingers'
direction at GRASP_TILT_DEG to the cable. The pads hold the cable over ~`grasp_hold` either side of F;
link 1 starts where the cable leaves them toward the plug. The joint from the hand to link 1 is where the
env puts the grasp's per-world pose (its control error): its anchor in the hand, written per world.

Gravity acts on the cable and the plug; the arm compensates the hand's (MuJoCo gravcomp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from cable_assembly_sim.newton_bnc.model import _tube_inertia  # noqa: F401  (configures newton)
from cable_assembly_sim.newton_usb.model import (GRASP_TILT_DEG, PLUG_DY, ModelCfg, _read_meshes, _sdf_mesh,
                                                 grasp_rotation, panel_mesh, PLUG_USD, SOCKET_USD)

import newton  # noqa: E402

OVERMOLD_BACK_Z = 0.043163
"""Where the cable leaves the overmold, up the plug frame's z [m] (build_usb.py)."""
CABLE_RADIUS = 0.00275


@dataclass
class CableModelCfg(ModelCfg):
    cable_free_len: float = 0.025
    """Cable between the overmold and the pads' edge [m]: what bends."""
    cable_segments: int = 10
    grasp_hold: float = 0.020
    """From the fingertip point to where the cable leaves the pads toward the plug [m]: the pads' faces
    (22.4 x 37.8 mm) cross a cable at 20 deg to their long side over ~±20 mm."""
    cable_density: float = 0.036
    """Linear density [kg/m] (a 5.5 mm PVC-jacketed cord, 1.5 g/cm^3)."""
    plug_mass: float = 0.009
    """Plug shell + insert + overmold [kg]."""
    bend_stiffness: float = 1.0e-3
    """Nominal EI [N m^2] (the env randomises it)."""
    twist_stiffness: float = 7.0e-4
    """Nominal GJ [N m^2]."""
    cable_damping_ratio: float = 1.0
    """Of each joint's critical damping (with its armature)."""
    cable_armature: float = 2.0e-6
    """Rotor-like inertia on each cable joint [kg m^2]: the links weigh 0.2 g, and MuJoCo integrates their
    springs explicitly; this keeps k dt^2 / I < 0.3 for EI up to 3e-3 at 1 kHz. A 7 g plug at 17 mm."""
    gravity: float = 9.81
    pads: bool = False
    cable_collide: bool = True
    """Capsules on the links, against the adapter and the panel: a cable folded over onto the face (a plug
    landing off the opening, pushed) must not pass through it."""
    """The pads are drawn only: on the cable, ~60 mm up, they cannot reach the adapter."""

    @property
    def link_len(self) -> float:
        return self.cable_free_len / self.cable_segments

    @property
    def grasp_z(self) -> float:
        """The fingertip point up the plug frame's z [m], nominal."""
        return OVERMOLD_BACK_Z + self.cable_free_len + self.grasp_hold


def nominal_hand(cfg: CableModelCfg) -> tuple[np.ndarray, np.ndarray]:
    """(R_PF, p_PF): the hand frame F in the plug frame, nominal grasp."""
    return grasp_rotation(GRASP_TILT_DEG), np.array([0.0, PLUG_DY, cfg.grasp_z])


@dataclass
class CableModel:
    model: newton.Model
    plug_body: int
    hand_body: int
    link_bodies: list[int]
    grasp_joint: int
    """The hand -> link 1 joint (index in a world): its anchor in the hand is the grasp's per-world pose."""
    bodies_per_world: int
    joints_per_world: int
    coords_per_world: int
    dofs_per_world: int
    shapes_per_world: int
    plug_shapes: list[int]
    socket_shapes: list[int]
    pad_shapes: list[int]
    arm_dof0: int
    cable_dof0: int
    plug_dof0: int


def _tf(R: np.ndarray, p) -> wp.transform:
    q = wp.quat_from_matrix(wp.mat33(*np.asarray(R, dtype=np.float64).flatten().tolist()))
    return wp.transform(wp.vec3(*[float(v) for v in p]), q)


def build(cfg: CableModelCfg, device=None) -> CableModel:
    plug = _read_meshes(PLUG_USD, ("plug",))
    socket = _read_meshes(SOCKET_USD, ("receptacle",))
    meshes = {
        "plug": _sdf_mesh(*plug["plug"], cfg.sdf_resolution, cfg.gap),
        "receptacle": _sdf_mesh(*socket["receptacle"], cfg.socket_sdf_resolution, cfg.gap),
    }
    if cfg.panel:
        meshes["panel"] = _sdf_mesh(*panel_mesh(), cfg.panel_sdf_resolution, cfg.gap)
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.3, ke=cfg.contact_ke, kd=cfg.contact_kd, gap=cfg.gap, density=0.0, mu_torsional=0.0, mu_rolling=0.0,
    )
    world = newton.ModelBuilder(gravity=-cfg.gravity)
    newton.solvers.SolverMuJoCo.register_custom_attributes(world)
    world.default_shape_cfg = shape_cfg
    shape_names = []

    def add_mesh(body, name):
        s = world.add_shape_mesh(body, mesh=meshes[name], cfg=shape_cfg, label=name)
        shape_names.append((s, name))

    add_mesh(-1, "receptacle")
    if cfg.panel:
        add_mesh(-1, "panel")

    R_PF, p_PF = nominal_hand(cfg)
    Dof = newton.ModelBuilder.JointDofConfig
    axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    # The arm and the cable (articulation A). The hand: the arm's reflected mass at the fingertips.
    m = cfg.hand_mass
    hand = world.add_link(
        xform=_tf(R_PF, p_PF), com=wp.vec3(0.0, 0.0, 0.0), mass=m,
        inertia=_tube_inertia(m, 0.0, 0.03, 0.06), lock_inertia=True, label="hand",
        custom_attributes={"mujoco:gravcomp": 1.0},
    )
    j_arm = world.add_joint_d6(
        -1, hand,
        linear_axes=[Dof(axis=a, target_ke=cfg.lin_ke, target_kd=cfg.lin_kd) for a in axes],
        angular_axes=[Dof(axis=a, target_ke=cfg.rot_ke, target_kd=cfg.rot_kd) for a in axes],
        parent_xform=wp.transform(wp.vec3(*[float(v) for v in p_PF]), wp.quat_identity()),
        child_xform=_tf(R_PF.T, (0.0, 0.0, 0.0)), label="arm",
    )
    # The links: body frames at their upper joint, axes the plug's (z up the cable).
    L = cfg.link_len
    ml = cfg.cable_density * L
    k_b, k_t = cfg.bend_stiffness / L, cfg.twist_stiffness / L
    links, joints = [], [j_arm]
    top = cfg.grasp_z - cfg.grasp_hold
    for i in range(cfg.cable_segments):
        z = top - i * L
        b = world.add_link(
            xform=wp.transform(wp.vec3(0.0, PLUG_DY, z), wp.quat_identity()), com=wp.vec3(0.0, 0.0, -0.5 * L), mass=ml,
            inertia=_tube_inertia(ml, 0.0, CABLE_RADIUS, L), lock_inertia=True, label=f"link_{i}",
        )
        if i == 0:
            parent, X_p = hand, _tf(R_PF.T, R_PF.T @ (np.array([0.0, PLUG_DY, top]) - p_PF))
        else:
            parent, X_p = links[-1], wp.transform(wp.vec3(0.0, 0.0, -L), wp.quat_identity())
        crit = lambda k: 2.0 * math.sqrt(k * cfg.cable_armature)  # noqa: E731
        ang = [Dof(axis=a, target_ke=(k_t if a[2] else k_b), target_kd=cfg.cable_damping_ratio * crit(k_t if a[2] else k_b),
                   armature=cfg.cable_armature) for a in axes]
        joints.append(world.add_joint_d6(parent, b, linear_axes=[], angular_axes=ang, parent_xform=X_p,
                                         child_xform=wp.transform_identity(), label=f"cable_{i}"))
        if cfg.cable_collide:
            sh = world.add_shape_capsule(b, xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * L), wp.quat_identity()),
                                         radius=CABLE_RADIUS, half_height=0.5 * L, cfg=shape_cfg, label=f"cable_shape_{i}")
            shape_names.append((sh, "cable"))
        links.append(b)
    world.add_articulation(joints)
    # The plug (articulation B): seated at zero, its D6 free; the detent's friction on its world-z DOF.
    mp = cfg.plug_mass
    plug_body = world.add_link(
        xform=wp.transform_identity(), com=wp.vec3(0.0, 0.0, 0.020), mass=mp,
        inertia=_tube_inertia(mp, 0.0, 0.006, 0.048), lock_inertia=True, label="plug_body",
    )
    add_mesh(plug_body, "plug")
    j_plug = world.add_joint_d6(
        -1, plug_body,
        linear_axes=[Dof(axis=a, friction=1.0e-4 if a[2] else 0.0) for a in axes],
        angular_axes=[Dof(axis=a) for a in axes], label="plug_free",
    )

    world.add_articulation([j_plug])
    # The weld: link N's lower end onto the overmold's back, on the cable's axis.
    world.add_joint_fixed(links[-1], plug_body, parent_xform=wp.transform(wp.vec3(0.0, 0.0, -L), wp.quat_identity()),
                          child_xform=wp.transform(wp.vec3(0.0, PLUG_DY, OVERMOLD_BACK_Z), wp.quat_identity()),
                          label="cable_to_plug")

    keep = {("plug", "receptacle"), ("plug", "panel"), ("cable", "receptacle"), ("cable", "panel")}
    for i, (sa, na) in enumerate(shape_names):
        for sb, nb in shape_names[i + 1:]:
            if (na, nb) not in keep and (nb, na) not in keep:
                world.add_shape_collision_filter_pair(sa, sb)

    builder = newton.ModelBuilder(gravity=-cfg.gravity)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(world, world_count=cfg.num_worlds, spacing=(0.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    model.rigid_contact_max = cfg.contacts_per_world * cfg.num_worlds
    names = dict(shape_names)
    return CableModel(
        model=model, plug_body=plug_body, hand_body=hand, link_bodies=links, grasp_joint=joints[1],
        bodies_per_world=world.body_count, joints_per_world=world.joint_count,
        coords_per_world=world.joint_coord_count, dofs_per_world=world.joint_dof_count,
        shapes_per_world=world.shape_count,
        plug_shapes=[s for s, nm in names.items() if nm == "plug"],
        socket_shapes=[s for s, nm in names.items() if nm in ("receptacle", "panel")],
        pad_shapes=[],
        arm_dof0=world.joint_qd_start[j_arm], cable_dof0=world.joint_qd_start[joints[1]],
        plug_dof0=world.joint_qd_start[j_plug],
    )
