"""Gaussian splats on a USD stage, written from numpy -- for the live view (scripts/live_view.py).

Isaac Sim 6.2's RTX renders the ParticleField3DGaussianSplat prim type. Its schema here is a
backport of OpenUSD 26.03's UsdVol one (omni.usd.schema.usd_particle_field) with no Python
bindings, so the prim is defined by its type name and the attributes by their schema names: what
omni.kit.converter.gsplat does when pxr.UsdVol lacks the class.

THE ARRAYS, one row per Gaussian, the 3DGS conventions (the converter's .ply reader undoes the
log/logit encodings a .ply stores; these are the decoded values):
    positions     (N, 3)  m, in the prim's frame
    scales        (N, 3)  standard deviations along the Gaussian's own axes, m
    orientations  (N, 4)  unit quaternions, (w, x, y, z)
    opacities     (N,)    0..1
    colors        (N, 3)  RGB 0..1, as a 3DGS model renders it; kept as the degree-0 spherical
                          harmonic, so the colour is the same from every side

write() sets all of them in one Sdf.ChangeBlock: a renderer that read them one at a time could
see positions of one length and scales of another while N changes.

Import this only once Kit is running (after launch_simulation): it imports pxr, and a pxr loaded
before Kit leaves Kit's own USD libraries failing to load (undefined pxrInternal_v0_25_11 symbols).
"""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, Vt

TYPE_NAME = "ParticleField3DGaussianSplat"
SH_C0 = 0.28209479177387814  # the degree-0 spherical harmonic: colour = 0.5 + SH_C0 * coefficient

_ARRAYS = (
    ("positions", Sdf.ValueTypeNames.Point3fArray),
    ("scales", Sdf.ValueTypeNames.Float3Array),
    ("orientations", Sdf.ValueTypeNames.QuatfArray),
    ("opacities", Sdf.ValueTypeNames.FloatArray),
    ("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array),
)


def define(stage, path):
    """An empty splat prim at `path`, with every attribute write() sets already created."""
    prim = stage.DefinePrim(path, TYPE_NAME)
    if not prim.IsValid() or prim.GetTypeName() != TYPE_NAME:
        raise RuntimeError(f"could not define a {TYPE_NAME} at {path}: is omni.usd.schema.usd_particle_field loaded?")
    for name, value in (("projectionModeHint", "perspective"), ("sortingModeHint", "zDepth")):
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Token, custom=False,
                             variability=Sdf.VariabilityUniform).Set(value)
    prim.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int).Set(0)
    for name, type_name in _ARRAYS:
        prim.CreateAttribute(name, type_name)
    sh = UsdGeom.Primvar(prim.GetAttribute("radiance:sphericalHarmonicsCoefficients"))
    sh.SetElementSize(1)
    sh.SetInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    UsdGeom.Boundable(prim).CreateExtentAttr()
    return prim


def _f32(a, cols):
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim != (2 if cols else 1) or (cols and a.shape[1] != cols):
        raise ValueError(f"expected shape (N, {cols})" if cols else "expected shape (N,)")
    return a


def write(prim, positions, scales, orientations, opacities, colors):
    """Replace the prim's Gaussians (see the module docstring for the arrays). N may change."""
    p, s, q = _f32(positions, 3), _f32(scales, 3), _f32(orientations, 4)
    o, c = _f32(opacities, 0), _f32(colors, 3)
    n = len(p)
    if not len(s) == len(q) == len(o) == len(c) == n:
        raise ValueError(f"array lengths differ: {[len(a) for a in (p, s, q, o, c)]}")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    lo, hi = (p.min(axis=0), p.max(axis=0)) if n else (np.zeros(3), np.zeros(3))
    values = {
        "positions": Vt.Vec3fArray.FromNumpy(p),
        "scales": Vt.Vec3fArray.FromNumpy(s),
        # Vt.QuatfArray.FromNumpy takes the real part LAST (Gf.Quatf takes it first).
        "orientations": Vt.QuatfArray.FromNumpy(np.ascontiguousarray(q[:, [1, 2, 3, 0]])),
        "opacities": Vt.FloatArray.FromNumpy(np.clip(o, 0.0, 1.0)),
        "radiance:sphericalHarmonicsCoefficients": Vt.Vec3fArray.FromNumpy(
            np.ascontiguousarray((c - 0.5) / SH_C0, dtype=np.float32)),
        "primvars:displayColor": Vt.Vec3fArray.FromNumpy(np.clip(c, 0.0, 1.0)),
        "extent": Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]),
    }
    with Sdf.ChangeBlock():
        for name, value in values.items():
            prim.GetAttribute(name).Set(value)
