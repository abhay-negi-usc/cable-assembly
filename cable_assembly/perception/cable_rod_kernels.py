"""Warp kernels for perception/cable_model.py's CableRod: one XPBD step of a chain of nodes --
predict under gravity, then stretch, bending, the pull toward observed points, and resting on the
surface the camera saw (a height field), then velocities.

Kept apart so `import cable_model` needs no Warp until a rod is made. Warp compiles these on first
use (cached under ~/.cache/warp)."""

import warp as wp


@wp.kernel
def predict(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=float),
            x_prev: wp.array(dtype=wp.vec3), gravity: wp.vec3, damping: float, dt: float):
    i = wp.tid()
    x_prev[i] = x[i]
    if w[i] > 0.0:
        v[i] = v[i] * damping + gravity * dt
        x[i] = x[i] + v[i] * dt


@wp.kernel
def stretch(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), rest: float, compliance: float,
            dt: float, parity: int):
    k = wp.tid() * 2 + parity                                  # alternate segments: no write races
    if k + 1 >= x.shape[0]:
        return
    a, b = x[k], x[k + 1]
    wa, wb = w[k], w[k + 1]
    d = b - a
    n = wp.length(d)
    if n < 1.0e-9 or wa + wb == 0.0:
        return
    alpha = compliance / (dt * dt)
    c = n - rest
    dl = -c / (wa + wb + alpha)
    g = d / n
    x[k] = a - g * dl * wa
    x[k + 1] = b + g * dl * wb


@wp.kernel
def bend(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), rest: float, compliance: float,
         dt: float, phase: int):
    k = wp.tid() * 3 + phase                                   # every third triplet at a time
    if k + 2 >= x.shape[0]:
        return
    a, c = x[k], x[k + 2]
    wa, wc = w[k], w[k + 2]
    d = c - a
    n = wp.length(d)
    if n < 1.0e-9 or wa + wc == 0.0:
        return
    alpha = compliance / (dt * dt)
    cc = n - 2.0 * rest                                       # straight: the ends two segments apart
    if cc >= 0.0:
        return
    dl = -cc / (wa + wc + alpha)
    g = d / n
    x[k] = a - g * dl * wa
    x[k + 2] = c + g * dl * wc


@wp.kernel
def pull(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), target: wp.array(dtype=wp.vec3),
         has: wp.array(dtype=int), compliance: float, dt: float):
    i = wp.tid()
    if has[i] == 0 or w[i] == 0.0:
        return
    alpha = compliance / (dt * dt)
    x[i] = x[i] + (target[i] - x[i]) * (w[i] / (w[i] + alpha))


@wp.func
def height_at(height: wp.array2d(dtype=float), lo: wp.vec2, cell: float, p: wp.vec3) -> float:
    q = wp.vec2((p[0] - lo[0]) / cell - 0.5, (p[1] - lo[1]) / cell - 0.5)   # cell centres
    nx, ny = height.shape[0], height.shape[1]
    fx = wp.clamp(q[0], 0.0, float(nx - 1) - 1.0e-4)
    fy = wp.clamp(q[1], 0.0, float(ny - 1) - 1.0e-4)
    i, j = int(fx), int(fy)
    tx, ty = fx - float(i), fy - float(j)
    return wp.lerp(wp.lerp(height[i, j], height[i + 1, j], tx), wp.lerp(height[i, j + 1], height[i + 1, j + 1], tx), ty)


@wp.kernel
def rest_on(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), height: wp.array2d(dtype=float),
            lo: wp.vec2, cell: float, radius: float):
    """Keep each node at least `radius` above the surface the camera saw beneath it."""
    i = wp.tid()
    if w[i] == 0.0:
        return
    p = x[i]
    floor = height_at(height, lo, cell, p) + radius
    if p[2] < floor:
        x[i] = wp.vec3(p[0], p[1], floor)


@wp.kernel
def velocities(x: wp.array(dtype=wp.vec3), x_prev: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), dt: float):
    i = wp.tid()
    v[i] = (x[i] - x_prev[i]) / dt
