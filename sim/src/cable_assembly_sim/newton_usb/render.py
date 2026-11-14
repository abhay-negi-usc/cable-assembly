"""Low-resolution videos of USB insertion episodes, from recorded trajectories (no GPU, no GL: matplotlib -> ffmpeg).

A trajectory is what UsbNewtonEnv.snapshot() returns, stacked over steps, for a few worlds, saved as an .npz
(record()). Each world is drawn as a row of two panels:
  - 3D: the adapter, the plug with its overmold and 5 cm of cable, and the 2F-85's pads at the tilted grasp;
  - SECTION through the socket's x = 0 plane, zoomed: the receptacle pocket and tongue, and the plug's shell
    and tongue cavity -- the 0.04-0.07 mm fit that decides the insertion -- with the depth, the push, the
    detent and the outcome written over it.

    python -m cable_assembly_sim.newton_usb.render traj.npz out.mp4
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

TIP_MM = 4.837  # model.TIP_DEPTH [mm]
PLUG_DY_MM = 0.456  # model.PLUG_DY: the shell's centreline in the plug frame's y

PAD_COLOUR = (0.25, 0.35, 0.75)
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def _decimate(points, faces, target):
    faces = faces.reshape(-1, 3)
    if len(faces) <= target:
        return points, faces
    import fast_simplification

    p, f = fast_simplification.simplify(points.astype(np.float32), faces.astype(np.int32), target_reduction=1 - target / len(faces))
    return p, f


def _shade(points, faces, rgb):
    n = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    k = 0.35 + 0.65 * np.abs(n @ LIGHT)
    return np.clip(np.asarray(rgb)[None, :] * k[:, None], 0, 1)


def _box(half):
    hx, hy, hz = half
    c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return c, f


class Scene:
    """The meshes, decimated for drawing, in the socket frame (the plug's seated)."""

    def __init__(self):
        from .model import PAD_HALF, _read_meshes, PLUG_USD, SOCKET_USD

        drawn = _read_meshes(PLUG_USD, ("plug", "overmold", "cable"))
        sock = _read_meshes(SOCKET_USD, ("adapter", "receptacle"))
        mm = lambda pf: (pf[0].astype(np.float64) * 1e3, pf[1].reshape(-1, 3))  # noqa: E731
        self.plug_full = mm(drawn["plug"])
        self.receptacle_full = mm(sock["receptacle"])
        self.parts = {  # name: (points [mm], faces, colour)
            "plug": (*_decimate(*mm(drawn["plug"]), 900), (0.78, 0.78, 0.82)),
            "overmold": (*_decimate(*mm(drawn["overmold"]), 700), (0.12, 0.12, 0.13)),
            "cable": (*_decimate(*mm(drawn["cable"]), 250), (0.10, 0.10, 0.10)),
        }
        ap, af = _decimate(*mm(sock["adapter"]), 2500)
        keep = ap[af].mean(1)[:, 2] > -6.0  # the front: what the camera sees
        self.adapter = (ap, af[keep], (0.30, 0.30, 0.33))
        self.pad = (*_box(np.asarray(PAD_HALF) * 1e3), PAD_COLOUR)
        self._socket_section = None

    def socket_section(self):
        if self._socket_section is None:
            self._socket_section = _section(*self.receptacle_full, np.eye(3), np.zeros(3))
        return self._socket_section


def _section(points, faces, R, p, x=0.0):
    """Polylines (y, z) [mm] of the mesh posed by (R, p [mm]) cut by the socket's x = x plane."""
    import trimesh

    m = trimesh.Trimesh(points @ R.T + p, faces, process=False)
    s = m.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
    return [] if s is None else [e[:, 1:3] for e in s.discrete]


def record(snaps: list[dict], path: str, meta: dict | None = None) -> None:
    """Stack per-step snapshots ({name: array (worlds, ...)}) into an .npz (steps first)."""
    out = {k: np.stack([s[k] for s in snaps]) for k in snaps[0]}
    for k, v in (meta or {}).items():
        out[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(path, **out)


def render(traj_path: str, out_path: str, worlds: int = 2, fps: int = 20, dpi: int = 80, stride: int = 1,
           title: str = "") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = dict(np.load(traj_path))
    T, W = d["p_true"].shape[:2]
    worlds = min(worlds, W)
    scene = Scene()
    dt = float(d.get("meta_step_dt", 0.05))
    title = title or str(d.get("meta_title", ""))

    fig = plt.figure(figsize=(11.0, 3.2 * worlds), dpi=dpi)
    rows = []
    th = np.linspace(0, 2 * np.pi, 97)
    for w in range(worlds):
        ax3 = fig.add_subplot(worlds, 3, 3 * w + 1, projection="3d")
        ax3.set_proj_type("ortho")
        ax3.view_init(elev=18, azim=-58)
        ax3.set_xlim(-22, 22), ax3.set_ylim(-22, 22), ax3.set_zlim(-8, 58)
        ax3.set_box_aspect((44, 44, 66))
        ax3.set_axis_off()
        p, f, c = scene.adapter
        ax3.add_collection3d(Poly3DCollection(p[f], facecolors=_shade(p, f, c), edgecolors="none"))
        # The panel (model.panel_mesh), clipped to the view: a flat ring at its face.
        from .model import PANEL_R, PANEL_TOP_Z
        r0, r1, zp = PANEL_R[0] * 1e3, 22.0, PANEL_TOP_Z * 1e3
        ring = [[(r0 * np.cos(a), r0 * np.sin(a), zp), (r1 * np.cos(a), r1 * np.sin(a), zp),
                 (r1 * np.cos(b), r1 * np.sin(b), zp), (r0 * np.cos(b), r0 * np.sin(b), zp)] for a, b in zip(th[:-1], th[1:])]
        ax3.add_collection3d(Poly3DCollection(ring, facecolors=(0.72, 0.72, 0.68, 0.35), edgecolors="none"))
        moving = {}
        for name, (pp, ff, cc) in scene.parts.items():
            col = Poly3DCollection(pp[ff], facecolors=_shade(pp, ff, cc), edgecolors="none")
            ax3.add_collection3d(col)
            moving[name] = (col, pp, ff)
        pads = []
        for _ in range(2):
            pp, ff, cc = scene.pad
            col = Poly3DCollection(pp[ff], facecolors=_shade(pp, ff, cc), edgecolors="none", alpha=0.55)
            ax3.add_collection3d(col)
            pads.append((col, pp, ff))
        ax2 = fig.add_subplot(worlds, 3, 3 * w + 2)
        ax2.set_aspect("equal")
        ax2.set_xlim(-12, 12), ax2.set_ylim(-6, 18)
        ax2.tick_params(labelsize=6)
        ax2.set_xlabel("socket y [mm]", fontsize=7), ax2.set_ylabel("z [mm]", fontsize=7)
        for line in scene.socket_section():
            ax2.plot(line[:, 0], line[:, 1], color="0.25", lw=0.8)
        ax2.plot([-12, -PANEL_R[0] * 1e3], [zp, zp], color="0.55", lw=0.8)
        ax2.plot([PANEL_R[0] * 1e3, 12], [zp, zp], color="0.55", lw=0.8)
        txt = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va="top", ha="left", fontsize=6.5, family="monospace")
        # Top view: the plug tip's track over the opening (12.5 x 5.1 mm) and the plate's rim.
        axt = fig.add_subplot(worlds, 3, 3 * w + 3)
        axt.set_aspect("equal")
        axt.set_xlim(-14, 14), axt.set_ylim(-14, 14)
        axt.tick_params(labelsize=6)
        axt.set_xlabel("socket x [mm]", fontsize=7), axt.set_ylabel("socket y [mm]", fontsize=7)
        axt.plot(11.7 * np.cos(th), 11.7 * np.sin(th), color="0.55", lw=0.8)
        axt.add_patch(plt.Rectangle((-6.25, -2.09), 12.5, 5.12, fill=False, color="0.25", lw=0.8))
        track, = axt.plot([], [], color="#2a78d6", lw=0.8)
        tip_box, = axt.plot([], [], color="tab:red", lw=0.9)
        axt.set_title("top: tip track", fontsize=7)
        rows.append((ax3, moving, pads, ax2, [], txt, (track, tip_box)))
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w_px, h_px = fig.canvas.get_width_height()
    part = out_path + ".part"  # renamed when complete: uploaders watching for *.mp4 never see a partial file
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                           "-s", f"{w_px}x{h_px}", "-r", str(fps / stride), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "28", "-f", "mp4", part], stdin=subprocess.PIPE)
    for t in range(0, T, stride):
        for w, (ax3, moving, pads, ax2, lines, txt, (track, tip_box)) in enumerate(rows):
            R, p = d["R_true"][t, w], d["p_true"][t, w] * 1e3
            tips = d["p_true"][: t + 1, w] * 1e3 - d["R_true"][: t + 1, w, :, 2] * TIP_MM
            track.set_data(tips[:, 0], tips[:, 1])
            corners = np.array([[-6, -2.25, 0], [6, -2.25, 0], [6, 2.25, 0], [-6, 2.25, 0], [-6, -2.25, 0]]) + [0, PLUG_DY_MM, 0]
            c = (corners + [0, 0, -TIP_MM]) @ R.T + p
            tip_box.set_data(c[:, 0], c[:, 1])
            for col, pp, ff_ in moving.values():
                col.set_verts((pp @ R.T + p)[ff_])
            for k, (col, pp, ff_) in enumerate(pads):
                Rk, pk = d["R_pad"][t, w, k], d["p_pad"][t, w, k] * 1e3
                col.set_verts((pp @ Rk.T + pk)[ff_])
            for ln in lines:
                ln.remove()
            lines.clear()
            for seg in _section(*scene.plug_full, R, p):
                lines.append(ax2.plot(seg[:, 0], seg[:, 1], color="tab:red", lw=0.9)[0])
            state = "SEATED" if d["seated"][t, w] else ("engaged" if d["engaged"][t, w] else "")
            if d.get("voided") is not None and d["voided"][t, w]:
                state = "VOID"
            err = d.get("err")
            err_s = "" if err is None else f"\nerr scale {float(d['scale'][t, w]):.2f}"
            spd = "" if d.get("speed") is None else f"  v {d['speed'][t, w] * 1e3:4.1f} mm/s"
            if d.get("slip") is not None and d["slip"][:, w].max() > 0:
                spd += f"\nslipped {d['slip'][t, w] * 1e3:4.1f} mm"
            txt.set_text(f"t {t * dt:5.2f} s   world {w}\nabove seat {d['h'][t, w] * 1e3:6.2f} mm\n"
                         f"|F| {d['force'][t, w]:5.1f} N  push {d['push'][t, w]:5.1f} N\n"
                         f"detent {d['detent'][t, w]:4.1f} N{spd}{err_s}\n{state}")
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    os.replace(part, out_path)
    return out_path


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]))
