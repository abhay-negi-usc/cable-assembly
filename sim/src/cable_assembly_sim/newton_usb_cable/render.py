"""Low-resolution videos of cable-held USB insertion episodes (UsbCableEnv.snapshot() stacked over steps): the
newton_usb renderer's three panels, with the bending cable drawn from its centreline (hand to overmold) and
the pads riding on the hand, up the cable.

    python -m cable_assembly_sim.newton_usb_cable.render traj.npz out.mp4 [worlds]
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from cable_assembly_sim.newton_usb.render import PLUG_DY_MM, TIP_MM, Scene, _section, _shade, record  # noqa: F401


def render(traj_path: str, out_path: str, worlds: int = 2, fps: int = 20, dpi: int = 80, stride: int = 1,
           title: str = "") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from cable_assembly_sim.newton_usb.model import PANEL_R, PANEL_TOP_Z

    d = dict(np.load(traj_path))
    T, W = d["p_true"].shape[:2]
    worlds = min(worlds, W)
    scene = Scene()
    scene.parts.pop("cable", None)  # the rigid stub: drawn from the centreline instead
    dt = float(d.get("meta_step_dt", 0.05))
    title = title or str(d.get("meta_title", ""))

    fig = plt.figure(figsize=(12.0, 3.6 * worlds), dpi=dpi)
    rows = []
    th = np.linspace(0, 2 * np.pi, 97)
    for w in range(worlds):
        ax3 = fig.add_subplot(worlds, 3, 3 * w + 1, projection="3d")
        ax3.set_proj_type("ortho")
        ax3.view_init(elev=14, azim=-58)
        ax3.set_xlim(-32, 32), ax3.set_ylim(-32, 32), ax3.set_zlim(-8, 140)
        ax3.set_box_aspect((64, 64, 148))
        ax3.set_axis_off()
        p, f, c = scene.adapter
        ax3.add_collection3d(Poly3DCollection(p[f], facecolors=_shade(p, f, c), edgecolors="none"))
        r0, r1, zp = PANEL_R[0] * 1e3, 30.0, PANEL_TOP_Z * 1e3
        ring = [[(r0 * np.cos(a), r0 * np.sin(a), zp), (r1 * np.cos(a), r1 * np.sin(a), zp),
                 (r1 * np.cos(b), r1 * np.sin(b), zp), (r0 * np.cos(b), r0 * np.sin(b), zp)] for a, b in zip(th[:-1], th[1:])]
        ax3.add_collection3d(Poly3DCollection(ring, facecolors=(0.72, 0.72, 0.68, 0.35), edgecolors="none"))
        moving = {}
        for name, (pp, ff, cc) in scene.parts.items():
            col = Poly3DCollection(pp[ff], facecolors=_shade(pp, ff, cc), edgecolors="none")
            ax3.add_collection3d(col)
            moving[name] = (col, pp, ff)
        cable, = ax3.plot([], [], [], color="0.08", lw=4.0, solid_capstyle="round")
        nodes, = ax3.plot([], [], [], "o", color="#d0a020", ms=2.0)
        pads = []
        for _ in range(2):
            pp, ff, cc = scene.pad
            col = Poly3DCollection(pp[ff], facecolors=_shade(pp, ff, cc), edgecolors="none", alpha=0.45)
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
        rows.append((ax3, moving, (cable, nodes), pads, ax2, [], txt, (track, tip_box)))
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w_px, h_px = fig.canvas.get_width_height()
    part = out_path + ".part"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                           "-s", f"{w_px}x{h_px}", "-r", str(fps / stride), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "28", "-f", "mp4", part], stdin=subprocess.PIPE)
    for t in range(0, T, stride):
        for w, (ax3, moving, (cable, nodes), pads, ax2, lines, txt, (track, tip_box)) in enumerate(rows):
            R, p = d["R_true"][t, w], d["p_true"][t, w] * 1e3
            tips = d["p_true"][: t + 1, w] * 1e3 - d["R_true"][: t + 1, w, :, 2] * TIP_MM
            track.set_data(tips[:, 0], tips[:, 1])
            corners = np.array([[-6, -2.25, 0], [6, -2.25, 0], [6, 2.25, 0], [-6, 2.25, 0], [-6, -2.25, 0]]) + [0, PLUG_DY_MM, 0]
            c = (corners + [0, 0, -TIP_MM]) @ R.T + p
            tip_box.set_data(c[:, 0], c[:, 1])
            for col, pp, ff_ in moving.values():
                col.set_verts((pp @ R.T + p)[ff_])
            cp = d["cable"][t, w] * 1e3
            cable.set_data_3d(cp[:, 0], cp[:, 1], cp[:, 2])
            nodes.set_data_3d(cp[2:-1, 0], cp[2:-1, 1], cp[2:-1, 2])
            for k, (col, pp, ff_) in enumerate(pads):
                Rk, pk = d["R_pad"][t, w, k], d["p_pad"][t, w, k] * 1e3
                col.set_verts((pp @ Rk.T + pk)[ff_])
            for ln in lines:
                ln.remove()
            lines.clear()
            for seg in _section(*scene.plug_full, R, p):
                lines.append(ax2.plot(seg[:, 0], seg[:, 1], color="tab:red", lw=0.9)[0])
            state = "SEATED" if d["seated"][t, w] else ("engaged" if d["engaged"][t, w] else "")
            if d["voided"][t, w]:
                state = "VOID"
            ctrl = d["ctrl"][t, w] * np.array([1.5, 3.0, 3.0, 5.0])
            txt.set_text(f"t {t * dt:5.2f} s   world {w}\nabove seat {d['h'][t, w] * 1e3:6.2f} mm\n"
                         f"|F| {d['force'][t, w]:5.1f} N  push {d['push'][t, w]:5.1f} N\n"
                         f"detent {d['detent'][t, w]:4.1f} N\n"
                         f"cable EI {d['EI'][t, w] * 1e3:4.2f}e-3  bend {np.degrees(d['bend'][t, w]):4.1f} deg\n"
                         f"grasp ctrl err y {ctrl[0]:+.1f} z {ctrl[1]:+.1f} mm\n"
                         f"  rot {ctrl[2]:+.1f} roll {ctrl[3]:+.1f} deg\n"
                         f"err scale {float(d['scale'][t, w]):.2f}\n{state}")
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    os.replace(part, out_path)
    return out_path


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]))
