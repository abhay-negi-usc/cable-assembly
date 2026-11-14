"""Plots of a USB training run's metrics, and a low-res video of its latest evaluation (newton_usb_train.py runs
this in the background at every monitor point; it also runs on its own, on a run that is going or done).

    <run>/plots/metrics.png         small multiples: curriculum, success, reward, forces, losses, speed
    <run>/videos/iter_<it>.mp4      the deterministic policy, from <run>/videos/iter_<it>.npz

    ./run_usb.sh python scripts/newton_usb_monitor.py logs/rsl_rl/usb_newton/<run> [--video <npz>]
"""

import argparse
import csv
import glob
import os

import numpy as np

INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6a64", "#e4e3de", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (dataviz reference palette, validated)

PANELS = [  # (tensorboard tag, title, y limits or None, reference lines (value, label))
    ("/metrics/noise_scale", "Curriculum scale (1 = 10 mm, 10 deg at tip)", (0, 1.05), ()),
    ("/metrics/seated_at_end", "Seated at episode end (training, stochastic)", (0, 1.02), ()),
    ("/metrics/curriculum_rate", "Success at the scale being judged", (0, 1.02), ()),
    ("/metrics/peak_force", "Mean peak hand force per episode [N]", (0, None), ((6, "penalised above"), (12, "cap"))),
    ("/metrics/end_force", "Mean hand force at episode end [N]", (0, None), ((2, "hold limit"),)),
    ("/metrics/peak_contact_speed_mm_s", "Mean peak axial speed in contact [mm/s]", (0, None), ((5, "penalised above"),)),
    ("Train/mean_reward", "Mean episode return", None, ()),
    ("Policy/mean_std", "Policy action std", None, ()),
    ("/metrics/artifact_kicked", "Simulator artifacts, cumulative (kicked)", None, ()),
    ("Loss/value", "Value loss", None, ()),
    ("Loss/surrogate", "Surrogate loss", None, ()),
    ("Perf/total_fps", "Env steps per second", None, ()),
]


def scalars(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for f in sorted(glob.glob(os.path.join(run, "events.out.tfevents.*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            ev = ea.Scalars(tag)
            s, v = out.setdefault(tag, ([], []))
            s.extend(e.step for e in ev)
            v.extend(e.value for e in ev)
    return {k: (np.asarray(s), np.asarray(v)) for k, (s, v) in out.items()}


def smooth(v, n=10):
    if len(v) < n:
        return v
    k = np.ones(n) / n
    return np.concatenate((np.cumsum(v[: n - 1]) / np.arange(1, n), np.convolve(v, k, mode="valid")))


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def plot(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = scalars(run)
    ev_path = os.path.join(run, "eval.csv")
    ev = list(csv.DictReader(open(ev_path))) if os.path.exists(ev_path) else []
    fig, axes = plt.subplots(5, 3, figsize=(12, 13.5), facecolor=SURFACE)
    axes = axes.ravel()
    for ax, (tag, title, ylim, refs) in zip(axes, PANELS):
        style(ax, title)
        if tag in sc and len(sc[tag][0]):
            s, v = sc[tag]
            ax.plot(s, v, color=BLUE, lw=0.8, alpha=0.3)
            ax.plot(s, smooth(v), color=BLUE, lw=2)
        else:
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", color=MUTED, fontsize=8)
        for y, lab in refs:
            ax.axhline(y, color=MUTED, lw=0.8, ls="--")
            ax.annotate(lab, (0.0, y), xycoords=("axes fraction", "data"), xytext=(3, 2), textcoords="offset points",
                        fontsize=6.5, color=MUTED, va="bottom")
        if ylim:
            ax.set_ylim(*ylim)
    # Deterministic evaluations: at the curriculum's scale and at full uncertainty. Two series: legend + labels.
    ax = axes[len(PANELS)]
    style(ax, "Evaluation: seated at end (deterministic)")
    if ev:
        it = np.array([int(r["iteration"]) for r in ev])
        for key, col, lab in (("seated_cur", BLUE, "at curriculum scale"), ("seated_full", ORANGE, "at full uncertainty")):
            y = np.array([float(r[key]) for r in ev])
            ax.plot(it, y, color=col, lw=2, marker="o", ms=4, label=lab)
            ax.annotate(f"{y[-1]:.2f}", (it[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK, va="center")
        ax.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK)
    ax.set_ylim(0, 1.05)
    ax = axes[len(PANELS) + 1]
    style(ax, "Evaluation: median time to seat [s]")
    if ev:
        it = np.array([int(r["iteration"]) for r in ev])
        y = np.array([float(r["t_seat_full"]) if r["t_seat_full"] not in ("", "nan") else np.nan for r in ev])
        ax.plot(it, y, color=ORANGE, lw=2, marker="o", ms=4)
    ax = axes[len(PANELS) + 2]
    style(ax, "Reward terms (per step)")
    terms = sorted(t for t in sc if t.startswith("/reward/"))
    if terms:
        for t in terms:
            s, v = sc[t]
            ax.plot(s, smooth(v), lw=1.5, color=MUTED)
            ax.annotate(t.split("/")[-1], (s[-1], smooth(v)[-1]), xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color=INK, va="center")
    for ax in axes:
        ax.set_xlabel("iteration", fontsize=7, color=MUTED)
    fig.suptitle(f"USB-A insertion -- {os.path.basename(os.path.normpath(run))}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.join(run, "plots"), exist_ok=True)
    out = os.path.join(run, "plots", "metrics.png")
    fig.savefig(out, dpi=90, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--video", default=None, help="an evaluation trajectory (.npz) to render beside it")
    p.add_argument("--worlds", type=int, default=2)
    a = p.parse_args()
    print("plots:", plot(a.run), flush=True)
    if a.video:
        from cable_assembly_sim.newton_usb import render

        print("video:", render.render(a.video, os.path.splitext(a.video)[0] + ".mp4", worlds=a.worlds), flush=True)


if __name__ == "__main__":
    main()
