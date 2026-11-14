"""Start the next training stage when the current one has saturated.

Watches a running newton_usb_train.py run's eval.csv. When the curriculum is at its top (scale >= --scale) and the
deterministic success at full uncertainty, averaged over the last --window evaluations, is at least --success, it
stops that run (its last checkpoint is the one that evaluation used: evaluations run at checkpoints) and starts
the next stage from it with the given training arguments. If the run ends first, it exits without starting one.

    ./run_usb.sh python scripts/newton_usb_chain.py --run logs/rsl_rl/usb_newton/<stage-1 run> --pid <its pid> \\
        --log logs/rsl_rl/usb_newton/train_stage2.log -- --slip 1 --hours 12 --curriculum_start 0.5
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def alive(pid):
    try:
        with open(f"/proc/{pid}/cmdline") as f:
            return "newton_usb_train.py" in f.read()
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--pid", type=int, required=True, help="the training process writing --run")
    p.add_argument("--log", required=True, help="stdout of the next stage")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--success", type=float, default=0.95)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--poll_s", type=float, default=120.0)
    p.add_argument("next", nargs=argparse.REMAINDER, help="-- then the next stage's newton_usb_train.py arguments")
    a = p.parse_args()
    nxt = [x for x in a.next if x != "--"]
    path = os.path.join(a.run, "eval.csv")
    pid = a.pid
    if not alive(pid):
        raise SystemExit(f"pid {pid} is not a newton_usb_train.py process")
    print(f"watching {path} (training pid {pid})", flush=True)
    while True:
        rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
        top = [r for r in rows if float(r["scale"]) >= a.scale - 1e-6]
        if len(top) >= a.window:
            last = top[-a.window:]
            mean = sum(float(r["seated_full"]) for r in last) / a.window
            print(f"{time.strftime('%H:%M')} iteration {last[-1]['iteration']}: full-uncertainty success over the last "
                  f"{a.window} evaluations {mean:.3f}", flush=True)
            if mean >= a.success:
                ckpt = os.path.join(a.run, f"model_{int(last[-1]['iteration'])}.pt")
                break
        if not alive(pid):
            print("the run ended before saturating: no next stage", flush=True)
            return 1
        time.sleep(a.poll_s)
    if alive(pid):
        os.kill(pid, signal.SIGINT)
        for _ in range(120):
            if not alive(pid):
                break
            time.sleep(1)
        else:
            os.kill(pid, signal.SIGKILL)
    cmd = [sys.executable, os.path.join(HERE, "newton_usb_train.py"), "--init_from", ckpt, *nxt]
    print(f"saturated: starting the next stage from {ckpt}:\n  {' '.join(cmd)}", flush=True)
    with open(a.log, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
