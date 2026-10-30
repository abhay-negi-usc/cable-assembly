"""The live view's parts that need no Isaac Sim, shared by its two viewers:

    scripts/live_view.py      Isaac Sim: Kit and the RTX renderer
    scripts/live_view_gl.py   Newton's OpenGL viewer: no Kit, a fraction of the GPU

The task's state stream (cable_assembly/robot/live_view.py), the cell file, the GPU memory check
before a viewer starts, the Robotiq counts -> finger angle mapping and what the status panel says.
"""

import collections
import json
import math
import os
import re
import socket
import struct
import subprocess
import threading
import time

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Robotiq counts -> the model's finger_joint, as scripts/bnc_cell_server.py maps them: the 2F-85's
# fingertip angle is linear in counts (cable_assembly/robot/gripper_kinematics.py, 0.2186 deg a
# count), and the model's linkage closes at 45 deg.
DEG_PER_COUNT = 0.2186
FINGER_MAX_DEG = 45.0

# GPU memory the task's SAM3 needs, GiB: measured 4.40 from its first frame on (3.33 of it weights),
# on the D415 test frame with the viewer running. The extra 0.6 is a margin.
TASK_GPU_GIB = 5.0
TASK_CMD = re.compile(r"cable_assembly[./]apps")  # python -m cable_assembly.apps.bnc_pickup ...

FRAMES_PORT = 6009


def finger_rad(counts):
    """The model's finger_joint for a Robotiq position, 0 open .. 255 closed."""
    return math.radians(min(FINGER_MAX_DEG, max(0.0, float(counts)) * DEG_PER_COUNT))


def load_cell(name):
    """(configs/cells/<name>.yaml as a dict, its path)."""
    path = os.path.join(REPO, "configs", "cells", f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}, path


class Stream:
    """The task's datagrams, read without ever blocking. Keeps the newest one a model with these
    arm `joints` can show.

    ONE SENDER AT A TIME. Anything that runs a Robot with the cell's live_view streams to the same
    port -- a dry-run rehearsal, the test suite -- and taking the newest sample from whoever sent
    it flashed their arm into the live one's. So the sender shown keeps the view until it has been
    silent for `hold_s`; others are ignored meanwhile (said once each). After that the first to
    send takes over. A real arm takes over from a dry run at once; a dry run never displaces a
    real arm that is still streaming."""

    def __init__(self, bind, port, joints, robot="iiwa14", hold_s=1.0):
        self.joints, self.robot, self.hold_s = list(joints), robot, hold_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port))
        self.sock.setblocking(False)
        self.address = f"udp://{bind}:{port}"
        self.latest = None  # the newest valid sample
        self.sender = None
        self.t_rx = None  # time.monotonic() it arrived
        self.live = False
        self._ever_live = False
        self._arrivals = []  # for the rate shown
        self._rejected = set()
        self._ignored = set()  # senders ignored while another held the view
        self._behind = collections.deque(maxlen=30)  # how old the sample shown was, s

    def poll(self):
        """Drain the socket, keep the newest sample this model can show. True if one arrived."""
        got = False
        while True:
            try:
                data, addr = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                return got
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            why = self._unusable(msg)
            if why:
                if why not in self._rejected:  # once each, not 60 times a second
                    self._rejected.add(why)
                    print(f"[live] ignoring the stream from {addr[0]}:{addr[1]}: {why}", flush=True)
                continue
            now = time.monotonic()
            real_over_dry = self.latest is not None and self.latest.get("dry_run") and not msg.get("dry_run")
            if self.sender is not None and addr != self.sender and now - self.t_rx < self.hold_s and not real_over_dry:
                if addr not in self._ignored:
                    self._ignored.add(addr)
                    print(f"[live] ignoring a second stream from {addr[0]}:{addr[1]}"
                          f"{' (a dry run)' if msg.get('dry_run') else ''} while {self.sender[0]}:{self.sender[1]} "
                          f"is live", flush=True)
                continue
            if self.sender != addr:
                print(f"[live] receiving {msg['robot']} from {addr[0]}:{addr[1]}"
                      f"{' (a DRY RUN)' if msg.get('dry_run') else ''}", flush=True)
                self._ignored.discard(addr)
            self.latest, self.sender, self.t_rx = msg, addr, now
            self._arrivals.append(now)
            got = True

    def _unusable(self, msg):
        if not isinstance(msg, dict) or msg.get("v") != 1:
            return f"not a version-1 live-view sample ({str(msg)[:60]})"
        if msg.get("robot") != self.robot or msg.get("joints") != self.joints:
            return (f"this model is the {self.robot} ({self.joints}); the task drives a {msg.get('robot')} "
                    f"({msg.get('joints')})")
        q = msg.get("q")
        if (not isinstance(q, list) or len(q) != len(self.joints)
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in q)):
            return f"bad joint vector {q}"
        return None

    def age(self):
        """Seconds since the newest sample arrived, or None before the first."""
        return None if self.t_rx is None else time.monotonic() - self.t_rx

    def rate_hz(self):
        now = time.monotonic()
        self._arrivals = [t for t in self._arrivals if now - t < 1.0]
        return len(self._arrivals)

    def check_live(self, stale_s):
        """Whether a sample came in the last `stale_s`; says so on the console when that changes
        (not for the wait before the first sample)."""
        age = self.age()
        now_live = age is not None and age < stale_s
        if now_live != self.live and self.latest is not None and (self._ever_live or not now_live):
            print(f"[live] {'stream back' if now_live else f'stream lost: no sample for {stale_s:.1f} s'}", flush=True)
        self.live = now_live
        self._ever_live |= now_live
        return now_live

    def describe(self, finger):
        """What a status panel shows: (state, status, phase, joints, gripper). state is one of
        waiting | live | dry_run | stale; `finger` is the finger angle shown, rad."""
        msg = self.latest
        if msg is None:
            return "waiting", f"waiting for the task on {self.address}", "phase: --", "A1..A7 [deg]  --", "gripper: --"
        if self.live:
            state = "dry_run" if msg.get("dry_run") else "live"
            status = f"LIVE  {self.rate_hz()} Hz from {self.sender[0]}" + ("  (DRY RUN)" if msg.get("dry_run") else "")
            # How long ago the arm was where it is drawn: the sample's age as this frame is
            # drawn, the task's clock against ours -- shown only when they agree (the same PC).
            behind = time.time() - float(msg.get("t", 0.0))
            if 0.0 <= behind < 2.0:
                self._behind.append(behind)
            if self._behind:
                status += f"  {1000.0 * sorted(self._behind)[len(self._behind) // 2]:.0f} ms behind"
        else:
            state, status = "stale", f"NO DATA for {self.age():.1f} s -- showing the last pose"
        q_deg = " ".join(f"{math.degrees(x):+6.1f}" for x in msg["q"])
        g = msg.get("gripper")
        return (state, status, f"phase: {msg.get('phase') or '--'}", f"A1..A7 [deg]  {q_deg}",
                f"gripper: {g} counts ({math.degrees(finger):.0f} deg)" if g is not None else "gripper: --")

    def close(self):
        self.sock.close()


class FrameReceiver:
    """The camera frames the task sends (cable_assembly/robot/live_view.py, FrameSender), over
    TCP. A thread of its own accepts the task -- again whenever it restarts -- and hands each
    frame to `on_frame` as rgbd.load() would return it from a capture: rgb, depth_m, K, T_base_cam,
    T_base_tool0 (and T_tool0_fingertip), plus phase, seq and t. The task's overlays go to
    `on_overlay` (the header dict: name, text, items), if given. Both run in that thread."""

    def __init__(self, bind, port, on_frame, on_overlay=None):
        self.on_frame, self.on_overlay = on_frame, on_overlay
        self.server = socket.create_server((bind, port), reuse_port=False)
        self.server.settimeout(0.5)
        self.port = self.server.getsockname()[1]  # the one given, or the free one port 0 got
        self.address = f"tcp://{bind}:{self.port}"
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-frames", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                conn, addr = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            print(f"[live] camera frames from {addr[0]}:{addr[1]}", flush=True)
            with conn:
                conn.settimeout(1.0)
                try:
                    while not self._stop.is_set():
                        frame = self._read(conn)
                        if frame is None:
                            break
                        if frame.get("kind") == "overlay":
                            if self.on_overlay is not None:
                                self.on_overlay(frame)
                            continue
                        self.frames += 1
                        self.on_frame(frame)
                except (OSError, ValueError) as exc:
                    print(f"[live] camera frames: connection dropped ({exc})", flush=True)

    def _read(self, conn):
        """One frame, or None when the task closed the connection."""
        head = self._exactly(conn, 4)
        if head is None:
            return None
        h = json.loads(self._exactly(conn, struct.unpack(">I", head)[0], required=True))
        if h.get("v") == 1 and h.get("kind") == "overlay":
            return h
        if h.get("v") != 1 or h.get("kind") != "rgbd":
            raise ValueError(f"not a version-1 rgbd frame or overlay: {str(h)[:80]}")
        w, ht = int(h["w"]), int(h["h"])
        rgb = np.frombuffer(self._exactly(conn, int(h["rgb_bytes"]), required=True), np.uint8).reshape(ht, w, 3)
        depth = np.frombuffer(self._exactly(conn, int(h["depth_bytes"]), required=True), np.uint16).reshape(ht, w)
        frame = {"kind": "rgbd", "rgb": rgb, "depth_m": depth.astype(np.float32) * np.float32(h["depth_unit"]),
                 "K": np.array(h["K"]), "T_base_cam": np.array(h["T_base_cam"]),
                 "T_base_tool0": np.array(h["T_base_tool0"]), "robot": h.get("robot"),
                 "phase": h.get("phase", ""), "seq": h.get("seq"), "t": h.get("t")}
        if h.get("T_tool0_fingertip") is not None:
            frame["T_tool0_fingertip"] = np.array(h["T_tool0_fingertip"])
        return frame

    def _exactly(self, conn, n, required=False):
        """n bytes, waiting out the socket's timeouts while the viewer runs; None if the peer
        closed before the first byte (and `required` is off)."""
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = conn.recv(min(n - len(buf), 1 << 20))
            except TimeoutError:
                if self._stop.is_set():
                    raise OSError("viewer closing") from None
                continue
            if not chunk:
                if buf or required:
                    raise OSError("the task closed the connection mid-frame")
                return None
            buf += chunk
        return bytes(buf)

    def close(self):
        self._stop.set()
        self.server.close()
        self._thread.join(timeout=2.0)


class Overlays:
    """The task's overlays by name, as the viewer last heard them (a newer one of a name replaces
    it); `version` counts the changes, for a viewer to redraw on."""

    def __init__(self):
        self._by_name = {}
        self.version = 0
        self._lock = threading.Lock()

    def put(self, overlay):
        with self._lock:
            self._by_name[overlay["name"]] = overlay
            self.version += 1

    def snapshot(self):
        """({name: overlay}, version)."""
        with self._lock:
            return dict(self._by_name), self.version

    def text(self):
        """Every overlay's panel lines, by name."""
        by_name, _ = self.snapshot()
        return [line for name in sorted(by_name) for line in by_name[name].get("text", [])]


def gpu_users():
    """(free GiB, [(GiB, pid, command)]) on GPU 0 -- where the viewer and SAM3 both run -- or None
    without nvidia-smi."""

    def smi(query):
        out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return [[f.strip() for f in line.split(",")] for line in out.splitlines() if line.strip()]

    try:
        uuid, free = smi("--query-gpu=uuid,memory.free")[0]
        users = []
        for gpu, pid, mib in smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
            if gpu != uuid:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = f.read().decode(errors="replace").strip("\0").split("\0")
                cmd = " ".join([os.path.basename(argv[0]), *argv[1:]])  # python, not its venv path
            except OSError:
                cmd = "?"
            users.append((float(mib) / 1024.0, int(pid), cmd))
        return float(free) / 1024.0, users
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def gpu_shortfall(free_gib, users, task_gib, viewer_gib):
    """Why the viewer must not start, or None. The viewer takes `viewer_gib` at start and keeps it.
    The task loads SAM3 as it starts and needs up to `task_gib` by its first frame, so a viewer
    started into a GPU that cannot hold both leaves the task to die of torch.OutOfMemoryError.
    Started after the task, the viewer only has to leave what the task has not taken yet."""
    task_has = sum(gib for gib, _, cmd in users if TASK_CMD.search(cmd))
    if free_gib >= viewer_gib + max(task_gib - task_has, 0.0):
        return None
    lines = [(f"not starting. The GPU has {free_gib:.1f} GiB free; this viewer takes {viewer_gib:.1f}, "
              f"and the task's SAM3 needs up to {task_gib:.1f} (it holds {task_has:.1f} now). The task "
              "would run out of GPU memory. Using the GPU:")]
    lines += [f"    {gib:4.1f} GiB  pid {pid:<8d} {cmd[:140]}" for gib, pid, cmd in sorted(users, reverse=True)]
    lines.append("Stop one of them, run the viewer on another PC (sim/README.md), or pass --task_gpu_gib 0 to start anyway.")
    return "\n".join(lines)


def check_gpu(task_gpu_gib, bind, viewer_gib):
    """Exit, saying why, if starting this viewer would leave the task's SAM3 short of GPU memory.
    `task_gpu_gib` None: TASK_GPU_GIB when `bind` is loopback, else 0 (the task is on another PC
    and does not share this GPU). 0 skips the check."""
    task_gib = task_gpu_gib
    if task_gib is None:
        task_gib = TASK_GPU_GIB if bind in ("127.0.0.1", "localhost") else 0.0
    if task_gib <= 0.0:
        return
    gpu = gpu_users()
    if gpu is None:
        print("[live] nvidia-smi failed: GPU memory not checked.", flush=True)
    elif why := gpu_shortfall(*gpu, task_gib, viewer_gib):
        raise SystemExit(f"[live] {why}")
