"""RSL-RL log writer for Weights & Biases, for the USB insertion runs (newton_usb_train.py --wandb_project).

Like rsl_rl's WandbLogWriter (scalars go to TensorBoard and W&B, at the learning iteration as the step), with:

- the run named <run dir>_<run_name>, in a group, with the script's arguments in its config;
- the evaluation videos (<run>/videos/iter_<it>.mp4, rendered in the background) logged once each under
  "eval/video", captioned with the iteration they show. rsl_rl offers every *.mp4 under the run at each
  iteration; render.render() writes to a .part file and renames it, so a video is never taken half-written;
- checkpoints uploaded only every `model_every` iterations (6.7 MB each; save_file() uploads the last one);
- W&B's own files kept beside the run directory, not inside it (its media copies are *.mp4 files too).
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import asdict, is_dataclass

import wandb
from rsl_rl.utils.wandb_log_writer import WandbLogWriter
from torch.utils.tensorboard import SummaryWriter


class UsbWandbWriter(WandbLogWriter):
    """A WandbLogWriter (rsl_rl then skips its wall-clock-time scalars, which would break W&B's step order)."""

    def __init__(self, log_dir: str, project_name: str, run_name: str = "", group: str | None = None,
                 tags: list[str] | None = None, model_every: int = 500, config: dict | None = None,
                 run_id: str | None = None) -> None:
        SummaryWriter.__init__(self, log_dir, flush_secs=10)  # not WandbLogWriter's: it starts its own run
        base = os.path.basename(os.path.normpath(log_dir))
        wandb_dir = os.path.dirname(os.path.abspath(log_dir))
        wandb.init(project=project_name, entity=os.environ.get("WANDB_ENTITY") or None,
                   name=f"{base}_{run_name}" if run_name else base, group=group, tags=tags or None, dir=wandb_dir,
                   config={"log_dir": os.path.abspath(log_dir), **(config or {})},
                   id=run_id, resume="allow" if run_id else None, allow_val_change=True)
        self.model_every = model_every
        # A run continued in place (--log_dir, --wandb_id) already uploaded the videos it has.
        self.logged_videos: set[str] = {v.name for v in pathlib.Path(log_dir).rglob("*.mp4")}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        SummaryWriter.add_scalar(self, tag, scalar_value, global_step=global_step, walltime=walltime,
                                 new_style=new_style)
        wandb.log({tag: scalar_value}, step=global_step)

    def store_config(self, env_cfg, train_cfg) -> None:
        env = asdict(env_cfg) if is_dataclass(env_cfg) else getattr(env_cfg, "to_dict", lambda: str(env_cfg))()
        wandb.config.update({"train_cfg": train_cfg, "env_cfg": env}, allow_val_change=True)

    def save_model(self, model_path: str, it: int) -> None:
        if self.model_every > 0 and it % self.model_every == 0:
            self.save_file(model_path)

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path), policy="now")

    def save_video(self, video: pathlib.Path, it: int) -> None:
        video = pathlib.Path(video)
        if video.name in self.logged_videos or "wandb" in video.parts:
            return
        m = re.search(r"iter_(\d+)", video.name)
        caption = f"iteration {int(m.group(1))}: full uncertainty, deterministic" if m else video.stem
        wandb.log({"eval/video": wandb.Video(str(video), format="mp4", caption=caption)}, step=it)
        self.logged_videos.add(video.name)

    def stop(self) -> None:
        wandb.finish()
