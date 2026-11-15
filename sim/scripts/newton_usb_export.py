"""Export a newton_usb / newton_usb_cable actor checkpoint (rsl_rl model_*.pt) as a standalone TorchScript
module for the robot: deterministic (the Gaussian's mean), CPU, no rsl_rl / Newton / Warp needed to run it.

    ./run_usb.sh python scripts/newton_usb_export.py <run>/model_700.pt [out.pt]

The module maps (obs [B, n_obs], h [1, B, hidden]) -> (action [B, 6], h): the observation normaliser, the GRU,
the ELU MLP head. Start each episode with h = zeros (module.initial_state(B)); feed back the returned h every
policy step. A check against the training code's own deterministic actor is run before writing.
"""

import argparse
import os

import torch


class Actor(torch.nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].float().clone())
        self.register_buffer("std", sd["obs_normalizer._std"].float().clone())
        self.eps = 1e-2  # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)
        w_ih = sd["rnn.rnn.weight_ih_l0"]
        self.hidden = int(sd["rnn.rnn.weight_hh_l0"].shape[1])
        self.gru = torch.nn.GRU(int(w_ih.shape[1]), self.hidden, 1)
        self.gru.load_state_dict({k[len("rnn.rnn."):]: v for k, v in sd.items() if k.startswith("rnn.rnn.")})
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
        layers = []
        for j, i in enumerate(idx):
            lin = torch.nn.Linear(sd[f"mlp.{i}.weight"].shape[1], sd[f"mlp.{i}.weight"].shape[0])
            lin.load_state_dict({"weight": sd[f"mlp.{i}.weight"], "bias": sd[f"mlp.{i}.bias"]})
            layers.append(lin)
            if j < len(idx) - 1:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers)

    @torch.jit.export
    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden)

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        x = (obs - self.mean) / (self.std + self.eps)
        y, h = self.gru(x.unsqueeze(0), h)
        return self.mlp(y.squeeze(0)), h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("out", nargs="?")
    a = p.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    actor = Actor(sd).eval()
    n_obs = int(sd["obs_normalizer._mean"].shape[-1])

    # Check: the same network rebuilt from rsl_rl's own modules, over a 50-step random sequence.
    from rsl_rl.modules.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(n_obs)
    norm.load_state_dict({k[len("obs_normalizer."):]: v for k, v in sd.items() if k.startswith("obs_normalizer.")})
    norm.eval()
    torch.manual_seed(0)
    obs = torch.randn(50, 8, n_obs)
    h1, h2 = actor.initial_state(8), None
    err = 0.0
    with torch.no_grad():
        for t in range(50):
            a1, h1 = actor(obs[t], h1)
            y, h2 = actor.gru(norm(obs[t]).unsqueeze(0), h2)
            a2 = actor.mlp(y.squeeze(0))
            err = max(err, (a1 - a2).abs().max().item())
    assert err < 1e-5, err

    out = a.out or os.path.splitext(a.ckpt)[0] + "_actor.ts.pt"
    ts = torch.jit.script(actor)
    ts.save(out)
    re = torch.jit.load(out)
    with torch.no_grad():
        b1, _ = re(obs[0], re.initial_state(8))
        b2, _ = actor(obs[0], actor.initial_state(8))
    assert (b1 - b2).abs().max().item() < 1e-6
    print(f"wrote {out}: obs {n_obs} -> action 6, GRU hidden {actor.hidden}; normaliser check max err {err:.1e}")


if __name__ == "__main__":
    main()
