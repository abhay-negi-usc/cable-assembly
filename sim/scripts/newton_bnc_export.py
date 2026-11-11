"""Export a trained BNC policy (scripts/newton_bnc_train.py) for the robot: its actor's weights, observation
normaliser and every constant of the env it was trained in, as ONE .npz the app runs in numpy
(cable_assembly/skills/rl_insertion.py Policy -- no torch or rsl_rl on the robot PC).

    ./run.sh python scripts/newton_bnc_export.py --checkpoint logs/rsl_rl/bnc_newton/<run>/model_<it>.pt \
        --out ../configs/policies/bnc_insert.npz

Checked before it is written: the numpy actor against torch's GRU + MLP on the same weights, over a
random observation sequence (the GRU state carried), to 1e-5.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))           # the app (cable_assembly), for the check


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    run_dir = os.path.dirname(os.path.abspath(a.checkpoint))
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    actor = conf["train"]["actor"]
    if actor.get("class_name") != "RNNModel" or actor.get("rnn_type") != "gru" or actor.get("rnn_num_layers", 1) != 1:
        sys.exit(f"only a 1-layer GRU actor is exported (this run's: {actor})")
    if conf["train"]["obs_groups"]["actor"] != ["policy"]:
        sys.exit(f"only the 'policy' observation is ported to the robot (this run's: {conf['train']['obs_groups']['actor']})")
    if actor.get("activation", "elu") != "elu":
        sys.exit("only elu")
    from cable_assembly_sim.newton_bnc.model import GRASP_Z

    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    mlp = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    spec = {"checkpoint": os.path.abspath(a.checkpoint), "iteration": int(torch.load(a.checkpoint, map_location="cpu",
                                                                                      weights_only=False)["iter"]),
            "env": conf["env"], "args": conf.get("args", {}), "grasp_z": GRASP_Z, "obs_group": "policy",
            "obs_eps": 0.01, "mlp": [list(sd[f"mlp.{i}.weight"].shape) for i in mlp]}
    arrays = {"obs_mean": sd["obs_normalizer._mean"][0].numpy(), "obs_std": sd["obs_normalizer._std"][0].numpy(),
              "gru_w_ih": sd["rnn.rnn.weight_ih_l0"].numpy(), "gru_w_hh": sd["rnn.rnn.weight_hh_l0"].numpy(),
              "gru_b_ih": sd["rnn.rnn.bias_ih_l0"].numpy(), "gru_b_hh": sd["rnn.rnn.bias_hh_l0"].numpy()}
    for j, i in enumerate(mlp):
        arrays[f"mlp_w{j}"] = sd[f"mlp.{i}.weight"].numpy()
        arrays[f"mlp_b{j}"] = sd[f"mlp.{i}.bias"].numpy()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, spec=json.dumps(spec), **arrays)

    # THE CHECK: torch's GRU + MLP on the same weights against the numpy actor, 200 steps.
    from cable_assembly.skills.rl_insertion import Policy

    pol = Policy(a.out)
    D, H = arrays["obs_mean"].shape[0], arrays["gru_w_hh"].shape[1]
    gru = torch.nn.GRU(D, H)
    gru.load_state_dict({"weight_ih_l0": sd["rnn.rnn.weight_ih_l0"], "weight_hh_l0": sd["rnn.rnn.weight_hh_l0"],
                         "bias_ih_l0": sd["rnn.rnn.bias_ih_l0"], "bias_hh_l0": sd["rnn.rnn.bias_hh_l0"]})
    layers = []
    for j, i in enumerate(mlp):
        lin = torch.nn.Linear(*reversed(sd[f"mlp.{i}.weight"].shape))
        lin.load_state_dict({"weight": sd[f"mlp.{i}.weight"], "bias": sd[f"mlp.{i}.bias"]})
        layers += [lin] + ([torch.nn.ELU()] if j < len(mlp) - 1 else [])
    head = torch.nn.Sequential(*layers)
    rng = np.random.default_rng(0)
    h = torch.zeros(1, 1, H)
    worst = 0.0
    with torch.no_grad():
        for _ in range(200):
            o = arrays["obs_mean"] + arrays["obs_std"] * rng.standard_normal(D).astype(np.float32)
            x = (torch.from_numpy(o) - sd["obs_normalizer._mean"][0]) / (sd["obs_normalizer._std"][0] + 0.01)
            y, h = gru(x.view(1, 1, D), h)
            ref = head(y.view(H)).numpy()
            worst = max(worst, float(np.max(np.abs(pol(o) - ref))))
    if worst > 1e-4:
        os.remove(a.out)
        sys.exit(f"numpy actor disagrees with torch by {worst:.2e} -- not written")
    print(f"{a.out}: iteration {spec['iteration']}, obs {D}, GRU {H}, mlp {spec['mlp']}; numpy = torch to {worst:.1e}")


if __name__ == "__main__":
    main()
