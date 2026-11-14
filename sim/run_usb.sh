#!/usr/bin/env bash
# The USB task's environment (.venv-usb): newton 1.6.1 + rsl-rl 5.5.1 over a CUDA 12 torch, for machines whose
# driver predates CUDA 13 (the main uv env's torch is cu130: driver >= 580). Made once with
#   uv venv --python <a python 3.12 with torch 2.6+cu12x> --system-site-packages .venv-usb
#   uv pip install --python .venv-usb/bin/python <warp-lang 1.17.0+cu12 wheel from NVIDIA's GitHub release> \
#       "newton[sim]==1.6.1" "mujoco-warp==3.12.0" "mujoco~=3.12.0" "rsl-rl-lib==5.5.1" "tensordict<0.9" \
#       usd-core gmsh trimesh networkx rtree scipy tensorboard fast-simplification matplotlib pytest
#   (then uninstall the cu13 torch / triton / nvidia-* that uv pulls in, so the base torch is used)
#   uv pip install --python .venv-usb/bin/python --no-deps -e .
# Like run.sh, it keeps ~/.bashrc's ROS paths out.
#
#   ./run_usb.sh python scripts/newton_usb_check.py
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH \
    ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION VIRTUAL_ENV
# Weights & Biases as the user (newton_usb_train.py --wandb_project), not the machine's default login in ~/.netrc.
# WANDB_KEY_FILE overrides the file; an explicit WANDB_API_KEY wins over both.
key_file="${WANDB_KEY_FILE:-$HOME/.config/wandb/api_key_abhay}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$key_file" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$key_file")"
    export WANDB_API_KEY
fi
exec .venv-usb/bin/"$@"
