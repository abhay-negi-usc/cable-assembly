#!/usr/bin/env bash
# `uv run` in this project's environment, minus what ~/.bashrc puts in every shell: ROS Humble,
# MoveIt and Webots paths. Their python3.10 / 3.8 PYTHONPATH and their libraries on LD_LIBRARY_PATH
# (CUDA 11.7 first) get loaded into a Python 3.12 / CUDA 13 Isaac process otherwise.
#
#   ./run.sh python scripts/check_iiwa14.py --physics isaacsim_physx
#   ./run.sh --extra isaacsim isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx
set -euo pipefail
cd "$(dirname "$0")"
# VIRTUAL_ENV too: the editor exports the `cable` conda env as one, and uv warns on every run.
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH \
    ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION VIRTUAL_ENV
exec uv run "$@"
