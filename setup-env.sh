#!/usr/bin/env bash
# Build the conda env for cable_assembly (Linux). Run from the repo root:
#
#   ./setup-env.sh              # env named "cable"
#   ./setup-env.sh myenv        # any other name
#   FRI_CLIENT_VERSION=1.16 ./setup-env.sh    # if the controller runs a different FRI version
#
# Steps, and why only the first lives in environment.yml:
#   1. conda env create -f environment.yml   -- Python 3.12, numpy<2, pybullet, cmake<4
#   2. pip install every requirements/ layer -- torch cu128 and SAM3 included. Run HERE, not as
#                                               an environment.yml pip: section, because conda
#                                               runs that with -U (see environment.yml)
#   3. activation hooks that park PYTHONPATH -- a sourced ROS workspace puts python3.10
#                                               site-packages and ROS pytest plugins on it, which
#                                               shadow this env's 3.12 packages
#   4. pyFRI, built against FRI_CLIENT_VERSION -- see requirements/iiwa.txt for its build flags
#   5. cuMotion (optional; CUMOTION=0 skips)  -- NVIDIA's GPU motion planner. Not on PyPI: the wheel
#                                               comes from the GitHub release tarball, which is
#                                               unpacked OUTSIDE the repo (CUMOTION_DIR, default
#                                               ../third_party) because its license (NVIDIA Isaac
#                                               ROS Software License) does not allow vendoring it
#                                               into this Apache-2.0 tree. The tarball's content/
#                                               holds the robot descriptions (ur10e.xrdf, ...).
#
# The SAM3 weights are gated and NOT downloaded here: request access at
# https://huggingface.co/facebook/sam3, then `hf auth login`.
set -euo pipefail

env_name="${1:-cable}"
fri_version="${FRI_CLIENT_VERSION:-1.15}"
pyfri_ref="302317a2e942699678bfa59315511d84a97a58c2"   # lbr-stack/pyfri main, 2024-09-08
cumotion="${CUMOTION:-1}"
cumotion_dir="${CUMOTION_DIR:-$(cd .. && pwd)/third_party}"
cumotion_pkg="cumotion-1.1.0-cuda12.6-x86_64"           # CUDA 12.6 build = the Ubuntu 22.04 one
cumotion_sha256="0dd7e9ecb4e0bd243e7441a0d9845322eddcb281b1d199f90766dd1b9a5c6ef0"

[ -f environment.yml ] || { echo "Run from the repo root." >&2; exit 1; }
command -v conda >/dev/null || { echo "conda not found on PATH." >&2; exit 1; }
# The same leak the hooks guard against would otherwise reach the pip installs below.
unset PYTHONPATH

echo "==> 1/5 conda env '$env_name' from environment.yml"
conda env create -n "$env_name" -f environment.yml
prefix="$(conda run -n "$env_name" python -c 'import sys; print(sys.prefix)')"

echo "==> 2/5 requirements/ (no -U: the conda numpy and pybullet must stay)"
conda run -n "$env_name" --no-capture-output \
    python -m pip install -r requirements/all.txt -r requirements/dev.txt

echo "==> 3/5 activation hooks (park PYTHONPATH while '$env_name' is active)"
mkdir -p "$prefix/etc/conda/activate.d" "$prefix/etc/conda/deactivate.d"
cat > "$prefix/etc/conda/activate.d/clear_ros_pythonpath.sh" <<'HOOK'
# A sourced ROS workspace puts python3.10 site-packages on PYTHONPATH; they shadow this env's 3.12
# packages and autoload ROS pytest plugins. Park it while the env is active.
export _CABLE_SAVED_PYTHONPATH="${PYTHONPATH-}"
unset PYTHONPATH
HOOK
cat > "$prefix/etc/conda/deactivate.d/clear_ros_pythonpath.sh" <<'HOOK'
if [ -n "${_CABLE_SAVED_PYTHONPATH-}" ]; then export PYTHONPATH="$_CABLE_SAVED_PYTHONPATH"; fi
unset _CABLE_SAVED_PYTHONPATH
HOOK

echo "==> 4/5 pyFRI against FRI client $fri_version"
conda run -n "$env_name" --no-capture-output env FRI_CLIENT_VERSION="$fri_version" \
    python -m pip install --no-build-isolation --no-deps \
    "pyfri @ git+https://github.com/lbr-stack/pyfri@${pyfri_ref}"

if [ "$cumotion" = 1 ]; then
    echo "==> 5/5 cuMotion 1.1.0 into $cumotion_dir/$cumotion_pkg"
    if [ ! -d "$cumotion_dir/$cumotion_pkg" ]; then
        mkdir -p "$cumotion_dir"
        tarball="$cumotion_dir/$cumotion_pkg.tar.gz"
        curl -fSL -o "$tarball" \
            "https://github.com/nvidia-isaac/cumotion/releases/download/v1.1.0/$cumotion_pkg.tar.gz"
        echo "$cumotion_sha256  $tarball" | sha256sum -c -
        tar -xzf "$tarball" -C "$cumotion_dir"
        rm "$tarball"
    fi
    # --no-deps: its only requirement is numpy>=1.26, and pip must not touch the conda numpy.
    conda run -n "$env_name" --no-capture-output python -m pip install --no-deps \
        "$cumotion_dir/$cumotion_pkg/python_wheels/cumotion-1.1.0-cp312-cp312-linux_x86_64.whl"
    conda run -n "$env_name" --no-capture-output python -c \
        "import cumotion; print('cumotion', cumotion.__version__)"
else
    echo "==> 5/5 cuMotion skipped (CUMOTION=0)"
fi

echo "==> editable install of cable_assembly"
conda run -n "$env_name" --no-capture-output python -m pip install --no-deps -e .

echo "==> verify"
conda run -n "$env_name" --no-capture-output python -W ignore -c "
import numpy, torch, sam3, pyfri, cv2, rtde_control, pyrealsense2, py_trees, ruckig, pybullet
import cable_assembly
assert numpy.__version__.startswith('1.'), 'numpy %s: SAM3 needs numpy < 2' % numpy.__version__
# Upstream SAM3 hard-codes CUDA, so a torch that cannot see the GPU is a broken env, not a slow one.
assert torch.cuda.is_available(), ('torch %s (CUDA %s) cannot use the GPU -- check that the '
    'NVIDIA driver supports that CUDA version (nvidia-smi)' % (torch.__version__, torch.version.cuda))
print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| FRI client', pyfri.FRI_CLIENT_VERSION)
"
echo
echo "Done. Next:  conda activate $env_name"
echo "             python -m pytest          # compare with tests/known_failures.txt"
echo "             hf auth login             # once, for the gated SAM3 weights"
