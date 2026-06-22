# robot_retargeter

English | [中文](README_zh.md)

A toolkit for retargeting human motion (SMPL-X) or source-robot motion onto
target humanoid robots, with support for side-by-side multi-robot visualization.

## Overview

The project provides an end-to-end motion retargeting pipeline with three stages:

1. **Replay / keypoint extraction**: Extract skeletal keypoints from SMPL-X
   motion (`.npz`) or source-robot motion (`.csv`).
2. **Retargeting**: Map the keypoints onto a target robot model via inverse
   kinematics ([mink](https://github.com/kevinzakka/mink) + MuJoCo) to produce
   robot motion.
3. **Visualization**: Play back the retargeted result for one or more target
   robots side by side.

Directory layout:

| Directory | Contents |
|---|---|
| `asset/` | Robot models (URDF/MJCF/mesh), SMPL-X models, skeleton |
| `config/` | Retargeting configs for robots and skeleton (YAML) |
| `dataset/` | Input motion data (SMPL-X `.npz` / LAFAN1 `.csv`, etc.) |
| `output_data/` | Output keypoints and robot motion |
| `scripts/` | Python pipeline scripts |
| `bash/` | One-command pipeline scripts |

## Installation

Requires Python >= 3.10 (developed on Python 3.11). A virtual environment
(conda / venv) is recommended.

```bash
# 1) Create and activate an environment (conda as an example)
conda create -n robot_retargeter python=3.11 -y
conda activate robot_retargeter

# 2) Install dependencies
pip install -e .
# or
pip install -r requirements.txt
```

Key dependencies: `mujoco`, `mink`, `numpy`, `torch`, `scipy`, `PyYAML`,
`tqdm`, `glfw`, `smplx`, `trimesh` (exact version floors are listed in
`setup.py` / `requirements.txt`).

### SMPL-X models (downloaded separately)

The SMPL-X model files are **not** included in this repository (they are
subject to their own license). To use the SMPL-X pipeline
(`retarget_from_smplx.sh`), download them yourself:

1. Register and download from the official site: https://smpl-x.is.tue.mpg.de/
   (the "SMPL-X" models, `.npz` format).
2. Place the files under `asset/smplx/` so the layout looks like:

   ```
   asset/smplx/
     SMPLX_NEUTRAL.npz
     SMPLX_MALE.npz
     SMPLX_FEMALE.npz
   ```

> By downloading the models you agree to the SMPL-X license terms. This
> directory is git-ignored and will not be committed.

> Note: Large robot mesh/texture files (`*.stl`, `*.obj`, `*.dae`, `*.png`,
> `*.mtl`) are tracked with [Git LFS](https://git-lfs.com/). Make sure Git LFS
> is installed (`git lfs install`) before cloning so the meshes are fetched.

## Usage

The `bash/` directory provides two one-command pipeline scripts that
automatically run all three stages (replay -> retarget -> visualize). By default
the scripts use the `python` from the currently active environment; you can
override it via `PYTHON_BIN`.

### 1) Retarget from SMPL-X motion

```bash
./bash/retarget_from_smplx.sh
```

Customize via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SMPL_MOTION_FILE` | `dataset/ACCAD/Extended_1_stageii.npz` | Input SMPL-X motion file |
| `VIS_ROBOTS` | `g1 h2 t800 r1` | Target robot list (space-separated, multiple allowed) |
| `KEYPOINTS_NAME` | derived from motion file name | Keypoints / output motion name |
| `SOURCE_FPS` | `120` | Source motion frame rate |
| `RENDER_FPS` | `30` | Visualization render frame rate |
| `PYTHON_BIN` | auto-detected | Python interpreter to use |

Example (custom robots and motion file):

```bash
VIS_ROBOTS="hightorque_hi h2 t800" \
SMPL_MOTION_FILE="dataset/ACCAD/Form_1_stageii.npz" \
./bash/retarget_from_smplx.sh
```

### 2) Retarget from source-robot motion

```bash
./bash/retarget_from_robot.sh
```

| Variable | Default | Description |
|---|---|---|
| `ROBOT_MOTION_FILE` | `dataset/lafan1_g1/dance1_subject2.csv` | Source robot motion file |
| `ORIGIN_ROBOT` | `g1` | Source robot name (provides the skeleton config) |
| `VIS_ROBOTS` | `h2 r1` | Target robot list (space-separated, multiple allowed) |
| `SOURCE_FPS` | `30` | Source motion frame rate |
| `RENDER_FPS` | `30` | Visualization render frame rate |
| `PYTHON_BIN` | auto-detected | Python interpreter to use |

Example:

```bash
VIS_ROBOTS="h2 t800 r1" \
ROBOT_MOTION_FILE="dataset/lafan1_g1/dance1_subject1.csv" \
ORIGIN_ROBOT="g1" \
./bash/retarget_from_robot.sh
```

After a run finishes, the retargeted robot motion is saved under
`output_data/robot_motion/`.
