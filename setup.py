#!/usr/bin/env python
"""Setup script for the robot_retargeter project.

Installs all third-party Python dependencies required to run the retargeting
pipeline (smpl_replay / robot_replay / robot_retarget / multi_robot_visualize).

Version lower bounds are pinned to the versions verified in the development
``mjlab`` conda environment (Python 3.11). They are expressed as ``>=`` so that
newer compatible releases are still allowed.

Usage::

    pip install -e .              # editable install (recommended for dev)
    pip install .                 # regular install
    pip install -r requirements.txt
"""

from setuptools import find_packages, setup

# Core runtime dependencies. Version floors come from the `mjlab` env:
#   mujoco 3.9.0, mink 1.1.1, numpy 2.4.6, torch 2.11.0, PyYAML 6.0.3,
#   scipy 1.17.1, tqdm 4.67.3, glfw 2.10.0, smplx 0.1.28, trimesh 4.11.5
INSTALL_REQUIRES = [
    "mujoco>=3.9.0",
    "mink>=1.1.1",
    "numpy>=2.0,<3.0",
    "torch>=2.0",
    "PyYAML>=6.0",
    "scipy>=1.13",
    "tqdm>=4.66",
    "glfw>=2.7",
    "smplx>=0.1.28",
    "trimesh>=4.0",
]

setup(
    name="robot_retargeter",
    version="0.1.0",
    description="Motion retargeting pipeline from SMPL-X / source robots to target humanoid robots.",
    long_description=(
        "Tools to replay SMPL-X or source-robot motion, retarget keypoints onto "
        "target robot models via inverse kinematics (mink/MuJoCo), and visualize "
        "multiple robots side by side."
    ),
    python_requires=">=3.10",
    packages=find_packages(exclude=("asset", "config", "dataset", "output_data", "bash")),
    py_modules=[],
    install_requires=INSTALL_REQUIRES,
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
)
