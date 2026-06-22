#!/usr/bin/env bash
set -euo pipefail

# Step 1) Run smpl_replay without viewer        (per robot)
# Step 2) Run robot_retarget                     (per robot)
# Step 3) Run multi_robot_visualize              (once, for all robots)
#
# VIS_ROBOTS may contain one or more robot names separated by spaces. Steps 1
# and 2 are repeated for each robot (using config/robot/<robot>.yaml), then
# step 3 visualizes them all together.
#   e.g. VIS_ROBOTS="hightorque_hi h2 t800" ./scripts/retarget_from_smplx.sh

SMPL_MOTION_FILE="${SMPL_MOTION_FILE:-dataset/ACCAD/Extended_1_stageii.npz}"
VIS_ROBOTS="${VIS_ROBOTS:-g1 h2 t800 r1}"
SOURCE_FPS="${SOURCE_FPS:-120}"
RENDER_FPS="${RENDER_FPS:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Resolve the Python interpreter. Override with PYTHON_BIN=/path/to/python.
# Otherwise prefer the active environment's python / python3.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # use user-provided PYTHON_BIN as-is
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[error] no python interpreter found; set PYTHON_BIN explicitly" >&2
  exit 127
fi
SKELETON_CONFIG="${SKELETON_CONFIG:-config/skeleton/skeleton.yaml}"
KEYPOINTS_NAME="${KEYPOINTS_NAME:-$(basename "${SMPL_MOTION_FILE}" .npz)}"

ROBOT_RENDER_DEBUG="${ROBOT_RENDER_DEBUG:-false}"

if ! "${PYTHON_BIN}" --version >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN is not a working python interpreter: ${PYTHON_BIN}" >&2
  exit 127
fi

# Parse the (possibly space-separated) robot list. Each robot gets its own
# smpl_replay (step 1) + robot_retarget (step 2) pass, then a single
# multi_robot_visualize (step 3) renders all of them together.
read -r -a VIS_ROBOT_ARR <<< "${VIS_ROBOTS}"
if [[ "${#VIS_ROBOT_ARR[@]}" -eq 0 ]]; then
  echo "[error] VIS_ROBOTS is empty" >&2
  exit 1
fi

NUM_ROBOTS="${#VIS_ROBOT_ARR[@]}"

for idx in "${!VIS_ROBOT_ARR[@]}"; do
  ROBOT="${VIS_ROBOT_ARR[$idx]}"
  STEP_NO=$((idx + 1))
  ROBOT_CONFIG="config/robot/${ROBOT}.yaml"

  if [[ ! -f "${ROBOT_CONFIG}" ]]; then
    echo "[error] robot config not found: ${ROBOT_CONFIG}" >&2
    exit 1
  fi

  echo "=== [robot ${STEP_NO}/${NUM_ROBOTS}] ${ROBOT} (config: ${ROBOT_CONFIG}) ==="

  echo "[1/3] smpl_replay (no viewer)"
  "${PYTHON_BIN}" scripts/smpl_replay.py \
    --no-viewer \
    --motion_file "${SMPL_MOTION_FILE}" \
    --robot-config "${ROBOT_CONFIG}" \
    --skeleton-config "${SKELETON_CONFIG}" \
    --fps "${SOURCE_FPS}"

  echo "[2/3] robot_retarget"
  RETARGET_ARGS=(--config "${ROBOT_CONFIG}")
  if [[ -n "${KEYPOINTS_NAME}" ]]; then
    RETARGET_ARGS+=(--keypoints-name "${KEYPOINTS_NAME}")
  fi
  if [[ "${ROBOT_RENDER_DEBUG}" == "true" ]]; then
    RETARGET_ARGS+=(--render-debug)
  elif [[ "${ROBOT_RENDER_DEBUG}" == "false" ]]; then
    RETARGET_ARGS+=(--no-render-debug)
  fi
  "${PYTHON_BIN}" scripts/robot_retarget.py "${RETARGET_ARGS[@]}"
done

# Resolve the motion name. It is shared across robots since it derives from the
# SMPL motion (KEYPOINTS_NAME), or from the keypoints_path of the first robot.
if [[ -n "${KEYPOINTS_NAME}" ]]; then
  MOTION_NAME="${KEYPOINTS_NAME}"
else
  FIRST_ROBOT="${VIS_ROBOT_ARR[0]}"
  FIRST_CONFIG="config/robot/${FIRST_ROBOT}.yaml"
  KEYPOINTS_LINE="$(grep -E '^keypoints_path:' "${FIRST_CONFIG}" | head -n 1 || true)"
  if [[ -z "${KEYPOINTS_LINE}" ]]; then
    echo "[error] keypoints_path not found in ${FIRST_CONFIG}" >&2
    exit 1
  fi
  KEYPOINTS_PATH="${KEYPOINTS_LINE#keypoints_path:}"
  KEYPOINTS_PATH="$(echo "${KEYPOINTS_PATH}" | xargs)"
  KEYPOINTS_BASENAME="$(basename "${KEYPOINTS_PATH}")"
  KEYPOINT_STEM="${KEYPOINTS_BASENAME%.pkl}"
  KEYPOINT_STEM="${KEYPOINT_STEM%_keypoints}"
  MOTION_NAME="${KEYPOINT_STEM}"
fi

echo "[3/3] multi_robot_visualize"
"${PYTHON_BIN}" scripts/multi_robot_visualize.py \
  --motion "${MOTION_NAME}" \
  --robots "${VIS_ROBOT_ARR[@]}" \
  --source_fps "${SOURCE_FPS}" \
  --render_fps "${RENDER_FPS}"

echo "[done] pipeline finished: motion=${MOTION_NAME}, robots=${VIS_ROBOTS}"
