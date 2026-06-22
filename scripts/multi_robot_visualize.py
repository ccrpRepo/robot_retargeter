"""Visualize motion trajectories of multiple robots simultaneously with MuJoCo.

File naming convention: <motion_name>_<robot_name>.csv, for example Form_1_stageii_g1.csv.
So you only need to provide one motion name and a list of robot names; the script will
construct the file paths automatically.

Note: MuJoCo free-joint qpos ordering is [pos(xyz), quat(wxyz)].
After loading CSV files, quaternions must be converted from xyzw to wxyz.

Usage example:
    python scripts/multi_robot_visualize.py \
        --motion body_check_001__A548_M_from_g1 \
        --robots h2 \
        --source_fps 120 \
        --render_fps 30
"""

import argparse
import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Robot name -> MJCF path
ROBOT_XML = {
    "g1": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "g1_description", "mjcf", "g1.xml"
    ),
    "h2": os.path.join(SCRIPT_DIR, "..", "asset", "robot", "h2_description", "H2.xml"),
    "r1": os.path.join(SCRIPT_DIR, "..", "asset", "robot", "r1_description", "R1.xml"),
    "t800": os.path.join(SCRIPT_DIR, "..", "asset", "robot", "t800", "xml", "serial_t800.xml"),
    "pm01": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "pm01_edu", "xml", "serial_pm01_edu.xml"
    ),
    "unitree_a2": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "a2_description", "a2.xml"),
    "agibot_x2": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "agibot_x2", "x2_ultra.xml"),
    "limx_oli": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "HU_D04_description", "xml", "HU_D04_01.xml"
    ),
    "pnd_adam": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "pnd_adam_lite", "adam_lite.xml"
    ),
    "booster_t1": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "booster_t1", "T1_serial.xml"
    ),
    "jaka_pi": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "jaka_pi", "Khan_mini_simplified.xml"
    ),
    "hightorque_hi": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "hightorque_hi", "hi_25dof.xml"
    ),
    "unitree_a2w": os.path.join(
        SCRIPT_DIR, "..", "asset", "robot", "a2w_description", "a2_wheel.xml"
    ),
}

MOTION_DIR = os.path.join(SCRIPT_DIR, "..", "output_data", "robot_motion")


def load_motion(csv_path: str) -> np.ndarray:
    """Load a motion CSV and convert the base quaternion from xyzw to wxyz.

    Each returned row can be written directly to MuJoCo data.qpos:
        [pos(xyz), quat(wxyz), joint_0, joint_1, ...]
    """
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise ValueError(
            f"Motion file {csv_path} must have at least 7 columns (pos3 + quat4), got {data.shape[1]}"
        )

    qpos = data.copy()
    # CSV: first 3 columns are position, columns 4-7 are quaternion (xyzw)
    quat_xyzw = data[:, 3:7]
    # MuJoCo free joint requires quaternion in wxyz order
    qpos[:, 3] = quat_xyzw[:, 3]  # w
    qpos[:, 4] = quat_xyzw[:, 0]  # x
    qpos[:, 5] = quat_xyzw[:, 1]  # y
    qpos[:, 6] = quat_xyzw[:, 2]  # z
    return qpos


def build_combined_spec(robots: list[str]) -> mujoco.MjSpec:
    """Merge multiple robot MJCF models into one scene.

    Use a prefix (robot_name + "_") to avoid body/joint/geom name conflicts across models.
    Each robot will have a free joint named "<robot>_floating_base_joint".
    """
    spec = mujoco.MjSpec()
    spec.modelname = "multi_robot"

    # Increase global headlight illumination. A non-reflective ground can make
    # the scene too dark under a single directional light, so raise ambient/diffuse.
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]

    # Skybox texture
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=512,
    )
    # White-ish grid texture: cell interior light gray, grid lines dark gray.
    grid_res = 256          # Pixel resolution per tile
    line_px = 3             # Grid line width (pixels)
    grid_rgba = np.zeros((grid_res, grid_res, 4), dtype=np.uint8)
    bg_color = (100, 100, 100)    # Cell background color
    line_color = (50, 50, 50)  # Grid line color
    # Fill background first, then draw border lines on all four sides
    grid_rgba[:, :, :3] = bg_color
    grid_rgba[:, :, 3] = 255
    grid_rgba[:line_px, :, :3] = line_color
    grid_rgba[-line_px:, :, :3] = line_color
    grid_rgba[:, :line_px, :3] = line_color
    grid_rgba[:, -line_px:, :3] = line_color

    grid_tex = spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=grid_res,
        height=grid_res,
        nchannel=4,
    )
    grid_tex.data = grid_rgba.reshape(-1).tobytes()

    spec.add_material(
        name="groundplane",
        textures=["", "groundplane"],
        texrepeat=[10, 10],
        texuniform=True,
        reflectance=0.0,
    )

    spec.worldbody.add_light(
        pos=[0, 0, 20.0],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
    )

    # Add a ground plane using the grid material defined above
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],  # For planes, the first two size values at 0 mean infinite extent
        material="groundplane",
        pos=[0, 0, 0],
    )

    for robot in robots:
        robot_spec = mujoco.MjSpec.from_file(ROBOT_XML[robot])
        for j in robot_spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                j.name = "floating_base_joint"
                break
        frame = spec.worldbody.add_frame()
        spec.attach(robot_spec, prefix=f"{robot}_", frame=frame)

    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0

    return spec


def get_qpos_start(model: mujoco.MjModel, freejoint_name: str) -> int:
    """Return the starting qpos index of a robot free joint in the merged model."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
    if jid < 0:
        raise ValueError(f"Joint not found in model: {freejoint_name}")
    return model.jnt_qposadr[jid]


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo multi-robot motion visualization")
    parser.add_argument(
        "--motion",
        default="body_check_001__A548_M_from_g1",
        help="Motion name (without robot suffix or extension), filename format: <motion_name>_<robot_name>.csv",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=[
                # "unitree_a2w",
                # "agibot_x2",
                #  "g1", 
                 "h2", 
                #  "r1", 
                #  "t800",
                #  "pm01", 
                #  "unitree_a2",
                #  "limx_oli",
                #  "pnd_adam",
                #  "booster_t1",
                #  "jaka_pi",
                #  "hightorque_hi",
                 
                 ],
        choices=list(ROBOT_XML.keys()),
        help="List of robot names to visualize, choices: " + ", ".join(ROBOT_XML.keys()),
    )
    parser.add_argument(
        "--motion_dir",
        default=MOTION_DIR,
        help="Directory containing motion CSV files",
    )
    parser.add_argument("--source_fps", type=float, default=30.0, help="Source data frame rate (original capture FPS)")
    parser.add_argument(
        "--render_fps",
        type=float,
        default=60.0,
        help="Actual rendering FPS, e.g. 60/30; if lower than source FPS, frames are skipped to reduce load while preserving playback speed",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    robots = args.robots

    # Build file paths and load each robot's motion data
    robot_qpos: dict[str, np.ndarray] = {}
    for robot in robots:
        csv_path = os.path.join(args.motion_dir, f"{args.motion}_{robot}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Motion file not found: {csv_path}")
        robot_qpos[robot] = load_motion(csv_path)

    n_frames = min(q.shape[0] for q in robot_qpos.values())
    print(
        "  ".join(f"{r} frames={robot_qpos[r].shape[0]} (qpos {robot_qpos[r].shape[1]})" for r in robots)
        + f"  ->  playing {n_frames} frames"
    )

    # Build the merged model
    spec = build_combined_spec(robots)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Locate each robot's starting qpos index (via free joint)
    robot_start: dict[str, int] = {}
    robot_dim: dict[str, int] = {}
    for robot in robots:
        start = get_qpos_start(model, f"{robot}_floating_base_joint")
        dim = robot_qpos[robot].shape[1]
        if start + dim > model.nq:
            raise ValueError(
                f"qpos dimension mismatch: {robot}[{start}:{start + dim}] exceeds model.nq={model.nq}"
            )
        robot_start[robot] = start
        robot_dim[robot] = dim

    # Precompute XY placement offsets on the ground in an outward grid from the origin.
    # Use a near-square grid layout so added robots spread out naturally.
    spacing = 2.0  # Robot spacing (meters)
    n = len(robots)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    robot_offset: dict[str, tuple[float, float]] = {}
    for idx, robot in enumerate(robots):
        col = idx % cols
        row = idx // cols
        # Center the grid so its center lies at the origin
        dx = (col - (cols - 1) / 2.0) * spacing
        dy = (row - (rows - 1) / 2.0) * spacing
        robot_offset[robot] = (dx, dy)

    def apply_frame(i: int) -> None:
        for idx, robot in enumerate(robots):
            start = robot_start[robot]
            dim = robot_dim[robot]
            data.qpos[start : start + dim] = robot_qpos[robot][i]
            # Spread robots on the XY plane by grid offsets
            # Free-joint qpos order is [x, y, z, qw, qx, qy, qz], so x=start and y=start+1
            dx, dy = robot_offset[robot]
            data.qpos[start] += dx
            data.qpos[start + 1] += dy
        # Visualization only: forward kinematics is enough for global body poses.
        # Skip heavier mj_forward stages like collision, constraints, and dynamics.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    # Frame skipping: source data at args.source_fps, render every step frames for render_fps.
    # Use dt = step / source_fps to preserve real-time playback speed.
    step = max(1, round(args.source_fps / args.render_fps))
    dt = step / args.source_fps
    print(f"source_fps={args.source_fps:g}, render_fps≈{args.source_fps / step:g}, render_every={step} frame(s)")
    apply_frame(0)

    # Playback control state, updated by keyboard callback
    state = {"frame": 0, "paused": False, "replay": False}

    def key_callback(keycode: int) -> None:
        # Space: pause/resume; R: replay from start
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
            print("Paused" if state["paused"] else "Resumed")
        elif keycode in (ord("r"), ord("R")):
            state["replay"] = True
            print("Replay")

    print("Controls: [Space] Pause/Resume  [R] Replay")

    # Progress bar in rendered frames (total is after step-based frame skipping)
    total_render_frames = math.ceil(n_frames / step)
    pbar = tqdm(total=total_render_frames, desc="Playback", unit="frame", dynamic_ncols=True)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            t0 = time.time()

            # Handle replay request
            if state["replay"]:
                state["frame"] = 0
                state["replay"] = False
                state["paused"] = False
                pbar.reset(total=total_render_frames)

            apply_frame(state["frame"])
            viewer.sync()

            # Sync progress bar to current frame (counted in rendered frames)
            pbar.n = min(state["frame"] // step, total_render_frames)
            pbar.refresh()

            if not state["paused"]:
                state["frame"] += step
                if state["frame"] >= n_frames:
                    if args.loop:
                        state["frame"] = 0
                        pbar.reset(total=total_render_frames)
                    else:
                        # Stay at the last frame; wait for R replay or window close
                        state["frame"] = n_frames - 1
                        state["paused"] = True
                        pbar.n = total_render_frames
                        pbar.refresh()

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                print(
                    f"Warning: Processing one frame took {elapsed:.3f}s, exceeding the target frame interval {dt:.3f}s"
                )

    pbar.close()


if __name__ == "__main__":
    main()
