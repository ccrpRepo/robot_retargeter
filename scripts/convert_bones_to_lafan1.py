#!/usr/bin/env python3
"""Convert G1 CSV motions to LAFAN1-like CSV format.

Input format (G1 CSV):
- Header row exists.
- Root pose uses Euler angles: root_rotateX/Y/Z (degrees).
- Joint columns end with "_dof" and are in degrees.

Output format (LAFAN1-like):
- No header.
- One numeric row per frame:
  root_tx, root_ty, root_tz, root_qx, root_qy, root_qz, root_qw, joint_0, joint_1, ...
- Root quaternion uses XYZW order.
- Joint values are radians.

Terminal examples:
    # Run from project root and convert all CSV files under dataset/bones_g1
    python scripts/convert_bones_to_lafan1.py

    # Convert one file only
    python scripts/convert_bones_to_lafan1.py \
        --input-csv dataset/bones_g1/body_check_001__A548_M.csv

    # Customize input/output directories and root scale
    python scripts/convert_bones_to_lafan1.py \
        --input-root dataset/bones_g1 \
        --output-root output_data/robot_motion/bones_g1
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dataset" / "bones_g1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output_data" / "robot_motion" / "bones_g1"


def euler_xyz_deg_to_quat_xyzw(rx_deg: float, ry_deg: float, rz_deg: float) -> tuple[float, float, float, float]:
    """Convert intrinsic XYZ Euler angles (deg) to quaternion in XYZW order."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)

    # wxyz for intrinsic XYZ
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return x, y, z, w


def get_motion_files(input_root: Path, include_m: bool) -> list[Path]:
    files = sorted(input_root.glob("**/*.csv"))
    if include_m:
        return files
    return [p for p in files if not p.name.endswith("_M.csv")]


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def convert_one_file(src: Path, dst: Path, root_scale: float) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {src}")

        joint_cols = [c for c in reader.fieldnames if c.endswith("_dof")]

        frames = 0
        with dst.open("w", newline="") as f_out:
            writer = csv.writer(f_out)
            for row in reader:
                tx = safe_float(row, "root_translateX") * root_scale
                ty = safe_float(row, "root_translateY") * root_scale
                tz = safe_float(row, "root_translateZ") * root_scale

                qx, qy, qz, qw = euler_xyz_deg_to_quat_xyzw(
                    safe_float(row, "root_rotateX"),
                    safe_float(row, "root_rotateY"),
                    safe_float(row, "root_rotateZ"),
                )

                out_row: list[float] = [tx, ty, tz, qx, qy, qz, qw]
                for jc in joint_cols:
                    out_row.append(math.radians(safe_float(row, jc)))

                writer.writerow([f"{v:.6f}" for v in out_row])
                frames += 1

    return frames, len(joint_cols)


def make_output_path(src: Path, input_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(input_root)
    stem = src.stem
    return output_root / rel.parent / f"{stem}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert G1 CSV to LAFAN1-like quaternion CSV")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Source root directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--input-csv", type=Path, default=None, help="Convert only one CSV file")
    # parser.add_argument("--include-m", action="store_true", help="Include *_M.csv files")
    parser.add_argument("--root-scale", type=float, default=0.01, help="Scale for root translation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_csv is not None:
        src_files = [args.input_csv]
    else:
        src_files = get_motion_files(args.input_root, include_m=True)

    if not src_files:
        raise FileNotFoundError("No source CSV files found.")

    total_frames = 0
    total_files = 0

    for src in src_files:
        if not src.exists():
            print(f"[skip] missing: {src}")
            continue

        if args.input_csv is not None:
            dst = args.output_root / f"{src.stem}.csv"
        else:
            dst = make_output_path(src, args.input_root, args.output_root)

        frames, joint_count = convert_one_file(src, dst, root_scale=args.root_scale)
        total_frames += frames
        total_files += 1
        print(f"[ok] {src} -> {dst} | frames={frames}, joints={joint_count}")

    print(f"done: files={total_files}, frames={total_frames}, output_root={args.output_root}")


if __name__ == "__main__":
    main()
