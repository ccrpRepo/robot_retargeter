# robot_retargeter

中文 | [English](README.md)

将人体动作（SMPL-X）或源机器人动作重定向到目标人形机器人，并支持多机器人并排可视化的工具集。

## 简介

本项目提供一条完整的动作重定向流水线，主要包含三步：

1. **回放 / 提取关键点**：从 SMPL-X 动作（`.npz`）或源机器人动作（`.csv`）中提取骨骼关键点。
2. **重定向（Retarget）**：基于逆运动学（[mink](https://github.com/kevinzakka/mink) + MuJoCo）把关键点映射到目标机器人模型上，生成机器人动作。
3. **可视化**：将一个或多个目标机器人的重定向结果并排播放查看。

目录结构概览：

| 目录 | 内容 |
|---|---|
| `asset/` | 机器人模型（URDF/MJCF/mesh）、SMPL-X 模型、骨架 |
| `config/` | 机器人与骨架的重定向配置（YAML） |
| `dataset/` | 输入动作数据（SMPL-X `.npz` / LAFAN1 `.csv` 等） |
| `output_data/` | 输出的关键点与机器人动作 |
| `scripts/` | Python 流水线脚本 |
| `bash/` | 一键运行的流水线脚本 |

## 安装
### 克隆仓库

仓库中的较大机器人 mesh / 纹理文件（`*.stl`、`*.obj`、`*.dae`、`*.png`、`*.mtl`）直接随仓库提交，使用普通 Git 即可克隆。

```bash
# 使用普通 Git 克隆
git clone https://github.com/ccrpRepo/robot_retargeter.git
cd robot_retargeter
```

### Python 环境
需要 Python ≥ 3.10（开发环境为 Python 3.11）。建议使用虚拟环境（conda / venv）。

```bash
# 1) 创建并激活环境（以 conda 为例）
conda create -n robot_retargeter python=3.11 -y
conda activate robot_retargeter

# 2) 安装依赖
pip install -e .
# 或
pip install -r requirements.txt
```

主要依赖：`mujoco`、`mink`、`numpy`、`torch`、`scipy`、`PyYAML`、`tqdm`、`glfw`、`smplx`、`trimesh`（精确版本下限见 `setup.py` / `requirements.txt`）。

### SMPL-X 模型（需单独下载）

SMPL-X 模型文件**不包含**在本仓库中（受其自身许可协议约束）。若要使用 SMPL-X 流水线（`retarget_from_smplx.sh`），请自行下载：

1. 在官网注册并下载：https://smpl-x.is.tue.mpg.de/ （下载 "SMPL-X" 模型，`.npz` 格式）。
2. 将文件放到 `asset/smplx/` 目录下，结构如下：

   ```
   asset/smplx/
     SMPLX_NEUTRAL.npz
     SMPLX_MALE.npz
     SMPLX_FEMALE.npz
   ```

> 下载即表示你同意 SMPL-X 的许可协议。该目录已被 git 忽略，不会被提交。

> 注意：较大的机器人 mesh / 纹理文件（`*.stl`、`*.obj`、`*.dae`、`*.png`、`*.mtl`）已直接存储在仓库中，因此首次克隆耗时可能更长。

### 动作数据集（AMASS / ACCAD）

`dataset/ACCAD/` 中提供的是 AMASS 数据集里 **ACCAD** 子集的少量开源示例动作（SMPL-X `.npz` 格式），仅用于快速体验流水线。若需要更多动作数据，请自行从 AMASS 官网下载：

- AMASS 官网（注册后下载）：https://amass.is.tue.mpg.de/
- ACCAD 子集下载页：https://amass.is.tue.mpg.de/download.php （在列表中选择 **ACCAD**，下载 `SMPL-X N`（neutral）格式）

下载后解压，将 `.npz` 动作文件放到 `dataset/` 下任意目录（例如 `dataset/ACCAD/`），再通过 `SMPL_MOTION_FILE` 指向对应文件即可。

> 下载即表示你同意 AMASS / ACCAD 的许可协议，相关数据仅可用于其许可允许的用途。

## 运行

`bash/` 目录提供了两个一键流水线脚本，会自动完成「关键映射点生成 → 重定向 → 可视化」三步。脚本默认使用当前激活环境中的 `python`，也可用 `PYTHON_BIN` 指定解释器。

### 1) 从 SMPL-X 动作重定向

```bash
./bash/retarget_from_smplx.sh
```

可通过环境变量自定义参数：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SMPL_MOTION_FILE` | `dataset/ACCAD/Form_1_stageii.npz` | 输入的 SMPL-X 动作文件 |
| `VIS_ROBOTS` | `g1 h2 t800 r1` | 目标机器人列表（空格分隔，支持多个） |
| `KEYPOINTS_NAME` | 由动作文件名自动推导 | 关键点 / 输出动作名称 |
| `SOURCE_FPS` | `120` | 源动作帧率 |
| `RENDER_FPS` | `30` | 可视化渲染帧率 |
| `PYTHON_BIN` | 自动检测 | 指定 Python 解释器 |

示例（自定义机器人与动作文件）：

```bash
VIS_ROBOTS="g1 jaka_pi h2 t800" \
SMPL_MOTION_FILE="dataset/ACCAD/Form_1_stageii.npz" \
./bash/retarget_from_smplx.sh
```

### 2) 从源机器人动作重定向

演示预览（点击 GIF 打开 MP4）：

[![retarget 预览](retarget_from_g1_dance1_subject2_preview.gif)](retarget_from_g1_dance1_subject2.mp4)

视频直链：
[retarget_from_g1_dance1_subject2.mp4](retarget_from_g1_dance1_subject2.mp4)

```bash
./bash/retarget_from_robot.sh
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ROBOT_MOTION_FILE` | `dataset/lafan1_g1/dance1_subject2.csv` | 源机器人动作文件 |
| `ORIGIN_ROBOT` | `g1` | 源机器人名称（提供骨架配置） |
| `VIS_ROBOTS` | `h2 r1` | 目标机器人列表（空格分隔，支持多个） |
| `SOURCE_FPS` | `30` | 源动作帧率 |
| `RENDER_FPS` | `30` | 可视化渲染帧率 |
| `PYTHON_BIN` | 自动检测 | 指定 Python 解释器 |

示例：

```bash
VIS_ROBOTS="jaka_pi h2 t800 agibot_x2" \
ROBOT_MOTION_FILE="dataset/lafan1_g1/dance1_subject2.csv" \
ORIGIN_ROBOT="g1" \
./bash/retarget_from_robot.sh
```

运行结束后，重定向得到的机器人动作会保存在 `output_data/robot_motion/` 下。
