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

### SMPL-X 模型（需单独下载）

SMPL-X 模型文件**不包含**在本仓库中（受其自身许可协议约束）。若要使用 SMPL-X 流水线（`retarget_from_smplx.sh`），请自行下载：

1. 在官网注册并下载：[smplx model download](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip) （下载 "SMPL-X" 模型，`.npz` 格式）。
2. 将文件放到 `asset/smplx/` 目录下，结构如下：

   ```
   asset/smplx/
     SMPLX_NEUTRAL.npz
     SMPLX_MALE.npz
     SMPLX_FEMALE.npz
   ```

### 动作数据集（AMASS）

`dataset/ACCAD/` 中提供的是 AMASS 数据集里 **ACCAD** 子集的少量开源示例动作（SMPL-X `.npz` 格式），仅用于快速体验流水线。若需要更多动作数据，请自行从 AMASS 官网下载：

- AMASS 官网（注册后下载）：https://amass.is.tue.mpg.de/

下载后解压，将 `.npz` 动作文件放到 `dataset/` 下任意目录（例如 `dataset/ACCAD/`），再通过 `SMPL_MOTION_FILE` 指向对应文件即可。

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

![retarget 预览](retarget_from_g1_dance1_subject2_preview.gif)

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
VIS_ROBOTS="jaka_pi h2 t800 pnd_adam" \
ROBOT_MOTION_FILE="dataset/bones_g1/grab_walk_ff_180_001__A550_M.csv" \
ORIGIN_ROBOT="g1" \
SOURCE_FPS=120 \
RENDER_FPS=30 \
./bash/retarget_from_robot.sh
```

运行结束后，重定向得到的机器人动作会保存在 `output_data/robot_motion/` 下。

#### 使用 Bones Seed G1 数据集

[Bones Studio](https://huggingface.co/datasets/bones-studio/seed) 发布了以 G1 机器人为载体的 **Seed** 动作数据集，其原始格式（根节点使用欧拉角/厘米单位，关节值为角度）与本项目所需的 LAFAN1 格式（根节点使用四元数/米单位，关节值为弧度）不同，需要先用 `scripts/convert_bones_to_lafan1.py` 进行转换。

**数据下载**

```bash
# 从 Hugging Face 下载 G1 Seed 数据集
wget https://huggingface.co/datasets/bones-studio/seed/blob/main/g1.tar.gz -O g1.tar.gz
tar -xzf g1.tar.gz -C dataset/bones_g1_origin/
```

**格式转换**

`convert_bones_to_lafan1.py` 将原始 G1 CSV 转换为 LAFAN1 兼容格式：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input-root` | `dataset/bones_g1_origin` | 原始 CSV 所在目录（批量转换） |
| `--output-root` | `dataset/bones_g1` | 转换后 CSV 输出目录 |
| `--input-csv` | 无 | 仅转换单个 CSV 文件 |
| `--root-scale` | `0.01` | 根节点位移缩放系数（cm→m） |

```bash
# 批量转换 dataset/bones_g1_origin/ 下所有 CSV 文件
python scripts/convert_bones_to_lafan1.py

# 仅转换单个文件
python scripts/convert_bones_to_lafan1.py \
    --input-csv dataset/bones_g1_origin/grab_walk_ff_180_001__A550_M.csv

# 自定义输入 / 输出目录
python scripts/convert_bones_to_lafan1.py \
    --input-root dataset/bones_g1_origin \
    --output-root dataset/bones_g1
```

转换完成后，即可将生成的 CSV 作为源动作传入重定向流水线：

```bash
VIS_ROBOTS="jaka_pi h2 t800 pnd_adam" \
ROBOT_MOTION_FILE="dataset/bones_g1/grab_walk_ff_180_001__A550_M.csv" \
ORIGIN_ROBOT="g1" \
SOURCE_FPS=120 \
RENDER_FPS=30 \
./bash/retarget_from_robot.sh
```

## 核心机制

### 1）骨架匹配
source：smplx模型  --->  agibot_x2模型

![smplx骨架与agibot_x2骨架匹配示意图](scale_fig.png)

由于源骨架（SMPL-X）与目标机器人的肢体长度不同，需要按「每段连杆（link）长度比」对源关键点逐段缩放，使其匹配目标机器人的尺寸，同时**保留每段连杆原有的朝向**。

骨架缩放伪代码（对应 `scripts/smpl_replay.py` 中的
`compute_robot_link_lengths` / `compute_link_geometry_from_positions` /
`compute_link_scale_factors` / `apply_link_scales_to_positions`）：

```text
# 输入：
#   robot_mjcf            目标机器人的 MJCF 模型
#   robot_links           连杆定义 {link_name: (parent_body, child_body)}
#   skeleton_positions    源骨架关键点序列, 形状 [T, K, 3]
#   skeleton_links        源骨架连杆定义 {link_name: (parent_body, child_body)}

# 步骤 1: 计算目标机器人每段连杆长度（零位姿、世界坐标下的两端距离）
mj_forward(robot_mjcf)                      # 前向运动学得到各 body 世界坐标
for link_name, (parent, child) in robot_links:
    robot_len[link_name] = || xpos[child] - xpos[parent] ||

# 步骤 2: 计算源骨架每段连杆的方向向量与长度（逐帧）
for link_name, (parent, child) in skeleton_links:
    link_vec[link_name]     = skeleton_positions[:, child] - skeleton_positions[:, parent]   # [T, 3]
    skeleton_len[link_name] = norm(link_vec[link_name], axis=-1)                             # [T]

# 步骤 3: 计算每段连杆的缩放系数（逐帧，目标长度 / 源长度）
for link_name in skeleton_links:
    scale[link_name] = robot_len[link_name] / skeleton_len[link_name]   # [T]

# 步骤 4: 按连杆拓扑从父到子重建关键点位置
#         保持源方向向量不变，仅按 scale 拉伸长度
scaled_pos = copy(skeleton_positions)
for link_name, (parent, child) in skeleton_links:    # 按 父 -> 子 顺序遍历
    scaled_pos[:, child] = scaled_pos[:, parent] + scale[link_name][:, None] * link_vec[link_name]

# 输出：scaled_pos —— 已匹配目标机器人尺寸的关键点
```

用数学符号表达，对第 $i$ 段连杆（父关键点 $p$、子关键点 $c$）在第 $t$ 帧：

$$
s_i^{(t)} = \frac{L_i^{\text{robot}}}{\lVert \mathbf{x}_c^{(t)} - \mathbf{x}_p^{(t)} \rVert}, \qquad
\hat{\mathbf{x}}_c^{(t)} = \hat{\mathbf{x}}_p^{(t)} + s_i^{(t)} \left( \mathbf{x}_c^{(t)} - \mathbf{x}_p^{(t)} \right)
$$

其中 $L_i^{\text{robot}}$ 是机器人该连杆的固定长度，$\mathbf{x}$ 为原始关键点，$\hat{\mathbf{x}}$ 为缩放后的关键点。

> 说明：
> - 缩放沿「父关键点 → 子关键点」逐段进行，子关键点位置由父关键点加上「原方向向量 × 缩放系数」得到，因此只改变长度、不改变朝向。
> - 缩放系数是**逐帧**计算的（`link_scale_is_static = False`），可适应源骨架长度随姿态的细微变化。
> - 此外，整体的根节点（root）平移还会按腿长比例缩放（`compute_leg_displacement_scale` / `scale_keypoint_frame_displacements`），使步幅与机器人腿长匹配。

#### Root 缩放

为使不同体型（尤其腿长差异较大）的源/目标模型在运动幅度上保持一致，
会根据「目标腿长 / 源腿长」计算位移缩放系数，并对 root 在 $x,y,z$ 三个方向的帧间位移统一缩放。

- 腿长定义：大腿长度 + 小腿长度（左右腿取平均）。
- 缩放目标：主要调节整体步幅与位移尺度，使动作在目标机器人上更自然。

#### 适配注意事项（斜胯构型）

在当前骨架定义中，`left_shoulder - neck - right_shoulder` 与
`left_hip - hips_mean - right_hip` 默认近似共线（水平）。
对于类似众擎 t800 的斜胯构型，建议在髋部左右两侧额外添加两个映射点，
并与对应 `hip_roll_link` 建立 fixed 连接，以更稳定地表达骨盆倾斜与左右髋差异。

![t800注意事项](hip_spheres.png)


### 2）膝关节弯曲

为提升下肢构型的可达性与力学表现，流程中会对膝部做两段骨（hip-knee-foot）几何重建。
该过程由 `knee_angle_offset_degrees` 控制弯曲强度：

- 文档示例图使用 `60.0` 度仅用于展示效果。
- 实际配置常用值为 `15.0` 度。
- 在 `smpl_replay.py` 中该偏置默认参与计算。
- 在 `robot_replay.py` 中需显式开启 `enable_knee_angle_offset_degrees: true` 才会生效。

核心是两段骨 IK 与空间余弦定理。设：

- $a$：大腿长度（hip->knee）
- $b$：小腿长度（knee->foot）
- $d$：目标髋踝距离（hip->target\_foot）

则沿髋踝方向的投影长度与垂向抬起量分别为：

$$
x = \frac{a^2 - b^2 + d^2}{2d}, \qquad
h = \sqrt{\max(a^2 - x^2,\, 0)}
$$

重建后的膝位置可写为：

$$
\mathbf{p}_{knee} = \mathbf{p}_{hip} + x\,\mathbf{u} + h\,\mathbf{v}
$$

其中 $\mathbf{u}$ 为髋到目标踝的单位方向，$\mathbf{v}$ 为在弯曲平面内的弯曲方向。 
这里特别说明：在膝关节弯曲步骤中，会保持「髋关节到踝关节」向量方向（即 $\mathbf{u}$ 方向）不变，
仅调整该向量的长度（对应目标髋踝距离 $d$），再据此重建膝位置。
代码实现中会保持弯曲方向与原始屈膝偏好一致，避免膝盖反转。

![blend_knee](blend_fig.png)

### 3）接触检测

接触判定通常覆盖双手与双脚；其中每只脚会设置前、后两个接触检测点，以更稳定地识别支撑与离地状态。
![foot_contact](foot_contact_fig.png)

接触状态按「低速度 + 低高度」双条件判定（见 `compute_contact_sequence` /
`compute_robot_contact_sequence`）：

$$
	ext{contact}(t,c)=\big(v_{t,c}\le v_{th}\big)\;\land\;\big(z_{t,c}\le h_{th}\big)
$$

其中：

- $v_{t,c}$：接触点 $c$ 在第 $t$ 帧的窗口速度（由 `contact_vel_calculate_window` 计算）
- $z_{t,c}$：接触点高度
- $v_{th}$：`contact_vel_threshold`
- $h_{th}$：`contact_height_threshold`

该设计可同时抑制“低空高速掠过”与“高空慢速摆动”导致的误判。

### 4）自适应高度

在检测到接触后，系统会估计每一帧的地面高度偏移，并整体下移关键点序列（仅 $z$ 方向），
使支撑脚更稳定地贴合地面。

1. 在每帧选择当前激活接触点中的最小高度作为基准高度。
2. 若该帧无激活接触，则沿用上一帧高度。
3. 对高度序列做一阶低通滤波（`contact_height_lpf_alpha`）：

$$
y_t = \alpha x_t + (1-\alpha)y_{t-1}, \qquad 0<\alpha\le1
$$

4. 对所有关键点执行 $z$ 向平移：

$$
z'_{t,k} = z_{t,k} - y_t
$$

其中，较小的 $\alpha$ 会带来更平滑但更滞后的地面跟随效果。

### 5）足端接触滑动抑制

在 `robot_retarget.py` 中，针对处于接触状态的足端（或配置的接触体）引入“接触锁定目标”：

1. 将连续接触区间内的源关键点位置取均值，得到该区间的固定目标点。
2. 接触为真时，IK 目标使用该固定点；接触为假时，恢复跟随原始目标。
3. 该约束以额外 `FrameTask` 形式加入优化，权重由 `contact_pos_fixed_factor` 控制。

这样可以显著减小支撑相内的足端漂移（foot sliding），同时在摆动相保持动作自由度。

