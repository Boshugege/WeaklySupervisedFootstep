# 弱监督DAS脚步检测系统

```bash
# 单人训练（默认：头部裁掉50s，尾部裁掉20s，跳过前15个通道）
python workflow_train.py wangdihai

# 自定义裁剪
python workflow_train.py wangdihai --trim_head 60 --trim_tail 30

# 多人联合训练（提高泛化性）
python workflow_train_multi.py wangdihai wangjiahui

# 推理
python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib
python workflow_infer.py wuwenxuan --model output/multi_wangdihai_wangjiahui/models/multi_wangdihai_wangjiahui.joblib
```

```
期望目录结构
Data/
├── Video/          # 视频文件（MP4）
│   ├── wangdihai.MP4
│   ├── wangjiahui.MP4
│   └── wuwenxuan.MP4
├── Airtag/         # Airtag时间标记CSV
│   ├── wangdihai.csv
│   ├── wangjiahui.csv
│   └── ...
└── DAS/            # TDMS原始数据
    └── *.tdms
```

## 项目概述

本项目实现了一个基于**弱监督学习**的分布式光纤声学传感(DAS)脚步事件检测系统。系统利用音频信号作为弱监督源，在没有精确标注的情况下，从DAS数据中自动检测脚步事件并定位其通道位置。

### 核心特性

- ✅ **弱监督学习**：无需人工标注，利用音频自动生成弱标签
- ✅ **自训练迭代**：通过多轮自训练逐步提高检测召回率
- ✅ **多频带分析**：支持多个频带特征提取
- ✅ **智能时间裁剪**：自动根据数据长度计算裁剪范围（头部/尾部）
- ✅ **通道屏蔽**：可跳过噪声通道（默认跳过前15个）
- ✅ **多人联合训练**：支持多人数据联合训练提高泛化性
- ✅ **丰富可视化**：热图、轨迹图、对比图等多种输出
- ✅ **调优工具**：独立的音频检测验证工具

---

## 文件结构

```
WeaklySupervisedFootstep/
├── WeaklySupervised_FootstepDetector.py   # 主检测程序
├── workflow_train.py                      # 单人训练工作流
├── workflow_train_multi.py                # 多人联合训练工作流
├── workflow_infer.py                      # 推理工作流
├── extract_name_signals_from_tdms.py      # TDMS信号提取（支持通道屏蔽）
├── audio_step_tuning.py                   # 音频检测调优工具
├── README.md                              # 本文档
├── Data/                                  # 数据目录
│   ├── Video/                             # 视频文件
│   ├── Airtag/                            # Airtag CSV
│   └── DAS/                               # TDMS原始数据
└── output/                                # 输出目录
```

---

## 脚本说明

### 1. WeaklySupervised_FootstepDetector.py

**主检测程序** - 完整的弱监督脚步检测流程

#### 功能

- 加载DAS CSV数据和音频/视频文件
- 多频带滤波处理
- 音频弱标签提取
- 弱监督模型训练
- 自训练迭代优化
- 通道位置估计
- 可视化输出

#### 命令行参数

| 参数                     | 类型  | 默认值                   | 说明                   |
| ------------------------ | ----- | ------------------------ | ---------------------- |
| `--das_csv`, `-d`        | str   | 必需                     | DAS CSV文件路径        |
| `--audio`, `-a`          | str   | None                     | 音频/视频文件路径      |
| `--trim_start`           | float | None                     | 数据起始裁剪时间（秒） |
| `--trim_end`             | float | None                     | 数据结束裁剪时间（秒） |
| `--das_fs`               | int   | 2000                     | DAS采样率 (Hz)         |
| `--audio_sr`             | int   | 48000                    | 音频重采样率 (Hz)      |
| `--align_dt`             | float | 0.0                      | 时间对齐偏移量         |
| `--self_train_rounds`    | int   | 3                        | 自训练迭代轮数         |
| `--confidence_threshold` | float | 0.7                      | 高置信预测阈值         |
| `--output_dir`, `-o`     | str   | output/weakly_supervised | 输出目录               |

#### 使用示例

```bash
python WeaklySupervised_FootstepDetector.py \
    --das_csv "path/to/data.csv" \
    --audio "path/to/video.mp4" \
    --trim_start 50 \
    --trim_end 200 \
    --output_dir "output/results"
```

---

### 2. workflow_train.py

**训练工作流** - 单人训练的完整流程

#### 功能

- 自动提取TDMS信号（支持通道屏蔽）
- 自动计算时间裁剪范围（根据数据长度）
- 训练弱监督模型并保存

#### 命令行参数

| 参数                     | 类型  | 默认值 | 说明                               |
| ------------------------ | ----- | ------ | ---------------------------------- |
| `name`                   | str   | 必需   | 目标名字（对应Video/Airtag文件名） |
| `--trim_head`            | float | 50.0   | 从数据开头裁掉的时长（秒）         |
| `--trim_tail`            | float | 20.0   | 从数据结尾裁掉的时长（秒）         |
| `--skip_channels`        | int   | 15     | 跳过前N个通道（设为0保留所有）     |
| `--das_fs`               | int   | 2000   | DAS采样率 (Hz)                     |
| `--self_train_rounds`    | int   | 3      | 自训练迭代轮数                     |
| `--confidence_threshold` | float | 0.7    | 高置信预测阈值                     |
| `--overwrite`            | flag  | -      | 覆盖已存在的输出文件               |

#### 使用示例

```bash
# 使用默认参数（头部-50s，尾部-20s，跳过前15通道）
python workflow_train.py wangdihai

# 自定义裁剪
python workflow_train.py wangdihai --trim_head 60 --trim_tail 30

# 保留所有通道
python workflow_train.py wangdihai --skip_channels 0
```

---

### 3. workflow_train_multi.py

**多人联合训练工作流** - 提高模型泛化性

#### 功能

- 合并多人数据进行联合训练
- 统一的时间裁剪配置
- 输出通用模型供推理使用

#### 命令行参数

| 参数                  | 类型  | 默认值          | 说明                       |
| --------------------- | ----- | --------------- | -------------------------- |
| `names`               | str[] | 必需（至少2个） | 训练数据的名字列表         |
| `--trim_head`         | float | 50.0            | 从数据开头裁掉的时长（秒） |
| `--trim_tail`         | float | 20.0            | 从数据结尾裁掉的时长（秒） |
| `--skip_channels`     | int   | 15              | 跳过前N个通道              |
| `--model_name`        | str   | auto            | 输出模型名称               |
| `--self_train_rounds` | int   | 3               | 自训练迭代轮数             |

#### 使用示例

```bash
# 两人联合训练
python workflow_train_multi.py wangdihai wangjiahui

# 三人联合训练，自定义模型名
python workflow_train_multi.py wangdihai wangjiahui wuwenxuan --model_name multi_3person

# 自定义裁剪
python workflow_train_multi.py wangdihai wangjiahui --trim_head 60 --trim_tail 30
```

---

### 4. workflow_infer.py

**推理工作流** - 使用已训练模型检测脚步

#### 功能

- 加载预训练模型
- 自动提取信号并检测脚步
- 支持无视频纯推理模式

#### 命令行参数

| 参数              | 类型  | 默认值 | 说明                       |
| ----------------- | ----- | ------ | -------------------------- |
| `name`            | str   | 必需   | 目标名字                   |
| `--model`, `-m`   | str   | 必需   | 模型文件路径（.joblib）    |
| `--trim_head`     | float | 50.0   | 从数据开头裁掉的时长（秒） |
| `--trim_tail`     | float | 20.0   | 从数据结尾裁掉的时长（秒） |
| `--skip_channels` | int   | 15     | 跳过前N个通道              |
| `--with_audio`    | flag  | -      | 同时使用音频进行对比验证   |

#### 使用示例

```bash
# 基本推理
python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib

# 使用联合模型
python workflow_infer.py wuwenxuan --model output/multi_wangdihai_wangjiahui/models/multi_wangdihai_wangjiahui.joblib

# 带音频对比
python workflow_infer.py wangjiahui --model model.joblib --with_audio
```

---

### 5. extract_name_signals_from_tdms.py

**信号提取脚本** - 从TDMS提取DAS信号

#### 功能

- 根据Airtag时间标记提取TDMS信号
- 支持跳过前N个通道（通道屏蔽）
- 输出CSV格式

#### 命令行参数

| 参数                | 类型  | 默认值 | 说明                           |
| ------------------- | ----- | ------ | ------------------------------ |
| `--skip-channels`   | int   | 15     | 跳过前N个通道（设为0保留所有） |
| `--fs`              | float | 2000   | DAS采样率 (Hz)                 |
| `--use-airtag-only` | flag  | -      | 仅使用Airtag（无需Video）      |

---

### 6. audio_step_tuning.py

**音频检测调优工具** - 验证和调整音频脚步检测参数

#### 功能

- 独立验证音频脚步检测的可信度
- 可视化音频波形和检测结果
- 输出调优建议
- 与DAS数据叠加对比

#### 命令行参数

| 参数                 | 类型  | 默认值              | 说明                              |
| -------------------- | ----- | ------------------- | --------------------------------- |
| `--audio`, `-a`      | str   | 必需                | 音频/视频文件路径                 |
| `--das_csv`, `-d`    | str   | None                | DAS CSV文件路径（可选，用于叠加） |
| `--output_dir`, `-o` | str   | output/audio_tuning | 输出目录                          |
| `--trim_start`       | float | 50.0                | 开始时间（秒）                    |
| `--trim_end`         | float | 200.0               | 结束时间（秒）                    |
| `--bp_low`           | float | 4000                | 带通低频 (Hz)                     |
| `--bp_high`          | float | 10000               | 带通高频 (Hz)                     |
| `--min_interval`     | float | 0.30                | 最小脚步间隔 (s)                  |
| `--peak_prom`        | float | 1.5                 | 峰值显著性阈值                    |
| `--peak_height`      | float | 0.8                 | 峰值高度阈值                      |

#### 使用示例

```bash
# 基本使用
python audio_step_tuning.py \
    --audio "path/to/video.mp4" \
    --trim_start 50 \
    --trim_end 200

# 调整检测参数
python audio_step_tuning.py \
    --audio "path/to/video.mp4" \
    --peak_height 0.5 \
    --peak_prom 1.0

# 与DAS叠加对比
python audio_step_tuning.py \
    --audio "path/to/video.mp4" \
    --das_csv "path/to/data.csv" \
    --trim_start 50 \
    --trim_end 200
```

---

## 输入数据格式

### DAS CSV文件

```csv
ch_0,ch_1,ch_2,...,ch_191
182,173,182,...,0
120,120,134,...,-4
...
```

- **列**：通道名 `ch_0`, `ch_1`, ..., `ch_N`
- **行**：时间采样点（每行一个时间点）
- **采样率**：2000 Hz
- **数值**：信号强度（整数或浮点数）

### 音频/视频文件

- 支持格式：MP4, MOV, WAV, M4A, MP3 等
- 建议采样率：≥ 44100 Hz（支持10kHz高频）
- 单声道或立体声均可

---

## 输出文件说明

### 1. 主检测结果

#### `*_steps.csv`

最终脚步检测结果，结构化CSV格式：

```csv
time,channel,confidence
0.68,143,0.988
0.98,16,0.506
1.31,16,0.457
...
```

| 字段         | 说明                                              |
| ------------ | ------------------------------------------------- |
| `time`       | 脚步发生时间（秒），相对于裁剪后的数据起点        |
| `channel`    | 脚步最可能发生的通道索引（0-N）                   |
| `confidence` | 置信度（0-1），综合时间检测置信度和通道估计置信度 |

**置信度解释**：

- `> 0.8`：高置信，可直接使用
- `0.5-0.8`：中置信，建议人工复核
- `< 0.5`：低置信，可能是误检

---

### 2. 可视化输出

#### `*_heatmap_steps.png`

**主热图** - DAS能量热图 + 脚步事件标记

![示例](examples/output/wangdihai_heatmap_steps.png)

- **上图**：时间×通道能量热图
  - 横轴：时间（秒）
  - 纵轴：通道索引
  - 颜色：对数能量（viridis色彩映射）
  - 散点：检测到的脚步事件，颜色表示置信度
- **下图**：时间轴上的脚步概率曲线

**用途**：

- 查看脚步事件在时间和通道上的分布
- 验证通道位置估计是否合理
- 观察脚步事件的密度和规律

---

#### `*_detailed_segments.png`

**分段详细视图** - 多个时间段的放大视图

- 将数据分成多个15秒片段
- 每段单独显示热图和脚步标记
- 标注每段检测到的脚步数量

**用途**：

- 详细检查特定时间段的检测结果
- 发现遗漏或误检的脚步

---

#### `*_channel_trajectory.png`

**通道轨迹图** - 脚步通道随时间的变化

- **上图**：通道轨迹散点图
  - 横轴：时间
  - 纵轴：通道
  - 颜色：置信度
- **下图**：步频间隔分布直方图
  - 显示平均和中位数间隔

**用途**：

- 观察行走轨迹（通道位置变化）
- 分析步频稳定性
- 检测异常区间（间隔过长或过短）

---

#### `*_comparison.png`

**对比图** - 音频检测 vs DAS检测

- **第1行**：音频包络曲线 + 音频检测的脚步
- **第2行**：DAS概率曲线 + DAS检测的脚步
- **第3行**：两种检测的叠加对比

**用途**：

- 验证DAS检测与音频检测的一致性
- 发现DAS独立发现的新脚步
- 评估弱监督效果

---

### 3. 音频调优输出

#### `*_audio_tuning_overview.png`

**调优总览图** - 全面的音频检测分析

- **第1行**：原始波形 + 带通滤波后波形
- **第2行**：Z-score包络 + 检测阈值线
- **第3行左**：脚步间隔分布直方图
- **第3行右**：峰值高度分布直方图
- **第4行**：详细片段视图（中间10秒）

**用途**：

- 验证音频检测参数是否合适
- 观察检测阈值与峰值的关系
- 分析步频分布是否合理

---

#### `*_step_segments.png`

**脚步波形片段** - 每个脚步附近的波形

- 显示12个均匀分布的脚步
- 每个图展示脚步前后0.5秒的波形
- 红色竖线标记脚步时刻

**用途**：

- 确认检测到的是真实脚步声
- 观察脚步信号的波形特征
- 识别误检（非脚步的峰值）

---

#### `*_das_audio_overlay.png`

**DAS+音频叠加图** - 两路数据的对齐验证

- **第1行**：DAS能量热图 + 音频脚步标记（红线）
- **第2行**：DAS总能量曲线（所有通道叠加）
- **第3行**：音频包络曲线

**用途**：

- 验证音频和DAS数据的时间对齐
- 观察音频脚步在DAS热图上的响应
- 确定是否需要调整align_dt参数

---

#### `*_audio_steps.csv`

音频检测的脚步时间列表：

```csv
time,height
0.223,1.625
0.538,1.336
1.012,1.101
...
```

| 字段     | 说明                |
| -------- | ------------------- |
| `time`   | 脚步时间（秒）      |
| `height` | 峰值高度（Z-score） |

---

## 参数调优指南

### 音频检测参数

#### 漏检太多（召回率低）

| 参数          | 调整方向 | 原因               |
| ------------- | -------- | ------------------ |
| `peak_height` | ↓ 降低   | 接受更弱的峰值     |
| `peak_prom`   | ↓ 降低   | 接受不太显著的峰值 |
| `bp_low`      | ↓ 降低   | 捕获更多低频成分   |

```bash
python audio_step_tuning.py --audio video.mp4 --peak_height 0.5 --peak_prom 1.0
```

#### 误检太多（精确率低）

| 参数           | 调整方向 | 原因             |
| -------------- | -------- | ---------------- |
| `peak_height`  | ↑ 提高   | 过滤弱信号       |
| `peak_prom`    | ↑ 提高   | 要求更显著的峰值 |
| `min_interval` | ↑ 提高   | 避免连续误检     |

```bash
python audio_step_tuning.py --audio video.mp4 --peak_height 1.0 --peak_prom 2.0
```

---

### 弱监督模型参数

#### 检测结果太少

| 参数                   | 调整方向 | 原因             |
| ---------------------- | -------- | ---------------- |
| `self_train_rounds`    | ↑ 增加   | 更多迭代补全漏检 |
| `confidence_threshold` | ↓ 降低   | 接受更多新检测   |

#### 检测结果噪声多

| 参数                   | 调整方向 | 原因             |
| ---------------------- | -------- | ---------------- |
| `confidence_threshold` | ↑ 提高   | 只保留高置信预测 |
| `self_train_rounds`    | ↓ 减少   | 避免过度扩展     |

---

### 滤波参数

#### DAS滤波

默认使用三个频带：

- `5-10 Hz`：主频带（脚步主要响应）
- `2-5 Hz`：低频辅助
- `10-20 Hz`：高频辅助

可在 `Config` 类中修改 `das_bp_bands`。

#### 音频滤波

- `4-10 kHz`：脚步声主要频带

如果环境噪声较大，可尝试：

- 提高 `bp_low` 到 5000 Hz
- 降低 `bp_high` 到 8000 Hz

---

## 示例输入输出

### 输入数据示例

**DAS CSV** (`wangdihai.csv`)：

- 形状：434001行 × 192列
- 采样率：2000 Hz
- 时长：约217秒
- 通道：ch_0 到 ch_191

**视频** (`wangdihai.MP4`)：

- 第一人称GoPro视频
- 包含脚步声音频
- 时长：与DAS数据对应

### 处理配置

```python
trim_start = 50.0    # 从第50秒开始
trim_end = 200.0     # 到第200秒结束（3分20秒）
align_dt = 0.0       # 无需额外时间对齐
```

### 输出结果统计

| 指标             | 值              |
| ---------------- | --------------- |
| 音频检测脚步数   | 202             |
| 弱监督检测脚步数 | 367             |
| 检测时间范围     | 0.68s - 149.06s |
| 平均置信度       | 0.639           |
| 通道范围         | 0 - 143         |
| 估计步频         | ~80 步/分钟     |

### 输出文件

见 `examples/output/` 目录下的所有文件。

---

## 常见问题

### Q: 为什么需要时间裁剪？

视频和DAS信号的开头结尾通常包含非步行部分（如设备启动、站立等），裁剪可以只保留有效的步行数据。

### Q: 如何确定合适的裁剪范围？

1. 先用 `audio_step_tuning.py` 大致浏览全时间段
2. 观察脚步分布，确定步行开始和结束时间
3. 设置 `trim_start` 和 `trim_end`

### Q: 音频和DAS时间不对齐怎么办？

1. 运行 `Tools/align_das_audio_qc.py` 计算偏移量
2. 使用 `--align_dt` 参数传入偏移值
3. DAS时间 = 音频时间 + align_dt

### Q: 没有对应的音频文件怎么办？

系统会自动切换到纯能量检测模式，但效果会下降。建议尽量提供音频文件。

### Q: 检测到的通道位置不准确？

通道位置估计基于能量分布，在信号较弱时可能不准确。可以：

1. 提高 `confidence_threshold` 只保留高置信结果
2. 检查是否存在热通道干扰
3. 调整DAS滤波频带

### Q: 训练好的模型可以用到其他数据上吗？

**可以！** 只要布设和环境条件相似。使用模型保存/加载功能：

```bash
# 训练时保存模型
python WeaklySupervised_FootstepDetector.py \
    --das_csv train_data.csv \
    --audio train_video.mp4 \
    --save_model models/my_model.joblib

# 在新数据上使用（无需音频）
python WeaklySupervised_FootstepDetector.py \
    --das_csv new_data.csv \
    --load_model models/my_model.joblib \
    --inference_only
```

**模型迁移适用场景**：

- ✅ 相同布设、相同环境
- ✅ 相似的行走方式
- ⚠️ 布设变化较大时建议重新训练

**模型文件内容**：

- 训练好的分类器（RandomForest/GradientBoosting）
- 特征标准化器（StandardScaler）
- 训练配置参数

---

## 依赖安装

```bash
pip install numpy pandas scipy matplotlib librosa scikit-learn
```

版本要求：

- Python >= 3.8
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7
- matplotlib >= 3.4
- librosa >= 0.9
- scikit-learn >= 1.0

---

## 算法原理

### 1. 音频弱标签提取

```
音频 → 4-10kHz带通滤波 → RMS包络 → 对数变换 → 鲁棒Z-score → 峰值检测 → 脚步候选时间
```

### 2. DAS多频带特征提取

```
DAS → 多频带带通滤波 → 短时能量 → 特征向量
     ├── 5-10Hz（主频带）
     ├── 2-5Hz（低频）
     └── 10-20Hz（高频）
```

### 3. 弱监督训练

- **正样本**：音频检测到的脚步时间点（允许±0.5s误差）
- **负样本**：远离任何脚步的时间点
- **分类器**：随机森林（默认）或梯度提升

### 4. 自训练迭代

```
初始弱标签 → 训练模型 → 预测 → 高置信预测 → 合并标签 → 重复
```

每轮迭代：

1. 用当前标签训练新模型
2. 在DAS数据上预测
3. 筛选置信度 > 0.7 的新检测
4. 合并到标签集，排除已有脚步附近的重复
5. 重复，逐步提高召回率

### 5. 通道位置估计

```
脚步时间 → 提取时间窗口 → 各通道能量 → Softmax归一化 → 选择最高能量通道
```

---

## 版本历史

- **v1.1** (2026-02-03)：模型保存/加载功能
  - 新增 `--save_model` 参数保存训练好的模型
  - 新增 `--load_model` 参数加载已有模型
  - 新增 `--inference_only` 仅推理模式（无需音频）
  - 支持模型跨数据迁移使用

- **v1.0** (2026-02-03)：初始版本
  - 弱监督脚步检测
  - 自训练迭代
  - 多种可视化输出
  - 音频调优工具
