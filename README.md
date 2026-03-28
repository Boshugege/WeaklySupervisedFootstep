# 弱监督 DAS 脚步检测项目使用说明

本项目用于在分布式光纤声学传感（DAS）数据中检测脚步事件，并估计脚步发生的通道位置。当前仓库包含三条主线：

1. 离线训练：从 `name` 出发，按 Airtag 时间从 TDMS 中切出 DAS CSV，再结合音频弱标签训练模型。
2. 离线推理：使用已有模型对新数据进行推理，可选带音频对比，也可将离线结果按在线 JSON 协议回放。
3. 在线推理与可视化：按 TDMS 分块模拟在线推理，并将 `signal/event` JSON 包发送给 `JSONStreamPlot.py` 实时显示。

这份 README 只描述当前仓库里仍然有效、仍然推荐使用的脚本和目录，不再以历史版本为准。

## 1. 当前建议使用的入口

| 脚本 | 作用 | 是否推荐作为主入口 |
| --- | --- | --- |
| `workflow_train.py` | 离线训练完整流程 | 是 |
| `workflow_infer.py` | 离线推理完整流程 | 是 |
| `WeaklySupervised_FootstepDetector.py` | 核心训练/推理脚本 | 是，适合高级用法 |
| `extract_name_signals_from_tdms.py` | 从 TDMS 按名字切出 DAS CSV | 是 |
| `realtime_infer_stream.py` | 在线推理模拟并发送 JSON 流 | 是 |
| `JSONStreamPlot.py` | 接收在线 JSON 流并实时可视化 | 是 |
| `experiment/WeaklySupervised_FootstepDetector_3D_CNN.py` | 3D-CNN 实验方向 | 实验性质 |

`deprecated/` 中的内容目前已废弃。

## 2. 目录与数据约定

当前仓库目录中，和实际运行最相关的是以下部分：

```text
WeaklySupervisedFootstep/
├── Data/
│   ├── Airtag/
│   ├── Audio/
│   ├── Video/
│   └── DAS/
├── output/
├── examples/
├── experiment/
├── deprecated/
├── workflow_train.py
├── workflow_infer.py
├── WeaklySupervised_FootstepDetector.py
├── wsfd/
│   ├── __init__.py
│   ├── config.py
│   ├── signal_utils.py
│   ├── audio_labels.py
│   ├── features.py
│   ├── models.py
│   ├── detector.py
│   ├── visualization.py
│   ├── pipeline.py
│   └── cli.py
├── extract_name_signals_from_tdms.py
├── realtime_infer_stream.py
├── JSONStreamPlot.py
└── requirements.txt
```

### 2.1 当前代码结构

`WeaklySupervised_FootstepDetector.py` 现在已经不是 2000 行的单体实现，而是一个兼容入口。实际实现集中在 `wsfd/`：

- [WeaklySupervised_FootstepDetector.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/WeaklySupervised_FootstepDetector.py)
  作用：根目录兼容入口，对外保留原来的命令行和导出名。
- [wsfd/config.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/config.py)
  作用：集中维护配置对象 `Config`。
- [wsfd/signal_utils.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/signal_utils.py)
  作用：带通、鲁棒 z-score、RMS、移动平均等通用信号处理函数。
- [wsfd/audio_labels.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/audio_labels.py)
  作用：从音频提取脚步候选时间，生成弱标签。
- [wsfd/features.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/features.py)
  作用：加载 DAS CSV、做多频带滤波、提特征、提 CNN patch、估计通道。
- [wsfd/models.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/models.py)
  作用：定义 CNN 结构和 PyTorch 二分类包装器。
- [wsfd/detector.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/detector.py)
  作用：负责准备训练样本、训练分类器、在时间网格上预测、自训练和模型保存加载。
- [wsfd/visualization.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/visualization.py)
  作用：集中管理热图、轨迹图、对比图等输出。
- [wsfd/pipeline.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/pipeline.py)
  作用：串联完整训练流程和纯推理流程。
- [wsfd/cli.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/cli.py)
  作用：核心脚本命令行入口。

### 2.2 数据目录含义

```text
Data/
├── Audio/
│   └── <name>.mp3
├── Video/
│   └── <name>.MP4 / <name>.mp4 / <name>.MOV / <name>.mov
├── Airtag/
│   └── <name>.csv
└── DAS/
    └── *.tdms
```

约定如下：

- `name` 是整个流程的主索引键。
- 训练工作流默认要求 `Data/Audio/<name>.mp3` 存在。
- 纯推理不要求 `Data/Video/<name>.*`。
- 只有在推理时显式使用 `--with_audio` 做对比时，才会尝试读取 `Data/Video/<name>.*`。
- `Data/Airtag/<name>.csv` 用于提供该人的起止时间窗口。
- `Data/DAS/*.tdms` 是原始 DAS 文件，脚本会根据 Airtag 时间自动找重叠段并切片。

### 2.3 Airtag CSV 基本要求

`extract_name_signals_from_tdms.py` 要求 Airtag CSV 第一列是时间列，并且表头以 `datetime` 开头。脚本支持多种常见时间格式，包括：

- `YYYY-mm-dd HH:MM:SS`
- `YYYY/mm/dd HH:MM:SS`
- 带毫秒版本

默认按 `UTC+8` 解释 CSV 时间，即 `--csv-utc-offset-hours 8.0`。

### 2.4 DAS CSV 输出格式

切片完成后的 DAS CSV 形如：

```csv
ch_18,ch_19,ch_20,...
182,173,182,...
120,120,134,...
...
```

说明：

- 每列对应一个通道。
- 每行对应一个时间采样点。
- 默认采样率是 `2000 Hz`。
- 默认会跳过前 `18` 个通道，因此列名通常从 `ch_18` 开始。

## 3. 环境准备

### 3.1 安装依赖

建议在虚拟环境中安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `librosa`
- `scikit-learn`
- `torch`
- `joblib`
- `soundfile`
- `nptdms`

如果你不使用虚拟环境，至少要保证运行这些脚本的 Python 解释器已经安装上述依赖。

### 3.2 Windows 终端编码问题

部分脚本打印了 Unicode 符号。在某些 Windows 终端中，如果默认编码不是 UTF-8，可能出现输出报错。可以先执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## 4. 最常用的标准流程

### 4.1 离线训练

最常用命令：

```powershell
python workflow_train.py wangdihai
```

这条命令会做以下事情：

1. 检查 `Data/Audio/wangdihai.mp3`
2. 检查 `Data/Airtag/wangdihai.csv`
3. 检查 `Data/DAS/`
4. 调用 `extract_name_signals_from_tdms.py` 从 TDMS 中切出 `output/wangdihai/signals/wangdihai.csv`
5. 计算实际裁剪范围
6. 调用 `WeaklySupervised_FootstepDetector.py` 训练模型
7. 保存模型到 `output/wangdihai/models/wangdihai_model.joblib`
8. 输出检测结果和图到 `output/wangdihai/results/`

### 4.2 离线推理

最常用命令：

```powershell
python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib
```

这条命令会：

1. 检查或生成 `output/wangjiahui/signals/wangjiahui.csv`
2. 加载 `.joblib` 模型
3. 调用 `WeaklySupervised_FootstepDetector.py` 进行推理
4. 将结果写到 `output/wangjiahui/results/`

### 4.3 在线推理模拟 + 实时显示

先开接收端：

```powershell
python JSONStreamPlot.py --protocol udp --port 9000
```

再开在线发送端：

```powershell
python realtime_infer_stream.py --name wangjiahui --model output/wangdihai/models/wangdihai_model.joblib
```

这条链路用于：

- 按 TDMS 分块模拟在线输入
- 做带固定延迟的在线推理
- 发送 `signal` 和 `event` JSON 包
- 在 `JSONStreamPlot.py` 中实时看信号和事件

### 4.4 离线结果按在线协议回放

如果你已经完成了离线推理，也可以直接把结果按在线协议发出去：

```powershell
python workflow_infer.py wangjiahui `
  --model output/wangdihai/models/wangdihai_model.joblib `
  --stream_after_infer `
  --host 127.0.0.1 `
  --port 9000
```

这个模式的用途是：

- 不重新做在线推理
- 直接把离线推理结果转成在线 `signal/event` JSON 进行可视化联调

## 5. 离线训练工作流：`workflow_train.py`

### 5.1 作用

`workflow_train.py` 是从 `name` 到模型的完整训练入口，适合日常使用。

当前脚本默认目录：

- 音频：`Data/Audio`
- Airtag：`Data/Airtag`
- TDMS：`Data/DAS`
- 输出：`output/<name>/`

### 5.2 常用命令

```powershell
python workflow_train.py wangdihai
python workflow_train.py wangdihai --trim_head 60 --trim_tail 30
python workflow_train.py wangdihai --skip_channels 0
python workflow_train.py wangdihai --model_type rf
python workflow_train.py wangdihai --model_type cnn --device cuda
python workflow_train.py wangdihai --dry_run
```

### 5.3 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `name` | 必填 | 对应 `Data/Audio/<name>.mp3` 与 `Data/Airtag/<name>.csv` |
| `--trim_head` | `50.0` | 从头裁掉多少秒 |
| `--trim_tail` | `20.0` | 从尾裁掉多少秒 |
| `--skip_channels` | `18` | 跳过前 N 个通道 |
| `--das_fs` | `2000` | DAS 采样率 |
| `--output_dir` | `output` | 输出根目录 |
| `--model_type` | `auto` | `auto/rf/cnn` |
| `--device` | `auto` | `auto/cuda/cpu` |
| `--self_train_rounds` | `0` | 自训练轮数 |
| `--confidence_threshold` | `0.45` | 高置信阈值 |
| `--torch_epochs` | `50` | CNN 训练轮数 |
| `--torch_batch_size` | `128` | CNN batch size |
| `--torch_lr` | `1e-4` | CNN 学习率 |
| `--das_filter_method` | `sosfilt` | DAS 带通滤波方法 |
| `--disable_das_bandpass` | 关闭 | 若开启则不做 DAS 带通 |
| `--skip_extract` | 关闭 | 跳过切片步骤 |
| `--overwrite` | 关闭 | 覆盖已有输出 |
| `--dry_run` | 关闭 | 只打印命令，不执行 |

### 5.4 训练产物

训练完成后通常会在：

```text
output/<name>/
├── signals/
│   └── <name>.csv
├── models/
│   └── <name>_model.joblib
└── results/
    ├── <name>_steps.csv
    ├── <name>_metrics.txt
    ├── <name>_heatmap_steps.png
    ├── <name>_detailed_segments.png
    ├── <name>_channel_trajectory.png
    ├── <name>_learned_pattern.png
    ├── <name>_heatmap_raw.png
    ├── <name>_heatmap_bp_*.png
    ├── <name>_comparison.png
    └── <name>_audio_envelope_window.png
```

其中：

- `*_steps.csv` 是最终脚步事件表。
- `*_metrics.txt` 是评估指标文本。
- `*_comparison.png` 和 `*_audio_envelope_window.png` 只有存在音频时才会生成。

## 6. 离线推理工作流：`workflow_infer.py`

### 6.1 作用

`workflow_infer.py` 是最常用的离线推理入口。  
它支持三种输入方式：

1. 只给 `name`，自动切 TDMS 再推理
2. 给 `name + --das_csv`，直接用已有 CSV 推理
3. 给 `--with_audio`，在纯推理之外额外生成音频对比图

### 6.2 常用命令

```powershell
python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib
python workflow_infer.py wangjiahui --model model.joblib --with_audio
python workflow_infer.py wangjiahui --model model.joblib --das_csv output/wangjiahui/signals/wangjiahui.csv
python workflow_infer.py wangjiahui --model model.joblib --dry_run
```

### 6.3 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `name` | 必填 | 对应 `Data/Airtag/<name>.csv` 的名字 |
| `--model`, `-m` | 必填 | 训练好的 `.joblib` 模型 |
| `--trim_head` | `50.0` | 从头裁掉多少秒 |
| `--trim_tail` | `20.0` | 从尾裁掉多少秒 |
| `--skip_channels` | `18` | 跳过前 N 个通道 |
| `--das_fs` | `2000` | DAS 采样率 |
| `--with_audio` | 关闭 | 推理时额外读取同名视频做音频对比；纯推理不需要它 |
| `--das_csv` | 空 | 直接指定已有 DAS CSV |
| `--output_dir` | `output/<name>` | 自定义输出根目录 |
| `--das_filter_method` | `sosfilt` | DAS 带通滤波方法 |
| `--disable_das_bandpass` | 关闭 | 关闭 DAS 带通 |
| `--skip_extract` | 关闭 | 跳过切片 |
| `--overwrite` | 关闭 | 覆盖已有结果 |
| `--dry_run` | 关闭 | 只打印命令 |

### 6.4 扩展参数

以下参数不是最小主线必需，但当前脚本仍支持：

| 参数 | 作用 |
| --- | --- |
| `--double` | 将原始信号与镜像信号叠加后推理 |
| `--mirror_only` | 仅用镜像信号推理 |
| `--channel_shift` | 推理前对通道整体平移 |
| `--stream_after_infer` | 推理后按在线协议发送结果 |
| `--protocol` / `--host` / `--port` | 回放目标设置 |
| `--signal_downsample` | 回放 `signal` 包降采样 |
| `--udp_max_samples` | UDP 单包最大样本数 |
| `--udp_max_bytes` | UDP 单包最大字节数 |
| `--replay_speed` | 回放速度，`1.0` 近实时，`0` 不等待 |

### 6.5 推理产物

推理结果通常位于：

```text
output/<name>/results/
├── <base>_steps_inference.csv
├── <base>_heatmap_inference.png
├── <base>_trajectory_inference.png
└── <base>_learned_pattern_inference.png
```

如果脚本内部走的不是纯推理分支，也可能看到 `<base>_steps.csv`。  
当前 `workflow_infer.py` 在回放离线结果时，会优先寻找：

1. `<base>_steps_inference.csv`
2. `<base>_steps.csv`

## 7. 核心脚本：`WeaklySupervised_FootstepDetector.py`

### 7.1 什么时候直接用它

当你已经有 DAS CSV，或者你想直接控制训练/推理参数时，可以绕过 workflow，直接调用核心脚本。

### 7.2 现在的内部实现方式

虽然你仍然可以像以前一样直接运行：

```powershell
python WeaklySupervised_FootstepDetector.py ...
```

但当前实现已经拆到 `wsfd/` 包里：

- `WeaklySupervised_FootstepDetector.py` 只是兼容入口
- `wsfd/cli.py` 负责解析参数和调用流程
- `wsfd/pipeline.py` 负责训练/推理主流程
- `wsfd/detector.py` 负责准备样本、训练模型和时间网格预测
- `wsfd/features.py` 负责特征提取、CNN patch 构造和通道估计
- `wsfd/models.py` 负责 CNN 模型本身

因此后续如果要理解或修改算法逻辑，优先看 `wsfd/`，不要只盯着根目录入口。

### 7.3 训练示例

```powershell
python WeaklySupervised_FootstepDetector.py `
  --das_csv output/wangdihai/signals/wangdihai.csv `
  --audio Data/Audio/wangdihai.mp3 `
  --trim_start 50 `
  --trim_end 180 `
  --output_dir output/wangdihai/results `
  --save_model output/wangdihai/models/wangdihai_model.joblib
```

### 7.4 仅推理示例

```powershell
python WeaklySupervised_FootstepDetector.py `
  --das_csv output/wangjiahui/signals/wangjiahui.csv `
  --load_model output/wangdihai/models/wangdihai_model.joblib `
  --inference_only `
  --output_dir output/wangjiahui/results
```

### 7.5 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--das_csv`, `-d` | 必填 | DAS CSV 路径 |
| `--audio`, `-a` | 空 | 音频或视频路径 |
| `--trim_start` | 空 | 起始时间裁剪 |
| `--trim_end` | 空 | 结束时间裁剪 |
| `--das_fs` | `2000` | DAS 采样率 |
| `--audio_sr` | `48000` | 音频重采样率 |
| `--das_filter_method` | `filtfilt` | 核心脚本默认滤波方式 |
| `--disable_das_bandpass` | 关闭 | 关闭 DAS 带通 |
| `--align_dt` | `0.0` | 音频与 DAS 对齐偏移 |
| `--model_type` | `auto` | `auto/rf/cnn` |
| `--self_train_rounds` | `0` | 自训练轮数 |
| `--confidence_threshold` | `0.35` | 高置信阈值 |
| `--device` | `auto` | `auto/cuda/cpu` |
| `--torch_epochs` | `50` | CNN 训练轮数 |
| `--torch_batch_size` | `64` | CNN batch size |
| `--torch_lr` | `1e-3` | CNN 学习率 |
| `--torch_hidden_dim` | `64` | CNN 隐层维度 |
| `--torch_dropout` | `0.1` | dropout |
| `--cnn_window_s` | `0.12` | CNN 输入窗口秒数 |
| `--cnn_predict_chunk` | `256` | CNN 预测分块大小 |
| `--output_dir`, `-o` | `output/weakly_supervised` | 输出目录 |
| `--save_model` | 空 | 训练后保存模型 |
| `--load_model` | 空 | 加载模型 |
| `--inference_only` | 关闭 | 仅推理；必须同时给 `--load_model` |

### 7.6 一个重要区别

`workflow_train.py` 和 `WeaklySupervised_FootstepDetector.py` 的默认值并不完全一致。  
例如：

- `workflow_train.py` 默认 `confidence_threshold=0.45`
- `WeaklySupervised_FootstepDetector.py` 默认 `confidence_threshold=0.35`
- `workflow_train.py` 默认 `torch_batch_size=128`
- `WeaklySupervised_FootstepDetector.py` 默认 `torch_batch_size=64`
- `workflow_train.py` 默认 `torch_lr=1e-4`
- `WeaklySupervised_FootstepDetector.py` 默认 `torch_lr=1e-3`

因此：

- 想要和日常 workflow 保持一致时，优先使用 workflow。
- 想做精细控制时，再直接调用核心脚本。

### 7.7 当前模型到底在学什么

这是当前项目最容易误解的地方。

当前系统并不是直接学习 `(time, channel)` 二维定位，而是分成两步：

1. 模型先学习“这个时间窗里有没有脚步”
2. 检测到脚步时间后，再根据该时间附近的 DAS 能量分布估计通道

对应实现：

- 弱标签时间来自 [wsfd/audio_labels.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/audio_labels.py)
- 训练样本准备在 [wsfd/detector.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/detector.py)
- 时间网格预测在 [wsfd/detector.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/detector.py)
- 通道估计在 [wsfd/features.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/features.py)

更具体地说：

- 音频先提供一组脚步候选时间 `audio_step_times`
- 这些时间用于构造 DAS 正负样本窗口
- 模型输出的是每个时间点的脚步概率曲线
- 检测出脚步时间后，再调用 `estimate_channel_for_time()` 估计通道

所以当前系统的监督目标是：

- `这个时间窗是否包含脚步`

而不是：

- `这个脚步发生在哪个通道`

### 7.8 RF 和 CNN 的区别

RF 和 CNN 现在都主要服务于“时间检测”，但它们看到的输入不同。

RF 分支在 [wsfd/features.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/features.py) 中使用汇总统计特征，包括：

- 总能量
- 最大通道能量
- 平均能量
- 能量标准差
- 最大能量通道
- 最大振幅
- 平均波动

因此 RF 更接近：

- 基于多频带统计特征判断这个窗口像不像脚步

CNN 分支则不使用这些汇总统计量，而是直接读取原始 patch。  
在 [wsfd/features.py](C:/Users/boshu/Desktop/srt/WeaklySupervisedFootstep/wsfd/features.py) 中，CNN 输入的形状是：

- `[N, B, W, C]`

其中：

- `B` 是频带数
- `W` 是时间窗长度
- `C` 是通道数

这意味着 CNN 输入是一块完整的：

- 频带 × 时间 × 通道

时空 patch，而不是单个总能量值。

### 7.9 为什么 CNN 学了时空结构，却仍然主要输出时间

原因不是“学到的空间信息丢了”，而是训练目标没有要求它输出通道。

当前 CNN 的损失函数只在乎：

- 这个窗口是不是脚步窗口

所以它虽然会利用时空结构来提高判断能力，但最终输出头仍然是二分类概率。  
这可以理解为：

- CNN 用时间-通道二维振动结构来做时间检测
- 但并没有被监督成“直接输出通道位置”

因此现在的系统是：

- CNN 学时空纹理
- 输出脚步时间概率
- 通道位置再靠后处理估计

如果后续研究要进一步往“端到端定位”走，才需要把通道预测也纳入模型监督目标。

## 8. TDMS 切片工具：`extract_name_signals_from_tdms.py`

### 8.1 作用

该脚本根据 Airtag CSV 里的起止时间，自动定位重叠的 TDMS 文件，并把对应时间段的 DAS 数据切成单个名字的 CSV。

### 8.2 常用命令

按视频或音频目录里的名字批量处理：

```powershell
python extract_name_signals_from_tdms.py `
  --video-dir Data/Video `
  --airtag-csv-dir Data/Airtag `
  --tdms-dir Data/DAS `
  --output-dir output/name_signals
```

只处理某个人：

```powershell
python extract_name_signals_from_tdms.py `
  --video-dir Data/Video `
  --airtag-csv-dir Data/Airtag `
  --tdms-dir Data/DAS `
  --output-dir output/name_signals `
  --name wangdihai
```

没有视频也可仅按 Airtag 文件名处理：

```powershell
python extract_name_signals_from_tdms.py `
  --video-dir Data/Audio `
  --airtag-csv-dir Data/Airtag `
  --tdms-dir Data/DAS `
  --output-dir output/name_signals `
  --use-airtag-only
```

### 8.3 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--video-dir` | 必填 | 用于提供名字列表的目录 |
| `--airtag-csv-dir` | 必填 | Airtag CSV 目录 |
| `--tdms-dir` | 必填 | TDMS 目录 |
| `--output-dir` | `output/name_signals` | 输出目录 |
| `--fs` | `2000.0` | DAS 采样率 |
| `--csv-utc-offset-hours` | `8.0` | CSV 时区偏移 |
| `--name` | 空 | 只处理指定名字，可重复传入 |
| `--encoding` | `utf-8-sig` | Airtag CSV 编码 |
| `--overwrite` | 关闭 | 覆盖已有 CSV |
| `--dry-run` | 关闭 | 只打印匹配，不写文件 |
| `--use-airtag-only` | 关闭 | 不依赖视频文件名 |
| `--skip-channels` | `18` | 跳过前 N 个通道 |

### 8.4 输出命名

默认输出：

```text
output/name_signals/<name>.csv
```

而 workflow 会把输出放进：

```text
output/<name>/signals/<name>.csv
```

## 9. 在线推理模拟：`realtime_infer_stream.py`

### 9.1 作用

该脚本从 TDMS 原始数据中按块读取信号，做在线滤波和在线推理，并通过 UDP/TCP 发送 JSON 数据流。

它的目标不是离线批处理，而是模拟“边到数据边检测边显示”的链路。

### 9.2 使用示例

```powershell
python realtime_infer_stream.py `
  --name wangjiahui `
  --model output/wangdihai/models/wangdihai_model.joblib `
  --protocol udp `
  --host 127.0.0.1 `
  --port 9000
```

### 9.3 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--name` | 必填 | 对应 Airtag CSV 文件名 |
| `--model`, `-m` | 必填 | 已训练模型路径 |
| `--airtag-csv-dir` | `Data/Airtag` | Airtag 目录 |
| `--tdms-dir` | `Data/DAS` | TDMS 目录 |
| `--das-fs` | `2000.0` | DAS 采样率 |
| `--skip-channels` | `18` | 跳过前 N 个通道 |
| `--trim-head` | `50.0` | 头部裁掉秒数 |
| `--trim-tail` | `20.0` | 尾部裁掉秒数 |
| `--csv-utc-offset-hours` | `8.0` | CSV 时区偏移 |
| `--chunk-seconds` | `1.0` | 每次读取的块长度 |
| `--time-step` | `0.03` | 推理时间步长 |
| `--buffer-seconds` | `10.0` | 环形缓冲区长度 |
| `--latency-seconds` | `1.0` | 事件输出延迟 |
| `--detrend-alpha` | `0.001` | 在线去趋势系数 |
| `--speed` | `1.0` | 仿真速度，`0` 为不等待 |
| `--max-seconds` | 空 | 最长回放时间 |
| `--protocol` | `udp` | 网络协议 |
| `--host` | `127.0.0.1` | 目标主机 |
| `--port` | `9000` | 目标端口 |
| `--signal-downsample` | `1` | `signal` 包降采样 |
| `--udp-max-samples` | `10` | UDP 单包最大样本数 |
| `--udp-max-bytes` | `60000` | UDP 单包最大字节数 |

### 9.4 适用模型

这个脚本走的是在线 CNN 推理思路，实际使用时建议传入由 CNN 路线得到的 `.joblib` 模型。

## 10. 在线可视化：`JSONStreamPlot.py`

### 10.1 作用

`JSONStreamPlot.py` 是在线接收端。它监听 UDP 或 TCP 端口，接收：

- `signal` 包：时间序列信号
- `event` 包：脚步事件

然后以实时曲线、热图和轨迹的方式显示。

### 10.2 基本使用

```powershell
python JSONStreamPlot.py --protocol udp --host 0.0.0.0 --port 9000
```

### 10.3 常用显示参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--window-seconds` | `5.0` | 显示窗口长度 |
| `--max-channels` | `0` | 最多显示多少通道，`0` 为全部 |
| `--refresh-hz` | `20.0` | 刷新率 |
| `--sample-rate` | 空 | 当包里不带采样率时的兜底值 |
| `--max-events` | `2000` | 内存中保留的事件数 |
| `--show-events` / `--hide-events` | 默认显示 | 是否显示事件点 |
| `--show-heatmap` | 默认关闭 | 是否显示热图 |

### 10.4 轨迹相关参数

如果你要把事件映射到一条空间轨迹上显示，可以用：

- `--trajectory-xz`
- `--trajectory-channels`
- `--channel-offset`
- `--channel-tail-trim`
- `--reverse-channel-direction`
- `--no-reverse-channel-direction`
- `--trajectory-fade-seconds`
- `--trajectory-lost-timeout`
- `--trajectory-position-alpha`
- `--trajectory-smooth-speed`
- `--trajectory-max-association-distance`
- `--trajectory-max-velocity`
- `--trajectory-velocity-alpha`
- `--trajectory-point-size`
- `--trajectory-glow-size`

默认已经内置了一组 `XZ` 折线控制点，适合当前项目里已有的轨迹显示逻辑。

## 11. 在线 JSON 协议

### 11.1 `signal` 包

在线发送端会发送如下结构的信号包：

```json
{
  "packet_type": "signal",
  "timestamp": 123.45,
  "sample_rate": 2000.0,
  "sample_count": 10,
  "total_channels": 174,
  "signals": [[...], [...]]
}
```

字段含义：

- `packet_type`: 固定为 `signal`
- `timestamp`: 该块对应的时间戳
- `sample_rate`: 采样率
- `sample_count`: 这个包里包含的样本数
- `total_channels`: 总通道数
- `signals`: 信号矩阵

### 11.2 `event` 包

事件包结构：

```json
{
  "packet_type": "event",
  "timestamp": 123.78,
  "channel_index": 96,
  "confidence": 0.82
}
```

字段含义：

- `packet_type`: 固定为 `event`
- `timestamp`: 事件时间
- `channel_index`: 检测到的通道
- `confidence`: 置信度

这套协议目前由两条链路共享：

1. `realtime_infer_stream.py` 在线模拟发送
2. `workflow_infer.py --stream_after_infer` 离线结果回放发送

## 12. 输出文件说明

### 12.1 训练阶段常见输出

| 文件 | 说明 |
| --- | --- |
| `*_steps.csv` | 最终脚步结果 |
| `*_metrics.txt` | 指标文本 |
| `*_heatmap_steps.png` | 总体热图 + 事件 |
| `*_detailed_segments.png` | 多段细看图 |
| `*_channel_trajectory.png` | 通道轨迹图 |
| `*_learned_pattern.png` | 学到的模式 |
| `*_heatmap_raw.png` | 原始热图 |
| `*_heatmap_bp_<band>.png` | 分频带热图 |
| `*_comparison.png` | 音频与 DAS 对比 |
| `*_audio_envelope_window.png` | 音频包络窗口图 |

### 12.2 推理阶段常见输出

| 文件 | 说明 |
| --- | --- |
| `*_steps_inference.csv` | 纯推理结果 |
| `*_heatmap_inference.png` | 推理热图 |
| `*_trajectory_inference.png` | 推理轨迹图 |
| `*_learned_pattern_inference.png` | 推理模式图 |

### 12.3 `*_steps.csv` 结果字段

结果 CSV 一般至少包含：

- 时间
- 通道
- 置信度

不同训练/推理路径下列名可能略有差别，但核心含义一致：在某个时间点、某个通道，模型认为存在脚步事件，并给出一个置信度分数。

## 13. 3D-CNN 实验脚本

当前 3D-CNN 方向位于：

```text
experiment/WeaklySupervised_FootstepDetector_3D_CNN.py
```

它是实验脚本，不是当前主线入口，但仍然是仓库中保留的研究方向。

### 13.1 主要能力

- 支持 `--name` 自动切 CSV
- 支持 `--das_csv` 直接输入已有 CSV
- 支持训练、保存模型、加载模型、仅推理
- 默认输出到 `output/3d_cnn/<tag>/`

### 13.2 关键参数

| 参数 | 默认值 |
| --- | --- |
| `--bands` | `5-10,10-20,20-50,50-100` |
| `--patch_frames` | `31` |
| `--epochs` | `24` |
| `--batch_size` | `96` |
| `--lr` | `1e-3` |
| `--score_threshold` | `0.40` |
| `--peak_time_dist_s` | `0.20` |
| `--peak_channel_dist` | `2` |

### 13.3 常见输出

- `<base>_steps_3dcnn.csv`
- `<base>_heatmap_3dcnn.png`
- `<base>_3dcnn.pt`

## 14. 示例与历史文件

### 14.1 示例输出

仓库中已经附带了一些输出示例，可参考：

```text
examples/output/
```

其中包括：

- `wangdihai_heatmap_steps.png`
- `wangdihai_channel_trajectory.png`
- `wangdihai_comparison.png`
- `wangdihai_steps.csv`

### 14.2 历史和废弃内容

以下脚本不建议作为当前项目的主要入口：

- `deprecated/workflow_train_multi.py`
- `deprecated/audio_step_tuning.py`
- `deprecated/batch_audio_spectrogram.py`
- `deprecated/make_trajectory_animation.py`
- `run_footstep_detection.py`

如果后人接手项目，应优先看本 README 里的主线，不要先从这些文件入手。

## 15. 常见问题

### 15.1 `ModuleNotFoundError: No module named 'numpy'`

说明当前 Python 环境没有装依赖。先执行：

```powershell
python -m pip install -r requirements.txt
```

或者使用虚拟环境里的解释器。

### 15.2 `workflow_train.py` 说找不到音频

当前训练工作流固定检查：

```text
Data/Audio/<name>.mp3
```

如果你的训练音频不是 `mp3`，需要先转成对应文件名的 `mp3`，或者修改脚本逻辑。

### 15.3 `--skip_extract` 什么时候有用

当你已经有：

```text
output/<name>/signals/<name>.csv
```

或者你通过 `--das_csv` 明确提供了 CSV 时，它可以避免再次从 TDMS 切片。

### 15.4 workflow 和核心脚本参数为什么对不上

这是当前仓库的真实状态，不是 README 写错。workflow 是“日常使用入口”，核心脚本是“底层实现入口”，二者默认值不完全一致。  
如果你的目标是复现实验记录，优先使用当时实际调用的入口脚本。

## 16. 推荐给后人的最短上手路径

如果是第一次接手本项目，建议只按下面顺序理解：

1. 先看本 README。
2. 确认 `Data/Audio`、`Data/Airtag`、`Data/DAS` 的命名是否一致；`Data/Video` 只在需要音频对比时再检查。
3. 跑 `python workflow_train.py <name> --dry_run` 看流程是否能串起来。
4. 再跑正式训练。
5. 用 `python workflow_infer.py <name> --model ...` 做离线推理。
6. 最后再看 `realtime_infer_stream.py + JSONStreamPlot.py` 的在线链路。

这样理解成本最低，也最接近这个仓库当前仍在使用的方式。
