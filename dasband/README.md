# DASBand

`dasband/` 是一套独立于旧仓库流程的弱监督 DAS 时空响应带建模实现。

核心思想：

1. 从候选脚步点 `(t_k, c_k)` 出发。
2. 用两段鲁棒直线拟合做轨迹级清洗。
3. 将清洗后的轨迹插值成连续中心线。
4. 生成带状伪标签 `T(t,c)`。
5. 训练 2D mask 分割模型预测 `M(t,c)`。
6. 用质心和动态规划从 `M(t,c)` 解码主轨迹，并输出不确定度。

## 目录

- 根目录只保留 `main.py` 和 `README.md`
- `scripts/` 下包含所有实现模块和子入口
- `scripts/prepare_labels.py`: 生成候选点清洗结果和伪带标签
- `scripts/train.py`: 从 `prepare_labels` 产物训练 2D mask 模型
- `scripts/infer.py`: 预测 mask，并解码主轨迹与宽度

## 最小使用方式

### 0. 主入口

```bash
python -m dasband.main wangdihai
```

默认流程：

1. 从 `Data/Airtag/wangdihai.csv` 读取时间范围
2. 从 `Data/DAS/*.tdms` 裁剪并输出 `output/wangdihai/signals/wangdihai.csv`
3. 生成伪标签
4. 训练 `dasband_model.pt`
5. 对同一个样本输出 mask、主轨迹和不确定度

常用调参示例：

```bash
python -m dasband.main wangdihai \
  --trim_start_s 40 \
  --trim_end_s 210 \
  --gaussian_sigma_ch 2.5 \
  --clean_outlier_threshold_ch 4 \
  --epochs 50 \
  --learning_rate 5e-4 \
  --model_channels 48
```

如果只想提取+准备伪标签：

```bash
python -m dasband.main wangdihai --stop_after_prepare
```

如果已有 checkpoint，只做提取+推理：

```bash
python -m dasband.main wangjiahui \
  --checkpoint output/wangdihai/dasband/train/dasband_model.pt
```

### 1. 准备伪标签

```bash
python -m dasband.scripts.prepare_labels \
  --das_csv output/wangdihai/signals/wangdihai.csv \
  --candidate_csv output/wangdihai/label_peak/wangdihai_label_peak_channels.csv \
  --name wangdihai
```

如果没有候选点 CSV，也可以让它从音频自动提时间点，再映射峰值通道：

```bash
python -m dasband.scripts.prepare_labels \
  --das_csv output/wangdihai/signals/wangdihai.csv \
  --audio_path Data/Audio/wangdihai.mp3 \
  --name wangdihai
```

### 2. 训练

```bash
python -m dasband.scripts.train \
  --prep_dir output/wangdihai/dasband/prepare \
  --name wangdihai
```

### 3. 推理

```bash
python -m dasband.scripts.infer \
  --das_csv output/wangjiahui/signals/wangjiahui.csv \
  --checkpoint output/wangdihai/dasband/train/dasband_model.pt \
  --name wangjiahui
```

## 主要输出

- `candidate_points_raw.csv`
- `candidate_points_clean.csv`
- `pseudo_label.npy`
- `feature_cube.npy`
- `dasband_model.pt`
- `pred_mask.npy`
- `track.csv`
- `candidate_cleaning.png`
- `pseudo_label.png`
- `inference_result.png`

## 说明

- `main.py` 支持从 `Data/DAS/*.tdms` 直接提取到 CSV；子脚本默认输入仍是 DAS CSV。
- `dasband/` 不依赖旧 `wsfd/` 包，可以整体搬走作为新仓库基础。
- 建议先用 `prepare_labels` 看清洗结果与伪标签是否合理，再开始训练。
