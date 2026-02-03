# 快速入门指南

## 5分钟上手

### 第1步：安装依赖

```bash
cd WeaklySupervisedFootstep
pip install -r requirements.txt
```

### 第2步：验证音频检测（推荐先做）

```bash
python audio_step_tuning.py \
    --audio "你的视频.mp4" \
    --trim_start 50 \
    --trim_end 200 \
    --output_dir "output/tuning"
```

查看生成的图片，确认：

- ✅ 脚步检测位置正确
- ✅ 步频合理（正常步行约80-120步/分钟）
- ✅ 无明显误检

### 第3步：运行弱监督检测

```bash
python WeaklySupervised_FootstepDetector.py \
    --das_csv "你的DAS数据.csv" \
    --audio "你的视频.mp4" \
    --trim_start 50 \
    --trim_end 200 \
    --output_dir "output/results"
```

### 第4步：查看结果

- `*_steps.csv` - 脚步事件列表 (time, channel, confidence)
- `*_heatmap_steps.png` - 可视化热图

---

## 常用命令速查

### 调整音频检测灵敏度

```bash
# 更灵敏（检测更多脚步）
python audio_step_tuning.py --audio video.mp4 --peak_height 0.5 --peak_prom 1.0

# 更严格（减少误检）
python audio_step_tuning.py --audio video.mp4 --peak_height 1.2 --peak_prom 2.0
```

### 调整弱监督参数

```bash
# 更多迭代（找到更多脚步）
python WeaklySupervised_FootstepDetector.py ... --self_train_rounds 5

# 更高置信阈值（减少噪声）
python WeaklySupervised_FootstepDetector.py ... --confidence_threshold 0.8
```

### 保存和加载模型 🆕

```bash
# 训练并保存模型
python WeaklySupervised_FootstepDetector.py \
    --das_csv wangdihai.csv \
    --audio video.mp4 \
    --trim_start 50 --trim_end 200 \
    --save_model "models/wangdihai_model.joblib"

# 使用已训练模型处理其他数据（无需音频！）
python WeaklySupervised_FootstepDetector.py \
    --das_csv wangjiahui.csv \
    --trim_start 50 --trim_end 200 \
    --load_model "models/wangdihai_model.joblib" \
    --inference_only

# 加载模型同时提供音频（用于对比验证）
python WeaklySupervised_FootstepDetector.py \
    --das_csv wangjiahui.csv \
    --audio video_jiahui.mp4 \
    --load_model "models/wangdihai_model.joblib"
```

**模型迁移适用场景**：

- ✅ 相同布设、相同环境
- ✅ 相似的行走方式
- ⚠️ 不同布设可能需要重新训练

---

## 🚀 一键工作流（推荐）

### 训练工作流

```bash
# 输入名字，自动完成：TDMS切分 → 训练 → 保存模型
python workflow_train.py wangdihai

# 指定时间范围
python workflow_train.py wangdihai --trim_start 50 --trim_end 200

# 查看帮助
python workflow_train.py --help
```

输出结构：

```
output/wangdihai/
├── signals/wangdihai.csv      # 提取的DAS信号
├── models/wangdihai_model.joblib  # 训练好的模型
└── results/                   # 检测结果和可视化
```

### 推理工作流

```bash
# 使用wangdihai的模型检测wangjiahui的脚步
python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib

# 带音频对比验证
python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib --with_audio

# 查看帮助
python workflow_infer.py --help
```

---

## 输出文件速查

| 文件                       | 内容     | 最重要的信息              |
| -------------------------- | -------- | ------------------------- |
| `*_steps.csv`              | 检测结果 | time, channel, confidence |
| `*_heatmap_steps.png`      | 主热图   | 脚步在时间×通道上的分布   |
| `*_channel_trajectory.png` | 轨迹图   | 行走轨迹和步频            |
| `*_comparison.png`         | 对比图   | 音频vs DAS检测效果        |

---

## 遇到问题？

1. **检测太少**：降低 `peak_height` 和 `peak_prom`
2. **误检太多**：提高 `peak_height` 和 `peak_prom`
3. **时间不对齐**：使用 `--align_dt` 调整偏移
4. **模型迁移效果差**：环境变化大时需重新训练
5. **更多帮助**：查看 `README.md` 详细文档
