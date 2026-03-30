#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流脚本：标签点最高能量通道映射（免训练）
========================================

功能：输入名字 -> 复用TDMS切分逻辑 -> 从音频提取时间标签 ->
     对每个标签时刻取DAS最高能量通道 -> 输出CSV和可视化图。

使用示例：
    python workflow_label_peak_channel.py wangdihai
    python workflow_label_peak_channel.py wangdihai --trim_head 50 --trim_tail 20
    python workflow_label_peak_channel.py wangdihai --skip_extract
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wsfd.audio_labels import AudioWeakLabelExtractor
from wsfd.config import Config
from wsfd.features import DASFeatureExtractor
from wsfd.visualization import Visualizer


# ============================================================================
# 默认配置
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "Data"
AUDIO_DIR = DATA_DIR / "Audio"
AIRTAG_DIR = DATA_DIR / "Airtag"
DAS_DIR = DATA_DIR / "DAS"

OUTPUT_BASE_DIR = SCRIPT_DIR / "output"

DEFAULT_DAS_FS = 2000
DEFAULT_UTC_OFFSET = 8.0
DEFAULT_TRIM_HEAD = 50.0
DEFAULT_TRIM_TAIL = 20.0
DEFAULT_SKIP_CHANNELS = 18


def parse_args():
    parser = argparse.ArgumentParser(
        description="标签通道工作流：直接将音频标签映射到DAS最高能量通道（不训练模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("name", help="目标名字（对应Audio中的mp3文件名，不含扩展名）")

    parser.add_argument("--trim_head", type=float, default=DEFAULT_TRIM_HEAD,
                        help=f"从数据开头裁掉的时长（秒），默认 {DEFAULT_TRIM_HEAD}")
    parser.add_argument("--trim_tail", type=float, default=DEFAULT_TRIM_TAIL,
                        help=f"从数据结尾裁掉的时长（秒），默认 {DEFAULT_TRIM_TAIL}")
    parser.add_argument("--skip_channels", type=int, default=DEFAULT_SKIP_CHANNELS,
                        help=f"跳过前N个通道（默认 {DEFAULT_SKIP_CHANNELS}），设为0保留所有通道")
    parser.add_argument("--das_fs", type=int, default=DEFAULT_DAS_FS,
                        help=f"DAS采样率 (Hz)，默认 {DEFAULT_DAS_FS}")

    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"输出根目录（默认 {OUTPUT_BASE_DIR}）")
    parser.add_argument("--das_filter_method", type=str, default="sosfilt",
                        choices=["filtfilt", "sosfilt"],
                        help="DAS带通滤波方法，默认 sosfilt")
    parser.add_argument("--disable_das_bandpass", action="store_true",
                        help="关闭DAS带通滤波，直接用原始DAS信号取最大能量通道")
    parser.add_argument("--label_window_s", type=float, default=0.15,
                        help="标签时刻能量统计窗口（秒），默认 0.15")

    parser.add_argument("--skip_extract", action="store_true",
                        help="跳过信号提取步骤（如果CSV已存在）")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出文件")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印将要执行的命令，不实际运行")

    return parser.parse_args()


def check_audio_exists(name: str) -> Path:
    audio_path = AUDIO_DIR / f"{name}.mp3"
    if audio_path.exists():
        return audio_path
    raise FileNotFoundError(f"找不到音频文件: {AUDIO_DIR / name}.mp3")


def check_airtag_exists(name: str) -> Path:
    csv_path = AIRTAG_DIR / f"{name}.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"找不到Airtag CSV: {csv_path}")


def run_command(cmd: list, description: str, dry_run: bool = False) -> int:
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")

    if dry_run:
        print("[DRY RUN] 跳过实际执行")
        return 0

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"[ERROR] 命令失败，返回码: {result.returncode}")
    return result.returncode


def compute_peak_channel(das_signal: np.ndarray, t: float, fs: int, window_s: float):
    t_idx = int(t * fs)
    half_win = int(window_s * fs / 2)

    start = max(0, t_idx - half_win)
    end = min(das_signal.shape[0], t_idx + half_win)
    if end <= start:
        return 0, 0.0

    window = das_signal[start:end, :]
    channel_energy = np.sum(window ** 2, axis=0)
    ch = int(np.argmax(channel_energy))
    energy = float(channel_energy[ch])
    return ch, energy


def main():
    args = parse_args()
    name = args.name.lower()

    if args.output_dir:
        output_base_dir = Path(args.output_dir)
        if not output_base_dir.is_absolute():
            output_base_dir = SCRIPT_DIR / output_base_dir
    else:
        output_base_dir = OUTPUT_BASE_DIR

    print("=" * 60)
    print("标签点最高能量通道工作流（免训练）")
    print("=" * 60)
    print(f"目标名字: {name}")
    print(f"时间裁剪: 头部 -{args.trim_head}s, 尾部 -{args.trim_tail}s")
    print(f"跳过通道: 前 {args.skip_channels} 个")
    print(f"DAS采样率: {args.das_fs} Hz")

    print("\n[CHECK] 检查输入文件...")
    try:
        audio_path = check_audio_exists(name)
        print(f"  ✓ 音频: {audio_path}")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return 1

    try:
        airtag_path = check_airtag_exists(name)
        print(f"  ✓ Airtag: {airtag_path}")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return 1

    if not DAS_DIR.exists():
        print(f"  ✗ DAS目录不存在: {DAS_DIR}")
        return 1
    print(f"  ✓ DAS目录: {DAS_DIR}")

    output_dir = output_base_dir / name
    csv_output_dir = output_dir / "signals"
    results_output_dir = output_dir / "label_peak"
    das_csv_path = csv_output_dir / f"{name}.csv"

    print(f"\n[OUTPUT] 输出根目录: {output_base_dir}")
    print(f"[OUTPUT] 输出目录: {output_dir}")

    if das_csv_path.exists() and not args.overwrite:
        print(f"\n[SKIP] 信号CSV已存在，跳过提取: {das_csv_path}")
        print("       （如需重新提取，请使用 --overwrite）")
    elif args.skip_extract and das_csv_path.exists():
        print(f"\n[SKIP] 跳过信号提取（CSV已存在: {das_csv_path}）")
    else:
        extract_cmd = [
            sys.executable, str(SCRIPT_DIR / "extract_name_signals_from_tdms.py"),
            "--video-dir", str(AUDIO_DIR),
            "--airtag-csv-dir", str(AIRTAG_DIR),
            "--tdms-dir", str(DAS_DIR),
            "--output-dir", str(csv_output_dir),
            "--fs", str(args.das_fs),
            "--csv-utc-offset-hours", str(DEFAULT_UTC_OFFSET),
            "--name", name,
            "--skip-channels", str(args.skip_channels),
            "--use-airtag-only",
        ]
        if args.overwrite:
            extract_cmd.append("--overwrite")

        ret = run_command(extract_cmd, "从TDMS提取DAS信号", args.dry_run)
        if ret != 0:
            print("[ABORT] 信号提取失败")
            return ret

    if not args.dry_run and not das_csv_path.exists():
        print(f"[ERROR] 信号CSV未生成: {das_csv_path}")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] 仅执行到提取阶段检查，后续分析已跳过")
        return 0

    n_samples = sum(1 for _ in open(das_csv_path, encoding="utf-8")) - 1
    total_duration = n_samples / args.das_fs

    trim_start = args.trim_head
    trim_end = total_duration - args.trim_tail
    if trim_end <= trim_start:
        print(
            f"[ERROR] 裁剪后无有效数据: 总时长={total_duration:.1f}s, "
            f"头部裁剪={args.trim_head}s, 尾部裁剪={args.trim_tail}s"
        )
        return 1

    effective_duration = trim_end - trim_start
    print(f"\n[TIME] 数据总时长: {total_duration:.1f}s")
    print(f"       裁剪后范围: {trim_start:.1f}s - {trim_end:.1f}s (有效 {effective_duration:.1f}s)")

    results_output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    config.das_fs = args.das_fs
    config.trim_start_s = trim_start
    config.trim_end_s = trim_end
    config.das_filter_method = args.das_filter_method
    config.disable_das_bandpass = bool(args.disable_das_bandpass)
    config.output_dir = str(results_output_dir)

    print("\n" + "=" * 60)
    print("[STEP] 标签映射到最高能量通道")
    print("=" * 60)

    das_extractor = DASFeatureExtractor(config)
    das_raw, _ = das_extractor.load_das_csv(str(das_csv_path))
    das_trimmed = das_extractor.trim_data(das_raw, trim_start, trim_end)

    das_bands = das_extractor.multi_band_filter(das_trimmed)
    primary_band = tuple(config.das_bp_bands[0])
    das_primary = das_bands[primary_band]

    energy_matrix, frame_times = das_extractor.compute_short_time_energy(das_primary)

    audio_extractor = AudioWeakLabelExtractor(config)
    audio_result = audio_extractor.process_audio(
        str(audio_path),
        trim_start=trim_start,
        trim_end=trim_end,
    )
    label_times = np.asarray(audio_result["step_times"], dtype=np.float64)

    if len(label_times) == 0:
        print("[ERROR] 音频未提取到任何时间标签，无法映射通道")
        return 1

    rows = []
    energies = []
    for t in label_times:
        if t < 0 or t >= effective_duration:
            continue
        ch, e = compute_peak_channel(das_primary, float(t), args.das_fs, args.label_window_s)
        rows.append((float(t), int(ch), float(e)))
        energies.append(float(e))

    if not rows:
        print("[ERROR] 标签均超出有效时间范围，无法输出")
        return 1

    energies_np = np.asarray(energies, dtype=np.float64)
    e_min = float(np.min(energies_np))
    e_max = float(np.max(energies_np))
    e_span = max(1e-12, e_max - e_min)

    # 可视化颜色使用0.3~1.0范围，和现有轨迹图的置信度范围一致
    step_events = []
    for t, ch, e in rows:
        conf = 0.3 + 0.7 * ((e - e_min) / e_span)
        step_events.append((t, ch, conf))

    df_out = pd.DataFrame(rows, columns=["time", "channel", "peak_energy"])
    df_out = df_out.sort_values("time").reset_index(drop=True)

    labels_csv = results_output_dir / f"{name}_label_peak_channels.csv"
    df_out.to_csv(labels_csv, index=False)

    viz = Visualizer(config)
    heatmap_path = results_output_dir / f"{name}_label_peak_heatmap.png"
    trajectory_path = results_output_dir / f"{name}_label_peak_trajectory.png"

    viz.plot_energy_heatmap_with_steps(
        energy_matrix,
        frame_times,
        step_events,
        str(heatmap_path),
        title=f"Label Peak Channels: {name}",
    )
    viz.plot_channel_trajectory(step_events, str(trajectory_path))

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"标签数量: {len(df_out)}")
    print(f"标签通道CSV: {labels_csv}")
    print(f"热图: {heatmap_path}")
    print(f"轨迹图: {trajectory_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
