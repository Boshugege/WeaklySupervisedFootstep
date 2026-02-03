#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版弱监督脚步检测 - 快速使用脚本
=====================================

用法示例:
    python run_footstep_detection.py wangdihai

这会自动找到对应的DAS CSV和视频文件，运行检测并生成结果。
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from WeaklySupervised_FootstepDetector import Config, run_pipeline

# ============================================================================
# 配置区域 - 根据你的数据修改以下参数
# ============================================================================

# 数据路径配置
DAS_CSV_DIR = r"C:\Users\boshu\Desktop\srt\DAS2Steps\output\b202_name_signals"
VIDEO_DIR = r"C:\Users\boshu\Desktop\srt\B202_Video"
OUTPUT_BASE_DIR = r"C:\Users\boshu\Desktop\srt\DAS2Steps\output\weakly_supervised"

# ============ 时间裁剪配置（核心参数） ============
# 用于去除视频/信号开头结尾的废弃部分
# 设置为 None 表示不裁剪

TRIM_CONFIG = {
    "wangdihai": {
        "trim_start": 50.0,     # 开始时间（秒）- 从50秒开始
        "trim_end": 200.0,      # 结束时间（秒）- 到3分20秒
        "align_dt": 0.0,        # 时间对齐偏移（DAS时间 = 音频时间 + align_dt）
    },
    "wangjiahui": {
        "trim_start": 5.0,
        "trim_end": None,
        "align_dt": 0.0,
    },
    "wuwenxuan": {
        "trim_start": 5.0,
        "trim_end": None,
        "align_dt": 0.0,
    },
    # 默认配置
    "default": {
        "trim_start": 5.0,
        "trim_end": None,
        "align_dt": 0.0,
    }
}

# ============ 检测参数配置 ============
DETECTION_CONFIG = {
    # DAS滤波配置
    "das_fs": 2000,                     # DAS采样率
    "das_bp_bands": [                   # 多频带滤波
        (5, 10),                        # 主频带（你指定的5-10Hz）
        (2, 5),                         # 低频辅助
        (10, 20),                       # 高频辅助
    ],
    
    # 音频滤波配置  
    "audio_bp_low": 4000,               # 音频带通低频（你指定的4kHz）
    "audio_bp_high": 10000,             # 音频带通高频（你指定的10kHz）
    
    # 脚步检测参数
    "step_min_interval": 0.30,          # 最小脚步间隔（秒）
    "weak_label_sigma": 0.15,           # 弱标签时间扩展sigma
    
    # 模型参数
    "self_train_rounds": 3,             # 自训练轮数
    "confidence_threshold": 0.6,        # 高置信阈值
}


def find_files(name):
    """根据名称查找对应的DAS CSV和视频文件"""
    # 查找CSV
    csv_candidates = [
        os.path.join(DAS_CSV_DIR, f"{name}.csv"),
        os.path.join(DAS_CSV_DIR, f"{name.lower()}.csv"),
        os.path.join(DAS_CSV_DIR, f"{name.upper()}.csv"),
    ]
    
    das_csv = None
    for path in csv_candidates:
        if os.path.exists(path):
            das_csv = path
            break
    
    if das_csv is None:
        raise FileNotFoundError(f"Cannot find DAS CSV for '{name}' in {DAS_CSV_DIR}")
    
    # 查找视频
    video_exts = ['.MP4', '.mp4', '.mov', '.MOV', '.avi', '.AVI']
    video_path = None
    
    for ext in video_exts:
        path = os.path.join(VIDEO_DIR, f"{name}{ext}")
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path is None:
        print(f"[WARNING] Cannot find video for '{name}' in {VIDEO_DIR}")
        print("[INFO] Will use energy-based detection only (no audio weak labels)")
    
    return das_csv, video_path


def run_detection(name, trim_start=None, trim_end=None, align_dt=None):
    """运行脚步检测"""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    # 查找文件
    das_csv, video_path = find_files(name)
    print(f"DAS CSV: {das_csv}")
    print(f"Video: {video_path}")
    
    # 获取配置
    trim_cfg = TRIM_CONFIG.get(name, TRIM_CONFIG["default"])
    
    # 覆盖配置（如果命令行指定了）
    if trim_start is not None:
        trim_cfg["trim_start"] = trim_start
    if trim_end is not None:
        trim_cfg["trim_end"] = trim_end
    if align_dt is not None:
        trim_cfg["align_dt"] = align_dt
    
    print(f"\nTime trim config:")
    print(f"  trim_start: {trim_cfg['trim_start']}s")
    print(f"  trim_end: {trim_cfg['trim_end']}s")
    print(f"  align_dt: {trim_cfg['align_dt']}s")
    
    # 创建配置对象
    config = Config()
    
    # 应用检测配置
    config.das_fs = DETECTION_CONFIG["das_fs"]
    config.das_bp_bands = DETECTION_CONFIG["das_bp_bands"]
    config.audio_bp_low = DETECTION_CONFIG["audio_bp_low"]
    config.audio_bp_high = DETECTION_CONFIG["audio_bp_high"]
    config.step_min_interval = DETECTION_CONFIG["step_min_interval"]
    config.weak_label_sigma = DETECTION_CONFIG["weak_label_sigma"]
    config.self_train_rounds = DETECTION_CONFIG["self_train_rounds"]
    config.confidence_threshold = DETECTION_CONFIG["confidence_threshold"]
    
    # 应用时间裁剪
    config.trim_start_s = trim_cfg["trim_start"]
    config.trim_end_s = trim_cfg["trim_end"]
    
    # 设置输出目录
    config.output_dir = os.path.join(OUTPUT_BASE_DIR, name)
    
    # 运行检测
    step_events, energy_matrix, frame_times = run_pipeline(
        das_csv=das_csv,
        audio_path=video_path,
        config=config,
        align_dt=trim_cfg["align_dt"]
    )
    
    return step_events


def main():
    parser = argparse.ArgumentParser(
        description="快速运行弱监督脚步检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 处理单个文件
    python run_footstep_detection.py wangdihai

    # 处理并指定时间范围
    python run_footstep_detection.py wangdihai --trim_start 10 --trim_end 120

    # 处理所有文件
    python run_footstep_detection.py --all
        """
    )
    
    parser.add_argument('name', nargs='?', default=None,
                        help='数据名称（不含扩展名）')
    parser.add_argument('--all', action='store_true',
                        help='处理所有已知文件')
    parser.add_argument('--trim_start', type=float, default=None,
                        help='覆盖配置的起始裁剪时间')
    parser.add_argument('--trim_end', type=float, default=None,
                        help='覆盖配置的结束裁剪时间')
    parser.add_argument('--align_dt', type=float, default=None,
                        help='覆盖配置的时间对齐偏移')
    
    args = parser.parse_args()
    
    if args.all:
        # 处理所有文件
        names = list(TRIM_CONFIG.keys())
        names.remove("default")
        
        for name in names:
            try:
                run_detection(name, args.trim_start, args.trim_end, args.align_dt)
            except Exception as e:
                print(f"[ERROR] Failed to process {name}: {e}")
    
    elif args.name:
        # 处理单个文件
        run_detection(args.name, args.trim_start, args.trim_end, args.align_dt)
    
    else:
        parser.print_help()
        print("\n可用的数据名称:")
        for name in TRIM_CONFIG.keys():
            if name != "default":
                print(f"  - {name}")


if __name__ == "__main__":
    main()
