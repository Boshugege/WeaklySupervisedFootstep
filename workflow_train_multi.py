#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多人联合训练工作流
==================

功能：使用多个人的数据联合训练，提高模型泛化性

使用示例：
    # 使用多人数据联合训练（默认：头部裁掉50s，尾部裁掉20s）
    python workflow_train_multi.py wangdihai wangjiahui

    # 自定义裁剪
    python workflow_train_multi.py wangdihai wangjiahui --trim_head 60 --trim_tail 30

    # 指定输出模型名称
    python workflow_train_multi.py wangdihai wangjiahui wuwenxuan --model_name multi_3person

    # 自定义参数
    python workflow_train_multi.py wangdihai wangjiahui --self_train_rounds 5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# ============================================================================
# 默认配置
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "Data"
VIDEO_DIR = DATA_DIR / "Video"
AIRTAG_DIR = DATA_DIR / "Airtag"
DAS_DIR = DATA_DIR / "DAS"
OUTPUT_BASE_DIR = SCRIPT_DIR / "output"

# 默认参数
DEFAULT_DAS_FS = 2000  # DAS采样率 (Hz)
DEFAULT_UTC_OFFSET = 8.0  # 时区偏移（北京时间 UTC+8）

# 时间裁剪默认值（秒）- 从头尾裁掉的时长
DEFAULT_TRIM_HEAD = 50.0   # 从开头裁掉50秒
DEFAULT_TRIM_TAIL = 20.0   # 从结尾裁掉20秒

# 通道屏蔽默认值
DEFAULT_SKIP_CHANNELS = 15  # 跳过前15个通道


def parse_args():
    parser = argparse.ArgumentParser(
        description="多人联合训练：使用多个人的数据一起训练，提高泛化性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 两人联合训练
  python workflow_train_multi.py wangdihai wangjiahui

  # 三人联合训练，自定义模型名
  python workflow_train_multi.py wangdihai wangjiahui wuwenxuan --model_name multi_3person

  # 自定义裁剪（从开头裁掉60秒，从结尾裁掉30秒）
  python workflow_train_multi.py wangdihai wangjiahui --trim_head 60 --trim_tail 30

  # 更多自训练轮数
  python workflow_train_multi.py wangdihai wangjiahui --self_train_rounds 5
        """
    )
    
    # 必需参数：多个名字
    parser.add_argument("names", nargs="+",
                        help="训练数据的名字列表（至少2个）")
    
    # 时间裁剪 - 从头尾裁掉的时长
    parser.add_argument("--trim_head", type=float, default=DEFAULT_TRIM_HEAD,
                        help=f"从数据开头裁掉的时长（秒），默认 {DEFAULT_TRIM_HEAD}")
    parser.add_argument("--trim_tail", type=float, default=DEFAULT_TRIM_TAIL,
                        help=f"从数据结尾裁掉的时长（秒），默认 {DEFAULT_TRIM_TAIL}")
    
    # 通道屏蔽
    parser.add_argument("--skip_channels", type=int, default=DEFAULT_SKIP_CHANNELS,
                        help=f"跳过前N个通道（默认 {DEFAULT_SKIP_CHANNELS}），设为0保留所有通道")
    
    # 采样率
    parser.add_argument("--das_fs", type=int, default=DEFAULT_DAS_FS,
                        help=f"DAS采样率 (Hz)，默认 {DEFAULT_DAS_FS}")
    
    # 训练参数
    parser.add_argument("--self_train_rounds", type=int, default=3,
                        help="自训练迭代轮数，默认 3")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="高置信预测阈值，默认 0.7")
    
    # 输出控制
    parser.add_argument("--model_name", type=str, default=None,
                        help="输出模型名称，默认为 'multi_<names>'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认 output/multi_<model_name>")
    
    # 其他选项
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出文件")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印将要执行的命令，不实际运行")
    
    return parser.parse_args()


def find_video(name: str) -> Optional[Path]:
    """查找视频文件"""
    for ext in [".MP4", ".mp4", ".MOV", ".mov"]:
        video_path = VIDEO_DIR / f"{name}{ext}"
        if video_path.exists():
            return video_path
    return None


def check_airtag_exists(name: str) -> Optional[Path]:
    """检查Airtag CSV是否存在"""
    csv_path = AIRTAG_DIR / f"{name}.csv"
    if csv_path.exists():
        return csv_path
    return None


def run_command(cmd: list, description: str, dry_run: bool = False) -> int:
    """运行命令"""
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


def extract_signals_for_name(name: str, output_dir: Path, das_fs: int, 
                              skip_channels: int, overwrite: bool, dry_run: bool) -> Tuple[bool, Path]:
    """为单个名字提取DAS信号"""
    csv_output_dir = output_dir / "signals"
    das_csv_path = csv_output_dir / f"{name}.csv"
    
    # 检查是否已存在
    if das_csv_path.exists() and not overwrite:
        print(f"  [SKIP] {name}: 信号CSV已存在")
        return True, das_csv_path
    
    # 检查Airtag
    airtag_path = check_airtag_exists(name)
    if not airtag_path:
        print(f"  [ERROR] {name}: 找不到Airtag CSV")
        return False, das_csv_path
    
    # 检查是否有视频（用于确定提取模式）
    video_path = find_video(name)
    
    extract_cmd = [
        sys.executable, str(SCRIPT_DIR / "extract_name_signals_from_tdms.py"),
        "--video-dir", str(VIDEO_DIR),
        "--airtag-csv-dir", str(AIRTAG_DIR),
        "--tdms-dir", str(DAS_DIR),
        "--output-dir", str(csv_output_dir),
        "--fs", str(das_fs),
        "--csv-utc-offset-hours", str(DEFAULT_UTC_OFFSET),
        "--name", name,
        "--skip-channels", str(skip_channels),
    ]
    
    if overwrite:
        extract_cmd.append("--overwrite")
    if not video_path:
        extract_cmd.append("--use-airtag-only")
    
    if dry_run:
        print(f"  [DRY RUN] {name}: 跳过提取")
        return True, das_csv_path
    
    result = subprocess.run(extract_cmd, cwd=str(SCRIPT_DIR), 
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {name}: 提取失败")
        print(result.stderr)
        return False, das_csv_path
    
    print(f"  [OK] {name}: 提取成功")
    return True, das_csv_path


def main():
    args = parse_args()
    names = [n.lower() for n in args.names]
    
    if len(names) < 2:
        print("[ERROR] 至少需要2个名字进行联合训练")
        return 1
    
    # 生成模型名称
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"multi_{'_'.join(names[:3])}"  # 最多取前3个名字
        if len(names) > 3:
            model_name += f"_+{len(names)-3}"
    
    print("=" * 60)
    print("多人联合训练工作流")
    print("=" * 60)
    print(f"训练数据: {', '.join(names)} ({len(names)}人)")
    print(f"时间裁剪: 头部 -{args.trim_head}s, 尾部 -{args.trim_tail}s")
    print(f"跳过通道: 前 {args.skip_channels} 个")
    print(f"DAS采样率: {args.das_fs} Hz")
    print(f"模型名称: {model_name}")
    
    # ===== 1. 检查输入文件 =====
    print(f"\n[CHECK] 检查输入文件...")
    valid_names = []
    video_paths = {}
    
    for name in names:
        video_path = find_video(name)
        airtag_path = check_airtag_exists(name)
        
        if not airtag_path:
            print(f"  ✗ {name}: 缺少Airtag CSV，跳过")
            continue
        
        if video_path:
            print(f"  ✓ {name}: 视频={video_path.name}, Airtag=✓")
            video_paths[name] = video_path
        else:
            print(f"  ⚠ {name}: 无视频（仅用于补充训练数据），Airtag=✓")
            video_paths[name] = None
        
        valid_names.append(name)
    
    if len(valid_names) < 2:
        print(f"\n[ERROR] 有效数据不足2人，无法进行联合训练")
        return 1
    
    # 检查至少有一个有视频（用于弱监督）
    names_with_video = [n for n in valid_names if video_paths[n] is not None]
    if not names_with_video:
        print(f"\n[ERROR] 所有数据都没有视频，无法进行弱监督训练")
        return 1
    
    print(f"\n[INFO] 有效训练数据: {len(valid_names)}人, 其中{len(names_with_video)}人有视频")
    
    # ===== 2. 设置输出路径 =====
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE_DIR / model_name
    model_output_dir = output_dir / "models"
    results_output_dir = output_dir / "results"
    model_path = model_output_dir / f"{model_name}.joblib"
    
    print(f"\n[OUTPUT] 输出目录: {output_dir}")
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        results_output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 3. 提取所有人的DAS信号 =====
    print(f"\n[EXTRACT] 提取DAS信号...")
    das_csv_paths = {}
    
    for name in valid_names:
        success, csv_path = extract_signals_for_name(
            name, output_dir, args.das_fs, args.skip_channels, args.overwrite, args.dry_run
        )
        if success:
            das_csv_paths[name] = csv_path
        else:
            print(f"  [WARN] {name}: 提取失败，从训练中排除")
    
    if len(das_csv_paths) < 2:
        print(f"\n[ERROR] 成功提取的数据不足2人")
        return 1
    
    # ===== 4. 联合训练 =====
    # 构建训练命令 - 使用第一个有视频的人作为主训练数据
    primary_name = names_with_video[0]
    primary_csv = das_csv_paths.get(primary_name)
    primary_video = video_paths[primary_name]
    
    if not primary_csv or not primary_csv.exists():
        print(f"[ERROR] 主训练数据 {primary_name} 的CSV不存在")
        return 1
    
    # 合并所有CSV数据用于训练
    print(f"\n[MERGE] 合并训练数据...")
    
    if not args.dry_run:
        merged_data = []
        merged_audio_steps = []
        
        for name in das_csv_paths.keys():
            csv_path = das_csv_paths[name]
            if not csv_path.exists():
                continue
            
            # 加载DAS数据
            df = pd.read_csv(csv_path)
            das_raw = df.values.astype(np.float32)
            
            # 计算每个人的实际时间范围
            total_samples = das_raw.shape[0]
            total_duration = total_samples / args.das_fs
            
            trim_start = args.trim_head
            trim_end = total_duration - args.trim_tail
            
            if trim_end <= trim_start:
                print(f"  [WARN] {name}: 裁剪后无数据 (总时长={total_duration:.1f}s, 头部-{args.trim_head}s, 尾部-{args.trim_tail}s)，跳过")
                continue
            
            start_idx = int(trim_start * args.das_fs)
            end_idx = int(trim_end * args.das_fs)
            end_idx = min(end_idx, das_raw.shape[0])
            
            das_trimmed = das_raw[start_idx:end_idx, :]
            merged_data.append(das_trimmed)
            
            duration = (end_idx - start_idx) / args.das_fs
            print(f"  [OK] {name}: {das_trimmed.shape[0]} 样本 ({duration:.1f}s), 裁剪范围 {trim_start:.1f}s-{trim_end:.1f}s")
        
        if not merged_data:
            print("[ERROR] 没有有效的训练数据")
            return 1
        
        # 保存合并后的数据
        merged_csv_path = output_dir / "signals" / f"{model_name}_merged.csv"
        merged_array = np.vstack(merged_data)
        
        # 创建DataFrame并保存
        columns = [f"ch_{i}" for i in range(merged_array.shape[1])]
        df_merged = pd.DataFrame(merged_array, columns=columns)
        df_merged.to_csv(merged_csv_path, index=False)
        
        merged_duration = merged_array.shape[0] / args.das_fs
        print(f"\n[MERGED] 合并数据: {merged_array.shape[0]} 样本 ({merged_duration:.1f}s)")
        print(f"[MERGED] 保存到: {merged_csv_path}")
    else:
        merged_csv_path = output_dir / "signals" / f"{model_name}_merged.csv"
        merged_duration = 999  # placeholder
    
    # ===== 5. 使用合并数据训练 =====
    # 注意：需要收集所有有视频的音频来生成弱标签
    # 这里我们使用主视频训练，但用合并的DAS数据
    
    train_cmd = [
        sys.executable, str(SCRIPT_DIR / "WeaklySupervised_FootstepDetector.py"),
        "--das_csv", str(merged_csv_path),
        "--audio", str(primary_video),
        "--trim_start", "0",  # 已经裁剪过了
        "--trim_end", str(merged_duration),  # 合并后的总时长
        "--das_fs", str(args.das_fs),
        "--self_train_rounds", str(args.self_train_rounds),
        "--confidence_threshold", str(args.confidence_threshold),
        "--output_dir", str(results_output_dir),
        "--save_model", str(model_path),
    ]
    
    ret = run_command(train_cmd, f"联合训练 ({len(das_csv_paths)}人数据)", args.dry_run)
    if ret != 0:
        print("[ABORT] 训练失败")
        return ret
    
    # ===== 6. 完成 =====
    print("\n" + "=" * 60)
    print("联合训练完成！")
    print("=" * 60)
    print(f"训练数据: {', '.join(das_csv_paths.keys())}")
    print(f"合并数据: {merged_csv_path}")
    print(f"训练模型: {model_path}")
    print(f"结果目录: {results_output_dir}")
    print()
    print("使用方法：")
    print(f"  python workflow_infer.py <名字> --model {model_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
