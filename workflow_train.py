#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流脚本1：训练模式
====================

功能：输入名字 → 从TDMS切分信号 → 使用音频弱监督训练 → 保存模型

使用示例：
    python workflow_train.py wangdihai
    python workflow_train.py wangdihai --trim_head 50 --trim_tail 20
    python workflow_train.py wangdihai --skip_channels 15
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ============================================================================
# 默认配置
# ============================================================================

# 数据目录（相对于脚本所在目录）
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "Data"
AUDIO_DIR = DATA_DIR / "Audio"  # 音频文件目录
AIRTAG_DIR = DATA_DIR / "Airtag"
DAS_DIR = DATA_DIR / "DAS"

# 输出目录
OUTPUT_BASE_DIR = SCRIPT_DIR / "output"

# 默认参数
DEFAULT_DAS_FS = 2000  # DAS采样率 (Hz) - 实际采样率2000Hz
DEFAULT_UTC_OFFSET = 8.0  # 时区偏移（北京时间 UTC+8）

# 时间裁剪默认值（秒）- 从头尾裁掉的时长
DEFAULT_TRIM_HEAD = 50.0   # 从开头裁掉50秒
DEFAULT_TRIM_TAIL = 20.0   # 从结尾裁掉20秒

# 通道屏蔽默认值
DEFAULT_SKIP_CHANNELS = 18  # 跳过前18个通道


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练工作流：从名字到模型的完整流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（使用默认裁剪：头50s，尾20s）
  python workflow_train.py wangdihai

  # 指定裁剪时长（从开头裁掉60秒，从结尾裁掉30秒）
  python workflow_train.py wangdihai --trim_head 60 --trim_tail 30

  # 跳过前20个通道
  python workflow_train.py wangdihai --skip_channels 20

  # 跳过信号提取（已有CSV）
  python workflow_train.py wangdihai --skip_extract
        """
    )
    
    # 必需参数
    parser.add_argument("name", 
                        help="目标名字（对应Audio中的mp3文件名，不含扩展名）")
    
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
    
    # 控制选项
    parser.add_argument("--skip_extract", action="store_true",
                        help="跳过信号提取步骤（如果CSV已存在）")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出文件")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印将要执行的命令，不实际运行")
    
    return parser.parse_args()


def check_audio_exists(name: str) -> Path:
    """检查音频文件是否存在"""
    audio_path = AUDIO_DIR / f"{name}.mp3"
    if audio_path.exists():
        return audio_path
    raise FileNotFoundError(f"找不到音频文件: {AUDIO_DIR / name}.mp3")


def check_airtag_exists(name: str) -> Path:
    """检查Airtag CSV是否存在"""
    csv_path = AIRTAG_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到Airtag CSV: {csv_path}")
    return csv_path


def run_command(cmd: list, description: str, dry_run: bool = False) -> int:
    """运行命令并打印输出"""
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


def main():
    args = parse_args()
    name = args.name.lower()  # 统一小写
    
    print("="*60)
    print("脚步检测训练工作流")
    print("="*60)
    print(f"目标名字: {name}")
    print(f"时间裁剪: 头部 -{args.trim_head}s, 尾部 -{args.trim_tail}s")
    print(f"跳过通道: 前 {args.skip_channels} 个")
    print(f"DAS采样率: {args.das_fs} Hz")
    
    # ===== 1. 检查输入文件 =====
    print(f"\n[CHECK] 检查输入文件...")
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
    
    # ===== 2. 设置输出路径 =====
    output_dir = OUTPUT_BASE_DIR / name
    csv_output_dir = output_dir / "signals"
    model_output_dir = output_dir / "models"
    results_output_dir = output_dir / "results"
    
    das_csv_path = csv_output_dir / f"{name}.csv"
    model_path = model_output_dir / f"{name}_model.joblib"
    
    print(f"\n[OUTPUT] 输出目录: {output_dir}")
    
    # ===== 3. 提取DAS信号 =====
    # 智能跳过：如果CSV已存在且未指定--overwrite，自动跳过提取
    if das_csv_path.exists() and not args.overwrite:
        print(f"\n[SKIP] 信号CSV已存在，跳过提取: {das_csv_path}")
        print(f"       （如需重新提取，请使用 --overwrite）")
    elif args.skip_extract and das_csv_path.exists():
        print(f"\n[SKIP] 跳过信号提取（CSV已存在: {das_csv_path}）")
    else:
        extract_cmd = [
            sys.executable, str(SCRIPT_DIR / "extract_name_signals_from_tdms.py"),
            "--video-dir", str(VIDEO_DIR),
            "--airtag-csv-dir", str(AIRTAG_DIR),
            "--tdms-dir", str(DAS_DIR),
            "--output-dir", str(csv_output_dir),
            "--fs", str(args.das_fs),
            "--csv-utc-offset-hours", str(DEFAULT_UTC_OFFSET),
            "--name", name,
            "--skip-channels", str(args.skip_channels),
        ]
        if args.overwrite:
            extract_cmd.append("--overwrite")
        
        ret = run_command(extract_cmd, "从TDMS提取DAS信号", args.dry_run)
        if ret != 0:
            print("[ABORT] 信号提取失败")
            return ret
    
    # 验证CSV生成成功
    if not args.dry_run and not das_csv_path.exists():
        print(f"[ERROR] 信号CSV未生成: {das_csv_path}")
        return 1
    
    # ===== 4. 计算实际时间范围 =====
    # 读取CSV获取总时长，然后计算 trim_start 和 trim_end
    import pandas as pd
    if not args.dry_run:
        df_info = pd.read_csv(das_csv_path, nrows=0)  # 只读header
        n_samples = sum(1 for _ in open(das_csv_path)) - 1  # 减去header行
        total_duration = n_samples / args.das_fs
        
        trim_start = args.trim_head
        trim_end = total_duration - args.trim_tail
        
        if trim_end <= trim_start:
            print(f"[ERROR] 裁剪后无有效数据: 总时长={total_duration:.1f}s, 头部裁剪={args.trim_head}s, 尾部裁剪={args.trim_tail}s")
            return 1
        
        effective_duration = trim_end - trim_start
        print(f"\n[TIME] 数据总时长: {total_duration:.1f}s")
        print(f"       裁剪后范围: {trim_start:.1f}s - {trim_end:.1f}s (有效 {effective_duration:.1f}s)")
    else:
        # dry_run 模式使用占位值
        trim_start = args.trim_head
        trim_end = 999  # placeholder
    
    # ===== 5. 创建模型目录 =====
    if not args.dry_run:
        model_output_dir.mkdir(parents=True, exist_ok=True)
        results_output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 6. 训练弱监督模型 =====
    train_cmd = [
        sys.executable, str(SCRIPT_DIR / "WeaklySupervised_FootstepDetector.py"),
        "--das_csv", str(das_csv_path),
        "--audio", str(audio_path),
        "--trim_start", str(trim_start),
        "--trim_end", str(trim_end),
        "--das_fs", str(args.das_fs),
        "--self_train_rounds", str(args.self_train_rounds),
        "--confidence_threshold", str(args.confidence_threshold),
        "--output_dir", str(results_output_dir),
        "--save_model", str(model_path),
    ]
    
    ret = run_command(train_cmd, "训练弱监督脚步检测模型", args.dry_run)
    if ret != 0:
        print("[ABORT] 模型训练失败")
        return ret
    
    # ===== 7. 完成 =====
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"DAS信号CSV: {das_csv_path}")
    print(f"训练好的模型: {model_path}")
    print(f"检测结果目录: {results_output_dir}")
    print()
    print("下一步：使用以下命令对其他数据进行推理：")
    print(f"  python workflow_infer.py <其他名字> --model {model_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
