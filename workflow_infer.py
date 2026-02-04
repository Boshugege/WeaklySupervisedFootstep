#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流脚本2：推理模式
====================

功能：输入名字和模型路径 → 从TDMS切分信号 → 加载模型识别脚步

使用示例：
    python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib
    python workflow_infer.py wangjiahui --model models/pretrained.joblib --trim_head 50 --trim_tail 20
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
VIDEO_DIR = DATA_DIR / "Video"
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
        description="推理工作流：使用已训练模型检测新数据中的脚步",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用wangdihai训练的模型检测wangjiahui的脚步
  python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib

  # 指定裁剪时长（从开头裁掉60秒，从结尾裁掉30秒）
  python workflow_infer.py wangjiahui --model model.joblib --trim_head 60 --trim_tail 30

  # 跳过前20个通道
  python workflow_infer.py wangjiahui --model model.joblib --skip_channels 20

  # 同时使用音频进行对比验证
  python workflow_infer.py wangjiahui --model model.joblib --with_audio

  # 跳过信号提取（已有CSV）
  python workflow_infer.py wangjiahui --model model.joblib --skip_extract
        """
    )
    
    # 必需参数
    parser.add_argument("name", 
                        help="目标名字（对应Video中的MP4文件名，不含扩展名）")
    parser.add_argument("--model", "-m", required=True,
                        help="已训练模型的路径（.joblib文件）")
    
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
    
    # 控制选项
    parser.add_argument("--with_audio", action="store_true",
                        help="同时使用音频进行对比验证（非纯推理模式）")
    parser.add_argument("--skip_extract", action="store_true",
                        help="跳过信号提取步骤（如果CSV已存在）")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出文件")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印将要执行的命令，不实际运行")
    
    # 自定义路径
    parser.add_argument("--das_csv", type=str, default=None,
                        help="直接指定DAS CSV文件路径（跳过提取步骤）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="自定义输出目录")
    
    return parser.parse_args()


from typing import Optional


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
    if not csv_path.exists():
        return None
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
    print("脚步检测推理工作流")
    print("="*60)
    print(f"目标名字: {name}")
    print(f"使用模型: {args.model}")
    print(f"时间裁剪: 头部 -{args.trim_head}s, 尾部 -{args.trim_tail}s")
    print(f"跳过通道: 前 {args.skip_channels} 个")
    print(f"DAS采样率: {args.das_fs} Hz")
    print(f"模式: {'带音频对比' if args.with_audio else '纯推理（无需音频）'}")
    
    # ===== 1. 检查模型文件 =====
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n[ERROR] 模型文件不存在: {model_path}")
        return 1
    print(f"\n[CHECK] 模型文件: {model_path} ✓")
    
    # ===== 2. 确定DAS CSV路径 =====
    if args.das_csv:
        # 直接使用指定的CSV
        das_csv_path = Path(args.das_csv)
        if not das_csv_path.exists():
            print(f"[ERROR] 指定的DAS CSV不存在: {das_csv_path}")
            return 1
        print(f"[CHECK] 使用指定的DAS CSV: {das_csv_path}")
        need_extract = False
    else:
        # 检查是否需要提取
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE_DIR / name
        csv_output_dir = output_dir / "signals"
        das_csv_path = csv_output_dir / f"{name}.csv"
        
        # 智能跳过：如果CSV已存在且未指定--overwrite，自动跳过提取
        if das_csv_path.exists() and not args.overwrite:
            need_extract = False
            print(f"[SKIP] 信号CSV已存在，跳过提取: {das_csv_path}")
        elif args.skip_extract and das_csv_path.exists():
            need_extract = False
            print(f"[SKIP] 使用已存在的DAS CSV: {das_csv_path}")
        else:
            need_extract = True
        
        if need_extract:
            # 需要提取，检查Airtag文件
            airtag_path = check_airtag_exists(name)
            if not airtag_path:
                print(f"[ERROR] 找不到Airtag CSV: {AIRTAG_DIR / name}.csv")
                print("  提示：如果CSV已存在，使用 --das_csv 直接指定路径")
                return 1
            print(f"[CHECK] Airtag CSV: {airtag_path} ✓")
        else:
            print(f"[SKIP] 使用已存在的DAS CSV: {das_csv_path}")
    
    # ===== 3. 设置输出路径 =====
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE_DIR / name
    results_output_dir = output_dir / "results"
    csv_output_dir = output_dir / "signals"
    
    print(f"\n[OUTPUT] 输出目录: {results_output_dir}")
    
    # ===== 4. 提取DAS信号（如需要） =====
    if need_extract and not args.das_csv:
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

        # 如果没有对应的视频文件，则使用 Airtag CSV 名称来提取（不依赖 MP4）
        video_path = find_video(name)
        if video_path:
            print(f"[INFO] Found video: {video_path}")
        else:
            print(f"[INFO] No video file found for '{name}'; using Airtag CSVs only (--use-airtag-only)")
            extract_cmd.append("--use-airtag-only")
        
        ret = run_command(extract_cmd, "从TDMS提取DAS信号", args.dry_run)
        if ret != 0:
            print("[ABORT] 信号提取失败")
            return ret
        
        # 验证CSV生成成功
        if not args.dry_run and not das_csv_path.exists():
            print(f"[ERROR] 信号CSV未生成: {das_csv_path}")
            return 1
    
    # ===== 5. 计算实际时间范围 =====
    import pandas as pd
    if not args.dry_run:
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
        trim_start = args.trim_head
        trim_end = 999  # placeholder
    
    # ===== 6. 创建输出目录 =====
    if not args.dry_run:
        results_output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 7. 运行推理 =====
    infer_cmd = [
        sys.executable, str(SCRIPT_DIR / "WeaklySupervised_FootstepDetector.py"),
        "--das_csv", str(das_csv_path),
        "--trim_start", str(trim_start),
        "--trim_end", str(trim_end),
        "--das_fs", str(args.das_fs),
        "--output_dir", str(results_output_dir),
        "--load_model", str(model_path),
    ]
    
    if args.with_audio:
        # 带音频对比模式
        video_path = find_video(name)
        if video_path:
            infer_cmd.extend(["--audio", str(video_path)])
            print(f"[INFO] 使用音频进行对比: {video_path}")
        else:
            print(f"[WARN] 未找到视频文件，将使用纯推理模式")
            infer_cmd.append("--inference_only")
    else:
        # 纯推理模式
        infer_cmd.append("--inference_only")
    
    ret = run_command(infer_cmd, "使用模型进行脚步检测", args.dry_run)
    if ret != 0:
        print("[ABORT] 推理失败")
        return ret
    
    # ===== 8. 完成 =====
    print("\n" + "="*60)
    print("推理完成！")
    print("="*60)
    print(f"DAS信号CSV: {das_csv_path}")
    print(f"使用的模型: {model_path}")
    print(f"检测结果目录: {results_output_dir}")
    
    # 列出生成的文件
    if not args.dry_run and results_output_dir.exists():
        print("\n生成的文件:")
        for f in sorted(results_output_dir.glob(f"{name}*")):
            print(f"  - {f.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
