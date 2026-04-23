#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流脚本2：推理模式
====================

功能：输入名字和模型路径 → 从TDMS切分信号 → 加载模型识别脚步

使用示例：
    python workflow_infer.py wangjiahui --model output/wangdihai/models/wangdihai_model.joblib
    python workflow_infer.py wangjiahui --model models/pretrained.joblib --trim_head 50 --trim_tail 20
    python workflow_infer.py wangjiahui --replay_only --stream_after_infer --replay_speed 1.0
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
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
DOUBLE_CHUNK_SIZE = 20000  # 双人模拟分块行数，平衡内存与速度


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

    # 推理后按在线格式回放发送结果
    python workflow_infer.py wangjiahui --model model.joblib --stream_after_infer --host 127.0.0.1 --port 9000

    # 仅回放之前推理过的结果（不重新推理）
    python workflow_infer.py wangjiahui --replay_only --replay_speed 1.0 --mirror_only

    # 指定steps通道模式，避免历史文件发生二次偏移
    python workflow_infer.py wangjiahui --replay_only --steps_channel_mode global
        """
    )
    
    # 必需参数
    parser.add_argument("name", 
                        help="目标名字（对应Video中的MP4文件名，不含扩展名）")
    parser.add_argument("--model", "-m", required=False,
                        help="已训练模型的路径（.joblib文件）；--replay_only 时可不提供")
    
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
    parser.add_argument("--disable_das_bandpass", action="store_true",
                        help="关闭DAS带通滤波，直接使用原始DAS信号推理（建议与训练时设置一致）")
    parser.add_argument("--das_filter_method", type=str, default="sosfilt",
                        choices=["filtfilt", "sosfilt"],
                        help="DAS带通滤波方法，默认 sosfilt")
    parser.add_argument("--double", action="store_true",
                        help="启用双人模拟：将DAS信号与其通道倒置版本逐点相加后再推理")
    parser.add_argument("--mirror_only", action="store_true",
                        help="仅使用通道镜像后的信号进行推理（不与原信号叠加）")
    parser.add_argument("--channel_shift", type=int, default=0,
                        help="对推理输入做通道偏移（正数向高通道移动，空位补0；默认0不偏移）")
    parser.add_argument("--replay_only", action="store_true",
                        help="仅回放历史推理结果并发送，不重新运行推理")

    # 推理后回放发送（在线同格式）
    parser.add_argument("--stream_after_infer", action="store_true",
                        help="离线推理完成后，按在线格式(signal/event)发送结果")
    parser.add_argument("--protocol", choices=["udp", "tcp"], default="udp",
                        help="回放发送协议，默认 udp")
    parser.add_argument("--host", default="127.0.0.1",
                        help="回放发送目标主机，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=9000,
                        help="回放发送目标端口，默认 9000")
    parser.add_argument("--signal_downsample", type=int, default=1,
                        help="发送signal包时降采样因子，默认 1")
    parser.add_argument("--udp_max_samples", type=int, default=10,
                        help="UDP每个signal包最大样本数，默认 10")
    parser.add_argument("--udp_max_bytes", type=int, default=60000,
                        help="UDP包最大字节数，默认 60000")
    parser.add_argument("--replay_speed", type=float, default=0.0,
                        help="回放速度（1.0=实时，0=不等待），默认 0")
    parser.add_argument("--steps_channel_mode", choices=["auto", "local", "global"], default="auto",
                        help="steps CSV 的 channel 字段模式：local(局部索引)/global(全局通道号)/auto(自动判断，默认)")
    
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


def open_sender(protocol: str, host: str, port: int):
    if protocol == "udp":
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock


def send_packet(sock, protocol: str, host: str, port: int, payload: dict):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if protocol == "udp":
        sock.sendto(data, (host, port))
    else:
        sock.sendall(data + b"\n")


def stream_offline_results(args, das_csv_path: Path, steps_csv_path: Path, trim_start: float, trim_end: float):
    import pandas as pd
    import numpy as np
    import re

    if not steps_csv_path.exists():
        print(f"[WARN] 未找到steps结果文件，跳过回放发送: {steps_csv_path}")
        return

    print(f"\n[STREAM] 回放离线结果到 {args.protocol.upper()} {args.host}:{args.port}")

    df_das = pd.read_csv(das_csv_path)
    all_data = df_das.values.astype(np.float32)

    start_idx = max(0, int(trim_start * args.das_fs))
    end_idx = min(all_data.shape[0], int(trim_end * args.das_fs))
    if end_idx <= start_idx:
        print("[WARN] 回放区间为空，跳过发送")
        return

    replay_data = all_data[start_idx:end_idx, :]

    channel_ids = []
    ch_pattern = re.compile(r"^ch_(\d+)$")
    for i, col in enumerate(df_das.columns):
        m = ch_pattern.match(str(col))
        if m:
            channel_ids.append(int(m.group(1)))
        else:
            channel_ids.append(i)

    if len(channel_ids) != replay_data.shape[1]:
        channel_ids = list(range(replay_data.shape[1]))

    total_channels = int(max(channel_ids) + 1) if channel_ids else int(replay_data.shape[1])
    need_signal_restore = (total_channels != replay_data.shape[1]) or (channel_ids and (min(channel_ids) != 0))

    df_steps = pd.read_csv(steps_csv_path)
    step_events = []
    local_max_ch = -1
    global_max_ch = -1
    min_channel_id = int(min(channel_ids)) if channel_ids else 0
    max_channel_id = int(max(channel_ids)) if channel_ids else (int(replay_data.shape[1]) - 1)

    step_mode = str(args.steps_channel_mode)
    if step_mode == "auto" and (not df_steps.empty) and ("channel" in df_steps.columns):
        step_min = int(df_steps["channel"].min())
        step_max = int(df_steps["channel"].max())
        # 自动规则：
        # 1) 出现小于最小有效通道号（例如 ch_18 前的 0..17）=> local
        # 2) 全部落在全局通道范围内 => global
        # 3) 兜底按 local（兼容旧结果）
        if step_min < min_channel_id:
            step_mode = "local"
        elif step_min >= min_channel_id and step_max <= max_channel_id:
            step_mode = "global"
        else:
            step_mode = "local"

    print(f"[STREAM] steps通道模式: {step_mode} (min_ch={min_channel_id}, max_ch={max_channel_id})")

    if not df_steps.empty and all(c in df_steps.columns for c in ["time", "channel", "confidence"]):
        for _, row in df_steps.sort_values("time").iterrows():
            ch = int(row["channel"])
            local_max_ch = max(local_max_ch, ch)
            if step_mode == "global":
                ch_global = int(ch)
            else:
                if 0 <= ch < len(channel_ids):
                    ch_global = int(channel_ids[ch])
                else:
                    ch_global = int(ch)
            global_max_ch = max(global_max_ch, ch_global)
            step_events.append((float(row["time"]), ch_global, float(row["confidence"])))

    if local_max_ch >= 0:
        inferred_tail_trim = max(0, int(total_channels - (global_max_ch + 1)))
        print(
            f"[STREAM] 通道映射恢复: local[0..{local_max_ch}] -> global[0..{global_max_ch}] "
            f"(total={total_channels}, restore_zeros={need_signal_restore}, tail_trim={inferred_tail_trim})"
        )
    else:
        print(f"[STREAM] 通道映射恢复: 无events，保持 total_channels={total_channels}, restore_zeros={need_signal_restore}")

    sock = open_sender(args.protocol, args.host, args.port)
    try:
        event_ptr = 0
        n = replay_data.shape[0]
        down = max(1, int(args.signal_downsample))
        udp_step = max(1, int(args.udp_max_samples))
        replay_dt = 1.0 / float(args.das_fs)

        for start in range(0, n, udp_step * down):
            chunk = replay_data[start:start + udp_step * down:down, :]
            if chunk.size == 0:
                continue

            sample_rate = float(args.das_fs) / down
            ts = start / float(args.das_fs)

            if need_signal_restore:
                restored_chunk = np.zeros((chunk.shape[0], total_channels), dtype=chunk.dtype)
                restored_chunk[:, channel_ids] = chunk
                out_chunk = restored_chunk
            else:
                out_chunk = chunk

            payload = {
                "packet_type": "signal",
                "timestamp": ts,
                "sample_rate": sample_rate,
                "sample_count": int(out_chunk.shape[0]),
                "total_channels": int(total_channels),
                "signals": out_chunk.tolist(),
            }

            if args.protocol == "udp" and args.udp_max_bytes and args.udp_max_bytes > 0:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if len(data) > args.udp_max_bytes:
                    reduced = out_chunk[: max(1, int(out_chunk.shape[0] / 2)), :]
                    payload["signals"] = reduced.tolist()
                    payload["sample_count"] = int(reduced.shape[0])

            send_packet(sock, args.protocol, args.host, args.port, payload)

            window_end_t = (start + chunk.shape[0] * down) / float(args.das_fs)
            while event_ptr < len(step_events) and step_events[event_ptr][0] < window_end_t:
                t, ch, conf = step_events[event_ptr]
                event_payload = {
                    "packet_type": "event",
                    "timestamp": float(t),
                    "channel_index": int(ch),
                    "confidence": float(conf),
                }
                send_packet(sock, args.protocol, args.host, args.port, event_payload)
                event_ptr += 1

            if args.replay_speed and args.replay_speed > 0:
                time.sleep((chunk.shape[0] * down * replay_dt) / args.replay_speed)

        while event_ptr < len(step_events):
            t, ch, conf = step_events[event_ptr]
            event_payload = {
                "packet_type": "event",
                "timestamp": float(t),
                "channel_index": int(ch),
                "confidence": float(conf),
            }
            send_packet(sock, args.protocol, args.host, args.port, event_payload)
            event_ptr += 1

        print(f"[STREAM] 已发送 signal + event（events={len(step_events)}）")
    finally:
        sock.close()


def _compute_mirror_indices(columns: list):
    import re

    ch_pattern = re.compile(r"^ch_(\d+)$")
    parsed_ids = []
    for col in columns:
        m = ch_pattern.match(str(col))
        if not m:
            return None, None
        parsed_ids.append(int(m.group(1)))

    if not parsed_ids:
        return None, None

    max_channel_id = max(parsed_ids)
    id_to_idx = {ch_id: idx for idx, ch_id in enumerate(parsed_ids)}
    mirror_indices = [id_to_idx.get(max_channel_id - ch_id, -1) for ch_id in parsed_ids]
    return mirror_indices, max_channel_id


def _apply_channel_shift(values, channel_shift: int):
    import numpy as np

    shift = int(channel_shift)
    if shift == 0:
        return values

    n_channels = values.shape[1]
    shifted = np.zeros_like(values)
    if abs(shift) >= n_channels:
        return shifted

    if shift > 0:
        shifted[:, shift:] = values[:, : n_channels - shift]
    else:
        k = -shift
        shifted[:, : n_channels - k] = values[:, k:]
    return shifted


def build_augmented_csv(
    src_csv_path: Path,
    dst_csv_path: Path,
    *,
    do_double: bool,
    do_mirror_only: bool,
    channel_shift: int,
    chunk_size: int = DOUBLE_CHUNK_SIZE,
):
    import pandas as pd

    dst_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_csv_path.exists():
        dst_csv_path.unlink()

    # 优先按全局通道号镜像（ch_k -> ch_(max_k - k)），避免 skip_channels 后简单翻转引起的通道位置偏移
    col_df = pd.read_csv(src_csv_path, nrows=0)
    columns = list(col_df.columns)
    mirror_indices, max_channel_id = _compute_mirror_indices(columns)
    if mirror_indices is not None:
        print(
            f"[AUG] 通道镜像映射: ch_k -> ch_{max_channel_id}-k "
            f"(示例: {columns[0]} -> ch_{max_channel_id - int(str(columns[0]).split('_')[-1])})"
        )
    else:
        print("[AUG] 通道镜像映射: 回退为列顺序反转（未检测到标准ch_x列名）")

    print(f"[AUG] 变换配置: mirror_only={do_mirror_only}, double={do_double}, channel_shift={channel_shift}")

    wrote = False
    for chunk_df in pd.read_csv(src_csv_path, chunksize=chunk_size):
        chunk_values = chunk_df.values.astype("float32", copy=False)

        if mirror_indices is None:
            mirrored_values = chunk_values[:, ::-1]
        else:
            mirrored_values = chunk_values.copy()
            for dst_idx, src_idx in enumerate(mirror_indices):
                if src_idx >= 0:
                    mirrored_values[:, dst_idx] = chunk_values[:, src_idx]
                else:
                    mirrored_values[:, dst_idx] = 0.0

        if do_mirror_only:
            transformed_values = mirrored_values
        elif do_double:
            transformed_values = chunk_values + mirrored_values
        else:
            transformed_values = chunk_values

        transformed_values = _apply_channel_shift(transformed_values, channel_shift)

        out_df = pd.DataFrame(transformed_values, columns=chunk_df.columns)
        out_df.to_csv(dst_csv_path, mode="a", header=not wrote, index=False)
        wrote = True

    if not wrote:
        raise RuntimeError(f"输入CSV为空，无法构建双人模拟信号: {src_csv_path}")


def main():
    args = parse_args()
    name = args.name.lower()  # 统一小写
    
    print("="*60)
    print("脚步检测推理工作流")
    print("="*60)
    print(f"目标名字: {name}")
    print(f"使用模型: {args.model if args.model else '(未指定)'}")
    print(f"时间裁剪: 头部 -{args.trim_head}s, 尾部 -{args.trim_tail}s")
    print(f"跳过通道: 前 {args.skip_channels} 个")
    print(f"DAS采样率: {args.das_fs} Hz")
    print(f"双人模拟: {'开启' if args.double else '关闭'}")
    print(f"纯镜像: {'开启' if args.mirror_only else '关闭'}")
    print(f"通道偏移: {args.channel_shift}")
    print(f"仅回放历史结果: {'开启' if args.replay_only else '关闭'}")
    run_mode = "仅回放历史结果" if args.replay_only else ("带音频对比" if args.with_audio else "纯推理（无需音频）")
    print(f"模式: {run_mode}")

    if args.double and args.mirror_only:
        print("[ERROR] 参数冲突：--double 与 --mirror_only 不能同时开启")
        return 1
    
    # ===== 1. 检查模型文件 =====
    model_path = None
    if not args.replay_only:
        if not args.model:
            print("\n[ERROR] 非 --replay_only 模式下必须提供 --model")
            return 1
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"\n[ERROR] 模型文件不存在: {model_path}")
            return 1
        print(f"\n[CHECK] 模型文件: {model_path} ✓")
    elif args.model:
        model_path = Path(args.model)
        if model_path.exists():
            print(f"\n[CHECK] 回放模式检测到模型文件（不会用于推理）: {model_path} ✓")
        else:
            print(f"\n[WARN] 回放模式下指定的模型文件不存在（可忽略）: {model_path}")
    
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
    
    # ===== 7. 运行推理（或仅回放） =====
    infer_das_csv_path = das_csv_path
    use_augmented_input = args.double or args.mirror_only or (int(args.channel_shift) != 0)
    if use_augmented_input and not args.dry_run:
        aug_suffix_parts = []
        if args.mirror_only:
            aug_suffix_parts.append("mirror")
        elif args.double:
            aug_suffix_parts.append("double")
        if int(args.channel_shift) != 0:
            shift = int(args.channel_shift)
            aug_suffix_parts.append(f"shift{'p' if shift > 0 else 'm'}{abs(shift)}")

        aug_suffix = "_".join(aug_suffix_parts)
        augmented_csv_path = csv_output_dir / f"{das_csv_path.stem}_{aug_suffix}.csv"
        print(f"\n[AUG] 构建增强信号: {augmented_csv_path}")
        build_augmented_csv(
            das_csv_path,
            augmented_csv_path,
            do_double=args.double,
            do_mirror_only=args.mirror_only,
            channel_shift=int(args.channel_shift),
        )
        infer_das_csv_path = augmented_csv_path
        print(f"[AUG] 输入切换为增强信号: {infer_das_csv_path}")

    if not args.replay_only:
        infer_cmd = [
            sys.executable, str(SCRIPT_DIR / "WeaklySupervised_FootstepDetector.py"),
            "--das_csv", str(infer_das_csv_path),
            "--trim_start", str(trim_start),
            "--trim_end", str(trim_end),
            "--das_fs", str(args.das_fs),
            "--das_filter_method", str(args.das_filter_method),
            "--output_dir", str(results_output_dir),
            "--load_model", str(model_path),
        ]
        if args.disable_das_bandpass:
            infer_cmd.append("--disable_das_bandpass")
        
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
    else:
        print("\n[SKIP] --replay_only 已开启，跳过推理阶段")

    # ===== 8. 按在线格式回放发送 =====
    if (args.stream_after_infer or args.replay_only) and not args.dry_run:
        base_name = Path(infer_das_csv_path).stem
        steps_infer_csv = results_output_dir / f"{base_name}_steps_inference.csv"
        steps_default_csv = results_output_dir / f"{base_name}_steps.csv"
        steps_csv_path = steps_infer_csv if steps_infer_csv.exists() else steps_default_csv
        if not steps_csv_path.exists():
            print(f"[ERROR] 未找到可回放的历史推理结果: {steps_infer_csv.name} / {steps_default_csv.name}")
            print("        可先运行一次正常推理，或调整 --double/--mirror_only/--channel_shift 对应到已存在结果")
            return 1
        print(f"[STREAM] 使用事件文件: {steps_csv_path.name}")
        stream_offline_results(args, infer_das_csv_path, steps_csv_path, trim_start, trim_end)
    
    # ===== 9. 完成 =====
    print("\n" + "="*60)
    print("推理完成！")
    print("="*60)
    print(f"DAS信号CSV: {das_csv_path}")
    if infer_das_csv_path != das_csv_path:
        print(f"推理输入CSV: {infer_das_csv_path}")
    if model_path is not None:
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
