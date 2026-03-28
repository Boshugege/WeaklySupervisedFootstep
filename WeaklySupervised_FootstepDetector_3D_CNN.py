#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简洁版多频带 3D-CNN 弱监督脚步检测
================================

特点：
- 使用多频带（默认 5-10, 10-20, 20-50, 50-100 Hz）
- 频带维度 + 时间维度 + 通道维度输入 3D-CNN
- 推理在 2D(通道×时间) 得分图上做局部峰值检测，支持同一时刻多通道事件
- 支持两种输入方式：
  1) 直接传 --das_csv
  2) 传 --name 自动从 TDMS 按 Airtag 时段切片并生成 CSV

示例：
  # 自动提取 + 训练 + 推理
  python WeaklySupervised_FootstepDetector_3D_CNN.py --name wangdihai

  # 直接CSV + 音频训练
  python WeaklySupervised_FootstepDetector_3D_CNN.py --das_csv output/wangdihai/signals/wangdihai.csv --audio Data/Audio/wangdihai.mp3

  # 仅推理
  python WeaklySupervised_FootstepDetector_3D_CNN.py --das_csv output/wangdihai/signals/wangdihai.csv --load_model output/3d_cnn/wangdihai_3dcnn.pt --inference_only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from scipy.ndimage import maximum_filter, gaussian_filter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "Data"
DEFAULT_VIDEO_DIR = DATA_DIR / "Video"
DEFAULT_AIRTAG_DIR = DATA_DIR / "Airtag"
DEFAULT_TDMS_DIR = DATA_DIR / "DAS"
DEFAULT_AUDIO_DIR = DATA_DIR / "Audio"
DEFAULT_SIGNALS_ROOT = SCRIPT_DIR / "output"


@dataclass
class Config:
    das_fs: int = 2000
    trim_start: float = 0.0
    trim_end: float | None = None

    bands: Tuple[Tuple[float, float], ...] = (
        (5.0, 10.0),
        (10.0, 20.0),
        (20.0, 50.0),
        (50.0, 100.0),
    )
    band_order: int = 4

    st_energy_win_ms: float = 80.0
    st_energy_step_ms: float = 20.0

    audio_sr: int = 48000
    audio_bp_low: float = 4000.0
    audio_bp_high: float = 10000.0
    audio_env_ms: float = 15.0
    audio_smooth_ms: float = 30.0
    audio_peak_prom: float = 1.5
    audio_peak_height: float = 0.8
    step_min_interval: float = 0.45
    weak_sigma: float = 0.12

    patch_frames: int = 31
    epochs: int = 24
    batch_size: int = 96
    lr: float = 1e-3
    weight_decay: float = 1e-4

    score_threshold: float = 0.40
    peak_time_dist_s: float = 0.20
    peak_channel_dist: int = 2
    use_adaptive_threshold: bool = True
    adaptive_threshold_quantile: float = 99.2
    topk_per_time: int = 2
    max_events: int = 600

    track_time_gap_s: float = 0.35
    track_channel_gap: int = 8
    min_track_len: int = 3
    strong_confidence_quantile: float = 99.8

    aux_center_loss_weight: float = 0.20
    aux_center_halfwidth: int = 1


def robust_zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad + eps)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, kernel, mode="same")


def short_time_rms(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win, dtype=np.float64) / float(win)
    ma = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ma + 1e-12)


def butter_bandpass(low: float, high: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    low_n = max(1e-6, low / nyq)
    high_n = min(0.9999, high / nyq)
    return butter(order, [low_n, high_n], btype="band")


def bandpass_filter_1d(x: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs, order=order)
    x0 = x.astype(np.float64) - np.mean(x)
    return filtfilt(b, a, x0)


def bandpass_filter_2d(x: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs, order=order)
    out = np.zeros_like(x, dtype=np.float64)
    for channel_index in range(x.shape[1]):
        col = x[:, channel_index].astype(np.float64) - np.mean(x[:, channel_index])
        out[:, channel_index] = filtfilt(b, a, col)
    return out


def parse_band_list(bands_text: str) -> Tuple[Tuple[float, float], ...]:
    segments = [s.strip() for s in str(bands_text).split(",") if s.strip()]
    out: List[Tuple[float, float]] = []
    for seg in segments:
        if "-" not in seg:
            raise ValueError(f"非法频带格式: {seg}，应形如 5-10,10-20")
        low_s, high_s = seg.split("-", 1)
        low = float(low_s)
        high = float(high_s)
        if high <= low:
            raise ValueError(f"频带上限必须大于下限: {seg}")
        out.append((low, high))
    if not out:
        raise ValueError("频带列表为空")
    return tuple(out)


def read_audio_mono(audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    ext = Path(audio_path).suffix.lower()
    direct_exts = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".au", ".caf", ".mp3", ".m4a"}

    if ext in direct_exts:
        y, sr = sf.read(audio_path, always_2d=True)
        y = np.mean(y, axis=1).astype(np.float32)
    else:
        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name
            cmd = ["ffmpeg", "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", str(target_sr), tmp_wav]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError("音频抽取失败，请安装ffmpeg或改用wav/flac/mp3")
            y, sr = sf.read(tmp_wav, always_2d=True)
            y = np.mean(y, axis=1).astype(np.float32)
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    sr = int(sr)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        y = resample_poly(y, up, down).astype(np.float32)
        sr = target_sr
    return y, sr


def detect_audio_steps(audio_path: str, cfg: Config, trim_start: float, trim_end: float | None) -> np.ndarray:
    print(f"[Audio] Loading: {audio_path}")
    y, sr = read_audio_mono(audio_path, cfg.audio_sr)

    start_sample = int(max(0.0, trim_start) * sr)
    end_sample = int(trim_end * sr) if trim_end is not None else len(y)
    end_sample = min(end_sample, len(y))
    y = y[start_sample:end_sample]

    y_bp = bandpass_filter_1d(y, sr, cfg.audio_bp_low, cfg.audio_bp_high, cfg.band_order)
    env = short_time_rms(y_bp, int(cfg.audio_env_ms * 1e-3 * sr))
    env = moving_average(env, int(cfg.audio_smooth_ms * 1e-3 * sr))
    t = np.arange(len(env), dtype=np.float64) / float(sr)

    z = robust_zscore(np.log(env + 1e-12))
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0 / sr
    z = moving_average(z, max(1, int(0.02 / max(dt, 1e-6))))

    min_dist = max(1, int(cfg.step_min_interval / max(dt, 1e-6)))
    peaks, _ = find_peaks(z, distance=min_dist, prominence=cfg.audio_peak_prom, height=cfg.audio_peak_height)
    step_times = t[peaks]
    print(f"[Audio] Weak labels: {len(step_times)}")
    return step_times


def find_video_for_name(name: str, video_dir: Path) -> Path | None:
    for ext in [".MP4", ".mp4", ".MOV", ".mov", ".AVI", ".avi"]:
        p = video_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


def find_audio_for_name(name: str) -> Path | None:
    audio_candidates = [
        DEFAULT_AUDIO_DIR / f"{name}.mp3",
        DEFAULT_AUDIO_DIR / f"{name}.wav",
        DEFAULT_AUDIO_DIR / f"{name}.flac",
        DEFAULT_AUDIO_DIR / f"{name}.m4a",
        DEFAULT_VIDEO_DIR / f"{name}.mp4",
        DEFAULT_VIDEO_DIR / f"{name}.MP4",
        DEFAULT_VIDEO_DIR / f"{name}.mov",
        DEFAULT_VIDEO_DIR / f"{name}.MOV",
    ]
    for p in audio_candidates:
        if p.exists():
            return p
    return None


def extract_das_csv_by_name(
    name: str,
    fs: int,
    signals_root: Path,
    video_dir: Path,
    airtag_dir: Path,
    tdms_dir: Path,
    overwrite: bool,
) -> Path:
    output_signal_dir = signals_root / name / "signals"
    output_signal_dir.mkdir(parents=True, exist_ok=True)
    das_csv = output_signal_dir / f"{name}.csv"

    if das_csv.exists() and not overwrite:
        print(f"[Extract] Reuse existing CSV: {das_csv}")
        return das_csv

    extract_script = SCRIPT_DIR / "extract_name_signals_from_tdms.py"
    if not extract_script.exists():
        raise FileNotFoundError(f"未找到提取脚本: {extract_script}")

    cmd = [
        str(Path(os.sys.executable)), str(extract_script),
        "--video-dir", str(video_dir),
        "--airtag-csv-dir", str(airtag_dir),
        "--tdms-dir", str(tdms_dir),
        "--output-dir", str(output_signal_dir),
        "--fs", str(fs),
        "--csv-utc-offset-hours", "8.0",
        "--name", name,
        "--skip-channels", "18",
    ]

    if overwrite:
        cmd.append("--overwrite")

    video_path = find_video_for_name(name, video_dir)
    if video_path is None:
        cmd.append("--use-airtag-only")
        print(f"[Extract] No video for {name}, use --use-airtag-only")
    else:
        print(f"[Extract] Video found: {video_path}")

    print(f"[Extract] Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if ret.returncode != 0:
        raise RuntimeError("DAS切片提取失败")
    if not das_csv.exists():
        raise RuntimeError(f"提取完成但未找到CSV: {das_csv}")

    return das_csv


def load_das_csv(das_csv: str, cfg: Config) -> Tuple[np.ndarray, List[int], List[str]]:
    print(f"[DAS] Loading: {das_csv}")
    df = pd.read_csv(das_csv)
    values = df.values.astype(np.float64)

    channel_ids: List[int] = []
    for col in df.columns:
        if str(col).startswith("ch_"):
            try:
                channel_ids.append(int(str(col).split("_", 1)[1]))
            except Exception:
                channel_ids.append(len(channel_ids))
        else:
            channel_ids.append(len(channel_ids))

    start_idx = int(max(0.0, cfg.trim_start) * cfg.das_fs)
    end_idx = int(cfg.trim_end * cfg.das_fs) if cfg.trim_end is not None else values.shape[0]
    end_idx = min(end_idx, values.shape[0])
    values = values[start_idx:end_idx, :]
    print(f"[DAS] Shape after trim: {values.shape}")
    return values, channel_ids, list(df.columns)


def compute_energy_map(filtered_das: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    win = max(1, int(cfg.st_energy_win_ms * 1e-3 * cfg.das_fs))
    step = max(1, int(cfg.st_energy_step_ms * 1e-3 * cfg.das_fs))
    total_samples = filtered_das.shape[0]

    frames = []
    frame_times = []
    for start in range(0, total_samples - win + 1, step):
        seg = filtered_das[start:start + win, :]
        energy = np.sum(seg ** 2, axis=0)
        frames.append(energy)
        frame_times.append((start + win // 2) / float(cfg.das_fs))

    energy_map = np.array(frames, dtype=np.float64).T  # [C,T]
    return energy_map, np.array(frame_times, dtype=np.float64)


def build_multiband_energy_maps(das: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    maps = []
    frame_times = None

    for low, high in cfg.bands:
        print(f"[DAS] Bandpass {low}-{high}Hz")
        das_bp = bandpass_filter_2d(das, cfg.das_fs, low, high, cfg.band_order)
        energy_map, times = compute_energy_map(das_bp, cfg)

        log_map = np.log10(energy_map + 1e-12)
        log_map = (log_map - np.mean(log_map)) / (np.std(log_map) + 1e-9)
        maps.append(log_map.astype(np.float32))

        if frame_times is None:
            frame_times = times

    stacked = np.stack(maps, axis=0)  # [B,C,T]
    return stacked, frame_times if frame_times is not None else np.array([], dtype=np.float64)


class Footstep3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)          # [N,32,B,T,C]
        vol = self.head(feat)            # [N,1,B,T,C]
        map2d = torch.amax(vol, dim=2)   # [N,1,T,C]，沿频带维聚合
        pooled = torch.amax(map2d, dim=(2, 3))
        return pooled.squeeze(1), map2d.squeeze(1)


def make_soft_labels(frame_times: np.ndarray, step_times: np.ndarray, sigma: float) -> np.ndarray:
    soft = np.zeros_like(frame_times, dtype=np.float64)
    for step_time in step_times:
        g = np.exp(-0.5 * ((frame_times - step_time) / sigma) ** 2)
        soft = np.maximum(soft, g)
    return soft


def build_training_patches(
    multiband_map: np.ndarray,
    frame_times: np.ndarray,
    weak_step_times: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    # multiband_map: [B,C,T]
    half = cfg.patch_frames // 2
    soft = make_soft_labels(frame_times, weak_step_times, cfg.weak_sigma)

    valid_centers = np.arange(half, len(frame_times) - half, dtype=np.int32)
    pos_idx = valid_centers[soft[valid_centers] >= 0.45]
    neg_pool = valid_centers[soft[valid_centers] <= 0.05]

    if len(pos_idx) == 0:
        raise RuntimeError("弱标签正样本为0，请检查音频质量、时间裁剪或阈值")
    if len(neg_pool) == 0:
        raise RuntimeError("可用负样本为0，请调整参数")

    n_neg = min(len(neg_pool), max(len(pos_idx) * 2, 200))
    neg_idx = np.random.choice(neg_pool, n_neg, replace=False)

    sample_idx = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate([
        np.ones(len(pos_idx), dtype=np.float32),
        np.zeros(len(neg_idx), dtype=np.float32),
    ])

    patches = []
    for center in sample_idx:
        left = center - half
        right = center + half + 1
        patch = multiband_map[:, :, left:right]       # [B,C,T]
        patch = np.transpose(patch, (0, 2, 1))        # [B,T,C]
        patches.append(patch.astype(np.float32))

    x = np.stack(patches, axis=0)                     # [N,B,T,C]
    x = x[:, None, :, :, :]                           # [N,1,B,T,C]

    perm = np.random.permutation(len(labels))
    x = x[perm]
    labels = labels[perm]

    print(f"[Train] positive={len(pos_idx)}, negative={len(neg_idx)}, patch={cfg.patch_frames}")
    return x, labels


def train_model(x: np.ndarray, y: np.ndarray, cfg: Config, device: torch.device) -> Footstep3DCNN:
    model = Footstep3DCNN().to(device)

    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    pos_weight = float((len(y) - np.sum(y)) / (np.sum(y) + 1e-6))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    map_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        total_cls_loss = 0.0
        total_map_loss = 0.0
        total = 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)

            logits, map2d = model(bx)
            cls_loss = criterion(logits, by)

            target_map = torch.zeros_like(map2d)
            pos_mask = by > 0.5
            if torch.any(pos_mask):
                energy_hint = torch.mean(bx[:, 0], dim=1)  # [N,T,C], 沿频带均值
                center_t = int(energy_hint.shape[1] // 2)
                hint = energy_hint[pos_mask, center_t, :]
                hint_min = torch.amin(hint, dim=1, keepdim=True)
                hint_max = torch.amax(hint, dim=1, keepdim=True)
                hint_norm = (hint - hint_min) / (hint_max - hint_min + 1e-6)

                half_width = max(0, int(cfg.aux_center_halfwidth))
                t0 = max(0, center_t - half_width)
                t1 = min(target_map.shape[1], center_t + half_width + 1)
                for t_index in range(t0, t1):
                    target_map[pos_mask, t_index, :] = hint_norm

            map_loss = map_criterion(map2d, target_map)
            loss = cls_loss + float(cfg.aux_center_loss_weight) * map_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * bx.shape[0]
            total_cls_loss += float(cls_loss.item()) * bx.shape[0]
            total_map_loss += float(map_loss.item()) * bx.shape[0]
            total += bx.shape[0]

        print(
            f"[Train] epoch {epoch:02d}/{cfg.epochs} "
            f"loss={total_loss / max(total, 1):.4f} "
            f"(cls={total_cls_loss / max(total, 1):.4f}, map={total_map_loss / max(total, 1):.4f})"
        )

    return model


def infer_score_map(
    model: Footstep3DCNN,
    multiband_map: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    # multiband_map: [B,C,T]
    model.eval()

    _, n_channels, n_frames = multiband_map.shape
    half = cfg.patch_frames // 2
    centers = np.arange(half, n_frames - half, dtype=np.int32)
    if len(centers) == 0:
        raise RuntimeError("有效帧不足，无法推理；请减小 --patch_frames")

    global_map = np.zeros((n_channels, n_frames), dtype=np.float64)
    global_cnt = np.zeros((n_channels, n_frames), dtype=np.float64)
    time_probs = np.zeros(n_frames, dtype=np.float64)

    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            sel = centers[start:start + batch_size]
            batch = []
            for center in sel:
                left = center - half
                right = center + half + 1
                patch = multiband_map[:, :, left:right]         # [B,C,T]
                patch = np.transpose(patch, (0, 2, 1))          # [B,T,C]
                batch.append(patch.astype(np.float32))

            bx = np.stack(batch, axis=0)[:, None, :, :, :]      # [N,1,B,T,C]
            tx = torch.from_numpy(bx).to(device)

            logits, map2d = model(tx)
            probs = torch.sigmoid(logits).cpu().numpy()         # [N]
            map_probs = torch.sigmoid(map2d).cpu().numpy()      # [N,T,C]

            for i, center in enumerate(sel):
                left = center - half
                right = center + half + 1
                patch_map = map_probs[i].T                       # [C,T]
                global_map[:, left:right] += patch_map
                global_cnt[:, left:right] += 1.0
                time_probs[center] = float(probs[i])

    valid = global_cnt > 0
    global_map[valid] = global_map[valid] / global_cnt[valid]
    global_map = gaussian_filter(global_map, sigma=(0.8, 0.8))

    return global_map, time_probs


def detect_2d_peaks(
    score_map: np.ndarray,
    frame_times: np.ndarray,
    channel_ids: List[int],
    cfg: Config,
) -> List[Tuple[float, int, float]]:
    dt = np.median(np.diff(frame_times)) if len(frame_times) > 1 else 0.02
    t_radius = max(1, int((cfg.peak_time_dist_s / max(dt, 1e-6)) / 2))
    c_radius = max(1, int(cfg.peak_channel_dist))

    if bool(cfg.use_adaptive_threshold):
        threshold = float(np.percentile(score_map, float(cfg.adaptive_threshold_quantile)))
    else:
        threshold = float(cfg.score_threshold)
    threshold = max(1e-6, threshold)

    local_max = score_map == maximum_filter(score_map, size=(2 * c_radius + 1, 2 * t_radius + 1), mode="nearest")
    mask = local_max & (score_map >= threshold)

    coords = np.argwhere(mask)  # [N,2], each row [c,t]
    if len(coords) == 0:
        print(f"[Detect] threshold={threshold:.4f}, no candidates")
        return []

    scores = score_map[coords[:, 0], coords[:, 1]]
    order = np.argsort(scores)[::-1]

    # 每个时间帧保留top-k，避免单帧过多离散通道噪声
    topk_per_time = max(1, int(cfg.topk_per_time))
    time_bucket_counts: dict[int, int] = {}
    filtered_candidates: List[Tuple[int, int, float]] = []
    for idx in order:
        c, t = int(coords[idx, 0]), int(coords[idx, 1])
        s = float(scores[idx])
        used = time_bucket_counts.get(t, 0)
        if used >= topk_per_time:
            continue
        time_bucket_counts[t] = used + 1
        filtered_candidates.append((c, t, s))

    if not filtered_candidates:
        print(f"[Detect] threshold={threshold:.4f}, no filtered candidates")
        return []

    # 轨迹连续性过滤：优先保留能组成连续路径的点，降低离群点影响
    time_gap_frames = max(1, int(float(cfg.track_time_gap_s) / max(dt, 1e-6)))
    channel_gap = max(1, int(cfg.track_channel_gap))
    min_track_len = max(1, int(cfg.min_track_len))

    filtered_candidates.sort(key=lambda x: x[1])
    tracks: List[List[Tuple[int, int, float]]] = []

    for c, t, s in filtered_candidates:
        best_track_index = -1
        best_cost = float("inf")
        for track_index, track in enumerate(tracks):
            last_c, last_t, _ = track[-1]
            dt_frames = t - last_t
            dc = abs(c - last_c)
            if dt_frames < 0 or dt_frames > time_gap_frames or dc > channel_gap:
                continue
            cost = dt_frames + 0.5 * dc
            if cost < best_cost:
                best_cost = cost
                best_track_index = track_index
        if best_track_index >= 0:
            tracks[best_track_index].append((c, t, s))
        else:
            tracks.append([(c, t, s)])

    selected: List[Tuple[int, int, float]] = []
    all_scores = np.array([x[2] for x in filtered_candidates], dtype=np.float64)
    strong_threshold = float(np.percentile(all_scores, float(cfg.strong_confidence_quantile)))

    for track in tracks:
        if len(track) >= min_track_len:
            selected.extend(track)
        else:
            for item in track:
                if item[2] >= strong_threshold:
                    selected.append(item)

    if not selected:
        selected = filtered_candidates

    selected.sort(key=lambda x: x[2], reverse=True)
    max_events = max(1, int(cfg.max_events))
    selected = selected[:max_events]

    events = []
    for c, t, s in selected:
        sec = float(frame_times[t])
        ch = int(channel_ids[c]) if c < len(channel_ids) else int(c)
        events.append((sec, ch, s))

    events.sort(key=lambda x: x[0])
    print(
        f"[Detect] threshold={threshold:.4f}, raw={len(coords)}, topk={len(filtered_candidates)}, "
        f"tracks={len(tracks)}, final={len(events)}"
    )
    return events


def save_outputs(
    output_dir: Path,
    base_name: str,
    score_map: np.ndarray,
    frame_times: np.ndarray,
    events: List[Tuple[float, int, float]],
    channel_ids: List[int],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{base_name}_steps_3dcnn.csv"
    pd.DataFrame(events, columns=["time", "channel", "confidence"]).to_csv(csv_path, index=False)
    print(f"[Output] {csv_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1]})
    ch_min = min(channel_ids) if channel_ids else 0
    ch_max = max(channel_ids) if channel_ids else score_map.shape[0] - 1

    ax1.imshow(
        score_map,
        aspect="auto",
        origin="lower",
        extent=[frame_times[0], frame_times[-1], ch_min, ch_max],
        cmap="viridis",
    )

    if events:
        times = [e[0] for e in events]
        channels = [e[1] for e in events]
        conf = [e[2] for e in events]
        sc = ax1.scatter(times, channels, c=conf, cmap="hot", s=40, edgecolors="white", linewidths=0.6)
        fig.colorbar(sc, ax=ax1, label="confidence")

    ax1.set_title("3D-CNN score map + detected events")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Channel")

    ax2.plot(frame_times, np.max(score_map, axis=0), color="steelblue", linewidth=1.2)
    ax2.set_title("Max-over-channel score")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("score")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / f"{base_name}_heatmap_3dcnn.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] {fig_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Multiband 3D-CNN weakly supervised footstep detector")

    p.add_argument("--name", default=None, help="目标名字；提供后可自动从TDMS切片生成CSV")
    p.add_argument("--das_csv", default=None, help="DAS CSV路径；若不给则需--name自动提取")
    p.add_argument("--audio", default=None, help="音频/视频路径（训练时必需，缺省会按name自动查找）")

    p.add_argument("--output_dir", default=None, help="输出目录；默认 output/3d_cnn/<name或csv_stem>")
    p.add_argument("--save_model", default=None, help="保存模型(.pt)")
    p.add_argument("--load_model", default=None, help="加载模型(.pt)")
    p.add_argument("--inference_only", action="store_true", help="仅推理（需--load_model）")

    p.add_argument("--trim_start", type=float, default=0.0)
    p.add_argument("--trim_end", type=float, default=None)
    p.add_argument("--das_fs", type=int, default=2000)

    p.add_argument("--bands", type=str, default="5-10,10-20,20-50,50-100", help="多频带列表")
    p.add_argument("--patch_frames", type=int, default=31)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--score_threshold", type=float, default=0.40)
    p.add_argument("--peak_time_dist_s", type=float, default=0.20)
    p.add_argument("--peak_channel_dist", type=int, default=2)
    p.add_argument("--disable_adaptive_threshold", action="store_true", help="关闭分位数自适应阈值，改用固定score_threshold")
    p.add_argument("--adaptive_threshold_quantile", type=float, default=99.2, help="自适应阈值分位数(0-100)")
    p.add_argument("--topk_per_time", type=int, default=2, help="每个时间帧最多保留的通道峰值数")
    p.add_argument("--max_events", type=int, default=600, help="最终最多输出事件数")

    p.add_argument("--track_time_gap_s", type=float, default=0.35, help="轨迹连接最大时间间隔(秒)")
    p.add_argument("--track_channel_gap", type=int, default=8, help="轨迹连接最大通道差")
    p.add_argument("--min_track_len", type=int, default=3, help="保留轨迹最短长度")
    p.add_argument("--strong_confidence_quantile", type=float, default=99.8, help="短轨迹保留的高置信分位数")

    p.add_argument("--aux_center_loss_weight", type=float, default=0.20, help="中心时刻辅助定位损失权重")
    p.add_argument("--aux_center_halfwidth", type=int, default=1, help="中心监督半宽(帧)")

    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    # 自动提取相关
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--airtag_dir", default=str(DEFAULT_AIRTAG_DIR))
    p.add_argument("--tdms_dir", default=str(DEFAULT_TDMS_DIR))
    p.add_argument("--signals_root", default=str(DEFAULT_SIGNALS_ROOT), help="自动切片CSV根目录")
    p.add_argument("--overwrite_extract", action="store_true", help="重新提取CSV，即使已有同名文件")

    return p.parse_args()


def pick_device(pref: str) -> torch.device:
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_input_paths(args) -> Tuple[Path, Path | None]:
    das_csv_path: Path

    if args.das_csv:
        das_csv_path = Path(args.das_csv)
    else:
        if not args.name:
            raise ValueError("未提供 --das_csv 时必须提供 --name")
        name = args.name.lower()
        das_csv_path = extract_das_csv_by_name(
            name=name,
            fs=int(args.das_fs),
            signals_root=Path(args.signals_root),
            video_dir=Path(args.video_dir),
            airtag_dir=Path(args.airtag_dir),
            tdms_dir=Path(args.tdms_dir),
            overwrite=bool(args.overwrite_extract),
        )

    if not das_csv_path.exists():
        raise FileNotFoundError(f"DAS CSV不存在: {das_csv_path}")

    if args.audio:
        audio_path = Path(args.audio)
    elif args.name:
        audio_path = find_audio_for_name(args.name.lower())
    else:
        audio_path = None

    if audio_path is not None and not audio_path.exists():
        raise FileNotFoundError(f"音频/视频路径不存在: {audio_path}")

    return das_csv_path, audio_path


def main():
    args = parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = Config(
        das_fs=int(args.das_fs),
        trim_start=float(args.trim_start),
        trim_end=args.trim_end,
        bands=parse_band_list(args.bands),
        patch_frames=max(9, int(args.patch_frames) | 1),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        score_threshold=float(args.score_threshold),
        peak_time_dist_s=float(args.peak_time_dist_s),
        peak_channel_dist=int(args.peak_channel_dist),
        use_adaptive_threshold=not bool(args.disable_adaptive_threshold),
        adaptive_threshold_quantile=float(args.adaptive_threshold_quantile),
        topk_per_time=int(args.topk_per_time),
        max_events=int(args.max_events),
        track_time_gap_s=float(args.track_time_gap_s),
        track_channel_gap=int(args.track_channel_gap),
        min_track_len=int(args.min_track_len),
        strong_confidence_quantile=float(args.strong_confidence_quantile),
        aux_center_loss_weight=float(args.aux_center_loss_weight),
        aux_center_halfwidth=int(args.aux_center_halfwidth),
    )

    device = pick_device(args.device)
    print(f"[Device] {device}")
    print(f"[Bands] {cfg.bands}")

    das_csv_path, audio_path = resolve_input_paths(args)

    das, channel_ids, _ = load_das_csv(str(das_csv_path), cfg)
    multiband_map, frame_times = build_multiband_energy_maps(das, cfg)

    model = Footstep3DCNN().to(device)

    base_name = das_csv_path.stem
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        tag = args.name.lower() if args.name else base_name
        output_dir = SCRIPT_DIR / "output" / "3d_cnn" / tag

    model_path = args.load_model

    if not args.inference_only:
        if audio_path is None:
            raise ValueError("训练模式需要音频弱标签，请提供 --audio 或 --name 对应可解析音频")

        weak_step_times = detect_audio_steps(str(audio_path), cfg, cfg.trim_start, cfg.trim_end)
        x, y = build_training_patches(multiband_map, frame_times, weak_step_times, cfg)
        model = train_model(x, y, cfg, device)

        save_path = args.save_model
        if not save_path:
            save_path = str(output_dir / f"{base_name}_3dcnn.pt")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
        }, save_path)
        model_path = save_path
        print(f"[Model] Saved: {save_path}")
    else:
        if not args.load_model:
            raise ValueError("--inference_only 需要 --load_model")

    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Model] Loaded: {model_path}")

    score_map, _ = infer_score_map(model, multiband_map, cfg, device)
    events = detect_2d_peaks(score_map, frame_times, channel_ids, cfg)

    save_outputs(output_dir, base_name, score_map, frame_times, events, channel_ids)

    print("=" * 60)
    print("3D-CNN Footstep Detection Done")
    print(f"events: {len(events)}")
    if events:
        arr_t = np.array([e[0] for e in events])
        arr_c = np.array([e[1] for e in events])
        print(f"time range: {arr_t.min():.2f}s - {arr_t.max():.2f}s")
        print(f"channel range: {arr_c.min()} - {arr_c.max()}")
    print(f"output dir: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
