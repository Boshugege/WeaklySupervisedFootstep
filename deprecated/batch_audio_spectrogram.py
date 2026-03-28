#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成音频频谱图，并输出一个可用于筛选样本质量的统计表。

默认行为：
- 输入目录: Data/mp3_output
- 输出目录: output/mp3_spectrograms
- 每个音频输出同名 PNG
- 额外输出 quality_summary.csv（按质量分数降序）
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, filtfilt, find_peaks


def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.9999)
    return butter(order, [low_n, high_n], btype="band")


def bandpass_filter(x, fs, low, high, order=4):
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, x)


def moving_average(x, n):
    if n <= 1:
        return x
    kernel = np.ones(n, dtype=np.float64) / n
    return np.convolve(x, kernel, mode="same")


def robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)


def short_time_rms(x, win):
    x2 = x * x
    return np.sqrt(moving_average(x2, win) + 1e-12)


def detect_like_pipeline(y, sr, bp_low=4000, bp_high=10000, min_interval=0.30, peak_prom=1.5, peak_height=0.8):
    y_bp = bandpass_filter(y, sr, bp_low, bp_high, order=4)
    env = short_time_rms(y_bp, max(1, int(0.015 * sr)))
    env = moving_average(env, max(1, int(0.030 * sr)))
    t = np.arange(len(env)) / sr

    z = robust_zscore(np.log(env + 1e-12))
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0 / sr
    z = moving_average(z, max(1, int(0.02 / dt)))
    min_dist = max(1, int(min_interval / dt))
    peaks, props = find_peaks(z, distance=min_dist, prominence=peak_prom, height=peak_height)
    return t, env, z, peaks, props


def analyze_one(audio_path, out_png, sr=48000, fmax=12000):
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration_s = len(y) / sr if sr > 0 else 0.0

    # 频谱
    n_fft = 2048
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask = (freqs >= 4000) & (freqs <= 10000)
    full_energy = float(np.mean(S) + 1e-12)
    band_energy = float(np.mean(S[band_mask, :]) + 1e-12) if np.any(band_mask) else 1e-12
    band_energy_ratio = band_energy / full_energy

    # 与主流程一致的峰值检测代理指标
    t, env, z, peaks, _ = detect_like_pipeline(y, sr)
    peak_count = int(len(peaks))
    peak_rate_per_min = float(peak_count / max(duration_s / 60.0, 1e-9))
    snr_proxy = float(np.percentile(z, 95) - np.percentile(z, 50))

    # 绘图：上频谱，下包络+峰值
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        fmax=fmax,
        ax=ax1,
    )
    ax1.set_title(f"{audio_path.stem} | Spectrogram")
    ax1.set_ylabel("Hz")

    t_short = np.linspace(0, duration_s, num=len(z), endpoint=False)
    ax2.plot(t_short, z, linewidth=1.0, label="log-envelope z-score")
    if peak_count > 0:
        ax2.scatter(t_short[peaks], z[peaks], s=12, c="r", label="peaks")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("z")
    ax2.set_title(
        f"peaks={peak_count}, rate={peak_rate_per_min:.1f}/min, "
        f"band_ratio={band_energy_ratio:.2f}, snr_proxy={snr_proxy:.2f}"
    )
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {
        "name": audio_path.stem,
        "audio_file": str(audio_path),
        "duration_s": duration_s,
        "peak_count": peak_count,
        "peak_rate_per_min": peak_rate_per_min,
        "band_energy_ratio": band_energy_ratio,
        "snr_proxy": snr_proxy,
        "spectrogram_png": str(out_png),
    }


def normalize_col(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return v
    mn, mx = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


def add_quality_score(df):
    # 打分策略：只使用 snr_score 和 band_score，去掉步频限制
    band_score = normalize_col(df["band_energy_ratio"].to_numpy())
    snr_score = normalize_col(df["snr_proxy"].to_numpy())
    # 新权重：snr 55%，band 45%
    quality = 0.55 * snr_score + 0.45 * band_score
    df["quality_score"] = quality
    return df.sort_values("quality_score", ascending=False).reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="批量生成同名频谱图并输出音频质量排序")
    p.add_argument("--input_dir", default="Data/Audio", help="输入音频目录")
    p.add_argument("--output_dir", default="output/audio_spectrograms", help="输出目录")
    p.add_argument("--sr", type=int, default=48000, help="重采样率")
    p.add_argument("--fmax", type=float, default=12000, help="频谱图最大频率显示")
    p.add_argument("--limit", type=int, default=0, help="仅处理前N个文件（0=全部）")
    p.add_argument("--trim_profile", default="trim_profile.csv", help="输出 trim_profile 路径")
    p.add_argument("--default_trim_head", type=float, default=20.0, help="默认头部裁剪秒数")
    p.add_argument("--default_trim_tail", type=float, default=10.0, help="默认尾部裁剪秒数")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 Airtag 目录，用于过滤有效样本
    airtag_dir = Path("Data/Airtag")
    airtag_names = set()
    if airtag_dir.exists():
        airtag_names = {p.stem.lower() for p in airtag_dir.glob("*.csv") if p.is_file()}

    files = sorted([p for p in in_dir.glob("*.mp3") if p.is_file()])
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise FileNotFoundError(f"No mp3 files found in: {in_dir}")

    rows = []
    for i, f in enumerate(files, 1):
        out_png = out_dir / f"{f.stem}.png"
        print(f"[{i}/{len(files)}] {f.name} -> {out_png.name}")
        row = analyze_one(f, out_png, sr=args.sr, fmax=args.fmax)
        # 标记是否有 Airtag
        row["has_airtag"] = f.stem.lower() in airtag_names
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_quality_score(df)
    csv_path = out_dir / "quality_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\nTop 10 by quality_score:")
    show_cols = ["name", "quality_score", "snr_proxy", "band_energy_ratio", "peak_rate_per_min", "has_airtag"]
    print(df[show_cols].head(10).to_string(index=False))
    print(f"\n[Done] Spectrograms: {out_dir}")
    print(f"[Done] Summary CSV: {csv_path}")

    # ===== 自动生成 trim_profile.csv =====
    # 只保留有 Airtag 的样本，按质量分数从高到低排序
    df_valid = df[df["has_airtag"]].copy()
    if len(df_valid) > 0:
        trim_df = pd.DataFrame({
            "filename": df_valid["name"],
            "trim_head_s": args.default_trim_head,
            "trim_tail_s": args.default_trim_tail
        })
        trim_path = Path(args.trim_profile)
        trim_df.to_csv(trim_path, index=False, encoding="utf-8-sig")
        print(f"\n[Done] trim_profile.csv: {trim_path} ({len(trim_df)} 个有效样本)")
        print(f"       (按 quality_score 从高到低排序，仅包含有 Airtag 的样本)")
    else:
        print("\n[WARN] 没有找到有 Airtag 的样本，未生成 trim_profile.csv")


if __name__ == "__main__":
    main()
