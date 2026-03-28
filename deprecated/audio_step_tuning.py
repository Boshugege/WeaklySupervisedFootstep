#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频脚步检测调优工具
====================

用于验证音频中脚步检测的可信度，在进行DAS弱监督训练之前确认音频标签质量。

输出：
1. 音频包络曲线 + 检测到的脚步标记
2. 音频波形片段 + 脚步位置
3. 脚步间隔统计
4. 交互式调参建议
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[ERROR] librosa is required. Install with: pip install librosa")

# ============================================================================
# 信号处理函数
# ============================================================================
def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low_n = max(1e-6, low / nyq)
    high_n = min(0.9999, high / nyq)
    b, a = butter(order, [low_n, high_n], btype='band')
    return b, a

def bandpass_filter(x, fs, low, high, order=4):
    b, a = butter_bandpass(low, high, fs, order=order)
    return filtfilt(b, a, x - np.mean(x))

def short_time_rms(x, win):
    win = max(1, int(win))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / float(win)
    ma = np.convolve(x2, kernel, mode='same')
    return np.sqrt(ma + 1e-12)

def moving_average(x, win):
    if win <= 1:
        return x
    kernel = np.ones(int(win)) / float(win)
    return np.convolve(x, kernel, mode='same')

def robust_zscore(x, eps=1e-9):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad + eps)


# ============================================================================
# 音频脚步检测类
# ============================================================================
class AudioStepDetectorTuner:
    """音频脚步检测调优器"""
    
    def __init__(self, audio_path, trim_start=0.0, trim_end=None):
        self.audio_path = audio_path
        self.trim_start = trim_start
        self.trim_end = trim_end
        
        # 默认参数
        self.params = {
            'sr': 48000,              # 采样率
            'bp_low': 4000,           # 带通低频 (Hz)
            'bp_high': 10000,         # 带通高频 (Hz)
            'bp_order': 4,            # 滤波器阶数
            'env_ms': 15,             # RMS包络窗口 (ms)
            'smooth_ms': 30,          # 平滑窗口 (ms)
            'min_interval': 0.30,     # 最小脚步间隔 (s)
            'peak_prom': 1.5,         # 峰值显著性
            'peak_height': 0.8,       # 峰值高度阈值
        }
        
        self.y = None
        self.sr = None
        self.t = None
        self.y_bp = None
        self.envelope = None
        self.z_envelope = None
        self.step_times = None
        self.step_peaks = None
        
    def load_audio(self):
        """加载音频"""
        print(f"[Audio] Loading: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=self.params['sr'], mono=True)
        
        # 时间裁剪
        start_idx = int(self.trim_start * self.sr)
        end_idx = int(self.trim_end * self.sr) if self.trim_end else len(self.y)
        self.y = self.y[start_idx:end_idx]
        
        self.t = np.arange(len(self.y)) / self.sr
        print(f"[Audio] Duration: {len(self.y)/self.sr:.2f}s, Samples: {len(self.y)}")
        
    def detect_steps(self):
        """检测脚步"""
        # 带通滤波
        self.y_bp = bandpass_filter(self.y, self.sr, 
                                    self.params['bp_low'], 
                                    self.params['bp_high'],
                                    self.params['bp_order'])
        
        # RMS包络
        env_win = max(1, int(self.params['env_ms'] * 1e-3 * self.sr))
        self.envelope = short_time_rms(self.y_bp, env_win)
        
        # 平滑
        smooth_win = max(1, int(self.params['smooth_ms'] * 1e-3 * self.sr))
        env_smooth = moving_average(self.envelope, smooth_win)
        
        # 对数Z-score
        log_env = np.log(env_smooth + 1e-12)
        self.z_envelope = robust_zscore(log_env)
        
        # 进一步平滑
        z_smooth = moving_average(self.z_envelope, max(1, int(0.02 * self.sr)))
        
        # 峰值检测
        dt = 1.0 / self.sr
        min_dist = max(1, int(self.params['min_interval'] / dt))
        
        peaks, props = find_peaks(z_smooth, 
                                  distance=min_dist,
                                  prominence=self.params['peak_prom'],
                                  height=self.params['peak_height'])
        
        self.step_peaks = peaks
        self.step_times = self.t[peaks]
        self.step_heights = props.get('peak_heights', z_smooth[peaks])
        
        print(f"[Audio] Detected {len(self.step_times)} steps")
        
        return self.step_times
    
    def update_params(self, **kwargs):
        """更新参数"""
        self.params.update(kwargs)
        
    def plot_tuning_overview(self, output_path):
        """绘制调优总览图"""
        fig = plt.figure(figsize=(18, 14))
        
        # 1. 原始波形 + 带通滤波后波形
        ax1 = plt.subplot(4, 1, 1)
        # 降采样显示
        ds = max(1, len(self.t) // 10000)
        ax1.plot(self.t[::ds], self.y[::ds], 'b-', alpha=0.5, linewidth=0.5, label='Original')
        ax1.plot(self.t[::ds], self.y_bp[::ds], 'r-', alpha=0.7, linewidth=0.5, 
                label=f'Bandpass {self.params["bp_low"]}-{self.params["bp_high"]}Hz')
        for t_step in self.step_times:
            ax1.axvline(t_step, color='lime', alpha=0.6, linewidth=1)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Audio Waveform (green lines = detected steps, n={len(self.step_times)})')
        ax1.legend(loc='upper right')
        ax1.set_xlim(self.t[0], self.t[-1])
        
        # 2. RMS包络 + Z-score
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(self.t[::ds], self.z_envelope[::ds], 'g-', linewidth=0.8, label='Z-score Envelope')
        ax2.axhline(self.params['peak_height'], color='orange', linestyle='--', 
                   label=f'Height threshold = {self.params["peak_height"]}')
        for t_step in self.step_times:
            ax2.axvline(t_step, color='red', alpha=0.6, linewidth=1)
        ax2.set_ylabel('Z-score')
        ax2.set_title('Log RMS Envelope (Z-score normalized)')
        ax2.legend(loc='upper right')
        ax2.set_xlim(self.t[0], self.t[-1])
        ax2.grid(True, alpha=0.3)
        
        # 3. 脚步间隔分析
        ax3 = plt.subplot(4, 2, 5)
        if len(self.step_times) > 1:
            intervals = np.diff(self.step_times)
            ax3.hist(intervals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax3.axvline(np.median(intervals), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(intervals):.3f}s')
            ax3.axvline(np.mean(intervals), color='orange', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(intervals):.3f}s')
            ax3.axvline(self.params['min_interval'], color='green', linestyle=':',
                       label=f'Min interval: {self.params["min_interval"]}s')
            ax3.legend()
            ax3.set_xlabel('Step Interval (s)')
            ax3.set_ylabel('Count')
            ax3.set_title('Step Interval Distribution')
        
        # 4. 峰值高度分布
        ax4 = plt.subplot(4, 2, 6)
        if len(self.step_heights) > 0:
            ax4.hist(self.step_heights, bins=20, color='coral', alpha=0.7, edgecolor='black')
            ax4.axvline(self.params['peak_height'], color='green', linestyle='--',
                       label=f'Height threshold: {self.params["peak_height"]}')
            ax4.legend()
            ax4.set_xlabel('Peak Height (Z-score)')
            ax4.set_ylabel('Count')
            ax4.set_title('Step Peak Height Distribution')
        
        # 5. 详细片段视图（选取几个典型片段）
        ax5 = plt.subplot(4, 1, 4)
        # 选取中间10秒
        mid_time = (self.t[0] + self.t[-1]) / 2
        t_start = mid_time - 5
        t_end = mid_time + 5
        
        mask = (self.t >= t_start) & (self.t <= t_end)
        if np.any(mask):
            ax5.plot(self.t[mask], self.z_envelope[mask], 'g-', linewidth=1, label='Z-score Envelope')
            ax5.axhline(self.params['peak_height'], color='orange', linestyle='--', alpha=0.7)
            
            # 标记这个时间段内的脚步
            segment_steps = self.step_times[(self.step_times >= t_start) & (self.step_times <= t_end)]
            for t_step in segment_steps:
                ax5.axvline(t_step, color='red', alpha=0.8, linewidth=2)
                ax5.annotate(f'{t_step:.2f}s', (t_step, self.params['peak_height'] + 0.5),
                           rotation=90, fontsize=8, ha='center')
            
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Z-score')
            ax5.set_title(f'Detail View: {t_start:.1f}s - {t_end:.1f}s (red lines = steps)')
            ax5.set_xlim(t_start, t_end)
            ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Output] Saved: {output_path}")
    
    def plot_step_segments(self, output_path, n_segments=12):
        """绘制脚步附近的波形片段"""
        if len(self.step_times) == 0:
            print("[WARNING] No steps detected, skipping segment plot")
            return
        
        # 选取均匀分布的脚步
        n_steps = len(self.step_times)
        indices = np.linspace(0, n_steps - 1, min(n_segments, n_steps), dtype=int)
        selected_steps = self.step_times[indices]
        
        n_cols = 4
        n_rows = (len(selected_steps) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        window_s = 0.5  # 显示脚步前后0.5秒
        
        for idx, (ax, t_step) in enumerate(zip(axes, selected_steps)):
            t_start = t_step - window_s
            t_end = t_step + window_s
            
            mask = (self.t >= t_start) & (self.t <= t_end)
            if not np.any(mask):
                continue
            
            # 绘制带通滤波后的波形
            ax.plot(self.t[mask] - t_step, self.y_bp[mask], 'b-', linewidth=0.8)
            ax.axvline(0, color='red', linewidth=2, label='Step')
            ax.fill_betweenx([-1, 1], -0.05, 0.05, color='red', alpha=0.2)
            
            ax.set_title(f'Step #{idx+1} @ {t_step:.2f}s', fontsize=10)
            ax.set_xlabel('Time relative to step (s)')
            ax.set_xlim(-window_s, window_s)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for ax in axes[len(selected_steps):]:
            ax.set_visible(False)
        
        plt.suptitle(f'Step Waveform Segments (Bandpass {self.params["bp_low"]}-{self.params["bp_high"]}Hz)', 
                    fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Output] Saved: {output_path}")
    
    def plot_das_overlay(self, das_csv_path, output_path, das_bp_low=5, das_bp_high=10):
        """绘制DAS数据叠加图"""
        print(f"[DAS] Loading: {das_csv_path}")
        df = pd.read_csv(das_csv_path)
        das = df.values.astype(np.float64)
        
        das_fs = 2000
        
        # 时间裁剪
        start_idx = int(self.trim_start * das_fs)
        end_idx = int(self.trim_end * das_fs) if self.trim_end else das.shape[0]
        das = das[start_idx:end_idx, :]
        
        print(f"[DAS] Shape: {das.shape}")
        
        # 5-10Hz带通滤波
        b, a = butter_bandpass(das_bp_low, das_bp_high, das_fs, order=4)
        das_bp = np.zeros_like(das)
        for c in range(das.shape[1]):
            col = das[:, c] - np.mean(das[:, c])
            das_bp[:, c] = filtfilt(b, a, col)
        
        # 计算短时能量
        win_ms = 50
        step_ms = 25
        win_samples = int(win_ms * 1e-3 * das_fs)
        step_samples = int(step_ms * 1e-3 * das_fs)
        
        T, C = das_bp.shape
        frames = []
        frame_times = []
        
        for start in range(0, T - win_samples + 1, step_samples):
            frame = das_bp[start:start + win_samples, :]
            energy = np.sum(frame ** 2, axis=0)
            frames.append(energy)
            frame_times.append((start + win_samples // 2) / das_fs)
        
        energy_matrix = np.array(frames).T  # [C, n_frames]
        frame_times = np.array(frame_times)
        
        # 绘制叠加图
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), 
                                  gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. DAS热图 + 脚步标记
        ax1 = axes[0]
        log_energy = np.log10(energy_matrix + 1e-10)
        im = ax1.imshow(log_energy, aspect='auto', origin='lower',
                       extent=[frame_times[0], frame_times[-1], 0, C],
                       cmap='viridis', interpolation='bilinear')
        
        # 标记音频检测的脚步
        for t_step in self.step_times:
            ax1.axvline(t_step, color='red', alpha=0.7, linewidth=1.5)
        
        ax1.set_ylabel('Channel')
        ax1.set_title(f'DAS Energy Heatmap ({das_bp_low}-{das_bp_high}Hz) + Audio Steps (red lines, n={len(self.step_times)})')
        plt.colorbar(im, ax=ax1, label='Log Energy', shrink=0.8)
        
        # 2. DAS总能量曲线（所有通道叠加）
        ax2 = axes[1]
        total_energy = np.sum(energy_matrix, axis=0)
        total_energy_z = robust_zscore(np.log(total_energy + 1e-12))
        ax2.plot(frame_times, total_energy_z, 'g-', linewidth=0.8, label='DAS Total Energy (Z-score)')
        
        for t_step in self.step_times:
            ax2.axvline(t_step, color='red', alpha=0.5, linewidth=1)
        
        ax2.set_ylabel('Z-score')
        ax2.set_title('DAS Total Energy (all channels summed)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(frame_times[0], frame_times[-1])
        
        # 3. 音频包络
        ax3 = axes[2]
        # 降采样到DAS帧率
        audio_env_ds = np.interp(frame_times, self.t, self.z_envelope)
        ax3.plot(frame_times, audio_env_ds, 'b-', linewidth=0.8, label='Audio Envelope (Z-score)')
        
        for t_step in self.step_times:
            ax3.axvline(t_step, color='red', alpha=0.5, linewidth=1)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Z-score')
        ax3.set_title(f'Audio Envelope ({self.params["bp_low"]}-{self.params["bp_high"]}Hz)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(frame_times[0], frame_times[-1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Output] Saved: {output_path}")
        
        return energy_matrix, frame_times
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("Audio Step Detection Statistics")
        print("=" * 60)
        
        print(f"\n[Parameters]")
        for key, val in self.params.items():
            print(f"  {key}: {val}")
        
        print(f"\n[Results]")
        print(f"  Total steps detected: {len(self.step_times)}")
        
        if len(self.step_times) > 1:
            intervals = np.diff(self.step_times)
            print(f"  Step intervals:")
            print(f"    Min:    {np.min(intervals):.3f}s")
            print(f"    Max:    {np.max(intervals):.3f}s")
            print(f"    Mean:   {np.mean(intervals):.3f}s")
            print(f"    Median: {np.median(intervals):.3f}s")
            print(f"    Std:    {np.std(intervals):.3f}s")
            
            # 估计步频
            avg_interval = np.mean(intervals)
            steps_per_second = 1.0 / avg_interval if avg_interval > 0 else 0
            steps_per_minute = steps_per_second * 60
            print(f"\n  Estimated cadence: {steps_per_minute:.1f} steps/min")
        
        if len(self.step_heights) > 0:
            print(f"\n  Peak heights:")
            print(f"    Min:    {np.min(self.step_heights):.2f}")
            print(f"    Max:    {np.max(self.step_heights):.2f}")
            print(f"    Mean:   {np.mean(self.step_heights):.2f}")
        
        print("\n" + "=" * 60)
        
        # 调优建议
        print("\n[Tuning Suggestions]")
        
        if len(self.step_times) > 1:
            intervals = np.diff(self.step_times)
            
            if np.min(intervals) < 0.25:
                print("  ⚠ Some intervals are very short (<0.25s).")
                print("    → Consider increasing 'min_interval' or 'peak_prom'")
            
            if np.max(intervals) > 2.0:
                print("  ⚠ Some intervals are very long (>2s), possible missed steps.")
                print("    → Consider decreasing 'peak_height' or 'peak_prom'")
            
            cv = np.std(intervals) / np.mean(intervals)
            if cv > 0.5:
                print(f"  ⚠ High variability in intervals (CV={cv:.2f}).")
                print("    → Check if detection is consistent")
        
        if len(self.step_times) < 10:
            print("  ⚠ Very few steps detected.")
            print("    → Consider lowering 'peak_height' and 'peak_prom'")
        
        print()


# ============================================================================
# 主程序
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Audio Step Detection Tuning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本使用
    python audio_step_tuning.py --audio video.mp4 --trim_start 50 --trim_end 200

    # 调整检测参数
    python audio_step_tuning.py --audio video.mp4 --peak_height 0.5 --peak_prom 1.0

    # 与DAS数据叠加
    python audio_step_tuning.py --audio video.mp4 --das_csv data.csv --trim_start 50 --trim_end 200
        """
    )
    
    parser.add_argument('--audio', '-a', required=True, help='音频/视频文件路径')
    parser.add_argument('--das_csv', '-d', default=None, help='DAS CSV文件路径（可选，用于叠加显示）')
    parser.add_argument('--output_dir', '-o', default='output/audio_tuning', help='输出目录')
    
    # 时间裁剪
    parser.add_argument('--trim_start', type=float, default=50.0, help='开始时间（秒）')
    parser.add_argument('--trim_end', type=float, default=200.0, help='结束时间（秒）')
    
    # 滤波参数
    parser.add_argument('--bp_low', type=float, default=4000, help='带通低频 (Hz)')
    parser.add_argument('--bp_high', type=float, default=10000, help='带通高频 (Hz)')
    
    # 检测参数
    parser.add_argument('--min_interval', type=float, default=0.30, help='最小脚步间隔 (s)')
    parser.add_argument('--peak_prom', type=float, default=1.5, help='峰值显著性阈值')
    parser.add_argument('--peak_height', type=float, default=0.8, help='峰值高度阈值')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建调优器
    tuner = AudioStepDetectorTuner(args.audio, args.trim_start, args.trim_end)
    
    # 更新参数
    tuner.update_params(
        bp_low=args.bp_low,
        bp_high=args.bp_high,
        min_interval=args.min_interval,
        peak_prom=args.peak_prom,
        peak_height=args.peak_height
    )
    
    # 加载和检测
    tuner.load_audio()
    tuner.detect_steps()
    
    # 打印统计
    tuner.print_statistics()
    
    # 生成图表
    base_name = os.path.splitext(os.path.basename(args.audio))[0]
    
    # 1. 调优总览图
    overview_path = os.path.join(args.output_dir, f"{base_name}_audio_tuning_overview.png")
    tuner.plot_tuning_overview(overview_path)
    
    # 2. 脚步波形片段
    segments_path = os.path.join(args.output_dir, f"{base_name}_step_segments.png")
    tuner.plot_step_segments(segments_path)
    
    # 3. 如果提供了DAS数据，绘制叠加图
    if args.das_csv:
        overlay_path = os.path.join(args.output_dir, f"{base_name}_das_audio_overlay.png")
        tuner.plot_das_overlay(args.das_csv, overlay_path)
    
    # 保存检测结果
    steps_csv = os.path.join(args.output_dir, f"{base_name}_audio_steps.csv")
    df_steps = pd.DataFrame({
        'time': tuner.step_times,
        'height': tuner.step_heights
    })
    df_steps.to_csv(steps_csv, index=False)
    print(f"[Output] Steps CSV: {steps_csv}")
    
    print("\n[DONE] Audio step tuning completed!")
    print(f"Check the output files in: {args.output_dir}")


if __name__ == "__main__":
    main()
