#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弱监督脚步检测系统 (Weakly Supervised Footstep Detection)
==========================================================

核心思路：
1. 从音频中提取脚步候选时间（4-10kHz带通滤波）作为弱标签
2. 使用这些时间弱标签训练DAS模型（5-10Hz带通滤波）
3. 模型先学习"何时有脚步"，再推断"脚步在哪个通道"
4. 自训练迭代：用高置信预测补全漏检脚步

输入：
- DAS CSV文件（列: ch_0, ch_1, ..., ch_N，行: 时间采样点，采样率2000Hz）
- 对应的MP4视频文件（提取音频作为弱监督）

输出：
- 可视化图：时间×通道热图 + 脚步标记点 (t, c)
- 结构化CSV：time, channel, confidence
"""

import os
import argparse
import subprocess
import tempfile
from math import gcd
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
import matplotlib.pyplot as plt

import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    class _NNStub:
        Module = object
    nn = _NNStub()
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

# ============================================================================
# [1] 配置参数类
# ============================================================================
class Config:
    """全局配置参数"""
    def __init__(self):
        # ===== 时间裁剪参数（核心需求）=====
        self.trim_start_s = 0.0      # 数据起始裁剪时间（秒）
        self.trim_end_s = None       # 数据结束裁剪时间（秒），None表示到结尾
        
        # ===== DAS参数 =====
        self.das_fs = 2000           # DAS采样率 (Hz)
        self.das_bp_bands = [        # 多频带滤波配置
            (5, 10),                 # 主频带（用户指定）
            (10, 20),                # 高频补充
        ]
        self.das_filter_order = 4
        
        # ===== 音频参数 =====
        self.audio_sr = 48000        # 音频重采样率
        self.audio_bp_low = 4000     # 音频带通低频截止 (Hz)
        self.audio_bp_high = 10000   # 音频带通高频截止 (Hz)
        self.audio_filter_order = 4
        self.audio_env_ms = 15       # RMS包络窗口 (ms)
        self.audio_smooth_ms = 30    # 包络平滑窗口 (ms)
        
        # ===== 脚步检测参数 =====
        self.step_min_interval = 0.45   # 最小脚步间隔 (秒)
        self.audio_peak_prom = 1.5      # 峰值显著性阈值
        self.audio_peak_height = 0.8    # 峰值高度阈值
        
        # ===== 弱标签参数 =====
        self.weak_label_sigma = 0.15    # 弱标签高斯扩展sigma (秒)
        self.time_tolerance = 0.5       # 时间对齐容差 (秒)
        
        # ===== 特征提取参数 =====
        self.feature_win_ms = 100       # 特征窗口 (ms)
        self.feature_step_ms = 25       # 特征步长 (ms)
        
        # ===== 模型参数 =====
        self.model_type = 'auto'        # auto: 有CUDA用cnn，否则用rf
        self.n_estimators = 100
        self.self_train_rounds = 0      # 自训练轮数
        self.confidence_threshold = 0.75 # 高置信预测阈值
        self.device = 'auto'            # 'auto'/'cuda'/'cpu'
        self.torch_epochs = 40
        self.torch_batch_size = 64
        self.torch_lr = 1e-3
        self.torch_weight_decay = 1e-4
        self.torch_hidden_dim = 64
        self.torch_dropout = 0.1
        self.torch_patience = 8
        self.torch_val_interval = 5
        self.torch_amp = True
        self.cnn_window_s = 0.12
        self.cnn_predict_chunk = 256
        
        # ===== 输出参数 =====
        self.output_dir = 'output/weakly_supervised'
        
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        if args.trim_start is not None:
            self.trim_start_s = args.trim_start
        if args.trim_end is not None:
            self.trim_end_s = args.trim_end
        if args.das_fs is not None:
            self.das_fs = args.das_fs
        if args.audio_sr is not None:
            self.audio_sr = args.audio_sr
        if args.output_dir is not None:
            self.output_dir = args.output_dir


# ============================================================================
# [2] 信号处理工具函数
# ============================================================================
def butter_bandpass(low, high, fs, order=4):
    """设计Butterworth带通滤波器"""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if low_n <= 0:
        low_n = 1e-6
    if high_n >= 1:
        high_n = 0.9999
    b, a = butter(order, [low_n, high_n], btype='band')
    return b, a


def bandpass_filter(x, fs, low, high, order=4):
    """一维带通滤波"""
    b, a = butter_bandpass(low, high, fs, order=order)
    return filtfilt(b, a, x - np.mean(x))


def bandpass_filter_2d(X, fs, low, high, order=4):
    """对多通道信号 [T, C] 按列进行带通滤波"""
    b, a = butter_bandpass(low, high, fs, order=order)
    Xf = np.zeros_like(X, dtype=np.float64)
    for c in range(X.shape[1]):
        col = X[:, c].astype(np.float64) - np.mean(X[:, c])
        Xf[:, c] = filtfilt(b, a, col)
    return Xf


def robust_zscore(x, eps=1e-9):
    """基于MAD的鲁棒Z-score"""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad + eps)


def short_time_rms(x, win):
    """短时RMS包络"""
    win = max(1, int(win))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / float(win)
    ma = np.convolve(x2, kernel, mode='same')
    return np.sqrt(ma + 1e-12)


def moving_average(x, win):
    """移动平均"""
    if win <= 1:
        return x
    kernel = np.ones(int(win)) / float(win)
    return np.convolve(x, kernel, mode='same')


# ============================================================================
# [3] 音频脚步候选提取（弱标签生成）
# ============================================================================
class AudioWeakLabelExtractor:
    """从音频中提取脚步候选时间作为弱标签"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_audio(self, audio_path):
        """加载音频文件并统一到目标采样率"""
        audio_path = str(audio_path)
        ext = os.path.splitext(audio_path)[1].lower()

        # 对常见无损/有损音频格式直接读取；视频容器统一走ffmpeg抽取，避免依赖即将移除的audioread路径
        direct_exts = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".au", ".caf"}

        if ext in direct_exts:
            y, sr = self._read_mono_soundfile(audio_path)
            y = self._resample_if_needed(y, sr, self.config.audio_sr)
            return y, self.config.audio_sr

        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name

            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-vn", "-ac", "1", "-ar", str(self.config.audio_sr),
                tmp_wav,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                msg = proc.stderr.strip().splitlines()
                tail = "\n".join(msg[-8:]) if msg else "unknown ffmpeg error"
                raise RuntimeError(
                    "Failed to extract audio via ffmpeg. "
                    "Please install ffmpeg or provide a WAV/FLAC audio file.\n"
                    f"ffmpeg stderr tail:\n{tail}"
                )

            y, sr = self._read_mono_soundfile(tmp_wav)
            y = self._resample_if_needed(y, sr, self.config.audio_sr)
            return y, self.config.audio_sr
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    @staticmethod
    def _read_mono_soundfile(path):
        data, sr = sf.read(path, always_2d=True)
        y = np.mean(data, axis=1).astype(np.float32)
        return y, int(sr)

    @staticmethod
    def _resample_if_needed(y, src_sr, dst_sr):
        if src_sr == dst_sr:
            return y
        g = gcd(int(src_sr), int(dst_sr))
        up = int(dst_sr) // g
        down = int(src_sr) // g
        y_rs = resample_poly(y, up, down).astype(np.float32)
        return y_rs
    
    def extract_envelope(self, y, sr):
        """提取带通滤波后的RMS包络"""
        # 4-10kHz带通滤波
        y_bp = bandpass_filter(y, sr, 
                               self.config.audio_bp_low, 
                               self.config.audio_bp_high,
                               self.config.audio_filter_order)
        
        # 短时RMS包络
        env_win = max(1, int(self.config.audio_env_ms * 1e-3 * sr))
        env = short_time_rms(y_bp, env_win)
        
        # 平滑
        smooth_win = max(1, int(self.config.audio_smooth_ms * 1e-3 * sr))
        env_smooth = moving_average(env, smooth_win)
        
        t = np.arange(len(env_smooth)) / sr
        return t, env_smooth
    
    def detect_step_candidates(self, t, env):
        """从包络中检测脚步候选峰值"""
        # 对数包络的鲁棒Z-score
        log_env = np.log(env + 1e-12)
        z = robust_zscore(log_env)
        
        # 进一步平滑
        dt = np.median(np.diff(t))
        smooth_samples = max(1, int(0.02 / dt))  # 20ms
        z_smooth = moving_average(z, smooth_samples)
        
        # 峰值检测
        min_dist = max(1, int(self.config.step_min_interval / dt))
        peaks, props = find_peaks(z_smooth, 
                                  distance=min_dist,
                                  prominence=self.config.audio_peak_prom,
                                  height=self.config.audio_peak_height)
        
        step_times = t[peaks]
        step_heights = props['peak_heights'] if 'peak_heights' in props else z_smooth[peaks]
        
        return step_times, step_heights, z_smooth
    
    def generate_soft_labels(self, step_times, time_grid, sigma=None):
        """生成软标签（高斯扩展）"""
        if sigma is None:
            sigma = self.config.weak_label_sigma
        
        soft_labels = np.zeros(len(time_grid), dtype=np.float64)
        for t_step in step_times:
            # 高斯窗
            gauss = np.exp(-0.5 * ((time_grid - t_step) / sigma) ** 2)
            soft_labels = np.maximum(soft_labels, gauss)
        
        return soft_labels
    
    def process_audio(self, audio_path, trim_start=0.0, trim_end=None):
        """完整的音频处理流程"""
        print(f"[Audio] Loading: {audio_path}")
        y, sr = self.load_audio(audio_path)
        
        # 时间裁剪
        start_sample = int(trim_start * sr)
        end_sample = int(trim_end * sr) if trim_end else len(y)
        y = y[start_sample:end_sample]
        
        print(f"[Audio] Duration after trim: {len(y)/sr:.2f}s")
        
        # 提取包络
        t, env = self.extract_envelope(y, sr)
        
        # 检测脚步候选
        step_times, step_heights, z_env = self.detect_step_candidates(t, env)
        
        print(f"[Audio] Detected {len(step_times)} step candidates")
        
        return {
            'time': t,
            'envelope': env,
            'z_envelope': z_env,
            'step_times': step_times,
            'step_heights': step_heights,
            'sr': sr
        }


# ============================================================================
# [4] DAS数据处理与特征提取
# ============================================================================
class DASFeatureExtractor:
    """DAS数据处理与多频带特征提取"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_das_csv(self, csv_path):
        """加载DAS CSV文件"""
        print(f"[DAS] Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        das = df.values.astype(np.float64)
        
        # 获取通道名
        ch_cols = [c for c in df.columns if c.startswith('ch_')]
        n_channels = len(ch_cols)
        
        print(f"[DAS] Shape: {das.shape}, Channels: {n_channels}")
        return das, ch_cols
    
    def trim_data(self, das, trim_start=0.0, trim_end=None):
        """时间裁剪"""
        fs = self.config.das_fs
        start_idx = int(trim_start * fs)
        end_idx = int(trim_end * fs) if trim_end else das.shape[0]
        
        das_trimmed = das[start_idx:end_idx, :]
        print(f"[DAS] Trimmed shape: {das_trimmed.shape}, Duration: {das_trimmed.shape[0]/fs:.2f}s")
        return das_trimmed
    
    def multi_band_filter(self, das):
        """多频带滤波"""
        bands_filtered = {}
        for low, high in self.config.das_bp_bands:
            print(f"[DAS] Applying {low}-{high}Hz bandpass filter...")
            filtered = bandpass_filter_2d(das, self.config.das_fs, low, high, 
                                          self.config.das_filter_order)
            bands_filtered[(low, high)] = filtered
        return bands_filtered
    
    def compute_short_time_energy(self, das_filtered, win_ms=None, step_ms=None):
        """计算短时能量矩阵 [C, T_frames]"""
        if win_ms is None:
            win_ms = self.config.feature_win_ms
        if step_ms is None:
            step_ms = self.config.feature_step_ms
        
        fs = self.config.das_fs
        win_samples = max(1, int(win_ms * 1e-3 * fs))
        step_samples = max(1, int(step_ms * 1e-3 * fs))
        
        T, C = das_filtered.shape
        
        frames = []
        frame_times = []
        
        for start in range(0, T - win_samples + 1, step_samples):
            frame = das_filtered[start:start + win_samples, :]
            energy = np.sum(frame ** 2, axis=0)  # [C]
            frames.append(energy)
            frame_times.append((start + win_samples // 2) / fs)
        
        E = np.array(frames).T  # [C, n_frames]
        frame_times = np.array(frame_times)
        
        return E, frame_times
    
    def extract_features_at_times(self, das_bands, sample_times, window_s=0.2):
        """在指定时间点提取多频带特征"""
        fs = self.config.das_fs
        half_win = int(window_s * fs / 2)
        
        features_list = []
        valid_times = []
        
        for t in sample_times:
            t_idx = int(t * fs)
            start = max(0, t_idx - half_win)
            end = min(das_bands[(5, 10)].shape[0], t_idx + half_win)
            
            if end - start < half_win:
                continue
            
            feat = []
            for (low, high), das_bp in das_bands.items():
                window = das_bp[start:end, :]  # [win, C]
                
                # 特征：每个通道的能量、最大值、标准差
                energy = np.sum(window ** 2, axis=0)  # [C]
                max_val = np.max(np.abs(window), axis=0)  # [C]
                std_val = np.std(window, axis=0)  # [C]
                
                # 汇总统计
                feat.extend([
                    np.sum(energy),           # 总能量
                    np.max(energy),           # 最大通道能量
                    np.mean(energy),          # 平均能量
                    np.std(energy),           # 能量标准差
                    np.argmax(energy),        # 最大能量通道
                    np.max(max_val),          # 最大振幅
                    np.mean(std_val),         # 平均波动
                ])
            
            features_list.append(feat)
            valid_times.append(t)
        
        return np.array(features_list), np.array(valid_times)

    def extract_cnn_windows_at_times(self, das_bands, sample_times, window_s=0.2):
        """为CNN提取原始时空窗口，输出 [N, B, W, C]。"""
        fs = self.config.das_fs
        half_win = int(window_s * fs / 2)
        band_keys = sorted(das_bands.keys())

        windows = []
        valid_times = []
        for t in sample_times:
            t_idx = int(t * fs)
            start = max(0, t_idx - half_win)
            end = min(das_bands[band_keys[0]].shape[0], t_idx + half_win)
            if end - start < 2 * half_win:
                continue

            band_windows = []
            for key in band_keys:
                w = das_bands[key][start:end, :].astype(np.float32)
                band_windows.append(w)
            x = np.stack(band_windows, axis=0)  # [B, W, C]
            windows.append(x)
            valid_times.append(t)

        if not windows:
            return np.empty((0, len(band_keys), 2 * half_win, 0), dtype=np.float32), np.array([])
        return np.stack(windows, axis=0), np.array(valid_times)
    
    def estimate_channel_for_time(self, das_filtered, t, window_s=0.15):
        """估计指定时间点最可能的脚步通道"""
        fs = self.config.das_fs
        t_idx = int(t * fs)
        half_win = int(window_s * fs / 2)
        
        start = max(0, t_idx - half_win)
        end = min(das_filtered.shape[0], t_idx + half_win)
        
        if end <= start:
            return 0, 0.5
        
        window = das_filtered[start:end, :]
        
        # 每个通道的能量
        channel_energy = np.sum(window ** 2, axis=0)
        
        # 使用softmax归一化（更平滑）
        # 先做z-score标准化
        mean_e = np.mean(channel_energy)
        std_e = np.std(channel_energy) + 1e-12
        z_energy = (channel_energy - mean_e) / std_e
        
        # 温度参数控制softmax的锐度
        temperature = 1.0
        exp_z = np.exp(z_energy / temperature)
        softmax_prob = exp_z / (np.sum(exp_z) + 1e-12)
        
        # 最可能的通道
        best_ch = np.argmax(softmax_prob)
        
        # 置信度：使用top-1与top-2/3的比值
        sorted_probs = np.sort(softmax_prob)[::-1]
        top1 = sorted_probs[0]
        top_avg = np.mean(sorted_probs[1:4]) if len(sorted_probs) > 3 else sorted_probs[1] if len(sorted_probs) > 1 else 0.01
        
        # 归一化置信度到0.3-1.0范围
        ratio = top1 / (top_avg + 1e-12)
        confidence = min(1.0, max(0.3, 0.3 + 0.7 * (1 - np.exp(-ratio / 5))))
        
        return best_ch, confidence


# ============================================================================
# [5] 弱监督模型训练与预测
# ============================================================================
def _resolve_torch_device(device_pref: str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install torch to use cnn model_type.")
    pref = (device_pref or 'auto').lower()
    if pref == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pref == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return torch.device('cuda')
    return torch.device('cpu')


class TemporalBlock(nn.Module):
    """时间维度残差块 - 捕捉振动的时序模式"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), 
                      padding=(padding, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), 
                      padding=(padding, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.conv(x))


class SpatialBlock(nn.Module):
    """空间维度残差块 - 捕捉相邻通道的振动关联"""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), 
                      padding=(0, padding)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), 
                      padding=(0, padding)),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.conv(x))


class FeatureCNN(nn.Module):
    """
    改进版CNN - 直接学习振动模式而非只关注能量
    
    设计原则:
    1. 使用小卷积核(3x3)捕捉局部振动波形
    2. 分离时间和空间卷积，分别学习振动序列和通道关联
    3. 使用残差连接保留原始波形信息
    4. 减少池化操作，用stride卷积逐步降采样
    5. 多尺度时序特征提取（不同dilation）
    """
    def __init__(self, in_bands: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        c1 = max(32, hidden_dim // 4)
        c2 = max(64, hidden_dim // 2)
        c3 = max(128, hidden_dim)
        
        # 初始特征提取 - 小卷积核保留振动细节
        self.stem = nn.Sequential(
            nn.Conv2d(in_bands, c1, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
        )
        
        # 多尺度时序振动模式学习（不同dilation捕捉不同时间尺度的振动）
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(c1, kernel_size=3, dilation=1),  # 短期振动
            TemporalBlock(c1, kernel_size=3, dilation=2),  # 中期振动
            TemporalBlock(c1, kernel_size=3, dilation=4),  # 长期振动模式
        ])
        
        # 空间降采样（stride=2，比MaxPool更平滑）
        self.downsample1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
        )
        
        # 空间关联学习 - 相邻通道的振动关联
        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(c2, kernel_size=3),
            SpatialBlock(c2, kernel_size=5),  # 更大范围的通道关联
        ])
        
        # 进一步特征压缩
        self.downsample2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
        )
        
        # 最终特征聚合 - 保留时间和空间维度的统计量
        # 不使用(1,1)全局池化，而是分别计算时间和空间统计
        self.temporal_pool = nn.AdaptiveAvgPool2d((4, 1))  # 保留4个时间点
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 4))   # 保留4个通道
        
        # 分类头 - 更大的隐藏层保留信息
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 + c3 * 4, hidden_dim * 2),  # 时间+空间特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.in_bands = in_bands

    def forward(self, x):
        # 初始特征
        x = self.stem(x)
        
        # 多尺度时序振动模式
        for block in self.temporal_blocks:
            x = block(x)
        
        # 降采样
        x = self.downsample1(x)
        
        # 空间关联
        for block in self.spatial_blocks:
            x = block(x)
        
        # 进一步压缩
        x = self.downsample2(x)
        
        # 分别提取时间和空间统计特征
        t_feat = self.temporal_pool(x)  # [B, C, 4, 1]
        s_feat = self.spatial_pool(x)   # [B, C, 1, 4]
        
        # 拼接时空特征
        t_flat = t_feat.view(t_feat.size(0), -1)
        s_flat = s_feat.view(s_feat.size(0), -1)
        combined = torch.cat([t_flat, s_flat], dim=1)
        
        return self.head(combined).squeeze(1)


class TorchBinaryClassifier:
    """提供与sklearn兼容接口的PyTorch二分类包装器。"""
    def __init__(self, config: Config, input_shape=None):
        self.config = config
        self.model_kind = 'cnn'
        self.input_shape = input_shape
        self.device = _resolve_torch_device(config.device)
        self.model = None
        self.best_val_loss = None

    def _build_network(self, input_shape):
        in_bands = int(input_shape[0])
        return FeatureCNN(
            in_bands=in_bands,
            hidden_dim=self.config.torch_hidden_dim,
            dropout=self.config.torch_dropout,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.input_shape = tuple(X.shape[1:])
        self.model = self._build_network(self.input_shape).to(self.device)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.astype(np.int32)
        )

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=self.config.torch_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.torch_batch_size, shuffle=False)

        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        pos_weight = torch.tensor([max(1.0, neg / (pos + 1e-12))], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.torch_lr,
            weight_decay=self.config.torch_weight_decay,
        )
        use_amp = bool(getattr(self.config, "torch_amp", True)) and (self.device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        val_interval = max(1, int(getattr(self.config, "torch_val_interval", 5)))

        best_state = None
        best_val = np.inf
        wait = 0
        print(
            f"[Torch] model={self.model_kind}, device={self.device}, input_shape={self.input_shape}, "
            f"amp={use_amp}, val_interval={val_interval}"
        )
        for epoch in range(1, self.config.torch_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            should_validate = (
                epoch == 1
                or epoch % val_interval == 0
                or epoch == self.config.torch_epochs
            )
            if not should_validate:
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = self.model(xb)
                    val_losses.append(float(criterion(logits, yb).item()))
            mean_val = float(np.mean(val_losses)) if val_losses else np.inf
            print(f"[Torch] Epoch {epoch:03d}/{self.config.torch_epochs} val_loss={mean_val:.4f}")

            if mean_val < best_val - 1e-5:
                best_val = mean_val
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.config.torch_patience:
                    print(f"[Torch] Early stopping at epoch {epoch}, best_val={best_val:.4f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_loss = best_val
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Torch model not trained.")
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        probs_all = []
        with torch.no_grad():
            for i in range(0, len(X), self.config.torch_batch_size):
                xb = torch.from_numpy(X[i:i + self.config.torch_batch_size]).to(self.device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                probs_all.append(probs)
        p1 = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float32)
        p1 = p1.astype(np.float64)
        return np.column_stack([1.0 - p1, p1])

    def score(self, X, y):
        probs = self.predict_proba(X)[:, 1]
        pred = (probs >= 0.5).astype(np.int32)
        y = np.asarray(y).astype(np.int32)
        return float(np.mean(pred == y))

    def to_serializable(self):
        if self.model is None or self.input_shape is None:
            raise ValueError("Torch model not trained.")
        return {
            'model_kind': self.model_kind,
            'input_shape': [int(x) for x in self.input_shape],
            'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            'best_val_loss': None if self.best_val_loss is None else float(self.best_val_loss),
        }

    @classmethod
    def from_serializable(cls, artifact: dict, config: Config):
        model_kind = artifact.get('model_kind', 'cnn')
        if model_kind != 'cnn':
            raise ValueError(f"Unsupported saved torch model_kind: {model_kind}")
        input_shape = tuple(artifact.get('input_shape', [2, 400, 174]))
        obj = cls(config=config, input_shape=input_shape)
        obj.model = obj._build_network(input_shape).to(obj.device)
        obj.model.load_state_dict(artifact['state_dict'])
        obj.model.eval()
        obj.best_val_loss = artifact.get('best_val_loss')
        return obj


class WeaklySupervisedDetector:
    """弱监督脚步检测器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.last_eval_report = None
        self.recommended_threshold = None

    def _effective_model_type(self):
        has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        selected = self.config.model_type
        if selected == 'auto':
            selected = 'cnn' if has_cuda else 'rf'
            print(f"[Model] Auto-selecting model: {'CNN (CUDA available)' if has_cuda else 'RandomForest (no CUDA)'}")
        elif selected == 'cnn' and not has_cuda:
            print("[Model] CUDA不可用，自动回退到rf")
            selected = 'rf'
        elif selected == 'cnn' and has_cuda:
            print("[Model] Using CNN with CUDA")
        elif selected == 'rf':
            print("[Model] Using RandomForest")
        return selected

    def _build_model(self):
        """按配置构建分类器"""
        selected = self._effective_model_type()

        if selected == 'cnn':
            return TorchBinaryClassifier(self.config)
        if selected == 'rf':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=10,
                random_state=42,
                n_jobs=1
            )
        raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    def _run_validation_report(self, X, y):
        """
        训练前做一次轻量验证：
        - 时间无关分层划分（默认 80/20）
        - 输出 PR-AUC / Precision / Recall / F1
        - 扫描阈值给出推荐值
        """
        self.last_eval_report = None
        self.recommended_threshold = None

        n_samples = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        min_class = min(n_pos, n_neg)

        print(f"[Eval] Samples={n_samples}, Pos={n_pos}, Neg={n_neg}")

        if n_samples < 30 or min_class < 8:
            print("[Eval] Skip validation: sample size too small for stable split")
            return

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model_eval = self._build_model()
        if self._effective_model_type() == 'cnn':
            model_eval.fit(X_train, y_train)
            val_probs = model_eval.predict_proba(X_val)[:, 1]
        else:
            scaler_eval = StandardScaler()
            X_train_scaled = scaler_eval.fit_transform(X_train)
            X_val_scaled = scaler_eval.transform(X_val)
            model_eval.fit(X_train_scaled, y_train)
            val_probs = model_eval.predict_proba(X_val_scaled)[:, 1]
        pr_auc = average_precision_score(y_val, val_probs)

        thresholds = np.arange(0.30, 0.91, 0.05)
        rows = []
        best = None
        best_key = (-1.0, -1.0, -1.0, 0.0)  # F1, Precision, Recall, -|thr-0.5|

        print("[Eval] Validation metrics by threshold:")
        print("       thr    P      R      F1")
        for thr in thresholds:
            y_pred = (val_probs >= thr).astype(np.int32)
            p = precision_score(y_val, y_pred, zero_division=0)
            r = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            rows.append({'threshold': float(thr), 'precision': float(p), 'recall': float(r), 'f1': float(f1)})
            print(f"       {thr:0.2f}  {p:0.3f}  {r:0.3f}  {f1:0.3f}")

            key = (f1, p, r, -abs(float(thr) - 0.5))
            if key > best_key:
                best_key = key
                best = rows[-1]

        if best is None:
            return

        self.recommended_threshold = float(best['threshold'])
        self.last_eval_report = {
            'n_samples': int(n_samples),
            'n_pos': int(n_pos),
            'n_neg': int(n_neg),
            'pr_auc': float(pr_auc),
            'best_threshold': float(best['threshold']),
            'best_precision': float(best['precision']),
            'best_recall': float(best['recall']),
            'best_f1': float(best['f1']),
            'grid': rows,
        }

        print(f"[Eval] PR-AUC={pr_auc:.4f}")
        print(f"[Eval] Best threshold={best['threshold']:.2f} (P={best['precision']:.3f}, "
              f"R={best['recall']:.3f}, F1={best['f1']:.3f})")
        print(f"[Eval] Current confidence_threshold={self.config.confidence_threshold:.2f}")
        print(f"[Eval] Suggestion: try --confidence_threshold {best['threshold']:.2f}")

    def prepare_training_data(self, das_bands, audio_step_times, 
                              neg_ratio=3.0, time_range=None):
        """
        准备训练数据
        - 正样本：音频检测到的脚步时间点附近
        - 负样本：远离任何脚步的时间点
        """
        if time_range is None:
            T = das_bands[(5, 10)].shape[0]
            time_range = (0, T / self.config.das_fs)
        
        t_min, t_max = time_range
        
        # 正样本时间
        pos_times = audio_step_times[(audio_step_times >= t_min) & 
                                      (audio_step_times <= t_max)]
        
        # 生成负样本时间（远离正样本）
        min_dist = self.config.step_min_interval * 1.5
        all_times = np.arange(t_min + 0.5, t_max - 0.5, 0.1)
        
        neg_times = []
        for t in all_times:
            if len(pos_times) == 0 or np.min(np.abs(t - pos_times)) > min_dist:
                neg_times.append(t)
        neg_times = np.array(neg_times)
        
        # 采样负样本
        n_neg = min(len(neg_times), int(len(pos_times) * neg_ratio))
        if n_neg > 0:
            neg_idx = np.random.choice(len(neg_times), n_neg, replace=False)
            neg_times = neg_times[neg_idx]
        
        print(f"[Train] Positive samples: {len(pos_times)}, Negative samples: {len(neg_times)}")
        
        model_type = self._effective_model_type()

        # 提取特征/窗口
        das_extractor = DASFeatureExtractor(self.config)
        if model_type == 'cnn':
            pos_x, _ = das_extractor.extract_cnn_windows_at_times(
                das_bands, pos_times, window_s=self.config.cnn_window_s)
            neg_x, _ = das_extractor.extract_cnn_windows_at_times(
                das_bands, neg_times, window_s=self.config.cnn_window_s)
            X = np.concatenate([pos_x, neg_x], axis=0)
            y = np.concatenate([np.ones(len(pos_x)), np.zeros(len(neg_x))])
        else:
            pos_features, _ = das_extractor.extract_features_at_times(
                das_bands, pos_times, window_s=0.2)
            neg_features, _ = das_extractor.extract_features_at_times(
                das_bands, neg_times, window_s=0.2)
            X = np.vstack([pos_features, neg_features])
            y = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])

        # 打乱
        perm = np.random.permutation(len(y))
        X = X[perm]
        y = y[perm]
        
        return X, y
    
    def train(self, X, y):
        """训练模型"""
        # 在正式训练前输出一份验证指标报告（不改变最终训练行为）
        self._run_validation_report(X, y)

        # 选择模型
        self.model = self._build_model()

        if isinstance(self.model, TorchBinaryClassifier):
            self.model.fit(X, y)
            train_acc = self.model.score(X, y)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            train_acc = self.model.score(X_scaled, y)

        # 训练集准确率
        print(f"[Train] Training accuracy: {train_acc:.4f}")
        
        return self.model
    
    def predict_on_grid(self, das_bands, time_step=0.05, time_range=None):
        """在时间网格上预测脚步概率"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if time_range is None:
            T = das_bands[(5, 10)].shape[0]
            time_range = (0.5, T / self.config.das_fs - 0.5)
        
        t_min, t_max = time_range
        grid_times = np.arange(t_min, t_max, time_step)
        
        das_extractor = DASFeatureExtractor(self.config)
        if isinstance(self.model, TorchBinaryClassifier):
            chunk = max(32, int(self.config.cnn_predict_chunk))
            all_times = []
            all_probs = []
            for i in range(0, len(grid_times), chunk):
                batch_times = grid_times[i:i + chunk]
                windows, valid_times = das_extractor.extract_cnn_windows_at_times(
                    das_bands, batch_times, window_s=self.config.cnn_window_s)
                if len(windows) == 0:
                    continue
                probs = self.model.predict_proba(windows)[:, 1]
                all_times.append(valid_times)
                all_probs.append(probs)
            if not all_times:
                return np.array([]), np.array([])
            return np.concatenate(all_times), np.concatenate(all_probs)
        else:
            features, valid_times = das_extractor.extract_features_at_times(
                das_bands, grid_times, window_s=0.2)
            if len(features) == 0:
                return np.array([]), np.array([])
            X_scaled = self.scaler.transform(features)
            probs = self.model.predict_proba(X_scaled)[:, 1]
            return valid_times, probs
    
    def detect_steps_from_probs(self, times, probs, threshold=0.75, min_interval=0.45):
        """从概率曲线中检测脚步事件"""
        # 峰值检测
        dt = np.median(np.diff(times)) if len(times) > 1 else 0.05
        min_dist = max(1, int(min_interval / dt))
        
        peaks, props = find_peaks(probs, distance=min_dist, height=threshold)
        
        step_times = times[peaks]
        step_probs = probs[peaks]
        
        return step_times, step_probs
    
    def save_model(self, model_path):
        """
        保存训练好的模型到文件
        
        Args:
            model_path: 模型保存路径 (.joblib 文件)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # 保存模型和标准化器以及配置
        if isinstance(self.model, TorchBinaryClassifier):
            model_payload = {
                'type': 'torch',
                'artifact': self.model.to_serializable(),
            }
        else:
            model_payload = {
                'type': 'sklearn',
                'artifact': self.model,
            }

        model_data = {
            'model_payload': model_payload,
            'scaler': self.scaler,
            'recommended_threshold': self.recommended_threshold,
            'config': {
                'das_fs': self.config.das_fs,
                'das_bp_bands': self.config.das_bp_bands,
                'das_filter_order': self.config.das_filter_order,
                'step_min_interval': self.config.step_min_interval,
                'feature_win_ms': self.config.feature_win_ms,
                'feature_step_ms': self.config.feature_step_ms,
                'model_type': self.config.model_type,
                'n_estimators': self.config.n_estimators,
                'device': self.config.device,
                'torch_epochs': self.config.torch_epochs,
                'torch_batch_size': self.config.torch_batch_size,
                'torch_lr': self.config.torch_lr,
                'torch_weight_decay': self.config.torch_weight_decay,
                'torch_hidden_dim': self.config.torch_hidden_dim,
                'torch_dropout': self.config.torch_dropout,
                'torch_patience': self.config.torch_patience,
                'torch_val_interval': self.config.torch_val_interval,
                'torch_amp': self.config.torch_amp,
                'cnn_window_s': self.config.cnn_window_s,
                'cnn_predict_chunk': self.config.cnn_predict_chunk,
            },
            'version': '2.0'
        }
        
        joblib.dump(model_data, model_path)
        print(f"[Model] Saved to: {model_path}")
    
    @classmethod
    def load_model(cls, model_path, config=None):
        """
        从文件加载已训练的模型
        
        Args:
            model_path: 模型文件路径 (.joblib)
            config: 可选的配置对象，如果不提供则使用模型中保存的配置
        
        Returns:
            WeaklySupervisedDetector: 加载好模型的检测器实例
        """
        print(f"[Model] Loading from: {model_path}")
        model_data = joblib.load(model_path)
        
        # 创建配置
        if config is None:
            config = Config()
        
        # 恢复保存的配置参数
        saved_config = model_data.get('config', {})
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 创建检测器实例
        detector = cls(config)
        model_payload = model_data.get('model_payload')
        if model_payload is None:
            # backward compatible with old format
            detector.model = model_data['model']
        elif model_payload.get('type') == 'torch':
            detector.model = TorchBinaryClassifier.from_serializable(model_payload['artifact'], config)
        else:
            detector.model = model_payload['artifact']
        detector.scaler = model_data['scaler']
        detector.recommended_threshold = model_data.get('recommended_threshold', None)
        
        print(f"[Model] Loaded successfully (version: {model_data.get('version', 'unknown')})")
        print(f"[Model] Config: DAS {saved_config.get('das_bp_bands', 'N/A')}, "
              f"model_type={saved_config.get('model_type', 'N/A')}")
        if detector.recommended_threshold is not None:
            print(f"[Model] Recommended threshold from training: {detector.recommended_threshold:.2f}")
        
        return detector


# ============================================================================
# [6] 自训练迭代优化
# ============================================================================
class SelfTrainingIterator:
    """自训练迭代优化器"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def run_iteration(self, das_bands, current_step_times, round_num):
        """运行一轮自训练迭代"""
        print(f"\n[Self-Train] Round {round_num}")
        
        # 创建新检测器
        detector = WeaklySupervisedDetector(self.config)
        
        # 准备训练数据
        X, y = detector.prepare_training_data(das_bands, current_step_times)
        
        # 训练
        detector.train(X, y)
        
        # 预测
        grid_times, probs = detector.predict_on_grid(das_bands, time_step=0.03)
        
        # 检测脚步
        new_step_times, new_step_probs = detector.detect_steps_from_probs(
            grid_times, probs, 
            threshold=self.config.confidence_threshold,
            min_interval=self.config.step_min_interval
        )
        
        # 合并新检测到的高置信脚步
        combined = self._merge_step_times(current_step_times, new_step_times, 
                                          new_step_probs, min_dist=0.2)
        
        print(f"[Self-Train] Previous steps: {len(current_step_times)}, "
              f"New detections: {len(new_step_times)}, Combined: {len(combined)}")
        
        return combined, detector, grid_times, probs
    
    def _merge_step_times(self, old_times, new_times, new_probs, min_dist=0.2):
        """合并新旧脚步时间"""
        if len(new_times) == 0:
            return old_times
        
        # 筛选高置信新检测
        high_conf_mask = new_probs >= self.config.confidence_threshold
        high_conf_times = new_times[high_conf_mask]
        
        # 只保留远离已有脚步的新检测
        new_to_add = []
        for t in high_conf_times:
            if len(old_times) == 0 or np.min(np.abs(t - old_times)) > min_dist:
                new_to_add.append(t)
        
        combined = np.sort(np.concatenate([old_times, new_to_add]))
        return combined


# ============================================================================
# [7] 可视化输出
# ============================================================================
class Visualizer:
    """可视化工具"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def plot_energy_heatmap_with_steps(self, energy_matrix, frame_times, 
                                       step_events, output_path,
                                       title="DAS Footstep Detection"):
        """
        绘制能量热图并标记脚步事件
        
        Args:
            energy_matrix: [C, T_frames] 能量矩阵
            frame_times: 帧时间数组
            step_events: [(time, channel, confidence), ...]
            output_path: 输出图片路径
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
        
        # === 上图：热图 + 脚步标记 ===
        ax1 = axes[0]
        
        # 对数能量
        log_energy = np.log10(energy_matrix + 1e-10)
        
        # 热图
        im = ax1.imshow(log_energy, aspect='auto', origin='lower',
                       extent=[frame_times[0], frame_times[-1], 
                               0, energy_matrix.shape[0]],
                       cmap='viridis', interpolation='bilinear')
        
        # 标记脚步事件
        if len(step_events) > 0:
            step_times = [e[0] for e in step_events]
            step_channels = [e[1] for e in step_events]
            step_confs = [e[2] for e in step_events]
            
            # 用散点标记
            scatter = ax1.scatter(step_times, step_channels, 
                                  c=step_confs, cmap='hot', 
                                  s=50, alpha=0.8, edgecolors='white', linewidths=0.5,
                                  vmin=0.5, vmax=1.0)
            plt.colorbar(scatter, ax=ax1, label='Confidence', shrink=0.8)
        
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Channel', fontsize=12)
        ax1.set_title(title, fontsize=14)
        
        plt.colorbar(im, ax=ax1, label='Log Energy', shrink=0.8, pad=0.12)
        
        # === 下图：时间轴上的脚步概率曲线 ===
        ax2 = axes[1]
        
        if len(step_events) > 0:
            # 创建脚步密度曲线
            t_grid = np.linspace(frame_times[0], frame_times[-1], 1000)
            step_density = np.zeros_like(t_grid)
            
            for t_step, ch, conf in step_events:
                gauss = conf * np.exp(-0.5 * ((t_grid - t_step) / 0.1) ** 2)
                step_density = np.maximum(step_density, gauss)
            
            ax2.fill_between(t_grid, step_density, alpha=0.5, color='orange', 
                            label='Step Probability')
            ax2.plot(t_grid, step_density, color='darkorange', linewidth=1)
            
            # 标记脚步时间
            for t_step, ch, conf in step_events:
                ax2.axvline(t_step, color='red', alpha=0.5, linewidth=0.8)
        
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Step Probability', fontsize=12)
        ax2.set_xlim(frame_times[0], frame_times[-1])
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Viz] Saved: {output_path}")
    
    def plot_detection_comparison(self, frame_times, audio_env, audio_step_times,
                                   das_prob_curve, das_step_times, output_path):
        """对比音频和DAS检测结果"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        
        # 音频包络
        ax1 = axes[0]
        ax1.plot(frame_times[:len(audio_env)], audio_env, 'b-', linewidth=0.8)
        for t in audio_step_times:
            ax1.axvline(t, color='cyan', alpha=0.7, linewidth=1)
        ax1.set_ylabel('Audio Envelope\n(4-10kHz BP)')
        ax1.set_title('Audio vs DAS Footstep Detection Comparison')
        ax1.legend(['Audio Envelope', 'Audio Steps'], loc='upper right')
        
        # DAS概率曲线
        ax2 = axes[1]
        if len(das_prob_curve) > 0:
            ax2.plot(frame_times[:len(das_prob_curve)], das_prob_curve, 
                    'g-', linewidth=0.8)
        for t in das_step_times:
            ax2.axvline(t, color='lime', alpha=0.7, linewidth=1)
        ax2.set_ylabel('DAS Step\nProbability')
        ax2.legend(['DAS Probability', 'DAS Steps'], loc='upper right')
        
        # 合并对比
        ax3 = axes[2]
        for t in audio_step_times:
            ax3.axvline(t, color='blue', alpha=0.5, linewidth=1.5, 
                       label='Audio' if t == audio_step_times[0] else '')
        for t in das_step_times:
            ax3.axvline(t, color='red', alpha=0.5, linewidth=1.5,
                       label='DAS' if t == das_step_times[0] else '')
        ax3.set_ylabel('Detection\nEvents')
        ax3.set_xlabel('Time (s)')
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Viz] Saved: {output_path}")
    
    def plot_detailed_segment(self, das_filtered, frame_times, step_events,
                               t_start, t_end, output_path):
        """绘制详细时间段的信号和检测"""
        # 选择时间范围
        mask = (frame_times >= t_start) & (frame_times <= t_end)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 热图
        ax1 = axes[0]
        # ... 绘制逻辑
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def plot_multi_segment_detail(self, energy_matrix, frame_times, step_events,
                                   output_path, segment_duration=10.0):
        """绘制多个时间段的详细视图"""
        total_time = frame_times[-1] - frame_times[0]
        n_segments = min(4, int(total_time / segment_duration))
        
        if n_segments < 1:
            n_segments = 1
        
        fig, axes = plt.subplots(n_segments, 1, figsize=(16, 4 * n_segments))
        if n_segments == 1:
            axes = [axes]
        
        # 对数能量
        log_energy = np.log10(energy_matrix + 1e-10)
        
        for i, ax in enumerate(axes):
            t_start = frame_times[0] + i * segment_duration
            t_end = t_start + segment_duration
            
            # 时间范围掩码
            mask = (frame_times >= t_start) & (frame_times <= t_end)
            if not np.any(mask):
                continue
            
            frame_idx = np.where(mask)[0]
            segment_energy = log_energy[:, frame_idx]
            segment_times = frame_times[mask]
            
            # 热图
            im = ax.imshow(segment_energy, aspect='auto', origin='lower',
                          extent=[segment_times[0], segment_times[-1], 
                                  0, energy_matrix.shape[0]],
                          cmap='viridis', interpolation='bilinear')
            
            # 标记这个时间段内的脚步
            segment_steps = [(t, ch, conf) for t, ch, conf in step_events 
                            if t_start <= t <= t_end]
            
            if len(segment_steps) > 0:
                step_t = [s[0] for s in segment_steps]
                step_ch = [s[1] for s in segment_steps]
                step_conf = [s[2] for s in segment_steps]
                
                ax.scatter(step_t, step_ch, c=step_conf, cmap='hot',
                          s=80, alpha=0.9, edgecolors='white', linewidths=1,
                          vmin=0.3, vmax=1.0, zorder=10)
                
                # 添加竖线
                for t in step_t:
                    ax.axvline(t, color='white', alpha=0.3, linewidth=0.5)
            
            ax.set_ylabel('Channel')
            ax.set_title(f'Segment {i+1}: {t_start:.1f}s - {t_end:.1f}s  '
                        f'({len(segment_steps)} steps detected)')
        
        axes[-1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Viz] Saved: {output_path}")
    
    def plot_channel_trajectory(self, step_events, output_path):
        """绘制脚步通道轨迹图"""
        if len(step_events) == 0:
            return
        
        times = [e[0] for e in step_events]
        channels = [e[1] for e in step_events]
        confidences = [e[2] for e in step_events]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                                  gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图：通道轨迹
        ax1 = axes[0]
        scatter = ax1.scatter(times, channels, c=confidences, cmap='RdYlGn',
                              s=60, alpha=0.8, edgecolors='black', linewidths=0.5,
                              vmin=0.3, vmax=1.0)
        ax1.plot(times, channels, 'k-', alpha=0.3, linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax1, label='Confidence')
        
        ax1.set_ylabel('Channel', fontsize=12)
        ax1.set_title('Footstep Channel Trajectory Over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 下图：步频直方图
        ax2 = axes[1]
        if len(times) > 1:
            intervals = np.diff(times)
            ax2.hist(intervals, bins=30, color='steelblue', alpha=0.7, 
                    edgecolor='black')
            ax2.axvline(np.median(intervals), color='red', linestyle='--',
                       label=f'Median: {np.median(intervals):.3f}s')
            ax2.axvline(np.mean(intervals), color='orange', linestyle='--',
                       label=f'Mean: {np.mean(intervals):.3f}s')
            ax2.legend()
        
        ax2.set_xlabel('Step Interval (s)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Step Interval Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Viz] Saved: {output_path}")


# ============================================================================
# [8] 主流程
# ============================================================================
def run_pipeline(das_csv, audio_path, config: Config, align_dt=0.0,
                 save_model_path=None, load_model_path=None):
    """
    完整的弱监督脚步检测流程
    
    Args:
        das_csv: DAS CSV文件路径
        audio_path: 音频/视频文件路径
        config: 配置对象
        align_dt: 时间对齐偏移量 (DAS时间 = 音频时间 + align_dt)
        save_model_path: 训练后保存模型的路径
        load_model_path: 加载已有模型的路径（跳过训练）
    
    Returns:
        step_events: [(time, channel, confidence), ...]
    """
    print("=" * 60)
    print("Weakly Supervised Footstep Detection Pipeline")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ===== 1. 加载和处理DAS数据 =====
    das_extractor = DASFeatureExtractor(config)
    das_raw, ch_cols = das_extractor.load_das_csv(das_csv)
    
    # 时间裁剪
    das_trimmed = das_extractor.trim_data(das_raw, 
                                          config.trim_start_s, 
                                          config.trim_end_s)
    
    # 多频带滤波
    das_bands = das_extractor.multi_band_filter(das_trimmed)
    
    # 计算主频带能量矩阵（用于可视化）
    das_5_10 = das_bands[(5, 10)]
    energy_matrix, frame_times = das_extractor.compute_short_time_energy(das_5_10)
    
    print(f"[DAS] Energy matrix shape: {energy_matrix.shape}")
    
    # ===== 2. 处理音频，提取弱标签 =====
    audio_result = {'envelope': None, 'step_times': np.array([])}
    audio_step_times = np.array([])
    if audio_path:
        audio_extractor = AudioWeakLabelExtractor(config)
        audio_result = audio_extractor.process_audio(
            audio_path,
            trim_start=config.trim_start_s,
            trim_end=config.trim_end_s
        )
        audio_step_times = audio_result['step_times'] + align_dt
        print(f"[Audio] Step candidates (after alignment): {len(audio_step_times)}")
    
    # ===== 3. 训练或加载弱监督模型 =====
    if load_model_path:
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Model file not found: {load_model_path}")
        # 加载已有模型
        print(f"\n[Model] Loading pre-trained model from: {load_model_path}")
        detector = WeaklySupervisedDetector.load_model(load_model_path, config)
        
        # 直接预测
        grid_times, probs = detector.predict_on_grid(das_bands, time_step=0.03)
        step_times_detected, step_probs = detector.detect_steps_from_probs(
            grid_times, probs,
            threshold=config.confidence_threshold,
            min_interval=config.step_min_interval
        )
        
        print(f"[Model] Detected {len(step_times_detected)} steps using loaded model")
    else:
        if not audio_path:
            raise ValueError("Training mode requires --audio. For model-only inference use --load_model with --inference_only.")
        if len(audio_step_times) <= 5:
            raise ValueError(f"Too few audio weak labels ({len(audio_step_times)}). Check audio quality or trim range.")

        print("\n[Model] Starting weakly supervised training...")
        
        # 初始训练
        detector = WeaklySupervisedDetector(config)
        X, y = detector.prepare_training_data(das_bands, audio_step_times)
        detector.train(X, y)
        
        # 初始预测
        grid_times, probs = detector.predict_on_grid(das_bands, time_step=0.03)
        step_times_detected, step_probs = detector.detect_steps_from_probs(
            grid_times, probs,
            threshold=config.confidence_threshold,
            min_interval=config.step_min_interval
        )
        
        # 自训练迭代
        if config.self_train_rounds > 0:
            self_trainer = SelfTrainingIterator(config)
            current_steps = audio_step_times.copy()
            
            for round_num in range(1, config.self_train_rounds + 1):
                current_steps, detector, grid_times, probs = self_trainer.run_iteration(
                    das_bands, current_steps, round_num
                )
            
            # 最终检测
            step_times_detected, step_probs = detector.detect_steps_from_probs(
                grid_times, probs,
                threshold=config.confidence_threshold,
                min_interval=config.step_min_interval
            )
        
        # 保存训练好的模型
        if save_model_path:
            detector.save_model(save_model_path)
    
    # ===== 4. 估计通道位置 =====
    step_events = []
    for t, prob in zip(step_times_detected, step_probs):
        ch, ch_conf = das_extractor.estimate_channel_for_time(das_5_10, t)
        step_events.append((t, ch, prob * ch_conf))
    
    print(f"\n[Result] Detected {len(step_events)} footstep events")
    
    # ===== 5. 可视化输出 =====
    viz = Visualizer(config)
    
    # 主热图
    base_name = os.path.splitext(os.path.basename(das_csv))[0]
    heatmap_path = os.path.join(config.output_dir, f"{base_name}_heatmap_steps.png")
    viz.plot_energy_heatmap_with_steps(energy_matrix, frame_times, step_events, 
                                       heatmap_path,
                                       title=f"Footstep Detection: {base_name}")
    
    # 多段详细视图
    detail_path = os.path.join(config.output_dir, f"{base_name}_detailed_segments.png")
    viz.plot_multi_segment_detail(energy_matrix, frame_times, step_events, 
                                  detail_path, segment_duration=15.0)
    
    # 通道轨迹图
    trajectory_path = os.path.join(config.output_dir, f"{base_name}_channel_trajectory.png")
    viz.plot_channel_trajectory(step_events, trajectory_path)
    
    # 对比图
    if audio_result['envelope'] is not None:
        compare_path = os.path.join(config.output_dir, f"{base_name}_comparison.png")
        # 重采样音频包络到DAS帧率
        audio_env_resampled = np.interp(frame_times, 
                                        audio_result['time'], 
                                        audio_result['envelope'])
        
        # DAS概率曲线重采样
        if len(grid_times) > 0 and len(probs) > 0:
            das_prob_resampled = np.interp(frame_times, grid_times, probs)
        else:
            das_prob_resampled = np.zeros(len(frame_times))
        
        viz.plot_detection_comparison(frame_times, audio_env_resampled, 
                                      audio_step_times,
                                      das_prob_resampled, step_times_detected,
                                      compare_path)
    
    # ===== 6. 输出CSV结果 =====
    csv_output_path = os.path.join(config.output_dir, f"{base_name}_steps.csv")
    df_out = pd.DataFrame(step_events, columns=['time', 'channel', 'confidence'])
    df_out = df_out.sort_values('time').reset_index(drop=True)
    df_out.to_csv(csv_output_path, index=False)
    print(f"[Output] Steps CSV: {csv_output_path}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)
    print(f"Total steps detected: {len(step_events)}")
    if len(step_events) > 0:
        print(f"Time range: {df_out['time'].min():.2f}s - {df_out['time'].max():.2f}s")
        print(f"Avg confidence: {df_out['confidence'].mean():.3f}")
        print(f"Channel range: {df_out['channel'].min()} - {df_out['channel'].max()}")
    
    return step_events, energy_matrix, frame_times

# ============================================================================
# [9] 命令行接口
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Weakly Supervised Footstep Detection from DAS + Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本使用
  python WeaklySupervised_FootstepDetector.py --das_csv data.csv --audio video.mp4

  # 指定时间范围（去除头尾废弃部分）
  python WeaklySupervised_FootstepDetector.py --das_csv data.csv --audio video.mp4 \\
      --trim_start 5.0 --trim_end 200.0

  # 自定义输出目录
  python WeaklySupervised_FootstepDetector.py --das_csv data.csv --audio video.mp4 \\
      --output_dir my_results
        """
    )
    
    # 必需参数
    parser.add_argument('--das_csv', '-d', required=True,
                        help='DAS CSV文件路径')
    parser.add_argument('--audio', '-a', default=None,
                        help='音频/视频文件路径（用于提取弱标签）')
    
    # 时间裁剪参数（核心需求）
    parser.add_argument('--trim_start', type=float, default=None,
                        help='数据起始裁剪时间（秒），去除开头废弃部分')
    parser.add_argument('--trim_end', type=float, default=None,
                        help='数据结束裁剪时间（秒），去除结尾废弃部分')
    
    # 采样率参数
    parser.add_argument('--das_fs', type=int, default=2000,
                        help='DAS采样率 (Hz)，默认2000')
    parser.add_argument('--audio_sr', type=int, default=48000,
                        help='音频重采样率 (Hz)，默认48000')
    
    # 时间对齐
    parser.add_argument('--align_dt', type=float, default=0.0,
                        help='时间对齐偏移量：DAS时间 = 音频时间 + align_dt')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'rf', 'cnn'],
                        help='模型类型：auto/rf/cnn；auto=有CUDA用cnn，无CUDA用rf')
    parser.add_argument('--self_train_rounds', type=int, default=0,
                        help='自训练迭代轮数，默认0')
    parser.add_argument('--confidence_threshold', type=float, default=0.75,
                        help='高置信预测阈值，默认0.75')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='深度模型设备选择，默认auto')
    parser.add_argument('--torch_epochs', type=int, default=50,
                        help='深度模型训练轮数，默认50')
    parser.add_argument('--torch_batch_size', type=int, default=64,
                        help='深度模型batch大小，默认64')
    parser.add_argument('--torch_lr', type=float, default=1e-3,
                        help='深度模型学习率，默认1e-3')
    parser.add_argument('--torch_hidden_dim', type=int, default=64,
                        help='深度模型隐藏维度，默认64')
    parser.add_argument('--torch_dropout', type=float, default=0.1,
                        help='深度模型dropout，默认0.1')
    parser.add_argument('--cnn_window_s', type=float, default=0.12,
                        help='CNN输入窗口时长（秒），默认0.12')
    parser.add_argument('--cnn_predict_chunk', type=int, default=256,
                        help='CNN网格预测分块大小，默认256')
    
    # 输出参数
    parser.add_argument('--output_dir', '-o', default=None,
                        help='输出目录，默认 output/weakly_supervised')
    
    # 模型保存/加载参数
    parser.add_argument('--save_model', type=str, default=None,
                        help='训练后保存模型到指定路径（.joblib文件）')
    parser.add_argument('--load_model', type=str, default=None,
                        help='加载已训练的模型文件（.joblib文件），跳过训练')
    parser.add_argument('--inference_only', action='store_true',
                        help='仅推理模式：使用--load_model指定的模型，无需音频')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建配置
    config = Config()
    config.update_from_args(args)
    
    if args.self_train_rounds is not None:
        config.self_train_rounds = args.self_train_rounds
    if args.confidence_threshold is not None:
        config.confidence_threshold = args.confidence_threshold
    if args.model_type is not None:
        config.model_type = args.model_type
    if args.device is not None:
        config.device = args.device
    if args.torch_epochs is not None:
        config.torch_epochs = args.torch_epochs
    if args.torch_batch_size is not None:
        config.torch_batch_size = args.torch_batch_size
    if args.torch_lr is not None:
        config.torch_lr = args.torch_lr
    if args.torch_hidden_dim is not None:
        config.torch_hidden_dim = args.torch_hidden_dim
    if args.torch_dropout is not None:
        config.torch_dropout = args.torch_dropout
    if args.cnn_window_s is not None:
        config.cnn_window_s = args.cnn_window_s
    if args.cnn_predict_chunk is not None:
        config.cnn_predict_chunk = args.cnn_predict_chunk
    
    # ===== 仅推理模式 =====
    if args.inference_only:
        if args.load_model is None:
            print("[ERROR] --inference_only requires --load_model to specify a trained model")
            return
        
        print("\n" + "=" * 60)
        print("INFERENCE ONLY MODE (using pre-trained model)")
        print("=" * 60)
        
        step_events, energy_matrix, frame_times = run_inference_only(
            das_csv=args.das_csv,
            model_path=args.load_model,
            config=config
        )
    else:
        # ===== 正常训练+检测模式 =====
        step_events, energy_matrix, frame_times = run_pipeline(
            das_csv=args.das_csv,
            audio_path=args.audio,
            config=config,
            align_dt=args.align_dt,
            save_model_path=args.save_model,
            load_model_path=args.load_model
        )
    
    print("\n[DONE] Weakly supervised footstep detection completed!")


def run_inference_only(das_csv, model_path, config):
    """
    仅使用已训练模型进行推理（无需音频）
    
    Args:
        das_csv: DAS数据CSV文件路径
        model_path: 已保存的模型文件路径
        config: 配置对象
    
    Returns:
        step_events: 检测到的脚步事件列表
        energy_matrix: 能量矩阵
        frame_times: 帧时间
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ===== 1. 加载模型 =====
    detector = WeaklySupervisedDetector.load_model(model_path, config)
    
    # ===== 2. 加载和处理DAS数据 =====
    print(f"\n[DAS] Loading: {das_csv}")
    das_extractor = DASFeatureExtractor(config)
    
    df_das = pd.read_csv(das_csv)
    das_raw = df_das.values.astype(np.float32)
    
    # 时间裁剪
    if config.trim_start_s is not None or config.trim_end_s is not None:
        start_idx = int((config.trim_start_s or 0) * config.das_fs)
        end_idx = int((config.trim_end_s or (das_raw.shape[0] / config.das_fs)) * config.das_fs)
        das_raw = das_raw[start_idx:end_idx, :]
        print(f"[DAS] Trimmed: {start_idx/config.das_fs:.2f}s - {end_idx/config.das_fs:.2f}s")
    
    print(f"[DAS] Shape: {das_raw.shape} ({das_raw.shape[0]/config.das_fs:.2f}s, {das_raw.shape[1]} channels)")
    
    # 多频段带通滤波
    das_bands = das_extractor.multi_band_filter(das_raw)
    das_5_10 = das_bands[(5, 10)]
    
    # 能量矩阵
    energy_matrix, frame_times = das_extractor.compute_short_time_energy(das_5_10)
    
    # ===== 3. 特征提取和预测 =====
    # 使用 detector.predict_on_grid 进行网格预测
    grid_times, probs = detector.predict_on_grid(das_bands, time_step=0.03)
    
    # 检测脚步
    step_times_detected, step_probs = detector.detect_steps_from_probs(
        grid_times, probs, 
        threshold=config.confidence_threshold,
        min_interval=config.step_min_interval
    )
    
    print(f"\n[Inference] Detected {len(step_times_detected)} steps from DAS data")
    
    # ===== 4. 估计通道位置 =====
    step_events = []
    for t, prob in zip(step_times_detected, step_probs):
        ch, ch_conf = das_extractor.estimate_channel_for_time(das_5_10, t)
        step_events.append((t, ch, prob * ch_conf))
    
    # ===== 5. 可视化输出 =====
    viz = Visualizer(config)
    
    base_name = os.path.splitext(os.path.basename(das_csv))[0]
    heatmap_path = os.path.join(config.output_dir, f"{base_name}_heatmap_inference.png")
    viz.plot_energy_heatmap_with_steps(energy_matrix, frame_times, step_events, 
                                       heatmap_path,
                                       title=f"Inference Mode: {base_name}")
    
    # 通道轨迹图
    trajectory_path = os.path.join(config.output_dir, f"{base_name}_trajectory_inference.png")
    viz.plot_channel_trajectory(step_events, trajectory_path)
    
    # ===== 6. 输出CSV =====
    csv_output_path = os.path.join(config.output_dir, f"{base_name}_steps_inference.csv")
    df_out = pd.DataFrame(step_events, columns=['time', 'channel', 'confidence'])
    df_out = df_out.sort_values('time').reset_index(drop=True)
    df_out.to_csv(csv_output_path, index=False)
    print(f"[Output] Steps CSV: {csv_output_path}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"Total steps detected: {len(step_events)}")
    if len(step_events) > 0:
        print(f"Time range: {df_out['time'].min():.2f}s - {df_out['time'].max():.2f}s")
        print(f"Avg confidence: {df_out['confidence'].mean():.3f}")
        print(f"Channel range: {df_out['channel'].min()} - {df_out['channel'].max()}")
    
    return step_events, energy_matrix, frame_times


if __name__ == "__main__":
    main()
