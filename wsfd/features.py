# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .config import Config
from .signal_utils import bandpass_filter_2d

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
        if self.config.disable_das_bandpass:
            print("[DAS] Bandpass disabled, using raw DAS for all configured bands")
            raw = das.astype(np.float64)
            return {tuple(band): raw for band in self.config.das_bp_bands}

        bands_filtered = {}
        for low, high in self.config.das_bp_bands:
            print(f"[DAS] Applying {low}-{high}Hz bandpass filter ({self.config.das_filter_method})...")
            filtered = bandpass_filter_2d(das, self.config.das_fs, low, high, 
                                          self.config.das_filter_order,
                                          method=self.config.das_filter_method)
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
        primary_band = tuple(self.config.das_bp_bands[0])
        
        features_list = []
        valid_times = []
        
        for t in sample_times:
            t_idx = int(t * fs)
            start = max(0, t_idx - half_win)
            end = min(das_bands[primary_band].shape[0], t_idx + half_win)
            
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
