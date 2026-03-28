# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile
from math import gcd

import numpy as np
from scipy.signal import find_peaks, resample_poly
import soundfile as sf

from .config import Config
from .signal_utils import bandpass_filter, moving_average, robust_zscore, short_time_rms

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
