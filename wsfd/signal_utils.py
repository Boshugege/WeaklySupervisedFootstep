# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt

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


def bandpass_filter(x, fs, low, high, order=4, method='filtfilt'):
    """一维带通滤波"""
    x0 = x - np.mean(x)
    if method == 'filtfilt':
        b, a = butter_bandpass(low, high, fs, order=order)
        return filtfilt(b, a, x0)
    if method == 'sosfilt':
        nyq = 0.5 * fs
        low_n = max(1e-6, low / nyq)
        high_n = min(0.9999, high / nyq)
        sos = butter(order, [low_n, high_n], btype='band', output='sos')
        return sosfilt(sos, x0)
    raise ValueError(f"Unsupported DAS filter method: {method}")


def bandpass_filter_2d(X, fs, low, high, order=4, method='filtfilt'):
    """对多通道信号 [T, C] 按列进行带通滤波"""
    Xf = np.zeros_like(X, dtype=np.float64)
    if method == 'filtfilt':
        b, a = butter_bandpass(low, high, fs, order=order)
        for c in range(X.shape[1]):
            col = X[:, c].astype(np.float64) - np.mean(X[:, c])
            Xf[:, c] = filtfilt(b, a, col)
        return Xf

    if method == 'sosfilt':
        nyq = 0.5 * fs
        low_n = max(1e-6, low / nyq)
        high_n = min(0.9999, high / nyq)
        sos = butter(order, [low_n, high_n], btype='band', output='sos')
        for c in range(X.shape[1]):
            col = X[:, c].astype(np.float64) - np.mean(X[:, c])
            Xf[:, c] = sosfilt(sos, col)
        return Xf

    raise ValueError(f"Unsupported DAS filter method: {method}")


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
