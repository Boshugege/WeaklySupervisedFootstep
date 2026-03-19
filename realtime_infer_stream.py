#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online inference simulation:
Read existing TDMS files chunk-by-chunk, run causal bandpass filtering,
perform CNN inference with fixed latency, and stream JSON packets that
match JSONStreamPlot expectations.
"""

import argparse
import json
import socket
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
import re

import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
from nptdms import TdmsFile
import joblib

import torch
import torch.nn as nn

from WeaklySupervised_FootstepDetector import Config, FeatureCNN
from extract_name_signals_from_tdms import (
    read_csv_start_end,
    build_tdms_index,
    choose_overlapping_files,
    extraction_indices,
)


TDMS_TIME_PATTERN = re.compile(r"_UTC_(\d{8}_\d{6}\.\d+)\.tdms$", re.IGNORECASE)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Online inference simulation with JSON streaming output."
    )
    p.add_argument("--name", required=True, help="Target name (same as Airtag CSV filename, without extension).")
    p.add_argument("--model", "-m", required=True, help="Path to CNN model (.joblib).")
    p.add_argument("--airtag-csv-dir", default="Data/Airtag", help="Directory with Airtag CSV files.")
    p.add_argument("--tdms-dir", default="Data/DAS", help="Directory with TDMS files.")
    p.add_argument("--das-fs", type=float, default=2000.0, help="DAS sample rate (Hz).")
    p.add_argument("--skip-channels", type=int, default=18, help="Skip first N channels.")
    p.add_argument("--trim-head", type=float, default=50.0, help="Trim seconds from start.")
    p.add_argument("--trim-tail", type=float, default=20.0, help="Trim seconds from end.")
    p.add_argument("--csv-utc-offset-hours", type=float, default=8.0, help="Airtag CSV UTC offset (hours).")
    p.add_argument("--chunk-seconds", type=float, default=1.0, help="Chunk duration (s).")
    p.add_argument("--time-step", type=float, default=0.03, help="Grid step for inference (s).")
    p.add_argument("--buffer-seconds", type=float, default=10.0, help="Ring buffer size (s), <=0 keeps all probs.")
    p.add_argument("--latency-seconds", type=float, default=1.0, help="Event output latency (s).")
    p.add_argument("--detrend-alpha", type=float, default=0.001,
                   help="EMA detrend factor before bandpass (online approximation of global de-mean).")
    p.add_argument("--speed", type=float, default=1.0, help="Simulation speed (1.0 = real-time, 0 = no sleep).")
    p.add_argument("--max-seconds", type=float, default=None, help="Max seconds to stream (optional).")
    p.add_argument("--protocol", choices=["udp", "tcp"], default="udp", help="Stream protocol.")
    p.add_argument("--host", default="127.0.0.1", help="Target host.")
    p.add_argument("--port", type=int, default=9000, help="Target port.")
    p.add_argument("--signal-downsample", type=int, default=1, help="Downsample factor for signal JSON.")
    p.add_argument("--udp-max-samples", type=int, default=10, help="Max samples per UDP packet.")
    p.add_argument("--udp-max-bytes", type=int, default=60000, help="Max bytes per UDP packet.")
    return p.parse_args(argv)


def list_channels(tdms_file: TdmsFile):
    channels = []
    for group in tdms_file.groups():
        for channel in group.channels():
            channels.append(channel)
    return channels


def find_any_cnn_model():
    models = sorted(Path("output").glob("**/models/*_model.joblib"))
    if not models:
        return None
    for path in models:
        try:
            data = joblib.load(path)
        except Exception:
            continue
        payload = data.get("model_payload", {})
        if payload.get("type") == "torch":
            return str(path)
    return str(models[0])


class LegacyFeatureCNN(nn.Module):
    def __init__(self, in_bands: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_bands, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)


class TorchInferWrapper:
    def __init__(self, model: nn.Module, device, batch_size: int):
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        probs_all = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i:i + self.batch_size]).to(self.device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                probs_all.append(probs)
        p1 = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float32)
        p1 = p1.astype(np.float64)
        return np.column_stack([1.0 - p1, p1])


def load_torch_model(model_path: str, config: Config):
    data = joblib.load(model_path)
    model_payload = data.get("model_payload", {})
    if model_payload.get("type") != "torch":
        raise ValueError("Model is not a torch CNN model.")

    saved_config = data.get("config", {})
    for key, value in saved_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    artifact = model_payload.get("artifact", {})
    input_shape = tuple(artifact.get("input_shape", [2, 400, 174]))
    state_dict = artifact.get("state_dict", {})

    device = torch.device("cuda" if (config.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    if any(k.startswith("net.") for k in state_dict.keys()):
        model = LegacyFeatureCNN(in_bands=int(input_shape[0]), dropout=getattr(config, "torch_dropout", 0.1)).to(device)
    else:
        model = FeatureCNN(in_bands=int(input_shape[0]), hidden_dim=config.torch_hidden_dim, dropout=config.torch_dropout).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    wrapper = TorchInferWrapper(model=model, device=device, batch_size=config.torch_batch_size)
    return wrapper


class RingBuffer:
    def __init__(self, capacity, n_channels, dtype=np.float32):
        self.capacity = int(capacity)
        self.n_channels = int(n_channels)
        self.data = np.zeros((self.capacity, self.n_channels), dtype=dtype)
        self.start = 0
        self.size = 0
        self.total_written = 0

    def append(self, chunk):
        if chunk.size == 0:
            return
        chunk = np.asarray(chunk)
        n = chunk.shape[0]
        if n >= self.capacity:
            chunk = chunk[-self.capacity:, :]
            self.data[:, :] = chunk
            self.start = 0
            self.size = self.capacity
            self.total_written += n
            return

        end = (self.start + self.size) % self.capacity
        space = self.capacity - self.size
        if n <= space:
            first = min(n, self.capacity - end)
            self.data[end:end + first, :] = chunk[:first, :]
            if n > first:
                self.data[0:n - first, :] = chunk[first:, :]
            self.size += n
        else:
            # Overwrite oldest
            overwrite = n - space
            first = min(n, self.capacity - end)
            self.data[end:end + first, :] = chunk[:first, :]
            if n > first:
                self.data[0:n - first, :] = chunk[first:, :]
            self.start = (self.start + overwrite) % self.capacity
            self.size = self.capacity

        self.total_written += n

    def get_slice(self, start_idx, end_idx):
        if end_idx <= start_idx:
            return np.empty((0, self.n_channels), dtype=self.data.dtype)
        oldest = self.total_written - self.size
        if start_idx < oldest or end_idx > self.total_written:
            return np.empty((0, self.n_channels), dtype=self.data.dtype)
        rel_start = (self.start + (start_idx - oldest)) % self.capacity
        length = end_idx - start_idx
        if rel_start + length <= self.capacity:
            return self.data[rel_start:rel_start + length, :]
        first = self.capacity - rel_start
        return np.vstack([self.data[rel_start:, :], self.data[:length - first, :]])


class OnlineBandpass:
    def __init__(self, bands, fs, order, n_channels):
        self.bands = bands
        self.fs = fs
        self.order = order
        self.n_channels = n_channels
        self.sos_map = {}
        self.zi_map = {}
        for low, high in bands:
            sos = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="band", output="sos")
            self.sos_map[(low, high)] = sos
            self.zi_map[(low, high)] = np.zeros((sos.shape[0], 2, n_channels), dtype=np.float64)

    def filter_chunk(self, x):
        out = {}
        for band, sos in self.sos_map.items():
            zi = self.zi_map[band]
            y, zf = sosfilt(sos, x, zi=zi, axis=0)
            self.zi_map[band] = zf
            out[band] = y
        return out


class OnlineDetrender:
    def __init__(self, n_channels, alpha=0.001):
        self.n_channels = int(n_channels)
        self.alpha = float(min(1.0, max(1e-6, alpha)))
        self.mean = np.zeros((self.n_channels,), dtype=np.float64)
        self.initialized = False

    def transform_chunk(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            return x
        out = np.empty_like(x, dtype=np.float64)
        if not self.initialized and x.shape[0] > 0:
            self.mean[:] = x[0, :]
            self.initialized = True
        one_minus = 1.0 - self.alpha
        for i in range(x.shape[0]):
            self.mean = one_minus * self.mean + self.alpha * x[i, :]
            out[i, :] = x[i, :] - self.mean
        return out


def estimate_channel(window):
    if window.size == 0:
        return 0, 0.5
    channel_energy = np.sum(window ** 2, axis=0)
    mean_e = np.mean(channel_energy)
    std_e = np.std(channel_energy) + 1e-12
    z_energy = (channel_energy - mean_e) / std_e
    exp_z = np.exp(z_energy)
    softmax_prob = exp_z / (np.sum(exp_z) + 1e-12)
    best_ch = int(np.argmax(softmax_prob))
    sorted_probs = np.sort(softmax_prob)[::-1]
    top1 = sorted_probs[0]
    top_avg = np.mean(sorted_probs[1:4]) if len(sorted_probs) > 3 else sorted_probs[1] if len(sorted_probs) > 1 else 0.01
    ratio = top1 / (top_avg + 1e-12)
    confidence = min(1.0, max(0.3, 0.3 + 0.7 * (1 - np.exp(-ratio / 5))))
    return best_ch, float(confidence)


def open_sender(protocol, host, port):
    if protocol == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock


def send_packet(sock, protocol, host, port, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if protocol == "udp":
        sock.sendto(data, (host, port))
    else:
        sock.sendall(data + b"\n")


def main(argv=None):
    args = _parse_args(argv)
    name = args.name.lower().strip()
    tdms_dir = Path(args.tdms_dir)
    airtag_dir = Path(args.airtag_csv_dir)
    csv_path = airtag_dir / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Airtag CSV not found: {csv_path}")

    model_path = args.model

    config = Config()
    config.das_fs = float(args.das_fs)
    config.model_type = "cnn"
    detector = load_torch_model(model_path, config)
    primary_band = tuple(config.das_bp_bands[0])

    fs = float(args.das_fs)
    chunk_samples = max(1, int(args.chunk_seconds * fs))
    buffer_samples = max(chunk_samples * 2, int(args.buffer_seconds * fs))
    time_step = float(args.time_step)
    latency = float(args.latency_seconds)

    sock = open_sender(args.protocol, args.host, args.port)

    total_channels = None
    bandpass = None
    detrender = None
    band_buffers = {}

    prob_times = deque()
    prob_vals = deque()
    last_emitted_time = -1e9
    next_grid_time = 0.5
    stream_time = 0.0

    try:
        start_local, end_local = read_csv_start_end(csv_path, encoding="utf-8-sig")
        offset = timedelta(hours=float(args.csv_utc_offset_hours))
        start_utc = start_local - offset + timedelta(seconds=float(args.trim_head))
        end_utc = end_local - offset - timedelta(seconds=float(args.trim_tail))
        if end_utc <= start_utc:
            raise ValueError("Invalid trim range: end_utc <= start_utc after trimming.")

        tdms_index = build_tdms_index(tdms_dir)
        overlaps = choose_overlapping_files(tdms_index, start_utc, end_utc)
        if not overlaps:
            raise ValueError("No TDMS files overlap the requested time range.")

        global_written = 0

        for item in overlaps:
            path = item.path
            tdms = TdmsFile.read(path)
            channels = list_channels(tdms)
            if not channels:
                continue

            channels = channels[args.skip_channels:] if args.skip_channels > 0 else channels
            if not channels:
                continue

            if total_channels is None:
                total_channels = len(channels)
                if not config.disable_das_bandpass:
                    bandpass = OnlineBandpass(config.das_bp_bands, fs, config.das_filter_order, total_channels)
                    detrender = OnlineDetrender(total_channels, alpha=args.detrend_alpha)
                for band in config.das_bp_bands:
                    band_buffers[band] = RingBuffer(buffer_samples, total_channels, dtype=np.float32)

            n_samples = len(channels[0])
            start_idx, end_idx = extraction_indices(
                file_start_utc=item.start_utc,
                fs=fs,
                n_samples=n_samples,
                seg_start_utc=start_utc,
                seg_end_utc=end_utc,
            )
            if end_idx <= start_idx:
                continue

            for start in range(start_idx, end_idx, chunk_samples):
                end = min(end_idx, start + chunk_samples)
                cols = [np.asarray(ch[start:end]) for ch in channels]
                raw_chunk = np.column_stack(cols).astype(np.float32)
                if raw_chunk.size == 0:
                    continue

                # JSON signal packet (optionally downsampled)
                if args.signal_downsample > 1:
                    send_chunk = raw_chunk[::args.signal_downsample, :]
                    sample_rate = fs / args.signal_downsample
                else:
                    send_chunk = raw_chunk
                    sample_rate = fs

                stream_time = global_written / fs
                if send_chunk.size > 0:
                    if args.protocol == "udp" and args.udp_max_samples > 0:
                        step = int(args.udp_max_samples)
                        base_ts = stream_time
                        dt = 1.0 / float(sample_rate)
                        for i in range(0, send_chunk.shape[0], step):
                            sub = send_chunk[i:i + step, :]
                            payload = {
                                "packet_type": "signal",
                                "timestamp": base_ts + i * dt,
                                "sample_rate": float(sample_rate),
                                "sample_count": int(sub.shape[0]),
                                "total_channels": int(total_channels),
                                "signals": sub.tolist(),
                            }
                            if args.udp_max_bytes and args.udp_max_bytes > 0:
                                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                                if len(data) > args.udp_max_bytes:
                                    continue
                            send_packet(sock, args.protocol, args.host, args.port, payload)
                    else:
                        payload = {
                            "packet_type": "signal",
                            "timestamp": stream_time,
                            "sample_rate": float(sample_rate),
                            "sample_count": int(send_chunk.shape[0]),
                            "total_channels": int(total_channels),
                            "signals": send_chunk.tolist(),
                        }
                        if args.protocol == "udp" and args.udp_max_bytes and args.udp_max_bytes > 0:
                            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                            if len(data) > args.udp_max_bytes:
                                send_chunk = send_chunk[: max(1, int(send_chunk.shape[0] / 2)), :]
                                payload["signals"] = send_chunk.tolist()
                                payload["sample_count"] = int(send_chunk.shape[0])
                        send_packet(sock, args.protocol, args.host, args.port, payload)

                # Online preprocessing: match training config (bandpass on/off)
                if config.disable_das_bandpass:
                    bands = {band: raw_chunk for band in config.das_bp_bands}
                else:
                    # continuous detrend (avoids chunk-boundary artifacts from per-chunk de-mean)
                    x_for_filter = detrender.transform_chunk(raw_chunk)
                    bands = bandpass.filter_chunk(x_for_filter)
                for band, data in bands.items():
                    band_buffers[band].append(data.astype(np.float32))

                # Online inference on grid times
                latest_time = band_buffers[primary_band].total_written / fs - (config.cnn_window_s / 2.0)
                if latest_time > next_grid_time:
                    grid_times = np.arange(next_grid_time, latest_time, time_step)
                    windows = []
                    valid_times = []
                    half_win = int(config.cnn_window_s * fs / 2)
                    for t in grid_times:
                        center_idx = int(t * fs)
                        start_idx = center_idx - half_win
                        end_idx = center_idx + half_win
                        band_windows = []
                        ok = True
                        for band in config.das_bp_bands:
                            w = band_buffers[band].get_slice(start_idx, end_idx)
                            if w.shape[0] != 2 * half_win:
                                ok = False
                                break
                            band_windows.append(w.astype(np.float32))
                        if not ok:
                            continue
                        x = np.stack(band_windows, axis=0)
                        windows.append(x)
                        valid_times.append(t)

                    if windows:
                        X = np.stack(windows, axis=0)
                        probs = detector.predict_proba(X)[:, 1]
                        for t, p in zip(valid_times, probs):
                            prob_times.append(float(t))
                            prob_vals.append(float(p))

                    next_grid_time = float(grid_times[-1] + time_step) if len(grid_times) else next_grid_time

                # Peak detection with fixed latency
                global_written += (end - start)
                current_time = global_written / fs
                if args.buffer_seconds > 0:
                    while prob_times and prob_times[0] < current_time - args.buffer_seconds:
                        prob_times.popleft()
                        prob_vals.popleft()

                if len(prob_times) >= 3:
                    times = np.array(prob_times, dtype=np.float64)
                    probs = np.array(prob_vals, dtype=np.float64)
                    smooth_points = max(1, int(config.prob_smooth_points))
                    if smooth_points > 1 and len(probs) >= smooth_points:
                        kernel = np.ones(smooth_points, dtype=np.float64) / float(smooth_points)
                        probs_used = np.convolve(probs, kernel, mode="same")
                    else:
                        probs_used = probs

                    dt = np.median(np.diff(times)) if len(times) > 1 else time_step
                    min_dist = max(1, int(config.step_min_interval / dt))
                    peaks, props = find_peaks(probs_used, distance=min_dist, height=config.confidence_threshold)
                    heights = props.get("peak_heights")
                    if heights is None:
                        heights = probs_used[peaks]
                    for idx, height in zip(peaks, heights):
                        t = float(times[idx])
                        if t > current_time - latency:
                            continue
                        half_ch = int(0.15 * fs / 2)
                        center_idx = int(t * fs)
                        ch_start = center_idx - half_ch
                        ch_end = center_idx + half_ch
                        w = band_buffers[primary_band].get_slice(ch_start, ch_end)
                        ch, ch_conf = estimate_channel(w)
                        event_payload = {
                            "packet_type": "event",
                            "timestamp": t,
                            "channel_index": int(ch),
                            "confidence": float(height * ch_conf),
                        }
                        send_packet(sock, args.protocol, args.host, args.port, event_payload)
                        last_emitted_time = t

                if args.max_seconds is not None and current_time >= args.max_seconds:
                    return

                if args.speed and args.speed > 0:
                    time.sleep((end - start) / fs / args.speed)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
