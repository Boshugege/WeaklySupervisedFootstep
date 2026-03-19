#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


DEFAULT_TRAJECTORY_XZ = (
    "-14.1,-75.8;"
    "-14.1,-34.3;"
    "-16.5,-34.3;"
    "-16.5,-28.1;"
    "-41.1,-28.1;"
    "-41.1,-36.4;"
    "-31.5,-36.4;"
    "-31.5,-43.3"
)


def parse_xz_polyline(text: str) -> np.ndarray:
    points = []
    for token in str(text).split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [part.strip() for part in token.split(",")]
        if len(parts) != 2:
            continue
        try:
            x_val = float(parts[0])
            z_val = float(parts[1])
        except ValueError:
            continue
        points.append([x_val, z_val])
    if len(points) < 2:
        points = [[0.0, 0.0], [1.0, 0.0]]
    return np.asarray(points, dtype=np.float64)


def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    n_points = int(max(2, n_points))
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return np.repeat(points[:1, :], n_points, axis=0)

    deltas = np.diff(points, axis=0)
    seg_lens = np.sqrt(np.sum(deltas ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum[-1])
    if total_len <= 1e-12:
        return np.repeat(points[:1, :], n_points, axis=0)

    targets = np.linspace(0.0, total_len, n_points)
    sampled = np.empty((n_points, 2), dtype=np.float64)
    seg_idx = 0
    for idx, target in enumerate(targets):
        while seg_idx < len(seg_lens) - 1 and target > cum[seg_idx + 1]:
            seg_idx += 1
        seg_start = points[seg_idx]
        seg_end = points[seg_idx + 1]
        seg_len = max(seg_lens[seg_idx], 1e-12)
        frac = (target - cum[seg_idx]) / seg_len
        sampled[idx, :] = seg_start + frac * (seg_end - seg_start)
    return sampled


def choose_steps_csv(results_dir: Path, name: str) -> Path:
    infer = results_dir / f"{name}_steps_inference.csv"
    normal = results_dir / f"{name}_steps.csv"
    if infer.exists():
        return infer
    if normal.exists():
        return normal
    raise FileNotFoundError(f"No steps csv found in {results_dir}")


def infer_total_channels(root: Path, name: str, steps_path: Path | None) -> int:
    signal_csv = root / "output" / name / "signals" / f"{name}.csv"
    if signal_csv.exists():
        with signal_csv.open("r", encoding="utf-8") as f:
            header = f.readline().strip()
        if header:
            return int(len(header.split(",")))

    if steps_path is not None and steps_path.exists():
        df = pd.read_csv(steps_path)
        if not df.empty and "channel" in df.columns:
            return int(df["channel"].max()) + 1

    return 120


def detect_axis_geometry_from_image(img: np.ndarray):
    if img.ndim == 3:
        gray = img[..., :3].mean(axis=2)
    else:
        gray = img.astype(np.float64)

    gray = gray.astype(np.float64)
    if gray.max() > 1.5:
        gray = gray / 255.0

    h, w = gray.shape[:2]

    non_white = gray < 0.97
    row_ratio = non_white.mean(axis=1)
    col_ratio = non_white.mean(axis=0)

    rows = np.where(row_ratio > 0.01)[0]
    cols = np.where(col_ratio > 0.01)[0]

    if len(rows) > 0:
        y_top = int(rows.min())
        y_bottom = int(rows.max())
    else:
        y_top = int(0.12 * h)
        y_bottom = int(0.90 * h)

    if len(cols) > 0:
        x_left = int(cols.min())
        x_right = int(cols.max())
    else:
        x_left = int(0.05 * w)
        x_right = int(0.95 * w)

    search_end = int(x_left + max(4, 0.35 * (x_right - x_left)))
    search_end = max(x_left + 1, min(search_end, w - 1))

    if y_bottom <= y_top + 2:
        y_top = int(0.12 * h)
        y_bottom = int(0.90 * h)

    band = gray[y_top:y_bottom + 1, x_left:search_end + 1]
    if band.size == 0:
        x_axis = int(x_left + 0.08 * (x_right - x_left))
    else:
        darkness = (band < 0.35).mean(axis=0)
        best = int(np.argmax(darkness))
        if darkness[best] < 0.02:
            x_axis = int(x_left + 0.08 * (x_right - x_left))
        else:
            x_axis = int(x_left + best)

    x_axis = int(np.clip(x_axis, 0, w - 1))
    y_top = int(np.clip(y_top, 0, h - 1))
    y_bottom = int(np.clip(y_bottom, 0, h - 1))
    if y_bottom <= y_top:
        y_top, y_bottom = 0, h - 1

    return x_axis, y_top, y_bottom


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate simple morph animation: all channel dots slide to polyline-mapped positions.")
    p.add_argument("--name", default="wangdihai", help="Target name")
    p.add_argument("--results-dir", default=None, help="Results directory (default: output/<name>/results)")
    p.add_argument("--image", default=None, help="Path to left trajectory image")
    p.add_argument("--steps-csv", default=None, help="Path to steps csv")
    p.add_argument("--output", default=None, help="Output animation path (.gif or .mp4)")
    p.add_argument("--fps", type=int, default=24, help="Animation fps")
    p.add_argument("--duration", type=float, default=2.2, help="Animation duration in seconds")
    p.add_argument("--hold-seconds", type=float, default=0.0, help="Hold time in seconds at both start and end")
    p.add_argument("--total-channels", type=int, default=0, help="Total channels for dot stack (0 = auto infer)")
    p.add_argument("--polyline-xz", default=DEFAULT_TRAJECTORY_XZ, help="Polyline points: x,z;x,z;...")
    p.add_argument("--reverse", action="store_true", default=True, help="Reverse channel->polyline direction")
    p.add_argument("--no-reverse", dest="reverse", action="store_false", help="Disable reverse mapping")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).parent.resolve()
    results_dir = Path(args.results_dir) if args.results_dir else (root / "output" / args.name / "results")

    image_path = Path(args.image) if args.image else (results_dir / f"{args.name}_channel_trajectory.png")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    steps_path = Path(args.steps_csv) if args.steps_csv else choose_steps_csv(results_dir, args.name)
    if not steps_path.exists():
        raise FileNotFoundError(f"Steps csv not found: {steps_path}")

    output_path = Path(args.output) if args.output else (results_dir / f"{args.name}_channel_morph_anim.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.total_channels and args.total_channels > 1:
        total_channels = int(args.total_channels)
    else:
        total_channels = infer_total_channels(root, args.name, steps_path)
    total_channels = max(2, int(total_channels))

    img = plt.imread(str(image_path))
    img_h, img_w = img.shape[:2]

    axis_x, axis_y_top, axis_y_bottom = detect_axis_geometry_from_image(img)

    fps = max(1, int(args.fps))
    duration = max(0.2, float(args.duration))
    hold_seconds = max(0.0, float(args.hold_seconds))

    n_frames_core = max(2, int(duration * fps) + 1)
    frame_t_core = np.linspace(0.0, 1.0, n_frames_core)
    n_hold = int(round(hold_seconds * fps))
    if n_hold > 0:
        frame_t = np.concatenate([
            np.zeros((n_hold,), dtype=np.float64),
            frame_t_core,
            np.ones((n_hold,), dtype=np.float64),
        ])
    else:
        frame_t = frame_t_core
    n_frames = len(frame_t)

    ch_min = 0
    ch_max = total_channels - 1

    channels = np.arange(total_channels, dtype=np.float64)
    polyline = resample_polyline(parse_xz_polyline(args.polyline_xz), n_points=total_channels)
    if args.reverse:
        polyline = polyline[::-1, :]

    px = polyline[:, 0]
    pz = polyline[:, 1]
    px_min, px_max = float(np.min(px)), float(np.max(px))
    pz_min, pz_max = float(np.min(pz)), float(np.max(pz))
    px_span = max(1e-6, px_max - px_min)
    pz_span = max(1e-6, pz_max - pz_min)

    # Right demo axis coordinates: preserve x:z geometry while y matches left-image vertical span
    y_span = float(axis_y_bottom - axis_y_top)
    y_span = max(1e-6, y_span)
    target_y = axis_y_bottom - y_span * ((pz - pz_min) / pz_span)

    px_mid = 0.5 * (px_min + px_max)
    x_span_geom = y_span * (px_span / pz_span)
    x_span_geom = max(1e-6, x_span_geom)
    target_x = ((px - px_mid) / px_span) * x_span_geom

    x_pad = 0.08 * x_span_geom
    x_min_plot = float(np.min(target_x) - x_pad)
    x_max_plot = float(np.max(target_x) + x_pad)

    start_x_data = np.full_like(channels, float(axis_x), dtype=np.float64)
    if total_channels <= 1:
        start_y_data = np.full_like(channels, 0.5 * (axis_y_top + axis_y_bottom), dtype=np.float64)
    else:
        start_y_data = axis_y_bottom - (channels / float(total_channels - 1)) * (axis_y_bottom - axis_y_top)

    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.20, 0.55])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.imshow(img)
    ax_left.axis("off")
    ax_left.set_xlim(0, img_w)
    ax_left.set_ylim(img_h, 0)

    # Right panel: simple schematic morph
    ax_right.plot(target_x, target_y, color="0.45", linewidth=1.8, label="Polyline target")

    ax_right.set_xlim(x_min_plot, x_max_plot)
    ax_right.set_ylim(axis_y_bottom + 2.0, axis_y_top - 2.0)
    ax_right.set_aspect("equal", adjustable="box")
    ax_right.grid(True, alpha=0.2)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")

    # Overlay axis for cross-subplot motion (figure-normalized coordinates)
    ax_overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    ax_overlay.set_xlim(0.0, 1.0)
    ax_overlay.set_ylim(0.0, 1.0)
    ax_overlay.axis("off")

    fig.canvas.draw()

    def data_to_fig(ax, x_vals, y_vals):
        pts = np.column_stack([x_vals, y_vals])
        disp = ax.transData.transform(pts)
        return fig.transFigure.inverted().transform(disp)

    start_fig = data_to_fig(ax_left, start_x_data, start_y_data)
    target_fig = data_to_fig(ax_right, target_x, target_y)

    moving_dots = ax_overlay.scatter(start_fig[:, 0], start_fig[:, 1], s=18, c="red", alpha=0.95)

    def update(frame_idx):
        p = float(frame_t[frame_idx])
        p = p * p * (3.0 - 2.0 * p)  # smoothstep easing

        cur_xy = (1.0 - p) * start_fig + p * target_fig

        moving_dots.set_offsets(cur_xy)
        return (moving_dots,)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)

    if output_path.suffix.lower() == ".mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2500)
            anim.save(str(output_path), writer=writer)
        except Exception:
            fallback = output_path.with_suffix(".gif")
            anim.save(str(fallback), writer=PillowWriter(fps=fps))
            output_path = fallback
    else:
        anim.save(str(output_path), writer=PillowWriter(fps=fps))

    plt.close(fig)
    print(f"[OK] Animation saved: {output_path}")
    print(f"[INFO] Image: {image_path.name} | Steps: {steps_path.name} | Channels: {total_channels} | Frames: {n_frames}")


if __name__ == "__main__":
    main()
