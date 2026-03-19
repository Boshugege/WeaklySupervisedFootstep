# -*- coding: utf-8 -*-
"""
Receive newline-delimited JSON packets from TDMS2JSONStream and plot in real time.
"""

import argparse
import json
import queue
import socket
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def _parse_xz_polyline(text):
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


def _resample_polyline(points, n_points):
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


def _channel_to_polyline_index(channel_idx, total_channels, polyline_channels, reverse_direction=False):
    if total_channels <= 1 or polyline_channels <= 1:
        return 0
    ratio = float(channel_idx) / float(total_channels - 1)
    if reverse_direction:
        ratio = 1.0 - ratio
    mapped = int(round(ratio * float(polyline_channels - 1)))
    return int(np.clip(mapped, 0, polyline_channels - 1))


def _map_event_channel(ch_local, total_channels, offset=0, tail_trim=0):
    total_channels = int(max(1, total_channels))
    offset = int(max(0, offset))
    tail_trim = int(max(0, tail_trim))
    valid_end = total_channels - tail_trim - 1
    valid_end = max(0, valid_end)
    mapped = int(round(float(ch_local))) + offset
    return int(np.clip(mapped, 0, valid_end))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Receive JSON stream and plot signals in real time."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9000, help="Bind port")
    parser.add_argument(
        "--protocol",
        choices=["udp", "tcp"],
        default="udp",
        help="Network protocol",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=5.0,
        help="Plot window in seconds",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=0,
        help="Max channels to plot (0 for all)",
    )
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=20.0,
        help="Plot refresh rate",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Fallback sample rate if packets omit it",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=2000,
        help="Max events to keep in memory",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        default=True,
        help="Display event markers",
    )
    parser.add_argument(
        "--hide-events",
        dest="show_events",
        action="store_false",
        help="Hide event markers",
    )
    parser.add_argument(
        "--show-heatmap",
        action="store_true",
        default=False,
        help="Display heatmap image",
    )
    parser.add_argument(
        "--trajectory-xz",
        default=DEFAULT_TRAJECTORY_XZ,
        help="Polyline control points in 'x,z;x,z;...' format",
    )
    parser.add_argument(
        "--trajectory-channels",
        type=int,
        default=0,
        help="Channel count used for polyline mapping (0 uses stream total_channels)",
    )
    parser.add_argument(
        "--channel-offset",
        type=int,
        default=0,
        help="Add this offset to incoming event channel indices before plotting/mapping",
    )
    parser.add_argument(
        "--channel-tail-trim",
        type=int,
        default=0,
        help="Reserve this many channels at tail (max index becomes total_channels-tail_trim-1)",
    )
    parser.add_argument(
        "--reverse-channel-direction",
        action="store_true",
        default=True,
        help="Reverse mapping so channel 0 maps to polyline tail and max channel maps to head",
    )
    parser.add_argument(
        "--no-reverse-channel-direction",
        dest="reverse_channel_direction",
        action="store_false",
        help="Disable channel-direction reversal",
    )
    parser.add_argument(
        "--trajectory-fade-seconds",
        type=float,
        default=3.0,
        help="Fade-in/out duration in seconds for trajectory light",
    )
    parser.add_argument(
        "--trajectory-lost-timeout",
        type=float,
        default=3.0,
        help="Seconds without event before trajectory light starts fading out",
    )
    parser.add_argument(
        "--trajectory-position-alpha",
        type=float,
        default=0.35,
        help="EMA alpha for channel-position update from new events",
    )
    parser.add_argument(
        "--trajectory-smooth-speed",
        type=float,
        default=5.0,
        help="Display position interpolation speed for trajectory light",
    )
    parser.add_argument(
        "--trajectory-max-association-distance",
        type=float,
        default=15.0,
        help="Max per-event channel jump allowed before gating",
    )
    parser.add_argument(
        "--trajectory-max-velocity",
        type=float,
        default=30.0,
        help="Max channel velocity (channels/s) for jump prevention",
    )
    parser.add_argument(
        "--trajectory-velocity-alpha",
        type=float,
        default=0.35,
        help="EMA alpha for light velocity update",
    )
    parser.add_argument(
        "--trajectory-point-size",
        type=float,
        default=30.0,
        help="Core light point size",
    )
    parser.add_argument(
        "--trajectory-glow-size",
        type=float,
        default=180.0,
        help="Outer glow point size",
    )
    return parser.parse_args(argv)


class StreamBuffer:
    def __init__(self, window_seconds, max_channels):
        self.window_seconds = window_seconds
        self.max_channels = max_channels
        self.sample_rate = None
        self.total_channels = None
        self.plot_channels = None
        self.window_samples = None
        self.time_deque = None
        self.channel_deques = None
        self.last_timestamp = None

    def _init_buffers(self, total_channels, sample_rate):
        self.sample_rate = sample_rate
        self.total_channels = total_channels
        if self.max_channels is None or self.max_channels <= 0:
            self.plot_channels = total_channels
        else:
            self.plot_channels = min(total_channels, self.max_channels)
        # Keep all received samples in memory for accumulated plotting
        self.window_samples = None
        self.time_deque = deque()
        self.channel_deques = [
            deque() for _ in range(self.plot_channels)
        ]
        self.last_timestamp = None

    def _ensure_buffers(self, total_channels, sample_rate):
        if self.time_deque is None:
            self._init_buffers(total_channels, sample_rate)
            return
        if total_channels != self.total_channels:
            self._init_buffers(total_channels, sample_rate)
            return
        if sample_rate != self.sample_rate:
            self._init_buffers(total_channels, sample_rate)
            return

    def append_sample(self, timestamp, values):
        if self.time_deque is None:
            return
        self.time_deque.append(float(timestamp))
        if not isinstance(values, (list, tuple, np.ndarray)):
            return
        limit = min(self.plot_channels, len(values))
        for idx in range(self.plot_channels):
            if idx < limit:
                self.channel_deques[idx].append(float(values[idx]))
            else:
                self.channel_deques[idx].append(0.0)

        self.last_timestamp = timestamp

    def update_from_packet(self, payload, fallback_sample_rate):
        total_channels = int(payload.get("total_channels", 0))
        signals = payload.get("signals")
        if not total_channels or signals is None:
            return
        sample_rate = payload.get("sample_rate") or fallback_sample_rate
        if sample_rate is None:
            return
        self._ensure_buffers(total_channels, float(sample_rate))

        sample_count = int(payload.get("sample_count", 1))
        timestamp = payload.get("timestamp")
        if timestamp is None:
            timestamp = time.time()

        if not isinstance(signals, list) or not signals:
            return

        # Handle batched samples if signals is list of lists.
        # NOTE: sender may send sample_count==1 with signals shaped as [[ch0,...]].
        if isinstance(signals[0], list):
            step = 1.0 / self.sample_rate if self.sample_rate else 0.0
            base_ts = float(timestamp)
            for offset, values in enumerate(signals):
                self.append_sample(base_ts + offset * step, values)
            return

        # Handle single sample as flat list [ch0, ch1, ...]
        if sample_count <= 1:
            self.append_sample(float(timestamp), signals)


class Receiver(threading.Thread):
    def __init__(self, host, port, protocol, out_queue):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.protocol = protocol
        self.out_queue = out_queue
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _run_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(0.5)
        try:
            while not self._stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                try:
                    payload = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                self.out_queue.put(payload)
        finally:
            sock.close()

    def _run_tcp(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(0.5)
        conn = None
        buffer = b""
        try:
            while not self._stop_event.is_set():
                if conn is None:
                    try:
                        conn, _ = server.accept()
                        conn.settimeout(0.5)
                    except socket.timeout:
                        continue
                try:
                    chunk = conn.recv(65535)
                except socket.timeout:
                    continue
                if not chunk:
                    conn.close()
                    conn = None
                    buffer = b""
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        payload = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    self.out_queue.put(payload)
        finally:
            if conn is not None:
                conn.close()
            server.close()

    def run(self):
        if self.protocol == "udp":
            self._run_udp()
        else:
            self._run_tcp()


def main(argv=None):
    args = _parse_args(argv)
    packet_queue = queue.Queue()
    buffer = StreamBuffer(args.window_seconds, args.max_channels)
    event_times = deque(maxlen=args.max_events)
    event_channels = deque(maxlen=args.max_events)
    raw_polyline = _parse_xz_polyline(args.trajectory_xz)
    state = {
        "observed_max_event_channel": -1,
        "last_event_time": None,
        "last_signal_time": None,
        "last_channel_count": None,
        "last_polyline_channels": None,
        "sampled_polyline": None,
        "fixed_vabs": None,
        "light_active": False,
        "light_channel_position": None,
        "light_display_xy": None,
        "light_target_alpha": 0.0,
        "light_alpha": 0.0,
        "light_last_event_wall": None,
        "light_last_update_wall": time.time(),
        "light_velocity": 0.0,
        "light_last_event_ts": None,
    }

    receiver = Receiver(args.host, args.port, args.protocol, packet_queue)
    receiver.start()

    fig, (ax, ax_traj) = plt.subplots(1, 2, figsize=(12, 5))
    ax.set_title("TDMS JSON Stream")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")

    ax_traj.set_title("Polyline Trajectory")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Z")
    ax_traj.grid(True, alpha=0.2)

    image = ax.imshow(
        np.zeros((1, 1), dtype=float),
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="seismic",
    )
    image.set_visible(args.show_heatmap)
    event_scatter = ax.scatter([], [], s=30, c="black", marker="x")
    event_scatter.set_visible(args.show_events)
    trajectory_line, = ax_traj.plot([], [], color="tab:gray", linewidth=1.5)
    trajectory_glow = ax_traj.scatter([], [], s=args.trajectory_glow_size, c="tab:red")
    trajectory_scatter = ax_traj.scatter([], [], s=args.trajectory_point_size, c="#fff2a8")
    empty_xy = np.empty((0, 2), dtype=float)
    trajectory_glow.set_offsets(empty_xy)
    trajectory_scatter.set_offsets(empty_xy)

    def _effective_polyline_channels():
        tail_trim = max(0, int(args.channel_tail_trim))
        if args.trajectory_channels and args.trajectory_channels > 1:
            return int(max(2, int(args.trajectory_channels) - tail_trim))
        if buffer.total_channels and buffer.total_channels > 1:
            return int(max(2, int(buffer.total_channels) - tail_trim))
        if state["observed_max_event_channel"] >= 1:
            return int(max(2, int(state["observed_max_event_channel"] + 1 - tail_trim)))
        return None

    def _ensure_polyline_ready():
        channels = _effective_polyline_channels()
        if channels is None:
            return
        if state["last_polyline_channels"] == channels and state["sampled_polyline"] is not None:
            return

        sampled_polyline = _resample_polyline(raw_polyline, channels)
        state["sampled_polyline"] = sampled_polyline
        trajectory_line.set_data(sampled_polyline[:, 0], sampled_polyline[:, 1])

        pad = 0.05
        x_min, x_max = float(np.min(sampled_polyline[:, 0])), float(np.max(sampled_polyline[:, 0]))
        z_min, z_max = float(np.min(sampled_polyline[:, 1])), float(np.max(sampled_polyline[:, 1]))
        x_span = max(1e-6, x_max - x_min)
        z_span = max(1e-6, z_max - z_min)
        ax_traj.set_xlim(x_min - pad * x_span, x_max + pad * x_span)
        ax_traj.set_ylim(z_min - pad * z_span, z_max + pad * z_span)
        ax_traj.set_aspect("equal", adjustable="box")

        state["last_polyline_channels"] = channels

    def _refresh(_frame):
        updated = False
        while True:
            try:
                payload = packet_queue.get_nowait()
            except queue.Empty:
                break
            if payload.get("packet_type") == "event":
                timestamp = payload.get("timestamp")
                channel_index = payload.get("channel_index")
                if timestamp is not None and channel_index is not None:
                    ts = float(timestamp)
                    total_for_map = int(buffer.total_channels) if buffer.total_channels else int(args.trajectory_channels or 1)
                    ch_idx = _map_event_channel(
                        channel_index,
                        total_channels=total_for_map,
                        offset=args.channel_offset,
                        tail_trim=args.channel_tail_trim,
                    )
                    event_times.append(ts)
                    event_channels.append(float(ch_idx))
                    state["observed_max_event_channel"] = max(state["observed_max_event_channel"], ch_idx)
                    state["last_event_time"] = ts

                    _ensure_polyline_ready()
                    sampled_polyline = state["sampled_polyline"]
                    if sampled_polyline is not None and len(sampled_polyline) > 0:
                        total_channels_for_map = _effective_polyline_channels()
                        if total_channels_for_map is not None and total_channels_for_map > 0:
                            pos_alpha = float(np.clip(args.trajectory_position_alpha, 0.01, 1.0))
                            vel_alpha = float(np.clip(args.trajectory_velocity_alpha, 0.01, 1.0))
                            max_assoc = max(1.0, float(args.trajectory_max_association_distance))
                            max_vel = max(0.5, float(args.trajectory_max_velocity))
                            if state["light_channel_position"] is None:
                                state["light_channel_position"] = float(ch_idx)
                                state["light_velocity"] = 0.0
                            else:
                                prev_pos = float(state["light_channel_position"])
                                prev_vel = float(state["light_velocity"])

                                prev_ts = state["light_last_event_ts"]
                                if prev_ts is not None and ts > float(prev_ts):
                                    dt_evt = max(1e-3, float(ts - float(prev_ts)))
                                else:
                                    wall_now_local = time.time()
                                    last_wall = state["light_last_event_wall"]
                                    if last_wall is None:
                                        dt_evt = 0.03
                                    else:
                                        dt_evt = max(1e-3, float(wall_now_local - float(last_wall)))

                                predicted = prev_pos + prev_vel * dt_evt
                                dist_pred = abs(float(ch_idx) - predicted)
                                dist_raw = abs(float(ch_idx) - prev_pos)
                                dist_use = min(dist_pred, dist_raw)
                                speed_raw = dist_raw / dt_evt

                                gated_ch = float(ch_idx)
                                if dist_use > max_assoc or speed_raw > max_vel:
                                    delta = float(ch_idx) - prev_pos
                                    clipped_delta = float(np.clip(delta, -max_assoc, max_assoc))
                                    gated_ch = prev_pos + clipped_delta

                                raw_vel = (gated_ch - prev_pos) / dt_evt
                                new_vel = (1.0 - vel_alpha) * prev_vel + vel_alpha * raw_vel
                                state["light_velocity"] = float(np.clip(new_vel, -max_vel, max_vel))
                                state["light_channel_position"] = (
                                    (1.0 - pos_alpha) * prev_pos + pos_alpha * gated_ch
                                )

                            mapped_idx = _channel_to_polyline_index(
                                float(state["light_channel_position"]),
                                total_channels_for_map,
                                len(sampled_polyline),
                                reverse_direction=bool(args.reverse_channel_direction),
                            )
                            x_val, z_val = sampled_polyline[mapped_idx]

                            if state["light_display_xy"] is None:
                                state["light_display_xy"] = np.array([x_val, z_val], dtype=np.float64)

                            state["light_active"] = True
                            state["light_target_alpha"] = 1.0
                            state["light_alpha"] = max(float(state["light_alpha"]), 0.35)
                            state["light_last_event_wall"] = time.time()
                            state["light_last_event_ts"] = ts
            else:
                buffer.update_from_packet(payload, args.sample_rate)
                if buffer.last_timestamp is not None:
                    state["last_signal_time"] = float(buffer.last_timestamp)
                _ensure_polyline_ready()
            updated = True

        if buffer.time_deque is not None:
            times = list(buffer.time_deque)
            if times:
                data = np.array([list(channel) for channel in buffer.channel_deques], dtype=float)

                x_max = times[-1]
                x_min = times[0]
                image.set_data(data)
                image.set_extent((x_min, x_max, -0.5, buffer.plot_channels - 0.5))
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(-0.5, buffer.plot_channels - 0.5)

                finite_vals = data[np.isfinite(data)]
                if finite_vals.size > 0:
                    if state["fixed_vabs"] is None:
                        v_abs = np.percentile(np.abs(finite_vals), 99.0)
                        if not np.isfinite(v_abs) or v_abs <= 0:
                            v_abs = max(1.0, float(np.std(finite_vals)))
                        state["fixed_vabs"] = float(v_abs)
                    if args.show_heatmap:
                        image.set_clim(-state["fixed_vabs"], state["fixed_vabs"])

                if state["last_channel_count"] != buffer.plot_channels:
                    tick_step = 10
                    ticks = np.arange(0, buffer.plot_channels, tick_step)
                    ax.set_yticks(ticks)
                    ax.set_yticklabels([str(idx) for idx in ticks])
                    state["last_channel_count"] = buffer.plot_channels

        if args.show_events:
            if event_times and event_channels:
                offsets = np.column_stack([list(event_times), list(event_channels)])
                event_scatter.set_offsets(offsets)
            else:
                event_scatter.set_offsets(np.empty((0, 2)))
        else:
            event_scatter.set_offsets(np.empty((0, 2)))

        now_wall = time.time()
        dt_wall = max(1e-4, now_wall - float(state["light_last_update_wall"]))
        state["light_last_update_wall"] = now_wall

        sampled_polyline = state["sampled_polyline"]
        total_channels_for_map = _effective_polyline_channels()
        if (
            state["light_active"]
            and state["light_channel_position"] is not None
            and sampled_polyline is not None
            and len(sampled_polyline) > 0
            and total_channels_for_map is not None
            and total_channels_for_map > 0
        ):
            mapped_idx = _channel_to_polyline_index(
                float(state["light_channel_position"]),
                total_channels_for_map,
                len(sampled_polyline),
                reverse_direction=bool(args.reverse_channel_direction),
            )
            target_xy = sampled_polyline[mapped_idx].astype(np.float64)

            if state["light_display_xy"] is None:
                state["light_display_xy"] = target_xy.copy()
            else:
                smooth = 1.0 - np.exp(-max(0.1, float(args.trajectory_smooth_speed)) * dt_wall)
                state["light_display_xy"] = (
                    (1.0 - smooth) * state["light_display_xy"] + smooth * target_xy
                )

            lost_timeout = max(0.01, float(args.trajectory_lost_timeout))
            last_evt_wall = state["light_last_event_wall"]
            if last_evt_wall is not None and (now_wall - float(last_evt_wall)) > lost_timeout:
                state["light_target_alpha"] = 0.0

            fade_s = max(0.01, float(args.trajectory_fade_seconds))
            if state["light_target_alpha"] > state["light_alpha"]:
                state["light_alpha"] = min(
                    float(state["light_target_alpha"]),
                    float(state["light_alpha"] + dt_wall / fade_s),
                )
            else:
                state["light_alpha"] = max(
                    float(state["light_target_alpha"]),
                    float(state["light_alpha"] - dt_wall / fade_s),
                )

            if state["light_alpha"] <= 0.01 and state["light_target_alpha"] <= 0.0:
                state["light_active"] = False

        if state["light_active"] and state["light_display_xy"] is not None and state["light_alpha"] > 0.0:
            xy = np.asarray(state["light_display_xy"], dtype=np.float64).reshape(1, 2)
            trajectory_glow.set_offsets(xy)
            trajectory_scatter.set_offsets(xy)

            glow_colors = np.array([[1.0, 0.2, 0.2, 0.35 * float(state["light_alpha"])]], dtype=np.float64)
            core_colors = np.array([[1.0, 0.95, 0.66, float(state["light_alpha"])]], dtype=np.float64)
            trajectory_glow.set_facecolors(glow_colors)
            trajectory_scatter.set_facecolors(core_colors)
        else:
            empty_xy = np.empty((0, 2), dtype=float)
            trajectory_glow.set_offsets(empty_xy)
            trajectory_scatter.set_offsets(empty_xy)

        return (image, event_scatter, trajectory_line, trajectory_glow, trajectory_scatter)

    interval_ms = max(10, int(1000.0 / args.refresh_hz))
    anim = FuncAnimation(fig, _refresh, interval=interval_ms, blit=False)

    try:
        plt.show()
    finally:
        receiver.stop()
        receiver.join(timeout=1.0)


if __name__ == "__main__":
    main()
