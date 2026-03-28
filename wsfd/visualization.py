# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .config import Config

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

    def plot_signal_heatmap(self, energy_matrix, frame_times, output_path, title):
        """绘制纯信号能量热图（无脚步标记）"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        log_energy = np.log10(energy_matrix + 1e-10)

        im = ax.imshow(log_energy, aspect='auto', origin='lower',
                       extent=[frame_times[0], frame_times[-1],
                               0, energy_matrix.shape[0]],
                       cmap='viridis', interpolation='bilinear')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title(title, fontsize=14)

        plt.colorbar(im, ax=ax, label='Log Energy', shrink=0.9)
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

    def plot_audio_envelope_window(self, frame_times, audio_env, audio_step_times,
                                   output_path, window_s=50.0):
        """输出单张音频包络图：使用全时间窗口。"""
        frame_times = np.asarray(frame_times, dtype=np.float64)
        audio_env = np.asarray(audio_env, dtype=np.float64)
        audio_step_times = np.asarray(audio_step_times, dtype=np.float64)

        if len(frame_times) < 2 or len(audio_env) < 2:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(frame_times, audio_env, 'b-', linewidth=1.0)

        for t_step in audio_step_times:
            ax.axvline(t_step, color='cyan', alpha=0.7, linewidth=1)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Audio Envelope')
        ax.set_xlim(frame_times[0], frame_times[-1])
        ax.set_title('Audio Envelope (Full Window)')
        ax.grid(True, alpha=0.3)

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
