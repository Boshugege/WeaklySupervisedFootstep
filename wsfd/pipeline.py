# -*- coding: utf-8 -*-
import os
import re

import numpy as np
import pandas as pd

from .audio_labels import AudioWeakLabelExtractor
from .config import Config
from .detector import SelfTrainingIterator, WeaklySupervisedDetector
from .features import DASFeatureExtractor
from .visualization import Visualizer
from .signal_utils import bandpass_filter_2d


def _parse_channel_ids_from_columns(columns):
    ids = []
    pat = re.compile(r"^ch_(\d+)$")
    for i, col in enumerate(columns):
        m = pat.match(str(col))
        if m:
            ids.append(int(m.group(1)))
        else:
            ids.append(i)
    return ids


def _restore_channel_matrix(energy_matrix, channel_ids):
    if energy_matrix is None or len(channel_ids) == 0:
        return energy_matrix
    c, t = energy_matrix.shape
    if len(channel_ids) != c:
        return energy_matrix
    total_channels = int(max(channel_ids) + 1)
    if total_channels == c and min(channel_ids) == 0:
        return energy_matrix
    restored = np.zeros((total_channels, t), dtype=energy_matrix.dtype)
    restored[channel_ids, :] = energy_matrix
    return restored


def _map_local_channel_to_global(ch_local, channel_ids):
    ch_local = int(ch_local)
    if 0 <= ch_local < len(channel_ids):
        return int(channel_ids[ch_local])
    return ch_local

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
    
    channel_ids = _parse_channel_ids_from_columns(ch_cols)

    # 时间裁剪
    das_trimmed = das_extractor.trim_data(das_raw, 
                                          config.trim_start_s, 
                                          config.trim_end_s)
    
    # 多频带滤波
    das_bands = das_extractor.multi_band_filter(das_trimmed)
    primary_band = tuple(config.das_bp_bands[0])
    
    # 计算主频带能量矩阵（用于可视化）
    das_primary = das_bands[primary_band]
    energy_matrix, frame_times = das_extractor.compute_short_time_energy(das_primary)
    energy_matrix = _restore_channel_matrix(energy_matrix, channel_ids)
    
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
        ch, ch_conf = das_extractor.estimate_channel_for_time(das_primary, t)
        ch_global = _map_local_channel_to_global(ch, channel_ids)
        step_events.append((t, ch_global, prob * ch_conf))
    
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

    # 额外输出：未经过带通滤波的原始信号热图
    raw_energy_matrix, raw_frame_times = das_extractor.compute_short_time_energy(das_trimmed)
    raw_energy_matrix = _restore_channel_matrix(raw_energy_matrix, channel_ids)
    raw_heatmap_path = os.path.join(config.output_dir, f"{base_name}_heatmap_raw.png")
    viz.plot_signal_heatmap(
        raw_energy_matrix,
        raw_frame_times,
        raw_heatmap_path,
        title=f"DAS Signal Heatmap (Raw, No Bandpass): {base_name}",
    )

    # 额外输出：多频段带通后的信号热图（配色保持与主热图一致）
    heatmap_bands = [
        (2.5, 5),
        (5, 10),
        (10, 20),
        (20, 50),
        (50, 100),
        (100, 1000),
    ]
    for low, high in heatmap_bands:
        print(f"[Viz] Building band heatmap: {low}-{high}Hz")
        das_bp = bandpass_filter_2d(
            das_trimmed,
            config.das_fs,
            low,
            high,
            config.das_filter_order,
            method=config.das_filter_method,
        )
        band_energy, band_frame_times = das_extractor.compute_short_time_energy(das_bp)
        band_energy = _restore_channel_matrix(band_energy, channel_ids)
        band_tag = f"{str(low).replace('.', 'p')}_{str(high).replace('.', 'p')}Hz"
        band_heatmap_path = os.path.join(config.output_dir, f"{base_name}_heatmap_bp_{band_tag}.png")
        viz.plot_signal_heatmap(
            band_energy,
            band_frame_times,
            band_heatmap_path,
            title=f"DAS Signal Heatmap ({low}-{high}Hz)",
        )

    # 导出模型学习到的pattern（仅CNN模型）
    pattern_path = os.path.join(config.output_dir, f"{base_name}_learned_pattern.png")
    detector.export_learned_pattern(pattern_path)
    
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

        # 单图输出：音频包络稳定窗口（约30秒）
        audio_window_path = os.path.join(config.output_dir, f"{base_name}_audio_envelope_window.png")
        viz.plot_audio_envelope_window(
            frame_times,
            audio_env_resampled,
            audio_step_times,
            audio_window_path,
            window_s=50.0,
        )
    
    # ===== 6. 输出CSV结果 =====
    csv_output_path = os.path.join(config.output_dir, f"{base_name}_steps.csv")
    df_out = pd.DataFrame(step_events, columns=['time', 'channel', 'confidence'])
    df_out = df_out.sort_values('time').reset_index(drop=True)
    df_out.to_csv(csv_output_path, index=False)
    print(f"[Output] Steps CSV: {csv_output_path}")

    # ===== 7. 输出指标文本 =====
    eval_report = detector.last_eval_report or {}
    train_acc = detector.last_train_accuracy

    energy_curve = np.mean(energy_matrix, axis=0)
    step_times_for_mask = np.array([e[0] for e in step_events], dtype=np.float64)
    event_window_s = max(0.08, config.feature_win_ms * 1e-3)

    event_mask = np.zeros_like(frame_times, dtype=bool)
    for t in step_times_for_mask:
        event_mask |= np.abs(frame_times - t) <= event_window_s

    event_energy_mean = float(np.mean(energy_curve[event_mask])) if np.any(event_mask) else np.nan
    bg_mask = ~event_mask
    bg_energy_mean = float(np.mean(energy_curve[bg_mask])) if np.any(bg_mask) else np.nan

    if np.isfinite(event_energy_mean) and np.isfinite(bg_energy_mean) and bg_energy_mean > 0:
        event_bg_ratio = event_energy_mean / bg_energy_mean
        snr_db = 10.0 * np.log10(event_bg_ratio + 1e-12)
    else:
        event_bg_ratio = np.nan
        snr_db = np.nan

    def _fmt(v, digits=4):
        if v is None:
            return "N/A"
        try:
            fv = float(v)
            if not np.isfinite(fv):
                return "N/A"
            return f"{fv:.{digits}f}"
        except Exception:
            return "N/A"

    # ===== 7.1 路径级指标 (CR / DC / PJ) =====
    t_gap = float(getattr(config, 'path_gap_s', 1.2))
    delta_c_max = float(getattr(config, 'path_delta_c_max', 8))
    eps_d = float(getattr(config, 'path_eps_d', 1))

    sorted_events = sorted(step_events, key=lambda x: x[0])
    trajectories = []
    if len(sorted_events) > 0:
        cur = [sorted_events[0]]
        for ev in sorted_events[1:]:
            if (ev[0] - cur[-1][0]) > t_gap:
                trajectories.append(cur)
                cur = [ev]
            else:
                cur.append(ev)
        trajectories.append(cur)

    traj_valid = [traj for traj in trajectories if len(traj) >= 3]
    K = len(traj_valid)

    cr_num = 0.0
    cr_den = 0.0
    dc_vals = []
    pj_vals = []

    for traj in traj_valid:
        ch = np.array([p[1] for p in traj], dtype=np.float64)
        d = np.diff(ch)
        if len(d) == 0:
            continue

        # CR_k
        deltas = np.abs(d)
        cr_hits = float(np.sum(deltas <= delta_c_max))
        cr_num += cr_hits
        cr_den += float(len(d))

        # DC_k
        n_pos = int(np.sum(d > eps_d))
        n_neg = int(np.sum(d < -eps_d))
        denom = n_pos + n_neg
        if denom > 0:
            dc_vals.append(max(n_pos, n_neg) / float(denom))

        # PJ_k
        pj_vals.append(float(np.std(d)))

    CR = (cr_num / cr_den) if cr_den > 0 else np.nan
    DC = float(np.mean(dc_vals)) if len(dc_vals) > 0 else np.nan
    PJ = float(np.mean(pj_vals)) if len(pj_vals) > 0 else np.nan

    metrics_txt_path = os.path.join(config.output_dir, f"{base_name}_metrics.txt")
    lines = [
        f"PR-AUC: {_fmt(eval_report.get('pr_auc'))}",
        f"Precision: {_fmt(eval_report.get('best_precision'))}",
        f"Recall: {_fmt(eval_report.get('best_recall'))}",
        f"F1: {_fmt(eval_report.get('best_f1'))}",
        f"Best_threshold: {_fmt(eval_report.get('best_threshold'))}",
        f"Training_accuracy: {_fmt(train_acc)}",
        f"SNR(dB): {_fmt(snr_db)}",
        f"事件/背景能量比: {_fmt(event_bg_ratio)}",
        "",
        f"T_gap(s): {_fmt(t_gap, 3)}",
        f"Delta_c_max(ch): {_fmt(delta_c_max, 3)}",
        f"Epsilon_d(ch): {_fmt(eps_d, 3)}",
        f"轨迹数量K: {K}",
        f"CR: {_fmt(CR)}",
        f"DC: {_fmt(DC)}",
        f"PJ: {_fmt(PJ)}",
        f"Path-MAE(ch): N/A",
    ]
    with open(metrics_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Output] Metrics TXT: {metrics_txt_path}")
    
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
    channel_ids = _parse_channel_ids_from_columns(list(df_das.columns))
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
    primary_band = tuple(config.das_bp_bands[0])
    das_primary = das_bands[primary_band]
    
    # 能量矩阵
    energy_matrix, frame_times = das_extractor.compute_short_time_energy(das_primary)
    energy_matrix = _restore_channel_matrix(energy_matrix, channel_ids)
    
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
        ch, ch_conf = das_extractor.estimate_channel_for_time(das_primary, t)
        ch_global = _map_local_channel_to_global(ch, channel_ids)
        step_events.append((t, ch_global, prob * ch_conf))
    
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

    # 导出模型学习到的pattern（仅CNN模型）
    pattern_path = os.path.join(config.output_dir, f"{base_name}_learned_pattern_inference.png")
    detector.export_learned_pattern(pattern_path)
    
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
