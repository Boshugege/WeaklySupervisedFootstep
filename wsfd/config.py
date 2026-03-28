# -*- coding: utf-8 -*-

class Config:
    """全局配置参数"""
    def __init__(self):
        # ===== 时间裁剪参数（核心需求）=====
        self.trim_start_s = 0.0      # 数据起始裁剪时间（秒）
        self.trim_end_s = None       # 数据结束裁剪时间（秒），None表示到结尾
        
        # ===== DAS参数 =====
        self.das_fs = 2000           # DAS采样率 (Hz)
        self.das_bp_bands = [        # 多频带滤波配置
            (5, 10),                # 主频带（用户指定）
            (10, 50),                # 高频补充
        ]
        self.das_filter_order = 4
        self.das_filter_method = 'filtfilt'  # 'filtfilt'（默认）或 'sosfilt'
        self.disable_das_bandpass = False
        
        # ===== 音频参数 =====
        self.audio_sr = 48000        # 音频重采样率
        self.audio_bp_low = 4000     # 音频带通低频截止 (Hz)
        self.audio_bp_high = 10000   # 音频带通高频截止 (Hz)
        self.audio_filter_order = 4
        self.audio_env_ms = 15       # RMS包络窗口 (ms)
        self.audio_smooth_ms = 30    # 包络平滑窗口 (ms)
        
        # ===== 脚步检测参数 =====
        self.step_min_interval = 0.5   # 最小脚步间隔 (秒)
        self.audio_peak_prom = 1.5      # 峰值显著性阈值
        self.audio_peak_height = 0.8    # 峰值高度阈值
        
        # ===== 弱标签参数 =====
        self.weak_label_sigma = 0.18    # 弱标签高斯扩展sigma (秒)
        self.time_tolerance = 0.5       # 时间对齐容差 (秒)
        
        # ===== 特征提取参数 =====
        self.feature_win_ms = 100       # 特征窗口 (ms)
        self.feature_step_ms = 25       # 特征步长 (ms)

        # ===== 路径级评估参数 =====
        self.path_gap_s = 1.2           # 轨迹分段时间阈值 T_gap (s)
        self.path_delta_c_max = 3       # 连续性阈值 Δc_max (ch)
        self.path_eps_d = 1             # 方向死区 ε_d (ch)
        
        # ===== 模型参数 =====
        self.model_type = 'auto'        # auto: 有CUDA用cnn，否则用rf
        self.n_estimators = 100
        self.self_train_rounds = 0      # 自训练轮数
        self.confidence_threshold = 0.40 # 高置信预测阈值
        self.prob_smooth_points = 4      # 概率曲线平滑窗口点数（>=1）
        self.device = 'auto'            # 'auto'/'cuda'/'cpu'
        self.torch_epochs = 50
        self.torch_batch_size = 128
        self.torch_lr = 1e-4
        self.torch_weight_decay = 1e-4
        self.torch_hidden_dim = 64
        self.torch_dropout = 0.1
        self.torch_patience = 8
        self.torch_val_interval = 5
        self.torch_amp = True
        self.cnn_window_s = 0.24
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
        if hasattr(args, 'das_filter_method') and args.das_filter_method is not None:
            self.das_filter_method = str(args.das_filter_method)
        if args.audio_sr is not None:
            self.audio_sr = args.audio_sr
        if args.output_dir is not None:
            self.output_dir = args.output_dir

