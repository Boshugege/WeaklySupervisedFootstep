# -*- coding: utf-8 -*-
import argparse

from .config import Config
from .pipeline import run_inference_only, run_pipeline

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
    parser.add_argument('--das_filter_method', type=str, default='filtfilt',
                        choices=['filtfilt', 'sosfilt'],
                        help='DAS带通滤波方法：filtfilt(默认, 零相位) 或 sosfilt(因果)')
    parser.add_argument('--disable_das_bandpass', action='store_true',
                        help='关闭DAS带通滤波，直接使用原始DAS信号训练/推理')
    
    # 时间对齐
    parser.add_argument('--align_dt', type=float, default=0.0,
                        help='时间对齐偏移量：DAS时间 = 音频时间 + align_dt')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'rf', 'cnn'],
                        help='模型类型：auto/rf/cnn；auto=有CUDA用cnn，无CUDA用rf')
    parser.add_argument('--self_train_rounds', type=int, default=0,
                        help='自训练迭代轮数，默认0')
    parser.add_argument('--confidence_threshold', type=float, default=0.35,
                        help='高置信预测阈值，默认0.35')
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
    if args.disable_das_bandpass:
        config.disable_das_bandpass = True
    
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
