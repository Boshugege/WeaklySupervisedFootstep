# -*- coding: utf-8 -*-
import os

import joblib
import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import DASFeatureExtractor
from .models import TORCH_AVAILABLE, TorchBinaryClassifier, torch

class WeaklySupervisedDetector:
    """弱监督脚步检测器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.last_eval_report = None
        self.recommended_threshold = None
        self.last_train_accuracy = None

    def _effective_model_type(self):
        has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        selected = self.config.model_type
        if selected == 'auto':
            selected = 'cnn' if has_cuda else 'rf'
            print(f"[Model] Auto-selecting model: {'CNN (CUDA available)' if has_cuda else 'RandomForest (no CUDA)'}")
        elif selected == 'cnn' and not has_cuda:
            print("[Model] CUDA不可用，自动回退到rf")
            selected = 'rf'
        elif selected == 'cnn' and has_cuda:
            print("[Model] Using CNN with CUDA")
        elif selected == 'rf':
            print("[Model] Using RandomForest")
        return selected

    def _build_model(self):
        """按配置构建分类器"""
        selected = self._effective_model_type()

        if selected == 'cnn':
            return TorchBinaryClassifier(self.config)
        if selected == 'rf':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=10,
                random_state=42,
                n_jobs=1
            )
        raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    def _run_validation_report(self, X, y):
        """
        训练前做一次轻量验证：
        - 时间无关分层划分（默认 80/20）
        - 输出 PR-AUC / Precision / Recall / F1
        - 扫描阈值给出推荐值
        """
        self.last_eval_report = None
        self.recommended_threshold = None

        n_samples = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        min_class = min(n_pos, n_neg)

        print(f"[Eval] Samples={n_samples}, Pos={n_pos}, Neg={n_neg}")

        if n_samples < 30 or min_class < 8:
            print("[Eval] Skip validation: sample size too small for stable split")
            return

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model_eval = self._build_model()
        if self._effective_model_type() == 'cnn':
            model_eval.fit(X_train, y_train)
            val_probs = model_eval.predict_proba(X_val)[:, 1]
        else:
            scaler_eval = StandardScaler()
            X_train_scaled = scaler_eval.fit_transform(X_train)
            X_val_scaled = scaler_eval.transform(X_val)
            model_eval.fit(X_train_scaled, y_train)
            val_probs = model_eval.predict_proba(X_val_scaled)[:, 1]
        pr_auc = average_precision_score(y_val, val_probs)

        thresholds = np.arange(0.30, 0.91, 0.05)
        rows = []
        best = None
        best_key = (-1.0, -1.0, -1.0, 0.0)  # F1, Precision, Recall, -|thr-0.5|

        print("[Eval] Validation metrics by threshold:")
        print("       thr    P      R      F1")
        for thr in thresholds:
            y_pred = (val_probs >= thr).astype(np.int32)
            p = precision_score(y_val, y_pred, zero_division=0)
            r = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            rows.append({'threshold': float(thr), 'precision': float(p), 'recall': float(r), 'f1': float(f1)})
            print(f"       {thr:0.2f}  {p:0.3f}  {r:0.3f}  {f1:0.3f}")

            key = (f1, p, r, -abs(float(thr) - 0.5))
            if key > best_key:
                best_key = key
                best = rows[-1]

        if best is None:
            return

        self.recommended_threshold = float(best['threshold'])
        self.last_eval_report = {
            'n_samples': int(n_samples),
            'n_pos': int(n_pos),
            'n_neg': int(n_neg),
            'pr_auc': float(pr_auc),
            'best_threshold': float(best['threshold']),
            'best_precision': float(best['precision']),
            'best_recall': float(best['recall']),
            'best_f1': float(best['f1']),
            'grid': rows,
        }

        print(f"[Eval] PR-AUC={pr_auc:.4f}")
        print(f"[Eval] Best threshold={best['threshold']:.2f} (P={best['precision']:.3f}, "
              f"R={best['recall']:.3f}, F1={best['f1']:.3f})")
        print(f"[Eval] Current confidence_threshold={self.config.confidence_threshold:.2f}")
        print(f"[Eval] Suggestion: try --confidence_threshold {best['threshold']:.2f}")

    def prepare_training_data(self, das_bands, audio_step_times, 
                              neg_ratio=2.0, time_range=None):
        """
        准备训练数据
        - 正样本：音频检测到的脚步时间点附近
        - 负样本：远离任何脚步的时间点
        """
        if time_range is None:
            primary_band = tuple(self.config.das_bp_bands[0])
            T = das_bands[primary_band].shape[0]
            time_range = (0, T / self.config.das_fs)
        
        t_min, t_max = time_range
        
        # 正样本时间
        pos_times = audio_step_times[(audio_step_times >= t_min) & 
                                      (audio_step_times <= t_max)]
        
        # 生成负样本时间（包含 hard negatives + easy negatives）
        # 旧逻辑使用过大的排除半径，容易导致负样本过少。
        exclude_radius = max(0.08, self.config.step_min_interval * 0.6)
        hard_upper = max(exclude_radius + 0.05, self.config.step_min_interval * 1.2)
        all_times = np.arange(t_min + 0.35, t_max - 0.35, 0.03)

        if len(pos_times) == 0:
            neg_times = all_times
        else:
            pos_sorted = np.sort(pos_times)
            idx = np.searchsorted(pos_sorted, all_times)
            left_d = np.where(idx > 0, np.abs(all_times - pos_sorted[np.clip(idx - 1, 0, len(pos_sorted) - 1)]), np.inf)
            right_d = np.where(idx < len(pos_sorted), np.abs(pos_sorted[np.clip(idx, 0, len(pos_sorted) - 1)] - all_times), np.inf)
            min_d = np.minimum(left_d, right_d)

            hard_pool = all_times[(min_d >= exclude_radius) & (min_d <= hard_upper)]
            easy_pool = all_times[min_d > hard_upper]

            n_neg_target = int(max(len(pos_times) * neg_ratio, 120))
            n_hard = min(len(hard_pool), int(n_neg_target * 0.6))
            n_easy = min(len(easy_pool), n_neg_target - n_hard)

            hard_sel = np.random.choice(hard_pool, n_hard, replace=False) if n_hard > 0 else np.array([])
            easy_sel = np.random.choice(easy_pool, n_easy, replace=False) if n_easy > 0 else np.array([])
            neg_times = np.sort(np.concatenate([hard_sel, easy_sel]))
        
        print(
            f"[Train] Positive samples: {len(pos_times)}, Negative samples: {len(neg_times)} "
            f"(neg_ratio={neg_ratio}, exclude_radius={exclude_radius:.2f}s)"
        )
        
        model_type = self._effective_model_type()

        # 提取特征/窗口
        das_extractor = DASFeatureExtractor(self.config)
        if model_type == 'cnn':
            pos_x, _ = das_extractor.extract_cnn_windows_at_times(
                das_bands, pos_times, window_s=self.config.cnn_window_s)
            neg_x, _ = das_extractor.extract_cnn_windows_at_times(
                das_bands, neg_times, window_s=self.config.cnn_window_s)
            X = np.concatenate([pos_x, neg_x], axis=0)
            y = np.concatenate([np.ones(len(pos_x)), np.zeros(len(neg_x))])
        else:
            pos_features, _ = das_extractor.extract_features_at_times(
                das_bands, pos_times, window_s=0.2)
            neg_features, _ = das_extractor.extract_features_at_times(
                das_bands, neg_times, window_s=0.2)
            X = np.vstack([pos_features, neg_features])
            y = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])

        # 打乱
        perm = np.random.permutation(len(y))
        X = X[perm]
        y = y[perm]
        
        return X, y
    
    def train(self, X, y):
        """训练模型"""
        # 在正式训练前输出一份验证指标报告（不改变最终训练行为）
        self._run_validation_report(X, y)

        # 选择模型
        self.model = self._build_model()

        if isinstance(self.model, TorchBinaryClassifier):
            self.model.fit(X, y)
            train_acc = self.model.score(X, y)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            train_acc = self.model.score(X_scaled, y)

        # 训练集准确率
        self.last_train_accuracy = float(train_acc)
        print(f"[Train] Training accuracy: {train_acc:.4f}")
        
        return self.model
    
    def predict_on_grid(self, das_bands, time_step=0.05, time_range=None):
        """在时间网格上预测脚步概率"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if time_range is None:
            primary_band = tuple(self.config.das_bp_bands[0])
            T = das_bands[primary_band].shape[0]
            time_range = (0.5, T / self.config.das_fs - 0.5)
        
        t_min, t_max = time_range
        grid_times = np.arange(t_min, t_max, time_step)
        
        das_extractor = DASFeatureExtractor(self.config)
        if isinstance(self.model, TorchBinaryClassifier):
            chunk = max(32, int(self.config.cnn_predict_chunk))
            all_times = []
            all_probs = []
            for i in range(0, len(grid_times), chunk):
                batch_times = grid_times[i:i + chunk]
                windows, valid_times = das_extractor.extract_cnn_windows_at_times(
                    das_bands, batch_times, window_s=self.config.cnn_window_s)
                if len(windows) == 0:
                    continue
                probs = self.model.predict_proba(windows)[:, 1]
                all_times.append(valid_times)
                all_probs.append(probs)
            if not all_times:
                return np.array([]), np.array([])
            return np.concatenate(all_times), np.concatenate(all_probs)
        else:
            features, valid_times = das_extractor.extract_features_at_times(
                das_bands, grid_times, window_s=0.2)
            if len(features) == 0:
                return np.array([]), np.array([])
            X_scaled = self.scaler.transform(features)
            probs = self.model.predict_proba(X_scaled)[:, 1]
            return valid_times, probs
    
    def detect_steps_from_probs(self, times, probs, threshold=0.35, min_interval=0.45):
        """从概率曲线中检测脚步事件"""
        probs = np.asarray(probs, dtype=np.float64)
        smooth_points = int(max(1, getattr(self.config, "prob_smooth_points", 1)))
        if smooth_points > 1 and len(probs) >= smooth_points:
            kernel = np.ones(smooth_points, dtype=np.float64) / float(smooth_points)
            probs_used = np.convolve(probs, kernel, mode='same')
        else:
            probs_used = probs

        # 峰值检测
        dt = np.median(np.diff(times)) if len(times) > 1 else 0.05
        min_dist = max(1, int(min_interval / dt))
        
        peaks, props = find_peaks(probs_used, distance=min_dist, height=threshold)
        
        step_times = times[peaks]
        step_probs = probs_used[peaks]
        
        return step_times, step_probs

    def export_learned_pattern(self, output_png_path):
        if isinstance(self.model, TorchBinaryClassifier):
            self.model.export_learned_pattern(output_png_path)
            return True
        print("[Viz] Learned pattern export skipped: current model is not CNN.")
        return False
    
    def save_model(self, model_path):
        """
        保存训练好的模型到文件
        
        Args:
            model_path: 模型保存路径 (.joblib 文件)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # 保存模型和标准化器以及配置
        if isinstance(self.model, TorchBinaryClassifier):
            model_payload = {
                'type': 'torch',
                'artifact': self.model.to_serializable(),
            }
        else:
            model_payload = {
                'type': 'sklearn',
                'artifact': self.model,
            }

        model_data = {
            'model_payload': model_payload,
            'scaler': self.scaler,
            'recommended_threshold': self.recommended_threshold,
            'config': {
                'das_fs': self.config.das_fs,
                'das_bp_bands': self.config.das_bp_bands,
                'das_filter_order': self.config.das_filter_order,
                'das_filter_method': self.config.das_filter_method,
                'disable_das_bandpass': self.config.disable_das_bandpass,
                'step_min_interval': self.config.step_min_interval,
                'feature_win_ms': self.config.feature_win_ms,
                'feature_step_ms': self.config.feature_step_ms,
                'model_type': self.config.model_type,
                'n_estimators': self.config.n_estimators,
                'device': self.config.device,
                'torch_epochs': self.config.torch_epochs,
                'torch_batch_size': self.config.torch_batch_size,
                'torch_lr': self.config.torch_lr,
                'torch_weight_decay': self.config.torch_weight_decay,
                'torch_hidden_dim': self.config.torch_hidden_dim,
                'torch_dropout': self.config.torch_dropout,
                'torch_patience': self.config.torch_patience,
                'torch_val_interval': self.config.torch_val_interval,
                'torch_amp': self.config.torch_amp,
                'prob_smooth_points': self.config.prob_smooth_points,
                'cnn_window_s': self.config.cnn_window_s,
                'cnn_predict_chunk': self.config.cnn_predict_chunk,
            },
            'version': '2.0'
        }
        
        joblib.dump(model_data, model_path)
        print(f"[Model] Saved to: {model_path}")
    
    @classmethod
    def load_model(cls, model_path, config=None):
        """
        从文件加载已训练的模型
        
        Args:
            model_path: 模型文件路径 (.joblib)
            config: 可选的配置对象，如果不提供则使用模型中保存的配置
        
        Returns:
            WeaklySupervisedDetector: 加载好模型的检测器实例
        """
        print(f"[Model] Loading from: {model_path}")
        model_data = joblib.load(model_path)
        
        # 创建配置
        if config is None:
            config = Config()
        
        # 恢复保存的配置参数
        saved_config = model_data.get('config', {})
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 创建检测器实例
        detector = cls(config)
        model_payload = model_data.get('model_payload')
        if model_payload is None:
            # backward compatible with old format
            detector.model = model_data['model']
        elif model_payload.get('type') == 'torch':
            detector.model = TorchBinaryClassifier.from_serializable(model_payload['artifact'], config)
        else:
            detector.model = model_payload['artifact']
        detector.scaler = model_data['scaler']
        detector.recommended_threshold = model_data.get('recommended_threshold', None)
        
        print(f"[Model] Loaded successfully (version: {model_data.get('version', 'unknown')})")
        print(f"[Model] Config: DAS {saved_config.get('das_bp_bands', 'N/A')}, "
              f"model_type={saved_config.get('model_type', 'N/A')}")
        if detector.recommended_threshold is not None:
            print(f"[Model] Recommended threshold from training: {detector.recommended_threshold:.2f}")
        
        return detector


# ============================================================================
# [6] 自训练迭代优化
# ============================================================================
class SelfTrainingIterator:
    """自训练迭代优化器"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def run_iteration(self, das_bands, current_step_times, round_num):
        """运行一轮自训练迭代"""
        print(f"\n[Self-Train] Round {round_num}")
        
        # 创建新检测器
        detector = WeaklySupervisedDetector(self.config)
        
        # 准备训练数据
        X, y = detector.prepare_training_data(das_bands, current_step_times)
        
        # 训练
        detector.train(X, y)
        
        # 预测
        grid_times, probs = detector.predict_on_grid(das_bands, time_step=0.03)
        
        # 检测脚步
        new_step_times, new_step_probs = detector.detect_steps_from_probs(
            grid_times, probs, 
            threshold=self.config.confidence_threshold,
            min_interval=self.config.step_min_interval
        )
        
        # 合并新检测到的高置信脚步
        combined = self._merge_step_times(current_step_times, new_step_times, 
                                          new_step_probs, min_dist=0.2)
        
        print(f"[Self-Train] Previous steps: {len(current_step_times)}, "
              f"New detections: {len(new_step_times)}, Combined: {len(combined)}")
        
        return combined, detector, grid_times, probs
    
    def _merge_step_times(self, old_times, new_times, new_probs, min_dist=0.2):
        """合并新旧脚步时间"""
        if len(new_times) == 0:
            return old_times
        
        # 筛选高置信新检测
        high_conf_mask = new_probs >= self.config.confidence_threshold
        high_conf_times = new_times[high_conf_mask]
        
        # 只保留远离已有脚步的新检测
        new_to_add = []
        for t in high_conf_times:
            if len(old_times) == 0 or np.min(np.abs(t - old_times)) > min_dist:
                new_to_add.append(t)
        
        combined = np.sort(np.concatenate([old_times, new_to_add]))
        return combined
