# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from .config import Config

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    class _NNStub:
        Module = object
    nn = _NNStub()
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

def _resolve_torch_device(device_pref: str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install torch to use cnn model_type.")
    pref = (device_pref or 'auto').lower()
    if pref == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pref == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return torch.device('cuda')
    return torch.device('cpu')


class TemporalBlock(nn.Module):
    """时间维度残差块 - 捕捉振动的时序模式"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), 
                      padding=(padding, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), 
                      padding=(padding, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.conv(x))


class SpatialBlock(nn.Module):
    """空间维度残差块 - 捕捉相邻通道的振动关联"""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), 
                      padding=(0, padding)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), 
                      padding=(0, padding)),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.conv(x))


class FeatureCNN(nn.Module):
    """
    改进版CNN - 直接学习振动模式而非只关注能量
    
    设计原则:
    1. 使用小卷积核(3x3)捕捉局部振动波形
    2. 分离时间和空间卷积，分别学习振动序列和通道关联
    3. 使用残差连接保留原始波形信息
    4. 减少池化操作，用stride卷积逐步降采样
    5. 多尺度时序特征提取（不同dilation）
    """
    def __init__(self, in_bands: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        c1 = max(32, hidden_dim // 4)
        c2 = max(64, hidden_dim // 2)
        c3 = max(128, hidden_dim)
        
        # 初始特征提取 - 小卷积核保留振动细节
        self.stem = nn.Sequential(
            nn.Conv2d(in_bands, c1, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
        )
        
        # 多尺度时序振动模式学习（不同dilation捕捉不同时间尺度的振动）
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(c1, kernel_size=3, dilation=1),  # 短期振动
            TemporalBlock(c1, kernel_size=3, dilation=2),  # 中期振动
            TemporalBlock(c1, kernel_size=3, dilation=4),  # 长期振动模式
        ])
        
        # 空间降采样（stride=2，比MaxPool更平滑）
        self.downsample1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
        )
        
        # 空间关联学习 - 相邻通道的振动关联
        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(c2, kernel_size=3),
            SpatialBlock(c2, kernel_size=5),  # 更大范围的通道关联
        ])
        
        # 进一步特征压缩
        self.downsample2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
        )
        
        # 最终特征聚合 - 保留时间和空间维度的统计量
        # 不使用(1,1)全局池化，而是分别计算时间和空间统计
        self.temporal_pool = nn.AdaptiveAvgPool2d((4, 1))  # 保留4个时间点
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 4))   # 保留4个通道
        
        # 分类头 - 更大的隐藏层保留信息
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 + c3 * 4, hidden_dim * 2),  # 时间+空间特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.in_bands = in_bands

    def forward(self, x):
        # 初始特征
        x = self.stem(x)
        
        # 多尺度时序振动模式
        for block in self.temporal_blocks:
            x = block(x)
        
        # 降采样
        x = self.downsample1(x)
        
        # 空间关联
        for block in self.spatial_blocks:
            x = block(x)
        
        # 进一步压缩
        x = self.downsample2(x)
        
        # 分别提取时间和空间统计特征
        t_feat = self.temporal_pool(x)  # [B, C, 4, 1]
        s_feat = self.spatial_pool(x)   # [B, C, 1, 4]
        
        # 拼接时空特征
        t_flat = t_feat.view(t_feat.size(0), -1)
        s_flat = s_feat.view(s_feat.size(0), -1)
        combined = torch.cat([t_flat, s_flat], dim=1)
        
        return self.head(combined).squeeze(1)


class TorchBinaryClassifier:
    """提供与sklearn兼容接口的PyTorch二分类包装器。"""
    def __init__(self, config: Config, input_shape=None):
        self.config = config
        self.model_kind = 'cnn'
        self.input_shape = input_shape
        self.device = _resolve_torch_device(config.device)
        self.model = None
        self.best_val_loss = None

    def _build_network(self, input_shape):
        in_bands = int(input_shape[0])
        return FeatureCNN(
            in_bands=in_bands,
            hidden_dim=self.config.torch_hidden_dim,
            dropout=self.config.torch_dropout,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.input_shape = tuple(X.shape[1:])
        self.model = self._build_network(self.input_shape).to(self.device)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.astype(np.int32)
        )

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=self.config.torch_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.torch_batch_size, shuffle=False)

        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        pos_weight = torch.tensor([max(1.0, neg / (pos + 1e-12))], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.torch_lr,
            weight_decay=self.config.torch_weight_decay,
        )
        use_amp = bool(getattr(self.config, "torch_amp", True)) and (self.device.type == "cuda")
        amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)
        val_interval = max(1, int(getattr(self.config, "torch_val_interval", 5)))

        best_state = None
        best_val = np.inf
        wait = 0
        print(
            f"[Torch] model={self.model_kind}, device={self.device}, input_shape={self.input_shape}, "
            f"amp={use_amp}, val_interval={val_interval}"
        )
        for epoch in range(1, self.config.torch_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(amp_device, enabled=use_amp):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            should_validate = (
                epoch == 1
                or epoch % val_interval == 0
                or epoch == self.config.torch_epochs
            )
            if not should_validate:
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    with torch.amp.autocast(amp_device, enabled=use_amp):
                        logits = self.model(xb)
                    val_losses.append(float(criterion(logits, yb).item()))
            mean_val = float(np.mean(val_losses)) if val_losses else np.inf
            print(f"[Torch] Epoch {epoch:03d}/{self.config.torch_epochs} val_loss={mean_val:.4f}")

            if mean_val < best_val - 1e-5:
                best_val = mean_val
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.config.torch_patience:
                    print(f"[Torch] Early stopping at epoch {epoch}, best_val={best_val:.4f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_loss = best_val
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Torch model not trained.")
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        probs_all = []
        with torch.no_grad():
            for i in range(0, len(X), self.config.torch_batch_size):
                xb = torch.from_numpy(X[i:i + self.config.torch_batch_size]).to(self.device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                probs_all.append(probs)
        p1 = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float32)
        p1 = p1.astype(np.float64)
        return np.column_stack([1.0 - p1, p1])

    def score(self, X, y):
        probs = self.predict_proba(X)[:, 1]
        pred = (probs >= 0.5).astype(np.int32)
        y = np.asarray(y).astype(np.int32)
        return float(np.mean(pred == y))

    def export_learned_pattern(self, output_png_path):
        """导出CNN首层卷积中最强滤波器对应的时空pattern。"""
        if self.model is None:
            raise ValueError("Torch model not trained.")
        first_conv = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                break
        if first_conv is None:
            raise ValueError("Unexpected model structure: no Conv2d layer found.")

        w = first_conv.weight.detach().cpu().numpy()  # [out_ch, in_bands, kh, kw]
        filt_norm = np.linalg.norm(w.reshape(w.shape[0], -1), axis=1)
        best_idx = int(np.argmax(filt_norm))
        pattern = np.mean(w[best_idx], axis=0)  # [kh, kw], 跨频带平均

        npy_path = os.path.splitext(output_png_path)[0] + ".npy"
        np.save(npy_path, pattern)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        im = ax.imshow(pattern, cmap='RdBu_r', aspect='auto')
        ax.set_title(f"Learned Pattern (conv1 #{best_idx})")
        ax.set_xlabel("Channel Kernel Axis")
        ax.set_ylabel("Time Kernel Axis")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"[Viz] Saved learned pattern: {output_png_path}")
        print(f"[Viz] Saved learned pattern array: {npy_path}")

    def to_serializable(self):
        if self.model is None or self.input_shape is None:
            raise ValueError("Torch model not trained.")
        return {
            'model_kind': self.model_kind,
            'input_shape': [int(x) for x in self.input_shape],
            'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            'best_val_loss': None if self.best_val_loss is None else float(self.best_val_loss),
        }

    @classmethod
    def from_serializable(cls, artifact: dict, config: Config):
        model_kind = artifact.get('model_kind', 'cnn')
        if model_kind != 'cnn':
            raise ValueError(f"Unsupported saved torch model_kind: {model_kind}")
        input_shape = tuple(artifact.get('input_shape', [2, 400, 174]))
        obj = cls(config=config, input_shape=input_shape)
        obj.model = obj._build_network(input_shape).to(obj.device)
        obj.model.load_state_dict(artifact['state_dict'])
        obj.model.eval()
        obj.best_val_loss = artifact.get('best_val_loss')
        return obj
