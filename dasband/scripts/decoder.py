# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from .config import DASBandConfig


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weighted_centroid(mask: np.ndarray, threshold: float = 0.0):
    mask = np.asarray(mask, dtype=np.float64)
    if threshold > 0:
        mask = np.where(mask >= threshold, mask, 0.0)
    channels = np.arange(mask.shape[1], dtype=np.float64)[None, :]
    numer = np.sum(mask * channels, axis=1)
    denom = np.sum(mask, axis=1) + 1e-8
    return (numer / denom).astype(np.float32)


def extract_path_dp(mask: np.ndarray, config: DASBandConfig):
    prob = np.clip(np.asarray(mask, dtype=np.float64), 1e-6, 1.0)
    emission = np.log(prob)
    T, C = emission.shape
    max_jump = max(1, int(config.dp_max_jump_ch))
    lam = float(config.dp_jump_penalty)

    score = np.full((T, C), -np.inf, dtype=np.float64)
    prev = np.full((T, C), -1, dtype=np.int32)
    score[0] = emission[0]

    for t in range(1, T):
        for c in range(C):
            lo = max(0, c - max_jump)
            hi = min(C, c + max_jump + 1)
            prev_cands = np.arange(lo, hi)
            candidate_score = score[t - 1, prev_cands] - lam * np.abs(prev_cands - c)
            if t >= 2 and config.dp_curvature_penalty > 0:
                ref = prev[t - 1, prev_cands]
                valid_ref = ref >= 0
                curvature = np.zeros_like(candidate_score)
                curvature[valid_ref] = float(config.dp_curvature_penalty) * np.abs(c - 2 * prev_cands[valid_ref] + ref[valid_ref])
                candidate_score = candidate_score - curvature
            best_idx = int(np.argmax(candidate_score))
            score[t, c] = emission[t, c] + candidate_score[best_idx]
            prev[t, c] = int(prev_cands[best_idx])

    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = max(0, int(prev[t, path[t]]))
    return path.astype(np.float32)


def estimate_uncertainty(mask: np.ndarray, path: np.ndarray):
    prob = np.asarray(mask, dtype=np.float64)
    channels = np.arange(prob.shape[1], dtype=np.float64)[None, :]
    path = np.asarray(path, dtype=np.float64)[:, None]
    numer = np.sum(((channels - path) ** 2) * prob, axis=1)
    denom = np.sum(prob, axis=1) + 1e-8
    return np.sqrt(numer / denom).astype(np.float32)
