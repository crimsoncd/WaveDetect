from __future__ import annotations

import cv2
import numpy as np

from .config import DetectorConfig


def preprocess_frame(frame_bgr: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Enhance strip visibility under mist/highlight noise."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
    )
    v = clahe.apply(v)

    gain = np.clip(v.astype(np.float32) * cfg.brightness_gain, 0, 255).astype(np.uint8)
    hsv_enhanced = cv2.merge([h, s, gain])
    bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    if cfg.gaussian_ksize > 1:
        k = cfg.gaussian_ksize | 1
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray
