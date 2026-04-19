from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class DetectorConfig:
    # ROI as ratio of full frame.
    roi_x0: float = 0.08
    roi_x1: float = 0.92
    roi_y0: float = 0.20
    roi_y1: float = 0.90

    # Preprocess params.
    clahe_clip_limit: float = 2.5
    clahe_grid_size: int = 8
    gaussian_ksize: int = 5
    brightness_gain: float = 1.15

    # Segmentation params.
    thr_percentile: float = 82.0
    min_contour_area: int = 2500
    morph_kernel: int = 7
    min_strip_area_ratio: float = 0.015
    min_strip_aspect_ratio: float = 0.9
    min_strip_width_ratio: float = 0.35
    hot_red_min: int = 180
    hot_rg_diff_min: int = 25
    hot_value_min: int = 220

    # Thickness sampling (relative to detected strip width).
    thickness_left_ratio: float = 0.15
    thickness_right_ratio: float = 0.85

    # Wave analysis.
    smooth_window: int = 17
    edge_ratio: float = 0.2
    rms_mm_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "low": 4.0,
            "mid": 8.0,
            "high": 14.0,
        }
    )
    slope_deg_threshold: float = 7.5
    bilateral_margin_mm: float = 3.5
    bilateral_from_center: bool = True
    bilateral_center_min_level: int = 1
    # Stage-aware tolerance:
    # middle stage (strip fully in view) suppresses short burr-like spikes.
    middle_phase_width_ratio: float = 0.92
    middle_required_consecutive: int = 3
    middle_min_level: int = 2

    # Pixel to mm conversion.
    mm_per_px_x: float = 2.0
    mm_per_px_y: float = 0.5
    ds_on_left: bool = True

    # Post process and runtime.
    temporal_alpha: float = 0.60
    frame_step: int = 1



def load_config(path: str | Path | None) -> DetectorConfig:
    if path is None:
        return DetectorConfig()

    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = DetectorConfig()
    for key, value in data.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg
