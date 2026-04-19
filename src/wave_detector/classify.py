from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig


@dataclass
class WaveResult:
    ds_level: int
    ws_level: int
    bilateral_level: int
    center_level: int
    thickness_mm: float
    width_mm: float
    rms_left_mm: float
    rms_right_mm: float
    center_rms_mm: float


@dataclass
class TemporalState:
    ng_streak: int = 0



def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")



def level_from_error(err_mm: float, cfg: DetectorConfig) -> int:
    if err_mm < cfg.rms_mm_thresholds["low"]:
        return 0
    if err_mm < cfg.rms_mm_thresholds["mid"]:
        return 1
    if err_mm < cfg.rms_mm_thresholds["high"]:
        return 2
    return 3



def fit_line_rms(y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.arange(len(y), dtype=np.float32)
    coeff = np.polyfit(x, y.astype(np.float32), deg=1)
    y_fit = coeff[0] * x + coeff[1]
    rms = float(np.sqrt(np.mean((y - y_fit) ** 2)))
    return y_fit, rms



def classify_wave(top_profile_px: np.ndarray, thickness_px: float, width_px: float, cfg: DetectorConfig) -> WaveResult:
    top_smooth = smooth_1d(top_profile_px, cfg.smooth_window)
    n = len(top_smooth)
    edge_len = max(8, int(n * cfg.edge_ratio))
    center_start = int(n * 0.35)
    center_end = int(n * 0.65)

    left = top_smooth[:edge_len]
    right = top_smooth[-edge_len:]
    center = top_smooth[center_start:center_end]

    _, left_rms_px = fit_line_rms(left)
    _, right_rms_px = fit_line_rms(right)
    _, center_rms_px = fit_line_rms(center)

    left_rms_mm = left_rms_px * cfg.mm_per_px_y
    right_rms_mm = right_rms_px * cfg.mm_per_px_y
    center_rms_mm = center_rms_px * cfg.mm_per_px_y

    left_level = level_from_error(left_rms_mm, cfg)
    right_level = level_from_error(right_rms_mm, cfg)
    if cfg.ds_on_left:
        ds_level, ws_level = left_level, right_level
    else:
        ds_level, ws_level = right_level, left_level

    center_level = level_from_error(center_rms_mm, cfg)
    bilateral_level = 0
    if ds_level > 0 and ws_level > 0:
        bilateral_level = min(3, max(ds_level, ws_level))
    elif cfg.bilateral_from_center and center_level >= cfg.bilateral_center_min_level:
        # In some camera angles one edge can be partially occluded; center undulation
        # is used as a fallback cue for bilateral-wave grading.
        bilateral_level = center_level

    thickness_mm = thickness_px * cfg.mm_per_px_y
    width_mm = width_px * cfg.mm_per_px_x

    return WaveResult(
        ds_level=ds_level,
        ws_level=ws_level,
        bilateral_level=bilateral_level,
        center_level=center_level,
        thickness_mm=thickness_mm,
        width_mm=width_mm,
        rms_left_mm=left_rms_mm,
        rms_right_mm=right_rms_mm,
        center_rms_mm=center_rms_mm,
    )


def stage_aware_filter(
    wave: WaveResult,
    width_px: float,
    roi_width_px: float,
    cfg: DetectorConfig,
    state: TemporalState,
) -> tuple[WaveResult, str]:
    phase = "transition"
    if roi_width_px > 1 and width_px / roi_width_px >= cfg.middle_phase_width_ratio:
        phase = "middle"

    ng_now = any(
        v > 0 for v in [wave.ds_level, wave.ws_level, wave.bilateral_level, wave.center_level]
    )
    if ng_now:
        state.ng_streak += 1
    else:
        state.ng_streak = 0

    if phase == "middle":
        # During steady forward motion, suppress brief low-level spark burrs.
        if state.ng_streak < cfg.middle_required_consecutive:
            wave.ds_level = 0
            wave.ws_level = 0
            wave.bilateral_level = 0
            wave.center_level = 0
        else:
            wave.ds_level = wave.ds_level if wave.ds_level >= cfg.middle_min_level else 0
            wave.ws_level = wave.ws_level if wave.ws_level >= cfg.middle_min_level else 0
            wave.bilateral_level = wave.bilateral_level if wave.bilateral_level >= cfg.middle_min_level else 0
            wave.center_level = wave.center_level if wave.center_level >= cfg.middle_min_level else 0

    return wave, phase
