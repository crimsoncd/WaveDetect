from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig


@dataclass
class StripContour:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    mask: np.ndarray



def make_roi(frame: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x0 = int(cfg.roi_x0 * w)
    x1 = int(cfg.roi_x1 * w)
    y0 = int(cfg.roi_y0 * h)
    y1 = int(cfg.roi_y1 * h)
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)



def segment_strip(roi_bgr: np.ndarray, gray_roi: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    thr = np.percentile(gray_roi, cfg.thr_percentile)
    _, gray_mask = cv2.threshold(gray_roi, thr, 255, cv2.THRESH_BINARY)

    g = roi_bgr[:, :, 1].astype(np.int16)
    r = roi_bgr[:, :, 2].astype(np.int16)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    hot_mask = (
        ((r >= cfg.hot_red_min) & ((r - g) >= cfg.hot_rg_diff_min))
        | (v >= cfg.hot_value_min)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(gray_mask, hot_mask)
    k = max(3, cfg.morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask



def find_main_contour(mask: np.ndarray, cfg: DetectorConfig) -> StripContour | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = mask.shape
    roi_area = float(h * w)
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_contour_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1)
        width_ratio = cw / max(w, 1)
        area_ratio = area / max(roi_area, 1.0)
        if area_ratio < cfg.min_strip_area_ratio:
            continue
        if aspect < cfg.min_strip_aspect_ratio:
            continue
        if width_ratio < cfg.min_strip_width_ratio:
            continue
        filtered.append((c, area))

    if not filtered:
        return None

    best = max(filtered, key=lambda item: item[1])[0]
    x, y, w, h = cv2.boundingRect(best)
    local_mask = np.zeros_like(mask)
    cv2.drawContours(local_mask, [best], -1, 255, thickness=-1)
    return StripContour(contour=best, bbox=(x, y, w, h), mask=local_mask)



def extract_top_bottom_profile(contour_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = contour_mask.shape
    xs = np.arange(w)
    top = np.full(w, np.nan, dtype=np.float32)
    bottom = np.full(w, np.nan, dtype=np.float32)

    for x in xs:
        ys = np.flatnonzero(contour_mask[:, x] > 0)
        if ys.size == 0:
            continue
        top[x] = ys.min()
        bottom[x] = ys.max()

    return xs, top, bottom



def fill_nan_1d(arr: np.ndarray) -> np.ndarray:
    idx = np.arange(arr.size)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return arr
    return np.interp(idx, idx[mask], arr[mask])
