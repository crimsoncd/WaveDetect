from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .classify import TemporalState, WaveResult, classify_wave, stage_aware_filter
from .config import DetectorConfig
from .detect import (
    extract_top_bottom_profile,
    fill_nan_1d,
    find_main_contour,
    make_roi,
    segment_strip,
)
from .preprocess import preprocess_frame


class WaveDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._last_result: WaveResult | None = None
        self._state = TemporalState()

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[WaveResult | None, dict]:
        roi_bgr, (x0, y0, _, _) = make_roi(frame_bgr, self.cfg)
        gray = preprocess_frame(roi_bgr, self.cfg)
        mask = segment_strip(roi_bgr, gray, self.cfg)

        contour = find_main_contour(mask, self.cfg)
        if contour is None:
            return None, {
                "roi": (x0, y0),
                "mask": mask,
                "bbox": None,
                "top_profile": None,
            }

        xs, top, bottom = extract_top_bottom_profile(contour.mask)
        top = fill_nan_1d(top)
        bottom = fill_nan_1d(bottom)
        valid = (~np.isnan(top)) & (~np.isnan(bottom))
        if valid.sum() < 30:
            return None, {
                "roi": (x0, y0),
                "mask": mask,
                "bbox": contour.bbox,
                "top_profile": None,
            }

        x_valid = xs[valid]
        top_valid = top[valid]
        bottom_valid = bottom[valid]

        width_px = float(x_valid.max() - x_valid.min())
        left_x = int(x_valid.min() + self.cfg.thickness_left_ratio * width_px)
        right_x = int(x_valid.min() + self.cfg.thickness_right_ratio * width_px)
        sample_mask = (x_valid >= left_x) & (x_valid <= right_x)
        if sample_mask.sum() > 5:
            thickness_px = float(np.median(bottom_valid[sample_mask] - top_valid[sample_mask]))
        else:
            thickness_px = float(np.median(bottom_valid - top_valid))

        wave = classify_wave(top_valid, thickness_px, width_px, self.cfg)
        wave, phase = stage_aware_filter(
            wave=wave,
            width_px=width_px,
            roi_width_px=float(roi_bgr.shape[1]),
            cfg=self.cfg,
            state=self._state,
        )
        if self._last_result is not None:
            a = self.cfg.temporal_alpha
            wave.thickness_mm = a * self._last_result.thickness_mm + (1 - a) * wave.thickness_mm
            wave.width_mm = a * self._last_result.width_mm + (1 - a) * wave.width_mm

        self._last_result = wave
        return wave, {
            "roi": (x0, y0),
            "mask": mask,
            "bbox": contour.bbox,
            "top_profile": np.stack([x_valid, top_valid], axis=1),
            "phase": phase,
        }


def annotate_frame(frame: np.ndarray, aux: dict, wave: WaveResult | None) -> np.ndarray:
    out = frame.copy()
    x0, y0 = aux["roi"]
    if aux.get("bbox") is not None:
        x, y, w, h = aux["bbox"]
        cv2.rectangle(out, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (40, 220, 255), 2)

    profile = aux.get("top_profile")
    if profile is not None:
        for x, y in profile[::3]:
            cv2.circle(out, (x0 + int(x), y0 + int(y)), 1, (0, 255, 0), -1)

    if wave is not None:
        phase = aux.get("phase", "transition")
        lines = [
            f"Phase:     {phase}",
            f"Thickness: {wave.thickness_mm:6.2f} mm",
            f"Width:     {wave.width_mm:6.2f} mm",
            f"DS/WS/Bi/C: {wave.ds_level}/{wave.ws_level}/{wave.bilateral_level}/{wave.center_level}",
            f"RMS L/R/C: {wave.rms_left_mm:.2f}/{wave.rms_right_mm:.2f}/{wave.center_rms_mm:.2f} mm",
        ]
        y = 28
        for text in lines:
            cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            y += 26

    return out


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    cfg: DetectorConfig,
    max_frames: int | None = None,
    save_video: bool = True,
    save_ng_images: bool = True,
) -> dict:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ng_frames").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"), fourcc, fps, (width, height))

    detector = WaveDetector(cfg)

    rows = []
    timings = []
    processed = 0
    ng_count = 0

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % cfg.frame_step != 0:
            continue

        if max_frames is not None and processed >= max_frames:
            break

        t0 = time.perf_counter()
        wave, aux = detector.process_frame(frame)
        dt_ms = (time.perf_counter() - t0) * 1000
        timings.append(dt_ms)

        annotated = annotate_frame(frame, aux, wave)
        if writer is not None:
            writer.write(annotated)

        timestamp = frame_idx / fps
        row = {
            "frame": frame_idx,
            "time_s": round(timestamp, 3),
            "latency_ms": round(dt_ms, 3),
            "strip_present": False,
            "phase": None,
            "ds_level": None,
            "ws_level": None,
            "bilateral_level": None,
            "center_level": None,
            "thickness_mm": None,
            "width_mm": None,
            "rms_left_mm": None,
            "rms_right_mm": None,
            "center_rms_mm": None,
        }

        if wave is not None:
            row["strip_present"] = True
            row["phase"] = aux.get("phase")
            row.update(asdict(wave))
            is_ng = any(
                value > 0
                for value in [wave.ds_level, wave.ws_level, wave.bilateral_level, wave.center_level]
            )
            if is_ng and save_ng_images:
                cv2.imwrite(str(output_dir / "ng_frames" / f"{frame_idx:06d}_result.jpg"), annotated)
                ng_count += 1

        rows.append(row)
        processed += 1

    cap.release()
    if writer is not None:
        writer.release()

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["frame", "time_s", "latency_ms"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "video": str(video_path),
        "frames_total": total,
        "frames_processed": processed,
        "fps": fps,
        "avg_latency_ms": float(np.mean(timings)) if timings else None,
        "p95_latency_ms": float(np.percentile(timings, 95)) if timings else None,
        "max_latency_ms": float(np.max(timings)) if timings else None,
        "ng_frames": ng_count,
        "output_csv": str(csv_path),
        "output_video": str(output_dir / "annotated.mp4") if save_video else None,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
