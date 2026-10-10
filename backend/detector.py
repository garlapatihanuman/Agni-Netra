from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    MODEL_PATH,
)

_model: YOLO | None = None


def load_model(model_path: str = MODEL_PATH, device: str = DEVICE) -> YOLO:
    """Load the YOLOv8 weights file."""
    global _model

    if not model_path:
        raise ValueError("MODEL_PATH is empty. Set it in config.py or the MODEL_PATH environment variable.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found: '{model_path}'. Place your trained weights here or update MODEL_PATH."
        )

    if _model is None or getattr(_model, "ckpt_path", None) != model_path:
        _model = YOLO(model_path)

    return _model


def get_model_names(model: YOLO) -> Dict[int, str]:
    """Retrieve model class names."""
    names = getattr(model, "names", {}) or {}
    if not isinstance(names, dict):
        return {int(k): str(v) for k, v in enumerate(names)}
    return {int(key): str(value) for key, value in names.items()}


def draw_tactical_bbox(
    annotated: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    confidence: float,
    is_live_flame: bool = False,
) -> None:
    """Renders a futuristic tactical bounding box with corner brackets, tint, and high-contrast badge."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # 1. Semi-transparent colored glow inside box
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.12 if not is_live_flame else 0.18
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    # 2. Main rectangle outline
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 3. High-tech corner brackets (tactical HUD style)
    corner_len = max(6, min(int(min(w, h) * 0.25), 24))
    bracket_thickness = 3
    # Top-Left
    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, bracket_thickness, cv2.LINE_AA)
    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, bracket_thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, bracket_thickness, cv2.LINE_AA)
    cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, bracket_thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, bracket_thickness, cv2.LINE_AA)
    cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, bracket_thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, bracket_thickness, cv2.LINE_AA)
    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, bracket_thickness, cv2.LINE_AA)

    # 4. Center crosshairs on detected live flame
    if is_live_flame and w > 18 and h > 18:
        cx, cy = x1 + w // 2, y1 + h // 2
        ch_len = 5
        cv2.line(annotated, (cx - ch_len, cy), (cx + ch_len, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - ch_len), (cx, cy + ch_len), (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Label badge background & text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    badge_h = text_h + 12
    badge_w = text_w + 14

    by1 = max(0, y1 - badge_h)
    by2 = by1 + badge_h
    bx1 = max(0, x1)
    bx2 = min(annotated.shape[1], bx1 + badge_w)

    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (18, 24, 38), -1)
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

    cv2.putText(
        annotated,
        label,
        (bx1 + 7, by2 - 6),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def detect_objects(
    frame: np.ndarray,
    model: YOLO,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    image_size: int = IMAGE_SIZE,
    max_detections: int = MAX_DETECTIONS,
    device: str = DEVICE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs high-precision inference using trained YOLOv8 weights (best.pt).
    
    Exclusively utilizes the neural network to identify real fire, live flame, and smoke,
    avoiding false positives on ordinary orange or red colored objects.
    """
    start_time = time.perf_counter()
    annotated = frame.copy()
    h_img, w_img = frame.shape[:2]
    total_area = max(1, w_img * h_img)

    # Run YOLOv8 prediction with user-configured confidence and IoU thresholds
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        max_det=max_detections,
        device=device,
        verbose=False,
    )

    model_names = get_model_names(model)
    detections: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        prediction = results[0]
        boxes = prediction.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for bbox, score, cls_id in zip(xyxy, confidences, class_ids):
                raw_name = model_names.get(int(cls_id), f"class_{int(cls_id)}").lower()

                # Discard non-hazard background class 'other'
                if raw_name == "other":
                    continue

                score_float = float(score)
                if score_float < conf_threshold:
                    continue

                x1, y1, x2, y2 = [int(v) for v in bbox]
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                box_area = bw * bh
                area_ratio = box_area / total_area

                # Class differentiation:
                # Class 0: 'fire' -> Distinguish compact 'Live Flame' (lighter, candle, match) vs larger 'Fire'
                # Class 2: 'smoke' -> 'Smoke'
                if raw_name == "fire":
                    if area_ratio <= 0.08 and bw <= w_img * 0.35:
                        final_class = "Live Flame"
                    else:
                        final_class = "Fire"
                elif raw_name == "smoke":
                    final_class = "Smoke"
                else:
                    final_class = raw_name.capitalize()

                detections.append(
                    {
                        "class_name": final_class,
                        "confidence": score_float,
                        "bbox": [x1, y1, x2, y2],
                        "source": "yolo_best_pt",
                    }
                )

    # Sort detections by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = detections[:max_detections]

    # Render bounding boxes and compute telemetry
    class_counts: Counter[str] = Counter()
    highest_conf = 0.0

    for det in detections:
        cname = det["class_name"]
        score = det["confidence"]
        bbox = det["bbox"]
        class_counts[cname] += 1
        highest_conf = max(highest_conf, score)

        if cname == "Live Flame":
            color = (0, 165, 255)  # Flame Neon Orange / Gold (BGR)
            label = f"LIVE FLAME {score * 100:.1f}%"
            is_live_flame = True
        elif cname == "Fire":
            color = (45, 25, 255)  # Crimson Laser Red (BGR)
            label = f"FIRE {score * 100:.1f}%"
            is_live_flame = False
        elif cname == "Smoke":
            color = (255, 225, 0)  # Cool Cyan (BGR)
            label = f"SMOKE {score * 100:.1f}%"
            is_live_flame = False
        else:
            color = (0, 230, 118)
            label = f"{cname.upper()} {score * 100:.1f}%"
            is_live_flame = False

        draw_tactical_bbox(
            annotated=annotated,
            bbox=bbox,
            label=label,
            color=color,
            confidence=score,
            is_live_flame=is_live_flame,
        )

    inference_time = (time.perf_counter() - start_time) * 1000.0

    # Determine overall hazard level
    total_det = len(detections)
    if total_det == 0:
        hazard_level = "CLEAR"
    elif "Fire" in class_counts or ("Live Flame" in class_counts and highest_conf >= 0.50):
        hazard_level = "CRITICAL"
    elif "Live Flame" in class_counts or "Smoke" in class_counts:
        hazard_level = "WARNING"
    else:
        hazard_level = "ADVISORY"

    stats = {
        "total_detections": total_det,
        "class_counts": dict(sorted(class_counts.items())),
        "detections": detections,
        "hazard_level": hazard_level,
        "highest_confidence": float(highest_conf),
        "inference_time_ms": float(round(inference_time, 1)),
        "flame_count": class_counts.get("Live Flame", 0),
        "fire_count": class_counts.get("Fire", 0),
        "smoke_count": class_counts.get("Smoke", 0),
    }

    return annotated, stats
