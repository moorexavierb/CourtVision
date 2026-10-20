"""Pick frames from your own broadcast clips that are worth labelling.

Validation mAP on the Roboflow data is already high (0.94 detector, 0.995
court), yet on a real NBA clip only 5.9 of 10 players were found per frame and
5.2 of 18 court keypoints. The fastest way to close that gap is to label the
frames the current models get wrong. Each frame gets a score from the models'
own outputs: missing players, no ball or several balls, low-confidence boxes
and too few court keypoints. The highest scoring frames, spaced apart in time
so near-duplicates are not labelled twice, are exported with the models'
predictions as pre-labels, so labelling is mostly correcting boxes in Roboflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrameStats:
    frame: int
    players: int = 0                 # player boxes at or above the keep confidence
    balls: int = 0                   # ball boxes at or above the keep confidence
    uncertain: int = 0               # boxes between the low and keep confidence
    keypoints: Optional[int] = None  # court keypoints above their confidence, if a court model ran
    extra: dict = field(default_factory=dict)


def frame_score(s: FrameStats, expected_players: int = 10, min_keypoints: int = 6) -> float:
    """Higher means the models struggled more on this frame."""
    score = max(0, expected_players - s.players) / expected_players       # missed players
    score += 1.0 if s.balls == 0 else 0.5 * (s.balls - 1)                 # no ball, or phantom balls
    score += 0.1 * s.uncertain                                            # hesitant boxes
    if s.keypoints is not None and s.keypoints < min_keypoints:
        score += (min_keypoints - s.keypoints) / min_keypoints            # court mapping at risk
    return score


def pick_frames(stats: list[FrameStats], k: int, min_gap: int = 15, **score_kw) -> list[int]:
    """The k highest-scoring frames, at least min_gap frames apart, in time order."""
    order = sorted(stats, key=lambda s: frame_score(s, **score_kw), reverse=True)
    chosen: list[int] = []
    for s in order:
        if len(chosen) == k:
            break
        if all(abs(s.frame - c) >= min_gap for c in chosen):
            chosen.append(s.frame)
    return sorted(chosen)


def yolo_rows(xyxy: np.ndarray, cls: np.ndarray, size: tuple[int, int]) -> list[str]:
    """Boxes in pixels to YOLO "cls cx cy w h" rows, normalised by size (width, height)."""
    W, H = size
    rows = []
    for (x0, y0, x1, y1), c in zip(np.asarray(xyxy, float), np.asarray(cls, int)):
        rows.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return rows


def pose_row(kpts_xy: np.ndarray, kpts_conf: np.ndarray, size: tuple[int, int], conf: float = 0.5) -> Optional[str]:
    """One court keypoint prediction to a YOLO pose row; low-confidence points become invisible."""
    W, H = size
    vis = np.asarray(kpts_conf) >= conf
    if not vis.any():
        return None
    pts = np.asarray(kpts_xy, float)
    vx, vy = pts[vis, 0], pts[vis, 1]
    x0, x1, y0, y1 = vx.min(), vx.max(), vy.min(), vy.max()
    vals = []
    for (x, y), v in zip(pts, vis):
        vals += [x / W, y / H, 2] if v else [0, 0, 0]
    box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, max(x1 - x0, 1) / W, max(y1 - y0, 1) / H]
    return " ".join(["0"] + [f"{v:.6f}" for v in box] + [f"{v:.6g}" for v in vals])
