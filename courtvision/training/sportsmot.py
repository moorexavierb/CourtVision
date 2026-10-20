"""Use SportsMOT basketball clips as extra detector training data.

SportsMOT (MCG-NJU, CC BY-NC 4.0, gated on Hugging Face) has broadcast
basketball clips with every on-court player boxed in every frame, including
the crowded, overlapping and blurred moments where our detector loses players.
It labels only players: no ball, referees or hoop. Training on it as is would
teach the detector that the ball and referees are background, so the missing
classes are filled from our own detector's confident predictions (pseudo-
labels), and frames where the detector is unsure about one of those classes
are skipped rather than half-labelled.

Expected layout (MOT17 style, after extracting the download):
    <root>/<split>/<sequence>/img1/000001.jpg
    <root>/<split>/<sequence>/gt/gt.txt   rows: frame,id,x,y,w,h,conf,class,vis
and optionally <root>/splits_txt/basketball.txt listing the basketball sequences.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def basketball_sequences(root: str, split: str) -> list[Path]:
    """Sequences of a split, filtered to basketball when splits_txt/basketball.txt exists."""
    root_p = Path(root)
    seqs = sorted(p for p in (root_p / split).glob("*") if (p / "img1").is_dir())
    listing = root_p / "splits_txt" / "basketball.txt"
    if listing.exists():
        keep = {l.strip() for l in open(listing) if l.strip()}
        seqs = [s for s in seqs if s.name in keep]
    return seqs


def read_gt(seq: Path) -> dict[int, np.ndarray]:
    """{frame number: (N, 4) player boxes as x0, y0, x1, y1 in pixels}."""
    boxes: dict[int, list] = {}
    gt = seq / "gt" / "gt.txt"
    if not gt.exists():
        return {}
    for line in open(gt):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        f, x, y, w, h = int(float(parts[0])), *map(float, parts[2:6])
        if len(parts) > 6 and float(parts[6]) == 0:  # MOT "ignore" rows
            continue
        boxes.setdefault(f, []).append((x, y, x + w, y + h))
    return {f: np.array(b, float) for f, b in boxes.items()}


def merge_gt_and_pseudo(gt_players: np.ndarray, pred_xyxy: np.ndarray, pred_cls: np.ndarray,
                        pred_conf: np.ndarray, player_cls: int, fill_classes: list[int],
                        keep_conf: float = 0.5, low_conf: float = 0.15
                        ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Ground-truth players plus confident detector boxes for the classes SportsMOT lacks.

    Detector player boxes are discarded (the ground truth covers players). For each class in
    fill_classes, boxes at or above keep_conf are kept. If any box of those classes falls between
    low_conf and keep_conf the frame is ambiguous (a ball the model half sees) and None is returned,
    so the frame is skipped instead of teaching that the object is background.
    Returns (xyxy, cls) or None.
    """
    pred_xyxy = np.asarray(pred_xyxy, float).reshape(-1, 4)
    pred_cls = np.asarray(pred_cls, int)
    pred_conf = np.asarray(pred_conf, float)
    fill = np.isin(pred_cls, fill_classes)
    if (fill & (pred_conf >= low_conf) & (pred_conf < keep_conf)).any():
        return None
    keep = fill & (pred_conf >= keep_conf)
    gt_players = np.asarray(gt_players, float).reshape(-1, 4)
    xyxy = np.vstack([gt_players, pred_xyxy[keep]])
    cls = np.concatenate([np.full(len(gt_players), player_cls, int), pred_cls[keep]])
    return xyxy, cls
