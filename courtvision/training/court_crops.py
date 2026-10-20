"""Make zoomed, half-court training views from full-court keypoint labels.

Every image in the reloc2-den7l court dataset shows the whole court centred in
frame, while NBA broadcasts mostly frame one half and zoom in on the ball (the
model saw 5.2 of 18 keypoints per frame on the lebron_block clip). Cropping a
full-court image to a window of the broadcast's aspect ratio and scaling it
back up gives the same kind of view: fewer landmarks, larger lines, keypoints
near the image edge. Keypoints outside the crop become invisible (0 0 0), and
crops with fewer than `min_visible` landmarks are skipped, because the
pipeline needs four to fit a homography.

Labels are YOLO pose rows: "cls cx cy w h x1 y1 v1 ... x18 y18 v18", normalised.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .datasets import _images, _label_for


def crop_pose_label(row: list[float], window: tuple[float, float, float, float],
                    min_visible: int = 4) -> Optional[list[float]]:
    """Re-express one pose label inside a normalised crop window (x0, y0, x1, y1).

    Returns the new row, or None when fewer than `min_visible` keypoints stay in the crop.
    """
    x0, y0, x1, y1 = window
    cw, ch = x1 - x0, y1 - y0
    cls, cx, cy, w, h = row[:5]
    kpts = np.asarray(row[5:], float).reshape(-1, 3)

    out = np.zeros_like(kpts)
    inside = (kpts[:, 2] > 0) & (kpts[:, 0] >= x0) & (kpts[:, 0] <= x1) & (kpts[:, 1] >= y0) & (kpts[:, 1] <= y1)
    if inside.sum() < min_visible:
        return None
    out[inside, 0] = (kpts[inside, 0] - x0) / cw
    out[inside, 1] = (kpts[inside, 1] - y0) / ch
    out[inside, 2] = kpts[inside, 2]

    # Court box clipped to the crop.
    bx0, by0 = max(cx - w / 2, x0), max(cy - h / 2, y0)
    bx1, by1 = min(cx + w / 2, x1), min(cy + h / 2, y1)
    box = [((bx0 + bx1) / 2 - x0) / cw, ((by0 + by1) / 2 - y0) / ch, (bx1 - bx0) / cw, (by1 - by0) / ch]
    return [cls] + box + out.ravel().tolist()


def random_window(rng: random.Random, zoom_range: tuple[float, float] = (1.6, 3.0),
                  centre_x: Optional[float] = None) -> tuple[float, float, float, float]:
    """A crop window keeping the image's aspect ratio, zoomed in by a factor from zoom_range.

    The crop is biased toward one end of the court (like a half-court broadcast view) unless
    centre_x is given. Vertically it stays near the middle, where the floor is in these images.
    """
    zoom = rng.uniform(*zoom_range)
    size = 1.0 / zoom
    cx = centre_x if centre_x is not None else rng.choice([rng.uniform(0.15, 0.4), rng.uniform(0.6, 0.85)])
    cy = rng.uniform(0.4, 0.65)
    x0 = min(max(cx - size / 2, 0.0), 1.0 - size)
    y0 = min(max(cy - size / 2, 0.0), 1.0 - size)
    return x0, y0, x0 + size, y0 + size


def make_zoomed_crops(root: str, split: str = "train", per_image: int = 2, zoom_range=(1.6, 3.0),
                      min_visible: int = 4, seed: int = 0, tries: int = 10) -> int:
    """Add up to `per_image` zoomed crops of every image in root/split, written beside the originals.

    Crops are saved at the original resolution as <stem>_zoomK.jpg. Returns the number written.
    """
    rng = random.Random(seed)
    img_dir = Path(root) / split / "images"
    written = 0
    for img_path in _images(img_dir):
        if "_zoom" in img_path.stem:
            continue
        lbl = _label_for(img_path)
        if not lbl.exists():
            continue
        rows = [[float(v) for v in line.split()] for line in open(lbl) if line.split()]
        if not rows:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        made = 0
        for _ in range(per_image * tries):
            if made == per_image:
                break
            win = random_window(rng, zoom_range)
            new_rows = [r for r in (crop_pose_label(row, win, min_visible) for row in rows) if r is not None]
            if not new_rows:
                continue
            x0, y0, x1, y1 = int(win[0] * W), int(win[1] * H), int(win[2] * W), int(win[3] * H)
            crop = cv2.resize(img[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_LINEAR)
            stem = f"{img_path.stem}_zoom{made}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop)
            (lbl.parent / f"{stem}.txt").write_text(
                "\n".join(" ".join(f"{v:.6g}" for v in r) for r in new_rows) + "\n")
            made += 1
            written += 1
    return written
