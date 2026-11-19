"""Zoomed, half-court views made from full-court keypoint labels.

The public court dataset shows the whole court centred in frame, while broadcasts mostly frame
one half and zoom toward the ball. Cropping a labelled image to a window with the same aspect
ratio and scaling it back up gives that kind of view; landmarks outside the window become
invisible, and crops with fewer than 4 landmarks (too few for a homography) are skipped.

Crops are added to a built dataset (sources.build) as new manifest rows that copy their parent's
component, so a crop is never on the other side of a split from its original. Only training images
are cropped: validation and test stay real frames. The manifest's crop_of column names each
crop's parent image (empty for originals).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .sources import write_lists


def crop_pose_row(row: list[float], window: tuple[float, float, float, float],
                  min_visible: int = 4) -> Optional[list[float]]:
    """One pose label re-expressed inside a normalised crop window (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = window
    cw, ch = x1 - x0, y1 - y0
    cls, cx, cy, w, h = row[:5]
    kp = np.asarray(row[5:], float).reshape(-1, 3)
    inside = (kp[:, 2] > 0) & (kp[:, 0] >= x0) & (kp[:, 0] <= x1) & (kp[:, 1] >= y0) & (kp[:, 1] <= y1)
    if inside.sum() < min_visible:
        return None
    out = np.zeros_like(kp)
    out[inside, 0] = (kp[inside, 0] - x0) / cw
    out[inside, 1] = (kp[inside, 1] - y0) / ch
    out[inside, 2] = kp[inside, 2]
    bx0, by0 = max(cx - w / 2, x0), max(cy - h / 2, y0)
    bx1, by1 = min(cx + w / 2, x1), min(cy + h / 2, y1)
    box = [((bx0 + bx1) / 2 - x0) / cw, ((by0 + by1) / 2 - y0) / ch, (bx1 - bx0) / cw, (by1 - by0) / ch]
    return [cls] + box + out.ravel().tolist()


def random_window(rng: random.Random, zoom=(1.6, 3.0)) -> tuple[float, float, float, float]:
    """A window 1/zoom of the image, biased toward one end of the court like a half-court view."""
    size = 1.0 / rng.uniform(*zoom)
    cx = rng.choice([rng.uniform(0.15, 0.4), rng.uniform(0.6, 0.85)])
    cy = rng.uniform(0.4, 0.65)
    x0 = min(max(cx - size / 2, 0.0), 1.0 - size)
    y0 = min(max(cy - size / 2, 0.0), 1.0 - size)
    return x0, y0, x0 + size, y0 + size


def add_zoomed_crops(root: str, sources: list[str], per_image: int = 2, zoom=(1.6, 3.0), seed: int = 0,
                     tries: int = 10) -> int:
    """Add up to per_image crops of every training image of the given sources in a built pose dataset.
    Updates manifest.csv and the lists; returns the number of crops written."""
    import yaml
    root_p = Path(root)
    m = pd.read_csv(root_p / "manifest.csv")
    if "crop_of" not in m:
        m["crop_of"] = None
    rng = random.Random(seed)
    new = []
    done = set(m.crop_of.dropna())  # parents cropped by an earlier call are left alone
    parents = m[m.source.isin(sources) & (m.split == "train") & m.crop_of.isna() & ~m.image.isin(done)]
    for r in parents.itertuples(index=False):
        img_path = Path(r.image)
        lbl = root_p / "labels" / r.source / f"{img_path.stem}.txt"
        rows = [[float(v) for v in l.split()] for l in open(lbl) if l.split()] if lbl.exists() else []
        img = cv2.imread(str(img_path)) if rows else None
        if img is None:
            continue
        H, W = img.shape[:2]
        made = 0
        for _ in range(per_image * tries):
            if made == per_image:
                break
            win = random_window(rng, zoom)
            labels = [c for c in (crop_pose_row(row, win) for row in rows) if c is not None]
            if not labels:
                continue
            x0, y0, x1, y1 = int(win[0] * W), int(win[1] * H), int(win[2] * W), int(win[3] * H)
            stem = f"{img_path.stem}_zoom{made}"
            cv2.imwrite(str(img_path.parent / f"{stem}.jpg"), cv2.resize(img[y0:y1, x0:x1], (W, H)))
            (lbl.parent / f"{stem}.txt").write_text("\n".join(" ".join(f"{v:.6g}" for v in c) for c in labels) + "\n")
            new.append({**r._asdict(), "image": str(img_path.parent / f"{stem}.jpg"), "crop_of": r.image})
            made += 1
    if new:
        m = pd.concat([m, pd.DataFrame(new)], ignore_index=True)
        m.to_csv(root_p / "manifest.csv", index=False)
        meta = yaml.safe_load(open(root_p / "build.yaml"))
        write_lists(m, meta["names"], root_p, meta.get("kpt_shape"), meta.get("flip_idx"))
    return len(new)
