"""Draw labels on images, to check a dataset by eye before training on it (notebooks 01 and 02).

A dataset where some players are not boxed, or a pseudo-label on the wrong landmark, is easiest
to catch by looking: a grid of nine random images per source is usually enough.
"""
from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

from .evaluate import label_path

COLORS = {"Player": (60, 220, 60), "Ref": (60, 60, 220), "Ball": (0, 165, 255), "Hoop": (220, 220, 0)}


def draw_label(image_path: str, names: list[str]) -> Optional[np.ndarray]:
    """The image (RGB) with its YOLO label drawn: boxes with class names, and court landmarks
    numbered by index for pose rows."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    lbl = label_path(image_path)
    rows = [l.split() for l in open(lbl)] if lbl.exists() else []
    for r in rows:
        if len(r) < 5:
            continue
        name = names[int(r[0])] if int(r[0]) < len(names) else r[0]
        cx, cy, w, h = (float(v) for v in r[1:5])
        col = COLORS.get(name, (255, 255, 255))
        if len(r) > 5:  # pose row: landmarks, not the box
            kp = np.array(r[5:], float).reshape(-1, 3)
            for i, (x, y, v) in enumerate(kp):
                if v > 0:
                    cv2.circle(img, (int(x * W), int(y * H)), 6, (0, 0, 255), -1)
                    cv2.putText(img, str(i), (int(x * W) + 7, int(y * H) - 7), 0, 0.8, (0, 255, 255), 2)
            continue
        p0, p1 = (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H))
        cv2.rectangle(img, p0, p1, col, 2)
        cv2.putText(img, name, (p0[0], p0[1] - 3), 0, 0.5, col, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_grid(images: list[str], names: list[str], title: str = "", n: int = 9, seed: int = 0, save: str = None):
    """n random images with labels in a 3-column grid."""
    import matplotlib.pyplot as plt
    pick = random.Random(seed).sample(list(images), min(n, len(images)))
    if not pick:
        return
    rows = (len(pick) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 4 * rows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, p in zip(np.atleast_1d(axes).flat, pick):
        img = draw_label(p, names)
        if img is not None:
            ax.imshow(img)
            ax.set_title(p.rsplit("/", 1)[-1][:60], fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
