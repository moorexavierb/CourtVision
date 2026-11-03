"""YOLO label rows, and labels made by a model where a dataset has none.

Detection: a dataset that boxes only some of our classes (SportsMOT boxes players, not the ball,
referees or hoops) would teach the detector that the missing classes are background. Its missing
classes are filled from the current detector's confident boxes; a frame where the detector is
unsure (a box between low_conf and keep_conf) is skipped rather than half-labelled.

Court keypoints: none of the extra datasets label court landmarks. The current court model's
keypoints are kept only when they agree with the court's real geometry: a homography is fitted
to the confident keypoints, and the frame is used only if they all fit within ``max_err_m``. The
fitted mapping then places *every* landmark in the frame, including the ones the model missed,
which is where the new labels add information. The same works from a mapping carried through a
pan by camera motion (``court_row_from_h``), which labels zoomed frames where the model sees
too few keypoints itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..court import CourtSpec, fit_homography, landmarks_in_image, orientation
from .evaluate import iou_matrix


def box_rows(xyxy, cls, size: tuple[int, int]) -> list[str]:
    """Pixel boxes to YOLO 'cls cx cy w h' rows, normalised by size (width, height)."""
    W, H = size
    return [f"{int(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
            for (x0, y0, x1, y1), c in zip(np.asarray(xyxy, float).reshape(-1, 4), np.asarray(cls, int))]


def pose_row(xy: np.ndarray, visible: np.ndarray, size: tuple[int, int], cls: int = 0) -> Optional[str]:
    """One court instance as a YOLO pose row; the box spans the visible landmarks."""
    W, H = size
    vis = np.asarray(visible, bool)
    if not vis.any():
        return None
    pts = np.asarray(xy, float)
    x0, x1 = pts[vis, 0].min(), pts[vis, 0].max()
    y0, y1 = pts[vis, 1].min(), pts[vis, 1].max()
    box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, max(x1 - x0, 1) / W, max(y1 - y0, 1) / H]
    kp = []
    for (x, y), v in zip(pts, vis):
        kp += [x / W, y / H, 2] if v else [0, 0, 0]
    return " ".join([str(cls)] + [f"{v:.6f}" for v in box] + [f"{v:.6g}" for v in kp])


def fill_missing(gt_xyxy, gt_cls, pred_xyxy, pred_cls, pred_conf, fill_classes: list[int],
                 keep_conf: float = 0.5, low_conf: float = 0.15, max_iou_with_gt: float = 0.5
                 ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Ground-truth boxes plus confident model boxes of the classes the dataset lacks.

    Model boxes overlapping a ground-truth box (IoU above max_iou_with_gt) are dropped: that
    object is already labelled. Returns (xyxy, cls), or None when the frame is ambiguous.
    """
    gt_xyxy = np.asarray(gt_xyxy, float).reshape(-1, 4)
    gt_cls = np.asarray(gt_cls, int)
    pred_xyxy = np.asarray(pred_xyxy, float).reshape(-1, 4)
    pred_cls, pred_conf = np.asarray(pred_cls, int), np.asarray(pred_conf, float)
    fill = np.isin(pred_cls, fill_classes)
    if (fill & (pred_conf >= low_conf) & (pred_conf < keep_conf)).any():
        return None
    keep = fill & (pred_conf >= keep_conf)
    if keep.any() and len(gt_xyxy):
        keep &= ~(iou_matrix(pred_xyxy, gt_xyxy).max(axis=1) > max_iou_with_gt)
    return np.vstack([gt_xyxy, pred_xyxy[keep]]), np.concatenate([gt_cls, pred_cls[keep]])


def court_row_from_h(H: np.ndarray, court: CourtSpec, size: tuple[int, int], min_visible: int = 4,
                     cls: int = 0) -> Optional[str]:
    """Pose row with every landmark the mapping puts inside the frame."""
    lm = landmarks_in_image(H, court, size)
    if lm[:, 2].sum() < min_visible:
        return None
    return pose_row(lm[:, :2], lm[:, 2] > 0, size, cls)


def court_row_from_keypoints(kps: np.ndarray, court: CourtSpec, size: tuple[int, int], min_conf: float = 0.6,
                             min_kp: int = 5, max_err_m: float = 0.25, max_px_err: float = 0.01,
                             orientation_sign: Optional[int] = None, cls: int = 0) -> tuple[Optional[str], dict]:
    """Geometry-checked court label from one frame of model keypoints (18, 3).

    Accepted when at least min_kp keypoints are confident, a homography fits them all (no RANSAC
    outliers) within max_err_m on the court, the fitted mapping puts each of them back within
    max_px_err of the image width of where the model saw it, and (when orientation_sign is given)
    the mapping is the right way round: a model that mixes up the two halves of the court can
    still fit perfectly, as a mirror image. Returns (row or None, info); info["reason"] says why
    a frame was rejected.
    """
    H, info = fit_homography(kps, court, min_conf, max_error_m=max_err_m)
    conf = kps[:, 2] >= min_conf
    if H is None or conf.sum() < min_kp:
        info["reason"] = info["reason"] if H is None else "too_few_keypoints"
        return None, info
    if info["used"].sum() < conf.sum():
        info["reason"] = "outlier_keypoints"
        return None, info
    if orientation_sign is not None and orientation(H, kps[conf, :2]) != orientation_sign:
        info["reason"] = "mirrored"
        return None, info
    lm = landmarks_in_image(H, court, size)
    err = np.linalg.norm(lm[conf, :2] - kps[conf, :2], axis=1).max()
    info["max_px_err"] = float(err)
    if err > max_px_err * size[0]:
        info["reason"] = "disagrees_with_geometry"
        return None, info
    info["added"] = int(lm[:, 2].sum() - conf.sum())  # landmarks labelled that the model missed
    return court_row_from_h(H, court, size, cls=cls), info


def label_orientation(row: str, court: CourtSpec) -> Optional[int]:
    """Which way round a hand-labelled court row maps (court.orientation), or None when it has
    fewer than 4 visible landmarks. Normalised coordinates are fine: scaling the image axes does
    not change the sign. The majority over a hand-labelled dataset is the sign real views have."""
    v = np.array(row.split()[5:], float).reshape(-1, 3)
    kps = np.column_stack([v[:, :2], (v[:, 2] > 0).astype(float)])
    H, _ = fit_homography(kps, court, 0.5, max_error_m=1e9)
    return None if H is None else orientation(H, kps[kps[:, 2] > 0, :2])
