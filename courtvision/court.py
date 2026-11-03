"""The court in metres, its 18 landmarks, and image <-> court homographies.

Court frame: origin at a corner, x along the sideline (baseline to baseline, 0 to 28.65 m on an
NBA court), y across (sideline to sideline, 0 to 15.24 m). The 18 landmarks use the order of the
public Roboflow court-keypoint dataset (reloc2-den7l), at true NBA (or FIBA) positions:

    0-5    left baseline: corner, corner-3, lane, lane, corner-3, corner
    6-7    half-court line ends
    8-9    left free-throw line ends
    10-15  right baseline (mirrored)
    16-17  right free-throw line ends
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import cv2
import numpy as np

# Index each landmark becomes when the image is flipped left-right. Goes into the pose data.yaml.
FLIP_IDX = [15, 14, 13, 12, 11, 10, 6, 7, 16, 17, 5, 4, 3, 2, 1, 0, 8, 9]
N_KEYPOINTS = 18


@dataclass(frozen=True)
class CourtSpec:
    name: str
    length: float
    width: float
    lane_width: float
    free_throw_dist: float       # baseline to free-throw line
    corner_three_offset: float   # sideline to corner-three line
    three_radius: float
    hoop_offset: float           # baseline to centre of rim

    @cached_property
    def keypoints(self) -> np.ndarray:
        """(18, 2) landmark positions in metres, in dataset order."""
        L, W, c, ft = self.length, self.width, self.corner_three_offset, self.free_throw_dist
        lo = (W - self.lane_width) / 2
        hi = W - lo
        return np.array([
            (0, 0), (0, c), (0, lo), (0, hi), (0, W - c), (0, W),
            (L / 2, W), (L / 2, 0),
            (ft, lo), (ft, hi),
            (L, W), (L, W - c), (L, hi), (L, lo), (L, c), (L, 0),
            (L - ft, lo), (L - ft, hi),
        ], dtype=np.float64)

    @cached_property
    def hoops(self) -> np.ndarray:
        """(2, 2) rim centres (x, y) for the left and right baskets."""
        return np.array([(self.hoop_offset, self.width / 2), (self.length - self.hoop_offset, self.width / 2)])

    def nearest_hoop(self, x: float, y: float) -> np.ndarray:
        return self.hoops[int(np.argmin(np.hypot(*(self.hoops - (x, y)).T)))]

    def in_bounds(self, x: float, y: float, margin: float = 1.5) -> bool:
        """On the court, with a margin for players standing on the lines or just out of bounds."""
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin


NBA = CourtSpec("nba", 28.65, 15.24, 4.88, 5.79, 0.91, 7.24, 1.575)
FIBA = CourtSpec("fiba", 28.0, 15.0, 4.9, 5.8, 0.9, 6.75, 1.575)


def get_court(standard: str) -> CourtSpec:
    return {"nba": NBA, "fiba": FIBA}[standard.lower()]


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    """Map (N, 2) points through a 3x3 homography."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return cv2.perspectiveTransform(pts[None], H)[0]


def fit_homography(kps: np.ndarray, court: CourtSpec, min_conf: float = 0.5,
                   max_error_m: float = 1.0) -> tuple[Optional[np.ndarray], dict]:
    """Image -> court homography from one frame's (18, 3) keypoints (x, y, conf).

    Returns (H or None, info). info: n_kp (keypoints above min_conf), err_m (median error of the
    RANSAC inliers on the court, in metres), used (mask of inlier keypoints), reason ("ok" or why
    the fit failed: too_few_keypoints, collinear, ransac_failed, high_error).
    """
    mask = kps[:, 2] >= min_conf
    info = {"n_kp": int(mask.sum()), "err_m": float("nan"), "used": np.zeros(len(kps), bool), "reason": "ok"}
    if mask.sum() < 4:
        info["reason"] = "too_few_keypoints"
        return None, info
    img, world = kps[mask, :2].astype(np.float64), court.keypoints[mask]
    if np.linalg.matrix_rank(np.column_stack([world, np.ones(len(world))])) < 3:
        info["reason"] = "collinear"  # four points on one line say nothing about the other axis
        return None, info
    H, inliers = cv2.findHomography(img, world, cv2.RANSAC, ransacReprojThreshold=0.75)
    if H is None or inliers is None or inliers.sum() < 4:
        info["reason"] = "ransac_failed"
        return None, info
    inl = inliers.ravel() == 1
    info["err_m"] = float(np.median(np.linalg.norm(apply_h(H, img) - world, axis=1)[inl]))
    info["used"][np.flatnonzero(mask)[inl]] = True
    if info["err_m"] > max_error_m:
        info["reason"] = "high_error"
        return None, info
    return H / H[2, 2], info


def orientation(H_img_to_court: np.ndarray, img_pts) -> int:
    """+1 or -1: which way round a mapping is (the sign of its determinant, with the points taken
    in front of the camera). Every real view of a court labelled one way has the same sign; a fit
    to labels with the left and right halves swapped (a mirror image) has the other. Which sign is
    right depends on the dataset's landmark convention, so it is measured on hand-labelled frames
    (notebook 02) rather than assumed here."""
    pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    w = np.column_stack([pts, np.ones(len(pts))]) @ H_img_to_court[2]
    return int(np.sign(np.linalg.det(H_img_to_court)) * np.sign(np.median(w)))


def landmarks_in_image(H_img_to_court: np.ndarray, court: CourtSpec, size: tuple[int, int],
                       margin: float = 0.0) -> np.ndarray:
    """Where every landmark falls in the image under a court mapping: (18, 3) x, y, visible.

    visible is 1 for landmarks inside the frame (and in front of the camera), else 0. Used to
    label training frames from a verified mapping, including landmarks the model missed.
    """
    G = np.linalg.inv(H_img_to_court)
    X = np.column_stack([court.keypoints, np.ones(N_KEYPOINTS)]) @ G.T
    w = X[:, 2]
    xy = X[:, :2] / np.where(np.abs(w) < 1e-12, 1e-12, w)[:, None]
    W, Hh = size
    front = np.sign(w) == np.sign(np.median(w))
    inside = (xy[:, 0] >= -margin) & (xy[:, 0] < W + margin) & (xy[:, 1] >= -margin) & (xy[:, 1] < Hh + margin)
    return np.column_stack([xy, (front & inside).astype(float)])
