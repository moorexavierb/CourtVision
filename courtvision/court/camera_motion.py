"""Frame-to-frame camera motion for a panning, zooming broadcast camera.

A broadcast camera turns and zooms about a fixed point, so consecutive frames
are related by a single homography for *every* pixel (floor and crowd alike).
We estimate it from background features tracked with optical flow:

* players and referees are masked out (they move on their own),
* features are spread over a grid so no single region dominates,
* screen overlays (scoreboard, clock, bug) are learned on the fly: grid cells
  whose features stay still while the rest of the picture moves are masked
  from then on,
* forward-backward flow checking and RANSAC reject bad tracks.

A frame whose motion cannot be trusted (too few inliers, a histogram jump, an
implausible zoom or shift) gets ``None``: that is a camera cut, and court
mappings are not carried across it.

This lets ``court.homography.track_homographies`` carry a court mapping from
frames where keypoints are visible through frames where they are not, instead
of holding a stale mapping while the camera pans. It runs as a forward filter
over a frame stream, so it also works for live input.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np
from tqdm.auto import tqdm

GRID = (16, 9)  # cells across, down


class CameraMotionEstimator:
    def __init__(self, width: int = 640, max_corners: int = 800, per_cell: int = 8,
                 min_inliers: int = 15, min_inlier_ratio: float = 0.3, cut_hist_corr: float = 0.6,
                 max_zoom_step: float = 0.15, max_shift_frac: float = 0.25):
        self.width = width
        self.max_corners = max_corners
        self.per_cell = per_cell
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.cut_hist_corr = cut_hist_corr
        self.max_zoom_step = max_zoom_step
        self.max_shift_frac = max_shift_frac

    def run(self, frames: Iterable[np.ndarray], boxes: list[list[list[float]]] | None = None,
            total: int | None = None) -> tuple[list[Optional[np.ndarray]], list[float]]:
        """Return (motion, inlier_ratio) per frame.

        ``motion[t]`` maps pixel coordinates of frame t-1 to frame t (full resolution), or is None
        for the first frame and at cuts. ``boxes[t]`` lists the moving-object boxes in frame t.
        Afterwards ``self.cuts`` lists the frames that start a new shot (a histogram jump), as
        opposed to frames where the motion fit merely failed.
        """
        self.cuts: list[int] = []
        motion: list[Optional[np.ndarray]] = []
        ratios: list[float] = []
        prev = prev_hist = None
        s = 1.0
        static_hits = np.zeros(GRID[::-1])
        static_obs = np.zeros(GRID[::-1])
        for t, frame in enumerate(tqdm(frames, total=total, desc="camera motion")):
            h, w = frame.shape[:2]
            s = self.width / w
            gray = cv2.cvtColor(cv2.resize(frame, (self.width, int(round(h * s)))), cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            M, ratio = None, 0.0
            if prev is not None:
                if cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < self.cut_hist_corr:
                    self.cuts.append(t)
                else:
                    mask = self._mask(prev.shape, boxes[t - 1] if boxes else [], s, static_hits, static_obs)
                    M, ratio = self._step(prev, gray, mask, static_hits, static_obs)
                if M is not None:
                    S = np.diag([s, s, 1.0])
                    M = np.linalg.inv(S) @ M @ S
                    M /= M[2, 2]
            motion.append(M)
            ratios.append(ratio)
            prev, prev_hist = gray, hist
        return motion, ratios

    # ------------------------------------------------------------------ internals
    def _mask(self, shape, boxes, s, static_hits, static_obs) -> np.ndarray:
        h, w = shape
        mask = np.full((h, w), 255, np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = (np.asarray(b) * s).astype(int)
            pad = int(0.15 * (y2 - y1)) + 2
            mask[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
        overlay = (static_obs >= 10) & (static_hits / np.maximum(static_obs, 1) > 0.6)
        ch, cw = h / GRID[1], w / GRID[0]
        for r, c in zip(*np.nonzero(overlay)):
            mask[int(r * ch):int((r + 1) * ch), int(c * cw):int((c + 1) * cw)] = 0
        return mask

    def _step(self, prev, cur, mask, static_hits, static_obs) -> tuple[Optional[np.ndarray], float]:
        pts = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 7, mask=mask)
        if pts is None or len(pts) < self.min_inliers:
            return None, 0.0
        pts = self._spread(pts.reshape(-1, 2), prev.shape)
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if len(a) < self.min_inliers:
            return None, 0.0
        M, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
        if M is None:
            return None, 0.0
        inl = inl.ravel().astype(bool)
        ratio = float(inl.mean())
        if inl.sum() < self.min_inliers or ratio < self.min_inlier_ratio or not self._plausible(M, prev.shape):
            return None, ratio
        # Learn overlays: features that stay put while the camera clearly moves.
        moved = np.linalg.norm(cv2.perspectiveTransform(a[None], M)[0] - a, axis=1)
        if np.median(moved) > 1.5:
            still = np.linalg.norm(b - a, axis=1) < 0.3
            h, w = prev.shape
            cols = np.clip((a[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1)
            rows = np.clip((a[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1)
            np.add.at(static_obs, (rows, cols), 1)
            np.add.at(static_hits, (rows, cols), still.astype(float))
        return M / M[2, 2], ratio

    def _spread(self, pts: np.ndarray, shape) -> np.ndarray:
        """Keep at most ``per_cell`` features per grid cell (strongest first, as returned)."""
        h, w = shape
        cell = (np.clip((pts[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1) * GRID[0]
                + np.clip((pts[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1))
        keep, count = [], {}
        for i, c in enumerate(cell):
            if count.get(c, 0) < self.per_cell:
                keep.append(i)
                count[c] = count.get(c, 0) + 1
        return pts[keep]

    def _plausible(self, M: np.ndarray, shape) -> bool:
        h, w = shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        moved = cv2.perspectiveTransform(corners[None], M)[0]
        area = cv2.contourArea(moved) / (w * h)
        shift = np.linalg.norm(moved.mean(axis=0) - corners.mean(axis=0)) / w
        return abs(area - 1) <= 2 * self.max_zoom_step and shift <= self.max_shift_frac
