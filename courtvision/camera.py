"""Follow the broadcast camera, so every frame has an image -> court mapping.

Court keypoints are only visible now and then: a zoomed half-court shot shows a few lines, a pan
shows none for a second. A broadcast camera pans and zooms about a fixed point, so consecutive
frames are related by one homography for every pixel. This module

1. estimates that frame-to-frame motion from background features (optical flow; players and
   screen graphics masked out) and marks cuts (``estimate_motion``),
2. carries each keypoint fit forwards and backwards through the motion, blends in fits that agree
   and rejects fits that don't (a left/right mix-up), and fills frames between fits
   (``track_homographies``). Fits that are a mirror image of the court (the model swapped its two
   halves, which can still fit perfectly) are dropped first: every real view of a game maps the
   same way round, so the orientation most fits in the video share is the real one,
3. keeps only the cuts where the court mapping really jumps (``confirm_cuts``): flashes and
   graphics swing the brightness histogram without the camera moving.

Everything runs as a forward pass plus one backward fill, so it can later run on a live stream
with a short delay.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np
from tqdm.auto import tqdm

from .court import CourtSpec, apply_h, fit_homography, orientation

GRID = (16, 9)  # feature grid: cells across, down


# --------------------------------------------------------------------------- camera motion

def estimate_motion(frames: Iterable[np.ndarray], boxes: Optional[list[list]] = None, width: int = 640,
                    cut_hist_corr: float = 0.6, total: Optional[int] = None
                    ) -> tuple[list[Optional[np.ndarray]], list[int]]:
    """(motion, cuts). motion[t] maps frame t-1 pixels to frame t pixels, or is None when it could
    not be trusted (first frame, cut, too few features). cuts lists frames that start a new shot.
    boxes[t] are the player/referee boxes of frame t, masked out because they move on their own."""
    motion: list[Optional[np.ndarray]] = []
    cuts: list[int] = []
    prev = prev_hist = None
    static_hits = np.zeros(GRID[::-1])
    static_obs = np.zeros(GRID[::-1])
    for t, frame in enumerate(tqdm(frames, total=total, desc="camera motion")):
        h, w = frame.shape[:2]
        s = width / w
        gray = cv2.cvtColor(cv2.resize(frame, (width, int(round(h * s)))), cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        M = None
        if prev is not None:
            if cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < cut_hist_corr:
                cuts.append(t)
            else:
                mask = _feature_mask(prev.shape, boxes[t - 1] if boxes else [], s, static_hits, static_obs)
                M = _frame_motion(prev, gray, mask, static_hits, static_obs)
                if M is not None:
                    S = np.diag([s, s, 1.0])
                    M = np.linalg.inv(S) @ M @ S
                    M /= M[2, 2]
        motion.append(M)
        prev, prev_hist = gray, hist
    return motion, cuts


def _feature_mask(shape, boxes, s, static_hits, static_obs) -> np.ndarray:
    """Where features may be taken: not on players, not on screen graphics."""
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = (np.asarray(b) * s).astype(int)
        pad = int(0.15 * (y2 - y1)) + 2
        mask[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
    # Grid cells whose features stay still while the camera moves are graphics (score bug, clock).
    overlay = (static_obs >= 10) & (static_hits / np.maximum(static_obs, 1) > 0.6)
    ch, cw = h / GRID[1], w / GRID[0]
    for r, c in zip(*np.nonzero(overlay)):
        mask[int(r * ch):int((r + 1) * ch), int(c * cw):int((c + 1) * cw)] = 0
    return mask


def _frame_motion(prev, cur, mask, static_hits, static_obs, min_inliers: int = 15,
                  min_ratio: float = 0.3) -> Optional[np.ndarray]:
    pts = cv2.goodFeaturesToTrack(prev, 800, 0.01, 7, mask=mask)
    if pts is None or len(pts) < min_inliers:
        return None
    pts = _spread(pts.reshape(-1, 2), prev.shape)
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, winSize=(21, 21), maxLevel=3)
    fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)       # forward-backward check
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    if len(a) < min_inliers:
        return None
    M, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
    if M is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_inliers or inl.mean() < min_ratio or not _plausible(M, prev.shape):
        return None
    moved = np.linalg.norm(cv2.perspectiveTransform(a[None], M)[0] - a, axis=1)
    if np.median(moved) > 1.5:  # the camera moved: learn which cells did not
        still = np.linalg.norm(b - a, axis=1) < 0.3
        h, w = prev.shape
        cols = np.clip((a[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1)
        rows = np.clip((a[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1)
        np.add.at(static_obs, (rows, cols), 1)
        np.add.at(static_hits, (rows, cols), still.astype(float))
    return M / M[2, 2]


def _spread(pts: np.ndarray, shape, per_cell: int = 8) -> np.ndarray:
    """At most per_cell features per grid cell, so no single region dominates the fit."""
    h, w = shape
    cell = (np.clip((pts[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1) * GRID[0]
            + np.clip((pts[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1))
    keep, count = [], {}
    for i, c in enumerate(cell):
        if count.get(c, 0) < per_cell:
            keep.append(i)
            count[c] = count.get(c, 0) + 1
    return pts[keep]


def _plausible(M: np.ndarray, shape, max_zoom_step: float = 0.15, max_shift: float = 0.25) -> bool:
    h, w = shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    moved = cv2.perspectiveTransform(corners[None], M)[0]
    area = cv2.contourArea(moved) / (w * h)
    shift = np.linalg.norm(moved.mean(axis=0) - corners.mean(axis=0)) / w
    return abs(area - 1) <= 2 * max_zoom_step and shift <= max_shift


# --------------------------------------------------------------------------- court mapping

def _blend(A: np.ndarray, B: np.ndarray, w_a: float) -> np.ndarray:
    H = w_a * A + (1 - w_a) * B
    return H / H[2, 2]


def _disagreement(H: np.ndarray, kps: np.ndarray, used: np.ndarray, court: CourtSpec) -> float:
    """Median distance (m) between where H sends this frame's keypoints and where they belong."""
    return float(np.median(np.linalg.norm(apply_h(H, kps[used, :2]) - court.keypoints[used], axis=1)))


def _drop_mirrored(fits: list, keypoints: np.ndarray, min_conf: float) -> tuple[list, list[bool]]:
    """Drop fits whose orientation (court.orientation) differs from the majority of the video's
    fits. Returns (fits, mirrored flag per frame)."""
    signs = [orientation(H, k[k[:, 2] >= min_conf, :2]) if H is not None else 0 for (H, _), k in zip(fits, keypoints)]
    major = int(np.sign(sum(signs)))
    mirrored = [major != 0 and s == -major for s in signs]
    out = [(None, {**info, "reason": "mirrored"}) if m else (H, info) for (H, info), m in zip(fits, mirrored)]
    return out, mirrored


def track_homographies(keypoints: np.ndarray, motion: list[Optional[np.ndarray]], court: CourtSpec,
                       min_conf: float = 0.5, max_flow_frames: int = 150, smoothing: float = 0.5,
                       reject_m: float = 1.5, reset_frames: int = 5) -> tuple[list[Optional[np.ndarray]], list[dict]]:
    """Per-frame image -> court homography, following the camera between keypoint fits.

    Returns (homographies, calibration). calibration[t]: n_kp, fit_err_m, source (keypoints:
    fresh fit; fused: fit blended with the tracked mapping; flow: carried by camera motion;
    none), flow_age (frames since a fit), kp_rejected (fit not used), mirrored (dropped as a
    mirror image).
    """
    n = len(keypoints)
    fits, mirrored = _drop_mirrored([fit_homography(k, court, min_conf) for k in keypoints], keypoints, min_conf)
    INF = 10 ** 9
    fwd: list[Optional[np.ndarray]] = [None] * n
    age_f = [INF] * n
    calib: list[dict] = []
    prev, prev_age = None, INF
    cand, cand_count = None, 0  # a run of rejected fits that agree with each other
    for t in range(n):
        M = motion[t] if t < len(motion) else None
        Hk, fi = fits[t]
        c = {"n_kp": fi["n_kp"], "fit_err_m": fi["err_m"], "kp_rejected": mirrored[t], "mirrored": mirrored[t],
             "source": "flow"}
        pred, pred_age = None, INF
        if prev is not None and M is not None and prev_age < max_flow_frames:
            pred, pred_age = prev @ np.linalg.inv(M), prev_age + 1
            pred = pred / pred[2, 2]
        if cand is not None:
            cand = None if M is None else cand @ np.linalg.inv(M)
            cand = None if cand is None else cand / cand[2, 2]
        H, age = pred, pred_age
        if Hk is not None:
            if pred is None:
                H, age, c["source"] = Hk, 0, "keypoints"
            elif _disagreement(pred, keypoints[t], fi["used"], court) < reject_m:
                H, age, cand, cand_count, c["source"] = _blend(pred, Hk, smoothing), 0, None, 0, "fused"
            else:
                c["kp_rejected"] = True
                if cand is not None and _disagreement(cand, keypoints[t], fi["used"], court) < reject_m:
                    cand, cand_count = _blend(cand, Hk, smoothing), cand_count + 1
                else:
                    cand, cand_count = Hk, 1
                if cand_count >= reset_frames:  # the keypoints keep saying the track drifted
                    H, age, cand, cand_count, c["source"] = cand, 0, None, 0, "keypoints"
        fwd[t], age_f[t] = H, age
        prev, prev_age = H, age
        calib.append(c)

    # Backward fill: carry each fit back through earlier frames, blend with the forward estimate.
    out = list(fwd)
    back, age_b = None, INF
    for t in range(n - 1, -1, -1):
        if age_f[t] == 0:
            back, age_b = fwd[t], 0
        elif back is not None and t + 1 < len(motion) and motion[t + 1] is not None and age_b < max_flow_frames:
            back, age_b = back @ motion[t + 1], age_b + 1
            back = back / back[2, 2]
        else:
            back, age_b = None, INF
        age = age_f[t]
        if age > 0 and back is not None:
            out[t] = back if fwd[t] is None else _blend(fwd[t], back, age_b / (age_f[t] + age_b))
            age = age_b if fwd[t] is None else min(age_f[t], age_b)
        calib[t]["flow_age"] = int(age) if age < INF else None
        if out[t] is None:
            calib[t]["source"] = "none"
    return out, calib


def confirm_cuts(cuts: list[int], homographies: list[Optional[np.ndarray]], size: tuple[int, int],
                 jump_m: float = 2.0) -> list[int]:
    """Histogram cuts where the court mapping also jumps (or is lost). With no mapping at all,
    every histogram cut is kept."""
    if not any(H is not None for H in homographies):
        return list(cuts)
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5) * w, np.linspace(0.5, 0.95, 4) * h)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    out = []
    for t in cuts:
        a = homographies[t - 1] if 0 < t <= len(homographies) else None
        b = homographies[t] if t < len(homographies) else None
        if a is None or b is None or np.median(np.linalg.norm(apply_h(a, pts) - apply_h(b, pts), axis=1)) > jump_m:
            out.append(t)
    return out
