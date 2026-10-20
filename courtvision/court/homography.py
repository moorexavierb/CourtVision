"""Image <-> court mapping and camera recovery.

``estimate_homographies`` turns per-frame keypoints into a per-frame image->court
homography (metres) with RANSAC, rejects fits with a large reprojection error,
holds the last good fit through short dropouts, and smooths jitter while
resetting on camera cuts.

``camera_from_homography`` recovers a full 3x4 projection matrix from the
floor homography (focal length from the orthogonality constraint, principal
point at the image centre). That is what lets us lift the ball off the floor
in 3D: a homography only knows about z = 0.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .geometry import CourtSpec


def fit_homography(kps: np.ndarray, court: CourtSpec, min_conf: float = 0.5,
                   max_error_m: float = 1.0) -> Optional[np.ndarray]:
    return fit_homography_info(kps, court, min_conf, max_error_m)[0]


def fit_homography_info(kps: np.ndarray, court: CourtSpec, min_conf: float = 0.5,
                        max_error_m: float = 1.0) -> tuple[Optional[np.ndarray], dict]:
    """Fit image->court from one frame's keypoints; also report what went into the fit.

    info: n_kp (keypoints above min_conf), kp_conf (their mean confidence), err_m (median
    reprojection error of the RANSAC inliers, in metres), used (bool mask of inlier keypoints),
    reason (why the fit failed, else "ok").
    """
    mask = kps[:, 2] >= min_conf
    info = {"n_kp": int(mask.sum()), "kp_conf": float(kps[mask, 2].mean()) if mask.any() else 0.0,
            "err_m": float("nan"), "used": np.zeros(len(kps), bool), "reason": "ok"}
    if mask.sum() < 4:
        info["reason"] = "too_few_keypoints"
        return None, info
    img = kps[mask, :2].astype(np.float64)
    world = court.keypoints[mask]
    # Four or more points on one line give a degenerate fit.
    if np.linalg.matrix_rank(np.column_stack([world, np.ones(len(world))])) < 3:
        info["reason"] = "collinear"
        return None, info
    H, inliers = cv2.findHomography(img, world, cv2.RANSAC, ransacReprojThreshold=0.75)
    if H is None or inliers is None or inliers.sum() < 4:
        info["reason"] = "ransac_failed"
        return None, info
    inl = inliers.ravel() == 1
    err = np.linalg.norm(apply_h(H, img) - world, axis=1)[inl]
    info["err_m"] = float(np.median(err))
    used = np.zeros(len(kps), bool)
    used[np.flatnonzero(mask)[inl]] = True
    info["used"] = used
    if np.median(err) > max_error_m:
        info["reason"] = "high_error"
        return None, info
    return H / H[2, 2], info


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return cv2.perspectiveTransform(pts[None], H)[0]


def estimate_homographies(keypoints: list[np.ndarray], court: CourtSpec, min_conf: float = 0.5,
                          hold_frames: int = 15, smoothing: float = 0.6,
                          cut_threshold_m: float = 2.0) -> list[Optional[np.ndarray]]:
    out: list[Optional[np.ndarray]] = []
    prev, since = None, 0
    probe = court.keypoints
    for kps in keypoints:
        H = fit_homography(kps, court, min_conf)
        if H is None:
            since += 1
            out.append(prev if prev is not None and since <= hold_frames else None)
            continue
        if prev is not None and since <= hold_frames:
            # Compare where both fits send the same image points; a big jump is a cut.
            img_probe = apply_h(np.linalg.inv(H), probe)
            jump = np.median(np.linalg.norm(apply_h(prev, img_probe) - probe, axis=1))
            if jump < cut_threshold_m:
                H = smoothing * prev + (1 - smoothing) * H
                H /= H[2, 2]
        prev, since = H, 0
        out.append(H)
    return out


def _blend(A: np.ndarray, B: np.ndarray, w_a: float) -> np.ndarray:
    H = w_a * A + (1 - w_a) * B
    return H / H[2, 2]


def _disagreement(H: np.ndarray, kps: np.ndarray, used: np.ndarray, court: CourtSpec) -> float:
    """Median distance (m) between where H sends this frame's keypoints and where they belong."""
    return float(np.median(np.linalg.norm(apply_h(H, kps[used, :2]) - court.keypoints[used], axis=1)))


def track_homographies(keypoints: list[np.ndarray], motion: list[Optional[np.ndarray]], court: CourtSpec,
                       min_conf: float = 0.5, max_flow_frames: int = 150, smoothing: float = 0.5,
                       reject_threshold_m: float = 1.5, reset_frames: int = 5
                       ) -> tuple[list[Optional[np.ndarray]], list[dict]]:
    """Per-frame image->court homographies that follow the camera between keypoint sightings.

    ``motion[t]`` maps frame t-1 pixels to frame t pixels (``CameraMotionEstimator``), or is None
    at a cut. Within a shot:

    * the mapping is predicted from the previous frame through the camera motion, so it follows
      pans and zooms with no lag;
    * a keypoint fit that agrees with the prediction is blended in (``smoothing`` = weight on the
      prediction), which removes jitter; one that disagrees (a mislabelled keypoint, a left/right
      mix-up) is rejected unless ``reset_frames`` consistent fits in a row say the track is wrong;
    * frames with no keypoint fit are filled from both directions (forward from the last fit,
      backward from the next), weighted by how many frames each had to travel, up to
      ``max_flow_frames``.

    Returns (homographies, info) where info[t] has n_kp, kp_conf, fit_err_m, source
    (fused | keypoints | flow | none), flow_age (frames since a keypoint fit), kp_rejected,
    disagreement_m (keypoint fit vs tracked mapping) and motion_ok.
    """
    n = len(keypoints)
    fits = [fit_homography_info(k, court, min_conf) for k in keypoints]
    INF = 10 ** 9
    fwd: list[Optional[np.ndarray]] = [None] * n
    age_f = [INF] * n
    info: list[dict] = []
    prev, prev_age = None, INF
    cand, cand_count = None, 0
    for t in range(n):
        M = motion[t] if t < len(motion) else None
        Hk, fi = fits[t]
        d = {"n_kp": fi["n_kp"], "kp_conf": fi["kp_conf"], "fit_err_m": fi["err_m"], "fit": fi["reason"],
             "kp_rejected": False, "disagreement_m": float("nan"), "motion_ok": M is not None}
        pred, pred_age = None, INF
        if prev is not None and M is not None and prev_age < max_flow_frames:
            pred, pred_age = prev @ np.linalg.inv(M), prev_age + 1
            pred = pred / pred[2, 2]
        if cand is not None:
            cand = cand @ np.linalg.inv(M) if M is not None else None
            cand = None if cand is None else cand / cand[2, 2]
        H, age = pred, pred_age
        d["source"] = "flow"
        if Hk is not None:
            if pred is None:
                H, age, d["source"] = Hk, 0, "keypoints"
            else:
                dis = _disagreement(pred, keypoints[t], fi["used"], court)
                d["disagreement_m"] = dis
                if dis < reject_threshold_m:
                    H, age, cand, cand_count = _blend(pred, Hk, smoothing), 0, None, 0
                    d["source"] = "fused"
                else:
                    d["kp_rejected"] = True
                    if cand is not None and _disagreement(cand, keypoints[t], fi["used"], court) < reject_threshold_m:
                        cand, cand_count = _blend(cand, Hk, smoothing), cand_count + 1
                    else:
                        cand, cand_count = Hk, 1
                    if cand_count >= reset_frames:  # the keypoints consistently say we drifted
                        H, age, cand, cand_count = cand, 0, None, 0
                        d["source"] = "keypoints"
        fwd[t], age_f[t] = H, age
        prev, prev_age = H, age
        info.append(d)

    # Backward pass: carry each keypoint-anchored mapping back through the motion of earlier frames.
    out: list[Optional[np.ndarray]] = list(fwd)
    back, age_b = None, INF
    for t in range(n - 1, -1, -1):
        if age_f[t] == 0:
            back, age_b = fwd[t], 0
        elif back is not None and t + 1 < len(motion) and motion[t + 1] is not None and age_b < max_flow_frames:
            back, age_b = back @ motion[t + 1], age_b + 1
            back = back / back[2, 2]
        else:
            back, age_b = None, INF
        af = age_f[t]
        if age_f[t] > 0 and back is not None:
            if fwd[t] is None:
                out[t], af = back, age_b
            else:
                out[t] = _blend(fwd[t], back, age_b / (age_f[t] + age_b))
                af = min(age_f[t], age_b)
        info[t]["flow_age"] = int(af) if af < INF else None
        if out[t] is None:
            info[t]["source"] = "none"
        elif af > 0:
            info[t]["source"] = "flow"
    return out, info


def confirm_cuts(cuts: set[int], homographies: list[Optional[np.ndarray]], image_size: tuple[int, int],
                 jump_m: float = 2.0) -> set[int]:
    """Keep the histogram cuts where the court mapping also jumps (or is lost).

    Flashes, replay wipes and graphics can swing the brightness histogram without the camera
    changing view. A real cut shows the court from somewhere else, so the same pixels land more
    than ``jump_m`` apart on the court before and after it. With no court mapping at all, every
    histogram cut is kept.
    """
    if not any(H is not None for H in homographies):
        return set(cuts)
    w, h = image_size
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5) * w, np.linspace(0.5, 0.95, 4) * h)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    out = set()
    for t in cuts:
        a = homographies[t - 1] if 0 < t <= len(homographies) else None
        b = homographies[t] if t < len(homographies) else None
        if a is None or b is None or np.median(np.linalg.norm(apply_h(a, pts) - apply_h(b, pts), axis=1)) > jump_m:
            out.add(t)
    return out


def camera_from_homography(H_img_to_court: np.ndarray, image_size: tuple[int, int],
                           focal: float | None = None) -> dict:
    """Recover K and a 3x4 projection P (court metres -> pixels) from a floor homography.

    The court frame has z pointing up (camera above the floor).
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    G = np.linalg.inv(H_img_to_court)  # court -> image
    G = G / G[2, 2]
    if focal is None:
        focal = _focal_from_homography(G, cx, cy) or 1.2 * w
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    B = np.linalg.inv(K) @ G
    lam = 1.0 / np.linalg.norm(B[:, 0])
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    # Court centre must be in front of the camera.
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    # Orthonormalise [r1 r2] (noise makes them slightly non-orthogonal).
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2]), full_matrices=False)
    r1, r2 = (U @ Vt).T
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    center = -R.T @ t
    if center[2] < 0:
        # Court axes are left-handed relative to "up": flip z so the camera sits above the floor.
        r3 = -r3
        R = np.column_stack([r1, r2, r3])
        center = -np.linalg.solve(R, t)
    P = K @ np.column_stack([R, t])
    return {"K": K, "R": R, "t": t, "P": P, "center": center, "focal": focal}


def _focal_from_homography(G: np.ndarray, cx: float, cy: float) -> float | None:
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    A = T @ G
    h1, h2 = A[:, 0], A[:, 1]
    cands = []
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
        if f2 > 0:
            cands.append(f2)
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        f2 = -(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den
        if f2 > 0:
            cands.append(f2)
    if not cands:
        return None
    return float(np.sqrt(np.median(cands)))


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    x = (P @ np.column_stack([X, np.ones(len(X))]).T).T
    return x[:, :2] / x[:, 2:3]
