"""Scores that match what the tracking dataset needs, per source dataset.

mAP rewards a good ranking of boxes over a whole dataset. The pipeline instead runs at one
confidence threshold and needs, in every frame, all ten players, the ball, and enough court
landmarks to map the frame. So besides Ultralytics' mAP, models are scored on:

* detector: per-frame player recall (players found / players labelled, averaged over frames) and
  the share of frames where every labelled player was found; ball precision and recall; the share
  of frames with a phantom ball;
* tracking (sequences with track ids, e.g. SportsMOT val): MOTA and IDF1, because a court
  position is only useful if it stays attached to one player;
* court model: share of frames that get a usable homography, landmarks per frame, and, where
  labels exist, landmark error in pixels and PCK (share within 2% of the image width).

Only classes a source labels itself are scored on it (``classes``): SportsMOT's ball labels are
the old detector's guesses, and scoring a new model against them would measure agreement with
the old model, not accuracy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> tuple[list[tuple[int, int]], int, int]:
    """One-to-one matches (Hungarian on IoU) above iou_thr. Returns (pairs, false positives, misses)."""
    ious = iou_matrix(gt, pred)
    if ious.size == 0:
        return [], len(np.asarray(pred).reshape(-1, 4)), len(np.asarray(gt).reshape(-1, 4))
    r, c = linear_sum_assignment(-ious)
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if ious[i, j] >= iou_thr]
    return pairs, ious.shape[1] - len(pairs), ious.shape[0] - len(pairs)


def label_path(image: str | Path) -> Path:
    """Ultralytics' rule: the label sits at the image path with the last /images/ swapped for /labels/."""
    s = str(image)
    i = s.rfind("/images/")
    s = s[:i] + "/labels/" + s[i + len("/images/"):] if i >= 0 else s
    return Path(s).with_suffix(".txt")


def read_yolo(label_path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(xyxy pixels, class ids) from a YOLO label file; empty arrays if missing."""
    W, H = size
    boxes, cls = [], []
    if Path(label_path).exists():
        for line in open(label_path):
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
            cls.append(c)
    return np.array(boxes, float).reshape(-1, 4), np.array(cls, int)


def frame_scores(gt_xyxy: np.ndarray, gt_cls: np.ndarray, pred_xyxy: np.ndarray, pred_cls: np.ndarray,
                 pred_conf: np.ndarray, classes: dict[str, int], conf: dict[str, float],
                 iou_thr: float = 0.5) -> dict:
    """True/false positives and misses per class for one frame, each class at its own threshold."""
    out = {}
    for name, cid in classes.items():
        g = gt_xyxy[gt_cls == cid]
        keep = (pred_cls == cid) & (pred_conf >= conf.get(name, 0.25))
        pairs, fp, fn = match(g, pred_xyxy[keep], iou_thr)
        out[name] = {"gt": len(g), "tp": len(pairs), "fp": fp, "fn": fn}
    return out


def summarise(frames: list[dict], expected_players: Optional[int] = None) -> dict:
    """Aggregate frame_scores over a dataset into the numbers the pipeline cares about."""
    report: dict = {"frames": len(frames)}
    if not frames:
        return report
    for name in frames[0]:
        tp = sum(f[name]["tp"] for f in frames)
        fp = sum(f[name]["fp"] for f in frames)
        gt = sum(f[name]["gt"] for f in frames)
        report[f"{name}_precision"] = round(tp / (tp + fp), 3) if tp + fp else None
        report[f"{name}_recall"] = round(tp / gt, 3) if gt else None
    if "Player" in frames[0]:
        with_players = [f["Player"] for f in frames if f["Player"]["gt"]]
        if with_players:
            report["player_recall_per_frame"] = round(float(np.mean([p["tp"] / p["gt"] for p in with_players])), 3)
            report["all_players_found_share"] = round(float(np.mean([p["fn"] == 0 for p in with_players])), 3)
            report["players_found_per_frame"] = round(float(np.mean([p["tp"] for p in with_players])), 2)
            report["players_labelled_per_frame"] = round(float(np.mean([p["gt"] for p in with_players])), 2)
    if "Ball" in frames[0]:
        report["phantom_ball_frame_share"] = round(float(np.mean([f["Ball"]["fp"] > 0 for f in frames])), 3)
    return report


def score_images(model, images: Iterable[str], class_names: list[str], conf: dict[str, float],
                 imgsz: int = 1280, iou_thr: float = 0.5, batch: int = 16) -> list[dict]:
    """Run an Ultralytics model over YOLO-labelled images and score every frame (needs a GPU stack).
    conf: {class name: threshold} for the classes to score."""
    images = list(images)
    wanted = {n: class_names.index(n) for n in conf if n in class_names}
    model_ids = {v: k for k, v in model.names.items()}
    remap = {model_ids[n]: class_names.index(n) for n in class_names if n in model_ids}
    min_conf = min(conf.values())
    frames = []
    for i in range(0, len(images), batch):
        for res in model.predict(images[i:i + batch], imgsz=imgsz, conf=min_conf, verbose=False):
            h, w = res.orig_shape
            g_xyxy, g_cls = read_yolo(label_path(res.path), (w, h))
            b = res.boxes
            p_cls = np.array([remap.get(int(c), -1) for c in b.cls.cpu().numpy()], int)
            frames.append(frame_scores(g_xyxy, g_cls, b.xyxy.cpu().numpy(), p_cls, b.conf.cpu().numpy(),
                                       wanted, conf, iou_thr))
    return frames


# ----------------------------------------------------------------- tracking

def mot_metrics(gt: dict[int, dict[int, list[float]]], pred: dict[int, dict[int, list[float]]],
                iou_thr: float = 0.5) -> dict:
    """CLEAR-MOT MOTA, identity switches and IDF1 for one sequence.

    gt, pred: {frame: {track id: [x0, y0, x1, y1]}}. MOTA = 1 - (misses + false positives +
    id switches) / ground-truth boxes. IDF1 is the F1 of boxes matched under the best
    one-to-one mapping between ground-truth and predicted identities over the whole sequence,
    so it drops whenever a player's track changes id or swaps with another player.
    """
    frames = sorted(set(gt) | set(pred))
    n_gt = sum(len(v) for v in gt.values())
    n_pred = sum(len(v) for v in pred.values())
    fp = fn = idsw = tp = 0
    last_match: dict[int, int] = {}
    gt_ids = sorted({i for v in gt.values() for i in v})
    pr_ids = sorted({i for v in pred.values() for i in v})
    gi = {g: k for k, g in enumerate(gt_ids)}
    pi = {p: k for k, p in enumerate(pr_ids)}
    overlap = np.zeros((len(gt_ids), len(pr_ids)))
    for f in frames:
        g, p = gt.get(f, {}), pred.get(f, {})
        g_ids, p_ids = list(g), list(p)
        ious = iou_matrix(np.array([g[i] for i in g_ids]), np.array([p[i] for i in p_ids]))
        if ious.size:
            for a, b in zip(*np.nonzero(ious >= iou_thr)):
                overlap[gi[g_ids[a]], pi[p_ids[b]]] += 1
        pairs, f_fp, f_fn = match(np.array([g[i] for i in g_ids]), np.array([p[i] for i in p_ids]), iou_thr)
        fp, fn, tp = fp + f_fp, fn + f_fn, tp + len(pairs)
        for a, b in pairs:
            gid, pid = g_ids[a], p_ids[b]
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
    idtp = 0.0
    if overlap.size:
        r, c = linear_sum_assignment(-overlap)
        idtp = float(overlap[r, c].sum())
    return {"gt_boxes": n_gt, "gt_ids": len(gt_ids), "pred_ids": len(pr_ids),
            "mota": round(1 - (fn + fp + idsw) / n_gt, 3) if n_gt else None,
            "idf1": round(2 * idtp / (n_gt + n_pred), 3) if n_gt + n_pred else None,
            "id_switches": idsw, "recall": round(tp / n_gt, 3) if n_gt else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None}


def mot_table(per_sequence: dict[str, dict]) -> pd.DataFrame:
    """One row per sequence plus a box-weighted overall row."""
    df = pd.DataFrame(per_sequence).T
    if df.empty:
        return df
    w = df["gt_boxes"].astype(float)
    overall = {c: round(float((df[c].astype(float) * w).sum() / w.sum()), 3) for c in ["mota", "idf1", "recall", "precision"]}
    overall.update(gt_boxes=int(w.sum()), gt_ids=int(df["gt_ids"].sum()), pred_ids=int(df["pred_ids"].sum()),
                   id_switches=int(df["id_switches"].sum()))
    df.loc["overall"] = overall
    return df


# ----------------------------------------------------------------- court keypoints

def court_scores(model, images: Iterable[str], court, imgsz: int = 640, kp_conf: float = 0.5,
                 max_err_m: float = 0.5, orientation_sign: Optional[int] = None, batch: int = 16) -> dict:
    """Court model on images: share mapped (its keypoints fit one homography within max_err_m,
    the right way round when orientation_sign is given), landmarks per frame, and against labels
    where they exist, median landmark error (px) and PCK@2% of width (a missed landmark counts
    as wrong)."""
    from ..court import fit_homography, orientation
    from ..detect import best_court_instance
    images = list(images)
    mapped, n_kp, errs, pck = [], [], [], []
    for i in range(0, len(images), batch):
        for res in model.predict(images[i:i + batch], imgsz=imgsz, conf=0.25, verbose=False):
            k = best_court_instance(res)
            H, info = fit_homography(k, court, kp_conf, max_error_m=max_err_m)
            if H is not None and orientation_sign is not None:
                H = H if orientation(H, k[k[:, 2] >= kp_conf, :2]) == orientation_sign else None
            mapped.append(H is not None)
            n_kp.append(info["n_kp"])
            lbl = label_path(res.path)
            if lbl.exists() and lbl.read_text().split():
                h, w = res.orig_shape
                v = np.array(lbl.read_text().split()[5:], float).reshape(-1, 3)
                vis = v[:, 2] > 0
                if vis.any():
                    d = np.linalg.norm(k[vis, :2] - v[vis, :2] * (w, h), axis=1)
                    d[k[vis, 2] < kp_conf] = np.inf  # a landmark the model missed counts as wrong
                    errs += d[np.isfinite(d)].tolist()
                    pck += (d <= 0.02 * w).tolist()
    return {"images": len(mapped), "mapped_share": round(float(np.mean(mapped)), 3) if mapped else None,
            "keypoints_per_frame": round(float(np.mean(n_kp)), 2) if n_kp else None,
            "median_px_error": round(float(np.median(errs)), 1) if errs else None,
            "pck_2pct": round(float(np.mean(pck)), 3) if pck else None}


# ----------------------------------------------------------------- new vs current

def compare(new: dict, current: dict, higher=(), lower=(), tol: float = 0.01) -> tuple[bool, list[str]]:
    """Is the new model at least as good? higher: metrics where more is better; lower: where less
    is better. A metric missing (None) on either side is not compared. Returns (ok, reasons)."""
    reasons = []
    for key, sign in [(k, 1) for k in higher] + [(k, -1) for k in lower]:
        a, b = new.get(key), current.get(key)
        if a is None or b is None or a != a or b != b:  # missing or NaN
            continue
        if sign * (a - b) < -tol:
            reasons.append(f"{key}: new {a} vs current {b}")
    return not reasons, reasons
