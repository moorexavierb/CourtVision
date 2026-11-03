"""Split player tracks into two teams by jersey colour.

For each track, a few torso crops are sampled across its life. In each crop a 2-cluster KMeans
separates jersey from background (the cluster that owns the crop's corners is background). Every
track's median jersey colour (LAB) is then clustered into two teams. One decision per track, so a
player never flickers between teams.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def sample_frames(tracks: pd.DataFrame, per_track: int = 12) -> dict[int, list[int]]:
    """{track_id: frames to sample}, spread evenly over each player track."""
    out = {}
    for tid, g in tracks[tracks.role == "player"].groupby("track_id"):
        fs = g.frame.to_numpy()
        out[int(tid)] = sorted(set(fs[np.linspace(0, len(fs) - 1, min(per_track, len(fs))).astype(int)].tolist()))
    return out


def torso_crop(frame: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 8:
        return None
    crop = frame[max(y1 + int(0.15 * h), 0):max(y1 + int(0.5 * h), 0),
                 max(x1 + int(0.2 * w), 0):max(x2 - int(0.2 * w), 0)]
    return crop if crop.size else None


def jersey_color(crop: np.ndarray) -> np.ndarray:
    """Dominant non-background colour of a torso crop, in LAB."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) < 4 or lab.std(axis=0).max() < 1.0:
        return lab.mean(axis=0)
    km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(lab)
    labels = km.labels_.reshape(crop.shape[:2])
    background = Counter([labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]).most_common(1)[0][0]
    return km.cluster_centers_[1 - background]


def assign_teams(frames: dict[int, np.ndarray], tracks: pd.DataFrame, per_track: int = 12
                 ) -> tuple[dict[int, int], np.ndarray | None]:
    """({track_id: 0 or 1}, (2, 3) team colours in LAB). frames: {frame index: image}."""
    colors = {}
    for tid, fs in sample_frames(tracks, per_track).items():
        rows = tracks[(tracks.track_id == tid) & tracks.frame.isin(fs)]
        cs = [jersey_color(c) for _, r in rows.iterrows() if r.frame in frames
              and (c := torso_crop(frames[r.frame], (r.x1, r.y1, r.x2, r.y2))) is not None]
        if cs:
            colors[tid] = np.median(np.stack(cs), axis=0)
    if len(colors) < 2:
        return {tid: 0 for tid in colors}, None
    ids = list(colors)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.stack([colors[t] for t in ids]))
    return {t: int(l) for t, l in zip(ids, km.labels_)}, km.cluster_centers_


def team_bgr(team_colors: np.ndarray | None) -> list[tuple[int, int, int]]:
    """Display colours (BGR) for the two teams: their jersey colour, else white and blue."""
    if team_colors is None:
        return [(235, 235, 235), (200, 60, 30)]
    lab = np.clip(team_colors, 0, 255).astype(np.uint8).reshape(2, 1, 3)
    return [tuple(int(c) for c in px[0]) for px in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)]
