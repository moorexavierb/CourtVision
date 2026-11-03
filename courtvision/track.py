"""Give every player and referee box a track id that follows them through camera pans.

ByteTrack predicts each track with a constant-velocity filter and matches by box overlap. In
screen pixels a pan makes every player appear to sprint sideways at the camera's speed, the
predictions miss and tracks break. So each box is first mapped into the coordinates of the first
frame of the current shot through the accumulated camera motion (camera.estimate_motion), and the
tracker only sees the players' own motion. At a cut the tracker restarts with fresh ids (an id is
never handed to whoever stands on the same screen spot in the next shot); positions.relink joins
ids across the cut on the court.

Players and referees share one tracker, and each track's role is a confidence-weighted vote over
its life, so a player the detector calls a referee for a few frames keeps his track.

Output table, one row per tracked box:

    frame | track_id | role (player, referee) | x1 y1 x2 y2 | conf
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["frame", "track_id", "role", "x1", "y1", "x2", "y2", "conf"]


def warp_boxes(boxes: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Axis-aligned bounds of (N, 4) boxes after the homography C."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                        np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)
    pts = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], 2) @ C.T
    pts = pts[..., :2] / pts[..., 2:3]
    return np.concatenate([pts.min(1), pts.max(1)], 1)


def track_people(dets: pd.DataFrame, n_frames: int, fps: float, motion: Optional[list] = None,
                 cuts: Optional[list[int]] = None, track_buffer: int = 60) -> pd.DataFrame:
    """Track the player and referee rows of a detections table (detect.detect_objects)."""
    import supervision as sv

    def new_tracker():
        return sv.ByteTrack(lost_track_buffer=track_buffer, frame_rate=int(round(fps)))

    people = dets[dets.role.isin(["player", "referee"])]
    by_frame = {f: g for f, g in people.groupby("frame")}
    cuts = set(cuts or [])
    tracker, C, offset, max_id = new_tracker(), np.eye(3), 0, 0
    rows = []
    for t in range(n_frames):
        if t in cuts:  # new shot: fresh tracker, fresh ids, new reference frame
            tracker, C, offset = new_tracker(), np.eye(3), max_id
        elif motion is not None and t < len(motion) and motion[t] is not None:
            C = C @ np.linalg.inv(motion[t])
            C /= C[2, 2]
        g = by_frame.get(t)
        if g is None or g.empty:
            tracker.update_with_detections(sv.Detections.empty())
            continue
        xyxy = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        det = sv.Detections(xyxy=warp_boxes(xyxy, C), confidence=g.conf.to_numpy(float),
                            class_id=(g.role == "referee").to_numpy().astype(int),
                            data={"idx": np.arange(len(g))})
        out = tracker.update_with_detections(det)
        for tid, i in zip(out.tracker_id, out.data.get("idx", [])):
            gid = int(tid) + offset
            max_id = max(max_id, gid)
            r = g.iloc[int(i)]
            rows.append((t, gid, r.role, r.x1, r.y1, r.x2, r.y2, r.conf))
    tracks = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    return vote_roles(tracks)


def vote_roles(tracks: pd.DataFrame) -> pd.DataFrame:
    """One role per track: the confidence-weighted majority of its boxes."""
    if tracks.empty:
        return tracks
    score = tracks.assign(w=tracks.conf * np.where(tracks.role == "referee", 1, -1)).groupby("track_id").w.sum()
    role = np.where(score > 0, "referee", "player")
    return tracks.assign(role=tracks.track_id.map(dict(zip(score.index, role))))
