"""Player and referee tracking that follows the broadcast camera.

ByteTrack predicts each track with a constant-velocity Kalman filter and
matches by box overlap. In screen pixels a pan makes every player appear to
sprint sideways at the camera's speed, the predictions miss, and tracks break.
Here every box is first mapped into a stabilised frame (the first frame of the
current shot) through the accumulated camera motion
(``court.camera_motion``), so the tracker only sees the players' own motion.
The output keeps the original screen boxes.

* At a cut (a histogram jump) the tracker is reset and new ids are issued, so
  an id is never handed to whoever stands at the same screen spot in the next
  shot. ``detection.stitch`` joins identities across the cut in court space.
* Frames whose motion could not be estimated (but are not cuts) are treated as
  a still camera rather than a reset.
* Players and referees share one tracker. Each track's class is a
  confidence-weighted vote over its life, so a player the detector briefly
  calls a referee keeps his track.

Everything runs frame by frame (``CameraCompensatedTracker.update``), so it
also works on a live stream; only the class vote needs the finished track, and
a running vote can stand in for it live.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from ..types import PlayerTracks
from .tracker import PLAYER, REFEREE


def warp_boxes(boxes: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Axis-aligned bounds of boxes (N, 4) after the homography C."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                        np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)  # (N, 4, 2)
    pts = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], 2) @ C.T
    pts = pts[..., :2] / pts[..., 2:3]
    return np.concatenate([pts.min(1), pts.max(1)], 1)


class CameraCompensatedTracker:
    def __init__(self, fps: float = 30.0, track_buffer: int = 60, compensate: bool = True,
                 reset_on_cut: bool = True):
        import supervision as sv

        self._sv = sv
        self.fps = fps
        self.track_buffer = track_buffer
        self.compensate = compensate
        self.reset_on_cut = reset_on_cut
        self.offset = 0      # ids of earlier shots, so ids stay unique across resets
        self.max_id = 0
        self.C = np.eye(3)   # current frame pixels -> stabilised (shot start) pixels
        self.tracker = self._new()

    def _new(self):
        return self._sv.ByteTrack(lost_track_buffer=self.track_buffer, frame_rate=int(round(self.fps)))

    def update(self, people: list[dict], motion: Optional[np.ndarray] = None, cut: bool = False) -> list[tuple[int, dict]]:
        """Track one frame. ``motion`` maps the previous frame's pixels to this frame's (None if unknown).
        Returns [(track_id, detection dict)]."""
        if cut and self.reset_on_cut:
            self.tracker = self._new()
            self.offset = self.max_id
            self.C = np.eye(3)
        elif self.compensate and motion is not None:
            self.C = self.C @ np.linalg.inv(motion)
            self.C /= self.C[2, 2]
        if not people:
            self.tracker.update_with_detections(self._sv.Detections.empty())
            return []
        boxes = np.array([p["bbox"] for p in people], float)
        det = self._sv.Detections(xyxy=warp_boxes(boxes, self.C) if self.compensate else boxes,
                                  confidence=np.array([p["conf"] for p in people], float),
                                  class_id=np.array([p.get("cls", PLAYER) for p in people], int),
                                  data={"idx": np.arange(len(people))})
        tracked = self.tracker.update_with_detections(det)
        out = []
        for tid, i in zip(tracked.tracker_id, tracked.data.get("idx", [])):
            gid = int(tid) + self.offset
            self.max_id = max(self.max_id, gid)
            out.append((gid, people[int(i)]))
        return out


def track_people(people: list[list[dict]], motion: Optional[list[Optional[np.ndarray]]] = None,
                 cuts: Optional[set[int]] = None, fps: float = 30.0, track_buffer: int = 60,
                 compensate: bool = True, reset_on_cut: bool = True) -> tuple[PlayerTracks, PlayerTracks]:
    """Track every frame, then split tracks into (players, referees) by their class vote."""
    trk = CameraCompensatedTracker(fps, track_buffer, compensate and motion is not None, reset_on_cut)
    cuts = cuts or set()
    frames: list[dict[int, dict]] = []
    votes: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2))
    for t, ppl in enumerate(people):
        M = motion[t] if motion is not None and t < len(motion) else None
        ft = {}
        for tid, p in trk.update(ppl, M, t in cuts):
            ft[tid] = {"bbox": list(p["bbox"]), "conf": p["conf"]}
            votes[tid][p.get("cls", PLAYER)] += p["conf"]
        frames.append(ft)
    is_ref = {tid: v[REFEREE] > v[PLAYER] for tid, v in votes.items()}
    players = [{t: d for t, d in ft.items() if not is_ref[t]} for ft in frames]
    referees = [{t: d for t, d in ft.items() if is_ref[t]} for ft in frames]
    return players, referees
