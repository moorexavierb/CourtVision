"""Score a detector the way the tracking dataset uses it: through the pipeline's tracker.

A box only becomes a court position if the tracker keeps it on one player, so the detector's
player threshold is tuned on tracking (IDF1 on labelled SportsMOT sequences) rather than on box
counts. The model runs once per sequence at a low threshold (``detect_sequence``); tracking at any
higher threshold (``track_players``) then needs no GPU, so a sweep costs one model pass.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from .. import camera, track
from ..config import Detection


def detect_sequence(weights: str, seq: Path, det: Detection, min_conf: float = 0.05) -> dict:
    """Detections (down to min_conf), camera motion and cuts for one SportsMOT sequence."""
    from ..detect import detect_objects
    from .sportsmot import frame_paths
    paths = [p for _, p in sorted(frame_paths(seq).items())]
    frames = lambda: (cv2.imread(str(p)) for p in paths)
    dets = detect_objects(frames(), weights, replace(det, conf=min_conf, ball_conf=min_conf), total=len(paths))
    people = dets[dets.role.isin(["player", "referee"]) & (dets.conf >= det.conf)].groupby("frame")
    boxes = [people.get_group(i)[["x1", "y1", "x2", "y2"]].values.tolist() if i in people.groups else []
             for i in range(len(paths))]
    motion, cuts = camera.estimate_motion(frames(), boxes, total=len(paths))
    return {"dets": dets, "motion": motion, "cuts": cuts, "frame_ids": [int(p.stem) for p in paths]}


def track_players(seq_dets: dict, conf: float, fps: float, track_buffer: int) -> dict[int, dict[int, list[float]]]:
    """{frame id: {track id: box}} of the player tracks when people boxes need conf or more."""
    d = seq_dets["dets"]
    d = d[~d.role.isin(["player", "referee"]) | (d.conf >= conf)]
    ids = seq_dets["frame_ids"]
    tracks = track.track_people(d, len(ids), fps, seq_dets["motion"], seq_dets["cuts"], track_buffer)
    players = tracks[tracks.role == "player"]
    return {ids[f]: {int(r.track_id): [r.x1, r.y1, r.x2, r.y2] for r in g.itertuples()}
            for f, g in players.groupby("frame")}


def ultralytics_tracks(weights: str, seq: Path, tracker: str = "botsort.yaml", conf: float = 0.1,
                       imgsz: int = 1280, track_buffer: Optional[int] = None, fps: Optional[float] = None,
                       player_names=("player",)) -> dict[int, dict[int, list[float]]]:
    """{frame id: {track id: box}} from Ultralytics' own trackers on one sequence, players only.

    A reference for the pipeline's tracker: "botsort.yaml" is BoT-SORT (Aharon et al. 2022) with
    sparse optical-flow camera-motion compensation, "bytetrack.yaml" is ByteTrack (Zhang et al.,
    ECCV 2022), both as shipped with Ultralytics (docs.ultralytics.com/modes/track). conf is the
    detector's floor; the tracker's own high/low thresholds then apply. track_buffer and fps, when
    given, override the yaml so all trackers keep lost players for the same time."""
    import tempfile
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils import ROOT
    from .sportsmot import frame_paths, seq_fps
    cfg_path = Path(tracker)
    if not cfg_path.exists():
        cfg_path = Path(ROOT) / "cfg" / "trackers" / tracker
    t_cfg = yaml.safe_load(open(cfg_path))
    if track_buffer is not None:
        # Frames a lost track is kept: supervision's ByteTrack (the pipeline) keeps it
        # track_buffer * fps / 30 frames; Ultralytics runs its trackers at frame_rate=30, i.e. track_buffer frames.
        t_cfg["track_buffer"] = max(1, int(round(track_buffer * (fps or seq_fps(seq)) / 30)))
    tmp = Path(tempfile.mkdtemp()) / Path(tracker).name
    yaml.safe_dump(t_cfg, open(tmp, "w"))
    model = YOLO(weights)
    classes = [i for i, n in model.names.items() if n.lower() in player_names]
    out = {}
    for fid, p in sorted(frame_paths(seq).items()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = model.track(img, persist=True, tracker=str(tmp), conf=conf, imgsz=imgsz, classes=classes, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            out[fid] = {int(i): b.tolist() for i, b in zip(r.boxes.id.cpu().numpy(), r.boxes.xyxy.cpu().numpy())}
    return out


def tracking_sweep(per_seq: dict[str, tuple[dict, dict, float]], grid, track_buffer: int) -> pd.DataFrame:
    """Overall HOTA / MOTA / IDF1 (TrackEval's combination over sequences) at each player threshold.
    per_seq: {name: (seq_dets, ground-truth tracks, fps)}."""
    from .evaluate import mot_metrics, mot_table
    rows = {}
    for c in grid:
        table = mot_table({name: mot_metrics(gt, track_players(sd, float(c), fps, track_buffer))
                           for name, (sd, gt, fps) in per_seq.items()})
        rows[round(float(c), 3)] = table.loc["overall"].to_dict()
    return pd.DataFrame(rows).T.rename_axis("conf")
