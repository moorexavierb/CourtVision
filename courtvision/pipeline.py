"""NBA broadcast video in, 10 Hz court tracking of every player and the ball out.

    video ─► detect (players, referees, ball, hoop)        detect.py     GPU, cached
          ─► court keypoints                               detect.py     GPU, cached
          ─► camera motion + cuts                          camera.py     cached
          ─► image -> court mapping for every frame        camera.py
          ─► track ids through pans and cuts               track.py
          ─► teams by jersey colour                        teams.py
          ─► player court positions (relinked, smoothed)   positions.py
          ─► ball: image track, holder, court position     ball.py
          ─► 10 Hz table                                   output.py

``run`` does the model passes (each cached per video, so re-runs only redo what changed) and
``analyze`` does the rest, which needs no GPU and is what the tests exercise. Every intermediate
table is kept on the result, so any stage can be inspected in a notebook.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import ball as ball_mod
from . import camera, diagnostics, output, positions as positions_mod, teams, track
from .config import Config
from .court import get_court
from .video import cached, iter_frames, read_frames_at, video_info

log = logging.getLogger(__name__)


@dataclass
class Result:
    fps: float
    size: tuple[int, int]
    detections: pd.DataFrame          # one row per box (detect.py)
    tracks: pd.DataFrame              # one row per tracked player/referee box (track.py)
    homographies: list                # per frame image -> court 3x3, or None
    calibration: list[dict]           # per frame: keypoints seen, mapping source (camera.py)
    cuts: list[int]
    team_of: dict[int, int]
    remap: dict[int, int]             # track id -> player id after relinking
    positions: pd.DataFrame           # players per frame, metres (positions.py)
    ball: pd.DataFrame                # ball per frame (ball.py)
    tracking: pd.DataFrame            # the 10 Hz dataset (output.py)
    team_colors: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)
    frames: Optional[pd.DataFrame] = None  # per-frame diagnostics table


def run(video: str | Path, cfg: Optional[Config] = None, out_dir: str | Path | None = None,
        write_outputs: bool = True) -> Result:
    """Run the whole pipeline on one video; outputs go to out_dir (default output.dir/<video name>)."""
    cfg = cfg or Config()
    video = Path(video)
    fps, n, size = video_info(video)
    c, cache = cfg, cfg.output.cache_dir

    from .detect import detect_keypoints, detect_objects
    dets = cached(cache, video, "detect", (c.models.detector, c.detection),
                  lambda: detect_objects(iter_frames(video), c.models.detector, c.detection, c.models.device, n))
    kps = cached(cache, video, "keypoints", (c.models.court_keypoints, c.court.keypoint_imgsz),
                 lambda: detect_keypoints(iter_frames(video), c.models.court_keypoints, c.court, c.models.device, total=n))
    n = len(kps)  # frames actually decoded (container frame counts can be off)

    def motion_pass():
        people = dets[dets.role.isin(["player", "referee"])]
        boxes = [[] for _ in range(n)]
        for r in people.itertuples():
            if r.frame < n:
                boxes[r.frame].append((r.x1, r.y1, r.x2, r.y2))
        return camera.estimate_motion(iter_frames(video), boxes, total=n)
    motion, cuts = cached(cache, video, "motion", (c.models.detector, c.detection, "v2"), motion_pass)

    result = analyze(dets, kps[:n], motion[:n], cuts, fps, size, cfg,
                     team_frames=lambda wanted: read_frames_at(video, wanted))
    if write_outputs:
        write(result, video, Path(out_dir or cfg.output.dir) / video.stem, cfg)
    return result


def analyze(dets: pd.DataFrame, keypoints: np.ndarray, motion: list, cuts: list[int], fps: float,
            size: tuple[int, int], cfg: Optional[Config] = None, team_frames=None,
            team_of: Optional[dict[int, int]] = None) -> Result:
    """Everything after the model passes. team_frames(frame list) -> {frame: image} supplies the
    crops for jersey colours; or pass team_of directly."""
    cfg = cfg or Config()
    court = get_court(cfg.court.standard)
    n = len(keypoints)

    H, calib = camera.track_homographies(keypoints, motion, court, cfg.court.keypoint_conf,
                                         cfg.court.max_flow_frames, cfg.court.smoothing, cfg.court.reject_m)
    cuts = camera.confirm_cuts(cuts, H, size)
    tracks = track.track_people(dets, n, fps, motion, cuts, cfg.tracking.track_buffer)

    colors = None
    if team_of is None:
        wanted = sorted({f for fs in teams.sample_frames(tracks).values() for f in fs})
        team_of, colors = teams.assign_teams(team_frames(wanted) if team_frames else {}, tracks)
    pos, remap = positions_mod.player_positions(tracks, H, court, team_of, fps, cfg.tracking)
    ball = ball_mod.ball_table(dets, tracks, remap, pos, H, court, fps, cfg.ball)
    table = output.tracking_table(pos, ball, n, fps, cfg.output.hz)

    n_samples = len(np.arange(0, (n - 1) / fps + 1e-9, 1.0 / cfg.output.hz)) if n else 0
    frames = diagnostics.frame_table(n, calib, dets, pos, ball)
    summary = diagnostics.summarize(frames, tracks, remap, cuts, fps, output.coverage(table, n_samples))
    return Result(fps, size, dets, tracks, H, calib, cuts, team_of, remap, pos, ball, table, colors, summary, frames)


def write(result: Result, video: Path, out_dir: Path, cfg: Config) -> None:
    """tracking.parquet/.csv (the dataset), per-frame tables for debugging, diagnostics, video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = cfg.output.team_names
    t = result.tracking.assign(team=result.tracking.team.map(lambda k: names[int(k)] if k in (0, 1) else None))
    t.to_parquet(out_dir / "tracking.parquet", index=False)
    t.to_csv(out_dir / "tracking.csv", index=False)
    result.positions.to_parquet(out_dir / "players_per_frame.parquet", index=False)
    result.ball.to_parquet(out_dir / "ball_per_frame.parquet", index=False)
    result.frames.to_csv(out_dir / "frames.csv", index=False)
    (out_dir / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2, default=float))
    log.info("\n%s", diagnostics.report(result.diagnostics))
    if cfg.output.write_video:
        from .render import render_video
        render_video(video, out_dir / f"{video.stem}_tracked.mp4", result.tracks, result.remap, result.team_of,
                     result.positions, result.ball, get_court(cfg.court.standard),
                     teams.team_bgr(result.team_colors), result.fps)
    log.info("Wrote %s", out_dir)
