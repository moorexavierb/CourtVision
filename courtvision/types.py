"""Shared data structures.

Per-frame tracks use plain dicts so they pickle cheaply and are easy to inspect
in a notebook:

    PlayerTracks = list[dict[int, dict]]   # frame -> {track_id: {"bbox": [x1,y1,x2,y2], "conf": float}}
    BallTrack    = list[list[float] | None] # frame -> [x1,y1,x2,y2] or None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BBox = list[float]
PlayerTracks = list[dict[int, dict]]
BallTrack = list[Optional[BBox]]


def bbox_center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_foot(b: BBox) -> tuple[float, float]:
    """Bottom-centre of a box: where the player touches the floor plane."""
    return ((b[0] + b[2]) / 2.0, float(b[3]))


def bbox_height(b: BBox) -> float:
    return float(b[3] - b[1])


def bbox_width(b: BBox) -> float:
    return float(b[2] - b[0])


@dataclass
class RawDetections:
    """Output of the model pass, before tracking."""
    people: list[list[dict]]            # frame -> [{"bbox": BBox, "conf": float, "cls": 0 player | 1 referee}]
    ball_candidates: list[list[dict]]   # frame -> [{"bbox": BBox, "conf": float}, ...]
    hoops: list[Optional[BBox]]


@dataclass
class DetectionResult:
    """Detections after tracking: players and referees carry track ids."""
    players: PlayerTracks
    referees: PlayerTracks
    ball_candidates: list[list[dict]]   # frame -> [{"bbox": BBox, "conf": float}, ...]
    hoops: list[Optional[BBox]]


@dataclass
class Possession:
    player_id: int
    team: int
    start_frame: int
    end_frame: int  # inclusive


@dataclass
class Event:
    frame: int
    type: str                     # pass | interception | turnover | shot | made_shot | possession
    player_id: int | None = None
    team: int | None = None
    target_id: int | None = None  # receiver for passes, defender for interceptions
    extra: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    fps: float
    frame_size: tuple[int, int]
    players: PlayerTracks
    referees: PlayerTracks
    ball: BallTrack
    hoops: list[Optional[BBox]]
    team_of: dict[int, int]
    possessor: list[int]
    possessions: list[Possession]
    events: list[Event]
    homographies: list[Optional[np.ndarray]]
    court_positions: dict[int, list[Optional[tuple[float, float]]]]
    speed_kmh: dict[int, list[Optional[float]]]
    distance_m: dict[int, list[Optional[float]]]
    ball_3d: list[Optional[tuple[float, float, float]]]
    # track id -> frame -> (17, 3) COCO-17 skeleton in court metres; empty unless pose is enabled
    pose_3d: dict[int, list[Optional[np.ndarray]]] = field(default_factory=dict)
    # per-frame court calibration info (keypoints seen, fit error, mapping source); see diagnostics.py
    calibration: list[dict] = field(default_factory=list)
    # track id -> id it was merged into by re-linking
    track_remap: dict[int, int] = field(default_factory=dict)
    # identity id -> 0-1 trust in its links (detection.stitch.assign_identities); tracks fitting no identity
    identity_conf: dict[int, float] = field(default_factory=dict)
    identity_extras: list[int] = field(default_factory=list)
    shot_cuts: list[int] = field(default_factory=list)  # frames that start a new camera shot
    diagnostics: Optional[dict] = None
    diagnostics_frames: Optional[object] = None  # pandas DataFrame, one row per frame
