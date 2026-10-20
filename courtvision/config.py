"""Typed configuration loaded from YAML with dataclass defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelsConfig:
    detector: str = "models/player_detector.pt"
    ball_detector: str | None = None
    court_keypoints: str | None = "models/court_keypoints.pt"
    device: str | None = None


@dataclass
class DetectionConfig:
    imgsz: int = 1280
    conf: float = 0.10
    ball_conf: float = 0.05
    batch_size: int = 16
    track_buffer: int = 60  # frames a lost player track is kept alive (occlusions in the paint)
    class_names: dict[str, list[str]] = field(default_factory=lambda: {
        "player": ["Player"],
        "referee": ["Ref"],
        "ball": ["Ball"],
        "hoop": ["Hoop"],
    })


@dataclass
class TeamsConfig:
    method: str = "color"
    team_names: list[str] = field(default_factory=lambda: ["Team A", "Team B"])
    clip_prompts: list[str] | None = None
    samples_per_track: int = 12


@dataclass
class BallConfig:
    max_speed_px_per_frame: float = 90.0
    max_gap_frames: int = 20


@dataclass
class PossessionConfig:
    max_distance_ratio: float = 0.35
    min_frames: int = 5


@dataclass
class EventsConfig:
    interception_max_gap_frames: int = 45
    shot_window_frames: int = 45


@dataclass
class CourtConfig:
    standard: str = "nba"
    keypoint_conf: float = 0.5
    homography_hold_frames: int = 15
    homography_smoothing: float = 0.6   # weight on the camera-tracked mapping when a keypoint fit agrees
    camera_motion: bool = True          # follow pans/zooms with optical flow between keypoint fits
    max_flow_frames: int = 150          # longest stretch carried on camera motion alone
    reject_threshold_m: float = 1.5     # keypoint fits further than this from the tracked mapping are dropped


@dataclass
class TrackingConfig:
    camera_compensation: bool = True    # track in camera-stabilised coordinates, so pans don't break tracks
    reset_on_cut: bool = True           # new track ids after a camera cut (identities re-joined in court space)
    relink: bool = True                 # join player tracks the tracker split in two
    relink_max_gap_s: float = 1.0
    relink_max_speed_mps: float = 6.0
    identities: bool = True             # at most max_per_team ids per team: re-join players who left the frame
    max_per_team: int = 5
    identity_max_gap_s: float = 10.0
    identity_max_speed_mps: float = 8.0
    smooth: bool = True                 # de-spike, gap-fill and smooth court positions
    max_fill_s: float = 0.5
    smooth_window_s: float = 0.3


@dataclass
class MetricsConfig:
    speed_window_frames: int = 5
    max_speed_mps: float = 11.0


@dataclass
class PoseConfig:
    enabled: bool = False          # adds a GPU pass; ~10 ms per player-frame to lift, so best on clips
    model: str = "yolo11m-pose.pt"  # any ultralytics COCO-17 pose model; downloaded on first use
    imgsz: int = 1280
    conf: float = 0.25
    batch_size: int = 16
    match_iou: float = 0.4         # pose box to player track box overlap needed to attach a pose
    min_keypoint_conf: float = 0.3
    player_height_m: float = 2.0   # bone-length prior; NBA average is about 1.98 m
    smooth_window: int = 5


@dataclass
class OutputConfig:
    dir: str = "output"
    cache_dir: str | None = "stubs"
    write_video: bool = True
    write_3d_html: bool = True
    viewer: str = "bodies"              # bodies (three.js animated players) | plotly (dots)
    character_glb: str | None = None    # rigged Mixamo-style .glb used for every player in the bodies viewer
    write_diagnostics: bool = True      # diagnostics.json / _frames.csv / .png: which stage is weak


@dataclass
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    teams: TeamsConfig = field(default_factory=TeamsConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        cfg = cls()
        if path is not None:
            with open(path) as f:
                _merge(cfg, yaml.safe_load(f) or {})
        _merge(cfg, overrides)
        return cfg


def _merge(obj: Any, data: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)
