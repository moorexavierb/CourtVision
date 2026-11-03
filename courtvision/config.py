"""All settings in one place. Defaults live here; configs/default.yaml overrides them."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Models:
    detector: str = "models/player_detector.pt"        # YOLO detect: Player, Ref, Ball, Hoop
    court_keypoints: str = "models/court_keypoints.pt"  # YOLO pose: 18 court landmarks (court.py)
    device: str | None = None                           # None = cuda if available


@dataclass
class Detection:
    imgsz: int = 1280            # broadcast balls are 15-30 px at 1280; smaller loses them
    conf: float = 0.25           # players and referees
    ball_conf: float = 0.10      # the ball is small and blurred, so it gets a lower bar
    batch_size: int = 16
    # Model class names for each role, so a model with other names only needs a config change.
    class_names: dict[str, list[str]] = field(default_factory=lambda: {
        "player": ["Player"], "referee": ["Ref", "Referee"], "ball": ["Ball"], "hoop": ["Hoop"]})


@dataclass
class Court:
    standard: str = "nba"            # nba | fiba
    keypoint_conf: float = 0.5       # keypoints below this are ignored
    keypoint_imgsz: int = 640
    max_flow_frames: int = 150       # longest stretch the mapping is carried on camera motion alone
    smoothing: float = 0.5           # weight on the camera-tracked mapping when a keypoint fit agrees
    reject_m: float = 1.5            # keypoint fits this far from the tracked mapping are rejected


@dataclass
class Tracking:
    track_buffer: int = 60           # frames a lost track survives (screens in the paint)
    relink_max_gap_s: float = 1.0    # join a track that ends to one that starts this soon after
    relink_max_speed_mps: float = 7.0
    smooth_window_s: float = 0.3     # moving average on court positions
    max_fill_s: float = 0.5          # gaps up to this long inside a track are interpolated


@dataclass
class Ball:
    max_speed_px: float = 90.0       # per frame; faster jumps are false detections
    max_gap_frames: int = 20         # image-space gaps up to this are interpolated
    hold_distance: float = 0.35      # ball-to-hand distance, as a fraction of the player's box height
    max_rel_speed: float = 1.5       # a held ball moves with the player: under this many box heights/s
    max_flight_s: float = 3.0        # longest free-ball stretch drawn as a straight ground line


@dataclass
class Output:
    dir: str = "output"
    cache_dir: str | None = "stubs"  # model passes are cached here per video; None disables
    hz: float = 10.0                 # samples per second in tracking.parquet
    team_names: list[str] = field(default_factory=lambda: ["Team A", "Team B"])
    write_video: bool = True         # annotated video with a top-down minimap, for checking by eye


@dataclass
class Config:
    models: Models = field(default_factory=Models)
    detection: Detection = field(default_factory=Detection)
    court: Court = field(default_factory=Court)
    tracking: Tracking = field(default_factory=Tracking)
    ball: Ball = field(default_factory=Ball)
    output: Output = field(default_factory=Output)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Defaults, then the YAML file, then keyword overrides (e.g. output={"hz": 25})."""
        cfg = cls()
        if path is not None:
            _merge(cfg, yaml.safe_load(open(path)) or {})
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
