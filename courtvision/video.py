"""Video reading/writing and the per-video cache for model passes."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np


def video_info(path: str | Path) -> tuple[float, int, tuple[int, int]]:
    """(fps, frame count, (width, height))."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, n, size


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """BGR frames one at a time, so a full game never sits in memory."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_frames_at(path: str | Path, wanted: list[int]) -> dict[int, np.ndarray]:
    """{frame index: frame} for a sorted list of indices, in one sequential pass."""
    want, out = set(wanted), {}
    for i, frame in enumerate(iter_frames(path)):
        if i in want:
            out[i] = frame
            if len(out) == len(want):
                break
    return out


def save_video(frames, path: str | Path, fps: float) -> None:
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(first)
    for f in frames:
        writer.write(f)
    writer.release()


def cached(cache_dir: str | Path | None, video: str | Path, stage: str, settings: Any,
           compute: Callable[[], Any]) -> Any:
    """Run compute() once per (video, stage, settings) and keep the result as a pickle.

    The key includes the video's name and size and a hash of the settings, so changing a model
    or a threshold recomputes that stage only. Delete the cache folder to force a full rerun.
    """
    if cache_dir is None:
        return compute()
    p = Path(video)
    digest = hashlib.md5(f"{p.name}|{p.stat().st_size}|{stage}|{settings!r}".encode()).hexdigest()[:10]
    path = Path(cache_dir) / f"{p.stem}_{stage}_{digest}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(value, open(path, "wb"))
    return value
