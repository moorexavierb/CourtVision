"""Model passes over a video: object detections and court keypoints. The only GPU code.

Detections come back as one table, one row per box, which is easy to filter and inspect:

    frame | role (player, referee, ball, hoop) | x1 y1 x2 y2 (pixels) | conf
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import Court, Detection

DET_COLUMNS = ["frame", "role", "x1", "y1", "x2", "y2", "conf"]


def _batches(frames: Iterable[np.ndarray], size: int):
    batch = []
    for f in frames:
        batch.append(f)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def role_of_class(names: dict[int, str], class_names: dict[str, list[str]]) -> dict[int, str]:
    """{model class id: role} from the config's role -> class-name lists (case-insensitive)."""
    lookup = {n.lower(): role for role, ns in class_names.items() for n in ns}
    return {i: lookup[n.lower()] for i, n in names.items() if n.lower() in lookup}


def detect_objects(frames: Iterable[np.ndarray], model_path: str, cfg: Detection, device=None,
                   total: int | None = None) -> pd.DataFrame:
    """Every player, referee, ball and hoop box in every frame."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    roles = role_of_class(model.names, cfg.class_names)
    min_conf = min(cfg.conf, cfg.ball_conf)
    rows, f = [], 0
    with tqdm(total=total, desc="detect") as bar:
        for batch in _batches(frames, cfg.batch_size):
            for res in model.predict(batch, imgsz=cfg.imgsz, conf=min_conf, device=device, verbose=False):
                b = res.boxes
                for (x1, y1, x2, y2), c, k in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
                    role = roles.get(int(k))
                    if role is None or c < (cfg.ball_conf if role == "ball" else cfg.conf):
                        continue
                    rows.append((f, role, float(x1), float(y1), float(x2), float(y2), float(c)))
                f += 1
            bar.update(len(batch))
    return pd.DataFrame(rows, columns=DET_COLUMNS)


def detect_keypoints(frames: Iterable[np.ndarray], model_path: str, cfg: Court, device=None,
                     batch_size: int = 16, total: int | None = None) -> np.ndarray:
    """(n_frames, 18, 3) court landmarks per frame: x, y, confidence (0 when not found)."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    out = []
    with tqdm(total=total, desc="court keypoints") as bar:
        for batch in _batches(frames, batch_size):
            for res in model.predict(batch, imgsz=cfg.keypoint_imgsz, conf=0.25, device=device, verbose=False):
                out.append(best_court_instance(res))
            bar.update(len(batch))
    return np.stack(out) if out else np.zeros((0, 18, 3))


def best_court_instance(res) -> np.ndarray:
    """The most confident court instance of one Ultralytics pose result as (18, 3)."""
    kp = res.keypoints
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return np.zeros((18, 3))
    i = int(res.boxes.conf.argmax()) if res.boxes is not None and len(res.boxes) else 0
    xy = kp.xy[i].cpu().numpy()
    conf = kp.conf[i].cpu().numpy() if kp.conf is not None else np.ones(len(xy))
    conf = np.where((xy[:, 0] > 0) & (xy[:, 1] > 0), conf, 0.0)  # Ultralytics puts missing points at 0,0
    return np.column_stack([xy, conf])


def boxes(dets: pd.DataFrame, frame: int, role: str) -> np.ndarray:
    """(N, 4) boxes of one role in one frame (handy when debugging in a notebook)."""
    d = dets[(dets.frame == frame) & (dets.role == role)]
    return d[["x1", "y1", "x2", "y2"]].to_numpy()
