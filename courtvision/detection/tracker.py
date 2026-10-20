"""Single-pass detection of players, referees, ball and hoop.

One YOLO pass per frame produces every class. This stage only detects: the
expensive model pass is cached on its own, and tracking (``detection.tracking``)
runs afterwards on the cached boxes, so it can use the camera motion and be
re-tuned without re-running the model. Players and referees are kept together
as "people" with their class, because the detector often calls a player a
referee for a few frames and a per-class tracker would cut his track there.
The ball and hoop are not tracked: a ball is a single object whose identity is
known, and its detections are filtered later by ``courtvision.ball``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from ..config import DetectionConfig
from ..types import RawDetections

PLAYER, REFEREE = 0, 1


class ObjectDetector:
    def __init__(self, model_path: str, cfg: DetectionConfig, device: str | None = None,
                 ball_model_path: str | None = None):
        from ultralytics import YOLO
        import supervision as sv

        self.cfg = cfg
        self.device = device
        self.model = YOLO(model_path)
        self.ball_model = YOLO(ball_model_path) if ball_model_path else None
        self._sv = sv
        self.class_ids = self._resolve_classes(self.model.names, cfg.class_names)
        self.ball_class_ids = (self._resolve_classes(self.ball_model.names, cfg.class_names)["ball"]
                               if self.ball_model else self.class_ids["ball"])

    @staticmethod
    def _resolve_classes(names: dict[int, str], wanted: dict[str, list[str]]) -> dict[str, set[int]]:
        lookup = {v.lower(): k for k, v in names.items()}
        out = {}
        for role, labels in wanted.items():
            out[role] = {lookup[l.lower()] for l in labels if l.lower() in lookup}
        return out

    def _predict(self, model, frames: list[np.ndarray], conf: float):
        return model.predict(frames, conf=conf, imgsz=self.cfg.imgsz, device=self.device, verbose=False)

    def run(self, frames: Iterable[np.ndarray], total: int | None = None) -> RawDetections:
        sv = self._sv
        people, balls, hoops = [], [], []
        batch: list[np.ndarray] = []

        def flush():
            min_conf = min(self.cfg.conf, self.cfg.ball_conf)
            results = self._predict(self.model, batch, min_conf)
            ball_results = self._predict(self.ball_model, batch, self.cfg.ball_conf) if self.ball_model else results
            for res, bres in zip(results, ball_results):
                det = sv.Detections.from_ultralytics(res)
                people.append(self._boxes(det, self.class_ids["player"], self.cfg.conf, PLAYER)
                              + self._boxes(det, self.class_ids["referee"], self.cfg.conf, REFEREE))
                bdet = sv.Detections.from_ultralytics(bres)
                balls.append(self._boxes(bdet, self.ball_class_ids, self.cfg.ball_conf))
                hoop = self._boxes(det, self.class_ids["hoop"], self.cfg.conf)
                hoops.append(max(hoop, key=lambda d: d["conf"])["bbox"] if hoop else None)
            batch.clear()

        for frame in tqdm(frames, total=total, desc="detect"):
            batch.append(frame)
            if len(batch) == self.cfg.batch_size:
                flush()
        if batch:
            flush()
        return RawDetections(people=people, ball_candidates=balls, hoops=hoops)

    @staticmethod
    def _boxes(det, class_ids: set[int], conf: float, cls: int | None = None) -> list[dict]:
        mask = np.isin(det.class_id, list(class_ids)) & (det.confidence >= conf)
        out = [{"bbox": b.tolist(), "conf": float(c)} for b, c in zip(det.xyxy[mask], det.confidence[mask])]
        if cls is not None:
            for d in out:
                d["cls"] = cls
        return out
