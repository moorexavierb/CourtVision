"""Annotated video for checking the tracking by eye: boxes with ids and team colours, the ball,
and a top-down minimap of the court positions in the corner."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .court import CourtSpec
from .video import iter_frames


def draw_court(court: CourtSpec, px_per_m: float = 12.0, pad: int = 10) -> np.ndarray:
    W, H = int(court.length * px_per_m) + 2 * pad, int(court.width * px_per_m) + 2 * pad
    img = np.full((H, W, 3), (60, 110, 160), np.uint8)
    p = lambda x, y: (int(pad + x * px_per_m), int(pad + y * px_per_m))
    white = (240, 240, 240)
    cv2.rectangle(img, p(0, 0), p(court.length, court.width), white, 1)
    cv2.line(img, p(court.length / 2, 0), p(court.length / 2, court.width), white, 1)
    lo, hi = (court.width - court.lane_width) / 2, (court.width + court.lane_width) / 2
    for x0, x1 in [(0, court.free_throw_dist), (court.length - court.free_throw_dist, court.length)]:
        cv2.rectangle(img, p(x0, lo), p(x1, hi), white, 1)
    for hx, hy in court.hoops:
        cv2.circle(img, p(hx, hy), int(court.three_radius * px_per_m), white, 1)
    return img


def minimap(court: CourtSpec, players: pd.DataFrame, ball_xy, colors, px_per_m: float = 12.0, pad: int = 10):
    img = draw_court(court, px_per_m, pad)
    p = lambda x, y: (int(pad + x * px_per_m), int(pad + y * px_per_m))
    for r in players.itertuples():
        c = colors[int(r.team)] if r.team in (0, 1) else (128, 128, 128)
        cv2.circle(img, p(r.x_m, r.y_m), 5, c, -1)
        cv2.circle(img, p(r.x_m, r.y_m), 5, (0, 0, 0), 1)
    if ball_xy is not None and not np.isnan(ball_xy[0]):
        cv2.circle(img, p(*ball_xy), 4, (0, 140, 255), -1)
    return img


def render_video(video, out_path, tracks: pd.DataFrame, remap: dict, team_of: dict, positions: pd.DataFrame,
                 ball: pd.DataFrame, court: CourtSpec, colors, fps: float) -> None:
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    pos_by_frame = {f: g for f, g in positions.groupby("frame")}
    writer = None
    for f, frame in enumerate(iter_frames(video)):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for r in by_frame.get(f, pd.DataFrame()).itertuples():
            pid = remap.get(r.track_id, r.track_id)
            team = team_of.get(pid, team_of.get(r.track_id, -1))
            c = (0, 220, 255) if r.role == "referee" else (colors[team] if team in (0, 1) else (128, 128, 128))
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 2)
            label = "REF" if r.role == "referee" else str(pid)
            cv2.putText(frame, label, (int(r.x1), int(r.y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if f < len(ball):
            b = ball.iloc[f]
            if not np.isnan(b.u):
                cv2.circle(frame, (int(b.u), int(b.v)), 8, (0, 140, 255), 2)
            mini = minimap(court, pos_by_frame.get(f, positions.iloc[:0]), (b.x_m, b.y_m), colors)
            mh, mw = mini.shape[:2]
            if mh < frame.shape[0] and mw < frame.shape[1]:
                frame[-mh - 10:-10, -mw - 10:-10] = mini
        cv2.putText(frame, str(f), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    if writer is not None:
        writer.release()
