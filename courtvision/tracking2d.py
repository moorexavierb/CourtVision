"""2D court tracking: x, y in metres for every on-court player and the ball, on a fixed clock.

This is the main output of the 2D pipeline (``output.mode: 2d``). It needs only the
floor homography, not the full camera, so it works wherever the court is mapped.

Players: the foot point of each box projected through the homography (re-linked,
de-spiked and smoothed upstream in ``tactical.projector``).

Ball: a floor homography is only right for points on the floor, and the ball rarely
is. So the ball's court position (the point on the floor under it) comes from what
the ball is doing:

* **held**: the holder's position, while the ball is at their hands. Possession has
  hysteresis (it stays with the passer a few frames after the release, so the
  touches table does not flicker), so ``contact`` (the raw per-frame holder) decides
  where the hold really ends.
* **flight** (a pass, or a shot until it reaches the rim): a ball in the air moves at
  constant horizontal velocity, so the point under it moves in a straight line at
  constant speed from the release to the catch (or to the rim, for a shot). Linear
  interpolation in time between those two positions is the exact ground track of a
  parabola, without having to solve for the height.
* **floor**: a loose ball with no catch to aim at (bouncing, rolling, out of bounds).
  The bottom of the ball box is close to the floor, so it is projected directly.

Every row carries ``source`` so the low-confidence parts (floor, flight with long
gaps) can be filtered.

The pipeline runs at the video's own frame rate (29.97 or 59.94 for NBA
broadcasts), because the tracker and the possession logic need every frame, and
``resample`` puts the result on a uniform clock, 10 Hz by default (0.1 s per sample:
up to ~0.9 m of movement for a sprinting player, enough for spacing, speed and
distance; set ``output.tracking_fps: 25`` for SportVU-rate data). Each sample is
interpolated from the smoothed per-frame positions, not just the nearest frame.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .court.geometry import CourtSpec
from .court.homography import apply_h
from .possession.acquisition import NO_POSSESSION
from .types import BallTrack, Event

XY = tuple[float, float]


def ball_court_track(ball: BallTrack, possessor: list[int], player_court: dict[int, list[Optional[XY]]],
                     homographies: list[Optional[np.ndarray]], court: CourtSpec,
                     events: Optional[list[Event]] = None, fps: float = 30.0,
                     max_flight_s: float = 3.0, contact: Optional[list[int]] = None) -> tuple[list[Optional[XY]], list[Optional[str]]]:
    """Per-frame ball court position (x, y in metres) and its source (held, flight, floor)."""
    n = len(possessor)
    xy: list[Optional[XY]] = [None] * n
    src: list[Optional[str]] = [None] * n
    for f in range(n):
        pid = possessor[f]
        released = (contact is not None and f < len(ball) and ball[f] is not None and contact[f] != pid)
        if pid != NO_POSSESSION and not released:
            p = player_court.get(pid, [None] * n)[f]
            if p is not None:
                xy[f], src[f] = (float(p[0]), float(p[1])), "held"

    shots = sorted((e for e in (events or []) if e.type == "shot"), key=lambda e: e.frame)
    made = {e.extra.get("release_frame"): e.frame for e in (events or []) if e.type == "made_shot"}
    max_flight = int(round(max_flight_s * fps))

    f = 0
    while f < n:
        if src[f] == "held":
            f += 1
            continue
        start = f
        while f < n and src[f] != "held":
            f += 1
        end = f  # exclusive; frame `end` (if < n) is the catch
        before = xy[start - 1] if start > 0 else None
        after = xy[end] if end < n else None
        # Waypoints (frame, xy) the ball passes over: release, rim for a shot, catch.
        way: list[tuple[int, XY]] = []
        if before is not None:
            way.append((start - 1, before))
        for s in shots:
            if start <= s.frame < end:
                rim_f = made.get(s.extra.get("release_frame"), s.frame)
                rim_f = min(max(rim_f, start), end - 1)
                ref = before or after
                if ref is not None:
                    hx, hy, _ = court.hoops[court.nearest_hoop(*ref)]
                    way.append((rim_f, (float(hx), float(hy))))
        if after is not None:
            way.append((end, after))
        way = sorted(dict(way).items())
        for (fa, pa), (fb, pb) in zip(way, way[1:]):
            if fb - fa > max_flight or fb <= fa:
                continue
            for g in range(max(fa, start), min(fb, end)):
                t = (g - fa) / (fb - fa)
                xy[g] = (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))
                src[g] = "flight"
        for g in range(start, end):
            if src[g] is None and g < len(ball) and ball[g] is not None and homographies[g] is not None:
                b = ball[g]
                x, y = apply_h(homographies[g], np.array([[(b[0] + b[2]) / 2, b[3]]]))[0]
                if court.in_bounds(x, y):
                    xy[g], src[g] = (float(x), float(y)), "floor"
    return xy, src


def resample(times: np.ndarray, values: list[Optional[XY]], out_times: np.ndarray,
             max_gap_s: float = 0.2) -> np.ndarray:
    """Linear resample of a per-frame (x, y) track onto out_times; NaN where the track is
    missing or the nearest observations are more than max_gap_s apart."""
    arr = np.array([v if v is not None else (np.nan, np.nan) for v in values], float).reshape(-1, 2)
    ok = ~np.isnan(arr[:, 0])
    out = np.full((len(out_times), 2), np.nan)
    if ok.sum() == 0:
        return out
    t_ok = times[ok]
    for k in range(2):
        out[:, k] = np.interp(out_times, t_ok, arr[ok, k], left=np.nan, right=np.nan)
    # Blank samples whose bracketing observations are too far apart.
    idx = np.searchsorted(t_ok, out_times)
    lo = t_ok[np.clip(idx - 1, 0, len(t_ok) - 1)]
    hi = t_ok[np.clip(idx, 0, len(t_ok) - 1)]
    exact = np.isclose(hi, out_times) | np.isclose(lo, out_times)
    out[(hi - lo > max_gap_s) & ~exact] = np.nan
    return out


def tracking_2d_table(result, ball_xy: list[Optional[XY]], ball_src: list[Optional[str]],
                      out_fps: Optional[float] = 10.0, max_gap_s: float = 0.2) -> pd.DataFrame:
    """Long table: one row per (sample, object). Columns: sample, time_s, frame (nearest source
    frame), object (player | ball), id (-1 for the ball), team, x_m, y_m, speed_kmh, has_ball,
    source (tracked for players; held / flight / floor for the ball).

    out_fps=None keeps the video's own frames.
    """
    n = len(result.possessor)
    fps = result.fps
    times = np.arange(n) / fps
    if out_fps:
        out_times = np.arange(0, (n - 1) / fps + 1e-9, 1.0 / out_fps)
    else:
        out_times = times
    nearest = np.clip(np.rint(out_times * fps).astype(int), 0, max(n - 1, 0))
    cols = ["sample", "time_s", "frame", "object", "id", "team", "x_m", "y_m", "speed_kmh", "has_ball", "source"]
    parts = []
    for tid, track in result.court_positions.items():
        xy = resample(times, track, out_times, max_gap_s)
        ok = ~np.isnan(xy[:, 0])
        if not ok.any():
            continue
        spd = np.array([v if v is not None else np.nan
                        for v in result.speed_kmh.get(tid, [None] * n)], float)
        spd_s = np.interp(out_times, times, spd) if (~np.isnan(spd)).any() else np.full(len(out_times), np.nan)
        fr = nearest[ok]
        parts.append(pd.DataFrame({
            "sample": np.flatnonzero(ok), "time_s": out_times[ok].round(3), "frame": fr, "object": "player",
            "id": tid, "team": result.team_of.get(tid), "x_m": xy[ok, 0].round(3), "y_m": xy[ok, 1].round(3),
            "speed_kmh": np.round(spd_s[ok], 2), "has_ball": np.array(result.possessor)[fr] == tid,
            "source": "tracked"}))
    bxy = resample(times, ball_xy, out_times, max_gap_s)
    ok = ~np.isnan(bxy[:, 0])
    if ok.any():
        fr = nearest[ok]
        parts.append(pd.DataFrame({
            "sample": np.flatnonzero(ok), "time_s": out_times[ok].round(3), "frame": fr, "object": "ball",
            "id": -1, "team": None, "x_m": bxy[ok, 0].round(3), "y_m": bxy[ok, 1].round(3),
            "speed_kmh": np.nan, "has_ball": False,
            "source": [ball_src[f] or "interpolated" for f in fr]}))
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, ignore_index=True)[cols]
    return df.sort_values(["sample", "object", "id"], ascending=[True, False, True]).reset_index(drop=True)


def coverage(table: pd.DataFrame, expected_players: int = 10) -> dict:
    """How complete the 2D output is: share of samples with the ball, and players per sample."""
    if table.empty:
        return {"samples": 0, "players_per_sample": 0.0, "full_court_share": 0.0, "ball_share": 0.0}
    samples = table["sample"].max() + 1
    per = table[table["object"] == "player"].groupby("sample").size().reindex(range(samples), fill_value=0)
    ball = table[table["object"] == "ball"]["sample"].nunique()
    return {"samples": int(samples), "players_per_sample": round(float(per.mean()), 2),
            "full_court_share": round(float((per >= expected_players).mean()), 3),
            "ball_share": round(ball / samples, 3),
            "ball_sources": table[table["object"] == "ball"]["source"].value_counts().to_dict()}
