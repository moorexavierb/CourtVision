"""The ball: one clean image track, who holds it, and where it is on the court.

Image track: per frame, the ball candidate that best continues the track (confidence weighted by
distance from the last position, not just the top box, which is often a head or a light);
impossible jumps removed; short gaps interpolated, long gaps left empty.

Holder: a player holds the ball when its centre is near one of their hand points (box sides at
chest and waist height, top of the box), measured in box heights so it works at any zoom, and it
moves with them (a pass flying past a player in the picture is not a catch).

Court position. A floor homography is only right for points on the floor, and the ball rarely is,
so its court x, y (the point on the floor under it) comes from what it is doing:

* held: the holder's position (also through a dribble: free frames between two frames held by
  the same player stay with that player).
* flight: between a release and the next catch, a ball in the air keeps a constant horizontal
  velocity, so the point under it moves in a straight line at constant speed. Linear
  interpolation in time between the two holders is the exact ground track of the arc, with no
  need to know its height. A shot goes release -> rim (the frame the ball reaches the hoop box)
  -> rebounder.
* floor: a loose ball with nothing to aim at (bouncing, rolling): the bottom of the ball box is
  close to the floor, so it is projected directly.

Table, one row per frame:

    frame | u v (ball centre, pixels) | detected | holder | x_m y_m | source (held, flight, floor)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .court import CourtSpec, apply_h

NO_HOLDER = -1


# --------------------------------------------------------------------------- image track

def pixel_track(dets: pd.DataFrame, n_frames: int, max_speed_px: float = 90.0, max_gap: int = 20) -> np.ndarray:
    """(n_frames, 4) ball box per frame (NaN where unknown)."""
    balls = dets[dets.role == "ball"]
    cands = {f: g[["x1", "y1", "x2", "y2", "conf"]].to_numpy() for f, g in balls.groupby("frame")}
    out = np.full((n_frames, 4), np.nan)
    last, since = None, 0
    for f in range(n_frames):  # 1. continuity-aware selection
        c = cands.get(f)
        if c is None or not len(c):
            since += 1
            continue
        centres = (c[:, :2] + c[:, 2:4]) / 2
        if last is None:
            k = int(c[:, 4].argmax())
        else:
            reach = 1.5 * max_speed_px * (since + 1)
            score = c[:, 4] * np.clip(1 - np.linalg.norm(centres - last, axis=1) / reach, 0, None)
            k = int(score.argmax()) if score.max() > 0 else (int(c[:, 4].argmax()) if since > 30 else -1)
        if k < 0:
            since += 1
            continue
        out[f], last, since = c[k, :4], centres[k], 0
    last_f = None
    for f in range(n_frames):  # 2. drop impossible jumps
        if np.isnan(out[f, 0]):
            continue
        if last_f is not None:
            d = np.linalg.norm((out[f, :2] + out[f, 2:]) / 2 - (out[last_f, :2] + out[last_f, 2:]) / 2)
            if d > max_speed_px * (f - last_f):
                out[f] = np.nan
                continue
        last_f = f
    df = pd.DataFrame(out)  # 3. fill gaps up to max_gap frames, never longer ones
    seen = df[0].notna().to_numpy()
    filled = df.interpolate(limit_area="inside")
    gap_id = np.cumsum(seen)
    gap_len = pd.Series(~seen).groupby(gap_id).transform("sum").to_numpy()
    filled.loc[(~seen) & (gap_len > max_gap)] = np.nan
    return filled.to_numpy()


# --------------------------------------------------------------------------- holder

def hand_points(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, cx = y2 - y1, (x1 + x2) / 2
    return np.array([(x1, y1 + .3 * h), (x2, y1 + .3 * h), (x1, y1 + .55 * h), (x2, y1 + .55 * h),
                     (cx, y1), (cx, y1 + .55 * h)])


def _box_velocities(tracks: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """{(frame, track_id): box-centre velocity in pixels per frame}."""
    out = {}
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        c = np.column_stack([(g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2])
        v = np.gradient(c, g.frame.to_numpy(), axis=0) if len(g) > 1 else np.zeros((1, 2))
        out.update({(int(f), int(tid)): vel for f, vel in zip(g.frame, v)})
    return out


def holders(ball_px: np.ndarray, tracks: pd.DataFrame, remap: dict[int, int], fps: float = 30.0,
            hold_distance: float = 0.35, max_rel_speed: float = 1.5, max_hidden: int = 15) -> np.ndarray:
    """Player id holding the ball per frame (NO_HOLDER if nobody).

    A player holds the ball when it is at their hands AND moves with them: its speed relative to
    the player's box is under max_rel_speed box heights per second (about 3 m/s). Without the
    second test, a pass that flies in front of or behind another player in the picture would be
    read as that player catching it. tracks: the tracks table; remap: positions.relink's output,
    so holders use the same ids as the positions table.
    """
    n = len(ball_px)
    out = np.full(n, NO_HOLDER)
    players = tracks[tracks.role == "player"]
    by_frame = {f: g for f, g in players.groupby("frame")}
    vel = _box_velocities(players)
    centre = (ball_px[:, :2] + ball_px[:, 2:]) / 2
    ball_v = np.gradient(centre, axis=0) if n > 1 else np.zeros((n, 2))
    for f in range(n):
        g = by_frame.get(f)
        if g is None or np.isnan(centre[f, 0]):
            continue
        best, best_d = NO_HOLDER, np.inf
        for r in g.itertuples():
            h = max(r.y2 - r.y1, 1.0)
            d = np.min(np.linalg.norm(hand_points((r.x1, r.y1, r.x2, r.y2)) - centre[f], axis=1)) / h
            if r.x1 <= centre[f, 0] <= r.x2 and r.y1 <= centre[f, 1] <= r.y2:
                d *= 0.5  # inside the box is strong evidence
            rel = np.linalg.norm(ball_v[f] - vel.get((f, int(r.track_id)), 0)) * fps / h
            if d <= hold_distance and not rel > max_rel_speed and d < best_d:
                best, best_d = remap.get(int(r.track_id), int(r.track_id)), d
        out[f] = best
    # Ball not seen (hidden in the hands) between two frames held by the same player: still held.
    f = 0
    while f < n:
        if not np.isnan(centre[f, 0]):
            f += 1
            continue
        s = f
        while f < n and np.isnan(centre[f, 0]):
            f += 1
        if 0 < s and f < n and f - s <= max_hidden and out[s - 1] != NO_HOLDER and out[s - 1] == out[f]:
            out[s:f] = out[s - 1]
    return out


# --------------------------------------------------------------------------- court position

def near_hoop(ball_px: np.ndarray, hoop_boxes: np.ndarray) -> np.ndarray:
    """Per frame: is the ball centre at the rim (inside the hoop box, widened by half its width)?"""
    c = (ball_px[:, :2] + ball_px[:, 2:]) / 2
    w = hoop_boxes[:, 2] - hoop_boxes[:, 0]
    with np.errstate(invalid="ignore"):
        return ((c[:, 0] >= hoop_boxes[:, 0] - w / 2) & (c[:, 0] <= hoop_boxes[:, 2] + w / 2)
                & (c[:, 1] >= hoop_boxes[:, 1] - w) & (c[:, 1] <= hoop_boxes[:, 3]))


def hoop_boxes(dets: pd.DataFrame, n_frames: int) -> np.ndarray:
    """(n_frames, 4) most confident hoop box per frame (NaN where none)."""
    out = np.full((n_frames, 4), np.nan)
    hoops = dets[dets.role == "hoop"].sort_values("conf")
    for r in hoops.itertuples():  # sorted by confidence, so the best box is written last
        out[r.frame] = (r.x1, r.y1, r.x2, r.y2)
    return out


def court_track(ball_px: np.ndarray, holder: np.ndarray, positions: pd.DataFrame,
                homographies: list, court: CourtSpec, at_rim: Optional[np.ndarray] = None,
                fps: float = 30.0, max_flight_s: float = 3.0) -> tuple[np.ndarray, list[Optional[str]]]:
    """((n, 2) ball court x, y per frame, NaN where unknown; source per frame)."""
    n = len(holder)
    pos = {(int(r.frame), int(r.player_id)): (r.x_m, r.y_m) for r in positions.itertuples()}
    xy = np.full((n, 2), np.nan)
    src: list[Optional[str]] = [None] * n
    for f in range(n):
        p = pos.get((f, int(holder[f])))
        if holder[f] != NO_HOLDER and p is not None:
            xy[f], src[f] = p, "held"
    max_flight = int(round(max_flight_s * fps))
    f = 0
    while f < n:
        if src[f] == "held":
            f += 1
            continue
        start = f
        while f < n and src[f] != "held":
            f += 1
        end = f  # free frames are start..end-1; end (if < n) is the catch
        if 0 < start and end < n and holder[start - 1] == holder[end] and end - start <= max_flight:
            # A dribble (or a bobble): the same player before and after, so the ball stays with them.
            for g in range(start, end):
                p = pos.get((g, int(holder[end])))
                if p is not None:
                    xy[g], src[g] = p, "held"
            continue
        way = []  # (frame, x, y) the ground track passes through
        if start > 0:
            way.append((start - 1, *xy[start - 1]))
        if at_rim is not None:
            rim = [g for g in range(start, end) if at_rim[g]]
            ref = xy[start - 1] if start > 0 else (xy[end] if end < n else None)
            if rim and ref is not None:
                way.append((rim[0], *court.nearest_hoop(*ref)))
        if end < n:
            way.append((end, *xy[end]))
        for (fa, xa, ya), (fb, xb, yb) in zip(way, way[1:]):
            if fb - fa > max_flight:
                continue
            for g in range(max(fa, start), min(fb, end)):
                t = (g - fa) / (fb - fa)
                xy[g], src[g] = (xa + t * (xb - xa), ya + t * (yb - ya)), "flight"
        for g in range(start, end):  # anything left: a loose ball on the floor
            H = homographies[g] if g < len(homographies) else None
            if src[g] is None and H is not None and not np.isnan(ball_px[g, 0]):
                x, y = apply_h(H, [((ball_px[g, 0] + ball_px[g, 2]) / 2, ball_px[g, 3])])[0]
                if court.in_bounds(x, y):
                    xy[g], src[g] = (x, y), "floor"
    return xy, src


def ball_table(dets: pd.DataFrame, tracks: pd.DataFrame, remap: dict[int, int], positions: pd.DataFrame,
               homographies: list, court: CourtSpec, fps: float, cfg) -> pd.DataFrame:
    """All of the above as one per-frame table."""
    n = len(homographies)
    px = pixel_track(dets, n, cfg.max_speed_px, cfg.max_gap_frames)
    who = holders(px, tracks, remap, fps, cfg.hold_distance, cfg.max_rel_speed)
    rim = near_hoop(px, hoop_boxes(dets, n))
    xy, src = court_track(px, who, positions, homographies, court, rim, fps, cfg.max_flight_s)
    detected = np.zeros(n, bool)
    detected[dets[dets.role == "ball"].frame.unique()] = True
    return pd.DataFrame({"frame": np.arange(n), "u": (px[:, 0] + px[:, 2]) / 2, "v": (px[:, 1] + px[:, 3]) / 2,
                         "detected": detected, "holder": who, "x_m": xy[:, 0], "y_m": xy[:, 1], "source": src})
