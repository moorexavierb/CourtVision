"""Player positions on the court, in metres, per video frame.

1. ``project``: the foot point of each player box (bottom centre) through that frame's
   homography. Boxes that land off the court (bench, coaches, fans) are dropped.
2. ``relink``: a track that ends is joined to one that starts soon after, on the same team,
   never on screen together, and reachable at running speed. This mends breaks from occlusions
   in the paint, missed detections and camera cuts (the tracker restarts ids at a cut).
3. ``smooth``: per player, a 5-frame median removes foot-point spikes (a box cut off by another
   player), short gaps are interpolated, a moving average takes out jitter, and speed comes from
   the smoothed path.

Table, one row per player per frame:

    frame | player_id | team | x_m y_m (smoothed) | raw_x_m raw_y_m | speed_mps | observed
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .court import CourtSpec, apply_h


def project(tracks: pd.DataFrame, homographies: list[Optional[np.ndarray]], court: CourtSpec) -> pd.DataFrame:
    """Player rows of the tracks table with raw court positions (x_m, y_m) added."""
    players = tracks[tracks.role == "player"]
    parts = []
    for f, g in players.groupby("frame"):
        H = homographies[f] if f < len(homographies) else None
        if H is None:
            continue
        feet = np.column_stack([(g.x1 + g.x2) / 2, g.y2])
        xy = apply_h(H, feet)
        g = g.assign(x_m=xy[:, 0], y_m=xy[:, 1])
        parts.append(g[[court.in_bounds(x, y) for x, y in xy]])
    cols = list(tracks.columns) + ["x_m", "y_m"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def relink(pos: pd.DataFrame, team_of: dict[int, int], fps: float, max_gap_s: float = 1.0,
           max_speed_mps: float = 7.0, slack_m: float = 1.0) -> dict[int, int]:
    """{track_id: earlier track_id it continues}. Cheapest links first; each track gets at most one
    predecessor and one successor. Chains resolve to the first track of the chain."""
    if pos.empty:
        return {}
    ends = {}
    for tid, g in pos.sort_values("frame").groupby("track_id"):
        xy = g[["x_m", "y_m"]].to_numpy()
        ends[tid] = (g.frame.iloc[0], g.frame.iloc[-1], np.median(xy[:5], 0), np.median(xy[-5:], 0))
    cands = []
    for a, (_, a_last, _, a_end) in ends.items():
        for b, (b_first, _, b_start, _) in ends.items():
            if a == b or team_of.get(a, -1) != team_of.get(b, -1):
                continue
            gap = b_first - a_last
            if gap <= 0 or gap > max_gap_s * fps:
                continue
            reach = slack_m + max_speed_mps * gap / fps
            dist = float(np.hypot(*(b_start - a_end)))
            if dist <= reach:
                cands.append((dist / reach + gap / (max_gap_s * fps), a, b))
    nxt, prv = {}, {}
    for _, a, b in sorted(cands):
        if a not in nxt and b not in prv:
            nxt[a], prv[b] = b, a
    remap = {}
    for tid in prv:
        root = tid
        while root in prv:
            root = prv[root]
        remap[int(tid)] = int(root)
    return remap


def smooth(pos: pd.DataFrame, fps: float, window_s: float = 0.3, max_fill_s: float = 0.5) -> pd.DataFrame:
    """Per player_id: de-spike, fill short gaps, smooth, and add speed (m/s)."""
    limit = max(1, int(round(max_fill_s * fps)))
    win = max(3, int(round(window_s * fps)) | 1)
    parts = []
    for pid, g in pos.groupby("player_id"):
        g = g.groupby("frame")[["x_m", "y_m"]].mean()  # a relinked id can have two boxes at a seam
        frames = np.arange(g.index.min(), g.index.max() + 1)
        raw = g.reindex(frames)
        observed = raw.x_m.notna().to_numpy()
        med = raw.rolling(5, center=True, min_periods=1).median().where(raw.notna())
        filled = med.interpolate(limit_area="inside")
        gap_id = np.cumsum(observed)                       # frames since the last observation
        gap_len = pd.Series(~observed).groupby(gap_id).transform("sum").to_numpy()
        filled.loc[(~observed) & (gap_len > limit)] = np.nan   # long gaps stay empty
        sm = filled.rolling(win, center=True, min_periods=1).mean().where(filled.notna())
        vx = np.gradient(sm.x_m.to_numpy()) * fps if len(sm) > 1 else np.zeros(len(sm))
        vy = np.gradient(sm.y_m.to_numpy()) * fps if len(sm) > 1 else np.zeros(len(sm))
        out = pd.DataFrame({"frame": frames, "player_id": pid, "x_m": sm.x_m.to_numpy(), "y_m": sm.y_m.to_numpy(),
                            "raw_x_m": raw.x_m.to_numpy(), "raw_y_m": raw.y_m.to_numpy(),
                            "speed_mps": np.hypot(vx, vy), "observed": observed})
        parts.append(out[out.x_m.notna()])
    cols = ["frame", "player_id", "x_m", "y_m", "raw_x_m", "raw_y_m", "speed_mps", "observed"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def player_positions(tracks: pd.DataFrame, homographies: list, court: CourtSpec, team_of: dict[int, int],
                     fps: float, cfg) -> tuple[pd.DataFrame, dict[int, int]]:
    """project -> relink -> smooth. Returns (positions table with team, track remap)."""
    pos = project(tracks, homographies, court)
    remap = relink(pos, team_of, fps, cfg.relink_max_gap_s, cfg.relink_max_speed_mps)
    pos = pos.assign(player_id=pos.track_id.map(lambda t: remap.get(t, t)) if len(pos) else [])
    out = smooth(pos, fps, cfg.smooth_window_s, cfg.max_fill_s)
    out["team"] = out.player_id.map(team_of)
    return out, remap
