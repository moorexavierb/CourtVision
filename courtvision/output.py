"""The deliverable: every on-court player and the ball, on a fixed 10 Hz clock.

The models run on every video frame (29.97 or 59.94 fps for NBA broadcasts), because the tracker
and the ball logic need consecutive frames. This module puts the per-frame positions on a uniform
clock (``hz``, 10 by default: one sample every 0.1 s, under a metre of movement for a sprinting
player) by linear interpolation between frames. A sample is left out when the nearest real
observations around it are more than ``max_gap_s`` apart, so gaps are never papered over.

tracking.parquet / tracking.csv, one row per sample per object:

    sample | time_s | frame (nearest video frame) | object (player, ball) | id (-1 for the ball)
    | team | x_m | y_m | speed_mps | has_ball | source (tracked; ball: held, flight, floor)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["sample", "time_s", "frame", "object", "id", "team", "x_m", "y_m", "speed_mps", "has_ball", "source"]


def resample(frames: np.ndarray, values: np.ndarray, fps: float, times: np.ndarray, max_gap_s: float = 0.2) -> np.ndarray:
    """Linear resample of per-frame values (N, k) observed at `frames` onto `times` (s).
    NaN outside the observed span and inside gaps longer than max_gap_s."""
    values = np.asarray(values, float).reshape(len(frames), -1)
    out = np.full((len(times), values.shape[1]), np.nan)
    ok = ~np.isnan(values[:, 0])
    if ok.sum() == 0:
        return out
    t = np.asarray(frames, float)[ok] / fps
    for k in range(values.shape[1]):
        out[:, k] = np.interp(times, t, values[ok, k], left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(t, times), 1, max(len(t) - 1, 1))
    lo, hi = t[np.clip(i - 1, 0, len(t) - 1)], t[np.clip(i, 0, len(t) - 1)]
    on_obs = np.isclose(times, lo) | np.isclose(times, hi)
    out[(hi - lo > max_gap_s) & ~on_obs] = np.nan
    return out


def tracking_table(positions: pd.DataFrame, ball: pd.DataFrame, n_frames: int, fps: float,
                   hz: float = 10.0, max_gap_s: float = 0.2) -> pd.DataFrame:
    """The 10 Hz table from the per-frame positions (positions.py) and ball (ball.py) tables."""
    times = np.arange(0, (n_frames - 1) / fps + 1e-9, 1.0 / hz)
    nearest = np.clip(np.rint(times * fps).astype(int), 0, max(n_frames - 1, 0))
    holder = ball.set_index("frame").holder.reindex(range(n_frames)).to_numpy() if len(ball) else np.full(n_frames, -1)
    parts = []
    for pid, g in positions.groupby("player_id"):
        v = resample(g.frame.to_numpy(), g[["x_m", "y_m", "speed_mps"]].to_numpy(), fps, times, max_gap_s)
        ok = ~np.isnan(v[:, 0])
        if ok.any():
            parts.append(pd.DataFrame({
                "sample": np.flatnonzero(ok), "time_s": times[ok].round(3), "frame": nearest[ok], "object": "player",
                "id": int(pid), "team": g.team.iloc[0], "x_m": v[ok, 0].round(3), "y_m": v[ok, 1].round(3),
                "speed_mps": v[ok, 2].round(2), "has_ball": holder[nearest[ok]] == pid, "source": "tracked"}))
    if len(ball):
        v = resample(ball.frame.to_numpy(), ball[["x_m", "y_m"]].to_numpy(), fps, times, max_gap_s)
        ok = ~np.isnan(v[:, 0])
        src = ball.source.to_numpy()
        if ok.any():
            parts.append(pd.DataFrame({
                "sample": np.flatnonzero(ok), "time_s": times[ok].round(3), "frame": nearest[ok], "object": "ball",
                "id": -1, "team": None, "x_m": v[ok, 0].round(3), "y_m": v[ok, 1].round(3), "speed_mps": np.nan,
                "has_ball": False, "source": [src[f] or "interpolated" for f in nearest[ok]]}))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts, ignore_index=True)[COLUMNS]
    return df.sort_values(["sample", "object", "id"], ascending=[True, False, True]).reset_index(drop=True)


def coverage(table: pd.DataFrame, n_samples: int, expected_players: int = 10) -> dict:
    """How complete the dataset is: players placed per sample, share of samples with all ten, and
    share of samples with the ball (by how it was placed)."""
    if not n_samples:
        return {}
    per = table[table.object == "player"].groupby("sample").size().reindex(range(n_samples), fill_value=0)
    ball = table[table.object == "ball"]
    return {"samples": int(n_samples),
            "players_per_sample": round(float(per.mean()), 2),
            "all_players_share": round(float((per >= expected_players).mean()), 3),
            "ball_share": round(ball["sample"].nunique() / n_samples, 3),
            "ball_sources": {k: round(v / n_samples, 3) for k, v in ball.source.value_counts().items()}}
