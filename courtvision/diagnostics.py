"""Which stage is weak on this clip? One number per stage, plus a per-frame table to plot.

Read it top to bottom: a stage can only be as good as the ones before it. If the court is mapped
in 60% of frames, at most 60% of samples can have positions, however good the detector is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def frame_table(n: int, calib: list[dict], dets: pd.DataFrame, positions: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
    """One row per video frame: court mapping, detections and placed players, ball."""
    cal = pd.DataFrame(calib) if calib else pd.DataFrame({"source": ["none"] * n, "n_kp": 0})
    count = lambda df: df.groupby("frame").size().reindex(range(n), fill_value=0).to_numpy()
    return pd.DataFrame({
        "frame": np.arange(n),
        "court_source": cal.source.to_numpy()[:n],
        "n_keypoints": cal.n_kp.to_numpy()[:n],
        "fit_rejected": cal.get("kp_rejected", pd.Series(False, index=cal.index)).to_numpy(bool)[:n],
        "fit_mirrored": cal.get("mirrored", pd.Series(False, index=cal.index)).to_numpy(bool)[:n],
        "players_detected": count(dets[dets.role == "player"]),
        "players_placed": count(positions) if len(positions) else np.zeros(n, int),
        "ball_detected": ball.detected.to_numpy() if len(ball) else np.zeros(n, bool),
        "ball_source": ball.source.to_numpy() if len(ball) else [None] * n,
    })


def summarize(frames: pd.DataFrame, tracks: pd.DataFrame, remap: dict, cuts: list[int], fps: float,
              coverage: dict) -> dict:
    mapped = frames.court_source != "none"
    life = tracks[tracks.role == "player"].groupby("track_id").size() / fps if len(tracks) else pd.Series(dtype=float)
    return {
        "frames": len(frames), "fps": round(fps, 3), "duration_s": round(len(frames) / fps, 2),
        "court": {"mapped_share": round(float(mapped.mean()), 3),
                  "from_keypoints_share": round(float(frames.court_source.isin(["keypoints", "fused"]).mean()), 3),
                  "keypoints_per_frame": round(float(frames.n_keypoints.mean()), 2),
                  "rejected_fits": int(frames.fit_rejected.sum()), "mirrored_fits": int(frames.fit_mirrored.sum()),
                  "cuts": len(cuts)},
        "players": {"detected_per_frame": round(float(frames.players_detected.mean()), 2),
                    "placed_per_frame": round(float(frames.players_placed.mean()), 2),
                    "tracks": int(life.size), "relinked": len(remap),
                    "tracks_under_1s_share": round(float((life < 1).mean()), 3) if life.size else 0.0},
        "ball": {"detected_share": round(float(frames.ball_detected.mean()), 3),
                 "placed_share": round(float(frames.ball_source.notna().mean()), 3)},
        "dataset": coverage,
    }


def report(s: dict) -> str:
    c, p, b, d = s["court"], s["players"], s["ball"], s.get("dataset") or {}
    pct = lambda v: f"{v:.0%}"
    return "\n".join([
        f"{s['frames']} frames, {s['duration_s']} s at {s['fps']} fps",
        f"court    mapped {pct(c['mapped_share'])} of frames ({pct(c['from_keypoints_share'])} from keypoints, "
        f"the rest carried by camera motion); {c['keypoints_per_frame']} keypoints per frame; "
        f"{c['rejected_fits']} keypoint fits rejected ({c['mirrored_fits']} mirror images); {c['cuts']} cuts",
        f"players  {p['detected_per_frame']} detected and {p['placed_per_frame']} placed on the court per frame; "
        f"{p['tracks']} tracks ({p['relinked']} re-linked), {pct(p['tracks_under_1s_share'])} shorter than 1 s",
        f"ball     detected in {pct(b['detected_share'])} of frames, placed on the court in {pct(b['placed_share'])}",
        f"dataset  {d.get('players_per_sample', 0)} players per sample, all ten in {pct(d.get('all_players_share', 0))}, "
        f"ball in {pct(d.get('ball_share', 0))}  {d.get('ball_sources', {})}",
    ])
