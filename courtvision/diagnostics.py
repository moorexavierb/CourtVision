"""Per-stage health report: which part of the pipeline is letting the 3D down.

``diagnose`` looks at every stage's intermediate output and returns

* a summary dict (written to ``diagnostics.json``) with one block per stage
  and a list of ``findings``: plain-language problems ranked by how much they
  hurt the reconstruction, each naming the stage and what would fix it;
* a per-frame table (``diagnostics_frames.csv``) for plotting or digging in.

``plot_diagnostics`` draws the per-frame table as a timeline
(``diagnostics.png``), and ``format_report`` prints the summary.

Thresholds are rules of thumb for NBA broadcast footage, not hard limits.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .court.geometry import CourtSpec
from .court.homography import apply_h
from .possession.acquisition import NO_POSSESSION
from .types import PipelineResult, PlayerTracks, bbox_foot


def _runs(mask: np.ndarray) -> list[int]:
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def _raw_positions(players: PlayerTracks, homographies, court: CourtSpec):
    """Unsmoothed foot positions and how many feet landed off the court."""
    pos: dict[int, dict[int, tuple[float, float]]] = {}
    total = oob = 0
    for f, (ft, H) in enumerate(zip(players, homographies)):
        if H is None or not ft:
            continue
        ids = list(ft)
        for tid, (x, y) in zip(ids, apply_h(H, np.array([bbox_foot(ft[t]["bbox"]) for t in ids]))):
            total += 1
            if court.in_bounds(x, y):
                pos.setdefault(tid, {})[f] = (x, y)
            else:
                oob += 1
    return pos, total, oob


def diagnose(raw_players: PlayerTracks, ball_candidates: list[list[dict]], result: PipelineResult,
             calibration: list[dict], court: CourtSpec, track_colors: Optional[dict[int, np.ndarray]] = None,
             team_colors: Optional[np.ndarray] = None, remap: Optional[dict[int, int]] = None,
             video: Optional[dict] = None) -> tuple[dict, pd.DataFrame]:
    """``video`` (optional): {"width", "height", "fps", "bitrate_mbps"} of the input clip."""
    n, fps = len(result.players), result.fps
    remap = remap or {}
    findings: list[tuple[int, str, str]] = []  # (severity 1-3, stage, text)

    # ------------------------------------------------------------ input video
    ball_px = [max(c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1])
               for cands in ball_candidates for c in cands]
    player_px = [p["bbox"][3] - p["bbox"][1] for ft in raw_players for p in ft.values()]
    video_block = dict(video or {})
    video_block["median_ball_px"] = round(float(np.median(ball_px)), 1) if ball_px else None
    video_block["median_player_px"] = round(float(np.median(player_px)), 1) if player_px else None
    h, vfps, br = video_block.get("height"), video_block.get("fps"), video_block.get("bitrate_mbps")
    if h and h < 720:
        findings.append((2, "input video",
                         f"The clip is {h}p. Below 720p the ball is only a few pixels across and court lines blur, "
                         "which hurts ball and keypoint detection. Use 720p or better (1080p ideal)."))
    if vfps and vfps < 25:
        findings.append((1, "input video",
                         f"The clip is {vfps:.0f} fps. Below 25 fps fast passes and cuts skip frames; use 30 or 60 fps."))
    if br and h and br < (2.5 if h < 1000 else 4.0):
        findings.append((1, "input video",
                         f"The clip is heavily compressed ({br:.1f} Mbit/s for {h}p), which smears the ball and lines "
                         "into blocks. Aim for at least 4 Mbit/s at 720p or 8 Mbit/s at 1080p."))
    if video_block["median_ball_px"] is not None and video_block["median_ball_px"] < 10:
        findings.append((1, "input video",
                         f"Detected balls are about {video_block['median_ball_px']:.0f} px across; under ~10 px "
                         "detection gets unreliable. A higher-resolution source helps most here."))

    # ------------------------------------------------------------ court mapping
    cal = pd.DataFrame(calibration) if calibration else pd.DataFrame(index=range(n))
    mapped = np.array([H is not None for H in result.homographies])
    src = cal.get("source", pd.Series(["keypoints" if m else "none" for m in mapped]))
    kp_fit = src.isin(["keypoints", "fused"]).to_numpy()
    fit_err = cal.get("fit_err_m", pd.Series(np.nan, index=range(n)))
    n_kp = cal.get("n_kp", pd.Series(0, index=range(n)))
    rejected = cal.get("kp_rejected", pd.Series(False, index=range(n))).astype(bool)
    motion_ok = cal.get("motion_ok", pd.Series(np.nan, index=range(n)))
    fail = cal.get("fit", pd.Series("", index=range(n)))
    has_motion = "motion_ok" in cal and cal["motion_ok"].any()
    cuts = int((~motion_ok.astype(bool)).iloc[1:].sum()) if has_motion else None
    unmapped_runs = _runs(~mapped)
    court_block = {
        "mapped_share": float(mapped.mean()) if n else 0.0,
        "keypoint_fit_share": float(kp_fit.mean()) if n else 0.0,
        "flow_share": float((src == "flow").mean()) if n else 0.0,
        "mean_keypoints_seen": float(n_kp.mean()) if n else 0.0,
        "mean_keypoint_conf": float(cal["kp_conf"][cal["kp_conf"] > 0].mean()) if "kp_conf" in cal and (cal["kp_conf"] > 0).any() else 0.0,
        "median_fit_error_m": float(np.nanmedian(fit_err)) if np.isfinite(fit_err).any() else None,
        "p90_fit_error_m": float(np.nanpercentile(fit_err, 90)) if np.isfinite(fit_err).any() else None,
        "rejected_fit_share": float(rejected.mean()) if n else 0.0,
        "fit_failures": {k: int(v) for k, v in Counter(fail[~kp_fit]).items() if k and k != "ok"},
        "camera_cuts": len(result.shot_cuts) if has_motion else None,
        "motion_failures": cuts,  # frames whose camera motion could not be estimated (cuts included)
        "longest_unmapped_s": round(max(unmapped_runs, default=0) / fps, 2),
    }
    if court_block["mapped_share"] < 0.9:
        if court_block["keypoint_fit_share"] < 0.3:
            findings.append((3, "court keypoints",
                             f"The court is mapped on only {court_block['mapped_share']:.0%} of frames, and the keypoint "
                             f"model gives a usable fit on {court_block['keypoint_fit_share']:.0%}. Players can't be placed "
                             "in 3D without it. The model needs NBA broadcast frames in its training set (zoomed-in "
                             "half-court views especially), and more landmarks than the 18 baseline/lane/half-court points."))
        else:
            findings.append((2, "camera tracking",
                             f"The court is mapped on {court_block['mapped_share']:.0%} of frames. The gaps are long "
                             f"(longest {court_block['longest_unmapped_s']} s), usually cuts to close-ups or replays."))
    if court_block["median_fit_error_m"] is not None and court_block["median_fit_error_m"] > 0.4:
        findings.append((2, "court keypoints",
                         f"Keypoint fits are off by {court_block['median_fit_error_m']:.2f} m (median), so players "
                         "shift around the court. Keypoint placement accuracy needs more or better-labelled training data."))
    if court_block["rejected_fit_share"] > 0.1:
        findings.append((2, "court keypoints",
                         f"{court_block['rejected_fit_share']:.0%} of keypoint fits contradicted the camera track and were "
                         "dropped. That usually means landmarks get the wrong label (left vs right baseline, or "
                         "mirror-image lane corners). Check the labels and flip_idx in notebook 02."))

    # ------------------------------------------------------------ players
    raw_pos, total_feet, oob = _raw_positions(raw_players, result.homographies, court)
    jumps = steps = 0
    for track in raw_pos.values():
        fs = sorted(track)
        for a, b in zip(fs, fs[1:]):
            if b == a + 1:
                steps += 1
                jumps += np.hypot(*np.subtract(track[b], track[a])) > 1.0  # 30 m/s at 30 fps
    lengths = Counter(tid for ft in raw_players for tid in ft)
    per_frame = np.array([len(ft) for ft in raw_players])
    short = sum(1 for v in lengths.values() if v < fps)
    players_block = {
        "tracks_raw": len(lengths),
        "tracks_after_relink": len(lengths) - len(remap),
        "relinked": len(remap),
        "short_tracks_share": round(short / max(len(lengths), 1), 3),
        "median_track_s": round(float(np.median(list(lengths.values()))) / fps, 2) if lengths else 0.0,
        "players_per_frame_mean": round(float(per_frame.mean()), 2) if n else 0.0,
        "players_per_frame_max": int(per_frame.max()) if n else 0,
        "frames_over_10_players_share": round(float((per_frame > 10).mean()), 3) if n else 0.0,
        "off_court_feet_share": round(oob / max(total_feet, 1), 3),
        "teleports_per_100_steps": round(100 * jumps / max(steps, 1), 2),
    }
    if players_block["off_court_feet_share"] > 0.1:
        findings.append((3, "court keypoints",
                         f"{players_block['off_court_feet_share']:.0%} of player feet map outside the court, so the "
                         "court mapping itself is wrong on many frames (not just noisy)."))
    if players_block["teleports_per_100_steps"] > 1.0:
        findings.append((2, "court mapping / detection",
                         f"Players jump more than 1 m between frames {players_block['teleports_per_100_steps']} times per "
                         "100 steps (before smoothing). Causes: court mapping jumps, or boxes cut off by other players."))
    ids_per_team = Counter(result.team_of.get(t, -1) for t in {t for ft in result.players for t in ft})
    weak = [k for k, v in result.identity_conf.items() if v < 0.5]
    players_block.update({
        "identities_per_team": {str(k): v for k, v in sorted(ids_per_team.items())},
        "extra_tracks": len(result.identity_extras),
        "weak_identities": len(weak),
    })
    if players_block["short_tracks_share"] > 0.3:
        findings.append((2, "player tracking",
                         f"{players_block['short_tracks_share']:.0%} of player tracks last under a second "
                         f"({players_block['tracks_raw']} tracks, {sum(ids_per_team.values())} identities after "
                         f"re-joining). The tracker loses players often, so bodies pop in and out; most of it is "
                         "missed detections (detector recall on broadcast frames)."))
    if len(weak) > max(2, 0.3 * len(result.identity_conf)):
        findings.append((2, "player identity",
                         f"{len(weak)} of {len(result.identity_conf)} player identities were re-joined on a close call "
                         "between teammates (court position alone can't tell them apart), so touches may be credited "
                         "to the wrong teammate. Reading jersey numbers is the fix."))
    if players_block["frames_over_10_players_share"] > 0.05:
        findings.append((2, "player detection",
                         f"More than 10 players are detected in {players_block['frames_over_10_players_share']:.0%} of "
                         "frames: bench players, coaches or referees are being detected as players."))

    # ------------------------------------------------------------ teams
    team_counts = Counter(result.team_of.values())
    over5 = np.array([max(Counter(result.team_of.get(t, -1) for t in ft if result.team_of.get(t, -1) in (0, 1)).values(),
                          default=0) > 5 for ft in result.players])
    teams_block = {"tracks_per_team": {str(k): v for k, v in sorted(team_counts.items())},
                   "frames_team_over_5_share": round(float(over5.mean()), 3) if n else 0.0}
    if team_colors is not None and track_colors:
        sep = float(np.linalg.norm(team_colors[0] - team_colors[1]))
        amb = 0
        for c in track_colors.values():
            d0, d1 = np.linalg.norm(c - team_colors[0]), np.linalg.norm(c - team_colors[1])
            amb += abs(d0 - d1) / max(d0 + d1, 1e-6) < 0.15
        teams_block["colour_separation_lab"] = round(sep, 1)
        teams_block["ambiguous_tracks_share"] = round(amb / len(track_colors), 3)
        if sep < 25 or teams_block["ambiguous_tracks_share"] > 0.25:
            findings.append((2, "team assignment",
                             f"The two jersey colours are hard to tell apart (separation {sep:.0f}, "
                             f"{teams_block['ambiguous_tracks_share']:.0%} of tracks ambiguous), so players flip teams. "
                             "Try teams.method = 'clip' with prompts naming both jerseys."))
    if teams_block["frames_team_over_5_share"] > 0.1:
        findings.append((1, "team assignment",
                         f"One team has more than 5 players on screen in {teams_block['frames_team_over_5_share']:.0%} "
                         "of frames: referees or bench players are counted, or tracks are on the wrong team."))

    # ------------------------------------------------------------ ball
    raw_ball = np.array([bool(c) for c in ball_candidates[:n]])
    clean = np.array([b is not None for b in result.ball])
    held = np.array([p != NO_POSSESSION for p in result.possessor])
    b3 = np.array([b is not None for b in result.ball_3d])
    ball_block = {"raw_detection_share": round(float(raw_ball.mean()), 3) if n else 0.0,
                  "after_cleaning_share": round(float(clean.mean()), 3) if n else 0.0,
                  "possession_share": round(float(held.mean()), 3) if n else 0.0,
                  "ball_3d_share": round(float(b3.mean()), 3) if n else 0.0}
    if ball_block["raw_detection_share"] < 0.5:
        findings.append((2, "ball detection",
                         f"The ball is detected in only {ball_block['raw_detection_share']:.0%} of frames, so possession, "
                         "passes and the 3D ball are guessed. Train a dedicated ball model at imgsz 1280 on broadcast "
                         "frames and set models.ball_detector."))

    pose_block = None
    if result.pose_3d:
        pf = sum(1 for tr in result.pose_3d.values() for s in tr if s is not None)
        pose_block = {"skeleton_share": round(pf / max(sum(per_frame), 1), 3)}

    findings.sort(key=lambda x: -x[0])
    summary = {"frames": n, "fps": fps, "duration_s": round(n / fps, 2), "video": video_block, "court": court_block,
               "players": players_block, "teams": teams_block, "ball": ball_block, "pose": pose_block,
               "findings": [{"severity": s, "stage": st, "finding": t} for s, st, t in findings]}

    frames = pd.DataFrame({
        "frame": np.arange(n), "mapped": mapped, "source": src.to_numpy()[:n],
        "n_keypoints": n_kp.to_numpy()[:n], "fit_err_m": fit_err.to_numpy()[:n],
        "kp_rejected": rejected.to_numpy()[:n], "players": per_frame,
        "ball_detected": raw_ball, "ball": clean, "possession": held,
    })
    if has_motion:
        frames["motion_ok"] = cal["motion_ok"].to_numpy()[:n]
    return summary, frames


def format_report(summary: dict) -> str:
    c, p, t, b = summary["court"], summary["players"], summary["teams"], summary["ball"]
    pct = lambda v: f"{v:.0%}"
    err = f"{c['median_fit_error_m']:.2f} m" if c["median_fit_error_m"] is not None else "n/a"
    lines = [
        f"CourtVision diagnostics: {summary['frames']} frames ({summary['duration_s']} s)",
        "",
        _video_line(summary.get("video") or {}),
        f"Court mapping   mapped {pct(c['mapped_share'])} of frames (keypoints {pct(c['keypoint_fit_share'])}, "
        f"camera flow {pct(c['flow_share'])}); {c['mean_keypoints_seen']:.1f} keypoints seen per frame, "
        f"fit error {err}, {pct(c['rejected_fit_share'])} fits rejected, longest gap {c['longest_unmapped_s']} s"
        + (f", {c['camera_cuts']} cuts" if c["camera_cuts"] is not None else ""),
        f"Players         {p['tracks_raw']} tracks ({p['relinked']} re-linked), {pct(p['short_tracks_share'])} under 1 s, "
        f"identities per team {p.get('identities_per_team', {})}, {p.get('weak_identities', 0)} weak, "
        f"{p['players_per_frame_mean']} per frame (max {p['players_per_frame_max']}), "
        f"{pct(p['off_court_feet_share'])} feet off court, {p['teleports_per_100_steps']} jumps per 100 steps",
        f"Teams           tracks per team {t['tracks_per_team']}"
        + (f", colour separation {t['colour_separation_lab']}, {pct(t['ambiguous_tracks_share'])} ambiguous"
           if "colour_separation_lab" in t else ""),
        f"Ball            detected {pct(b['raw_detection_share'])}, after cleaning {pct(b['after_cleaning_share'])}, "
        f"possession {pct(b['possession_share'])}, 3D {pct(b['ball_3d_share'])}",
        "",
    ]
    if summary["findings"]:
        lines.append("Weakest stages first:")
        for i, f in enumerate(summary["findings"], 1):
            lines.append(f"{i}. [{f['stage']}] {f['finding']}")
    else:
        lines.append("No stage stands out as broken.")
    return "\n".join(lines)


def _video_line(v: dict) -> str:
    parts = []
    if v.get("width"):
        parts.append(f"{v['width']}x{v['height']} at {v['fps']:.0f} fps")
    if v.get("bitrate_mbps"):
        parts.append(f"{v['bitrate_mbps']:.1f} Mbit/s")
    if v.get("median_player_px"):
        parts.append(f"players ~{v['median_player_px']:.0f} px tall")
    if v.get("median_ball_px"):
        parts.append(f"ball ~{v['median_ball_px']:.0f} px")
    return "Input video     " + (", ".join(parts) or "unknown")


def plot_diagnostics(frames: pd.DataFrame, fps: float, path: str | Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    t = frames["frame"] / fps
    fig, ax = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    colors = {"keypoints": "#2a9d8f", "fused": "#2a9d8f", "flow": "#e9c46a", "none": "#e76f51"}
    ax[0].scatter(t, np.ones(len(t)), c=[colors.get(s, "#999") for s in frames["source"]], marker="|", s=300)
    ax[0].set_yticks([]); ax[0].set_title("Court mapping: green keypoints, yellow camera flow, red none", fontsize=10)
    ax[1].plot(t, frames["n_keypoints"], lw=0.8, label="keypoints seen")
    ax2 = ax[1].twinx(); ax2.plot(t, frames["fit_err_m"], lw=0.8, color="#e76f51", label="fit error (m)")
    ax[1].set_ylabel("keypoints"); ax2.set_ylabel("error m")
    ax[2].plot(t, frames["players"], lw=0.8); ax[2].axhline(10, color="#999", ls="--", lw=0.8)
    ax[2].set_ylabel("players")
    ax[3].fill_between(t, 0, frames["ball_detected"].astype(int), step="mid", alpha=0.5, label="ball detected")
    ax[3].fill_between(t, 0, frames["possession"].astype(int) * 0.5, step="mid", alpha=0.5, label="possession")
    ax[3].set_ylim(0, 1.05); ax[3].set_yticks([]); ax[3].legend(loc="upper right", fontsize=8); ax[3].set_xlabel("seconds")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
