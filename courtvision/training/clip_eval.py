"""Does the pipeline track what it should? Run it on game clips and look, by number and by eye.

Box scores on single frames (recall, mAP) say whether players are found; they do not say whether
the tracking dataset follows each player. This module runs the *real* pipeline
(``pipeline.run``: detector, court model, camera-compensated tracker, relinking, 10 Hz output) on
short clips and saves, per clip, a small tracking dataset plus what is needed to check it:

Clips with ground truth (SportsMOT basketball sequences of games held out of training):

    metrics.json            HOTA / DetA / AssA / MOTA / IDF1 / ID switches (TrackEval), for the
                            tracker's own ids and for the ids after relinking
    players_image.csv       every tracked player box, with the ground-truth player it covers
                            (gt_id, iou; -1 = a box on nobody) and its court x, y
    tracking.parquet        the 10 Hz court dataset for the clip (as notebook 00 writes it)
    gt_vs_pred.mp4          ground truth (thin white) vs tracks (coloured by id); red = missed
                            player, orange = box on nobody
    id_timeline.png         one row per real player, coloured by the track id that covers him:
                            one colour per row = tracked through; a colour change = an ID switch;
                            white = lost
    counts.png              players labelled, tracked correctly and false boxes, per frame
    pred_mot.txt, gt_mot.txt  MOTChallenge files, to re-score with TrackEval's own scripts

Your own clips (no labels): the same pipeline outputs and ``sanity_checks``: things a correct
tracking dataset of basketball cannot do (more than five players a team at once, running faster
than about 10 m/s, jumping metres in 0.1 s, two ids standing on one spot), plus plots of players
per sample, id lifetimes and court paths.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .evaluate import iou_matrix, match, mot_metrics, mot_table

# Physical limits used by sanity_checks. NBA tracking data puts top in-game speeds at about
# 9-10 m/s (20-22 mph); anything well above is a mis-placed foot point or a swapped id.
MAX_SPEED_MPS = 10.0
MAX_STEP_M = 1.5          # in 0.1 s: 15 m/s
SAME_SPOT_M = 0.5         # two ids this close for SAME_SPOT_S are one player tracked twice
SAME_SPOT_S = 1.0


# ----------------------------------------------------------------- video helpers

def frames_to_video(paths: list[Path], out: Path, fps: float) -> Path:
    """Write frames to an H.264 mp4 (ffmpeg, near-lossless) or, without ffmpeg, mp4v (OpenCV)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise FileNotFoundError(paths[0])
    h, w = first.shape[:2]
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "14", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None and img.shape[:2] == (h, w):
                proc.stdin.write(img.tobytes())
        proc.stdin.close()
        if proc.wait() == 0:
            return out
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None and img.shape[:2] == (h, w):
            writer.write(img)
    writer.release()
    return out


def browser_mp4(path: Path) -> Path:
    """Re-encode an OpenCV mp4v file as H.264 so it plays in a browser / Colab; no-op without ffmpeg."""
    path = Path(path)
    if not shutil.which("ffmpeg") or not path.exists():
        return path
    tmp = path.with_name(path.stem + "_h264.mp4")
    ok = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-crf", "23",
                         "-pix_fmt", "yuv420p", str(tmp)]).returncode == 0
    if ok:
        tmp.replace(path)
    return path


def cut_clip(video: Path, out: Path, start_s: float = 0.0, seconds: Optional[float] = None) -> Path:
    """A stretch of a video, re-encoded so the cut starts on a real frame (stream copy cuts on
    keyframes and can start on a frozen or missing frame)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start_s), "-i", str(video)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-an", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    return out


# ----------------------------------------------------------------- ground-truth clips

def pred_tracks(tracks: pd.DataFrame, frame_ids: list[int], remap: Optional[dict] = None
                ) -> dict[int, dict[int, list[float]]]:
    """{frame id: {id: box}} of the pipeline's player tracks; remap applies relinking. If two
    tracks end up with one id in a frame (a relink seam), the more confident box is kept."""
    players = tracks[tracks.role == "player"]
    out: dict[int, dict[int, list[float]]] = {}
    for r in players.sort_values("conf").itertuples():
        if r.frame >= len(frame_ids):
            continue
        pid = int(remap.get(r.track_id, r.track_id)) if remap else int(r.track_id)
        out.setdefault(frame_ids[r.frame], {})[pid] = [r.x1, r.y1, r.x2, r.y2]
    return out


def match_frames(gt: dict, pred: dict, iou_thr: float = 0.5) -> pd.DataFrame:
    """One row per ground-truth or predicted box per frame: gt_id, pred_id (-1 when unmatched), iou."""
    rows = []
    for f in sorted(set(gt) | set(pred)):
        g, p = gt.get(f, {}), pred.get(f, {})
        gi, pi = list(g), list(p)
        ga, pa = np.array([g[i] for i in gi], float).reshape(-1, 4), np.array([p[i] for i in pi], float).reshape(-1, 4)
        pairs, _, _ = match(ga, pa, iou_thr)
        ious = iou_matrix(ga, pa)
        mg, mp = {a for a, _ in pairs}, {b for _, b in pairs}
        rows += [(f, gi[a], pi[b], float(ious[a, b])) for a, b in pairs]
        rows += [(f, gi[a], -1, 0.0) for a in range(len(gi)) if a not in mg]
        rows += [(f, -1, pi[b], 0.0) for b in range(len(pi)) if b not in mp]
    return pd.DataFrame(rows, columns=["frame", "gt_id", "pred_id", "iou"])


def write_mot(tracks: dict, path: Path) -> None:
    """MOTChallenge text: frame,id,x,y,w,h,conf,-1,-1,-1."""
    with open(path, "w") as f:
        for fr in sorted(tracks):
            for tid, (x0, y0, x1, y1) in sorted(tracks[fr].items()):
                f.write(f"{fr},{tid},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},1,-1,-1,-1\n")


def evaluate_gt_clip(seq: Path, cfg, out_dir: Path, max_frames: Optional[int] = 750, artifacts: bool = True) -> dict:
    """The pipeline on one SportsMOT sequence (first max_frames frames), scored against its tracks.
    Returns {"raw": metrics of the tracker's ids, "linked": after relinking, "result": pipeline Result}."""
    from .. import pipeline
    from .sportsmot import frame_paths, read_tracks, seq_fps
    seq, out_dir = Path(seq), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = seq_fps(seq)
    paths = sorted(frame_paths(seq).items())[:max_frames]
    if not paths:
        raise FileNotFoundError(f"no frames in {seq}/img1")
    ids = [f for f, _ in paths]
    video = frames_to_video([p for _, p in paths], out_dir / f"{seq.name}.mp4", fps)
    cfg = copy.deepcopy(cfg)
    cfg.output.write_video = artifacts
    result = pipeline.run(video, cfg, out_dir=out_dir, write_outputs=True)
    ids = ids[:len(result.homographies)]
    keep = set(ids)
    gt = {f: b for f, b in read_tracks(seq).items() if f in keep}
    raw = pred_tracks(result.tracks, ids)
    linked = pred_tracks(result.tracks, ids, result.remap)
    m_raw, m_linked = mot_metrics(gt, raw), mot_metrics(gt, linked)
    raw_te, linked_te = m_raw.pop("_raw", None), m_linked.pop("_raw", None)
    metrics = {"clip": seq.name, "frames": len(ids), "fps": fps, "tracker_ids": m_raw, "after_relink": m_linked,
               "pipeline": result.diagnostics}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_mot(gt, out_dir / "gt_mot.txt")
    write_mot(linked, out_dir / "pred_mot.txt")
    matches = match_frames(gt, linked)
    image_table(result, ids, matches).to_csv(out_dir / "players_image.csv", index=False)
    if artifacts:
        plot_id_timeline(matches, fps, out_dir / "id_timeline.png", title=seq.name)
        plot_counts(matches, fps, out_dir / "counts.png", title=seq.name)
        draw_gt_vs_pred([p for _, p in paths][:len(ids)], ids, gt, linked, matches, out_dir / "gt_vs_pred.mp4", fps)
    m_raw["_raw"], m_linked["_raw"] = raw_te, linked_te   # kept for mot_table's combined row
    return {"raw": m_raw, "linked": m_linked, "result": result, "matches": matches}


def image_table(result, frame_ids: list[int], matches: pd.DataFrame) -> pd.DataFrame:
    """Tracked player boxes with the real player each covers and the court position given to it."""
    t = result.tracks[result.tracks.role == "player"].copy()
    t = t[t.frame < len(frame_ids)]
    t["player_id"] = t.track_id.map(lambda k: result.remap.get(k, k))
    t["frame_id"] = t.frame.map(lambda f: frame_ids[f])
    t["time_s"] = (t.frame / result.fps).round(3)
    m = matches[matches.pred_id >= 0][["frame", "pred_id", "gt_id", "iou"]].rename(columns={"frame": "frame_id", "pred_id": "player_id"})
    t = t.merge(m, on=["frame_id", "player_id"], how="left")
    t["gt_id"] = t.gt_id.fillna(-1).astype(int)
    pos = result.positions[["frame", "player_id", "x_m", "y_m"]] if len(result.positions) else None
    if pos is not None:
        t = t.merge(pos, on=["frame", "player_id"], how="left")
    cols = ["frame", "frame_id", "time_s", "track_id", "player_id", "gt_id", "iou", "conf", "x1", "y1", "x2", "y2", "x_m", "y_m"]
    return t.reindex(columns=cols).sort_values(["frame", "player_id"])


def _color(i: int) -> tuple[int, int, int]:
    """A stable, distinct BGR colour per id (golden-angle hue steps)."""
    hue = int((i * 137.508) % 180)
    c = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def draw_gt_vs_pred(paths, frame_ids, gt, pred, matches: pd.DataFrame, out: Path, fps: float) -> Path:
    by_frame = {f: g for f, g in matches.groupby("frame")}
    writer = None
    for p, f in zip(paths, frame_ids):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        m = by_frame.get(f, matches.iloc[:0])
        missed = set(m[(m.pred_id < 0)].gt_id)
        false = set(m[(m.gt_id < 0)].pred_id)
        for gid, (x0, y0, x1, y1) in gt.get(f, {}).items():
            col = (0, 0, 255) if gid in missed else (255, 255, 255)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 3 if gid in missed else 1)
            cv2.putText(img, f"gt{gid}", (int(x0), int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        for pid, (x0, y0, x1, y1) in pred.get(f, {}).items():
            col = (0, 140, 255) if pid in false else _color(pid)
            cv2.rectangle(img, (int(x0) + 2, int(y0) + 2), (int(x1) - 2, int(y1) - 2), col, 2)
            cv2.putText(img, str(pid), (int(x0), int(y0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        n_gt, n_ok = len(gt.get(f, {})), int(((m.gt_id >= 0) & (m.pred_id >= 0)).sum())
        cv2.putText(img, f"frame {f}  tracked {n_ok}/{n_gt}  missed {len(missed)}  false {len(false)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(img)
    if writer is not None:
        writer.release()
    return browser_mp4(out)


def plot_id_timeline(matches: pd.DataFrame, fps: float, out: Path, title: str = "") -> Path:
    """One row per real player: coloured by the track id covering him, white where he is lost."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    m = matches[matches.gt_id >= 0].sort_values("frame")
    players = sorted(m.gt_id.unique())
    fig, ax = plt.subplots(figsize=(14, 0.35 * len(players) + 1.5))
    switches = 0
    for row, gid in enumerate(players):
        g = m[m.gt_id == gid]
        prev = None
        for pid, run in g.groupby((g.pred_id != g.pred_id.shift()).cumsum()):
            p = int(run.pred_id.iloc[0])
            t0, t1 = run.frame.iloc[0] / fps, (run.frame.iloc[-1] + 1) / fps
            if p >= 0:
                c = _color(p)
                ax.broken_barh([(t0, t1 - t0)], (row - 0.4, 0.8), color=(c[2] / 255, c[1] / 255, c[0] / 255))
                if prev is not None and prev != p:
                    ax.plot([t0, t0], [row - 0.5, row + 0.5], "k-", lw=1.5)
                    switches += 1
                prev = p
    ax.set_yticks(range(len(players)), [f"gt {g}" for g in players])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{title}: track id covering each real player (black tick = id change: {switches}; white = lost)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return Path(out)


def plot_counts(matches: pd.DataFrame, fps: float, out: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    frames = np.sort(matches.frame.unique())
    count = lambda sel: matches[sel].groupby("frame").size().reindex(frames, fill_value=0)
    per = pd.DataFrame({"labelled": count(matches.gt_id >= 0), "tracked": count((matches.gt_id >= 0) & (matches.pred_id >= 0)),
                        "false": count(matches.gt_id < 0)})
    t = per.index.to_numpy() / fps
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.plot(t, per.labelled, "k-", lw=1, label="players labelled")
    ax.plot(t, per.tracked, "g-", lw=1, label="tracked (IoU >= 0.5)")
    ax.plot(t, per["false"], "-", color="orange", lw=1, label="boxes on nobody")
    ax.set_xlabel("time (s)")
    ax.legend(loc="lower left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return Path(out)


# ----------------------------------------------------------------- any clip: sanity checks

def sanity_checks(tracking: pd.DataFrame, diagnostics: dict, hz: float = 10.0) -> dict:
    """Numbers a correct basketball tracking dataset keeps near zero (except the first block)."""
    players = tracking[tracking.object == "player"] if len(tracking) else tracking
    n_samples = (diagnostics.get("dataset") or {}).get("samples") or (int(tracking["sample"].max()) + 1 if len(tracking) else 0)
    out = {"court_mapped_share": diagnostics.get("court", {}).get("mapped_share"),
           "players_per_sample": (diagnostics.get("dataset") or {}).get("players_per_sample"),
           "all_ten_share": (diagnostics.get("dataset") or {}).get("all_players_share"),
           "ids": int(players.id.nunique()) if len(players) else 0,
           "ids_under_1s_share": None}
    if not len(players):
        return out
    per = players.groupby("sample").size()
    life = players.groupby("id").size() / hz
    out["ids_under_1s_share"] = round(float((life < 1).mean()), 3)
    out["over_ten_share"] = round(float((per > 10).sum() / max(n_samples, 1)), 3)
    team = players[players.team.notna()].groupby(["sample", "team"]).size()
    out["over_five_a_team_share"] = round(float((team > 5).groupby("sample").any().sum() / max(n_samples, 1)), 3)
    out["too_fast_share"] = round(float((players.speed_mps > MAX_SPEED_MPS).mean()), 4)
    p = players.sort_values(["id", "sample"])
    step = np.hypot(p.x_m.diff(), p.y_m.diff())[(p.id == p.id.shift()) & (p["sample"] == p["sample"].shift() + 1)]
    out["jumps_over_1.5m_share"] = round(float((step > MAX_STEP_M).mean()), 4) if len(step) else 0.0
    out["same_spot_pairs"] = same_spot_pairs(players, hz)
    return out


def same_spot_pairs(players: pd.DataFrame, hz: float) -> int:
    """Pairs of ids within SAME_SPOT_M of each other for SAME_SPOT_S or longer: one player, two tracks."""
    close = {}
    for s, g in players.groupby("sample"):
        xy, ids = g[["x_m", "y_m"]].to_numpy(), g.id.to_numpy()
        if len(g) < 2:
            continue
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        for a, b in zip(*np.nonzero(np.triu(d < SAME_SPOT_M, 1))):
            close[(ids[a], ids[b])] = close.get((ids[a], ids[b]), 0) + 1
    return int(sum(v >= SAME_SPOT_S * hz for v in close.values()))


def plot_dataset(tracking: pd.DataFrame, court, out: Path, title: str = "", hz: float = 10.0) -> Path:
    """Three panels: players per sample, id lifetimes (by team), court paths."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    players = tracking[tracking.object == "player"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={"width_ratios": [1.2, 1.2, 1.6]})
    if not len(players):
        for ax in axes:
            ax.text(0.5, 0.5, "no players placed on the court", ha="center")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return Path(out)
    per = players.groupby("time_s").size()
    axes[0].plot(per.index, per.values, lw=1)
    axes[0].axhline(10, color="k", ls="--", lw=0.8)
    axes[0].set(xlabel="time (s)", ylabel="players placed", title="players per 10 Hz sample (10 = all)")
    team_col = {0: "tab:blue", 1: "tab:red"}
    order = players.groupby("id").time_s.min().sort_values().index
    for row, pid in enumerate(order):
        g = players[players.id == pid]
        axes[1].plot(g.time_s, np.full(len(g), row), ".", ms=2, color=team_col.get(g.team.iloc[0], "gray"))
    axes[1].set(xlabel="time (s)", ylabel="id (by first seen)", title="id lifetimes (colour = team)")
    ax = axes[2]
    ax.add_patch(plt.Rectangle((0, 0), court.length, court.width, fill=False))
    ax.plot([court.length / 2] * 2, [0, court.width], "k-", lw=0.8)
    for pid, g in players.groupby("id"):
        ax.plot(g.x_m, g.y_m, "-", lw=0.8, color=team_col.get(g.team.iloc[0], "gray"), alpha=0.7)
    ball = tracking[tracking.object == "ball"]
    if len(ball):
        ax.plot(ball.x_m, ball.y_m, ".", ms=2, color="orange", label="ball")
    ax.set(xlim=(-2, court.length + 2), ylim=(court.width + 2, -2), aspect="equal", title="court paths (m)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return Path(out)


def evaluate_own_clip(video: Path, cfg, out_dir: Path, start_s: float = 0.0, seconds: Optional[float] = 30.0) -> dict:
    """The pipeline on a stretch of one of your clips; tracking.parquet, the tracked video,
    sanity checks and plots in out_dir."""
    from .. import pipeline
    from ..court import get_court
    out_dir = Path(out_dir)
    clip = cut_clip(video, out_dir / f"{Path(video).stem}.mp4", start_s, seconds)
    cfg = copy.deepcopy(cfg)
    cfg.output.write_video = True
    result = pipeline.run(clip, cfg, out_dir=out_dir, write_outputs=True)
    tracked = out_dir / f"{clip.stem}_tracked.mp4"
    browser_mp4(tracked)
    checks = sanity_checks(result.tracking, result.diagnostics, cfg.output.hz)
    (out_dir / "checks.json").write_text(json.dumps(checks, indent=2, default=float))
    plot_dataset(result.tracking, get_court(cfg.court.standard), out_dir / "dataset.png", title=Path(video).stem, hz=cfg.output.hz)
    return {"checks": checks, "result": result}


def summary_table(per_clip: dict[str, dict], which: str = "linked") -> pd.DataFrame:
    """mot_table over the GT clips (overall row combined by TrackEval)."""
    return mot_table({k: dict(v[which]) for k, v in per_clip.items()})
