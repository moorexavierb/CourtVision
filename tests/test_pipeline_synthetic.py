"""End-to-end run of everything after the models, on a synthetic broadcast clip.

Four players (two teams, white and dark-blue jerseys) are placed on a real NBA
court, imaged through a synthetic broadcast camera, and drawn into a video.
Player 1 passes to player 2. The test checks team assignment, the detected
pass, court-space tracking, the 3D ball and every output file.
"""
import json

import numpy as np
import cv2
import pandas as pd

from courtvision.config import Config
from courtvision.court.geometry import NBA
from courtvision.court.homography import project
from courtvision.io.video import save_video
from courtvision.pipeline import Pipeline
from courtvision.teams.team_assigner import TeamAssigner
from courtvision.types import DetectionResult
from conftest import floor_homography
from test_pose import standing_skeleton

JERSEYS = {0: (240, 240, 240), 1: (140, 40, 20)}  # BGR: white, dark blue


def player_box(P, x, y, height=2.0, width=0.6):
    foot = project(P, np.array([x, y, 0.0]))[0]
    head = project(P, np.array([x, y, height]))[0]
    h = foot[1] - head[1]
    w = h * width / height
    return [foot[0] - w / 2, head[1], foot[0] + w / 2, foot[1]]


def test_synthetic_play(tmp_path, broadcast_cam):
    P, K, R, t, size = broadcast_cam
    H = floor_homography(P)
    fps, n = 30.0, 90
    starts = {1: ((8.0, 6.0), 0), 2: ((12.0, 9.0), 0), 3: ((9.0, 4.0), 1), 4: ((13.0, 10.5), 1)}

    players, ball_c, frames = [], [], []
    for f in range(n):
        img = np.full((size[1], size[0], 3), (120, 170, 214), np.uint8)
        fp = {}
        for tid, ((x, y), team) in starts.items():
            x += 0.05 * f  # everyone drifts at 1.5 m/s
            b = player_box(P, x, y)
            fp[tid] = {"bbox": b, "conf": 0.9}
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), JERSEYS[team], -1)
        players.append(fp)
        frames.append(img)

        p1 = np.array([8.0 + 0.05 * f, 6.0, 1.0])
        p2 = np.array([12.0 + 0.05 * f, 9.0, 1.0])
        if f < 25:
            pos = p1
        elif f < 55:  # one-second lob pass under real gravity: peaks ~1.23 m above release
            T, tt = 1.0, (f - 25) / fps
            pos = p1 + (p2 - p1) * tt / T + np.array([0, 0, 0.5 * 9.81 * tt * (T - tt)])
        else:
            pos = p2
        u, v = project(P, pos)[0]
        ball_c.append([{"bbox": [u - 4, v - 4, u + 4, v + 4], "conf": 0.6}])

    video = tmp_path / "synthetic.mp4"
    save_video(frames, video, fps)
    det = DetectionResult(players=players, referees=[{} for _ in range(n)], ball_candidates=ball_c,
                          hoops=[None] * n)

    cfg = Config.load(None, output={"dir": str(tmp_path / "out"), "cache_dir": None, "mode": "3d"})
    pipe = Pipeline(cfg)
    assigner = TeamAssigner(cfg.teams)
    team_of = assigner.assign(frames, players)
    assert team_of[1] == team_of[2] != team_of[3] == team_of[4]

    # 2D body keypoints for player 3, as the pose model would report them.
    poses2d = []
    for f in range(n):
        uv = np.array([project(P, X)[0] for X in standing_skeleton(9.0 + 0.05 * f, 4.0)])
        poses2d.append({3: np.column_stack([uv, np.full(17, 0.9)])})

    result = pipe.analyze(det, [H] * n, team_of, fps, size, poses2d)
    passes = [e for e in result.events if e.type == "pass"]
    assert len(passes) == 1 and passes[0].player_id == 1 and passes[0].target_id == 2

    # Feet land where we put them on the court.
    assert np.allclose(result.court_positions[3][0], (9.0, 4.0), atol=0.05)
    assert abs(result.speed_kmh[1][-1] - 1.5 * 3.6) < 0.3

    # 2D: the ball's floor point follows the straight ground line of the lob, at constant speed.
    # (Possession hysteresis keeps the passer as holder for a few frames after the release.)
    flight = [f for f in range(25, 55) if result.ball_court_source[f] == "flight"]
    assert len(flight) >= 15, result.ball_court_source[20:60]
    for f in flight:
        tt = (f - 25) / fps
        truth = (8.0 + 0.05 * f + 4.0 * tt, 6.0 + 3.0 * tt)
        assert np.hypot(*np.subtract(result.ball_court[f], truth)) < 0.8, (f, result.ball_court[f], truth)  # ball is an arm away from the feet
    assert result.ball_court_source[10] == "held" and result.ball_court_source[80] == "held"

    # The lob peaks well above the 1 m hold height.
    peak = max(b[2] for b in result.ball_3d[25:55] if b is not None)
    assert abs(peak - 2.23) < 0.3, peak

    out_dir = tmp_path / "out" / "synthetic"
    out_dir.mkdir(parents=True)
    pipe.write_outputs(result, video, out_dir)
    for name in ["tracking_2d.csv", "tracking_2d.parquet", "touches.csv", "touches.parquet", "events.csv",
                 "box_score.csv", "tracking.csv", "scene.json",
                 "play_3d.html", "synthetic_annotated.mp4"]:
        assert (out_dir / name).exists(), name
    scene = json.loads((out_dir / "scene.json").read_text())
    p3 = next(p for p in scene["frames"][10]["players"] if p["id"] == 3)
    assert scene["pose_format"]["name"] == "coco17" and len(p3["pose"]) == 17
    assert abs(p3["pose"][0][2] - 1.95) < 0.25   # nose at head height
    html = (out_dir / "play_3d.html").read_text()
    assert html.startswith("<!doctype html>") and '"pose_format"' in html and "/*__CV_SCENE__*/null" not in html
    box = pd.read_csv(out_dir / "box_score.csv")
    assert box.set_index("player_id").loc[1, "passes"] == 1

    # Touch-by-touch: player 1 passes to player 2, who holds it to the end of the clip.
    touches = pd.read_parquet(out_dir / "touches.parquet")
    assert list(touches.player_id) == [1, 2]
    t1, t2 = touches.iloc[0], touches.iloc[1]
    assert (t1.action, t1.outcome, t1.target_id) == ("pass", "completed", 2)
    assert (t2.start_action, t2.action) == ("catch", "end_of_clip")
    assert t1.possession_id == t2.possession_id
    assert t1.tm1_id == 2 and {t1.op1_id, t1.op2_id} == {3, 4} and pd.isna(t1.op3_id)
    assert t1.nearest_defender_id == 3 and abs(t1.nearest_defender_m - np.hypot(1, 2)) < 0.1
    assert abs(t1.speed_kmh - 5.4) < 0.5 and abs(t1.op1_speed_kmh - 5.4) < 0.5
    assert 0 < t1.confidence <= 1


def test_2d_mode_outputs(tmp_path, broadcast_cam):
    """Default mode: tracking_2d at 10 Hz with every player and the ball, and no 3D files."""
    P, K, R, t, size = broadcast_cam
    H = floor_homography(P)
    fps, n = 30.0, 60
    players, ball_c = [], []
    for f in range(n):
        players.append({tid: {"bbox": player_box(P, x + 0.1 * f, y), "conf": 0.9}
                        for tid, (x, y) in {1: (8.0, 6.0), 2: (12.0, 9.0)}.items()})
        u, v = project(P, np.array([8.0 + 0.1 * f, 6.0, 1.0]))[0]
        ball_c.append([{"bbox": [u - 4, v - 4, u + 4, v + 4], "conf": 0.6}])
    det = DetectionResult(players=players, referees=[{} for _ in range(n)], ball_candidates=ball_c,
                          hoops=[None] * n)
    frames = [np.full((size[1], size[0], 3), 128, np.uint8) for _ in range(n)]
    video = tmp_path / "clip.mp4"
    save_video(frames, video, fps)
    cfg = Config.load(None, output={"dir": str(tmp_path / "out"), "cache_dir": None, "write_video": False})
    assert cfg.output.mode == "2d" and cfg.output.tracking_fps == 10.0
    pipe = Pipeline(cfg)
    result = pipe.analyze(det, [H] * n, {1: 0, 2: 1}, fps, size)
    assert all(b is None for b in result.ball_3d)
    out_dir = tmp_path / "out" / "clip"
    out_dir.mkdir(parents=True)
    pipe.write_outputs(result, video, out_dir)
    assert not (out_dir / "scene.json").exists() and not (out_dir / "play_3d.html").exists()
    t2d = pd.read_csv(out_dir / "tracking_2d.csv")
    assert t2d["sample"].nunique() == 20                     # 59 frames at 30 fps = 1.97 s -> 20 samples
    assert np.allclose(np.diff(sorted(t2d.time_s.unique())), 0.1)
    per = t2d.groupby("sample")["object"].value_counts().unstack()
    assert (per["player"] == 2).all() and (per["ball"] == 1).all()
    p1 = t2d[(t2d.object == "player") & (t2d.id == 1)].set_index("time_s")
    assert abs(p1.loc[1.0, "x_m"] - (8.0 + 0.1 * 30)) < 0.1     # 3 m/s drift, resampled on the 10 Hz clock
    ball = t2d[t2d.object == "ball"]
    assert (ball.source == "held").mean() > 0.8 and set(p1.team) == {"Team A"}
