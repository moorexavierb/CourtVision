"""End to end after the models: a synthetic broadcast play with ten players and a lob pass.

Ten players (two teams, white and dark-blue jerseys) jog across a real NBA court seen by a
broadcast camera. Player 0 holds the ball, throws a one-second lob to player 1, who keeps it.
Detections and keypoints are what perfect models would report. The test checks the 10 Hz dataset:
every player and the ball in every sample, positions to within centimetres, one id per player,
teams, and the ball's ground track during the pass.
"""
import cv2
import numpy as np
import pandas as pd

from courtvision.config import Config
from courtvision.court import NBA, apply_h
from courtvision.pipeline import analyze, write
from courtvision.video import save_video
from conftest import floor_homography

FPS, N = 30.0, 90
JERSEY = {0: (240, 240, 240), 1: (140, 40, 20)}  # BGR
START = [(6, 4), (11, 9), (8, 12), (13, 3), (4, 7.6), (9, 5), (12, 7), (7, 10), (14, 11), (5, 2)]
VEL = 1.5  # m/s along x


def truth(i, f):
    x, y = START[i]
    return np.array([x + VEL * f / FPS, y])


def ball_pos(f):
    a, b = np.r_[truth(0, f), 1.0], np.r_[truth(1, f), 1.0]
    if f < 25:
        return a
    if f < 55:
        t = (f - 25) / FPS
        return a + (b - a) * t + np.array([0, 0, 0.5 * 9.81 * t * (1.0 - t)])
    return b


def project(P, X):
    x = P @ np.r_[X, 1.0]
    return x[:2] / x[2]


def scene(P):
    rows, kps = [], np.zeros((N, 18, 3))
    G = np.linalg.inv(floor_homography(P))
    for f in range(N):
        for i in range(10):
            x, y = truth(i, f)
            foot, head = project(P, [x, y, 0]), project(P, [x, y, 2.0])
            h = foot[1] - head[1]
            rows.append((f, "player", foot[0] - 0.15 * h, head[1], foot[0] + 0.15 * h, foot[1], 0.9))
        u, v = project(P, ball_pos(f))
        rows.append((f, "ball", u - 5, v - 5, u + 5, v + 5, 0.6))
        img = apply_h(G, NBA.keypoints)
        kps[f] = np.column_stack([img, np.full(18, 0.9)])
    return pd.DataFrame(rows, columns=["frame", "role", "x1", "y1", "x2", "y2", "conf"]), kps


def frames_for(dets, size):
    def render(wanted):
        out = {}
        for f in wanted:
            img = np.full((size[1], size[0], 3), (120, 170, 214), np.uint8)
            for k, r in enumerate(dets[(dets.frame == f) & (dets.role == "player")].itertuples()):
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), JERSEY[int(k >= 5)], -1)
            out[f] = img
        return out
    return render


def test_synthetic_play(tmp_path, broadcast_cam):
    P, *_, size = broadcast_cam
    dets, kps = scene(P)
    motion = [None] + [np.eye(3)] * (N - 1)
    cfg = Config.load(None, output={"cache_dir": None, "write_video": True})
    r = analyze(dets, kps, motion, [], FPS, size, cfg, team_frames=frames_for(dets, size))

    # One id per player, matched to the truth by position at frame 0.
    pos = r.positions
    assert pos.player_id.nunique() == 10
    first = pos[pos.frame == 0].set_index("player_id")[["x_m", "y_m"]]
    who = {i: int(((first - truth(i, 0)) ** 2).sum(axis=1).idxmin()) for i in range(10)}
    assert len(set(who.values())) == 10
    assert len({r.team_of[who[i]] for i in range(5)}) == 1 and r.team_of[who[0]] != r.team_of[who[5]]

    # The 10 Hz table: 30 samples (0 to 2.9 s), all ten players and the ball in each.
    t = r.tracking
    assert t["sample"].nunique() == 30 and np.allclose(np.diff(sorted(t.time_s.unique())), 0.1)
    per = t.groupby(["sample", "object"]).size().unstack()
    assert (per.player == 10).all() and (per.ball == 1).all()
    p0 = t[(t.object == "player") & (t.id == who[0])].set_index("time_s")
    assert np.allclose(p0.loc[1.5, ["x_m", "y_m"]].to_numpy(float), truth(0, 45), atol=0.05)
    assert abs(p0.speed_mps.median() - VEL) < 0.05
    assert r.diagnostics["dataset"]["all_players_share"] == 1.0

    # Ball: held by player 0, then a straight ground line during the lob, then held by player 1.
    b = r.ball
    assert b.holder[10] == who[0] and b.holder[80] == who[1]
    flight = b[(b.frame >= 25) & (b.frame < 55) & (b.source == "flight")]
    assert len(flight) >= 15, b.source[20:60].tolist()
    for row in flight.itertuples():
        assert np.hypot(row.x_m - ball_pos(row.frame)[0], row.y_m - ball_pos(row.frame)[1]) < 0.8  # an arm away
    ball10 = t[t.object == "ball"]
    assert set(ball10.source) <= {"held", "flight"} and t[t.has_ball].id.isin([who[0], who[1]]).all()

    video = tmp_path / "play.mp4"
    save_video((np.zeros((size[1], size[0], 3), np.uint8) for _ in range(N)), video, FPS)
    write(r, video, tmp_path / "out", cfg)
    out = pd.read_parquet(tmp_path / "out" / "tracking.parquet")
    assert set(out.team.dropna()) == {"Team A", "Team B"} and len(out) == len(t)
    for name in ["tracking.csv", "players_per_frame.parquet", "ball_per_frame.parquet", "frames.csv",
                 "diagnostics.json", "play_tracked.mp4"]:
        assert (tmp_path / "out" / name).exists(), name
