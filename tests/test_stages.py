"""Single stages: tracking through pans and cuts, relinking, smoothing, ball, 10 Hz resampling."""
import numpy as np
import pandas as pd

from courtvision.ball import NO_HOLDER, court_track, pixel_track
from courtvision.court import NBA
from courtvision.output import resample
from courtvision.positions import relink, smooth
from courtvision.track import track_people


def dets_from(boxes_per_frame):
    rows = [(f, "player", *b, 0.9) for f, bs in enumerate(boxes_per_frame) for b in bs]
    return pd.DataFrame(rows, columns=["frame", "role", "x1", "y1", "x2", "y2", "conf"])


def test_tracker_follows_fast_pan_and_restarts_at_cut():
    # Five players standing still on the court while the camera pans 40 px per frame (they slide
    # across the screen); at frame 30 the shot cuts.
    n, pan = 40, 40.0
    base = [(200 + 160 * k, 400) for k in range(5)]
    frames = [[(x - pan * f, y, x - pan * f + 40, y + 120) for x, y in base] for f in range(n)]
    motion = [None] + [np.array([[1, 0, -pan], [0, 1, 0], [0, 0, 1.0]])] * (n - 1)
    tr = track_people(dets_from(frames), n, 30.0, motion, cuts=[30])
    before, after = tr[tr.frame < 30], tr[tr.frame >= 30]
    assert before.track_id.nunique() == 5 and after.track_id.nunique() == 5
    assert not set(before.track_id) & set(after.track_id)       # fresh ids after the cut
    assert (before.groupby("track_id").size() == 30).all()      # nobody lost through the pan


def test_relink_joins_split_track_and_smoothing_cleans_it():
    fps = 30.0
    rows = [(f, 1, 5 + 0.05 * f, 5.0) for f in range(30)] + [(f, 7, 5 + 0.05 * f, 5.0) for f in range(33, 60)]
    rows += [(f, 2, 5 + 0.05 * f, 12.0) for f in range(33, 60)]            # far away: not a match
    pos = pd.DataFrame(rows, columns=["frame", "track_id", "x_m", "y_m"])
    remap = relink(pos, {1: 0, 7: 0, 2: 0}, fps)
    assert remap == {7: 1}
    pos["player_id"] = pos.track_id.map(lambda t: remap.get(t, t))
    rng = np.random.default_rng(0)
    pos.loc[pos.player_id == 1, "x_m"] += rng.normal(0, 0.15, (pos.player_id == 1).sum())
    pos.loc[(pos.player_id == 1) & (pos.frame == 40), "x_m"] = 15.0          # one-frame spike
    sm = smooth(pos, fps).query("player_id == 1").set_index("frame")
    assert 31 in sm.index and abs(sm.loc[31, "x_m"] - 6.55) < 0.2           # gap filled
    assert abs(sm.loc[40, "x_m"] - 7.0) < 0.2                               # spike removed
    assert abs(sm.speed_mps.median() - 1.5) < 0.5


def test_ball_selection_outliers_and_gaps():
    rows = [(f, "ball", 100 + 10 * f, 100, 110 + 10 * f, 110, 0.5) for f in range(10) if f != 6]
    rows += [(1, "ball", 900, 50, 910, 60, 0.9), (4, "ball", 800, 800, 810, 810, 0.95)]  # decoys
    px = pixel_track(pd.DataFrame(rows, columns=["frame", "role", "x1", "y1", "x2", "y2", "conf"]), 40)
    assert px[1, 0] == 110 and px[4, 0] == 140 and abs(px[6, 0] - 160) < 1e-6
    assert np.isnan(px[10:]).all()


def test_ball_dribble_rim_and_floor():
    n = 70
    holder = np.array([1] * 10 + [NO_HOLDER] * 6 + [1] * 4 + [NO_HOLDER] * 20 + [2] * 10 + [NO_HOLDER] * 20)
    pos = pd.DataFrame([(f, 1, 20.0, 7.6) for f in range(n)] + [(f, 2, 26.0, 7.0) for f in range(n)],
                       columns=["frame", "player_id", "x_m", "y_m"])
    at_rim = np.zeros(n, bool)
    at_rim[30] = True
    px = np.full((n, 4), np.nan)
    px[55:] = (500, 500, 510, 510)
    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])    # pixel (505, 510) -> (5.05, 5.1) m
    xy, src = court_track(px, holder, pos, [H] * n, NBA, at_rim, fps=30.0)
    assert src[12] == "held" and tuple(xy[12]) == (20.0, 7.6)                    # dribble stays with 1
    assert src[30] == "flight" and np.allclose(xy[30], NBA.hoops[1])             # shot reaches the rim
    assert np.allclose(xy[25], np.array([20, 7.6]) + (NBA.hoops[1] - [20, 7.6]) * 6 / 11)
    assert src[45] == "held" and src[60] == "floor" and np.allclose(xy[60], (5.05, 5.1))


def test_resample_interpolates_and_blanks_long_gaps():
    frames = np.r_[np.arange(4), np.arange(8, 10)]
    vals = np.column_stack([frames.astype(float), np.zeros(len(frames))])
    out = resample(frames, vals, 30.0, np.arange(0, 0.3, 0.04), max_gap_s=0.1)
    assert np.isclose(out[1, 0], 1.2)          # 0.04 s = frame 1.2
    assert np.isnan(out[3, 0])                 # 0.12 s falls in the 5-frame gap


def test_default_config_file_matches_the_code():
    from dataclasses import asdict
    from courtvision.config import Config
    assert asdict(Config.load("configs/default.yaml")) == asdict(Config())
