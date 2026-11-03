"""Multi-dataset build, grouped splits, cross-dataset folds, evaluation metrics, 2D ball track."""
import cv2
import numpy as np
import pandas as pd
import yaml

from courtvision.court.geometry import NBA
from courtvision.possession.acquisition import NO_POSSESSION
from courtvision.tracking2d import ball_court_track, resample
from courtvision.training.datasets import write_data_yaml
from courtvision.training.evaluate import frame_scores, label_path, mot_metrics, summarise
from courtvision.training.sources import (Source, assign_groups, build_multisource, kfold_folds, lodo_folds,
                                          original_stem, video_key)
from courtvision.types import Event


def _img(path, seed):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))


def _dataset(root, names, files, seed):
    """files: {split: [(stem, [class ids])]}"""
    for split, items in files.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for k, (stem, classes) in enumerate(items):
            _img(root / split / "images" / f"{stem}.jpg", seed * 1000 + hash((split, k)) % 1000)
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(f"{c} 0.5 0.5 0.1 0.1" for c in classes))
    write_data_yaml(root, names)


def test_group_keys():
    assert original_stem("clip3_frame_0012_jpg.rf.0123456789abcdef0123456789abcdef") == "clip3_frame_0012"
    assert video_key("clip3_frame_0012_jpg.rf.0123456789abcdef0123456789abcdef") == "clip3"
    assert video_key("v_abc_c008_000123") == video_key("v_abc_c008_000456")
    # Too few videos to split by: fall back to one group per original image (augmented copies together).
    stems = ["a_jpg.rf.%032x" % i for i in range(3)] + ["b_jpg.rf.%032x" % i for i in range(3)]
    assert assign_groups(stems) == ["a"] * 3 + ["b"] * 3


def test_build_splits_by_group_and_writes_folds(tmp_path):
    # Roboflow-style source: 12 videos x 5 frames, each frame exported with 2 augmented copies.
    rf = []
    for v in range(12):
        for f in range(5):
            for aug in range(2):
                rf.append((f"game{v}_frame{f:03d}_jpg.rf.{v:04d}{f:04d}{aug:024d}", [0, 1]))
    _dataset(tmp_path / "rf", ["Ball", "Player"], {"train": rf}, seed=1)
    # SportsMOT-style source with other class names.
    smot = [(f"seq{s}_{f:06d}", [0]) for s in range(10) for f in range(6)]
    _dataset(tmp_path / "smot", ["person"], {"train": smot}, seed=2)
    # Held-out clips, eval only; one image is a byte-identical copy of a Roboflow image.
    _dataset(tmp_path / "own", ["Player", "Ball"], {"train": [(f"lakers_{i:04d}", [0, 1]) for i in range(8)]}, seed=3)
    dup = next((tmp_path / "rf" / "train" / "images").glob("*.jpg"))
    (tmp_path / "own" / "train" / "images" / "lakers_0000.jpg").write_bytes(dup.read_bytes())
    negs = tmp_path / "negs"
    negs.mkdir()
    for i in range(3):
        _img(negs / f"crowd{i}.jpg", 900 + i)

    names = ["Player", "Ball"]
    out = tmp_path / "multi"
    m = build_multisource([Source("rf", str(tmp_path / "rf")),
                           Source("smot", str(tmp_path / "smot"), rename={"person": "Player"}, weight=2),
                           Source("own", str(tmp_path / "own"), role="test"),
                           Source("negs", str(negs), role="negatives")], names, str(out), seed=0)

    assert len(m[m.source == "own"]) == 7                          # duplicate dropped
    assert set(m[m.source == "own"].split) == {"test"}
    assert set(m[m.source == "negs"].split) == {"train"} and m[m.source == "negs"].Player.sum() == 0
    assert m[m.source == "smot"].Player.sum() == 60 and m[m.source == "smot"].Ball.sum() == 0
    # No video on both sides of a split.
    assert (m.groupby("group").split.nunique() == 1).all()
    rf_m = m[m.source == "rf"]
    assert rf_m.group.nunique() == 12 and set(rf_m.split) == {"train", "valid", "test"}
    # Training list repeats SportsMOT twice.
    train = (out / "lists" / "train.txt").read_text().split()
    smot_train = m[(m.source == "smot") & (m.split == "train")]
    assert sum(p in set(smot_train.image) for p in train) == 2 * len(smot_train)
    # Labels sit where Ultralytics looks for them, with our class ids.
    first = m[m.source == "smot"].image.iloc[0]
    assert label_path(first).read_text().split()[0] == "0"
    d = yaml.safe_load(open(out / "data_own.yaml"))
    assert d["val"] == d["test"] and open(d["test"]).read().count("\n") == 7

    folds = lodo_folds(str(out), names, min_images=10)
    assert set(folds) == {"rf", "smot"}
    f = yaml.safe_load(open(folds["smot"]))
    tr, te = set(open(f["train"]).read().split()), set(open(f["test"]).read().split())
    assert te == set(m[m.source == "smot"].image) and not tr & te
    assert not tr & set(m[m.source == "own"].image)

    k = kfold_folds(str(out), names, k=3)
    val_sets = [set(open(yaml.safe_load(open(p))["val"]).read().split()) for p in k]
    assert sum(len(v) for v in val_sets) == len(set().union(*val_sets))
    for p, v in zip(k, val_sets):
        assert not v & set(open(yaml.safe_load(open(p))["train"]).read().split())


def test_frame_scores_and_summary():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], float)
    gt_cls = np.array([0, 0, 1])
    pred = np.array([[0, 0, 10, 10], [41, 0, 51, 10], [70, 0, 75, 5]], float)
    s = frame_scores(gt, gt_cls, pred, np.array([0, 1, 1]), np.array([0.9, 0.3, 0.2]),
                     {"Player": 0, "Ball": 1}, {"Player": 0.25, "Ball": 0.1})
    assert s["Player"] == {"gt": 2, "tp": 1, "fp": 0, "fn": 1}
    assert s["Ball"] == {"gt": 1, "tp": 1, "fp": 1, "fn": 0}
    r = summarise([s, s])
    assert r["player_recall_per_frame"] == 0.5 and r["all_players_found_share"] == 0.0
    assert r["phantom_ball_frame_share"] == 1.0 and r["Ball_precision"] == 0.5


def test_mot_metrics_counts_id_switch():
    box = lambda x: [x, 0, x + 10, 20]
    gt = {f: {1: box(0), 2: box(50)} for f in range(10)}
    # The tracker swaps ids at frame 5 and misses player 2 in frame 9.
    pred = {f: ({10: box(0), 20: box(50)} if f < 5 else {20: box(0), 10: box(50)}) for f in range(10)}
    del pred[9][10]
    m = mot_metrics(gt, pred)
    assert m["id_switches"] == 2 and m["recall"] == 0.95
    assert m["mota"] == round(1 - (1 + 0 + 2) / 20, 3)
    assert m["idf1"] == round(2 * 10 / (20 + 19), 3)
    perfect = mot_metrics(gt, {f: {7: box(0), 8: box(50)} for f in range(10)})
    assert perfect["mota"] == 1.0 and perfect["idf1"] == 1.0


def test_ball_track_shot_goes_to_rim_then_rebounder():
    n = 40
    possessor = [1] * 10 + [NO_POSSESSION] * 20 + [2] * 10
    court = {1: [(20.0, 7.6)] * n, 2: [(26.0, 7.0)] * n}
    events = [Event(18, "shot", 1, 0, extra={"release_frame": 9, "made": False})]
    xy, src = ball_court_track([None] * n, possessor, court, [None] * n, NBA, events, fps=30.0)
    rim = NBA.hoops[1][:2]
    assert src[5] == "held" and xy[5] == (20.0, 7.6)
    assert src[18] == "flight" and np.allclose(xy[18], rim)
    assert np.allclose(xy[14], np.array([20.0, 7.6]) + (rim - [20.0, 7.6]) * 5 / 9)
    assert src[29] == "flight" and src[35] == "held"


def test_resample_blanks_long_gaps():
    times = np.arange(10) / 30.0
    vals = [(float(i), 0.0) for i in range(4)] + [None] * 4 + [(8.0, 0.0), (9.0, 0.0)]
    out = resample(times, vals, np.arange(0, 0.3, 0.04), max_gap_s=0.1)
    assert np.isclose(out[1, 0], 1.2)                 # 0.04 s = 1.2 frames, interpolated
    assert np.isnan(out[3, 0])                        # 0.12 s sits in the 5-frame gap


def test_near_duplicates_survive_reencoding(tmp_path):
    from courtvision.training.sources import near_duplicate_share
    rng = np.random.default_rng(0)
    refs, cands = [], []
    for i in range(3):
        img = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (15, 15), 5)
        p = tmp_path / f"ref{i}.png"
        cv2.imwrite(str(p), img)
        refs.append(str(p))
        q = tmp_path / f"fork{i}.jpg"                     # a fork re-exported at another size and quality
        cv2.imwrite(str(q), cv2.resize(img, (128, 96)), [cv2.IMWRITE_JPEG_QUALITY, 60])
        cands.append(str(q))
    new = tmp_path / "new.jpg"
    cv2.imwrite(str(new), cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (15, 15), 5))
    share, hits = near_duplicate_share(cands + [str(new)], refs)
    assert share == 0.75 and str(new) not in hits
