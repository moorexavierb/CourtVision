"""Edge cases of the dataset build, TrackEval scores and the game-clip evaluation."""
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from courtvision.training import clip_eval
from courtvision.training.evaluate import label_path, mot_metrics, mot_table
from courtvision.training.labels import fill_dataset, labelled_classes
from courtvision.training.sources import (DETECT_CLASSES, Source, build, clean_label, leakage_report,
                                          near_duplicate_pairs, resolution_report)


def _img(path, seed, size=(48, 64)):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), cv2.GaussianBlur(rng.integers(0, 255, (*size, 3), dtype=np.uint8), (9, 9), 3))


def _dataset(root, names, items):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    for i, (stem, rows) in enumerate(items):
        _img(root / "train" / "images" / f"{stem}.jpg", i)
        if rows is not None:
            (root / "train" / "labels" / f"{stem}.txt").write_text("\n".join(rows) + "\n")
    yaml.safe_dump({"names": dict(enumerate(names))}, open(root / "data.yaml", "w"))


def test_clean_label_fixes_and_drops_bad_rows():
    rows, notes = clean_label(["0 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2",           # duplicate
                               "1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4",                # polygon -> box
                               "0 0.95 0.5 0.2 0.2",                               # clipped at the right edge
                               "0 0.5 0.5 0 0.1", "x 1 2 3 4", "7 0.5 0.5 0.1 0.1",  # no area, bad, unknown id
                               "2 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1"],          # class we drop, NaN
                              {0: 0, 1: 2}, 3)
    assert rows[0] == "0 0.500000 0.500000 0.200000 0.200000"
    assert rows[1] == "2 0.200000 0.250000 0.200000 0.300000"
    assert rows[2] == "0 0.925000 0.500000 0.150000 0.200000"
    assert len(rows) == 3
    assert notes == {"bad_rows": 2, "unknown_class": 1, "polygon_to_box": 1, "clipped": 1, "no_area": 1,
                     "duplicates": 1, "wrong_keypoint_count": 0, "dropped_class": 1}
    pose, n = clean_label(["0 0.5 0.5 0.4 0.4 " + " ".join(["0.5 0.5"] * 18), "0 0.5 0.5 0.4 0.4 0.1 0.1 2"],
                          {0: 0}, 1, kpt_shape=[18, 3])
    assert len(pose) == 1 and pose[0].split()[5:8] == ["0.5", "0.5", "2"] and n["wrong_keypoint_count"] == 1


def test_build_survives_bad_files_and_flags_unmapped_players(tmp_path):
    items = [(f"clip{v}_{f:03d}", ["0 0.5 0.5 0.2 0.3"]) for v in range(12) for f in range(3)]
    items += [("clip99_000", None)]                         # no label file: unknown, left out
    items += [(f"empty{i}", []) for i in range(6)]            # empty label files: background
    _dataset(tmp_path / "a", ["player", "scoreboard"], items)
    imgs = tmp_path / "a" / "train" / "images"
    (imgs / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "a" / "train" / "labels" / "broken.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    good = (imgs / "clip0_000.jpg").read_bytes()
    (imgs / "clip0_001.jpg").write_bytes(good[:-2])          # truncated jpeg: repaired
    negs = tmp_path / "negs"
    negs.mkdir()
    for i in range(20):
        _img(negs / f"crowd{i}.jpg", 100 + i)
    m = build([Source("a", str(tmp_path / "a")), Source("negs", str(negs), role="negatives")], DETECT_CLASSES,
              str(tmp_path / "out"), max_background=0.2)
    c = m.attrs["checks"]["a"]
    assert c["unreadable"] == 1 and c["no_label_file"] == 1 and c["repaired_jpeg"] == 1 and c["empty_label"] == 6
    assert m.attrs["classes"]["a"] == {"player": "Player", "scoreboard": None}
    train = m[m.split == "train"]
    assert train.background.sum() <= 0.2 / 0.8 * (~train.background).sum() + 1
    assert (m.split == "unused").sum() > 0
    if (m[m.source == "a"].split == "unused").any():   # in-domain empty frames go only after every negative
        assert (m[m.source == "negs"].split == "unused").all()
    assert leakage_report(m)["components_in_two_splits"] == 0
    assert (tmp_path / "out" / "checks.yaml").exists()
    assert resolution_report(m, 1280).loc["a", "upscaled_share"] == 1.0

    _dataset(tmp_path / "b", ["team_a", "team_b", "ball"], [("x_000", ["0 0.5 0.5 0.1 0.1"])])
    with pytest.raises(ValueError, match="team_a"):
        build([Source("b", str(tmp_path / "b"))], DETECT_CLASSES, str(tmp_path / "out2"))
    m2 = build([Source("b", str(tmp_path / "b"), rename={"team_a": "Player", "team_b": "Player"})], DETECT_CLASSES,
               str(tmp_path / "out2"))
    assert m2.Player.sum() == 1


def test_flat_images_are_not_near_duplicates():
    hashes = np.array([0, 0, 1, 2**64 - 1, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0E], dtype=np.uint64)
    assert near_duplicate_pairs(hashes, 6) == [(4, 5)]


def test_fill_dataset_adds_missing_classes(tmp_path):
    _dataset(tmp_path / "gd", ["player"], [(f"f{i}", ["0 0.25 0.5 0.1 0.2"]) for i in range(3)])
    assert labelled_classes(tmp_path / "gd", DETECT_CLASSES) == {"Player": 3, "Ref": 0, "Ball": 0, "Hoop": 0}
    calls = iter([0.9, 0.3, 0.9])   # second frame: unsure referee -> skipped

    def predict(img):
        h, w = img.shape[:2]
        return (np.array([[0.6 * w, 0.2 * h, 0.7 * w, 0.8 * h], [0.2 * w, 0.3 * h, 0.3 * w, 0.7 * h]]),
                np.array([1, 0]), np.array([next(calls), 0.95]))

    counts = fill_dataset(tmp_path / "gd", tmp_path / "filled", DETECT_CLASSES, ["Ref"], predict)
    assert counts == {"images": 2, "skipped_unsure": 1, "unreadable": 0, "boxes_added": 2}
    assert labelled_classes(tmp_path / "filled", DETECT_CLASSES) == {"Player": 2, "Ref": 2, "Ball": 0, "Hoop": 0}


def _toy_tracks(n=60, switch_at=30):
    gt = {f: {i: [40 * i + f, 10, 40 * i + f + 20, 60] for i in range(4)} for f in range(1, n + 1)}
    pred = {f: {(i if f < switch_at or i else 9): [40 * i + f + 1, 10, 40 * i + f + 20, 60] for i in range(4) if not (i == 3 and f % 7 == 0)}
            for f in range(1, n + 1)}
    return gt, pred


def test_trackeval_and_simple_agree_and_hota_is_one_when_perfect():
    gt, pred = _toy_tracks()
    te, simple = mot_metrics(gt, pred), mot_metrics(gt, pred, backend="simple")
    assert te["metrics_by"] == "TrackEval"
    for k in ["mota", "idf1", "id_switches", "recall", "precision"]:
        assert te[k] == pytest.approx(simple[k], abs=1e-3), k
    perfect = mot_metrics(gt, gt)
    assert perfect["hota"] == perfect["mota"] == perfect["idf1"] == 1.0
    assert 0 < te["hota"] < 1 and te["id_switches"] == 1
    table = mot_table({"a": te, "b": perfect})
    assert table.loc["overall", "metrics_by"] == "TrackEval" and table.loc["a", "hota"] < table.loc["overall", "hota"] < 1


def _sequence(root, n=40):
    (root / "img1").mkdir(parents=True)
    (root / "gt").mkdir()
    for f in range(1, n + 1):
        img = np.full((180, 320, 3), 90, np.uint8)
        cv2.imwrite(str(root / "img1" / f"{f:06d}.jpg"), img)
    (root / "gt" / "gt.txt").write_text("".join(f"{f},{i},{20 + 60 * i + f},40,30,80,1,1,1\n"
                                                for f in range(1, n + 1) for i in (1, 2, 3)))
    (root / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\n")


def test_evaluate_gt_clip_end_to_end_with_a_stand_in_pipeline(tmp_path, monkeypatch):
    seq = tmp_path / "v_game_c001"
    _sequence(seq)
    from courtvision import pipeline
    from courtvision.config import Config

    def fake_run(video, cfg, out_dir=None, write_outputs=True):
        rows = []
        for f in range(40):   # frame f is SportsMOT frame f + 1; player 3's track breaks at frame 20
            for i in (1, 2, 3):
                tid = 30 if (i == 3 and f >= 20) else i
                x = 20 + 60 * i + f + 1
                rows.append((f, tid, "player", x, 40, x + 30, 120, 0.9))
            rows.append((f, 50, "player", 0, 0, 10, 10, 0.5))   # a box on nobody
        tracks = pd.DataFrame(rows, columns=["frame", "track_id", "role", "x1", "y1", "x2", "y2", "conf"])
        return SimpleNamespace(tracks=tracks, homographies=[None] * 40, remap={30: 3}, fps=25.0,
                               positions=pd.DataFrame(columns=["frame", "player_id", "x_m", "y_m"]),
                               diagnostics={"court": {"mapped_share": 0.0}})

    monkeypatch.setattr(pipeline, "run", fake_run)
    out = clip_eval.evaluate_gt_clip(seq, Config(), tmp_path / "out", max_frames=40)
    assert out["raw"]["id_switches"] == 1 and out["linked"]["id_switches"] == 0   # relinking mends the break
    assert out["linked"]["recall"] == 1.0 and out["linked"]["precision"] == 0.75
    for f in ["metrics.json", "players_image.csv", "gt_mot.txt", "pred_mot.txt", "id_timeline.png", "counts.png", "gt_vs_pred.mp4"]:
        assert (tmp_path / "out" / f).exists(), f
    img = pd.read_csv(tmp_path / "out" / "players_image.csv")
    assert set(img[img.player_id == 50].gt_id) == {-1} and set(img[img.player_id == 3].gt_id) == {3}
    table = clip_eval.summary_table({"c1": out})
    assert table.loc["overall", "idf1"] == 1.0 or table.loc["overall", "idf1"] > 0.8


def test_sanity_checks_catch_impossible_tracking(tmp_path):
    rows = []
    for s in range(30):
        for pid in range(12):   # twelve players, six a team, ids 0 and 1 on one spot
            x = 5.0 + pid * 2 if pid != 1 else 5.1
            if pid == 11 and s == 15:
                x += 4   # a jump of 4 m in 0.1 s
            rows.append((s, s / 10, s * 3, "player", pid, pid % 2, x, 7.0, 20.0 if pid == 5 else 3.0, False, "tracked"))
    t = pd.DataFrame(rows, columns=["sample", "time_s", "frame", "object", "id", "team", "x_m", "y_m", "speed_mps", "has_ball", "source"])
    c = clip_eval.sanity_checks(t, {"dataset": {"samples": 30}})
    assert c["over_ten_share"] == 1.0 and c["over_five_a_team_share"] == 1.0
    assert c["too_fast_share"] == pytest.approx(1 / 12, abs=1e-3) and c["jumps_over_1.5m_share"] > 0
    assert c["same_spot_pairs"] == 1
    from courtvision.court import NBA
    assert clip_eval.plot_dataset(t, NBA, tmp_path / "d.png").exists()
