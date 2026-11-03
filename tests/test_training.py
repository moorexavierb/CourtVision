"""Training data: leakage-free multi-dataset build, cross-validation folds, labels, crops, scores."""
import cv2
import numpy as np
import pandas as pd
import yaml

from courtvision.court import FLIP_IDX, NBA, apply_h, orientation
from courtvision.training.crops import add_zoomed_crops, crop_pose_row
from courtvision.training.evaluate import compare, frame_scores, label_path, mot_metrics, summarise
from courtvision.training.labels import court_row_from_keypoints, fill_missing, label_orientation, pose_row
from courtvision.training.mining import FrameStats, frame_score, pick_frames
from courtvision.training.sources import (DETECT_CLASSES, Source, assign_groups, build, hold_out_groups,
                                          kfold_folds, leakage_report, lodo_folds, original_stem, video_key)
from courtvision.training import sportsmot
from conftest import floor_homography


def _img(rng, path):
    cv2.imwrite(str(path), cv2.GaussianBlur(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), (9, 9), 3))


def _dataset(root, names, items, seed):
    """items: [(stem, [class ids])], all in a train split."""
    rng = np.random.default_rng(seed)
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    for stem, classes in items:
        _img(rng, root / "train" / "images" / f"{stem}.jpg")
        (root / "train" / "labels" / f"{stem}.txt").write_text("\n".join(f"{c} 0.5 0.5 0.1 0.1" for c in classes))
    yaml.safe_dump({"names": {i: n for i, n in enumerate(names)}}, open(root / "data.yaml", "w"))


def test_group_keys():
    rf = "clip3_frame_0012_jpg.rf.0123456789abcdef0123456789abcdef"
    assert original_stem(rf) == "clip3_frame_0012" and video_key(rf) == "clip3"
    assert video_key("v_abc_c008_000123") == video_key("v_abc_c008_000456")
    stems = ["a_jpg.rf.%032x" % i for i in range(3)] + ["b_jpg.rf.%032x" % i for i in range(3)]
    assert assign_groups(stems) == ["a"] * 3 + ["b"] * 3   # too few videos: one group per original image
    assert assign_groups(["v_x__v_x_c001_000010", "v_x__v_x_c002_000010"], "prefix") == ["v_x", "v_x"]
    held = hold_out_groups([f"clip{v}_{f:03d}" for v in range(10) for f in range(5)], 0.2)
    assert held.sum() == 10 and len({f"clip{i // 5}" for i in held.nonzero()[0]}) == 2   # two whole clips


def test_sportsmot_to_yolo_groups_by_game(tmp_path):
    root = tmp_path / "sm"
    for split, seq in [("train", "v_game1_c001"), ("train", "v_game2_c003"), ("val", "v_game1_c002")]:
        d = root / split / seq
        (d / "img1").mkdir(parents=True)
        (d / "gt").mkdir()
        rng = np.random.default_rng(len(seq))
        for f in range(1, 21):
            _img(rng, d / "img1" / f"{f:06d}.jpg")
        (d / "gt" / "gt.txt").write_text("".join(f"{f},{i},{5 + 20 * i},10,8,20,1,1,1\n" for f in range(1, 21) for i in (1, 2)))
    counts = sportsmot.to_yolo(str(root), str(tmp_path / "yolo"), DETECT_CLASSES, stride=5)
    assert counts == {"frames": 12, "skipped": 0, "sequences": 3, "games": 2}
    stems = [p.stem for p in (tmp_path / "yolo" / "train" / "images").glob("*.jpg")]
    assert sorted(set(assign_groups(stems, "prefix"))) == ["v_game1", "v_game2"]   # val clip c002 joins game1
    lbl = (tmp_path / "yolo" / "train" / "labels" / "v_game1__v_game1_c001_000001.txt").read_text().split()
    assert lbl[:5] == ["0", "0.453125", "0.416667", "0.125000", "0.416667"] and len(lbl) == 10
    assert [s.name for s in sportsmot.sequences_of_games(str(root), {"v_game1"})] == ["v_game1_c001", "v_game1_c002"]


def test_build_has_no_leakage_and_folds_hold_out_whole_datasets(tmp_path):
    # v17-like: 12 videos x 4 frames, each exported twice (augmented copies).
    v17 = [(f"game{v}_frame{f:03d}_jpg.rf.{v:04d}{f:04d}{a:024d}", [0, 1, 4])
           for v in range(12) for f in range(4) for a in range(2)]
    _dataset(tmp_path / "v17", ["Ball", "Clock", "Hoop", "Overlay", "Player", "Ref", "Scoreboard"], v17, 1)
    # Basketball Game Detection style class names; one frame is a re-encoded copy of a v17 frame.
    gd = [(f"gsw_lal_{v}_{f:04d}", [0, 1, 2, 3]) for v in range(10) for f in range(5)]
    _dataset(tmp_path / "gd", ["ball", "basket", "player", "referee", "scoreboard"], gd, 2)
    copy_src = next((tmp_path / "v17" / "train" / "images").glob("game3_*"))
    img = cv2.imread(str(copy_src))
    cv2.imwrite(str(tmp_path / "gd" / "train" / "images" / "gsw_lal_0_0000.jpg"), cv2.resize(img, (80, 60)),
                [cv2.IMWRITE_JPEG_QUALITY, 70])
    # Held-out clips; one is a copy of a Game Detection frame.
    _dataset(tmp_path / "own", ["Player", "Ball"], [(f"lakers_{i:04d}", [0, 1]) for i in range(6)], 3)
    (tmp_path / "own" / "train" / "images" / "lakers_0000.jpg").write_bytes(
        (tmp_path / "gd" / "train" / "images" / "gsw_lal_5_0002.jpg").read_bytes())
    negs = tmp_path / "negs"
    negs.mkdir()
    for i in range(3):
        _img(np.random.default_rng(90 + i), negs / f"crowd{i}.jpg")

    out = tmp_path / "det"
    m = build([Source("v17", str(tmp_path / "v17")), Source("gd", str(tmp_path / "gd"), weight=2),
               Source("own", str(tmp_path / "own"), role="test"), Source("negs", str(negs), role="negatives")],
              DETECT_CLASSES, str(out), seed=0)

    report = leakage_report(m)
    assert report["components_in_two_splits"] == report["groups_in_two_splits"] == 0
    assert report["test_source_images_not_in_test"] == 0
    assert report["images_tied_across_datasets"] >= 2 + 2 + 5 + 6   # the two copies tie two videos each
    # Class names mapped: basket -> Hoop, referee -> Ref; Clock/Overlay/Scoreboard dropped.
    gd_m = m[m.source == "gd"]
    assert (gd_m[["Player", "Ref", "Ball", "Hoop"]].sum() == 50).all()
    assert set(open(label_path(gd_m.image.iloc[1])).read().split()[::5]) == {"0", "1", "2", "3"}
    # The re-encoded copy is tied to its v17 video; the copied held-out frame drags its video to test.
    copy = m[m.image.str.endswith("gsw_lal_0_0000.jpg")].iloc[0]
    assert copy.component in set(m[m.group == "v17/game3"].component)
    assert set(m[m.group == "gd/gsw_lal_5"].split) == {"test"}
    assert set(m[m.source == "negs"].split) == {"train"}
    assert {"train", "valid", "test"} <= set(m[m.source == "v17"].split)
    train = (out / "lists" / "train.txt").read_text().split()
    gd_train = gd_m[gd_m.split == "train"].image
    assert sum(p in set(gd_train) for p in train) == 2 * len(gd_train)   # weight 2

    folds = lodo_folds(str(out), min_images=10)
    assert set(folds) == {"v17", "gd"}
    f = yaml.safe_load(open(folds["gd"]))
    tr, te = set(open(f["train"]).read().split()), set(open(f["test"]).read().split())
    assert te == set(gd_m.image) and not tr & te
    assert not set(m[m.component.isin(set(gd_m.component))].image) & tr   # v17 game3 (copy's video) left out
    assert not tr & set(m[m.source == "own"].image)

    val_sets = []
    for p in kfold_folds(str(out), k=3):
        y = yaml.safe_load(open(p))
        v, t = set(open(y["val"]).read().split()), set(open(y["train"]).read().split())
        assert not v & t
        val_sets.append(v)
    assert sum(len(v) for v in val_sets) == len(set().union(*val_sets))


def test_court_label_from_keypoints_adds_missed_landmarks(broadcast_cam):
    P, *_, size = broadcast_cam
    img = apply_h(np.linalg.inv(floor_homography(P)), NBA.keypoints)
    inside = (img[:, 0] >= 0) & (img[:, 0] < size[0]) & (img[:, 1] >= 0) & (img[:, 1] < size[1])
    kps = np.column_stack([img, np.where(inside, 0.9, 0.0)])
    kps[np.flatnonzero(inside)[:3], 2] = 0.1                    # the model misses three landmarks
    sign = orientation(floor_homography(P), img[inside])
    row, info = court_row_from_keypoints(kps, NBA, size, orientation_sign=sign)
    assert row is not None and info["added"] == 3
    v = np.array(row.split()[5:], float).reshape(-1, 3)
    assert ((v[:, 2] > 0) == inside).all()
    assert np.allclose(v[inside, :2] * size, img[inside], atol=0.5)
    bad = kps.copy()
    bad[np.flatnonzero(inside)[4], :2] += 60                     # one keypoint in the wrong place
    assert court_row_from_keypoints(bad, NBA, size)[0] is None
    flipped = kps[FLIP_IDX]                                      # left/right swapped labels
    assert court_row_from_keypoints(flipped, NBA, size)[0] is not None     # fits fine as a mirror image
    assert court_row_from_keypoints(flipped, NBA, size, orientation_sign=sign)[1]["reason"] == "mirrored"
    # The sign real views have is read off hand labels; a mirrored labelling reads the other way.
    assert label_orientation(row, NBA) == sign
    assert label_orientation(pose_row(flipped[:, :2], flipped[:, 2] > 0.5, size), NBA) == -sign


def test_fill_missing_classes():
    gt = np.array([[0, 0, 10, 20]], float)
    pred = np.array([[1, 1, 10, 20], [50, 50, 55, 55], [80, 0, 90, 20]], float)
    xyxy, cls = fill_missing(gt, [0], pred, [1, 2, 1], [0.9, 0.8, 0.7], fill_classes=[1, 2])
    assert cls.tolist() == [0, 2, 1]                             # ref box on top of a player dropped
    assert fill_missing(gt, [0], pred, [1, 2, 1], [0.9, 0.3, 0.7], fill_classes=[1, 2]) is None


def test_zoomed_crops_inherit_split(tmp_path):
    root = tmp_path / "kp"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    rng = np.random.default_rng(0)
    xs = np.linspace(0.1, 0.9, 6)
    row = "0 0.5 0.5 0.8 0.6 " + " ".join(f"{x:.3f} {y:.3f} 2" for x in xs for y in (0.4, 0.6, 0.5))
    for i in range(12):
        _img(rng, root / "train" / "images" / f"court{i}_{i:03d}.jpg")
        (root / "train" / "labels" / f"court{i}_{i:03d}.txt").write_text(row + "\n")
    yaml.safe_dump({"names": {0: "court"}, "kpt_shape": [18, 3]}, open(root / "data.yaml", "w"))
    out = tmp_path / "built"
    m = build([Source("kp", str(root))], ["court"], str(out), kpt_shape=[18, 3], flip_idx=FLIP_IDX)
    assert add_zoomed_crops(str(out), ["kp"], per_image=1) > 0
    m2 = pd.read_csv(out / "manifest.csv")
    crops = m2[m2.crop_of.notna()]
    assert len(crops) == (m.split == "train").sum() and add_zoomed_crops(str(out), ["kp"], per_image=1) == 0  # not cropped twice
    parent = crops.crop_of.map(dict(zip(m.image, m.split)))
    assert (crops.split.to_numpy() == parent.to_numpy()).all() and (crops.split == "train").all()
    assert yaml.safe_load(open(out / "data.yaml"))["flip_idx"] == FLIP_IDX
    assert crop_pose_row([0, .5, .5, 1, 1] + [0.9, 0.5, 2] * 18, (0, 0, 0.5, 1)) is None


def test_frame_scores_and_mot():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], float)
    s = frame_scores(gt, np.array([0, 0, 2]), np.array([[0, 0, 10, 10], [41, 0, 51, 10], [70, 0, 75, 5]], float),
                     np.array([0, 2, 2]), np.array([0.9, 0.3, 0.2]), {"Player": 0, "Ball": 2}, {"Player": .25, "Ball": .1})
    r = summarise([s, s])
    assert r["player_recall_per_frame"] == 0.5 and r["phantom_ball_frame_share"] == 1.0 and r["Ball_precision"] == 0.5
    box = lambda x: [x, 0, x + 10, 20]
    gt_t = {f: {1: box(0), 2: box(50)} for f in range(10)}
    pred = {f: ({10: box(0), 20: box(50)} if f < 5 else {20: box(0), 10: box(50)}) for f in range(10)}
    del pred[9][10]
    mm = mot_metrics(gt_t, pred)
    assert mm["id_switches"] == 2 and mm["mota"] == round(1 - 3 / 20, 3) and mm["idf1"] == round(20 / 39, 3)


def test_compare_new_and_current():
    cur = {"player_recall_per_frame": 0.80, "Ball_recall": 0.70, "phantom_ball_frame_share": 0.10}
    ok, why = compare({**cur, "player_recall_per_frame": 0.85}, cur, ["player_recall_per_frame", "Ball_recall"],
                      ["phantom_ball_frame_share"])
    assert ok and not why
    ok, why = compare({**cur, "phantom_ball_frame_share": 0.2, "Ball_recall": None}, cur, ["Ball_recall"],
                      ["phantom_ball_frame_share"])
    assert not ok and why == ["phantom_ball_frame_share: new 0.2 vs current 0.1"]


def test_mining_prefers_hard_frames_spaced_out():
    stats = [FrameStats(frame=f, players=10, balls=1, keypoints=10) for f in range(0, 300, 10)]
    stats[5].players, stats[6].players, stats[20].balls = 4, 5, 0
    assert frame_score(stats[5]) > frame_score(stats[0])
    assert pick_frames(stats, 2, min_gap=30) == [50, 200]


def test_sportsmot_basketball_zip_keeps_only_basketball(tmp_path):
    import tarfile, zipfile
    names = {"sportsmot_publish/splits_txt/basketball.txt": b"v_b1_c001\nv_b2_c001\n",
             "sportsmot_publish/splits_txt/football.txt": b"v_f1_c001\n"}
    for split, seq in [("train", "v_b1_c001"), ("val", "v_b2_c001"), ("train", "v_f1_c001"), ("test", "v_b3_c001")]:
        names[f"sportsmot_publish/dataset/{split}/{seq}/img1/000001.jpg"] = b"jpg"
        names[f"sportsmot_publish/dataset/{split}/{seq}/gt/gt.txt"] = b"1,1,0,0,1,1,1,1,1\n"
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "sportsmot_publish.zip", "w") as z:
        for k, v in names.items():
            z.writestr(k, v)
    info = sportsmot.basketball_zip(str(raw), str(tmp_path / "bb.zip"))
    assert info["sequences"] == {"train": 1, "val": 1}
    got = set(zipfile.ZipFile(tmp_path / "bb.zip").namelist())
    assert got == {"splits_txt/basketball.txt", "splits_txt/football.txt", "train/v_b1_c001/img1/000001.jpg",
                   "train/v_b1_c001/gt/gt.txt", "val/v_b2_c001/img1/000001.jpg", "val/v_b2_c001/gt/gt.txt"}
    root = sportsmot.unpack(str(tmp_path / "bb.zip"), str(tmp_path / "sm"))
    assert [s.name for s in sportsmot.basketball_sequences(root, "train")] == ["v_b1_c001"]
    # The same from a .tar.gz download.
    raw2 = tmp_path / "raw2"
    raw2.mkdir()
    import io
    with tarfile.open(raw2 / "sportsmot.tar.gz", "w:gz") as t:
        for k, v in names.items():
            ti = tarfile.TarInfo(k)
            ti.size = len(v)
            t.addfile(ti, io.BytesIO(v))
    assert sportsmot.basketball_zip(str(raw2), str(tmp_path / "bb2.zip"))["files"] == 6
