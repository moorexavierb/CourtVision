import random

import cv2
import numpy as np
import yaml

from courtvision.training.court_crops import crop_pose_label, make_zoomed_crops, random_window
from courtvision.training.datasets import class_counts, merge_datasets, resplit, write_data_yaml
from courtvision.training.mining import FrameStats, frame_score, pick_frames, pose_row, yolo_rows


def _make_dataset(root, names, per_split):
    for split, items in per_split.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i, classes in enumerate(items):
            cv2.imwrite(str(root / split / "images" / f"im{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
            rows = [f"{c} 0.5 0.5 0.1 0.1" for c in classes]
            (root / split / "labels" / f"im{i}.txt").write_text("\n".join(rows))
    write_data_yaml(root, names)


def test_merge_renames_and_drops(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _make_dataset(a, ["Ball", "Player"], {"train": [[0, 1]], "valid": [[1]]})
    _make_dataset(b, ["basketball", "person", "rim"], {"train": [[0, 1, 2]]})
    negs = tmp_path / "negs"
    negs.mkdir()
    cv2.imwrite(str(negs / "crowd.jpg"), np.zeros((8, 8, 3), np.uint8))

    out = tmp_path / "merged"
    counts = merge_datasets([(str(a), None), (str(b), {"basketball": "Ball", "person": "Player", "rim": None})],
                            names=["Player", "Ball"], out=str(out), negatives=str(negs))
    assert counts == {"Player": 3, "Ball": 2}
    assert (out / "train" / "labels" / "neg_crowd.txt").read_text() == ""
    assert sorted(l.split()[0] for l in open(out / "train" / "labels" / "s1_im0.txt")) == ["0", "1"]
    assert yaml.safe_load(open(out / "data.yaml"))["names"] == {0: "Player", 1: "Ball"}
    assert class_counts(str(out))["train"]["images"] == 3


def test_resplit_stratifies_rare_class(tmp_path):
    root = tmp_path / "d"
    # 20 ball images and 80 player-only images, all starting in train.
    _make_dataset(root, ["Ball", "Player"], {"train": [[0, 1]] * 20 + [[1]] * 80, "valid": [[1]] * 2})
    sizes = resplit(str(root), val_frac=0.2, stratify_class=0)
    assert sizes == {"train": 82, "valid": 20}  # 4 of 20 ball images + 16 of 82 others
    counts = class_counts(str(root))
    assert counts["valid"]["Ball"] == 4 and counts["train"]["Ball"] == 16


def test_crop_pose_label_hides_outside_points():
    kpts = np.zeros((18, 3))
    kpts[:, 0] = np.linspace(0.05, 0.95, 18)
    kpts[:, 1] = 0.5
    kpts[:, 2] = 2
    row = [0, 0.5, 0.5, 0.9, 0.4] + kpts.ravel().tolist()
    out = crop_pose_label(row, (0.0, 0.25, 0.5, 0.75))
    k = np.array(out[5:]).reshape(-1, 3)
    assert (k[:9, 2] == 2).all() and (k[9:, 2] == 0).all()
    assert np.allclose(k[:9, 0], kpts[:9, 0] / 0.5) and np.allclose(k[:9, 1], 0.5)
    assert np.allclose(out[1:5], [0.55, 0.5, 0.9, 0.8])
    assert crop_pose_label(row, (0.45, 0.25, 0.5, 0.3)) is None  # too few points left


def test_random_window_is_square_in_normalised_units():
    rng = random.Random(1)
    for _ in range(50):
        x0, y0, x1, y1 = random_window(rng)
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert abs((x1 - x0) - (y1 - y0)) < 1e-9


def test_make_zoomed_crops_writes_images_and_labels(tmp_path):
    root = tmp_path / "court"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "train" / "images" / "c.jpg"), np.full((90, 160, 3), 128, np.uint8))
    kpts = np.stack([np.tile(np.linspace(0.05, 0.95, 6), 3), np.repeat([0.4, 0.5, 0.6], 6), np.full(18, 2)], 1)
    (root / "train" / "labels" / "c.txt").write_text(" ".join(map(str, [0, 0.5, 0.5, 0.9, 0.3] + kpts.ravel().tolist())))
    n = make_zoomed_crops(str(root), per_image=2, zoom_range=(1.5, 2.0))
    assert n == 2
    crop = cv2.imread(str(root / "train" / "images" / "c_zoom0.jpg"))
    assert crop.shape == (90, 160, 3)
    vals = np.array(open(root / "train" / "labels" / "c_zoom0.txt").read().split(), float)[5:].reshape(-1, 3)
    assert 4 <= (vals[:, 2] > 0).sum() < 18
    assert make_zoomed_crops(str(root), per_image=2) == 2  # existing crops are not cropped again


def test_mining_prefers_hard_frames_and_spaces_them():
    stats = [FrameStats(frame=f, players=10, balls=1, keypoints=10) for f in range(100)]
    stats[10] = FrameStats(frame=10, players=4, balls=0, keypoints=3)
    stats[12] = FrameStats(frame=12, players=5, balls=3)        # close to 10: skipped
    stats[60] = FrameStats(frame=60, players=9, balls=2)
    assert frame_score(stats[10]) > frame_score(stats[60]) > frame_score(stats[0])
    assert pick_frames(stats, k=2, min_gap=15) == [10, 60]


def test_label_rows():
    rows = yolo_rows(np.array([[100, 50, 200, 250]]), np.array([3]), (1000, 500))
    assert rows == ["3 0.150000 0.300000 0.100000 0.400000"]
    xy = np.array([[100, 100], [300, 200], [0, 0]])
    r = pose_row(xy, np.array([0.9, 0.8, 0.1]), (400, 400)).split()
    assert r[0] == "0" and len(r) == 5 + 9 and r[-3:] == ["0", "0", "0"]


def test_overlap_and_blur_raise_score():
    from courtvision.training.mining import overlap_pairs, sharpness
    boxes = np.array([[0, 0, 10, 20], [5, 0, 15, 20], [100, 100, 110, 120]])
    assert overlap_pairs(boxes) == 1 and overlap_pairs(boxes[:1]) == 0
    base = FrameStats(frame=0, players=10, balls=1)
    assert frame_score(FrameStats(frame=0, players=10, balls=1, overlaps=3)) > frame_score(base)
    assert frame_score(FrameStats(frame=0, players=10, balls=1, sharpness=20.0)) > frame_score(base)
    sharp = np.zeros((64, 64), np.uint8)
    sharp[::2] = 255
    assert sharpness(sharp) > sharpness(cv2.GaussianBlur(sharp, (9, 9), 3))


def test_sportsmot_reading_and_pseudo_label_merge(tmp_path):
    from courtvision.training.sportsmot import basketball_sequences, merge_gt_and_pseudo, read_gt
    for name in ["bball_1", "soccer_1"]:
        (tmp_path / "train" / name / "img1").mkdir(parents=True)
        (tmp_path / "train" / name / "gt").mkdir()
        (tmp_path / "train" / name / "gt" / "gt.txt").write_text("1,1,10,20,30,40,1,1,1\n1,2,0,0,5,5,0,1,1\n2,1,11,20,30,40,1,1,1\n")
    (tmp_path / "splits_txt").mkdir()
    (tmp_path / "splits_txt" / "basketball.txt").write_text("bball_1\n")
    seqs = basketball_sequences(str(tmp_path), "train")
    assert [s.name for s in seqs] == ["bball_1"]
    gt = read_gt(seqs[0])
    assert np.allclose(gt[1], [[10, 20, 40, 60]]) and len(gt[2]) == 1  # ignore row dropped

    PLAYER, BALL, REF = 4, 0, 5
    pred = np.array([[0, 0, 5, 5], [50, 50, 55, 55], [70, 0, 90, 60]])
    out = merge_gt_and_pseudo(gt[1], pred, [PLAYER, BALL, REF], [0.9, 0.8, 0.7], PLAYER, [BALL, REF])
    assert out is not None and list(out[1]) == [PLAYER, BALL, REF]  # detector player dropped, GT kept
    assert merge_gt_and_pseudo(gt[1], pred, [PLAYER, BALL, REF], [0.9, 0.3, 0.7], PLAYER, [BALL, REF]) is None
