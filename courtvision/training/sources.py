"""Train on several detection datasets at once and measure each one separately.

PR #5's ``merge_datasets`` + ``resplit`` pooled every image and split at random.
That has two problems once several datasets are in play:

* **Leakage.** Neighbouring video frames, and Roboflow's augmented copies of one
  image (``frame12_jpg.rf.<hash>.jpg`` three times with different hashes), land on
  both sides of the split. Validation then scores memorised frames, which is how
  0.94 validation recall turned into 6 of 10 players on a real clip.
* **One number for everything.** A pooled mAP is dominated by the largest dataset
  and says nothing about whether the model works on footage it has not seen.

Here every image keeps its source dataset and a *group* (the video, sequence or
original image it came from). Splits are made per source and by group, so no
video is on both sides. The build writes:

    <out>/images/<source>/...           <out>/labels/<source>/...
    <out>/manifest.csv                  one row per image: source, group, split, boxes per class
    <out>/data.yaml                     train on every source, validate on every source
    <out>/data_<source>.yaml            the same model scored on one source (val and test)
    <out>/folds/lodo_<source>.yaml      leave-one-dataset-out: train on the others, test on this one
    <out>/folds/kfold_<i>.yaml          grouped k-fold over the pooled training data (optional)

Leave-one-dataset-out (LODO) is the cross-dataset check: a detector that holds up
on a dataset it never trained on will hold up on a new broadcast. Roles:

* ``train``: split into train / valid / test by group.
* ``test``: never trained on; all of it is test (e.g. hand-labelled frames from
  clips you keep back to score the pipeline).
* ``negatives``: images with nothing to label (crowd, bench, replays); train only.
"""
from __future__ import annotations

import hashlib
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .datasets import IMAGE_EXTS, SPLITS, _images, _label_for, _names

_RF_SUFFIX = re.compile(r"(_(jpg|jpeg|png|webp|bmp))?\.rf\.[0-9a-f]{16,}$", re.I)
_FRAME_SUFFIX = re.compile(r"[_\-. ]?(frame|frm|img|image|f)?[_\-. ]?\d+$", re.I)


@dataclass
class Source:
    name: str                                         # short id, used in file names
    root: str                                         # YOLO dataset root with data.yaml (or an image folder for negatives)
    rename: Optional[dict[str, Optional[str]]] = None  # source class -> our class, None drops it
    role: str = "train"                               # train | test | negatives
    group_by: str = "auto"                            # auto | video | image
    weight: int = 1                                   # repeat its training images this many times
    keep_splits: bool = False                         # keep the source's own train/valid/test


def original_stem(stem: str) -> str:
    """File stem without Roboflow's export suffix, so augmented copies share one name."""
    return _RF_SUFFIX.sub("", stem)


def video_key(stem: str) -> str:
    """Best guess at the video a frame came from: the original stem minus a trailing frame number."""
    base = original_stem(stem)
    key = _FRAME_SUFFIX.sub("", base)
    return key or base


def assign_groups(stems: list[str], group_by: str = "auto", min_groups: int = 10) -> list[str]:
    """Group id per image. auto: by video if that gives at least min_groups groups, else by
    original image (which still keeps augmented copies together)."""
    if group_by == "image":
        return [original_stem(s) for s in stems]
    videos = [video_key(s) for s in stems]
    if group_by == "video" or len(set(videos)) >= min_groups:
        return videos
    return [original_stem(s) for s in stems]


def split_groups(groups: list[str], val_frac: float, test_frac: float, seed: int = 0) -> dict[str, str]:
    """Assign whole groups to test, valid, train so each split gets about its share of images."""
    sizes: dict[str, int] = defaultdict(int)
    for g in groups:
        sizes[g] += 1
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    total = sum(sizes.values())
    targets = [("test", test_frac * total), ("valid", val_frac * total)]
    out, filled, k = {}, 0, 0
    for g in order:
        while k < len(targets) and filled >= targets[k][1]:
            k, filled = k + 1, 0
        if k < len(targets) and len(sizes) - len(out) > 1:
            out[g] = targets[k][0]
            filled += sizes[g]
        else:
            out[g] = "train"
    return out


def _file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build_multisource(sources: list[Source], names: list[str], out: str, val_frac: float = 0.15,
                      test_frac: float = 0.15, seed: int = 0, dedupe: bool = True) -> pd.DataFrame:
    """Copy every source into `out` with our class list and write the manifest and the yamls.

    Returns the manifest (one row per image). Exact duplicate images across sources
    (Roboflow Universe sets are often forks of each other) are kept once, in the first source.
    """
    out_root = Path(out)
    shutil.rmtree(out_root, ignore_errors=True)
    rows, seen = [], {}
    for src in sources:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", src.name):
            raise ValueError(f"source name '{src.name}' must be letters, digits, - or _")
        root = Path(src.root)
        if src.role == "negatives":
            items = [(img, None, "train") for img in _images(root)]
            to_dst: dict[int, int] = {}
        else:
            src_names = _names(root / "data.yaml")
            rename = src.rename or {}
            to_dst = {i: names.index(rename.get(n, n)) for i, n in enumerate(src_names)
                      if rename.get(n, n) is not None and rename.get(n, n) in names}
            items = [(img, _label_for(img), split) for split in SPLITS if (root / split / "images").exists()
                     for img in _images(root / split / "images")]
        stems = [img.stem for img, _, _ in items]
        groups = assign_groups(stems, src.group_by)
        if src.role == "test":
            split_of = {g: "test" for g in groups}
        elif src.role == "negatives":
            split_of = {g: "train" for g in groups}
        elif not src.keep_splits:
            split_of = split_groups(groups, val_frac, test_frac, seed)
        else:
            split_of = None
        img_dir, lbl_dir = out_root / "images" / src.name, out_root / "labels" / src.name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        used = set()
        for (img, lbl, orig_split), group in zip(items, groups):
            if dedupe:
                h = _file_hash(img)
                if h in seen:
                    continue
                seen[h] = src.name
            split = split_of[group] if split_of is not None else orig_split
            stem = img.stem if img.stem not in used else f"{orig_split}_{img.stem}"
            used.add(stem)
            counts = {n: 0 for n in names}
            label_rows = []
            if lbl is not None and lbl.exists():
                for line in open(lbl):
                    parts = line.split()
                    if not parts or int(parts[0]) not in to_dst:
                        continue
                    cls = to_dst[int(parts[0])]
                    counts[names[cls]] += 1
                    label_rows.append(" ".join([str(cls)] + parts[1:]))
            dst = img_dir / (stem + img.suffix.lower())
            shutil.copy(img, dst)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(label_rows) + ("\n" if label_rows else ""))
            rows.append({"image": str(dst), "source": src.name, "role": src.role, "group": f"{src.name}/{group}",
                         "split": split, "weight": src.weight, **counts})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_root / "manifest.csv", index=False)
    write_split_files(manifest, names, out_root)
    return manifest


def _write_list(path: Path, images: list[str]) -> str:
    path.write_text("\n".join(images) + ("\n" if images else ""))
    return str(path)


def _yaml(path: Path, root: Path, names: list[str], train: str, val: str, test: Optional[str] = None) -> Path:
    cfg = {"path": str(root), "train": train, "val": val, "names": {i: n for i, n in enumerate(names)}}
    if test:
        cfg["test"] = test
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


def write_split_files(manifest: pd.DataFrame, names: list[str], root: Path) -> None:
    lists = root / "lists"
    lists.mkdir(exist_ok=True)

    def imgs(df, weighted=False):
        if not weighted:
            return df["image"].tolist()
        return [p for p, w in zip(df["image"], df["weight"]) for _ in range(int(w))]

    tr = _write_list(lists / "train.txt", imgs(manifest[manifest.split == "train"], weighted=True))
    va = _write_list(lists / "val.txt", imgs(manifest[manifest.split == "valid"]))
    te = _write_list(lists / "test.txt", imgs(manifest[manifest.split == "test"]))
    _yaml(root / "data.yaml", root, names, tr, va, te)
    for name, df in manifest.groupby("source"):
        v = _write_list(lists / f"val_{name}.txt", imgs(df[df.split == "valid"]))
        t = _write_list(lists / f"test_{name}.txt", imgs(df[df.split == "test"]))
        # An eval-only source has no valid split; score its test images under both names.
        _yaml(root / f"data_{name}.yaml", root, names, tr, v if (df.split == "valid").any() else t, t)


def lodo_folds(root: str, names: list[str], min_images: int = 50) -> dict[str, Path]:
    """Leave-one-dataset-out folds: for each training source, train (and early-stop) on the
    other sources and test on every image of the held-out one. Returns {source: yaml}."""
    root_p = Path(root)
    manifest = pd.read_csv(root_p / "manifest.csv")
    folds = root_p / "folds"
    folds.mkdir(exist_ok=True)
    out = {}
    train_sources = sorted(manifest[manifest.role == "train"].source.unique())
    for held in train_sources:
        rest = manifest[(manifest.source != held) & (manifest.role != "test")]
        mine = manifest[manifest.source == held]
        if len(mine) < min_images or rest[rest.split == "train"].empty:
            continue
        tr = _write_list(folds / f"lodo_{held}_train.txt",
                         [p for p, w in zip(rest[rest.split == "train"].image, rest[rest.split == "train"].weight)
                          for _ in range(int(w))])
        va = _write_list(folds / f"lodo_{held}_val.txt", rest[rest.split == "valid"].image.tolist())
        te = _write_list(folds / f"lodo_{held}_test.txt", mine.image.tolist())
        out[held] = _yaml(folds / f"lodo_{held}.yaml", root_p, names, tr, va, te)
    return out


def kfold_folds(root: str, names: list[str], k: int = 5, seed: int = 0) -> list[Path]:
    """Grouped k-fold over the pooled train + valid images (groups never straddle folds).
    Fold i validates on group-fold i and trains on the rest; the test split stays untouched."""
    root_p = Path(root)
    manifest = pd.read_csv(root_p / "manifest.csv")
    pool = manifest[manifest.split.isin(["train", "valid"])]
    groups = sorted(pool.group.unique())
    random.Random(seed).shuffle(groups)
    # Greedy: biggest groups first into the currently smallest fold, so folds have similar sizes.
    sizes = pool.group.value_counts()
    fold_of, load = {}, [0] * k
    for g in sorted(groups, key=lambda g: -sizes[g]):
        i = load.index(min(load))
        fold_of[g] = i
        load[i] += int(sizes[g])
    folds = root_p / "folds"
    folds.mkdir(exist_ok=True)
    out = []
    for i in range(k):
        val = pool[pool.group.map(fold_of) == i]
        tr_df = pool[pool.group.map(fold_of) != i]
        tr = _write_list(folds / f"kfold_{i}_train.txt",
                         [p for p, w in zip(tr_df.image, tr_df.weight) for _ in range(int(w))])
        va = _write_list(folds / f"kfold_{i}_val.txt", val.image.tolist())
        out.append(_yaml(folds / f"kfold_{i}.yaml", root_p, names, tr, va))
    return out


def summary(manifest: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Images and boxes per class for each source and split."""
    agg = {"image": "count", **{n: "sum" for n in names}}
    return manifest.groupby(["source", "split"]).agg(agg).rename(columns={"image": "images"})


def dhash(image_path: str, size: int = 8) -> int:
    """64-bit difference hash: survives re-encoding and resizing, so it finds the same frame
    re-exported by a Roboflow fork (where exact-byte dedupe misses it)."""
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_share(candidate: list[str], reference: list[str], max_bits: int = 4) -> tuple[float, list[str]]:
    """Share of candidate images within max_bits Hamming distance of any reference image,
    and the candidate paths that matched. A fork of a set you already have scores near 1."""
    import numpy as np
    ref = np.array([dhash(p) for p in reference], dtype=np.uint64)
    hits = []
    for p in candidate:
        if not len(ref):
            break
        h = np.uint64(dhash(p))
        x = np.bitwise_xor(ref, h)
        dist = np.array([bin(int(v)).count("1") for v in x])
        if dist.min() <= max_bits:
            hits.append(p)
    return (len(hits) / len(candidate) if candidate else 0.0), hits
