"""Grow and re-split YOLO detection datasets.

The v17 detector data has 213 training balls and a 32-image validation split,
so the ball model is short of examples and every per-class metric is noisy.
These helpers merge extra YOLO-format datasets into one class list (renaming
or dropping classes per source), add unlabelled hard-negative frames (crowd,
bench, replays) that teach the model what is not a player or a ball, and
re-split train/valid so the validation set is large enough to read.

Layout everywhere is the Roboflow YOLO export: <root>/<split>/images/*.jpg and
<root>/<split>/labels/*.txt, one "cls cx cy w h" row per box.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import yaml

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(open(data_yaml))["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def _label_for(image: Path) -> Path:
    return image.parent.parent / "labels" / (image.stem + ".txt")


def write_data_yaml(root: Path, names: list[str], kpt_shape: Optional[list[int]] = None,
                    flip_idx: Optional[list[int]] = None) -> Path:
    cfg = {"path": str(root), "train": "train/images", "val": "valid/images",
           "names": {i: n for i, n in enumerate(names)}}
    if (root / "test" / "images").exists():
        cfg["test"] = "test/images"
    if kpt_shape is not None:
        cfg["kpt_shape"] = kpt_shape
    if flip_idx is not None:
        cfg["flip_idx"] = flip_idx
    path = root / "data.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


def merge_datasets(sources: list[tuple[str, Optional[dict[str, Optional[str]]]]], names: list[str],
                   out: str, negatives: Optional[str] = None) -> dict[str, int]:
    """Merge YOLO datasets into `out` with the class list `names`.

    sources: (dataset root with a data.yaml, rename map). The map sends a source class name to a
    name in `names`, or to None to drop it; source classes not in the map keep their name if it is
    in `names` and are dropped otherwise. Pass None for no renames.
    negatives: optional folder of images with no objects of interest; each gets an empty label.
    Files are prefixed with the source index so names never collide. Returns box counts per class.
    """
    out_root = Path(out)
    counts = {n: 0 for n in names}
    for si, (src, rename) in enumerate(sources):
        src_root = Path(src)
        src_names = _names(src_root / "data.yaml")
        rename = rename or {}
        to_dst = {}
        for i, n in enumerate(src_names):
            target = rename.get(n, n)
            if target is not None and target in names:
                to_dst[i] = names.index(target)
        for split in SPLITS:
            img_dir = src_root / split / "images"
            if not img_dir.exists():
                continue
            (out_root / split / "images").mkdir(parents=True, exist_ok=True)
            (out_root / split / "labels").mkdir(parents=True, exist_ok=True)
            for img in _images(img_dir):
                stem = f"s{si}_{img.stem}"
                shutil.copy(img, out_root / split / "images" / (stem + img.suffix))
                rows = []
                lbl = _label_for(img)
                if lbl.exists():
                    for line in open(lbl):
                        parts = line.split()
                        if not parts or int(parts[0]) not in to_dst:
                            continue
                        cls = to_dst[int(parts[0])]
                        counts[names[cls]] += 1
                        rows.append(" ".join([str(cls)] + parts[1:]))
                (out_root / split / "labels" / (stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
    if negatives:
        (out_root / "train" / "images").mkdir(parents=True, exist_ok=True)
        (out_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
        for img in _images(Path(negatives)):
            stem = f"neg_{img.stem}"
            shutil.copy(img, out_root / "train" / "images" / (stem + img.suffix))
            (out_root / "train" / "labels" / (stem + ".txt")).write_text("")
    write_data_yaml(out_root, names)
    return counts


def resplit(root: str, val_frac: float = 0.2, seed: int = 0, stratify_class: Optional[int] = None,
            keep_test: bool = True) -> dict[str, int]:
    """Pool train + valid in place and re-split them with `val_frac` going to valid.

    stratify_class: images containing this class id are split separately, so the rare class
    (the ball) gets `val_frac` of its images in validation instead of whatever chance gives it.
    The test split is left alone when keep_test is True. Returns image counts per split.
    """
    root_p = Path(root)
    pool = [img for s in ("train", "valid") if (root_p / s / "images").exists()
            for img in _images(root_p / s / "images")]

    def has_cls(img: Path) -> bool:
        lbl = _label_for(img)
        return lbl.exists() and any(l.split() and int(l.split()[0]) == stratify_class for l in open(lbl))

    groups = [pool] if stratify_class is None else [[p for p in pool if has_cls(p)], [p for p in pool if not has_cls(p)]]
    rng = random.Random(seed)
    moves = []
    for group in groups:
        group = sorted(group)
        rng.shuffle(group)
        n_val = round(len(group) * val_frac)
        moves += [(p, "valid") for p in group[:n_val]] + [(p, "train") for p in group[n_val:]]

    staging = root_p / "_resplit"
    for split in ("train", "valid"):
        (staging / split / "images").mkdir(parents=True, exist_ok=True)
        (staging / split / "labels").mkdir(parents=True, exist_ok=True)
    for img, split in moves:
        lbl = _label_for(img)
        stem = img.stem
        if (staging / "train" / "labels" / f"{stem}.txt").exists() or (staging / "valid" / "labels" / f"{stem}.txt").exists():
            stem = f"{img.parent.parent.name}_{stem}"  # same file name in train and valid
        shutil.move(str(img), staging / split / "images" / (stem + img.suffix))
        if lbl.exists():
            shutil.move(str(lbl), staging / split / "labels" / f"{stem}.txt")
        else:
            (staging / split / "labels" / f"{stem}.txt").write_text("")
    for split in ("train", "valid"):
        shutil.rmtree(root_p / split, ignore_errors=True)
        shutil.move(str(staging / split), root_p / split)
    shutil.rmtree(staging)
    if not keep_test:
        shutil.rmtree(root_p / "test", ignore_errors=True)
    return {s: len(_images(root_p / s / "images")) for s in SPLITS if (root_p / s / "images").exists()}


def class_counts(root: str) -> dict[str, dict[str, int]]:
    """Box counts per class per split, plus image counts, for a quick sanity check."""
    root_p = Path(root)
    names = _names(root_p / "data.yaml")
    report = {}
    for split in SPLITS:
        img_dir = root_p / split / "images"
        if not img_dir.exists():
            continue
        c = {n: 0 for n in names}
        imgs = _images(img_dir)
        for img in imgs:
            lbl = _label_for(img)
            if lbl.exists():
                for line in open(lbl):
                    if line.split():
                        c[names[int(line.split()[0])]] += 1
        c["images"] = len(imgs)
        report[split] = c
    return report
